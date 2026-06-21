#!/usr/bin/env python3
"""
backtest.py — Signal-validation backtesting harness for index options strategies.

HOW IT WORKS
------------
Synthetic mode (default, fast):
  Builds option chains from the underlying's 1-min close + India VIX as the
  ATM IV, then applies a calibrated skew surface (steep left wing for puts,
  mild right wing for calls) using the BS model. No real option price data
  needed — good for testing signal logic, risk management, and P&L distribution.

KNOWN LIMITATIONS
  - SL / target checks use the BAR CLOSE, not intrabar high/low. A few extra
    SL hits that should have triggered mid-bar will be missed. Makes the
    backtest slightly optimistic on win rate. Upgrade to tick data when available.
  - Volume and OI are modelled (proportional to distance from ATM), so
    SkewHunter alpha1 (which uses OI changes) will be synthetic. Alpha2 (IV
    skew) is driven by the modelled surface and is more meaningful.
  - The synthetic skew surface is a static calibration. Real skew varies daily
    and around events (expiry, RBI, earnings).

USAGE
-----
  # Set env vars first
  export BROKER_APP_ID=XXXXXXXXXX-100
  export BROKER_ACCESS_TOKEN=<from Fyers dashboard or after auth flow>

  python backtest.py --start 2025-01-01 --end 2025-05-30
  python backtest.py --instrument banknifty --start 2025-01-01 --end 2025-05-30
  python backtest.py --list-instruments
  python backtest.py --capital 1000000 --output my_run.csv
"""

import argparse
import asyncio
import csv
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, date, time as dt_time
from typing import Dict, List, Optional

import httpx
import numpy as np

from bs_engine import BlackScholesEngine, OptionType
from broker_gateway import BrokerGateway, OrderResponse, Quote, Position, Order
from config import Exchange, ProductType, OrderType, TransactionType, StrategyConfig
from data_feed import OptionChainSnapshot, OptionQuote
from db_lock import TradingDatabase, TradeStatus
from instruments import InstrumentSpec, INSTRUMENTS
from risk_manager import RiskManager, RiskThresholds
from strategies import (
    FixedRR13Strategy,
    CurvatureCreditSpreadStrategy,
    SkewHunterStrategy,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("backtest")


# ── Fyers Historical Data ─────────────────────────────────────────────────────

class FyersHistoryClient:
    """Fetches OHLCV candles from Fyers v3 /history endpoint."""

    # Correct Fyers data history endpoint (confirmed via probe)
    BASE = "https://api-t1.fyers.in/data"
    # Fyers caps 1-min data at ~100 calendar days per request
    CHUNK_DAYS = 59

    def __init__(self, app_id: str, access_token: str) -> None:
        # Fyers v3 Authorization header format: "{app_id}:{access_token}"
        self._headers = {
            "Authorization": f"{app_id}:{access_token}",
        }

    def _safe_json(self, r: httpx.Response) -> dict:
        """Parse JSON with a clear error message showing the raw response."""
        try:
            return r.json()
        except Exception:
            text = r.text[:500]
            raise RuntimeError(
                f"\n  Fyers API returned non-JSON (HTTP {r.status_code}):\n"
                f"  {text}\n\n"
                f"  Possible causes:\n"
                f"  1. Access token has expired — run: python3 get_token_browser.py\n"
                f"  2. Symbol format is wrong — check the symbol name\n"
                f"  3. API rate limit hit — wait 60 seconds and retry\n"
                f"  4. Fyers API is temporarily down"
            )

    async def fetch_candles(
        self,
        symbol: str,
        from_date: str,   # "YYYY-MM-DD"
        to_date: str,
        resolution: int = 1,
    ) -> List[dict]:
        """Return list of dicts: {timestamp, open, high, low, close, volume}."""
        all_candles: List[dict] = []
        start = datetime.strptime(from_date, "%Y-%m-%d")
        end   = datetime.strptime(to_date,   "%Y-%m-%d")
        chunk = timedelta(days=self.CHUNK_DAYS)
        cur   = start

        async with httpx.AsyncClient(timeout=30) as client:
            while cur <= end:
                chunk_end = min(cur + chunk, end)
                params = {
                    "symbol":      symbol,
                    "resolution":  str(resolution),
                    "date_format": "1",
                    "range_from":  cur.strftime("%Y-%m-%d"),
                    "range_to":    chunk_end.strftime("%Y-%m-%d"),
                    "cont_flag":   "1",
                }
                r = await client.get(
                    f"{self.BASE}/history",
                    headers=self._headers,
                    params=params,
                )
                data = self._safe_json(r)
                status = data.get("s")
                if status == "no_data":
                    # Chunk is in the future or a holiday — skip silently
                    cur = chunk_end + timedelta(days=1)
                    continue
                if status != "ok":
                    raise RuntimeError(
                        f"Fyers history error [{symbol}] "
                        f"{cur.strftime('%Y-%m-%d')} → "
                        f"{chunk_end.strftime('%Y-%m-%d')}: {data}"
                    )
                for ts, o, h, l, cl, v in data.get("candles", []):
                    all_candles.append({
                        "timestamp": datetime.fromtimestamp(ts),
                        "open": o, "high": h, "low": l, "close": cl,
                        "volume": int(v),
                    })
                cur = chunk_end + timedelta(days=1)

        logger.info("Fetched %d candles for %s", len(all_candles), symbol)
        return all_candles

    async def fetch_daily_vix(self, from_date: str, to_date: str) -> Dict[date, float]:
        """Return {trading_date: india_vix_close}."""
        candles = await self.fetch_candles(
            "NSE:INDIAVIX-INDEX", from_date, to_date, resolution=1
        )
        daily: Dict[date, float] = {}
        for c in candles:
            daily[c["timestamp"].date()] = c["close"]   # last tick of day wins
        return daily


# ── Synthetic Option Chain ────────────────────────────────────────────────────

class SyntheticChainBuilder:
    """
    Builds OptionChainSnapshot from underlying close + VIX.

    Skew calibration (typical Nifty surface):
      OTM put IV  = atm_iv × exp(1.5 × put_moneyness)   — steep left wing
      OTM call IV = atm_iv × exp(0.3 × call_moneyness)  — mild right wing
    """

    def __init__(self, bs: BlackScholesEngine, spec: InstrumentSpec,
                 seed: int = 42) -> None:
        self.bs   = bs
        self.spec = spec
        self._rng  = np.random.default_rng(seed)
        self._bias: float = 0.0          # sentiment state: +bias=bullish, -bias=bearish
        self._oi:   Dict[str, float] = {}

    def build(
        self,
        spot:      float,
        vix_pct:   float,
        timestamp: datetime,
        expiry_dt: datetime,
        tte:       float,
    ) -> OptionChainSnapshot:
        atm_iv  = max(0.05, vix_pct / 100)
        si      = self.spec.strike_interval
        atm     = round(spot / si) * si
        exp_str = expiry_dt.strftime("%d%b%y").upper()

        snap = OptionChainSnapshot(
            underlying=self.spec.name,
            spot_price=spot,
            timestamp=timestamp,
            expiry=exp_str,
            atm_strike=atm,
        )
        if tte < 1e-6:
            return snap

        pfx = f"{self.spec.segment}:{self.spec.name}"

        # Stochastic sentiment bias — same model as OfflineChainBuilder.
        # Drives call/put skew slope ratio AND volume imbalance so alpha
        # signals cross thresholds naturally on real market data.
        self._bias += float(self._rng.normal(0, 0.04))
        self._bias  = float(np.clip(self._bias, -3.0, 3.0))

        put_slope  = max(0.00005, 0.0003 * (1.0 - 0.18 * self._bias))
        call_slope = max(0.00005, 0.0001 * (1.0 + 0.60 * self._bias))
        call_vol_mult = max(0.15, 1.0 + 0.50 * self._bias)
        put_vol_mult  = max(0.15, 1.0 - 0.35 * self._bias)

        # ±25 strikes — covers all moneyness levels for FixedRR energy calc
        indices  = np.arange(-25, 26)
        strikes  = atm + indices * si
        valid    = strikes > 0
        strikes  = strikes[valid]
        indices  = indices[valid]

        dk_put   = np.maximum(0.0, atm - strikes)
        dk_call  = np.maximum(0.0, strikes - atm)
        c_iv_arr = np.maximum(0.04, atm_iv + call_slope * dk_call)
        p_iv_arr = np.maximum(0.04, atm_iv + put_slope  * dk_put)

        c_px_arr = np.maximum(0.05, self.bs.call_price(spot, strikes, tte, c_iv_arr))
        p_px_arr = np.maximum(0.05, self.bs.put_price( spot, strikes, tte, p_iv_arr))
        c_dl_arr = self.bs.delta(spot, strikes, tte, c_iv_arr, OptionType.CALL)
        p_dl_arr = self.bs.delta(spot, strikes, tte, p_iv_arr, OptionType.PUT)
        n_vol    = self._rng.lognormal(0, 0.15, size=(len(strikes), 2))

        for j, (i, strike) in enumerate(zip(indices, strikes)):
            spread  = 0.004 + abs(i) * 0.001
            v_base  = max(20, int(8_000 - abs(i) * 290))
            c_vol   = max(10, int(v_base * call_vol_mult * n_vol[j, 0]))
            p_vol   = max(10, int(v_base * put_vol_mult  * n_vol[j, 1]))
            oi_base = max(200, 80_000 - int(abs(i) * 2_800))

            def _oi(key: str, base: int) -> int:
                prev  = self._oi.get(key, float(base))
                swing = prev * (0.006 if abs(i) <= 3 else 0.002)
                new   = max(100.0, prev + float(self._rng.normal(0, swing)))
                self._oi[key] = new
                return int(new)

            # Differentiate bid_qty vs ask_qty based on sentiment bias
            # so Curvature's viscosity signal is non-trivial.
            # When _bias > 0 (bullish): more buyers on calls → bid_qty > ask_qty
            # When _bias < 0 (bearish): more sellers on calls → ask_qty > bid_qty
            bias_mult_bid = max(0.3, 1.0 + 0.4 * self._bias)
            bias_mult_ask = max(0.3, 1.0 - 0.3 * self._bias)

            snap.calls[float(strike)] = OptionQuote(
                symbol=f"{pfx}{exp_str}{int(strike)}CE", strike=float(strike),
                expiry=exp_str, option_type=OptionType.CALL,
                ltp=round(float(c_px_arr[j]), 2),
                bid=round(float(c_px_arr[j]) * (1 - spread), 2),
                ask=round(float(c_px_arr[j]) * (1 + spread), 2),
                bid_qty=max(10, int(c_vol * bias_mult_bid)),
                ask_qty=max(10, int(c_vol * bias_mult_ask)),
                volume=c_vol * 10,
                oi=_oi(f"{pfx}{exp_str}{int(strike)}CE", oi_base),
                iv=float(c_iv_arr[j]), delta=float(c_dl_arr[j]),
            )
            # Puts: reversed — when bullish, sellers dominate put side
            snap.puts[float(strike)] = OptionQuote(
                symbol=f"{pfx}{exp_str}{int(strike)}PE", strike=float(strike),
                expiry=exp_str, option_type=OptionType.PUT,
                ltp=round(float(p_px_arr[j]), 2),
                bid=round(float(p_px_arr[j]) * (1 - spread), 2),
                ask=round(float(p_px_arr[j]) * (1 + spread), 2),
                bid_qty=max(10, int(p_vol * bias_mult_ask)),  # reversed for puts
                ask_qty=max(10, int(p_vol * bias_mult_bid)),
                volume=p_vol * 10,
                oi=_oi(f"{pfx}{exp_str}{int(strike)}PE", oi_base),
                iv=float(p_iv_arr[j]), delta=float(p_dl_arr[j]),
            )

        return snap


# ── Simulated Broker ──────────────────────────────────────────────────────────

class BacktestBroker(BrokerGateway):
    """
    Fills limit orders at the limit price (assumes liquid index options).
    Fills market orders at current price ± 0.1% slippage.
    """

    SLIPPAGE = 0.001

    def __init__(self) -> None:
        # BrokerGateway expects config + compliance; we bypass both
        self._prices:  Dict[str, float] = {}
        self._next_id: int = 0
        self._client       = None
        self._session_token = "backtest"

    def update_prices(self, chain: OptionChainSnapshot) -> None:
        for q in chain.calls.values():
            self._prices[q.symbol] = q.ltp
        for q in chain.puts.values():
            self._prices[q.symbol] = q.ltp

    def price_of(self, symbol: str) -> Optional[float]:
        return self._prices.get(symbol)

    # ── BrokerGateway interface ───────────────────────────────────────────

    async def authenticate(self) -> bool:
        return True

    async def place_order(
        self,
        symbol: str,
        exchange: Exchange,
        transaction_type: TransactionType,
        order_type: OrderType,
        quantity: int,
        price: float = 0.0,
        trigger_price: float = 0.0,
        product_type: ProductType = ProductType.INTRADAY,
    ) -> OrderResponse:
        self._next_id += 1
        oid = f"BT{self._next_id:05d}"

        if order_type == OrderType.MARKET:
            fill = self._prices.get(symbol, price)
            fill *= (1 + self.SLIPPAGE) if transaction_type == TransactionType.BUY \
                   else (1 - self.SLIPPAGE)
        else:
            fill = price   # limit → assume fills at limit

        return OrderResponse(
            success=True,
            order_id=oid,
            broker_order_id=oid,
            message=f"BT fill @ {fill:.2f}",
            status="FILLED",
        )

    async def modify_order(self, order_id: str, **kwargs) -> OrderResponse:
        return OrderResponse(success=True, order_id=order_id)

    async def cancel_order(self, order_id: str) -> OrderResponse:
        return OrderResponse(success=True, order_id=order_id)

    async def get_positions(self) -> List[Position]:
        return []

    async def get_orders(self) -> List[Order]:
        return []

    async def get_quote(self, symbol: str, exchange: Exchange) -> Optional[Quote]:
        px = self._prices.get(symbol)
        if px is None:
            return None
        return Quote(
            symbol=symbol, ltp=px,
            bid_price=px * 0.999, ask_price=px * 1.001,
            bid_qty=500, ask_qty=500,
            volume=10_000, oi=100_000,
            timestamp=datetime.now().isoformat(),
        )


# ── Expiry / TTE Helpers ──────────────────────────────────────────────────────

def nearest_expiry(ref: datetime, weekday: int) -> datetime:
    """Next occurrence of the given weekday (0=Mon … 4=Fri) from ref."""
    days = (weekday - ref.weekday()) % 7
    if days == 0 and ref.time() >= dt_time(15, 30):
        days = 7
    return ref + timedelta(days=days)


def trading_tte(now: datetime, expiry_dt: datetime) -> float:
    """TTE in years counting Mon–Fri trading days only."""
    days = 0.0
    cur  = now
    while cur.date() < expiry_dt.date():
        if cur.weekday() < 5:
            days += 1
        cur += timedelta(days=1)
    if now.weekday() < 5:
        close_ = now.replace(hour=15, minute=30, second=0, microsecond=0)
        open_  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
        if now < close_:
            elapsed = max(0.0, (now - open_).total_seconds())
            session = (close_ - open_).total_seconds()
            days += max(0.0, 1.0 - elapsed / session)
    return max(1 / 252, days / 252)


# ── Performance Report ────────────────────────────────────────────────────────

def _sharpe(pnls: list, initial_capital: float, trading_days: int = 244) -> float:
    """Annualised Sharpe ratio from trade P&L.
    Groups by exit day, fills missing days with 0, then computes mean/std.
    """
    if not pnls or len(pnls) < 2:
        return 0.0
    daily_ret = np.array(pnls) / initial_capital
    mean_r = float(np.mean(daily_ret))
    std_r  = float(np.std(daily_ret, ddof=1))
    if std_r < 1e-12:
        return 0.0
    # Annualise: assume each trade ≈ 1 day, scale to 252 trading days
    return float((mean_r / std_r) * np.sqrt(252))


def _strategy_metrics(pnls: list, initial_capital: float = 500_000) -> dict:
    """Compute full metrics dict for a list of P&L values."""
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total  = sum(pnls)
    wr     = len(wins) / max(1, len(pnls)) * 100
    aw     = float(np.mean(wins))   if wins   else 0.0
    al     = float(np.mean(losses)) if losses else 0.0
    loss_sum = abs(sum(losses)) if losses else 0.0
    pf       = sum(wins) / loss_sum if wins and loss_sum > 1e-9 else float("inf")
    exp    = (wr / 100 * aw) + ((1 - wr / 100) * al)
    sharpe = _sharpe(pnls, initial_capital)
    # Drawdown as % of initial capital — comparable to the portfolio-level figure
    cap = peak = initial_capital
    dd  = 0.0
    for p in pnls:
        cap  += p
        peak  = max(peak, cap)
        dd    = max(dd, (peak - cap) / peak * 100)
    ann_return = (total / initial_capital) * (252 / max(1, len(pnls)))
    calmar = (ann_return * 100 / dd) if dd > 0 else 0.0
    return dict(total=total, trades=len(pnls), wr=wr,
                avg_win=aw, avg_loss=al, pf=pf, exp=exp,
                max_dd=dd, sharpe=sharpe, calmar=calmar)


def _print_metrics_block(title: str, m: dict, initial_capital: float,
                          indent: str = "  ") -> None:
    """Print a full metrics block for any set of trades."""
    W = 57
    print(f"\n{indent}{'─' * W}")
    print(f"{indent}  {title}")
    print(f"{indent}{'─' * W}")
    print(f"{indent}  Total Trades    :  {m['trades']}")
    print(f"{indent}  Win Rate        :  {m['wr']:.1f}%")
    print(f"{indent}  Avg Win         :  ₹{m['avg_win']:>10,.0f}")
    print(f"{indent}  Avg Loss        :  ₹{m['avg_loss']:>10,.0f}")
    if m['avg_loss'] != 0:
        print(f"{indent}  Actual RR       :  {abs(m['avg_win'] / m['avg_loss']):.2f} : 1")
    print(f"{indent}  Profit Factor   :  {m['pf']:.2f}")
    print(f"{indent}  Expectancy/trade:  ₹{m['exp']:>10,.0f}")
    print(f"{indent}  Total P&L       :  ₹{m['total']:>10,.0f}  "
          f"({m['total'] / initial_capital * 100:+.2f}%)")
    print(f"{indent}  Max Drawdown    :  {m['max_dd']:.2f}%")
    print(f"{indent}  Sharpe Ratio    :  {m.get('sharpe', 0):.2f}")
    print(f"{indent}  Calmar Ratio    :  {m.get('calmar', 0):.2f}")


def generate_report(
    db: TradingDatabase,
    initial_capital: float,
    output_csv: str,
    label: str = "",
    output_dir: str = "backtest_results_output",
) -> dict:
    """Print overall + per-strategy metrics, save CSV, return summary dict."""
    conn = db._get_connection()
    rows = conn.execute(
        """SELECT * FROM trades
           WHERE status IN ('CLOSED','SQUARED_OFF','EMERGENCY_EXIT')
           ORDER BY exit_time"""
    ).fetchall()

    if not rows:
        print("\n  No completed trades to report.\n")
        return {}

    all_pnls = [r["pnl"] for r in rows if r["pnl"] is not None]
    m_all    = _strategy_metrics(all_pnls)

    # ── Overall portfolio summary ─────────────────────────────────────────────
    W = 55
    print("\n" + "═" * W)
    hdr = f"  OVERALL  —  {label}" if label else "  OVERALL PERFORMANCE"
    print(hdr)
    print("═" * W)
    print(f"  Initial Capital :  ₹{initial_capital:>12,.0f}")
    print(f"  Final Capital   :  ₹{initial_capital + m_all['total']:>12,.0f}")
    _print_metrics_block("Portfolio Total", m_all, initial_capital, indent="")
    print("═" * W)

    # ── Per-strategy full breakdown ───────────────────────────────────────────
    strat_names = sorted({r["strategy_name"] for r in rows})
    if len(strat_names) > 1 or strat_names:
        print(f"\n  {'─'*W}")
        print(f"  PER-STRATEGY BREAKDOWN")

    strat_summaries = {}
    for sname in strat_names:
        s_rows = [r for r in rows if r["strategy_name"] == sname]
        s_pnls = [r["pnl"] for r in s_rows if r["pnl"] is not None]
        if not s_pnls:
            continue
        m = _strategy_metrics(s_pnls, initial_capital)
        strat_summaries[sname] = m
        _print_metrics_block(sname, m, initial_capital)
        reasons: dict = {}
        for r in s_rows:
            reason = r["exit_reason"] or "UNKNOWN"
            reasons[reason] = reasons.get(reason, 0) + 1
        for k, v in sorted(reasons.items()):
            print(f"      {k}: {v} trades")

    print()

    # ── Save CSVs to output directory ────────────────────────────────────────
    import os as _os
    _os.makedirs(output_dir, exist_ok=True)

    fields = [
        "strategy_name", "symbol", "option_type", "strike", "direction",
        "entry_time", "exit_time", "entry_price", "exit_price",
        "quantity", "pnl", "exit_reason",
    ]

    # Combined file (all strategies)
    combined_path = _os.path.join(output_dir, _os.path.basename(output_csv))
    with open(combined_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields})

    # One file per strategy
    base_name = _os.path.splitext(_os.path.basename(output_csv))[0]
    for sname in strat_names:
        s_rows = [r for r in rows if r["strategy_name"] == sname]
        safe   = sname.replace(" ", "_").replace("/", "-")
        spath  = _os.path.join(output_dir, f"{base_name}_{safe}.csv")
        with open(spath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in s_rows:
                writer.writerow({k: r[k] for k in fields})
        print(f"  [{sname}] → {spath}")

    print(f"  [All]            → {combined_path}\n")

    return {
        "label":         label or output_csv,
        "total_pnl":     m_all["total"],
        "trades":        m_all["trades"],
        "win_rate":      m_all["wr"],
        "profit_factor": m_all["pf"],
        "max_dd":        m_all["max_dd"],
        "return_pct":    m_all["total"] / initial_capital * 100,
        "expectancy":    m_all["exp"],
        "sharpe":        m_all.get("sharpe", 0),
        "calmar":        m_all.get("calmar", 0),
        "by_strategy":   strat_summaries,
    }


# ── Main Backtest Engine ──────────────────────────────────────────────────────

STRATEGY_NAMES = {
    "FixedRR_1to3":           "FixedRR_1to3",
    "fixedrr":                "FixedRR_1to3",
    "CurvatureCreditSpread":  "CurvatureCreditSpread",
    "curvature":              "CurvatureCreditSpread",
    "SkewHunter":             "SkewHunter",
    "skewhunter":             "SkewHunter",
    "ExpiryShortStrangle":    "ExpiryShortStrangle",
    "strangle":               "ExpiryShortStrangle",
    "ironcondor":             "ExpiryShortStrangle",
    "ZenCreditSpread":        "ZenCreditSpread",
    "zen":                    "ZenCreditSpread",
    "LyapunovCreditSpread":   "LyapunovCreditSpread",
    "lyapunov":               "LyapunovCreditSpread",
}


async def run_backtest(
    instrument_key:  str,
    start_date:      str,
    end_date:        str,
    initial_capital: float = 500_000,
    risk_free_rate:  float = 0.065,
    output_csv:      str   = "backtest_results.csv",
    output_dir:      str   = "backtest_results_output",
    strategy_filter: Optional[str] = None,   # None = run all; else one strategy name
) -> dict:
    spec = INSTRUMENTS[instrument_key]
    # ── Check for real NSE option cache ──────────────────────────────────────
    from nse_data_fetcher import RealOptionDayData, CalibratedChainBuilder, CACHE_ROOT as NSE_CACHE
    use_real_nse = RealOptionDayData.is_cache_available(NSE_CACHE)

    print(f"\n{'='*60}")
    print(f"  Instrument : {spec.display_name}")
    print(f"  Period     : {start_date}  →  {end_date}")
    print(f"  Capital    : ₹{initial_capital:,.0f}")
    if use_real_nse:
        nse_stats = RealOptionDayData.cache_stats(NSE_CACHE)
        print(f"  Chain data : REAL NSE ({nse_stats['trading_days_cached']} days, "
              f"{nse_stats['cache_size_mb']} MB)")
        print(f"               ← Real IV, real OI, real volume from NSE bhavcopy")
    else:
        print(f"  Chain data : Synthetic BS model")
        print(f"               Run: python3 nse_data_fetcher.py --start 2025-06-25")
    print(f"{'='*60}")

    app_id       = os.environ.get("BROKER_APP_ID", "")
    access_token = os.environ.get("BROKER_ACCESS_TOKEN", "")
    if not app_id or not access_token:
        raise RuntimeError(
            "BROKER_APP_ID and BROKER_ACCESS_TOKEN must be set.\n"
            "Run:  python3 get_token_browser.py  to get a token."
        )
    data_client  = FyersHistoryClient(app_id, access_token)

    # ── Quick smoke test: fetch 2 days of data first ─────────────────────────
    print("  Verifying Fyers data API access ...", end=" ", flush=True)
    test_end   = end_date
    test_start = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=3)).strftime("%Y-%m-%d")
    try:
        test_candles = await data_client.fetch_candles(spec.spot_symbol, test_start, test_end)
        print(f"OK ({len(test_candles)} test bars fetched)")
    except RuntimeError as e:
        print(f"FAILED\n{e}")
        return {}

    print("  Fetching 1-min underlying candles ...", flush=True)
    candles = await data_client.fetch_candles(spec.spot_symbol, start_date, end_date)
    if not candles:
        print("  ERROR: no candle data returned. Check credentials and symbol.")
        return

    print("  Fetching India VIX (daily) ...", flush=True)
    vix_map = await data_client.fetch_daily_vix(start_date, end_date)
    print(f"  Data ready: {len(candles)} bars, {len(vix_map)} VIX days\n")

    # Silence noisy-but-expected backtest warnings:
    #   db_lock   — lock repeats across bars (simulation faster than 5-min timeout)
    #   risk_manager — daily-loss-limit fires on every bar after it's breached
    for _log in ("trading_system.db_lock", "trading_system.risk_manager"):
        logging.getLogger(_log).setLevel(logging.ERROR)

    # Core objects
    bs = BlackScholesEngine(risk_free_rate)

    # Use real NSE calibrated data if cache exists, otherwise synthetic
    if use_real_nse:
        day_data = RealOptionDayData(NSE_CACHE)
        builder  = CalibratedChainBuilder(
            bs=bs, spec=spec, day_data=day_data,
            synthetic_fallback=SyntheticChainBuilder(bs, spec),
        )
    else:
        builder = SyntheticChainBuilder(bs, spec)

    broker  = BacktestBroker()
    db      = TradingDatabase(":memory:")   # fresh per run

    risk = RiskManager(
        database=db,
        capital=initial_capital,
        thresholds=RiskThresholds(
            max_daily_loss_percent=2.0,
            max_drawdown_percent=5.0,
            trailing_drawdown_percent=3.0,
            max_position_size=spec.lot_size * 10,
            max_open_positions=4,
            position_size_per_trade=spec.lot_size,
            # Phase 1/3: Per-strategy capital sleeves — read from env (set by CLI --capital)
            skewhunter_allocated_capital=float(
                os.environ.get("SLEEVE_SKEWHUNTER", "100000")),
            strangle_allocated_capital=float(
                os.environ.get("SLEEVE_STRANGLE", "300000")),
            credit_spread_allocated_capital=float(
                os.environ.get("SLEEVE_CREDIT_SPREAD", "100000")),
            risk_per_trade_percent=2.0,
        ),
        on_risk_event=lambda e, d: logger.warning("Risk event: %s", e.value),
    )

    # Thresholds calibrated for synthetic chain model.
    # The chain uses modelled (not real) volume/OI, so signals are weaker
    # than live. These thresholds match what's used in backtest_offline.py.
    cfg = StrategyConfig(
        fixed_rr_alpha1_long_threshold=0.60,
        fixed_rr_alpha2_long_threshold=0.58,
        fixed_rr_alpha1_short_threshold=0.40,
        fixed_rr_alpha2_short_threshold=0.42,
        skewhunter_alpha1_long=0.62,
        skewhunter_alpha2_long=0.60,
        skewhunter_alpha1_short=0.38,
        skewhunter_alpha2_short=0.40,
    )
    from strategies import (
        ExpiryShortStrangleStrategy, ZenCreditSpreadStrategy,
        LyapunovCreditSpreadStrategy,
    )

    strat_fixed    = FixedRR13Strategy(cfg, bs, db, risk, broker)
    strat_curv     = CurvatureCreditSpreadStrategy(cfg, bs, db, risk, broker)
    strat_skew     = SkewHunterStrategy(cfg, bs, db, risk, broker)
    strat_strangle = ExpiryShortStrangleStrategy(cfg, bs, db, risk, broker)
    strat_zen      = ZenCreditSpreadStrategy(cfg, bs, db, risk, broker)
    strat_lyap     = LyapunovCreditSpreadStrategy(cfg, bs, db, risk, broker)

    _all = [strat_fixed, strat_curv, strat_skew, strat_strangle, strat_zen, strat_lyap]

    # ── Strategy filter: run only the requested strategy ─────────────────────
    if strategy_filter:
        canonical = STRATEGY_NAMES.get(strategy_filter) or STRATEGY_NAMES.get(
            strategy_filter.lower()
        )
        if not canonical:
            raise ValueError(
                f"Unknown strategy '{strategy_filter}'. "
                f"Valid names: {sorted(set(STRATEGY_NAMES.values()))}"
            )
        all_strategies = [s for s in _all if s.name == canonical]
        if not all_strategies:
            raise ValueError(f"Strategy '{canonical}' not found in instantiated list.")
        print(f"  Strategy filter : {canonical} only")
    else:
        all_strategies = _all

    # Monkey-patch time checks: BacktestEngine controls windows via bar_time.
    for s in all_strategies:
        s.is_trading_window = lambda: True          # type: ignore[method-assign]
    strat_curv.is_entry_window = lambda *a, **kw: True  # accepts ts param

    # Time window constants
    T_OPEN           = dt_time(9, 15)
    T_STRAT_START    = dt_time(10, 15)
    T_STRAT_END      = dt_time(14, 15)
    T_STRANGLE_START = dt_time(10, 30)   # strangle entry: after morning vol settles
    T_STRANGLE_END   = dt_time(11, 15)
    T_CURV_START     = dt_time(15, 0)
    T_CURV_END       = dt_time(15, 25)
    T_SQUAREOFF      = dt_time(15, 15)
    T_CLOSE          = dt_time(15, 30)

    # Group candles by trading date
    by_date: Dict[date, List[dict]] = defaultdict(list)
    for c in candles:
        by_date[c["timestamp"].date()].append(c)

    total_bars   = 0
    trading_days = 0
    nse_cal_days = 0   # days where NSE real data calibration succeeded

    for day in sorted(by_date.keys()):
        day_bars   = sorted(by_date[day], key=lambda b: b["timestamp"])
        vix        = vix_map.get(day, 15.0)
        expiry_dt  = nearest_expiry(
            datetime.combine(day, T_OPEN), spec.expiry_weekday
        )
        exp_str    = expiry_dt.strftime("%d%b%y").upper()
        trading_days += 1

        # Show which data source will calibrate today's option chain
        if use_real_nse and isinstance(builder, CalibratedChainBuilder):
            has_data = day_data.load(day)
            if has_data:
                nse_cal_days += 1
                if trading_days <= 3 or trading_days % 50 == 0:
                    stats = day_data.get_daily_stats(spec.name)
                    print(f"  {day}  NSE real data: strikes={stats.get('strikes',0)}  "
                          f"total_OI={stats.get('total_oi',0):,}  "
                          f"spot=₹{stats.get('spot',0):,.0f}  VIX={vix:.1f}%")

        # Reset daily counters and Phase 4 kill-switch each morning.
        with risk._lock:
            risk._metrics.realized_pnl       = 0.0
            risk._metrics.unrealized_pnl     = 0.0
            risk._metrics.daily_loss_percent = 0.0
        risk.reset_daily_kill_switch()   # Phase 4: fresh start each day

        for bar in day_bars:
            ts       = bar["timestamp"]
            bar_time = ts.time()

            if bar_time < T_OPEN or bar_time > T_CLOSE:
                continue

            tte   = trading_tte(ts, expiry_dt)
            chain = builder.build(bar["close"], vix, ts, expiry_dt, tte)
            broker.update_prices(chain)
            total_bars += 1

            # Track spot price every bar for Curvature intraday momentum signal.
            strat_curv._track_spot(chain)

            # ── Phase 4: Intrabar high/low trailing stop check ───────────
            # Using close price alone masks extreme intrabar volatility.
            # Use bar['high'] and bar['low'] to trigger SL/target/trailing
            # stop before candle close, matching real market behaviour.
            bar_high = bar.get("high", bar["close"])
            bar_low  = bar.get("low",  bar["close"])

            for active_trade in db.get_active_trades():
                if active_trade.direction != "BUY":
                    continue
                entry = active_trade.entry_price
                if entry <= 0:
                    continue

                # Check if bar's LOW hit the stop loss
                pos = risk.get_position(active_trade.trade_id)
                current_sl = pos.stop_loss if pos else active_trade.stop_loss
                if bar_low <= current_sl:
                    px = max(bar_low, current_sl)   # fill at SL (realistic)
                    db.update_trade_status(
                        active_trade.trade_id, TradeStatus.CLOSED,
                        exit_price=px, exit_reason="INTRABAR_SL",
                    )
                    pnl = (px - entry) * active_trade.quantity
                    risk.remove_position(active_trade.trade_id, pnl)
                    db.release_trade_lock(active_trade.strategy_name, active_trade.symbol)
                    continue

                # Check if bar's HIGH hit the target
                if active_trade.target and bar_high >= active_trade.target:
                    px = min(bar_high, active_trade.target)
                    db.update_trade_status(
                        active_trade.trade_id, TradeStatus.CLOSED,
                        exit_price=px, exit_reason="INTRABAR_TARGET",
                    )
                    pnl = (px - entry) * active_trade.quantity
                    risk.remove_position(active_trade.trade_id, pnl)
                    db.release_trade_lock(active_trade.strategy_name, active_trade.symbol)
                    continue

                # Update dynamic trailing stop using bar's HIGH
                from strategies import BaseStrategy as _BS
                new_sl = _BS._dynamic_trailing_sl(entry, bar_high)
                if pos and new_sl > pos.stop_loss:
                    risk.update_trailing_stop(active_trade.trade_id, new_sl)

            # Phase 4: Intraday kill-switch — checked EVERY BAR using bar['low']
            # for worst-case intrabar marking (close prices hide intrabar extremes).
            # Combined capital = sum of all three strategy sleeves.
            combined_capital = (
                getattr(risk.thresholds, "skewhunter_allocated_capital",   100_000)
                + getattr(risk.thresholds, "strangle_allocated_capital",   300_000)
                + getattr(risk.thresholds, "credit_spread_allocated_capital", 100_000)
            )
            risk.check_intraday_kill_switch(
                bar_high=bar.get("high", bar["close"]),
                bar_low =bar.get("low",  bar["close"]),
                combined_capital=combined_capital,
            )

            # Phase 4: EMERGENCY_SQUARE_OFF — exit all positions at bar['low']
            # (worst-case intrabar fill for long positions)
            if risk._kill_switch_active:
                for kst in db.get_active_trades():
                    # Fill at bar_low for BUY positions (worst realistic exit)
                    # Fill at bar_high for SELL positions
                    if kst.direction == "BUY":
                        kpx = bar.get("low", broker.price_of(kst.symbol) or kst.entry_price)
                    else:
                        kpx = bar.get("high", broker.price_of(kst.symbol) or kst.entry_price)
                    kpx = max(0.05, kpx)   # floor at tick size
                    db.update_trade_status(
                        kst.trade_id, TradeStatus.SQUARED_OFF,
                        exit_price=kpx, exit_reason="EMERGENCY_SQUARE_OFF",
                    )
                    km  = 1 if kst.direction == "BUY" else -1
                    kpl = (kpx - kst.entry_price) * kst.quantity * km
                    risk.remove_position(kst.trade_id, kpl)
                    db.release_trade_lock(kst.strategy_name, kst.symbol)

            # ── Strategy entry windows ────────────────────────────────────
            # Each block guards on `in all_strategies` so filtered runs skip
            # strategies that weren't requested via --strategy.

            # FixedRR, SkewHunter, Zen, Lyapunov — main strategy window
            if T_STRAT_START <= bar_time <= T_STRAT_END:
                for strategy in [strat_fixed, strat_skew, strat_zen, strat_lyap]:
                    if strategy not in all_strategies:
                        continue
                    try:
                        sig = await strategy.evaluate(chain)
                        if sig:
                            await strategy.execute_signal(sig)
                    except Exception as e:
                        logger.debug("%s eval: %s", strategy.name, e)

            # Strangle / Iron Condor enters 10:30-11:15 AM
            if T_STRANGLE_START <= bar_time <= T_STRANGLE_END:
                if strat_strangle in all_strategies:
                    try:
                        sig = await strat_strangle.evaluate(chain)
                        if sig:
                            await strat_strangle.execute_signal(sig)
                    except Exception as e:
                        logger.debug("Strangle eval: %s", e)

            # Curvature fires in the overnight entry window (15:00-15:25)
            if T_CURV_START <= bar_time <= T_CURV_END:
                if strat_curv in all_strategies:
                    try:
                        sig = await strat_curv.evaluate(chain)
                        if sig:
                            await strat_curv.execute_signal(sig)
                    except Exception as e:
                        logger.debug("Curvature eval: %s", e)

            # ── Position management (every bar) ──────────────────────────

            for strategy in all_strategies:
                try:
                    await strategy.manage_positions()
                except Exception as e:
                    logger.debug("%s manage: %s", strategy.name, e)

            # ── Mandatory intraday square-off at 3:15 PM ─────────────────

            if bar_time >= T_SQUAREOFF:
                for trade in db.get_active_trades():
                    if trade.product_type != "INTRADAY":
                        continue
                    px = broker.price_of(trade.symbol) or trade.entry_price
                    db.update_trade_status(
                        trade.trade_id,
                        TradeStatus.SQUARED_OFF,
                        exit_price=px,
                        exit_reason="EOD_SQUAREOFF",
                    )
                    direction_mult = 1 if trade.direction == "BUY" else -1
                    pnl = (px - trade.entry_price) * trade.quantity * direction_mult
                    risk.remove_position(trade.trade_id, pnl)
                    # Release lock so strategy can enter a new position tomorrow
                    db.release_trade_lock(trade.strategy_name, trade.symbol)

        # Per-day summary (only print days with activity)
        stats = db.get_daily_stats(day.isoformat())
        if stats["total_trades"] > 0:
            print(f"  {day}  trades={stats['total_trades']:2d}  "
                  f"pnl=₹{stats['realized_pnl']:+10,.0f}")

    if use_real_nse:
        print(f"\n  Data sources used:")
        print(f"    Fyers API  : {total_bars:,} 1-min Nifty/BankNifty/Sensex bars (intraday price moves)")
        print(f"    NSE bhavcopy: {nse_cal_days}/{trading_days} days calibrated  "
              f"(real IV + OI + volume per strike)")
        if nse_cal_days < trading_days * 0.8:
            print(f"    NOTE: {trading_days-nse_cal_days} days used synthetic fallback "
                  f"(NSE holidays / missing bhavcopy)")
    else:
        print(f"\n  Processed {total_bars:,} bars across {trading_days} trading days (synthetic chains)")
    print(f"\n  Processed {total_bars:,} bars across {trading_days} trading days")
    lbl = f"{spec.display_name}  ({trading_days} days, {'NSE real IV+OI+Vol' if use_real_nse else 'synthetic BS'})"
    return generate_report(db, initial_capital, output_csv, label=lbl,
                           output_dir=output_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest index options strategies using Fyers historical data"
    )
    parser.add_argument(
        "--instrument", default="nifty",
        choices=list(INSTRUMENTS.keys()),
        help="Which index to backtest (default: nifty)",
    )
    parser.add_argument(
        "--strategy", default=None,
        metavar="NAME",
        help=(
            "Run a single strategy only. "
            "Valid names: SkewHunter, FixedRR_1to3, CurvatureCreditSpread, "
            "ExpiryShortStrangle, ZenCreditSpread, LyapunovCreditSpread  "
            "(aliases: skewhunter, fixedrr, curvature, strangle, zen, lyapunov)"
        ),
    )
    parser.add_argument("--start",  default="2025-01-01", metavar="YYYY-MM-DD")
    parser.add_argument("--end",    default="2025-05-30", metavar="YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=500_000,
                        help="Initial capital in INR (default: 500000)")
    parser.add_argument("--output", default="backtest_results.csv",
                        help="Output CSV path")
    parser.add_argument("--list-instruments", action="store_true",
                        help="Print all supported instruments and exit")
    parser.add_argument("--list-strategies", action="store_true",
                        help="Print all supported strategy names and exit")
    args = parser.parse_args()

    if args.list_instruments:
        print(f"\n{'Key':<14} {'Name':<32} {'Lot':>4}  {'ΔK':>6}  {'Expiry':<6}  {'Min ₹'}")
        print("-" * 75)
        days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        for key, spec in INSTRUMENTS.items():
            print(f"  {key:<12} {spec.display_name:<32} {spec.lot_size:>4}  "
                  f"₹{spec.strike_interval:>5.0f}  {days[spec.expiry_weekday]:<6}  "
                  f"₹{spec.min_capital:,}")
        print()
        return

    if args.list_strategies:
        print("\n  Available strategies (--strategy NAME):\n")
        seen = set()
        for alias, canonical in sorted(STRATEGY_NAMES.items()):
            if canonical not in seen:
                print(f"  {canonical}")
                seen.add(canonical)
        print("\n  Aliases accepted: "
              + ", ".join(k for k in STRATEGY_NAMES if k not in seen))
        print()
        return

    asyncio.run(run_backtest(
        instrument_key=args.instrument,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        output_csv=args.output,
        strategy_filter=args.strategy,
    ))


if __name__ == "__main__":
    main()
