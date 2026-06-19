#!/usr/bin/env python3
"""
backtest_offline.py — Zero-credential offline strategy backtester.

Generates realistic index price paths using Geometric Brownian Motion (GBM)
with calibrated parameters, then runs all 3 strategies and prints a full
performance report. No Fyers API keys needed.

Instruments simulated:
  Nifty 50    S0 = ₹23,000   σ = 14 % p.a.
  Bank Nifty  S0 = ₹50,000   σ = 22 % p.a.
  Sensex      S0 = ₹75,000   σ = 13 % p.a.

India VIX is modelled as an Ornstein-Uhlenbeck mean-reverting process
(long-run mean = 14 %, daily vol = 1.5 %).

LIMITATIONS (vs real data):
  • SL / target checked at bar close, not intrabar (slightly optimistic WR).
  • Volume is modelled; OI uses a random walk (SkewHunter alpha1 is semi-real).
  • No NSE holiday calendar — weekends only are excluded.
  • Curvature strategy fires rarely because VIX changes are smooth in simulation.

Usage:
  python3 backtest_offline.py                        # 3 months, ₹5L capital
  python3 backtest_offline.py --months 6             # extend period
  python3 backtest_offline.py --capital 1000000      # ₹10L
  python3 backtest_offline.py --instrument banknifty # single instrument
"""

import argparse
import asyncio
import csv
import dataclasses
import sys
from collections import defaultdict
from datetime import datetime, timedelta, date, time as dt_time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── project imports ────────────────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Synthetic price-path generation
# ══════════════════════════════════════════════════════════════════════════════

BARS_PER_DAY = 375   # 9:15 AM → 3:29 PM (375 one-minute bars)


def _trading_calendar(n_months: int, start: datetime = datetime(2025, 1, 2)) -> List[date]:
    """Return a list of Mon-Fri trading dates for ~n_months."""
    days: List[date] = []
    d = start
    target = int(n_months * 21)   # ~21 trading days/month
    while len(days) < target:
        if d.weekday() < 5:
            days.append(d.date())
        d += timedelta(days=1)
    return days


def generate_price_path(
    s0: float,
    mu: float,
    sigma: float,
    trading_days: List[date],
    seed: int = 42,
) -> List[dict]:
    """
    GBM 1-minute bars with a mild intraday U-shape volatility pattern
    (higher vol at open and close — realistic for Indian markets).
    """
    rng    = np.random.default_rng(seed)
    n_bars = len(trading_days) * BARS_PER_DAY
    dt     = 1 / 252 / BARS_PER_DAY

    z          = rng.standard_normal(n_bars)
    log_ret    = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z

    # Intraday volatility U-shape: higher vol first 30 and last 30 minutes
    bar_of_day  = np.arange(n_bars) % BARS_PER_DAY
    u_shape     = 1 + 0.4 * np.exp(-bar_of_day / 40) + \
                      0.3 * np.exp(-(BARS_PER_DAY - 1 - bar_of_day) / 25)
    log_ret    *= u_shape

    prices = s0 * np.exp(np.cumsum(log_ret))

    bars: List[dict] = []
    idx = 0
    prev_close = s0
    for day in trading_days:
        for minute in range(BARS_PER_DAY):
            total_min   = 9 * 60 + 15 + minute
            hour        = total_min // 60
            min_of_hour = total_min % 60
            ts          = datetime.combine(day, dt_time(hour, min_of_hour))
            close       = float(prices[idx])

            # Simulate realistic OHLC within the bar
            bar_vol  = abs(sigma * np.sqrt(dt) * close)
            open_    = prev_close
            hi       = max(open_, close) + abs(float(rng.normal(0, bar_vol * 0.3)))
            lo       = min(open_, close) - abs(float(rng.normal(0, bar_vol * 0.3)))
            vol      = max(100, int(rng.poisson(8_000)))

            bars.append({
                "timestamp": ts,
                "open": open_, "high": hi, "low": lo, "close": close,
                "volume": vol,
            })
            prev_close = close
            idx += 1

    return bars


def generate_vix(trading_days: List[date], seed: int = 99) -> Dict[date, float]:
    """
    Ornstein-Uhlenbeck VIX path.
      κ  = 0.08  (daily mean-reversion speed)
      μ  = 14.0  (long-run mean VIX %)
      σ  = 1.5   (daily volatility of VIX)
    Occasional jump-spikes to simulate event-driven vol expansions.
    """
    rng = np.random.default_rng(seed)
    kappa, mu_v, sigma_v = 0.08, 14.0, 1.5
    vix = mu_v
    result: Dict[date, float] = {}
    for i, d in enumerate(trading_days):
        vix += kappa * (mu_v - vix) + sigma_v * float(rng.normal())
        # Rare jump spikes (~once per 40 days)
        if rng.random() < 0.025:
            vix += float(rng.uniform(3, 8))
        vix = float(np.clip(vix, 8.0, 45.0))
        result[d] = round(vix, 2)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Option chain builder with live OI variation
# ══════════════════════════════════════════════════════════════════════════════

class OfflineChainBuilder:
    """
    Builds OptionChainSnapshot using BS model with a stochastic market-sentiment
    state variable (_bias) that drives both the call/put skew slope ratio AND the
    call/put volume imbalance each bar.  This is what makes alpha signals non-trivial:

      _bias > 0  (bullish)  → OTM calls bid up, call vol > put vol
      _bias < 0  (bearish)  → OTM puts bid up, put vol > call vol

    _bias does a random walk ∈ [-3, 3] with step_size 0.04/bar.  At extreme
    values it produces the IV and flow imbalances that cross alpha thresholds.
    """

    def __init__(self, bs: BlackScholesEngine, spec: InstrumentSpec, seed: int = 7) -> None:
        self.bs     = bs
        self.spec   = spec
        self._rng   = np.random.default_rng(seed)
        self._oi:   Dict[str, float] = {}   # symbol → running OI
        self._bias: float = 0.0             # market sentiment state

    def build(
        self,
        spot:      float,
        vix_pct:   float,
        timestamp: datetime,
        expiry_dt: datetime,
        tte:       float,
    ) -> OptionChainSnapshot:
        atm_iv  = max(0.06, vix_pct / 100)
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

        # ── Stochastic sentiment state (slow random walk ∈ [-3, 3]) ─────────
        # _bias drives BOTH the IV skew shape AND the volume imbalance:
        #   +3 = maximum bullish: OTM calls become as expensive as OTM puts;
        #         call volume dominates → FixedRR α2 ↑, SkewHunter α2 ↑
        #   -3 = maximum fearful: OTM puts become very expensive;
        #         put volume dominates → SkewHunter α2 ↓ (Long Put trigger)
        self._bias += float(self._rng.normal(0, 0.04))
        self._bias  = float(np.clip(self._bias, -3.0, 3.0))

        # Slope of put/call wing (absolute IV per point OTM, calibrated to Nifty)
        # Base: ATM-100 put ≈ +3pp above ATM (0.0003/pt), ATM+100 call ≈ +0.3pp
        put_slope  = max(0.00005, 0.0003 * (1.0 - 0.18 * self._bias))
        call_slope = max(0.00005, 0.0001 * (1.0 + 0.60 * self._bias))

        # Volume imbalance: which side has more activity this bar
        call_vol_mult = max(0.15, 1.0 + 0.50 * self._bias)
        put_vol_mult  = max(0.15, 1.0 - 0.35 * self._bias)

        # ── Vectorised BS over ±25 strikes ───────────────────────────────────
        indices  = np.arange(-25, 26)
        strikes  = atm + indices * si
        valid    = strikes > 0
        strikes  = strikes[valid]
        indices  = indices[valid]

        dk_put   = np.maximum(0.0, atm - strikes)
        dk_call  = np.maximum(0.0, strikes - atm)
        c_iv_arr = np.maximum(0.04, atm_iv + call_slope * dk_call)
        p_iv_arr = np.maximum(0.04, atm_iv + put_slope  * dk_put)

        # BS price + delta in one numpy pass (no gamma/theta/vega — not used)
        c_px_arr = np.maximum(0.05, self.bs.call_price(spot, strikes, tte, c_iv_arr))
        p_px_arr = np.maximum(0.05, self.bs.put_price( spot, strikes, tte, p_iv_arr))
        c_dl_arr = self.bs.delta(spot, strikes, tte, c_iv_arr, OptionType.CALL)
        p_dl_arr = self.bs.delta(spot, strikes, tte, p_iv_arr, OptionType.PUT)

        n_vol    = self._rng.lognormal(0, 0.15, size=(len(strikes), 2))

        for idx_j, (i, strike) in enumerate(zip(indices, strikes)):
            spread  = 0.004 + abs(i) * 0.001
            v_base  = max(20, int(8_000 - abs(i) * 290))
            c_vol   = max(10, int(v_base * call_vol_mult * n_vol[idx_j, 0]))
            p_vol   = max(10, int(v_base * put_vol_mult  * n_vol[idx_j, 1]))
            oi_base = max(200, 80_000 - int(abs(i) * 2_800))

            def _oi(key: str, base: int) -> int:
                prev  = self._oi.get(key, float(base))
                swing = prev * (0.006 if abs(i) <= 3 else 0.002)
                new   = max(100.0, prev + float(self._rng.normal(0, swing)))
                self._oi[key] = new
                return int(new)

            c_oi = _oi(f"{pfx}{exp_str}{int(strike)}CE", oi_base)
            p_oi = _oi(f"{pfx}{exp_str}{int(strike)}PE", oi_base)

            c_iv = float(c_iv_arr[idx_j]); p_iv = float(p_iv_arr[idx_j])
            c_px = float(c_px_arr[idx_j]); p_px = float(p_px_arr[idx_j])

            snap.calls[float(strike)] = OptionQuote(
                symbol=f"{pfx}{exp_str}{int(strike)}CE", strike=float(strike),
                expiry=exp_str, option_type=OptionType.CALL,
                ltp=round(c_px,2), bid=round(c_px*(1-spread),2),
                ask=round(c_px*(1+spread),2),
                bid_qty=c_vol, ask_qty=c_vol, volume=c_vol*10, oi=c_oi,
                iv=c_iv, delta=float(c_dl_arr[idx_j]),
            )
            snap.puts[float(strike)] = OptionQuote(
                symbol=f"{pfx}{exp_str}{int(strike)}PE", strike=float(strike),
                expiry=exp_str, option_type=OptionType.PUT,
                ltp=round(p_px,2), bid=round(p_px*(1-spread),2),
                ask=round(p_px*(1+spread),2),
                bid_qty=p_vol, ask_qty=p_vol, volume=p_vol*10, oi=p_oi,
                iv=p_iv, delta=float(p_dl_arr[idx_j]),
            )

        return snap


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Simulated broker
# ══════════════════════════════════════════════════════════════════════════════

class BacktestBroker(BrokerGateway):
    """Fills limit orders at limit price; market orders at price ± 0.1%."""

    SLIPPAGE = 0.001

    def __init__(self) -> None:
        self._prices:  Dict[str, float] = {}
        self._next_id: int = 0
        self._client        = None
        self._session_token = "backtest"

    def update_prices(self, chain: OptionChainSnapshot) -> None:
        for q in chain.calls.values():
            self._prices[q.symbol] = q.ltp
        for q in chain.puts.values():
            self._prices[q.symbol] = q.ltp

    def price_of(self, sym: str) -> Optional[float]:
        return self._prices.get(sym)

    async def authenticate(self) -> bool:
        return True

    async def place_order(
        self, symbol, exchange, transaction_type, order_type,
        quantity, price=0.0, trigger_price=0.0,
        product_type=ProductType.INTRADAY,
    ) -> OrderResponse:
        self._next_id += 1
        oid = f"BT{self._next_id:06d}"
        if order_type == OrderType.MARKET:
            fill = self._prices.get(symbol, price)
            fill *= (1 + self.SLIPPAGE) if transaction_type == TransactionType.BUY \
                   else (1 - self.SLIPPAGE)
        else:
            fill = price
        return OrderResponse(
            success=True, order_id=oid, broker_order_id=oid,
            message=f"BT fill @ {fill:.2f}", status="FILLED",
        )

    async def modify_order(self, order_id, **_) -> OrderResponse:
        return OrderResponse(success=True, order_id=order_id)

    async def cancel_order(self, order_id) -> OrderResponse:
        return OrderResponse(success=True, order_id=order_id)

    async def get_positions(self) -> List[Position]:
        return []

    async def get_orders(self) -> List[Order]:
        return []

    async def get_quote(self, symbol: str, exchange) -> Optional[Quote]:
        px = self._prices.get(symbol)
        if px is None:
            return None
        return Quote(
            symbol=symbol, ltp=px,
            bid_price=px * 0.999, ask_price=px * 1.001,
            bid_qty=500, ask_qty=500, volume=10_000, oi=100_000,
            timestamp=datetime.now().isoformat(),
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def nearest_expiry(ref: datetime, weekday: int) -> datetime:
    days = (weekday - ref.weekday()) % 7
    if days == 0 and ref.time() >= dt_time(15, 30):
        days = 7
    return ref + timedelta(days=days)


def trading_tte(now: datetime, expiry_dt: datetime) -> float:
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
            days   += max(0.0, 1.0 - elapsed / session)
    return max(1 / 252, days / 252)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Performance report
# ══════════════════════════════════════════════════════════════════════════════

def _report(
    db: TradingDatabase,
    label: str,
    initial_capital: float,
    output_csv: Optional[str] = None,
) -> dict:
    """Print a formatted report and return summary dict."""
    conn = db._get_connection()
    rows = conn.execute(
        "SELECT * FROM trades "
        "WHERE status IN ('CLOSED','SQUARED_OFF','EMERGENCY_EXIT') "
        "ORDER BY exit_time"
    ).fetchall()

    pnls   = [r["pnl"] for r in rows if r["pnl"] is not None]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    if not pnls:
        print(f"\n  [{label}]  No closed trades.\n")
        return {}

    total_pnl = sum(pnls)
    wr        = len(wins) / len(pnls) * 100
    avg_w     = float(np.mean(wins))   if wins   else 0.0
    avg_l     = float(np.mean(losses)) if losses else 0.0
    pf        = sum(wins) / abs(sum(losses)) if wins and losses else float("inf")
    exp_val   = (wr / 100 * avg_w) + ((1 - wr / 100) * avg_l)

    # Max drawdown from trade sequence
    cap = peak = initial_capital
    max_dd = 0.0
    for p in pnls:
        cap  += p
        peak  = max(peak, cap)
        max_dd = max(max_dd, (peak - cap) / peak * 100)

    W = 56
    bar = "═" * W
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    print(f"  Capital         ₹{initial_capital:>12,.0f}  →  ₹{initial_capital+total_pnl:>12,.0f}")
    print(f"  Total P&L     : ₹{total_pnl:>10,.0f}  ({total_pnl/initial_capital*100:+.2f} %)")
    print(f"  Total Trades  : {len(pnls)}")
    print(f"  Win Rate      : {wr:.1f} %")
    print(f"  Avg Win       : ₹{avg_w:>9,.0f}    Avg Loss : ₹{avg_l:>9,.0f}")
    if avg_l != 0:
        print(f"  Actual RR     : {abs(avg_w/avg_l):.2f} : 1")
    print(f"  Profit Factor : {pf:.2f}")
    print(f"  Expectancy    : ₹{exp_val:>9,.0f} / trade")
    print(f"  Max Drawdown  : {max_dd:.2f} %")
    print(bar)

    # Per-strategy breakdown
    strats = sorted({r["strategy_name"] for r in rows})
    for s in strats:
        sp  = [r["pnl"] for r in rows if r["strategy_name"] == s and r["pnl"] is not None]
        sw  = sum(1 for x in sp if x > 0)
        print(f"  [{s}]  trades={len(sp):3d}  wins={sw:3d}  "
              f"wr={sw/max(1,len(sp))*100:5.1f}%  pnl=₹{sum(sp):>9,.0f}")
    print()

    if output_csv:
        with open(output_csv, "w", newline="") as f:
            fields = [
                "strategy_name","symbol","option_type","strike","direction",
                "entry_time","exit_time","entry_price","exit_price",
                "quantity","pnl","exit_reason",
            ]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in fields})
        print(f"  Trade log → {output_csv}\n")

    return {
        "label": label, "total_pnl": total_pnl,
        "trades": len(pnls), "win_rate": wr,
        "profit_factor": pf, "max_dd": max_dd,
        "return_pct": total_pnl / initial_capital * 100,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Core simulation loop
# ══════════════════════════════════════════════════════════════════════════════

async def _run_one(
    spec:            InstrumentSpec,
    trading_days:    List[date],
    vix_map:         Dict[date, float],
    price_bars:      List[dict],
    initial_capital: float,
    risk_free_rate:  float,
    output_csv:      Optional[str],
    seed:            int,
) -> dict:
    bs      = BlackScholesEngine(risk_free_rate)
    builder = OfflineChainBuilder(bs, spec, seed=seed)
    broker  = BacktestBroker()
    db      = TradingDatabase(":memory:")

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
        ),
        on_risk_event=lambda e, d: None,
    )

    # In synthetic mode signals are weaker than live (simplified IV model, no
    # real order flow).  Thresholds are lowered to ~1.0 sigma vs live ~1.5 sigma
    # so the demo generates trades.  Recalibrate with real data before going live.
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
    strat_fixed    = FixedRR13Strategy(cfg, bs, db, risk, broker)
    strat_curv     = CurvatureCreditSpreadStrategy(cfg, bs, db, risk, broker)
    strat_skew     = SkewHunterStrategy(cfg, bs, db, risk, broker)
    all_strategies = [strat_fixed, strat_curv, strat_skew]

    # Disable internal time checks — BacktestEngine controls entry windows
    for s in all_strategies:
        s.is_trading_window  = lambda: True        # type: ignore[method-assign]
    strat_curv.is_entry_window = lambda: True      # type: ignore[method-assign]

    T_STRAT_START = dt_time(10, 15)
    T_STRAT_END   = dt_time(14, 15)
    T_CURV_START  = dt_time(15,  0)
    T_CURV_END    = dt_time(15, 25)
    T_SQUAREOFF   = dt_time(15, 15)
    T_CLOSE       = dt_time(15, 30)

    by_date: Dict[date, List[dict]] = defaultdict(list)
    for bar in price_bars:
        by_date[bar["timestamp"].date()].append(bar)

    total_bars = 0
    for day in trading_days:
        day_bars  = sorted(by_date.get(day, []), key=lambda b: b["timestamp"])
        vix       = vix_map.get(day, 14.0)
        expiry_dt = nearest_expiry(
            datetime.combine(day, dt_time(9, 15)), spec.expiry_weekday
        )

        for bar in day_bars:
            ts       = bar["timestamp"]
            bar_time = ts.time()
            if bar_time > T_CLOSE:
                continue

            tte   = trading_tte(ts, expiry_dt)
            chain = builder.build(bar["close"], vix, ts, expiry_dt, tte)
            broker.update_prices(chain)
            total_bars += 1

            # ── Entry windows ─────────────────────────────────────────────
            if T_STRAT_START <= bar_time <= T_STRAT_END:
                for s in [strat_fixed, strat_skew]:
                    try:
                        sig = await s.evaluate(chain)
                        if sig:
                            await s.execute_signal(sig)
                    except Exception:
                        pass

            if T_CURV_START <= bar_time <= T_CURV_END:
                try:
                    sig = await strat_curv.evaluate(chain)
                    if sig:
                        await strat_curv.execute_signal(sig)
                except Exception:
                    pass

            # ── Position management every bar ────────────────────────────
            for s in all_strategies:
                try:
                    await s.manage_positions()
                except Exception:
                    pass

            # ── EOD square-off for INTRADAY at 3:15 PM ───────────────────
            if bar_time >= T_SQUAREOFF:
                for trade in db.get_active_trades():
                    if trade.product_type != "INTRADAY":
                        continue
                    px = broker.price_of(trade.symbol) or trade.entry_price
                    db.update_trade_status(
                        trade.trade_id, TradeStatus.SQUARED_OFF,
                        exit_price=px, exit_reason="EOD_SQUAREOFF",
                    )
                    dm  = 1 if trade.direction == "BUY" else -1
                    pnl = (px - trade.entry_price) * trade.quantity * dm
                    risk.remove_position(trade.trade_id, pnl)

    label = f"{spec.display_name}  ({len(trading_days)} trading days)"
    return _report(db, label, initial_capital, output_csv)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Comparative summary
# ══════════════════════════════════════════════════════════════════════════════

def _comparative_table(results: List[dict]) -> None:
    if not results:
        return
    W = 70
    print("\n" + "═" * W)
    print("  COMPARATIVE SUMMARY")
    print("═" * W)
    print(f"  {'Instrument':<22} {'Trades':>6} {'WinRate':>8} "
          f"{'P&L':>12} {'Return':>8} {'MaxDD':>7} {'PF':>6}")
    print("  " + "─" * (W - 2))
    for r in results:
        if not r:
            continue
        print(f"  {r['label'][:22]:<22} {r['trades']:>6} "
              f"{r['win_rate']:>7.1f}% "
              f"₹{r['total_pnl']:>10,.0f} "
              f"{r['return_pct']:>+7.2f}% "
              f"{r['max_dd']:>6.2f}% "
              f"{r['profit_factor']:>6.2f}")
    print("═" * W)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Entry point
# ══════════════════════════════════════════════════════════════════════════════

INSTRUMENT_PARAMS = {
    #            s0        mu    sigma  seed
    "nifty":   (23_000,  0.12, 0.14,  42),
    "banknifty":(50_000, 0.10, 0.22,  77),
    "sensex":  (75_000,  0.12, 0.13,  13),
    "finnifty": (22_000, 0.11, 0.17,  55),
}

async def _main_async(args) -> None:
    n_months        = args.months
    initial_capital = args.capital
    instruments     = (
        [args.instrument] if args.instrument != "all"
        else ["nifty", "banknifty", "sensex"]
    )
    risk_free_rate = 0.065  # RBI repo rate proxy

    trading_days = _trading_calendar(n_months)
    vix_map      = generate_vix(trading_days)

    print(f"\n{'━'*60}")
    print(f"  OFFLINE STRATEGY BACKTEST")
    print(f"  Period  : {trading_days[0]}  →  {trading_days[-1]}")
    print(f"  Days    : {len(trading_days)} trading days")
    print(f"  Capital : ₹{initial_capital:,.0f}")
    print(f"  Mode    : Synthetic GBM + OU-VIX  (no API keys needed)")
    print(f"{'━'*60}")

    results = []
    for ikey in instruments:
        if ikey not in INSTRUMENTS:
            print(f"  WARNING: '{ikey}' not in INSTRUMENTS — skipping")
            continue
        spec = INSTRUMENTS[ikey]
        s0, mu, sigma, seed = INSTRUMENT_PARAMS.get(
            ikey, (23_000, 0.12, 0.14, 42)
        )
        print(f"\n  Generating {spec.display_name} price path "
              f"(S₀=₹{s0:,}  σ={sigma*100:.0f}%  μ={mu*100:.0f}%) ...")
        bars = generate_price_path(s0, mu, sigma, trading_days, seed=seed)

        print(f"  Running strategies on {len(bars):,} bars ...")
        csv_path = f"backtest_{ikey}.csv" if not args.no_csv else None
        result   = await _run_one(
            spec=spec,
            trading_days=trading_days,
            vix_map=vix_map,
            price_bars=bars,
            initial_capital=initial_capital,
            risk_free_rate=risk_free_rate,
            output_csv=csv_path,
            seed=seed,
        )
        results.append(result)

    if len(results) > 1:
        _comparative_table(results)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Offline strategy backtest (no API credentials required)"
    )
    p.add_argument(
        "--instrument", default="all",
        choices=["all"] + list(INSTRUMENT_PARAMS.keys()),
        help="Which instrument to run (default: all three)",
    )
    p.add_argument("--months",   type=int,   default=3,
                   help="Simulation length in months (default: 3)")
    p.add_argument("--capital",  type=float, default=500_000,
                   help="Starting capital in INR (default: 500000)")
    p.add_argument("--no-csv",   action="store_true",
                   help="Skip writing CSV trade log")
    args = p.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
