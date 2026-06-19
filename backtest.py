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

    BASE = "https://api-t1.fyers.in/api/v3"
    # Fyers caps 1-min data at ~100 calendar days per request
    CHUNK_DAYS = 59

    def __init__(self, app_id: str, access_token: str) -> None:
        self._headers = {
            "Authorization": f"{app_id}:{access_token}",
            "Content-Type": "application/json",
        }

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
                data = r.json()
                if data.get("s") != "ok":
                    raise RuntimeError(f"Fyers history error [{symbol}]: {data}")
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

    def __init__(self, bs: BlackScholesEngine, spec: InstrumentSpec) -> None:
        self.bs   = bs
        self.spec = spec

    def build(
        self,
        spot:      float,
        vix_pct:   float,
        timestamp: datetime,
        expiry_dt: datetime,
        tte:       float,       # years
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

        for i in range(-10, 11):
            strike = atm + i * si
            if strike <= 0:
                continue

            m_put  = max(0.0, (atm - strike) / spot)
            m_call = max(0.0, (strike - atm) / spot)
            call_iv = atm_iv * np.exp(0.30 * m_call)
            put_iv  = atm_iv * np.exp(1.50 * m_put)

            if tte < 1e-6:
                continue

            call_px = max(0.05, float(self.bs.call_price(spot, strike, tte, call_iv)))
            put_px  = max(0.05, float(self.bs.put_price(spot, strike, tte, put_iv)))

            spread   = 0.004 + abs(i) * 0.002
            vol_base = max(50,    8_000 - abs(i) * 700)
            oi_base  = max(500,  80_000 - abs(i) * 7_000)
            pfx      = f"{self.spec.segment}:{self.spec.name}"

            def _make_quote(otype: str, px: float, iv: float) -> OptionQuote:
                opt = OptionType.CALL if otype == "CE" else OptionType.PUT
                return OptionQuote(
                    symbol=f"{pfx}{exp_str}{int(strike)}{otype}",
                    strike=strike,
                    expiry=exp_str,
                    option_type=opt,
                    ltp=px,
                    bid=round(px * (1 - spread), 2),
                    ask=round(px * (1 + spread), 2),
                    bid_qty=vol_base,
                    ask_qty=vol_base,
                    volume=vol_base * 10,
                    oi=oi_base,
                    iv=iv,
                    delta=float(self.bs.delta(spot, strike, tte, iv, opt)),
                    gamma=float(self.bs.gamma(spot, strike, tte, iv)),
                    theta=float(self.bs.theta(spot, strike, tte, iv, opt)),
                    vega=float(self.bs.vega(spot, strike, tte, iv)),
                )

            snap.calls[strike] = _make_quote("CE", call_px, call_iv)
            snap.puts[strike]  = _make_quote("PE", put_px,  put_iv)

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

def generate_report(
    db: TradingDatabase,
    initial_capital: float,
    output_csv: str,
) -> None:
    """Read closed trades from DB and print metrics + save CSV."""
    conn = db._get_connection()
    rows = conn.execute(
        """SELECT * FROM trades
           WHERE status IN ('CLOSED','SQUARED_OFF','EMERGENCY_EXIT')
           ORDER BY exit_time"""
    ).fetchall()

    if not rows:
        print("\n  No completed trades to report.\n")
        return

    pnls  = [r["pnl"] for r in rows if r["pnl"] is not None]
    wins  = [p for p in pnls if p > 0]
    losses= [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate  = len(wins) / max(1, len(pnls)) * 100
    avg_win   = float(np.mean(wins))   if wins   else 0.0
    avg_loss  = float(np.mean(losses)) if losses else 0.0
    pf        = sum(wins) / abs(sum(losses)) if losses and wins else float("inf")
    expectancy= (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    # Rolling max-drawdown from trade sequence
    cap = initial_capital
    peak = cap
    max_dd = 0.0
    for p in pnls:
        cap += p
        peak = max(peak, cap)
        max_dd = max(max_dd, (peak - cap) / peak * 100)

    W = 55
    print("\n" + "=" * W)
    print("  BACKTEST PERFORMANCE SUMMARY")
    print("=" * W)
    print(f"  Initial Capital :  ₹{initial_capital:>12,.0f}")
    print(f"  Final Capital   :  ₹{initial_capital + total_pnl:>12,.0f}")
    print(f"  Total P&L       :  ₹{total_pnl:>12,.0f}  ({total_pnl/initial_capital*100:+.2f}%)")
    print(f"  Total Trades    :  {len(pnls)}")
    print(f"  Win Rate        :  {win_rate:.1f}%")
    print(f"  Avg Win         :  ₹{avg_win:>10,.0f}")
    print(f"  Avg Loss        :  ₹{avg_loss:>10,.0f}")
    if avg_loss != 0:
        print(f"  Actual RR       :  {abs(avg_win / avg_loss):.2f} : 1")
    print(f"  Profit Factor   :  {pf:.2f}")
    print(f"  Expectancy/trade:  ₹{expectancy:>10,.0f}")
    print(f"  Max Drawdown    :  {max_dd:.2f}%")
    print("=" * W)

    # Per-strategy breakdown
    strat_names = sorted({r["strategy_name"] for r in rows})
    for sname in strat_names:
        s_pnls = [r["pnl"] for r in rows
                  if r["strategy_name"] == sname and r["pnl"] is not None]
        s_wins = sum(1 for p in s_pnls if p > 0)
        print(f"\n  [{sname}]")
        print(f"    Trades={len(s_pnls)}  Wins={s_wins}  "
              f"WinRate={s_wins/max(1,len(s_pnls))*100:.1f}%  "
              f"P&L=₹{sum(s_pnls):,.0f}")

    # Save CSV
    with open(output_csv, "w", newline="") as f:
        fields = [
            "strategy_name", "symbol", "option_type", "strike", "direction",
            "entry_time", "exit_time", "entry_price", "exit_price",
            "quantity", "pnl", "exit_reason",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields})
    print(f"\n  Full trade log → {output_csv}\n")


# ── Main Backtest Engine ──────────────────────────────────────────────────────

async def run_backtest(
    instrument_key: str,
    start_date:     str,
    end_date:       str,
    initial_capital: float = 500_000,
    risk_free_rate:  float = 0.065,
    output_csv:      str   = "backtest_results.csv",
) -> None:
    spec = INSTRUMENTS[instrument_key]
    print(f"\n{'='*55}")
    print(f"  Instrument : {spec.display_name}")
    print(f"  Period     : {start_date}  →  {end_date}")
    print(f"  Capital    : ₹{initial_capital:,.0f}")
    print(f"  Note: synthetic chain (spot + India VIX via BS model)")
    print(f"{'='*55}")

    app_id       = os.environ["BROKER_APP_ID"]
    access_token = os.environ["BROKER_ACCESS_TOKEN"]
    data_client  = FyersHistoryClient(app_id, access_token)

    print("  Fetching 1-min underlying candles ...", flush=True)
    candles = await data_client.fetch_candles(spec.spot_symbol, start_date, end_date)
    if not candles:
        print("  ERROR: no candle data returned. Check credentials and symbol.")
        return

    print("  Fetching India VIX (daily) ...", flush=True)
    vix_map = await data_client.fetch_daily_vix(start_date, end_date)
    print(f"  Data ready: {len(candles)} bars, {len(vix_map)} VIX days\n")

    # Core objects
    bs      = BlackScholesEngine(risk_free_rate)
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
        ),
        on_risk_event=lambda e, d: logger.warning("Risk event: %s", e.value),
    )

    cfg      = StrategyConfig()
    strat_fixed    = FixedRR13Strategy(cfg, bs, db, risk, broker)
    strat_curv     = CurvatureCreditSpreadStrategy(cfg, bs, db, risk, broker)
    strat_skew     = SkewHunterStrategy(cfg, bs, db, risk, broker)
    all_strategies = [strat_fixed, strat_curv, strat_skew]

    # Monkey-patch is_trading_window: BacktestEngine controls entry windows,
    # so we disable the internal time check (which would use real wall-clock time).
    for s in all_strategies:
        s.is_trading_window = lambda: True  # type: ignore[method-assign]

    # Time window constants
    T_OPEN        = dt_time(9, 15)
    T_STRAT_START = dt_time(10, 15)
    T_STRAT_END   = dt_time(14, 15)
    T_CURV_START  = dt_time(15, 0)
    T_CURV_END    = dt_time(15, 25)
    T_SQUAREOFF   = dt_time(15, 15)
    T_CLOSE       = dt_time(15, 30)

    # Group candles by trading date
    by_date: Dict[date, List[dict]] = defaultdict(list)
    for c in candles:
        by_date[c["timestamp"].date()].append(c)

    total_bars   = 0
    trading_days = 0

    for day in sorted(by_date.keys()):
        day_bars   = sorted(by_date[day], key=lambda b: b["timestamp"])
        vix        = vix_map.get(day, 15.0)
        expiry_dt  = nearest_expiry(
            datetime.combine(day, T_OPEN), spec.expiry_weekday
        )
        exp_str    = expiry_dt.strftime("%d%b%y").upper()
        trading_days += 1

        for bar in day_bars:
            ts       = bar["timestamp"]
            bar_time = ts.time()

            if bar_time < T_OPEN or bar_time > T_CLOSE:
                continue

            tte   = trading_tte(ts, expiry_dt)
            chain = builder.build(bar["close"], vix, ts, expiry_dt, tte)
            broker.update_prices(chain)
            total_bars += 1

            # ── Strategy entry windows ────────────────────────────────────

            if T_STRAT_START <= bar_time <= T_STRAT_END:
                for strategy in [strat_fixed, strat_skew]:
                    try:
                        sig = await strategy.evaluate(chain)
                        if sig:
                            await strategy.execute_signal(sig)
                    except Exception as e:
                        logger.debug("%s eval: %s", strategy.name, e)

            if T_CURV_START <= bar_time <= T_CURV_END:
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

        # Per-day summary (only print days with activity)
        stats = db.get_daily_stats(day.isoformat())
        if stats["total_trades"] > 0:
            print(f"  {day}  trades={stats['total_trades']:2d}  "
                  f"pnl=₹{stats['realized_pnl']:+10,.0f}")

    print(f"\n  Processed {total_bars:,} bars across {trading_days} trading days")
    generate_report(db, initial_capital, output_csv)


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
    parser.add_argument("--start",  default="2025-01-01", metavar="YYYY-MM-DD")
    parser.add_argument("--end",    default="2025-05-30", metavar="YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=500_000,
                        help="Initial capital in INR (default: 500000)")
    parser.add_argument("--output", default="backtest_results.csv",
                        help="Output CSV path")
    parser.add_argument("--list-instruments", action="store_true",
                        help="Print all supported instruments and exit")
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

    asyncio.run(run_backtest(
        instrument_key=args.instrument,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        output_csv=args.output,
    ))


if __name__ == "__main__":
    main()
