"""
ORB Signal Quality Test — Futures-style P&L (no theta drag).

Why: Options backtests show negative results because:
  1. Debit spread theta decay eats into intraday profits
  2. Spread is capped at 100pts but underlying needs 200+ to make spread worth buying
  3. Stop fires at underlying re-entering range (tight, 8-9pt risk), but option
     loses more than 8-9pts of delta when ATM → OTM

This script tests the SIGNAL QUALITY in isolation using futures-like delta P&L:
  P&L = lots × lot_size × (exit_price - entry_price) × ±1

If signals have positive expected value here, we tune the options execution separately.
If signals are random here, the strategy has no edge at all.

Usage:
  python3 backtest_orb_delta.py --instrument NIFTY --start 2022-01-01 --end 2024-12-31
  python3 backtest_orb_delta.py --instrument NIFTY --start 2022-01-01 --end 2024-12-31 --stop_atr 1.5 --target_rr 3
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from datetime import date, time as dtime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("orb_delta")

# ── Instrument specs ──────────────────────────────────────────────────────────

LOT_SIZES = {
    "NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 40,
    "SENSEX": 10, "BANKEX": 15,
}

EXPIRY_WEEKDAYS = {
    "NIFTY": 1, "BANKNIFTY": 2, "FINNIFTY": 1,
    "SENSEX": 3, "BANKEX": 0,
}

SLIPPAGE = {
    "NIFTY": 1.0, "BANKNIFTY": 3.0, "FINNIFTY": 2.0,
    "SENSEX": 2.0, "BANKEX": 2.0,
}

# ── Trade dataclass ────────────────────────────────────────────────────────────

class DeltaTrade:
    def __init__(self, date, direction, entry, stop, target, range_h, range_l, range_pct):
        self.date      = date
        self.direction = direction   # 1 = LONG, -1 = SHORT
        self.entry     = entry
        self.stop      = stop
        self.target    = target
        self.range_h   = range_h
        self.range_l   = range_l
        self.range_pct = range_pct
        self.exit_price = None
        self.exit_reason = None
        self.pnl        = 0.0

    @property
    def risk_pts(self):
        return abs(self.entry - self.stop)

    @property
    def reward_pts(self):
        return abs(self.target - self.entry)


# ── Main backtest ──────────────────────────────────────────────────────────────

def run(
    instrument:    str,
    start:         date,
    end:           date,
    # Range params
    min_range_pct: float = 0.002,
    max_range_pct: float = 0.015,
    buffer_pct:    float = 0.0003,   # tighter buffer for delta test
    # Exit params
    stop_at_range: bool  = True,     # stop = back inside range
    stop_atr_mult: float = 0.0,      # if >0: stop = entry ± N×ATR instead
    target_rr:     float = 2.0,
    hard_exit_time: str  = "15:15",
    skip_expiry:   bool  = True,
    # Capital
    capital:       float = 200_000,
    lots:          int   = 1,
) -> None:
    key      = instrument.upper()
    lot_size = LOT_SIZES.get(key, 75)
    exp_wd   = EXPIRY_WEEKDAYS.get(key, 3)
    slip     = SLIPPAGE.get(key, 1.0)

    cache = f"data/cache/{key}_1min.csv"
    df    = pd.read_csv(cache, index_col="datetime", parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")

    lo = pd.Timestamp(start, tz="Asia/Kolkata")
    hi = pd.Timestamp(end,   tz="Asia/Kolkata") + pd.Timedelta(days=1)
    df = df[(df.index >= lo) & (df.index < hi)].copy()
    df["atr14"] = _atr(df, 14)

    sq_h, sq_m = [int(x) for x in hard_exit_time.split(":")]

    trades: List[DeltaTrade] = []
    nav    = capital
    equity: List[Tuple[date, float]] = []

    for day, grp in df.groupby(df.index.date):
        if skip_expiry and day.weekday() == exp_wd:
            continue

        rng_bars = grp.between_time("09:15", "09:29")
        if len(rng_bars) < 10:
            continue

        rng_h = float(rng_bars["high"].max())
        rng_l = float(rng_bars["low"].min())
        rng_close = float(rng_bars["close"].iloc[-1])
        rng_pts   = rng_h - rng_l
        rng_pct   = rng_pts / rng_close if rng_close > 0 else 0.0

        if not (min_range_pct <= rng_pct <= max_range_pct):
            continue

        # Compute ATR at range end
        day_atr = float(grp["atr14"].between_time("09:29", "09:29").iloc[-1]
                        if len(grp.between_time("09:29", "09:29")) > 0 else 0)

        bull_trig = rng_h + buffer_pct * rng_close
        bear_trig = rng_l - buffer_pct * rng_close

        trade: Optional[DeltaTrade] = None
        signal_bar: Optional[pd.Series] = None

        post_bars = grp.between_time("09:30", "11:30")
        for ts, row in post_bars.iterrows():
            if row["high"] > bull_trig:
                entry = bull_trig + slip
                direction = 1
                if stop_atr_mult > 0 and day_atr > 0:
                    stop  = entry - stop_atr_mult * day_atr
                else:
                    stop  = rng_h - slip   # just inside range
                target = entry + target_rr * abs(entry - stop)
                trade = DeltaTrade(day, direction, entry, stop, target, rng_h, rng_l, rng_pct)
                signal_bar = row
                break
            elif row["low"] < bear_trig:
                entry = bear_trig - slip
                direction = -1
                if stop_atr_mult > 0 and day_atr > 0:
                    stop  = entry + stop_atr_mult * day_atr
                else:
                    stop  = rng_l + slip   # just inside range
                target = entry - target_rr * abs(entry - stop)
                trade = DeltaTrade(day, direction, entry, stop, target, rng_h, rng_l, rng_pct)
                signal_bar = row
                break

        if trade is None:
            equity.append((day, nav))
            continue

        # ── Simulate trade execution ───────────────────────────────────────────
        all_remaining = grp[grp.index > signal_bar.name]

        for ts, row in all_remaining.iterrows():
            t = ts.time()
            if t >= dtime(sq_h, sq_m):
                trade.exit_price  = float(row["open"])
                trade.exit_reason = "time_stop"
                break

            if trade.direction == 1:
                if row["low"] <= trade.stop:
                    trade.exit_price  = trade.stop - slip
                    trade.exit_reason = "stop_loss"
                    break
                if row["high"] >= trade.target:
                    trade.exit_price  = trade.target + slip
                    trade.exit_reason = "target_hit"
                    break
            else:
                if row["high"] >= trade.stop:
                    trade.exit_price  = trade.stop + slip
                    trade.exit_reason = "stop_loss"
                    break
                if row["low"] <= trade.target:
                    trade.exit_price  = trade.target - slip
                    trade.exit_reason = "target_hit"
                    break

        if trade.exit_price is None:
            last = grp.iloc[-1]
            trade.exit_price  = float(last["close"])
            trade.exit_reason = "end_of_day"

        trade.pnl = trade.direction * (trade.exit_price - trade.entry) * lots * lot_size
        nav += trade.pnl
        trades.append(trade)
        equity.append((day, nav))

    # ── Report ─────────────────────────────────────────────────────────────────
    print_report(trades, equity, instrument, start, end, capital, target_rr)


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(high-low), (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def print_report(trades, equity, instrument, start, end, capital, target_rr):
    print("\n" + "=" * 65)
    print(f"  ORB SIGNAL QUALITY — {instrument} FUTURES P&L  [{start} → {end}]")
    print(f"  Target R:R = {target_rr}:1  (no theta drag)")
    print("=" * 65)

    if not trades:
        print("  No trades.")
        return

    n = len(trades)
    wins   = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    wr     = len(wins) / n
    aw     = np.mean([t.pnl for t in wins])  if wins   else 0
    al     = np.mean([t.pnl for t in losses]) if losses else 0
    gw     = sum(t.pnl for t in wins)
    gl     = abs(sum(t.pnl for t in losses))
    pf     = gw / gl if gl > 0 else float("inf")
    exp    = np.mean([t.pnl for t in trades])

    print(f"  Trades:         {n}")
    print(f"  Win Rate:       {wr:.1%}")
    print(f"  Avg Win:        ₹{aw:,.0f}  ({np.mean([t.reward_pts for t in wins]):.1f} pts)" if wins else "  Avg Win:   N/A")
    print(f"  Avg Loss:       ₹{al:,.0f}  ({np.mean([abs(t.risk_pts) for t in losses]):.1f} pts)" if losses else "  Avg Loss:  N/A")
    print(f"  Profit Factor:  {pf:.2f}")
    print(f"  Expectancy:     ₹{exp:,.0f}  per trade")

    if equity:
        eq_d, eq_v = zip(*equity)
        eq_s = pd.Series(eq_v, index=pd.DatetimeIndex(eq_d))
        peak  = eq_s.cummax()
        dd    = float((eq_s / peak - 1).min())
        td    = (pd.Timestamp(end) - pd.Timestamp(start)).days
        final = eq_s.iloc[-1]
        cagr  = float((final / capital) ** (365 / max(1, td)) - 1)
        vol   = float(eq_s.pct_change().dropna().std() * (252 ** 0.5))
        sharpe = (cagr - 0.065) / vol if vol > 0 else 0
        print(f"\n  Final NAV:      ₹{final:,.0f}  (started ₹{capital:,.0f})")
        print(f"  CAGR:           {cagr:.1%}")
        print(f"  Sharpe:         {sharpe:.2f}")
        print(f"  Max Drawdown:   {dd:.1%}")

    # Exit breakdown
    from collections import Counter
    reasons = Counter(t.exit_reason for t in trades)
    print("\n  Exit breakdown:")
    for r, cnt in reasons.most_common():
        avg_pnl = np.mean([t.pnl for t in trades if t.exit_reason == r])
        print(f"    {r:20s} {cnt:4d}  avg ₹{avg_pnl:,.0f}")

    # Direction breakdown
    longs  = [t for t in trades if t.direction ==  1]
    shorts = [t for t in trades if t.direction == -1]
    print(f"\n  Long:   {len(longs):3d} trades  WR={len([t for t in longs if t.pnl>0])/max(1,len(longs)):.1%}")
    print(f"  Short:  {len(shorts):3d} trades  WR={len([t for t in shorts if t.pnl>0])/max(1,len(shorts)):.1%}")

    # Monthly P&L
    monthly: dict = defaultdict(float)
    for t in trades:
        monthly[t.date.strftime("%Y-%m")] += t.pnl
    if monthly:
        print("\n  Monthly P&L:")
        ym_sorted = sorted(monthly)
        for i, ym in enumerate(ym_sorted):
            sign = "+" if monthly[ym] >= 0 else ""
            print(f"    {ym}  {sign}₹{monthly[ym]:,.0f}")

    print("=" * 65)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ORB Signal Quality (Futures P&L)")
    parser.add_argument("--instrument", default="NIFTY",
                        choices=list(LOT_SIZES))
    parser.add_argument("--start",       default="2022-01-01")
    parser.add_argument("--end",         default="2024-12-31")
    parser.add_argument("--min_range",   type=float, default=0.002)
    parser.add_argument("--max_range",   type=float, default=0.015)
    parser.add_argument("--buffer",      type=float, default=0.0003)
    parser.add_argument("--target_rr",   type=float, default=2.0)
    parser.add_argument("--stop_atr",    type=float, default=0.0,
                        help="If >0: stop = entry ± N×ATR. Default: stop=range edge")
    parser.add_argument("--capital",     type=float, default=200_000)
    parser.add_argument("--lots",        type=int,   default=1)
    args = parser.parse_args()

    run(
        instrument    = args.instrument,
        start         = date.fromisoformat(args.start),
        end           = date.fromisoformat(args.end),
        min_range_pct = args.min_range,
        max_range_pct = args.max_range,
        buffer_pct    = args.buffer,
        target_rr     = args.target_rr,
        stop_atr_mult = args.stop_atr,
        capital       = args.capital,
        lots          = args.lots,
    )
