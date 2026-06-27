"""
Strategy F — MCX Commodity Donchian Trend (Turtle System modernized)

Thesis: MCX commodities — Gold, Crude Oil, Natural Gas, Silver — exhibit multi-day
trending behavior driven by global supply/demand fundamentals, USD/INR, and festival/
seasonal cycles. The classic Turtle Trading system (20-period Donchian channel breakout)
captures these trends with a systematic approach. Indian commodity markets are less
algorithmically efficient than US futures, giving the system a longer exploitable edge.

Specific India edges:
  - MCX Gold: wedding/festival demand cycles (Oct–Dec, Apr–May) create predictable rallies
  - MCX Crude Oil: INR weakens during oil spikes → amplified returns for Indian traders
  - MCX Natural Gas: monsoon-correlated power demand; winter = structural bull
  - Gold-Silver ratio: oscillates between 70–90 in India, clear mean-reversion patterns

Signal: 20-bar Donchian channel on 15-minute bars (each bar = 15 min of MCX session)
  - Buy:  close > highest close of last 20 bars → NEW HIGH BREAKOUT
  - Sell: close < lowest  close of last 20 bars → NEW LOW BREAKDOWN
  - Exit: 10-bar reverse Donchian (tighter channel for exit)
  - Stop: 2 × ATR(14) from entry (Turtle system stop)

Capital management:
  - 1% capital at risk per trade (classic turtle sizing)
  - Position in lots = (1% × capital) / (2 × ATR × lot_size × lot_value)
  - Max 4 concurrent positions across all MCX instruments

Standalone backtester — uses MCX daily OHLCV (works offline once data is downloaded).

Usage:
  python -m algo_platform.strategies.mcx_trend --instrument GOLD --start 2020-01-01
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("platform.strategies.mcx_trend")

# ── Instrument specs ───────────────────────────────────────────────────────────

MCX_SPECS = {
    #          lot_size  lot_unit  tick  margin_pct
    "GOLD":     (100,    "grams",  1.0,  0.05),   # 100g lot, ₹1 tick
    "GOLDMINI": (10,     "grams",  1.0,  0.05),   # 10g mini lot
    "SILVER":   (30,     "kg",     1.0,  0.05),   # 30 kg lot
    "CRUDEOIL": (100,    "bbl",    1.0,  0.05),   # 100 barrels
    "NATURALGAS":(1250,  "mmbtu",  0.10, 0.06),   # 1250 mmbtu
    "COPPER":   (2500,   "kg",     0.05, 0.05),   # 2500 kg
    "ZINC":     (5000,   "kg",     0.05, 0.05),   # 5000 kg
    "ALUMINIUM":(5000,   "kg",     0.05, 0.05),
    "NICKEL":   (250,    "kg",     0.10, 0.05),
    "LEAD":     (5000,   "kg",     0.05, 0.05),
}


@dataclass
class MCXTrade:
    instrument:  str
    direction:   str       # "LONG" or "SHORT"
    entry_date:  date
    entry_price: float
    exit_date:   Optional[date]
    exit_price:  Optional[float]
    stop_price:  float
    lots:        int
    lot_size:    int
    pnl:         float = 0.0
    exit_reason: str   = ""


@dataclass
class MCXBacktestResult:
    trades:    List[MCXTrade]
    equity:    pd.Series
    sharpe:    float
    cagr:      float
    max_dd:    float
    win_rate:  float
    pf:        float        # profit factor
    avg_win:   float
    avg_loss:  float


class MCXTrendBacktester:
    """
    Donchian Channel trend-following backtest on MCX commodity daily data.

    Usage
    -----
    bt = MCXTrendBacktester(capital=200_000, cache_dir="data/cache")
    result = bt.run("GOLD", date(2020, 1, 1), date(2026, 6, 1))
    bt.print_report(result, "GOLD")
    """

    def __init__(
        self,
        capital:      float = 200_000,
        cache_dir:    str   = "data/cache",
        risk_per_trade: float = 0.01,   # 1% of capital per trade
        entry_period: int   = 20,       # Donchian entry: 20-bar high/low
        exit_period:  int   = 10,       # Donchian exit: 10-bar reverse channel
        atr_period:   int   = 14,
        atr_stop_mult:float = 2.0,      # stop = 2 × ATR
        max_positions:int   = 4,
        brokerage_per_lot: float = 30.0, # ₹30/lot round-trip MCX
        slippage_pct:     float = 0.001, # 0.1% slippage
    ) -> None:
        self._capital       = capital
        self._cache         = Path(cache_dir)
        self._risk_per_trade= risk_per_trade
        self._entry_period  = entry_period
        self._exit_period   = exit_period
        self._atr_period    = atr_period
        self._atr_stop_mult = atr_stop_mult
        self._max_positions = max_positions
        self._brokerage     = brokerage_per_lot
        self._slippage      = slippage_pct

    # ── Data ──────────────────────────────────────────────────────────────────

    def _load(self, instrument: str) -> pd.DataFrame:
        key = instrument.upper()
        # Try daily cache first
        for suffix in ["_daily.csv", "_1min.csv"]:
            path = self._cache / f"{key}{suffix}"
            if path.exists():
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df.index = pd.DatetimeIndex(df.index).tz_localize(None)
                if suffix == "_1min.csv":
                    df = df.resample("D").agg({
                        "open":  "first", "high": "max",
                        "low":   "min",   "close": "last",
                        "volume":"sum",
                    }).dropna()
                return df[["open", "high", "low", "close", "volume"]]
        raise FileNotFoundError(
            f"No data for {instrument}. Download with:\n"
            f"  python download_mcx.py --instrument {instrument}"
        )

    # ── Indicators ────────────────────────────────────────────────────────────

    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def _donchian(df: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series]:
        high_n = df["high"].rolling(period).max().shift(1)   # shift to avoid lookahead
        low_n  = df["low"].rolling(period).min().shift(1)
        return high_n, low_n

    # ── Backtest ──────────────────────────────────────────────────────────────

    def run(
        self,
        instrument: str,
        start:      date,
        end:        date,
    ) -> MCXBacktestResult:
        key = instrument.upper()
        spec = MCX_SPECS.get(key)
        if spec is None:
            raise ValueError(f"Unknown MCX instrument: {key}. Valid: {sorted(MCX_SPECS)}")

        lot_size, lot_unit, tick, margin_pct = spec

        df = self._load(key)
        df = df[(df.index.date >= start) & (df.index.date <= end)].copy()

        if len(df) < self._entry_period + self._atr_period + 2:
            raise ValueError(f"Insufficient data for {key} ({len(df)} bars).")

        # Compute indicators
        df["atr"]       = self._atr(df, self._atr_period)
        df["entry_hi"], df["entry_lo"] = self._donchian(df, self._entry_period)
        df["exit_hi"],  df["exit_lo"]  = self._donchian(df, self._exit_period)

        nav    = self._capital
        equity_curve: List[Tuple[date, float]] = []
        trades: List[MCXTrade]  = []
        open_trades: List[MCXTrade] = []

        for i in range(self._entry_period + self._atr_period, len(df)):
            row  = df.iloc[i]
            prev = df.iloc[i - 1]
            day  = df.index[i].date()

            # ── Check exits ──────────────────────────────────────────────────
            still_open = []
            for ot in open_trades:
                exit_price = None
                reason     = ""

                if ot.direction == "LONG":
                    if row["low"] <= ot.stop_price:
                        exit_price = ot.stop_price * (1 - self._slippage)
                        reason = "atr_stop"
                    elif prev["close"] < prev["exit_lo"]:
                        exit_price = row["open"] * (1 - self._slippage)
                        reason = "donchian_exit"
                else:
                    if row["high"] >= ot.stop_price:
                        exit_price = ot.stop_price * (1 + self._slippage)
                        reason = "atr_stop"
                    elif prev["close"] > prev["exit_hi"]:
                        exit_price = row["open"] * (1 + self._slippage)
                        reason = "donchian_exit"

                if exit_price is not None:
                    sign = 1 if ot.direction == "LONG" else -1
                    gross_pnl = sign * (exit_price - ot.entry_price) * ot.lots * lot_size
                    cost = self._brokerage * ot.lots * 2   # entry + exit
                    ot.pnl         = gross_pnl - cost
                    ot.exit_date   = day
                    ot.exit_price  = exit_price
                    ot.exit_reason = reason
                    nav += ot.pnl
                    trades.append(ot)
                else:
                    still_open.append(ot)

            open_trades = still_open

            # ── Generate new entry ───────────────────────────────────────────
            if len(open_trades) < self._max_positions:
                atr = row["atr"]
                if atr <= 0:
                    equity_curve.append((day, nav))
                    continue

                already_long  = any(t.direction == "LONG"  for t in open_trades if t.instrument == key)
                already_short = any(t.direction == "SHORT" for t in open_trades if t.instrument == key)

                new_trade: Optional[MCXTrade] = None

                if not already_long and prev["close"] > prev["entry_hi"]:
                    entry_px   = row["open"] * (1 + self._slippage)
                    stop_px    = entry_px - self._atr_stop_mult * atr
                    risk_per_lot = (entry_px - stop_px) * lot_size
                    lots = max(1, int((self._risk_per_trade * nav) / max(risk_per_lot, 1)))

                    new_trade = MCXTrade(
                        instrument  = key,
                        direction   = "LONG",
                        entry_date  = day,
                        entry_price = entry_px,
                        exit_date   = None,
                        exit_price  = None,
                        stop_price  = stop_px,
                        lots        = lots,
                        lot_size    = lot_size,
                    )

                elif not already_short and prev["close"] < prev["entry_lo"]:
                    entry_px   = row["open"] * (1 - self._slippage)
                    stop_px    = entry_px + self._atr_stop_mult * atr
                    risk_per_lot = (stop_px - entry_px) * lot_size
                    lots = max(1, int((self._risk_per_trade * nav) / max(risk_per_lot, 1)))

                    new_trade = MCXTrade(
                        instrument  = key,
                        direction   = "SHORT",
                        entry_date  = day,
                        entry_price = entry_px,
                        exit_date   = None,
                        exit_price  = None,
                        stop_price  = stop_px,
                        lots        = lots,
                        lot_size    = lot_size,
                    )

                if new_trade is not None:
                    entry_cost = self._brokerage * new_trade.lots
                    nav -= entry_cost
                    open_trades.append(new_trade)

            equity_curve.append((day, nav))

        # Force-close open trades at last price
        last_row = df.iloc[-1]
        for ot in open_trades:
            sign = 1 if ot.direction == "LONG" else -1
            exit_px  = last_row["close"] * (1 - sign * self._slippage)
            gross_pnl = sign * (exit_px - ot.entry_price) * ot.lots * lot_size
            cost = self._brokerage * ot.lots * 2
            ot.pnl         = gross_pnl - cost
            ot.exit_date   = df.index[-1].date()
            ot.exit_price  = exit_px
            ot.exit_reason = "end_of_data"
            nav += ot.pnl
            trades.append(ot)

        # Build equity series
        ec_dates, ec_vals = zip(*equity_curve) if equity_curve else ([], [])
        equity = pd.Series(ec_vals, index=pd.DatetimeIndex(ec_dates))

        # Performance metrics
        daily_rets  = equity.pct_change().dropna()
        n_trades    = len(trades)
        wins        = [t.pnl for t in trades if t.pnl > 0]
        losses      = [t.pnl for t in trades if t.pnl <= 0]
        win_rate    = len(wins) / n_trades if n_trades > 0 else 0.0
        avg_win     = float(np.mean(wins))  if wins   else 0.0
        avg_loss    = float(np.mean(losses)) if losses else 0.0
        gross_wins  = sum(wins)
        gross_loss  = abs(sum(losses))
        pf          = gross_wins / gross_loss if gross_loss > 0 else float("inf")

        peak   = equity.cummax()
        max_dd = float((equity / peak - 1).min())
        total_days = (pd.Timestamp(end) - pd.Timestamp(start)).days
        cagr   = float((nav / self._capital) ** (365 / max(1, total_days)) - 1)
        ann_vol= float(daily_rets.std() * np.sqrt(252)) if len(daily_rets) > 1 else 0.01
        sharpe = (cagr - 0.065) / ann_vol if ann_vol > 0 else 0.0

        return MCXBacktestResult(
            trades   = trades,
            equity   = equity,
            sharpe   = sharpe,
            cagr     = cagr,
            max_dd   = max_dd,
            win_rate = win_rate,
            pf       = pf,
            avg_win  = avg_win,
            avg_loss = avg_loss,
        )

    def print_report(self, result: MCXBacktestResult, instrument: str) -> None:
        print("\n" + "=" * 60)
        print(f"MCX DONCHIAN TREND — {instrument} — BACKTEST REPORT")
        print("=" * 60)
        print(f"  Trades:       {len(result.trades)}")
        print(f"  Win Rate:     {result.win_rate:.1%}")
        print(f"  Avg Win:      ₹{result.avg_win:,.0f}")
        print(f"  Avg Loss:     ₹{result.avg_loss:,.0f}")
        print(f"  Profit Factor:{result.pf:.2f}")
        print(f"  CAGR:         {result.cagr:.1%}")
        print(f"  Sharpe:       {result.sharpe:.2f}")
        print(f"  Max Drawdown: {result.max_dd:.1%}")
        print(f"  Final NAV:    ₹{result.equity.iloc[-1]:,.0f}")
        print("=" * 60)

        long_trades  = [t for t in result.trades if t.direction == "LONG"]
        short_trades = [t for t in result.trades if t.direction == "SHORT"]
        print(f"  Long trades:  {len(long_trades)}")
        print(f"  Short trades: {len(short_trades)}")

        if result.trades:
            sorted_by_pnl = sorted(result.trades, key=lambda t: t.pnl)
            print(f"\n  Best trade:   ₹{sorted_by_pnl[-1].pnl:,.0f} ({sorted_by_pnl[-1].direction} entered {sorted_by_pnl[-1].entry_date})")
            print(f"  Worst trade:  ₹{sorted_by_pnl[0].pnl:,.0f} ({sorted_by_pnl[0].direction} entered {sorted_by_pnl[0].entry_date})")
        print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="MCX Donchian Trend Backtest")
    parser.add_argument("--instrument", default="GOLD",  choices=list(MCX_SPECS))
    parser.add_argument("--start",      default="2020-01-01")
    parser.add_argument("--end",        default="2026-06-01")
    parser.add_argument("--capital",    type=float, default=200_000)
    parser.add_argument("--period",     type=int,   default=20)
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    bt = MCXTrendBacktester(capital=args.capital, entry_period=args.period)
    try:
        result = bt.run(args.instrument, start, end)
        bt.print_report(result, args.instrument)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
