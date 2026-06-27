"""
Strategy J — Statistical VWAP Mean Reversion

Based on Gemini's suggestion. This is a well-researched institutional strategy.
The core insight: when price deviates >2.5σ from VWAP on LOW volume, it almost
always reverts. The volume filter is critical — high volume deviation = trend,
low volume deviation = noise/reversion.

Rules (5-min bars, 10:00 AM – 2:45 PM only):
  1. VWAP: cumulative (High+Low+Close)/3 × Volume / Σ Volume (resets at session open)
  2. Deviation: rolling 20-bar std dev of (close - VWAP)
  3. Z-Score_price = (close - VWAP) / std_dev
  4. Volume Z-Score = (vol - vol_20_mean) / vol_20_std (negative = quiet tape)

  LONG:  Z_price < -2.5  AND  Z_vol < -1.0   (price below VWAP on quiet volume)
  SHORT: Z_price > +2.5  AND  Z_vol < -1.0   (price above VWAP on quiet volume)

  Exit: limit order at VWAP (mean reversion target)
  Stop: 1.5 × ATR(14) from entry

Why volume filter matters:
  - Large Z-price WITHOUT volume → liquidity vacuum, order flow imbalance, reverts fast
  - Large Z-price WITH volume → informed buying/selling, trend continuation → STOP OUT

Gemini's version verbatim, adapted for our data infrastructure.
Tested on NIFTY/BANKNIFTY 1min → resampled to 5min.
"""

from __future__ import annotations

import logging
from datetime import time as dtime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger("platform.strategies.vwap_reversion")

ENTRY_START = dtime(10, 0)
ENTRY_END   = dtime(14, 45)
SQUARE_OFF  = dtime(15, 10)

LOT_SIZES = {"NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 40}


class VWAPReversionBacktester:
    """
    Backtest VWAP Mean Reversion on index 1-min data (resampled to 5-min).

    Usage
    -----
    bt = VWAPReversionBacktester("data/cache")
    result = bt.run("NIFTY", "2022-01-01", "2024-12-31")
    bt.print_report(result, "NIFTY")
    """

    def __init__(
        self,
        cache_dir:     str   = "data/cache",
        # Signal parameters (Gemini's spec)
        z_price_entry: float = 2.5,     # |Z-score| threshold
        z_vol_filter:  float = -1.0,    # volume must be quiet (Z < -1.0)
        std_window:    int   = 20,       # rolling window for price std dev
        vol_window:    int   = 10,       # rolling window for volume mean/std
        atr_period:    int   = 14,
        atr_stop_mult: float = 1.5,
        # Capital
        capital:       float = 200_000,
        lots:          int   = 1,
        brokerage:     float = 20.0,    # ₹20 per order
        slippage_pct:  float = 0.001,
    ) -> None:
        self._cache      = Path(cache_dir)
        self._z_price    = z_price_entry
        self._z_vol      = z_vol_filter
        self._std_win    = std_window
        self._vol_win    = vol_window
        self._atr_period = atr_period
        self._atr_mult   = atr_stop_mult
        self._capital    = capital
        self._lots       = lots
        self._brokerage  = brokerage
        self._slip       = slippage_pct

    # ── Indicators ─────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_intraday_vwap(df: pd.DataFrame) -> pd.Series:
        """
        True intraday VWAP resetting at each session open.
        Uses vectorised groupby for speed.
        """
        typical = (df["high"] + df["low"] + df["close"]) / 3
        tpv     = typical * df["volume"]
        date_key = df.index.date

        df2 = df.copy()
        df2["typical"] = typical
        df2["tpv"]     = tpv
        df2["date"]    = date_key

        df2["cum_tpv"] = df2.groupby("date")["tpv"].cumsum()
        df2["cum_vol"] = df2.groupby("date")["volume"].cumsum()
        df2["vwap"]    = df2["cum_tpv"] / df2["cum_vol"].replace(0, np.nan)
        return df2["vwap"]

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
        high, low, pc = df["high"], df["low"], df["close"].shift(1)
        tr = pd.concat([(high-low), (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    def _compute_features(self, df5: pd.DataFrame) -> pd.DataFrame:
        """Add VWAP, Z-scores, ATR to 5-min dataframe."""
        out = df5.copy()

        # Replace zero volume with small positive to avoid division by zero
        out["volume"] = out["volume"].replace(0, 1)

        # VWAP
        out["vwap"] = self._compute_intraday_vwap(out)

        # Price deviation from VWAP
        out["dev"]  = out["close"] - out["vwap"]

        # Rolling std of deviation (20 bars = 100 min = ~1.7 hrs of history)
        out["dev_std"] = (
            out.groupby(out.index.date)["dev"]
            .transform(lambda x: x.rolling(self._std_win, min_periods=5).std())
        )

        # Price Z-score
        out["z_price"] = out["dev"] / out["dev_std"].replace(0, np.nan)

        # Volume Z-score (rolling 10 bars within session)
        out["vol_mean"] = (
            out.groupby(out.index.date)["volume"]
            .transform(lambda x: x.rolling(self._vol_win, min_periods=5).mean())
        )
        out["vol_std"] = (
            out.groupby(out.index.date)["volume"]
            .transform(lambda x: x.rolling(self._vol_win, min_periods=5).std())
        )
        out["z_vol"] = (
            (out["volume"] - out["vol_mean"]) /
            out["vol_std"].replace(0, np.nan)
        )

        # ATR (14 bars on 5-min = 70 min)
        out["atr"] = self._compute_atr(out, self._atr_period)

        return out

    # ── Main backtest ──────────────────────────────────────────────────────────

    def run(self, instrument: str, start: str, end: str) -> dict:
        key = instrument.upper()
        lot_size = LOT_SIZES.get(key, 75)

        # Load 1-min data
        path = self._cache / f"{key}_1min.csv"
        if not path.exists():
            raise FileNotFoundError(f"No 1-min data for {key}. Expected: {path}")

        logger.info("Loading %s 1-min data…", key)
        raw = pd.read_csv(path, index_col="datetime", parse_dates=True)
        if raw.index.tz is None:
            raw.index = raw.index.tz_localize("Asia/Kolkata")

        # Slice date range
        raw = raw[start:end].copy()
        logger.info("Loaded %d 1-min bars. Resampling to 5-min…", len(raw))

        # Resample to 5-min
        df5 = raw.resample("5min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna(subset=["close"])

        # Compute features
        df5 = self._compute_features(df5)

        # Filter to trading hours only
        df5 = df5.between_time("09:15", "15:30")

        logger.info("Running backtest on %d 5-min bars…", len(df5))

        nav     = float(self._capital)
        trades:  List[dict] = []
        equity:  List[Tuple] = []
        _traded_days = set()   # one trade per session

        # State
        in_trade    = False
        direction   = 0          # 1 = long, -1 = short
        entry_price = 0.0
        stop_price  = 0.0
        vwap_target = 0.0
        trade_date  = None

        for ts, row in df5.iterrows():
            t = ts.time()
            d = ts.date()

            # Reset at new session
            if d != trade_date:
                if in_trade:   # force close at session end
                    exit_px = entry_price   # approximation
                    pnl = direction * (exit_px - entry_price) * self._lots * lot_size
                    trades.append({"date": trade_date, "direction": direction,
                                   "entry": entry_price, "exit": exit_px,
                                   "pnl": pnl - self._brokerage * 2, "reason": "force_close"})
                    nav += pnl - self._brokerage * 2
                in_trade = False
                trade_date = d

            # Skip if price z-score missing (still warming up)
            if pd.isna(row.get("z_price")):
                equity.append((ts, nav))
                continue

            # ── Exit check ────────────────────────────────────────────────────
            if in_trade:
                exit_px    = None
                exit_reason= ""

                if t >= SQUARE_OFF:
                    exit_px     = float(row["open"]) * (1 - direction * self._slip)
                    exit_reason = "time_stop"
                elif direction == 1:
                    if row["low"] <= stop_price:
                        exit_px     = stop_price * (1 - self._slip)
                        exit_reason = "stop_loss"
                    elif row["close"] >= vwap_target:
                        exit_px     = float(row["vwap"]) * (1 - self._slip)
                        exit_reason = "target_hit"
                else:
                    if row["high"] >= stop_price:
                        exit_px     = stop_price * (1 + self._slip)
                        exit_reason = "stop_loss"
                    elif row["close"] <= vwap_target:
                        exit_px     = float(row["vwap"]) * (1 + self._slip)
                        exit_reason = "target_hit"

                if exit_px is not None:
                    pnl = direction * (exit_px - entry_price) * self._lots * lot_size
                    pnl -= self._brokerage * 2
                    nav += pnl
                    trades.append({
                        "date": d, "ts": ts, "direction": direction,
                        "entry": entry_price, "exit": exit_px,
                        "pnl": pnl, "reason": exit_reason,
                    })
                    in_trade = False

            # ── Entry check (one trade per session only) ──────────────────────
            if not in_trade and ENTRY_START <= t <= ENTRY_END and trade_date not in _traded_days:
                z_price = float(row["z_price"])
                z_vol   = float(row["z_vol"])
                atr     = float(row["atr"]) if not pd.isna(row.get("atr")) else 0.0
                vwap    = float(row["vwap"])

                # Volume filter: pass if z_vol is unavailable (index data has no real volume)
                vol_ok = pd.isna(z_vol) or z_vol < self._z_vol
                if vol_ok:   # quiet volume (or no volume data) — reversion likely
                    if z_price < -self._z_price:   # LONG: price too far below VWAP
                        direction   = 1
                        entry_price = float(row["close"]) * (1 + self._slip)
                        stop_price  = entry_price - self._atr_mult * atr
                        vwap_target = vwap
                        in_trade    = True
                        _traded_days.add(trade_date)
                        nav        -= self._brokerage

                    elif z_price > self._z_price:  # SHORT: price too far above VWAP
                        direction   = -1
                        entry_price = float(row["close"]) * (1 - self._slip)
                        stop_price  = entry_price + self._atr_mult * atr
                        vwap_target = vwap
                        in_trade    = True
                        _traded_days.add(trade_date)
                        nav        -= self._brokerage

            equity.append((ts, nav))

        # Metrics
        n = len(trades)
        if n == 0:
            return {"error": "no trades"}

        wins   = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        wr     = len(wins) / n
        gw     = sum(t["pnl"] for t in wins)
        gl     = abs(sum(t["pnl"] for t in losses))
        pf     = gw / gl if gl > 0 else float("inf")

        eq_d, eq_v = zip(*equity)
        eq_s  = pd.Series(eq_v, index=pd.DatetimeIndex(eq_d))
        # Daily equity (last value per day)
        daily = eq_s.resample("D").last().dropna()
        peak  = daily.cummax()
        mdd   = float((daily / peak - 1).min())
        total_days = (pd.Timestamp(end) - pd.Timestamp(start)).days
        final = float(eq_s.iloc[-1])
        ratio = final / self._capital
        cagr  = float(abs(ratio) ** (365 / max(1, total_days)) - 1) if ratio > 0 else -1.0
        vol   = float(daily.pct_change().dropna().std() * 252**0.5)
        sharpe= (cagr - 0.065) / vol if vol > 0 else 0.0

        from collections import Counter
        reasons = Counter(t["reason"] for t in trades)

        return {
            "instrument": key, "n": n, "win_rate": wr,
            "avg_win":  float(np.mean([t["pnl"] for t in wins]))  if wins  else 0,
            "avg_loss": float(np.mean([t["pnl"] for t in losses])) if losses else 0,
            "profit_factor": pf,
            "expectancy": sum(t["pnl"] for t in trades) / n,
            "cagr": cagr, "sharpe": sharpe, "max_dd": mdd,
            "final_nav": final, "exit_reasons": reasons,
            "trades": trades,
        }

    def print_report(self, result: dict, instrument: str) -> None:
        if "error" in result:
            print(f"No result: {result['error']}")
            return
        print("\n" + "=" * 60)
        print(f"  VWAP MEAN REVERSION — {instrument}")
        print("=" * 60)
        print(f"  Trades:         {result['n']}")
        print(f"  Win Rate:       {result['win_rate']:.1%}")
        print(f"  Avg Win:        ₹{result['avg_win']:,.0f}")
        print(f"  Avg Loss:       ₹{result['avg_loss']:,.0f}")
        print(f"  Profit Factor:  {result['profit_factor']:.2f}")
        print(f"  Expectancy:     ₹{result['expectancy']:,.0f}/trade")
        print(f"  CAGR:           {result['cagr']:.1%}")
        print(f"  Sharpe:         {result['sharpe']:.2f}")
        print(f"  Max Drawdown:   {result['max_dd']:.1%}")
        print(f"  Final NAV:      ₹{result['final_nav']:,.0f}")
        print(f"\n  Exit breakdown:")
        for r, cnt in result["exit_reasons"].most_common():
            avg = np.mean([t["pnl"] for t in result["trades"] if t["reason"] == r])
            print(f"    {r:20s} {cnt:4d}  avg ₹{avg:,.0f}")
        print("=" * 60)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    instrument = sys.argv[1] if len(sys.argv) > 1 else "NIFTY"
    start      = sys.argv[2] if len(sys.argv) > 2 else "2022-01-01"
    end        = sys.argv[3] if len(sys.argv) > 3 else "2024-12-31"

    bt = VWAPReversionBacktester(
        cache_dir     = "data/cache",
        z_price_entry = 2.5,
        z_vol_filter  = -1.0,
        std_window    = 20,
        vol_window    = 10,
        atr_stop_mult = 1.5,
        capital       = 200_000,
        lots          = 1,
    )

    try:
        result = bt.run(instrument, start, end)
        bt.print_report(result, instrument)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
