"""
Strategy E — Sector Momentum Rotation

Thesis: NSE sector indices show persistent short-term momentum (3–10 days).
The top-ranked sector over the last 5 trading days outperforms the market by 1–2%
over the next 1–3 days with ~58% consistency (documented in Indian equity literature).
Sector rotation in India is slower than the US because fewer algorithmic sector ETF
rebalancers exist, creating exploitable persistence in flows.

Signal: Daily, generated after market close for next-day entry.
Trade:
  - Morning (9:30 AM): Enter long position in top-1 sector index via ATM call
  - OR short position in bottom-1 sector via ATM put (optional, higher risk)
  - Map sector → instrument:
      "NIFTYIT"    → FINNIFTY options (IT = 25% of FINNIFTY)
      "BANKNIFTY" (NIFTYBANKING) → BANKNIFTY options
      default      → NIFTY options
  - Exit: next morning at open OR at -1% stop loss on underlying

Standalone backtester:
  Uses daily sector index data (not 1min). Results in sector-relative return series.
  Run: python -m algo_platform.strategies.sector_momentum
  Requires: data/cache/{SECTOR}_daily.csv files (download via FyersDownloader first).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("platform.strategies.sector_momentum")

# ── Sector universe ────────────────────────────────────────────────────────────

SECTORS: List[str] = [
    "NIFTYIT", "NIFTYAUTO", "NIFTYMETAL", "NIFTYPHARMA",
    "NIFTYFMCG", "NIFTYENERGY", "NIFTYREALTY", "NIFTYINFRA",
]

# Map sector → best tradeable instrument for options
SECTOR_TO_INSTRUMENT: Dict[str, str] = {
    "NIFTYIT":     "FINNIFTY",     # IT heavy in FINNIFTY
    "NIFTYAUTO":   "NIFTY",
    "NIFTYMETAL":  "NIFTY",
    "NIFTYPHARMA": "NIFTY",
    "NIFTYFMCG":   "NIFTY",
    "NIFTYENERGY": "NIFTY",
    "NIFTYREALTY": "NIFTY",
    "NIFTYINFRA":  "NIFTY",
}

# Sector-BANKNIFTY correlation override
_BANK_CORR_OVERRIDE = {"NIFTYBANK": "BANKNIFTY"}


@dataclass
class SectorSignal:
    date:          date
    top_sector:    str
    bottom_sector: str
    top_momentum:  float    # 5-day momentum of top sector
    bot_momentum:  float
    instrument:    str      # NIFTY / BANKNIFTY / FINNIFTY
    direction:     str      # "LONG" or "SHORT"
    rank_table:    Dict[str, float] = field(default_factory=dict)


@dataclass
class SectorBacktestResult:
    signals:         List[SectorSignal]
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    sharpe:          float
    cagr:            float
    max_dd:          float
    win_rate:        float
    avg_win:         float
    avg_loss:        float
    n_trades:        int


class SectorMomentumBacktester:
    """
    Daily sector momentum backtest.

    Usage
    -----
    bt = SectorMomentumBacktester(cache_dir="data/cache")
    result = bt.run(
        start=date(2021, 1, 1),
        end=date(2026, 6, 1),
        momentum_window=5,
        hold_days=1,
        top_n=1,
        long_only=True,
    )
    bt.print_report(result)
    """

    def __init__(self, cache_dir: str = "data/cache") -> None:
        self._cache = Path(cache_dir)

    # ── Data loading ───────────────────────────────────────────────────────────

    def _load_sector(self, sector: str) -> Optional[pd.DataFrame]:
        path = self._cache / f"{sector}_daily.csv"
        if not path.exists():
            logger.warning("Sector data not found: %s — run download first.", path)
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        return df[["close"]].rename(columns={"close": sector})

    def _load_benchmark(self, instrument: str = "NIFTY") -> Optional[pd.DataFrame]:
        path = self._cache / f"{instrument}_daily.csv"
        if not path.exists():
            # Fallback: daily from 1min (pick last bar each day)
            path_1min = self._cache / f"{instrument}_1min.csv"
            if path_1min.exists():
                df = pd.read_csv(path_1min, index_col=0, parse_dates=True)
                df.index = pd.DatetimeIndex(df.index).tz_localize(None)
                daily = df["close"].resample("D").last().dropna()
                return daily.rename("NIFTY").to_frame()
            return None
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        return df[["close"]].rename(columns={"close": instrument})

    # ── Core logic ─────────────────────────────────────────────────────────────

    def run(
        self,
        start:            date,
        end:              date,
        momentum_window:  int   = 5,     # lookback in trading days
        hold_days:        int   = 1,     # hold period after signal
        top_n:            int   = 1,     # number of top sectors to long
        long_only:        bool  = True,  # also short bottom sector if False
        min_momentum:     float = 0.005, # minimum 5-day momentum to trade (0.5%)
        transaction_cost: float = 0.001, # 0.1% round-trip (ETF proxy cost)
    ) -> SectorBacktestResult:
        # Load all sector data
        sector_frames = []
        available_sectors = []
        for s in SECTORS:
            df = self._load_sector(s)
            if df is not None:
                sector_frames.append(df)
                available_sectors.append(s)

        if not sector_frames:
            raise RuntimeError(
                "No sector data found. Run:\n"
                "  python -m algo_platform.run download-sectors --start 2020-01-01"
            )

        sector_data = pd.concat(sector_frames, axis=1).sort_index()
        benchmark   = self._load_benchmark("NIFTY")

        # Align dates
        lo = pd.Timestamp(start)
        hi = pd.Timestamp(end)
        sector_data = sector_data[(sector_data.index >= lo) & (sector_data.index <= hi)]

        # Compute daily returns
        sector_rets = sector_data.pct_change()

        # Compute rolling momentum (lookback window)
        sector_mom = (sector_data / sector_data.shift(momentum_window) - 1)

        signals:          List[SectorSignal]  = []
        strat_ret_list:   List[Tuple[date, float]] = []

        dates = sector_data.index.tolist()

        for i in range(momentum_window + 1, len(dates) - hold_days):
            sig_date = dates[i].date()
            row_mom  = sector_mom.iloc[i].dropna()
            if row_mom.empty:
                continue

            rank    = row_mom.sort_values(ascending=False)
            top_sec = rank.index[0]
            bot_sec = rank.index[-1]
            top_val = rank.iloc[0]
            bot_val = rank.iloc[-1]

            if top_val < min_momentum:
                continue   # momentum not strong enough to trade

            instrument = SECTOR_TO_INSTRUMENT.get(top_sec, "NIFTY")
            direction  = "LONG"

            # Signal: use next-day-to-next-day return (i → i+hold_days)
            bench_col = "NIFTY" if benchmark is not None else None
            trade_returns = []

            # For simplicity: proxy trade return = sector ETF return over hold period
            for n in range(1, hold_days + 1):
                if i + n < len(dates):
                    day_ret = sector_rets[top_sec].iloc[i + n]
                    if not np.isnan(day_ret):
                        trade_returns.append(day_ret)

            if not trade_returns:
                continue

            total_ret = (1 + pd.Series(trade_returns)).prod() - 1 - transaction_cost

            signals.append(SectorSignal(
                date          = sig_date,
                top_sector    = top_sec,
                bottom_sector = bot_sec,
                top_momentum  = top_val,
                bot_momentum  = bot_val,
                instrument    = instrument,
                direction     = direction,
                rank_table    = rank.to_dict(),
            ))
            strat_ret_list.append((sig_date, total_ret))

        if not strat_ret_list:
            raise RuntimeError("No signals generated — check data range and filters.")

        sig_dates, strat_rets = zip(*strat_ret_list)
        strat_series = pd.Series(strat_rets, index=pd.DatetimeIndex(sig_dates))

        # Benchmark: buy-and-hold NIFTY over same period
        if benchmark is not None:
            bm = benchmark["NIFTY"].pct_change().reindex(strat_series.index).fillna(0)
        else:
            bm = pd.Series(0.0, index=strat_series.index)

        # Performance metrics
        n_trades   = len(strat_series)
        wins       = strat_series[strat_series > 0]
        losses     = strat_series[strat_series <= 0]
        win_rate   = len(wins) / n_trades if n_trades > 0 else 0.0
        avg_win    = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss   = float(losses.mean()) if len(losses) > 0 else 0.0

        cum        = (1 + strat_series).cumprod()
        peak       = cum.cummax()
        dd         = (cum / peak - 1).min()
        total_days = (pd.Timestamp(end) - pd.Timestamp(start)).days
        cagr       = float((cum.iloc[-1]) ** (365 / max(1, total_days)) - 1)
        vol        = float(strat_series.std() * np.sqrt(252))
        sharpe     = (cagr - 0.065) / vol if vol > 0 else 0.0

        return SectorBacktestResult(
            signals          = signals,
            strategy_returns = strat_series,
            benchmark_returns= bm,
            sharpe           = sharpe,
            cagr             = cagr,
            max_dd           = float(dd),
            win_rate         = win_rate,
            avg_win          = avg_win,
            avg_loss         = avg_loss,
            n_trades         = n_trades,
        )

    def print_report(self, result: SectorBacktestResult) -> None:
        print("\n" + "=" * 60)
        print("SECTOR MOMENTUM ROTATION — BACKTEST REPORT")
        print("=" * 60)
        print(f"  Trades:    {result.n_trades}")
        print(f"  Win Rate:  {result.win_rate:.1%}")
        print(f"  Avg Win:   {result.avg_win:.2%}")
        print(f"  Avg Loss:  {result.avg_loss:.2%}")
        print(f"  CAGR:      {result.cagr:.1%}")
        print(f"  Sharpe:    {result.sharpe:.2f}")
        print(f"  Max DD:    {result.max_dd:.1%}")
        print()

        # Top sectors summary
        from collections import Counter
        top_freq = Counter(s.top_sector for s in result.signals)
        print("  Top Sectors by Frequency:")
        for sec, cnt in top_freq.most_common(5):
            print(f"    {sec:20s} {cnt:4d} days")

        print("=" * 60)

    def get_todays_signal(
        self,
        momentum_window: int = 5,
        min_momentum: float = 0.005,
    ) -> Optional[SectorSignal]:
        """Return today's sector signal from cached data (for live/paper trading)."""
        sector_frames = []
        for s in SECTORS:
            df = self._load_sector(s)
            if df is not None:
                sector_frames.append(df)

        if not sector_frames:
            return None

        sector_data = pd.concat(sector_frames, axis=1).sort_index()
        if len(sector_data) < momentum_window + 1:
            return None

        row_mom  = (sector_data.iloc[-1] / sector_data.iloc[-(momentum_window + 1)] - 1)
        rank     = row_mom.dropna().sort_values(ascending=False)
        if rank.empty or rank.iloc[0] < min_momentum:
            return None

        top_sec = rank.index[0]
        bot_sec = rank.index[-1]
        today   = sector_data.index[-1].date()

        return SectorSignal(
            date          = today,
            top_sector    = top_sec,
            bottom_sector = bot_sec,
            top_momentum  = float(rank.iloc[0]),
            bot_momentum  = float(rank.iloc[-1]),
            instrument    = SECTOR_TO_INSTRUMENT.get(top_sec, "NIFTY"),
            direction     = "LONG",
            rank_table    = rank.to_dict(),
        )


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from datetime import date as _date

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    start = _date(2021, 1, 1)
    end   = _date(2026, 6, 1)

    bt = SectorMomentumBacktester(cache_dir="data/cache")
    try:
        result = bt.run(start=start, end=end, momentum_window=5, hold_days=1, top_n=1)
        bt.print_report(result)

        # Show recent signals
        print("\nLast 10 signals:")
        for sig in result.signals[-10:]:
            print(f"  {sig.date}  top={sig.top_sector:15s}  mom={sig.top_momentum:.2%}  → {sig.instrument}")

    except RuntimeError as e:
        print(f"\nERROR: {e}")
        print("\nDownload sector data first:")
        print("  python download_sectors.py")
        sys.exit(1)
