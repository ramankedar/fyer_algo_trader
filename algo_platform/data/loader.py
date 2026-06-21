"""
Converts cached CSVs into MarketBar lists and per-date VIX lookups.
Filters to NSE/BSE market hours (9:15 AM – 3:30 PM IST).

Volume synthesis
----------------
All Indian index data from Fyers has volume = 0 (indices don't have
own trading volume). To support volume-dependent features (VWAP weighting,
volume-spike detection in Strategy A), we synthesize a relative volume
proxy that captures:

  1. Intraday U-shape  — high at open/close, low at midday
                         (mirrors actual NSE/BSE cash market patterns)
  2. Range amplification — bars with wider range have proportionally
                          more synthetic volume (range = proxy for activity)
  3. Daily normalization — volumes are comparable across days

The absolute numbers don't matter; only the *relative* pattern matters
for volume spike detection (spike = current bar >> recent median bar).
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from algo_platform.core.types import MarketBar
from algo_platform.data.downloader import FyersDownloader

logger = logging.getLogger("algo_platform.data.loader")

IST          = ZoneInfo("Asia/Kolkata")
MARKET_OPEN  = (9,  15)   # 09:15 IST
MARKET_CLOSE = (15, 30)   # 15:30 IST

# Minutes from open at which intraday U-shape is minimum (midday)
_SESSION_MINUTES = 375    # 9:15 AM → 3:30 PM
_MIDDAY_MINUTE   = 187    # ≈ 12:17 PM (trough of U-shape)

# Synthetic volume base (relative scale; what matters is the ratio, not absolute)
_BASE_VOL = 100_000.0


class MarketDataLoader:
    """
    Thin wrapper around FyersDownloader that produces typed objects
    consumed by the backtest engine and research pipeline.
    """

    def __init__(self, downloader: FyersDownloader) -> None:
        self._dl = downloader

    # ── Bars ───────────────────────────────────────────────────────────────────

    def load_bars(
        self,
        instrument:          str,
        start:               date,
        end:                 date,
        resolution:          str  = "1",
        filter_market_hours: bool = True,
    ) -> List[MarketBar]:
        """
        Return list of MarketBar objects, chronologically sorted.
        Automatically synthesises relative volume when raw volume = 0.
        """
        df = self._dl.download(instrument, start, end, resolution)
        if df.empty:
            logger.warning("No data for %s [%s→%s]", instrument, start, end)
            return []

        if filter_market_hours and resolution in ("1", "5", "15", "60"):
            df = self._filter_market(df)

        # Detect and fix zero-volume index data
        if df["volume"].sum() == 0:
            logger.info(
                "[%s] Volume is all-zero (index data) — synthesising "
                "relative volume proxy.", instrument
            )
            df = self._synthesize_volume(df)

        bars: List[MarketBar] = []
        for ts, row in df.iterrows():
            dt = ts.to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=IST)
            bars.append(MarketBar(
                timestamp = dt,
                open      = float(row["open"]),
                high      = float(row["high"]),
                low       = float(row["low"]),
                close     = float(row["close"]),
                volume    = float(row["volume"]),
            ))

        logger.info("Loaded %d bars: %s [%s→%s] res=%s",
                    len(bars), instrument, start, end, resolution)
        return bars

    # ── VIX ────────────────────────────────────────────────────────────────────

    def load_vix(self, start: date, end: date) -> Dict[date, float]:
        """
        Returns {date: vix_close} for the requested range.
        Used by the chain builder to set ATM IV.
        """
        df = self._dl.download("VIX", start, end, resolution="D")
        if df.empty:
            logger.warning("No VIX data — chain builder will use default 15.0.")
            return {}

        result: Dict[date, float] = {}
        for ts, row in df.iterrows():
            result[ts.date()] = float(row["close"])

        logger.info("Loaded VIX for %d days.", len(result))
        return result

    # ── Volume synthesis ───────────────────────────────────────────────────────

    @staticmethod
    def _synthesize_volume(df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace zero volume with a realistic intraday relative volume proxy.

        Algorithm
        ---------
        vol(bar) = BASE × U(t) × R(bar)

        U(t)   = intraday shape factor ∈ [0.35, 1.0]:
                 highest at open (9:15) and close (15:30),
                 lowest around 12:15–12:30 (midday lull).

        R(bar) = bar_range / daily_avg_range ∈ [0.1, 10]:
                 bars with wide range (high activity) get proportionally
                 higher volume than quiet bars on the same day.
        """
        df = df.copy()

        # ── Intraday U-shape factor ───────────────────────────────────────────
        idx = df.index
        # Minutes elapsed since market open (9:15 = minute 0)
        minutes_from_open = (idx.hour - 9) * 60 + idx.minute - 15
        minutes_from_open = np.clip(minutes_from_open, 0, _SESSION_MINUTES)

        # Normalised distance from midday (0 = midday, 1 = open/close)
        dist_from_mid = np.abs(minutes_from_open - _MIDDAY_MINUTE) / _MIDDAY_MINUTE
        dist_from_mid = np.clip(dist_from_mid, 0.0, 1.0)

        # Smooth U-shape: 0.35 at midday, 1.0 at open/close
        u_shape = 0.35 + 0.65 * (dist_from_mid ** 0.6)

        # ── Range-based relative activity ─────────────────────────────────────
        bar_range = (df["high"] - df["low"]).clip(lower=0.0)

        # Daily average range (forward-fill on days with no data)
        daily_avg = (
            bar_range
            .groupby(df.index.normalize())
            .transform("mean")
        )
        daily_avg = daily_avg.replace(0, bar_range.mean() if bar_range.mean() > 0 else 1.0)

        rel_range = (bar_range / daily_avg).clip(0.1, 10.0)

        # ── Combine and add mild noise to avoid artificial regularity ─────────
        np.random.seed(0)   # deterministic so reruns are identical
        noise = np.random.lognormal(mean=0.0, sigma=0.15, size=len(df))

        df["volume"] = (_BASE_VOL * u_shape * rel_range.values * noise).round(0)
        df["volume"] = df["volume"].clip(lower=1.0)

        return df

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _filter_market(df: pd.DataFrame) -> pd.DataFrame:
        h, m = df.index.hour, df.index.minute
        after_open   = (h > MARKET_OPEN[0]) | ((h == MARKET_OPEN[0])  & (m >= MARKET_OPEN[1]))
        before_close = (h < MARKET_CLOSE[0])| ((h == MARKET_CLOSE[0]) & (m <= MARKET_CLOSE[1]))
        return df[after_open & before_close]
