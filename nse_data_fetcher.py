#!/usr/bin/env python3
"""
nse_data_fetcher.py — Download real NSE F&O bhavcopy option chain data.

Source: NSE India official archives (free, no auth needed)
  URL:  https://archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip

Provides per-day per-strike per-expiry:
  • Real OHLC option prices         → real IV from actual traded prices
  • Real Open Interest (OI)          → genuine SkewHunter alpha1 signal
  • Real daily ΔOI                   → OI change (institutional flow)
  • Real volume                      → actual call/put volume for alpha2
  • Real underlying spot price       → NSE's own settlement reference

How it integrates with the backtest:
  Daily bhavcopy data calibrates the 1-min synthetic chain:
    1. Real ATM IV (from actual option prices) → better IV surface
    2. Real OI per strike → genuine SkewHunter flow signals
    3. Real put/call volume ratio → better FixedRR alpha2
    4. Intraday option prices = BS(real_IV, real_spot) → realistic greeks

Usage:
    python3 nse_data_fetcher.py --start 2025-06-25 --end 2026-06-20
    python3 nse_data_fetcher.py --start 2025-06-25 --end 2026-06-20 --stats
"""

import argparse
import asyncio
import csv
import io
import json
import logging
import os
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple

import httpx
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nse_fetcher")

CACHE_ROOT  = "nse_option_cache"
NSE_URL     = "https://archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date}_F_0000.csv.zip"
HEADERS     = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
CONCURRENCY = 3
DELAY       = 0.5   # be gentle with NSE servers

UNDERLYINGS = {"NIFTY", "BANKNIFTY", "FINNIFTY"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _bhavcopy_url(dt: date) -> str:
    return NSE_URL.format(date=dt.strftime("%Y%m%d"))


def _cache_path(dt: date) -> str:
    return os.path.join(CACHE_ROOT, f"{dt.strftime('%Y-%m-%d')}.parquet")


def _meta_path() -> str:
    return os.path.join(CACHE_ROOT, "metadata.json")


def _load_meta() -> dict:
    if os.path.exists(_meta_path()):
        with open(_meta_path()) as f:
            return json.load(f)
    return {"fetched_dates": [], "failed_dates": []}


def _save_meta(meta: dict) -> None:
    with open(_meta_path(), "w") as f:
        json.dump(meta, f, indent=2)


def _trading_days(start: date, end: date) -> List[date]:
    """Mon–Fri dates (NSE holidays not excluded — missing files handled gracefully)."""
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


# ── NSE bhavcopy downloader ───────────────────────────────────────────────────

class NSEBhavCopyFetcher:
    """Downloads and caches NSE F&O bhavcopy for a date range."""

    def __init__(self, cache_root: str = CACHE_ROOT, force: bool = False):
        self.cache_root = cache_root
        self.force      = force
        self._sem       = asyncio.Semaphore(CONCURRENCY)
        os.makedirs(cache_root, exist_ok=True)

    async def _download_one(
        self, dt: date, client: httpx.AsyncClient, meta: dict
    ) -> bool:
        """Download and parse one day's bhavcopy. Returns True if data saved."""
        path = _cache_path(dt)
        if os.path.exists(path) and not self.force:
            return True   # already cached

        async with self._sem:
            await asyncio.sleep(DELAY)
            try:
                url = _bhavcopy_url(dt)
                r = await client.get(url)
                if r.status_code == 404:
                    # NSE holiday or non-trading day
                    meta["failed_dates"].append(dt.isoformat())
                    return False
                if r.status_code != 200:
                    logger.warning("  HTTP %d for %s", r.status_code, dt)
                    return False

                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    with z.open(z.namelist()[0]) as f:
                        df = pd.read_csv(f, low_memory=False)

                # Filter to supported underlyings (options only)
                opts = df[
                    df["TckrSymb"].isin(UNDERLYINGS) &
                    df["OptnTp"].isin(["CE", "PE"])
                ].copy()

                if opts.empty:
                    logger.debug("  No options data for %s", dt)
                    return False

                # Keep only columns we need
                opts = opts[[
                    "TckrSymb",    # NIFTY / BANKNIFTY / FINNIFTY
                    "XpryDt",      # expiry date string YYYY-MM-DD
                    "StrkPric",    # strike price
                    "OptnTp",      # CE / PE
                    "OpnPric",     # open
                    "HghPric",     # high
                    "LwPric",      # low
                    "ClsPric",     # close (settlement for expiry day)
                    "SttlmPric",   # settlement price
                    "OpnIntrst",   # open interest (contracts)
                    "ChngInOpnIntrst",  # change in OI
                    "TtlTradgVol", # total volume (contracts)
                    "UndrlygPric", # underlying spot price (NSE reference)
                ]].rename(columns={
                    "TckrSymb":         "underlying",
                    "XpryDt":           "expiry",
                    "StrkPric":         "strike",
                    "OptnTp":           "option_type",
                    "OpnPric":          "open",
                    "HghPric":          "high",
                    "LwPric":           "low",
                    "ClsPric":          "close",
                    "SttlmPric":        "settlement",
                    "OpnIntrst":        "oi",
                    "ChngInOpnIntrst":  "delta_oi",
                    "TtlTradgVol":      "volume",
                    "UndrlygPric":      "spot",
                })

                opts["trade_date"] = dt.isoformat()
                opts["strike"]     = pd.to_numeric(opts["strike"], errors="coerce")
                for col in ["open","high","low","close","settlement","oi","delta_oi","volume","spot"]:
                    opts[col] = pd.to_numeric(opts[col], errors="coerce").fillna(0)

                opts.to_parquet(path, index=False)
                return True

            except Exception as e:
                logger.warning("  Error fetching %s: %s", dt, e)
                return False

    async def fetch_range(self, start: date, end: date) -> None:
        """Download all trading days in [start, end]."""
        days = _trading_days(start, end)
        meta = _load_meta()
        cached_set = set(meta.get("fetched_dates", []))
        failed_set = set(meta.get("failed_dates", []))

        to_fetch = [
            d for d in days
            if d.isoformat() not in cached_set
            and d.isoformat() not in failed_set
        ]

        if not to_fetch and not self.force:
            logger.info(
                "All %d days already cached. Use --force to re-download.", len(days)
            )
            return

        logger.info(
            "Fetching %d / %d trading days from NSE archives ...",
            len(to_fetch), len(days),
        )

        limits = httpx.Limits(max_connections=CONCURRENCY + 2, max_keepalive_connections=CONCURRENCY)
        async with httpx.AsyncClient(
            timeout=30, follow_redirects=True,
            headers=HEADERS, limits=limits,
        ) as client:
            tasks = [self._download_one(d, client, meta) for d in to_fetch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        ok = sum(1 for r in results if r is True)
        logger.info("  Downloaded: %d / %d  |  Holiday/missing: %d", ok, len(to_fetch), len(to_fetch) - ok)

        for d in to_fetch:
            if os.path.exists(_cache_path(d)):
                cached_set.add(d.isoformat())

        meta["fetched_dates"] = sorted(cached_set)
        meta["last_updated"]  = datetime.now().isoformat()
        _save_meta(meta)
        logger.info("Cache: %s/  (%d days)", self.cache_root, len(cached_set))


# ── RealOptionDayData ─────────────────────────────────────────────────────────

class RealOptionDayData:
    """
    Loads and serves real NSE option data for a given trading date.

    For each (underlying, expiry, strike, type) provides:
      close_price, oi, delta_oi, volume, spot

    Used by CalibratedChainBuilder to inject real data into the 1-min simulation.
    """

    def __init__(self, cache_root: str = CACHE_ROOT):
        self.cache_root = cache_root
        self._loaded_date: Optional[str] = None
        self._data: Optional[pd.DataFrame] = None

    def load(self, dt: date) -> bool:
        """Load data for the given date. Returns True if data available."""
        key = dt.isoformat()
        if self._loaded_date == key:
            return self._data is not None and not self._data.empty
        self._loaded_date = key
        path = os.path.join(self.cache_root, f"{key}.parquet")
        if not os.path.exists(path):
            self._data = None
            return False
        try:
            self._data = pd.read_parquet(path)
            return not self._data.empty
        except Exception as e:
            logger.debug("Failed to load %s: %s", path, e)
            self._data = None
            return False

    def get_option_row(
        self,
        underlying: str,
        expiry_dt: datetime,
        strike: float,
        option_type: str,   # "CE" or "PE"
    ) -> Optional[dict]:
        """Return real option data for this specific contract, or None."""
        if self._data is None:
            return None
        expiry_str = expiry_dt.strftime("%Y-%m-%d")
        mask = (
            (self._data["underlying"]   == underlying) &
            (self._data["expiry"]       == expiry_str) &
            (self._data["strike"]       == strike) &
            (self._data["option_type"]  == option_type)
        )
        rows = self._data[mask]
        if rows.empty:
            return None
        row = rows.iloc[0]
        return {
            "close":     float(row["close"]),
            "oi":        int(row["oi"]),
            "delta_oi":  int(row["delta_oi"]),
            "volume":    int(row["volume"]),
            "spot":      float(row["spot"]),
        }

    def get_atm_iv_from_straddle(
        self,
        underlying: str,
        expiry_dt:  datetime,
        atm_strike: float,
        spot:       float,
        tte:        float,
        bs_engine,
    ) -> Optional[float]:
        """
        Compute real ATM IV from the actual ATM call + put settlement prices.
        Returns IV as decimal (e.g., 0.155 = 15.5%) or None if not available.
        """
        from bs_engine import OptionType
        call_row = self.get_option_row(underlying, expiry_dt, atm_strike, "CE")
        put_row  = self.get_option_row(underlying, expiry_dt, atm_strike, "PE")

        if not call_row or not put_row:
            return None
        if call_row["close"] <= 0 and put_row["close"] <= 0:
            return None

        ivs = []
        for row, otype in [(call_row, OptionType.CALL), (put_row, OptionType.PUT)]:
            if row["close"] > 0.5 and tte > 0:
                iv = bs_engine.implied_volatility_newton_raphson(
                    market_price=row["close"],
                    spot=spot, strike=atm_strike,
                    time_to_expiry=tte,
                    option_type=otype,
                )
                if iv and 0.05 <= iv <= 2.0:
                    ivs.append(iv)

        return float(np.mean(ivs)) if ivs else None

    def get_daily_stats(self, underlying: str) -> dict:
        """Aggregate statistics for monitoring."""
        if self._data is None:
            return {}
        sub = self._data[self._data["underlying"] == underlying]
        return {
            "total_rows":   len(sub),
            "expiries":     sub["expiry"].nunique(),
            "strikes":      sub["strike"].nunique(),
            "total_oi":     int(sub["oi"].sum()),
            "total_volume": int(sub["volume"].sum()),
            "spot":         float(sub["spot"].iloc[0]) if not sub.empty else 0,
        }

    @staticmethod
    def is_cache_available(cache_root: str = CACHE_ROOT) -> bool:
        meta = _load_meta() if os.path.exists(os.path.join(cache_root, "metadata.json")) else {}
        return len(meta.get("fetched_dates", [])) > 5

    @staticmethod
    def cache_stats(cache_root: str = CACHE_ROOT) -> dict:
        meta = _load_meta() if os.path.exists(os.path.join(cache_root, "metadata.json")) else {}
        dates = meta.get("fetched_dates", [])
        size_mb = sum(
            os.path.getsize(os.path.join(cache_root, f"{d}.parquet"))
            for d in dates
            if os.path.exists(os.path.join(cache_root, f"{d}.parquet"))
        ) / 1_048_576
        return {
            "trading_days_cached": len(dates),
            "date_range":  f"{dates[0]} → {dates[-1]}" if dates else "none",
            "cache_size_mb": round(size_mb, 1),
            "last_updated":  meta.get("last_updated", "never"),
        }


# ── CalibratedChainBuilder ────────────────────────────────────────────────────

class CalibratedChainBuilder:
    """
    Builds OptionChainSnapshot using REAL NSE daily data for calibration,
    then generates intraday 1-min pricing via BS model.

    Each bar's option chain reflects:
      ┌─────────────────────────────────────────────────────────────────┐
      │ REAL (from NSE bhavcopy loaded at day-start):                   │
      │   • ATM IV      — from actual straddle settlement price         │
      │   • IV skew     — slope calibrated from OTM strikes' real prices│
      │   • OI per strike — real institutional positioning              │
      │   • Volume per strike — real call/put flow ratios               │
      │   • ΔOI         — actual OI change from previous session        │
      │                                                                  │
      │ MODELLED (BS model applied intraday):                           │
      │   • Bid/ask spread — proportional to vega                       │
      │   • Intraday price — BS(real_IV, current_spot, tte)             │
      │   • Delta, Gamma, Vega, Theta — from real IV                    │
      └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        bs,
        spec,
        day_data: RealOptionDayData,
        synthetic_fallback=None,
    ):
        from bs_engine import OptionType
        from data_feed import OptionChainSnapshot, OptionQuote

        self.bs       = bs
        self.spec     = spec
        self._day     = day_data
        self._synth   = synthetic_fallback
        self._OT      = OptionType
        self._OCS     = OptionChainSnapshot
        self._OQ      = OptionQuote

        # Calibration state (updated at start of each trading day)
        self._cal_date:      Optional[str]   = None
        self._cal_atm_iv:    float           = 0.14
        self._cal_put_slope: float           = 0.0003   # IV per point OTM (put side)
        self._cal_call_slope:float           = 0.00005  # IV per point OTM (call side)
        self._cal_oi:        Dict           = {}        # {(strike, otype): real_oi}
        self._cal_vol:       Dict           = {}        # {(strike, otype): real_vol}
        self._cal_doi:       Dict           = {}        # {(strike, otype): delta_oi}
        self._bias:          float          = 0.0       # intraday sentiment
        self._rng                           = __import__("numpy").random.default_rng(42)

    def _calibrate(self, trade_date: date, expiry_dt: datetime,
                   spot: float, tte: float) -> None:
        """
        Calibrate IV surface from real NSE option data.
        Called once per trading day at 9:15 AM bar.
        """
        key = trade_date.isoformat()
        if self._cal_date == key:
            return   # already calibrated for today

        self._cal_date = key
        si   = self.spec.strike_interval
        atm  = round(spot / si) * si
        und  = self.spec.name

        if not self._day.load(trade_date):
            logger.debug("No real data for %s — using synthetic", key)
            return

        # ── 1. Real ATM IV ─────────────────────────────────────────────
        real_iv = self._day.get_atm_iv_from_straddle(
            und, expiry_dt, atm, spot, tte, self.bs
        )
        if real_iv:
            self._cal_atm_iv = real_iv
            logger.debug(
                "  Calibrated ATM IV: %.2f%%  (from real NSE straddle price)", real_iv * 100
            )

        # ── 2. Real IV skew slope ─────────────────────────────────────
        put_diffs, call_diffs = [], []
        for i_offset in [1, 2, 3, 4, 5]:
            for otype, diffs_list, sign in [("PE", put_diffs, -1), ("CE", call_diffs, 1)]:
                s_k = atm + sign * i_offset * si
                row = self._day.get_option_row(und, expiry_dt, s_k, otype)
                if row and row["close"] > 0.5 and tte > 0:
                    from bs_engine import OptionType as OT
                    ot = OT.PUT if otype == "PE" else OT.CALL
                    iv = self.bs.implied_volatility_newton_raphson(
                        row["close"], spot, s_k, tte, ot
                    )
                    if iv and 0.05 <= iv <= 2.0:
                        dist = abs(s_k - atm)
                        slope = (iv - self._cal_atm_iv) / max(1, dist)
                        diffs_list.append(slope)

        if put_diffs:
            self._cal_put_slope  = max(0.00005, float(np.mean(put_diffs)))
        if call_diffs:
            self._cal_call_slope = max(0.00001, float(np.mean(call_diffs)))

        # ── 3. Real OI and Volume per strike ──────────────────────────
        self._cal_oi.clear()
        self._cal_vol.clear()
        self._cal_doi.clear()
        for i in range(-20, 21):
            strike = atm + i * si
            for ot in ("CE", "PE"):
                row = self._day.get_option_row(und, expiry_dt, strike, ot)
                if row:
                    self._cal_oi[ (strike, ot)] = row["oi"]
                    self._cal_vol[(strike, ot)] = row["volume"]
                    self._cal_doi[(strike, ot)] = row["delta_oi"]

        logger.debug(
            "  Calibrated: put_slope=%.5f  call_slope=%.5f  "
            "strikes_with_data=%d",
            self._cal_put_slope, self._cal_call_slope, len(self._cal_oi),
        )

    def build(self, spot, vix_pct, timestamp, expiry_dt, tte):
        from bs_engine import OptionType
        from data_feed import OptionChainSnapshot, OptionQuote

        # Calibrate once per day (9:15 bar triggers calibration)
        self._calibrate(timestamp.date(), expiry_dt, spot, tte)

        si  = self.spec.strike_interval
        atm = round(spot / si) * si
        und = self.spec.name
        pfx = f"{self.spec.segment}:{und}"
        exp_str = expiry_dt.strftime("%d%b%y").upper()

        snap = OptionChainSnapshot(
            underlying=und,
            spot_price=spot,
            timestamp=timestamp,
            expiry=exp_str,
            atm_strike=atm,
        )
        if tte < 1e-6:
            return snap

        # Intraday sentiment bias (small random walk, adds intraday signal variance)
        self._bias += float(self._rng.normal(0, 0.02))
        self._bias  = float(np.clip(self._bias, -2.0, 2.0))

        for i in range(-20, 21):
            strike = atm + i * si
            if strike <= 0:
                continue

            dk_put  = max(0.0, atm - strike)
            dk_call = max(0.0, strike - atm)

            # IV surface: real calibrated slope + small intraday variation
            bias_adj = self._bias * 0.00002
            c_iv = max(0.04, self._cal_atm_iv + self._cal_call_slope * dk_call + bias_adj)
            p_iv = max(0.04, self._cal_atm_iv + self._cal_put_slope  * dk_put  - bias_adj)

            c_px = max(0.05, float(self.bs.call_price(spot, strike, tte, c_iv)))
            p_px = max(0.05, float(self.bs.put_price( spot, strike, tte, p_iv)))

            c_delta = float(self.bs.delta(spot, strike, tte, c_iv, OptionType.CALL))
            p_delta = float(self.bs.delta(spot, strike, tte, p_iv, OptionType.PUT))

            # Real OI / Volume from bhavcopy (or proportional synthetic if missing)
            oi_base = max(100, 80_000 - abs(i) * 3_000)
            c_oi  = self._cal_oi.get( (strike, "CE"), oi_base)
            p_oi  = self._cal_oi.get( (strike, "PE"), oi_base)
            c_vol = self._cal_vol.get((strike, "CE"), max(50, 8_000 - abs(i)*300))
            p_vol = self._cal_vol.get((strike, "PE"), max(50, 8_000 - abs(i)*300))

            # Bid-ask spread: tighter near ATM (realistic)
            spread = max(0.003, 0.003 + abs(i) * 0.001)

            # Bias-driven bid/ask imbalance (for Curvature viscosity signal)
            b_bid = max(0.3, 1.0 + 0.35 * self._bias)
            b_ask = max(0.3, 1.0 - 0.25 * self._bias)

            snap.calls[float(strike)] = OptionQuote(
                symbol=f"{pfx}{exp_str}{int(strike)}CE",
                strike=float(strike), expiry=exp_str,
                option_type=OptionType.CALL,
                ltp=round(c_px, 2),
                bid=round(c_px * (1 - spread), 2),
                ask=round(c_px * (1 + spread), 2),
                bid_qty=max(10, int(c_vol * b_bid / 10)),
                ask_qty=max(10, int(c_vol * b_ask / 10)),
                volume=c_vol,
                oi=c_oi,
                iv=c_iv,
                delta=c_delta,
            )
            snap.puts[float(strike)] = OptionQuote(
                symbol=f"{pfx}{exp_str}{int(strike)}PE",
                strike=float(strike), expiry=exp_str,
                option_type=OptionType.PUT,
                ltp=round(p_px, 2),
                bid=round(p_px * (1 - spread), 2),
                ask=round(p_px * (1 + spread), 2),
                bid_qty=max(10, int(p_vol * b_ask / 10)),
                ask_qty=max(10, int(p_vol * b_bid / 10)),
                volume=p_vol,
                oi=p_oi,
                iv=p_iv,
                delta=p_delta,
            )

        return snap


# ── CLI ───────────────────────────────────────────────────────────────────────

async def _main(args):
    if args.stats:
        stats = RealOptionDayData.cache_stats()
        print(f"\n  NSE Option Cache Statistics")
        print(f"  {'─'*40}")
        for k, v in stats.items():
            print(f"  {k:<25}: {v}")
        print()
        return

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end,   "%Y-%m-%d").date()

    print(f"\n{'━'*60}")
    print(f"  NSE F&O Bhavcopy Downloader")
    print(f"  Period : {start} → {end}")
    print(f"  Days   : ~{len(_trading_days(start,end))} trading days")
    print(f"  Source : NSE India Official Archives (free)")
    print(f"  Cache  : {CACHE_ROOT}/")
    print(f"{'━'*60}\n")

    fetcher = NSEBhavCopyFetcher(force=args.force)
    await fetcher.fetch_range(start, end)

    stats = RealOptionDayData.cache_stats()
    print(f"\n  Done.  {stats['trading_days_cached']} days cached  "
          f"({stats['cache_size_mb']} MB)  [{stats['date_range']}]")
    print(f"\n  Now run the backtest:")
    print(f"  python3 run_full_backtest.py --months 12 --capital 500000\n")


def main():
    p = argparse.ArgumentParser(
        description="Download real NSE F&O bhavcopy option chain data"
    )
    p.add_argument("--start",  default="2025-06-25", metavar="YYYY-MM-DD")
    p.add_argument("--end",    default=datetime.today().strftime("%Y-%m-%d"),
                   metavar="YYYY-MM-DD")
    p.add_argument("--force",  action="store_true", help="Re-download cached days")
    p.add_argument("--stats",  action="store_true", help="Show cache stats and exit")
    asyncio.run(_main(p.parse_args()))


if __name__ == "__main__":
    main()
