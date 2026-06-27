"""
Fyers v3 historical data downloader with incremental CSV caching.

Cache strategy
--------------
Each instrument + resolution gets one CSV file: data/cache/{INSTR}_{res}.csv
On every download() call:
  1. Load existing cache (if any).
  2. Compute which date ranges are NOT in cache.
  3. Fetch only those ranges via chunked Fyers API calls (≤90 days per request).
  4. Merge, dedup, and save back to cache.
  5. Return the requested slice.

This means subsequent calls for overlapping date ranges are instant (disk only).
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("algo_platform.data.downloader")

FYERS_SYMBOLS: Dict[str, str] = {
    # NSE broad indices
    "NIFTY":     "NSE:NIFTY50-INDEX",
    "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
    "FINNIFTY":  "NSE:FINNIFTY-INDEX",
    "NIFTY100":  "NSE:NIFTY100-INDEX",
    "NIFTYMID":  "NSE:NIFTYMIDCAP50-INDEX",
    # BSE indices
    "SENSEX":    "BSE:SENSEX-INDEX",
    "BANKEX":    "BSE:BANKEX-INDEX",
    "BSEIT":     "BSE:IT-INDEX",
    # NSE Sector indices (Tier 2 — sector momentum strategy)
    "NIFTYIT":      "NSE:NIFTYIT-INDEX",
    "NIFTYAUTO":    "NSE:NIFTYAUTO-INDEX",
    "NIFTYMETAL":   "NSE:NIFTYMETAL-INDEX",
    "NIFTYPHARMA":  "NSE:NIFTYPHARMA-INDEX",
    "NIFTYFMCG":    "NSE:NIFTYFMCG-INDEX",
    "NIFTYENERGY":  "NSE:NIFTYENERGY-INDEX",
    "NIFTYREALTY":  "NSE:NIFTYREALTY-INDEX",
    "NIFTYINFRA":   "NSE:NIFTYINFRA-INDEX",
    "NIFTYMEDIA":   "NSE:NIFTYMEDIA-INDEX",
    "NIFTYPS":      "NSE:NIFTYPSE-INDEX",      # public sector enterprises
    # MCX Commodities — near-month contracts with cont_flag=1 for continuous history.
    # Fyers requires the SPECIFIC contract symbol (MCX:GOLD26AUGFUT), not a generic "-I" form.
    # Run refresh_mcx_symbols() to auto-update these from the Fyers symbol master each month.
    # Last refreshed: 2026-06-26
    "GOLD":         "MCX:GOLD26AUGFUT",
    "GOLDMINI":     "MCX:GOLDM26AUGFUT",
    "SILVER":       "MCX:SILVER26JULFUT",
    "CRUDEOIL":     "MCX:CRUDEOIL26JULFUT",
    "NATURALGAS":   "MCX:NATURALGAS26JULFUT",
    "COPPER":       "MCX:COPPER26JULFUT",
    "ZINC":         "MCX:ZINC26JULFUT",
    "ALUMINIUM":    "MCX:ALUMINIUM26JULFUT",
    "NICKEL":       "MCX:NICKEL26JULFUT",
    "LEAD":         "MCX:LEAD26JULFUT",
    # Volatility
    "VIX":       "NSE:INDIAVIX-INDEX",
}

_RES_LABEL: Dict[str, str] = {
    "1": "1min", "5": "5min", "15": "15min", "60": "60min", "D": "daily",
}

# 90 days safely under Fyers 100-day cap per request for 1-min data
CHUNK_DAYS = 90


class FyersDownloader:
    """
    Downloads OHLCV from Fyers v3 and caches to CSV.
    Handles chunking, rate-limiting, and incremental updates transparently.
    """

    def __init__(
        self,
        client_id:         str,
        access_token:      str,
        cache_dir:         str = "data/cache",
        rate_limit_sleep:  float = 1.0,
    ) -> None:
        self._dir    = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._sleep  = rate_limit_sleep
        self._fyers  = self._make_client(client_id, access_token)

    # ── Public ─────────────────────────────────────────────────────────────────

    def download(
        self,
        instrument: str,         # "NIFTY" | "BANKNIFTY" | "FINNIFTY" | "VIX"
        start:      date,
        end:        date,
        resolution: str = "1",   # "1" | "5" | "15" | "D"
    ) -> pd.DataFrame:
        """
        Returns DataFrame indexed by tz-aware IST datetimes.
        Columns: open, high, low, close, volume.
        """
        key       = instrument.upper()
        fyers_sym = FYERS_SYMBOLS.get(key)
        if fyers_sym is None:
            raise ValueError(
                f"Unknown instrument '{key}'. Valid: {sorted(FYERS_SYMBOLS)}"
            )

        cache_path = self._cache_path(key, resolution)
        existing   = self._load_cache(cache_path)

        missing = self._missing_ranges(existing, start, end)
        if not missing:
            logger.info("[%s %s] Cache hit — no download needed.", key, resolution)
        else:
            new_frames = []
            for rs, re in missing:
                logger.info("[%s] Downloading %s → %s (res=%s)", key, rs, re, resolution)
                df = self._fetch_chunked(fyers_sym, resolution, rs, re)
                if not df.empty:
                    new_frames.append(df)
                    logger.info("[%s] Got %d bars.", key, len(df))

            if new_frames:
                parts    = ([existing] if existing is not None else []) + new_frames
                combined = pd.concat(parts).sort_index()
                combined = combined[~combined.index.duplicated(keep="last")]
                self._save_cache(combined, cache_path)
                existing = combined
            else:
                logger.warning("[%s] No new data fetched for ranges: %s", key, missing)

        return self._slice(existing, start, end)

    def status(self) -> Dict[str, str]:
        """Return cache status: {filename: date_range} for each cached file."""
        result = {}
        for f in sorted(self._dir.glob("*.csv")):
            try:
                df = pd.read_csv(f, index_col="datetime", nrows=1)
                df2 = pd.read_csv(f, index_col="datetime")
                result[f.name] = f"{df.index[0][:10]} → {df2.index[-1][:10]} ({len(df2)} rows)"
            except Exception:
                result[f.name] = "unreadable"
        return result

    # ── Internals ──────────────────────────────────────────────────────────────

    def _fetch_chunked(
        self, fyers_sym: str, resolution: str, start: date, end: date,
    ) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        cur = start
        while cur <= end:
            chunk_end = min(cur + timedelta(days=CHUNK_DAYS), end)
            payload = {
                "symbol":     fyers_sym,
                "resolution": resolution,
                "date_format": "1",              # YYYY-MM-DD strings
                "range_from":  cur.strftime("%Y-%m-%d"),
                "range_to":    chunk_end.strftime("%Y-%m-%d"),
                "cont_flag":   "1",
            }
            try:
                resp = self._fyers.history(data=payload)
                if resp.get("s") == "ok" and resp.get("candles"):
                    frames.append(self._parse_candles(resp["candles"]))
                else:
                    msg = resp.get("message", resp.get("errmsg", str(resp)))
                    logger.warning("  [%s→%s] empty/error: %s", cur, chunk_end, msg)
            except Exception as exc:
                logger.error("  [%s→%s] fetch failed: %s", cur, chunk_end, exc)

            cur = chunk_end + timedelta(days=1)
            time.sleep(self._sleep)

        return pd.concat(frames) if frames else pd.DataFrame()

    @staticmethod
    def _parse_candles(candles: list) -> pd.DataFrame:
        df = pd.DataFrame(
            candles, columns=["epoch", "open", "high", "low", "close", "volume"]
        )
        df["datetime"] = (
            pd.to_datetime(df["epoch"], unit="s")
            .dt.tz_localize("UTC")
            .dt.tz_convert("Asia/Kolkata")
        )
        df.set_index("datetime", inplace=True)
        df.drop(columns=["epoch"], inplace=True)
        return df.astype(float).sort_index()

    def _cache_path(self, instrument: str, resolution: str) -> Path:
        label = _RES_LABEL.get(resolution, resolution)
        return self._dir / f"{instrument}_{label}.csv"

    def _load_cache(self, path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, index_col="datetime", parse_dates=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize("Asia/Kolkata")
            return df.astype(float)
        except Exception as exc:
            logger.warning("Cache '%s' unreadable (%s) — will re-fetch.", path.name, exc)
            return None

    @staticmethod
    def _save_cache(df: pd.DataFrame, path: Path) -> None:
        df.to_csv(path)
        logger.info("Saved %d rows → %s", len(df), path.name)

    @staticmethod
    def _missing_ranges(
        existing:   Optional[pd.DataFrame],
        want_start: date,
        want_end:   date,
    ) -> List[Tuple[date, date]]:
        if existing is None or existing.empty:
            return [(want_start, want_end)]
        have_start = existing.index.min().date()
        have_end   = existing.index.max().date()
        gaps: List[Tuple[date, date]] = []
        if want_start < have_start:
            gaps.append((want_start, have_start - timedelta(days=1)))
        if want_end > have_end:
            gaps.append((have_end + timedelta(days=1), want_end))
        return gaps

    @staticmethod
    def _slice(df: Optional[pd.DataFrame], start: date, end: date) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        lo = pd.Timestamp(start, tz="Asia/Kolkata")
        hi = pd.Timestamp(end,   tz="Asia/Kolkata") + pd.Timedelta(days=1)
        return df[(df.index >= lo) & (df.index < hi)]

    @staticmethod
    def _make_client(client_id: str, access_token: str):
        try:
            from fyers_apiv3 import fyersModel
            return fyersModel.FyersModel(
                client_id=client_id, token=access_token,
                is_async=False, log_path="",
            )
        except ImportError:
            raise ImportError(
                "fyers-apiv3 not installed. Run: pip install fyers-apiv3"
            )


def refresh_mcx_symbols(update_module: bool = True) -> Dict[str, str]:
    """
    Auto-detect current near-month MCX symbols from Fyers public symbol master.

    Call this once per month (or add to download_mcx.py startup) to keep
    FYERS_SYMBOLS up to date as contracts expire and roll to next month.

    Parameters
    ----------
    update_module : bool
        If True (default), updates FYERS_SYMBOLS in this module in-place
        so subsequent download() calls use the refreshed symbols.

    Returns
    -------
    Dict[str, str]
        {commodity_key: fyers_symbol} mapping of near-month contracts.
    """
    import requests

    MCX_WANT = {
        "GOLD":       lambda n: "GOLD" == n.split()[0] and "MINI" not in n and "GUINEA" not in n and "PETAL" not in n,
        "GOLDMINI":   lambda n: n.startswith("GOLDM "),
        "SILVER":     lambda n: "SILVER" == n.split()[0] and "MICRO" not in n and "MINI" not in n,
        "CRUDEOIL":   lambda n: "CRUDEOIL" == n.split()[0] and "MINI" not in n,
        "NATURALGAS": lambda n: "NATURALGAS" == n.split()[0] and "MINI" not in n,
        "COPPER":     lambda n: "COPPER" == n.split()[0] and "MINI" not in n,
        "ZINC":       lambda n: "ZINC" == n.split()[0] and "MINI" not in n,
        "ALUMINIUM":  lambda n: "ALUMINIUM" == n.split()[0] and "MINI" not in n,
        "NICKEL":     lambda n: "NICKEL" == n.split()[0] and "MINI" not in n,
        "LEAD":       lambda n: "LEAD" == n.split()[0] and "MINI" not in n,
    }

    try:
        resp = requests.get(
            "https://public.fyers.in/sym_details/MCX_COM.csv",
            timeout=20,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("MCX symbol refresh failed: %s — using existing symbols", exc)
        return {}

    contracts: Dict[str, list] = {k: [] for k in MCX_WANT}
    for line in resp.text.strip().split("\n"):
        parts = line.split(",")
        if len(parts) < 10:
            continue
        sym  = parts[9]
        name = parts[1].strip().upper()
        if not sym.startswith("MCX:") or "FUT" not in sym:
            continue
        for key, matcher in MCX_WANT.items():
            if matcher(name):
                contracts[key].append(sym)
                break

    result: Dict[str, str] = {}
    for key, syms in contracts.items():
        if syms:
            result[key] = syms[0]   # first = nearest expiry
            logger.info("MCX near-month: %-12s → %s", key, syms[0])

    if update_module and result:
        FYERS_SYMBOLS.update(result)
        logger.info("FYERS_SYMBOLS updated with %d MCX near-month contracts", len(result))

    return result
