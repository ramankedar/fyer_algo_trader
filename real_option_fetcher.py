#!/usr/bin/env python3
"""
real_option_fetcher.py — Fetch real 1-min option candle data from Fyers API.

WHY THIS MATTERS
───────────────
Our current backtest uses synthetic option chains built from the BS model.
Every alpha signal (IV skew, volume ratios, OI changes) is computed on random
synthetic data — giving near-random signals.

With real option data we get:
  • Real option prices       → accurate IV per strike per minute
  • Real traded volume       → genuine call/put flow for SkewHunter alpha1
  • Real bid-ask spreads     → actual liquidity (viscosity) for Curvature
  • Real IV surface changes  → actual skew dynamics for FixedRR

DATA STRUCTURE
─────────────
option_data_cache/
  nifty/
    metadata.json           ← list of expiries + strikes fetched
    2025-06-19/             ← one folder per expiry date
      NFO_NIFTY2561924000CE.csv
      NFO_NIFTY2561924000PE.csv
      ...

Each CSV: timestamp,open,high,low,close,volume  (1-min bars)

USAGE
─────
Step 1 — Fetch and cache (~7-10 min for 12 months of Nifty):
    python3 real_option_fetcher.py --months 12 --instrument nifty

Step 2 — Run backtest with real data:
    python3 run_full_backtest.py --months 12 --use-real-options

Step 3 — Refresh cache (e.g. after a month):
    python3 real_option_fetcher.py --months 1 --instrument nifty --force
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

# ── Load credentials ──────────────────────────────────────────────────────────
load_dotenv(override=True)
try:
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.removeprefix("export").strip()
            if "=" in line:
                k, _, v = line.partition("=")
                os.environ[k.strip()] = v.strip().strip('"').strip("'")
except FileNotFoundError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("option_fetcher")

# ── Constants ─────────────────────────────────────────────────────────────────

CACHE_ROOT = "option_data_cache"
FYERS_BASE = "https://api-t1.fyers.in/data"
CHUNK_DAYS_SPOT    = 59  # max calendar days per request for index (spot/VIX)
CHUNK_DAYS_OPTION  = 10  # Fyers caps 1-min option data at ~10 days per request
CONCURRENCY = 4         # parallel API calls (be gentle with Fyers rate limits)
CALL_DELAY  = 0.25      # seconds between calls per coroutine slot

INSTRUMENT_PARAMS = {
    "nifty": {
        "underlying":      "NIFTY",
        "spot_symbol":     "NSE:NIFTY50-INDEX",
        "data_prefix":     "NSE",    # prefix for data/history API
        "trade_segment":   "NFO",    # prefix for order placement API
        "strike_interval": 50.0,
        "expiry_weekday":  1,         # Tuesday (NSE changed from Thursday in 2024)
    },
    "banknifty": {
        "underlying":      "BANKNIFTY",
        "spot_symbol":     "NSE:NIFTYBANK-INDEX",
        "data_prefix":     "NSE",
        "trade_segment":   "NFO",
        "strike_interval": 100.0,
        "expiry_weekday":  2,    # Wednesday
    },
    "sensex": {
        "underlying":      "SENSEX",
        "spot_symbol":     "BSE:SENSEX-INDEX",
        "data_prefix":     "BSE",
        "trade_segment":   "BFO",
        "strike_interval": 100.0,
        "expiry_weekday":  4,    # Friday
    },
}

# ─── IMPORTANT NOTE ───────────────────────────────────────────────────────────
# Fyers purges 1-min option data for EXPIRED contracts immediately after expiry.
# Historical 1-min option backtesting via Fyers API is NOT possible.
#
# This fetcher is useful for:
#   (a) Fetching CURRENT WEEK's real option prices for live trading validation
#   (b) Calibrating the synthetic BS model parameters against real market prices
#
# For real historical option data, consider:
#   - NSEpy (free, daily OHLC from NSE)
#   - Opstra Definedge (paid, 1-min chain history)
#   - TickerPlant (professional feed)
# ─────────────────────────────────────────────────────────────────────────────

# Month chars for Fyers weekly option symbols (Oct→O, Nov→N, Dec→D)
_MONTH_CHAR = {
    1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6",
    7:"7", 8:"8", 9:"9", 10:"O", 11:"N", 12:"D",
}


# ── Symbol helpers ────────────────────────────────────────────────────────────

def fyers_weekly_symbol(data_prefix: str, underlying: str, expiry_dt: datetime,
                         strike: float, otype: str) -> str:
    """
    Build Fyers weekly option symbol for DATA API (history/quotes).
    Format: {NSE/BSE}:{UNDERLYING}{YY}{M}{DD}{STRIKE}{CE/PE}
    Example: NSE:NIFTY2562324000CE  (June 23 2026 expiry)

    IMPORTANT: data_prefix is 'NSE' for Nifty/BankNifty, 'BSE' for Sensex.
    This is DIFFERENT from the trading segment ('NFO'/'BFO') used for orders.
    """
    yy = expiry_dt.strftime("%y")
    mc = _MONTH_CHAR[expiry_dt.month]
    dd = expiry_dt.strftime("%d")
    return f"{data_prefix}:{underlying}{yy}{mc}{dd}{int(strike)}{otype}"


def csv_safe_symbol(symbol: str) -> str:
    """Convert Fyers symbol to a safe filename."""
    return symbol.replace(":", "_").replace("/", "-")


def weekly_expiries(start: date, end: date, weekday: int) -> List[datetime]:
    """
    All occurrences of `weekday` (0=Mon…4=Fri) between start and end (inclusive).
    """
    result = []
    d = start
    # Fast-forward to first occurrence
    while d.weekday() != weekday:
        d += timedelta(days=1)
    while d <= end:
        result.append(datetime.combine(d, datetime.min.time()))
        d += timedelta(weeks=1)
    return result


def atm_from_spot(spot: float, strike_interval: float) -> float:
    return round(spot / strike_interval) * strike_interval


# ── Fyers API client (re-uses backtest.py logic) ──────────────────────────────

class FyersClient:
    """Thin async Fyers data-history client."""

    def __init__(self, app_id: str, access_token: str):
        self._headers = {"Authorization": f"{app_id}:{access_token}"}

    async def candles(
        self,
        symbol:     str,
        from_date:  str,
        to_date:    str,
        resolution: int  = 1,
        is_option:  bool = False,
        client: Optional[httpx.AsyncClient] = None,
    ) -> List[dict]:
        """
        Fetch 1-min OHLCV candles for `symbol`.

        Key differences between spot and option requests:
          - cont_flag=1  → only for continuous futures/spot; OMIT for options (causes 422)
          - chunk size   → 59 days for spot; 10 days for options (Fyers limit)
          - date_format  → 1 (date strings) for both
        """
        start = datetime.strptime(from_date, "%Y-%m-%d")
        end   = datetime.strptime(to_date,   "%Y-%m-%d")
        chunk = timedelta(days=CHUNK_DAYS_OPTION if is_option else CHUNK_DAYS_SPOT)
        cur   = start
        all_c: List[dict] = []

        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=30)

        try:
            while cur <= end:
                ce = min(cur + chunk, end)

                params: dict = {
                    "symbol":      symbol,
                    "resolution":  str(resolution),
                    "date_format": "1",
                    "range_from":  cur.strftime("%Y-%m-%d"),
                    "range_to":    ce.strftime("%Y-%m-%d"),
                }
                # cont_flag=1 = continuous contract rolling; valid only for spot/index
                if not is_option:
                    params["cont_flag"] = "1"

                r = await client.get(
                    f"{FYERS_BASE}/history",
                    headers=self._headers,
                    params=params,
                )

                if r.status_code != 200:
                    logger.debug(
                        "HTTP %d for %s (%s→%s): %s",
                        r.status_code, symbol[-25:],
                        cur.strftime("%Y-%m-%d"), ce.strftime("%Y-%m-%d"),
                        r.text[:120],
                    )
                    cur = ce + timedelta(days=1)
                    continue

                d = r.json()
                if d.get("s") == "ok":
                    for row in d.get("candles", []):
                        ts, o, h, l, c, v = row
                        if c > 0:
                            if isinstance(ts, (int, float)):
                                ts_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:00")
                            else:
                                ts_str = str(ts)[:16].replace("T", " ") + ":00"
                            all_c.append({
                                "t": ts_str,
                                "o": o, "h": h, "l": l, "c": c, "v": int(v),
                            })
                elif d.get("s") == "no_data":
                    pass   # no trades in this window — normal for illiquid strikes
                else:
                    logger.debug(
                        "API error for %s: code=%s msg=%s",
                        symbol[-25:], d.get("code"), d.get("message"),
                    )

                cur = ce + timedelta(days=1)
        finally:
            if owns_client:
                await client.aclose()

        return all_c

    async def spot_close(self, spot_symbol: str, from_date: str, to_date: str) -> Dict[date, float]:
        """Return {date: closing spot price} for underlying."""
        raw   = await self.candles(spot_symbol, from_date, to_date,
                                   resolution=1, is_option=False)
        daily: Dict[date, float] = {}
        for row in raw:
            d = datetime.strptime(row["t"], "%Y-%m-%d %H:%M:00").date()
            daily[d] = row["c"]   # last bar of day wins
        return daily


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _expiry_dir(cache_root: str, ikey: str, expiry_dt: datetime) -> str:
    return os.path.join(cache_root, ikey, expiry_dt.strftime("%Y-%m-%d"))


def _candle_path(cache_root: str, ikey: str, expiry_dt: datetime, symbol: str) -> str:
    return os.path.join(
        _expiry_dir(cache_root, ikey, expiry_dt),
        csv_safe_symbol(symbol) + ".csv",
    )


def _save_candles(path: str, candles: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "o", "h", "l", "c", "v"])
        w.writeheader()
        w.writerows(candles)


def _load_candles(path: str) -> Dict[str, dict]:
    """Return {timestamp_str: candle_dict}."""
    result = {}
    if not os.path.exists(path):
        return result
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            result[row["t"]] = {
                "o": float(row["o"]), "h": float(row["h"]),
                "l": float(row["l"]), "c": float(row["c"]),
                "v": int(row["v"]),
            }
    return result


def _meta_path(cache_root: str, ikey: str) -> str:
    return os.path.join(cache_root, ikey, "metadata.json")


def _load_meta(cache_root: str, ikey: str) -> dict:
    p = _meta_path(cache_root, ikey)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"fetched_expiries": []}


def _save_meta(cache_root: str, ikey: str, meta: dict) -> None:
    os.makedirs(os.path.join(cache_root, ikey), exist_ok=True)
    with open(_meta_path(cache_root, ikey), "w") as f:
        json.dump(meta, f, indent=2)


# ── Main fetcher ──────────────────────────────────────────────────────────────

class OptionDataFetcher:
    """
    Orchestrates fetching and caching of real option candle data from Fyers.

    For each weekly expiry:
      1. Determine the ATM strike from the underlying spot at start of expiry week
      2. Generate ±n_strikes strikes at the instrument's interval
      3. Fetch 1-min candles for each (symbol, expiry_period) pair
      4. Cache results to disk as CSV
    """

    def __init__(
        self,
        client: FyersClient,
        ikey: str,
        cache_root: str = CACHE_ROOT,
        n_strikes: int = 10,
        force_refresh: bool = False,
    ):
        self.client       = client
        self.ikey         = ikey
        self.params       = INSTRUMENT_PARAMS[ikey]
        self.cache_root   = cache_root
        self.n_strikes    = n_strikes
        self.force        = force_refresh
        self._semaphore   = asyncio.Semaphore(CONCURRENCY)

    async def _fetch_one(
        self,
        symbol: str,
        from_date: str,
        to_date:   str,
        path:      str,
        http_client: httpx.AsyncClient,
        pbar_desc: str = "",
    ) -> None:
        async with self._semaphore:
            await asyncio.sleep(CALL_DELAY)
            if os.path.exists(path) and not self.force:
                return   # cached
            candles = await self.client.candles(
                symbol, from_date, to_date,
                is_option=True,      # disables cont_flag, uses smaller chunks
                client=http_client,
            )
            _save_candles(path, candles)
            if candles:
                logger.debug("  ✓ %s  %d bars", symbol[-20:], len(candles))

    async def fetch_expiry(
        self,
        expiry_dt:  datetime,
        spot_daily: Dict[date, float],
        http_client: httpx.AsyncClient,
    ) -> List[str]:
        """
        Fetch all option strikes for one expiry. Returns list of symbol names fetched.
        """
        params = self.params
        si     = params["strike_interval"]
        seg    = params["data_prefix"]    # NSE/BSE for data API (not NFO/BFO)
        und    = params["underlying"]

        # Determine the Monday of this expiry week
        days_to_monday  = expiry_dt.weekday()   # 0=Mon, so Thursday=3
        monday          = expiry_dt - timedelta(days=days_to_monday)
        fetch_start_dt  = monday - timedelta(days=7)   # include previous week for context
        fetch_end_dt    = expiry_dt

        from_date = fetch_start_dt.strftime("%Y-%m-%d")
        to_date   = fetch_end_dt.strftime("%Y-%m-%d")

        # Get ATM from spot close on Monday (or nearest available day)
        atm_spot = None
        for delta in range(0, 5):
            check_date = (monday - timedelta(days=delta)).date()
            if check_date in spot_daily:
                atm_spot = spot_daily[check_date]
                break

        if atm_spot is None:
            # Fall back: use any price within ±3 weeks
            nearby = [v for k, v in spot_daily.items()
                      if abs((datetime.combine(k, datetime.min.time()) - expiry_dt).days) < 21]
            atm_spot = nearby[0] if nearby else 24000.0

        atm = atm_from_spot(atm_spot, si)

        # Build list of (symbol, cache_path)
        tasks    = []
        symbols  = []
        for i in range(-self.n_strikes, self.n_strikes + 1):
            strike = atm + i * si
            for otype in ("CE", "PE"):
                sym  = fyers_weekly_symbol(seg, und, expiry_dt, strike, otype)
                path = _candle_path(self.cache_root, self.ikey, expiry_dt, sym)
                symbols.append(sym)
                tasks.append(
                    self._fetch_one(sym, from_date, to_date, path, http_client)
                )

        await asyncio.gather(*tasks)
        return symbols

    async def fetch_all(self, start_date: str, end_date: str) -> None:
        """
        Fetch all weekly expiries between start_date and end_date.
        Shows progress. Skips expiries already in cache (unless --force).
        """
        params  = self.params
        sd      = datetime.strptime(start_date, "%Y-%m-%d").date()
        ed      = datetime.strptime(end_date,   "%Y-%m-%d").date()
        expiries = weekly_expiries(sd, ed, params["expiry_weekday"])

        if not expiries:
            logger.warning("No expiries found in the date range.")
            return

        meta = _load_meta(self.cache_root, self.ikey)
        fetched_set = set(meta.get("fetched_expiries", []))

        logger.info(
            "Instrument: %s  |  expiries: %d  |  strikes/expiry: %d  |  "
            "total symbols: ~%d",
            self.ikey.upper(), len(expiries),
            (2 * self.n_strikes + 1) * 2,
            len(expiries) * (2 * self.n_strikes + 1) * 2,
        )

        # Pre-fetch spot prices (fast, single call)
        logger.info("Fetching spot prices ...")
        spot_daily = await self.client.spot_close(
            params["spot_symbol"], start_date, end_date
        )
        logger.info("  Got %d spot close values.", len(spot_daily))

        async with httpx.AsyncClient(timeout=30, limits=httpx.Limits(
            max_connections=CONCURRENCY + 2, max_keepalive_connections=CONCURRENCY,
        )) as http_client:

            for idx, exp_dt in enumerate(expiries):
                exp_str = exp_dt.strftime("%Y-%m-%d")
                skip_key = exp_str

                if skip_key in fetched_set and not self.force:
                    logger.info(
                        "  [%3d/%d]  %s  (cached — skip)", idx+1, len(expiries), exp_str
                    )
                    continue

                logger.info("  [%3d/%d]  Fetching expiry %s ...", idx+1, len(expiries), exp_str)
                syms = await self.fetch_expiry(exp_dt, spot_daily, http_client)

                fetched_set.add(skip_key)
                meta["fetched_expiries"] = sorted(fetched_set)
                meta["last_updated"]     = datetime.now().isoformat()
                meta["instrument"]       = self.ikey
                meta["n_strikes"]        = self.n_strikes
                _save_meta(self.cache_root, self.ikey, meta)

                non_empty = sum(
                    1 for s in syms
                    if os.path.exists(_candle_path(self.cache_root, self.ikey, exp_dt, s))
                    and os.path.getsize(_candle_path(self.cache_root, self.ikey, exp_dt, s)) > 50
                )
                logger.info("    → %d / %d symbols have real data", non_empty, len(syms))

        logger.info("Done. Cache: %s/%s/", self.cache_root, self.ikey)


# ── RealOptionChainBuilder (used by backtest.py) ──────────────────────────────

class RealOptionChainBuilder:
    """
    Builds OptionChainSnapshot from cached real option data.

    Priority:
      1. Real data from CSV cache (genuine prices + volume)
      2. BS-model fallback for strikes not in cache or with no trades that bar

    The real data provides accurate IV, volume, and bid-ask spreads for
    the alpha signals that synthetic chains cannot replicate.
    """

    def __init__(
        self,
        bs,                       # BlackScholesEngine
        spec,                     # InstrumentSpec
        cache_root: str = CACHE_ROOT,
        synthetic_fallback = None,  # SyntheticChainBuilder for missing data
    ):
        from bs_engine import OptionType
        from data_feed import OptionChainSnapshot, OptionQuote

        self.bs         = bs
        self.spec       = spec
        self.cache_root = cache_root
        self._synth     = synthetic_fallback
        self._OT        = OptionType
        self._OCS       = OptionChainSnapshot
        self._OQ        = OptionQuote

        # Loaded data: {symbol: {ts_str: candle}}
        self._data: Dict[str, Dict[str, dict]] = {}
        self._loaded_expiry: Optional[str] = None   # avoid reloading same expiry

    def _load_expiry(self, expiry_dt) -> None:
        """Load CSV data for a given expiry into memory."""
        exp_str = expiry_dt.strftime("%Y-%m-%d")
        if self._loaded_expiry == exp_str:
            return

        exp_dir = _expiry_dir(self.cache_root, self.spec.name.lower(), expiry_dt)
        self._data.clear()

        if not os.path.isdir(exp_dir):
            self._loaded_expiry = exp_str
            return

        for fname in os.listdir(exp_dir):
            if not fname.endswith(".csv"):
                continue
            path   = os.path.join(exp_dir, fname)
            symbol = fname[:-4].replace("_", ":", 1)   # undo csv_safe_symbol
            self._data[symbol] = _load_candles(path)

        self._loaded_expiry = exp_str
        loaded = sum(1 for v in self._data.values() if v)
        logger.debug(
            "Loaded expiry %s: %d symbols, %d with data",
            exp_str, len(self._data), loaded,
        )

    def build(self, spot, vix_pct, timestamp, expiry_dt, tte):
        """
        Build OptionChainSnapshot.
        Uses real Fyers data for current-expiry bars; falls back to BS model
        for expired/missing strikes (which is ALL bars in historical backtest).
        """
        from bs_engine import OptionType
        from data_feed import OptionChainSnapshot, OptionQuote
        import numpy as np

        self._load_expiry(expiry_dt)

        atm_iv  = max(0.05, vix_pct / 100)
        si      = self.spec.strike_interval
        atm     = round(spot / si) * si
        exp_str = expiry_dt.strftime("%d%b%y").upper()
        pfx     = f"{self.spec.segment}:{self.spec.name}"
        ts_key  = timestamp.strftime("%Y-%m-%d %H:%M:00")

        snap = OptionChainSnapshot(
            underlying=self.spec.name,
            spot_price=spot,
            timestamp=timestamp,
            expiry=exp_str,
            atm_strike=atm,
        )
        if tte < 1e-6:
            return snap

        for i in range(-20, 21):
            strike = atm + i * si
            if strike <= 0:
                continue

            for otype_str, otype_enum, quotes_dict in (
                ("CE", OptionType.CALL, snap.calls),
                ("PE", OptionType.PUT,  snap.puts),
            ):
                # Data API prefix (NSE/BSE) differs from trading segment (NFO/BFO)
                data_pfx = INSTRUMENT_PARAMS.get(
                    self.spec.name.lower(), {}
                ).get("data_prefix", "NSE")
                sym = fyers_weekly_symbol(
                    data_pfx, self.spec.name, expiry_dt, strike, otype_str
                )

                # ── Try real data ────────────────────────────────────────
                real_candle = self._data.get(sym, {}).get(ts_key)
                if real_candle and real_candle.get("c", 0) > 0:
                    price = float(real_candle["c"])
                    vol   = int(real_candle["v"])

                    # Compute real IV from actual traded price
                    iv = self.bs.implied_volatility_newton_raphson(
                        market_price=price,
                        spot=spot, strike=strike,
                        time_to_expiry=tte,
                        option_type=otype_enum,
                    )
                    if iv is None or iv < 0.01:
                        iv = atm_iv + (0.0003 if otype_str == "PE" else 0.00005) * abs(atm - strike)

                    # Simulate bid/ask from 0.3% spread (typical for liquid strikes)
                    spread = max(0.5, price * 0.003)
                    delta  = float(self.bs.delta(spot, strike, tte, iv, otype_enum))

                    quotes_dict[float(strike)] = OptionQuote(
                        symbol=f"{pfx}{exp_str}{int(strike)}{otype_str}",
                        strike=float(strike), expiry=exp_str,
                        option_type=otype_enum,
                        ltp=round(price, 2),
                        bid=round(max(0.05, price - spread), 2),
                        ask=round(price + spread, 2),
                        bid_qty=max(10, vol // 10),
                        ask_qty=max(10, vol // 12),
                        volume=vol,
                        oi=0,   # historical OI not available from Fyers candles
                        iv=iv, delta=delta,
                    )
                    continue

                # ── Fallback: BS model ───────────────────────────────────
                dk = max(0.0, (atm - strike) if otype_str == "PE" else (strike - atm))
                iv = atm_iv + (0.0003 * dk if otype_str == "PE" else 0.00005 * dk)
                iv = max(0.04, iv)

                if otype_str == "CE":
                    price = float(self.bs.call_price(spot, strike, tte, iv))
                else:
                    price = float(self.bs.put_price(spot, strike, tte, iv))
                price = max(0.05, price)

                delta = float(self.bs.delta(spot, strike, tte, iv, otype_enum))
                spread = 0.005 + abs(i) * 0.001

                quotes_dict[float(strike)] = OptionQuote(
                    symbol=f"{pfx}{exp_str}{int(strike)}{otype_str}",
                    strike=float(strike), expiry=exp_str,
                    option_type=otype_enum,
                    ltp=round(price, 2),
                    bid=round(price * (1 - spread), 2),
                    ask=round(price * (1 + spread), 2),
                    bid_qty=50, ask_qty=50, volume=0, oi=0,
                    iv=iv, delta=delta,
                )

        return snap

    @staticmethod
    def is_cache_available(cache_root: str, ikey: str) -> bool:
        """Check if real option cache exists for this instrument."""
        meta = _load_meta(cache_root, ikey)
        return len(meta.get("fetched_expiries", [])) > 0

    @staticmethod
    def cache_stats(cache_root: str, ikey: str) -> dict:
        """Return stats about the current cache."""
        meta     = _load_meta(cache_root, ikey)
        exp_list = meta.get("fetched_expiries", [])
        total_files = 0
        for exp in exp_list:
            exp_dir = os.path.join(cache_root, ikey, exp)
            if os.path.isdir(exp_dir):
                total_files += len([f for f in os.listdir(exp_dir) if f.endswith(".csv")])
        return {
            "expiries_cached": len(exp_list),
            "csv_files":       total_files,
            "last_updated":    meta.get("last_updated", "never"),
            "n_strikes":       meta.get("n_strikes", 0),
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

async def _main(args):
    app_id = os.environ.get("BROKER_APP_ID",       "").strip()
    token  = os.environ.get("BROKER_ACCESS_TOKEN", "").strip()

    if not token:
        print("ERROR: BROKER_ACCESS_TOKEN not set. Run: python3 get_token_browser.py")
        sys.exit(1)

    # Verify token
    print("Verifying token ...", end=" ", flush=True)
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            "https://api-t1.fyers.in/api/v3/profile",
            headers={"Authorization": f"{app_id}:{token}"},
        )
        d = r.json()
    if d.get("s") != "ok":
        print(f"INVALID: {d}")
        sys.exit(1)
    name = (d.get("data") or {}).get("name", "")
    print(f"OK ({name})")

    client   = FyersClient(app_id, token)
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=args.months * 30)).strftime("%Y-%m-%d")

    instruments = args.instruments.split(",") if args.instruments else [args.instrument]

    for ikey in instruments:
        ikey = ikey.strip().lower()
        if ikey not in INSTRUMENT_PARAMS:
            print(f"Unknown instrument: {ikey}. Choose from: {list(INSTRUMENT_PARAMS)}")
            continue

        print(f"\n{'━'*60}")
        print(f"  {ikey.upper()}  {start_date} → {end_date}")
        print(f"{'━'*60}")

        if args.stats:
            stats = RealOptionChainBuilder.cache_stats(args.cache_dir, ikey)
            print(f"  Cache stats: {stats}")
            continue

        fetcher = OptionDataFetcher(
            client       = client,
            ikey         = ikey,
            cache_root   = args.cache_dir,
            n_strikes    = args.n_strikes,
            force_refresh= args.force,
        )
        await fetcher.fetch_all(start_date, end_date)

        # Print final stats
        stats = RealOptionChainBuilder.cache_stats(args.cache_dir, ikey)
        print(f"\n  Cache complete: {stats}")


def main():
    p = argparse.ArgumentParser(
        description="Fetch real Fyers option chain data and cache to disk"
    )
    p.add_argument("--instrument",  default="nifty",
                   choices=list(INSTRUMENT_PARAMS),
                   help="Instrument to fetch (default: nifty)")
    p.add_argument("--instruments", default=None,
                   help="Comma-separated list, e.g. 'nifty,banknifty,sensex'")
    p.add_argument("--months",     type=int, default=12,
                   help="Months of history (default: 12)")
    p.add_argument("--n-strikes",  type=int, default=10,
                   help="Strikes each side of ATM to fetch (default: 10 → 21 strikes)")
    p.add_argument("--cache-dir",  default=CACHE_ROOT,
                   help=f"Cache directory (default: {CACHE_ROOT})")
    p.add_argument("--force",      action="store_true",
                   help="Re-fetch even if already cached")
    p.add_argument("--stats",      action="store_true",
                   help="Show cache stats and exit")
    asyncio.run(_main(p.parse_args()))


if __name__ == "__main__":
    main()
