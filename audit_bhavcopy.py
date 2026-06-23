#!/usr/bin/env python3
"""
audit_bhavcopy.py — NSE bhavcopy cache coverage audit.

Scans nse_option_cache/ against all trade dates used in verification and
reports which files are missing, which have content for the expected instrument,
and gives a copy-pasteable list of missing dates to download.

Usage:
  python3 audit_bhavcopy.py --start 2023-01-01 --end 2026-06-19
  python3 audit_bhavcopy.py --cache-dir /custom/path/to/cache
"""

from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

_BNF_CUTOFF = date(2024, 11, 13)   # last Wednesday with BANKNIFTY weekly options

INSTRUMENTS = {
    "NIFTY":     {"weekday": 3, "nse_sym": "NIFTY",     "cutoff": None},
    "BANKNIFTY": {"weekday": 2, "nse_sym": "BANKNIFTY", "cutoff": _BNF_CUTOFF},
}


def _required_dates(start: date, end: date) -> dict:
    out = {k: [] for k in INSTRUMENTS}
    d = start
    while d <= end:
        for inst, cfg in INSTRUMENTS.items():
            if d.weekday() == cfg["weekday"]:
                cutoff = cfg["cutoff"]
                if cutoff is None or d <= cutoff:
                    out[inst].append(d)
        d += timedelta(days=1)
    return out


def _check_file(path: Path, inst: str, trade_date: date) -> str:
    """Return 'ok', 'missing', 'no_data', or 'no_expiry_match'."""
    if not path.exists():
        return "missing"
    try:
        df = pd.read_parquet(path)
        sym = INSTRUMENTS[inst]["nse_sym"]
        sub = df[df["underlying"] == sym]
        if sub.empty:
            return "no_data"
        # For 0DTE options, the bhavcopy should have rows with expiry = trade_date
        exp_match = sub[sub["expiry"] == trade_date]
        if exp_match.empty:
            return "no_expiry_match"
        return "ok"
    except Exception as e:
        return f"corrupt({e})"


def main(start: date, end: date, cache_dir: str, verbose: bool = False) -> None:
    cache = Path(cache_dir)
    required = _required_dates(start, end)

    print(f"\n{'━'*62}")
    print(f"  BHAVCOPY CACHE AUDIT")
    print(f"{'━'*62}")
    print(f"  Cache dir : {cache_dir}")
    print(f"  Period    : {start} → {end}")
    print()

    all_missing: List[date] = []
    all_no_expiry: List[date] = []

    for inst, dates in required.items():
        cutoff  = INSTRUMENTS[inst]["cutoff"]
        results = {d: _check_file(cache / f"{d}.parquet", inst, d) for d in dates}

        ok           = [d for d, s in results.items() if s == "ok"]
        missing      = [d for d, s in results.items() if s == "missing"]
        no_data      = [d for d, s in results.items() if s == "no_data"]
        no_exp       = [d for d, s in results.items() if s == "no_expiry_match"]
        corrupt      = [d for d, s in results.items() if s.startswith("corrupt")]

        coverage = len(ok) / len(dates) * 100 if dates else 0
        all_missing.extend(missing)
        all_no_expiry.extend(no_exp)

        cutoff_note = f" (discontinued after {cutoff})" if cutoff else ""
        print(f"  {inst}{cutoff_note}")
        print(f"  {'─'*48}")
        print(f"    Required dates : {len(dates)}")
        print(f"    ✓ Full match   : {len(ok):3d}  ({coverage:.0f}%)")
        print(f"    ✗ Missing file : {len(missing):3d}")
        print(f"    ✗ No inst data : {len(no_data):3d}")
        print(f"    ✗ No exp match : {len(no_exp):3d}  ← bhavcopy file exists but options"
              f"\n                              have different expiry (e.g., monthly only)")
        if corrupt:
            print(f"    ✗ Corrupt      : {len(corrupt):3d}")
        print()

        if verbose and missing:
            print(f"    Missing files:")
            for d in missing:
                print(f"      {d}")
        if verbose and no_exp:
            print(f"    No expiry match (file exists but no 0DTE options):")
            for d in no_exp[:10]:
                print(f"      {d}")
            if len(no_exp) > 10:
                print(f"      ... and {len(no_exp)-10} more")
        if verbose and (missing or no_exp):
            print()

    # Summary
    print(f"  {'━'*58}")
    print(f"  SUMMARY")
    print(f"  {'━'*58}")
    if not all_missing and not all_no_expiry:
        print(f"  ✓ All required files present with matching 0DTE options.")
    else:
        if all_missing:
            print(f"  ✗ {len(all_missing)} files missing from cache entirely.")
        if all_no_expiry:
            print(f"  ✗ {len(all_no_expiry)} files exist but have no 0DTE option expiry match.")
            print(f"     (These fall back to intrinsic-value exit in the verifier.)")

    download_needed = sorted(set(all_missing))
    if download_needed:
        print()
        print(f"  Files to download ({len(download_needed)} dates):")
        for d in download_needed:
            print(f"    {d}")
        print()
        print(f"  Download command:")
        print(f"    python3 -c \"")
        print(f"    from datetime import date")
        print(f"    from algo_platform.data.real_options import NseBhavcopDownloader")
        print(f"    dl = NseBhavcopDownloader('nse_option_cache')")
        print(f"    # Download Thursdays (NIFTY) and Wednesdays (BANKNIFTY pre-2024-11-13)")
        print(f"    dl.download_range(date({start.year},{start.month},{start.day}),")
        print(f"                      date({end.year},{end.month},{end.day}), expiry_weekday=3)")
        print(f"    dl.download_range(date({start.year},{start.month},{start.day}),")
        print(f"                      date(2024,11,13), expiry_weekday=2)")
        print(f"    \"")
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Audit NSE bhavcopy cache coverage")
    p.add_argument("--start",     default="2023-01-01")
    p.add_argument("--end",       default=str(date.today()))
    p.add_argument("--cache-dir", default="nse_option_cache")
    p.add_argument("--verbose",   action="store_true",
                   help="List every missing/mismatched date")
    args = p.parse_args()
    main(date.fromisoformat(args.start), date.fromisoformat(args.end),
         args.cache_dir, args.verbose)
