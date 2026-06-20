#!/usr/bin/env python3
"""
run_full_backtest.py — Run all 3 strategies × 3 instruments = 9 P&L reports.

Uses real Fyers historical data (token from get_token_browser.py).
All CSVs saved to backtest_results_output/.

Usage:
    python3 run_full_backtest.py
    python3 run_full_backtest.py --months 12 --capital 500000
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List

import httpx
from dotenv import load_dotenv

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

OUTPUT_DIR = "backtest_results_output"
INSTRUMENTS = ["nifty", "banknifty", "sensex"]
STRATEGIES  = [
    "FixedRR_1to3",
    "CurvatureCreditSpread",
    "SkewHunter",
    "ExpiryShortStrangle",
    "ZenCreditSpread",
    "LyapunovCreditSpread",
]


async def _verify_token() -> tuple[str, str]:
    app_id = os.environ.get("BROKER_APP_ID", "").strip()
    token  = os.environ.get("BROKER_ACCESS_TOKEN", "").strip()
    if not token:
        print("ERROR: BROKER_ACCESS_TOKEN not set. Run: python3 get_token_browser.py")
        sys.exit(1)
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            "https://api-t1.fyers.in/api/v3/profile",
            headers={"Authorization": f"{app_id}:{token}"},
        )
        d = r.json()
    if d.get("s") != "ok":
        print(f"Token invalid or expired: {d}")
        print("Run: python3 get_token_browser.py")
        sys.exit(1)
    name = (d.get("data") or {}).get("name", "")
    print(f"  Token valid — logged in as: {name}\n")
    return app_id, token


def _print_9x_matrix(matrix: Dict[str, Dict[str, dict]], initial_capital: float) -> None:
    """Print 3×3 comparative table: rows=strategies, cols=instruments."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cols = INSTRUMENTS
    rows = STRATEGIES

    W = 100
    print("\n" + "═" * W)
    print("  9-COMBINATION PERFORMANCE MATRIX  (real Fyers data)")
    print("  Rows = Strategies   |   Columns = Instruments")
    print("═" * W)

    # Header
    hdr = f"  {'Strategy':<26}"
    for inst in cols:
        hdr += f"  {inst.upper():^28}"
    print(hdr)

    sub = f"  {'':<26}"
    for _ in cols:
        sub += f"  {'Trades  WR%   P&L%  Sharpe':^28}"
    print(sub)
    print("  " + "─" * (W - 2))

    for strat in rows:
        line = f"  {strat:<26}"
        for inst in cols:
            m = matrix.get(strat, {}).get(inst)
            if m and m.get("trades", 0) > 0:
                cell = (
                    f"{m['trades']:>4}  "
                    f"{m['wr']:>5.1f}%  "
                    f"{m['total']/initial_capital*100:>+6.2f}%  "
                    f"{m.get('sharpe', 0):>5.2f}"
                )
            else:
                cell = "  (no trades)        "
            line += f"  {cell:<28}"
        print(line)

    print("═" * W)
    print(f"\n  Best by Return  : ", end="")
    best_r = ("", "", float("-inf"))
    best_s = ("", "", float("-inf"))
    for strat in rows:
        for inst in cols:
            m = matrix.get(strat, {}).get(inst, {})
            if m.get("trades", 0) > 0:
                r = m["total"] / initial_capital * 100
                s = m.get("sharpe", 0)
                if r > best_r[2]:
                    best_r = (strat, inst, r)
                if s > best_s[2]:
                    best_s = (strat, inst, s)
    if best_r[0]:
        print(f"{best_r[0]} on {best_r[1].upper()} ({best_r[2]:+.2f}%)")
    print(f"  Best by Sharpe  : ", end="")
    if best_s[0]:
        print(f"{best_s[0]} on {best_s[1].upper()} (Sharpe={best_s[2]:.2f})")

    # Save matrix CSV
    import csv
    matrix_path = os.path.join(OUTPUT_DIR, "performance_matrix.csv")
    with open(matrix_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Strategy"] + [f"{i.upper()}_Trades" for i in cols]
                   + [f"{i.upper()}_WinRate%" for i in cols]
                   + [f"{i.upper()}_PnL%" for i in cols]
                   + [f"{i.upper()}_Sharpe" for i in cols]
                   + [f"{i.upper()}_PF" for i in cols]
                   + [f"{i.upper()}_MaxDD%" for i in cols])
        for strat in rows:
            row = [strat]
            for field in ["trades", "wr", "return_pct", "sharpe", "pf", "max_dd"]:
                for inst in cols:
                    m = matrix.get(strat, {}).get(inst, {})
                    row.append(round(m.get(field, 0), 2) if m else "")
            w.writerow(row)
    print(f"\n  Matrix saved → {matrix_path}\n")


async def _main(args):
    print(f"\n{'━'*65}")
    print("  FULL BACKTEST  —  3 Strategies × 3 Instruments  =  9 Reports")
    print(f"{'━'*65}")

    app_id, token = await _verify_token()
    os.environ["BROKER_APP_ID"]       = app_id
    os.environ["BROKER_ACCESS_TOKEN"] = token

    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=args.months * 30)).strftime("%Y-%m-%d")
    print(f"  Period   : {start_date}  →  {end_date}")
    print(f"  Capital  : ₹{args.capital:,.0f}")
    print(f"  Output   : {OUTPUT_DIR}/\n")

    from backtest import run_backtest

    # matrix[strategy_name][instrument_key] = metrics dict
    matrix: Dict[str, Dict[str, dict]] = {s: {} for s in STRATEGIES}

    for ikey in INSTRUMENTS:
        print(f"\n{'━'*65}")
        print(f"  INSTRUMENT: {ikey.upper()}")
        print(f"{'━'*65}")

        csv_name = f"{ikey}_{start_date}_{end_date}.csv"
        result = await run_backtest(
            instrument_key=ikey,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.capital,
            output_csv=csv_name,
            output_dir=OUTPUT_DIR,
        )

        # Populate per-strategy rows in matrix
        by_strat = result.get("by_strategy", {})
        for strat_name, m in by_strat.items():
            m["return_pct"] = m["total"] / args.capital * 100
            if strat_name in matrix:
                matrix[strat_name][ikey] = m

    # Print the 3×3 matrix
    _print_9x_matrix(matrix, args.capital)


def main():
    p = argparse.ArgumentParser(
        description="Full 9-combination backtest (3 strategies × 3 instruments)"
    )
    p.add_argument("--months",  type=int,   default=12)
    p.add_argument("--capital", type=float, default=500_000)
    asyncio.run(_main(p.parse_args()))


if __name__ == "__main__":
    main()
