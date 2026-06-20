#!/usr/bin/env python3
"""
run_backtest_with_token.py — Run real-data backtest using an existing access token.

Use this after running get_token_browser.py to get a token, OR if you already
have BROKER_ACCESS_TOKEN set in your .env from any other source.

The token is valid for the current trading day (resets at midnight).

Usage:
    python3 run_backtest_with_token.py
    python3 run_backtest_with_token.py --all --months 12
    python3 run_backtest_with_token.py --instrument banknifty --months 6
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta

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


async def _main(args):
    # Verify token is available
    token = os.environ.get("BROKER_ACCESS_TOKEN", "").strip()
    app_id = os.environ.get("BROKER_APP_ID", "").strip()

    if not token:
        print("\n  ERROR: BROKER_ACCESS_TOKEN is not set in .env")
        print("  Run this first:  python3 get_token_browser.py")
        sys.exit(1)

    if not app_id:
        print("\n  ERROR: BROKER_APP_ID is not set in .env")
        sys.exit(1)

    print(f"\n{'━'*60}")
    print("  REAL DATA BACKTEST — Using saved access token")
    print(f"{'━'*60}")
    print(f"  Token length : {len(token)} chars  (valid until midnight)")
    print(f"  App ID       : {app_id[:6]}...{app_id[-4:]}\n")

    # Quick token validity check before burning API calls
    import httpx
    print("  Verifying token is still valid ...", end=" ", flush=True)
    async with httpx.AsyncClient(timeout=15) as c:
        r = await c.get(
            "https://api-t1.fyers.in/api/v3/profile",
            headers={"Authorization": f"{app_id}:{token}"},
        )
        d = r.json()
    if d.get("s") != "ok":
        print(f"EXPIRED or INVALID: {d}")
        print("\n  Token has expired. Run:  python3 get_token_browser.py")
        sys.exit(1)
    name = (d.get("data") or {}).get("name", "")
    print(f"OK  (logged in as: {name})")

    from backtest import run_backtest

    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=args.months * 30)).strftime("%Y-%m-%d")

    instruments = (
        ["nifty", "banknifty", "sensex"] if args.all
        else [args.instrument]
    )

    results = []
    for ikey in instruments:
        print(f"\n{'━'*60}")
        print(f"  {ikey.upper()}  {start_date} → {end_date}")
        print(f"{'━'*60}")
        result = await run_backtest(
            instrument_key=ikey,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.capital,
            output_csv=f"real_{ikey}_{start_date}_{end_date}.csv",
        )
        results.append(result)

    # Comparative table
    if len(results) > 1:
        W = 70
        print("\n" + "═" * W)
        print("  COMPARATIVE SUMMARY  (real Fyers data)")
        print("═" * W)
        print(f"  {'Instrument':<22} {'Trades':>6} {'WinRate':>8} "
              f"{'P&L':>12} {'Return':>8} {'MaxDD':>7} {'PF':>6}")
        print("  " + "─" * (W - 2))
        for r in results:
            if not r:
                continue
            print(f"  {r.get('label','')[:22]:<22} {r.get('trades',0):>6} "
                  f"{r.get('win_rate',0):>7.1f}% "
                  f"₹{r.get('total_pnl',0):>10,.0f} "
                  f"{r.get('return_pct',0):>+7.2f}% "
                  f"{r.get('max_dd',0):>6.2f}% "
                  f"{r.get('profit_factor',0):>6.2f}")
        print("═" * W + "\n")


def main():
    p = argparse.ArgumentParser(
        description="Real-data backtest using saved Fyers access token"
    )
    p.add_argument("--instrument", default="nifty",
                   choices=["nifty", "banknifty", "finnifty", "sensex", "bankex"])
    p.add_argument("--all",     action="store_true", help="Run Nifty+BankNifty+Sensex")
    p.add_argument("--months",  type=int,   default=6)
    p.add_argument("--capital", type=float, default=500_000)
    asyncio.run(_main(p.parse_args()))


if __name__ == "__main__":
    main()
