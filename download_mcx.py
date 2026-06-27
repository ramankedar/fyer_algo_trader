"""
Download MCX commodity daily data for Strategy F (Donchian Trend).

Usage:
  python download_mcx.py --start 2020-01-01
  python download_mcx.py --instrument GOLD --start 2020-01-01
"""

import argparse
import logging
import os
from datetime import date

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MCX_KEYS = ["GOLD", "SILVER", "CRUDEOIL", "NATURALGAS", "COPPER", "ZINC"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MCX commodity daily data")
    parser.add_argument("--instrument", default=None, help="Single instrument (default: all)")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end",   default=str(date.today()))
    parser.add_argument("--cache", default="data/cache")
    args = parser.parse_args()

    client_id    = os.getenv("BROKER_APP_ID", "")       # e.g. 4CVZTA9AEG-200
    access_token = os.getenv("BROKER_ACCESS_TOKEN", "")

    if not client_id or not access_token:
        print("Set BROKER_APP_ID and BROKER_ACCESS_TOKEN in .env before running.")
        print("Refresh token: python3 get_token_browser.py")
        return

    from algo_platform.data.downloader import FyersDownloader, refresh_mcx_symbols

    # Auto-detect current near-month contracts (symbols change each expiry cycle)
    print("Refreshing MCX near-month symbols from Fyers symbol master...")
    refreshed = refresh_mcx_symbols(update_module=True)
    if refreshed:
        print(f"  Updated {len(refreshed)} symbols: {', '.join(refreshed)}\n")

    dl = FyersDownloader(
        client_id    = client_id,
        access_token = access_token,
        cache_dir    = args.cache,
    )

    keys  = [args.instrument.upper()] if args.instrument else MCX_KEYS
    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    print(f"\nDownloading MCX commodities: {keys}  ({start} → {end})\n")

    for key in keys:
        try:
            df = dl.download(key, start, end, resolution="D")
            print(f"  {key:16s} — {len(df):5d} daily bars  [{df.index[0].date()} → {df.index[-1].date()}]")
        except Exception as e:
            print(f"  {key:16s} — FAILED: {e}")

    print("\nDone. Run MCX backtest with:")
    print("  python -m algo_platform.strategies.mcx_trend --instrument GOLD")
    print("  python -m algo_platform.strategies.mcx_trend --instrument CRUDEOIL")


if __name__ == "__main__":
    main()
