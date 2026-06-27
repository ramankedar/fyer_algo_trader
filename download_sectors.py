"""
Download NSE sector index daily data for Strategy E (Sector Momentum Rotation).

Requires a valid Fyers access token.
Usage:
  python download_sectors.py --start 2020-01-01
"""

import argparse
import logging
import os
from datetime import date

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SECTOR_KEYS = [
    "NIFTYIT", "NIFTYAUTO", "NIFTYMETAL", "NIFTYPHARMA",
    "NIFTYFMCG", "NIFTYENERGY", "NIFTYREALTY", "NIFTYINFRA",
    "NIFTYMEDIA",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NSE sector index daily data")
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

    from algo_platform.data.downloader import FyersDownloader

    dl = FyersDownloader(
        client_id    = client_id,
        access_token = access_token,
        cache_dir    = args.cache,
    )

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    print(f"\nDownloading {len(SECTOR_KEYS)} sector indices ({start} → {end})\n")

    for key in SECTOR_KEYS:
        try:
            df = dl.download(key, start, end, resolution="D")
            print(f"  {key:18s} — {len(df):5d} daily bars  [{df.index[0].date()} → {df.index[-1].date()}]")
        except Exception as e:
            print(f"  {key:18s} — FAILED: {e}")

    print("\nDone. Run sector backtest with:")
    print("  python -m algo_platform.strategies.sector_momentum")


if __name__ == "__main__":
    main()
