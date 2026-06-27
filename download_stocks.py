"""
Download daily OHLCV for all NSE sector constituent stocks.

This enables the RS Momentum Cascade strategy (Strategy H) —
identifying the strongest stocks within the strongest sectors.

Usage:
  python3 download_stocks.py --start 2020-01-01
  python3 download_stocks.py --sector NIFTYIT --start 2020-01-01
  python3 download_stocks.py --tickers TCS,INFY,HCLTECH --start 2022-01-01
"""

import argparse
import logging
import os
import time
from datetime import date

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_stocks")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NSE stock daily data")
    parser.add_argument("--start",   default="2020-01-01")
    parser.add_argument("--end",     default=str(date.today()))
    parser.add_argument("--sector",  default=None, help="Download one sector only")
    parser.add_argument("--tickers", default=None, help="Comma-separated ticker list")
    parser.add_argument("--cache",   default="data/cache")
    parser.add_argument("--pause",   type=float, default=1.2,
                        help="Pause between API calls (seconds). Default 1.2s")
    args = parser.parse_args()

    client_id    = os.getenv("BROKER_APP_ID", "")       # e.g. 4CVZTA9AEG-200
    access_token = os.getenv("BROKER_ACCESS_TOKEN", "")

    if not client_id or not access_token:
        print("Set BROKER_APP_ID and BROKER_ACCESS_TOKEN in .env before running.")
        print("Refresh token: python3 get_token_browser.py")
        return

    from algo_platform.data.downloader import FyersDownloader
    from algo_platform.data.universe import (
        SECTOR_STOCKS, stocks_in_sector, all_sector_stocks, fyers_eq
    )

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    # Determine ticker list
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    elif args.sector:
        tickers = stocks_in_sector(args.sector.upper())
        if not tickers:
            print(f"Unknown sector '{args.sector}'. Valid: {list(SECTOR_STOCKS)}")
            return
        print(f"Sector {args.sector}: {len(tickers)} stocks")
    else:
        tickers = all_sector_stocks()
        print(f"All sectors: {len(tickers)} stocks")

    dl = FyersDownloader(
        client_id    = client_id,
        access_token = access_token,
        cache_dir    = args.cache,
        rate_limit_sleep = args.pause,
    )

    print(f"\nDownloading {len(tickers)} stocks  ({start} → {end})\n")

    ok, failed = [], []
    for i, ticker in enumerate(tickers, 1):
        fyers_sym = fyers_eq(ticker)

        # Temporarily add this ticker to downloader's symbol map
        from algo_platform.data import downloader as dl_mod
        dl_mod.FYERS_SYMBOLS[ticker] = fyers_sym

        try:
            df = dl.download(ticker, start, end, resolution="D")
            if df.empty:
                failed.append(ticker)
                logger.warning("  [%2d/%d] %-20s — empty (check symbol)", i, len(tickers), ticker)
            else:
                ok.append(ticker)
                logger.info("  [%2d/%d] %-20s — %d bars  [%s → %s]",
                            i, len(tickers), ticker, len(df),
                            df.index[0].date(), df.index[-1].date())
        except Exception as e:
            failed.append(ticker)
            logger.error("  [%2d/%d] %-20s — ERROR: %s", i, len(tickers), ticker, e)

    print(f"\n{'='*50}")
    print(f"  Downloaded: {len(ok)}/{len(tickers)} stocks")
    if failed:
        print(f"  Failed:     {failed}")
    print(f"\nRun RS Momentum backtest:")
    print(f"  python3 backtest_rs_momentum.py --start 2022-01-01")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
