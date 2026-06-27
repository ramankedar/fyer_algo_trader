"""
Backtest Strategy H: RS Momentum Cascade.

Requires sector index data + stock daily data.
Run downloads first:
  python3 download_sectors.py --start 2020-01-01
  python3 download_stocks.py  --start 2020-01-01

Usage:
  python3 backtest_rs_momentum.py --start 2022-01-01 --end 2026-01-01
  python3 backtest_rs_momentum.py --signals          # today's live signals
"""

import argparse
import logging
import sys
from datetime import date

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from algo_platform.strategies.rs_momentum import RSMomentumBacktester


def main() -> None:
    parser = argparse.ArgumentParser(description="RS Momentum Cascade Backtest")
    parser.add_argument("--start",       default="2022-01-01")
    parser.add_argument("--end",         default=str(date.today()))
    parser.add_argument("--capital",     type=float, default=200_000)
    parser.add_argument("--max_stocks",  type=int,   default=3)
    parser.add_argument("--hold_days",   type=int,   default=5)
    parser.add_argument("--rs_lookback", type=int,   default=5)
    parser.add_argument("--min_rs",      type=float, default=0.005)
    parser.add_argument("--stop_pct",    type=float, default=0.03)
    parser.add_argument("--signals",     action="store_true",
                        help="Print today's live signals only")
    args = parser.parse_args()

    # Use validated focused universe by default (PF=1.54 backtest, PF=1.04 OOS)
    from algo_platform.data.universe import RS_MOMENTUM_UNIVERSE
    import algo_platform.data.universe as u
    u.SECTOR_STOCKS = RS_MOMENTUM_UNIVERSE

    bt = RSMomentumBacktester(
        cache_dir      = "data/cache",
        rs_lookback    = args.rs_lookback,
        min_rs_excess  = 0.003,
        min_vol_ratio  = 0.8,
        max_stocks     = args.max_stocks,
        hold_days      = args.hold_days,
        stop_pct       = args.stop_pct,
        capital        = args.capital,
    )

    if args.signals:
        bt.print_todays_signals()
        return

    try:
        result = bt.run(
            start = date.fromisoformat(args.start),
            end   = date.fromisoformat(args.end),
        )
        bt.print_report(result)

    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
