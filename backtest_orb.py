"""
Backtest Strategy D (Opening Range Breakout) and Strategy G (Gap Trade)
using existing 1min NIFTY/BANKNIFTY data.

Usage:
  python backtest_orb.py --strategy orb --instrument NIFTY --start 2022-01-01
  python backtest_orb.py --strategy gap --instrument NIFTY --start 2022-01-01
  python backtest_orb.py --strategy orb --instrument BANKNIFTY --start 2021-01-01
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import date

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("backtest_orb")


def main() -> None:
    parser = argparse.ArgumentParser(description="ORB / Gap Strategy Backtest")
    parser.add_argument("--strategy",   choices=["orb", "gap"], default="orb")
    parser.add_argument("--instrument", default="NIFTY",
                        choices=["NIFTY", "BANKNIFTY", "FINNIFTY", "SENSEX"])
    parser.add_argument("--start",      default="2022-01-01")
    parser.add_argument("--end",        default=str(date.today()))
    parser.add_argument("--capital",    type=float, default=200_000)
    parser.add_argument("--quantity",   type=int,   default=1)
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    cache_dir  = "data/cache"
    instr      = args.instrument.upper()
    cache_file = f"{cache_dir}/{instr}_1min.csv"

    if not os.path.exists(cache_file):
        print(f"ERROR: {cache_file} not found.")
        print("Download data first: python -m algo_platform.run download --start 2020-01-01")
        return

    logger.info("Loading %s 1min data from %s …", instr, cache_file)
    df = pd.read_csv(cache_file, index_col="datetime", parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")

    start = pd.Timestamp(args.start, tz="Asia/Kolkata")
    end   = pd.Timestamp(args.end,   tz="Asia/Kolkata") + pd.Timedelta(days=1)
    df    = df[(df.index >= start) & (df.index < end)].copy()
    logger.info("Loaded %d bars  [%s → %s]", len(df), df.index[0].date(), df.index[-1].date())

    # ── Build instrument + config ─────────────────────────────────────────────
    from algo_platform.core.config import load_config, PlatformConfig
    from algo_platform.core.types import Instrument, MarketBar
    from algo_platform.backtest.engine import BacktestEngine
    from algo_platform.data.chain_builder import SyntheticChainBuilder

    config = load_config()
    config.risk.capital = args.capital

    try:
        instrument = Instrument(instr)
    except ValueError:
        instrument = Instrument.NIFTY

    # ── Select strategy ───────────────────────────────────────────────────────
    if args.strategy == "orb":
        from algo_platform.strategies.orb import ORBStrategy
        strategy = ORBStrategy(
            instrument = instrument,
            config     = config,
            quantity   = args.quantity,
        )
        label = "Opening Range Breakout"
    else:
        from algo_platform.strategies.gap_fade import GapTradeStrategy
        strategy = GapTradeStrategy(
            instrument = instrument,
            config     = config,
            quantity   = args.quantity,
        )
        label = "Gap & Go / Gap Fade"

    # ── Build MarketBar list ───────────────────────────────────────────────────
    bars: list[MarketBar] = []
    for ts, row in df.iterrows():
        bars.append(MarketBar(
            timestamp = ts,
            open      = float(row["open"]),
            high      = float(row["high"]),
            low       = float(row["low"]),
            close     = float(row["close"]),
            volume    = float(row.get("volume", 0)),
        ))

    # ── Load VIX for chain builder ─────────────────────────────────────────────
    vix_by_date = {}
    try:
        vix_df = pd.read_csv(f"{cache_dir}/VIX_daily.csv", index_col=0, parse_dates=True)
        if vix_df.index.tz is None:
            vix_df.index = vix_df.index.tz_localize("Asia/Kolkata")
        for ts, row in vix_df.iterrows():
            vix_by_date[ts.date()] = float(row["close"])
        logger.info("Loaded VIX for %d days", len(vix_by_date))
    except Exception as e:
        logger.warning("VIX daily not found (%s) — using flat 15%% IV for chains", e)

    chain_builder = SyntheticChainBuilder(risk_free_rate=config.risk_free_rate)

    # ── Run backtest ───────────────────────────────────────────────────────────
    engine = BacktestEngine(config)
    logger.info("Running %s on %s …", label, instr)
    report = engine.run(
        strategy      = strategy,
        bars          = bars,
        vix_by_date   = vix_by_date,
        chain_builder = chain_builder,
    )

    # ── Print results ──────────────────────────────────────────────────────────
    _print_report(report, label, instr, args)


def _print_report(report, label: str, instr: str, args) -> None:
    print("\n" + "=" * 65)
    print(f"  {label} — {instr}  [{args.start} → {args.end}]")
    print("=" * 65)

    if report.total_trades == 0:
        print("  No trades generated — check data / filters / date range.")
        print("=" * 65)
        return

    pnl = report.final_capital - report.initial_capital

    print(f"  Trades:         {report.total_trades}")
    print(f"  Win Rate:       {report.win_rate:.1%}")
    print(f"  Avg Win:        ₹{report.avg_win:,.0f}")
    print(f"  Avg Loss:       ₹{report.avg_loss:,.0f}")
    print(f"  Profit Factor:  {report.profit_factor:.2f}")
    print(f"  Expectancy:     ₹{report.expectancy:,.0f}  per trade")
    print(f"  Total P&L:      ₹{pnl:,.0f}")
    print(f"\n  CAGR:           {report.cagr:.1%}")
    print(f"  Sharpe:         {report.sharpe:.2f}")
    print(f"  Sortino:        {report.sortino:.2f}")
    print(f"  Max Drawdown:   {report.max_drawdown:.1%}")
    print(f"  Time in Market: {report.exposure:.1%}")
    print(f"\n  Passes gates:   {'YES ✓' if report.passes_validation else 'NO — ' + report.validation_notes}")
    print("=" * 65)


if __name__ == "__main__":
    main()
