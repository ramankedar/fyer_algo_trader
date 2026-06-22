#!/usr/bin/env python3
"""
run_barbell_backtest.py — Full Barbell Portfolio Backtest.

Portfolio architecture
----------------------
  THETA SLEEVE (₹2L or ₹1.2L):
    BarbellStrangle on NIFTY (Thursday) + BANKNIFTY (Wednesday).
    Same capital recycled across days — no simultaneous exposure.

  CONVEXITY SLEEVE (₹0.8L, optional):
    WeeklyMomentumBuyer — directional debit spread Monday morning.
    Adds convexity (explosive payoff on trending weeks).

  --full-capital flag:
    Use all ₹2L for theta (no convexity). Each strangle gets bigger
    budget → can size up to 2 lots instead of 1.

Note on pricing
---------------
  Underlying prices : REAL (Fyers API historical data)
  Option prices     : BS-estimated from real VIX. Apply ~15% haircut.
  Leg stops         : Triggered by REAL underlying movement + BS re-price.

Usage
-----
  python3 run_barbell_backtest.py --start 2025-07-01 --end 2026-06-19
  python3 run_barbell_backtest.py --start 2022-01-01 --end 2025-01-01 --no-convexity
  python3 run_barbell_backtest.py --start 2022-01-01 --end 2025-01-01 --full-capital
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from datetime import date

import numpy as np

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("barbell")

from algo_platform.core.config import load_config, PlatformConfig
from algo_platform.core.types import Instrument
from algo_platform.data.downloader import FyersDownloader
from algo_platform.data.loader import MarketDataLoader
from algo_platform.data.chain_builder import SyntheticChainBuilder
from algo_platform.backtest.engine import BacktestEngine
from algo_platform.backtest.metrics import compute_cagr, compute_sharpe, compute_max_drawdown
from algo_platform.strategies import BarbellStrangleStrategy, WeeklyMomentumBuyerStrategy


# ── Config sleeve helpers ─────────────────────────────────────────────────────

def _sleeve(base: PlatformConfig, capital: float,
            risk_pct_override: float = 0.0) -> PlatformConfig:
    """Deep-copy config with capital set to the sleeve amount."""
    cfg = copy.deepcopy(base)
    cfg.risk.capital        = capital
    cfg.risk.margin_reserve = max(10_000, capital * 0.12)
    if risk_pct_override > 0:
        cfg.risk.risk_per_trade_pct = risk_pct_override
    return cfg


# ── Report printer ────────────────────────────────────────────────────────────

def _report(title: str, equity: np.ndarray, capital: float,
            n_days: int, trades: dict) -> None:
    if len(equity) < 2:
        print(f"\n  {title}: insufficient data")
        return
    cagr   = compute_cagr(equity, n_days) if n_days > 0 else 0.0
    rets   = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
    sharpe = compute_sharpe(rets)
    maxdd  = compute_max_drawdown(equity)
    final  = equity[-1]

    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")
    print(f"  Capital (initial) : ₹{capital:,.0f}")
    print(f"  Capital (final)   : ₹{final:,.0f}")
    print(f"  Total P&L         : ₹{final - capital:+,.0f}")
    print(f"  CAGR              : {cagr:+.1%}")
    print(f"  Sharpe            : {sharpe:.2f}")
    print(f"  Max Drawdown      : {maxdd:.1%}")
    for k, v in trades.items():
        print(f"  {k:<18}: {v}")
    print(f"{'='*62}")


# ── Combine equity curves (correct version) ───────────────────────────────────

def _add_pnl(base_capital: float, *equity_curves: np.ndarray) -> np.ndarray:
    """
    Combine multiple equity curves into one starting at base_capital.
    Each curve must start at its own initial capital.
    min_len is taken across all curves so no 2-element dummy array skews things.
    """
    min_len = min(len(e) for e in equity_curves)
    pnl_sum = sum(e[:min_len] - e[0] for e in equity_curves)
    return base_capital + pnl_sum


# ── Main ──────────────────────────────────────────────────────────────────────

def run_barbell(
    start:         date,
    end:           date,
    with_convexity: bool = True,
    full_capital:   bool = False,
) -> None:
    base_cfg  = load_config()
    total_cap = base_cfg.risk.capital             # ₹2.0L

    # Capital allocation
    if full_capital:
        theta_cap = total_cap      # all ₹2L → theta only
        conv_cap  = 0.0
        # With 2× capital, use 2× risk_pct to maintain 2 lots on each trade
        theta_risk_pct = 0.02   # 2% → 2 lots at ₹1800 loss/lot
        with_convexity = False
    else:
        theta_cap = base_cfg.risk.theta_capital       # ₹1.2L
        conv_cap  = base_cfg.risk.convexity_capital   # ₹0.8L
        theta_risk_pct = 0.0    # use config default

    print(f"\n{'━'*62}")
    print(f"  BARBELL PORTFOLIO BACKTEST")
    print(f"{'━'*62}")
    print(f"  Period       : {start} → {end}")
    print(f"  Total capital: ₹{total_cap:,.0f}")
    if full_capital:
        print(f"  Mode         : FULL CAPITAL (all ₹{total_cap:,.0f} → theta, 2 lots)")
    else:
        print(f"  Theta sleeve : ₹{theta_cap:,.0f}  (NIFTY Thu + BANKNIFTY Wed)")
        if with_convexity:
            print(f"  Conv sleeve  : ₹{conv_cap:,.0f}  (Weekly momentum buyer)")
        else:
            print(f"  Conv sleeve  : disabled  (₹{conv_cap:,.0f} stays as margin buffer)")
    print(f"\n  ⚠  Underlying prices: REAL | Option prices: BS-estimated")
    print(f"     Apply ~15% haircut to CAGR for real-world estimate.")

    dl      = FyersDownloader(base_cfg.broker.app_id, base_cfg.broker.access_token,
                              "data/cache")
    ldr     = MarketDataLoader(dl)
    builder = SyntheticChainBuilder(base_cfg.risk_free_rate)

    # ── Load instrument-specific data ────────────────────────────────────────
    # Data mapping:
    #   NIFTY 1-min bars   → used by BarbellStrangleStrategy(NIFTY) on THURSDAYS
    #   BANKNIFTY 1-min    → used by BarbellStrangleStrategy(BANKNIFTY) on WEDNESDAYS
    #
    # Why we load ALL weekdays (not just expiry days) for the synthetic backtest:
    #   The BacktestEngine's feature engine (ATR, VWAP, RV, entropy) requires a
    #   rolling warmup window of 21+ bars. It must see Mon-Fri bars to correctly
    #   compute ATR percentile at Thursday 1:30 PM. Filtering to Thursdays-only
    #   would break the percentile context (every bar would look like "same week").
    print("\nLoading market data...")
    nifty_bars = ldr.load_bars("NIFTY",     start, end, "1")  # ALL weekdays for NIFTY
    bnf_bars   = ldr.load_bars("BANKNIFTY", start, end, "1")  # ALL weekdays for BANKNIFTY
    vix_data   = ldr.load_vix(start, end)
    n_days     = len(nifty_bars) // 375
    print(f"  NIFTY (all weekdays):     {len(nifty_bars):,} bars  → strategy uses Thursdays only")
    print(f"  BANKNIFTY (all weekdays): {len(bnf_bars):,} bars  → strategy uses Wednesdays only")
    print(f"  VIX:                      {len(vix_data)} days  |  Est trading days: {n_days}")

    if not nifty_bars or not bnf_bars:
        print("ERROR: No data. Run 'algo-trade download --start 2020-01-01' first.")
        sys.exit(1)

    lots = 2 if full_capital else 1

    # ── Instrument config: each instrument uses its own expiry day ─────────────
    # Capital is recycled each day — the same ₹ serves all instruments
    # because every trade closes by 3:15 PM before the next day's trade opens.
    theta_instruments = [
        (Instrument.NIFTY,     nifty_bars, "Thursday"),
        (Instrument.BANKNIFTY, bnf_bars,   "Wednesday"),
    ]

    # Load BANKEX and FINNIFTY if requested (--full-capital mode or default)
    bankex_bars   = ldr.load_bars("BANKEX",   start, end, "1")
    finnifty_bars = ldr.load_bars("FINNIFTY", start, end, "1")

    if bankex_bars:
        theta_instruments.append((Instrument.BANKEX,   bankex_bars,   "Monday"))
    if finnifty_bars:
        theta_instruments.append((Instrument.FINNIFTY, finnifty_bars, "Tuesday"))

    # ── Run each instrument independently ─────────────────────────────────────
    instrument_reports = []
    for inst, inst_bars, expiry_day in theta_instruments:
        print(f"\nRunning theta: {inst.value} BarbellStrangle ({expiry_day}, {lots} lot)...")
        inst_cfg   = _sleeve(base_cfg, theta_cap, theta_risk_pct)
        inst_strat = BarbellStrangleStrategy(inst, inst_cfg, quantity=lots)
        inst_report = BacktestEngine(inst_cfg).run(
            inst_strat, inst_bars, chain_builder=builder, vix_by_date=vix_data,
        )
        instrument_reports.append((inst.value, expiry_day, inst_report))
        print(f"  ✓ {inst_report.total_trades} trades | WR={inst_report.win_rate:.0%} | "
              f"CAGR={inst_report.cagr:+.1%} | MaxDD={inst_report.max_drawdown:.1%} | "
              f"Exp=₹{inst_report.expectancy:+,.0f}")

    # ── Theta combined: ALL instruments share the same capital (recycled daily) ─
    nifty_report = instrument_reports[0][2]   # for convexity correlation
    bnf_report   = instrument_reports[1][2]
    theta_equity = _add_pnl(theta_cap,
                            *[r.equity_curve for _, _, r in instrument_reports])

    # ── CONVEXITY LEG (optional) ───────────────────────────────────────────────
    if with_convexity and conv_cap > 0:
        print("Running convexity: WeeklyMomentumBuyer (Monday)...")
        conv_cfg    = _sleeve(base_cfg, conv_cap)
        conv_strat  = WeeklyMomentumBuyerStrategy(Instrument.NIFTY, conv_cfg, quantity=1)
        conv_report = BacktestEngine(conv_cfg).run(
            conv_strat, nifty_bars, chain_builder=builder, vix_by_date=vix_data,
        )
        print(f"  ✓ {conv_report.total_trades} trades | WR={conv_report.win_rate:.0%} | "
              f"CAGR={conv_report.cagr:+.1%} | MaxDD={conv_report.max_drawdown:.1%} | "
              f"Exp=₹{conv_report.expectancy:+,.0f}")
        conv_equity = conv_report.equity_curve
        conv_trades = conv_report.total_trades
    else:
        conv_equity = None
        conv_trades = 0

    # ── COMBINED PORTFOLIO ─────────────────────────────────────────────────────
    if conv_equity is not None:
        # Both sleeves: combine theta PnL + convexity PnL on total capital
        combined_equity = _add_pnl(total_cap,
                                   nifty_report.equity_curve,
                                   bnf_report.equity_curve,
                                   conv_equity)
    else:
        # Theta-only: theta PnL is the whole portfolio gain on total capital
        # Idle conv_cap stays as cash (₹0.8L untouched if not full_capital)
        combined_equity = total_cap + (theta_equity - theta_equity[0])

    # ── PRINT RESULTS ──────────────────────────────────────────────────────────
    for inst_name, expiry_day, r in instrument_reports:
        _report(f"{inst_name} ({expiry_day})",
                r.equity_curve, theta_cap, n_days,
                {"Trades": r.total_trades,
                 "Win rate": f"{r.win_rate:.0%}",
                 "Expectancy": f"₹{r.expectancy:+,.0f}",
                 "Lots per trade": lots})

    total_trades_theta = sum(r.total_trades for _, _, r in instrument_reports)
    _report(f"THETA COMBINED ({lots} lot, {len(instrument_reports)} instruments recycled)",
            theta_equity, theta_cap, n_days,
            {"Total trades": total_trades_theta,
             "Instruments": " + ".join(f"{n}({d})" for n, d, _ in instrument_reports),
             "Capital": "Same ₹ recycled every day — no simultaneous exposure"})

    if with_convexity and conv_equity is not None:
        _report("CONVEXITY — WeeklyMomentumBuyer",
                conv_equity, conv_cap, n_days,
                {"Trades": conv_trades,
                 "Win rate": f"{conv_report.win_rate:.0%}",
                 "Expectancy": f"₹{conv_report.expectancy:+,.0f}"})

    combined_cagr = compute_cagr(combined_equity, n_days) if n_days > 0 else 0.0
    _report("━━ FINAL PORTFOLIO ━━",
            combined_equity, total_cap, n_days,
            {"Theta trades": nifty_report.total_trades + bnf_report.total_trades,
             "Convexity trades": conv_trades,
             "Realworld est. (~-15%)": f"{combined_cagr * 0.85:+.1%}",
             "Max loss/trade (1 lot)": "₹3,600–5,400 (capped by leg stops)"})

    # Correlation check
    if with_convexity and conv_equity is not None and conv_trades > 0:
        min_l = min(len(theta_equity), len(conv_equity))
        t_ret = np.diff(theta_equity[:min_l]) / np.where(
            theta_equity[:min_l-1] > 0, theta_equity[:min_l-1], 1)
        c_ret = np.diff(conv_equity[:min_l]) / np.where(
            conv_equity[:min_l-1] > 0, conv_equity[:min_l-1], 1)
        if np.std(t_ret) > 1e-8 and np.std(c_ret) > 1e-8:
            corr = float(np.corrcoef(t_ret, c_ret)[0, 1])
            hedge = sum(1 for t, c in zip(np.diff(theta_equity), np.diff(conv_equity))
                        if t < 0 and c > 0)
            print(f"\n  Theta/Convexity correlation : {corr:.3f} "
                  f"({'✓ hedge effect' if corr < 0 else 'correlated — not ideal hedge'})")
            print(f"  Hedge days (theta↓, conv↑)  : {hedge}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Barbell Portfolio Backtest")
    p.add_argument("--start",          default="2022-01-01")
    p.add_argument("--end",            default=str(date.today()))
    p.add_argument("--no-convexity",   action="store_true",
                   help="Theta-only: disable convexity sleeve, ₹0.8L stays idle")
    p.add_argument("--full-capital",   action="store_true",
                   help="Use all ₹2L for theta (2 lots), no convexity sleeve")
    args = p.parse_args()

    if args.full_capital and not args.no_convexity:
        args.no_convexity = True   # full-capital implies no-convexity

    run_barbell(
        start          = date.fromisoformat(args.start),
        end            = date.fromisoformat(args.end),
        with_convexity = not args.no_convexity,
        full_capital   = args.full_capital,
    )
