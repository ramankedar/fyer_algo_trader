#!/usr/bin/env python3
"""
run_theta_sensitivity.py — Parameter sensitivity map for ProductionThetaStrategy.

Tests 36 parameter combinations around the research baseline (13:30 / 0.50 / 4.0)
to answer: does this baseline sit on a stable plateau or a narrow peak?

Grid (otm_sigma fixed at 0.30 throughout):
  Entry time  :  13:00   13:15   13:30*   13:45
  Buffer mult :  0.40    0.50*   0.60
  Cat mult    :  3.0     4.0*    5.0

Each entry time keeps a 10-minute entry window (same width as baseline).
morning_range_end stays at "12:59" (default) for all runs — consistent pre-entry range.

Execution:
  Data is loaded once per worker via OS-cached parquet files (no per-job pickling).
  Workers share the I/O via the existing data/cache/ directory.

Usage:
  python3 run_theta_sensitivity.py --start 2023-01-01 --end 2026-06-19
  python3 run_theta_sensitivity.py --start 2023-01-01 --end 2026-06-19 --workers 6 --capital 120000
"""

from __future__ import annotations

import argparse
import copy
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from typing import Dict, List, Optional, Tuple

# ── Parameter grid ────────────────────────────────────────────────────────────

ENTRY_TIMES   = ["13:00", "13:15", "13:30", "13:45"]
BUFFER_MULTS  = [0.40, 0.50, 0.60]
CAT_MULTS     = [3.0, 4.0, 5.0]
OTM_SIGMA     = 0.30   # locked across all runs

BASELINE      = ("13:30", 0.50, 4.0)

# Fixed 10-minute entry window width, matching baseline (13:30–13:40)
_ENTRY_END: Dict[str, str] = {
    "13:00": "13:10",
    "13:15": "13:25",
    "13:30": "13:40",
    "13:45": "13:55",
}

# ── Worker-level data store (populated by initializer, read-only in jobs) ─────

_BARS: list = []
_VIX:  dict = {}


def _init_worker(start_iso: str, end_iso: str, cache_dir: str) -> None:
    """
    Load NIFTY bars and VIX from disk once per worker process.
    Runs in the worker before any jobs are dispatched — avoids pickling ~90K bars
    for every one of the 36 jobs.
    """
    global _BARS, _VIX
    from algo_platform.core.config import load_config
    from algo_platform.data.downloader import FyersDownloader
    from algo_platform.data.loader import MarketDataLoader

    cfg   = load_config()
    dl    = FyersDownloader(cfg.broker.app_id, cfg.broker.access_token, cache_dir)
    ldr   = MarketDataLoader(dl)
    start = date.fromisoformat(start_iso)
    end   = date.fromisoformat(end_iso)

    _BARS = ldr.load_bars("NIFTY", start, end, "1")
    _VIX  = ldr.load_vix(start, end)


def _run_one(job: Tuple[str, str, float, float, float]) -> dict:
    """
    Worker function: instantiate strategy + engine, run backtest, return flat metrics dict.

    Args:
        job: (entry_time, entry_end, buffer_mult, cat_mult, capital)
    """
    entry_time, entry_end, buffer_mult, cat_mult, capital = job

    try:
        from algo_platform.core.config import load_config
        from algo_platform.core.types import Instrument
        from algo_platform.strategies import ProductionThetaStrategy
        from algo_platform.backtest.engine import BacktestEngine
        from algo_platform.data.chain_builder import SyntheticChainBuilder

        cfg = load_config()
        sleeve = copy.deepcopy(cfg)
        sleeve.risk.capital        = capital
        sleeve.risk.margin_reserve = max(10_000.0, capital * 0.12)

        strategy = ProductionThetaStrategy(
            Instrument("NIFTY"),
            sleeve,
            entry_time       = entry_time,
            entry_end_time   = entry_end,
            buffer_mult      = buffer_mult,
            cat_premium_mult = cat_mult,
            otm_sigma_mult   = OTM_SIGMA,
            # All other params stay at research defaults
        )

        builder = SyntheticChainBuilder(sleeve.risk_free_rate)
        engine  = BacktestEngine(sleeve)
        report  = engine.run(strategy, _BARS, chain_builder=builder, vix_by_date=_VIX)

        return {
            "entry":     entry_time,
            "buffer":    buffer_mult,
            "cat":       cat_mult,
            "trades":    report.total_trades,
            "win_rate":  report.win_rate,
            "exp":       report.expectancy,
            "pf":        report.profit_factor,
            "sharpe":    report.sharpe,
            "cagr":      report.cagr,
            "max_dd":    report.max_drawdown,
            "ok":        True,
        }

    except Exception as exc:
        import traceback
        return {
            "entry":     entry_time,
            "buffer":    buffer_mult,
            "cat":       cat_mult,
            "trades":    0,
            "win_rate":  0.0,
            "exp":       float("-inf"),
            "pf":        0.0,
            "sharpe":    0.0,
            "cagr":      0.0,
            "max_dd":    0.0,
            "ok":        False,
            "error":     f"{exc}\n{traceback.format_exc()}",
        }


# ── Output formatting ─────────────────────────────────────────────────────────

def _sign(v: float) -> str:
    return "+" if v >= 0 else ""


def _fmt_delta(v: float, unit: str = "") -> str:
    if unit == "%":
        return f"{_sign(v)}{v*100:.1f}%"
    if unit == "pct":
        return f"{_sign(v)}{v:.2f}"
    return f"{_sign(v)}{v:,.0f}"


def _is_baseline(r: dict) -> bool:
    return (r["entry"] == BASELINE[0]
            and abs(r["buffer"] - BASELINE[1]) < 1e-9
            and abs(r["cat"]    - BASELINE[2]) < 1e-9)


def _print_table(results: List[dict], start: str, end: str) -> None:
    """Print the full sensitivity table sorted descending by expectancy."""

    # Locate baseline
    base = next((r for r in results if _is_baseline(r)), None)
    if base is None:
        print("  ⚠  Baseline result not found in results.")
        base = {"exp": 0.0, "sharpe": 0.0, "max_dd": 0.0}

    base_exp    = base["exp"]
    base_sharpe = base["sharpe"]
    base_maxdd  = base["max_dd"]

    # Sort descending by expectancy
    sorted_results = sorted(results, key=lambda r: r["exp"], reverse=True)

    # Column widths
    W = 72 + 28   # total ~100 chars

    print(f"\n{'━'*W}")
    print(f"  PARAMETER SENSITIVITY — ProductionThetaStrategy  (NIFTY, σ=0.30 fixed)")
    print(f"  Period  : {start} → {end}     "
          f"Baseline: Entry={BASELINE[0]}  Buffer={BASELINE[1]}  Cat={BASELINE[2]}")
    print(f"{'━'*W}")

    hdr = (f"  {'Entry':<7} {'Buf':>5} {'Cat':>5}  "
           f"{'Trades':>6} {'Win%':>5} {'Exp(₹)':>8} {'PF':>5} {'Sharpe':>7} "
           f"{'CAGR%':>7} {'MaxDD%':>7}  "
           f"{'ΔExp':>8} {'ΔSharpe':>8} {'ΔMaxDD':>8}")
    print(hdr)
    print(f"  {'─'*98}")

    for r in sorted_results:
        if not r["ok"]:
            err = r.get("error", "unknown error")[:50]
            print(f"  {r['entry']:<7} {r['buffer']:>5.2f} {r['cat']:>5.1f}  ERROR: {err}")
            continue

        is_base = _is_baseline(r)

        d_exp    = r["exp"]    - base_exp
        d_sharpe = r["sharpe"] - base_sharpe
        d_maxdd  = r["max_dd"] - base_maxdd

        flag = " ◀ BASELINE" if is_base else ""

        print(
            f"  {r['entry']:<7} {r['buffer']:>5.2f} {r['cat']:>5.1f}  "
            f"{r['trades']:>6}  {r['win_rate']:>4.0%}  {r['exp']:>+8,.0f} "
            f"{min(r['pf'], 9.99):>5.2f}  {r['sharpe']:>7.2f} "
            f"{r['cagr']:>+6.1%} {r['max_dd']:>7.1%}  "
            f"{_fmt_delta(d_exp):>8} {_fmt_delta(d_sharpe, 'pct'):>8} "
            f"{_fmt_delta(d_maxdd, '%'):>8}{flag}"
        )

    print(f"  {'─'*98}")


def _print_robustness(results: List[dict], start: str, end: str) -> None:
    """Compute and print the robustness score."""
    ok_results = [r for r in results if r["ok"]]
    if not ok_results:
        print("  No valid results to score.")
        return

    base = next((r for r in ok_results if _is_baseline(r)), None)
    if base is None:
        print("  Baseline not found.")
        return

    base_exp = base["exp"]
    threshold = 0.90 * base_exp   # within 10% degradation of baseline

    # ── Robustness: % of combos within 10% of baseline expectancy ─────────────
    robust = [r for r in ok_results if r["exp"] >= threshold]
    score  = len(robust) / len(ok_results) * 100

    # ── Additional slices ─────────────────────────────────────────────────────
    profitable = [r for r in ok_results if r["exp"] > 0]
    same_entry = [r for r in ok_results if r["entry"] == BASELINE[0]]
    same_buf   = [r for r in ok_results if abs(r["buffer"] - BASELINE[1]) < 1e-9]
    same_cat   = [r for r in ok_results if abs(r["cat"]    - BASELINE[2]) < 1e-9]

    exp_values = [r["exp"] for r in ok_results]
    exp_min    = min(exp_values)
    exp_max    = max(exp_values)
    exp_range  = exp_max - exp_min

    W = 100
    print(f"\n{'━'*W}")
    print(f"  ROBUSTNESS ANALYSIS")
    print(f"{'━'*W}")
    print(f"  Baseline expectancy         : ₹{base_exp:+,.0f}/trade")
    print(f"  Robustness threshold (≥90%) : ₹{threshold:+,.0f}/trade")
    print()
    print(f"  Robust combinations         : {len(robust):>2}/{len(ok_results)} = {score:.0f}%")
    print(f"  Profitable combinations     : {len(profitable):>2}/{len(ok_results)} = "
          f"{len(profitable)/len(ok_results)*100:.0f}%")
    print()
    print(f"  Expectancy range across grid: ₹{exp_min:+,.0f}  →  ₹{exp_max:+,.0f}  "
          f"(spread = ₹{exp_range:,.0f})")
    print()

    # Slice analysis: which parameter matters most?
    def _slice_stats(group, label):
        if not group:
            print(f"    {label}: no data")
            return
        exps = [r["exp"] for r in group]
        print(f"    {label:<30} min=₹{min(exps):+,.0f}  max=₹{max(exps):+,.0f}  "
              f"mean=₹{sum(exps)/len(exps):+,.0f}")

    print(f"  Expectancy by fixed-parameter slice (other two vary):")
    _slice_stats(same_entry, f"Entry fixed at {BASELINE[0]} (9 combos)")
    _slice_stats(same_buf,   f"Buffer fixed at {BASELINE[1]} (12 combos)")
    _slice_stats(same_cat,   f"Cat fixed at {BASELINE[2]} (12 combos)")

    # Entry time sensitivity: average expectancy per entry time
    print()
    print(f"  Average expectancy by entry time:")
    for et in ENTRY_TIMES:
        grp  = [r for r in ok_results if r["entry"] == et]
        avg  = sum(r["exp"] for r in grp) / len(grp) if grp else 0.0
        flag = " ◀ baseline" if et == BASELINE[0] else ""
        print(f"    Entry {et}:  ₹{avg:+,.0f}/trade (avg over {len(grp)} combos){flag}")

    print()
    print(f"  CONCLUSION:")
    if score >= 80:
        verdict = ("PLATEAU. The baseline sits on a broad, stable region. "
                   f"{score:.0f}% of the parameter neighborhood preserves ≥90% of baseline alpha. "
                   "Strategy is robust to small parameter perturbations.")
    elif score >= 50:
        verdict = (f"MODERATE ROBUSTNESS ({score:.0f}%). Some sensitivity to parameter choice. "
                   "The edge is real but may require recalibration if market regime shifts.")
    else:
        verdict = (f"NARROW PEAK ({score:.0f}%). The baseline appears to be curve-fitted. "
                   "Most neighboring combinations lose significant alpha. "
                   "Do NOT proceed to live trading without re-running attribution on fresh data.")

    # Word-wrap the verdict
    words = verdict.split()
    line, lines = "  ", []
    for w in words:
        if len(line) + len(w) + 1 > 95:
            lines.append(line)
            line = "  "
        line += w + " "
    if line.strip():
        lines.append(line)
    for l in lines:
        print(l)
    print(f"{'━'*W}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(start: date, end: date, capital: float, n_workers: int,
         cache_dir: str) -> None:

    # Build the 36-job list
    jobs: List[Tuple] = [
        (entry, _ENTRY_END[entry], buf, cat, capital)
        for entry, buf, cat in itertools.product(ENTRY_TIMES, BUFFER_MULTS, CAT_MULTS)
    ]
    n_jobs = len(jobs)

    print(f"\n{'━'*72}")
    print(f"  THETA SENSITIVITY SCAN — ProductionThetaStrategy")
    print(f"{'━'*72}")
    print(f"  Instrument  : NIFTY (Thursday expiry only)")
    print(f"  Period      : {start} → {end}")
    print(f"  Capital     : ₹{capital:,.0f}")
    print(f"  Grid        : {len(ENTRY_TIMES)} entry × {len(BUFFER_MULTS)} buffer × "
          f"{len(CAT_MULTS)} cat = {n_jobs} combinations")
    print(f"  Workers     : {n_workers}")
    print(f"  Cache dir   : {cache_dir}")
    print(f"  Fixed param : otm_sigma = {OTM_SIGMA}")
    print(f"  Baseline    : entry={BASELINE[0]}  buffer={BASELINE[1]}  cat={BASELINE[2]}")
    print(f"  Note: morning_range_end='12:59' is identical for all entry times.")
    print()

    t0 = time.perf_counter()

    # Run in parallel
    results: List[dict] = []
    completed = 0

    with ProcessPoolExecutor(
        max_workers   = n_workers,
        initializer   = _init_worker,
        initargs      = (start.isoformat(), end.isoformat(), cache_dir),
    ) as pool:
        future_to_job = {pool.submit(_run_one, job): job for job in jobs}

        for future in as_completed(future_to_job):
            completed += 1
            job = future_to_job[future]
            try:
                res = future.result()
            except Exception as exc:
                entry, entry_end, buf, cat, _ = job
                res = {
                    "entry": entry, "buffer": buf, "cat": cat,
                    "trades": 0, "win_rate": 0.0, "exp": float("-inf"),
                    "pf": 0.0, "sharpe": 0.0, "cagr": 0.0, "max_dd": 0.0,
                    "ok": False, "error": str(exc),
                }

            results.append(res)
            entry, _, buf, cat, _ = job
            status = (f"Exp=₹{res['exp']:+,.0f}" if res["ok"]
                      else f"ERROR: {res.get('error', '')[:40]}")
            print(f"  [{completed:>2}/{n_jobs}] entry={entry} buf={buf:.2f} "
                  f"cat={cat:.1f}  →  {status}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\n  Completed {n_jobs} runs in {elapsed:.1f}s  "
          f"({elapsed/n_jobs:.2f}s/run avg)\n")

    _print_table(results, str(start), str(end))
    _print_robustness(results, str(start), str(end))


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Parameter sensitivity map for ProductionThetaStrategy"
    )
    p.add_argument("--start",    default="2023-01-01",
                   help="Backtest start date (YYYY-MM-DD)")
    p.add_argument("--end",      default=str(date.today()),
                   help="Backtest end date (YYYY-MM-DD)")
    p.add_argument("--capital",  type=float, default=120_000.0,
                   help="Theta sleeve capital in ₹ (default: 120000)")
    p.add_argument("--workers",  type=int,
                   default=min(os.cpu_count() or 4, 8),
                   help="Parallel worker processes (default: min(cpu_count, 8))")
    p.add_argument("--cache-dir", default="data/cache",
                   help="Fyers data cache directory (default: data/cache)")
    args = p.parse_args()

    main(
        start     = date.fromisoformat(args.start),
        end       = date.fromisoformat(args.end),
        capital   = args.capital,
        n_workers = args.workers,
        cache_dir = args.cache_dir,
    )
