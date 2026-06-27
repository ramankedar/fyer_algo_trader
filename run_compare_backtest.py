#!/usr/bin/env python3
"""
run_compare_backtest.py — Comparison: Real NSE data vs Synthetic, multiple periods.

Part 1 (Real-data): NIFTY + BANKNIFTY using actual NSE bhavcopy settlement prices.
  NIFTY   → NSE bhavcopy Thursday files (real settlement)
  BANKNIFTY → NSE bhavcopy Wednesday files (real settlement)

Part 2 (Synthetic): All 4 indices using Fyers 1-min data + BS pricing for 1 year.
  Note: BANKEX/FINNIFTY exits are BS-estimated (BSE indices not in NSE bhavcopy).

Usage: python3 run_compare_backtest.py
"""

import copy, math, sys, time
from collections import defaultdict
from datetime import date
from scipy.stats import norm

from algo_platform.core.config import load_config, LOT_SIZES
from algo_platform.core.types import Instrument
from algo_platform.data.downloader import FyersDownloader
from algo_platform.data.loader import MarketDataLoader
from algo_platform.data.real_options import NseBhavcopDownloader
from algo_platform.data.chain_builder import SyntheticChainBuilder
from algo_platform.backtest.engine import BacktestEngine
from algo_platform.strategies import BarbellStrangleStrategy

cfg = load_config()
dl  = FyersDownloader(cfg.broker.app_id, cfg.broker.access_token, "data/cache")
ldr = MarketDataLoader(dl)
bh  = NseBhavcopDownloader("nse_option_cache")

THETA = 120_000   # ₹1.2L theta capital per instrument


# ── Helper: Black-Scholes pricer ──────────────────────────────────────────────

def _bs(S, K, T, sigma, r=0.065, is_call=True):
    if T < 1e-8 or sigma < 0.01:
        return max(0.0, S - K) if is_call else max(0.0, K - S)
    sq = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    if is_call:
        return max(0.0, S * float(norm.cdf(d1)) - K * math.exp(-r * T) * float(norm.cdf(d2)))
    return max(0.0, K * math.exp(-r * T) * float(norm.cdf(-d2)) - S * float(norm.cdf(-d1)))


def _tte_years(ts):
    mins_left = max(1, (15 * 60 + 30) - (ts.hour * 60 + ts.minute))
    return mins_left / (375 * 252 * 60 / 60)


def _get_settlement(bhavcopy, sym, expiry_date, strike, opt_type):
    if bhavcopy is None:
        return None
    sub = bhavcopy[
        (bhavcopy["underlying"] == sym) &
        (bhavcopy["expiry"] == expiry_date) &
        (abs(bhavcopy["strike"] - strike) < 0.5) &
        (bhavcopy["option_type"] == opt_type)
    ]
    if sub.empty:
        return None
    v = float(sub["settlement"].iloc[0])
    # Old bhavcopy format: settlement = underlying index level
    idx_sub = bhavcopy[(bhavcopy["underlying"] == sym) & (bhavcopy["expiry"] == expiry_date)]
    if not idx_sub.empty and idx_sub["settlement"].std() < 1.0 and v > 1000:
        return max(0.0, (v - strike) if opt_type == "CE" else (strike - v))
    return max(0.0, v) if v >= 0 else float(sub["close"].iloc[0])


# ── Part 1: Real-data backtest ────────────────────────────────────────────────

def run_real(instrument: str, start: date, end: date) -> dict:
    """
    Backtest BarbellStrangle using real NSE bhavcopy settlement for exits.
    Entry: BS-estimated (real VIX). Leg stops: real underlying + BS.
    """
    SPEC = {
        "NIFTY":     {"step": 50.0,  "wd": 3, "sym": "NIFTY"},
        "BANKNIFTY": {"step": 100.0, "wd": 2, "sym": "BANKNIFTY"},
    }
    spec = SPEC[instrument]
    step, wd, sym = spec["step"], spec["wd"], spec["sym"]
    lot_size = LOT_SIZES[instrument].lot_size

    # Load only expiry-day bars for this instrument
    all_bars = ldr.load_bars(instrument, start, end, "1")
    bars = [b for b in all_bars if b.timestamp.weekday() == wd]
    vix  = ldr.load_vix(start, end)

    bars_by_date = defaultdict(list)
    for b in bars:
        bars_by_date[b.timestamp.date()].append(b)

    nav = THETA
    trade_pnls = []
    real_exit_count = 0

    for d, day_bars in sorted(bars_by_date.items()):
        vix_val = vix.get(d, 14.0)
        iv = vix_val / 100.0

        # Find 1:30 PM entry bar
        target_min = 13 * 60 + 30
        eb = min(
            (b for b in day_bars if abs(b.timestamp.hour * 60 + b.timestamp.minute - target_min) <= 5),
            key=lambda b: abs(b.timestamp.hour * 60 + b.timestamp.minute - target_min),
            default=None,
        )
        if eb is None:
            continue

        spot = eb.close
        atm  = round(spot / step) * step
        tte_e = _tte_years(eb.timestamp)

        # OTM strikes at 0.30σ
        two_hr_move = spot * iv * math.sqrt(tte_e)
        otm_off = max(step, round(two_hr_move * 0.30 / step) * step)
        sc = round((atm + otm_off) / step) * step
        sp = round((atm - otm_off) / step) * step

        ce_entry = _bs(spot, sc, tte_e, iv, is_call=True)
        pe_entry = _bs(spot, sp, tte_e, iv, is_call=False)
        if ce_entry < 1.0 or pe_entry < 1.0:
            continue

        # Load real settlement prices from bhavcopy
        bhavcopy = bh.load(d)
        ce_settle = _get_settlement(bhavcopy, sym, d, sc, "CE")
        pe_settle = _get_settlement(bhavcopy, sym, d, sp, "PE")

        # Leg stop simulation using real underlying movement + BS
        ce_stopped = pe_stopped = None
        for bar in day_bars:
            if bar.timestamp <= eb.timestamp:
                continue
            tte = _tte_years(bar.timestamp)
            if not ce_stopped and _bs(bar.close, sc, tte, iv, True) >= ce_entry * 1.30:
                ce_stopped = _bs(bar.close, sc, tte, iv, True)
            if not pe_stopped and _bs(bar.close, sp, tte, iv, False) >= pe_entry * 1.30:
                pe_stopped = _bs(bar.close, sp, tte, iv, False)

        # Exit prices: stopped leg = BS at stop, other leg = real settlement
        last_close = day_bars[-1].close
        ce_exit = ce_stopped if ce_stopped else (
            ce_settle if ce_settle is not None
            else _bs(last_close, sc, 1e-6, iv, True)
        )
        pe_exit = pe_stopped if pe_stopped else (
            pe_settle if pe_settle is not None
            else _bs(last_close, sp, 1e-6, iv, False)
        )

        if ce_settle is not None and not ce_stopped:
            real_exit_count += 1

        credit = (ce_entry + pe_entry) * lot_size
        cost   = (ce_exit  + pe_exit)  * lot_size
        tx     = 300.0  # approx transaction costs
        pnl    = credit - cost - tx
        nav   += pnl
        trade_pnls.append(pnl)

    if not trade_pnls:
        return {"n": 0}

    wins   = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p <= 0]
    years  = len(trade_pnls) / 52.0
    ratio  = nav / THETA
    cagr   = (ratio ** (1 / years) - 1) if (years > 0 and ratio > 0) else (ratio - 1)

    return {
        "n":       len(trade_pnls),
        "wr":      len(wins) / len(trade_pnls),
        "exp":     sum(trade_pnls) / len(trade_pnls),
        "pnl":     sum(trade_pnls),
        "cagr":    cagr,
        "avg_win": sum(wins)   / max(1, len(wins)),
        "avg_loss":sum(losses) / max(1, len(losses)),
        "real_pct":real_exit_count / len(trade_pnls),
    }


# ── Part 2: Synthetic backtest ─────────────────────────────────────────────────

def run_synthetic(instrument: str, inst_enum: Instrument, start: date, end: date) -> dict:
    """Synthetic backtest using Fyers 1-min data + BS pricing for entry and exit."""
    bars = ldr.load_bars(instrument, start, end, "1")
    vix  = ldr.load_vix(start, end)
    builder = SyntheticChainBuilder(cfg.risk_free_rate)
    cfg2 = copy.deepcopy(cfg)
    cfg2.risk.capital = THETA
    cfg2.risk.margin_reserve = 18_000
    strat  = BarbellStrangleStrategy(inst_enum, cfg2, quantity=1)
    report = BacktestEngine(cfg2).run(strat, bars, chain_builder=builder, vix_by_date=vix)
    return {
        "n":    report.total_trades,
        "wr":   report.win_rate,
        "exp":  report.expectancy,
        "pnl":  report.final_capital - THETA,
        "cagr": report.cagr,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── PART 1: Real-data, NIFTY + BANKNIFTY, multiple periods ───────────────
    print("\n" + "="*70)
    print("  PART 1 — REAL NSE DATA (bhavcopy settlement exits)")
    print("  NIFTY Thursday | BANKNIFTY Wednesday | 1yr / 2yr / 3yr")
    print("="*70)
    print(f"{'Instrument':<12} {'Period':<6} {'Trades':>7} {'WR':>5} "
          f"{'Exp/trade':>10} {'Total PnL':>11} {'CAGR/₹1.2L':>11} {'Real exits':>11}")
    print("-"*70)

    PERIODS = [
        ("1yr", date(2025, 6, 25), date(2026, 6, 19)),
        ("2yr", date(2024, 6, 25), date(2026, 6, 19)),
        ("3yr", date(2023, 6, 25), date(2026, 6, 19)),
    ]

    for inst in ["NIFTY", "BANKNIFTY"]:
        for label, s, e in PERIODS:
            r = run_real(inst, s, e)
            if not r.get("n"):
                print(f"{inst:<12} {label:<6}  NO DATA")
                continue
            print(f"{inst:<12} {label:<6} {r['n']:>7} {r['wr']:>5.0%} "
                  f"{r['exp']:>+10,.0f} {r['pnl']:>+11,.0f} "
                  f"{r['cagr']:>+11.1%} {r['real_pct']:>11.0%}")
            sys.stdout.flush()

    # ── PART 2: Synthetic, all 4 indices, 1yr ─────────────────────────────────
    print("\n" + "="*70)
    print("  PART 2 — SYNTHETIC (Fyers 1-min + BS pricing, 1yr Jun25-Jun26)")
    print("  All 4 indices | Exit prices: BS-estimated (not real settlement)")
    print("  BANKEX/FINNIFTY: BSE/NSE index, no NSE bhavcopy available")
    print("="*70)
    print(f"{'Instrument':<12} {'Expiry':>10} {'Trades':>7} {'WR':>5} "
          f"{'Exp/trade':>10} {'Total PnL':>11} {'CAGR/₹1.2L':>11} {'Data source':>14}")
    print("-"*70)

    S1, E1 = date(2025, 6, 25), date(2026, 6, 19)
    SYNTH_INSTS = [
        ("NIFTY",     Instrument.NIFTY,     "Thursday",  "NSE+BS"),
        ("BANKNIFTY", Instrument.BANKNIFTY, "Wednesday", "NSE+BS"),
        ("BANKEX",    Instrument.BANKEX,    "Monday",    "BSE+BS only"),
        ("FINNIFTY",  Instrument.FINNIFTY,  "Tuesday",   "NSE+BS"),
    ]
    for inst_name, inst_enum, expiry, note in SYNTH_INSTS:
        r = run_synthetic(inst_name, inst_enum, S1, E1)
        print(f"{inst_name:<12} {expiry:>10} {r['n']:>7} {r['wr']:>5.0%} "
              f"{r['exp']:>+10,.0f} {r['pnl']:>+11,.0f} "
              f"{r['cagr']:>+11.1%} {note:>14}")
        sys.stdout.flush()

    print()
    print("  KEY: Real-data exits are conservative (settlement = intrinsic only,")
    print("       no time value). Synthetic exits include residual time value (+10-15%).")


if __name__ == "__main__":
    main()
