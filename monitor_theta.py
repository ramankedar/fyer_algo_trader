#!/usr/bin/env python3
"""
monitor_theta.py — Terminal dashboard for ProductionTheta live/paper session.

Reads the JSONL trade log written by run_live_theta.py and displays:
  - Today's open positions + P&L
  - Historical trade summary (this week, all time)
  - CAGR, Sharpe, expectancy from all closed trades

Usage:
  python3 monitor_theta.py                    # today's date
  python3 monitor_theta.py --date 2026-06-26  # specific date
  python3 monitor_theta.py --all              # full history analysis
  python3 monitor_theta.py --watch            # refresh every 30s (like top)

Requires no live connection — reads log files only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional


TRADES_DIR = "trades"
CAPITAL    = 120_000.0   # per strategy sleeve


def _load_events(trades_dir: str, day: Optional[date] = None,
                 mode: str = "paper") -> List[dict]:
    """Load all JSONL events from one or all trade log files."""
    events = []
    p = Path(trades_dir)
    if not p.exists():
        return events

    if day:
        path = p / f"theta_{mode}_{day}.jsonl"
        files = [path] if path.exists() else []
    else:
        files = sorted(p.glob(f"theta_{mode}_*.jsonl"))

    for f in files:
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return events


def _pair_trades(events: List[dict]) -> List[dict]:
    """Match entry and exit events into complete trade dicts."""
    entries = {e["trade_id"]: e for e in events if e.get("event") == "entry"}
    exits   = {e["trade_id"]: e for e in events if e.get("event") == "exit"}
    trades  = []
    for tid, entry in entries.items():
        t = {**entry}
        if tid in exits:
            t.update(exits[tid])
            t["closed"] = True
        else:
            t["closed"]  = False
            t["pnl_net"] = None
        trades.append(t)
    return sorted(trades, key=lambda t: t.get("logged_at", ""))


def _cagr(pnls: List[float], dates: List[date], capital: float) -> float:
    if not pnls or not dates:
        return 0.0
    total = sum(pnls)
    days  = (max(dates) - min(dates)).days or 1
    years = days / 365.25
    nav   = capital + total
    return (nav / capital) ** (1 / years) - 1 if years > 0 else 0.0


def _sharpe(pnls: List[float], capital: float, freq: float = 52.0) -> float:
    if len(pnls) < 2:
        return 0.0
    import statistics
    rets = [p / capital for p in pnls]
    mu   = statistics.mean(rets)
    sd   = statistics.stdev(rets) or 1e-9
    return (mu / sd) * math.sqrt(freq)


def display(trades: List[dict], title: str) -> None:
    closed  = [t for t in trades if t["closed"]]
    open_   = [t for t in trades if not t["closed"]]
    pnls    = [t["pnl_net"] for t in closed if t["pnl_net"] is not None]
    wins    = [p for p in pnls if p > 0]
    losses  = [p for p in pnls if p <= 0]

    print(f"\n{'━'*70}")
    print(f"  {title}")
    print(f"{'━'*70}")

    # Open positions
    if open_:
        print(f"\n  OPEN POSITIONS ({len(open_)}):")
        for t in open_:
            print(f"    [{t['strategy']}]  SC={t.get('sc_strike','-')}  "
                  f"SP={t.get('sp_strike','-')}  "
                  f"credit=₹{(t.get('credit_actual',0)*75):.0f}  "
                  f"buf={t.get('buffer',0):.0f}pts")
    else:
        print(f"\n  No open positions.")

    # Closed today / in range
    if closed:
        print(f"\n  CLOSED TRADES ({len(closed)}):")
        print(f"  {'Strategy':<12} {'SC':>6} {'SP':>6} {'PnL':>9} {'Reason':<20}")
        print(f"  {'─'*58}")
        for t in closed[-20:]:
            pnl_str = f"₹{t['pnl_net']:+,.0f}" if t['pnl_net'] is not None else "—"
            print(f"  {t['strategy']:<12} {t.get('sc_strike',0):>6.0f} "
                  f"{t.get('sp_strike',0):>6.0f} {pnl_str:>9} "
                  f"{t.get('exit_reason','open'):<20}")

    # Summary stats
    if pnls:
        total    = sum(pnls)
        avg_win  = sum(wins)  / len(wins)   if wins   else 0
        avg_loss = sum(losses)/ len(losses) if losses else 0
        wr       = len(wins) / len(pnls) * 100
        exp      = total / len(pnls)

        trade_dates = []
        for t in closed:
            d = t.get("date") or ""
            if d:
                try: trade_dates.append(date.fromisoformat(d))
                except: pass

        cagr   = _cagr(pnls, trade_dates, CAPITAL) if trade_dates else 0.0
        sharpe = _sharpe(pnls, CAPITAL)

        print(f"\n  PERFORMANCE SUMMARY")
        print(f"  {'─'*50}")
        print(f"    Closed trades      : {len(closed)}")
        print(f"    Win rate           : {wr:.0f}%  ({len(wins)}W / {len(losses)}L)")
        print(f"    Expectancy         : ₹{exp:+,.0f}/trade")
        print(f"    Avg win            : ₹{avg_win:+,.0f}")
        print(f"    Avg loss           : ₹{avg_loss:+,.0f}")
        print(f"    Total P&L (net)    : ₹{total:+,.0f}")
        print(f"    CAGR (per strat)   : {cagr:+.1%}")
        print(f"    Sharpe             : {sharpe:.2f}")

        # Slippage analysis (model vs actual fills)
        entry_slips = [t.get("fill_slippage", 0) for t in closed]
        exit_slips  = [t.get("exit_slippage",  0) for t in closed]
        if any(s != 0 for s in entry_slips):
            avg_entry_slip = sum(entry_slips) / len(entry_slips)
            avg_exit_slip  = sum(exit_slips)  / len(exit_slips)
            print(f"\n  FILL SLIPPAGE (model bid − actual fill):")
            print(f"    Entry (per lot)    : ₹{avg_entry_slip*75:+.0f}  "
                  f"({avg_entry_slip:+.2f}/unit avg)")
            print(f"    Exit  (per lot)    : ₹{avg_exit_slip*75:+.0f}  "
                  f"({avg_exit_slip:+.2f}/unit avg)")
            print(f"    Total drag/trade   : ₹{(avg_entry_slip+avg_exit_slip)*75:+.0f}")

    print(f"\n{'━'*70}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Monitor ProductionTheta trades")
    p.add_argument("--date",  default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--all",   action="store_true", help="Full history analysis")
    p.add_argument("--live",  action="store_true", help="Show live trades (default: paper)")
    p.add_argument("--watch", action="store_true", help="Refresh every 30s")
    p.add_argument("--trades-dir", default=TRADES_DIR)
    args = p.parse_args()

    mode = "live" if args.live else "paper"
    day  = date.fromisoformat(args.date) if args.date else date.today()

    while True:
        os.system("clear")
        if args.all:
            events = _load_events(args.trades_dir, day=None, mode=mode)
            trades = _pair_trades(events)
            display(trades, f"ProductionTheta — Full History ({mode.upper()})")
        else:
            events = _load_events(args.trades_dir, day=day, mode=mode)
            trades = _pair_trades(events)
            display(trades, f"ProductionTheta — {day}  ({mode.upper()})")

        if not args.watch:
            break
        print(f"  Refreshing in 30s... (Ctrl-C to stop)")
        time.sleep(30)


if __name__ == "__main__":
    main()
