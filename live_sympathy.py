"""
Live Sector Sympathy Signal — Run at 9:25 AM before market pickup.

Polls Fyers live quotes for all sector stocks, finds:
  1. Which sector is strongest at open (up >0.8%)
  2. Who's the LEADER inside that sector (biggest gap up)
  3. Who are the LAGGARDS (same sector, flat/small gap — hasn't caught up yet)

Gives you 2-3 stocks to BUY at 9:30 AM expecting them to catch up with the leader.

Usage:
  python3 live_sympathy.py              # show live signals now
  python3 live_sympathy.py --watch      # refresh every 30 sec until 10:00 AM

Needs: BROKER_APP_ID + BROKER_ACCESS_TOKEN in .env (refresh with get_token_browser.py)
"""

from __future__ import annotations

import os
import sys
import time
import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

IST = ZoneInfo("Asia/Kolkata")

APP_ID = os.getenv("BROKER_APP_ID", "")
TOKEN  = os.getenv("BROKER_ACCESS_TOKEN", "")

# ── Universe ──────────────────────────────────────────────────────────────────

from algo_platform.data.universe import SECTOR_STOCKS, stocks_in_sector

# All unique tickers
ALL_TICKERS = list({t for stocks in SECTOR_STOCKS.values() for t in stocks})

# Fyers NSE symbol for each
def _sym(ticker: str) -> str:
    return f"NSE:{ticker}-EQ"

# ── Fyers quote fetcher ───────────────────────────────────────────────────────

def _fetch_quotes(tickers: List[str]) -> Dict[str, dict]:
    """Fetch live quotes in batches of 50 (Fyers limit)."""
    try:
        from fyers_apiv3 import fyersModel
        fyers = fyersModel.FyersModel(
            client_id=APP_ID, token=TOKEN,
            is_async=False, log_path=""
        )
    except ImportError:
        print("ERROR: fyers-apiv3 not installed. Run: pip install fyers-apiv3")
        sys.exit(1)

    result: Dict[str, dict] = {}
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        syms  = ",".join(_sym(t) for t in batch)

        try:
            resp = fyers.quotes({"symbols": syms})
            if resp.get("s") != "ok":
                logging.warning("Quote fetch failed: %s", resp)
                continue

            for item in resp.get("d", []):
                ticker = item["n"].split(":")[1].replace("-EQ", "")
                v      = item.get("v", {})
                result[ticker] = {
                    "ltp":        v.get("lp",  0),
                    "prev_close": v.get("prev_close_price", 0),
                    "open":       v.get("open_price", v.get("lp", 0)),
                    "high":       v.get("high_price",  0),
                    "low":        v.get("low_price",   0),
                    "volume":     v.get("volume",      0),
                    "avg_volume": v.get("avg_traded_price", 0),
                }
        except Exception as exc:
            logging.warning("Batch %d failed: %s", i, exc)

    return result


# ── Gap calculation ───────────────────────────────────────────────────────────

def _gap_pct(quote: dict) -> float:
    """(open - prev_close) / prev_close"""
    prev = quote.get("prev_close", 0)
    op   = quote.get("open", quote.get("ltp", 0))
    if prev <= 0:
        return 0.0
    return (op - prev) / prev


def _intraday_change(quote: dict) -> float:
    """(ltp - prev_close) / prev_close — for realtime tracking after open"""
    prev = quote.get("prev_close", 0)
    ltp  = quote.get("ltp", 0)
    if prev <= 0:
        return 0.0
    return (ltp - prev) / prev


# ── Signal generation ─────────────────────────────────────────────────────────

class SympathydSignal:
    def __init__(self):
        self.sector:        str   = ""
        self.sector_avg:    float = 0.0
        self.leader:        str   = ""
        self.leader_gap:    float = 0.0
        self.leader_price:  float = 0.0
        self.laggards:      list  = []   # [(ticker, gap, price, vol_ratio)]
        self.conviction:    float = 0.0


def generate_signals(
    quotes:          Dict[str, dict],
    min_sector_gap:  float = 0.008,   # sector avg must be up >0.8%
    min_leader_gap:  float = 0.015,   # leader must be up >1.5%
    max_laggard_gap: float = 0.005,   # laggard up < 0.5%
    min_laggard_gap: float = -0.003,  # laggard not down >0.3%
    use_ltp:         bool  = False,   # False=gap, True=intraday change (use post 9:30)
) -> List[SympathydSignal]:
    """
    Core signal logic. Call with use_ltp=False before 9:30 (gap analysis),
    use_ltp=True after 9:30 (real-time catching up).
    """
    gap_fn = _intraday_change if use_ltp else _gap_pct

    # Compute gaps for all available tickers
    ticker_gaps: Dict[str, float] = {
        t: gap_fn(q)
        for t, q in quotes.items()
        if q.get("prev_close", 0) > 0 and q.get("ltp", 0) > 0
    }

    signals: List[SympathydSignal] = []

    for sector, tickers in SECTOR_STOCKS.items():
        avail = [t for t in tickers if t in ticker_gaps]
        if len(avail) < 3:
            continue

        gaps    = {t: ticker_gaps[t] for t in avail}
        sec_avg = float(np.mean(list(gaps.values())))

        if sec_avg < min_sector_gap:
            continue   # sector not strong enough

        # Leader: biggest gapper
        leader = max(avail, key=lambda t: gaps[t])
        if gaps[leader] < min_leader_gap:
            continue

        # Laggards: same sector, small gap, not already down
        laggards = []
        for t in avail:
            if t == leader:
                continue
            g = gaps[t]
            if min_laggard_gap <= g <= max_laggard_gap:
                q = quotes.get(t, {})
                vol_ratio = q.get("volume", 0) / max(q.get("avg_volume", 1), 1)
                laggards.append((t, g, q.get("ltp", 0), vol_ratio))

        if not laggards:
            continue

        # Sort laggards by vol_ratio desc (ones with rising volume first)
        laggards.sort(key=lambda x: x[3], reverse=True)

        sig = SympathydSignal()
        sig.sector       = sector
        sig.sector_avg   = sec_avg
        sig.leader       = leader
        sig.leader_gap   = gaps[leader]
        sig.leader_price = quotes[leader].get("ltp", 0)
        sig.laggards     = laggards[:3]
        sig.conviction   = min(1.0, (gaps[leader] - sec_avg) / 0.02 * sec_avg / 0.015)
        signals.append(sig)

    return sorted(signals, key=lambda s: s.conviction, reverse=True)


# ── Display ───────────────────────────────────────────────────────────────────

def print_signals(signals: List[SympathydSignal], quotes: Dict[str, dict], use_ltp: bool) -> None:
    now = datetime.now(IST)
    mode = "INTRADAY CHANGE" if use_ltp else "GAP AT OPEN"

    print("\n" + "═" * 65)
    print(f"  SECTOR SYMPATHY SIGNALS  —  {now.strftime('%H:%M:%S IST')}  ({mode})")
    print("═" * 65)

    if not signals:
        print("  No sympathy setups (no sector up >0.8% with clear leader)")
        return

    for i, sig in enumerate(signals[:3], 1):
        print(f"\n  [{i}] {sig.sector}  →  sector avg {sig.sector_avg:+.2%}")
        print(f"      LEADER:   {sig.leader:15s}  {sig.leader_gap:+.2%}  ₹{sig.leader_price:,.0f}")
        print(f"\n      BUY THESE (haven't caught up yet):")
        print(f"      {'Stock':15s}  {'Gap':>8s}  {'Price':>10s}  {'Vol Ratio':>10s}  Stop")
        print("      " + "-" * 55)
        for ticker, gap, price, vol_r in sig.laggards:
            stop  = price * 0.975   # 2.5% stop
            print(f"      {ticker:15s}  {gap:+8.2%}  ₹{price:>9,.0f}  {vol_r:>9.1f}×  ₹{stop:,.0f}")

        catch_up = sig.sector_avg - (sum(x[1] for x in sig.laggards) / max(len(sig.laggards), 1))
        print(f"\n      Target:    catch up {catch_up:+.2%} to sector average")
        print(f"      Stop loss: 2.5% from entry")
        print(f"      Exit by:   10:30 AM (if sector weakens, exit earlier)")
        print(f"      Conviction: {sig.conviction:.0%}")

    print("\n  HOW TO TRADE:")
    print("  1. Enter the BUY stocks as close to 9:30-9:35 AM as possible")
    print("  2. Use limit orders near current LTP (avoid market orders at open)")
    print("  3. Watch the LEADER stock — if it starts reversing, exit laggards immediately")
    print("  4. Take profit when laggard's % gain = sector average")
    print("═" * 65)


def print_sector_dashboard(quotes: Dict[str, dict]) -> None:
    """Show all sectors sorted by performance."""
    now = datetime.now(IST)
    print(f"\n  SECTOR DASHBOARD — {now.strftime('%H:%M IST')}")
    print(f"  {'Sector':20s}  {'Avg Gap':>8s}  {'Leader Stock':>15s}  {'Leader Gap':>10s}")
    print("  " + "-" * 60)

    sector_summary = []
    for sector, tickers in SECTOR_STOCKS.items():
        avail = [t for t in tickers if t in quotes and quotes[t].get("prev_close", 0) > 0]
        if not avail:
            continue
        gaps   = [_gap_pct(quotes[t]) for t in avail]
        avg    = float(np.mean(gaps))
        leader = max(avail, key=lambda t: _gap_pct(quotes[t]))
        lead_g = _gap_pct(quotes[leader])
        sector_summary.append((sector, avg, leader, lead_g))

    for sector, avg, leader, lead_g in sorted(sector_summary, key=lambda x: -x[1]):
        bar = "█" * min(20, max(0, int(avg * 500))) if avg > 0 else "▌" * min(20, max(0, int(-avg * 500)))
        color_marker = "★" if avg > 0.008 else "·"
        print(f"  {color_marker} {sector:18s}  {avg:+8.2%}  {leader:>15s}  {lead_g:+10.2%}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    watch_mode = "--watch" in sys.argv
    dashboard  = "--dashboard" in sys.argv

    if not APP_ID or not TOKEN:
        print("ERROR: Set BROKER_APP_ID and BROKER_ACCESS_TOKEN in .env")
        print("Run: python3 get_token_browser.py")
        sys.exit(1)

    now = datetime.now(IST)
    use_ltp = now.hour >= 9 and now.minute >= 30   # use real-time after 9:30

    def _run_once():
        print(f"\nFetching live quotes for {len(ALL_TICKERS)} stocks…", end=" ", flush=True)
        quotes = _fetch_quotes(ALL_TICKERS)
        print(f"got {len(quotes)} quotes.")

        if dashboard or watch_mode:
            print_sector_dashboard(quotes)

        sigs = generate_signals(quotes, use_ltp=use_ltp)
        print_signals(sigs, quotes, use_ltp)

    if not watch_mode:
        _run_once()
        return

    # Watch mode: refresh every 30 seconds until 10:00 AM
    print("Watch mode: refreshing every 30 seconds. Press Ctrl+C to stop.")
    while True:
        now = datetime.now(IST)
        if now.hour >= 10:
            print("\n10:00 AM — sympathy window closed. Exiting.")
            break
        _run_once()
        time.sleep(30)


if __name__ == "__main__":
    main()
