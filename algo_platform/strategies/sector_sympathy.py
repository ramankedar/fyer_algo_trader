"""
Strategy I — Intraday Sector Sympathy (Catch-Up Trade)

The core insight:
  When a sector moves strongly at open (driven by 1-2 leader stocks), the
  laggard stocks in the SAME sector catch up within 45-90 minutes. Why:
    1. FII/DII sector-ETF buying pushes ALL constituents, not just leaders
    2. Retail sector-followers pile into whatever sector stock they own
    3. Index rebalancing creates proportional demand across all constituents
    4. Arbitrageurs close sector-internal dispersion

This is the most powerful 9:30-10:30 AM trade in Indian markets.

Signal generation (live, 9:30 AM):
  1. At 9:15–9:30 AM: rank sectors by intraday return (live quotes)
  2. Top sector must be up > 0.8% from previous close
  3. Within sector, find LEADER (up >1.5%) and LAGGARDS (up <0.5% or flat)
  4. Laggard qualifications: same direction as sector, volume starting to pick up
  5. Enter laggard at 9:30-9:45 AM entry window
  6. Stop: laggard crosses below its VWAP or sector index turns red
  7. Target: laggard gains = sector average gain (catch-up complete)

Backtest mode: uses daily open/prev_close to approximate the gap pattern.
Live mode: uses Fyers real-time quotes for precise entry.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from algo_platform.data.universe import SECTOR_STOCKS, stocks_in_sector

logger = logging.getLogger("platform.strategies.sector_sympathy")

SLIPPAGE = 0.002   # 0.2% round-trip (stock intraday)


@dataclass
class SympathySignal:
    date:            date
    sector:          str
    sector_gap_pct:  float      # sector index gap at open
    leader:          str        # stock with largest gap
    leader_gap:      float
    laggard:         str        # stock to BUY (hasn't caught up yet)
    laggard_gap:     float
    catch_up_target: float      # expected return = sector avg or leader
    conviction:      float


@dataclass
class SympathyTrade:
    signal:       SympathySignal
    entry_price:  float
    exit_price:   Optional[float] = None
    pnl:          float = 0.0
    pnl_pct:      float = 0.0
    exit_reason:  str   = ""
    shares:       int   = 0


class SectorSympathyBacktester:
    """
    Backtest the sector sympathy catch-up trade using daily OHLCV.

    Approximation: daily open vs. prev_close ≈ first-15-min gap signal.
    For exact results, use 1-min stock data (15-bar VWAP approach in live mode).

    Usage
    -----
    bt = SectorSympathyBacktester("data/cache")
    result = bt.run(date(2022,1,1), date(2026,1,1))
    bt.print_report(result)
    """

    def __init__(
        self,
        cache_dir:         str   = "data/cache",
        # Sector filter
        min_sector_gap:    float = 0.008,   # sector must be up >0.8% at open
        # Leader definition
        min_leader_gap:    float = 0.015,   # leader must be up >1.5%
        # Laggard definition
        max_laggard_gap:   float = 0.005,   # laggard is up <0.5% (hasn't moved yet)
        min_laggard_gap:   float = -0.002,  # laggard must be at least flat (not down)
        # Exit
        stop_pct:          float = 0.025,   # 2.5% stop on entry price
        target_as_sector:  bool  = True,    # target = catch up to sector average
        hold_days_max:     int   = 1,       # max hold = same day (intraday)
        # Capital
        capital:           float = 200_000,
        risk_per_trade:    float = 0.01,    # 1% of capital per trade
        brokerage:         float = 20.0,
    ) -> None:
        self._cache          = Path(cache_dir)
        self._min_sec_gap    = min_sector_gap
        self._min_lead_gap   = min_leader_gap
        self._max_lag_gap    = max_laggard_gap
        self._min_lag_gap    = min_laggard_gap
        self._stop_pct       = stop_pct
        self._target_sector  = target_as_sector
        self._capital        = capital
        self._risk_per_trade = risk_per_trade
        self._brokerage      = brokerage

    # ── Data ──────────────────────────────────────────────────────────────────

    def _load(self, ticker: str) -> Optional[pd.DataFrame]:
        for path in [
            self._cache / f"{ticker}_daily.csv",
            self._cache / f"{ticker.upper()}_daily.csv",
        ]:
            if path.exists():
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
                return df[["open", "high", "low", "close", "volume"]].astype(float)
        return None

    def _load_sector_index(self, sector: str) -> Optional[pd.DataFrame]:
        """Load sector index daily data for trend confirmation."""
        # Try NSE sector index
        for key in [sector, sector.replace("NIFTY", "")]:
            path = self._cache / f"{key}_daily.csv"
            if path.exists():
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
                return df[["open", "close"]].astype(float)
        return None

    # ── Signal logic ──────────────────────────────────────────────────────────

    def _sector_gap(
        self, stock_opens: Dict[str, float], stock_prev: Dict[str, float],
        tickers: List[str],
    ) -> float:
        """Approximate sector gap as average stock gap, weighted equally."""
        gaps = []
        for t in tickers:
            if t in stock_opens and t in stock_prev and stock_prev[t] > 0:
                gaps.append((stock_opens[t] / stock_prev[t]) - 1.0)
        return float(np.mean(gaps)) if gaps else 0.0

    def _find_sympathy(
        self,
        sector: str,
        stock_opens: Dict[str, float],
        stock_prev: Dict[str, float],
        today: date,
    ) -> Optional[SympathySignal]:
        tickers = stocks_in_sector(sector)
        available = [t for t in tickers if t in stock_opens and t in stock_prev
                     and stock_prev[t] > 0]

        if len(available) < 4:   # need enough stocks to identify pattern
            return None

        gaps = {t: (stock_opens[t] / stock_prev[t]) - 1.0 for t in available}
        sec_gap = self._sector_gap(stock_opens, stock_prev, available)

        if sec_gap < self._min_sec_gap:
            return None    # sector not strong enough at open

        # Identify leader (biggest gapper)
        leader = max(available, key=lambda t: gaps[t])
        if gaps[leader] < self._min_lead_gap:
            return None    # no clear leader

        # Identify laggards (up less, but same positive direction)
        laggards = [
            t for t in available
            if t != leader
            and self._min_lag_gap <= gaps[t] <= self._max_lag_gap
        ]
        if not laggards:
            return None

        # Best laggard = one with highest volume or widest gap to catch up
        laggard = laggards[0]   # first one; could rank by volume

        sec_avg = np.mean([gaps[t] for t in available])
        conviction = min(1.0, (gaps[leader] - gaps[laggard]) / 0.03)

        return SympathySignal(
            date           = today,
            sector         = sector,
            sector_gap_pct = sec_gap,
            leader         = leader,
            leader_gap     = gaps[leader],
            laggard        = laggard,
            laggard_gap    = gaps[laggard],
            catch_up_target= sec_avg,      # target = reach sector average
            conviction     = conviction,
        )

    # ── Backtest ──────────────────────────────────────────────────────────────

    def run(self, start: date, end: date) -> dict:
        from algo_platform.data.universe import SECTOR_STOCKS
        sectors = list(SECTOR_STOCKS.keys())

        # Load all stock data
        stock_data: Dict[str, pd.DataFrame] = {}
        for sector in sectors:
            for ticker in stocks_in_sector(sector):
                df = self._load(ticker)
                if df is not None:
                    stock_data[ticker] = df

        if len(stock_data) < 10:
            raise RuntimeError(
                "Need stock daily data. Run: python3 download_stocks.py --start 2020-01-01"
            )

        # Build a combined date index
        all_dates = sorted(set(
            ts.date() for df in stock_data.values()
            for ts in df.index
            if start <= ts.date() <= end
        ))

        nav    = float(self._capital)
        trades: List[SympathyTrade] = []
        equity: List[Tuple[date, float]] = []

        for day in all_dates:
            ts = pd.Timestamp(day)
            prev_ts = None
            # Find previous trading day
            for past in reversed(all_dates[:all_dates.index(day)]):
                prev_ts = pd.Timestamp(past)
                break

            if prev_ts is None:
                equity.append((day, nav))
                continue

            # Get open and prev_close for all stocks today
            stock_opens: Dict[str, float] = {}
            stock_prev:  Dict[str, float] = {}
            stock_high:  Dict[str, float] = {}
            stock_low:   Dict[str, float] = {}

            for ticker, df in stock_data.items():
                if ts in df.index and prev_ts in df.index:
                    row      = df.loc[ts]
                    prev_row = df.loc[prev_ts]
                    stock_opens[ticker] = float(row["open"])
                    stock_prev[ticker]  = float(prev_row["close"])
                    stock_high[ticker]  = float(row["high"])
                    stock_low[ticker]   = float(row["low"])

            # Check all sectors for a sympathy setup
            best_signal: Optional[SympathySignal] = None
            best_sec_gap = 0.0

            for sector in sectors:
                sig = self._find_sympathy(sector, stock_opens, stock_prev, day)
                if sig and sig.sector_gap_pct > best_sec_gap:
                    best_signal  = sig
                    best_sec_gap = sig.sector_gap_pct

            if best_signal is None:
                equity.append((day, nav))
                continue

            sig = best_signal
            laggard = sig.laggard
            if laggard not in stock_opens:
                equity.append((day, nav))
                continue

            entry_price = stock_opens[laggard] * (1 + SLIPPAGE)
            stop_price  = entry_price * (1 - self._stop_pct)
            target_price= entry_price * (1 + sig.catch_up_target)

            shares = max(1, int(self._risk_per_trade * nav / (entry_price * self._stop_pct)))
            cost   = self._brokerage * 2

            # Simulate using day's H/L
            high = stock_high.get(laggard, entry_price)
            low  = stock_low.get(laggard,  entry_price)
            prev_close = stock_prev[laggard]

            if low <= stop_price:
                exit_price  = stop_price
                exit_reason = "stop_loss"
            elif high >= target_price:
                exit_price  = target_price
                exit_reason = "target_hit"
            else:
                # Approximate intraday exit at close (couldn't reach target today)
                # Use daily close as proxy for 10:30 AM exit
                if ts in stock_data[laggard].index:
                    exit_price = float(stock_data[laggard].loc[ts, "close"]) * (1 - SLIPPAGE)
                else:
                    exit_price = entry_price
                exit_reason = "time_stop"

            pnl     = (exit_price - entry_price) * shares - cost
            pnl_pct = (exit_price / entry_price) - 1.0

            trade = SympathyTrade(
                signal      = sig,
                entry_price = entry_price,
                exit_price  = exit_price,
                pnl         = pnl,
                pnl_pct     = pnl_pct,
                exit_reason = exit_reason,
                shares      = shares,
            )
            nav += pnl
            trades.append(trade)
            equity.append((day, nav))

        return self._metrics(trades, equity, start, end)

    def _metrics(self, trades, equity, start, end) -> dict:
        n = len(trades)
        if n == 0:
            return {"error": "No trades"}

        wins   = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        wr     = len(wins) / n
        gw     = sum(t.pnl for t in wins)
        gl     = abs(sum(t.pnl for t in losses))
        pf     = gw / gl if gl > 0 else float("inf")
        exp    = sum(t.pnl for t in trades) / n

        if equity:
            eq_d, eq_v = zip(*equity)
            eq  = pd.Series(eq_v, index=pd.DatetimeIndex(eq_d))
            peak = eq.cummax()
            mdd  = float((eq / peak - 1).min())
            td   = (pd.Timestamp(end) - pd.Timestamp(start)).days
            cagr = float((eq.iloc[-1] / self._capital) ** (365 / max(1, td)) - 1)
            vol  = float(eq.pct_change().dropna().std() * 252**0.5)
            sharpe = (cagr - 0.065) / vol if vol > 0 else 0.0
        else:
            mdd = cagr = sharpe = 0.0

        from collections import Counter
        reasons = Counter(t.exit_reason for t in trades)

        return {
            "trades": trades, "n": n, "win_rate": wr,
            "avg_win":  float(np.mean([t.pnl for t in wins]))  if wins   else 0,
            "avg_loss": float(np.mean([t.pnl for t in losses])) if losses else 0,
            "profit_factor": pf, "expectancy": exp,
            "cagr": cagr, "sharpe": sharpe, "max_dd": mdd,
            "exit_reasons": reasons,
        }

    def print_report(self, result: dict) -> None:
        if "error" in result:
            print(f"No result: {result['error']}")
            return

        print("\n" + "=" * 60)
        print("  SECTOR SYMPATHY CATCH-UP — BACKTEST REPORT")
        print("=" * 60)
        print(f"  Trades:         {result['n']}")
        print(f"  Win Rate:       {result['win_rate']:.1%}")
        print(f"  Avg Win:        ₹{result['avg_win']:,.0f}")
        print(f"  Avg Loss:       ₹{result['avg_loss']:,.0f}")
        print(f"  Profit Factor:  {result['profit_factor']:.2f}")
        print(f"  Expectancy:     ₹{result['expectancy']:,.0f}  per trade")
        print(f"  CAGR:           {result['cagr']:.1%}")
        print(f"  Sharpe:         {result['sharpe']:.2f}")
        print(f"  Max Drawdown:   {result['max_dd']:.1%}")
        print(f"\n  Exit Reasons:")
        for r, cnt in result["exit_reasons"].most_common():
            pnl = np.mean([t.pnl for t in result["trades"] if t.exit_reason == r])
            print(f"    {r:20s} {cnt:4d}  avg ₹{pnl:,.0f}")
        trades = result["trades"]
        if trades:
            by_sector = defaultdict(list)
            for t in trades:
                by_sector[t.signal.sector].append(t.pnl)
            print(f"\n  By Sector:")
            for s, pnls in sorted(by_sector.items(), key=lambda x: -sum(x[1])):
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
                print(f"    {s:20s}  {len(pnls):4d} trades  WR={wr:.0%}  total ₹{sum(pnls):,.0f}")
        print("=" * 60)

    # ── Live signal (use at 9:30 AM with live Fyers quotes) ───────────────────

    def get_live_signals(self, live_gaps: Dict[str, float]) -> List[SympathySignal]:
        """
        Generate live signals at 9:30 AM.

        Parameters
        ----------
        live_gaps : {ticker: gap_pct} e.g. {"TCS": 0.021, "INFY": 0.008, ...}
                    Computed as (current_price / prev_close) - 1

        Returns
        -------
        List of SympathySignal sorted by conviction
        """
        from algo_platform.data.universe import SECTOR_STOCKS
        signals = []

        for sector in SECTOR_STOCKS:
            tickers = stocks_in_sector(sector)
            avail = [t for t in tickers if t in live_gaps]
            if len(avail) < 3:
                continue

            sec_gap = np.mean([live_gaps[t] for t in avail])
            if sec_gap < self._min_sec_gap:
                continue

            leader_t = max(avail, key=lambda t: live_gaps[t])
            if live_gaps[leader_t] < self._min_lead_gap:
                continue

            laggards = [
                t for t in avail
                if t != leader_t
                and self._min_lag_gap <= live_gaps[t] <= self._max_lag_gap
            ]
            if not laggards:
                continue

            for lag in laggards[:2]:   # take top 2 laggards
                sig = SympathySignal(
                    date           = date.today(),
                    sector         = sector,
                    sector_gap_pct = sec_gap,
                    leader         = leader_t,
                    leader_gap     = live_gaps[leader_t],
                    laggard        = lag,
                    laggard_gap    = live_gaps[lag],
                    catch_up_target= sec_gap,
                    conviction     = min(1.0, (live_gaps[leader_t] - live_gaps[lag]) / 0.03),
                )
                signals.append(sig)

        return sorted(signals, key=lambda s: s.conviction, reverse=True)

    def print_live_signals(self, signals: List[SympathySignal]) -> None:
        print("\n" + "=" * 60)
        print(f"  SYMPATHY SIGNALS — {date.today()}")
        print("=" * 60)
        if not signals:
            print("  No sympathy setups today (no sector up >0.8% with clear leader)")
            return

        for sig in signals[:5]:
            print(f"\n  Sector: {sig.sector}  (up {sig.sector_gap_pct:+.2%} at open)")
            print(f"  Leader:  {sig.leader:15s} up {sig.leader_gap:+.2%}")
            print(f"  → BUY:   {sig.laggard:15s} up only {sig.laggard_gap:+.2%}  (hasn't caught up)")
            print(f"  Target:  +{sig.catch_up_target:.2%}  (sector average)")
            print(f"  Stop:    -{self._stop_pct:.1%} from entry")
            print(f"  Conviction: {sig.conviction:.0%}")
        print("=" * 60)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    bt = SectorSympathyBacktester(
        cache_dir      = "data/cache",
        min_sector_gap = 0.008,
        min_leader_gap = 0.015,
        max_laggard_gap= 0.005,
        stop_pct       = 0.025,
        capital        = 200_000,
    )

    try:
        result = bt.run(date(2022, 1, 1), date(2026, 1, 1))
        bt.print_report(result)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
