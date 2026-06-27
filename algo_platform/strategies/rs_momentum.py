"""
Strategy H — Relative Strength (RS) Momentum Cascade

The hierarchy: Market → Sector → Industry Group → Stock
At each level, we pick the leader by 5-day RS. The key insight:
stocks that are outperforming their sector are absorbing institutional
accumulation. This leadership persists for 5–15 trading days before
mean-reversion kicks in.

Why this works in Indian markets:
  - FII/FPI flows are sector-concentrated. When a sector theme plays out
    (e.g., strong US tech → Indian IT rally), buying is UNEVEN — a few
    stocks lead and the rest follow 2–5 days later.
  - Identify the leaders early via RS → capture the biggest move.
  - Nifty50/100 stocks are liquid enough for option strategies.
  - NSE sector momentum reversal is slower than US (fewer algos
    rebalancing sector ETFs), so the edge lasts longer.

Signal generation (daily, run at 3:30 PM or pre-market):
  1. Sector rank: compute 5-day return of each sector index → pick top-1.
  2. Industry rank: within top sector, compute 5-day RS of each industry
     group → pick top group.
  3. Stock rank: compute 5-day RS vs sector index for all stocks →
     score = (stock_5d_ret - sector_5d_ret) × RS_vol_weight
  4. Select top-3 stocks with:
     a. RS score > threshold
     b. Price > 20-day SMA (in uptrend)
     c. Volume trend positive (5-day avg vol > 20-day avg vol)
     d. Not at 52-week LOW (avoid value traps)
  5. Optional: also score the WORST stock per sector for short signal.

Entry (next day, 9:30–10:00 AM):
  - Long: Buy ATM or slightly OTM call (1–2 weeks to expiry) on selected stocks.
  - OR: Direct long on stock futures.
  - Entry at open or VWAP by 10:00 AM.

Exit:
  a. RS drops below threshold for 2 consecutive days → exit.
  b. Price crosses below 5-day SMA → exit.
  c. Time stop: 5 trading days.
  d. Stop loss: -3% on stock price from entry.

Usage:
  bt = RSMomentumBacktester("data/cache")
  result = bt.run(date(2022,1,1), date(2026,1,1))
  bt.print_report(result)

  # Today's signals (live):
  signal = bt.get_todays_signal()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from algo_platform.data.universe import (
    SECTOR_STOCKS, INDUSTRY_GROUPS, NIFTY50, stocks_in_sector, industries_in_sector
)

logger = logging.getLogger("platform.strategies.rs_momentum")

SECTORS = list(SECTOR_STOCKS.keys())
SLIPPAGE = 0.002   # 0.2% round-trip for stock

# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class StockSignal:
    date:           date
    sector:         str
    industry_group: str
    ticker:         str
    rs_score:       float        # stock_5d_ret - sector_5d_ret (excess return)
    rs_ratio:       float        # stock_5d_ret / sector_5d_ret (ratio, > 1 = leader)
    vol_ratio:      float        # 5d_avg_vol / 20d_avg_vol (> 1 = volume rising)
    above_sma20:    bool
    pct_from_52wk_high: float    # negative = distance below 52-week high
    direction:      str          # "LONG" or "SHORT"
    conviction:     float        # 0–1


@dataclass
class RSMomentumTrade:
    signal:       StockSignal
    entry_date:   date
    entry_price:  float
    exit_date:    Optional[date] = None
    exit_price:   Optional[float] = None
    exit_reason:  str = ""
    pnl:          float = 0.0
    pnl_pct:      float = 0.0


@dataclass
class RSBacktestResult:
    trades:     List[RSMomentumTrade]
    daily_pnl:  pd.Series
    cagr:       float
    sharpe:     float
    max_dd:     float
    win_rate:   float
    pf:         float
    n_trades:   int
    top_sectors:       Dict[str, int]   # how often each sector was picked
    top_stocks:        Dict[str, int]   # most-traded stocks


# ── Core backtester ────────────────────────────────────────────────────────────

class RSMomentumBacktester:

    def __init__(
        self,
        cache_dir: str = "data/cache",
        # Signal parameters
        rs_lookback:    int   = 5,     # days for momentum calculation
        min_rs_excess:  float = 0.005, # stock must beat sector by 0.5%+
        min_vol_ratio:  float = 1.0,   # 5d avg vol ≥ 20d avg vol
        max_stocks:     int   = 3,     # max positions per signal day
        # Exit parameters
        stop_pct:       float = 0.03,  # 3% hard stop on stock price
        hold_days:      int   = 5,     # max hold period
        exit_rs_below:  float = -0.002,# exit if RS turns negative (loses leadership)
        # Cost
        brokerage:      float = 20.0,  # ₹20 per order
        capital:        float = 200_000,
        risk_per_trade: float = 0.01,  # 1% of capital per trade
    ) -> None:
        self._cache       = Path(cache_dir)
        self._rs_lb       = rs_lookback
        self._min_rs      = min_rs_excess
        self._min_vol     = min_vol_ratio
        self._max_stocks  = max_stocks
        self._stop_pct    = stop_pct
        self._hold_days   = hold_days
        self._exit_rs     = exit_rs_below
        self._brokerage   = brokerage
        self._capital     = capital
        self._risk_per_trade = risk_per_trade

    # ── Data loading ───────────────────────────────────────────────────────────

    def _load(self, key: str, suffix: str = "_daily.csv") -> Optional[pd.DataFrame]:
        """Load daily OHLCV from cache. Returns None if unavailable."""
        path = self._cache / f"{key}{suffix}"
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
            return df[["open", "high", "low", "close", "volume"]].astype(float)
        except Exception:
            return None

    def _load_close_series(self, key: str) -> Optional[pd.Series]:
        df = self._load(key)
        if df is None:
            return None
        return df["close"].rename(key)

    # ── Signal generation ──────────────────────────────────────────────────────

    def _rank_sectors(self, sector_close: pd.DataFrame, today_idx: int) -> List[Tuple[str, float]]:
        """Returns sectors ranked by 5-day momentum, descending."""
        if today_idx < self._rs_lb + 1:
            return []
        row = sector_close.iloc[today_idx]
        prev = sector_close.iloc[today_idx - self._rs_lb]
        mom = {}
        for col in sector_close.columns:
            if not np.isnan(row[col]) and not np.isnan(prev[col]) and prev[col] > 0:
                mom[col] = (row[col] / prev[col]) - 1.0
        return sorted(mom.items(), key=lambda x: x[1], reverse=True)

    def _rank_industry_groups(
        self, stock_data: Dict[str, pd.Series], sector: str,
        today_idx: int, sector_ret_5d: float,
    ) -> List[Tuple[str, float]]:
        """Returns industry groups within sector ranked by avg RS vs sector."""
        groups = industries_in_sector(sector)
        if not groups:
            return [("all", sector_ret_5d)]

        group_rs: Dict[str, float] = {}
        for group_name, tickers in groups.items():
            rets = []
            for t in tickers:
                if t not in stock_data:
                    continue
                s = stock_data[t]
                if today_idx >= len(s) or today_idx < self._rs_lb:
                    continue
                cur  = s.iloc[today_idx]
                prev = s.iloc[today_idx - self._rs_lb]
                if prev > 0 and not np.isnan(cur) and not np.isnan(prev):
                    rets.append((cur / prev) - 1.0)
            if rets:
                group_rs[group_name] = float(np.mean(rets)) - sector_ret_5d
        return sorted(group_rs.items(), key=lambda x: x[1], reverse=True)

    def _score_stocks(
        self, stock_data: Dict[str, pd.DataFrame],
        sector: str, sector_ret_5d: float,
        today_idx: int, today_date: date,
    ) -> List[StockSignal]:
        """Score all stocks in sector for RS leadership."""
        tickers = stocks_in_sector(sector)
        signals = []

        for ticker in tickers:
            sdf = stock_data.get(ticker)
            if sdf is None or today_idx < max(self._rs_lb, 20) + 1:
                continue

            close  = sdf["close"]
            volume = sdf["volume"]

            if today_idx >= len(close):
                continue

            cur_price = float(close.iloc[today_idx])
            prev_price = float(close.iloc[today_idx - self._rs_lb])
            if prev_price <= 0 or cur_price <= 0:
                continue

            stock_ret_5d = (cur_price / prev_price) - 1.0
            rs_excess    = stock_ret_5d - sector_ret_5d
            rs_ratio     = (1 + stock_ret_5d) / (1 + sector_ret_5d) if sector_ret_5d > -1 else 1.0

            # SMA20 filter
            sma20 = float(close.iloc[today_idx - 19: today_idx + 1].mean())
            above_sma20 = cur_price > sma20

            # Volume trend (5d avg > 20d avg)
            vol_5d  = float(volume.iloc[today_idx - 4 : today_idx + 1].mean())
            vol_20d = float(volume.iloc[today_idx - 19: today_idx + 1].mean())
            vol_ratio = vol_5d / vol_20d if vol_20d > 0 else 1.0

            # 52-week high (proximity)
            if today_idx >= 252:
                high_52w = float(close.iloc[today_idx - 251: today_idx + 1].max())
                pct_from_52w = (cur_price / high_52w) - 1.0
            else:
                pct_from_52w = 0.0

            # Industry group
            ind_group = "all"
            for grp, gtickers in industries_in_sector(sector).items():
                if ticker in gtickers:
                    ind_group = grp
                    break

            # Conviction score (0-1)
            rs_norm   = min(1.0, max(0.0, rs_excess / 0.05))   # normalize to 5%
            vol_norm  = min(1.0, max(0.0, (vol_ratio - 1.0) / 1.0))
            sma_bonus = 0.2 if above_sma20 else 0.0
            h52_bonus = 0.1 if pct_from_52w > -0.05 else 0.0  # within 5% of 52w high
            conviction = min(1.0, 0.5 * rs_norm + 0.2 * vol_norm + sma_bonus + h52_bonus)

            signals.append(StockSignal(
                date              = today_date,
                sector            = sector,
                industry_group    = ind_group,
                ticker            = ticker,
                rs_score          = rs_excess,
                rs_ratio          = rs_ratio,
                vol_ratio         = vol_ratio,
                above_sma20       = above_sma20,
                pct_from_52wk_high = pct_from_52w,
                direction         = "LONG",
                conviction        = conviction,
            ))

        # Sort by RS excess (strongest first)
        return sorted(signals, key=lambda s: s.rs_score, reverse=True)

    # ── Main backtest loop ─────────────────────────────────────────────────────

    def run(
        self,
        start: date,
        end:   date,
        long_only: bool = True,
    ) -> RSBacktestResult:
        # ── Load sector index daily data ───────────────────────────────────────
        sector_frames = []
        avail_sectors = []
        for s in SECTORS:
            cs = self._load_close_series(s)
            if cs is not None:
                sector_frames.append(cs)
                avail_sectors.append(s)

        if not sector_frames:
            raise RuntimeError(
                "No sector data. Run: python3 download_sectors.py --start 2020-01-01"
            )

        sector_close = pd.concat(sector_frames, axis=1).sort_index()
        lo = pd.Timestamp(start); hi = pd.Timestamp(end)
        sector_close = sector_close[(sector_close.index >= lo) & (sector_close.index <= hi)]

        # ── Load stock daily data ──────────────────────────────────────────────
        stock_data: Dict[str, pd.DataFrame] = {}
        for sector in avail_sectors:
            for ticker in stocks_in_sector(sector):
                df = self._load(ticker)
                if df is not None:
                    df_filt = df[(df.index >= lo) & (df.index <= hi)]
                    stock_data[ticker] = df_filt

        if not stock_data:
            raise RuntimeError(
                "No stock data. Run: python3 download_stocks.py --start 2020-01-01"
            )

        logger.info("Loaded %d sectors, %d stocks for RS backtest",
                    len(avail_sectors), len(stock_data))

        # ── Backtest loop ──────────────────────────────────────────────────────
        nav   = float(self._capital)
        trades: List[RSMomentumTrade] = []
        daily_pnl: List[Tuple[date, float]] = []
        open_trades: List[RSMomentumTrade] = []
        top_sectors: Dict[str, int] = {}
        top_stocks:  Dict[str, int] = {}

        dates = sector_close.index.tolist()

        for i, ts in enumerate(dates):
            today = ts.date()

            # Mark exits first
            still_open = []
            for ot in open_trades:
                # Check price for this stock today
                sdf = stock_data.get(ot.signal.ticker)
                if sdf is None or ts not in sdf.index:
                    still_open.append(ot)
                    continue

                row = sdf.loc[ts]
                cur_price = float(row["close"])
                entry     = ot.entry_price
                days_held = (today - ot.entry_date).days

                exit_price = None
                reason     = ""

                # Stop loss
                if cur_price <= entry * (1 - self._stop_pct):
                    exit_price = cur_price * (1 - SLIPPAGE)
                    reason     = "stop_loss"

                # Time stop
                elif days_held >= self._hold_days * 1.5:  # 1.5× to allow for weekends
                    exit_price = cur_price * (1 - SLIPPAGE)
                    reason     = "time_stop"

                # RS leadership lost (if sector data available)
                elif i >= self._rs_lb and ot.signal.sector in sector_close.columns:
                    sect_prev = float(sector_close[ot.signal.sector].iloc[i - self._rs_lb])
                    sect_cur  = float(sector_close[ot.signal.sector].iloc[i])
                    sect_ret  = (sect_cur / sect_prev - 1) if sect_prev > 0 else 0
                    stk_prev  = float(sdf["close"].iloc[max(0, i - self._rs_lb - 1)])
                    stk_ret   = (cur_price / stk_prev - 1) if stk_prev > 0 else 0
                    if (stk_ret - sect_ret) < self._exit_rs:
                        exit_price = cur_price * (1 - SLIPPAGE)
                        reason     = "rs_lost"

                if exit_price is not None:
                    shares = max(1, int(self._risk_per_trade * nav / max(entry, 1) / self._stop_pct))
                    gross  = (exit_price - entry) * shares
                    cost   = self._brokerage * 2
                    ot.exit_date  = today
                    ot.exit_price = exit_price
                    ot.exit_reason= reason
                    ot.pnl        = gross - cost
                    ot.pnl_pct    = (exit_price / entry) - 1.0
                    nav += ot.pnl
                    trades.append(ot)
                else:
                    still_open.append(ot)

            open_trades = still_open

            # Generate new signals
            if i < self._rs_lb + 20:
                daily_pnl.append((today, nav))
                continue

            sector_ranking = self._rank_sectors(sector_close, i)
            if not sector_ranking:
                daily_pnl.append((today, nav))
                continue

            top_sector, sector_ret_5d = sector_ranking[0]
            if sector_ret_5d < 0.003:   # sector must be up > 0.3% over 5 days
                daily_pnl.append((today, nav))
                continue

            top_sectors[top_sector] = top_sectors.get(top_sector, 0) + 1

            # Score stocks in top sector
            candidates = self._score_stocks(
                stock_data, top_sector, sector_ret_5d, i, today
            )

            # Filter by quality
            filtered = [
                s for s in candidates
                if s.rs_score >= self._min_rs
                and s.vol_ratio >= self._min_vol
                and s.above_sma20
                and s.pct_from_52wk_high >= -0.20   # not in a bear market
            ]

            # Take top N that aren't already in a trade
            current_tickers = {ot.signal.ticker for ot in open_trades}
            new_signals = [s for s in filtered if s.ticker not in current_tickers]
            new_signals = new_signals[:self._max_stocks]

            for sig in new_signals:
                # Entry next available price (use today's close as proxy for next open)
                sdf = stock_data.get(sig.ticker)
                if sdf is None or ts not in sdf.index:
                    continue
                entry_price = float(sdf.loc[ts, "close"]) * (1 + SLIPPAGE)

                trade = RSMomentumTrade(
                    signal      = sig,
                    entry_date  = today,
                    entry_price = entry_price,
                )
                open_trades.append(trade)
                top_stocks[sig.ticker] = top_stocks.get(sig.ticker, 0) + 1

            daily_pnl.append((today, nav))

        # Force-close remaining open trades
        for ot in open_trades:
            sdf = stock_data.get(ot.signal.ticker)
            if sdf is not None and len(sdf) > 0:
                exit_price = float(sdf["close"].iloc[-1]) * (1 - SLIPPAGE)
                shares = max(1, int(self._risk_per_trade * nav / max(ot.entry_price, 1) / self._stop_pct))
                ot.pnl    = (exit_price - ot.entry_price) * shares - self._brokerage * 2
                ot.pnl_pct= (exit_price / ot.entry_price) - 1.0
                ot.exit_reason = "end_of_data"
                nav += ot.pnl
                trades.append(ot)

        # Build equity series
        if daily_pnl:
            dp_d, dp_v = zip(*daily_pnl)
            equity = pd.Series(dp_v, index=pd.DatetimeIndex(dp_d))
        else:
            equity = pd.Series([self._capital])

        # Metrics
        n      = len(trades)
        wins   = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        wr     = len(wins) / n if n > 0 else 0.0
        gw     = sum(t.pnl for t in wins)
        gl     = abs(sum(t.pnl for t in losses))
        pf     = gw / gl if gl > 0 else float("inf")

        dr     = equity.pct_change().dropna()
        peak   = equity.cummax()
        mdd    = float((equity / peak - 1).min()) if len(equity) > 1 else 0.0
        td     = (pd.Timestamp(end) - pd.Timestamp(start)).days
        ratio  = nav / self._capital
        cagr   = float(abs(ratio) ** (365 / max(1, td)) - 1) if ratio > 0 else -1.0
        vol    = float(dr.std() * 252 ** 0.5) if len(dr) > 1 else 0.01
        sharpe = (cagr - 0.065) / vol if vol > 0 else 0.0

        return RSBacktestResult(
            trades      = trades,
            daily_pnl   = equity,
            cagr        = cagr,
            sharpe      = sharpe,
            max_dd      = mdd,
            win_rate    = wr,
            pf          = pf,
            n_trades    = n,
            top_sectors = top_sectors,
            top_stocks  = top_stocks,
        )

    # ── Report ─────────────────────────────────────────────────────────────────

    def print_report(self, result: RSBacktestResult) -> None:
        print("\n" + "=" * 65)
        print("  RS MOMENTUM CASCADE — BACKTEST REPORT")
        print("=" * 65)
        print(f"  Trades:         {result.n_trades}")
        print(f"  Win Rate:       {result.win_rate:.1%}")
        print(f"  Profit Factor:  {result.pf:.2f}")
        print(f"  CAGR:           {result.cagr:.1%}")
        print(f"  Sharpe:         {result.sharpe:.2f}")
        print(f"  Max Drawdown:   {result.max_dd:.1%}")

        print("\n  Top Sectors (days as leader):")
        for s, cnt in sorted(result.top_sectors.items(), key=lambda x: -x[1])[:5]:
            print(f"    {s:20s}  {cnt:4d} days")

        print("\n  Most Traded Stocks:")
        for t, cnt in sorted(result.top_stocks.items(), key=lambda x: -x[1])[:10]:
            wins = [tr for tr in result.trades if tr.signal.ticker == t and tr.pnl > 0]
            total = [tr for tr in result.trades if tr.signal.ticker == t]
            wr = len(wins) / len(total) if total else 0
            avg_pnl = np.mean([tr.pnl for tr in total]) if total else 0
            print(f"    {t:15s}  {cnt:3d}× | WR={wr:.0%} | avg ₹{avg_pnl:.0f}")

        if result.trades:
            sorted_by_pnl = sorted(result.trades, key=lambda t: t.pnl)
            print(f"\n  Best trade:  {sorted_by_pnl[-1].signal.ticker} "
                  f"₹{sorted_by_pnl[-1].pnl:,.0f} "
                  f"({sorted_by_pnl[-1].signal.sector})")
            print(f"  Worst trade: {sorted_by_pnl[0].signal.ticker} "
                  f"₹{sorted_by_pnl[0].pnl:,.0f} "
                  f"({sorted_by_pnl[0].signal.sector})")

            # Exit reason breakdown
            from collections import Counter
            reasons = Counter(t.exit_reason for t in result.trades)
            print("\n  Exit reasons:")
            for r, cnt in reasons.most_common():
                avg = np.mean([t.pnl for t in result.trades if t.exit_reason == r])
                print(f"    {r:20s}  {cnt:4d}  avg ₹{avg:.0f}")

        print("=" * 65)

    # ── Live signal (for paper/live trading) ───────────────────────────────────

    def get_todays_signals(self) -> List[StockSignal]:
        """Generate today's RS signals from cached data (pre-market use)."""
        sector_frames = []
        avail_sectors = []
        for s in SECTORS:
            cs = self._load_close_series(s)
            if cs is not None:
                sector_frames.append(cs)
                avail_sectors.append(s)

        if not sector_frames:
            return []

        sector_close = pd.concat(sector_frames, axis=1).sort_index()
        i = len(sector_close) - 1

        sector_ranking = self._rank_sectors(sector_close, i)
        if not sector_ranking or sector_ranking[0][1] < 0.003:
            return []

        top_sector, sector_ret_5d = sector_ranking[0]

        stock_data: Dict[str, pd.DataFrame] = {}
        for ticker in stocks_in_sector(top_sector):
            df = self._load(ticker)
            if df is not None:
                stock_data[ticker] = df

        if not stock_data:
            return []

        ts = sector_close.index[i]
        candidates = self._score_stocks(
            stock_data, top_sector, sector_ret_5d, i, ts.date()
        )
        return [
            s for s in candidates
            if s.rs_score >= self._min_rs
            and s.vol_ratio >= self._min_vol
            and s.above_sma20
        ][:self._max_stocks]

    def print_todays_signals(self) -> None:
        sigs = self.get_todays_signals()
        print("\n" + "=" * 60)
        print("  RS MOMENTUM — TODAY'S SIGNALS")
        print("=" * 60)
        if not sigs:
            print("  No signals today (no strong sector or insufficient data)")
        for sig in sigs:
            print(f"\n  {sig.ticker:15s} [{sig.sector} / {sig.industry_group}]")
            print(f"    RS Excess:   {sig.rs_score:+.2%}  (vs sector 5d)")
            print(f"    RS Ratio:    {sig.rs_ratio:.2f}×")
            print(f"    Vol Ratio:   {sig.vol_ratio:.2f}×  (5d/20d avg vol)")
            print(f"    Above SMA20: {'✓' if sig.above_sma20 else '✗'}")
            print(f"    52w Prox:    {sig.pct_from_52wk_high:+.1%}")
            print(f"    Conviction:  {sig.conviction:.0%}")
            print(f"    → Enter: Buy ATM call or stock futures at 9:30–10:00 AM")
        print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    bt = RSMomentumBacktester(
        cache_dir      = "data/cache",
        rs_lookback    = 5,
        min_rs_excess  = 0.005,
        min_vol_ratio  = 0.9,
        max_stocks     = 3,
        hold_days      = 5,
        stop_pct       = 0.03,
        capital        = 200_000,
    )

    if "--signals" in sys.argv:
        bt.print_todays_signals()
        sys.exit(0)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end",   default="2026-01-01")
    args = parser.parse_args()

    try:
        result = bt.run(
            start = date.fromisoformat(args.start),
            end   = date.fromisoformat(args.end),
        )
        bt.print_report(result)
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        print("\nDownload data first:")
        print("  python3 download_sectors.py --start 2020-01-01")
        print("  python3 download_stocks.py  --start 2020-01-01")
        sys.exit(1)
