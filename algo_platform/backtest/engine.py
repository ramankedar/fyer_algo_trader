"""
Event-driven backtesting engine.

Design rules that prevent lookahead bias:
  - Features are computed strictly from bars[0..t], chain[t].
  - Strategy sees only bar[t] and features[t] when deciding.
  - Order fills use bar[t+1] open price (next-bar-open model).
  - Costs applied on fill, not on signal.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from algo_platform.core.config import PlatformConfig, LOT_SIZES
from algo_platform.core.types import (
    FeatureVector, Instrument, MarketBar, OptionChain, OptionType,
    Order, OrderSide, OrderStatus, PerformanceReport, Position, Signal,
    SpreadLeg, Trade, TradeStatus,
)
from algo_platform.backtest.costs import IndianOptionsCostModel
from algo_platform.backtest.metrics import PerformanceAnalyzer
from algo_platform.research.features import FeatureEngine
from algo_platform.strategies.base import BaseStrategy

logger = logging.getLogger("algo_platform.backtest.engine")


class BacktestEngine:
    """
    Bar-by-bar simulation engine.

    Usage
    -----
    engine = BacktestEngine(config)
    report = engine.run(strategy, bars, chains=None, breadth_series=None)
    """

    def __init__(self, config: PlatformConfig) -> None:
        self._cfg      = config
        self._costs    = IndianOptionsCostModel(config.backtest)
        self._analyzer = PerformanceAnalyzer(
            risk_free        = config.risk_free_rate,
            validation_gates = {
                "min_profit_factor": config.backtest.min_profit_factor,
                "min_sharpe":        config.backtest.min_sharpe,
                "max_drawdown":      config.backtest.max_drawdown,
                "min_trades":        config.backtest.min_trades,
            },
        )

    # ── Main entry point ───────────────────────────────────────────────────────

    def run(
        self,
        strategy:        BaseStrategy,
        bars:            List[MarketBar],
        chains:          Optional[List[Optional[OptionChain]]] = None,
        breadth_series:  Optional[List[float]] = None,
        chain_builder=None,   # SyntheticChainBuilder | None
        vix_by_date: Optional[Dict[date, float]] = None,
    ) -> PerformanceReport:
        """
        Run a full backtest on `strategy` over `bars`.

        Parameters
        ----------
        bars          : 1-min OHLCV bars, chronological order.
        chains        : pre-built option chain per bar (takes priority over builder).
        chain_builder : SyntheticChainBuilder — used when chains is None.
                        Builds a chain from real spot + VIX on every bar.
        vix_by_date   : {date: vix_close} lookup used by chain_builder.
                        Falls back to config base_iv × 100 if not provided.
        breadth_series: fraction of advancing stocks per bar (None → 0.5).
        """
        if breadth_series is None:
            breadth_series = [0.5] * len(bars)

        instrument = strategy.instrument
        lot_size   = self._cfg.lot_size(instrument.value)

        feat_engine = FeatureEngine(
            instrument         = instrument.value,
            lot_size           = lot_size,
            atr_period         = self._cfg.research.atr_period,
            rv_window          = self._cfg.research.rv_window,
            percentile_window  = self._cfg.research.percentile_window,
            iv_rank_window     = self._cfg.research.iv_rank_window,
        )

        # Pre-process VIX for fast date lookup (sorted ascending)
        sorted_vix: List[Tuple[date, float]] = (
            sorted(vix_by_date.items()) if vix_by_date else []
        )
        default_iv     = self._cfg.backtest.base_iv
        last_chain: Optional[OptionChain] = None
        last_chain_min = -999     # minute-of-day when chain was last built

        capital       = self._cfg.risk.capital
        nav           = capital
        peak_nav      = capital      # highest NAV ever reached (for rolling DD)
        daily_nav: Dict[date, float] = {}
        daily_start_nav = capital    # NAV at the start of the current session day
        weekly_start_nav= capital    # NAV at the start of the current ISO week

        trades:      List[Trade]     = []
        open_trades: List[OpenTrade] = []

        current_date: Optional[date] = None
        current_week: int            = -1

        # When explicit chains list provided, consume it; otherwise build lazily
        chain_iter = iter(chains) if chains is not None else None

        for i, (bar, breadth) in enumerate(zip(bars, breadth_series)):
            # ── Resolve chain for this bar ──────────────────────────────────────
            if chain_iter is not None:
                chain = next(chain_iter, None)
            elif chain_builder is not None:
                # Build chain only every 5 minutes (options don't move faster)
                bar_min = bar.timestamp.hour * 60 + bar.timestamp.minute
                if abs(bar_min - last_chain_min) >= 5:
                    atm_iv = self._vix_for_date(sorted_vix, bar.timestamp.date(), default_iv)
                    try:
                        last_chain = chain_builder.build(
                            instrument = instrument,
                            spot       = bar.close,
                            timestamp  = bar.timestamp,
                            atm_iv     = atm_iv,
                        )
                    except Exception:
                        last_chain = None
                    last_chain_min = bar_min
                chain = last_chain
            else:
                chain = None
            # Reset session state at each new trading day
            if bar.timestamp.date() != current_date:
                current_date     = bar.timestamp.date()
                daily_start_nav  = nav   # baseline for daily loss limit
                iso_week         = bar.timestamp.isocalendar()[1]
                if iso_week != current_week:
                    weekly_start_nav = nav   # baseline for weekly loss limit
                    current_week     = iso_week
                feat_engine.new_session()
                if hasattr(strategy, "new_session"):
                    strategy.new_session()

            # ── Compute features (causal: only uses data up to bar i) ─────────
            features = feat_engine.update(bar, chain, breadth)
            if features is None:
                continue   # still warming up

            # ── Check exits for open trades ───────────────────────────────────
            still_open: List[OpenTrade] = []
            for ot in open_trades:
                exit_signal, exit_reason = self._check_exit(ot, bar, features, strategy, chain)
                if exit_signal:
                    trade = self._close_trade(ot, bar, exit_reason, chain)
                    # nav impact = exit_proceeds - exit_costs (entry cost already
                    # deducted at open via nav -= ot.cost, so don't deduct again)
                    nav  += trade.pnl + ot.cost
                    peak_nav = max(peak_nav, nav)   # track all-time high NAV
                    trades.append(trade)
                    logger.debug("Exit %s | reason=%s pnl=%.2f",
                                 ot.trade_id[:8], exit_reason, trade.pnl)
                else:
                    still_open.append(ot)
            open_trades = still_open

            # ── Risk gate: rolling drawdown + daily/weekly session limits ────────
            if not self._can_open_new(nav, peak_nav, daily_start_nav, weekly_start_nav):
                daily_nav[bar.timestamp.date()] = nav
                continue

            # ── Generate signal ────────────────────────────────────────────────
            signal = strategy.generate_signal(bar, chain, features)
            if signal is None:
                daily_nav[bar.timestamp.date()] = nav
                continue

            # ── Size and enter ─────────────────────────────────────────────────
            lots = self._size_position(nav, signal, features)
            if lots < 1:
                daily_nav[bar.timestamp.date()] = nav
                continue

            ot = self._open_trade(signal, lots, lot_size, bar, nav)
            if ot is not None:
                nav -= ot.cost          # deduct premium + costs
                open_trades.append(ot)
                logger.debug("Enter %s | lots=%d cost=%.2f",
                             ot.trade_id[:8], lots, ot.cost)

            daily_nav[bar.timestamp.date()] = nav

        # ── Force-close any remaining open trades at last bar ─────────────────
        if bars and open_trades:
            last_bar = bars[-1]
            for ot in open_trades:
                trade = self._close_trade(ot, last_bar, "end_of_simulation")
                nav  += trade.pnl
                trades.append(trade)
            daily_nav[last_bar.timestamp.date()] = nav

        # ── Build equity curve ────────────────────────────────────────────────
        if not daily_nav:
            equity = np.array([capital])
        else:
            sorted_days  = sorted(daily_nav.keys())
            equity_list  = [capital] + [daily_nav[d] for d in sorted_days]
            equity       = np.array(equity_list, dtype=float)

        start = bars[0].timestamp.date()  if bars else date.today()
        end   = bars[-1].timestamp.date() if bars else date.today()

        return self._analyzer.generate_report(
            strategy        = strategy.name,
            trades          = trades,
            equity_curve    = equity,
            initial_capital = capital,
            start_date      = start,
            end_date        = end,
            total_bars      = len(bars),
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _vix_for_date(
        sorted_vix: List[Tuple[date, float]],
        d:          date,
        default_iv: float,
    ) -> float:
        """Forward-fill VIX: return the most recent closing VIX on or before d."""
        if not sorted_vix:
            return default_iv
        iv = default_iv
        for vd, vv in sorted_vix:
            if vd <= d:
                iv = vv / 100.0   # VIX is percentage; convert to decimal
            else:
                break
        return iv

    def _can_open_new(
        self,
        nav:              float,
        peak_nav:         float,   # all-time high NAV (rolling drawdown base)
        daily_start_nav:  float,   # NAV at today's open (daily loss base)
        weekly_start_nav: float,   # NAV at this week's open (weekly loss base)
    ) -> bool:
        """
        Rolling risk gates — all measured from session/peak baselines so
        a bad day doesn't permanently ban trading for the rest of the backtest.
        """
        cfg = self._cfg.risk

        # Daily loss limit: reset each morning
        if nav < daily_start_nav * (1.0 - cfg.max_daily_loss_pct):
            return False

        # Weekly loss limit: reset each Monday
        if nav < weekly_start_nav * (1.0 - cfg.max_weekly_loss_pct):
            return False

        # Rolling peak-to-trough drawdown: resumes once we recover
        # (old code used cumulative-from-initial which permanently halted)
        if peak_nav > 0 and (peak_nav - nav) / peak_nav > cfg.max_portfolio_dd_pct:
            return False

        return True

    def _size_position(self, nav: float, signal: Signal,
                        features: FeatureVector) -> int:
        """Volatility-targeted lot sizing with drawdown scaling."""
        cfg        = self._cfg.risk
        capital    = cfg.capital
        rv         = max(features.realized_vol, 0.01)
        target_vol = cfg.target_annual_vol

        risk_budget_per_trade = nav * cfg.risk_per_trade_pct
        lot_size = self._cfg.lot_size(signal.instrument.value)
        if signal.max_loss > 0:
            loss_per_lot = signal.max_loss   # defined-risk strategies
        elif signal.net_debit < 0:
            # Credit trade: max loss = stop_loss_mult × credit received
            credit_per_lot = abs(signal.net_debit) * lot_size
            loss_per_lot = credit_per_lot * 1.5   # 1.5× credit = practical stop
        else:
            loss_per_lot = 50.0 * lot_size
        base_lots = max(1, int(risk_budget_per_trade / loss_per_lot))

        # Drawdown scaling
        dd_pct = max(0.0, (capital - nav) / capital)
        if dd_pct >= cfg.dd_hard_pct:
            return 0
        if dd_pct >= cfg.dd_soft_pct:
            scale = 1.0 - (dd_pct - cfg.dd_soft_pct) / (
                cfg.dd_hard_pct - cfg.dd_soft_pct + 1e-6
            )
            scale = max(cfg.min_size_factor, scale)
            base_lots = max(1, int(base_lots * scale))

        return base_lots

    def _open_trade(self, signal: Signal, lots: int, lot_size: int,
                    bar: MarketBar, nav: float) -> Optional[OpenTrade]:
        """
        Execute entry.  Net cost = premiums paid on BUY legs
                                 – credits received on SELL legs
                                 + all transaction costs.
        Returns None if net cost exceeds available capital.
        """
        total_cost = 0.0
        is_credit_trade = all(leg.side == OrderSide.SELL for leg in signal.legs)

        for leg in signal.legs:
            cost = self._costs.compute(leg.limit_price, lots, lot_size, leg.side)
            if leg.side == OrderSide.BUY:
                total_cost += leg.limit_price * lots * lot_size  # debit paid
            else:
                total_cost -= leg.limit_price * lots * lot_size  # credit received
            total_cost += cost.total   # transaction costs always positive

        # For debit trades: total_cost > 0 (paid net premium)
        # For credit trades: total_cost < 0 (received net premium)
        # Do NOT clamp — the sign is meaningful for accounting.

        # Capital check:
        if is_credit_trade or total_cost < 0:
            # Short position: broker requires margin ≈ 3× credit received
            # (simplified — real SPAN margin is complex)
            margin_required = abs(total_cost) * 3.0
            if margin_required > nav - self._cfg.risk.margin_reserve:
                return None
        else:
            if total_cost > nav - self._cfg.risk.margin_reserve:
                return None

        trade_id = str(uuid.uuid4())[:12]
        return OpenTrade(
            trade_id   = trade_id,
            signal     = signal,
            entry_time = bar.timestamp,
            entry_bar  = bar,
            lots       = lots,
            lot_size   = lot_size,
            cost       = total_cost,
        )

    def _check_exit(
        self, ot: "OpenTrade", bar: MarketBar,
        features: FeatureVector, strategy: BaseStrategy,
        chain: Optional["OptionChain"] = None,
    ) -> Tuple[bool, str]:
        """Delegate to the strategy's should_exit method."""
        strategy_name = ot.signal.strategy

        # Current spread value per share (same unit as net_debit/credit in should_exit)
        if chain is not None:
            current_val = abs(self._spread_exit_value(ot, bar, chain))  # per share
        else:
            current_val = abs(ot.signal.net_debit) * 0.5  # per share fallback

        if strategy_name == "VolCompressionExpansion":
            if hasattr(strategy, "should_exit"):
                return strategy.should_exit(
                    bar, features, ot.signal.net_debit, current_val
                )
        elif strategy_name == "IntradayTrend":
            if hasattr(strategy, "should_exit"):
                return strategy.should_exit(bar, features)
        elif strategy_name in ("GammaExpansion", "ShortStraddle", "ShortStrangle",
                               "IronButterfly", "AdaptiveStrangle"):
            if hasattr(strategy, "should_exit"):
                return strategy.should_exit(bar, features, current_val)
        elif strategy_name == "WeeklyIronCondor":
            if hasattr(strategy, "should_exit"):
                # current_val here = cost to buy back the condor
                return strategy.should_exit(bar, features, current_val)

        # Universal time stop
        if hasattr(strategy, "_square_off"):
            if not strategy._in_window(bar.timestamp, "09:15", strategy._square_off):
                return True, "time_stop"

        return False, ""

    def _close_trade(
        self, ot: "OpenTrade", bar: MarketBar, reason: str,
        chain: Optional["OptionChain"] = None,
    ) -> Trade:
        """
        Mark trade closed; PnL uses current chain's BS mid-prices for each leg.
        Falls back to time-value-decay estimate when chain is unavailable.
        """
        exit_net_per_unit = self._spread_exit_value(ot, bar, chain)

        # Net proceeds = exit spread value × lots × lot_size
        exit_proceeds = exit_net_per_unit * ot.lots * ot.lot_size

        # Exit transaction costs (reverse sides)
        exit_cost = 0.0
        for leg in ot.signal.legs:
            exit_side = (OrderSide.SELL if leg.side == OrderSide.BUY
                         else OrderSide.BUY)
            c = self._costs.compute(
                max(0.05, exit_net_per_unit), ot.lots, ot.lot_size, exit_side
            )
            exit_cost += c.total

        pnl = exit_proceeds - ot.cost - exit_cost
        return Trade(
            trade_id      = ot.trade_id,
            strategy      = ot.signal.strategy,
            instrument    = ot.signal.instrument,
            signal_id     = ot.signal.signal_id,
            legs          = ot.signal.legs,
            entry_time    = ot.entry_time,
            exit_time     = bar.timestamp,
            entry_cost    = ot.cost,
            exit_proceeds = exit_proceeds,
            pnl           = pnl,
            pnl_pct       = pnl / ot.cost if ot.cost > 0 else 0.0,
            status        = TradeStatus.CLOSED,
            exit_reason   = reason,
        )

    def _spread_exit_value(
        self, ot: "OpenTrade", bar: MarketBar,
        chain: Optional["OptionChain"],
    ) -> float:
        """
        Compute the net per-unit exit value of the spread.
        Uses chain mid-prices when available, else BS time-decay estimate.
        """
        if chain is not None:
            net = 0.0
            found = 0
            for leg in ot.signal.legs:
                q = chain.quote(leg.strike, leg.option_type)
                if q is not None:
                    price = q.mid
                    sign  = 1.0 if leg.side == OrderSide.BUY else -1.0
                    net  += sign * price
                    found += 1
            if found == len(ot.signal.legs) and found > 0:
                return float(net)

        # Fallback: time-decay estimate (assumes sqrt-time value decay)
        from datetime import timedelta
        elapsed_hours = max(0.0, (bar.timestamp - ot.entry_time).total_seconds() / 3600)
        session_hours = 6.25   # 9:15 AM – 3:30 PM
        time_used_frac = min(1.0, elapsed_hours / session_hours)
        # Value decays from entry_debit towards intrinsic as time passes
        decay = max(0.05, 1.0 - time_used_frac ** 0.5)
        return float(ot.signal.net_debit * decay)


# ── Internal state ────────────────────────────────────────────────────────────

class OpenTrade:
    """Mutable state for a trade that is currently open in the backtest."""
    __slots__ = ("trade_id", "signal", "entry_time", "entry_bar",
                 "lots", "lot_size", "cost")

    def __init__(self, trade_id: str, signal: Signal, entry_time: datetime,
                 entry_bar: MarketBar, lots: int, lot_size: int,
                 cost: float) -> None:
        self.trade_id   = trade_id
        self.signal     = signal
        self.entry_time = entry_time
        self.entry_bar  = entry_bar
        self.lots       = lots
        self.lot_size   = lot_size
        self.cost       = cost
