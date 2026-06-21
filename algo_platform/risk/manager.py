"""
Platform Risk Manager — real-time risk state, Greek aggregation, VaR, and
strategy PnL attribution. Designed to be called once per bar in live trading.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict, deque
from datetime import date, datetime
from typing import Dict, List, Optional

import numpy as np

from algo_platform.core.config import PlatformConfig
from algo_platform.core.types import Position, RiskState, Trade, TradeStatus
from algo_platform.risk.sizing import VolatilityTargetSizer

logger = logging.getLogger("platform.risk.manager")

_TRADING_DAYS_PER_YEAR = 252


class PlatformRiskManager:
    """
    Thread-safe risk manager that:
      - Tracks real-time drawdown
      - Enforces daily / weekly / portfolio loss limits
      - Aggregates Greeks across all open positions
      - Computes parametric daily VaR
      - Maintains per-strategy PnL attribution
    """

    def __init__(self, config: PlatformConfig) -> None:
        self._cfg     = config
        self._sizer   = VolatilityTargetSizer(config.risk)
        self._lock    = threading.Lock()

        self._capital: float = config.risk.capital
        self._nav:     float = config.risk.capital
        self._peak_nav:float = config.risk.capital

        # P&L tracking
        self._daily_pnl:   float = 0.0
        self._weekly_pnl:  float = 0.0
        self._total_pnl:   float = 0.0
        self._current_day: Optional[date] = None
        self._current_week:int = -1

        # Returns history for VaR
        self._daily_returns: deque = deque(maxlen=config.monitoring.var_window)

        # Strategy attribution
        self._strategy_pnl: Dict[str, float] = defaultdict(float)

        self._halted      = False
        self._halt_reason = ""

    # ── Public ─────────────────────────────────────────────────────────────────

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def nav(self) -> float:
        return self._nav

    def on_trade_closed(self, trade: Trade) -> None:
        """Register a completed trade; update all PnL ledgers."""
        with self._lock:
            self._total_pnl             += trade.pnl
            self._daily_pnl             += trade.pnl
            self._weekly_pnl            += trade.pnl
            self._strategy_pnl[trade.strategy] += trade.pnl
            self._nav                   += trade.pnl
            self._peak_nav = max(self._peak_nav, self._nav)

        self._check_limits()

    def on_bar(self, ts: datetime, positions: List[Position]) -> RiskState:
        """
        Called once per bar. Updates session P&L windows and returns state.
        """
        with self._lock:
            # Reset daily ledger at session start
            today = ts.date()
            if self._current_day != today:
                if self._current_day is not None:
                    # Record yesterday's return
                    daily_ret = self._daily_pnl / (self._nav - self._daily_pnl + 1e-8)
                    self._daily_returns.append(daily_ret)
                self._daily_pnl   = 0.0
                self._current_day = today

            # Reset weekly ledger on Mondays
            iso_week = ts.isocalendar()[1]
            if self._current_week != iso_week:
                self._weekly_pnl  = 0.0
                self._current_week = iso_week

            # Aggregate Greeks
            td = tg = tv = tth = 0.0
            unreal_pnl = 0.0
            for p in positions:
                td  += p.delta   * p.quantity
                tg  += p.gamma   * p.quantity
                tv  += p.vega    * p.quantity
                tth += p.theta   * p.quantity
                unreal_pnl += p.unrealized_pnl

            current_dd = (self._peak_nav - self._nav) / self._peak_nav \
                if self._peak_nav > 0 else 0.0
            max_dd = current_dd  # would normally track peak historically

            var_99 = self._parametric_var()

            state = RiskState(
                timestamp        = ts,
                capital          = self._capital,
                nav              = self._nav + unreal_pnl,
                daily_pnl        = self._daily_pnl + unreal_pnl,
                weekly_pnl       = self._weekly_pnl,
                total_pnl        = self._total_pnl,
                peak_nav         = self._peak_nav,
                current_drawdown = current_dd,
                max_drawdown     = max_dd,
                daily_var_99     = var_99,
                open_positions   = len(positions),
                total_delta      = td,
                total_gamma      = tg,
                total_theta      = tth,
                total_vega       = tv,
                is_trading_halted= self._halted,
                halt_reason      = self._halt_reason,
                strategy_pnl     = dict(self._strategy_pnl),
            )

        return state

    def can_open(self, strategy_name: str) -> bool:
        """True if the risk manager allows opening a new position."""
        if self._halted:
            return False
        cfg = self._cfg.risk
        if self._daily_pnl < -self._capital * cfg.max_daily_loss_pct:
            logger.warning("Daily loss limit hit — no new trades")
            return False
        if self._weekly_pnl < -self._capital * cfg.max_weekly_loss_pct:
            logger.warning("Weekly loss limit hit — no new trades")
            return False
        if not self._sizer.check_portfolio_limit(self._nav):
            logger.warning("Portfolio drawdown limit hit — no new trades")
            return False
        return True

    def lot_size_for(
        self,
        strategy_name: str,
        nav:           float,
        signal,                 # Signal object
        lot_size:      int,
        features=None,
    ) -> int:
        """Delegate to the sizer; returns 0 if any limit breached."""
        if not self.can_open(strategy_name):
            return 0
        return self._sizer.compute_lots(nav, signal, lot_size, features)

    # ── Private ────────────────────────────────────────────────────────────────

    def _check_limits(self) -> None:
        """Halt trading if any hard limit is breached."""
        cfg = self._cfg.risk
        with self._lock:
            if self._daily_pnl < -self._capital * cfg.max_daily_loss_pct:
                self._halt("Daily loss limit exceeded")
            elif self._weekly_pnl < -self._capital * cfg.max_weekly_loss_pct:
                self._halt("Weekly loss limit exceeded")
            elif not self._sizer.check_portfolio_limit(self._nav):
                self._halt("Portfolio drawdown limit exceeded")

    def _halt(self, reason: str) -> None:
        if not self._halted:
            self._halted      = True
            self._halt_reason = reason
            logger.critical("TRADING HALTED: %s", reason)

    def _parametric_var(self, confidence: float = 0.99) -> float:
        """
        One-day 99% parametric VaR based on recent daily returns.
        Assumes normal distribution — serves as a quick dashboard figure.
        """
        if len(self._daily_returns) < 5:
            return 0.0
        rets  = np.array(list(self._daily_returns))
        mu    = float(np.mean(rets))
        sigma = float(np.std(rets, ddof=1))
        from scipy.stats import norm
        z     = norm.ppf(1.0 - confidence)
        var   = -(mu + z * sigma) * self._nav
        return max(0.0, var)
