"""
Volatility-targeted position sizer with drawdown scaling.

Algorithm
---------
1.  Compute target daily vol = target_annual_vol / sqrt(252).
2.  Risk budget per trade    = capital × risk_per_trade_pct.
3.  Intrinsic max-loss of the spread determines loss_per_lot.
4.  base_lots = risk_budget / loss_per_lot.
5.  Scale down linearly when portfolio drawdown is in [soft, hard] range.
6.  Return 0 if drawdown >= hard (halt trading).
"""

from __future__ import annotations

import math
from typing import Optional

from algo_platform.core.config import RiskConfig
from algo_platform.core.types import FeatureVector, Signal


class VolatilityTargetSizer:
    """
    Computes integer lot count for each signal, respecting:
      - 0.5% risk per trade
      - 15% annual volatility target
      - Drawdown scaling (soft=8% → hard=15%)
    """

    def __init__(self, cfg: RiskConfig) -> None:
        self._cfg = cfg

    @property
    def capital(self) -> float:
        return self._cfg.capital

    def compute_lots(
        self,
        nav:           float,
        signal:        Signal,
        lot_size:      int,
        features:      Optional[FeatureVector] = None,
    ) -> int:
        """
        Returns integer number of lots to trade.
        Returns 0 if risk limits are breached or trade should be skipped.
        """
        cfg = self._cfg

        # Drawdown check
        dd_pct = self._drawdown_pct(nav)
        if dd_pct >= cfg.dd_hard_pct:
            return 0

        # Risk budget: ₹ we are willing to lose on this trade
        risk_budget = nav * cfg.risk_per_trade_pct

        # Max loss per lot (intrinsic to the spread structure)
        if signal.max_loss > 0 and len(signal.legs) > 0:
            base_lots_for_signal = max(1, len(set(
                leg.quantity for leg in signal.legs
            )))
            loss_per_lot = signal.max_loss / max(1, base_lots_for_signal)
        else:
            # Fallback: assume we risk the full debit per lot
            loss_per_lot = signal.net_debit * lot_size

        if loss_per_lot <= 0:
            return 0

        base_lots = max(1, int(risk_budget / loss_per_lot))

        # Volatility target adjustment
        if features is not None:
            rv = max(features.realized_vol, 0.05)
            vol_scale = cfg.target_annual_vol / rv
            # Apply vol scaling only when vol is elevated (don't over-size in low-vol)
            if rv > cfg.target_annual_vol:
                base_lots = max(1, int(base_lots * vol_scale))

        # Drawdown scaling
        base_lots = self._apply_drawdown_scale(base_lots, dd_pct)

        # Capital guard: never risk more than 20% of NAV on one trade
        max_lots_by_capital = max(1, int(nav * 0.20 / (loss_per_lot + 1e-8)))
        return min(base_lots, max_lots_by_capital)

    # ── Limit checks ──────────────────────────────────────────────────────────

    def check_daily_limit(self, daily_pnl: float) -> bool:
        """True if daily loss limit is NOT yet breached."""
        limit = -self._cfg.capital * self._cfg.max_daily_loss_pct
        return daily_pnl >= limit

    def check_weekly_limit(self, weekly_pnl: float) -> bool:
        """True if weekly loss limit is NOT yet breached."""
        limit = -self._cfg.capital * self._cfg.max_weekly_loss_pct
        return weekly_pnl >= limit

    def check_portfolio_limit(self, nav: float) -> bool:
        """True if portfolio drawdown limit is NOT yet breached."""
        return self._drawdown_pct(nav) < self._cfg.max_portfolio_dd_pct

    # ── Private ───────────────────────────────────────────────────────────────

    def _drawdown_pct(self, nav: float) -> float:
        loss = self._cfg.capital - nav
        return max(0.0, loss / self._cfg.capital)

    def _apply_drawdown_scale(self, lots: int, dd_pct: float) -> int:
        cfg  = self._cfg
        soft = cfg.dd_soft_pct
        hard = cfg.dd_hard_pct

        if dd_pct < soft:
            return lots

        # Linear ramp-down from 1.0 at soft → min_size_factor at hard
        scale = 1.0 - (dd_pct - soft) / (hard - soft + 1e-8)
        scale = max(cfg.min_size_factor, min(1.0, scale))
        return max(1, int(lots * scale))
