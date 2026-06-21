"""
Strategy B — Intraday Trend Following.

Logic:
  1. ADX > 25 confirms a trending regime.
  2. Price above VWAP + positive breadth  → Bull Call Spread.
     Price below VWAP + negative breadth → Bear Put Spread.
  3. Exit: VWAP crossover | end of session.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, List, Optional

import numpy as np

from algo_platform.core.config import PlatformConfig, StrategyBConfig
from algo_platform.core.types import (
    FeatureVector, Instrument, OptionChain, Signal, SignalDirection, MarketBar,
)
from algo_platform.strategies.base import BaseStrategy

logger = logging.getLogger("platform.strategies.trend_following")


def _wilder_smooth(values: List[float], period: int) -> List[float]:
    """Wilder smoothing (like an EMA with alpha=1/period)."""
    if len(values) < period:
        return [sum(values) / len(values)] * len(values)
    result = [sum(values[:period]) / period]
    for v in values[period:]:
        result.append(result[-1] * (period - 1) / period + v / period)
    return result


def compute_adx(bars: List[MarketBar], period: int = 14) -> tuple[float, float, float]:
    """Returns (ADX, +DI, -DI)."""
    if len(bars) < period + 1:
        return 0.0, 0.0, 0.0

    plus_dm:  List[float] = []
    minus_dm: List[float] = []
    trs:      List[float] = []

    for i in range(1, len(bars)):
        up   = bars[i].high - bars[i - 1].high
        down = bars[i - 1].low - bars[i].low
        pdm  = up   if up > down and up > 0   else 0.0
        ndm  = down if down > up and down > 0 else 0.0
        tr   = max(bars[i].high - bars[i].low,
                   abs(bars[i].high - bars[i - 1].close),
                   abs(bars[i].low  - bars[i - 1].close))
        plus_dm.append(pdm)
        minus_dm.append(ndm)
        trs.append(tr)

    s_pdm = _wilder_smooth(plus_dm, period)
    s_ndm = _wilder_smooth(minus_dm, period)
    s_tr  = _wilder_smooth(trs,  period)

    plus_di  = [100.0 * p / t if t > 0 else 0.0 for p, t in zip(s_pdm, s_tr)]
    minus_di = [100.0 * n / t if t > 0 else 0.0 for n, t in zip(s_ndm, s_tr)]
    dx       = [abs(p - n) / (p + n) * 100 if p + n > 0 else 0.0
                for p, n in zip(plus_di, minus_di)]

    if len(dx) < period:
        adx = float(np.mean(dx)) if dx else 0.0
    else:
        adx_series = _wilder_smooth(dx, period)
        adx = adx_series[-1]

    return float(adx), float(plus_di[-1]), float(minus_di[-1])


class TrendFollowingStrategy(BaseStrategy):
    """Strategy B: ADX-gated intraday trend trade via directional spreads."""

    name = "IntradayTrend"

    def __init__(self, instrument: Instrument, config: PlatformConfig,
                 quantity: int = 1) -> None:
        super().__init__(instrument, config)
        self._cfg: StrategyBConfig = config.strategy_b
        self._quantity  = quantity
        self._lot_size  = config.lot_size(instrument.value)
        self._width     = config.spread_width(instrument.value, "B")

        self._bars: List[MarketBar] = []
        self._session_bars: List[MarketBar] = []

        # Track active trade direction to detect VWAP crossover exit
        self._active_direction: SignalDirection = SignalDirection.NEUTRAL
        self._signal_emitted_today: bool = False

    def new_session(self) -> None:
        self._session_bars = []
        self._signal_emitted_today = False
        self._active_direction = SignalDirection.NEUTRAL

    def generate_signal(
        self,
        bar:      MarketBar,
        chain:    Optional[OptionChain],
        features: FeatureVector,
    ) -> Optional[Signal]:
        if not self._is_active or chain is None:
            return None

        self._bars.append(bar)
        self._session_bars.append(bar)

        if not self._in_window(bar.timestamp,
                               self._cfg.entry_start, self._cfg.entry_end):
            return None

        if len(self._bars) < self._cfg.min_warmup_bars:
            return None

        # ── Only one signal per session ────────────────────────────────────────
        if self._signal_emitted_today:
            return None

        # ── ADX regime filter ──────────────────────────────────────────────────
        adx, plus_di, minus_di = compute_adx(self._bars[-60:], self._cfg.adx_period)
        if adx < self._cfg.adx_threshold:
            return None

        # ── VWAP + breadth signal ──────────────────────────────────────────────
        above_vwap = features.vwap_distance > 0
        below_vwap = features.vwap_distance < 0
        pos_breadth = features.breadth > self._cfg.breadth_threshold
        neg_breadth = features.breadth < (1.0 - self._cfg.breadth_threshold)

        # DI alignment adds conviction (bull: +DI > -DI)
        di_bull = plus_di > minus_di
        di_bear = minus_di > plus_di

        if above_vwap and pos_breadth and di_bull:
            direction = SignalDirection.LONG
        elif below_vwap and neg_breadth and di_bear:
            direction = SignalDirection.SHORT
        else:
            return None

        # ── Build spread ───────────────────────────────────────────────────────
        atm = chain.atm_strike()
        if direction == SignalDirection.LONG:
            legs = self._build_call_spread(chain, atm, self._width,
                                           self._quantity, self._lot_size)
        else:
            legs = self._build_put_spread(chain, atm, self._width,
                                          self._quantity, self._lot_size)

        if not legs:
            return None

        self._active_direction     = direction
        self._signal_emitted_today = True
        self._in_trade             = True

        debit = self._net_debit(legs)
        logger.info(
            "StrategyB SIGNAL %s %s | ADX=%.1f VWAP_dist=%.3f breadth=%.2f",
            direction.value, self.instrument.value,
            adx, features.vwap_distance, features.breadth,
        )

        return Signal(
            strategy   = self.name,
            instrument = self.instrument,
            direction  = direction,
            timestamp  = bar.timestamp,
            legs       = legs,
            net_debit  = debit,
            max_loss   = self._max_loss(legs),
            max_profit = self._max_profit(legs, self._width),
            confidence = self._confidence(adx, features),
            features   = features,
            metadata   = {"adx": adx, "+DI": plus_di, "-DI": minus_di},
        )

    def should_exit(
        self,
        bar:      MarketBar,
        features: FeatureVector,
    ) -> tuple[bool, str]:
        if not self._in_trade:
            return False, ""

        # Time stop (only exit — VWAP crossover removed after diagnosis showed it
        # fires on 67% of trades at 6% win rate while time-stop trades have 80% win rate)
        if not self._in_window(bar.timestamp,
                               self._cfg.entry_start, self._cfg.square_off):
            self._in_trade = False
            return True, "time_stop"

        return False, ""

    def _confidence(self, adx: float, f: FeatureVector) -> float:
        adx_score     = min(1.0, (adx - self._cfg.adx_threshold) / 25.0)
        breadth_score = abs(f.breadth - 0.5) * 2.0
        return float(min(1.0, 0.4 + 0.35 * adx_score + 0.25 * breadth_score))
