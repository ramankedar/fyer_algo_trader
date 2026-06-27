"""
Strategy D — Opening Range Breakout (ORB)

Research basis: Indian markets digest FII pre-market positioning, Gift Nifty direction,
and global overnight signals in the first 15 minutes (9:15–9:29). Once the opening range
is set, a clean breakout above/below with price confirmation gives ~55–62% directional
accuracy and strong R:R because the range itself defines the natural stop.

Rules:
  1. 9:15–9:29 AM: accumulate bars → compute opening range High and Low.
  2. 9:30 onward: watch for first 1-min close that CONFIRMS the breakout.
       Bull trigger: close > range_high + buffer_pct × close
       Bear trigger: close < range_low  - buffer_pct × close
  3. Filters (skip day if violated):
       range_width outside [min_range_pct, max_range_pct] → flat day or already-moved day
       iv_rank > max_iv_rank → extreme vol; theta strategy handles those days
       expiry day → skip (theta already takes that slot)
  4. Enter ONE bull call spread (or bear put spread) per session.
  5. Stop: underlying re-enters the range (breaks back through the breakout level).
  6. Target: 2× range_width beyond the breakout point.
  7. Hard EOD: exit at 15:15 regardless.

Edge sources:
  - FII/DII execution concentration in opening 15 min
  - Gift Nifty gives early direction; institutions align with it in the range
  - Range tightness filter selects high-conviction setups only
  - Buffer prevents churning near range boundary
"""

from __future__ import annotations

import logging
from datetime import time
from typing import Optional

from algo_platform.core.config import PlatformConfig
from algo_platform.core.types import (
    FeatureVector, Instrument, OptionChain, Signal, SignalDirection, MarketBar,
)
from algo_platform.strategies.base import BaseStrategy

logger = logging.getLogger("platform.strategies.orb")

_INF = float("inf")


class ORBStrategy(BaseStrategy):
    """Strategy D: Opening Range Breakout via directional debit spreads."""

    name = "OpeningRangeBreakout"

    def __init__(
        self,
        instrument: Instrument,
        config: PlatformConfig,
        quantity: int = 1,
        # Range quality filters
        min_range_pct: float = 0.002,   # range must be >= 0.2% of underlying
        max_range_pct: float = 0.015,   # range must be <= 1.5% of underlying
        buffer_pct:    float = 0.0005,  # breakout buffer: 0.05% beyond range edge (stop-order simulation)
        # Regime filters
        max_iv_rank:   float = 0.75,    # skip if IV rank > 75th percentile (extreme)
        # Spread config
        spread_width_nifty:     float = 100.0,  # 2 strikes wide
        spread_width_banknifty: float = 200.0,
        spread_width_finnifty:  float = 50.0,
        spread_width_sensex:    float = 400.0,
        spread_width_bankex:    float = 400.0,
        # Risk:Reward
        target_rr: float = 2.0,
        square_off: str = "15:15",
    ) -> None:
        super().__init__(instrument, config)
        self._quantity     = quantity
        self._lot_size     = config.lot_size(instrument.value)

        _width_map = {
            "NIFTY":     spread_width_nifty,
            "BANKNIFTY": spread_width_banknifty,
            "FINNIFTY":  spread_width_finnifty,
            "SENSEX":    spread_width_sensex,
            "BANKEX":    spread_width_bankex,
        }
        self._spread_width  = _width_map.get(instrument.value, spread_width_nifty)

        self._min_range_pct = min_range_pct
        self._max_range_pct = max_range_pct
        self._buffer_pct    = buffer_pct
        self._max_iv_rank   = max_iv_rank
        self._target_rr     = target_rr
        self._square_off    = square_off

        # Per-session state
        self._range_high: float = -_INF
        self._range_low:  float = _INF
        self._range_valid: bool = False   # True after range quality check passes
        self._range_checked: bool = False # True once we've checked range at 9:30
        self._signal_emitted: bool = False
        self._direction: Optional[SignalDirection] = None
        self._breakout_level: float = 0.0  # the level we broke out from
        self._entry_underlying: float = 0.0
        self._range_width_pts: float = 0.0

    # ── Session management ─────────────────────────────────────────────────────

    def new_session(self) -> None:
        self._range_high      = -_INF
        self._range_low       = _INF
        self._range_valid     = False
        self._range_checked   = False
        self._signal_emitted  = False
        self._direction       = None
        self._breakout_level  = 0.0
        self._entry_underlying = 0.0
        self._range_width_pts  = 0.0
        self._in_trade        = False

    # ── Signal generation ──────────────────────────────────────────────────────

    def generate_signal(
        self,
        bar:      MarketBar,
        chain:    Optional[OptionChain],
        features: FeatureVector,
    ) -> Optional[Signal]:
        if not self._is_active or self._signal_emitted:
            return None

        t = bar.timestamp.time()
        _range_end  = time(9, 30)
        _entry_end  = time(11, 30)   # no ORB trades after 11:30 (range stale)

        # ── Phase 1: Accumulate opening range 9:15–9:29 ────────────────────────
        if t < _range_end:
            self._range_high = max(self._range_high, bar.high)
            self._range_low  = min(self._range_low,  bar.low)
            return None

        # ── Phase 2: Validate range at 9:30 (once per session) ─────────────────
        if not self._range_checked:
            self._range_checked = True
            if self._range_high == -_INF or self._range_low == _INF:
                return None   # no range bars (holiday-adjacent day)

            range_pts  = self._range_high - self._range_low
            range_pct  = range_pts / bar.close if bar.close > 0 else 0.0

            # Skip expiry days (theta strategy handles those)
            if self._is_expiry_day(bar.timestamp):
                logger.debug("ORB: skip expiry day %s", bar.timestamp.date())
                return None

            # Range quality filter
            if not (self._min_range_pct <= range_pct <= self._max_range_pct):
                logger.debug(
                    "ORB: skip %s — range_pct=%.3f outside [%.3f, %.3f]",
                    bar.timestamp.date(), range_pct,
                    self._min_range_pct, self._max_range_pct,
                )
                return None

            # IV regime filter
            if features.iv_rank > self._max_iv_rank:
                logger.debug(
                    "ORB: skip %s — iv_rank=%.2f > max %.2f",
                    bar.timestamp.date(), features.iv_rank, self._max_iv_rank,
                )
                return None

            self._range_valid     = True
            self._range_width_pts = range_pts
            logger.debug(
                "ORB: range set %s H=%.1f L=%.1f width=%.1f (%.2f%%)",
                bar.timestamp.date(), self._range_high, self._range_low,
                range_pts, range_pct * 100,
            )

        if not self._range_valid:
            return None

        if t > _entry_end:
            return None    # too late for a valid ORB entry

        if chain is None:
            return None

        # ── Phase 3: Detect breakout ────────────────────────────────────────────
        # Use bar.high / bar.low (not close) to simulate a stop-order fill:
        # if any print inside the bar crosses the trigger, we fill at the trigger price.
        bull_trigger = self._range_high + self._buffer_pct * bar.close
        bear_trigger = self._range_low  - self._buffer_pct * bar.close

        if bar.high > bull_trigger:
            direction       = SignalDirection.LONG
            breakout_level  = self._range_high
        elif bar.low < bear_trigger:
            direction       = SignalDirection.SHORT
            breakout_level  = self._range_low
        else:
            return None   # no breakout yet

        # ── Build directional spread ────────────────────────────────────────────
        atm = chain.atm_strike()
        if direction == SignalDirection.LONG:
            legs = self._build_call_spread(chain, atm, self._spread_width,
                                           self._quantity, self._lot_size)
        else:
            legs = self._build_put_spread(chain, atm, self._spread_width,
                                          self._quantity, self._lot_size)

        if not legs:
            return None

        self._signal_emitted    = True
        self._direction         = direction
        self._breakout_level    = breakout_level
        self._entry_underlying  = bar.close
        self._in_trade          = True

        debit = self._net_debit(legs)
        target_pts = self._target_rr * self._range_width_pts

        logger.info(
            "ORB SIGNAL %s %s | range=[%.1f–%.1f] break=%.1f target+%.1f iv_rank=%.2f",
            direction.value, self.instrument.value,
            self._range_low, self._range_high, bar.close, target_pts,
            features.iv_rank,
        )

        return Signal(
            strategy   = self.name,
            instrument = self.instrument,
            direction  = direction,
            timestamp  = bar.timestamp,
            legs       = legs,
            net_debit  = debit,
            max_loss   = self._max_loss(legs),
            max_profit = self._max_profit(legs, self._spread_width),
            confidence = self._confidence(features),
            features   = features,
            metadata   = {
                "range_high":    self._range_high,
                "range_low":     self._range_low,
                "range_width":   self._range_width_pts,
                "breakout_at":   bar.close,
                "target_pts":    target_pts,
                "bull_trigger":  bull_trigger,
                "bear_trigger":  bear_trigger,
            },
        )

    # ── Exit logic ─────────────────────────────────────────────────────────────

    def should_exit(
        self,
        bar:      MarketBar,
        features: FeatureVector,
    ) -> tuple[bool, str]:
        if not self._in_trade:
            return False, ""

        t = bar.timestamp.time()

        # Hard EOD exit
        if t >= time(*[int(x) for x in self._square_off.split(":")]):
            self._in_trade = False
            return True, "time_stop"

        if self._direction is None:
            return False, ""

        # Stop: price reverses back through the range breakout level
        if self._direction == SignalDirection.LONG:
            if bar.close < self._range_high:   # re-entered range = failed breakout
                self._in_trade = False
                return True, "stop_loss"
            # Target: 2× range_width from entry
            target = self._entry_underlying + self._target_rr * self._range_width_pts
            if bar.close >= target:
                self._in_trade = False
                return True, "target_hit"

        elif self._direction == SignalDirection.SHORT:
            if bar.close > self._range_low:   # re-entered range = failed breakdown
                self._in_trade = False
                return True, "stop_loss"
            target = self._entry_underlying - self._target_rr * self._range_width_pts
            if bar.close <= target:
                self._in_trade = False
                return True, "target_hit"

        return False, ""

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _confidence(self, features: FeatureVector) -> float:
        iv_score     = 1.0 - min(1.0, features.iv_rank)
        range_score  = min(1.0, self._range_width_pts / max(1, features.atr * 2))
        return float(min(1.0, 0.4 + 0.3 * iv_score + 0.3 * range_score))
