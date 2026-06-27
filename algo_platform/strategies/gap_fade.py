"""
Strategy G — Gap & Go / Gap Fade

Research basis: When NIFTY opens with a significant gap vs. prior close, two patterns dominate:

  Gap & GO  (momentum)  — large gaps (>0.8%) with first-bar volume surge get extended.
                          Institutional program trades and momentum algos pile in.
  Gap FADE  (reversion) — small-to-medium gaps (0.3–0.8%) with below-average first-bar
                          volume fill ~58% of the time within 60–75 minutes.

Edge in India:
  - Gift Nifty pre-market often overshoots. Large Gift Nifty gaps = Go; small = Fade.
  - India-US correlation spikes at NSE open then mean-reverts by 11 AM.
  - Institutional desks use VWAP for execution — gaps on low volume revert to VWAP.

Rules:
  1. Track last session's close in prev_close (set in new_session from prior-day last bar).
  2. At 9:15 bar: gap% = (open − prev_close) / prev_close.
  3. |gap%| < min_gap_pct or > max_gap_pct → no trade.
  4. Large gap (≥ large_gap_pct) + first-bar vol > vol_surge_mult × 20-day avg → Go.
  5. Small gap (< large_gap_pct) + first-bar vol ≤ avg → Fade.
  6. Build ATM debit spread in signal direction.
  7. Stop: underlying crosses prev_close (Go) or gap open extreme (Fade).
  8. Target: 2R. Hard exit: 10:30 AM.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import time
from typing import Deque, Optional

from algo_platform.core.config import PlatformConfig
from algo_platform.core.types import (
    FeatureVector, Instrument, OptionChain, Signal, SignalDirection, MarketBar,
)
from algo_platform.strategies.base import BaseStrategy

logger = logging.getLogger("platform.strategies.gap_fade")


class GapTradeStrategy(BaseStrategy):
    """Strategy G: Gap & Go / Gap Fade via directional debit spreads."""

    name = "GapTrade"

    def __init__(
        self,
        instrument: Instrument,
        config: PlatformConfig,
        quantity: int = 1,
        min_gap_pct:    float = 0.003,   # 0.3% minimum gap
        large_gap_pct:  float = 0.008,   # 0.8%: Go vs. Fade boundary
        max_gap_pct:    float = 0.025,   # 2.5% max (crisis = skip)
        vol_surge_mult: float = 1.5,     # Go: first-bar vol > 1.5× avg
        vol_lookback:   int   = 20,      # days for avg first-bar volume
        max_iv_rank:    float = 0.80,
        spread_width_nifty:     float = 100.0,
        spread_width_banknifty: float = 200.0,
        spread_width_finnifty:  float = 50.0,
        spread_width_sensex:    float = 400.0,
        spread_width_bankex:    float = 400.0,
        target_rr:      float = 2.0,
        square_off:     str   = "10:30",
    ) -> None:
        super().__init__(instrument, config)
        self._quantity = quantity
        self._lot_size = config.lot_size(instrument.value)

        _width_map = {
            "NIFTY":     spread_width_nifty,
            "BANKNIFTY": spread_width_banknifty,
            "FINNIFTY":  spread_width_finnifty,
            "SENSEX":    spread_width_sensex,
            "BANKEX":    spread_width_bankex,
        }
        self._spread_width = _width_map.get(instrument.value, spread_width_nifty)

        self._min_gap_pct    = min_gap_pct
        self._large_gap_pct  = large_gap_pct
        self._max_gap_pct    = max_gap_pct
        self._vol_surge_mult = vol_surge_mult
        self._max_iv_rank    = max_iv_rank
        self._target_rr      = target_rr
        self._square_off     = square_off

        # Rolling first-bar volume (needed to classify volume surge)
        self._first_bar_vols: Deque[float] = deque(maxlen=vol_lookback)

        # Carry-over between sessions
        self._last_bar_close: float = 0.0   # updated every bar; becomes prev_close next session

        # Per-session state
        self._prev_close:      float = 0.0
        self._gap_signal_done: bool  = False
        self._signal_emitted:  bool  = False
        self._direction:       Optional[SignalDirection] = None
        self._stop_level:      float = 0.0
        self._target_level:    float = 0.0

    def new_session(self) -> None:
        # Capture carry-over before resetting
        self._prev_close      = self._last_bar_close   # prior session's last close
        self._gap_signal_done = False
        self._signal_emitted  = False
        self._direction       = None
        self._stop_level      = 0.0
        self._target_level    = 0.0
        self._in_trade        = False

    def generate_signal(
        self,
        bar:      MarketBar,
        chain:    Optional[OptionChain],
        features: FeatureVector,
    ) -> Optional[Signal]:
        # Always update carry-over close (runs even if we return early)
        self._last_bar_close = bar.close

        if not self._is_active or self._signal_emitted or self._gap_signal_done:
            return None

        t = bar.timestamp.time()
        if t != time(9, 15):
            if t > time(9, 15):
                self._gap_signal_done = True   # missed first bar window
            return None

        # ── First bar of session: compute gap ──────────────────────────────────
        self._gap_signal_done = True

        if self._prev_close <= 0:
            return None   # no prior close (first session ever)

        gap_pct  = (bar.open - self._prev_close) / self._prev_close
        abs_gap  = abs(gap_pct)

        # Record first-bar volume for future sessions' avg computation
        first_bar_vol = bar.volume
        avg_first_vol = (sum(self._first_bar_vols) / len(self._first_bar_vols)
                         if self._first_bar_vols else 0.0)
        if first_bar_vol > 0:
            self._first_bar_vols.append(first_bar_vol)

        if abs_gap < self._min_gap_pct or abs_gap > self._max_gap_pct:
            return None

        if features.iv_rank > self._max_iv_rank:
            return None

        if self._is_expiry_day(bar.timestamp):
            return None   # theta strategy owns expiry days

        # Classify: Go or Fade
        volume_surge = (first_bar_vol > self._vol_surge_mult * avg_first_vol
                        and avg_first_vol > 0)

        if abs_gap >= self._large_gap_pct and volume_surge:
            trade_type = "go"
            direction  = SignalDirection.LONG if gap_pct > 0 else SignalDirection.SHORT
        elif abs_gap < self._large_gap_pct and not volume_surge:
            trade_type = "fade"
            direction  = SignalDirection.SHORT if gap_pct > 0 else SignalDirection.LONG
        else:
            return None   # conflicting signals (large gap, low vol OR small gap, high vol)

        if chain is None:
            return None

        atm = chain.atm_strike()
        if direction == SignalDirection.LONG:
            legs = self._build_call_spread(chain, atm, self._spread_width,
                                           self._quantity, self._lot_size)
        else:
            legs = self._build_put_spread(chain, atm, self._spread_width,
                                          self._quantity, self._lot_size)

        if not legs:
            return None

        # Stops and targets on the underlying
        risk_pts = abs(bar.open - self._prev_close)
        if trade_type == "go":
            if direction == SignalDirection.LONG:
                stop_level   = self._prev_close
                target_level = bar.open + self._target_rr * risk_pts
            else:
                stop_level   = self._prev_close
                target_level = bar.open - self._target_rr * risk_pts
        else:  # fade
            stop_level   = bar.open   # if gap keeps extending, stop out
            target_level = self._prev_close   # target = fill the gap

        self._direction      = direction
        self._stop_level     = stop_level
        self._target_level   = target_level
        self._signal_emitted = True
        self._in_trade       = True

        debit = self._net_debit(legs)
        logger.info(
            "GAP SIGNAL %s %s | type=%s gap=%.2f%% vol_surge=%s avg_vol=%.0f",
            direction.value, self.instrument.value,
            trade_type, gap_pct * 100, volume_surge, avg_first_vol,
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
            confidence = self._confidence(abs_gap, volume_surge, features),
            features   = features,
            metadata   = {
                "gap_pct":    gap_pct,
                "trade_type": trade_type,
                "stop":       stop_level,
                "target":     target_level,
                "vol_surge":  volume_surge,
                "prev_close": self._prev_close,
            },
        )

    def should_exit(
        self,
        bar:      MarketBar,
        features: FeatureVector,
    ) -> tuple[bool, str]:
        self._last_bar_close = bar.close

        if not self._in_trade:
            return False, ""

        t = bar.timestamp.time()
        sq_h, sq_m = [int(x) for x in self._square_off.split(":")]
        if t >= time(sq_h, sq_m):
            self._in_trade = False
            return True, "time_stop"

        if self._direction is None:
            return False, ""

        if self._direction == SignalDirection.LONG:
            if bar.close < self._stop_level:
                self._in_trade = False
                return True, "stop_loss"
            if bar.close >= self._target_level:
                self._in_trade = False
                return True, "target_hit"
        else:
            if bar.close > self._stop_level:
                self._in_trade = False
                return True, "stop_loss"
            if bar.close <= self._target_level:
                self._in_trade = False
                return True, "target_hit"

        return False, ""

    def _confidence(self, abs_gap: float, vol_surge: bool, f: FeatureVector) -> float:
        gap_score = min(1.0, abs_gap / self._large_gap_pct)
        vol_score = 0.8 if vol_surge else 0.4
        iv_score  = 1.0 - min(1.0, f.iv_rank)
        return float(min(1.0, 0.3 + 0.3 * gap_score + 0.2 * vol_score + 0.2 * iv_score))
