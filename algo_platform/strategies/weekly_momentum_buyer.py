"""
WeeklyMomentumBuyerStrategy — Convexity sleeve of the Barbell Portfolio.

Role in the Barbell
-------------------
The BarbellStrangle collects theta every week and LOSES when markets
make big directional moves. This strategy does the opposite: it loses
small premium each quiet week but WINS BIG on trending weeks.

Together they form the true barbell:
  Quiet market week  → Strangle wins ₹1,500  |  Buyer loses ₹200
  Trend/crisis week  → Strangle loses ₹3,000  |  Buyer wins ₹8,000+

Why this strategy is "convex"
------------------------------
Convexity = asymmetric payoff.
  Buy 1 ATM call on Monday: cost = ₹75 × price (say ₹200/share = ₹15,000)
  If NIFTY up 2% by Thursday: call worth 2×–5× entry → gain ₹15K–45K
  If NIFTY flat/down:          call decays → lose ₹15K max

  Win probability ≈ 35%, but WIN/LOSS ratio ≈ 4:1
  → Positive expected value on trending weeks

Entry logic
-----------
Every Monday (or Tuesday if Monday holiday) at 09:30 AM.
Direction: determined by 5-day momentum (last Friday close vs prior Friday).
  • Upward momentum  → BUY ATM call spread (limited cost, directional)
  • Downward momentum→ BUY ATM put spread
  • Flat             → Skip (saves capital for truly trending weeks)

ADX confirmation: only enter if ADX > 22 (some trend exists).

Capital sleeve: config.risk.convexity_capital (₹80K default).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Deque, List, Optional

from algo_platform.core.config import PlatformConfig
from algo_platform.core.types import (
    FeatureVector, Instrument, OptionChain, OptionType,
    OrderSide, Signal, SignalDirection, SpreadLeg, MarketBar,
)
from algo_platform.strategies.base import BaseStrategy
from algo_platform.strategies.trend_following import compute_adx

logger = logging.getLogger("platform.strategies.weekly_momentum_buyer")

_STEP = {
    "NIFTY": 50.0, "BANKNIFTY": 100.0, "FINNIFTY": 25.0,
    "SENSEX": 100.0, "BANKEX": 100.0,
}
_FIVE_DAY_BARS = 375 * 5   # approx bars in 5 trading days


class WeeklyMomentumBuyerStrategy(BaseStrategy):
    """
    Convexity sleeve: buys a directional debit spread on Monday morning
    when 5-day price momentum is confirmed by ADX.

    On a quiet week: pays small theta → small loss.
    On a trending week: directional payoff → large win.
    This creates the negative correlation needed for the Barbell.
    """

    name = "WeeklyMomentumBuyer"

    def __init__(
        self,
        instrument:         Instrument,
        config:             PlatformConfig,
        quantity:           int   = 1,
        spread_width_pts:   float = 0.0,   # 0 → use config default
        adx_threshold:      float = 22.0,
        momentum_min_pct:   float = 0.003,  # 0.3% min 5-day move to qualify
        profit_target_pct:  float = 1.50,   # take profit at 150% gain
        stop_loss_pct:      float = 0.70,   # stop if spread worth <30% of entry
    ) -> None:
        super().__init__(instrument, config)
        self._quantity         = quantity
        self._lot_size         = config.lot_size(instrument.value)
        self._step             = _STEP.get(instrument.value, 50.0)
        self._spread_width     = spread_width_pts or config.spread_width(instrument.value, "B")
        self._adx_threshold    = adx_threshold
        self._momentum_min_pct = momentum_min_pct
        self._profit_target    = profit_target_pct    # >1 = profit multiple
        self._stop_loss_pct    = stop_loss_pct        # fraction of entry cost

        self._entry_start = "09:25"
        self._entry_end   = "09:45"
        self._square_off  = "15:15"

        self._bars:            List[MarketBar] = []
        self._entry_cost:      float = 0.0
        self._signal_this_week:bool  = False
        self._direction:       SignalDirection = SignalDirection.NEUTRAL

    def new_session(self) -> None:
        if hasattr(self, '_bars') and self._bars:
            # Reset weekly signal on Mondays only
            if self._bars[-1].timestamp.weekday() == 0:
                self._signal_this_week = False
        self._in_trade = False

    def generate_signal(
        self,
        bar:      MarketBar,
        chain:    Optional[OptionChain],
        features: FeatureVector,
    ) -> Optional[Signal]:
        if not self._is_active or chain is None:
            return None

        self._bars.append(bar)

        # Only enter on Monday (or Tuesday if Monday is holiday proxy)
        wd = bar.timestamp.weekday()
        if wd not in (0, 1):    # Mon=0, Tue=1
            return None
        if self._signal_this_week or self._in_trade:
            return None
        if not self._in_window(bar.timestamp, self._entry_start, self._entry_end):
            return None
        if len(self._bars) < _FIVE_DAY_BARS:
            return None   # warmup: need 5 days of history

        # 5-day momentum filter
        five_days_ago_close = self._bars[-_FIVE_DAY_BARS].close
        current_close       = bar.close
        momentum_pct        = (current_close - five_days_ago_close) / five_days_ago_close

        if abs(momentum_pct) < self._momentum_min_pct:
            return None   # market too flat this week → skip

        # ADX confirmation (need some trend to justify buy)
        adx, plus_di, minus_di = compute_adx(self._bars[-60:], 14)
        if adx < self._adx_threshold:
            return None

        # Direction: momentum + DI alignment
        if momentum_pct > 0 and plus_di > minus_di:
            direction = SignalDirection.LONG    # bullish momentum → buy call spread
        elif momentum_pct < 0 and minus_di > plus_di:
            direction = SignalDirection.SHORT   # bearish momentum → buy put spread
        else:
            return None   # conflicting signals → skip

        # Build the spread
        atm = chain.atm_strike()
        if direction == SignalDirection.LONG:
            legs = self._build_call_spread(chain, atm, self._spread_width,
                                           self._quantity, self._lot_size)
        else:
            legs = self._build_put_spread(chain, atm, self._spread_width,
                                          self._quantity, self._lot_size)

        if not legs:
            return None

        debit      = self._net_debit(legs)
        max_loss   = self._max_loss(legs)
        max_profit = self._max_profit(legs, self._spread_width)

        self._entry_cost       = debit
        self._signal_this_week = True
        self._in_trade         = True
        self._direction        = direction

        logger.info(
            "WeeklyMomentumBuyer %s %s | ATM=%.0f | momentum=%.2f%% "
            "ADX=%.1f | debit=%.2f | potential_gain=%.2f",
            direction.value, self.instrument.value, atm,
            momentum_pct * 100, adx, debit, max_profit / self._lot_size,
        )

        return Signal(
            strategy   = self.name,
            instrument = self.instrument,
            direction  = direction,
            timestamp  = bar.timestamp,
            legs       = legs,
            net_debit  = debit,
            max_loss   = max_loss,
            max_profit = max_profit,
            confidence = min(1.0, abs(momentum_pct) / 0.02 * 0.5 + adx / 40 * 0.5),
            features   = features,
            metadata   = {"momentum_pct": momentum_pct, "adx": adx,
                          "spread_width": self._spread_width},
        )

    def should_exit(
        self,
        bar:           MarketBar,
        features:      FeatureVector,
        current_value: float,   # per-share current spread value
    ) -> tuple[bool, str]:
        if not self._in_trade:
            return False, ""

        # Time stop: always close by Thursday/Friday square-off
        if not self._in_window(bar.timestamp, self._entry_start, self._square_off):
            # Allow holding through mid-week (Mon entry, Thu exit)
            wd = bar.timestamp.weekday()
            if wd >= 3:   # Thursday or Friday → close
                self._in_trade = False
                return True, "weekly_time_stop"
            # Don't exit on Tuesday/Wednesday even after entry window
            return False, ""

        if self._entry_cost <= 0:
            return False, ""

        # Profit target: spread has grown 150% (2.5× entry cost)
        if current_value > self._entry_cost * (1.0 + self._profit_target):
            self._in_trade = False
            return True, "profit_target"

        # Stop loss: spread lost 70% of value
        if current_value < self._entry_cost * self._stop_loss_pct:
            self._in_trade = False
            return True, "stop_loss"

        return False, ""
