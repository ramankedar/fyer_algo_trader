"""
Strategy G — Short Iron Butterfly (Defined-Risk Straddle).

Structure: Sell ATM straddle + Buy OTM wings.

Why this is better than naked straddle
---------------------------------------
Naked straddle:   unlimited risk if market gaps ↑↓↑ 500pts (rare but catastrophic)
Iron butterfly:   max loss = wing_width - credit_received (KNOWN before entry)

Payoff at expiry:
  Market at ATM      → max profit (collect full net credit)
  Market at ±wing    → max loss = wing_width - net_credit
  Market beyond wing → loss stays CAPPED (wing absorbs it)

The tradeoff vs naked straddle
-------------------------------
  Naked straddle:  credit=84, break-even=±84, max_loss=∞
  Iron butterfly:  credit=54, break-even=±54, max_loss=46pts

  Win rate LOWER for butterfly (smaller break-even zone)
  BUT loss is bounded → better risk management, lower margin requirement

Dynamic wing width (key improvement over standard butterfly)
--------------------------------------------------------------
Wing width is sized to collect a MINIMUM credit ratio:
  min_credit_ratio = 0.5  → net credit ≥ 50% of ATM straddle value
  → wing_width = credit / (1 - min_ratio)
  → ensures we're not selling wings too cheaply

VIX-based entry filter (learning from Strategy F testing)
----------------------------------------------------------
  If VIX < 10: skip (premium too thin relative to actual moves)
  If VIX > 22: skip (market too volatile, moves exceed any break-even)
  Sweet spot: VIX 11-20 → consistent vol premium exists
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from algo_platform.core.config import PlatformConfig
from algo_platform.core.types import (
    FeatureVector, Instrument, OptionChain, OptionType,
    OrderSide, Signal, SignalDirection, SpreadLeg, MarketBar,
)
from algo_platform.strategies.base import BaseStrategy

logger = logging.getLogger("platform.strategies.iron_butterfly")

_STEP = {"NIFTY": 50.0, "BANKNIFTY": 100.0, "FINNIFTY": 25.0,
         "SENSEX": 100.0, "BANKEX": 100.0}


class IronButterflyStrategy(BaseStrategy):
    """
    Strategy G: Short iron butterfly on expiry afternoon.
    Defined-risk version of the ATM straddle — capped max loss,
    lower premium, same directional neutrality.
    """

    name = "IronButterfly"

    def __init__(self, instrument: Instrument, config: PlatformConfig,
                 quantity: int = 1,
                 wing_width_mult: float = 1.5,   # wing = 1.5× straddle break-even
                 min_credit_pct: float = 0.55,   # net credit ≥ 55% of wing spread
                 vix_min: float = 11.0,
                 vix_max: float = 20.0,
                 profit_target: float = 0.65,    # close at 65% decay
                 stop_loss_mult: float = 1.5,    # loss-based stop
                 ) -> None:
        super().__init__(instrument, config)
        self._quantity      = quantity
        self._lot_size      = config.lot_size(instrument.value)
        self._step          = _STEP.get(instrument.value, 50.0)
        self._wing_mult     = wing_width_mult
        self._min_credit    = min_credit_pct
        self._vix_min       = vix_min
        self._vix_max       = vix_max
        self._profit_target = profit_target
        self._stop_mult     = stop_loss_mult

        self._entry_start = "13:25"
        self._entry_end   = "13:40"
        self._square_off  = "15:15"

        self._credit:     float = 0.0
        self._wing_width: float = 0.0
        self._signal_today = False

    def new_session(self) -> None:
        self._in_trade = False
        self._signal_today = False
        self._credit = 0.0

    def generate_signal(
        self,
        bar:      MarketBar,
        chain:    Optional[OptionChain],
        features: FeatureVector,
    ) -> Optional[Signal]:
        if not self._is_active or chain is None:
            return None
        if not self._is_thursday(bar.timestamp):
            return None
        if self._signal_today or self._in_trade:
            return None
        if not self._in_window(bar.timestamp, self._entry_start, self._entry_end):
            return None

        vix = chain.india_vix
        if vix > 0 and not (self._vix_min <= vix <= self._vix_max):
            return None

        atm   = chain.atm_strike()
        step  = self._step
        iv    = (vix / 100.0) if vix > 0 else max(features.realized_vol, 0.12)

        # Compute break-even (= approx ATM straddle premium)
        tte_frac = 2.0 / (6.25 * 252)
        straddle_value = bar.close * iv * math.sqrt(tte_frac) * math.sqrt(2.0 / math.pi)

        # Wing width: 1.5× break-even, rounded to strike step
        wing_pts = max(step, round(straddle_value * self._wing_mult / step) * step)
        long_call_strike = self._nearest_listed_strike(chain, atm + wing_pts)
        long_put_strike  = self._nearest_listed_strike(chain, atm - wing_pts)
        actual_wing      = long_call_strike - atm   # actual width in pts

        # Look up all four quotes
        sc = chain.quote(atm,              OptionType.CALL)  # sell ATM call
        sp = chain.quote(atm,              OptionType.PUT)   # sell ATM put
        lc = chain.quote(long_call_strike, OptionType.CALL)  # buy OTM call (wing)
        lp = chain.quote(long_put_strike,  OptionType.PUT)   # buy OTM put (wing)
        if not all([sc, sp, lc, lp]):
            return None

        net_credit = (sc.bid + sp.bid) - (lc.ask + lp.ask)
        # Ensure minimum credit ratio: credit ≥ min_credit_pct × wing_width
        if net_credit < actual_wing * self._min_credit or net_credit <= 0:
            logger.debug("IronButterfly: credit %.1f < min %.1f — skip",
                        net_credit, actual_wing * self._min_credit)
            return None

        max_loss_pts   = actual_wing - net_credit   # defined max loss per share
        max_profit_pts = net_credit

        legs = [
            SpreadLeg(sc.symbol, atm,              OptionType.CALL,
                      OrderSide.SELL, self._quantity, self._lot_size, sc.bid),
            SpreadLeg(sp.symbol, atm,              OptionType.PUT,
                      OrderSide.SELL, self._quantity, self._lot_size, sp.bid),
            SpreadLeg(lc.symbol, long_call_strike, OptionType.CALL,
                      OrderSide.BUY,  self._quantity, self._lot_size, lc.ask),
            SpreadLeg(lp.symbol, long_put_strike,  OptionType.PUT,
                      OrderSide.BUY,  self._quantity, self._lot_size, lp.ask),
        ]

        self._credit       = net_credit
        self._wing_width   = actual_wing
        self._signal_today = True
        self._in_trade     = True

        logger.info(
            "IronButterfly SELL %s | ATM=%.0f ±%.0f wing | credit=%.1f | "
            "max_loss=%.0f pts | VIX=%.1f",
            self.instrument.value, atm, actual_wing, net_credit, max_loss_pts, vix,
        )

        return Signal(
            strategy   = self.name,
            instrument = self.instrument,
            direction  = SignalDirection.NEUTRAL,
            timestamp  = bar.timestamp,
            legs       = legs,
            net_debit  = -net_credit,
            max_loss   = max_loss_pts * self._quantity * self._lot_size,
            max_profit = max_profit_pts * self._quantity * self._lot_size,
            confidence = self._confidence(features, vix),
            features   = features,
            metadata   = {"atm": atm, "wing": actual_wing, "credit": net_credit,
                          "max_loss_pts": max_loss_pts, "vix": vix},
        )

    def should_exit(
        self,
        bar:           MarketBar,
        features:      FeatureVector,
        current_value: float,   # current per-share cost to close (abs value)
    ) -> tuple[bool, str]:
        if not self._in_trade:
            return False, ""

        if not self._in_window(bar.timestamp, self._entry_start, self._square_off):
            self._in_trade = False
            return True, "time_stop"

        credit = self._credit
        if credit <= 0:
            return False, ""

        # Profit target: 65% of credit has decayed
        if current_value < credit * (1.0 - self._profit_target):
            self._in_trade = False
            return True, "profit_target"

        # Loss stop: loss > 1.5× credit (still controlled since wing caps absolute loss)
        loss = current_value - credit
        if loss > credit * self._stop_mult:
            self._in_trade = False
            return True, "stop_loss"

        return False, ""

    def _confidence(self, f: FeatureVector, vix: float) -> float:
        vix_score = 1.0 - abs(vix - 15.0) / 7.0 if vix > 0 else 0.5
        return float(min(1.0, 0.4 + 0.6 * vix_score))
