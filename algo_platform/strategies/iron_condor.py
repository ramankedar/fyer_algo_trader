"""
Strategy E — Weekly Iron Condor (Systematic Premium Selling).

An iron condor = sell OTM call + sell OTM put + buy farther wings.
Collected premium if market stays between the short strikes.
Max loss is capped by the wings — defined-risk theta harvesting.

Entry logic
-----------
Enter Monday morning (or Tuesday if Monday is holiday).
Short strikes: 0.75σ × weekly_IV from ATM (roughly 1σ / 1-week move).
Wing protection: 1.5× the short strike distance.

Why 0.75σ
----------
• At 0.75σ: ~55% of time market stays between both short strikes.
• Win rate ≈ 55% × (wing_spread / short_spread) credit-to-risk ratio.
• With 60-65% win rate and 1:1.5 risk-reward, expectancy is positive.

Exit logic
----------
1. Profit target: 50% of max credit.
2. Stop loss: if EITHER short strike breached with 2+ days left.
3. Time stop: Thursday 3:15 PM.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Optional

import numpy as np

from algo_platform.core.config import PlatformConfig
from algo_platform.core.types import (
    FeatureVector, Instrument, OptionChain, OptionType,
    OrderSide, Signal, SignalDirection, SpreadLeg, MarketBar,
)
from algo_platform.strategies.base import BaseStrategy

logger = logging.getLogger("platform.strategies.iron_condor")


class IronCondorStrategy(BaseStrategy):
    """
    Strategy E: Weekly iron condor — sell OTM call spread + OTM put spread.
    Positive theta, defined max risk, no directional view needed.

    Position structure:
        Sell OTM call  (short call)
        Buy  FAR call  (long call, wing)
        Sell OTM put   (short put)
        Buy  FAR put   (long put, wing)
    """

    name = "WeeklyIronCondor"

    def __init__(self, instrument: Instrument, config: PlatformConfig,
                 quantity: int = 1) -> None:
        super().__init__(instrument, config)
        self._quantity = quantity
        self._lot_size = config.lot_size(instrument.value)
        self._step     = _strike_step(instrument.value)

        # Entry window: Monday / Tuesday early morning
        self._entry_days  = {0, 1}   # Monday=0, Tuesday=1
        self._entry_start = "09:30"
        self._entry_end   = "10:00"
        self._square_off  = "15:15"

        # Risk config
        self._sigma_mult      = 0.75   # short strikes at 0.75σ weekly move
        self._wing_mult       = 1.80   # wings at 1.8× short distance
        self._profit_target   = 0.50   # close at 50% of max credit
        self._stop_loss_mult  = 2.0    # stop if loss > 2× credit received

        # State
        self._credit_received: float = 0.0
        self._short_call_strike: float = 0.0
        self._short_put_strike:  float = 0.0
        self._signal_this_week:  bool  = False

    def new_session(self) -> None:
        if datetime.now().weekday() == 0:   # reset on Monday
            self._signal_this_week = False
            self._credit_received  = 0.0
        self._in_trade = False

    def generate_signal(
        self,
        bar:      MarketBar,
        chain:    Optional[OptionChain],
        features: FeatureVector,
    ) -> Optional[Signal]:
        if not self._is_active or chain is None:
            return None
        if bar.timestamp.weekday() not in self._entry_days:
            return None
        if self._signal_this_week or self._in_trade:
            return None
        if not self._in_window(bar.timestamp, self._entry_start, self._entry_end):
            return None
        if features.realized_vol <= 0:
            return None

        # Weekly move estimate: weekly_σ = daily_σ × sqrt(5 days)
        weekly_iv = features.realized_vol / math.sqrt(252) * math.sqrt(5)
        weekly_move = bar.close * weekly_iv   # expected ±1σ weekly move in pts

        atm = chain.atm_strike()

        # Short strikes: 0.75σ OTM on each side (rounded to step)
        short_call_dist = max(self._step,
                              round(weekly_move * self._sigma_mult / self._step) * self._step)
        short_call = self._nearest_listed_strike(chain, atm + short_call_dist)
        short_put  = self._nearest_listed_strike(chain, atm - short_call_dist)

        # Wing strikes: 1.8× beyond short strikes
        wing_dist  = round(short_call_dist * self._wing_mult / self._step) * self._step
        long_call  = self._nearest_listed_strike(chain, atm + wing_dist)
        long_put   = self._nearest_listed_strike(chain, atm - wing_dist)

        # Validate: need quotes for all 4 strikes
        sc_q = chain.quote(short_call, OptionType.CALL)
        lc_q = chain.quote(long_call,  OptionType.CALL)
        sp_q = chain.quote(short_put,  OptionType.PUT)
        lp_q = chain.quote(long_put,   OptionType.PUT)
        if not all([sc_q, lc_q, sp_q, lp_q]):
            return None

        # Net credit: sell short strikes, buy wings
        credit = (sc_q.bid + sp_q.bid) - (lc_q.ask + lp_q.ask)
        if credit <= 0:
            logger.debug("Iron condor credit ≤ 0 — skip %s", bar.timestamp.date())
            return None

        legs = [
            SpreadLeg(sc_q.symbol, short_call, OptionType.CALL,
                      OrderSide.SELL, self._quantity, self._lot_size, sc_q.bid),
            SpreadLeg(lc_q.symbol, long_call,  OptionType.CALL,
                      OrderSide.BUY,  self._quantity, self._lot_size, lc_q.ask),
            SpreadLeg(sp_q.symbol, short_put,  OptionType.PUT,
                      OrderSide.SELL, self._quantity, self._lot_size, sp_q.bid),
            SpreadLeg(lp_q.symbol, long_put,   OptionType.PUT,
                      OrderSide.BUY,  self._quantity, self._lot_size, lp_q.ask),
        ]

        max_profit = credit * self._quantity * self._lot_size
        wing_spread_pts = long_call - short_call   # same as long_put - short_put
        max_loss   = (wing_spread_pts - credit) * self._quantity * self._lot_size

        self._credit_received     = credit
        self._short_call_strike   = short_call
        self._short_put_strike    = short_put
        self._signal_this_week    = True
        self._in_trade            = True

        logger.info(
            "IronCondor SELL %s | ATM=%.0f | SC=%.0f SP=%.0f | credit=%.2f",
            self.instrument.value, atm, short_call, short_put, credit,
        )

        return Signal(
            strategy   = self.name,
            instrument = self.instrument,
            direction  = SignalDirection.NEUTRAL,
            timestamp  = bar.timestamp,
            legs       = legs,
            net_debit  = -credit,               # negative = credit received
            max_loss   = max_loss,
            max_profit = max_profit,
            confidence = self._confidence(features),
            features   = features,
            metadata   = {
                "short_call": short_call, "short_put": short_put,
                "long_call":  long_call,  "long_put":  long_put,
                "credit":     credit,     "weekly_move_est": weekly_move,
            },
        )

    def should_exit(
        self,
        bar:           MarketBar,
        features:      FeatureVector,
        current_debit: float,   # current cost to close the condor
    ) -> tuple[bool, str]:
        if not self._in_trade:
            return False, ""

        # Time stop
        if not self._in_window(bar.timestamp, self._entry_start, self._square_off):
            self._in_trade = False
            return True, "time_stop"

        credit = self._credit_received
        if credit <= 0:
            return False, ""

        # Profit target: 50% of credit has decayed → lock in profit
        if current_debit < credit * (1.0 - self._profit_target):
            self._in_trade = False
            return True, "profit_target"

        # Stop loss: based on LOSS relative to credit received, NOT absolute cost.
        # Wrong approach (old): stop when cost > 2× credit → fires immediately on any
        # breach because a breached short option has 100 pts value vs 24 pts credit.
        # Correct approach: stop when loss > 2× credit.
        # Loss = (cost_to_close - credit_received). Stop when loss > 2× credit.
        loss = current_debit - credit   # how much MORE we'd pay vs what we received
        if loss > credit * self._stop_loss_mult:
            self._in_trade = False
            return True, "stop_loss"

        return False, ""

    def _confidence(self, f: FeatureVector) -> float:
        # Higher volatility compression = better for condor (market stays range-bound)
        comp = 1.0 - max(f.atr_pct, f.rv_pct)
        return float(0.4 + 0.6 * comp)


def _strike_step(instrument: str) -> float:
    return {"NIFTY": 50.0, "BANKNIFTY": 100.0, "FINNIFTY": 25.0,
            "SENSEX": 100.0, "BANKEX": 100.0}.get(instrument.upper(), 50.0)
