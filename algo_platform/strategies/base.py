"""
Abstract base class and shared helpers for all strategies.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, time
from typing import Optional

from algo_platform.core.types import (
    FeatureVector, Instrument, OptionChain, OptionType, OrderSide,
    Signal, SpreadLeg, MarketBar,
)
from algo_platform.core.config import PlatformConfig

logger = logging.getLogger("platform.strategies")


def _parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


class BaseStrategy(ABC):
    """
    Contract all strategies must satisfy.
    Subclasses override `generate_signal`; base handles time-gating and logging.
    """

    def __init__(self, instrument: Instrument, config: PlatformConfig) -> None:
        self.instrument  = instrument
        self.config      = config
        self._is_active  = True
        self._in_trade   = False
        logger.info("Strategy %s initialised for %s", self.name, instrument.value)

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate_signal(
        self,
        bar:      MarketBar,
        chain:    Optional[OptionChain],
        features: FeatureVector,
    ) -> Optional[Signal]:
        """
        Evaluate current bar and return a Signal (or None if no trade).
        Must never reference future data.
        """

    # ── Shared helpers ─────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self._is_active

    def halt(self, reason: str = "") -> None:
        self._is_active = False
        logger.warning("Strategy %s halted. %s", self.name, reason)

    def resume(self) -> None:
        self._is_active = True
        logger.info("Strategy %s resumed.", self.name)

    def _in_window(self, ts: datetime, start: str, end: str) -> bool:
        t = ts.time()
        return _parse_hhmm(start) <= t <= _parse_hhmm(end)

    def _is_expiry_day(self, ts: datetime) -> bool:
        """True on this instrument's weekly expiry day (reads from config.LOT_SIZES)."""
        from algo_platform.core.config import LOT_SIZES
        spec = LOT_SIZES.get(self.instrument.value)
        wd   = spec.expiry_weekday if spec else 3   # default Thursday
        return ts.weekday() == wd

    # Keep old name as a shim so existing code doesn't silently break during transition
    def _is_thursday(self, ts: datetime) -> bool:
        return self._is_expiry_day(ts)

    def _build_call_spread(
        self,
        chain:     OptionChain,
        atm:       float,
        width:     float,
        quantity:  int,
        lot_size:  int,
    ) -> list[SpreadLeg]:
        """Bull call spread: buy ATM call, sell OTM call."""
        short_strike = self._nearest_listed_strike(chain, atm + width)
        atm_call = chain.quote(atm, OptionType.CALL)
        otm_call = chain.quote(short_strike, OptionType.CALL)
        if atm_call is None or otm_call is None:
            return []
        return [
            SpreadLeg(atm_call.symbol, atm, OptionType.CALL,
                      OrderSide.BUY,  quantity, lot_size, atm_call.ask),
            SpreadLeg(otm_call.symbol, short_strike, OptionType.CALL,
                      OrderSide.SELL, quantity, lot_size, otm_call.bid),
        ]

    def _build_put_spread(
        self,
        chain:     OptionChain,
        atm:       float,
        width:     float,
        quantity:  int,
        lot_size:  int,
    ) -> list[SpreadLeg]:
        """Bear put spread: buy ATM put, sell OTM put."""
        short_strike = self._nearest_listed_strike(chain, atm - width)
        atm_put  = chain.quote(atm, OptionType.PUT)
        otm_put  = chain.quote(short_strike, OptionType.PUT)
        if atm_put is None or otm_put is None:
            return []
        return [
            SpreadLeg(atm_put.symbol,  atm, OptionType.PUT,
                      OrderSide.BUY,  quantity, lot_size, atm_put.ask),
            SpreadLeg(otm_put.symbol,  short_strike, OptionType.PUT,
                      OrderSide.SELL, quantity, lot_size, otm_put.bid),
        ]

    def _build_straddle(
        self,
        chain:    OptionChain,
        atm:      float,
        quantity: int,
        lot_size: int,
    ) -> list[SpreadLeg]:
        """Long straddle: buy ATM call + buy ATM put."""
        call_q = chain.quote(atm, OptionType.CALL)
        put_q  = chain.quote(atm, OptionType.PUT)
        if call_q is None or put_q is None:
            return []
        return [
            SpreadLeg(call_q.symbol, atm, OptionType.CALL,
                      OrderSide.BUY, quantity, lot_size, call_q.ask),
            SpreadLeg(put_q.symbol,  atm, OptionType.PUT,
                      OrderSide.BUY, quantity, lot_size, put_q.ask),
        ]

    def _nearest_listed_strike(self, chain: OptionChain, target: float) -> float:
        strikes = sorted({q.strike for q in chain.quotes})
        if not strikes:
            return target
        return min(strikes, key=lambda s: abs(s - target))

    def _net_debit(self, legs: list[SpreadLeg]) -> float:
        total = 0.0
        for leg in legs:
            sign = 1.0 if leg.side == OrderSide.BUY else -1.0
            total += sign * leg.limit_price
        return total

    def _max_loss(self, legs: list[SpreadLeg]) -> float:
        """Max loss for a debit spread = net premium paid × quantity × lot_size."""
        debit = self._net_debit(legs)
        if not legs:
            return 0.0
        qty = legs[0].quantity * legs[0].lot_size
        return max(0.0, debit * qty)

    def _max_profit(self, legs: list[SpreadLeg], spread_width: float) -> float:
        """Max profit for a debit spread = (width - debit) × qty × lot_size."""
        debit = self._net_debit(legs)
        if not legs:
            return 0.0
        qty = legs[0].quantity * legs[0].lot_size
        return max(0.0, (spread_width - debit) * qty)
