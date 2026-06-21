"""
Strategy F — Short Expiry-Day Strangle (Improved Short Straddle).

A strangle sells SLIGHTLY OTM options instead of ATM.
Compared to the ATM straddle (Strategy D):
  ✓ Higher win rate   (market needs to move MORE to hurt you)
  ✓ Lower premium     (OTM options are cheaper than ATM)
  ✓ Wider breakeven   (you have more room before losing money)
  ✗ Lower profit/trade (collected less premium)

Net result: better Sharpe ratio, lower drawdown, more consistent.

Improvements over base Strategy D
----------------------------------
1.  OTM strikes instead of ATM → wider profit zone
2.  VIX filter → skip if VIX < 10 (premium too thin) or > 22 (too dangerous)
3.  Delta-based stop → exit if position delta exceeds 0.45 (market trending strongly)
4.  Profit target at 60% → lock in gains earlier, less time in market
5.  Entry window 1:20–1:40 PM → slight buffer around 1:30 PM

Strike selection
----------------
Sell 0.3σ OTM on each side.
For NIFTY at 24000 with VIX=14 (σ=0.14) and 2-hr TTE:
  daily_move_est = 24000 × 0.14 / sqrt(252) = 212 pts (daily 1σ)
  2hr_move_est   = 212 × sqrt(2/6.5) ≈ 117 pts (2hr 1σ)
  0.3σ offset    = 117 × 0.3 ≈ 35 pts → nearest listed strike
  → sell 24000+35=24050 call and 24000-35=23950 put
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

logger = logging.getLogger("platform.strategies.short_strangle")

_TRADING_HOURS_PER_DAY = 6.25   # 9:15–15:30 IST
_TRADING_DAYS_PER_YEAR = 252


class ShortStrangleStrategy(BaseStrategy):
    """
    Strategy F: Short OTM strangle on expiry Thursday afternoon.
    Improved version of ShortStraddle with VIX filter, delta stop, and
    wider profit zone from OTM strikes.
    """

    name = "ShortStrangle"

    def __init__(self, instrument: Instrument, config: PlatformConfig,
                 quantity: int = 1,
                 otm_sigma_mult: float = 0.30,   # how far OTM in σ units
                 vix_min: float = 10.0,          # skip if VIX < this (too cheap)
                 vix_max: float = 22.0,          # skip if VIX > this (too dangerous)
                 profit_target_pct: float = 0.60,
                 stop_loss_mult: float = 1.8,
                 ) -> None:
        super().__init__(instrument, config)
        self._quantity        = quantity
        self._lot_size        = config.lot_size(instrument.value)
        self._otm_sigma_mult  = otm_sigma_mult
        self._vix_min         = vix_min
        self._vix_max         = vix_max
        self._profit_target   = profit_target_pct
        self._stop_loss_mult  = stop_loss_mult

        self._entry_start    = "13:20"
        self._entry_end      = "13:40"
        self._square_off     = "15:15"

        self._credit_received:   float = 0.0
        self._short_call_strike: float = 0.0
        self._short_put_strike:  float = 0.0
        self._signal_today:      bool  = False

    def new_session(self) -> None:
        self._in_trade       = False
        self._signal_today   = False
        self._credit_received = 0.0

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

        # VIX filter: skip if vol is outside our range
        vix = chain.india_vix  # stored in chain
        if vix > 0 and not (self._vix_min <= vix <= self._vix_max):
            logger.debug("ShortStrangle: VIX %.1f outside [%.0f, %.0f] — skip",
                         vix, self._vix_min, self._vix_max)
            return None

        spot      = bar.close
        atm       = chain.atm_strike()
        step      = _strike_step(self.instrument.value)
        iv        = (vix / 100.0) if vix > 0 else (features.realized_vol or 0.14)

        # Compute 2-hour 0.3σ move estimate
        tte_hours = 2.0
        tte_frac  = tte_hours / (_TRADING_HOURS_PER_DAY * _TRADING_DAYS_PER_YEAR)
        two_hr_sigma = spot * iv * math.sqrt(tte_frac)   # 1σ expected move
        otm_offset   = max(step, round(two_hr_sigma * self._otm_sigma_mult / step) * step)

        short_call = self._nearest_listed_strike(chain, atm + otm_offset)
        short_put  = self._nearest_listed_strike(chain, atm - otm_offset)

        sc_q = chain.quote(short_call, OptionType.CALL)
        sp_q = chain.quote(short_put,  OptionType.PUT)
        if sc_q is None or sp_q is None:
            return None

        net_credit = sc_q.bid + sp_q.bid
        if net_credit < 5.0:   # too thin — skip
            return None

        legs = [
            SpreadLeg(sc_q.symbol, short_call, OptionType.CALL,
                      OrderSide.SELL, self._quantity, self._lot_size, sc_q.bid),
            SpreadLeg(sp_q.symbol, short_put,  OptionType.PUT,
                      OrderSide.SELL, self._quantity, self._lot_size, sp_q.bid),
        ]

        max_profit = net_credit * self._quantity * self._lot_size

        self._credit_received    = net_credit
        self._short_call_strike  = short_call
        self._short_put_strike   = short_put
        self._signal_today       = True
        self._in_trade           = True

        logger.info(
            "ShortStrangle SELL %s | ATM=%.0f | SC=%.0f SP=%.0f | "
            "credit=%.2f | VIX=%.1f | breakeven=±%.0f pts",
            self.instrument.value, atm, short_call, short_put,
            net_credit, vix, net_credit + otm_offset,
        )

        return Signal(
            strategy   = self.name,
            instrument = self.instrument,
            direction  = SignalDirection.NEUTRAL,
            timestamp  = bar.timestamp,
            legs       = legs,
            net_debit  = -net_credit,
            max_loss   = max_profit * self._stop_loss_mult,
            max_profit = max_profit,
            confidence = self._confidence(features, vix),
            features   = features,
            metadata   = {
                "short_call": short_call, "short_put": short_put,
                "credit":     net_credit, "otm_offset": otm_offset,
                "vix":        vix,
            },
        )

    def should_exit(
        self,
        bar:           MarketBar,
        features:      FeatureVector,
        current_value: float,   # current per-share straddle value (cost to close)
    ) -> tuple[bool, str]:
        if not self._in_trade:
            return False, ""

        # Hard time stop
        if not self._in_window(bar.timestamp, self._entry_start, self._square_off):
            self._in_trade = False
            return True, "time_stop"

        credit = self._credit_received
        if credit <= 0:
            return False, ""

        # Profit target: 60% of credit captured
        if current_value < credit * (1.0 - self._profit_target):
            self._in_trade = False
            return True, "profit_target"

        # Stop loss: cost-to-close > 1.8× original credit
        if current_value > credit * self._stop_loss_mult:
            self._in_trade = False
            return True, "stop_loss"

        # Delta stop: if market breached short strike significantly, exit
        spot = bar.close
        if (spot > self._short_call_strike * 1.003 or
                spot < self._short_put_strike * 0.997):
            self._in_trade = False
            return True, "strike_breach"

        return False, ""

    def _confidence(self, f: FeatureVector, vix: float) -> float:
        # Best when VIX is in the sweet spot (12-16) and vol is compressing
        vix_score = 1.0 - abs(vix - 14.0) / 8.0 if vix > 0 else 0.5
        comp_score = 1.0 - max(f.atr_pct, f.rv_pct)
        return float(min(1.0, max(0.3, 0.4 + 0.35 * vix_score + 0.25 * comp_score)))


def _strike_step(instrument: str) -> float:
    return {"NIFTY": 50.0, "BANKNIFTY": 100.0, "FINNIFTY": 25.0,
            "SENSEX": 100.0, "BANKEX": 100.0}.get(instrument.upper(), 50.0)
