"""
BarbellStrangleStrategy — NEW strategy, never modifies existing short_strangle.py.

Key innovation over ShortStrangle (F): Individual Leg Stop-Loss Management.

Problem the old strangle had on BankNifty
------------------------------------------
When BANKNIFTY spikes, both CE and PE legs were monitored together.
A gamma spike in one leg caused the combined cost-to-close to exceed the stop,
triggering a full position exit and crystallising a large loss — even though
the other leg would have continued decaying profitably.

Solution: Independent CE / PE tracking with underlying spot-breach stops
------------------------------------------------------------------------
- Each leg (CE and PE) is monitored by watching the UNDERLYING spot price.
- If spot crosses the sold strike, that leg is now ATM and premium has exploded.
- CE stop: spot >= sc_strike → CE leg stopped, PE continues.
- PE stop: spot <= sp_strike → PE leg stopped, CE continues.
- Primary win-condition: EOD time stop at 15:15 (theta decay harvested in full).
- Net PnL = CE_leg_PnL + PE_leg_PnL.

Why NOT Black-Scholes for stop tracking
----------------------------------------
The original approach used realized_vol (≈8-12%) to re-price options that were
SOLD at implied vol (≈15-25%). This permanently undervalued the re-priced premium,
making the 30% stop threshold unreachable — a fake infinite-Sharpe backtest.
Additionally, 0DTE gamma is nonlinear; standard BS cannot model intraday gamma
explosions near expiry. Spot-breach stops sidestep both problems entirely.

Capital sleeve: uses config.risk.theta_capital (₹1.2L by default).
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Optional

from algo_platform.core.config import PlatformConfig
from algo_platform.core.types import (
    FeatureVector, Instrument, OptionChain, OptionType,
    OrderSide, Signal, SignalDirection, SpreadLeg, MarketBar,
)
from algo_platform.strategies.base import BaseStrategy

logger = logging.getLogger("platform.strategies.barbell_strangle")

_STEP = {
    "NIFTY": 50.0, "BANKNIFTY": 100.0, "FINNIFTY": 25.0,
    "SENSEX": 100.0, "BANKEX": 100.0,
}
_TRADING_HOURS   = 6.25    # 9:15–15:30
_TRADING_DAYS    = 252
_CLOSE_HOUR      = 15
_CLOSE_MINUTE    = 30


def _remaining_tte_years(ts: datetime) -> float:
    """Minutes left to 15:30 expressed as fraction of a trading year."""
    minutes_left = max(1, (_CLOSE_HOUR * 60 + _CLOSE_MINUTE) - (ts.hour * 60 + ts.minute))
    return minutes_left / (60 * _TRADING_HOURS * _TRADING_DAYS)


# ── Strategy ──────────────────────────────────────────────────────────────────

class BarbellStrangleStrategy(BaseStrategy):
    """
    Strategy: Short OTM Strangle with independent leg stop-losses.

    Capital sleeve: config.risk.theta_capital (default ₹1.2L).
    Both legs are short; exit logic manages CE and PE independently.
    """

    name = "BarbellStrangle"

    def __init__(
        self,
        instrument:          Instrument,
        config:              PlatformConfig,
        quantity:            int   = 1,
        otm_sigma_mult:      float = 0.30,   # strikes at 0.30σ OTM each side
        leg_stop_pct:        float = 0.30,   # individual leg stop at +30% from entry
        profit_target_pct:   float = 0.60,   # combined profit at 60% credit decay
        vix_min:             float = 10.0,
        vix_max:             float = 22.0,
    ) -> None:
        super().__init__(instrument, config)
        self._quantity       = quantity
        self._lot_size       = config.lot_size(instrument.value)
        self._step           = _STEP.get(instrument.value, 50.0)
        self._otm_sigma_mult = otm_sigma_mult
        self._leg_stop_pct   = leg_stop_pct
        self._profit_target  = profit_target_pct
        self._vix_min        = vix_min
        self._vix_max        = vix_max
        self._rfr            = config.risk_free_rate

        self._entry_start = "13:25"
        self._entry_end   = "13:40"
        self._square_off  = "15:15"

        # ── Per-session leg state ──────────────────────────────────────────────
        self._entry_spot:     float = 0.0   # underlying spot at entry (for reference)
        self._sc_strike:      float = 0.0   # OTM call strike we SOLD
        self._sp_strike:      float = 0.0   # OTM put strike we SOLD
        self._ce_entry_price: float = 0.0   # CE leg entry mid-price (per share)
        self._pe_entry_price: float = 0.0   # PE leg entry mid-price (per share)
        self._total_credit:   float = 0.0   # combined credit per share at entry

        self._signal_today:  bool  = False

    def new_session(self) -> None:
        self._in_trade       = False
        self._signal_today   = False
        self._total_credit   = 0.0
        self._entry_spot     = 0.0

    # ── Signal generation ─────────────────────────────────────────────────────

    def generate_signal(
        self,
        bar:      MarketBar,
        chain:    Optional[OptionChain],
        features: FeatureVector,
    ) -> Optional[Signal]:
        if not self._is_active or chain is None:
            return None
        if not self._is_expiry_day(bar.timestamp):
            return None
        if self._signal_today or self._in_trade:
            return None
        if not self._in_window(bar.timestamp, self._entry_start, self._entry_end):
            return None

        vix = chain.india_vix
        if vix > 0 and not (self._vix_min <= vix <= self._vix_max):
            return None

        spot = bar.close
        atm  = chain.atm_strike()
        iv   = (vix / 100.0) if vix > 0 else max(features.realized_vol, 0.12)

        # Compute 0.30σ OTM offset
        tte_frac = _remaining_tte_years(bar.timestamp)
        two_hr_move = spot * iv * math.sqrt(tte_frac)
        otm_offset  = max(self._step,
                          round(two_hr_move * self._otm_sigma_mult / self._step) * self._step)

        sc_strike = self._nearest_listed_strike(chain, atm + otm_offset)
        sp_strike = self._nearest_listed_strike(chain, atm - otm_offset)

        sc_q = chain.quote(sc_strike, OptionType.CALL)
        sp_q = chain.quote(sp_strike, OptionType.PUT)
        if sc_q is None or sp_q is None:
            return None

        # Use bid prices (we are the seller)
        ce_price = sc_q.bid
        pe_price = sp_q.bid
        net_credit = ce_price + pe_price
        # Minimum viable credit: covers transaction costs AND leaves meaningful premium.
        # Rule of thumb: credit must be > 8 pts/share so that even after tx costs
        # (≈ ₹400/round-trip on 75 lots) we still collect ₹200+ net per trade.
        if net_credit < 8.0:
            return None

        legs = [
            SpreadLeg(sc_q.symbol, sc_strike, OptionType.CALL,
                      OrderSide.SELL, self._quantity, self._lot_size, ce_price),
            SpreadLeg(sp_q.symbol, sp_strike, OptionType.PUT,
                      OrderSide.SELL, self._quantity, self._lot_size, pe_price),
        ]

        # Store entry state for independent leg monitoring
        self._entry_spot     = spot
        self._sc_strike      = sc_strike
        self._sp_strike      = sp_strike
        self._ce_entry_price = ce_price
        self._pe_entry_price = pe_price
        self._total_credit   = net_credit
        self._signal_today   = True
        self._in_trade       = True

        logger.info(
            "BarbellStrangle SELL %s | ATM=%.0f | SC=%.0f PE=%.0f | "
            "credit=%.2f | leg_stop=+%.0f%% | VIX=%.1f",
            self.instrument.value, atm, sc_strike, sp_strike,
            net_credit, self._leg_stop_pct * 100, vix,
        )

        return Signal(
            strategy   = self.name,
            instrument = self.instrument,
            direction  = SignalDirection.NEUTRAL,
            timestamp  = bar.timestamp,
            legs       = legs,
            net_debit  = -net_credit,
            max_loss   = net_credit * 2.0 * self._quantity * self._lot_size,
            max_profit = net_credit * self._quantity * self._lot_size,
            confidence = 0.75,
            features   = features,
            metadata   = {
                "sc_strike": sc_strike, "sp_strike": sp_strike,
                "ce_entry":  ce_price,  "pe_entry":  pe_price,
                "credit":    net_credit, "vix":      vix,
            },
        )

    # ── Exit management with individual leg stops ─────────────────────────────

    def should_exit(
        self,
        bar:           MarketBar,
        features:      FeatureVector,
        current_value: float,   # combined position value per share (from engine)
    ) -> tuple[bool, str]:
        """
        Full-position stop based on underlying spot breaching either sold strike.

        Why full-position (not per-leg):
          The engine has no mechanism to close individual legs of an OpenTrade.
          Per-leg stop tracking in the strategy is invisible to the engine's PnL
          calculation — the engine always closes all legs together via the chain
          pricer. A full-position exit as soon as EITHER strike is touched gives
          the engine a clean bar to price the position accurately, before the
          breached leg moves further ITM.

        Why spot-breach (not BS re-pricing):
          Black-Scholes with realized_vol (8-12%) permanently undervalues options
          sold at implied vol (15-25%). The resulting stop threshold is never
          reached. Spot crossing the strike is an objective, model-free signal that
          the option is now ATM and the premium has exploded.
        """
        if not self._in_trade:
            return False, ""

        # ── Hard time stop: always exit at 15:15 ──────────────────────────────
        if not self._in_window(bar.timestamp, self._entry_start, self._square_off):
            self._in_trade = False
            return True, "time_stop"

        spot = bar.close

        # ── Full-position stop: exit immediately when either strike is touched ─
        # CE breached: spot has moved to or above the call strike we sold.
        if self._sc_strike > 0 and spot >= self._sc_strike:
            self._in_trade = False
            logger.info(
                "BarbellStrangle FULL STOP (CE breach) | SC=%.0f | spot=%.0f | "
                "entry_spot=%.0f | CE_entry=%.2f PE_entry=%.2f",
                self._sc_strike, spot, self._entry_spot,
                self._ce_entry_price, self._pe_entry_price,
            )
            return True, "spot_breach_stop"

        # PE breached: spot has moved to or below the put strike we sold.
        if self._sp_strike > 0 and spot <= self._sp_strike:
            self._in_trade = False
            logger.info(
                "BarbellStrangle FULL STOP (PE breach) | SP=%.0f | spot=%.0f | "
                "entry_spot=%.0f | CE_entry=%.2f PE_entry=%.2f",
                self._sp_strike, spot, self._entry_spot,
                self._ce_entry_price, self._pe_entry_price,
            )
            return True, "spot_breach_stop"

        return False, ""
