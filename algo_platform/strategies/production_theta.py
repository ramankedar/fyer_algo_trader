"""
ProductionThetaStrategy — EXPERIMENTAL, not production-ready.

Research basis: analyze_whipsaws.py forensic study on 2023-2026 NIFTY data.

Key findings that drove this design:
  - 73% of BarbellStrangle (Variant A) stops were whipsaws that recovered by EOD
  - 58% of profitable (Variant C) trades briefly crossed a sold strike intraday
  - The exact-strike stop fires at the point of maximum gamma noise (ATM)
  - Variant E (morning-range buffer) filters 82% of whipsaws at near-zero alpha cost
  - Catastrophic stop (4× entry premium) reduces worst loss 71% with ₹4/trade cost

Architecture (CatE):
  1. Sell OTM strangle at entry window open on expiry day
  2. Compute buffer from pre-entry morning range
  3. Independent legs: buffer or catastrophic stop closes ONE leg, hold the other
  4. EOD time stop closes remaining open leg(s)

All thresholds are constructor parameters with research-validated defaults.
The defaults must not be treated as optimised — they reflect one instrument
(NIFTY) over one period (2023-2026). Validate before changing any default.

DO NOT use in live trading without further out-of-sample validation.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Optional

from scipy.stats import norm

from algo_platform.core.config import PlatformConfig
from algo_platform.core.types import (
    FeatureVector, Instrument, OptionChain, OptionType,
    OrderSide, Signal, SignalDirection, SpreadLeg, MarketBar,
)
from algo_platform.strategies.base import BaseStrategy

logger = logging.getLogger("platform.strategies.production_theta")

_STEP: dict[str, float] = {
    "NIFTY": 50.0, "BANKNIFTY": 100.0, "FINNIFTY": 25.0,
    "SENSEX": 100.0, "BANKEX": 100.0,
}
_TRADING_HOURS = 6.25   # 9:15–15:30
_TRADING_DAYS  = 252
_CLOSE_HOUR    = 15
_CLOSE_MINUTE  = 30


# ── Module-level BS pricer (used only for catastrophic stop estimation) ───────

def _bs(S: float, K: float, T: float, sigma: float,
        r: float, is_call: bool) -> float:
    if T <= 1e-7 or sigma <= 0:
        return max(0.0, S - K) if is_call else max(0.0, K - S)
    sq = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    if is_call:
        return max(0.0, S * float(norm.cdf(d1)) - K * math.exp(-r * T) * float(norm.cdf(d2)))
    return max(0.0, K * math.exp(-r * T) * float(norm.cdf(-d2)) - S * float(norm.cdf(-d1)))


def _remaining_tte(ts: datetime) -> float:
    """Trading minutes remaining to 15:30, expressed as fraction of a trading year."""
    mins = max(1, (_CLOSE_HOUR * 60 + _CLOSE_MINUTE) - (ts.hour * 60 + ts.minute))
    return mins / (60 * _TRADING_HOURS * _TRADING_DAYS)


# ── Strategy ──────────────────────────────────────────────────────────────────

class ProductionThetaStrategy(BaseStrategy):
    """
    EXPERIMENTAL 0DTE short OTM strangle — research framework only.

    Parameters
    ----------
    entry_time : str
        IST time (HH:MM) at which the entry window opens. Default: "13:30".
        Research value: first bar on or after 13:30 is used as entry.
    entry_end_time : str
        Entry window closes. Default: "13:40".
    square_off_time : str
        Hard EOD exit for any remaining open leg(s). Default: "15:15".
    otm_sigma_mult : float
        OTM offset multiplier. Offset = spot × iv × sqrt(tte) × otm_sigma_mult.
        Research value: 0.30. Do not change without re-running attribution study.
    min_credit : float
        Minimum combined credit (per share) required to enter. Default: 8.0.
    vix_min, vix_max : float
        VIX filter bounds. Trades outside this range are skipped. Default: 10–22.
    morning_range_end : str
        All bars from 09:15 to this time contribute to the morning H/L range.
        Research: bars with hour < 13 → end = "12:59". Default: "12:59".
    buffer_mult : float
        Buffer = morning_range × buffer_mult. Minimum buffer = buffer_min_steps × step.
        Research value: 0.50 (half the morning range). Do not optimise.
    buffer_min_steps : int
        Minimum buffer expressed in strike steps. Default: 1 (one step = 50 pts NIFTY).
    buffer_iv_mult : float
        IV expansion factor applied when pricing a buffer-stopped leg at exit.
        Research used 1.30 (30% IV expansion). Default: 1.30.
    cat_premium_mult : float
        Catastrophic stop fires when BS-estimated option value ≥ entry × this.
        Research value: 4.0 (300% above entry = 4× entry). Do not optimise.
    cat_iv_mult : float
        IV expansion assumed when computing the catastrophic threshold.
        Research used 1.50. Higher = more conservative trigger. Default: 1.50.
    """

    name = "ProductionTheta"

    def __init__(
        self,
        instrument:          Instrument,
        config:              PlatformConfig,
        quantity:            int   = 1,
        # Entry / exit windows
        entry_time:          str   = "13:30",
        entry_end_time:      str   = "13:40",
        square_off_time:     str   = "15:15",
        # Strike selection
        otm_sigma_mult:      float = 0.30,
        min_credit:          float = 8.0,
        # VIX filter
        vix_min:             float = 10.0,
        vix_max:             float = 22.0,
        # Morning range window
        morning_range_end:   str   = "12:59",
        # Buffer stop — all from research, do not optimise individually
        buffer_mult:         float = 0.50,
        buffer_min_steps:    int   = 1,
        buffer_iv_mult:      float = 1.30,
        # Catastrophic stop
        cat_premium_mult:    float = 4.0,
        cat_iv_mult:         float = 1.50,
    ) -> None:
        super().__init__(instrument, config)

        self._quantity         = quantity
        self._lot_size         = config.lot_size(instrument.value)
        self._step             = _STEP.get(instrument.value, 50.0)
        self._rfr              = config.risk_free_rate

        # Time windows
        self._entry_time       = entry_time
        self._entry_end_time   = entry_end_time
        self._square_off       = square_off_time

        # Strike / credit
        self._otm_sigma_mult   = otm_sigma_mult
        self._min_credit       = min_credit

        # VIX
        self._vix_min          = vix_min
        self._vix_max          = vix_max

        # Morning range
        self._morning_range_end = morning_range_end

        # Buffer stop
        self._buffer_mult      = buffer_mult
        self._buffer_min_steps = buffer_min_steps
        self._buffer_iv_mult   = buffer_iv_mult

        # Catastrophic stop
        self._cat_premium_mult = cat_premium_mult
        self._cat_iv_mult      = cat_iv_mult

        # ── Per-session state ──────────────────────────────────────────────────
        self._signal_today:    bool  = False

        # Entry details (set in generate_signal)
        self._entry_spot:      float = 0.0
        self._entry_iv:        float = 0.0
        self._sc_strike:       float = 0.0   # sold call strike
        self._sp_strike:       float = 0.0   # sold put strike
        self._ce_entry_price:  float = 0.0
        self._pe_entry_price:  float = 0.0
        self._morning_range:   float = 0.0
        self._buffer:          float = 0.0   # effective buffer for this session

        # Intraday H/L accumulator (reset each session)
        self._day_high:        float = -float("inf")
        self._day_low:         float =  float("inf")

        # Independent leg stop state
        self._ce_stopped:      bool          = False
        self._pe_stopped:      bool          = False
        self._ce_stop_price:   Optional[float] = None   # option value when CE was stopped
        self._pe_stop_price:   Optional[float] = None   # option value when PE was stopped

    def new_session(self) -> None:
        self._in_trade         = False
        self._signal_today     = False
        self._entry_spot       = 0.0
        self._entry_iv         = 0.0
        self._sc_strike        = 0.0
        self._sp_strike        = 0.0
        self._ce_entry_price   = 0.0
        self._pe_entry_price   = 0.0
        self._morning_range    = 0.0
        self._buffer           = 0.0
        self._day_high         = -float("inf")
        self._day_low          =  float("inf")
        self._ce_stopped       = False
        self._pe_stopped       = False
        self._ce_stop_price    = None
        self._pe_stop_price    = None

    # ── Signal generation ──────────────────────────────────────────────────────

    def generate_signal(
        self,
        bar:      MarketBar,
        chain:    Optional[OptionChain],
        features: FeatureVector,
    ) -> Optional[Signal]:

        if not self._is_active:
            return None
        if not self._is_expiry_day(bar.timestamp):
            return None
        if self._signal_today or self._in_trade:
            return None

        # ── Accumulate morning range before entry window opens ─────────────────
        # Track H/L regardless of chain availability so the range is complete
        # even on bars where the chain builder hasn't fired yet.
        if self._in_window(bar.timestamp, "09:15", self._morning_range_end):
            self._day_high = max(self._day_high, bar.high)
            self._day_low  = min(self._day_low,  bar.low)

        # Entry window check (chain required from here)
        if not self._in_window(bar.timestamp, self._entry_time, self._entry_end_time):
            return None
        if chain is None:
            return None

        # VIX filter
        vix = chain.india_vix
        if vix > 0 and not (self._vix_min <= vix <= self._vix_max):
            return None

        spot = bar.close
        atm  = chain.atm_strike()
        iv   = (vix / 100.0) if vix > 0 else max(features.realized_vol, 0.12)

        # OTM offset: same formula as BarbellStrangle
        tte_frac    = _remaining_tte(bar.timestamp)
        two_hr_move = spot * iv * math.sqrt(tte_frac)
        otm_offset  = max(
            self._step,
            round(two_hr_move * self._otm_sigma_mult / self._step) * self._step,
        )

        sc_strike = self._nearest_listed_strike(chain, atm + otm_offset)
        sp_strike = self._nearest_listed_strike(chain, atm - otm_offset)

        sc_q = chain.quote(sc_strike, OptionType.CALL)
        sp_q = chain.quote(sp_strike, OptionType.PUT)
        if sc_q is None or sp_q is None:
            return None

        ce_price   = sc_q.bid
        pe_price   = sp_q.bid
        net_credit = ce_price + pe_price
        if net_credit < self._min_credit:
            return None

        # Buffer for this session
        morning_range = (
            max(0.0, self._day_high - self._day_low)
            if math.isfinite(self._day_high) and math.isfinite(self._day_low)
            else 0.0
        )
        buffer = max(
            self._buffer_min_steps * self._step,
            morning_range * self._buffer_mult,
        )

        legs = [
            SpreadLeg(sc_q.symbol, sc_strike, OptionType.CALL,
                      OrderSide.SELL, self._quantity, self._lot_size, ce_price),
            SpreadLeg(sp_q.symbol, sp_strike, OptionType.PUT,
                      OrderSide.SELL, self._quantity, self._lot_size, pe_price),
        ]

        # Persist all entry state needed by should_exit and compute_exit_value
        self._entry_spot      = spot
        self._entry_iv        = iv
        self._sc_strike       = sc_strike
        self._sp_strike       = sp_strike
        self._ce_entry_price  = ce_price
        self._pe_entry_price  = pe_price
        self._morning_range   = morning_range
        self._buffer          = buffer
        self._signal_today    = True
        self._in_trade        = True

        logger.info(
            "ProductionTheta SELL %s | ATM=%.0f | SC=%.0f SP=%.0f | "
            "credit=%.2f | buffer=%.0f (morning_range=%.0f) | VIX=%.1f",
            self.instrument.value, atm, sc_strike, sp_strike,
            net_credit, buffer, morning_range, vix,
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
            confidence = 0.70,   # deliberately conservative; not production-validated
            features   = features,
            metadata   = {
                "sc_strike":     sc_strike,
                "sp_strike":     sp_strike,
                "ce_entry":      ce_price,
                "pe_entry":      pe_price,
                "credit":        net_credit,
                "vix":           vix,
                "morning_range": morning_range,
                "buffer":        buffer,
                "otm_offset":    otm_offset,
            },
        )

    # ── Exit management ────────────────────────────────────────────────────────

    def should_exit(
        self,
        bar:           MarketBar,
        features:      FeatureVector,
        current_value: float,
    ) -> tuple[bool, str]:
        """
        Independent leg management with buffer and catastrophic stops.

        Stop priority (checked in order):
          1. EOD time stop — hard wall, always fires
          2. Catastrophic stop — BS-estimated option value ≥ entry × cat_premium_mult
          3. Buffer stop — spot beyond sold_strike ± buffer (spot-based, no BS)

        When one leg triggers but the other is still open:
          - Records the stop price for the breached leg
          - Returns (False, "") so the surviving leg continues accumulating theta
          - The engine keeps the trade open; compute_exit_value will blend the two
            leg values correctly when the trade is eventually closed

        Returns (True, reason) only when:
          - Both legs have been independently stopped
          - The EOD time stop fires
          - A catastrophic stop fires (either leg, immediately closes full position)

        Note for live trading:
          _ce_stopped / _pe_stopped flags and _ce_stop_price / _pe_stop_price
          must be read by the execution layer to place actual partial-close orders.
          The backtest engine approximates this via compute_exit_value.
        """
        if not self._in_trade:
            return False, ""

        # ── 1. EOD time stop ──────────────────────────────────────────────────
        if not self._in_window(bar.timestamp, self._entry_time, self._square_off):
            self._in_trade = False
            return True, "time_stop"

        spot = bar.close
        tte  = _remaining_tte(bar.timestamp)
        cat_iv = self._entry_iv * self._cat_iv_mult

        # ── 2. Catastrophic stop (BS-based) ───────────────────────────────────
        # Close the ENTIRE position immediately if either leg explodes past
        # the catastrophic threshold. This is a full-position exit, not per-leg,
        # because a large move invalidates the entire strangle rationale.
        if not self._ce_stopped and self._ce_entry_price > 0:
            ce_est = _bs(spot, self._sc_strike, tte, cat_iv, self._rfr, True)
            if ce_est >= self._ce_entry_price * self._cat_premium_mult:
                # Estimate exit price for the surviving PE leg at this moment too
                pe_est = _bs(spot, self._sp_strike, tte, self._entry_iv, self._rfr, False)
                self._ce_stop_price = ce_est
                self._pe_stop_price = pe_est   # close PE simultaneously
                self._ce_stopped    = True
                self._pe_stopped    = True
                self._in_trade      = False
                logger.warning(
                    "ProductionTheta CATASTROPHIC STOP (CE) | SC=%.0f | spot=%.0f | "
                    "ce_est=%.2f (%.1f× entry=%.2f) | cat_mult=%.1f",
                    self._sc_strike, spot, ce_est,
                    ce_est / max(self._ce_entry_price, 0.01),
                    self._ce_entry_price, self._cat_premium_mult,
                )
                return True, "cat_stop_ce"

        if not self._pe_stopped and self._pe_entry_price > 0:
            pe_est = _bs(spot, self._sp_strike, tte, cat_iv, self._rfr, False)
            if pe_est >= self._pe_entry_price * self._cat_premium_mult:
                ce_est = _bs(spot, self._sc_strike, tte, self._entry_iv, self._rfr, True)
                self._pe_stop_price = pe_est
                self._ce_stop_price = ce_est
                self._ce_stopped    = True
                self._pe_stopped    = True
                self._in_trade      = False
                logger.warning(
                    "ProductionTheta CATASTROPHIC STOP (PE) | SP=%.0f | spot=%.0f | "
                    "pe_est=%.2f (%.1f× entry=%.2f) | cat_mult=%.1f",
                    self._sp_strike, spot, pe_est,
                    pe_est / max(self._pe_entry_price, 0.01),
                    self._pe_entry_price, self._cat_premium_mult,
                )
                return True, "cat_stop_pe"

        # ── 3. Buffer stop (spot-based, per-leg) ──────────────────────────────
        # Closes only the breached leg; the surviving leg continues.
        # The IV expansion (buffer_iv_mult) is applied when recording the stop price
        # to approximate the real cost of closing at that moment.
        if not self._ce_stopped and self._sc_strike > 0:
            if spot >= self._sc_strike + self._buffer:
                ce_exit = _bs(spot, self._sc_strike, tte,
                              self._entry_iv * self._buffer_iv_mult, self._rfr, True)
                self._ce_stop_price = ce_exit
                self._ce_stopped    = True
                logger.info(
                    "ProductionTheta CE BUFFER STOP | SC=%.0f | spot=%.0f | "
                    "trigger=%.0f | ce_exit_est=%.2f | PE still open",
                    self._sc_strike, spot,
                    self._sc_strike + self._buffer, ce_exit,
                )

        if not self._pe_stopped and self._sp_strike > 0:
            if spot <= self._sp_strike - self._buffer:
                pe_exit = _bs(spot, self._sp_strike, tte,
                              self._entry_iv * self._buffer_iv_mult, self._rfr, False)
                self._pe_stop_price = pe_exit
                self._pe_stopped    = True
                logger.info(
                    "ProductionTheta PE BUFFER STOP | SP=%.0f | spot=%.0f | "
                    "trigger=%.0f | pe_exit_est=%.2f | CE still open",
                    self._sp_strike, spot,
                    self._sp_strike - self._buffer, pe_exit,
                )

        # Both legs independently stopped → exit full position
        if self._ce_stopped and self._pe_stopped:
            self._in_trade = False
            return True, "both_legs_stopped"

        # One or zero legs stopped — keep position open for surviving leg
        return False, ""

    # ── Engine integration hook ────────────────────────────────────────────────

    def compute_exit_value(
        self,
        ot:    Any,          # OpenTrade (typed as Any to avoid circular import)
        bar:   MarketBar,
        chain: Optional[OptionChain],
    ) -> float:
        """
        Returns net per-unit exit value for the backtest engine's _close_trade.

        For legs that were independently stopped: uses the recorded stop price
        (the BS-estimated option value at the moment the stop triggered).
        For legs still alive: uses the current chain mid-price or BS fallback.

        The engine uses this value as: exit_proceeds = value × lots × lot_size.
        Sign convention matches _spread_exit_value: SELL legs contribute negative
        values (cost to buy back), BUY legs contribute positive values (proceeds).

        Without this method the engine would use the chain price for ALL legs
        at the final exit time, incorrectly treating independently-stopped CE/PE
        as if they were held to EOD — which would understate losses on stopped legs
        that did not recover, and overstate losses on those that did.
        """
        spot    = bar.close
        elapsed = max(0.0, (bar.timestamp - ot.entry_time).total_seconds() / 3600)
        decay   = max(0.05, 1.0 - min(1.0, elapsed / _TRADING_HOURS) ** 0.5)

        net = 0.0
        for leg in ot.signal.legs:
            sign    = 1.0 if leg.side == OrderSide.BUY else -1.0
            is_call = (leg.option_type == OptionType.CALL)

            # Independently stopped leg — use the recorded price, not the chain
            if is_call and self._ce_stopped and self._ce_stop_price is not None:
                net += sign * self._ce_stop_price
                continue
            if not is_call and self._pe_stopped and self._pe_stop_price is not None:
                net += sign * self._pe_stop_price
                continue

            # Active leg — prefer chain mid-price
            if chain is not None:
                q = chain.quote(leg.strike, leg.option_type)
                if q is not None:
                    net += sign * q.mid
                    continue

            # Fallback: intrinsic + decayed time value (same as engine's per-leg fallback)
            intrinsic = (max(0.0, spot - leg.strike) if is_call
                         else max(0.0, leg.strike - spot))
            time_val  = leg.limit_price * decay
            net      += sign * (intrinsic + time_val)

        return float(net)
