"""
Synthetic option chain builder — vectorised implementation.

Uses a single numpy+scipy batch call for all strikes on each chain build,
giving 30-50× speedup vs scalar per-option Black-Scholes loops.

Skew model (Indian equity options)
------------------------------------
  put_iv(K)  = atm_iv * (1 + 0.15 * max(0, (S-K)/S))   OTM puts get more IV
  call_iv(K) = atm_iv * (1 - 0.08 * max(0, (K-S)/S))   OTM calls slightly less

Weekly expiry calendar (as of 2023-2025)
  NIFTY: Thursday | BANKNIFTY: Wednesday | FINNIFTY: Tuesday
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Dict, List
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import norm

from algo_platform.core.types import (
    Instrument, OptionChain, OptionQuote, OptionType,
)

IST = ZoneInfo("Asia/Kolkata")

STRIKE_STEP: Dict[str, float] = {
    "NIFTY":     50.0,
    "BANKNIFTY": 100.0,
    "FINNIFTY":  25.0,
    "SENSEX":    100.0,
    "BANKEX":    100.0,   # BSE BANKEX ~63000; 100pt steps
    "BSEIT":     200.0,   # BSE IT ~36000; 200pt steps
}
N_STRIKES_OTM: Dict[str, int] = {
    "NIFTY":     12,   # 12 × 50pt  = ±600  = ±2.7% of ~22000
    "BANKNIFTY": 12,   # 12 × 100pt = ±1200 = ±2.4% of ~50000
    "FINNIFTY":  16,   # 16 × 25pt  = ±400  = ±2.0% of ~20000
    "SENSEX":    20,   # 20 × 100pt = ±2000 = ±2.8% of ~72000
    "BANKEX":    20,   # 20 × 100pt = ±2000 = ±3.2% of ~63000
    "BSEIT":     15,   # 15 × 200pt = ±3000 = ±8.1% of ~37000
}
# Expiry weekdays live in algo_platform.core.config.LOT_SIZES — single source of truth.
# This helper bridges to that config so callers that import from here still work.
def _expiry_weekday(instrument: str) -> int:
    from algo_platform.core.config import LOT_SIZES
    spec = LOT_SIZES.get(instrument.upper())
    return spec.expiry_weekday if spec else 3   # default Thursday

PUT_SKEW_SLOPE  = 0.15
CALL_SKEW_SLOPE = 0.08
OI_BASE         = 100_000.0   # OI at ATM
OI_DECAY        = 25.0        # exponential decay with |moneyness|
SPREAD_HALF     = 0.005       # half bid-ask as fraction of price


def next_expiry(ts: datetime, instrument: str, min_days: int = 0) -> date:
    """
    Return the nearest weekly expiry that is at least `min_days` calendar days away.
    min_days=0  → current week (for Strategy B, C)
    min_days=5  → forces next-week expiry (for Strategy A — gives spread time to work)

    Skip to NEXT week only AFTER market close (15:30 IST), not at 15:00.
    This is critical: options still trade until 15:30 and the exit at 15:15
    should use today's expiry, not next week's.
    """
    wd       = _expiry_weekday(instrument)
    d        = ts.date()
    days_fwd = (wd - d.weekday()) % 7
    # Market closes at 15:30 — only skip to next week after close
    after_close = ts.hour > 15 or (ts.hour == 15 and ts.minute >= 30)
    if days_fwd == 0 and after_close:
        days_fwd = 7
    expiry = d + timedelta(days=days_fwd)
    # Skip to next week's expiry if minimum days requirement not satisfied
    if (expiry - d).days < min_days:
        expiry += timedelta(days=7)
    return expiry


def _iv_surface(atm_iv: float, moneyness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (call_ivs, put_ivs) arrays shaped like moneyness.
    moneyness = (K - S) / S.
    """
    call_iv = atm_iv * (1.0 - CALL_SKEW_SLOPE * np.maximum(0.0, moneyness))
    put_iv  = atm_iv * (1.0 + PUT_SKEW_SLOPE  * np.maximum(0.0, -moneyness))
    call_iv = np.clip(call_iv, 0.05, atm_iv * 4.0)
    put_iv  = np.clip(put_iv,  0.05, atm_iv * 4.0)
    return call_iv, put_iv


def _bs_vectorized(
    S: float, K: np.ndarray, T: float, r: float, sigma: np.ndarray, is_call: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Batch Black-Scholes for N strikes simultaneously.
    Returns arrays: (price, delta, gamma, theta_per_day, vega_per_1pct).
    """
    sqrt_T  = math.sqrt(max(T, 1e-8))
    sigma   = np.maximum(sigma, 1e-6)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    Nd1  = norm.cdf(d1)
    Nd2  = norm.cdf(d2)
    npd1 = norm.pdf(d1)

    disc = math.exp(-r * T)

    if is_call:
        price = S * Nd1 - K * disc * Nd2
        delta = Nd1
        theta = (-S * npd1 * sigma / (2.0 * sqrt_T) - r * K * disc * Nd2) / 365.0
    else:
        price = K * disc * (1.0 - Nd2) - S * (1.0 - Nd1)
        delta = Nd1 - 1.0
        theta = (-S * npd1 * sigma / (2.0 * sqrt_T) + r * K * disc * (1.0 - Nd2)) / 365.0

    gamma = npd1 / (S * sigma * sqrt_T)
    vega  = S * npd1 * sqrt_T / 100.0

    price = np.maximum(price, 0.0)
    return price, delta, gamma, theta, vega


class SyntheticChainBuilder:
    """
    Constructs a realistic OptionChain from real spot + VIX using
    vectorised Black-Scholes. ~30× faster than scalar per-option loops.
    """

    def __init__(self, risk_free_rate: float = 0.065) -> None:
        self._rfr = risk_free_rate

    def build(
        self,
        instrument: str | Instrument,
        spot:       float,
        timestamp:  datetime,
        atm_iv:     float,
        min_days_to_expiry: int = 0,    # 0=current-week, 5=next-week for Strategy A
    ) -> OptionChain:
        inst_str = (instrument.value
                    if isinstance(instrument, Instrument)
                    else str(instrument).upper())
        inst_obj = (instrument
                    if isinstance(instrument, Instrument)
                    else Instrument(inst_str))

        expiry   = next_expiry(timestamp, inst_str, min_days=min_days_to_expiry)
        cal_days = (expiry - timestamp.date()).days
        hour_frac = max(0.0, (15.5 - timestamp.hour - timestamp.minute / 60.0)) / 6.25
        # Floor: 1 minute in trading-time years (not 1 calendar day — that was the bug
        # that caused exit chains at 3:15 PM to show 1-day TTE, inflating OTM options)
        MIN_TTE = 1.0 / (375 * 252)   # ≈ 1 trading minute
        tte = max(MIN_TTE, cal_days / 365.0 + hour_frac / 365.0)

        quotes = self._build_quotes(inst_str, spot, expiry, tte, atm_iv)

        return OptionChain(
            instrument = inst_obj,
            spot       = spot,
            timestamp  = timestamp,
            expiry     = expiry,
            quotes     = quotes,
            india_vix  = round(atm_iv * 100.0, 2),
        )

    # ── Private ────────────────────────────────────────────────────────────────

    def _build_quotes(
        self, instrument: str, spot: float, expiry: date,
        tte: float, atm_iv: float,
    ) -> List[OptionQuote]:
        step  = STRIKE_STEP.get(instrument, 50.0)
        n_otm = N_STRIKES_OTM.get(instrument, 12)

        atm_k   = round(spot / step) * step
        strikes  = np.array([atm_k + i * step for i in range(-n_otm, n_otm + 1)])
        money    = (strikes - spot) / spot          # moneyness array

        call_iv, put_iv = _iv_surface(atm_iv, money)

        # Vectorised BS for calls and puts
        c_price, c_delta, c_gamma, c_theta, c_vega = _bs_vectorized(
            spot, strikes, tte, self._rfr, call_iv, True
        )
        p_price, p_delta, p_gamma, p_theta, p_vega = _bs_vectorized(
            spot, strikes, tte, self._rfr, put_iv, False
        )

        # Synthetic OI: exponential decay from ATM
        oi = OI_BASE * np.exp(-OI_DECAY * np.abs(money))

        exp_str   = expiry.strftime("%y%b").upper()
        inst_name = instrument.upper()

        quotes: List[OptionQuote] = []
        for j, K in enumerate(strikes):
            for is_call, price, delta, gamma, theta, vega, iv in [
                (True,  c_price[j], c_delta[j], c_gamma[j], c_theta[j], c_vega[j], call_iv[j]),
                (False, p_price[j], p_delta[j], p_gamma[j], p_theta[j], p_vega[j], put_iv[j]),
            ]:
                if price < 0.10:
                    continue

                spread  = max(0.05, price * SPREAD_HALF)
                suffix  = "CE" if is_call else "PE"
                opt_type= OptionType.CALL if is_call else OptionType.PUT
                oi_val  = float(oi[j])

                quotes.append(OptionQuote(
                    symbol      = f"NSE:{inst_name}{exp_str}{int(K)}{suffix}",
                    instrument  = Instrument(inst_name),
                    strike      = float(K),
                    option_type = opt_type,
                    expiry      = expiry,
                    ltp         = round(float(price),  2),
                    bid         = round(max(0.05, float(price) - spread), 2),
                    ask         = round(float(price) + spread, 2),
                    oi          = round(oi_val),
                    oi_change   = 0.0,
                    volume      = round(oi_val * 0.03),
                    iv          = round(float(iv), 4),
                    delta       = round(float(delta), 4),
                    gamma       = round(float(gamma), 6),
                    theta       = round(float(theta), 4),
                    vega        = round(float(vega),  4),
                ))

        return quotes
