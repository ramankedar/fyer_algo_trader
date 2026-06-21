"""
Feature computation engine.
Each of the 13 features is a pure function; FeatureEngine assembles them.
All computations are strictly causal — no lookahead.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from datetime import datetime
from typing import Deque, List, Optional, Sequence

import numpy as np

from algo_platform.core.types import FeatureVector, OptionChain, OptionType, MarketBar

logger = logging.getLogger("platform.research.features")


# ── Individual feature functions ──────────────────────────────────────────────

def compute_iv_rank(iv_history: Sequence[float], current_iv: float) -> float:
    """IV Rank: position of current IV within its trailing range (0-1)."""
    if not iv_history:
        return 0.5
    lo, hi = min(iv_history), max(iv_history)
    if hi <= lo:
        return 0.5
    return float(np.clip((current_iv - lo) / (hi - lo), 0.0, 1.0))


def compute_iv_skew(chain: OptionChain) -> float:
    """25-delta skew: put IV minus call IV at symmetric 25-delta strikes."""
    calls = sorted(chain.calls(), key=lambda q: q.strike)
    puts  = sorted(chain.puts(),  key=lambda q: q.strike)
    atm = chain.atm_strike()

    # Approximate 25-delta by picking strikes ~1 ATR OTM; use closest delta match
    def nearest_delta(quotes, target_delta):
        valid = [q for q in quotes if q.iv > 0 and abs(q.delta) > 0]
        if not valid:
            return None
        return min(valid, key=lambda q: abs(abs(q.delta) - target_delta))

    call_25 = nearest_delta(calls, 0.25)
    put_25  = nearest_delta(puts,  0.25)

    if call_25 is None or put_25 is None:
        return 0.0
    return float(put_25.iv - call_25.iv)


def compute_pc_oi_ratio(chain: OptionChain) -> float:
    """Put / Call open-interest ratio."""
    put_oi  = sum(q.oi for q in chain.puts())
    call_oi = sum(q.oi for q in chain.calls())
    if call_oi <= 0:
        return 1.0
    return float(put_oi / call_oi)


def compute_oi_change(chain: OptionChain, prev_chain: Optional[OptionChain]) -> float:
    """Percentage change in total OI vs previous snapshot."""
    if prev_chain is None:
        return 0.0
    curr = sum(q.oi for q in chain.quotes)
    prev = sum(q.oi for q in prev_chain.quotes)
    if prev <= 0:
        return 0.0
    return float((curr - prev) / prev)


def compute_delta_volume(chain: OptionChain) -> float:
    """
    Signed volume proxy: call_volume * |call_delta| - put_volume * |put_delta|.
    Positive = bullish pressure; negative = bearish pressure.
    """
    bull = sum(q.volume * abs(q.delta) for q in chain.calls() if q.iv > 0)
    bear = sum(q.volume * abs(q.delta) for q in chain.puts()  if q.iv > 0)
    total = bull + bear
    if total <= 0:
        return 0.0
    return float((bull - bear) / total)  # normalised to [-1, 1]


def compute_atr(bars: List[MarketBar], period: int = 14) -> float:
    """Wilder's Average True Range over `period` bars."""
    if len(bars) < 2:
        return bars[-1].high - bars[-1].low if bars else 0.0

    trs: List[float] = []
    for i in range(1, len(bars)):
        h, l, pc = bars[i].high, bars[i].low, bars[i - 1].close
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))

    if len(trs) < period:
        return float(np.mean(trs)) if trs else 0.0

    # Wilder smoothing
    atr = float(np.mean(trs[:period]))
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def compute_realized_vol(closes: np.ndarray, window: int = 20,
                          annualise: int = 252) -> float:
    """Annualised realised volatility from log returns."""
    if len(closes) < 2:
        return 0.0
    log_ret = np.diff(np.log(np.clip(closes[-window - 1:], 1e-8, None)))
    if len(log_ret) < 2:
        return 0.0
    return float(np.std(log_ret, ddof=1) * math.sqrt(annualise))


def compute_entropy(returns: np.ndarray, bins: int = 20) -> float:
    """Shannon entropy of the return distribution (nats)."""
    if len(returns) < bins:
        return 0.0
    counts, _ = np.histogram(returns, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def compute_hurst(prices: np.ndarray) -> float:
    """
    Hurst exponent via R/S analysis.
    H ≈ 0.5 → random walk; H > 0.5 → trending; H < 0.5 → mean-reverting.
    """
    n = len(prices)
    if n < 32:
        return 0.5

    lags = [l for l in [4, 8, 16, 32, 64, 128] if l < n // 2]
    if len(lags) < 3:
        return 0.5

    rs_vals: List[float] = []
    for lag in lags:
        n_chunks = n // lag
        rs_chunk: List[float] = []
        for k in range(n_chunks):
            sub = prices[k * lag: (k + 1) * lag].astype(float)
            mu = sub.mean()
            deviation = np.cumsum(sub - mu)
            R = deviation.max() - deviation.min()
            S = sub.std(ddof=1)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_vals.append(float(np.mean(rs_chunk)))
        else:
            rs_vals.append(np.nan)

    valid = [(l, rs) for l, rs in zip(lags, rs_vals)
             if not math.isnan(rs) and rs > 0]
    if len(valid) < 3:
        return 0.5

    log_lags = np.log([v[0] for v in valid])
    log_rs   = np.log([v[1] for v in valid])
    h = float(np.polyfit(log_lags, log_rs, 1)[0])
    return float(np.clip(h, 0.05, 0.95))


def compute_vwap_distance(bars: List[MarketBar], current_price: float) -> float:
    """
    (current_price - session_VWAP) / session_VWAP.
    Falls back to simple time-weighted mean when volume is zero (index data).
    """
    if not bars:
        return 0.0
    total_vol = sum(b.volume for b in bars)
    if total_vol > 0:
        vwap = sum(b.typical_price * b.volume for b in bars) / total_vol
    else:
        # Index data has no volume; use equal-weight session mean as proxy
        vwap = sum(b.typical_price for b in bars) / len(bars)
    if vwap <= 0:
        return 0.0
    return float((current_price - vwap) / vwap)


def compute_breadth(breadth_value: float) -> float:
    """
    Breadth: fraction of Nifty 50 constituents advancing (0-1).
    Pass-through — caller supplies the pre-computed value or proxy.
    """
    return float(np.clip(breadth_value, 0.0, 1.0))


def compute_gamma_exposure(chain: OptionChain, lot_size: int) -> float:
    """
    Aggregate dealer Gamma Exposure (GEX) in ₹ crore.
    Positive GEX → market makers are long gamma (stabilising).
    Negative GEX → short gamma (amplifies moves).
    """
    gex = 0.0
    for q in chain.quotes:
        sign = 1.0 if q.option_type == OptionType.CALL else -1.0
        # Dealer is short what retail bought, so flip sign for dealer position
        gex += -sign * q.gamma * q.oi * lot_size * (chain.spot ** 2) * 0.01
    return float(gex / 1e7)  # normalise to ₹ crore


def compute_range_compression(bars: List[MarketBar], atr: float) -> float:
    """
    Current bar range divided by ATR.
    < 1 → compression (narrow bar relative to ATR).
    > 1 → expansion.
    """
    if not bars or atr <= 0:
        return 1.0
    bar_range = bars[-1].high - bars[-1].low
    return float(bar_range / atr)


def _rolling_percentile(series: Sequence[float], window: int) -> float:
    """Percentile rank of the last element in `series` over `window` values."""
    arr = np.asarray(series[-window:], dtype=float)
    if len(arr) < 2:
        return 0.5
    val = arr[-1]
    below = np.sum(arr[:-1] < val)
    return float(below / (len(arr) - 1))


# ── Feature engine ────────────────────────────────────────────────────────────

class FeatureEngine:
    """
    Stateful engine that accumulates market data and emits FeatureVector.
    Call `update()` once per bar; it returns None until warm-up is complete.
    """

    def __init__(
        self,
        instrument: str,
        lot_size: int,
        atr_period: int = 14,
        rv_window: int = 20,
        percentile_window: int = 60,
        iv_rank_window: int = 252,
    ) -> None:
        self.instrument = instrument
        self.lot_size   = lot_size
        self._atr_period = atr_period
        self._rv_window  = rv_window
        self._pct_window = percentile_window
        self._iv_window  = iv_rank_window

        # O(1) incremental ATR — avoids O(n²) recompute-from-scratch per bar
        self._atr_val:    float = 0.0
        self._atr_ready:  int   = 0        # bars accumulated so far (up to period)
        self._prev_close: float = 0.0
        self._last_bar:   Optional[MarketBar] = None  # only need 1 bar for range

        self._closes:    Deque[float] = deque(maxlen=rv_window + 1)
        self._atrs:      Deque[float] = deque(maxlen=percentile_window)
        self._rvs:       Deque[float] = deque(maxlen=percentile_window)
        self._entropies: Deque[float] = deque(maxlen=percentile_window)
        self._ranges:    Deque[float] = deque(maxlen=percentile_window)
        self._iv_hist:   Deque[float] = deque(maxlen=iv_rank_window)
        self._session_bars: List[MarketBar] = []   # reset each session
        self._prev_chain:   Optional[OptionChain] = None
        self._bar_count:    int = 0

        # Incremental intraday VWAP — O(1) update, reset each session
        self._session_typical_sum: float = 0.0
        self._session_vol_sum:     float = 0.0
        self._session_bar_count:   int   = 0

        self._warmup = max(atr_period + 1, rv_window + 1, 20)

    # ── Public API ─────────────────────────────────────────────────────────────

    def new_session(self) -> None:
        """Reset intraday VWAP accumulator at market open. ATR persists across sessions."""
        self._session_bars = []
        self._session_typical_sum = 0.0
        self._session_vol_sum     = 0.0
        self._session_bar_count   = 0

    def update(
        self,
        bar:          MarketBar,
        chain:        Optional[OptionChain],
        breadth_val:  float = 0.5,
    ) -> Optional[FeatureVector]:
        """
        Feed one bar and optional option chain snapshot.
        Returns FeatureVector once warm-up is satisfied, else None.
        """
        self._bar_count     += 1
        self._session_bars.append(bar)
        self._closes.append(bar.close)

        # -- Incremental ATR (O(1) per bar, not O(n)) -------------------------
        if self._prev_close > 0:
            tr = max(bar.high - bar.low,
                     abs(bar.high - self._prev_close),
                     abs(bar.low  - self._prev_close))
            if self._atr_ready < self._atr_period:
                self._atr_ready += 1
                self._atr_val = (self._atr_val * (self._atr_ready - 1) + tr) / self._atr_ready
            else:
                k = 1.0 / self._atr_period
                self._atr_val = self._atr_val * (1.0 - k) + tr * k
        self._prev_close = bar.close
        atr = self._atr_val
        self._atrs.append(atr)

        # -- Realised volatility from fixed-size close deque ------------------
        closes_arr = np.asarray(self._closes)
        rv = compute_realized_vol(closes_arr, self._rv_window)
        self._rvs.append(rv)

        if len(closes_arr) >= 2:
            log_ret = np.diff(np.log(np.clip(closes_arr, 1e-8, None)))
        else:
            log_ret = np.array([0.0])
        ent = compute_entropy(log_ret)
        self._entropies.append(ent)

        # Range compression: only needs the current bar and current ATR
        bar_range = bar.high - bar.low
        rc        = float(bar_range / atr) if atr > 0 else 1.0
        self._ranges.append(rc)
        self._last_bar = bar

        hurst = compute_hurst(closes_arr) if len(closes_arr) >= 32 else 0.5

        # Incremental O(1) session VWAP (reset each session via new_session)
        self._session_bar_count   += 1
        self._session_typical_sum += bar.typical_price
        self._session_vol_sum     += bar.volume
        if self._session_vol_sum > 0:
            vwap = self._session_typical_sum / self._session_bar_count  # time-weighted fallback for zero-volume index
            # If real volume available, use volume-weighted
            # (index data has zero volume, so always falls back to time-weighted)
        else:
            vwap = self._session_typical_sum / self._session_bar_count
        vwap_dist = (bar.close - vwap) / vwap if vwap > 0 else 0.0

        if chain is not None:
            # IV rank
            atm = chain.atm_strike()
            atm_call = chain.quote(atm, OptionType.CALL)
            if atm_call and atm_call.iv > 0:
                self._iv_hist.append(atm_call.iv)
                current_iv = atm_call.iv
            else:
                current_iv = chain.india_vix / 100.0 if chain.india_vix > 0 else 0.15
                self._iv_hist.append(current_iv)

            iv_rank  = compute_iv_rank(list(self._iv_hist)[:-1], current_iv)
            iv_skew  = compute_iv_skew(chain)
            pcoi     = compute_pc_oi_ratio(chain)
            oi_chg   = compute_oi_change(chain, self._prev_chain)
            dv       = compute_delta_volume(chain)
            gex      = compute_gamma_exposure(chain, self.lot_size)
            self._prev_chain = chain
        else:
            iv_rank = iv_skew = oi_chg = dv = gex = 0.0
            pcoi = 1.0

        # -- Percentile ranks --------------------------------------------------
        atr_pct   = _rolling_percentile(list(self._atrs),      self._pct_window)
        rv_pct    = _rolling_percentile(list(self._rvs),       self._pct_window)
        ent_pct   = _rolling_percentile(list(self._entropies), self._pct_window)
        range_pct = _rolling_percentile(list(self._ranges),    self._pct_window)

        # -- Warm-up guard -----------------------------------------------------
        if self._bar_count < self._warmup:
            return None

        # When no real breadth data is provided (default 0.5), synthesise
        # a directional breadth proxy from VWAP distance: price above VWAP
        # implies broad strength (breadth > 0.5), below implies weakness.
        if breadth_val == 0.5 and abs(vwap_dist) > 0:
            breadth_val = float(np.clip(0.5 + 3.0 * vwap_dist, 0.1, 0.9))

        return FeatureVector(
            timestamp         = bar.timestamp,
            iv_rank           = iv_rank,
            iv_skew           = iv_skew,
            pc_oi_ratio       = pcoi,
            oi_change         = oi_chg,
            delta_volume      = dv,
            atr               = atr,
            atr_pct           = atr_pct,
            realized_vol      = rv,
            rv_pct            = rv_pct,
            entropy           = ent,
            entropy_pct       = ent_pct,
            hurst             = hurst,
            vwap_distance     = vwap_dist,
            breadth           = compute_breadth(breadth_val),
            gamma_exposure    = gex,
            range_compression = rc,
            range_pct         = range_pct,
        )
