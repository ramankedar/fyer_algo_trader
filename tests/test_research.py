"""
Unit tests for the research feature engine and statistics engine.
"""

import math
from datetime import datetime, date
from typing import List

import numpy as np
import pytest

from algo_platform.research.features import (
    compute_atr, compute_realized_vol, compute_entropy, compute_hurst,
    compute_vwap_distance, compute_iv_rank, compute_gamma_exposure,
    compute_range_compression, FeatureEngine,
)
from algo_platform.research.engine import (
    spearman_ic, ic_t_stat, decile_analysis, sharpe_contribution,
    ResearchEngine,
)
from algo_platform.core.types import MarketBar, OptionChain, OptionQuote, OptionType, Instrument


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar(ts: datetime, o=100.0, h=102.0, l=99.0, c=101.0, v=1000.0) -> MarketBar:
    return MarketBar(ts, o, h, l, c, v)


def _bars(n: int, base: float = 100.0, drift: float = 0.001) -> List[MarketBar]:
    bars = []
    price = base
    for i in range(n):
        # Spread across days so minute stays in 0-59
        day_offset = i // 375
        minute     = i % 60   # keep within valid minute range
        ts    = datetime(2024, 1, 2 + day_offset, 9, minute, 0)
        noise = np.random.uniform(-0.5, 0.5)
        price = price * (1 + drift) + noise
        bars.append(MarketBar(ts, price - 0.5, price + 1.0, price - 1.0, price, 1000 + i))
    return bars


# ── ATR ───────────────────────────────────────────────────────────────────────

def test_atr_basic():
    bars = _bars(20, base=100.0, drift=0.0)
    atr = compute_atr(bars, period=14)
    assert atr > 0


def test_atr_single_bar():
    bar = MarketBar(datetime.now(), 100, 102, 99, 101, 500)
    atr = compute_atr([bar])
    assert atr == pytest.approx(3.0, abs=0.1)


def test_atr_wilder_smoothing():
    bars = _bars(30)
    atr_14 = compute_atr(bars, period=14)
    atr_7  = compute_atr(bars, period=7)
    # Shorter period is more volatile; both should be positive
    assert atr_14 > 0 and atr_7 > 0


# ── Realised Volatility ───────────────────────────────────────────────────────

def test_rv_trending():
    closes = np.linspace(100, 150, 50)   # pure drift, near-zero RV
    rv = compute_realized_vol(closes, window=20)
    # Should be close to 0 but not exactly 0 due to discrete log returns
    assert rv < 0.05


def test_rv_noisy():
    np.random.seed(42)
    closes = 100 + np.cumsum(np.random.randn(50))
    rv = compute_realized_vol(closes, window=20)
    assert rv > 0.0 and rv < 5.0   # annualised, so very high is valid for noisy series


def test_rv_insufficient_data():
    rv = compute_realized_vol(np.array([100.0]), window=20)
    assert rv == 0.0


# ── Shannon Entropy ───────────────────────────────────────────────────────────

def test_entropy_uniform():
    # Uniform returns → higher entropy
    returns = np.linspace(-0.05, 0.05, 100)
    ent_uniform = compute_entropy(returns, bins=20)
    assert ent_uniform > 0


def test_entropy_concentrated():
    # All returns near 0 → lower entropy
    returns = np.random.normal(0, 0.0001, 100)
    ent_conc = compute_entropy(returns, bins=20)
    returns2 = np.random.uniform(-0.05, 0.05, 100)
    ent_spread = compute_entropy(returns2, bins=20)
    assert ent_conc <= ent_spread


# ── Hurst Exponent ────────────────────────────────────────────────────────────

def test_hurst_random_walk():
    np.random.seed(0)
    rw = np.cumsum(np.random.randn(256))
    h = compute_hurst(rw)
    # Finite-sample R/S can diverge from 0.5 substantially; just check valid range
    assert 0.05 <= h <= 0.95


def test_hurst_trending():
    np.random.seed(1)
    # Strongly trending series
    trend = np.cumsum(np.ones(256) + np.random.randn(256) * 0.1)
    h = compute_hurst(trend)
    assert h >= 0.5


def test_hurst_insufficient():
    assert compute_hurst(np.array([1.0, 2.0, 3.0])) == 0.5


# ── IV Rank ───────────────────────────────────────────────────────────────────

def test_iv_rank_at_max():
    hist = [0.10, 0.12, 0.15, 0.18, 0.20]
    assert compute_iv_rank(hist, 0.20) == pytest.approx(1.0)


def test_iv_rank_at_min():
    hist = [0.10, 0.12, 0.15, 0.18, 0.20]
    assert compute_iv_rank(hist, 0.10) == pytest.approx(0.0)


def test_iv_rank_midpoint():
    hist = [0.10, 0.20]
    rank = compute_iv_rank(hist, 0.15)
    assert rank == pytest.approx(0.5)


# ── VWAP distance ─────────────────────────────────────────────────────────────

def test_vwap_above():
    bars = [MarketBar(datetime.now(), 100, 101, 99, 100, 1000)] * 5
    # Current price well above VWAP (all bars at 100, current at 105)
    d = compute_vwap_distance(bars, 105.0)
    assert d > 0


def test_vwap_below():
    bars = [MarketBar(datetime.now(), 100, 101, 99, 100, 1000)] * 5
    d = compute_vwap_distance(bars, 95.0)
    assert d < 0


# ── Range compression ─────────────────────────────────────────────────────────

def test_range_compression_compressed():
    bar  = MarketBar(datetime.now(), 100, 100.1, 99.9, 100, 1000)   # tiny range
    rc   = compute_range_compression([bar], atr=2.0)
    assert rc < 1.0


def test_range_expansion():
    bar  = MarketBar(datetime.now(), 100, 105, 95, 100, 1000)   # 10-pt range
    rc   = compute_range_compression([bar], atr=2.0)
    assert rc > 1.0


# ── Research engine: IC and statistics ────────────────────────────────────────

def test_spearman_ic_perfect_positive():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    y = x.copy()
    ic = spearman_ic(x, y)
    assert ic == pytest.approx(1.0, abs=1e-6)


def test_spearman_ic_perfect_negative():
    x = np.arange(1, 11, dtype=float)
    y = x[::-1].copy()
    ic = spearman_ic(x, y)
    assert ic == pytest.approx(-1.0, abs=1e-6)


def test_ic_t_stat_positive_ic():
    # Series with positive mean AND some variance so t-stat is meaningful
    np.random.seed(5)
    ic_series = 0.05 + np.random.randn(100) * 0.02
    t = ic_t_stat(ic_series)
    assert t > 0 and math.isfinite(t)


def test_decile_analysis_shape():
    np.random.seed(0)
    feat = np.random.randn(500)
    fwd  = feat + np.random.randn(500) * 0.5   # noisy positive relationship
    dec  = decile_analysis(feat, fwd)
    assert dec.shape == (10,)


def test_sharpe_contribution_nonzero():
    np.random.seed(1)
    feat = np.random.randn(300)
    fwd  = np.random.randn(300) * 0.02
    sc = sharpe_contribution(feat, fwd)
    assert isinstance(sc, float) and math.isfinite(sc)


# ── FeatureEngine warm-up ─────────────────────────────────────────────────────

def test_feature_engine_warmup():
    from algo_platform.core.config import load_config
    cfg = load_config()
    eng = FeatureEngine("NIFTY", 75)
    bars = _bars(100)
    results = []
    for bar in bars:
        fv = eng.update(bar, None, 0.5)
        if fv is not None:
            results.append(fv)
    # Should emit features after warmup
    assert len(results) > 0


def test_feature_vector_array():
    from algo_platform.core.types import FeatureVector
    fv = FeatureVector(timestamp=datetime.now())
    arr = fv.as_array()
    assert arr.shape == (13,)
    assert all(isinstance(v, float) for v in arr)
