"""
Unit tests for the three strategy implementations.
Tests verify signal-generation logic, time-gating, and exit conditions.
"""

import pytest
from datetime import datetime, date
from typing import List, Optional

import numpy as np

from algo_platform.core.config import load_config
from algo_platform.core.types import (
    FeatureVector, Instrument, MarketBar, OptionChain, OptionQuote,
    OptionType, SignalDirection,
)
from algo_platform.strategies import (
    VolatilityCompressionStrategy,
    TrendFollowingStrategy,
    GammaExpansionStrategy,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar(ts: datetime = None, close: float = 22000.0,
         high: float = 22050.0, low: float = 21950.0,
         volume: float = 10000.0) -> MarketBar:
    ts = ts or datetime(2024, 6, 6, 11, 0)  # Thursday
    return MarketBar(ts, close - 50, high, low, close, volume)


def _feature(
    atr_pct: float = 0.5, rv_pct: float = 0.5, entropy_pct: float = 0.5,
    range_pct: float = 0.5, volume_spike: bool = False,
    vwap_dist: float = 0.0, breadth: float = 0.5,
    hurst: float = 0.5, atr: float = 200.0,
    rv: float = 0.15, gex: float = 0.0,
    pc_oi: float = 1.0, oi_chg: float = 0.0,
) -> FeatureVector:
    return FeatureVector(
        timestamp        = datetime.now(),
        iv_rank          = 0.5,
        iv_skew          = 0.0,
        pc_oi_ratio      = pc_oi,
        oi_change        = oi_chg,
        delta_volume     = 0.0,
        atr              = atr,
        atr_pct          = atr_pct,
        realized_vol     = rv,
        rv_pct           = rv_pct,
        entropy          = 2.0,
        entropy_pct      = entropy_pct,
        hurst            = hurst,
        vwap_distance    = vwap_dist,
        breadth          = breadth,
        gamma_exposure   = gex,
        range_compression= 0.5 if range_pct < 0.5 else 1.5,
        range_pct        = range_pct,
    )


def _make_chain(spot: float = 22000.0, ts: datetime = None) -> OptionChain:
    ts = ts or datetime(2024, 6, 6, 11, 0)
    strikes = [spot - 100, spot, spot + 100]
    quotes  = []
    for s in strikes:
        for ot, sign in [(OptionType.CALL, 1), (OptionType.PUT, -1)]:
            delta = max(0.01, min(0.99, 0.5 - sign * 0.25 * (s - spot) / 100))
            quotes.append(OptionQuote(
                symbol      = f"NSE:NIFTY{'CE' if ot == OptionType.CALL else 'PE'}{int(s)}",
                instrument  = Instrument.NIFTY,
                strike      = s,
                option_type = ot,
                expiry      = date(2024, 6, 6),
                ltp         = max(5.0, 200.0 - abs(s - spot) * 0.5),
                bid         = max(4.0, 195.0 - abs(s - spot) * 0.5),
                ask         = max(6.0, 205.0 - abs(s - spot) * 0.5),
                oi          = 50000.0,
                oi_change   = 100.0,
                volume      = 10000.0,
                iv          = 0.15,
                delta       = delta * sign,
                gamma       = 0.002,
                theta       = -5.0,
                vega        = 50.0,
            ))
    return OptionChain(
        instrument = Instrument.NIFTY,
        spot       = spot,
        timestamp  = ts,
        expiry     = date(2024, 6, 6),
        quotes     = quotes,
        india_vix  = 15.0,
    )


# ── Strategy A: Volatility Compression ────────────────────────────────────────

class TestVolCompressionStrategy:
    def setup_method(self):
        self.cfg      = load_config()
        self.strategy = VolatilityCompressionStrategy(Instrument.NIFTY, self.cfg)

    def test_no_signal_during_warmup(self):
        chain = _make_chain()
        f     = _feature(atr_pct=0.10, rv_pct=0.10, entropy_pct=0.10, range_pct=0.10)
        bar   = _bar(ts=datetime(2024, 6, 3, 11, 0))  # Monday
        # Only one bar — warmup not complete
        sig = self.strategy.generate_signal(bar, chain, f)
        assert sig is None

    def test_no_signal_outside_time_window(self):
        chain = _make_chain()
        f     = _feature(atr_pct=0.10, rv_pct=0.10, entropy_pct=0.10, range_pct=0.10)
        # 15:30 = outside entry window
        ts    = datetime(2024, 6, 3, 15, 30)
        sig   = self.strategy.generate_signal(_bar(ts=ts), chain, f)
        assert sig is None

    def test_compression_gate_requires_all_four(self):
        # Only 3 of 4 conditions in compression zone
        f = _feature(atr_pct=0.10, rv_pct=0.10, entropy_pct=0.10, range_pct=0.50)
        assert not self.strategy._in_compression(f)

    def test_compression_gate_all_four(self):
        f = _feature(atr_pct=0.10, rv_pct=0.10, entropy_pct=0.10, range_pct=0.10)
        assert self.strategy._in_compression(f)


# ── Strategy B: Trend Following ────────────────────────────────────────────────

class TestTrendFollowingStrategy:
    def setup_method(self):
        self.cfg      = load_config()
        self.strategy = TrendFollowingStrategy(Instrument.NIFTY, self.cfg)

    def test_no_signal_without_warmup(self):
        chain = _make_chain()
        f     = _feature(vwap_dist=0.01, breadth=0.7)
        sig   = self.strategy.generate_signal(_bar(), chain, f)
        assert sig is None   # not enough bars yet

    def test_new_session_resets_flag(self):
        self.strategy._signal_emitted_today = True
        self.strategy.new_session()
        assert not self.strategy._signal_emitted_today

    def test_adx_below_threshold_no_signal(self):
        # Inject enough bars to pass warmup
        chain = _make_chain()
        f     = _feature(vwap_dist=0.02, breadth=0.8)
        for _ in range(50):
            self.strategy._bars.append(_bar(ts=datetime(2024, 6, 3, 10, 30)))

        # ADX will be computed on these flat bars and should be very low
        ts  = datetime(2024, 6, 3, 11, 0)
        sig = self.strategy.generate_signal(_bar(ts=ts), chain, f)
        # ADX on flat bars < 25 → no signal
        assert sig is None

    def test_should_exit_at_time_stop(self):
        self.strategy._in_trade        = True
        self.strategy._active_direction = SignalDirection.LONG
        ts   = datetime(2024, 6, 3, 15, 30)   # past square-off
        f    = _feature()
        exit_, reason = self.strategy.should_exit(_bar(ts=ts), f)
        assert exit_ is True
        assert reason == "time_stop"

    def test_no_early_exit_on_vwap_cross(self):
        # VWAP crossover exit removed — only time_stop exits now (data showed it
        # was firing 67% of trades at 6% win rate vs time_stop's 80% win rate)
        self.strategy._in_trade         = True
        self.strategy._active_direction = SignalDirection.LONG
        ts   = datetime(2024, 6, 3, 11, 30)   # inside window
        f    = _feature(vwap_dist=-0.005)       # would have triggered old VWAP exit
        exit_, reason = self.strategy.should_exit(_bar(ts=ts), f)
        assert exit_ is False   # no early exit any more


# ── Strategy C: Gamma Expansion ────────────────────────────────────────────────

class TestGammaExpansionStrategy:
    def setup_method(self):
        self.cfg      = load_config()
        self.strategy = GammaExpansionStrategy(Instrument.NIFTY, self.cfg)

    def test_only_thursday(self):
        chain = _make_chain()
        f     = _feature(gex=100.0, pc_oi=2.0)
        # Monday = weekday 0
        ts    = datetime(2024, 6, 3, 14, 0)   # Monday
        sig   = self.strategy.generate_signal(_bar(ts=ts), chain, f)
        assert sig is None

    def test_outside_time_window(self):
        chain = _make_chain()
        f     = _feature(gex=100.0, pc_oi=2.0)
        # Thursday at 10:00 — before the 13:30-15:00 window
        ts    = datetime(2024, 6, 6, 10, 0)   # Thursday
        sig   = self.strategy.generate_signal(_bar(ts=ts), chain, f)
        assert sig is None

    def test_one_signal_per_session(self):
        self.strategy._signal_today = True
        chain = _make_chain()
        f     = _feature(gex=100.0, pc_oi=2.0)
        ts    = datetime(2024, 6, 6, 14, 0)   # Thursday, within window
        sig   = self.strategy.generate_signal(_bar(ts=ts), chain, f)
        assert sig is None

    def test_should_exit_time_stop(self):
        self.strategy._in_trade    = True
        self.strategy._entry_debit = 100.0
        ts   = datetime(2024, 6, 6, 15, 20)   # past square-off
        f    = _feature()
        exit_, reason = self.strategy.should_exit(_bar(ts=ts), f, 100.0)
        assert exit_ is True
        assert reason == "time_stop"

    def test_should_exit_fifty_pct_loss(self):
        self.strategy._in_trade    = True
        self.strategy._entry_debit = 100.0
        ts   = datetime(2024, 6, 6, 14, 0)   # inside window
        f    = _feature()
        # Spread now worth only 40% of entry
        exit_, reason = self.strategy.should_exit(_bar(ts=ts), f, 40.0)
        assert exit_ is True
        assert reason == "fifty_pct_loss"
