"""
Strategy A — Volatility Compression Expansion.

Logic:
  1. Wait until all 4 conditions are simultaneously in their 20th percentile or below
     (ATR, realised vol, entropy, range width).
  2. Enter on breakout above the compression range HIGH + volume spike > 2× median.
  3. Build an ATM debit spread in the breakout direction.
  4. Exit: dynamic ATR trailing stop | time stop | vol stop (ATR doubles).
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from statistics import median
from typing import Deque, List, Optional

from algo_platform.core.config import PlatformConfig, StrategyAConfig
from algo_platform.core.types import (
    FeatureVector, Instrument, OptionChain, Signal, SignalDirection, MarketBar,
)
from algo_platform.strategies.base import BaseStrategy

logger = logging.getLogger("platform.strategies.vol_compression")

_COMPRESSION_BARS = 20   # lookback to define the compression range


class VolatilityCompressionStrategy(BaseStrategy):
    """Strategy A: Volatility Compression → Breakout trade via debit spread."""

    name = "VolCompressionExpansion"

    def __init__(self, instrument: Instrument, config: PlatformConfig,
                 quantity: int = 1) -> None:
        super().__init__(instrument, config)
        self._cfg: StrategyAConfig = config.strategy_a
        self._quantity   = quantity
        self._lot_size   = config.lot_size(instrument.value)
        self._width      = config.spread_width(instrument.value, "A")

        # Compression range tracking
        self._highs: Deque[float] = deque(maxlen=_COMPRESSION_BARS)
        self._lows:  Deque[float] = deque(maxlen=_COMPRESSION_BARS)
        self._vols:  Deque[float] = deque(maxlen=50)   # for median volume

        # Live trade state
        self._entry_atr:   float = 0.0
        self._peak_close:  float = 0.0
        self._entry_close: float = 0.0
        self._entry_direction: SignalDirection = SignalDirection.NEUTRAL

    def new_session(self) -> None:
        self._in_trade = False

    # ── Main signal generator ──────────────────────────────────────────────────

    def generate_signal(
        self,
        bar:      MarketBar,
        chain:    Optional[OptionChain],
        features: FeatureVector,
    ) -> Optional[Signal]:
        if not self._is_active:
            return None
        if chain is None:
            return None

        self._highs.append(bar.high)
        self._lows.append(bar.low)
        self._vols.append(bar.volume)

        if not self._in_window(bar.timestamp,
                               self._cfg.entry_start, self._cfg.entry_end):
            return None
        if len(self._highs) < _COMPRESSION_BARS:
            return None

        # ── Compression gate ──────────────────────────────────────────────────
        if not self._in_compression(features):
            return None

        # ── Breakout gate ─────────────────────────────────────────────────────
        comp_high = max(list(self._highs)[:-1])   # excluding current bar
        comp_low  = min(list(self._lows)[:-1])
        med_vol   = median(self._vols) if self._vols else 0.0

        direction = self._detect_breakout(bar, comp_high, comp_low, med_vol)
        if direction == SignalDirection.NEUTRAL:
            return None

        # ── Build debit spread using NEXT-WEEK options ───────────────────────────
        # Compression breakouts need time to develop. Same-week options expire
        # in hours → 94% lose to theta even when direction is correct.
        # Min 5 days to expiry ensures the spread has a full week to capture the move.
        from algo_platform.data.chain_builder import SyntheticChainBuilder
        nw_builder = SyntheticChainBuilder(self.config.risk_free_rate)
        # Use current chain's IV as input
        current_iv = chain.india_vix / 100.0 if chain.india_vix > 0 else 0.14
        nw_chain = nw_builder.build(
            self.instrument, bar.close, bar.timestamp,
            atm_iv=current_iv, min_days_to_expiry=5
        )
        atm = nw_chain.atm_strike()
        if direction == SignalDirection.LONG:
            legs = self._build_call_spread(nw_chain, atm, self._width,
                                           self._quantity, self._lot_size)
        else:
            legs = self._build_put_spread(nw_chain, atm, self._width,
                                          self._quantity, self._lot_size)

        if not legs:
            logger.warning("StrategyA: could not build spread at ATM=%.0f", atm)
            return None

        debit      = self._net_debit(legs)
        max_loss   = self._max_loss(legs)
        max_profit = self._max_profit(legs, self._width)

        # Save state for exit management
        self._entry_atr       = features.atr
        self._peak_close      = bar.close
        self._entry_close     = bar.close
        self._entry_direction = direction
        self._in_trade        = True

        logger.info(
            "StrategyA SIGNAL %s %s | ATM=%.0f debit=%.2f max_loss=%.2f",
            direction.value, self.instrument.value, atm, debit, max_loss,
        )

        return Signal(
            strategy   = self.name,
            instrument = self.instrument,
            direction  = direction,
            timestamp  = bar.timestamp,
            legs       = legs,
            net_debit  = debit,
            max_loss   = max_loss,
            max_profit = max_profit,
            confidence = self._confidence(features),
            features   = features,
            metadata   = {
                "comp_high": comp_high,
                "comp_low":  comp_low,
                "atr":       features.atr,
            },
        )

    # ── Exit logic (called by backtest engine / execution layer) ───────────────

    def should_exit(
        self,
        bar:      MarketBar,
        features: FeatureVector,
        entry_debit: float,
        current_value: float,
    ) -> tuple[bool, str]:
        """
        Returns (should_exit, reason).
        Called by the execution/backtest layer on each bar once in a trade.
        """
        if not self._in_trade:
            return False, ""

        # Update peak
        if self._entry_direction == SignalDirection.LONG:
            self._peak_close = max(self._peak_close, bar.close)
        else:
            self._peak_close = min(self._peak_close, bar.close)

        # 1. Time stop
        if not self._in_window(bar.timestamp, self._cfg.entry_start,
                                self._cfg.square_off):
            self._in_trade = False
            return True, "time_stop"

        # 2. ATR trailing stop
        trail = self._cfg.trail_atr_mult * self._entry_atr
        if self._entry_direction == SignalDirection.LONG:
            if bar.close < self._peak_close - trail:
                self._in_trade = False
                return True, "atr_trail_stop"
        else:
            if bar.close > self._peak_close + trail:
                self._in_trade = False
                return True, "atr_trail_stop"

        # 3. Volatility stop (ATR has doubled from entry)
        if features.atr > self._entry_atr * self._cfg.vol_stop_mult:
            self._in_trade = False
            return True, "vol_stop"

        # 4. Max-loss stop (spread worthless)
        if current_value <= 0.05 * entry_debit:
            self._in_trade = False
            return True, "max_loss"

        return False, ""

    # ── Private helpers ────────────────────────────────────────────────────────

    def _in_compression(self, f: FeatureVector) -> bool:
        t = self._cfg
        return (
            f.atr_pct     <= t.atr_pct_threshold     and
            f.rv_pct      <= t.rv_pct_threshold       and
            f.entropy_pct <= t.entropy_pct_threshold  and
            f.range_pct   <= t.range_pct_threshold
        )

    def _detect_breakout(
        self, bar: MarketBar, comp_high: float, comp_low: float, med_vol: float,
    ) -> SignalDirection:
        """
        Volume spike gate:
          - Real volume (futures/ETFs): require bar.volume > 2× median.
          - Synthetic/zero volume (index data): require bar range > compression
            range width — a bar that widens beyond the compression box signals
            genuine expansion energy without needing real volume.
        """
        if med_vol > 0 and bar.volume > 0:
            vol_spike = bar.volume > self._cfg.volume_spike_mult * med_vol
        else:
            # No reliable volume — use range expansion vs compression width as proxy
            comp_width = comp_high - comp_low
            bar_range  = bar.high - bar.low
            vol_spike  = bar_range > max(comp_width * 0.5, 1.0)  # bar expands ≥ 50% of comp box

        if bar.close > comp_high and vol_spike:
            return SignalDirection.LONG
        if bar.close < comp_low  and vol_spike:
            return SignalDirection.SHORT
        return SignalDirection.NEUTRAL

    def _confidence(self, f: FeatureVector) -> float:
        """Higher hurst + cleaner compression = higher confidence."""
        hurst_bonus = max(0.0, f.hurst - 0.5) * 2.0   # 0-1
        comp_depth  = 1.0 - max(f.atr_pct, f.rv_pct, f.entropy_pct, f.range_pct)
        return float(min(1.0, 0.5 + 0.25 * hurst_bonus + 0.25 * comp_depth))
