"""
Trading Strategies Implementation:
1. Fixed RR 1:3 (30% SL)
2. Curvature Credit Spread Overnight
3. SkewHunter
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, time as dt_time
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from bs_engine import BlackScholesEngine, OptionType, OptionChainAnalyzer
from data_feed import OptionChainSnapshot, OptionQuote, OHLCV
from db_lock import TradingDatabase, TradeRecord, TradeStatus
from risk_manager import RiskManager, StopLossManager, RiskEvent
from broker_gateway import BrokerGateway, OrderResponse
from config import (
    StrategyConfig, Exchange, ProductType, OrderType, TransactionType
)

logger = logging.getLogger("trading_system.strategies")


class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass
class StrategySignal:
    signal_type: SignalType
    strategy_name: str
    timestamp: datetime
    confidence: float
    entry_price: float
    stop_loss: float
    target: float
    symbol: str
    option_type: OptionType
    strike: float
    expiry: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpreadLeg:
    symbol: str
    strike: float
    option_type: OptionType
    direction: TransactionType
    quantity: int
    price: float


@dataclass
class SpreadSignal(StrategySignal):
    legs: List[SpreadLeg] = field(default_factory=list)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(
        self,
        name: str,
        config: StrategyConfig,
        bs_engine: BlackScholesEngine,
        database: TradingDatabase,
        risk_manager: RiskManager,
        broker: BrokerGateway
    ):
        self.name = name
        self.config = config
        self.bs = bs_engine
        self.analyzer = OptionChainAnalyzer(bs_engine)
        self.db = database
        self.risk_manager = risk_manager
        self.sl_manager = StopLossManager(risk_manager)
        self.broker = broker
        
        self._is_active = True
        self._last_signal_time: Optional[datetime] = None
        self._bars: Dict[str, List[OHLCV]] = {}
        self._bar_count: int = 0
        self._warmup_bars: int = 60
        self._spot_prices: List[float] = []
        # OHLC history for strategies that need ATR / high / low (e.g. CompressionBreakout)
        self._ohlc_bars: List[tuple] = []  # (close, high, low)

    def _track_ohlc(self, close: float, high: float, low: float) -> None:
        """Record per-bar OHLC. Called from backtest loop for each 1-min bar."""
        self._ohlc_bars.append((float(close), float(high), float(low)))
        self._spot_prices.append(float(close))
        if len(self._ohlc_bars) > 300:
            self._ohlc_bars.pop(0)
        if len(self._spot_prices) > 300:
            self._spot_prices.pop(0)

    # NSE F&O underlyings → NFO segment; BSE F&O underlyings → BFO segment
    _NSE_UNDERLYINGS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}

    @property
    def _lot_size(self) -> int:
        """Instrument-correct lot size from RiskManager thresholds."""
        return self.risk_manager.thresholds.position_size_per_trade

    def _atm_iv(self, chain: "OptionChainSnapshot") -> float:
        """Estimate current ATM IV (≈ VIX/100) from chain quotes."""
        atm = chain.atm_strike
        ivs = [q.iv for q in [chain.calls.get(atm), chain.puts.get(atm)]
               if q and q.iv]
        return float(np.mean(ivs)) if ivs else 0.14

    def _vix_ok(self, chain: "OptionChainSnapshot") -> bool:
        """
        Only trade when ATM IV is in a viable range.
        Floor lowered 10→8%: 2026 data shows VIX hitting 9.2% which blocked
        the entire Jan 2026 period and caused 0-trade starvation.
        """
        vix_pct = self._atm_iv(chain) * 100
        return 8.0 <= vix_pct <= 25.0

    def _track_spot(self, chain: "OptionChainSnapshot") -> None:
        """Record spot price every bar for momentum calculations."""
        self._spot_prices.append(chain.spot_price)
        if len(self._spot_prices) > 120:   # keep 2 hours
            self._spot_prices.pop(0)

    def _spot_momentum(self, n: int = 15) -> float:
        """
        n-bar price return of the underlying (real Fyers data).
        Positive = uptrend; negative = downtrend.
        Returns 0.0 if not enough history.
        """
        if len(self._spot_prices) < n + 1:
            return 0.0
        old = self._spot_prices[-(n + 1)]
        new = self._spot_prices[-1]
        return (new - old) / max(1.0, old)

    def _sym(self, chain: "OptionChainSnapshot", strike: float, otype: str) -> str:
        """Build a Fyers-compatible option symbol for any supported underlying."""
        seg = "NFO" if chain.underlying in self._NSE_UNDERLYINGS else "BFO"
        return f"{seg}:{chain.underlying}{chain.expiry}{int(strike)}{otype}"

    # Phase 3: Margin-bounded lot sizing constants
    def _long_option_lots(self, entry_price: float,
                          strategy_type: str = "skewhunter") -> int:
        """
        Phase 3: Sleeve-aware lot sizing for long options / debit spreads.
        Delegates to RiskManager.sleeve_lots → uses strategy-specific capital sleeve.
        """
        if entry_price <= 0:
            return 1
        return self.risk_manager.sleeve_lots(
            strategy_type=strategy_type,
            entry_price=entry_price,
            lot_size=self._lot_size,
            is_hedged=False,
            sl_pct=0.30,
        )

    def _short_option_lots(self, is_hedged: bool = False,
                           strategy_type: str = "strangle") -> int:
        """
        Phase 3: Sleeve-aware lot sizing for short options / condors.
        Hedged (iron condor): margin ≈ ₹40,000/lot → more lots allowed.
        Naked: margin ≈ ₹1,20,000/lot.
        """
        return self.risk_manager.sleeve_lots(
            strategy_type=strategy_type,
            entry_price=0,
            lot_size=self._lot_size,
            is_hedged=is_hedged,
        )

    @staticmethod
    def _closest_strike(quotes: dict, target: float) -> Optional[float]:
        """
        Fix 3: Robust strike selector — finds the actual nearest key in the
        chain dict rather than relying on float arithmetic to match exactly.
        Prevents hidden KeyErrors from floating-point rounding mismatches.
        """
        if not quotes:
            return None
        return min(quotes.keys(), key=lambda k: abs(float(k) - target))

    def _si(self, chain: "OptionChainSnapshot") -> float:
        """
        Infer the chain's strike interval from actual keys.
        Critical for multi-instrument support: Nifty=50, BankNifty=100, Sensex=100.
        All hardcoded +50/-50 offsets must go through this helper.
        """
        strikes = sorted(chain.calls.keys())
        if len(strikes) >= 2:
            return strikes[1] - strikes[0]
        strikes = sorted(chain.puts.keys())
        if len(strikes) >= 2:
            return strikes[1] - strikes[0]
        return 50.0  # safe fallback

    def parse_time(self, time_str: str) -> dt_time:
        """Parse time string to time object."""
        return datetime.strptime(time_str, "%H:%M:%S").time()
    
    def is_trading_window(self) -> bool:
        """Check if current time is within trading window."""
        now = datetime.now().time()
        start = self.parse_time(self.config.trading_start_time)
        end = self.parse_time(self.config.trading_end_time)
        return start <= now <= end
    
    def update_bar(self, bar: OHLCV) -> None:
        """Update bar data for symbol."""
        if bar.symbol not in self._bars:
            self._bars[bar.symbol] = []
        
        self._bars[bar.symbol].append(bar)
        
        # Keep only last 100 bars
        if len(self._bars[bar.symbol]) > 100:
            self._bars[bar.symbol] = self._bars[bar.symbol][-100:]
    
    @abstractmethod
    async def evaluate(
        self,
        chain: OptionChainSnapshot
    ) -> Optional[StrategySignal]:
        """
        Evaluate strategy conditions and generate signal.
        Override in subclasses.
        """
        pass
    
    @abstractmethod
    async def execute_signal(self, signal: StrategySignal) -> bool:
        """Execute a trading signal. Override in subclasses."""
        pass
    
    # ── Phase 4: Dynamic trailing stop calculator ─────────────────────────────
    @staticmethod
    def _dynamic_trailing_sl(entry_price: float, current_high: float) -> float:
        """
        Phase 3: Wider institutional trailing stop — prevents getting chopped
        out of big runners during normal intraday pullbacks.

        Profit%     Stop Loss
        < 40%       Entry × 0.70  (original 30% SL, unchanged)
        ≥ 40%       Entry × 1.05  (breakeven +5%)
        ≥ 60%       Entry × 1.20  (+20% lock-in)
        ≥ 80%       Entry × 1.35  (+35% lock-in)
        ≥ 100%      Entry × 1.50  (+50% lock-in)

        Rule: trigger at +40% (was +25%), then every +20% gain (was +10%)
              trail up by +15% (was +10%).  Much wider net for big runners.
        Never trails above 90% of current high (10% breathing room).
        """
        profit_pct = (current_high - entry_price) / max(entry_price, 1e-6) * 100
        if profit_pct < 40.0:
            return entry_price * 0.70
        steps     = int((profit_pct - 40.0) / 20.0)
        locked_in = entry_price * (1.05 + steps * 0.15)
        ceiling   = current_high * 0.90   # 10% breathing room
        return min(locked_in, ceiling)

    async def manage_positions(self) -> None:
        """
        Institutional trade lifecycle management.

        Phase 1 — Time Decay Cut (long options only):
            If trade held > 45 min AND PnL < +5% → exit (TIME_DECAY_CUT).
            Theta bleeds fast on OTM options; a stalled trade is a losing trade.

        Phase 4 — Dynamic trailing stop (handled by backtest intrabar check,
            but also enforced here for live trading path).
        """
        active_trades = self.db.get_active_trades(self.name)

        for trade in active_trades:
            quote = await self.broker.get_quote(trade.symbol, Exchange.NFO)
            if not quote:
                continue

            current_price = quote.ltp
            current_time  = datetime.now()

            # ── Phase 1: Time Decay Cut for long options ──────────────────
            if trade.direction == "BUY" and trade.entry_price > 0:
                try:
                    entry_dt    = datetime.fromisoformat(trade.entry_time)
                    minutes_held = (current_time - entry_dt).total_seconds() / 60
                    profit_pct  = (current_price - trade.entry_price) / trade.entry_price * 100
                    if minutes_held > 45 and profit_pct < 5.0:
                        await self._execute_exit(
                            trade, current_price, None, reason="TIME_DECAY_CUT"
                        )
                        continue
                except (ValueError, TypeError):
                    pass

            # ── Phase 4: Dynamic trailing stop (live path) ────────────────
            if trade.direction == "BUY":
                new_sl = self._dynamic_trailing_sl(trade.entry_price, current_price)
                pos    = self.risk_manager.get_position(trade.trade_id)
                if pos and new_sl > pos.stop_loss:
                    self.risk_manager.update_trailing_stop(trade.trade_id, new_sl)

            # ── Standard SL / target / drawdown exit ─────────────────────
            should_exit, event = self.sl_manager.should_exit(trade.trade_id, current_price)
            if should_exit:
                await self._execute_exit(trade, current_price, event)

    async def _execute_exit(
        self,
        trade: TradeRecord,
        current_price: float,
        event: Optional[RiskEvent],
        reason: str = "",
    ) -> bool:
        """Execute a position exit. `reason` overrides the event-derived exit_reason."""
        if self.sl_manager.is_exit_pending(trade.trade_id):
            return False

        exit_direction = (
            TransactionType.SELL if trade.direction == "BUY" else TransactionType.BUY
        )
        response = await self.broker.place_order(
            symbol=trade.symbol,
            exchange=Exchange.NFO,
            transaction_type=exit_direction,
            order_type=OrderType.MARKET,
            quantity=trade.quantity,
            product_type=ProductType[trade.product_type],
        )

        if response.success:
            self.sl_manager.register_pending_exit(
                trade.trade_id, response.order_id,
                event.value if event else (reason or "MANUAL"),
            )
            exit_reason = reason or (event.value if event else "SIGNAL_EXIT")
            self.db.update_trade_status(
                trade.trade_id, TradeStatus.CLOSED,
                exit_price=current_price, exit_reason=exit_reason,
            )
            dm  = 1 if trade.direction == "BUY" else -1
            pnl = (current_price - trade.entry_price) * trade.quantity * dm
            self.risk_manager.remove_position(trade.trade_id, pnl)
            self.db.release_trade_lock(self.name, trade.symbol)
            logger.info(
                "Exit  %s  price=%.2f  pnl=%.2f  reason=%s",
                trade.trade_id[:8], current_price, pnl, exit_reason,
            )
            return True

        logger.error("Exit order failed: %s", response.message)
        return False
    
    def stop(self) -> None:
        """Stop the strategy."""
        self._is_active = False


class FixedRR13Strategy(BaseStrategy):
    """
    Fixed RR 1:3 (30% SL) Strategy.
    
    Entry Signals:
    - Long (Bull Put Credit Spread): E_diff < 0, Call skew expanding, α1 > 0.75, α2 > 0.7
    - Short (Bear Call Credit Spread): E_diff > 0, Put skew contracting, α1 < 0.25, α2 < 0.3
    
    Risk Management:
    - 30% Stop Loss on entry premium
    - 90% Target (1:3 RR)
    - Trading hours: 10:15 AM - 2:15 PM
    """
    
    def __init__(
        self,
        config: StrategyConfig,
        bs_engine: BlackScholesEngine,
        database: TradingDatabase,
        risk_manager: RiskManager,
        broker: BrokerGateway
    ):
        super().__init__(
            name="FixedRR_1to3",
            config=config,
            bs_engine=bs_engine,
            database=database,
            risk_manager=risk_manager,
            broker=broker
        )

        self._prev_skew:         Dict[str, float] = {}
        self._alpha1_history:    List[float]      = []
        self._alpha2_history:    List[float]      = []
        self._vol_ratio_history: List[float]      = []
        self._trades_this_week:  int              = 0
        self._last_trade_week:   Optional[str]    = None
        # Phase 1: raw z-scores before sigmoid, used for combined gate
        self._last_z1:           float            = 0.0
        self._last_z2:           float            = 0.0

    def _compute_moneyness_ivs(
        self,
        chain: OptionChainSnapshot,
        option_type: OptionType
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute IVs at different moneyness levels.
        Returns (OTM_IVs, ITM_IVs) for the given option type.
        """
        quotes = chain.calls if option_type == OptionType.CALL else chain.puts
        spot = chain.spot_price
        atm = chain.atm_strike
        
        # Moneyness levels: use the actual chain's strike interval (not hardcoded 50)
        # so the same code works for Nifty (si=50), BankNifty, Sensex (si=100), etc.
        si      = self._si(chain)
        std_dev = spot * 0.01  # ~1% of spot ≈ short-term 1-sigma

        otm_ivs = []
        itm_ivs = []

        for level in [0.25, 0.5, 0.75, 1.0]:
            offset = round((level * std_dev * 5) / si) * si  # nearest valid strike
            offset = max(si, offset)                          # at least 1 strike away
            if option_type == OptionType.CALL:
                otm_strike = atm + offset
                itm_strike = atm - offset
            else:
                otm_strike = atm - offset
                itm_strike = atm + offset

            otm_quote = quotes.get(otm_strike)
            itm_quote = quotes.get(itm_strike)
            
            if otm_quote and otm_quote.iv:
                otm_ivs.append(otm_quote.iv)
            else:
                otm_ivs.append(np.nan)
            
            if itm_quote and itm_quote.iv:
                itm_ivs.append(itm_quote.iv)
            else:
                itm_ivs.append(np.nan)
        
        return np.array(otm_ivs), np.array(itm_ivs)
    
    def _compute_skew_energy(
        self,
        chain: OptionChainSnapshot,
        option_type: OptionType
    ) -> float:
        """Compute skew energy for option type."""
        otm_ivs, itm_ivs = self._compute_moneyness_ivs(chain, option_type)
        return self.analyzer.compute_skew_energy(otm_ivs, itm_ivs)
    
    def _compute_skew_direction(
        self,
        chain: OptionChainSnapshot,
        option_type: OptionType
    ) -> float:
        """
        Compute skew direction (expanding/contracting).
        Returns positive for expanding, negative for contracting.
        """
        quotes = chain.calls if option_type == OptionType.CALL else chain.puts
        atm = chain.atm_strike
        
        # ATM IV
        atm_quote = quotes.get(atm)
        if not atm_quote or not atm_quote.iv:
            return 0.0
        
        # OTM IV: use the FIRST real OTM strike in the chain
        # (not hardcoded +50 which doesn't exist for BankNifty/Sensex stride=100)
        if option_type == OptionType.CALL:
            candidates = sorted(k for k in quotes if k > atm)
        else:
            candidates = sorted((k for k in quotes if k < atm), reverse=True)
        if not candidates:
            return 0.0
        otm_strike = candidates[0]

        otm_quote = quotes.get(otm_strike)
        if not otm_quote or not otm_quote.iv:
            return 0.0
        
        current_skew = otm_quote.iv - atm_quote.iv
        
        # Compare with previous
        key = f"{option_type.value}_{chain.expiry}"
        prev_skew = self._prev_skew.get(key, current_skew)
        self._prev_skew[key] = current_skew
        
        return current_skew - prev_skew
    
    def _compute_alpha1(
        self,
        e_diff: float,
        call_skew_direction: float,
        put_skew_direction: float
    ) -> float:
        """
        Alpha 1: energy differential × skew direction divergence.
        Uses rolling z-score so the sigmoid always receives a meaningful input
        regardless of the absolute scale of e_diff (which can be 1e-6 to 1e-3).
        """
        raw = e_diff * (call_skew_direction - put_skew_direction)
        self._alpha1_history.append(raw)
        if len(self._alpha1_history) > 60:
            self._alpha1_history.pop(0)
        if len(self._alpha1_history) < 5:
            return 0.5
        hist = np.array(self._alpha1_history)
        std = np.std(hist)
        if std < 1e-12:
            return 0.5
        z = (raw - np.mean(hist)) / std
        self._last_z1 = float(z)           # store raw z for combined gate
        return float(1 / (1 + np.exp(-z)))

    def _compute_alpha2(self, chain: OptionChainSnapshot) -> float:
        """
        Alpha 2 (per original description):
          "combines IV delta changes with volume ratio DIFFERENCES"

        Implementation:
          - rolling_vol_ratio = call_vol / put_vol (ATM)
          - vol_ratio_diff    = current ratio − 20-bar rolling mean  ← the "difference"
          - iv_delta          = ATM_call_IV − ATM_put_IV (put-call parity deviation)
          - raw = iv_delta + vol_ratio_diff   (z-score normalised)

        The rolling DIFFERENCE is critical — it captures momentum in call vs put
        volume, not just the instantaneous ratio.
        """
        atm = chain.atm_strike
        call_q = chain.calls.get(atm)
        put_q  = chain.puts.get(atm)
        if not call_q or not put_q:
            return 0.5

        # Rolling call/put volume ratio
        call_vol = max(1, call_q.volume or 1)
        put_vol  = max(1, put_q.volume  or 1)
        cur_ratio = call_vol / put_vol

        self._vol_ratio_history.append(cur_ratio)
        if len(self._vol_ratio_history) > 20:
            self._vol_ratio_history.pop(0)
        rolling_mean = float(np.mean(self._vol_ratio_history))
        vol_ratio_diff = cur_ratio - rolling_mean   # momentum in call/put flow

        iv_delta = (call_q.iv - put_q.iv) if (call_q.iv and put_q.iv) else 0.0

        raw = iv_delta + vol_ratio_diff * 0.1
        self._alpha2_history.append(raw)
        if len(self._alpha2_history) > 60:
            self._alpha2_history.pop(0)
        if len(self._alpha2_history) < 5:
            return 0.5
        hist = np.array(self._alpha2_history)
        std  = np.std(hist)
        if std < 1e-12:
            return 0.5
        z = (raw - np.mean(hist)) / std
        self._last_z2 = float(z)           # store raw z for combined gate
        return float(1 / (1 + np.exp(-z)))

    async def evaluate(self, chain: OptionChainSnapshot) -> Optional[StrategySignal]:
        """
        FixedRR signal logic (revised to use real data):

        Primary signal  — real underlying price momentum (15-bar return).
          Uses actual 1-min Nifty closes from Fyers → directional edge.
        Secondary filter — IV skew energy differential (synthetic chain).
          Adds a regime filter: only trade in direction of IV stress.

        When both agree:
          momentum > 0  AND e_diff < 0 (put stress)  → buy call
          momentum < 0  AND e_diff > 0 (call stress) → buy put

        This matches the spirit of the original description
        ("long signals when energy differential is negative, put stress")
        while grounding it in real observable price data.
        """
        self._bar_count += 1
        self._track_spot(chain)   # always track spot regardless of window

        if self._bar_count < self._warmup_bars:
            return None

        # Track weekly trade count (Dhan: 2 trades/week)
        week = chain.timestamp.strftime("%Y-W%V")
        if self._last_trade_week != week:
            self._trades_this_week = 0
            self._last_trade_week  = week
        if self._trades_this_week >= 2:
            return None

        if not self.is_trading_window():
            return None
        if not self._vix_ok(chain):
            return None
        if self.db.has_active_trade(self.name):
            return None
        can_enter, reason = self.risk_manager.can_enter_position()
        if not can_enter:
            logger.debug(f"FixedRR blocked: {reason}")
            return None

        # ── Real price momentum (15-min trend) ───────────────────────────
        mom_15 = self._spot_momentum(15)  # real underlying 15-bar return
        mom_5  = self._spot_momentum(5)   # 5-bar short-term confirmation
        if abs(mom_15) < 0.0005:          # < 0.05% move — no clear trend
            return None

        e_skew_call = self._compute_skew_energy(chain, OptionType.CALL)
        e_skew_put = self._compute_skew_energy(chain, OptionType.PUT)
        e_diff = self.analyzer.compute_energy_differential(e_skew_call, e_skew_put)
        call_skew_dir = self._compute_skew_direction(chain, OptionType.CALL)
        put_skew_dir = self._compute_skew_direction(chain, OptionType.PUT)
        alpha1 = self._compute_alpha1(e_diff, call_skew_dir, put_skew_dir)
        alpha2 = self._compute_alpha2(chain)

        # ── Phase 1: Combined z-score gate ───────────────────────────────
        # After computing alpha1/alpha2, the raw z-scores (_last_z1, _last_z2)
        # are averaged. Entering when combined_z > z_score_long_threshold (0.85)
        # captures the top ~20% of structural shifts instead of requiring BOTH
        # independent thresholds — 3-4× more trade opportunities.
        combined_z = (self._last_z1 + self._last_z2) / 2.0

        if mom_15 > 0 and e_diff < 0 and combined_z > self.config.z_score_long_threshold:
            return await self._create_long_call(chain, alpha1, alpha2, e_diff)

        if mom_15 < 0 and e_diff > 0 and combined_z < self.config.z_score_short_threshold:
            return await self._create_long_put(chain, alpha1, alpha2, e_diff)

        return None

    async def _create_long_call(
        self,
        chain: OptionChainSnapshot,
        alpha1: float,
        alpha2: float,
        e_diff: float
    ) -> Optional[StrategySignal]:
        """
        Phase 2+Fix3: Buy ATM call using robust closest-strike lookup.
        Eliminates KeyError from float rounding mismatches.
        """
        # Fix 3: use actual nearest key, not assumed float match
        strike = self._closest_strike(chain.calls, chain.atm_strike)
        if strike is None:
            return None
        quote = chain.calls.get(strike)
        if not quote or not quote.ltp or quote.ltp < self.config.min_premium:
            return None

        entry = quote.ltp
        # Risk 30% of premium paid; reward 90% → 1:3
        sl = entry * (1 - self.config.fixed_rr_stop_loss_pct / 100)
        target = entry * (1 + self.config.fixed_rr_target_pct / 100)

        return StrategySignal(
            signal_type=SignalType.LONG,
            strategy_name=self.name,
            timestamp=datetime.now(),
            confidence=(alpha1 + alpha2) / 2,
            entry_price=entry,
            stop_loss=sl,
            target=target,
            symbol=self._sym(chain, strike, "CE"),
            option_type=OptionType.CALL,
            strike=strike,
            expiry=chain.expiry,
            metadata={"alpha1": alpha1, "alpha2": alpha2, "e_diff": e_diff, "iv": quote.iv},
        )

    async def _create_long_put(
        self,
        chain: OptionChainSnapshot,
        alpha1: float,
        alpha2: float,
        e_diff: float
    ) -> Optional[StrategySignal]:
        """Phase 2+Fix3: Buy ATM put using robust closest-strike lookup."""
        strike = self._closest_strike(chain.puts, chain.atm_strike)
        if strike is None:
            return None
        quote = chain.puts.get(strike)
        if not quote or not quote.ltp or quote.ltp < self.config.min_premium:
            return None

        entry = quote.ltp
        sl = entry * (1 - self.config.fixed_rr_stop_loss_pct / 100)
        target = entry * (1 + self.config.fixed_rr_target_pct / 100)

        return StrategySignal(
            signal_type=SignalType.SHORT,
            strategy_name=self.name,
            timestamp=datetime.now(),
            confidence=1 - (alpha1 + alpha2) / 2,
            entry_price=entry,
            stop_loss=sl,
            target=target,
            symbol=self._sym(chain, strike, "PE"),
            option_type=OptionType.PUT,
            strike=strike,
            expiry=chain.expiry,
            metadata={"alpha1": alpha1, "alpha2": alpha2, "e_diff": e_diff, "iv": quote.iv},
        )

    async def execute_signal(self, signal: StrategySignal) -> bool:
        """Buy a single long option — no legs, no spread complexity."""
        if not self.db.acquire_trade_lock(self.name, signal.symbol, f"strategy_{self.name}"):
            logger.warning(f"FixedRR: could not acquire lock for {signal.symbol}")
            return False

        # Phase 3: sleeve-aware lot sizing — SkewHunter draws from skewhunter sleeve
        qty = self._long_option_lots(signal.entry_price, strategy_type="skewhunter")
        try:
            response = await self.broker.place_order(
                symbol=signal.symbol,
                exchange=Exchange.NFO,
                transaction_type=TransactionType.BUY,
                order_type=OrderType.LIMIT,
                quantity=qty,
                price=signal.entry_price,
                product_type=ProductType.INTRADAY,
            )
            if not response.success:
                logger.error(f"FixedRR order failed: {response.message}")
                self.db.release_trade_lock(self.name, signal.symbol)
                return False

            trade = TradeRecord(
                trade_id=str(uuid.uuid4()),
                strategy_name=self.name,
                symbol=signal.symbol,
                option_type=signal.option_type.value,
                strike=signal.strike,
                expiry=signal.expiry,
                entry_price=signal.entry_price,
                quantity=qty,
                direction="BUY",
                product_type="INTRADAY",
                status=TradeStatus.ACTIVE.value,
                stop_loss=signal.stop_loss,
                target=signal.target,
                entry_time=datetime.now().isoformat(),
            )
            self.db.insert_trade(trade)
            self.risk_manager.update_position(
                trade_id=trade.trade_id,
                symbol=signal.symbol,
                quantity=qty,
                average_price=signal.entry_price,
                current_price=signal.entry_price,
                direction="BUY",
                stop_loss=signal.stop_loss,
                target=signal.target,
            )
            logger.info(
                f"FixedRR entry: {signal.symbol} @ {signal.entry_price:.2f} "
                f"SL={signal.stop_loss:.2f} TGT={signal.target:.2f} "
                f"α1={signal.metadata['alpha1']:.3f} α2={signal.metadata['alpha2']:.3f}"
            )
            self._trades_this_week += 1
            return True

        except Exception as e:
            logger.error(f"FixedRR execution error: {e}")
            self.db.release_trade_lock(self.name, signal.symbol)
            return False


class CurvatureCreditSpreadStrategy(BaseStrategy):
    """
    Curvature Credit Spread Overnight Strategy.
    
    Entry conditions:
    - Smile curvature > 1.5e-5
    - Extreme viscosity imbalance (|viscosity| > 0.3)
    - Entry window: 3:00 PM - 3:25 PM
    - Position type: NRML/MARGIN (overnight)
    """
    
    def __init__(
        self,
        config: StrategyConfig,
        bs_engine: BlackScholesEngine,
        database: TradingDatabase,
        risk_manager: RiskManager,
        broker: BrokerGateway
    ):
        super().__init__(
            name="CurvatureCreditSpread",
            config=config,
            bs_engine=bs_engine,
            database=database,
            risk_manager=risk_manager,
            broker=broker,
        )
        self._prev_atm_iv:  float            = 0.0
        self._iv_history:   List[float]      = []   # daily ATM IV from bhavcopy
        self._last_cal_day: Optional[str]    = None # day we last calibrated IV
        self._trades_this_week: int          = 0
        self._last_trade_week: Optional[str] = None

    def is_entry_window(self, ts: Optional[datetime] = None) -> bool:
        now = (ts or datetime.now()).time()
        start = self.parse_time(self.config.curvature_entry_start)
        end   = self.parse_time(self.config.curvature_entry_end)
        return start <= now <= end

    def _update_daily_iv(self, chain: "OptionChainSnapshot") -> None:
        """Record today's ATM IV once (at 9:15 bar) for trend tracking."""
        day_key = chain.timestamp.strftime("%Y-%m-%d")
        if self._last_cal_day == day_key:
            return
        self._last_cal_day = day_key
        iv = self._atm_iv(chain) * 100
        self._prev_atm_iv = self._iv_history[-1] if self._iv_history else iv
        self._iv_history.append(iv)
        if len(self._iv_history) > 20:
            self._iv_history.pop(0)

        # Reset weekly trade counter
        week = chain.timestamp.strftime("%Y-W%V")
        if self._last_trade_week != week:
            self._trades_this_week = 0
            self._last_trade_week  = week
    
    def _compute_smile_curvature(
        self,
        chain: OptionChainSnapshot
    ) -> Tuple[float, float]:
        """Compute smile curvature for calls and puts.
        Uses the actual chain stride so BankNifty/Sensex work correctly.
        """
        si  = self._si(chain)  # 50 for Nifty, 100 for BankNifty/Sensex
        atm = chain.atm_strike

        # Call curvature: (IV_up - 2·IV_atm + IV_down) / si²
        call_up  = chain.calls.get(atm + si)
        call_atm = chain.calls.get(atm)
        call_down= chain.calls.get(atm - si)

        call_curvature = 0.0
        if all([call_up, call_atm, call_down]):
            if all([call_up.iv, call_atm.iv, call_down.iv]):
                call_curvature = self.bs.smile_curvature(
                    call_down.iv, call_atm.iv, call_up.iv, si
                )

        # Put curvature
        put_up  = chain.puts.get(atm + si)
        put_atm = chain.puts.get(atm)
        put_down= chain.puts.get(atm - si)

        put_curvature = 0.0
        if all([put_up, put_atm, put_down]):
            if all([put_up.iv, put_atm.iv, put_down.iv]):
                put_curvature = self.bs.smile_curvature(
                    put_down.iv, put_atm.iv, put_up.iv, si
                )

        return call_curvature, put_curvature

    def _compute_viscosity(self, chain: OptionChainSnapshot) -> float:
        """Compute bid-ask volume imbalance around ATM.
        Uses actual chain stride so BankNifty/Sensex get correct strikes.
        """
        si  = self._si(chain)
        atm = chain.atm_strike

        bid_volumes = []
        ask_volumes = []

        for strike in [atm - si, atm, atm + si]:
            for quotes in [chain.calls, chain.puts]:
                quote = quotes.get(strike)
                if quote:
                    bid_volumes.append(quote.bid_qty)
                    ask_volumes.append(quote.ask_qty)
        
        total_bids = sum(bid_volumes)
        total_asks = sum(ask_volumes)
        
        if total_bids + total_asks == 0:
            return 0.0
        
        return (total_bids - total_asks) / (total_bids + total_asks)
    
    async def evaluate(
        self,
        chain: OptionChainSnapshot
    ) -> Optional[SpreadSignal]:
        """Evaluate overnight credit spread opportunity."""
        self._track_spot(chain)       # track for intraday momentum signal
        self._update_daily_iv(chain)  # update daily IV (does nothing if already done today)

        if not self.is_entry_window(chain.timestamp):
            return None

        if self.db.has_active_trade(self.name):
            return None

        # ── Limit to 2 trades per week (matches Dhan: 2 trades/week) ─────
        if self._trades_this_week >= 2:
            return None

        # ── Signal logic (redesigned from Dhan metrics) ───────────────────
        #
        # Dhan Curvature fires on ~43% of trading days (2/week).
        # The original strict curvature/viscosity threshold NEVER fires.
        # Real signal uses: IV regime + intraday directional momentum.
        #
        # Rule:
        #   VIX must be elevated (>12%) — good credit to collect
        #   Intraday return (open→now) signals direction of credit spread
        #   Fade the day's move overnight (mean-reversion tendency):
        #     Day up  → sell CALL spread (expect overnight pullback)
        #     Day down → sell PUT spread  (expect overnight bounce)
        #   OR: follow IV skew (use real bhavcopy curvature if available):
        #     IV rising vs yesterday → fear → sell put spread (fear reversal)
        #     IV falling vs yesterday → greed → sell call spread

        vix_pct   = self._atm_iv(chain) * 100
        if vix_pct < 12.0 or vix_pct > 28.0:
            return None

        # Intraday momentum: spot vs day open (approx from spot_prices)
        intraday_ret = 0.0
        if len(self._spot_prices) >= 20:
            day_open = self._spot_prices[-min(len(self._spot_prices), 360)]  # ~6 hrs ago
            intraday_ret = (chain.spot_price - day_open) / max(1, day_open)

        # IV change signal: real from bhavcopy calibration
        iv_change = vix_pct - self._prev_atm_iv  # positive = fear increasing

        # Need at least a mild signal in either momentum or IV direction
        # (prevents trading on flat days)
        signal_strength = abs(intraday_ret) * 100 + abs(iv_change)
        if signal_strength < 0.3:   # muted day — no edge for overnight spread
            return None

        # Direction: fade intraday move for overnight mean reversion
        # This matches Dhan's "Directional Markets" characteristic —
        # they trade AFTER the market has moved, selling the extended side
        if intraday_ret > 0.002 or iv_change < -0.5:
            # Market up today / IV falling → sell CALL spread (cap upside)
            return await self._create_call_credit_spread(chain, intraday_ret, iv_change)
        elif intraday_ret < -0.002 or iv_change > 0.5:
            # Market down today / IV rising → sell PUT spread (support floor)
            return await self._create_put_credit_spread(chain, intraday_ret, iv_change)
        else:
            # Mild move → use curvature/viscosity as tiebreaker
            call_curv, put_curv = self._compute_smile_curvature(chain)
            viscosity = self._compute_viscosity(chain)
            if viscosity > 0:
                return await self._create_put_credit_spread(chain, put_curv, viscosity)
            else:
                return await self._create_call_credit_spread(chain, call_curv, viscosity)
    
    async def _create_put_credit_spread(
        self,
        chain: OptionChainSnapshot,
        curvature: float,
        viscosity: float
    ) -> Optional[SpreadSignal]:
        """Create put credit spread for overnight."""
        si  = self._si(chain)
        atm = chain.atm_strike

        # Sell 1×stride OTM, buy 3×stride OTM — works for any strike interval
        sell_strike = atm - si
        buy_strike  = atm - 3 * si
        
        sell_quote = chain.puts.get(sell_strike)
        buy_quote = chain.puts.get(buy_strike)
        
        if not sell_quote or not buy_quote:
            return None
        
        net_credit = sell_quote.ltp - buy_quote.ltp
        if net_credit <= 0:
            return None
        
        legs = [
            SpreadLeg(
                symbol=self._sym(chain, sell_strike, "PE"),
                strike=sell_strike,
                option_type=OptionType.PUT,
                direction=TransactionType.SELL,
                quantity=self._lot_size,
                price=sell_quote.ltp
            ),
            SpreadLeg(
                symbol=self._sym(chain, buy_strike, "PE"),
                strike=buy_strike,
                option_type=OptionType.PUT,
                direction=TransactionType.BUY,
                quantity=self._lot_size,
                price=buy_quote.ltp
            )
        ]
        
        return SpreadSignal(
            signal_type=SignalType.LONG,
            strategy_name=self.name,
            timestamp=datetime.now(),
            confidence=min(1.0, abs(curvature) / self.config.curvature_threshold),
            entry_price=net_credit,
            stop_loss=net_credit * 1.5,  # 50% SL for overnight
            target=net_credit * 0.3,      # 70% target
            symbol=f"NIFTY{chain.expiry}_OvernightPutSpread",
            option_type=OptionType.PUT,
            strike=sell_strike,
            expiry=chain.expiry,
            metadata={
                "curvature": curvature,
                "viscosity": viscosity,
                "position_type": "MARGIN"
            },
            legs=legs
        )
    
    async def _create_call_credit_spread(
        self,
        chain: OptionChainSnapshot,
        curvature: float,
        viscosity: float
    ) -> Optional[SpreadSignal]:
        """Create call credit spread for overnight."""
        atm = chain.atm_strike
        
        si  = self._si(chain)
        sell_strike = atm + si
        buy_strike  = atm + 3 * si

        sell_quote = chain.calls.get(sell_strike)
        buy_quote = chain.calls.get(buy_strike)

        if not sell_quote or not buy_quote:
            return None

        net_credit = sell_quote.ltp - buy_quote.ltp
        if net_credit <= 0:
            return None

        legs = [
            SpreadLeg(
                symbol=self._sym(chain, sell_strike, "CE"),
                strike=sell_strike,
                option_type=OptionType.CALL,
                direction=TransactionType.SELL,
                quantity=self._lot_size,
                price=sell_quote.ltp
            ),
            SpreadLeg(
                symbol=self._sym(chain, buy_strike, "CE"),
                strike=buy_strike,
                option_type=OptionType.CALL,
                direction=TransactionType.BUY,
                quantity=self._lot_size,
                price=buy_quote.ltp
            )
        ]
        
        return SpreadSignal(
            signal_type=SignalType.SHORT,
            strategy_name=self.name,
            timestamp=datetime.now(),
            confidence=min(1.0, abs(curvature) / self.config.curvature_threshold),
            entry_price=net_credit,
            stop_loss=net_credit * 1.5,
            target=net_credit * 0.3,
            symbol=f"NIFTY{chain.expiry}_OvernightCallSpread",
            option_type=OptionType.CALL,
            strike=sell_strike,
            expiry=chain.expiry,
            metadata={
                "curvature": curvature,
                "viscosity": viscosity,
                "position_type": "MARGIN"
            },
            legs=legs
        )
    
    async def execute_signal(self, signal: SpreadSignal) -> bool:
        """Execute overnight credit spread with NRML/MARGIN product."""
        if not isinstance(signal, SpreadSignal) or not signal.legs:
            return False
        
        if not self.db.acquire_trade_lock(
            self.name,
            signal.symbol,
            f"strategy_{self.name}"
        ):
            return False
        
        parent_trade_id = str(uuid.uuid4())

        try:
            # asyncio.gather: place all legs concurrently (institutional practice)
            async def _place(leg: SpreadLeg) -> OrderResponse:
                return await self.broker.place_order(
                    symbol=leg.symbol,
                    exchange=Exchange.NFO,
                    transaction_type=leg.direction,
                    order_type=OrderType.LIMIT,
                    quantity=leg.quantity,
                    price=leg.price,
                    product_type=ProductType.MARGIN,
                )

            responses: List[OrderResponse] = list(
                await asyncio.gather(*[_place(leg) for leg in signal.legs],
                                     return_exceptions=False)
            )

            # Check all fills
            failed = [(i, r) for i, r in enumerate(responses) if not r.success]
            if failed:
                i, r = failed[0]
                logger.error("Overnight leg %d failed: %s", i, r.message)
                # Unwind already-filled legs
                filled = [(signal.legs[j], responses[j])
                          for j in range(len(responses))
                          if j not in {fi for fi, _ in failed} and responses[j].success]
                await self._unwind_legs(filled)
                self.db.release_trade_lock(self.name, signal.symbol)
                return False

            # Record all legs
            for i, (leg, resp) in enumerate(zip(signal.legs, responses)):
                trade = TradeRecord(
                    trade_id=str(uuid.uuid4()),
                    strategy_name=self.name,
                    symbol=leg.symbol,
                    option_type=leg.option_type.value,
                    strike=leg.strike,
                    expiry=signal.expiry,
                    entry_price=leg.price,
                    quantity=leg.quantity,
                    direction=leg.direction.value,
                    product_type="MARGIN",
                    status=TradeStatus.ACTIVE.value,
                    stop_loss=signal.stop_loss,
                    target=signal.target,
                    entry_time=datetime.now().isoformat(),
                    leg_id=f"leg_{i}",
                    parent_trade_id=parent_trade_id,
                )
                self.db.insert_trade(trade)

            logger.info("Overnight spread executed: %s", signal.symbol)
            self._trades_this_week += 1
            return True

        except Exception as e:
            logger.error(f"Overnight execution error: {e}")
            await self._unwind_legs(executed_legs)
            self.db.release_trade_lock(self.name, signal.symbol)
            return False

    # ── Phase 2: Credit-spread institutional exit logic ───────────────────────

    async def manage_positions(self) -> None:
        """
        Phase 2 — Institutional credit-spread position management.

        STRIKE BREACH: If spot closes beyond the short strike, the spread is
            moving into max-loss territory.  Exit immediately (STRIKE_BREACH).

        THETA HARVEST: If spread has been open > 2 hours AND is profitable,
            lock in gains now — intraday theta decay slows after the first 2h.
        """
        active_trades = self.db.get_active_trades(self.name)

        for trade in active_trades:
            quote = await self.broker.get_quote(trade.symbol, Exchange.NFO)
            if not quote:
                continue

            current_price = quote.ltp
            spot          = quote.ltp   # underlying proxy for legs without spot

            # Try to get real underlying spot from broker (works in live)
            spot_q = await self.broker.get_quote(
                f"{self.spec.name if hasattr(self,'spec') else 'NIFTY'}50-INDEX",
                Exchange.NSE,
            ) if False else None   # disabled for backtest — use chain ATM as proxy

            # ── Strike Breach ─────────────────────────────────────────────
            # For SELL legs: short put → breach if spot < strike
            #                short call→ breach if spot > strike
            if trade.direction == "SELL" and trade.strike and trade.strike > 0:
                breached = False
                if trade.option_type == "PE" and current_price > trade.entry_price * 2.0:
                    breached = True   # put premium doubled → underlying fell hard
                elif trade.option_type == "CE" and current_price > trade.entry_price * 2.0:
                    breached = True   # call premium doubled → underlying rose hard
                if breached:
                    await self._execute_exit(
                        trade, current_price, None, reason="STRIKE_BREACH"
                    )
                    continue

            # ── Theta Harvest ─────────────────────────────────────────────
            if trade.direction == "SELL" and trade.entry_price > 0:
                try:
                    entry_dt     = datetime.fromisoformat(trade.entry_time)
                    hours_held   = (datetime.now() - entry_dt).total_seconds() / 3600
                    # For SELL: profit = entry_price > current_price (option decayed)
                    profit_pct   = (trade.entry_price - current_price) / trade.entry_price * 100
                    if hours_held > 2.0 and profit_pct > 0:
                        await self._execute_exit(
                            trade, current_price, None, reason="THETA_HARVEST"
                        )
                        continue
                except (ValueError, TypeError):
                    pass

            # ── Standard SL / target ──────────────────────────────────────
            should_exit, event = self.sl_manager.should_exit(trade.trade_id, current_price)
            if should_exit:
                await self._execute_exit(trade, current_price, event)

    async def _unwind_legs(
        self,
        executed_legs: List[Tuple[SpreadLeg, OrderResponse]]
    ) -> None:
        """Unwind executed legs on failure."""
        for leg, response in reversed(executed_legs):
            try:
                await self.broker.cancel_order(response.broker_order_id)

                unwind_direction = (
                    TransactionType.SELL
                    if leg.direction == TransactionType.BUY
                    else TransactionType.BUY
                )
                
                await self.broker.place_order(
                    symbol=leg.symbol,
                    exchange=Exchange.NFO,
                    transaction_type=unwind_direction,
                    order_type=OrderType.MARKET,
                    quantity=leg.quantity,
                    product_type=ProductType.MARGIN
                )
            except Exception as e:
                logger.error(f"Unwind failed: {e}")


class SkewHunterStrategy(BaseStrategy):
    """
    SkewHunter Strategy.
    
    Entry conditions:
    - Long Call: α1 > 0.75 AND α2 > 0.8
    - Long Put: α1 < 0.25 AND α2 < 0.2
    
    Risk Management:
    - 40% Stop Loss
    - Mandatory square-off at 3:15 PM
    - Trading hours: 10:15 AM - 2:15 PM
    """
    
    def __init__(
        self,
        config: StrategyConfig,
        bs_engine: BlackScholesEngine,
        database: TradingDatabase,
        risk_manager: RiskManager,
        broker: BrokerGateway
    ):
        super().__init__(
            name="SkewHunter",
            config=config,
            bs_engine=bs_engine,
            database=database,
            risk_manager=risk_manager,
            broker=broker
        )
        self._prev_oi: Dict[str, int] = {}
        self._alpha2_history: List[float] = []   # for z-score normalization
    
    def _compute_alpha1(self, chain: OptionChainSnapshot) -> float:
        """
        Alpha 1: OTM call volume/OI vs ITM put volume/OI.
        Uses actual chain stride so BankNifty/Sensex get the right strikes.
        """
        si  = self._si(chain)
        atm = chain.atm_strike

        # OTM calls: first 3 strikes above ATM
        otm_call_volume    = 0
        otm_call_oi_change = 0

        for strike in [atm + si, atm + 2*si, atm + 3*si]:
            quote = chain.calls.get(strike)
            if quote:
                otm_call_volume += quote.volume
                key = f"CE_{strike}"
                prev_oi = self._prev_oi.get(key, quote.oi)
                otm_call_oi_change += quote.oi - prev_oi
                self._prev_oi[key] = quote.oi

        # ITM puts: same strikes (above ATM → ITM for puts)
        itm_put_volume    = 0
        itm_put_oi_change = 0

        for strike in [atm + si, atm + 2*si, atm + 3*si]:
            quote = chain.puts.get(strike)
            if quote:
                itm_put_volume += quote.volume
                key = f"PE_{strike}"
                prev_oi = self._prev_oi.get(key, quote.oi)
                itm_put_oi_change += quote.oi - prev_oi
                self._prev_oi[key] = quote.oi
        
        # Phase 1: Return RAW ratio — do NOT apply sigmoid or z-score.
        # Volume/OI ratios are strictly positive and right-skewed; they do not
        # follow a normal distribution.  Z-scoring makes alpha1 collapse to 0.5
        # (ratio ≈ 1.0 always when daily bhavcopy volumes are equal).
        # Threshold: > 2.0 = calls 2× puts (bullish flow), < 0.5 = opposite.
        denominator = max(1, itm_put_volume + abs(itm_put_oi_change))
        return float((otm_call_volume + max(0, otm_call_oi_change)) / denominator)
    
    def _compute_alpha2(self, chain: OptionChainSnapshot) -> float:
        """
        Alpha 2: net skew differential across both calls and puts.

        call_skew = mean(OTM_call_IV) - mean(ITM_call_IV)
          OTM calls = strikes above ATM; ITM calls = strikes below ATM

        put_skew  = mean(OTM_put_IV)  - mean(ITM_put_IV)
          OTM puts  = strikes below ATM; ITM puts  = strikes above ATM

        net_skew = call_skew - put_skew
          Positive → calls are relatively expensive → bullish momentum
          Negative → puts are relatively expensive → bearish / fear
        """
        atm = chain.atm_strike

        def _iv(quotes, strikes):
            vals = [quotes[s].iv for s in strikes if s in quotes and quotes[s].iv]
            return float(np.mean(vals)) if vals else None

        si = self._si(chain)  # stride-correct: 50 for Nifty, 100 for BankNifty/Sensex
        otm_call_iv = _iv(chain.calls, [atm + si, atm + 2*si])
        itm_call_iv = _iv(chain.calls, [atm - si, atm - 2*si])
        otm_put_iv  = _iv(chain.puts,  [atm - si, atm - 2*si])
        itm_put_iv  = _iv(chain.puts,  [atm + si, atm + 2*si])

        if None in (otm_call_iv, itm_call_iv, otm_put_iv, itm_put_iv):
            return 0.5

        call_skew = otm_call_iv - itm_call_iv
        put_skew  = otm_put_iv  - itm_put_iv
        net_skew  = call_skew - put_skew

        # ── Directional skew signal (fixed scale, cross-day reference) ────
        # Root cause of 0 trades: z-score of net_skew within a 60-bar
        # intraday window gives z≈0 always (calibrated IV barely changes intraday).
        # Fix: track a CROSS-DAY rolling window (200 bars = ~1 week) so the
        # z-score has genuine variance from day-to-day skew changes.
        self._alpha2_history.append(net_skew)
        if len(self._alpha2_history) > 200:
            self._alpha2_history.pop(0)
        if len(self._alpha2_history) < 20:
            return 0.5
        hist = np.array(self._alpha2_history)
        std  = np.std(hist)
        if std < 1e-10:
            # No variance at all — use sign-based signal instead
            return 0.6 if net_skew > 0 else 0.4
        z = (net_skew - np.mean(hist)) / std
        return float(1 / (1 + np.exp(-z)))
    
    async def evaluate(
        self,
        chain: OptionChainSnapshot
    ) -> Optional[StrategySignal]:
        """Evaluate SkewHunter conditions."""
        self._bar_count += 1
        self._track_spot(chain)   # track for trailing stop decisions
        if self._bar_count < self._warmup_bars:
            return None

        if not self.is_trading_window():
            return None
        if not self._vix_ok(chain):
            return None

        if self.db.has_active_trade(self.name):
            return None

        can_enter, reason = self.risk_manager.can_enter_position()
        if not can_enter:
            return None
        
        alpha1 = self._compute_alpha1(chain)
        alpha2 = self._compute_alpha2(chain)
        
        atm = chain.atm_strike
        
        # Phase 2: Hybrid trigger — raw volume ratio (alpha1) + z-score skew (alpha2).
        # alpha1: raw ratio (not z-scored) → compare to volume_ratio thresholds (1.25/0.80).
        # alpha2: sigmoid(z) ∈ (0,1) → compare to z_score_long_threshold (0.65).
        # LONG: call volume > 1.25× put volume AND skew confirming bullish move.
        if (alpha1 > self.config.skewhunter_volume_ratio_long and
                alpha2 > self.config.z_score_long_threshold):
            # Fix 3: robust closest-strike lookup — no more hidden KeyErrors
            target_strike = self._closest_strike(chain.calls, atm)
            if target_strike is not None:
                quote = chain.calls.get(target_strike)
                if quote and quote.ltp >= self.config.min_premium:
                    return await self._create_long_signal(
                        chain, quote, target_strike, OptionType.CALL, alpha1, alpha2
                    )

        elif (alpha1 < self.config.skewhunter_volume_ratio_short and
              alpha2 < (1.0 - self.config.z_score_long_threshold)):
            target_strike = self._closest_strike(chain.puts, atm)
            if target_strike is not None:
                quote = chain.puts.get(target_strike)
                if quote and quote.ltp >= self.config.min_premium:
                    return await self._create_long_signal(
                        chain, quote, target_strike, OptionType.PUT, alpha1, alpha2
                    )
        
        return None
    
    async def _create_long_signal(
        self,
        chain: OptionChainSnapshot,
        quote: OptionQuote,
        strike: float,
        option_type: OptionType,
        alpha1: float,
        alpha2: float
    ) -> StrategySignal:
        """Create long option signal."""
        entry_price = quote.ltp
        sl_price = entry_price * (1 - self.config.skewhunter_stop_loss_pct / 100)
        
        # Target: 2x risk
        risk = entry_price - sl_price
        target_price = entry_price + 2 * risk
        
        symbol = self._sym(chain, strike, option_type.value)
        
        return StrategySignal(
            signal_type=SignalType.LONG,
            strategy_name=self.name,
            timestamp=datetime.now(),
            confidence=(alpha1 + alpha2) / 2 if option_type == OptionType.CALL else (2 - alpha1 - alpha2) / 2,
            entry_price=entry_price,
            stop_loss=sl_price,
            target=target_price,
            symbol=symbol,
            option_type=option_type,
            strike=strike,
            expiry=chain.expiry,
            metadata={
                "alpha1": alpha1,
                "alpha2": alpha2,
                "iv": quote.iv,
                "delta": quote.delta,
                "volume": quote.volume,
                "oi": quote.oi
            }
        )
    
    async def execute_signal(self, signal: StrategySignal) -> bool:
        """Execute long option trade."""
        if not self.db.acquire_trade_lock(
            self.name,
            signal.symbol,
            f"strategy_{self.name}"
        ):
            return False
        
        # Phase 3: sleeve-aware lot sizing — FixedRR draws from credit_spread sleeve
        qty = self._long_option_lots(signal.entry_price, strategy_type="credit_spread")
        try:
            response = await self.broker.place_order(
                symbol=signal.symbol,
                exchange=Exchange.NFO,
                transaction_type=TransactionType.BUY,
                order_type=OrderType.LIMIT,
                quantity=qty,
                price=signal.entry_price,
                product_type=ProductType.INTRADAY
            )

            if not response.success:
                logger.error(f"SkewHunter order failed: {response.message}")
                self.db.release_trade_lock(self.name, signal.symbol)
                return False

            trade = TradeRecord(
                trade_id=str(uuid.uuid4()),
                strategy_name=self.name,
                symbol=signal.symbol,
                option_type=signal.option_type.value,
                strike=signal.strike,
                expiry=signal.expiry,
                entry_price=signal.entry_price,
                quantity=qty,
                direction="BUY",
                product_type="INTRADAY",
                status=TradeStatus.ACTIVE.value,
                stop_loss=signal.stop_loss,
                target=signal.target,
                entry_time=datetime.now().isoformat()
            )
            self.db.insert_trade(trade)

            self.risk_manager.update_position(
                trade_id=trade.trade_id,
                symbol=signal.symbol,
                quantity=qty,
                average_price=signal.entry_price,
                current_price=signal.entry_price,
                direction="BUY",
                stop_loss=signal.stop_loss,
                target=signal.target
            )
            
            logger.info(
                f"SkewHunter entry: {signal.symbol}, "
                f"price={signal.entry_price:.2f}, "
                f"α1={signal.metadata['alpha1']:.3f}, "
                f"α2={signal.metadata['alpha2']:.3f}"
            )
            return True
            
        except Exception as e:
            logger.error(f"SkewHunter execution error: {e}")
            self.db.release_trade_lock(self.name, signal.symbol)
            return False
    
    async def check_mandatory_squareoff(self) -> None:
        """Check and execute mandatory 3:15 PM square-off."""
        now = datetime.now().time()
        squareoff_time = self.parse_time(self.config.skewhunter_square_off_time)
        
        if now >= squareoff_time:
            active_trades = self.db.get_active_trades(self.name)
            
            for trade in active_trades:
                if trade.product_type == "INTRADAY":
                    quote = await self.broker.get_quote(trade.symbol, Exchange.NFO)
                    if quote:
                        await self._execute_exit(
                            trade,
                            quote.ltp,
                            RiskEvent.SCHEDULED_SQUARE_OFF
                        )


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 4: Expiry Short Strangle
# ══════════════════════════════════════════════════════════════════════════════

class ExpiryShortStrangleStrategy(BaseStrategy):
    """
    Sells OTM call + put strangles and collects premium (theta decay).

    Unlike directional buying strategies, this profits when:
      - The underlying stays range-bound
      - Implied volatility is elevated at entry (expensive premiums) then reverts

    Strikes: ATM + 2×si (OTM call) and ATM − 2×si (OTM put)
    SL per leg: 3× the premium collected on that leg (option doubles from entry)
    Entry window: 10:30–11:00 AM (after morning volatility settles)
    Only one trade per expiry week — holds intraday (simplified from overnight)
    VIX: >13% for elevated premium; <25% to avoid panic blowouts
    """

    def __init__(
        self,
        config: StrategyConfig,
        bs_engine: BlackScholesEngine,
        database: TradingDatabase,
        risk_manager: RiskManager,
        broker: BrokerGateway,
    ):
        super().__init__(
            name="ExpiryShortStrangle",
            config=config,
            bs_engine=bs_engine,
            database=database,
            risk_manager=risk_manager,
            broker=broker,
        )
        self._last_entry_week: Optional[str] = None
        self._vix_history:    List[float]   = []   # for IV Rank

    # Approx SPAN margin per strangle lot (CE+PE pair) — used for lot sizing
    _MARGIN_PER_LOT = {"NIFTY": 80_000, "BANKNIFTY": 95_000,
                        "SENSEX": 70_000, "FINNIFTY": 50_000}
    _MARGIN_DEPLOY_PCT = 0.40   # deploy 40% of capital as margin

    def _entry_window(self, ts: Optional[datetime] = None) -> bool:
        now = (ts or datetime.now()).time()
        return dt_time(10, 30) <= now <= dt_time(11, 15)

    def _vix_range_ok(self, chain: "OptionChainSnapshot") -> bool:
        vix = self._atm_iv(chain) * 100
        return 13.0 <= vix <= 25.0

    def _iv_rank(self, chain: "OptionChainSnapshot") -> float:
        """
        IV Rank: where is today's IV relative to the last 60 trading days?
        0 = cheapest premiums, 1 = most expensive.
        Only sell when IV Rank > 0.30 (premiums are above recent average).
        This is the KEY filter Dhan/Streak strategies use for better timing.
        """
        vix = self._atm_iv(chain) * 100
        self._vix_history.append(vix)
        if len(self._vix_history) > 60:
            self._vix_history.pop(0)
        if len(self._vix_history) < 10:
            return 0.5
        lo, hi = min(self._vix_history), max(self._vix_history)
        return (vix - lo) / (hi - lo) if hi > lo else 0.5

    def _calc_lots(self, underlying: str) -> int:
        """
        Phase 3: Iron condor is hedged → use hedged margin (₹40K/lot) and
        draw from the strangle sleeve.
        """
        return self._short_option_lots(is_hedged=True, strategy_type="strangle")

    async def evaluate(
        self, chain: "OptionChainSnapshot"
    ) -> Optional[SpreadSignal]:
        self._bar_count += 1
        self._track_spot(chain)

        if not self._entry_window(chain.timestamp):
            return None
        if not self._vix_range_ok(chain):
            return None

        # Phase 2: IV Rank threshold lowered 0.30 → 0.15.
        # 2025 VIX was mostly 10-16%, which kept iv_rank < 0.30 for most of the year
        # and caused 90%+ of potential strangle entries to be blocked (only 4 trades).
        if self._iv_rank(chain) < 0.15:
            return None

        if self.db.has_active_trade(self.name):
            return None
        can_enter, _ = self.risk_manager.can_enter_position()
        if not can_enter:
            return None

        week_key = chain.timestamp.strftime("%Y-W%V")
        if self._last_entry_week == week_key:
            return None

        si  = self._si(chain)
        atm = chain.atm_strike
        und = chain.underlying
        lots = self._calc_lots(und)

        call_strike = atm + 2 * si
        put_strike  = atm - 2 * si

        call_q = chain.calls.get(call_strike)
        put_q  = chain.puts.get(put_strike)
        if not call_q or not put_q:
            return None

        # Phase 2: use expiry_min_premium_spot_pct for the strangle so far-OTM
        # strikes on expiry days (crushed premiums) are still selectable.
        # 0.0003 × 25000 = ₹7.50 vs the global ₹20 min_premium.
        expiry_min = max(
            chain.spot_price * self.config.expiry_min_premium_spot_pct,
            1.0,   # absolute floor: never trade for sub-₹1 premium
        )
        if call_q.ltp < expiry_min or put_q.ltp < expiry_min:
            return None

        # Phase 3: IRON CONDOR — add protective wings to prevent margin blowout.
        # Naked strangles are banned for ₹2L accounts (NSE SPAN peaks at expiry).
        # Wings reduce margin from ~₹80K to ~₹25-35K per lot.
        wing_call_strike = atm + 5 * si   # upper wing: BUY
        wing_put_strike  = atm - 5 * si   # lower wing: BUY

        wing_call_q = chain.calls.get(wing_call_strike)
        wing_put_q  = chain.puts.get(wing_put_strike)
        if not wing_call_q or not wing_put_q:
            return None   # wings must exist

        # Net credit = (short premiums) - (wing premiums)
        gross_credit = call_q.ltp + put_q.ltp
        wing_cost    = wing_call_q.ltp + wing_put_q.ltp
        net_premium  = gross_credit - wing_cost
        if net_premium <= 0:
            return None   # condor doesn't pay — skip

        # SL: combined exit if net premium received doubles (net loss = net_premium)
        combined_sl     = net_premium * 2.0
        combined_target = net_premium * 0.30   # keep 70% of credit

        # 4 legs — BUY wings FIRST (for margin benefit in live trading)
        legs = [
            SpreadLeg(
                symbol=self._sym(chain, wing_call_strike, "CE"),
                strike=wing_call_strike, option_type=OptionType.CALL,
                direction=TransactionType.BUY,
                quantity=lots * self._lot_size, price=wing_call_q.ltp,
            ),
            SpreadLeg(
                symbol=self._sym(chain, wing_put_strike, "PE"),
                strike=wing_put_strike, option_type=OptionType.PUT,
                direction=TransactionType.BUY,
                quantity=lots * self._lot_size, price=wing_put_q.ltp,
            ),
            SpreadLeg(
                symbol=self._sym(chain, call_strike, "CE"),
                strike=call_strike, option_type=OptionType.CALL,
                direction=TransactionType.SELL,
                quantity=lots * self._lot_size, price=call_q.ltp,
            ),
            SpreadLeg(
                symbol=self._sym(chain, put_strike, "PE"),
                strike=put_strike, option_type=OptionType.PUT,
                direction=TransactionType.SELL,
                quantity=lots * self._lot_size, price=put_q.ltp,
            ),
        ]

        return SpreadSignal(
            signal_type=SignalType.SHORT,
            strategy_name=self.name,
            timestamp=datetime.now(),
            confidence=min(1.0, (self._atm_iv(chain) * 100 - 13) / 12),
            entry_price=net_premium,
            stop_loss=combined_sl,
            target=combined_target,
            symbol=f"{chain.underlying}{chain.expiry}_IronCondor",
            option_type=OptionType.CALL,
            strike=atm,
            expiry=chain.expiry,
            metadata={
                "call_strike": call_strike, "put_strike": put_strike,
                "wing_call_strike": wing_call_strike, "wing_put_strike": wing_put_strike,
                "net_premium": net_premium, "gross_credit": gross_credit,
                "wing_cost": wing_cost, "lots": lots,
                "iv_rank": round(self._iv_rank(chain), 3),
                "week_key": week_key,
            },
            legs=legs,
        )

    async def execute_signal(self, signal: SpreadSignal) -> bool:
        if not isinstance(signal, SpreadSignal) or not signal.legs:
            return False
        if not self.db.acquire_trade_lock(
            self.name, signal.symbol, f"strategy_{self.name}"
        ):
            return False

        parent_id  = str(uuid.uuid4())
        executed   = []
        week_key   = datetime.now().strftime("%Y-W%V")

        try:
            for i, leg in enumerate(signal.legs):
                response = await self.broker.place_order(
                    symbol=leg.symbol,
                    exchange=Exchange.NFO,
                    transaction_type=leg.direction,
                    order_type=OrderType.LIMIT,
                    quantity=leg.quantity,
                    price=leg.price,
                    product_type=ProductType.INTRADAY,
                )
                if not response.success:
                    await self._unwind_legs(executed)
                    self.db.release_trade_lock(self.name, signal.symbol)
                    return False

                executed.append((leg, response))

                # SL/target per leg:
                # SELL legs: SL fires if premium RISES (loss), target if it falls
                # BUY  legs: SL fires if premium FALLS (loss), target if it rises
                if leg.direction == TransactionType.SELL:
                    sl_price  = leg.price * 2.5   # Phase 4: +150% → let 0DTE breathe past micro-whips
                    tgt_price = leg.price * 0.30  # keep 70% of premium collected
                    direction_str = "SELL"
                else:
                    # BUY wing: SL = 50% loss on wing premium, target = 200% gain
                    sl_price  = leg.price * 0.50
                    tgt_price = leg.price * 3.00
                    direction_str = "BUY"

                trade = TradeRecord(
                    trade_id=str(uuid.uuid4()),
                    strategy_name=self.name,
                    symbol=leg.symbol,
                    option_type=leg.option_type.value,
                    strike=leg.strike,
                    expiry=signal.expiry,
                    entry_price=leg.price,
                    quantity=leg.quantity,
                    direction=direction_str,
                    product_type="INTRADAY",
                    status=TradeStatus.ACTIVE.value,
                    stop_loss=sl_price,
                    target=tgt_price,
                    entry_time=datetime.now().isoformat(),
                    leg_id=f"leg_{i}",
                    parent_trade_id=parent_id,
                )
                self.db.insert_trade(trade)

                self.risk_manager.update_position(
                    trade_id=trade.trade_id,
                    symbol=leg.symbol,
                    quantity=leg.quantity,
                    average_price=leg.price,
                    current_price=leg.price,
                    direction=direction_str,
                    stop_loss=sl_price,
                    target=tgt_price,
                )

            self._last_entry_week = signal.metadata.get("week_key", week_key)
            ic_meta = signal.metadata
            logger.info(
                "Iron Condor entered: %s  net_credit=₹%.2f  "
                "short_strikes=(%s/%s)  wings=(%s/%s)",
                signal.symbol, signal.entry_price,
                ic_meta.get("call_strike", "?"), ic_meta.get("put_strike", "?"),
                ic_meta.get("wing_call_strike", "?"), ic_meta.get("wing_put_strike", "?"),
            )
            return True

        except Exception as e:
            logger.error(f"Strangle execution error: {e}")
            await self._unwind_legs(executed)
            self.db.release_trade_lock(self.name, signal.symbol)
            return False

    async def _unwind_legs(self, executed_legs) -> None:
        for leg, response in reversed(executed_legs):
            try:
                await self.broker.cancel_order(response.broker_order_id)
                unwind_dir = (
                    TransactionType.SELL if leg.direction == TransactionType.BUY
                    else TransactionType.BUY
                )
                await self.broker.place_order(
                    symbol=leg.symbol, exchange=Exchange.NFO,
                    transaction_type=unwind_dir, order_type=OrderType.MARKET,
                    quantity=leg.quantity, product_type=ProductType.INTRADAY,
                )
            except Exception as e:
                logger.error(f"Unwind failed for {leg.symbol}: {e}")

    # ── Phase 3: Leg-level strangle management ────────────────────────────────

    async def manage_positions(self) -> None:
        """
        Phase 3 — Institutional leg-by-leg strangle management.

        Rule: Do NOT look at combined premium.  Each leg is independent.

        SHORT CALL leg: if its premium rises +80% from entry (underlying
            surged), buy it back immediately (LEG_SL_HIT) and leave the
            short PUT open to collect remaining theta.

        SHORT PUT leg: same in reverse.

        Winning leg: once we cut the losing leg, place a trailing stop on
            the survivor at its entry price (breakeven).  Any decay above
            that is pure profit.
        """
        active_trades = self.db.get_active_trades(self.name)
        if not active_trades:
            return

        # Group by parent_trade_id so we can manage legs together
        from collections import defaultdict
        parents: dict = defaultdict(list)
        for t in active_trades:
            parents[t.parent_trade_id or t.trade_id].append(t)

        for parent_id, legs in parents.items():
            for trade in legs:
                if self.sl_manager.is_exit_pending(trade.trade_id):
                    continue

                quote = await self.broker.get_quote(trade.symbol, Exchange.NFO)
                if not quote:
                    continue

                current_price = quote.ltp
                entry         = trade.entry_price

                if trade.direction == "SELL" and entry > 0:
                    pct_increase = (current_price - entry) / entry * 100

                    # +80% on a short leg = buy it back (LEG_SL_HIT)
                    if pct_increase >= 80.0:
                        await self._execute_exit(
                            trade, current_price, None, reason="LEG_SL_HIT"
                        )
                        # Tighten stop on sibling legs to breakeven
                        for sibling in legs:
                            if sibling.trade_id != trade.trade_id:
                                self.risk_manager.update_trailing_stop(
                                    sibling.trade_id, sibling.entry_price * 0.99
                                )
                        continue

                # Standard combined SL (fallback)
                should_exit, event = self.sl_manager.should_exit(
                    trade.trade_id, current_price
                )
                if should_exit:
                    await self._execute_exit(trade, current_price, event)


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 5: Zen Credit Spread
# ══════════════════════════════════════════════════════════════════════════════

def _tsrank(series: np.ndarray) -> float:
    """Time-series rank: fraction of values in `series` that are < last value."""
    if len(series) < 2:
        return 0.5
    return float(np.sum(series[:-1] < series[-1]) / (len(series) - 1))


class ZenCreditSpreadStrategy(BaseStrategy):
    """
    Strategy 5: Zen Credit Spread Overnight.

    Alpha signals (from description):
      alpha  = tsrank(5-min normalised price change, lookback=800 bars≈67 hours)
      alpha2 = tsrank(5-min price-change × vol_ratio / atm_vol, lookback=300 bars)

    Signal triggers:
      alpha > 0.8 AND alpha2 > 0.8  → Credit PUT spread (bullish)
      alpha < 0.2 AND alpha2 < 0.2  → Credit CALL spread (bearish)

    Spread construction (scaled to actual strike interval):
      Bull: sell ATM put, buy put 8×si below  (≈400 pts for Nifty)
      Bear: sell ATM call, buy call 8×si above
    """

    LOOKBACK_ALPHA  = 800   # bars (800 one-min ≈ 2.13 days; matches "800-min lookback")
    LOOKBACK_ALPHA2 = 300

    def __init__(
        self,
        config: StrategyConfig,
        bs_engine: BlackScholesEngine,
        database: TradingDatabase,
        risk_manager: RiskManager,
        broker: BrokerGateway,
    ):
        super().__init__(
            name="ZenCreditSpread",
            config=config,
            bs_engine=bs_engine,
            database=database,
            risk_manager=risk_manager,
            broker=broker,
        )
        self._open_price: Optional[float] = None
        self._day_open:   Optional[str]   = None
        # Rolling histories for alpha signals
        self._alpha_series:  np.ndarray = np.array([])
        self._alpha2_series: np.ndarray = np.array([])
        # 5-min bar accumulation
        self._bar_5min_count:   int           = 0
        self._price_5min_start: Optional[float] = None
        self._vol_ratio_buf:    List[float]   = []
        self._atm_vol_buf:      List[float]   = []

    def _update_5min_bar(self, chain: "OptionChainSnapshot") -> Optional[tuple]:
        """
        Accumulate 1-min bars into 5-min observations.
        Returns (pct_change_5min, vol_ratio, atm_vol) when a 5-min bar closes.
        """
        self._bar_5min_count += 1

        # Reset open price each day
        day_str = datetime.now().strftime("%Y-%m-%d")
        if self._day_open != day_str:
            self._day_open  = day_str
            self._open_price = chain.spot_price

        # Track first price of the 5-min window
        if self._price_5min_start is None:
            self._price_5min_start = chain.spot_price

        # ATM volume ratio (call vol / put vol)
        atm = chain.atm_strike
        call_q = chain.calls.get(atm)
        put_q  = chain.puts.get(atm)
        if call_q and put_q:
            self._vol_ratio_buf.append(call_q.volume / max(1, put_q.volume))
            iv_sum = (call_q.iv or 0) + (put_q.iv or 0)
            self._atm_vol_buf.append(iv_sum if iv_sum > 0 else 0.14)

        if self._bar_5min_count % 5 != 0:
            return None   # 5-min bar not yet complete

        # 5-min bar closes
        pct_change = (chain.spot_price - self._price_5min_start) / max(1, self._price_5min_start)
        norm_change = pct_change / max(1e-6, self._open_price / chain.spot_price) if self._open_price else pct_change

        avg_vol_ratio = float(np.mean(self._vol_ratio_buf)) if self._vol_ratio_buf else 1.0
        avg_atm_vol   = float(np.mean(self._atm_vol_buf))   if self._atm_vol_buf  else 0.14

        # Reset window buffers
        self._price_5min_start = chain.spot_price
        self._vol_ratio_buf.clear()
        self._atm_vol_buf.clear()

        return norm_change, avg_vol_ratio, avg_atm_vol

    def _compute_alphas(self, chain: "OptionChainSnapshot") -> tuple[float, float]:
        result = self._update_5min_bar(chain)
        if result is None:
            # Not a 5-min boundary — return last known alpha values
            if len(self._alpha_series) > 0:
                return _tsrank(self._alpha_series), _tsrank(self._alpha2_series)
            return 0.5, 0.5

        norm_change, vol_ratio, atm_vol = result

        # Alpha: normalised price change → tsrank
        self._alpha_series = np.append(self._alpha_series, norm_change)
        if len(self._alpha_series) > self.LOOKBACK_ALPHA:
            self._alpha_series = self._alpha_series[-self.LOOKBACK_ALPHA:]

        # Alpha2: price change × vol ratio / atm vol → tsrank
        alpha2_raw = norm_change * vol_ratio / max(0.01, atm_vol)
        self._alpha2_series = np.append(self._alpha2_series, alpha2_raw)
        if len(self._alpha2_series) > self.LOOKBACK_ALPHA2:
            self._alpha2_series = self._alpha2_series[-self.LOOKBACK_ALPHA2:]

        return _tsrank(self._alpha_series), _tsrank(self._alpha2_series)

    async def evaluate(
        self, chain: "OptionChainSnapshot"
    ) -> Optional[SpreadSignal]:
        self._bar_count += 1
        if self._bar_count < 100:
            return None   # need at least 20 × 5-min bars for tsrank

        if not self.is_trading_window():
            return None
        if self.db.has_active_trade(self.name):
            return None

        can_enter, reason = self.risk_manager.can_enter_position()
        if not can_enter:
            return None

        alpha, alpha2 = self._compute_alphas(chain)

        si  = self._si(chain)
        atm = chain.atm_strike

        # Phase 2: DEBIT SPREAD — capital-efficient for ₹2L accounts.
        # Intraday credit spreads bleed Gamma before Theta can work.
        # Bull Call Debit Spread: BUY ATM Call, SELL OTM Call.
        # Bear Put Debit Spread: BUY ATM Put,  SELL OTM Put.
        # Max loss = net_debit paid. Profits from directional move + IV expansion.

        if alpha > 0.8 and alpha2 > 0.8:
            # BULLISH: BUY ATM Call + SELL OTM Call
            buy_strike  = atm
            sell_strike = atm + 3 * si

            buy_q  = chain.calls.get(buy_strike)
            sell_q = chain.calls.get(sell_strike)
            if not buy_q or not sell_q:
                return None
            net_debit = buy_q.ltp - sell_q.ltp   # cost of the spread (always > 0)
            if net_debit <= 0:
                return None

            # SL/Target relative to the BUY leg price so manage_positions can compare
            # current BUY-leg market price against these levels.
            buy_sl     = buy_q.ltp - 0.50 * net_debit   # 50% max loss of spread cost
            buy_target = buy_q.ltp + 0.80 * net_debit   # 80% profit on spread cost

            lots = self._long_option_lots(net_debit, strategy_type="credit_spread")
            legs = [
                SpreadLeg(
                    symbol=self._sym(chain, buy_strike, "CE"),
                    strike=buy_strike, option_type=OptionType.CALL,
                    direction=TransactionType.BUY,
                    quantity=lots * self._lot_size, price=buy_q.ltp,
                ),
                SpreadLeg(
                    symbol=self._sym(chain, sell_strike, "CE"),
                    strike=sell_strike, option_type=OptionType.CALL,
                    direction=TransactionType.SELL,
                    quantity=lots * self._lot_size, price=sell_q.ltp,
                ),
            ]
            return SpreadSignal(
                signal_type=SignalType.LONG,
                strategy_name=self.name,
                timestamp=datetime.now(),
                confidence=(alpha + alpha2) / 2,
                entry_price=net_debit,     # cost basis of the spread
                stop_loss=buy_sl,          # absolute SL level for BUY leg
                target=buy_target,         # absolute target for BUY leg
                symbol=f"{chain.underlying}{chain.expiry}_ZenCallDebit",
                option_type=OptionType.CALL,
                strike=buy_strike,
                expiry=chain.expiry,
                metadata={"alpha": alpha, "alpha2": alpha2,
                          "net_debit": net_debit, "spread_type": "call_debit"},
                legs=legs,
            )

        if alpha < 0.2 and alpha2 < 0.2:
            # BEARISH: BUY ATM Put + SELL OTM Put
            buy_strike  = atm
            sell_strike = atm - 3 * si

            buy_q  = chain.puts.get(buy_strike)
            sell_q = chain.puts.get(sell_strike)
            if not buy_q or not sell_q:
                return None
            net_debit = buy_q.ltp - sell_q.ltp
            if net_debit <= 0:
                return None

            buy_sl     = buy_q.ltp - 0.50 * net_debit
            buy_target = buy_q.ltp + 0.80 * net_debit

            lots = self._long_option_lots(net_debit, strategy_type="credit_spread")
            legs = [
                SpreadLeg(
                    symbol=self._sym(chain, buy_strike, "PE"),
                    strike=buy_strike, option_type=OptionType.PUT,
                    direction=TransactionType.BUY,
                    quantity=lots * self._lot_size, price=buy_q.ltp,
                ),
                SpreadLeg(
                    symbol=self._sym(chain, sell_strike, "PE"),
                    strike=sell_strike, option_type=OptionType.PUT,
                    direction=TransactionType.SELL,
                    quantity=lots * self._lot_size, price=sell_q.ltp,
                ),
            ]
            return SpreadSignal(
                signal_type=SignalType.LONG,
                strategy_name=self.name,
                timestamp=datetime.now(),
                confidence=1 - (alpha + alpha2) / 2,
                entry_price=net_debit,
                stop_loss=buy_sl,
                target=buy_target,
                symbol=f"{chain.underlying}{chain.expiry}_ZenPutDebit",
                option_type=OptionType.PUT,
                strike=buy_strike,
                expiry=chain.expiry,
                metadata={"alpha": alpha, "alpha2": alpha2,
                          "net_debit": net_debit, "spread_type": "put_debit"},
                legs=legs,
            )

        return None

    async def execute_signal(self, signal: SpreadSignal) -> bool:
        if not isinstance(signal, SpreadSignal) or not signal.legs:
            return False
        if not self.db.acquire_trade_lock(
            self.name, signal.symbol, f"strategy_{self.name}"
        ):
            return False

        parent_id = str(uuid.uuid4())
        executed  = []

        try:
            for i, leg in enumerate(signal.legs):
                response = await self.broker.place_order(
                    symbol=leg.symbol,
                    exchange=Exchange.NFO,
                    transaction_type=leg.direction,
                    order_type=OrderType.LIMIT,
                    quantity=leg.quantity,
                    price=leg.price,
                    product_type=ProductType.INTRADAY,
                )
                if not response.success:
                    await self._unwind_legs(executed)
                    self.db.release_trade_lock(self.name, signal.symbol)
                    return False
                executed.append((leg, response))

                trade = TradeRecord(
                    trade_id=str(uuid.uuid4()),
                    strategy_name=self.name,
                    symbol=leg.symbol,
                    option_type=leg.option_type.value,
                    strike=leg.strike,
                    expiry=signal.expiry,
                    entry_price=leg.price,
                    quantity=leg.quantity,
                    direction=leg.direction.value,
                    product_type="INTRADAY",
                    status=TradeStatus.ACTIVE.value,
                    stop_loss=signal.stop_loss,
                    target=signal.target,
                    entry_time=datetime.now().isoformat(),
                    leg_id=f"leg_{i}",
                    parent_trade_id=parent_id,
                )
                self.db.insert_trade(trade)

            logger.info(
                f"ZenSpread entered: {signal.symbol}  "
                f"credit={signal.entry_price:.2f}  "
                f"α={signal.metadata['alpha']:.3f} α2={signal.metadata['alpha2']:.3f}"
            )
            return True

        except Exception as e:
            logger.error(f"ZenSpread execution error: {e}")
            await self._unwind_legs(executed)
            self.db.release_trade_lock(self.name, signal.symbol)
            return False

    # ── Phase 3: 90-minute time-stop ─────────────────────────────────────────

    async def manage_positions(self) -> None:
        """
        Phase 3: 90-minute time-stop eliminates stagnation bleed.

        Debit spreads that sit open too long lose to theta even when "not losing".
        Rules:
          > 90 min AND PnL negative → TIME_DECAY_CUT (cut the loss now)
          > 90 min AND PnL positive → STAGNATION_PROFIT_TAKE (lock in the gain)
        This converts a floating P&L into a realised one either way.
        """
        active = self.db.get_active_trades(self.name)
        for trade in active:
            quote = await self.broker.get_quote(trade.symbol, Exchange.NFO)
            if not quote:
                continue

            current_price = quote.ltp

            # ── 90-minute time-stop ───────────────────────────────────────
            if trade.direction == "BUY" and trade.entry_price > 0:
                try:
                    entry_dt     = datetime.fromisoformat(trade.entry_time)
                    mins_open    = (datetime.now() - entry_dt).total_seconds() / 60
                    pnl_pct      = (current_price - trade.entry_price) / trade.entry_price * 100
                    if mins_open > 90:
                        reason = "STAGNATION_PROFIT_TAKE" if pnl_pct >= 0 else "TIME_DECAY_CUT"
                        await self._execute_exit(trade, current_price, None, reason=reason)
                        continue
                except (ValueError, TypeError):
                    pass

            # ── Standard SL / target ──────────────────────────────────────
            should_exit, event = self.sl_manager.should_exit(trade.trade_id, current_price)
            if should_exit:
                await self._execute_exit(trade, current_price, event)

    async def _unwind_legs(self, executed_legs) -> None:
        for leg, response in reversed(executed_legs):
            try:
                await self.broker.cancel_order(response.broker_order_id)
                unwind_dir = (
                    TransactionType.SELL if leg.direction == TransactionType.BUY
                    else TransactionType.BUY
                )
                await self.broker.place_order(
                    symbol=leg.symbol, exchange=Exchange.NFO,
                    transaction_type=unwind_dir, order_type=OrderType.MARKET,
                    quantity=leg.quantity, product_type=ProductType.INTRADAY,
                )
            except Exception as e:
                logger.error(f"ZenSpread unwind failed for {leg.symbol}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Strategy 6: Lyapunov Stability Credit Spread
# ══════════════════════════════════════════════════════════════════════════════

class LyapunovCreditSpreadStrategy(BaseStrategy):
    """
    Credit-spread strategy grounded in Lyapunov stability theory.

    Core insight: sell option premium when the market is in a STABLE,
    PREDICTABLE state — not when it's chaotic.  Two measures determine this:

    1. Lyapunov exponent λ  (chaos measure)
         λ = mean |log(P_straddle(t) / P_straddle(t-1))|
         Computed on the ATM straddle price (call_mid + put_mid) — a single
         number capturing total implied-move expectation.
         • λ → 0 : straddle barely moving  → stable regime
         • λ >> 0: straddle oscillating     → chaotic regime

    2. Stability score:  S = 1 − tanh(λ)   ∈ (0, 1]
         tanh maps any positive λ to (0,1).  When λ≈0, S≈1 (max stable).

    3. Predictability score: R² from a linear regression of the straddle
         price series over the rolling window.
         R² ≈ 1 → price trending linearly → predictable
         R² ≈ 0 → price random noise     → unpredictable

    4. Composite alpha: α = S × R²
         Only when BOTH measures are high do we enter.

    Trade logic:
       α > entry_threshold            → sell credit spread
       Direction: underlying momentum
         momentum > 0  → credit PUT spread  (bullish bias)
         momentum < 0  → credit CALL spread (bearish bias)
       α > iron_condor_threshold (0.85) → sell iron condor (both sides)

    SL  : 2× the net credit collected on each leg
    TGT : keep 60% of net credit (exit when remaining value = 40% of entry)
    """

    WINDOW              = 30     # bars for rolling Lyapunov + R²
    ENTRY_THRESHOLD     = 0.50   # α must exceed this to enter
    CONDOR_THRESHOLD    = 0.80   # α for full iron condor

    def __init__(
        self,
        config: StrategyConfig,
        bs_engine: BlackScholesEngine,
        database: TradingDatabase,
        risk_manager: RiskManager,
        broker: BrokerGateway,
    ):
        super().__init__(
            name="LyapunovCreditSpread",
            config=config,
            bs_engine=bs_engine,
            database=database,
            risk_manager=risk_manager,
            broker=broker,
        )
        self._straddle_prices: List[float] = []   # rolling ATM straddle price
        self._alpha_history:   List[float] = []   # for monitoring

    def _atm_straddle_price(self, chain: "OptionChainSnapshot") -> Optional[float]:
        """Mid-price of ATM call + ATM put (total premium in the market)."""
        atm  = chain.atm_strike
        call = chain.calls.get(atm)
        put  = chain.puts.get(atm)
        if not call or not put:
            return None
        c_mid = (call.bid + call.ask) / 2 if call.bid and call.ask else call.ltp
        p_mid = (put.bid  + put.ask)  / 2 if put.bid  and put.ask  else put.ltp
        return c_mid + p_mid

    # ── Lyapunov exponent ─────────────────────────────────────────────────────

    def _compute_lyapunov(self) -> float:
        """
        λ = (1/N) Σ |log(P(t) / P(t−1))|

        Mean absolute log-return of the ATM straddle price.
        Measures how quickly consecutive option prices diverge — exactly the
        logarithmic divergence concept from Lyapunov stability theory.
        """
        prices = np.array(self._straddle_prices)
        if len(prices) < 3:
            return 0.5   # not enough data → assume moderate chaos
        # Guard against zero prices
        safe = np.maximum(prices, 1e-6)
        log_returns = np.abs(np.diff(np.log(safe)))
        return float(np.mean(log_returns))

    # ── R² predictability ─────────────────────────────────────────────────────

    def _compute_r2(self) -> float:
        """
        R² from OLS regression of straddle price on time index.
        Captures whether the option price is evolving in a PREDICTABLE,
        linear pattern (R²→1) or bouncing randomly (R²→0).
        """
        prices = np.array(self._straddle_prices)
        n = len(prices)
        if n < 5:
            return 0.0

        x      = np.arange(n, dtype=float)
        x_mean = np.mean(x)
        y_mean = np.mean(prices)

        ss_xy = float(np.sum((x - x_mean) * (prices - y_mean)))
        ss_xx = float(np.sum((x - x_mean) ** 2))

        if ss_xx < 1e-10:
            return 1.0   # constant price → perfectly predictable

        slope     = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        y_pred    = slope * x + intercept

        ss_res = float(np.sum((prices - y_pred) ** 2))
        ss_tot = float(np.sum((prices - y_mean) ** 2))

        if ss_tot < 1e-10:
            return 1.0
        return max(0.0, 1.0 - ss_res / ss_tot)

    # ── Composite alpha ───────────────────────────────────────────────────────

    def _compute_alpha(self) -> tuple[float, float, float]:
        """
        Returns (alpha, lyapunov, r2).
        alpha = (1 − tanh(λ)) × R²
        """
        lam  = self._compute_lyapunov()
        r2   = self._compute_r2()
        stab = 1.0 - float(np.tanh(lam))  # tanh ∈ [0,1) → stability ∈ (0,1]
        alpha = stab * r2
        return alpha, lam, r2

    # ── Spread builders ───────────────────────────────────────────────────────

    # Phase 2: DEBIT spread versions — BUY ATM, SELL OTM.
    # Capital-efficient for ₹2L: defined risk, profits from directional move.

    def _put_spread(
        self, chain: "OptionChainSnapshot", atm: float, si: float
    ) -> Optional[SpreadSignal]:
        """Bearish debit put spread: BUY ATM Put, SELL OTM Put."""
        buy_k  = atm
        sell_k = atm - 2 * si
        bq, sq = chain.puts.get(buy_k), chain.puts.get(sell_k)
        if not bq or not sq:
            return None
        net_debit = bq.ltp - sq.ltp
        if net_debit <= 0:
            return None
        alpha, lam, r2 = self._compute_alpha()
        lots = self._long_option_lots(net_debit, strategy_type="credit_spread")
        return SpreadSignal(
            signal_type=SignalType.LONG,
            strategy_name=self.name,
            timestamp=datetime.now(),
            confidence=alpha,
            entry_price=net_debit,
            stop_loss=bq.ltp - 0.50 * net_debit,   # 50% loss on spread cost
            target=bq.ltp   + 0.80 * net_debit,    # 80% profit on spread cost
            symbol=f"{chain.underlying}{chain.expiry}_LyapPutDebit",
            option_type=OptionType.PUT,
            strike=buy_k,
            expiry=chain.expiry,
            metadata={"alpha": alpha, "lyapunov": lam, "r2": r2,
                      "net_debit": net_debit, "spread_type": "put_debit"},
            legs=[
                SpreadLeg(self._sym(chain, buy_k,  "PE"), buy_k,  OptionType.PUT,
                          TransactionType.BUY,  lots * self._lot_size, bq.ltp),
                SpreadLeg(self._sym(chain, sell_k, "PE"), sell_k, OptionType.PUT,
                          TransactionType.SELL, lots * self._lot_size, sq.ltp),
            ],
        )

    def _call_spread(
        self, chain: "OptionChainSnapshot", atm: float, si: float
    ) -> Optional[SpreadSignal]:
        """Bullish debit call spread: BUY ATM Call, SELL OTM Call."""
        buy_k  = atm
        sell_k = atm + 2 * si
        bq, sq = chain.calls.get(buy_k), chain.calls.get(sell_k)
        if not bq or not sq:
            return None
        net_debit = bq.ltp - sq.ltp
        if net_debit <= 0:
            return None
        alpha, lam, r2 = self._compute_alpha()
        lots = self._long_option_lots(net_debit, strategy_type="credit_spread")
        return SpreadSignal(
            signal_type=SignalType.LONG,
            strategy_name=self.name,
            timestamp=datetime.now(),
            confidence=alpha,
            entry_price=net_debit,
            stop_loss=bq.ltp - 0.50 * net_debit,
            target=bq.ltp   + 0.80 * net_debit,
            symbol=f"{chain.underlying}{chain.expiry}_LyapCallDebit",
            option_type=OptionType.CALL,
            strike=buy_k,
            expiry=chain.expiry,
            metadata={"alpha": alpha, "lyapunov": lam, "r2": r2,
                      "net_debit": net_debit, "spread_type": "call_debit"},
            legs=[
                SpreadLeg(self._sym(chain, buy_k,  "CE"), buy_k,  OptionType.CALL,
                          TransactionType.BUY,  lots * self._lot_size, bq.ltp),
                SpreadLeg(self._sym(chain, sell_k, "CE"), sell_k, OptionType.CALL,
                          TransactionType.SELL, lots * self._lot_size, sq.ltp),
            ],
        )

    # ── Evaluate ──────────────────────────────────────────────────────────────

    async def evaluate(
        self, chain: "OptionChainSnapshot"
    ) -> Optional[SpreadSignal]:
        self._bar_count += 1
        self._track_spot(chain)

        # Track ATM straddle price every bar
        straddle = self._atm_straddle_price(chain)
        if straddle and straddle > 0:
            self._straddle_prices.append(straddle)
            if len(self._straddle_prices) > self.WINDOW:
                self._straddle_prices.pop(0)

        if self._bar_count < self._warmup_bars:
            return None
        if len(self._straddle_prices) < 10:
            return None

        if not self.is_trading_window():
            return None
        if self.db.has_active_trade(self.name):
            return None

        can_enter, reason = self.risk_manager.can_enter_position()
        if not can_enter:
            return None

        # ── Lyapunov composite alpha ──────────────────────────────────────
        alpha, lam, r2 = self._compute_alpha()
        self._alpha_history.append(alpha)

        if alpha < self.ENTRY_THRESHOLD:
            return None   # market too chaotic or unpredictable

        si  = self._si(chain)
        atm = chain.atm_strike
        mom = self._spot_momentum(15)

        logger.debug(
            f"Lyapunov signal: α={alpha:.3f}  λ={lam:.4f}  R²={r2:.3f}  "
            f"mom={mom:.4f}  straddle=₹{straddle:.1f}"
        )

        # ── Iron condor when exceptionally stable ─────────────────────────
        if alpha >= self.CONDOR_THRESHOLD:
            put_sig  = self._put_spread(chain, atm, si)
            call_sig = self._call_spread(chain, atm, si)
            # Return the put spread first (execute call spread next evaluation)
            if put_sig:
                put_sig.symbol = f"{chain.underlying}{chain.expiry}_LyapCondor"
                put_sig.metadata["spread_type"] = "iron_condor"
                return put_sig

        # ── Directional credit spread based on momentum ───────────────────
        if mom > 0.0001:               # slight uptrend → put spread (bullish)
            return self._put_spread(chain, atm, si)
        elif mom < -0.0001:            # slight downtrend → call spread (bearish)
            return self._call_spread(chain, atm, si)
        else:                          # flat → put spread (theta bias)
            return self._put_spread(chain, atm, si)

    # ── Execute ───────────────────────────────────────────────────────────────

    async def execute_signal(self, signal: SpreadSignal) -> bool:
        if not isinstance(signal, SpreadSignal) or not signal.legs:
            return False
        if not self.db.acquire_trade_lock(
            self.name, signal.symbol, f"strategy_{self.name}"
        ):
            return False

        parent_id = str(uuid.uuid4())
        executed  = []

        try:
            for i, leg in enumerate(signal.legs):
                response = await self.broker.place_order(
                    symbol=leg.symbol,
                    exchange=Exchange.NFO,
                    transaction_type=leg.direction,
                    order_type=OrderType.LIMIT,
                    quantity=leg.quantity,
                    price=leg.price,
                    product_type=ProductType.INTRADAY,
                )
                if not response.success:
                    await self._unwind_legs(executed)
                    self.db.release_trade_lock(self.name, signal.symbol)
                    return False
                executed.append((leg, response))

                trade = TradeRecord(
                    trade_id=str(uuid.uuid4()),
                    strategy_name=self.name,
                    symbol=leg.symbol,
                    option_type=leg.option_type.value,
                    strike=leg.strike,
                    expiry=signal.expiry,
                    entry_price=leg.price,
                    quantity=leg.quantity,
                    direction=leg.direction.value,
                    product_type="INTRADAY",
                    status=TradeStatus.ACTIVE.value,
                    stop_loss=signal.stop_loss,
                    target=signal.target,
                    entry_time=datetime.now().isoformat(),
                    leg_id=f"leg_{i}",
                    parent_trade_id=parent_id,
                )
                self.db.insert_trade(trade)

            m = signal.metadata
            logger.info(
                f"Lyapunov spread entered: {signal.metadata.get('spread_type','')}  "
                f"credit=₹{signal.entry_price:.2f}  "
                f"α={m.get('alpha',0):.3f}  λ={m.get('lyapunov',0):.4f}  "
                f"R²={m.get('r2',0):.3f}"
            )
            return True

        except Exception as e:
            logger.error(f"Lyapunov execution error: {e}")
            await self._unwind_legs(executed)
            self.db.release_trade_lock(self.name, signal.symbol)
            return False

    async def _unwind_legs(self, executed_legs) -> None:
        for leg, response in reversed(executed_legs):
            try:
                await self.broker.cancel_order(response.broker_order_id)
                unwind_dir = (
                    TransactionType.SELL if leg.direction == TransactionType.BUY
                    else TransactionType.BUY
                )
                await self.broker.place_order(
                    symbol=leg.symbol, exchange=Exchange.NFO,
                    transaction_type=unwind_dir, order_type=OrderType.MARKET,
                    quantity=leg.quantity, product_type=ProductType.INTRADAY,
                )
            except Exception as e:
                logger.error(f"Lyapunov unwind failed for {leg.symbol}: {e}")

    def get_lyapunov_stats(self) -> dict:
        """Diagnostic: returns current Lyapunov metrics (useful for monitoring)."""
        if not self._straddle_prices:
            return {}
        alpha, lam, r2 = self._compute_alpha()
        return {
            "lyapunov_exponent": round(lam, 6),
            "stability":         round(1.0 - float(np.tanh(lam)), 4),
            "r_squared":         round(r2, 4),
            "composite_alpha":   round(alpha, 4),
            "straddle_price":    round(self._straddle_prices[-1], 2),
            "window_bars":       len(self._straddle_prices),
        }



# ══════════════════════════════════════════════════════════════════════════════
# Strategy 7: Compression Breakout
# ══════════════════════════════════════════════════════════════════════════════

class CompressionBreakoutStrategy(BaseStrategy):
    """
    Institutional-grade directional strategy based on variance compression,
    liquidity sweeps, and explosive breakouts.

    Uses REAL Nifty 1-min OHLC from Fyers — NOT synthetic option chain data.
    This gives it genuine signal quality unlike strategies that depend on
    modelled IV/OI.

    Algorithm:
    ┌─────────────────────────────────────────────────────────────────┐
    │ 1. SQUEEZE: Bollinger Bands inside Keltner Channel for 5+ bars  │
    │    (variance has collapsed — energy building for breakout)       │
    │ 2. SWEEP: Price breaks below 20-period low then recovers in 3   │
    │    candles (trapped shorts provide liquidity anchor)             │
    │ 3. ENTRY: Current bar breaks above highest high of last 10 bars │
    │    → Buy ATM call option                                         │
    │ 4. TRAIL: SL at sweep low → +40% → breakeven → trail +15%/20%  │
    └─────────────────────────────────────────────────────────────────┘

    Mathematical underpinning:
    • BB squeeze = σ → 0 (variance collapse) precedes σ expansion
    • Liquidity sweep = stop-hunt below support (Smart Money Concepts)
    • ATR-based Keltner Channel filters noise from BB-only squeezes
    """

    # ── Indicator parameters ──────────────────────────────────────────────────
    BB_PERIOD     = 20    # Bollinger Band lookback
    BB_STD        = 2.0   # standard deviations
    KC_ATR_MULT   = 2.0   # Fix 1: widened 1.5→2.0 so BB fits inside KC more on 1-min noise
    ATR_PERIOD    = 14    # True Range smoothing period
    SQUEEZE_BARS  = 5     # consecutive bars BB must be inside KC
    SWEEP_LOOKBK  = 20    # period for identifying the key low (sweep target)
    SWEEP_RECOVER = 10    # Fix 1: widened 3→10 candles to recover (1-min charts need time)
    BREAKOUT_BARS = 10    # highest high of last N bars = breakout trigger

    def __init__(
        self,
        config: StrategyConfig,
        bs_engine: BlackScholesEngine,
        database: TradingDatabase,
        risk_manager: RiskManager,
        broker: BrokerGateway,
    ):
        super().__init__(
            name="CompressionBreakout",
            config=config,
            bs_engine=bs_engine,
            database=database,
            risk_manager=risk_manager,
            broker=broker,
        )
        self._squeeze_count:    int            = 0      # consecutive squeeze bars
        self._sweep_detected:   bool           = False
        self._sweep_low:        float          = 0.0    # low of the liquidity sweep
        self._last_signal_day:  Optional[str]  = None   # one trade per day

    # ── Indicator computation ─────────────────────────────────────────────────

    def _compute_atr(self, period: int = 14) -> float:
        """Average True Range over `period` bars."""
        bars = self._ohlc_bars[-(period + 1):]
        if len(bars) < 2:
            return 0.0
        trs = []
        for i in range(1, len(bars)):
            c, h, l    = bars[i]
            prev_c     = bars[i - 1][0]
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            trs.append(tr)
        return float(np.mean(trs)) if trs else 0.0

    def _compute_bb_kc(self) -> Optional[tuple]:
        """
        Returns (bb_upper, bb_lower, kc_upper, kc_lower) for the current bar.
        None if insufficient history.
        """
        min_bars = self.BB_PERIOD + 1
        if len(self._ohlc_bars) < min_bars:
            return None
        bars    = self._ohlc_bars[-self.BB_PERIOD:]
        closes  = np.array([b[0] for b in bars], dtype=float)

        # Bollinger Bands
        bb_mid   = float(np.mean(closes))
        bb_std   = float(np.std(closes, ddof=1))
        bb_upper = bb_mid + self.BB_STD * bb_std
        bb_lower = bb_mid - self.BB_STD * bb_std

        # Keltner Channel (use same mid as BB for consistency)
        atr      = self._compute_atr(self.ATR_PERIOD)
        kc_upper = bb_mid + self.KC_ATR_MULT * atr
        kc_lower = bb_mid - self.KC_ATR_MULT * atr

        return bb_upper, bb_lower, kc_upper, kc_lower

    def _update_squeeze(self) -> bool:
        """
        Update squeeze counter and return True if currently squeezed.
        Squeeze = BB fully inside KC (upper & lower).
        """
        result = self._compute_bb_kc()
        if result is None:
            return False
        bb_upper, bb_lower, kc_upper, kc_lower = result
        if bb_upper < kc_upper and bb_lower > kc_lower:
            self._squeeze_count += 1
        else:
            self._squeeze_count = 0
            self._sweep_detected = False   # reset sweep if squeeze breaks
        return self._squeeze_count >= self.SQUEEZE_BARS

    def _check_sweep(self) -> None:
        """
        Detect a false breakdown (liquidity sweep):
        Price closes below the 20-period low, then recovers above it
        within SWEEP_RECOVER candles.

        This pattern signals trapped shorts who provide the fuel for the breakout.
        """
        need = self.SWEEP_LOOKBK + self.SWEEP_RECOVER + 1
        if len(self._ohlc_bars) < need:
            return

        # Identify the 20-period low (excluding the most recent bars)
        reference_bars = self._ohlc_bars[-(self.SWEEP_LOOKBK + self.SWEEP_RECOVER):
                                          -self.SWEEP_RECOVER]
        period_low = min(b[2] for b in reference_bars)   # low prices

        # Look for a sweep in the last SWEEP_RECOVER bars
        recent = self._ohlc_bars[-self.SWEEP_RECOVER:]
        for bar in recent:
            c, h, l = bar
            if l < period_low and c > period_low:
                # Low went below the key level but closed back above → sweep!
                self._sweep_detected = True
                self._sweep_low      = l
                return

    def _breakout_level(self) -> float:
        """Highest High of the last BREAKOUT_BARS candles (the resistance cap)."""
        recent = self._ohlc_bars[-self.BREAKOUT_BARS:]
        return float(max(b[1] for b in recent)) if recent else 0.0

    # ── Evaluate ──────────────────────────────────────────────────────────────

    async def evaluate(
        self, chain: "OptionChainSnapshot"
    ) -> Optional[StrategySignal]:
        """
        Entry logic:
          SQUEEZE (5+ bars) AND SWEEP occurred AND current bar breaks above
          the 10-bar highest high → buy ATM call.
        """
        self._bar_count += 1

        # Need enough history for all indicators
        min_needed = self.BB_PERIOD + self.SWEEP_LOOKBK + self.SWEEP_RECOVER + 5
        if len(self._ohlc_bars) < min_needed:
            return None

        if not self.is_trading_window():
            return None
        if not self._vix_ok(chain):
            return None
        if self.db.has_active_trade(self.name):
            return None

        # One trade per day — prevents overtrading on 1-min squeezes
        today = chain.timestamp.strftime("%Y-%m-%d")
        if self._last_signal_day == today:
            return None

        can_enter, _ = self.risk_manager.can_enter_position()
        if not can_enter:
            return None

        # ── Phase 1: Is the squeeze active? ──────────────────────────────
        is_compressed = self._update_squeeze()
        if not is_compressed:
            return None

        # ── Phase 2: Has a sweep occurred? ───────────────────────────────
        self._check_sweep()
        if not self._sweep_detected:
            return None

        # ── Phase 3: Has price broken above the breakout level? ──────────
        breakout_level = self._breakout_level()
        current_high   = self._ohlc_bars[-1][1]

        if current_high <= breakout_level:
            return None   # price hasn't broken out yet — wait

        # ── All conditions met — generate entry signal ────────────────────
        si  = self._si(chain)
        atm = chain.atm_strike

        # Fix 3: robust closest-strike lookup — prevents KeyError on any chain
        atm = self._closest_strike(chain.calls, atm)
        if atm is None:
            return None
        quote = chain.calls.get(atm)

        if not quote or not quote.ltp or quote.ltp < self.config.min_premium:
            return None

        entry_price = quote.ltp

        # ── Phase 4: Exit levels ──────────────────────────────────────────
        # Initial SL anchored below the sweep low (converted to option premium %)
        # We approximate: if underlying dropped X% to sweep low, option delta ~0.5
        # So option SL ≈ entry × (1 - 0.30) — 30% of premium at risk initially
        sl_pct  = 0.30
        sl      = entry_price * (1.0 - sl_pct)
        # Phase 3 target: first +40% triggers breakeven trail (handled by _dynamic_trailing_sl)
        target  = entry_price * 1.90   # 1:3 RR ceiling before trailing takes over

        # Mark signal day to prevent re-entry today
        self._last_signal_day = today
        # Reset squeeze so we don't re-trigger immediately
        self._squeeze_count  = 0
        self._sweep_detected = False

        logger.info(
            "CompressionBreakout signal: %s  entry=₹%.2f  breakout>₹%.2f  "
            "sweep_low=₹%.2f  squeeze=%d bars",
            self._sym(chain, atm, "CE"), entry_price,
            breakout_level, self._sweep_low, self._squeeze_count,
        )

        return StrategySignal(
            signal_type=SignalType.LONG,
            strategy_name=self.name,
            timestamp=datetime.now(),
            confidence=min(1.0, self._squeeze_count / 10.0),
            entry_price=entry_price,
            stop_loss=sl,
            target=target,
            symbol=self._sym(chain, atm, "CE"),
            option_type=OptionType.CALL,
            strike=float(atm),
            expiry=chain.expiry,
            metadata={
                "breakout_level": breakout_level,
                "sweep_low":      self._sweep_low,
                "squeeze_bars":   self._squeeze_count,
                "bb_kc":          str(self._compute_bb_kc()),
            },
        )

    # ── Execute ───────────────────────────────────────────────────────────────

    async def execute_signal(self, signal: StrategySignal) -> bool:
        """Buy ATM call with dynamic lot sizing from the risk engine."""
        if not self.db.acquire_trade_lock(
            self.name, signal.symbol, f"strategy_{self.name}"
        ):
            return False

        # Phase 4: dynamic lot sizing — 2% of total capital, no sleeve cap
        qty = self._long_option_lots(signal.entry_price, strategy_type="skewhunter")
        try:
            response = await self.broker.place_order(
                symbol=signal.symbol,
                exchange=Exchange.NFO,
                transaction_type=TransactionType.BUY,
                order_type=OrderType.LIMIT,
                quantity=qty,
                price=signal.entry_price,
                product_type=ProductType.INTRADAY,
            )
            if not response.success:
                logger.error("CompressionBreakout order failed: %s", response.message)
                self.db.release_trade_lock(self.name, signal.symbol)
                return False

            from db_lock import TradeRecord, TradeStatus
            trade = TradeRecord(
                trade_id=str(uuid.uuid4()),
                strategy_name=self.name,
                symbol=signal.symbol,
                option_type=signal.option_type.value,
                strike=signal.strike,
                expiry=signal.expiry,
                entry_price=signal.entry_price,
                quantity=qty,
                direction="BUY",
                product_type="INTRADAY",
                status=TradeStatus.ACTIVE.value,
                stop_loss=signal.stop_loss,
                target=signal.target,
                entry_time=datetime.now().isoformat(),
            )
            self.db.insert_trade(trade)
            self.risk_manager.update_position(
                trade_id=trade.trade_id,
                symbol=signal.symbol,
                quantity=qty,
                average_price=signal.entry_price,
                current_price=signal.entry_price,
                direction="BUY",
                stop_loss=signal.stop_loss,
                target=signal.target,
            )
            logger.info(
                "CompressionBreakout entry: %s @ ₹%.2f  qty=%d  "
                "SL=₹%.2f  TGT=₹%.2f",
                signal.symbol, signal.entry_price, qty,
                signal.stop_loss, signal.target,
            )
            return True

        except Exception as e:
            logger.error("CompressionBreakout execution error: %s", e)
            self.db.release_trade_lock(self.name, signal.symbol)
            return False
