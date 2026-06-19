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
    
    async def manage_positions(self) -> None:
        """Monitor and manage active positions."""
        active_trades = self.db.get_active_trades(self.name)
        
        for trade in active_trades:
            # Get current price
            quote = await self.broker.get_quote(
                trade.symbol,
                Exchange.NFO
            )
            
            if not quote:
                continue
            
            current_price = quote.ltp
            
            # Check exit conditions
            should_exit, event = self.sl_manager.should_exit(
                trade.trade_id,
                current_price
            )
            
            if should_exit:
                await self._execute_exit(trade, current_price, event)
    
    async def _execute_exit(
        self,
        trade: TradeRecord,
        current_price: float,
        event: Optional[RiskEvent]
    ) -> bool:
        """Execute position exit."""
        if self.sl_manager.is_exit_pending(trade.trade_id):
            return False
        
        # Determine exit direction
        exit_direction = (
            TransactionType.SELL
            if trade.direction == "BUY"
            else TransactionType.BUY
        )
        
        # Place exit order
        response = await self.broker.place_order(
            symbol=trade.symbol,
            exchange=Exchange.NFO,
            transaction_type=exit_direction,
            order_type=OrderType.MARKET,
            quantity=trade.quantity,
            product_type=ProductType[trade.product_type]
        )
        
        if response.success:
            self.sl_manager.register_pending_exit(
                trade.trade_id,
                response.order_id,
                event.value if event else "MANUAL"
            )
            
            # Update trade status
            exit_reason = event.value if event else "SIGNAL_EXIT"
            self.db.update_trade_status(
                trade.trade_id,
                TradeStatus.CLOSED,
                exit_price=current_price,
                exit_reason=exit_reason
            )
            
            # Update risk manager
            direction_mult = 1 if trade.direction == "BUY" else -1
            pnl = (current_price - trade.entry_price) * trade.quantity * direction_mult
            self.risk_manager.remove_position(trade.trade_id, pnl)
            
            # Release database lock
            self.db.release_trade_lock(self.name, trade.symbol)
            
            logger.info(
                f"Exit executed for {trade.trade_id}: "
                f"price={current_price}, pnl={pnl:.2f}, reason={exit_reason}"
            )
            return True
        
        logger.error(f"Exit order failed: {response.message}")
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

        self._prev_skew: Dict[str, float] = {}
        # Rolling histories for z-score normalization of alpha signals
        self._alpha1_history: List[float] = []
        self._alpha2_history: List[float] = []
    
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
        
        # Moneyness levels: 0.25, 0.5, 0.75, 1.0 standard deviations
        std_dev = spot * 0.01  # Approximate 1% as 1 std for short-term
        
        otm_ivs = []
        itm_ivs = []
        
        for level in [0.25, 0.5, 0.75, 1.0]:
            if option_type == OptionType.CALL:
                # OTM calls have strike > spot
                otm_strike = round((atm + level * std_dev * 5) / 50) * 50
                itm_strike = round((atm - level * std_dev * 5) / 50) * 50
            else:
                # OTM puts have strike < spot
                otm_strike = round((atm - level * std_dev * 5) / 50) * 50
                itm_strike = round((atm + level * std_dev * 5) / 50) * 50
            
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
        
        # OTM IV (1 strike away)
        if option_type == OptionType.CALL:
            otm_strike = atm + 50
        else:
            otm_strike = atm - 50
        
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
        return float(1 / (1 + np.exp(-z)))

    def _compute_alpha2(self, chain: OptionChainSnapshot) -> float:
        """
        Alpha 2: ATM put-call IV parity deviation + ATM volume imbalance.
        Both signals are combined and z-score normalised before sigmoid.
        """
        atm = chain.atm_strike
        call_q = chain.calls.get(atm)
        put_q = chain.puts.get(atm)
        if not call_q or not put_q:
            return 0.5

        iv_delta = (call_q.iv - put_q.iv) if (call_q.iv and put_q.iv) else 0.0
        call_vol = max(1, call_q.volume or 1)
        put_vol = max(1, put_q.volume or 1)
        vol_ratio = (call_vol - put_vol) / (call_vol + put_vol)  # [-1, 1]

        raw = iv_delta + vol_ratio * 0.05
        self._alpha2_history.append(raw)
        if len(self._alpha2_history) > 60:
            self._alpha2_history.pop(0)
        if len(self._alpha2_history) < 5:
            return 0.5
        hist = np.array(self._alpha2_history)
        std = np.std(hist)
        if std < 1e-12:
            return 0.5
        z = (raw - np.mean(hist)) / std
        return float(1 / (1 + np.exp(-z)))
    
    async def evaluate(self, chain: OptionChainSnapshot) -> Optional[StrategySignal]:
        """Evaluate strategy conditions and generate a directional long-option signal."""
        if not self.is_trading_window():
            return None
        if self.db.has_active_trade(self.name):
            return None
        can_enter, reason = self.risk_manager.can_enter_position()
        if not can_enter:
            logger.debug(f"FixedRR blocked: {reason}")
            return None

        e_skew_call = self._compute_skew_energy(chain, OptionType.CALL)
        e_skew_put = self._compute_skew_energy(chain, OptionType.PUT)
        e_diff = self.analyzer.compute_energy_differential(e_skew_call, e_skew_put)
        call_skew_dir = self._compute_skew_direction(chain, OptionType.CALL)
        put_skew_dir = self._compute_skew_direction(chain, OptionType.PUT)
        alpha1 = self._compute_alpha1(e_diff, call_skew_dir, put_skew_dir)
        alpha2 = self._compute_alpha2(chain)

        # Long call: put stress (e_diff < 0), call skew expanding, both alphas high
        if (e_diff < 0
                and call_skew_dir > 0
                and alpha1 > self.config.fixed_rr_alpha1_long_threshold
                and alpha2 > self.config.fixed_rr_alpha2_long_threshold):
            return await self._create_long_call(chain, alpha1, alpha2, e_diff)

        # Long put: call stress (e_diff > 0), put skew contracting, both alphas low
        if (e_diff > 0
                and put_skew_dir < 0
                and alpha1 < self.config.fixed_rr_alpha1_short_threshold
                and alpha2 < self.config.fixed_rr_alpha2_short_threshold):
            return await self._create_long_put(chain, alpha1, alpha2, e_diff)

        return None

    async def _create_long_call(
        self,
        chain: OptionChainSnapshot,
        alpha1: float,
        alpha2: float,
        e_diff: float
    ) -> Optional[StrategySignal]:
        """Buy a slightly OTM call. SL=30%, Target=90% → 1:3 RR."""
        strike = chain.atm_strike + 50
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
            symbol=f"NFO:NIFTY{chain.expiry}{int(strike)}CE",
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
        """Buy a slightly OTM put. SL=30%, Target=90% → 1:3 RR."""
        strike = chain.atm_strike - 50
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
            symbol=f"NFO:NIFTY{chain.expiry}{int(strike)}PE",
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

        qty = self.risk_manager.thresholds.position_size_per_trade
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
            broker=broker
        )
    
    def is_entry_window(self) -> bool:
        """Check if current time is within overnight entry window."""
        now = datetime.now().time()
        start = self.parse_time(self.config.curvature_entry_start)
        end = self.parse_time(self.config.curvature_entry_end)
        return start <= now <= end
    
    def _compute_smile_curvature(
        self,
        chain: OptionChainSnapshot
    ) -> Tuple[float, float]:
        """Compute smile curvature for calls and puts."""
        strike_interval = 50.0
        atm = chain.atm_strike
        
        # Call curvature
        call_up = chain.calls.get(atm + strike_interval)
        call_atm = chain.calls.get(atm)
        call_down = chain.calls.get(atm - strike_interval)
        
        call_curvature = 0.0
        if all([call_up, call_atm, call_down]):
            if all([call_up.iv, call_atm.iv, call_down.iv]):
                call_curvature = self.bs.smile_curvature(
                    call_down.iv, call_atm.iv, call_up.iv, strike_interval
                )
        
        # Put curvature
        put_up = chain.puts.get(atm + strike_interval)
        put_atm = chain.puts.get(atm)
        put_down = chain.puts.get(atm - strike_interval)
        
        put_curvature = 0.0
        if all([put_up, put_atm, put_down]):
            if all([put_up.iv, put_atm.iv, put_down.iv]):
                put_curvature = self.bs.smile_curvature(
                    put_down.iv, put_atm.iv, put_up.iv, strike_interval
                )
        
        return call_curvature, put_curvature
    
    def _compute_viscosity(self, chain: OptionChainSnapshot) -> float:
        """Compute liquidity viscosity around ATM."""
        atm = chain.atm_strike
        
        bid_volumes = []
        ask_volumes = []
        
        for strike in [atm - 50, atm, atm + 50]:
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
        # Check entry window
        if not self.is_entry_window():
            return None
        
        # Check for existing positions
        if self.db.has_active_trade(self.name):
            return None
        
        # Compute curvature and viscosity
        call_curv, put_curv = self._compute_smile_curvature(chain)
        viscosity = self._compute_viscosity(chain)
        
        # Check curvature threshold
        max_curv = max(abs(call_curv), abs(put_curv))
        if max_curv < self.config.curvature_threshold:
            return None
        
        # Check viscosity threshold
        if abs(viscosity) < self.config.viscosity_threshold:
            return None
        
        # Determine direction based on viscosity
        if viscosity > 0:
            # More bids than asks -> bullish -> sell put spread
            return await self._create_put_credit_spread(chain, call_curv, viscosity)
        else:
            # More asks than bids -> bearish -> sell call spread
            return await self._create_call_credit_spread(chain, put_curv, viscosity)
    
    async def _create_put_credit_spread(
        self,
        chain: OptionChainSnapshot,
        curvature: float,
        viscosity: float
    ) -> Optional[SpreadSignal]:
        """Create put credit spread for overnight."""
        atm = chain.atm_strike
        
        # Sell ATM-50, Buy ATM-150 put spread
        sell_strike = atm - 50
        buy_strike = atm - 150
        
        sell_quote = chain.puts.get(sell_strike)
        buy_quote = chain.puts.get(buy_strike)
        
        if not sell_quote or not buy_quote:
            return None
        
        net_credit = sell_quote.ltp - buy_quote.ltp
        if net_credit <= 0:
            return None
        
        legs = [
            SpreadLeg(
                symbol=f"NIFTY{chain.expiry}{sell_strike}PE",
                strike=sell_strike,
                option_type=OptionType.PUT,
                direction=TransactionType.SELL,
                quantity=25,
                price=sell_quote.ltp
            ),
            SpreadLeg(
                symbol=f"NIFTY{chain.expiry}{buy_strike}PE",
                strike=buy_strike,
                option_type=OptionType.PUT,
                direction=TransactionType.BUY,
                quantity=25,
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
        
        sell_strike = atm + 50
        buy_strike = atm + 150
        
        sell_quote = chain.calls.get(sell_strike)
        buy_quote = chain.calls.get(buy_strike)
        
        if not sell_quote or not buy_quote:
            return None
        
        net_credit = sell_quote.ltp - buy_quote.ltp
        if net_credit <= 0:
            return None
        
        legs = [
            SpreadLeg(
                symbol=f"NIFTY{chain.expiry}{sell_strike}CE",
                strike=sell_strike,
                option_type=OptionType.CALL,
                direction=TransactionType.SELL,
                quantity=25,
                price=sell_quote.ltp
            ),
            SpreadLeg(
                symbol=f"NIFTY{chain.expiry}{buy_strike}CE",
                strike=buy_strike,
                option_type=OptionType.CALL,
                direction=TransactionType.BUY,
                quantity=25,
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
        executed_legs = []
        
        try:
            for i, leg in enumerate(signal.legs):
                response = await self.broker.place_order(
                    symbol=leg.symbol,
                    exchange=Exchange.NFO,
                    transaction_type=leg.direction,
                    order_type=OrderType.LIMIT,
                    quantity=leg.quantity,
                    price=leg.price,
                    product_type=ProductType.MARGIN  # NRML for overnight
                )
                
                if not response.success:
                    logger.error(f"Overnight leg {i} failed: {response.message}")
                    await self._unwind_legs(executed_legs)
                    self.db.release_trade_lock(self.name, signal.symbol)
                    return False
                
                executed_legs.append((leg, response))
                
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
                    parent_trade_id=parent_trade_id
                )
                self.db.insert_trade(trade)
            
            logger.info(f"Overnight spread executed: {signal.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Overnight execution error: {e}")
            await self._unwind_legs(executed_legs)
            self.db.release_trade_lock(self.name, signal.symbol)
            return False
    
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
    
    def _compute_alpha1(self, chain: OptionChainSnapshot) -> float:
        """
        Compute Alpha 1: OTM call volume ratio + OI changes vs ITM puts.
        """
        atm = chain.atm_strike
        
        # OTM calls (strikes above ATM)
        otm_call_volume = 0
        otm_call_oi_change = 0
        
        for strike in [atm + 50, atm + 100, atm + 150]:
            quote = chain.calls.get(strike)
            if quote:
                otm_call_volume += quote.volume
                key = f"CE_{strike}"
                prev_oi = self._prev_oi.get(key, quote.oi)
                otm_call_oi_change += quote.oi - prev_oi
                self._prev_oi[key] = quote.oi
        
        # ITM puts (strikes above ATM for puts)
        itm_put_volume = 0
        itm_put_oi_change = 0
        
        for strike in [atm + 50, atm + 100, atm + 150]:
            quote = chain.puts.get(strike)
            if quote:
                itm_put_volume += quote.volume
                key = f"PE_{strike}"
                prev_oi = self._prev_oi.get(key, quote.oi)
                itm_put_oi_change += quote.oi - prev_oi
                self._prev_oi[key] = quote.oi
        
        # Calculate ratio
        if itm_put_volume + itm_put_oi_change == 0:
            return 0.5
        
        raw = (otm_call_volume + otm_call_oi_change) / max(
            1, itm_put_volume + abs(itm_put_oi_change)
        )
        
        # Normalize to [0, 1]
        return 1 / (1 + np.exp(-(raw - 1) * 2))
    
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

        otm_call_iv = _iv(chain.calls, [atm + 50, atm + 100])
        itm_call_iv = _iv(chain.calls, [atm - 50, atm - 100])
        otm_put_iv  = _iv(chain.puts,  [atm - 50, atm - 100])
        itm_put_iv  = _iv(chain.puts,  [atm + 50, atm + 100])

        if None in (otm_call_iv, itm_call_iv, otm_put_iv, itm_put_iv):
            return 0.5

        call_skew = otm_call_iv - itm_call_iv
        put_skew  = otm_put_iv  - itm_put_iv
        net_skew  = call_skew - put_skew

        # Scale of 50: a 2% IV difference yields sigmoid(1.0) ≈ 0.73, giving clear signal
        return float(1 / (1 + np.exp(-net_skew * 50)))
    
    async def evaluate(
        self,
        chain: OptionChainSnapshot
    ) -> Optional[StrategySignal]:
        """Evaluate SkewHunter conditions."""
        if not self.is_trading_window():
            return None
        
        if self.db.has_active_trade(self.name):
            return None
        
        can_enter, reason = self.risk_manager.can_enter_position()
        if not can_enter:
            return None
        
        alpha1 = self._compute_alpha1(chain)
        alpha2 = self._compute_alpha2(chain)
        
        atm = chain.atm_strike
        
        # Long Call trigger
        if (alpha1 > self.config.skewhunter_alpha1_long and
            alpha2 > self.config.skewhunter_alpha2_long):
            
            # Buy slightly OTM call
            target_strike = atm + 50
            quote = chain.calls.get(target_strike)
            
            if quote and quote.ltp >= self.config.min_premium:
                return await self._create_long_signal(
                    chain, quote, target_strike, OptionType.CALL,
                    alpha1, alpha2
                )
        
        # Long Put trigger
        elif (alpha1 < self.config.skewhunter_alpha1_short and
              alpha2 < self.config.skewhunter_alpha2_short):
            
            # Buy slightly OTM put
            target_strike = atm - 50
            quote = chain.puts.get(target_strike)
            
            if quote and quote.ltp >= self.config.min_premium:
                return await self._create_long_signal(
                    chain, quote, target_strike, OptionType.PUT,
                    alpha1, alpha2
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
        
        symbol = f"NIFTY{chain.expiry}{int(strike)}{option_type.value}"
        
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
        
        try:
            response = await self.broker.place_order(
                symbol=signal.symbol,
                exchange=Exchange.NFO,
                transaction_type=TransactionType.BUY,
                order_type=OrderType.LIMIT,
                quantity=25,
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
                quantity=25,
                direction="BUY",
                product_type="INTRADAY",
                status=TradeStatus.ACTIVE.value,
                stop_loss=signal.stop_loss,
                target=signal.target,
                entry_time=datetime.now().isoformat()
            )
            self.db.insert_trade(trade)
            
            # Register with risk manager
            self.risk_manager.update_position(
                trade_id=trade.trade_id,
                symbol=signal.symbol,
                quantity=25,
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



