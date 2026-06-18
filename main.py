#!/usr/bin/env python3
"""
Main Execution Module for Indian Options Algorithmic Trading System.
Handles WebSocket data, strategy orchestration, and lifecycle management.
"""

import asyncio
import signal
import sys
import logging
from datetime import datetime, time as dt_time
from typing import Optional, List, Dict

from config import (
    AppConfig, setup_logging, BrokerType, Exchange
)
from bs_engine import BlackScholesEngine, OptionType
from db_lock import TradingDatabase
from compliance import ComplianceManager
from risk_manager import RiskManager, RiskThresholds, RiskEvent
from broker_gateway import create_broker_gateway, BrokerGateway
from data_feed import (
    WebSocketFeed, BarAggregator, OptionChainManager, Tick, OHLCV
)
from strategies import (
    FixedRR13Strategy, CurvatureCreditSpreadStrategy, SkewHunterStrategy,
    BaseStrategy, StrategySignal
)


class TradingEngine:
    """
    Main trading engine orchestrating all components.
    """
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.logger = setup_logging(config.logging)
        
        # Core components
        self.db: Optional[TradingDatabase] = None
        self.compliance: Optional[ComplianceManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.broker: Optional[BrokerGateway] = None
        self.bs_engine: Optional[BlackScholesEngine] = None
        
        # Data components
        self.ws_feed: Optional[WebSocketFeed] = None
        self.bar_aggregator: Optional[BarAggregator] = None
        self.chain_manager: Optional[OptionChainManager] = None
        
        # Strategies
        self.strategies: List[BaseStrategy] = []
        
        # State
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
    
    async def initialize(self) -> bool:
        """Initialize all trading components."""
        self.logger.info("Initializing trading engine...")
        
        try:
            # Initialize database
            self.db = TradingDatabase(
                self.config.database.db_path,
                self.config.database.lock_timeout
            )
            
            # Check if trading is locked
            if self.db.is_trading_locked():
                self.logger.error("Trading is locked for today")
                return False
            
            # Initialize compliance manager
            self.compliance = ComplianceManager(
                whitelisted_ip=self.config.compliance.whitelisted_ip,
                totp_key=self.config.compliance.totp_key,
                max_ops=self.config.compliance.max_orders_per_second,
                enable_ip_check=self.config.compliance.enable_ip_check,
                login_time=self.config.compliance.daily_login_time
            )
            
            # Validate environment
            if not await self.compliance.validate_environment():
                self.logger.error("Compliance validation failed")
                return False
            
            # Initialize broker gateway
            self.broker = create_broker_gateway(
                self.config.broker,
                self.compliance
            )
            
            # Authenticate with broker
            if not await self.broker.authenticate():
                self.logger.error("Broker authentication failed")
                return False
            
            # Initialize Black-Scholes engine
            self.bs_engine = BlackScholesEngine(self.config.risk_free_rate)
            
            # Initialize risk manager
            capital = 500000.0  # Default capital, should be from account
            self.risk_manager = RiskManager(
                database=self.db,
                capital=capital,
                thresholds=RiskThresholds(
                    max_daily_loss_percent=self.config.risk.max_daily_loss_percent,
                    max_drawdown_percent=self.config.risk.max_drawdown_percent,
                    trailing_drawdown_percent=self.config.risk.trailing_drawdown_percent,
                    max_position_size=self.config.risk.max_position_size,
                    max_open_positions=self.config.risk.max_open_positions,
                    position_size_per_trade=self.config.risk.position_size_per_trade
                ),
                emergency_square_off_time=self.config.risk.emergency_square_off_time,
                on_risk_event=self._handle_risk_event
            )
            
            # Initialize option chain manager
            self.chain_manager = OptionChainManager(
                bs_engine=self.bs_engine,
                underlying=self.config.underlying_symbol,
                strike_interval=50.0
            )
            
            # Initialize bar aggregator
            self.bar_aggregator = BarAggregator(
                on_bar_complete=self._on_bar_complete
            )
            
            # Initialize strategies
            self._initialize_strategies()
            
            self.logger.info("Trading engine initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def _initialize_strategies(self) -> None:
        """Initialize all trading strategies."""
        # Strategy 1: Fixed RR 1:3
        self.strategies.append(FixedRR13Strategy(
            config=self.config.strategy,
            bs_engine=self.bs_engine,
            database=self.db,
            risk_manager=self.risk_manager,
            broker=self.broker
        ))
        
        # Strategy 2: Curvature Credit Spread Overnight
        self.strategies.append(CurvatureCreditSpreadStrategy(
            config=self.config.strategy,
            bs_engine=self.bs_engine,
            database=self.db,
            risk_manager=self.risk_manager,
            broker=self.broker
        ))
        
        # Strategy 3: SkewHunter
        self.strategies.append(SkewHunterStrategy(
            config=self.config.strategy,
            bs_engine=self.bs_engine,
            database=self.db,
            risk_manager=self.risk_manager,
            broker=self.broker
        ))
        
        self.logger.info(f"Initialized {len(self.strategies)} strategies")
    
    def _on_tick(self, tick: Tick) -> None:
        """Handle incoming tick data."""
        try:
            # Update bar aggregator
            completed_bar = self.bar_aggregator.process_tick(tick)
            
            # Update chain manager if option tick
            if "CE" in tick.symbol or "PE" in tick.symbol:
                self._update_option_chain(tick)
            
            # Update spot price
            if tick.symbol == f"{self.config.underlying_symbol}-EQ":
                self.chain_manager.update_spot(tick.ltp)
                
        except Exception as e:
            self.logger.error(f"Tick processing error: {e}")
    
    def _update_option_chain(self, tick: Tick) -> None:
        """Update option chain from tick."""
        try:
            # Parse option symbol (e.g., NIFTY24JUN23000CE)
            symbol = tick.symbol
            
            # Extract components (simplified parsing)
            option_type = OptionType.CALL if "CE" in symbol else OptionType.PUT
            
            # Extract strike (last 5 digits before CE/PE typically)
            strike_str = ""
            for i in range(len(symbol) - 2, 0, -1):
                if symbol[i].isdigit():
                    strike_str = symbol[i] + strike_str
                else:
                    break
            
            if not strike_str:
                return
            
            strike = float(strike_str)
            
            # Extract expiry (between underlying and strike)
            # This is simplified - actual parsing depends on symbol format
            expiry = datetime.now().strftime("%d%b%Y").upper()
            
            # Calculate time to expiry
            # Simplified: assume weekly expiry
            days_to_expiry = 7  # Would need proper expiry calendar
            time_to_expiry = days_to_expiry / 365
            
            self.chain_manager.update_option_quote(
                symbol=symbol,
                strike=strike,
                expiry=expiry,
                option_type=option_type,
                ltp=tick.ltp,
                bid=tick.bid,
                ask=tick.ask,
                bid_qty=tick.bid_qty,
                ask_qty=tick.ask_qty,
                volume=tick.volume,
                oi=tick.oi,
                time_to_expiry=time_to_expiry
            )
            
        except Exception as e:
            self.logger.debug(f"Option chain update error: {e}")
    
    def _on_bar_complete(self, bar: OHLCV) -> None:
        """Handle completed 1-minute bar."""
        # Update strategies with new bar
        for strategy in self.strategies:
            strategy.update_bar(bar)
    
    def _handle_risk_event(self, event: RiskEvent, data: Dict) -> None:
        """Handle risk management events."""
        self.logger.warning(f"Risk event: {event.value}, data: {data}")
        
        if event in (RiskEvent.MAX_DRAWDOWN, RiskEvent.DAILY_LOSS_LIMIT):
            # Trigger emergency shutdown
            asyncio.create_task(self._emergency_shutdown(event.value))
    
    async def _emergency_shutdown(self, reason: str) -> None:
        """Execute emergency shutdown."""
        self.logger.critical(f"Emergency shutdown triggered: {reason}")
        
        async def execute_order(symbol, direction, quantity, order_type):
            from config import TransactionType, OrderType as OT
            response = await self.broker.place_order(
                symbol=symbol,
                exchange=Exchange.NFO,
                transaction_type=TransactionType[direction],
                order_type=OT[order_type],
                quantity=quantity
            )
            return response.order_id if response.success else None
        
        await self.risk_manager.execute_emergency_shutdown(
            order_executor=execute_order,
            reason=reason
        )
    
    async def _strategy_loop(self) -> None:
        """Main strategy evaluation loop."""
        self.logger.info("Strategy loop started")
        
        while self._running:
            try:
                # Check market hours
                now = datetime.now().time()
                market_open = dt_time(9, 15)
                market_close = dt_time(15, 30)
                
                if not (market_open <= now <= market_close):
                    await asyncio.sleep(60)
                    continue
                
                # Get current option chain
                expiry = datetime.now().strftime("%d%b%Y").upper()
                chain = self.chain_manager.get_chain(expiry)
                
                if not chain or chain.spot_price <= 0:
                    await asyncio.sleep(1)
                    continue
                
                # Evaluate each strategy
                for strategy in self.strategies:
                    try:
                        signal = await strategy.evaluate(chain)
                        
                        if signal:
                            self.logger.info(
                                f"Signal generated: {strategy.name}, "
                                f"type={signal.signal_type.value}, "
                                f"confidence={signal.confidence:.3f}"
                            )
                            
                            # Execute signal
                            success = await strategy.execute_signal(signal)
                            
                            if success:
                                self.logger.info(
                                    f"Trade executed: {signal.symbol}"
                                )
                        
                        # Manage existing positions
                        await strategy.manage_positions()
                        
                    except Exception as e:
                        self.logger.error(
                            f"Strategy {strategy.name} error: {e}"
                        )
                
                # Check for SkewHunter mandatory square-off
                for strategy in self.strategies:
                    if isinstance(strategy, SkewHunterStrategy):
                        await strategy.check_mandatory_squareoff()
                
                # Check emergency square-off time
                if self.risk_manager.is_square_off_time():
                    await self._emergency_shutdown("Scheduled square-off time")
                    break
                
                await asyncio.sleep(1)  # 1-second evaluation cycle
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Strategy loop error: {e}")
                await asyncio.sleep(5)
        
        self.logger.info("Strategy loop stopped")
    
    async def _position_monitor_loop(self) -> None:
        """Monitor positions for SL/TP triggers."""
        self.logger.info("Position monitor started")
        
        while self._running:
            try:
                # Update position prices
                positions = self.risk_manager.get_positions()
                
                for position in positions:
                    quote = await self.broker.get_quote(
                        position.symbol,
                        Exchange.NFO
                    )
                    
                    if quote:
                        event = self.risk_manager.update_price(
                            position.trade_id,
                            quote.ltp
                        )
                        
                        if event:
                            self.logger.info(
                                f"Position event: {position.symbol}, "
                                f"event={event.value}"
                            )
                
                await asyncio.sleep(0.5)  # 500ms monitoring cycle
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Position monitor error: {e}")
                await asyncio.sleep(1)
        
        self.logger.info("Position monitor stopped")
    
    async def _subscribe_symbols(self) -> None:
        """Subscribe to required symbols."""
        # Get current expiry symbols
        expiry = datetime.now().strftime("%y%b").upper()
        
        # Generate symbol list
        symbols = [
            f"NSE:{self.config.underlying_symbol}-EQ",  # Spot
        ]
        
        # Add option strikes around ATM
        spot_quote = await self.broker.get_quote(
            f"{self.config.underlying_symbol}-EQ",
            Exchange.NSE
        )
        
        if spot_quote:
            atm = round(spot_quote.ltp / 50) * 50
            
            for offset in [-200, -150, -100, -50, 0, 50, 100, 150, 200]:
                strike = int(atm + offset)
                symbols.append(f"NFO:{self.config.underlying_symbol}{expiry}{strike}CE")
                symbols.append(f"NFO:{self.config.underlying_symbol}{expiry}{strike}PE")
        
        self.logger.info(f"Subscribing to {len(symbols)} symbols")
        await self.ws_feed.subscribe(symbols)
    
    async def run(self) -> None:
        """Run the trading engine."""
        self._running = True
        
        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._signal_handler)
        
        try:
            # Initialize WebSocket feed
            session_token = self.compliance.get_session_token()
            
            self.ws_feed = WebSocketFeed(
                ws_url=self.config.broker.ws_url,
                session_token=session_token,
                on_tick=self._on_tick,
                broker_type=self.config.broker.broker_type
            )
            
            # Start WebSocket feed
            ws_task = asyncio.create_task(self.ws_feed.run())
            self._tasks.append(ws_task)
            
            # Wait for connection and subscribe
            await asyncio.sleep(2)
            await self._subscribe_symbols()
            
            # Start strategy and monitor loops
            strategy_task = asyncio.create_task(self._strategy_loop())
            monitor_task = asyncio.create_task(self._position_monitor_loop())
            
            self._tasks.extend([strategy_task, monitor_task])
            
            self.logger.info("Trading engine running")
            
            # Wait for shutdown signal
            await self._shutdown_event.wait()
            
        except Exception as e:
            self.logger.error(f"Engine error: {e}")
        finally:
            await self.shutdown()
    
    def _signal_handler(self) -> None:
        """Handle shutdown signals."""
        self.logger.info("Shutdown signal received")
        self._running = False
        self._shutdown_event.set()
    
    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self.logger.info("Shutting down trading engine...")
        
        self._running = False
        
        # Stop strategies
        for strategy in self.strategies:
            strategy.stop()
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Stop WebSocket
        if self.ws_feed:
            await self.ws_feed.stop()
        
        # Flush pending bars
        if self.bar_aggregator:
            self.bar_aggregator.flush_all()
        
        # Close broker connection
        if self.broker:
            await self.broker.close()
        
        # Close database
        if self.db:
            self.db.close()
        
        self.logger.info("Trading engine shutdown complete")


async def main():
    """Main entry point."""
    print("=" * 60)
    print("Indian Options Algorithmic Trading System")
    print("=" * 60)
    
    # Load configuration
    config = AppConfig.from_env()
    
    # Create and initialize engine
    engine = TradingEngine(config)
    
    if not await engine.initialize():
        print("Failed to initialize trading engine")
        sys.exit(1)
    
    # Run engine
    await engine.run()


if __name__ == "__main__":
    asyncio.run(main())


