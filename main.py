#!/usr/bin/env python3
"""
Main Execution Module for Indian Options Algorithmic Trading System.
Handles WebSocket data, strategy orchestration, and lifecycle management.
"""

import asyncio
import signal
import sys
import logging
from datetime import datetime, timedelta, time as dt_time
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
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # WS tick LTP cache: symbol → last known LTP (avoids HTTP polling)
        self._tick_ltp: Dict[str, float] = {}
    
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
    
    # ── Expiry / TTE helpers ──────────────────────────────────────────────────

    def _get_nearest_weekly_expiry(self) -> datetime:
        """Return the next NSE weekly expiry (Thursday). If today is Thursday
        and market has closed, roll to next week."""
        today = datetime.now()
        days_ahead = (3 - today.weekday()) % 7  # 3 = Thursday
        if days_ahead == 0 and today.time() >= dt_time(15, 30):
            days_ahead = 7
        return today + timedelta(days=days_ahead)

    def _expiry_key(self) -> str:
        """Canonical expiry string used as the chain-manager dict key.
        Format: DDMMMYY  e.g. '19JUN25'.  Used consistently everywhere."""
        return self._get_nearest_weekly_expiry().strftime("%d%b%y").upper()

    def _compute_tte(self, expiry_dt: datetime) -> float:
        """Time-to-expiry in years counting only Mon-Fri trading days."""
        now = datetime.now()
        current = now
        trading_days = 0.0
        while current.date() < expiry_dt.date():
            if current.weekday() < 5:
                trading_days += 1
            current += timedelta(days=1)
        # Fractional day: fraction of 6h15m NSE session remaining today
        if expiry_dt.date() >= now.date() and now.weekday() < 5:
            session_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            session_open  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
            if now < session_close:
                elapsed = max(0, (now - session_open).total_seconds())
                session_len = (session_close - session_open).total_seconds()
                trading_days += max(0.0, 1.0 - elapsed / session_len)
        return max(1 / 252, trading_days / 252)

    # ── Tick handling ─────────────────────────────────────────────────────────

    def _on_tick(self, tick: Tick) -> None:
        """Handle incoming tick data."""
        try:
            # Always cache LTP for position monitor (avoids HTTP polling)
            self._tick_ltp[tick.symbol] = tick.ltp

            # Update bar aggregator
            self.bar_aggregator.process_tick(tick)

            # Update chain manager if option tick
            if "CE" in tick.symbol or "PE" in tick.symbol:
                self._update_option_chain(tick)

            # Update spot price
            if tick.symbol.endswith(f"{self.config.underlying_symbol}-EQ") or \
               tick.symbol == self.config.underlying_symbol:
                self.chain_manager.update_spot(tick.ltp)

        except Exception as e:
            self.logger.error(f"Tick processing error: {e}")
    
    def _update_option_chain(self, tick: Tick) -> None:
        """Update option chain from tick data."""
        try:
            symbol = tick.symbol
            option_type = OptionType.CALL if symbol.endswith("CE") else OptionType.PUT
            suffix = "CE" if option_type == OptionType.CALL else "PE"

            # Strip exchange prefix e.g. "NFO:" → bare symbol
            bare = symbol.split(":")[-1]

            # Extract numeric strike from the end: everything before CE/PE suffix
            bare_no_suffix = bare[:-2]  # remove "CE" or "PE"
            strike_str = ""
            for ch in reversed(bare_no_suffix):
                if ch.isdigit():
                    strike_str = ch + strike_str
                else:
                    break
            if not strike_str:
                return
            strike = float(strike_str)

            # Use the canonical expiry key (same format used in _subscribe_symbols)
            expiry = self._expiry_key()
            expiry_dt = self._get_nearest_weekly_expiry()
            time_to_expiry = self._compute_tte(expiry_dt)

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
                time_to_expiry=time_to_expiry,
            )

        except Exception as e:
            self.logger.debug(f"Option chain update error: {e}")
    
    def _on_bar_complete(self, bar: OHLCV) -> None:
        """Handle completed 1-minute bar."""
        # Update strategies with new bar
        for strategy in self.strategies:
            strategy.update_bar(bar)
    
    def _handle_risk_event(self, event: RiskEvent, data: Dict) -> None:
        """Handle risk management events.

        This may be called from a threading.Lock context inside RiskManager,
        so we must never call asyncio.create_task() directly here.
        Instead, schedule the coroutine onto the event loop thread-safely.
        """
        self.logger.warning(f"Risk event: {event.value}, data: {data}")
        if event in (RiskEvent.MAX_DRAWDOWN, RiskEvent.DAILY_LOSS_LIMIT):
            if self._loop and self._loop.is_running():
                ev_val = event.value
                self._loop.call_soon_threadsafe(
                    lambda: self._loop.create_task(self._emergency_shutdown(ev_val))
                )
    
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
                
                # Get current option chain — use the same key as _update_option_chain
                expiry = self._expiry_key()
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
        """Monitor positions for SL/TP triggers.

        Reads LTP from the WebSocket tick cache (_tick_ltp) so we never burn
        the SEBI 9-OPS rate limit on HTTP quote polling.
        Falls back to a broker REST call only when the symbol is missing from
        the cache (e.g. a freshly subscribed symbol with no tick yet).
        """
        self.logger.info("Position monitor started")

        while self._running:
            try:
                for position in self.risk_manager.get_positions():
                    ltp = self._tick_ltp.get(position.symbol)

                    if ltp is None:
                        # One-time REST fallback — cache the result immediately
                        quote = await self.broker.get_quote(position.symbol, Exchange.NFO)
                        if quote:
                            ltp = quote.ltp
                            self._tick_ltp[position.symbol] = ltp

                    if ltp is not None:
                        event = self.risk_manager.update_price(position.trade_id, ltp)
                        if event:
                            self.logger.info(
                                f"Position event: {position.symbol} event={event.value}"
                            )

                await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Position monitor error: {e}")
                await asyncio.sleep(1)

        self.logger.info("Position monitor stopped")
    
    async def _subscribe_symbols(self) -> None:
        """Subscribe to Nifty spot + options around ATM for the nearest weekly expiry."""
        # Canonical expiry string — must match what _update_option_chain uses
        expiry = self._expiry_key()          # e.g. "19JUN25"
        expiry_dt = self._get_nearest_weekly_expiry()

        # Fyers weekly option symbol format: NFO:NIFTY{YY}{M}{DD}{strike}CE
        # where M is single digit for months 1-9, or O/N/D for Oct/Nov/Dec
        month_map = {10: "O", 11: "N", 12: "D"}
        m = expiry_dt.month
        month_char = month_map.get(m, str(m))
        yy = expiry_dt.strftime("%y")
        dd = expiry_dt.strftime("%d")
        fyers_expiry = f"{yy}{month_char}{dd}"   # e.g. "2561​9"

        symbols = [f"NSE:{self.config.underlying_symbol}50-INDEX"]  # Nifty spot

        spot_quote = await self.broker.get_quote(
            f"{self.config.underlying_symbol}50-INDEX", Exchange.NSE
        )
        if spot_quote:
            atm = round(spot_quote.ltp / 50) * 50
            und = self.config.underlying_symbol  # "NIFTY"
            for offset in range(-200, 250, 50):
                strike = int(atm + offset)
                symbols.append(f"NFO:{und}{fyers_expiry}{strike}CE")
                symbols.append(f"NFO:{und}{fyers_expiry}{strike}PE")

        self.logger.info(f"Subscribing to {len(symbols)} symbols (expiry={expiry})")
        await self.ws_feed.subscribe(symbols)
    
    async def run(self) -> None:
        """Run the trading engine."""
        self._running = True
        self._loop = asyncio.get_running_loop()  # stored for thread-safe scheduling

        # Setup signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._loop.add_signal_handler(sig, self._signal_handler)
        
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


