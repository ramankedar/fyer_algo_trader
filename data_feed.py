"""
Market Data Feed Module: WebSocket streaming and 1-minute bar aggregation.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import struct

import websockets
from websockets.exceptions import ConnectionClosed

from config import BrokerConfig, BrokerType, Exchange
from bs_engine import BlackScholesEngine, OptionType, OptionChainAnalyzer

logger = logging.getLogger("trading_system.datafeed")


@dataclass
class Tick:
    symbol: str
    ltp: float
    bid: float
    ask: float
    bid_qty: int
    ask_qty: int
    volume: int
    oi: int
    timestamp: datetime


@dataclass
class OHLCV:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class OptionQuote:
    symbol: str
    strike: float
    expiry: str
    option_type: OptionType
    ltp: float
    bid: float
    ask: float
    bid_qty: int
    ask_qty: int
    volume: int
    oi: int
    iv: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None


@dataclass
class OptionChainSnapshot:
    underlying: str
    spot_price: float
    timestamp: datetime
    expiry: str
    atm_strike: float
    calls: Dict[float, OptionQuote] = field(default_factory=dict)
    puts: Dict[float, OptionQuote] = field(default_factory=dict)


class BarAggregator:
    """Aggregates ticks into 1-minute OHLCV bars."""
    
    def __init__(
        self,
        on_bar_complete: Callable[[OHLCV], None],
        bar_interval_seconds: int = 60
    ):
        self.on_bar_complete = on_bar_complete
        self.bar_interval = bar_interval_seconds
        self._current_bars: Dict[str, dict] = {}
        self._last_bar_time: Dict[str, datetime] = {}
    
    def _get_bar_start_time(self, timestamp: datetime) -> datetime:
        """Get the start time of the bar containing this timestamp."""
        seconds = timestamp.second + timestamp.microsecond / 1_000_000
        bar_start_second = (int(timestamp.timestamp()) // self.bar_interval) * self.bar_interval
        return datetime.fromtimestamp(bar_start_second)
    
    def process_tick(self, tick: Tick) -> Optional[OHLCV]:
        """
        Process a tick and potentially emit a completed bar.
        Returns completed bar if one was finalized.
        """
        symbol = tick.symbol
        bar_start = self._get_bar_start_time(tick.timestamp)
        
        # Check if this tick belongs to a new bar
        if symbol in self._last_bar_time:
            if bar_start > self._last_bar_time[symbol]:
                # Complete the previous bar
                completed = self._finalize_bar(symbol)
                
                # Start new bar
                self._start_new_bar(symbol, tick, bar_start)
                
                return completed
        else:
            # First tick for this symbol
            self._start_new_bar(symbol, tick, bar_start)
            return None
        
        # Update current bar
        self._update_bar(symbol, tick)
        return None
    
    def _start_new_bar(self, symbol: str, tick: Tick, bar_start: datetime) -> None:
        """Start a new bar."""
        self._current_bars[symbol] = {
            "open": tick.ltp,
            "high": tick.ltp,
            "low": tick.ltp,
            "close": tick.ltp,
            "volume": tick.volume,
            "timestamp": bar_start
        }
        self._last_bar_time[symbol] = bar_start
    
    def _update_bar(self, symbol: str, tick: Tick) -> None:
        """Update current bar with new tick."""
        bar = self._current_bars[symbol]
        bar["high"] = max(bar["high"], tick.ltp)
        bar["low"] = min(bar["low"], tick.ltp)
        bar["close"] = tick.ltp
        bar["volume"] = tick.volume  # Cumulative
    
    def _finalize_bar(self, symbol: str) -> OHLCV:
        """Finalize and return the current bar."""
        bar = self._current_bars[symbol]
        ohlcv = OHLCV(
            symbol=symbol,
            timestamp=bar["timestamp"],
            open=bar["open"],
            high=bar["high"],
            low=bar["low"],
            close=bar["close"],
            volume=bar["volume"]
        )
        
        # Call callback
        self.on_bar_complete(ohlcv)
        
        return ohlcv
    
    def flush_all(self) -> List[OHLCV]:
        """Flush all pending bars."""
        bars = []
        for symbol in list(self._current_bars.keys()):
            bars.append(self._finalize_bar(symbol))
        return bars


class OptionChainManager:
    """
    Manages option chain data with Greeks calculation.
    """
    
    def __init__(
        self,
        bs_engine: BlackScholesEngine,
        underlying: str = "NIFTY",
        strike_interval: float = 50.0
    ):
        self.bs = bs_engine
        self.analyzer = OptionChainAnalyzer(bs_engine)
        self.underlying = underlying
        self.strike_interval = strike_interval
        
        self._spot_price: float = 0.0
        self._chain_data: Dict[str, OptionChainSnapshot] = {}
        self._subscribers: List[Callable[[OptionChainSnapshot], None]] = []
    
    def update_spot(self, price: float) -> None:
        """Update underlying spot price."""
        self._spot_price = price
    
    def update_option_quote(
        self,
        symbol: str,
        strike: float,
        expiry: str,
        option_type: OptionType,
        ltp: float,
        bid: float,
        ask: float,
        bid_qty: int,
        ask_qty: int,
        volume: int,
        oi: int,
        time_to_expiry: float  # In years
    ) -> None:
        """Update option quote and calculate Greeks."""
        # Calculate IV
        mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else ltp
        
        iv = self.bs.implied_volatility_newton_raphson(
            market_price=mid_price,
            spot=self._spot_price,
            strike=strike,
            time_to_expiry=time_to_expiry,
            option_type=option_type
        )
        
        # Calculate Greeks if IV available
        delta = gamma = theta = vega = None
        if iv:
            delta = self.bs.delta(
                self._spot_price, strike, time_to_expiry, iv, option_type
            )
            gamma = self.bs.gamma(
                self._spot_price, strike, time_to_expiry, iv
            )
            theta = self.bs.theta(
                self._spot_price, strike, time_to_expiry, iv, option_type
            )
            vega = self.bs.vega(
                self._spot_price, strike, time_to_expiry, iv
            )
        
        quote = OptionQuote(
            symbol=symbol,
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            ltp=ltp,
            bid=bid,
            ask=ask,
            bid_qty=bid_qty,
            ask_qty=ask_qty,
            volume=volume,
            oi=oi,
            iv=iv,
            delta=float(delta) if delta else None,
            gamma=float(gamma) if gamma else None,
            theta=float(theta) if theta else None,
            vega=float(vega) if vega else None
        )
        
        # Update chain snapshot
        if expiry not in self._chain_data:
            self._chain_data[expiry] = OptionChainSnapshot(
                underlying=self.underlying,
                spot_price=self._spot_price,
                timestamp=datetime.now(),
                expiry=expiry,
                atm_strike=self.analyzer.get_atm_strike(
                    self._spot_price, self.strike_interval
                )
            )
        
        snapshot = self._chain_data[expiry]
        snapshot.spot_price = self._spot_price
        snapshot.timestamp = datetime.now()
        snapshot.atm_strike = self.analyzer.get_atm_strike(
            self._spot_price, self.strike_interval
        )
        
        if option_type == OptionType.CALL:
            snapshot.calls[strike] = quote
        else:
            snapshot.puts[strike] = quote
        
        # Notify subscribers
        for callback in self._subscribers:
            callback(snapshot)
    
    def get_chain(self, expiry: str) -> Optional[OptionChainSnapshot]:
        """Get option chain snapshot for expiry."""
        return self._chain_data.get(expiry)
    
    def get_atm_strike(self) -> float:
        """Get current ATM strike."""
        return self.analyzer.get_atm_strike(self._spot_price, self.strike_interval)
    
    def subscribe(self, callback: Callable[[OptionChainSnapshot], None]) -> None:
        """Subscribe to chain updates."""
        self._subscribers.append(callback)
    
    def get_iv_surface(self, expiry: str) -> Dict[str, Dict[float, float]]:
        """
        Get IV surface for an expiry.
        Returns {"calls": {strike: iv}, "puts": {strike: iv}}
        """
        chain = self._chain_data.get(expiry)
        if not chain:
            return {"calls": {}, "puts": {}}
        
        return {
            "calls": {s: q.iv for s, q in chain.calls.items() if q.iv},
            "puts": {s: q.iv for s, q in chain.puts.items() if q.iv}
        }
    
    def compute_smile_curvature(
        self,
        expiry: str,
        option_type: OptionType
    ) -> Optional[float]:
        """
        Compute volatility smile curvature at ATM.
        """
        chain = self._chain_data.get(expiry)
        if not chain:
            return None
        
        atm = chain.atm_strike
        up_strike = atm + self.strike_interval
        down_strike = atm - self.strike_interval
        
        quotes = chain.calls if option_type == OptionType.CALL else chain.puts
        
        if atm not in quotes or up_strike not in quotes or down_strike not in quotes:
            return None
        
        iv_atm = quotes[atm].iv
        iv_up = quotes[up_strike].iv
        iv_down = quotes[down_strike].iv
        
        if not all([iv_atm, iv_up, iv_down]):
            return None
        
        return self.bs.smile_curvature(iv_down, iv_atm, iv_up, self.strike_interval)
    
    def compute_viscosity(self, expiry: str) -> float:
        """
        Compute liquidity viscosity (bid-ask imbalance) at ATM.
        """
        chain = self._chain_data.get(expiry)
        if not chain:
            return 0.0
        
        atm = chain.atm_strike
        
        # Collect bid/ask volumes around ATM
        bid_volumes = []
        ask_volumes = []
        
        for strike in [atm - self.strike_interval, atm, atm + self.strike_interval]:
            for quotes in [chain.calls, chain.puts]:
                if strike in quotes:
                    bid_volumes.append(quotes[strike].bid_qty)
                    ask_volumes.append(quotes[strike].ask_qty)
        
        import numpy as np
        return self.analyzer.compute_viscosity(
            np.array(bid_volumes),
            np.array(ask_volumes)
        )


class WebSocketFeed:
    """
    WebSocket-based market data feed.
    """
    
    def __init__(
        self,
        ws_url: str,
        session_token: str,
        on_tick: Callable[[Tick], None],
        broker_type: BrokerType = BrokerType.FYERS
    ):
        self.ws_url = ws_url
        self.session_token = session_token
        self.on_tick = on_tick
        self.broker_type = broker_type
        
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._subscribed_symbols: set = set()
        self._running = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
    
    async def connect(self) -> bool:
        """Establish WebSocket connection."""
        try:
            if self.broker_type == BrokerType.FYERS:
                headers = {"Authorization": f"Bearer {self.session_token}"}
            else:
                headers = {"access-token": self.session_token}
            
            self._ws = await websockets.connect(
                self.ws_url,
                extra_headers=headers,
                ping_interval=30,
                ping_timeout=10
            )
            
            logger.info(f"WebSocket connected to {self.ws_url}")
            self._running = True
            self._reconnect_delay = 1.0
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def subscribe(self, symbols: List[str]) -> bool:
        """Subscribe to symbols."""
        if not self._ws:
            return False
        
        try:
            if self.broker_type == BrokerType.FYERS:
                # Send all symbols in one message; L=2 = Quote mode (LTP + bid/ask)
                message = {"T": "SUB_DATA", "L": "2", "SLIST": symbols}
                await self._ws.send(json.dumps(message))
            else:  # Dhan
                message = {
                    "RequestCode": 15,
                    "InstrumentCount": len(symbols),
                    "InstrumentList": [
                        {"ExchangeSegment": "NSE_FNO", "SecurityId": s}
                        for s in symbols
                    ],
                }
                await self._ws.send(json.dumps(message))

            for s in symbols:
                self._subscribed_symbols.add(s)

            logger.info(f"Subscribed to {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Subscribe failed: {e}")
            return False
    
    async def unsubscribe(self, symbols: List[str]) -> bool:
        """Unsubscribe from symbols."""
        if not self._ws:
            return False
        
        try:
            for symbol in symbols:
                if self.broker_type == BrokerType.FYERS:
                    message = {
                        "T": "UNSUB_DATA",
                        "SLIST": [symbol]
                    }
                else:
                    message = {
                        "RequestCode": 16,
                        "InstrumentList": [
                            {"ExchangeSegment": "NSE_FNO", "SecurityId": symbol}
                        ]
                    }
                
                await self._ws.send(json.dumps(message))
                self._subscribed_symbols.discard(symbol)
            
            return True
            
        except Exception as e:
            logger.error(f"Unsubscribe failed: {e}")
            return False
    
    def _parse_fyers_tick(self, data: dict) -> Optional[Tick]:
        """Parse Fyers v3 WebSocket tick.

        Fyers v3 wraps tick fields inside a nested 'v' object:
          {"symbol": "NFO:NIFTY...", "v": {"lp": 150.0, "bp": 149.9, "sp": 150.1, ...}}

        Field mapping (Fyers names → our Tick fields):
          lp  = last price (ltp)
          bp  = best bid price
          sp  = best ask price
          bq  = best bid quantity
          sq  = best ask quantity
        """
        try:
            symbol = data.get("symbol", "")
            if not symbol:
                return None
            v = data.get("v", data)  # fall back to flat structure if no 'v' key
            ltp = v.get("lp", v.get("ltp", 0.0))
            if not ltp:
                return None
            return Tick(
                symbol=symbol,
                ltp=float(ltp),
                bid=float(v.get("bp", v.get("bid", 0.0))),
                ask=float(v.get("sp", v.get("ask", 0.0))),
                bid_qty=int(v.get("bq", v.get("bidQty", 0))),
                ask_qty=int(v.get("sq", v.get("askQty", 0))),
                volume=int(v.get("volume", 0)),
                oi=int(v.get("oi", 0)),
                timestamp=datetime.now(),
            )
        except Exception as e:
            logger.debug(f"Failed to parse Fyers tick: {e}")
            return None
    
    def _parse_dhan_tick(self, data: bytes) -> Optional[Tick]:
        """Parse Dhan binary WebSocket tick."""
        try:
            # Dhan sends binary data
            # Format varies - this is simplified
            if len(data) < 32:
                return None
            
            # Parse binary structure (simplified)
            # Actual format: refer to Dhan documentation
            ltp = struct.unpack('f', data[8:12])[0]
            
            return Tick(
                symbol="",  # Extract from data
                ltp=ltp,
                bid=0,
                ask=0,
                bid_qty=0,
                ask_qty=0,
                volume=0,
                oi=0,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.debug(f"Failed to parse Dhan tick: {e}")
            return None
    
    async def _receive_loop(self) -> None:
        """Main receive loop."""
        while self._running and self._ws:
            try:
                message = await asyncio.wait_for(
                    self._ws.recv(),
                    timeout=30.0
                )
                
                tick = None

                if isinstance(message, bytes):
                    if self.broker_type == BrokerType.DHAN:
                        tick = self._parse_dhan_tick(message)
                    elif self.broker_type == BrokerType.FYERS:
                        # Fyers v3 occasionally sends bytes; try JSON decode first
                        try:
                            tick = self._parse_fyers_tick(json.loads(message.decode()))
                        except Exception:
                            pass  # binary Fyers format not yet decoded
                else:
                    data = json.loads(message)
                    if self.broker_type == BrokerType.FYERS:
                        tick = self._parse_fyers_tick(data)
                
                if tick and tick.ltp > 0:
                    self.on_tick(tick)
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await self._ws.ping()
                except Exception:
                    break
                    
            except ConnectionClosed:
                logger.warning("WebSocket connection closed")
                break
                
            except Exception as e:
                logger.error(f"Receive error: {e}")
                await asyncio.sleep(0.1)
    
    async def run(self) -> None:
        """Run the WebSocket feed with auto-reconnection."""
        while True:
            try:
                if not await self.connect():
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(
                        self._reconnect_delay * 2,
                        self._max_reconnect_delay
                    )
                    continue
                
                # Resubscribe to symbols
                if self._subscribed_symbols:
                    await self.subscribe(list(self._subscribed_symbols))
                
                # Enter receive loop
                await self._receive_loop()
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            
            if not self._running:
                break
            
            # Reconnect delay
            logger.info(f"Reconnecting in {self._reconnect_delay}s...")
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(
                self._reconnect_delay * 2,
                self._max_reconnect_delay
            )
    
    async def stop(self) -> None:
        """Stop the WebSocket feed."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        logger.info("WebSocket feed stopped")




