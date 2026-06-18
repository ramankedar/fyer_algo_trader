"""
Unified Broker Gateway for Fyers, Dhan, and OpenAlgo.
Handles authentication, order management, and position tracking.
"""

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass
from enum import Enum
import json

import httpx
import pyotp

from config import (
    BrokerConfig, BrokerType, Exchange, ProductType,
    OrderType, TransactionType
)
from compliance import ComplianceManager, rate_limited

logger = logging.getLogger("trading_system.broker")


@dataclass
class OrderResponse:
    success: bool
    order_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    message: Optional[str] = None
    status: Optional[str] = None


@dataclass
class Position:
    symbol: str
    exchange: str
    quantity: int
    average_price: float
    ltp: float
    pnl: float
    product_type: str


@dataclass
class Order:
    order_id: str
    symbol: str
    exchange: str
    transaction_type: str
    order_type: str
    quantity: int
    price: float
    trigger_price: Optional[float]
    status: str
    filled_quantity: int
    average_price: float
    rejection_reason: Optional[str]


@dataclass
class Quote:
    symbol: str
    ltp: float
    bid_price: float
    ask_price: float
    bid_qty: int
    ask_qty: int
    volume: int
    oi: int
    timestamp: str


class BrokerGateway(ABC):
    """Abstract base class for broker integrations."""
    
    def __init__(
        self,
        config: BrokerConfig,
        compliance: ComplianceManager
    ):
        self.config = config
        self.compliance = compliance
        self._session_token: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with broker and obtain session token."""
        pass
    
    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        exchange: Exchange,
        transaction_type: TransactionType,
        order_type: OrderType,
        quantity: int,
        price: float = 0,
        trigger_price: float = 0,
        product_type: ProductType = ProductType.INTRADAY
    ) -> OrderResponse:
        """Place an order."""
        pass
    
    @abstractmethod
    async def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        order_type: Optional[OrderType] = None
    ) -> OrderResponse:
        """Modify an existing order."""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel an order."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get all open positions."""
        pass
    
    @abstractmethod
    async def get_orders(self) -> List[Order]:
        """Get all orders for the day."""
        pass
    
    @abstractmethod
    async def get_quote(self, symbol: str, exchange: Exchange) -> Optional[Quote]:
        """Get current quote for a symbol."""
        pass
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class FyersGateway(BrokerGateway):
    """Fyers API v3 integration."""
    
    AUTH_BASE_URL = "[api-t1.fyers.in](https://api-t1.fyers.in/api/v3)"
    DATA_URL = "[api-t2.fyers.in](https://api-t2.fyers.in/data)"
    
    def __init__(
        self,
        config: BrokerConfig,
        compliance: ComplianceManager
    ):
        super().__init__(config, compliance)
        self._refresh_token: Optional[str] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={"Content-Type": "application/json"}
            )
        return self._client
    
    def _generate_auth_code_hash(self, auth_code: str) -> str:
        """Generate SHA256 hash for authentication."""
        data = f"{self.config.app_id}:{self.config.secret_key}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    async def authenticate(self) -> bool:
        """
        Authenticate with Fyers using TOTP.
        This is a simplified flow - actual implementation requires
        handling the OAuth redirect flow.
        """
        try:
            client = await self._get_client()
            
            # Step 1: Generate auth code URL
            # In production, this would involve browser redirect
            logger.info("Starting Fyers authentication flow")
            
            # Get TOTP code
            totp_code = self.compliance.get_totp_code()
            if not totp_code:
                logger.error("Failed to get TOTP code")
                return False
            
            # Step 2: Validate TOTP and get auth code
            # Note: Actual Fyers flow requires interactive login
            # This is a placeholder for the token validation
            
            validate_url = f"{self.AUTH_BASE_URL}/validate-authcode"
            
            app_id_hash = self._generate_auth_code_hash("")
            
            payload = {
                "grant_type": "authorization_code",
                "appIdHash": app_id_hash,
                "code": totp_code  # This would be the auth_code from redirect
            }
            
            response = await client.post(validate_url, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("s") == "ok":
                    self._session_token = data.get("access_token")
                    self._refresh_token = data.get("refresh_token")
                    self.compliance.set_session(self._session_token)
                    logger.info("Fyers authentication successful")
                    return True
            
            logger.error(f"Fyers authentication failed: {response.text}")
            return False
            
        except Exception as e:
            logger.error(f"Fyers authentication error: {e}")
            return False
    
    def _get_auth_headers(self) -> dict:
        """Get authorization headers."""
        return {
            "Authorization": f"{self.config.app_id}:{self._session_token}",
            "Content-Type": "application/json"
        }
    
    def _format_symbol(
        self,
        symbol: str,
        exchange: Exchange,
        expiry: Optional[str] = None,
        strike: Optional[float] = None,
        option_type: Optional[str] = None
    ) -> str:
        """Format symbol for Fyers API."""
        if exchange in (Exchange.NFO, Exchange.BFO):
            # Option symbol format: NSE:NIFTY24JUN23000CE
            return f"{exchange.value}:{symbol}"
        return f"{exchange.value}:{symbol}-EQ"
    
    async def place_order(
        self,
        symbol: str,
        exchange: Exchange,
        transaction_type: TransactionType,
        order_type: OrderType,
        quantity: int,
        price: float = 0,
        trigger_price: float = 0,
        product_type: ProductType = ProductType.INTRADAY
    ) -> OrderResponse:
        """Place order via Fyers API."""
        if not await self.compliance.acquire_order_slot_async(exchange.value):
            return OrderResponse(
                success=False,
                message="Rate limit exceeded"
            )
        
        try:
            client = await self._get_client()
            
            # Map product type
            product_map = {
                ProductType.INTRADAY: "INTRADAY",
                ProductType.MARGIN: "MARGIN",
                ProductType.CNC: "CNC"
            }
            
            # Map order type
            order_type_map = {
                OrderType.MARKET: 2,
                OrderType.LIMIT: 1,
                OrderType.SL_MARKET: 3,
                OrderType.SL_LIMIT: 4
            }
            
            payload = {
                "symbol": self._format_symbol(symbol, exchange),
                "qty": quantity,
                "type": order_type_map.get(order_type, 2),
                "side": 1 if transaction_type == TransactionType.BUY else -1,
                "productType": product_map.get(product_type, "INTRADAY"),
                "limitPrice": price if order_type in (OrderType.LIMIT, OrderType.SL_LIMIT) else 0,
                "stopPrice": trigger_price if order_type in (OrderType.SL_MARKET, OrderType.SL_LIMIT) else 0,
                "validity": "DAY",
                "disclosedQty": 0,
                "offlineOrder": False
            }
            
            response = await client.post(
                f"{self.AUTH_BASE_URL}/orders",
                json=payload,
                headers=self._get_auth_headers()
            )
            
            data = response.json()
            
            if data.get("s") == "ok":
                return OrderResponse(
                    success=True,
                    order_id=data.get("id"),
                    broker_order_id=data.get("id"),
                    message="Order placed successfully"
                )
            
            return OrderResponse(
                success=False,
                message=data.get("message", "Order placement failed")
            )
            
        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return OrderResponse(success=False, message=str(e))
    
    async def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        order_type: Optional[OrderType] = None
    ) -> OrderResponse:
        """Modify an existing order."""
        if not await self.compliance.acquire_order_slot_async("NFO"):
            return OrderResponse(success=False, message="Rate limit exceeded")
        
        try:
            client = await self._get_client()
            
            payload = {"id": order_id}
            
            if quantity is not None:
                payload["qty"] = quantity
            if price is not None:
                payload["limitPrice"] = price
            if trigger_price is not None:
                payload["stopPrice"] = trigger_price
            if order_type is not None:
                order_type_map = {
                    OrderType.MARKET: 2,
                    OrderType.LIMIT: 1,
                    OrderType.SL_MARKET: 3,
                    OrderType.SL_LIMIT: 4
                }
                payload["type"] = order_type_map.get(order_type, 2)
            
            response = await client.patch(
                f"{self.AUTH_BASE_URL}/orders/{order_id}",
                json=payload,
                headers=self._get_auth_headers()
            )
            
            data = response.json()
            
            if data.get("s") == "ok":
                return OrderResponse(
                    success=True,
                    order_id=order_id,
                    message="Order modified successfully"
                )
            
            return OrderResponse(
                success=False,
                message=data.get("message", "Modification failed")
            )
            
        except Exception as e:
            logger.error(f"Order modification error: {e}")
            return OrderResponse(success=False, message=str(e))
    
    async def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel an order."""
        if not await self.compliance.acquire_order_slot_async("NFO"):
            return OrderResponse(success=False, message="Rate limit exceeded")
        
        try:
            client = await self._get_client()
            
            response = await client.delete(
                f"{self.AUTH_BASE_URL}/orders/{order_id}",
                headers=self._get_auth_headers()
            )
            
            data = response.json()
            
            if data.get("s") == "ok":
                return OrderResponse(
                    success=True,
                    order_id=order_id,
                    message="Order cancelled"
                )
            
            return OrderResponse(
                success=False,
                message=data.get("message", "Cancellation failed")
            )
            
        except Exception as e:
            logger.error(f"Order cancellation error: {e}")
            return OrderResponse(success=False, message=str(e))
    
    async def get_positions(self) -> List[Position]:
        """Get all positions."""
        try:
            client = await self._get_client()
            
            response = await client.get(
                f"{self.AUTH_BASE_URL}/positions",
                headers=self._get_auth_headers()
            )
            
            data = response.json()
            positions = []
            
            if data.get("s") == "ok":
                for pos in data.get("netPositions", []):
                    positions.append(Position(
                        symbol=pos.get("symbol", ""),
                        exchange=pos.get("exchange", ""),
                        quantity=pos.get("netQty", 0),
                        average_price=pos.get("avgPrice", 0),
                        ltp=pos.get("ltp", 0),
                        pnl=pos.get("pl", 0),
                        product_type=pos.get("productType", "")
                    ))
            
            return positions
            
        except Exception as e:
            logger.error(f"Get positions error: {e}")
            return []
    
    async def get_orders(self) -> List[Order]:
        """Get all orders for the day."""
        try:
            client = await self._get_client()
            
            response = await client.get(
                f"{self.AUTH_BASE_URL}/orders",
                headers=self._get_auth_headers()
            )
            
            data = response.json()
            orders = []
            
            if data.get("s") == "ok":
                for ord in data.get("orderBook", []):
                    orders.append(Order(
                        order_id=ord.get("id", ""),
                        symbol=ord.get("symbol", ""),
                        exchange=ord.get("exchange", ""),
                        transaction_type="BUY" if ord.get("side") == 1 else "SELL",
                        order_type=str(ord.get("type", "")),
                        quantity=ord.get("qty", 0),
                        price=ord.get("limitPrice", 0),
                        trigger_price=ord.get("stopPrice"),
                        status=str(ord.get("status", "")),
                        filled_quantity=ord.get("filledQty", 0),
                        average_price=ord.get("tradedPrice", 0),
                        rejection_reason=ord.get("message")
                    ))
            
            return orders
            
        except Exception as e:
            logger.error(f"Get orders error: {e}")
            return []
    
    async def get_quote(self, symbol: str, exchange: Exchange) -> Optional[Quote]:
        """Get current quote."""
        try:
            client = await self._get_client()
            
            formatted_symbol = self._format_symbol(symbol, exchange)
            
            response = await client.get(
                f"{self.DATA_URL}/quotes",
                params={"symbols": formatted_symbol},
                headers=self._get_auth_headers()
            )
            
            data = response.json()
            
            if data.get("s") == "ok" and data.get("d"):
                q = data["d"][0]["v"]
                return Quote(
                    symbol=symbol,
                    ltp=q.get("lp", 0),
                    bid_price=q.get("bp", 0),
                    ask_price=q.get("sp", 0),
                    bid_qty=q.get("bq", 0),
                    ask_qty=q.get("sq", 0),
                    volume=q.get("volume", 0),
                    oi=q.get("oi", 0),
                    timestamp=datetime.now().isoformat()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Get quote error: {e}")
            return None


class DhanGateway(BrokerGateway):
    """Dhan API integration."""
    
    def __init__(
        self,
        config: BrokerConfig,
        compliance: ComplianceManager
    ):
        super().__init__(config, compliance)
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.api_base_url,
                timeout=30.0,
                headers={
                    "Content-Type": "application/json",
                    "access-token": self.config.secret_key
                }
            )
        return self._client
    
    async def authenticate(self) -> bool:
        """Dhan uses API key authentication, no TOTP flow needed."""
        try:
            # Validate credentials by making a test request
            client = await self._get_client()
            
            response = await client.get("/funds/margincalculator")
            
            if response.status_code == 200:
                self._session_token = self.config.secret_key
                self.compliance.set_session(self._session_token)
                logger.info("Dhan authentication successful")
                return True
            
            logger.error(f"Dhan authentication failed: {response.text}")
            return False
            
        except Exception as e:
            logger.error(f"Dhan authentication error: {e}")
            return False
    
    def _map_exchange(self, exchange: Exchange) -> str:
        """Map exchange enum to Dhan format."""
        mapping = {
            Exchange.NSE: "NSE_EQ",
            Exchange.NFO: "NSE_FNO",
            Exchange.BSE: "BSE_EQ",
            Exchange.BFO: "BSE_FNO"
        }
        return mapping.get(exchange, "NSE_FNO")
    
    async def place_order(
        self,
        symbol: str,
        exchange: Exchange,
        transaction_type: TransactionType,
        order_type: OrderType,
        quantity: int,
        price: float = 0,
        trigger_price: float = 0,
        product_type: ProductType = ProductType.INTRADAY
    ) -> OrderResponse:
        """Place order via Dhan API."""
        if not await self.compliance.acquire_order_slot_async(exchange.value):
            return OrderResponse(success=False, message="Rate limit exceeded")
        
        try:
            client = await self._get_client()
            
            # Map product type
            product_map = {
                ProductType.INTRADAY: "INTRADAY",
                ProductType.MARGIN: "MARGIN",
                ProductType.CNC: "CNC"
            }
            
            # Map order type
            order_type_map = {
                OrderType.MARKET: "MARKET",
                OrderType.LIMIT: "LIMIT",
                OrderType.SL_MARKET: "STOP_LOSS_MARKET",
                OrderType.SL_LIMIT: "STOP_LOSS"
            }
            
            payload = {
                "dhanClientId": self.config.client_id,
                "transactionType": transaction_type.value,
                "exchangeSegment": self._map_exchange(exchange),
                "productType": product_map.get(product_type, "INTRADAY"),
                "orderType": order_type_map.get(order_type, "MARKET"),
                "validity": "DAY",
                "tradingSymbol": symbol,
                "securityId": symbol,  # Would need symbol mapping
                "quantity": quantity,
                "price": price if order_type in (OrderType.LIMIT, OrderType.SL_LIMIT) else 0,
                "triggerPrice": trigger_price if order_type in (OrderType.SL_MARKET, OrderType.SL_LIMIT) else 0,
                "disclosedQuantity": 0,
                "afterMarketOrder": False
            }
            
            response = await client.post("/orders", json=payload)
            data = response.json()
            
            if response.status_code == 200 and data.get("orderId"):
                return OrderResponse(
                    success=True,
                    order_id=data["orderId"],
                    broker_order_id=data["orderId"],
                    message="Order placed successfully"
                )
            
            return OrderResponse(
                success=False,
                message=data.get("remarks", "Order placement failed")
            )
            
        except Exception as e:
            logger.error(f"Dhan order error: {e}")
            return OrderResponse(success=False, message=str(e))
    
    async def modify_order(
        self,
        order_id: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        order_type: Optional[OrderType] = None
    ) -> OrderResponse:
        """Modify an existing order."""
        if not await self.compliance.acquire_order_slot_async("NFO"):
            return OrderResponse(success=False, message="Rate limit exceeded")
        
        try:
            client = await self._get_client()
            
            payload = {
                "dhanClientId": self.config.client_id,
                "orderId": order_id
            }
            
            if quantity is not None:
                payload["quantity"] = quantity
            if price is not None:
                payload["price"] = price
            if trigger_price is not None:
                payload["triggerPrice"] = trigger_price
            if order_type is not None:
                order_type_map = {
                    OrderType.MARKET: "MARKET",
                    OrderType.LIMIT: "LIMIT",
                    OrderType.SL_MARKET: "STOP_LOSS_MARKET",
                    OrderType.SL_LIMIT: "STOP_LOSS"
                }
                payload["orderType"] = order_type_map.get(order_type, "MARKET")
            
            response = await client.put(f"/orders/{order_id}", json=payload)
            data = response.json()
            
            if response.status_code == 200:
                return OrderResponse(
                    success=True,
                    order_id=order_id,
                    message="Order modified"
                )
            
            return OrderResponse(
                success=False,
                message=data.get("remarks", "Modification failed")
            )
            
        except Exception as e:
            logger.error(f"Dhan modify error: {e}")
            return OrderResponse(success=False, message=str(e))
    
    async def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel an order."""
        if not await self.compliance.acquire_order_slot_async("NFO"):
            return OrderResponse(success=False, message="Rate limit exceeded")
        
        try:
            client = await self._get_client()
            
            response = await client.delete(f"/orders/{order_id}")
            
            if response.status_code == 200:
                return OrderResponse(
                    success=True,
                    order_id=order_id,
                    message="Order cancelled"
                )
            
            data = response.json()
            return OrderResponse(
                success=False,
                message=data.get("remarks", "Cancellation failed")
            )
            
        except Exception as e:
            logger.error(f"Dhan cancel error: {e}")
            return OrderResponse(success=False, message=str(e))
    
    async def get_positions(self) -> List[Position]:
        """Get all positions."""
        try:
            client = await self._get_client()
            
            response = await client.get("/positions")
            data = response.json()
            
            positions = []
            if response.status_code == 200:
                for pos in data:
                    positions.append(Position(
                        symbol=pos.get("tradingSymbol", ""),
                        exchange=pos.get("exchangeSegment", ""),
                        quantity=pos.get("netQty", 0),
                        average_price=pos.get("costPrice", 0),
                        ltp=pos.get("ltp", 0),
                        pnl=pos.get("realizedProfit", 0) + pos.get("unrealizedProfit", 0),
                        product_type=pos.get("productType", "")
                    ))
            
            return positions
            
        except Exception as e:
            logger.error(f"Dhan get positions error: {e}")
            return []
    
    async def get_orders(self) -> List[Order]:
        """Get all orders for the day."""
        try:
            client = await self._get_client()
            
            response = await client.get("/orders")
            data = response.json()
            
            orders = []
            if response.status_code == 200:
                for ord in data:
                    orders.append(Order(
                        order_id=ord.get("orderId", ""),
                        symbol=ord.get("tradingSymbol", ""),
                        exchange=ord.get("exchangeSegment", ""),
                        transaction_type=ord.get("transactionType", ""),
                        order_type=ord.get("orderType", ""),
                        quantity=ord.get("quantity", 0),
                        price=ord.get("price", 0),
                        trigger_price=ord.get("triggerPrice"),
                        status=ord.get("orderStatus", ""),
                        filled_quantity=ord.get("filledQty", 0),
                        average_price=ord.get("tradedPrice", 0),
                        rejection_reason=ord.get("remarks")
                    ))
            
            return orders
            
        except Exception as e:
            logger.error(f"Dhan get orders error: {e}")
            return []
    
    async def get_quote(self, symbol: str, exchange: Exchange) -> Optional[Quote]:
        """Get current quote."""
        try:
            client = await self._get_client()
            
            response = await client.get(
                f"/marketfeed/ltp",
                params={
                    "exchangeSegment": self._map_exchange(exchange),
                    "securityId": symbol
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return Quote(
                    symbol=symbol,
                    ltp=data.get("ltp", 0),
                    bid_price=data.get("bidPrice", 0),
                    ask_price=data.get("askPrice", 0),
                    bid_qty=data.get("bidQty", 0),
                    ask_qty=data.get("askQty", 0),
                    volume=data.get("volume", 0),
                    oi=data.get("openInterest", 0),
                    timestamp=datetime.now().isoformat()
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Dhan get quote error: {e}")
            return None


def create_broker_gateway(
    config: BrokerConfig,
    compliance: ComplianceManager
) -> BrokerGateway:
    """Factory function to create appropriate broker gateway."""
    if config.broker_type == BrokerType.FYERS:
        return FyersGateway(config, compliance)
    elif config.broker_type == BrokerType.DHAN:
        return DhanGateway(config, compliance)
    else:
        raise ValueError(f"Unsupported broker type: {config.broker_type}")



