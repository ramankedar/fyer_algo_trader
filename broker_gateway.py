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

    # Fyers v3 session-based login endpoints (no browser required)
    VAGATOR_URL = "https://api-t2.fyers.in/vagator/v2"
    AUTH_BASE_URL = "https://api-t1.fyers.in/api/v3"
    DATA_URL = "https://api-t2.fyers.in/data"

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

    def _app_id_hash(self, auth_code: str) -> str:
        """SHA-256 of '<app_id>:<secret_key>' for token exchange."""
        return hashlib.sha256(
            f"{self.config.app_id}:{self.config.secret_key}".encode()
        ).hexdigest()

    def _pin_hash(self, pin: str) -> str:
        """SHA-256 of the 4-digit Fyers PIN (required by verify_pin)."""
        return hashlib.sha256(pin.encode()).hexdigest()

    async def authenticate(self) -> bool:
        """
        Fyers API v3 automated login (no browser).

        Flow:
          1. send_login_otp  → request_key
          2. verify_otp      → request_key  (uses TOTP)
          3. verify_pin      → session token
          4. generate-authcode → auth_code
          5. validate-authcode → access_token
        """
        if not self.config.pin:
            logger.error("BROKER_PIN is required for Fyers authentication")
            return False

        client = await self._get_client()

        try:
            # ── Step 1: Initiate login, get request_key ──────────────────
            logger.info("Fyers auth step 1: send_login_otp")
            r = await client.post(
                f"{self.VAGATOR_URL}/send_login_otp",
                json={"fy_id": self.config.client_id, "app_id": "2"},
            )
            data = r.json()
            if data.get("s") != "ok" or not data.get("request_key"):
                logger.error(f"send_login_otp failed: {data}")
                return False
            request_key = data["request_key"]

            # ── Step 2: Verify TOTP ──────────────────────────────────────
            logger.info("Fyers auth step 2: verify_otp (TOTP)")
            totp_code = await self.compliance.get_totp_code_async()
            if not totp_code:
                logger.error("Failed to get TOTP code")
                return False

            r = await client.post(
                f"{self.VAGATOR_URL}/verify_otp",
                json={"request_key": request_key, "otp": totp_code},
            )
            data = r.json()
            if data.get("s") != "ok" or not data.get("request_key"):
                logger.error(f"verify_otp failed: {data}")
                return False
            request_key = data["request_key"]

            # ── Step 3: Verify PIN → session token ───────────────────────
            logger.info("Fyers auth step 3: verify_pin")
            r = await client.post(
                f"{self.VAGATOR_URL}/verify_pin",
                json={
                    "request_key": request_key,
                    "identity_type": "pin",
                    "recaptcha_token": "",
                    "pin": self._pin_hash(self.config.pin),
                },
            )
            data = r.json()
            session_token = (data.get("data") or {}).get("token")
            if not session_token:
                logger.error(f"verify_pin failed: {data}")
                return False

            # ── Step 4: Generate auth code ───────────────────────────────
            logger.info("Fyers auth step 4: generate-authcode")
            r = await client.post(
                f"{self.AUTH_BASE_URL}/generate-authcode",
                headers={"Authorization": session_token},
                json={
                    "fyers_id": self.config.client_id,
                    "app_id": self.config.app_id,
                    "redirect_uri": self.config.redirect_uri,
                    "appType": "100",
                    "code_challenge": "",
                    "state": "None",
                    "scope": "",
                    "nonce": "",
                    "response_type": "code",
                    "create_cookie": True,
                },
            )
            data = r.json()
            redirect_url: str = data.get("Url", "")
            if not redirect_url or "auth_code=" not in redirect_url:
                logger.error(f"generate-authcode failed: {data}")
                return False

            # Parse auth_code from redirect URL query string
            from urllib.parse import urlparse, parse_qs
            auth_code = parse_qs(urlparse(redirect_url).query).get("auth_code", [None])[0]
            if not auth_code:
                logger.error(f"auth_code not found in redirect URL: {redirect_url}")
                return False

            # ── Step 5: Exchange auth_code for access_token ──────────────
            logger.info("Fyers auth step 5: validate-authcode")
            r = await client.post(
                f"{self.AUTH_BASE_URL}/validate-authcode",
                json={
                    "grant_type": "authorization_code",
                    "appIdHash": self._app_id_hash(auth_code),
                    "code": auth_code,
                },
            )
            data = r.json()
            if data.get("s") == "ok" and data.get("access_token"):
                self._session_token = data["access_token"]
                self._refresh_token = data.get("refresh_token")
                self.compliance.set_session(self._session_token)
                logger.info("Fyers authentication successful")
                return True

            logger.error(f"validate-authcode failed: {data}")
            return False

        except Exception as e:
            logger.error(f"Fyers authentication error: {e}")
            return False
    
    def _get_auth_headers(self) -> dict:
        """Fyers v3 authorization header format: '<app_id>:<access_token>'."""
        return {
            "Authorization": f"{self.config.app_id}:{self._session_token}",
            "Content-Type": "application/json",
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



