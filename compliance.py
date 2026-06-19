"""
SEBI Compliance Module: Rate limiting, IP validation, and 2FA automation.
Ensures adherence to SEBI 2026 retail algo trading mandates.
"""

import time
import threading
import asyncio
import logging
import socket
import pyotp
import httpx
from datetime import datetime, timedelta
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
from functools import wraps

logger = logging.getLogger("trading_system.compliance")


@dataclass
class RateLimitStats:
    total_requests: int = 0
    requests_last_second: int = 0
    requests_blocked: int = 0
    last_reset: datetime = field(default_factory=datetime.now)


class TokenBucketRateLimiter:
    """
    Thread-safe token bucket rate limiter for SEBI OPS compliance.
    Maintains orders below 10 OPS threshold with 9 OPS limit.
    """
    
    def __init__(
        self,
        max_tokens: int = 9,
        refill_rate: float = 9.0,  # tokens per second
        segment_name: str = "NFO"
    ):
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.segment_name = segment_name
        
        self.tokens = float(max_tokens)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = RateLimitStats()
        self._request_times: deque = deque(maxlen=1000)
        
        logger.info(
            f"Rate limiter initialized for {segment_name}: "
            f"max={max_tokens} OPS, refill={refill_rate}/s"
        )
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def acquire(self, timeout: float = 5.0) -> bool:
        """
        Acquire a token for making an order request.
        Blocks until token available or timeout.
        Returns True if token acquired, False on timeout.
        """
        start_time = time.monotonic()
        
        while True:
            with self._lock:
                self._refill()
                
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    self.stats.total_requests += 1
                    self._request_times.append(time.monotonic())
                    return True
                
                # Calculate wait time
                tokens_needed = 1.0 - self.tokens
                wait_time = tokens_needed / self.refill_rate
            
            # Check timeout
            elapsed = time.monotonic() - start_time
            if elapsed + wait_time > timeout:
                with self._lock:
                    self.stats.requests_blocked += 1
                logger.warning(f"Rate limit timeout for {self.segment_name}")
                return False
            
            # Wait for refill
            time.sleep(min(wait_time, 0.1))
    
    async def acquire_async(self, timeout: float = 5.0) -> bool:
        """Async version of acquire for asyncio contexts."""
        start_time = time.monotonic()
        
        while True:
            with self._lock:
                self._refill()
                
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    self.stats.total_requests += 1
                    self._request_times.append(time.monotonic())
                    return True
                
                tokens_needed = 1.0 - self.tokens
                wait_time = tokens_needed / self.refill_rate
            
            elapsed = time.monotonic() - start_time
            if elapsed + wait_time > timeout:
                with self._lock:
                    self.stats.requests_blocked += 1
                return False
            
            await asyncio.sleep(min(wait_time, 0.05))
    
    def get_current_ops(self) -> float:
        """Get current orders per second rate."""
        now = time.monotonic()
        with self._lock:
            # Count requests in last second
            recent = sum(1 for t in self._request_times if now - t < 1.0)
            return float(recent)
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                "segment": self.segment_name,
                "total_requests": self.stats.total_requests,
                "blocked_requests": self.stats.requests_blocked,
                "current_tokens": self.tokens,
                "current_ops": self.get_current_ops()
            }


class SegmentRateLimiter:
    """
    Manages rate limiters for multiple exchange segments.
    """
    
    def __init__(self, max_ops: int = 9):
        self.limiters = {
            "NSE": TokenBucketRateLimiter(max_ops, max_ops, "NSE"),
            "NFO": TokenBucketRateLimiter(max_ops, max_ops, "NFO"),
            "BSE": TokenBucketRateLimiter(max_ops, max_ops, "BSE"),
            "BFO": TokenBucketRateLimiter(max_ops, max_ops, "BFO"),
        }
    
    def acquire(self, segment: str, timeout: float = 5.0) -> bool:
        """Acquire token for specified segment."""
        limiter = self.limiters.get(segment.upper())
        if limiter is None:
            logger.error(f"Unknown segment: {segment}")
            return False
        return limiter.acquire(timeout)
    
    async def acquire_async(self, segment: str, timeout: float = 5.0) -> bool:
        """Async acquire for specified segment."""
        limiter = self.limiters.get(segment.upper())
        if limiter is None:
            return False
        return await limiter.acquire_async(timeout)
    
    def get_all_stats(self) -> dict:
        """Get statistics for all segments."""
        return {seg: lim.get_stats() for seg, lim in self.limiters.items()}


def rate_limited(segment: str = "NFO"):
    """Decorator for rate-limited functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if hasattr(self, "rate_limiter"):
                if not self.rate_limiter.acquire(segment):
                    raise RateLimitExceeded(f"Rate limit exceeded for {segment}")
            return func(self, *args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            if hasattr(self, "rate_limiter"):
                if not await self.rate_limiter.acquire_async(segment):
                    raise RateLimitExceeded(f"Rate limit exceeded for {segment}")
            return await func(self, *args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    return decorator


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass


class IPValidator:
    """
    Validates that outgoing connections use whitelisted IP.
    """
    
    PUBLIC_IP_SERVICES = [
        "https://api.ipify.org",
        "https://ifconfig.me/ip",
        "https://icanhazip.com",
    ]
    
    def __init__(self, whitelisted_ip: str, enable_check: bool = True):
        self.whitelisted_ip = whitelisted_ip.strip()
        self.enable_check = enable_check
        self._cached_ip: Optional[str] = None
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=5)
    
    async def get_public_ip(self) -> Optional[str]:
        """Get current public IP address."""
        # Check cache
        if (self._cached_ip and self._cache_time and 
            datetime.now() - self._cache_time < self._cache_duration):
            return self._cached_ip
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for service in self.PUBLIC_IP_SERVICES:
                try:
                    response = await client.get(service)
                    if response.status_code == 200:
                        ip = response.text.strip()
                        self._cached_ip = ip
                        self._cache_time = datetime.now()
                        return ip
                except Exception as e:
                    logger.debug(f"Failed to get IP from {service}: {e}")
                    continue
        
        logger.error("Failed to determine public IP from all services")
        return None
    
    def get_public_ip_sync(self) -> Optional[str]:
        """Synchronous version of get_public_ip."""
        if (self._cached_ip and self._cache_time and 
            datetime.now() - self._cache_time < self._cache_duration):
            return self._cached_ip
        
        with httpx.Client(timeout=10.0) as client:
            for service in self.PUBLIC_IP_SERVICES:
                try:
                    response = client.get(service)
                    if response.status_code == 200:
                        ip = response.text.strip()
                        self._cached_ip = ip
                        self._cache_time = datetime.now()
                        return ip
                except Exception:
                    continue
        
        return None
    
    async def validate(self) -> bool:
        """
        Validate that current public IP matches whitelisted IP.
        """
        if not self.enable_check:
            logger.debug("IP validation disabled")
            return True
        
        if not self.whitelisted_ip:
            logger.warning("No whitelisted IP configured, skipping validation")
            return True
        
        current_ip = await self.get_public_ip()
        
        if current_ip is None:
            logger.error("Could not determine public IP for validation")
            return False
        
        if current_ip != self.whitelisted_ip:
            logger.error(
                f"IP mismatch: current={current_ip}, "
                f"whitelisted={self.whitelisted_ip}"
            )
            return False
        
        logger.info(f"IP validation passed: {current_ip}")
        return True
    
    def validate_sync(self) -> bool:
        """Synchronous validation."""
        if not self.enable_check or not self.whitelisted_ip:
            return True
        
        current_ip = self.get_public_ip_sync()
        
        if current_ip is None:
            return False
        
        return current_ip == self.whitelisted_ip


class TOTPAuthenticator:
    """
    TOTP-based 2FA automation for daily broker login.
    """
    
    def __init__(self, totp_key: str):
        if not totp_key:
            raise ValueError("TOTP key is required")
        
        # Clean the key (remove spaces, convert to uppercase)
        clean_key = totp_key.replace(" ", "").upper()
        self.totp = pyotp.TOTP(clean_key)
        
        logger.info("TOTP authenticator initialized")
    
    def get_current_code(self) -> str:
        """Get current TOTP code."""
        return self.totp.now()
    
    def get_code_with_validity(self) -> tuple[str, int]:
        """
        Get current code and seconds until expiry.
        Returns (code, seconds_remaining).
        """
        code = self.totp.now()
        
        # TOTP codes change every 30 seconds
        current_time = int(time.time())
        seconds_remaining = 30 - (current_time % 30)
        
        return code, seconds_remaining
    
    def verify_code(self, code: str) -> bool:
        """Verify a TOTP code (for testing setup)."""
        return self.totp.verify(code)
    
    def wait_for_fresh_code(self, min_validity: int = 5) -> str:
        """Blocking wait for a TOTP code with at least min_validity seconds remaining."""
        while True:
            code, remaining = self.get_code_with_validity()
            if remaining >= min_validity:
                return code
            wait_time = remaining + 1
            logger.debug(f"Waiting {wait_time}s for fresh TOTP code")
            time.sleep(wait_time)

    async def wait_for_fresh_code_async(self, min_validity: int = 5) -> str:
        """Async wait for a TOTP code with at least min_validity seconds remaining."""
        while True:
            code, remaining = self.get_code_with_validity()
            if remaining >= min_validity:
                return code
            wait_time = remaining + 1
            logger.debug(f"Waiting {wait_time}s for fresh TOTP code (async)")
            await asyncio.sleep(wait_time)


class DailyLoginManager:
    """
    Manages automated daily broker login with 2FA.
    """
    
    def __init__(
        self,
        totp_authenticator: TOTPAuthenticator,
        login_time: str = "08:25:00",  # Before pre-market at 08:30
    ):
        self.totp = totp_authenticator
        self.login_time = datetime.strptime(login_time, "%H:%M:%S").time()
        self._last_login_date: Optional[datetime] = None
        self._session_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
    
    def needs_login(self) -> bool:
        """Check if login is required."""
        today = datetime.now().date()
        
        # Need login if never logged in
        if self._last_login_date is None:
            return True
        
        # Need login if last login was on a different day
        if self._last_login_date.date() != today:
            return True
        
        # Need login if token expired
        if self._token_expiry and datetime.now() >= self._token_expiry:
            return True
        
        return False
    
    def is_login_time(self) -> bool:
        """Check if current time is appropriate for login."""
        now = datetime.now().time()
        
        # Allow login from configured time until market close
        market_close = datetime.strptime("15:30:00", "%H:%M:%S").time()
        
        return self.login_time <= now <= market_close
    
    def set_session(
        self,
        token: str,
        expiry: Optional[datetime] = None
    ) -> None:
        """Set session token after successful login."""
        self._session_token = token
        self._last_login_date = datetime.now()
        self._token_expiry = expiry or (
            datetime.now() + timedelta(hours=8)  # Default 8-hour session
        )
        logger.info(f"Session set, expires at {self._token_expiry}")
    
    def get_session_token(self) -> Optional[str]:
        """Get current session token if valid."""
        if self.needs_login():
            return None
        return self._session_token
    
    def get_totp_code(self) -> str:
        """Get fresh TOTP code for login."""
        return self.totp.wait_for_fresh_code(min_validity=10)


class ComplianceManager:
    """
    Central compliance management combining all SEBI requirements.
    """
    
    def __init__(
        self,
        whitelisted_ip: str,
        totp_key: str,
        max_ops: int = 9,
        enable_ip_check: bool = True,
        login_time: str = "08:25:00"
    ):
        self.rate_limiter = SegmentRateLimiter(max_ops)
        self.ip_validator = IPValidator(whitelisted_ip, enable_ip_check)
        self.totp = TOTPAuthenticator(totp_key) if totp_key else None
        self.login_manager = (
            DailyLoginManager(self.totp, login_time) if self.totp else None
        )
        
        self._is_validated = False
        
        logger.info("Compliance manager initialized")
    
    async def validate_environment(self) -> bool:
        """
        Perform all pre-trading compliance checks.
        Returns True if all checks pass.
        """
        # Validate IP
        if not await self.ip_validator.validate():
            logger.error("IP validation failed")
            return False
        
        # Check login status
        if self.login_manager and self.login_manager.needs_login():
            logger.warning("Broker login required")
            return False
        
        self._is_validated = True
        logger.info("All compliance checks passed")
        return True
    
    def validate_environment_sync(self) -> bool:
        """Synchronous version of validate_environment."""
        if not self.ip_validator.validate_sync():
            return False
        
        if self.login_manager and self.login_manager.needs_login():
            return False
        
        self._is_validated = True
        return True
    
    def is_ready(self) -> bool:
        """Check if system is ready for trading."""
        return self._is_validated
    
    def acquire_order_slot(self, segment: str = "NFO", timeout: float = 5.0) -> bool:
        """Acquire order slot respecting rate limits."""
        return self.rate_limiter.acquire(segment, timeout)
    
    async def acquire_order_slot_async(
        self,
        segment: str = "NFO",
        timeout: float = 5.0
    ) -> bool:
        """Async version of acquire_order_slot."""
        return await self.rate_limiter.acquire_async(segment, timeout)
    
    def get_totp_code(self) -> Optional[str]:
        """Get TOTP code for authentication (blocking)."""
        if self.totp:
            return self.totp.wait_for_fresh_code()
        return None

    async def get_totp_code_async(self) -> Optional[str]:
        """Get TOTP code for authentication (non-blocking async)."""
        if self.totp:
            return await self.totp.wait_for_fresh_code_async()
        return None
    
    def set_session(self, token: str, expiry: Optional[datetime] = None) -> None:
        """Set broker session after login."""
        if self.login_manager:
            self.login_manager.set_session(token, expiry)
    
    def get_session_token(self) -> Optional[str]:
        """Get current valid session token."""
        if self.login_manager:
            return self.login_manager.get_session_token()
        return None
    
    def get_compliance_status(self) -> dict:
        """Get full compliance status report."""
        return {
            "is_validated": self._is_validated,
            "ip_check_enabled": self.ip_validator.enable_check,
            "whitelisted_ip": self.ip_validator.whitelisted_ip,
            "current_ip": self.ip_validator._cached_ip,
            "rate_limits": self.rate_limiter.get_all_stats(),
            "login_required": (
                self.login_manager.needs_login() if self.login_manager else None
            ),
            "session_valid": (
                self.login_manager.get_session_token() is not None
                if self.login_manager else None
            )
        }



