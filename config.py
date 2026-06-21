"""
Configuration module for the Indian Options Algorithmic Trading System.
All sensitive values should be set via environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import logging

from dotenv import load_dotenv
load_dotenv() 

from dataclasses import dataclass, field


class BrokerType(Enum):
    FYERS = "fyers"
    DHAN = "dhan"
    OPENALGO = "openalgo"


class Exchange(Enum):
    NSE = "NSE"
    NFO = "NFO"
    BSE = "BSE"
    BFO = "BFO"


class ProductType(Enum):
    INTRADAY = "INTRADAY"  # MIS
    MARGIN = "MARGIN"      # NRML
    CNC = "CNC"            # Delivery


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL_MARKET = "SL-M"
    SL_LIMIT = "SL-L"


class TransactionType(Enum):
    BUY = "BUY"
    SELL = "SELL"


# @dataclass
# class BrokerConfig:
#     broker_type: BrokerType
#     client_id: str
#     app_id: str
#     secret_key: str
#     totp_key: str
#     redirect_uri: str = "[localhost](https://localhost:8080/callback)"
#     api_base_url: str = ""
#     ws_url: str = ""
    
#     def __post_init__(self):
#         if self.broker_type == BrokerType.FYERS:
#             self.api_base_url = "[api-t1.fyers.in](https://api-t1.fyers.in/api/v3)"
#             self.ws_url = "wss://api-t1.fyers.in/socket/v2/dataSock"
#         elif self.broker_type == BrokerType.DHAN:
#             self.api_base_url = "[api.dhan.co](https://api.dhan.co/v2)"
#             self.ws_url = "wss://api-feed.dhan.co"
#         elif self.broker_type == BrokerType.OPENALGO:
#             self.api_base_url = os.getenv("OPENALGO_API_URL", "[localhost](http://localhost:5000)")
#             self.ws_url = os.getenv("OPENALGO_WS_URL", "ws://localhost:5000/ws")



@dataclass
class BrokerConfig:
    broker_type: BrokerType
    client_id: str
    app_id: str
    secret_key: str
    totp_key: str
    pin: str = ""
    redirect_uri: str = "http://127.0.0.1:8080/callback"
    api_base_url: str = ""
    ws_url: str = ""

    def __post_init__(self):
        if self.broker_type == BrokerType.FYERS:
            self.api_base_url = "https://api-t1.fyers.in/api/v3"
            self.ws_url = "wss://api-t1.fyers.in/socket/v2/dataSock"
        elif self.broker_type == BrokerType.DHAN:
            self.api_base_url = "https://api.dhan.co/v2"
            self.ws_url = "wss://api-feed.dhan.co"
        elif self.broker_type == BrokerType.OPENALGO:
            self.api_base_url = os.getenv("OPENALGO_API_URL", "http://127.0.0.1:5000")
            self.ws_url = os.getenv("OPENALGO_WS_URL", "ws://127.0.0.1:5000/ws")

@dataclass
class RiskConfig:
    # ── Portfolio-level guards ────────────────────────────────────────────────
    max_daily_loss_percent:    float = 2.0
    max_position_size:         int   = 1800
    max_open_positions:        int   = 4
    max_drawdown_percent:      float = 5.0
    trailing_drawdown_percent: float = 3.0
    emergency_square_off_time: str   = "15:15:00"
    position_size_per_trade:   int   = 50

    # ── Dynamic Margin-Aware Risk Engine (replaces rigid sleeves) ────────────
    #
    # Single capital pool — no hard sleeves. Any strategy can use available
    # capital, subject to three guardrails:
    #
    #   max_capital_per_trade_pct  — no single trade consumes > 30% of total
    #                                (prevents one position crowding out others)
    #   margin_reserve_minimum     — always keep ₹50K free for SPAN margin
    #                                (prevents option-seller margin starvation)
    #   risk_per_trade_percent     — risk exactly 2% of TOTAL capital per trade
    #                                (Kelly-aligned; larger than sleeve-based 2%)
    max_capital_per_trade_pct: float = 0.30    # max 30% of capital in one trade
    margin_reserve_minimum:    float = 50_000.0 # always keep ₹50K for margin buffer
    risk_per_trade_percent:    float = 2.0      # 2% of total capital at risk

    # Legacy aliases (kept so RiskThresholds dataclass stays backward-compatible)
    skewhunter_allocated_capital:    float = 200_000.0   # points to full pool now
    strangle_allocated_capital:      float = 200_000.0
    credit_spread_allocated_capital: float = 200_000.0


@dataclass
class ComplianceConfig:
    max_orders_per_second: int = 9
    whitelisted_ip: str = ""
    enable_ip_check: bool = True
    totp_key: str = ""
    daily_login_time: str = "08:25:00"


@dataclass
class StrategyConfig:
    # Common
    min_premium: float = 20.0
    trading_start_time: str = "10:15:00"
    trading_end_time: str = "14:15:00"

    # Phase 2: Z-score entry gates — relaxed from 0.85/−0.85 to 0.65/−0.65.
    # 0.65 ≈ top/bottom 26% of observations (was 20%).  Eliminates trade starvation
    # seen with the tighter threshold in low-VIX 2025 environments.
    z_score_long_threshold:  float = 0.65
    z_score_short_threshold: float = -0.65

    # Phase 2: Expiry-day strangle minimum premium (as % of spot)
    # 0.0003 × 25000 = ₹7.50 — allows far-OTM strikes on expiry days where
    # premium is already crushed before 10:30 AM.
    expiry_min_premium_spot_pct: float = 0.0001   # Phase 4: ₹2.50 on Nifty — allows 0DTE wings

    # Fixed RR 1:3
    fixed_rr_stop_loss_pct: float = 30.0
    fixed_rr_target_pct: float = 90.0
    fixed_rr_alpha1_long_threshold: float = 0.75
    fixed_rr_alpha2_long_threshold: float = 0.7
    fixed_rr_alpha1_short_threshold: float = 0.25
    fixed_rr_alpha2_short_threshold: float = 0.3

    # Curvature Credit Spread
    # Phase 5: CurvatureCreditSpread removed — dead weight (1 trade, -100% in 18M)
    # curvature_threshold / viscosity_threshold deleted

    # SkewHunter
    skewhunter_stop_loss_pct: float = 40.0
    # Phase 2: raw volume-ratio thresholds (NOT z-scored — OI/Vol not normally distributed)
    # Relaxed: 1.25 = calls 25% > puts (was 2.0×); 0.80 = puts 25% > calls (was 0.5×).
    # Prevents trade starvation — real NSE data rarely shows 2× imbalance intraday.
    skewhunter_volume_ratio_long:  float = 1.05  # lowered: 1.25 was too strict with bhavcopy volumes
    skewhunter_volume_ratio_short: float = 0.95  # symmetrical: puts > calls even slightly
    # Legacy (kept for config compatibility)
    skewhunter_alpha1_long:  float = 0.75
    skewhunter_alpha2_long:  float = 0.8
    skewhunter_alpha1_short: float = 0.25
    skewhunter_alpha2_short: float = 0.2
    skewhunter_square_off_time: str = "15:15:00"


@dataclass
class DatabaseConfig:
    db_path: str = "trading_state.db"
    lock_timeout: int = 30


@dataclass
class LogConfig:
    log_level: str = "INFO"
    log_file: str = "trading_system.log"
    max_bytes: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5


@dataclass
class AppConfig:
    broker: BrokerConfig = field(default_factory=lambda: BrokerConfig(
        broker_type=BrokerType(os.getenv("BROKER_TYPE", "fyers")),
        client_id=os.getenv("BROKER_CLIENT_ID", ""),
        app_id=os.getenv("BROKER_APP_ID", ""),
        secret_key=os.getenv("BROKER_SECRET_KEY", ""),
        totp_key=os.getenv("BROKER_TOTP_KEY", ""),
        pin=os.getenv("BROKER_PIN", ""),
        redirect_uri=os.getenv("BROKER_REDIRECT_URI", "http://127.0.0.1:8080/callback"),
    ))
    risk: RiskConfig = field(default_factory=RiskConfig)
    compliance: ComplianceConfig = field(default_factory=lambda: ComplianceConfig(
        whitelisted_ip=os.getenv("WHITELISTED_IP", ""),
        totp_key=os.getenv("BROKER_TOTP_KEY", ""),
    ))
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LogConfig = field(default_factory=LogConfig)
    
    # Market configuration
    underlying_symbol: str = "NIFTY"
    risk_free_rate: float = 0.065  # 6.5% RBI repo rate
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load configuration from environment variables."""
        return cls()


def setup_logging(config: LogConfig) -> logging.Logger:
    """Configure application logging with rotation."""
    from logging.handlers import RotatingFileHandler
    
    logger = logging.getLogger("trading_system")
    logger.setLevel(getattr(logging, config.log_level.upper()))
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        config.log_file,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

