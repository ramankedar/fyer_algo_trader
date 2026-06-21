"""
Central type definitions for the trading platform.
All layers import from here — no cross-layer type dependencies.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


# ── Enumerations ──────────────────────────────────────────────────────────────

class Instrument(str, Enum):
    NIFTY     = "NIFTY"
    BANKNIFTY = "BANKNIFTY"
    FINNIFTY  = "FINNIFTY"
    SENSEX    = "SENSEX"
    BANKEX    = "BANKEX"
    BSEIT     = "BSEIT"


class OptionType(str, Enum):
    CALL = "CE"
    PUT  = "PE"


class SignalDirection(str, Enum):
    LONG    = "LONG"
    SHORT   = "SHORT"
    NEUTRAL = "NEUTRAL"


class OrderSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING   = "PENDING"
    OPEN      = "OPEN"
    FILLED    = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED  = "REJECTED"


class TradeStatus(str, Enum):
    OPEN   = "OPEN"
    CLOSED = "CLOSED"


class RegimeType(str, Enum):
    TRENDING      = "TRENDING"
    MEAN_REVERTING = "MEAN_REVERTING"
    CHOPPY        = "CHOPPY"


# ── Market data ───────────────────────────────────────────────────────────────

@dataclass(slots=True)
class MarketBar:
    timestamp: datetime
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3

    @property
    def true_range(self) -> float:
        return self.high - self.low  # intrabar; cross-bar TR computed in feature engine


@dataclass(slots=True)
class OptionQuote:
    symbol:      str
    instrument:  Instrument
    strike:      float
    option_type: OptionType
    expiry:      date
    ltp:         float
    bid:         float
    ask:         float
    oi:          float
    oi_change:   float
    volume:      float
    iv:          float
    delta:       float
    gamma:       float
    theta:       float
    vega:        float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.mid if self.mid > 0 else 0.0


@dataclass
class OptionChain:
    instrument:  Instrument
    spot:        float
    timestamp:   datetime
    expiry:      date
    quotes:      List[OptionQuote] = field(default_factory=list)
    india_vix:   float = 0.0

    def calls(self) -> List[OptionQuote]:
        return [q for q in self.quotes if q.option_type == OptionType.CALL]

    def puts(self) -> List[OptionQuote]:
        return [q for q in self.quotes if q.option_type == OptionType.PUT]

    def atm_strike(self) -> float:
        """Nearest listed strike to spot."""
        strikes = sorted({q.strike for q in self.quotes})
        if not strikes:
            return self.spot
        return min(strikes, key=lambda s: abs(s - self.spot))

    def quote(self, strike: float, option_type: OptionType) -> Optional[OptionQuote]:
        for q in self.quotes:
            if q.strike == strike and q.option_type == option_type:
                return q
        return None


# ── Features ─────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class FeatureVector:
    timestamp:         datetime
    iv_rank:           float  = 0.0   # 0-1; 1 = historically high IV
    iv_skew:           float  = 0.0   # put IV - call IV at 25-delta
    pc_oi_ratio:       float  = 1.0   # put OI / call OI
    oi_change:         float  = 0.0   # % change in total OI
    delta_volume:      float  = 0.0   # signed volume proxy (calls - puts weighted by delta)
    atr:               float  = 0.0   # 14-bar Average True Range
    atr_pct:           float  = 0.0   # ATR percentile in rolling window
    realized_vol:      float  = 0.0   # 20-bar annualised realised vol
    rv_pct:            float  = 0.0   # realised vol percentile
    entropy:           float  = 0.0   # Shannon entropy of return distribution
    entropy_pct:       float  = 0.0   # entropy percentile
    hurst:             float  = 0.5   # Hurst exponent (0.5 = random walk)
    vwap_distance:     float  = 0.0   # (price - VWAP) / VWAP
    breadth:           float  = 0.5   # % advancing constituents (0-1)
    gamma_exposure:    float  = 0.0   # aggregate dealer GEX (₹ crore)
    range_compression: float  = 1.0   # (high - low) / ATR; < 1 = compression
    range_pct:         float  = 0.5   # range_compression percentile

    def as_array(self) -> np.ndarray:
        return np.array([
            self.iv_rank, self.iv_skew, self.pc_oi_ratio, self.oi_change,
            self.delta_volume, self.atr_pct, self.rv_pct, self.entropy_pct,
            self.hurst, self.vwap_distance, self.breadth,
            self.gamma_exposure, self.range_pct,
        ], dtype=np.float64)

    FEATURE_NAMES = [
        "iv_rank", "iv_skew", "pc_oi_ratio", "oi_change", "delta_volume",
        "atr_pct", "rv_pct", "entropy_pct", "hurst", "vwap_distance",
        "breadth", "gamma_exposure", "range_pct",
    ]


@dataclass
class FeatureStats:
    name:               str
    ic:                 float          # mean Information Coefficient
    ic_std:             float          # std of IC time-series
    ic_pvalue:          float          # two-tailed p-value
    mutual_info:        float          # mutual information with forward return
    t_stat:             float          # t-stat of IC (IC_IR * sqrt(n))
    sharpe_contribution:float          # Sharpe of long/short portfolio on feature sign
    decile_returns:     np.ndarray     # 10-element array (D1..D10 mean fwd return)
    feature_importance: float          # gradient-boosting feature importance
    is_significant:     bool           # passes all significance gates
    n_observations:     int = 0


# ── Signals ───────────────────────────────────────────────────────────────────

@dataclass
class SpreadLeg:
    symbol:      str
    strike:      float
    option_type: OptionType
    side:        OrderSide
    quantity:    int             # number of lots
    lot_size:    int
    limit_price: float

    @property
    def notional(self) -> float:
        return self.limit_price * self.quantity * self.lot_size


@dataclass
class Signal:
    strategy:       str
    instrument:     Instrument
    direction:      SignalDirection
    timestamp:      datetime
    legs:           List[SpreadLeg]
    net_debit:      float          # total premium paid (debit spread > 0)
    max_loss:       float
    max_profit:     float
    confidence:     float          # 0-1
    features:       FeatureVector
    metadata:       Dict[str, Any] = field(default_factory=dict)
    signal_id:      str            = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def expiry(self) -> Optional[date]:
        for leg in self.legs:
            return None  # caller resolves via OptionChain
        return None


# ── Orders and positions ──────────────────────────────────────────────────────

@dataclass
class Order:
    order_id:    str
    symbol:      str
    side:        OrderSide
    quantity:    int
    limit_price: Optional[float]
    status:      OrderStatus
    timestamp:   datetime
    filled_price:Optional[float] = None
    strategy:    str = ""
    trade_id:    str = ""

    @classmethod
    def from_leg(cls, leg: SpreadLeg, strategy: str, trade_id: str) -> "Order":
        return cls(
            order_id=str(uuid.uuid4())[:12],
            symbol=leg.symbol,
            side=leg.side,
            quantity=leg.quantity * leg.lot_size,
            limit_price=leg.limit_price,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(),
            strategy=strategy,
            trade_id=trade_id,
        )


@dataclass
class Position:
    position_id:   str
    trade_id:      str
    strategy:      str
    instrument:    Instrument
    symbol:        str
    side:          OrderSide
    quantity:      int
    entry_price:   float
    current_price: float
    stop_loss:     float
    target:        float
    delta:         float = 0.0
    gamma:         float = 0.0
    theta:         float = 0.0
    vega:          float = 0.0
    entry_time:    datetime = field(default_factory=datetime.now)

    @property
    def unrealized_pnl(self) -> float:
        sign = 1 if self.side == OrderSide.BUY else -1
        return sign * (self.current_price - self.entry_price) * self.quantity

    @property
    def pnl_pct(self) -> float:
        cost = self.entry_price * self.quantity
        return self.unrealized_pnl / cost if cost else 0.0


@dataclass
class Trade:
    trade_id:     str
    strategy:     str
    instrument:   Instrument
    signal_id:    str
    legs:         List[SpreadLeg]
    entry_time:   datetime
    exit_time:    Optional[datetime]
    entry_cost:   float           # net premium paid (cost basis)
    exit_proceeds:float           # net proceeds on exit
    pnl:          float
    pnl_pct:      float
    status:       TradeStatus
    exit_reason:  str = ""
    metadata:     Dict[str, Any] = field(default_factory=dict)


# ── Risk ─────────────────────────────────────────────────────────────────────

@dataclass
class RiskState:
    timestamp:         datetime
    capital:           float
    nav:               float          # net asset value
    daily_pnl:         float
    weekly_pnl:        float
    total_pnl:         float
    peak_nav:          float
    current_drawdown:  float          # % from peak
    max_drawdown:      float          # worst drawdown ever seen
    daily_var_99:      float          # 1-day 99% Value at Risk
    open_positions:    int
    total_delta:       float
    total_gamma:       float
    total_theta:       float
    total_vega:        float
    is_trading_halted: bool = False
    halt_reason:       str  = ""
    strategy_pnl:      Dict[str, float] = field(default_factory=dict)


# ── Performance ───────────────────────────────────────────────────────────────

@dataclass
class PerformanceReport:
    strategy:      str
    start_date:    date
    end_date:      date
    initial_capital: float
    final_capital:   float

    cagr:          float
    sharpe:        float
    sortino:       float
    calmar:        float
    max_drawdown:  float
    profit_factor: float
    expectancy:    float           # per-trade expectancy in ₹
    exposure:      float           # % of time in market
    turnover:      float           # annualised turnover ratio

    total_trades:  int
    win_rate:      float
    avg_win:       float
    avg_loss:      float
    avg_hold_days: float

    equity_curve:  np.ndarray     # daily NAV series
    trade_dates:   List[date]

    passes_validation: bool = False
    validation_notes:  str  = ""

    def summary(self) -> str:
        lines = [
            f"Strategy       : {self.strategy}",
            f"Period         : {self.start_date} → {self.end_date}",
            f"CAGR           : {self.cagr:.1%}",
            f"Sharpe         : {self.sharpe:.2f}",
            f"Sortino        : {self.sortino:.2f}",
            f"Calmar         : {self.calmar:.2f}",
            f"Max Drawdown   : {self.max_drawdown:.1%}",
            f"Profit Factor  : {self.profit_factor:.2f}",
            f"Expectancy     : ₹{self.expectancy:,.0f}",
            f"Win Rate       : {self.win_rate:.1%}  ({self.total_trades} trades)",
            f"Exposure       : {self.exposure:.1%}",
            f"Passes Gate    : {'YES' if self.passes_validation else 'NO'} {self.validation_notes}",
        ]
        return "\n".join(lines)
