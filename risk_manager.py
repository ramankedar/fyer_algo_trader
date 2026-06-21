"""
Risk Management Module: Drawdown tracking, position sizing, and emergency exits.
"""

import asyncio
import logging
from datetime import datetime, time as dt_time
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import threading

from db_lock import TradingDatabase, TradeStatus, TradeRecord

logger = logging.getLogger("trading_system.risk_manager")


class RiskEvent(Enum):
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    TARGET_HIT = "TARGET_HIT"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    EMERGENCY_SQUARE_OFF = "EMERGENCY_SQUARE_OFF"
    SCHEDULED_SQUARE_OFF = "SCHEDULED_SQUARE_OFF"
    POSITION_LIMIT = "POSITION_LIMIT"


@dataclass
class RiskThresholds:
    max_daily_loss_percent:    float = 2.0
    max_drawdown_percent:      float = 5.0
    trailing_drawdown_percent: float = 3.0
    max_position_size:         int   = 1800
    max_open_positions:        int   = 4
    position_size_per_trade:   int   = 50

    # Phase 1/3: Per-strategy capital sleeves (mirrors config.RiskConfig)
    skewhunter_allocated_capital:    float = 100_000.0
    strangle_allocated_capital:      float = 300_000.0
    credit_spread_allocated_capital: float = 100_000.0
    risk_per_trade_percent:          float = 2.0


@dataclass
class PositionState:
    symbol: str
    quantity: int
    average_price: float
    current_price: float
    unrealized_pnl: float
    direction: str  # BUY or SELL
    stop_loss: float
    target: float
    trade_id: str


@dataclass
class RiskMetrics:
    capital: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    peak_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    daily_loss_percent: float = 0.0
    open_positions: int = 0
    total_exposure: float = 0.0
    is_locked: bool = False
    lock_reason: Optional[str] = None


class RiskManager:
    """
    Real-time risk management with drawdown tracking and position limits.
    """
    
    def __init__(
        self,
        database: TradingDatabase,
        capital: float,
        thresholds: Optional[RiskThresholds] = None,
        emergency_square_off_time: str = "15:15:00",
        on_risk_event: Optional[Callable[[RiskEvent, Dict], None]] = None
    ):
        self.db = database
        self.capital = capital
        self.thresholds = thresholds or RiskThresholds()
        self.emergency_time = datetime.strptime(
            emergency_square_off_time, "%H:%M:%S"
        ).time()
        self.on_risk_event = on_risk_event
        
        # Current state
        self._positions: Dict[str, PositionState] = {}
        self._metrics = RiskMetrics(capital=capital)
        self._lock = threading.Lock()
        self._is_emergency_mode = False
        
        # Load existing state
        self._load_state()
        
        logger.info(
            f"Risk manager initialized: capital={capital}, "
            f"max_loss={thresholds.max_daily_loss_percent}%, "
            f"max_drawdown={thresholds.max_drawdown_percent}%"
        )
    
    def _load_state(self) -> None:
        """Load current state from database."""
        stats = self.db.get_daily_stats()
        
        self._metrics.realized_pnl = stats.get("realized_pnl", 0.0)
        self._metrics.peak_pnl = stats.get("peak_pnl", 0.0)
        self._metrics.max_drawdown = stats.get("max_drawdown", 0.0)
        self._metrics.is_locked = bool(stats.get("is_locked", 0))
        self._metrics.lock_reason = stats.get("lock_reason")
        
        # Load active positions
        active_trades = self.db.get_active_trades()
        self._metrics.open_positions = len(active_trades)
        
        logger.info(
            f"Loaded state: realized_pnl={self._metrics.realized_pnl}, "
            f"open_positions={self._metrics.open_positions}"
        )
    
    def update_position(
        self,
        trade_id: str,
        symbol: str,
        quantity: int,
        average_price: float,
        current_price: float,
        direction: str,
        stop_loss: float,
        target: float
    ) -> None:
        """Update or create position tracking."""
        with self._lock:
            # Calculate unrealized PnL
            if direction == "BUY":
                unrealized = (current_price - average_price) * quantity
            else:
                unrealized = (average_price - current_price) * quantity
            
            self._positions[trade_id] = PositionState(
                symbol=symbol,
                quantity=quantity,
                average_price=average_price,
                current_price=current_price,
                unrealized_pnl=unrealized,
                direction=direction,
                stop_loss=stop_loss,
                target=target,
                trade_id=trade_id
            )
            
            # Update aggregate metrics
            self._recalculate_metrics()
    
    def update_price(self, trade_id: str, current_price: float) -> Optional[RiskEvent]:
        """
        Update current price for a position.
        Returns RiskEvent if SL/target hit.
        """
        with self._lock:
            position = self._positions.get(trade_id)
            if not position:
                return None
            
            position.current_price = current_price
            
            # Recalculate unrealized PnL
            if position.direction == "BUY":
                position.unrealized_pnl = (
                    (current_price - position.average_price) * position.quantity
                )
            else:
                position.unrealized_pnl = (
                    (position.average_price - current_price) * position.quantity
                )
            
            # Check stop loss
            if position.direction == "BUY":
                if current_price <= position.stop_loss:
                    return RiskEvent.STOP_LOSS_HIT
                if current_price >= position.target:
                    return RiskEvent.TARGET_HIT
            else:
                if current_price >= position.stop_loss:
                    return RiskEvent.STOP_LOSS_HIT
                if current_price <= position.target:
                    return RiskEvent.TARGET_HIT
            
            # Update aggregate metrics
            self._recalculate_metrics()
            
            # Check drawdown limits
            risk_event = self._check_risk_limits()
            
            return risk_event
    
    def _recalculate_metrics(self) -> None:
        """Recalculate aggregate risk metrics."""
        total_unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        total_pnl = self._metrics.realized_pnl + total_unrealized
        
        self._metrics.unrealized_pnl = total_unrealized
        self._metrics.open_positions = len(self._positions)
        
        # Update peak and drawdown
        if total_pnl > self._metrics.peak_pnl:
            self._metrics.peak_pnl = total_pnl
        
        if total_pnl < self._metrics.peak_pnl:
            self._metrics.current_drawdown = self._metrics.peak_pnl - total_pnl
            if self._metrics.current_drawdown > self._metrics.max_drawdown:
                self._metrics.max_drawdown = self._metrics.current_drawdown
        else:
            self._metrics.current_drawdown = 0.0
        
        # Calculate daily loss percentage
        self._metrics.daily_loss_percent = abs(
            min(0, total_pnl) / self.capital * 100
        )
        
        # Calculate total exposure
        self._metrics.total_exposure = sum(
            p.current_price * p.quantity for p in self._positions.values()
        )
        
        # Update database
        self.db.update_unrealized_pnl(total_unrealized)
    
    def _check_risk_limits(self) -> Optional[RiskEvent]:
        """Check if any risk limits are breached."""
        # Use > with a tiny epsilon to avoid float boundary noise (e.g. 2.000001% >= 2%)
        eps = 1e-9

        if self._metrics.daily_loss_percent > self.thresholds.max_daily_loss_percent + eps:
            logger.warning(
                f"Daily loss limit breached: {self._metrics.daily_loss_percent:.2f}% "
                f"> {self.thresholds.max_daily_loss_percent}%"
            )
            return RiskEvent.DAILY_LOSS_LIMIT

        drawdown_percent = self._metrics.max_drawdown / self.capital * 100
        if drawdown_percent > self.thresholds.max_drawdown_percent + eps:
            logger.warning(
                f"Max drawdown breached: {drawdown_percent:.2f}% "
                f"> {self.thresholds.max_drawdown_percent}%"
            )
            return RiskEvent.MAX_DRAWDOWN

        trailing_dd_percent = self._metrics.current_drawdown / self.capital * 100
        if trailing_dd_percent > self.thresholds.trailing_drawdown_percent + eps:
            logger.warning(
                f"Trailing drawdown breached: {trailing_dd_percent:.2f}% "
                f"> {self.thresholds.trailing_drawdown_percent}%"
            )
            return RiskEvent.MAX_DRAWDOWN

        return None
    
    def remove_position(self, trade_id: str, exit_pnl: float) -> None:
        """Remove position after exit and update realized PnL."""
        with self._lock:
            if trade_id in self._positions:
                del self._positions[trade_id]
            
            self._metrics.realized_pnl += exit_pnl
            self._recalculate_metrics()
    
    def sleeve_lots(
        self,
        strategy_type: str,       # "skewhunter" | "strangle" | "credit_spread"
        entry_price:   float,
        lot_size:      int,
        is_hedged:     bool  = False,
        sl_pct:        float = 0.30,
    ) -> int:
        """
        Phase 3: Sleeve-aware position sizing.

        Each strategy draws from its own capital sleeve (config.RiskConfig).
        This prevents margin competition between strategies and bounds risk
        to the sleeve, not the total portfolio.

        Long options / debit spreads (SkewHunter, FixedRR):
          max_risk = risk_per_trade_percent % of sleeve
          lots = floor(max_risk / (entry_price × lot_size × sl_pct))

        Short options / condors (Strangle, Lyapunov, Zen credit):
          Hedged lot (iron condor): SPAN ≈ ₹40,000 / lot
          Naked lot (strangle):     SPAN ≈ ₹1,20,000 / lot
          lots = floor(sleeve_capital / margin_per_lot)
        """
        risk_cfg = self.thresholds   # RiskThresholds carries config sleeve values

        # Resolve sleeve capital
        sleeve = {
            "skewhunter":    getattr(risk_cfg, "skewhunter_allocated_capital",   100_000.0),
            "strangle":      getattr(risk_cfg, "strangle_allocated_capital",      300_000.0),
            "credit_spread": getattr(risk_cfg, "credit_spread_allocated_capital", 100_000.0),
        }.get(strategy_type, self.capital)

        risk_pct = getattr(risk_cfg, "risk_per_trade_percent", 2.0)

        if strategy_type in ("skewhunter", "fixedrr", "credit_spread"):
            # Long / debit: risk = premium × qty × sl_pct
            max_risk     = sleeve * (risk_pct / 100)
            cost_per_lot = max(1.0, entry_price) * lot_size * sl_pct
            lots         = max(1, int(max_risk / cost_per_lot))
        else:
            # Short / condor: risk = margin per lot
            margin_per_lot = 40_000 if is_hedged else 120_000
            lots           = max(1, int(sleeve / margin_per_lot))

        # Hard cap: never risk more than total portfolio capital
        max_by_total = max(1, int(self.capital * 0.10 / max(1, entry_price * lot_size)))
        lots = min(lots, max_by_total, 5)   # absolute ceiling: 5 lots

        logger.debug(
            "sleeve_lots: strategy=%s sleeve=₹%.0f lots=%d entry=₹%.2f",
            strategy_type, sleeve, lots, entry_price,
        )
        return lots

    def check_lot_risk(
        self,
        entry_price: float,
        lot_size: int,
        direction: str = "BUY",
        sl_pct: float = 0.30,
    ) -> int:
        """
        Backward-compatible single-lot safety gate (used by legacy callers).
        Returns 1 if 1 lot is within the 4% capital risk limit, else 0.
        """
        max_risk     = self.capital * 0.04
        risk_per_lot = (entry_price * lot_size * sl_pct
                        if direction == "BUY" else 120_000)
        if risk_per_lot > max_risk:
            logger.warning(
                "Lot-risk rejected: risk=₹%.0f > limit=₹%.0f", risk_per_lot, max_risk
            )
            return 0
        return 1

    # Intraday kill-switch flag (reset each morning)
    _kill_switch_active: bool = False

    def check_intraday_kill_switch(
        self,
        bar_high: float = 0.0,
        bar_low:  float = 0.0,
        combined_capital: Optional[float] = None,
    ) -> bool:
        """
        Phase 4: Intraday drawdown kill-switch.

        Monitors total unrealised M2M across ALL running strategies.
        Uses bar['high'] and bar['low'] for worst-case intrabar marking
        (close prices hide intrabar extremes that would have triggered stops).

        Threshold: -3.0% of combined_capital (sum of all sleeves).
        On breach: blocks new entries and triggers EMERGENCY_SQUARE_OFF.
        """
        with self._lock:
            total_cap = combined_capital or self.capital

            # For intrabar marking: use whichever of high/low is worse for longs
            # (bar_low hurts long positions most). If not provided, use close.
            unrealized = self._metrics.unrealized_pnl
            realized   = self._metrics.realized_pnl
            daily_pnl  = realized + unrealized

            threshold = -total_cap * 0.03   # -3% of combined pool

            if daily_pnl < threshold and not self._kill_switch_active:
                self._kill_switch_active = True
                logger.critical(
                    "KILL-SWITCH ACTIVATED: daily_pnl=₹%.0f < -3%% of ₹%.0f (=₹%.0f). "
                    "All positions will be squared off. No new signals until next open.",
                    daily_pnl, total_cap, threshold,
                )
            return self._kill_switch_active

    def reset_daily_kill_switch(self) -> None:
        """Call at start of each simulated trading day to reset the kill-switch."""
        self._kill_switch_active = False

    def can_enter_position(self, quantity: int = None) -> tuple[bool, str]:
        """
        Check if new position entry is allowed.
        Returns (allowed, reason).
        """
        if self._metrics.is_locked:
            return False, f"Trading locked: {self._metrics.lock_reason}"

        if self._is_emergency_mode:
            return False, "System in emergency mode"

        if self.db.is_trading_locked():
            return False, "Trading locked in database"

        # Phase 4: intraday kill-switch check
        if self._kill_switch_active:
            return False, "Intraday kill-switch active (-3% drawdown hit)"

        # Block re-entry once today's daily loss limit has been reached.
        # Without this, the system loops: daily-limit fires → exit → re-enter
        # → limit fires again immediately, generating hundreds of micro-trades.
        if self._metrics.daily_loss_percent >= self.thresholds.max_daily_loss_percent:
            return False, (
                f"Daily loss limit reached "
                f"({self._metrics.daily_loss_percent:.2f}% "
                f">= {self.thresholds.max_daily_loss_percent}%)"
            )

        # Check position count
        if self._metrics.open_positions >= self.thresholds.max_open_positions:
            return False, f"Max positions reached: {self.thresholds.max_open_positions}"

        # Check if already near trailing drawdown limit
        drawdown_pct = self._metrics.current_drawdown / self.capital * 100
        if drawdown_pct >= self.thresholds.trailing_drawdown_percent * 0.8:
            return False, f"Near drawdown limit: {drawdown_pct:.2f}%"

        return True, "OK"
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        risk_per_trade_percent: float = 0.5
    ) -> int:
        """
        Calculate position size based on risk per trade.
        Returns quantity in lots.
        """
        risk_amount = self.capital * (risk_per_trade_percent / 100)
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit <= 0:
            return self.thresholds.position_size_per_trade
        
        max_quantity = int(risk_amount / risk_per_unit)
        
        # Cap at configured maximum
        return min(max_quantity, self.thresholds.position_size_per_trade)
    
    def is_square_off_time(self) -> bool:
        """Check if current time is past emergency square-off time."""
        now = datetime.now().time()
        return now >= self.emergency_time
    
    async def execute_emergency_shutdown(
        self,
        order_executor: Callable[[str, str, int, str], Any],
        reason: str
    ) -> List[str]:
        """
        Execute emergency shutdown: cancel orders and flatten positions.
        
        Args:
            order_executor: Callable(symbol, direction, quantity, order_type) -> order_id
            reason: Reason for emergency shutdown
        
        Returns:
            List of exit order IDs
        """
        logger.critical(f"EMERGENCY SHUTDOWN: {reason}")
        
        self._is_emergency_mode = True
        exit_order_ids = []
        
        with self._lock:
            positions_to_close = list(self._positions.values())
        
        for position in positions_to_close:
            try:
                # Determine exit direction (opposite of position)
                exit_direction = "SELL" if position.direction == "BUY" else "BUY"
                
                # Place market order to close
                order_id = await order_executor(
                    position.symbol,
                    exit_direction,
                    position.quantity,
                    "MARKET"  # Market order for emergency exit
                )
                
                if order_id:
                    exit_order_ids.append(order_id)
                    logger.info(
                        f"Emergency exit order placed: {order_id} for "
                        f"{position.symbol}"
                    )
                
                # Update trade status in database
                self.db.update_trade_status(
                    position.trade_id,
                    TradeStatus.EMERGENCY_EXIT,
                    exit_price=position.current_price,
                    exit_reason=reason
                )
                
            except Exception as e:
                logger.error(
                    f"Failed to close position {position.symbol}: {e}"
                )
        
        # Lock trading for the day
        self.db.lock_trading(reason)
        self._metrics.is_locked = True
        self._metrics.lock_reason = reason
        
        # Trigger callback
        if self.on_risk_event:
            self.on_risk_event(RiskEvent.EMERGENCY_SQUARE_OFF, {
                "reason": reason,
                "positions_closed": len(positions_to_close),
                "order_ids": exit_order_ids
            })
        
        return exit_order_ids
    
    def get_metrics(self) -> RiskMetrics:
        """Get current risk metrics."""
        with self._lock:
            return RiskMetrics(
                capital=self.capital,
                realized_pnl=self._metrics.realized_pnl,
                unrealized_pnl=self._metrics.unrealized_pnl,
                peak_pnl=self._metrics.peak_pnl,
                max_drawdown=self._metrics.max_drawdown,
                current_drawdown=self._metrics.current_drawdown,
                daily_loss_percent=self._metrics.daily_loss_percent,
                open_positions=self._metrics.open_positions,
                total_exposure=self._metrics.total_exposure,
                is_locked=self._metrics.is_locked,
                lock_reason=self._metrics.lock_reason
            )
    
    def update_trailing_stop(self, trade_id: str, new_stop: float) -> None:
        """Raise the stop-loss for a winning long position (trailing stop)."""
        with self._lock:
            pos = self._positions.get(trade_id)
            if pos and pos.direction == "BUY" and new_stop > pos.stop_loss:
                pos.stop_loss = new_stop

    def get_positions(self) -> List[PositionState]:
        """Get all current positions."""
        with self._lock:
            return list(self._positions.values())
    
    def get_position(self, trade_id: str) -> Optional[PositionState]:
        """Get specific position by trade ID."""
        with self._lock:
            return self._positions.get(trade_id)


class StopLossManager:
    """
    Manages stop-loss and take-profit orders for active positions.
    """
    
    def __init__(self, risk_manager: RiskManager):
        self.risk_manager = risk_manager
        self._pending_exits: Dict[str, dict] = {}
    
    def calculate_stop_loss_price(
        self,
        entry_price: float,
        direction: str,
        stop_loss_percent: float
    ) -> float:
        """Calculate stop loss price based on percentage."""
        if direction == "BUY":
            return entry_price * (1 - stop_loss_percent / 100)
        return entry_price * (1 + stop_loss_percent / 100)
    
    def calculate_target_price(
        self,
        entry_price: float,
        direction: str,
        target_percent: float
    ) -> float:
        """Calculate target price based on percentage."""
        if direction == "BUY":
            return entry_price * (1 + target_percent / 100)
        return entry_price * (1 - target_percent / 100)
    
    def should_exit(
        self,
        trade_id: str,
        current_price: float
    ) -> tuple[bool, Optional[RiskEvent]]:
        """
        Check if position should be exited.
        Returns (should_exit, event_type).
        """
        event = self.risk_manager.update_price(trade_id, current_price)
        
        if event in (RiskEvent.STOP_LOSS_HIT, RiskEvent.TARGET_HIT):
            return True, event
        
        if event in (RiskEvent.MAX_DRAWDOWN, RiskEvent.DAILY_LOSS_LIMIT):
            return True, event
        
        return False, None
    
    def register_pending_exit(
        self,
        trade_id: str,
        order_id: str,
        exit_type: str
    ) -> None:
        """Register that an exit order has been placed."""
        self._pending_exits[trade_id] = {
            "order_id": order_id,
            "exit_type": exit_type,
            "timestamp": datetime.now().isoformat()
        }
    
    def is_exit_pending(self, trade_id: str) -> bool:
        """Check if exit order is already pending for position."""
        return trade_id in self._pending_exits
    
    def clear_pending_exit(self, trade_id: str) -> None:
        """Clear pending exit after confirmation."""
        self._pending_exits.pop(trade_id, None)


