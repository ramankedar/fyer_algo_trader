"""
SQLite-based persistent state manager for trade management.
Implements database locking to prevent concurrent trade collisions.
"""

import sqlite3
import threading
import json
import logging
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
import time

logger = logging.getLogger("trading_system.db_lock")


class TradeStatus(Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    SQUARED_OFF = "SQUARED_OFF"
    EMERGENCY_EXIT = "EMERGENCY_EXIT"


class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass
class TradeRecord:
    trade_id: str
    strategy_name: str
    symbol: str
    option_type: str
    strike: float
    expiry: str
    entry_price: float
    quantity: int
    direction: str  # BUY or SELL
    product_type: str  # INTRADAY or MARGIN
    status: str
    stop_loss: float
    target: float
    entry_time: str
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None
    leg_id: Optional[str] = None  # For multi-leg strategies
    parent_trade_id: Optional[str] = None


@dataclass
class OrderRecord:
    order_id: str
    trade_id: str
    broker_order_id: Optional[str]
    symbol: str
    order_type: str
    transaction_type: str
    quantity: int
    price: float
    trigger_price: Optional[float]
    status: str
    created_at: str
    updated_at: str
    filled_quantity: int = 0
    average_price: float = 0.0
    rejection_reason: Optional[str] = None


class TradingDatabase:
    """
    Thread-safe SQLite database manager for trading state persistence.
    Implements pessimistic locking for concurrent access control.
    """
    
    def __init__(self, db_path: str = "trading_state.db", lock_timeout: int = 30):
        self.db_path = db_path
        self.lock_timeout = lock_timeout
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.db_path,
                timeout=self.lock_timeout,
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA busy_timeout=30000")
        return self._local.connection
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                strategy_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                option_type TEXT NOT NULL,
                strike REAL NOT NULL,
                expiry TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                direction TEXT NOT NULL,
                product_type TEXT NOT NULL,
                status TEXT NOT NULL,
                stop_loss REAL NOT NULL,
                target REAL NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                exit_price REAL,
                pnl REAL,
                exit_reason TEXT,
                leg_id TEXT,
                parent_trade_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                trade_id TEXT NOT NULL,
                broker_order_id TEXT,
                symbol TEXT NOT NULL,
                order_type TEXT NOT NULL,
                transaction_type TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                trigger_price REAL,
                status TEXT NOT NULL,
                filled_quantity INTEGER DEFAULT 0,
                average_price REAL DEFAULT 0.0,
                rejection_reason TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
            );
            
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                realized_pnl REAL DEFAULT 0.0,
                unrealized_pnl REAL DEFAULT 0.0,
                total_trades INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                max_drawdown REAL DEFAULT 0.0,
                peak_pnl REAL DEFAULT 0.0,
                is_locked INTEGER DEFAULT 0,
                lock_reason TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS trade_locks (
                strategy_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                locked_at TEXT NOT NULL,
                lock_holder TEXT NOT NULL,
                PRIMARY KEY (strategy_name, symbol)
            );
            
            CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
            CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_name);
            CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
            CREATE INDEX IF NOT EXISTS idx_orders_trade_id ON orders(trade_id);
            CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
        """)
        
        conn.commit()
        logger.info(f"Database initialized at {self.db_path}")
    
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction rolled back: {e}")
            raise
    
    def acquire_trade_lock(
        self,
        strategy_name: str,
        symbol: str,
        lock_holder: str
    ) -> bool:
        """
        Acquire exclusive lock for a strategy-symbol combination.
        Returns True if lock acquired, False if already locked.
        """
        with self._write_lock:
            try:
                with self.transaction() as conn:
                    # Check for existing lock
                    cursor = conn.execute(
                        """
                        SELECT lock_holder, locked_at 
                        FROM trade_locks 
                        WHERE strategy_name = ? AND symbol = ?
                        """,
                        (strategy_name, symbol)
                    )
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Lock exists - check if stale (older than 5 minutes)
                        locked_at = datetime.fromisoformat(existing["locked_at"])
                        age_seconds = (datetime.now() - locked_at).total_seconds()
                        
                        if age_seconds < 300:  # 5 minutes
                            logger.warning(
                                f"Lock held by {existing['lock_holder']} for "
                                f"{strategy_name}:{symbol}"
                            )
                            return False
                        
                        # Stale lock - remove and acquire
                        logger.info(f"Removing stale lock for {strategy_name}:{symbol}")
                        conn.execute(
                            "DELETE FROM trade_locks WHERE strategy_name = ? AND symbol = ?",
                            (strategy_name, symbol)
                        )
                    
                    # Acquire new lock
                    conn.execute(
                        """
                        INSERT INTO trade_locks (strategy_name, symbol, locked_at, lock_holder)
                        VALUES (?, ?, ?, ?)
                        """,
                        (strategy_name, symbol, datetime.now().isoformat(), lock_holder)
                    )
                    
                    logger.debug(f"Lock acquired for {strategy_name}:{symbol}")
                    return True
                    
            except sqlite3.IntegrityError:
                logger.warning(f"Failed to acquire lock for {strategy_name}:{symbol}")
                return False
    
    def release_trade_lock(self, strategy_name: str, symbol: str) -> bool:
        """Release trade lock."""
        with self._write_lock:
            with self.transaction() as conn:
                cursor = conn.execute(
                    "DELETE FROM trade_locks WHERE strategy_name = ? AND symbol = ?",
                    (strategy_name, symbol)
                )
                released = cursor.rowcount > 0
                if released:
                    logger.debug(f"Lock released for {strategy_name}:{symbol}")
                return released
    
    def has_active_trade(self, strategy_name: str, symbol: Optional[str] = None) -> bool:
        """Check if strategy has active trades."""
        conn = self._get_connection()
        
        if symbol:
            cursor = conn.execute(
                """
                SELECT COUNT(*) as count FROM trades 
                WHERE strategy_name = ? AND symbol = ? 
                AND status IN ('PENDING', 'ACTIVE', 'PARTIALLY_FILLED')
                """,
                (strategy_name, symbol)
            )
        else:
            cursor = conn.execute(
                """
                SELECT COUNT(*) as count FROM trades 
                WHERE strategy_name = ? 
                AND status IN ('PENDING', 'ACTIVE', 'PARTIALLY_FILLED')
                """,
                (strategy_name,)
            )
        
        return cursor.fetchone()["count"] > 0
    
    def insert_trade(self, trade: TradeRecord) -> bool:
        """Insert new trade record."""
        with self._write_lock:
            with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO trades (
                        trade_id, strategy_name, symbol, option_type, strike, expiry,
                        entry_price, quantity, direction, product_type, status,
                        stop_loss, target, entry_time, leg_id, parent_trade_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade.trade_id, trade.strategy_name, trade.symbol,
                        trade.option_type, trade.strike, trade.expiry,
                        trade.entry_price, trade.quantity, trade.direction,
                        trade.product_type, trade.status, trade.stop_loss,
                        trade.target, trade.entry_time, trade.leg_id,
                        trade.parent_trade_id
                    )
                )
                logger.info(f"Trade inserted: {trade.trade_id}")
                return True
    
    def update_trade_status(
        self,
        trade_id: str,
        status: TradeStatus,
        exit_price: Optional[float] = None,
        exit_reason: Optional[str] = None
    ) -> bool:
        """Update trade status and calculate PnL if closing."""
        with self._write_lock:
            with self.transaction() as conn:
                # Get current trade
                cursor = conn.execute(
                    "SELECT * FROM trades WHERE trade_id = ?",
                    (trade_id,)
                )
                trade = cursor.fetchone()
                
                if not trade:
                    logger.error(f"Trade not found: {trade_id}")
                    return False
                
                pnl = None
                exit_time = None
                
                if status in (TradeStatus.CLOSED, TradeStatus.SQUARED_OFF, 
                             TradeStatus.EMERGENCY_EXIT) and exit_price:
                    exit_time = datetime.now().isoformat()
                    
                    # Calculate PnL
                    direction_mult = 1 if trade["direction"] == "BUY" else -1
                    pnl = (exit_price - trade["entry_price"]) * trade["quantity"] * direction_mult
                    
                    # Update daily stats
                    self._update_daily_pnl(conn, pnl)
                
                conn.execute(
                    """
                    UPDATE trades SET 
                        status = ?,
                        exit_time = COALESCE(?, exit_time),
                        exit_price = COALESCE(?, exit_price),
                        pnl = COALESCE(?, pnl),
                        exit_reason = COALESCE(?, exit_reason),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE trade_id = ?
                    """,
                    (status.value, exit_time, exit_price, pnl, exit_reason, trade_id)
                )
                
                logger.info(f"Trade {trade_id} updated to {status.value}, PnL: {pnl}")
                return True
    
    def _update_daily_pnl(self, conn: sqlite3.Connection, pnl: float) -> None:
        """Update daily realized PnL."""
        today = date.today().isoformat()
        
        conn.execute(
            """
            INSERT INTO daily_stats (date, realized_pnl, total_trades)
            VALUES (?, ?, 1)
            ON CONFLICT(date) DO UPDATE SET
                realized_pnl = realized_pnl + ?,
                total_trades = total_trades + 1,
                winning_trades = winning_trades + CASE WHEN ? > 0 THEN 1 ELSE 0 END,
                losing_trades = losing_trades + CASE WHEN ? < 0 THEN 1 ELSE 0 END,
                updated_at = CURRENT_TIMESTAMP
            """,
            (today, pnl, pnl, pnl, pnl)
        )
    
    def get_active_trades(self, strategy_name: Optional[str] = None) -> List[TradeRecord]:
        """Get all active trades, optionally filtered by strategy."""
        conn = self._get_connection()
        
        if strategy_name:
            cursor = conn.execute(
                """
                SELECT * FROM trades 
                WHERE status IN ('PENDING', 'ACTIVE', 'PARTIALLY_FILLED')
                AND strategy_name = ?
                ORDER BY entry_time DESC
                """,
                (strategy_name,)
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM trades 
                WHERE status IN ('PENDING', 'ACTIVE', 'PARTIALLY_FILLED')
                ORDER BY entry_time DESC
                """
            )
        
        trades = []
        for row in cursor.fetchall():
            trades.append(TradeRecord(
                trade_id=row["trade_id"],
                strategy_name=row["strategy_name"],
                symbol=row["symbol"],
                option_type=row["option_type"],
                strike=row["strike"],
                expiry=row["expiry"],
                entry_price=row["entry_price"],
                quantity=row["quantity"],
                direction=row["direction"],
                product_type=row["product_type"],
                status=row["status"],
                stop_loss=row["stop_loss"],
                target=row["target"],
                entry_time=row["entry_time"],
                exit_time=row["exit_time"],
                exit_price=row["exit_price"],
                pnl=row["pnl"],
                exit_reason=row["exit_reason"],
                leg_id=row["leg_id"],
                parent_trade_id=row["parent_trade_id"]
            ))
        
        return trades
    
    def get_daily_stats(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """Get daily trading statistics."""
        conn = self._get_connection()
        
        if date_str is None:
            date_str = date.today().isoformat()
        
        cursor = conn.execute(
            "SELECT * FROM daily_stats WHERE date = ?",
            (date_str,)
        )
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        
        return {
            "date": date_str,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "max_drawdown": 0.0,
            "peak_pnl": 0.0,
            "is_locked": 0,
            "lock_reason": None
        }
    
    def update_unrealized_pnl(self, unrealized_pnl: float) -> None:
        """Update current unrealized PnL for drawdown tracking."""
        today = date.today().isoformat()
        
        with self._write_lock:
            with self.transaction() as conn:
                # Get current stats
                cursor = conn.execute(
                    "SELECT peak_pnl, realized_pnl FROM daily_stats WHERE date = ?",
                    (today,)
                )
                row = cursor.fetchone()
                
                if row:
                    total_pnl = row["realized_pnl"] + unrealized_pnl
                    peak_pnl = max(row["peak_pnl"], total_pnl)
                    drawdown = peak_pnl - total_pnl if total_pnl < peak_pnl else 0
                    
                    conn.execute(
                        """
                        UPDATE daily_stats SET
                            unrealized_pnl = ?,
                            peak_pnl = ?,
                            max_drawdown = MAX(max_drawdown, ?),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE date = ?
                        """,
                        (unrealized_pnl, peak_pnl, drawdown, today)
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO daily_stats (date, unrealized_pnl, peak_pnl)
                        VALUES (?, ?, ?)
                        """,
                        (today, unrealized_pnl, max(0, unrealized_pnl))
                    )
    
    def lock_trading(self, reason: str) -> None:
        """Lock trading for the day."""
        today = date.today().isoformat()
        
        with self._write_lock:
            with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO daily_stats (date, is_locked, lock_reason)
                    VALUES (?, 1, ?)
                    ON CONFLICT(date) DO UPDATE SET
                        is_locked = 1,
                        lock_reason = ?,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (today, reason, reason)
                )
        
        logger.warning(f"Trading locked: {reason}")
    
    def is_trading_locked(self) -> bool:
        """Check if trading is locked for today."""
        today = date.today().isoformat()
        conn = self._get_connection()
        
        cursor = conn.execute(
            "SELECT is_locked FROM daily_stats WHERE date = ?",
            (today,)
        )
        row = cursor.fetchone()
        
        return bool(row and row["is_locked"])
    
    def insert_order(self, order: OrderRecord) -> bool:
        """Insert new order record."""
        with self._write_lock:
            with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO orders (
                        order_id, trade_id, broker_order_id, symbol, order_type,
                        transaction_type, quantity, price, trigger_price, status,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        order.order_id, order.trade_id, order.broker_order_id,
                        order.symbol, order.order_type, order.transaction_type,
                        order.quantity, order.price, order.trigger_price,
                        order.status, order.created_at, order.updated_at
                    )
                )
                return True
    
    def update_order_status(
        self,
        order_id: str,
        status: OrderStatus,
        broker_order_id: Optional[str] = None,
        filled_quantity: Optional[int] = None,
        average_price: Optional[float] = None,
        rejection_reason: Optional[str] = None
    ) -> bool:
        """Update order status."""
        with self._write_lock:
            with self.transaction() as conn:
                conn.execute(
                    """
                    UPDATE orders SET
                        status = ?,
                        broker_order_id = COALESCE(?, broker_order_id),
                        filled_quantity = COALESCE(?, filled_quantity),
                        average_price = COALESCE(?, average_price),
                        rejection_reason = COALESCE(?, rejection_reason),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE order_id = ?
                    """,
                    (
                        status.value, broker_order_id, filled_quantity,
                        average_price, rejection_reason, order_id
                    )
                )
                return True
    
    def cleanup_old_records(self, days_to_keep: int = 30) -> int:
        """Remove old closed trades and orders."""
        cutoff_date = (
            datetime.now() - 
            __import__("datetime").timedelta(days=days_to_keep)
        ).isoformat()
        
        with self._write_lock:
            with self.transaction() as conn:
                # Get trade IDs to delete
                cursor = conn.execute(
                    """
                    SELECT trade_id FROM trades 
                    WHERE status IN ('CLOSED', 'CANCELLED', 'SQUARED_OFF', 'EMERGENCY_EXIT')
                    AND updated_at < ?
                    """,
                    (cutoff_date,)
                )
                trade_ids = [row["trade_id"] for row in cursor.fetchall()]
                
                if trade_ids:
                    # Delete associated orders
                    placeholders = ",".join("?" * len(trade_ids))
                    conn.execute(
                        f"DELETE FROM orders WHERE trade_id IN ({placeholders})",
                        trade_ids
                    )
                    
                    # Delete trades
                    deleted = conn.execute(
                        f"DELETE FROM trades WHERE trade_id IN ({placeholders})",
                        trade_ids
                    ).rowcount
                    
                    logger.info(f"Cleaned up {deleted} old trade records")
                    return deleted
        
        return 0
    
    def close(self) -> None:
        """Close database connections."""
        if hasattr(self._local, "connection") and self._local.connection:
            self._local.connection.close()
            self._local.connection = None



