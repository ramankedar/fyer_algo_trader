"""
Performance metrics for strategy evaluation.
All metrics use purely out-of-sample (post-cost) trade records.
"""

from __future__ import annotations

import math
from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from algo_platform.core.types import PerformanceReport, Trade, TradeStatus


# ── Individual metric functions ───────────────────────────────────────────────

def compute_cagr(equity: np.ndarray, n_trading_days: int,
                 trading_days_per_year: int = 252) -> float:
    """Compound Annual Growth Rate from an equity curve (NAV series)."""
    if equity[0] <= 0 or n_trading_days <= 0:
        return 0.0
    years = n_trading_days / trading_days_per_year
    return float((equity[-1] / equity[0]) ** (1.0 / years) - 1.0)


def compute_sharpe(returns: np.ndarray, risk_free: float = 0.065,
                   annualise: int = 252) -> float:
    """Annualised Sharpe ratio. `returns` are daily fractional returns."""
    if len(returns) < 2:
        return 0.0
    rf_daily  = (1 + risk_free) ** (1 / annualise) - 1
    excess    = returns - rf_daily
    std       = float(np.std(excess, ddof=1))
    if std < 1e-10:
        return 0.0
    return float(np.mean(excess) / std * math.sqrt(annualise))


def compute_sortino(returns: np.ndarray, risk_free: float = 0.065,
                    annualise: int = 252) -> float:
    """Sortino ratio (penalises only downside deviation)."""
    if len(returns) < 2:
        return 0.0
    rf_daily   = (1 + risk_free) ** (1 / annualise) - 1
    excess     = returns - rf_daily
    downside   = excess[excess < 0]
    if len(downside) < 2:
        return 0.0
    std_down   = float(np.std(downside, ddof=1))
    if std_down < 1e-10:
        return 0.0
    return float(np.mean(excess) / std_down * math.sqrt(annualise))


def compute_max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a fraction (positive number)."""
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd   = (peak - equity) / np.where(peak > 0, peak, 1)
    return float(np.max(dd))


def compute_calmar(cagr: float, max_dd: float) -> float:
    """Calmar ratio = CAGR / Max Drawdown."""
    return float(cagr / max_dd) if max_dd > 1e-6 else 0.0


def compute_profit_factor(trades: List[Trade]) -> float:
    """Sum of gross wins / |sum of gross losses|."""
    wins   = sum(t.pnl for t in trades if t.pnl > 0)
    losses = sum(t.pnl for t in trades if t.pnl < 0)
    return float(wins / -losses) if losses < 0 else float("inf")


def compute_expectancy(trades: List[Trade]) -> float:
    """Per-trade expectancy in ₹: E[PnL per trade]."""
    if not trades:
        return 0.0
    return float(np.mean([t.pnl for t in trades]))


def compute_exposure(trades: List[Trade], total_bars: int,
                     bars_per_day: int = 375) -> float:
    """Fraction of bars during which the strategy held a position (0-1)."""
    if total_bars <= 0 or not trades:
        return 0.0
    # Approximate: assume each trade lasted its hold period
    # For a more precise measure the engine can track this directly
    avg_bars_held = np.mean([
        (t.exit_time - t.entry_time).total_seconds() / 60
        if t.exit_time else bars_per_day
        for t in trades
    ])
    total_trade_bars = len(trades) * avg_bars_held
    return float(min(1.0, total_trade_bars / total_bars))


def compute_turnover(trades: List[Trade], initial_capital: float,
                     annualise: int = 252) -> float:
    """Annualised turnover (total notional / capital / trading_days × 252)."""
    if not trades or initial_capital <= 0:
        return 0.0
    total_notional = sum(abs(t.entry_cost) for t in trades)
    # number of unique days
    days = len({t.entry_time.date() for t in trades})
    if days <= 0:
        return 0.0
    daily_turnover = total_notional / initial_capital / days
    return float(daily_turnover * annualise)


def compute_win_rate(trades: List[Trade]) -> float:
    if not trades:
        return 0.0
    return float(sum(1 for t in trades if t.pnl > 0) / len(trades))


def compute_avg_win_loss(trades: List[Trade]) -> tuple[float, float]:
    wins   = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl < 0]
    avg_w  = float(np.mean(wins))   if wins   else 0.0
    avg_l  = float(np.mean(losses)) if losses else 0.0
    return avg_w, avg_l


# ── Validation gates ──────────────────────────────────────────────────────────

VALIDATION_GATES = {
    "min_profit_factor": 1.3,
    "min_sharpe":        1.2,
    "max_drawdown":      0.20,
    "min_trades":        500,
}


def validate_strategy(report: PerformanceReport,
                       gates: dict = VALIDATION_GATES) -> tuple[bool, str]:
    """Apply the four mandatory validation gates."""
    notes: List[str] = []
    passed = True

    if report.profit_factor < gates["min_profit_factor"]:
        notes.append(f"PF={report.profit_factor:.2f}<{gates['min_profit_factor']}")
        passed = False
    if report.sharpe < gates["min_sharpe"]:
        notes.append(f"Sharpe={report.sharpe:.2f}<{gates['min_sharpe']}")
        passed = False
    if report.max_drawdown > gates["max_drawdown"]:
        notes.append(f"MaxDD={report.max_drawdown:.1%}>{gates['max_drawdown']:.0%}")
        passed = False
    if report.total_trades < gates["min_trades"]:
        notes.append(f"Trades={report.total_trades}<{gates['min_trades']}")
        passed = False

    return passed, " | ".join(notes)


# ── Main analyzer ─────────────────────────────────────────────────────────────

class PerformanceAnalyzer:
    """Computes the full performance report from a list of trades + equity curve."""

    def __init__(self, risk_free: float = 0.065,
                 validation_gates: dict = VALIDATION_GATES) -> None:
        self.risk_free = risk_free
        self.gates     = validation_gates

    def generate_report(
        self,
        strategy:        str,
        trades:          List[Trade],
        equity_curve:    np.ndarray,   # daily NAV, length = n_trading_days
        initial_capital: float,
        start_date:      date,
        end_date:        date,
        total_bars:      int,
    ) -> PerformanceReport:
        closed = [t for t in trades if t.status == TradeStatus.CLOSED]

        if len(equity_curve) < 2:
            daily_returns = np.array([0.0])
        else:
            daily_returns = np.diff(equity_curve) / np.where(equity_curve[:-1] > 0,
                                                              equity_curve[:-1], 1)

        n_days   = len(equity_curve) - 1
        cagr     = compute_cagr(equity_curve, n_days)
        sharpe   = compute_sharpe(daily_returns, self.risk_free)
        sortino  = compute_sortino(daily_returns, self.risk_free)
        max_dd   = compute_max_drawdown(equity_curve)
        calmar   = compute_calmar(cagr, max_dd)
        pf       = compute_profit_factor(closed)
        exp      = compute_expectancy(closed)
        exposure = compute_exposure(closed, total_bars)
        turnover = compute_turnover(closed, initial_capital)
        win_rate = compute_win_rate(closed)
        avg_w, avg_l = compute_avg_win_loss(closed)

        if closed:
            hold_secs = []
            for t in closed:
                if t.exit_time:
                    hold_secs.append((t.exit_time - t.entry_time).total_seconds())
            avg_hold = float(np.mean(hold_secs)) / 86400.0 if hold_secs else 0.0
        else:
            avg_hold = 0.0

        report = PerformanceReport(
            strategy        = strategy,
            start_date      = start_date,
            end_date        = end_date,
            initial_capital = initial_capital,
            final_capital   = float(equity_curve[-1]),
            cagr            = cagr,
            sharpe          = sharpe,
            sortino         = sortino,
            calmar          = calmar,
            max_drawdown    = max_dd,
            profit_factor   = pf,
            expectancy      = exp,
            exposure        = exposure,
            turnover        = turnover,
            total_trades    = len(closed),
            win_rate        = win_rate,
            avg_win         = avg_w,
            avg_loss        = avg_l,
            avg_hold_days   = avg_hold,
            equity_curve    = equity_curve,
            trade_dates     = [t.entry_time.date() for t in closed],
        )

        passed, notes = validate_strategy(report, self.gates)
        report.passes_validation = passed
        report.validation_notes  = notes
        return report
