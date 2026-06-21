"""
Unit tests for the backtest metrics and cost model.
"""

import pytest
from datetime import datetime, date
from typing import List

import numpy as np

from algo_platform.backtest.costs import IndianOptionsCostModel, CostBreakdown
from algo_platform.backtest.metrics import (
    compute_cagr, compute_sharpe, compute_sortino, compute_max_drawdown,
    compute_calmar, compute_profit_factor, compute_expectancy, compute_win_rate,
    PerformanceAnalyzer, validate_strategy,
)
from algo_platform.core.config import load_config, BacktestConfig
from algo_platform.core.types import (
    Instrument, OrderSide, PerformanceReport, Trade, TradeStatus,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_trade(pnl: float, entry_day: int = 1) -> Trade:
    entry = datetime(2024, 1, entry_day, 9, 30)
    exit_ = datetime(2024, 1, entry_day, 15, 0)
    return Trade(
        trade_id      = f"t{entry_day}",
        strategy      = "test",
        instrument    = Instrument.NIFTY,
        signal_id     = "s1",
        legs          = [],
        entry_time    = entry,
        exit_time     = exit_,
        entry_cost    = 5000.0,
        exit_proceeds = 5000.0 + pnl,
        pnl           = pnl,
        pnl_pct       = pnl / 5000.0,
        status        = TradeStatus.CLOSED,
    )


def _equity(returns: list, start: float = 100_000.0) -> np.ndarray:
    nav = [start]
    for r in returns:
        nav.append(nav[-1] * (1 + r))
    return np.array(nav)


# ── Transaction costs ─────────────────────────────────────────────────────────

class TestCostModel:
    def setup_method(self):
        cfg       = BacktestConfig()
        self.model = IndianOptionsCostModel(cfg)

    def test_buy_has_no_stt(self):
        bd = self.model.compute(100.0, 1, 75, OrderSide.BUY)
        assert bd.stt == 0.0

    def test_sell_has_stt(self):
        bd = self.model.compute(100.0, 1, 75, OrderSide.SELL)
        assert bd.stt > 0.0

    def test_brokerage_flat_cap(self):
        # Small trade: ₹20 flat should dominate
        bd = self.model.compute(1.0, 1, 75, OrderSide.BUY)
        assert bd.brokerage == pytest.approx(20.0, abs=1.0)

    def test_gst_on_brokerage_plus_exchange(self):
        bd = self.model.compute(100.0, 1, 75, OrderSide.BUY)
        expected_gst = 0.18 * (bd.brokerage + bd.exchange_fee)
        assert bd.gst == pytest.approx(expected_gst, rel=0.01)

    def test_stamp_duty_buy_only(self):
        buy_bd  = self.model.compute(100.0, 1, 75, OrderSide.BUY)
        sell_bd = self.model.compute(100.0, 1, 75, OrderSide.SELL)
        assert buy_bd.stamp_duty > 0.0
        assert sell_bd.stamp_duty == 0.0

    def test_total_is_sum_of_components(self):
        bd = self.model.compute(100.0, 2, 75, OrderSide.BUY)
        assert bd.total == pytest.approx(
            bd.brokerage + bd.stt + bd.exchange_fee + bd.gst
            + bd.sebi + bd.stamp_duty + bd.slippage,
            rel=1e-6
        )


# ── CAGR ─────────────────────────────────────────────────────────────────────

def test_cagr_flat_equity():
    eq = np.ones(253)          # flat at 1.0 for one year
    cagr = compute_cagr(eq, 252)
    assert cagr == pytest.approx(0.0, abs=1e-6)


def test_cagr_doubling():
    eq   = np.array([100_000.0, 200_000.0])
    cagr = compute_cagr(eq, 252)
    assert cagr == pytest.approx(1.0, abs=0.01)   # ~100% in one year


def test_cagr_ten_pct_annual():
    annual_ret = 0.10
    eq = _equity([annual_ret / 252] * 252)
    cagr = compute_cagr(eq, 252)
    assert cagr == pytest.approx(annual_ret, abs=0.01)


# ── Sharpe ────────────────────────────────────────────────────────────────────

def test_sharpe_zero_std():
    returns = np.zeros(252)
    sharpe  = compute_sharpe(returns)
    assert sharpe == 0.0


def test_sharpe_positive_for_positive_excess_returns():
    np.random.seed(42)
    rf_daily = (1.065) ** (1 / 252) - 1
    returns  = np.random.normal(rf_daily + 0.001, 0.01, 252)
    sharpe   = compute_sharpe(returns)
    assert sharpe > 0


# ── Sortino ───────────────────────────────────────────────────────────────────

def test_sortino_no_negative_returns():
    returns = np.full(252, 0.001)   # all positive
    sortino = compute_sortino(returns)
    assert sortino == 0.0   # no downside → std_down = 0 → returns 0


# ── Max Drawdown ──────────────────────────────────────────────────────────────

def test_max_drawdown_no_drawdown():
    eq = np.array([100.0, 110.0, 120.0, 130.0])
    assert compute_max_drawdown(eq) == pytest.approx(0.0, abs=1e-6)


def test_max_drawdown_half():
    eq = np.array([100.0, 200.0, 100.0])   # 50% drop from peak
    assert compute_max_drawdown(eq) == pytest.approx(0.5, abs=1e-6)


# ── Profit Factor ─────────────────────────────────────────────────────────────

def test_profit_factor_all_wins():
    trades = [_make_trade(1000.0), _make_trade(500.0)]
    assert compute_profit_factor(trades) == float("inf")


def test_profit_factor_equal_wins_losses():
    trades = [_make_trade(1000.0), _make_trade(-1000.0)]
    assert compute_profit_factor(trades) == pytest.approx(1.0, abs=1e-6)


def test_profit_factor_greater_than_one():
    trades = [_make_trade(2000.0), _make_trade(-1000.0)]
    assert compute_profit_factor(trades) == pytest.approx(2.0, abs=1e-6)


# ── Expectancy ────────────────────────────────────────────────────────────────

def test_expectancy_zero_sum():
    trades = [_make_trade(1000.0), _make_trade(-1000.0)]
    assert compute_expectancy(trades) == pytest.approx(0.0, abs=1e-6)


def test_expectancy_positive():
    trades = [_make_trade(2000.0), _make_trade(-1000.0), _make_trade(500.0)]
    assert compute_expectancy(trades) == pytest.approx(500.0, abs=1e-6)


# ── Win rate ──────────────────────────────────────────────────────────────────

def test_win_rate_all_wins():
    trades = [_make_trade(100.0), _make_trade(200.0)]
    assert compute_win_rate(trades) == 1.0


def test_win_rate_half():
    trades = [_make_trade(100.0), _make_trade(-100.0)]
    assert compute_win_rate(trades) == 0.5


# ── Validation gates ──────────────────────────────────────────────────────────

def test_validation_fails_low_sharpe():
    report = PerformanceReport(
        strategy="test", start_date=date(2024,1,1), end_date=date(2025,1,1),
        initial_capital=300_000, final_capital=330_000,
        cagr=0.10, sharpe=0.90, sortino=1.5, calmar=1.0,   # sharpe < 1.2
        max_drawdown=0.10, profit_factor=1.5, expectancy=1000,
        exposure=0.4, turnover=2.0, total_trades=600,
        win_rate=0.55, avg_win=2000.0, avg_loss=-1000.0, avg_hold_days=0.3,
        equity_curve=np.array([300_000, 330_000]), trade_dates=[],
    )
    passed, notes = validate_strategy(report)
    assert not passed
    assert "Sharpe" in notes


def test_validation_passes_all_gates():
    report = PerformanceReport(
        strategy="test", start_date=date(2024,1,1), end_date=date(2025,1,1),
        initial_capital=300_000, final_capital=360_000,
        cagr=0.20, sharpe=1.5, sortino=2.0, calmar=2.0,
        max_drawdown=0.08, profit_factor=1.6, expectancy=1500,
        exposure=0.5, turnover=3.0, total_trades=600,
        win_rate=0.55, avg_win=2500.0, avg_loss=-1000.0, avg_hold_days=0.3,
        equity_curve=np.array([300_000, 360_000]), trade_dates=[],
    )
    passed, notes = validate_strategy(report)
    assert passed
    assert notes == ""


# ── Full analyzer round-trip ──────────────────────────────────────────────────

def test_performance_analyzer_basic():
    trades    = [_make_trade(1000.0, i) for i in range(1, 11)] + \
                [_make_trade(-500.0, i) for i in range(11, 21)]
    np.random.seed(7)
    rets   = np.random.normal(0.001, 0.005, 252)  # positive drift, some noise
    equity = _equity(rets.tolist(), start=300_000.0)
    analyzer  = PerformanceAnalyzer()
    report    = analyzer.generate_report(
        strategy        = "test",
        trades          = trades,
        equity_curve    = equity,
        initial_capital = 300_000.0,
        start_date      = date(2024, 1, 1),
        end_date        = date(2025, 1, 1),
        total_bars      = 252 * 375,
    )
    assert report.sharpe > 0
    assert report.cagr > 0
    assert 0 <= report.max_drawdown <= 1.0
    assert report.total_trades == 20
