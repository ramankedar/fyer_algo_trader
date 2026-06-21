"""
Unit tests for the risk and position sizing modules.
"""

import pytest
from datetime import datetime

from algo_platform.core.config import load_config, RiskConfig
from algo_platform.core.types import (
    FeatureVector, Instrument, OptionType, OrderSide, Signal,
    SignalDirection, SpreadLeg,
)
from algo_platform.risk.sizing import VolatilityTargetSizer
from algo_platform.risk.manager import PlatformRiskManager


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_signal(max_loss: float = 5000.0, net_debit: float = 100.0) -> Signal:
    leg = SpreadLeg(
        symbol="NSE:NIFTY24D19500CE",
        strike=19500.0,
        option_type=OptionType.CALL,
        side=OrderSide.BUY,
        quantity=1,
        lot_size=75,
        limit_price=100.0,
    )
    fv = FeatureVector(timestamp=datetime.now(), realized_vol=0.15)
    return Signal(
        strategy   = "test",
        instrument = Instrument.NIFTY,
        direction  = SignalDirection.LONG,
        timestamp  = datetime.now(),
        legs       = [leg],
        net_debit  = net_debit,
        max_loss   = max_loss,
        max_profit = 10000.0,
        confidence = 0.7,
        features   = fv,
    )


# ── Sizer: basic lot computation ──────────────────────────────────────────────

def test_sizer_returns_positive_lots():
    cfg    = RiskConfig(capital=300_000.0, risk_per_trade_pct=0.005)
    sizer  = VolatilityTargetSizer(cfg)
    signal = _make_signal(max_loss=5000.0)
    lots   = sizer.compute_lots(nav=300_000.0, signal=signal, lot_size=75)
    assert lots >= 1


def test_sizer_zero_on_hard_drawdown():
    cfg    = RiskConfig(capital=300_000.0, dd_hard_pct=0.15)
    sizer  = VolatilityTargetSizer(cfg)
    signal = _make_signal()
    # NAV at 80% of capital = 20% drawdown > 15% hard
    lots   = sizer.compute_lots(nav=240_000.0, signal=signal, lot_size=75)
    assert lots == 0


def test_sizer_reduced_on_soft_drawdown():
    cfg    = RiskConfig(capital=300_000.0, dd_soft_pct=0.08, dd_hard_pct=0.15)
    sizer  = VolatilityTargetSizer(cfg)
    signal = _make_signal()
    lots_no_dd   = sizer.compute_lots(nav=300_000.0, signal=signal, lot_size=75)
    # 10% drawdown — in soft zone
    lots_with_dd = sizer.compute_lots(nav=270_000.0, signal=signal, lot_size=75)
    assert lots_with_dd <= lots_no_dd


# ── Sizer: limit checks ───────────────────────────────────────────────────────

def test_daily_limit_ok():
    cfg   = RiskConfig(capital=300_000.0, max_daily_loss_pct=0.02)
    sizer = VolatilityTargetSizer(cfg)
    assert sizer.check_daily_limit(-5000.0) is True   # -5K < 2% of 300K = -6K


def test_daily_limit_breached():
    cfg   = RiskConfig(capital=300_000.0, max_daily_loss_pct=0.02)
    sizer = VolatilityTargetSizer(cfg)
    assert sizer.check_daily_limit(-7000.0) is False  # -7K > -6K limit


def test_weekly_limit_ok():
    cfg   = RiskConfig(capital=300_000.0, max_weekly_loss_pct=0.05)
    sizer = VolatilityTargetSizer(cfg)
    assert sizer.check_weekly_limit(-10_000.0) is True  # -10K < 5% of 300K = -15K


def test_portfolio_limit_ok():
    cfg   = RiskConfig(capital=300_000.0, max_portfolio_dd_pct=0.15)
    sizer = VolatilityTargetSizer(cfg)
    assert sizer.check_portfolio_limit(280_000.0) is True  # 6.7% dd < 15%


def test_portfolio_limit_breached():
    cfg   = RiskConfig(capital=300_000.0, max_portfolio_dd_pct=0.15)
    sizer = VolatilityTargetSizer(cfg)
    assert sizer.check_portfolio_limit(250_000.0) is False  # 16.7% dd > 15%


# ── RiskManager ───────────────────────────────────────────────────────────────

def test_risk_manager_can_open_initially():
    config  = load_config()
    manager = PlatformRiskManager(config)
    assert manager.can_open("test_strategy") is True


def test_risk_manager_halts_on_daily_loss():
    config  = load_config()
    manager = PlatformRiskManager(config)
    # Simulate a large loss trade
    from algo_platform.core.types import Trade, TradeStatus
    trade = Trade(
        trade_id      = "t1",
        strategy      = "test",
        instrument    = Instrument.NIFTY,
        signal_id     = "s1",
        legs          = [],
        entry_time    = datetime.now(),
        exit_time     = datetime.now(),
        entry_cost    = 10000.0,
        exit_proceeds = 0.0,
        pnl           = -config.risk.capital * (config.risk.max_daily_loss_pct + 0.01),
        pnl_pct       = -0.021,
        status        = TradeStatus.CLOSED,
    )
    manager.on_trade_closed(trade)
    assert manager.can_open("test_strategy") is False
    assert manager.is_halted is True


def test_risk_manager_on_bar_returns_state():
    config  = load_config()
    manager = PlatformRiskManager(config)
    state   = manager.on_bar(datetime.now(), [])
    assert state.capital == config.risk.capital
    assert state.open_positions == 0
    assert not state.is_trading_halted
