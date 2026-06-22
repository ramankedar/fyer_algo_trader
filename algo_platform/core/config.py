"""
Unified platform configuration — all parameters in one place.
Load from environment / YAML / direct instantiation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()


# ── Instrument specs ──────────────────────────────────────────────────────────

@dataclass
class InstrumentSpec:
    symbol:         str
    lot_size:       int    # units per lot
    tick:           float  # minimum price movement
    expiry_weekday: int    # 0=Mon 1=Tue 2=Wed 3=Thu 4=Fri  (weekly options expiry)

# ── Single source of truth for all instrument constants ───────────────────────
# To change SENSEX expiry day (e.g., from Thu to Fri), edit this one line.
LOT_SIZES: Dict[str, InstrumentSpec] = {
    # NSE indices
    "NIFTY":     InstrumentSpec("NIFTY",     75, 0.05, 3),  # Thursday
    "BANKNIFTY": InstrumentSpec("BANKNIFTY", 30, 0.05, 2),  # Wednesday
    "FINNIFTY":  InstrumentSpec("FINNIFTY",  40, 0.05, 1),  # Tuesday
    # BSE indices
    "SENSEX":    InstrumentSpec("SENSEX",    10, 0.05, 3),  # Thursday
    "BANKEX":    InstrumentSpec("BANKEX",    15, 0.05, 0),  # Monday (BSE BANKEX weekly)
    "BSEIT":     InstrumentSpec("BSEIT",     25, 0.05, 4),  # Friday  (BSE IT; verify lot size)
}


# ── Sub-configs ────────────────────────────────────────────────────────────────

@dataclass
class BrokerConfig:
    app_id:       str = field(default_factory=lambda: os.getenv("BROKER_APP_ID", ""))
    access_token: str = field(default_factory=lambda: os.getenv("BROKER_ACCESS_TOKEN", ""))
    client_id:    str = field(default_factory=lambda: os.getenv("BROKER_CLIENT_ID", ""))
    secret_key:   str = field(default_factory=lambda: os.getenv("BROKER_SECRET_KEY", ""))
    totp_key:     str = field(default_factory=lambda: os.getenv("BROKER_TOTP_KEY", ""))
    pin:          str = field(default_factory=lambda: os.getenv("BROKER_PIN", ""))
    base_url:     str = "https://api-t1.fyers.in/api/v3"
    data_url:     str = "https://api-t1.fyers.in/data"


@dataclass
class ResearchConfig:
    # Feature computation windows
    atr_period:       int   = 14
    rv_window:        int   = 20
    entropy_bins:     int   = 20
    hurst_max_lag:    int   = 100
    iv_rank_window:   int   = 252   # trading days
    percentile_window:int   = 60    # bars for percentile context

    # Statistical significance thresholds
    min_ic_abs:       float = 0.03  # |IC| must exceed this
    min_t_stat:       float = 2.0   # |t-stat| must exceed this (95% CI)
    max_pvalue:       float = 0.05
    min_mi:           float = 0.01

    # Forward return horizon (bars) for IC computation
    forward_horizon:  int   = 1


@dataclass
class StrategyAConfig:
    """Volatility Compression Expansion."""
    name:                  str   = "VolCompressionExpansion"
    # Compression thresholds (percentile rank, 0-1)
    atr_pct_threshold:     float = 0.20
    rv_pct_threshold:      float = 0.20
    entropy_pct_threshold: float = 0.20
    range_pct_threshold:   float = 0.20
    # Breakout parameters
    volume_spike_mult:     float = 2.0   # volume > N × median
    # Spread parameters (OTM distance in index points)
    spread_width_nifty:    float = 50.0
    spread_width_banknifty:float = 100.0
    spread_width_finnifty: float = 25.0
    spread_width_sensex:   float = 200.0
    spread_width_bankex:   float = 200.0
    spread_width_bseit:    float = 400.0
    # Exit parameters
    trail_atr_mult:        float = 1.5   # trailing stop = entry_high - N × ATR
    vol_stop_mult:         float = 2.0   # exit if ATR doubles from entry ATR
    # Time parameters (IST)
    entry_start:           str   = "09:30"
    entry_end:             str   = "14:30"
    square_off:            str   = "15:15"
    min_warmup_bars:       int   = 60


@dataclass
class StrategyBConfig:
    """Intraday Trend Following."""
    name:              str   = "IntradayTrend"
    adx_period:        int   = 14
    adx_threshold:     float = 25.0
    breadth_threshold: float = 0.50    # > 0.5 = positive breadth
    spread_width_nifty:    float = 50.0
    spread_width_banknifty:float = 100.0
    spread_width_finnifty: float = 25.0
    spread_width_sensex:   float = 200.0
    spread_width_bankex:   float = 200.0
    spread_width_bseit:    float = 400.0
    entry_start:       str   = "10:00"
    entry_end:         str   = "13:30"
    square_off:        str   = "15:15"
    min_warmup_bars:   int   = 30


@dataclass
class StrategyCConfig:
    """Expiry Gamma Expansion (Thursday only)."""
    name:                       str   = "GammaExpansion"
    gex_concentration_threshold:float = 0.25   # ATM GEX / total GEX (lowered for synthetic chains)
    entry_start:                str   = "13:30"
    entry_end:                  str   = "15:00"
    square_off:                 str   = "15:15"
    # Straddle vs strangle
    use_strangle:               bool  = False
    strangle_offset_nifty:      float = 100.0  # OTM offset for strangle
    strangle_offset_banknifty:  float = 200.0
    strangle_offset_finnifty:   float = 50.0
    strangle_offset_sensex:     float = 400.0
    strangle_offset_bankex:     float = 400.0
    strangle_offset_bseit:      float = 800.0


@dataclass
class RiskConfig:
    capital:                float = 200_000.0  # ₹2 lakh

    # Per-trade risk
    risk_per_trade_pct:     float = 0.005      # 0.5% of capital

    # Volatility targeting
    target_annual_vol:      float = 0.15       # 15% annual vol target
    vol_lookback:           int   = 20         # bars for realised vol estimate

    # Daily / weekly / portfolio limits
    max_daily_loss_pct:     float = 0.02       # 2% daily loss limit
    max_weekly_loss_pct:    float = 0.05       # 5% weekly loss limit
    max_portfolio_dd_pct:   float = 0.15       # 15% portfolio drawdown limit

    # Drawdown-based size scaling
    # At dd_soft, begin linear scaling; at dd_hard, halt trading
    dd_soft_pct:            float = 0.08       # 8% → start reducing
    dd_hard_pct:            float = 0.15       # 15% → halt
    min_size_factor:        float = 0.25       # never below 25% of base size

    # Position limits
    max_open_trades:        int   = 4
    margin_reserve:         float = 25_000.0   # always keep ₹25K free (scaled to ₹2L capital)

    # ── Barbell Portfolio Capital Sleeves (NEW — does not affect other strategies) ──
    # The Barbell splits total capital into two specialised pools:
    #   theta_capital    → ShortStrangle / BarbellStrangle (steady theta income)
    #   convexity_capital → Option buyers: FixedRR / CompressionBreakout (convex payoff)
    # Sum = capital (200_000). Adjust the split here; nothing else needs changing.
    theta_capital:          float = 120_000.0  # ₹1.2L → strangle margin
    convexity_capital:      float =  80_000.0  # ₹0.8L → directional option buyers


@dataclass
class BacktestConfig:
    # Walk-forward windows
    train_months:    int   = 12
    test_months:     int   = 3
    min_train_trades:int   = 100   # abort fold if train has fewer trades

    # Transaction cost model
    brokerage_flat:  float = 20.0   # ₹20 per executed order
    stt_sell_pct:    float = 0.000625  # 0.0625% on sell premium (F&O options)
    exchange_pct:    float = 0.0000530 # NSE F&O charge
    sebi_pct:        float = 0.0000001 # ₹10 / crore
    gst_pct:         float = 0.18   # on brokerage + exchange charges
    stamp_duty_pct:  float = 0.00003   # 0.003% on buy side only
    slippage_pct:    float = 0.001  # 0.1% of premium as slippage

    # Synthetic IV model (used when live chain not available)
    base_iv:         float = 0.15
    skew_slope:      float = -0.002  # per point OTM for puts

    # Validation gates
    min_profit_factor:float = 1.3
    min_sharpe:       float = 1.2
    max_drawdown:     float = 0.20
    min_trades:       int   = 500


@dataclass
class MonitoringConfig:
    refresh_rate:    float = 1.0   # dashboard refresh seconds
    var_window:      int   = 60    # bars for VaR estimation
    var_confidence:  float = 0.99
    log_level:       str   = "INFO"
    log_file:        str   = "platform.log"


@dataclass
class PlatformConfig:
    broker:     BrokerConfig   = field(default_factory=BrokerConfig)
    research:   ResearchConfig = field(default_factory=ResearchConfig)
    strategy_a: StrategyAConfig = field(default_factory=StrategyAConfig)
    strategy_b: StrategyBConfig = field(default_factory=StrategyBConfig)
    strategy_c: StrategyCConfig = field(default_factory=StrategyCConfig)
    risk:       RiskConfig     = field(default_factory=RiskConfig)
    backtest:   BacktestConfig = field(default_factory=BacktestConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Active instruments
    instruments: List[str] = field(default_factory=lambda: ["NIFTY", "BANKNIFTY"])
    risk_free_rate: float = 0.065   # RBI repo rate

    def lot_size(self, instrument: str) -> int:
        return LOT_SIZES[instrument.upper()].lot_size

    def expiry_weekday(self, instrument: str) -> int:
        """Weekly options expiry weekday (0=Mon … 4=Fri). Change in LOT_SIZES above."""
        return LOT_SIZES[instrument.upper()].expiry_weekday

    def spread_width(self, instrument: str, strategy: str = "A") -> float:
        key = instrument.upper()
        if strategy == "A":
            cfg = self.strategy_a
        elif strategy == "B":
            cfg = self.strategy_b
        else:
            cfg = self.strategy_c
        attr = {
            "NIFTY":     "spread_width_nifty",
            "BANKNIFTY": "spread_width_banknifty",
            "FINNIFTY":  "spread_width_finnifty",
            "SENSEX":    "spread_width_sensex",
            "BANKEX":    "spread_width_bankex",
            "BSEIT":     "spread_width_bseit",
        }.get(key, "spread_width_nifty")
        return getattr(cfg, attr, 50.0)


def load_config() -> PlatformConfig:
    cfg = PlatformConfig()
    # Override capital from env if set
    capital_env = os.getenv("TRADING_CAPITAL")
    if capital_env:
        cfg.risk.capital = float(capital_env)
    return cfg
