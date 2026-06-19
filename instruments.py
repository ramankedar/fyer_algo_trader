"""
instruments.py — Supported index option instruments.

Add a new entry to INSTRUMENTS to extend backtesting and live trading
to a different underlying without touching any strategy code.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class InstrumentSpec:
    name: str              # Used in Fyers option symbol (e.g. "NIFTY", "BANKNIFTY")
    display_name: str
    exchange: str          # "NSE" or "BSE"
    segment: str           # "NFO" (NSE F&O) or "BFO" (BSE F&O)
    lot_size: int          # Contracts per lot
    strike_interval: float # Strike spacing in ₹
    expiry_weekday: int    # 0=Mon 1=Tue 2=Wed 3=Thu 4=Fri
    spot_symbol: str       # Fyers symbol for the cash index
    min_capital: int       # Rough minimum capital for 1 credit spread lot (₹)


INSTRUMENTS: Dict[str, InstrumentSpec] = {
    # ── NSE ────────────────────────────────────────────────────────────────
    "nifty": InstrumentSpec(
        name="NIFTY",
        display_name="Nifty 50",
        exchange="NSE",
        segment="NFO",
        lot_size=25,
        strike_interval=50.0,
        expiry_weekday=3,               # Thursday
        spot_symbol="NSE:NIFTY50-INDEX",
        min_capital=300_000,
    ),
    "banknifty": InstrumentSpec(
        name="BANKNIFTY",
        display_name="Bank Nifty",
        exchange="NSE",
        segment="NFO",
        lot_size=15,
        strike_interval=100.0,
        expiry_weekday=2,               # Wednesday
        spot_symbol="NSE:NIFTYBANK-INDEX",
        min_capital=250_000,
    ),
    "finnifty": InstrumentSpec(
        name="FINNIFTY",
        display_name="Nifty Financial Services",
        exchange="NSE",
        segment="NFO",
        lot_size=40,
        strike_interval=50.0,
        expiry_weekday=1,               # Tuesday
        spot_symbol="NSE:FINNIFTY-INDEX",
        min_capital=200_000,
    ),
    "midcpnifty": InstrumentSpec(
        name="MIDCPNIFTY",
        display_name="Nifty Midcap Select",
        exchange="NSE",
        segment="NFO",
        lot_size=75,
        strike_interval=25.0,
        expiry_weekday=0,               # Monday
        spot_symbol="NSE:MIDCPNIFTY-INDEX",
        min_capital=150_000,
    ),
    # ── BSE ────────────────────────────────────────────────────────────────
    "sensex": InstrumentSpec(
        name="SENSEX",
        display_name="BSE Sensex",
        exchange="BSE",
        segment="BFO",
        lot_size=10,
        strike_interval=100.0,
        expiry_weekday=4,               # Friday
        spot_symbol="BSE:SENSEX-INDEX",
        min_capital=200_000,
    ),
    "bankex": InstrumentSpec(
        name="BANKEX",
        display_name="BSE Bankex",
        exchange="BSE",
        segment="BFO",
        lot_size=15,
        strike_interval=100.0,
        expiry_weekday=4,               # Friday
        spot_symbol="BSE:BANKEX-INDEX",
        min_capital=250_000,
    ),
}
