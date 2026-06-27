"""
Download commodity data from Yahoo Finance (free, no token needed).

Uses US futures prices in USD — same trend signals as MCX INR data.
Useful when Fyers token is expired or for cross-validation.

Symbols:
  GC=F  → Gold (COMEX, USD/troy oz)
  SI=F  → Silver (COMEX, USD/troy oz)
  CL=F  → Crude Oil WTI (USD/barrel)
  NG=F  → Natural Gas (USD/mmbtu)
  HG=F  → Copper (USD/lb)

Usage:
  pip install yfinance
  python3 download_free_commodities.py
"""

import sys
try:
    import yfinance as yf
except ImportError:
    print("Run: pip install yfinance")
    sys.exit(1)

import pandas as pd
from pathlib import Path

SYMBOLS = {
    "GOLD_USD":   "GC=F",   # Gold futures
    "SILVER_USD": "SI=F",   # Silver futures
    "CRUDE_USD":  "CL=F",   # WTI Crude Oil
    "NATGAS_USD": "NG=F",   # Natural Gas
}

cache = Path("data/cache")

for name, sym in SYMBOLS.items():
    print(f"Downloading {name} ({sym})…", end=" ", flush=True)
    try:
        df = yf.download(sym, start="2020-01-01", progress=False, auto_adjust=True)
        if df.empty:
            print("empty")
            continue
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        df.columns = [c.lower() for c in df.columns]
        path = cache / f"{name}_daily.csv"
        df[["open","high","low","close","volume"]].to_csv(path)
        print(f"{len(df)} bars [{df.index[0].date()} → {df.index[-1].date()}]")
    except Exception as e:
        print(f"ERROR: {e}")

print("\nTo backtest with USD data:")
print("  python3 -m algo_platform.strategies.mcx_trend --instrument SILVER_USD")
print("  (update MCX_SPECS in mcx_trend.py to include SILVER_USD)")
