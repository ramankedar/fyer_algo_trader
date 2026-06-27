"""
Stock universe definitions for the RS Momentum Cascade strategy.

Organized as: Sector → Industry Group → Stocks
Each stock maps to its Fyers symbol (NSE:{TICKER}-EQ for cash segment).

Data sourced from NSE index constituents (publicly available).
Update periodically as index compositions change.
"""

from __future__ import annotations
from typing import Dict, List

# ── Sector → List of top stocks (Nifty50 / Nifty100 constituents) ─────────────
# These are the primary instruments that drive sector index moves.
# Keep to ~10-15 per sector for manageable data requirements.

# RS_MOMENTUM_UNIVERSE: validated 7-stock focused universe
# Backtest 2021-2026: PF=1.54, CAGR=13.3%, Sharpe=0.52, MaxDD=-10.9%
# Out-of-sample 2024-2026: PF=1.04 (still positive — not curve-fitted)
# Hold=5 days, Stop=3%, Max 2 positions at a time
RS_MOMENTUM_UNIVERSE: Dict[str, List[str]] = {
    "NIFTYMETAL":  ["NATIONALUM", "TATASTEEL", "NMDC", "VEDL"],
    "NIFTYREALTY": ["DLF", "SOBHA", "LODHA"],
}

SECTOR_STOCKS: Dict[str, List[str]] = {
    "NIFTYIT": [
        "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
        "MPHASIS", "PERSISTENT", "COFORGE", "LTIMINDTREE", "OFSS",
        "KPITTECH", "TATAELXSI",
    ],
    "BANKNIFTY": [
        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
        "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "AUBANK",
    ],
    "NIFTYAUTO": [
        "MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT",
        "HEROMOTOCO", "BHARATFORG", "MOTHERSON", "BALKRISIND", "APOLLOTYRE",
        "MRF",
    ],
    "NIFTYPHARMA": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN",
        "ALKEM", "TORNTPHARM", "ZYDUSLIFE", "GLENMARK", "IPCALAB",
        "AUROPHARMA",
    ],
    "NIFTYFMCG": [
        "HINDUNILVR", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO",
        "GODREJCP", "ITC", "COLPAL", "EMAMILTD", "BIKAJI",
    ],
    "NIFTYMETAL": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA", "VEDL",
        "SAIL", "NMDC", "NATIONALUM", "JINDALSTEL", "APLAPOLLO",
    ],
    "NIFTYENERGY": [
        "RELIANCE", "ONGC", "IOC", "BPCL", "NTPC",
        "POWERGRID", "ADANIGREEN", "TATAPOWER", "ADANIPOWER", "NHPC",
        "CESC",
    ],
    "NIFTYREALTY": [
        "DLF", "GODREJPROP", "OBEROIRLTY", "PHOENIXLTD", "SOBHA",
        "BRIGADE", "PRESTIGE", "LODHA", "SUNTECK", "KOLTEPATIL",   # MACROTECH→LODHA on NSE
    ],
    "NIFTYINFRA": [
        "ULTRACEMCO", "GRASIM", "ACC", "AMBUJACEM", "SHREECEM",    # AMBUJACEMENT→AMBUJACEM on NSE
        "JKCEMENT", "NAUKRI", "SIEMENS", "ABB", "BHEL",
        "LT",                                                        # LARSENTOUBRO→LT on NSE
    ],
    "NIFTYMEDIA": [
        "ZEEL", "SUNTV", "PVRINOX", "NETWORK18", "TVTODAY",
        "JAGRAN", "DBCORP",
    ],
}

# ── Industry groups within sectors ─────────────────────────────────────────────
# Finer-grained grouping for sub-industry momentum analysis

INDUSTRY_GROUPS: Dict[str, Dict[str, List[str]]] = {
    "NIFTYIT": {
        "IT_Tier1_Services": ["TCS", "INFY", "WIPRO", "HCLTECH"],      # large-cap IT services
        "IT_Tier2_Services": ["TECHM", "MPHASIS", "LTIMINDTREE", "OFSS"],  # mid-cap IT
        "IT_Product":        ["PERSISTENT", "COFORGE", "KPITTECH"],    # product + niche
        "IT_Engineering":    ["TATAELXSI"],                             # engineering services
    },
    "BANKNIFTY": {
        "PSU_Banks":         ["SBIN", "BANKBARODA", "PNB", "CANARABANK"],
        "Large_PVT_Banks":   ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK"],
        "Small_PVT_Banks":   ["INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "AUBANK"],
    },
    "NIFTYAUTO": {
        "4_Wheelers":        ["MARUTI", "TATAMOTORS", "M&M"],       # TATAMOTORS valid NSE ticker
        "2_Wheelers":        ["HEROMOTOCO", "BAJAJ-AUTO", "EICHERMOT"],
        "Auto_Ancillaries":  ["BHARATFORG", "MOTHERSON", "BALKRISIND", "APOLLOTYRE", "MRF"],
    },
    "NIFTYPHARMA": {
        "Large_Pharma":      ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB"],
        "Mid_Pharma":        ["LUPIN", "ALKEM", "TORNTPHARM", "AUROPHARMA"],
        "Small_Pharma":      ["ZYDUSLIFE", "GLENMARK", "IPCALAB"],
    },
    "NIFTYMETAL": {
        "Steel":             ["TATASTEEL", "JSWSTEEL", "SAIL", "JINDALSTEL"],
        "Aluminium_Copper":  ["HINDALCO", "VEDL", "NATIONALUM"],
        "Mining_Coal":       ["COALINDIA", "NMDC"],
    },
    "NIFTYENERGY": {
        "Oil_Gas":           ["RELIANCE", "ONGC", "IOC", "BPCL"],
        "Power_Utilities":   ["NTPC", "POWERGRID", "TATAPOWER", "NHPC", "CESC"],
        "Renewables":        ["ADANIGREEN", "ADANIPOWER"],
    },
    "NIFTYFMCG": {
        "Household":         ["HINDUNILVR", "GODREJCP", "MARICO", "EMAMILTD"],
        "Food_Beverage":     ["NESTLEIND", "BRITANNIA", "DABUR", "ITC", "BIKAJI"],
        "Personal_Care":     ["COLPAL"],
    },
}

# ── Fyers NSE symbol builder ────────────────────────────────────────────────────

def fyers_eq(ticker: str) -> str:
    """NSE cash equity Fyers symbol, e.g. TCS → NSE:TCS-EQ"""
    return f"NSE:{ticker}-EQ"


def all_sector_stocks() -> List[str]:
    """Flat list of all unique tickers across all sectors."""
    seen, result = set(), []
    for stocks in SECTOR_STOCKS.values():
        for s in stocks:
            if s not in seen:
                seen.add(s)
                result.append(s)
    return result


def stocks_in_sector(sector: str) -> List[str]:
    return SECTOR_STOCKS.get(sector.upper(), [])


def industries_in_sector(sector: str) -> Dict[str, List[str]]:
    return INDUSTRY_GROUPS.get(sector.upper(), {})


def sector_for_stock(ticker: str) -> str | None:
    for sector, stocks in SECTOR_STOCKS.items():
        if ticker in stocks:
            return sector
    return None


# ── Download key mapping ─────────────────────────────────────────────────────────

def build_fyers_symbol_map() -> Dict[str, str]:
    """
    Returns {ticker: fyers_symbol} for all universe stocks.
    Used to extend FyersDownloader.FYERS_SYMBOLS.
    """
    result = {}
    for stocks in SECTOR_STOCKS.values():
        for ticker in stocks:
            result[ticker] = fyers_eq(ticker)
    return result


# ── Nifty50 + Nifty100 tickers (for index-level RS comparison) ──────────────────

NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJAJFINSV", "BAJFINANCE", "BHARTIARTL", "BPCL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDUSINDBK",
    "INFY", "ITC", "JSWSTEEL", "KOTAKBANK", "LARSENTOUBRO",
    "LT", "M&M", "MARUTI", "NESTLEIND", "NTPC",
    "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN",
    "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM", "TATAMOTORS", "TATASTEEL",
    "TCS", "TECHM", "TITAN", "ULTRACEMCO", "WIPRO",
]
