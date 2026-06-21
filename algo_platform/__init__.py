"""
Algorithmic Options Trading Platform for Indian Markets (NIFTY / BANKNIFTY / FINNIFTY).
Capital: ₹3 lakh  |  Objective: maximise risk-adjusted returns.
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("algo-trader-platform")
except PackageNotFoundError:
    __version__ = "0.1.0"
