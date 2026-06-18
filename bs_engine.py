"""
Black-Scholes Engine with Newton-Raphson IV Solver.
High-performance vectorized implementation using NumPy/SciPy.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from enum import Enum
import logging

logger = logging.getLogger("trading_system.bs_engine")


class OptionType(Enum):
    CALL = "CE"
    PUT = "PE"


@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    iv: float


@dataclass
class OptionData:
    spot: float
    strike: float
    time_to_expiry: float  # In years
    risk_free_rate: float
    option_type: OptionType
    market_price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[int] = None
    oi: Optional[int] = None


class BlackScholesEngine:
    """
    High-performance Black-Scholes engine with vectorized operations.
    Implements Newton-Raphson method for IV calculation.
    """
    
    # Constants for numerical stability
    MIN_VOL = 0.001
    MAX_VOL = 5.0
    MIN_TIME = 1e-10
    NEWTON_TOLERANCE = 1e-8
    NEWTON_MAX_ITERATIONS = 100
    VEGA_MIN = 1e-10
    
    def __init__(self, risk_free_rate: float = 0.065):
        self.risk_free_rate = risk_free_rate
        self._norm_pdf = norm.pdf
        self._norm_cdf = norm.cdf
    
    def _validate_inputs(
        self,
        spot: Union[float, np.ndarray],
        strike: Union[float, np.ndarray],
        time_to_expiry: Union[float, np.ndarray],
        volatility: Union[float, np.ndarray]
    ) -> bool:
        """Validate inputs for numerical stability."""
        arrays = [spot, strike, time_to_expiry, volatility]
        for arr in arrays:
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                return False
        if np.any(spot <= 0) or np.any(strike <= 0):
            return False
        if np.any(time_to_expiry <= 0) or np.any(volatility <= 0):
            return False
        return True
    
    def _d1(
        self,
        spot: Union[float, np.ndarray],
        strike: Union[float, np.ndarray],
        time_to_expiry: Union[float, np.ndarray],
        volatility: Union[float, np.ndarray],
        risk_free_rate: float
    ) -> Union[float, np.ndarray]:
        """Calculate d1 parameter."""
        sqrt_t = np.sqrt(np.maximum(time_to_expiry, self.MIN_TIME))
        vol_sqrt_t = volatility * sqrt_t
        
        numerator = (
            np.log(spot / strike) + 
            (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry
        )
        return numerator / np.maximum(vol_sqrt_t, self.MIN_VOL)
    
    def _d2(
        self,
        d1: Union[float, np.ndarray],
        volatility: Union[float, np.ndarray],
        time_to_expiry: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Calculate d2 parameter."""
        sqrt_t = np.sqrt(np.maximum(time_to_expiry, self.MIN_TIME))
        return d1 - volatility * sqrt_t
    
    def call_price(
        self,
        spot: Union[float, np.ndarray],
        strike: Union[float, np.ndarray],
        time_to_expiry: Union[float, np.ndarray],
        volatility: Union[float, np.ndarray],
        risk_free_rate: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate Black-Scholes call option price.
        C(S,K,T,r,σ) = S·N(d1) - K·e^(-rT)·N(d2)
        """
        r = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        d1 = self._d1(spot, strike, time_to_expiry, volatility, r)
        d2 = self._d2(d1, volatility, time_to_expiry)
        
        discount = np.exp(-r * time_to_expiry)
        price = spot * self._norm_cdf(d1) - strike * discount * self._norm_cdf(d2)
        
        return np.maximum(price, 0.0)
    
    def put_price(
        self,
        spot: Union[float, np.ndarray],
        strike: Union[float, np.ndarray],
        time_to_expiry: Union[float, np.ndarray],
        volatility: Union[float, np.ndarray],
        risk_free_rate: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate Black-Scholes put option price.
        P(S,K,T,r,σ) = K·e^(-rT)·N(-d2) - S·N(-d1)
        """
        r = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        d1 = self._d1(spot, strike, time_to_expiry, volatility, r)
        d2 = self._d2(d1, volatility, time_to_expiry)
        
        discount = np.exp(-r * time_to_expiry)
        price = strike * discount * self._norm_cdf(-d2) - spot * self._norm_cdf(-d1)
        
        return np.maximum(price, 0.0)
    
    def option_price(
        self,
        spot: Union[float, np.ndarray],
        strike: Union[float, np.ndarray],
        time_to_expiry: Union[float, np.ndarray],
        volatility: Union[float, np.ndarray],
        option_type: OptionType,
        risk_free_rate: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """Calculate option price based on type."""
        if option_type == OptionType.CALL:
            return self.call_price(spot, strike, time_to_expiry, volatility, risk_free_rate)
        return self.put_price(spot, strike, time_to_expiry, volatility, risk_free_rate)
    
    def vega(
        self,
        spot: Union[float, np.ndarray],
        strike: Union[float, np.ndarray],
        time_to_expiry: Union[float, np.ndarray],
        volatility: Union[float, np.ndarray],
        risk_free_rate: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate vega (sensitivity to volatility).
        Vega = S·√T·N'(d1)
        """
        r = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        d1 = self._d1(spot, strike, time_to_expiry, volatility, r)
        sqrt_t = np.sqrt(np.maximum(time_to_expiry, self.MIN_TIME))
        
        return spot * sqrt_t * self._norm_pdf(d1) / 100  # Per 1% vol change
    
    def delta(
        self,
        spot: Union[float, np.ndarray],
        strike: Union[float, np.ndarray],
        time_to_expiry: Union[float, np.ndarray],
        volatility: Union[float, np.ndarray],
        option_type: OptionType,
        risk_free_rate: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate delta (sensitivity to underlying price).
        Call Delta = N(d1)
        Put Delta = N(d1) - 1
        """
        r = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        d1 = self._d1(spot, strike, time_to_expiry, volatility, r)
        
        if option_type == OptionType.CALL:
            return self._norm_cdf(d1)
        return self._norm_cdf(d1) - 1
    
    def gamma(
        self,
        spot: Union[float, np.ndarray],
        strike: Union[float, np.ndarray],
        time_to_expiry: Union[float, np.ndarray],
        volatility: Union[float, np.ndarray],
        risk_free_rate: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate gamma (rate of change of delta).
        Gamma = N'(d1) / (S·σ·√T)
        """
        r = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        d1 = self._d1(spot, strike, time_to_expiry, volatility, r)
        sqrt_t = np.sqrt(np.maximum(time_to_expiry, self.MIN_TIME))
        
        denominator = spot * volatility * sqrt_t
        return self._norm_pdf(d1) / np.maximum(denominator, self.VEGA_MIN)
    
    def theta(
        self,
        spot: Union[float, np.ndarray],
        strike: Union[float, np.ndarray],
        time_to_expiry: Union[float, np.ndarray],
        volatility: Union[float, np.ndarray],
        option_type: OptionType,
        risk_free_rate: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate theta (time decay per day).
        """
        r = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        d1 = self._d1(spot, strike, time_to_expiry, volatility, r)
        d2 = self._d2(d1, volatility, time_to_expiry)
        sqrt_t = np.sqrt(np.maximum(time_to_expiry, self.MIN_TIME))
        
        # First term (common to both)
        term1 = -(spot * volatility * self._norm_pdf(d1)) / (2 * sqrt_t)
        
        if option_type == OptionType.CALL:
            term2 = -r * strike * np.exp(-r * time_to_expiry) * self._norm_cdf(d2)
        else:
            term2 = r * strike * np.exp(-r * time_to_expiry) * self._norm_cdf(-d2)
        
        # Convert to per-day theta (divide by 365)
        return (term1 + term2) / 365
    
    def rho(
        self,
        spot: Union[float, np.ndarray],
        strike: Union[float, np.ndarray],
        time_to_expiry: Union[float, np.ndarray],
        volatility: Union[float, np.ndarray],
        option_type: OptionType,
        risk_free_rate: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate rho (sensitivity to interest rate).
        """
        r = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        d1 = self._d1(spot, strike, time_to_expiry, volatility, r)
        d2 = self._d2(d1, volatility, time_to_expiry)
        
        if option_type == OptionType.CALL:
            return strike * time_to_expiry * np.exp(-r * time_to_expiry) * self._norm_cdf(d2) / 100
        return -strike * time_to_expiry * np.exp(-r * time_to_expiry) * self._norm_cdf(-d2) / 100
    
    def implied_volatility_newton_raphson(
        self,
        market_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        option_type: OptionType,
        risk_free_rate: Optional[float] = None,
        initial_guess: float = 0.2
    ) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        σ_{n+1} = σ_n - (C_BS(σ_n) - C_market) / Vega(σ_n)
        """
        r = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        # Validate inputs
        if market_price <= 0 or spot <= 0 or strike <= 0 or time_to_expiry <= 0:
            logger.warning(f"Invalid inputs for IV calculation: price={market_price}, "
                          f"spot={spot}, strike={strike}, T={time_to_expiry}")
            return None
        
        # Check intrinsic value bounds
        if option_type == OptionType.CALL:
            intrinsic = max(0, spot - strike * np.exp(-r * time_to_expiry))
        else:
            intrinsic = max(0, strike * np.exp(-r * time_to_expiry) - spot)
        
        if market_price < intrinsic * 0.99:  # Allow small tolerance
            logger.debug(f"Market price {market_price} below intrinsic {intrinsic}")
            return None
        
        sigma = initial_guess
        
        for iteration in range(self.NEWTON_MAX_ITERATIONS):
            # Calculate theoretical price
            theo_price = self.option_price(
                spot, strike, time_to_expiry, sigma, option_type, r
            )
            
            # Calculate vega (raw, not per %)
            d1 = self._d1(spot, strike, time_to_expiry, sigma, r)
            sqrt_t = np.sqrt(time_to_expiry)
            vega_raw = spot * sqrt_t * self._norm_pdf(d1)
            
            # Check convergence
            price_diff = theo_price - market_price
            
            if abs(price_diff) < self.NEWTON_TOLERANCE:
                if self.MIN_VOL <= sigma <= self.MAX_VOL:
                    return sigma
                return None
            
            # Prevent division by zero
            if vega_raw < self.VEGA_MIN:
                # Switch to bisection for this region
                logger.debug(f"Low vega at iteration {iteration}, sigma={sigma}")
                break
            
            # Newton-Raphson update
            sigma_new = sigma - price_diff / vega_raw
            
            # Clamp to valid range
            sigma = np.clip(sigma_new, self.MIN_VOL, self.MAX_VOL)
            
            # Check for oscillation
            if iteration > 10 and abs(sigma_new - sigma) < self.NEWTON_TOLERANCE:
                return sigma if self.MIN_VOL <= sigma <= self.MAX_VOL else None
        
        # Fallback to Brent's method if Newton-Raphson fails
        return self.implied_volatility_brent(
            market_price, spot, strike, time_to_expiry, option_type, r
        )
    
    def implied_volatility_brent(
        self,
        market_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        option_type: OptionType,
        risk_free_rate: Optional[float] = None
    ) -> Optional[float]:
        """
        Calculate IV using Brent's method as fallback.
        More robust but slower than Newton-Raphson.
        """
        r = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        def objective(sigma: float) -> float:
            return self.option_price(
                spot, strike, time_to_expiry, sigma, option_type, r
            ) - market_price
        
        try:
            iv = brentq(
                objective,
                self.MIN_VOL,
                self.MAX_VOL,
                xtol=self.NEWTON_TOLERANCE
            )
            return iv
        except ValueError:
            logger.debug(f"Brent's method failed for strike={strike}, price={market_price}")
            return None
    
    def compute_all_greeks(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        market_price: float,
        option_type: OptionType,
        risk_free_rate: Optional[float] = None
    ) -> Optional[Greeks]:
        """Compute all Greeks including IV for an option."""
        r = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        # First solve for IV
        iv = self.implied_volatility_newton_raphson(
            market_price, spot, strike, time_to_expiry, option_type, r
        )
        
        if iv is None:
            return None
        
        return Greeks(
            delta=float(self.delta(spot, strike, time_to_expiry, iv, option_type, r)),
            gamma=float(self.gamma(spot, strike, time_to_expiry, iv, r)),
            theta=float(self.theta(spot, strike, time_to_expiry, iv, option_type, r)),
            vega=float(self.vega(spot, strike, time_to_expiry, iv, r)),
            rho=float(self.rho(spot, strike, time_to_expiry, iv, option_type, r)),
            iv=iv
        )
    
    def compute_iv_surface(
        self,
        spot: float,
        strikes: np.ndarray,
        time_to_expiry: float,
        market_prices: np.ndarray,
        option_type: OptionType,
        risk_free_rate: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute IV for multiple strikes vectorized.
        Returns array of IVs, NaN where calculation failed.
        """
        r = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        ivs = np.full(len(strikes), np.nan)
        
        for i, (strike, price) in enumerate(zip(strikes, market_prices)):
            if price > 0 and strike > 0:
                iv = self.implied_volatility_newton_raphson(
                    price, spot, strike, time_to_expiry, option_type, r
                )
                if iv is not None:
                    ivs[i] = iv
        
        return ivs
    
    def smile_curvature(
        self,
        iv_down: float,
        iv_atm: float,
        iv_up: float,
        delta_strike: float
    ) -> float:
        """
        Calculate second derivative of volatility smile (curvature).
        curvature = (IV_up - 2·IV_atm + IV_down) / (Δstrike)²
        """
        if delta_strike <= 0:
            return 0.0
        return (iv_up - 2 * iv_atm + iv_down) / (delta_strike ** 2)
    
    def skew(
        self,
        iv_otm: float,
        iv_atm: float,
        delta_moneyness: float = 0.25
    ) -> float:
        """
        Calculate volatility skew.
        skew = (IV_OTM - IV_ATM) / Δmoneyness
        """
        if delta_moneyness <= 0:
            return 0.0
        return (iv_otm - iv_atm) / delta_moneyness


class OptionChainAnalyzer:
    """
    Analyzes option chain data to compute strategy signals.
    """
    
    def __init__(self, bs_engine: BlackScholesEngine):
        self.bs = bs_engine
    
    def get_atm_strike(self, spot: float, strike_interval: float = 50.0) -> float:
        """Get ATM strike rounded to nearest interval."""
        return round(spot / strike_interval) * strike_interval
    
    def get_moneyness_strikes(
        self,
        spot: float,
        atm_strike: float,
        strike_interval: float,
        std_dev_levels: list[float] = [0.25, 0.5, 0.75, 1.0]
    ) -> dict[float, float]:
        """
        Get strikes at various standard deviation levels.
        Returns dict mapping std_dev to strike.
        """
        # Approximate using 20% annualized vol and sqrt(T) = 0.1 (about 5 days)
        approx_std = spot * 0.20 * 0.1
        
        strikes = {}
        for level in std_dev_levels:
            strikes[level] = round((atm_strike + level * approx_std) / strike_interval) * strike_interval
            strikes[-level] = round((atm_strike - level * approx_std) / strike_interval) * strike_interval
        
        return strikes
    
    def compute_skew_energy(
        self,
        ivs_otm: np.ndarray,
        ivs_itm: np.ndarray
    ) -> float:
        """
        Calculate skew energy as sum of squared differences.
        E_skew = Σ(IV_OTM - IV_ITM)²
        """
        valid_mask = ~(np.isnan(ivs_otm) | np.isnan(ivs_itm))
        if not np.any(valid_mask):
            return 0.0
        
        diffs = ivs_otm[valid_mask] - ivs_itm[valid_mask]
        return float(np.sum(diffs ** 2))
    
    def compute_energy_differential(
        self,
        call_skew_energy: float,
        put_skew_energy: float
    ) -> float:
        """
        Calculate energy differential between calls and puts.
        E_diff = E_skew_Call - E_skew_Put
        """
        return call_skew_energy - put_skew_energy
    
    def compute_viscosity(
        self,
        bid_volumes: np.ndarray,
        ask_volumes: np.ndarray
    ) -> float:
        """
        Calculate liquidity viscosity (bid-ask imbalance).
        viscosity = (Σbids - Σasks) / (Σbids + Σasks)
        """
        total_bids = np.nansum(bid_volumes)
        total_asks = np.nansum(ask_volumes)
        
        if total_bids + total_asks == 0:
            return 0.0
        
        return (total_bids - total_asks) / (total_bids + total_asks)

