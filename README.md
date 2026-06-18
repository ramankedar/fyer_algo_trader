# Indian Options Algorithmic Trading System

A production-ready, SEBI-compliant algorithmic trading system for Indian options markets implementing three high-performing strategies.

## Features

### Strategies
1. **Fixed RR 1:3 (30% SL)** - Skew-based credit spreads with 1:3 risk-reward
2. **Curvature Credit Spread Overnight** - Volatility smile curvature exploitation
3. **SkewHunter** - IV skew and volume-based directional trades

### SEBI Compliance
- Token bucket rate limiting (<10 OPS per segment)
- Daily TOTP-based 2FA authentication
- Whitelisted IP validation
- Real-time drawdown controls

### Technical Features
- Newton-Raphson IV solver with Brent's method fallback
- WebSocket-based real-time data streaming
- SQLite persistent state management with locking
- Graceful shutdown with position protection

## Installation

```bash
# Clone repository
git clone <repository-url>
cd indian-options-algo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

