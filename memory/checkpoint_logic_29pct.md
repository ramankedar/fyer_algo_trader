---
name: checkpoint-logic-29pct
description: Strategy configuration and parameters that produced the 32.25% Nifty / +29.92% SkewHunter result — the "logic of 29%" checkpoint
metadata:
  type: project
---

## Checkpoint: "Logic of 29%" — NIFTY +32.25% Overall, SkewHunter +29.92%

**Date saved:** 2026-06-20
**Backtest period:** 2025-06-25 → 2026-06-20 (244 trading days)
**Capital:** ₹5,00,000

### Key Results
| Strategy         | Trades | Win Rate | P&L%    | Sharpe |
|------------------|--------|----------|---------|--------|
| SkewHunter       | 51     | 100.0%   | +29.92% | 291.39 |
| FixedRR_1to3     | 4      | 100.0%   | +2.08%  | 182.28 |
| ExpiryShortStrangle | 4   | 100.0%   | +1.30%  | 23.20  |
| ZenCreditSpread  | 10     | 60.0%    | +0.46%  | 3.80   |
| LyapunovCredit   | 10     | 50.0%    | -1.20%  | -4.90  |
| CurvatureCredit  | 1      | 0.0%     | -0.31%  | 0.00   |
| **TOTAL NIFTY**  | **80** | **87.5%**| **+32.25%** | **19.85** |

BankNifty: -3.94% (trade starvation), Sensex: -1.51% (no NSE bhavcopy data for BSE)

### What made this work
1. Phase 1-4 institutional exit improvements (Gemini's first set):
   - Time decay cut: 45-min / +5% PnL rule for long options
   - Intrabar high/low SL/target checks (Phase 4) — SkewHunter hits targets intrabar
   - Dynamic trailing stop: +25% trigger → breakeven+1%, trail +10% per +10% gain
   - asyncio.gather for spread legs
2. NSE bhavcopy real data calibration (243/244 days)
3. ExpiryShortStrangle: IV Rank > 0.30 filter, lot sizing via margin cap

### Critical parameters at this checkpoint
```python
# StrategyConfig
skewhunter_alpha1_long = 0.62
skewhunter_alpha2_long = 0.60
skewhunter_alpha1_short = 0.38
skewhunter_alpha2_short = 0.40
fixed_rr_alpha1_long_threshold = 0.60
fixed_rr_alpha2_long_threshold = 0.58
min_premium = 20.0

# BaseStrategy._dynamic_trailing_sl
# Trigger: +25%, Breakeven: +1%, Trail step: +10% per +10% gain
```

### What was causing trade starvation
- FixedRR: only 4 trades in 244 days (alpha conditions too strict)
- ExpiryShortStrangle: only 4 trades (IV rank > 0.30 filter blocked entries in low-VIX 2025)
- CurvatureCreditSpread: only 1 trade (intraday momentum signal rarely strong enough)

**How to apply:** Rollback strategies.py, config.py, backtest.py to these parameter values.
