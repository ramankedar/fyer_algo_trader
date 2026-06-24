---
name: prod-theta-deployment-decision
description: Cross-window sensitivity analysis confirmed 13:15 entry for ProductionThetaStrategy deployment
metadata:
  type: project
---

ProductionThetaStrategy deployment parameters settled after 4-window sensitivity analysis.

**Recommended parameters: entry=13:15 / entry_end=13:25 / buffer=0.50 / cat=4.0 / sigma=0.30**

**Why:** 13:15 outperforms 13:30 in all 4 windows (1yr, 2yr, 3yr, full 3.25yr). In the most
recent 1yr (2025-2026), 13:30 produces NEGATIVE expectancy (-₹695/trade) while 13:15 stays
positive (+₹481/trade). The 1yr 13:30 failure is real: the engine's risk gate cut trade count
from 42 to 32 because cat-stop losses compounded into portfolio drawdown. The causal mechanism
is structural: 13:15 gives 2h15m of theta vs 1h45m for 13:30.

**Why:** buffer=0.50 unchanged — empirically insensitive across all combos in all windows.
**Why:** cat=4.0 unchanged — forensic study showed 73% whipsaw rate for tighter stops.

**Deployment sequence**: Paper trade 8 Thursdays → review fills vs assumed → deploy 1 lot.

**Code change**: Entry parameters are constructor args, no file edits needed.

```python
strategy = ProductionThetaStrategy(
    Instrument("NIFTY"),
    config,
    entry_time="13:15",
    entry_end_time="13:25",
    buffer_mult=0.50,
    cat_premium_mult=4.0,
    otm_sigma_mult=0.30,
)
```

**NOT yet live** — pending paper trading validation as of 2026-06-24.
