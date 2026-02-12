# Two-Wallet Framework

## TL;DR
Portfolio allocation strategy: core wallet (80%, stable, diversified) + bandit sleeve (20%, adaptive, learning). Core protects against disasters, bandit adapts to find winners. Small sleeve limits damage from learning mistakes.

## Visual Explanation

```
TOTAL PORTFOLIO: $100K
┌─────────────────────────────────────────────────────────┐
│                                                          │
│  CORE WALLET (80% = $80K)                               │
│  ┌────────────────────────────────────────────┐         │
│  │ Equal-weight across all commodities        │         │
│  │ Rebalanced MONTHLY                         │         │
│  │ Stable, predictable, boring                │         │
│  │                                            │         │
│  │ WTI: $16K │ Gold: $16K │ Copper: $16K      │         │
│  │ NatGas: $16K │ Corn: $16K                  │         │
│  └────────────────────────────────────────────┘         │
│                                                          │
│  BANDIT SLEEVE (20% = $20K)                             │
│  ┌────────────────────────────────────────────┐         │
│  │ Thompson Sampling allocation                │         │
│  │ Rebalanced WEEKLY                          │         │
│  │ Tilts toward winners                       │         │
│  │                                            │         │
│  │ THIS WEEK:                                 │         │
│  │ Gold: $8K (40%) ← Tilted UP                │         │
│  │ Copper: $6K (30%)                          │         │
│  │ WTI: $4K (20%)                             │         │
│  │ NatGas: $1K (5%) ← Tilted DOWN             │         │
│  │ Corn: $1K (5%)                             │         │
│  └────────────────────────────────────────────┘         │
│                                                          │
└─────────────────────────────────────────────────────────┘

COMBINED ALLOCATION:
Gold: 16% (core) + 8% (bandit) = 24% total ← Learning it's good
NatGas: 16% (core) + 1% (bandit) = 17% total ← Learning it's weak
```

## Code (< 15 lines)

```python
import numpy as np

# Core: Always equal-weight
core_weights = np.ones(5) / 5  # [0.2, 0.2, 0.2, 0.2, 0.2]

# Bandit: Thompson Sampling (adapts weekly)
bandit_weights = thompson_sampling.get_weights()  # e.g., [0.4, 0.3, 0.2, 0.05, 0.05]

# Combine with 80-20 split
total_weights = 0.8 * core_weights + 0.2 * bandit_weights

# Example: Gold gets 16% + 8% = 24% total
# Even if bandit goes to 0% Gold, core keeps 16% exposure
```

## Common Pitfall

**Bandit sleeve too large**

Beginners think: "Why not 100% bandit? Adapt fully!"

**Why it fails:**
- Bandit can overfit to noise → concentrate in one arm
- One bad week → lose 20% of portfolio
- No safety net

**The fix:**
- Core (60-80%) guarantees diversification
- Bandit (20-40%) adapts but can't destroy portfolio
- Start with 20% bandit, increase only if proven safe

---

**Key Insight:** Don't fight diversification. Let the core keep you safe while the bandit learns.
