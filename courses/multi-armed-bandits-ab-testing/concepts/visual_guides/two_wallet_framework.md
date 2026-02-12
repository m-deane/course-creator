# Two-Wallet Framework

```
┌─────────────────────────────────────────────────┐
│        TWO-WALLET FRAMEWORK                      │
│        (Commodity Allocation)                    │
├─────────────────────────────────────────────────┤
│                                                  │
│  Total Capital                                   │
│  ┌──────────────────────────────────────────┐   │
│  │                                          │   │
│  │  ┌────────────────────┐ ┌──────────┐    │   │
│  │  │   CORE WALLET      │ │  BANDIT  │    │   │
│  │  │      80%           │ │  WALLET  │    │   │
│  │  │                    │ │   20%    │    │   │
│  │  │  Fixed allocation  │ │ Adaptive │    │   │
│  │  │  Boring by design  │ │ Learning │    │   │
│  │  │  Rebalance monthly │ │ Weekly   │    │   │
│  │  │                    │ │ tilts    │    │   │
│  │  │  WTI:  20%         │ │          │    │   │
│  │  │  Gold: 20%         │ │ Thompson │    │   │
│  │  │  Corn: 20%         │ │ Sampling │    │   │
│  │  │  Cu:   20%         │ │ picks    │    │   │
│  │  │  NatGas:20%        │ │ 1-2 arms │    │   │
│  │  └────────────────────┘ └──────────┘    │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  GUARDRAILS:                                     │
│  ├── Max 40% in any single commodity             │
│  ├── Min 5% in any commodity (never zero-out)    │
│  ├── Max 10% tilt change per week                │
│  └── Reduce tilt when volatility spikes          │
│                                                  │
├─────────────────────────────────────────────────┤
│ TL;DR: Keep your long-term plan (core), learn   │
│ with a small slice (bandit). Never let the       │
│ learning system become a gambling system.        │
├─────────────────────────────────────────────────┤
│ Code (< 15 lines):                               │
│                                                  │
│   core_weight = 0.80                             │
│   bandit_weight = 0.20                           │
│   core_alloc = np.ones(5) / 5  # equal weight   │
│   # Thompson Sampling for bandit sleeve          │
│   samples = np.random.normal(mu_hat, sigma_hat)  │
│   bandit_alloc = np.zeros(5)                     │
│   top2 = np.argsort(samples)[-2:]                │
│   bandit_alloc[top2] = 0.5  # split between top2│
│   # Combined allocation                          │
│   total = core_weight*core_alloc + \             │
│           bandit_weight*bandit_alloc             │
│   total = np.clip(total, 0.05, 0.40)  # guard   │
│   total /= total.sum()  # normalize             │
│                                                  │
├─────────────────────────────────────────────────┤
│ Common Pitfall: Making the bandit wallet too     │
│ large. If > 30%, a bad streak hurts your whole   │
│ portfolio. Start at 10-20%, increase only after  │
│ you trust the system.                            │
└─────────────────────────────────────────────────┘
```
