# Thompson Sampling

## TL;DR
Sample a plausible reward from each arm's belief distribution, then pick the best sample. Exploration happens automatically when uncertain (wide distributions give diverse samples). Exploitation happens when confident (narrow distributions give consistent samples). No parameters to tune.

## Visual Explanation

```
THOMPSON SAMPLING: SAMPLE → PICK BEST → UPDATE

Round 5 (Early, Uncertain):
┌────────────────────────────────────────┐
│ Arm A: Beta(3,3)                       │
│   ░░░░█████░░░░  Sample: 0.45          │
│                                        │
│ Arm B: Beta(4,2)                       │
│        ░░███████░  Sample: 0.72 ← Pick!│
│                                        │
│ Arm C: Beta(2,4)                       │
│  ░░░███░░░░░░  Sample: 0.31            │
└────────────────────────────────────────┘

Round 100 (Late, Confident):
┌────────────────────────────────────────┐
│ Arm A: Beta(42,59)                     │
│      ▓█▓  Sample: 0.41                 │
│                                        │
│ Arm B: Beta(68,33)                     │
│            ▓███▓  Sample: 0.67 ← Pick! │
│                                        │
│ Arm C: Beta(34,67)                     │
│    ▓█▓  Sample: 0.33                   │
└────────────────────────────────────────┘

Notice: Arm B's posterior tightened around 0.67 → gets picked ~80% of time
        But sometimes A or C gets lucky sample → maintains exploration
```

## Code (< 15 lines)

```python
from scipy.stats import beta
import numpy as np

# Initialize: Beta(1,1) = uniform prior
alpha = np.ones(3)
beta_param = np.ones(3)

def thompson_sampling():
    # Sample from each posterior
    samples = beta.rvs(alpha, beta_param)
    return np.argmax(samples)

def update(arm, reward):
    # Bernoulli reward: 1=success, 0=failure
    alpha[arm] += reward
    beta_param[arm] += (1 - reward)
```

## Common Pitfall

**Using wrong priors for your reward distribution**

Thompson Sampling assumes rewards match the prior family:
- **Bernoulli rewards** (0/1) → Beta prior ✅
- **Gaussian rewards** (continuous) → Normal prior ✅
- **Count data** (0,1,2,...) → Gamma prior ✅

**Commodity mistake:**
Returns are Gaussian, not Bernoulli!

```python
# ❌ WRONG for commodity returns
alpha += return  # Returns can be negative!

# ✅ RIGHT for commodity returns
# Use Normal-Normal conjugacy
mean[arm] = (n*mean[arm] + return) / (n+1)
std[arm] = std[arm] / sqrt(n+1)
```

---

**Key Insight:** Posterior width = uncertainty = exploration. As you learn, posteriors tighten and exploration naturally decreases.