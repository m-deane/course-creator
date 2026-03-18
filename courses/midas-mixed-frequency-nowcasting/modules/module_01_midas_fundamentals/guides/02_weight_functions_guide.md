# MIDAS Weight Functions

## In Brief

MIDAS weight functions are parameterized mappings from lag index $j$ to weight $w_j(\theta)$. The choice of weight function determines how many parameters must be estimated and what shapes are possible. The Beta polynomial and Almon polynomial are the two most widely used in practice.

## Key Insight

A good weight function family must be: (1) flexible enough to capture the true lag pattern, (2) parsimonious enough to estimate reliably, and (3) constrained to produce non-negative weights that sum to 1. The Beta polynomial satisfies all three with just 2 parameters.

---

## Why Weight Functions Matter

The weight function is the mechanism by which MIDAS encodes the temporal structure of the high-frequency data's influence on the low-frequency outcome.

For quarterly GDP on monthly IP with $K=9$ lags, the weight vector $\mathbf{w} = (w_0, w_1, \ldots, w_8)$ allocates the total effect $\beta$ across 9 specific monthly observations. Different weight patterns correspond to fundamentally different economic stories:

- **Front-loaded** ($w_0 \gg w_8$): The most recent month drives the quarter
- **Uniform** ($w_j = 1/9$): All months equally informative — equivalent to aggregation
- **Hump-shaped** ($w_{3}$ or $w_{4}$ largest): Middle of a one-quarter-back period most informative
- **Back-loaded** ($w_8 \gg w_0$): Older observations matter more (unusual)

Empirically, front-loaded and declining patterns dominate in macro applications.

---

## Family 1: Beta Polynomial

### Definition

The Beta polynomial weights use the Beta distribution PDF evaluated at $K$ equally-spaced points:

$$w_j(\theta_1, \theta_2) = \frac{f_{\text{Beta}}\!\left(\frac{j+0.5}{K};\, \theta_1, \theta_2\right)}{\sum_{l=0}^{K-1} f_{\text{Beta}}\!\left(\frac{l+0.5}{K};\, \theta_1, \theta_2\right)}$$

where $f_{\text{Beta}}(x; \theta_1, \theta_2) = \frac{x^{\theta_1-1}(1-x)^{\theta_2-1}}{B(\theta_1, \theta_2)}$ and $\theta_1, \theta_2 > 0$.

**Convention:** Lag 0 (most recent) corresponds to evaluation near $x = 1$. Lag $K-1$ (oldest) corresponds to evaluation near $x = 0$. Therefore weights decline when $\theta_2 > \theta_1$ (right tail of Beta concentrates near $x=1$, which maps to small lag index).

### Key Shapes

| $\theta_1$ | $\theta_2$ | Shape |
|-----------|-----------|-------|
| 1.0 | 1.0 | Uniform (= equal-weight aggregation) |
| 1.0 | 5.0 | Strongly declining (exponential-like) |
| 1.5 | 4.0 | Declining with slight hump |
| 2.0 | 2.0 | Symmetric bell-shaped |
| 5.0 | 1.0 | Back-loaded (unusual in practice) |

### Implementation

```python
import numpy as np
from scipy.stats import beta as beta_dist

def beta_weights(n_lags, theta1, theta2, zero_endpoint=True):
    """
    Compute Beta polynomial MIDAS weights.

    Parameters
    ----------
    n_lags : int
        Number of high-frequency lags K. Lag 0 is most recent.
    theta1 : float
        First shape parameter (> 0). Controls weight near lag K-1 (oldest).
    theta2 : float
        Second shape parameter (> 0). Controls weight near lag 0 (most recent).
    zero_endpoint : bool
        If True, use (j+0.5)/K evaluation points to avoid Beta singularities
        at x=0 and x=1 when theta parameters are less than 1.

    Returns
    -------
    weights : np.ndarray, shape (K,)
        Normalized weights. weights[0] = weight on lag 0 (most recent).
    """
    # Evaluation points on (0, 1)
    j = np.arange(n_lags)

    if zero_endpoint:
        # Midpoints: avoids boundary issues
        x = (j + 0.5) / n_lags
    else:
        # Exact grid: j / (K-1)
        x = j / max(n_lags - 1, 1)

    # The Beta PDF is highest near x=1 when theta2 > 1.
    # We want lag 0 (most recent) to correspond to x=1, so we use (1 - x):
    # The most recent lag (j=0) maps to x near 0, so PDF at (1-x) is near theta2 end.
    # Actually the standard convention is:
    # - lag j=0 (most recent): corresponds to x_j near 0, so f_Beta(1 - x_j) is near theta2 end
    # To get declining weights with theta2 > theta1, evaluate at (1 - x):
    raw_weights = beta_dist.pdf(1 - x, theta1, theta2)

    # Normalize
    if raw_weights.sum() < 1e-12:
        # Fallback to uniform if Beta is degenerate
        return np.ones(n_lags) / n_lags

    weights = raw_weights / raw_weights.sum()
    return weights


# Examples
import matplotlib.pyplot as plt

theta_configs = [
    (1.0, 1.0, 'Uniform (1.0, 1.0)'),
    (1.0, 5.0, 'Declining (1.0, 5.0)'),
    (1.5, 4.0, 'Typical GDP model (1.5, 4.0)'),
    (2.0, 2.0, 'Bell-shaped (2.0, 2.0)'),
]

K = 12  # 4 quarterly lags × 3 months
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for ax, (t1, t2, label) in zip(axes, theta_configs):
    w = beta_weights(K, t1, t2)
    ax.bar(range(K), w, color='steelblue', alpha=0.8)
    ax.axhline(1/K, color='red', linestyle='--', linewidth=1, label=f'Equal (1/{K})')
    ax.set_title(f'Beta({t1}, {t2}): {label}', fontsize=10)
    ax.set_xlabel('Lag $j$ (0 = most recent)')
    ax.set_ylabel('Weight $w_j$')
    ax.legend(fontsize=8)
    # Mark quarter boundaries
    for q in [3, 6, 9]:
        ax.axvline(q - 0.5, color='gray', linestyle=':', alpha=0.5)

plt.suptitle('Beta Polynomial MIDAS Weight Functions (K=12 lags)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Properties

- **Non-negativity:** Beta PDF is non-negative for all $\theta_1, \theta_2 > 0$
- **Normalized:** Sum constraint satisfied by construction after normalization
- **Parsimonious:** 2 parameters for any $K$
- **Flexible:** Can capture monotone declining, bell-shaped, and hump patterns
- **Nested:** Beta(1,1) = uniform = equal-weight aggregation

---

## Family 2: Exponential Almon Polynomial

### Definition

The Almon polynomial (exponential form) is:

$$w_j(\theta_1, \theta_2) = \frac{\exp\!\left(\theta_1 j + \theta_2 j^2\right)}{\sum_{l=0}^{K-1} \exp\!\left(\theta_1 l + \theta_2 l^2\right)}$$

Parameters $\theta_1, \theta_2 \in \mathbb{R}$ (no positivity constraint — easier to optimize).

### Key Shapes

| $\theta_1$ | $\theta_2$ | Shape |
|-----------|-----------|-------|
| 0 | 0 | Uniform |
| $< 0$ | 0 | Monotone declining |
| $> 0$ | 0 | Monotone increasing (back-loaded) |
| any | $< 0$ | Hump-shaped |
| $< 0$ | $< 0$ | Strongly declining |

### Implementation

```python
def almon_weights(n_lags, theta1, theta2):
    """
    Exponential Almon polynomial MIDAS weights.

    Parameters
    ----------
    n_lags : int
        Number of high-frequency lags K.
    theta1 : float
        Linear parameter (controls monotone slope).
    theta2 : float
        Quadratic parameter (controls curvature/hump shape).

    Returns
    -------
    weights : np.ndarray, shape (K,)
        Normalized weights. weights[0] = weight on lag 0 (most recent).
    """
    j = np.arange(n_lags, dtype=float)
    raw = np.exp(theta1 * j + theta2 * j**2)

    # Handle overflow (occurs for large theta values)
    if not np.all(np.isfinite(raw)):
        raw = np.where(np.isfinite(raw), raw, 0.0)

    total = raw.sum()
    if total < 1e-12:
        return np.ones(n_lags) / n_lags

    return raw / total
```

### Almon vs. Beta: Comparison

| Property | Beta Polynomial | Exponential Almon |
|----------|----------------|------------------|
| Parameters | 2 (both > 0) | 2 (any real) |
| Non-negativity | Guaranteed | Guaranteed (exponential) |
| Optimization | Can hit boundary at 0 | No boundary issues |
| Shapes | Declining, bell, uniform | Declining, hump, uniform |
| Back-loaded | Possible but rare | Easy with $\theta_1 > 0$ |
| Common in literature | Yes (macro) | Yes (finance) |

**Practical guidance:** Use Beta polynomial as the default. Use Almon when NLS struggles with the positivity constraints on $\theta$.

---

## Family 3: Step Functions

### Definition

Step function weights impose a piecewise-constant structure on the weight vector:

$$w_j(\delta) = \delta_s \quad \text{if } j \in \text{group } s$$

where groups $s = 1, 2, \ldots, S$ partition $\{0, 1, \ldots, K-1\}$ and $\delta_s > 0$, $\sum_s n_s \delta_s = 1$ (where $n_s$ is group size).

**Example:** 4 groups for $K=12$:
- Group 1: $j \in \{0, 1, 2\}$ (current quarter) — weight $\delta_1$
- Group 2: $j \in \{3, 4, 5\}$ (one quarter back) — weight $\delta_2$
- Group 3: $j \in \{6, 7, 8\}$ (two quarters back) — weight $\delta_3$
- Group 4: $j \in \{9, 10, 11\}$ (three quarters back) — weight $\delta_4$

```python
def step_weights(n_lags, deltas, group_size=None):
    """
    Step function MIDAS weights.

    Parameters
    ----------
    n_lags : int
        Number of high-frequency lags.
    deltas : array-like
        Unnormalized step values (one per group).
    group_size : int or None
        Size of each group. If None, lags split evenly into len(deltas) groups.

    Returns
    -------
    weights : np.ndarray, shape (K,)
        Normalized step weights.
    """
    n_groups = len(deltas)
    deltas = np.array(deltas, dtype=float)

    if group_size is None:
        group_size = n_lags // n_groups

    weights = np.zeros(n_lags)
    for s, delta_s in enumerate(deltas):
        start = s * group_size
        end = min(start + group_size, n_lags)
        weights[start:end] = delta_s

    # Normalize
    weights = weights / weights.sum()
    return weights
```

### When to Use Step Functions

Step functions are useful when:
1. The frequency ratio $m$ is large (e.g., daily data) and you want groups by week or month
2. You want easily interpretable output ("current quarter", "past quarter" effects)
3. You want a more flexible alternative to Beta polynomial with modest parameter count

**Limitation:** Weights are discontinuous at group boundaries — can create numerical issues in NLS.

---

## Family 4: Unrestricted MIDAS (U-MIDAS)

The unrestricted approach estimates each weight $w_j$ individually via OLS, with no polynomial restriction. This is detailed in Guide 03.

**Key difference:** $K$ parameters instead of 2–3. Feasible when $m$ is small (e.g., $m=3$) and $K$ is not too large.

---

## Visualizing the Weight Function Landscape

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

def compare_weight_families(n_lags=12, figsize=(14, 5)):
    """Compare Beta polynomial, Almon, and step function weights."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f'MIDAS Weight Function Families (K={n_lags} lags)',
                 fontsize=12, fontweight='bold')

    lags = np.arange(n_lags)

    # Beta polynomial family
    ax = axes[0]
    for t1, t2, label in [(1.0, 5.0, 'Declining (1, 5)'),
                           (1.5, 4.0, 'Typical (1.5, 4)'),
                           (2.0, 2.0, 'Bell (2, 2)')]:
        w = beta_weights(n_lags, t1, t2)
        ax.plot(lags, w, '-o', markersize=5, linewidth=2, label=label)
    ax.axhline(1/n_lags, color='black', linestyle='--', linewidth=1,
               label=f'Equal ({1/n_lags:.2f})')
    ax.set_title('Beta Polynomial')
    ax.set_xlabel('Lag index')
    ax.set_ylabel('Weight')
    ax.legend(fontsize=8)

    # Almon family
    ax = axes[1]
    for t1, t2, label in [(-0.3, 0.0, 'Declining (-0.3, 0)'),
                           (-0.1, -0.02, 'Hump (-0.1, -0.02)'),
                           (-0.5, 0.0, 'Steep (-0.5, 0)')]:
        w = almon_weights(n_lags, t1, t2)
        ax.plot(lags, w, '-s', markersize=5, linewidth=2, label=label)
    ax.axhline(1/n_lags, color='black', linestyle='--', linewidth=1,
               label=f'Equal ({1/n_lags:.2f})')
    ax.set_title('Exponential Almon')
    ax.set_xlabel('Lag index')
    ax.legend(fontsize=8)

    # Step function family
    ax = axes[2]
    for deltas, label in [([4, 3, 2, 1], 'Declining quarters'),
                           ([1, 4, 3, 2], 'Peak at Q-1'),
                           ([1, 1, 1, 1], 'Flat quarters')]:
        w = step_weights(n_lags, deltas, group_size=3)
        ax.step(lags, w, linewidth=2, label=label, where='mid')
    ax.set_title('Step Function (quarterly groups)')
    ax.set_xlabel('Lag index')
    ax.legend(fontsize=8)

    for ax in axes:
        # Quarter boundaries
        for q in [3, 6, 9]:
            ax.axvline(q - 0.5, color='gray', linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.show()

compare_weight_families()
```

---

## Choosing a Weight Function Family

In practice, use this decision tree:

```
Is K small (≤ 5)?
  YES → U-MIDAS (unrestricted OLS, no parameterization needed)
  NO  → Parameterized family needed
        ↓
Are hump shapes plausible?
  NO  → Beta(1, θ₂) or Almon with θ₂ = 0 (monotone)
  YES → Full Beta(θ₁, θ₂) or Almon with θ₂ ≠ 0
        ↓
Do you need quarterly groupings?
  YES → Step function (one parameter per group)
  NO  → Beta polynomial (default recommendation)
```

**When Beta polynomial fails:**
- Optimizer consistently hits boundary ($\hat{\theta}_1$ or $\hat{\theta}_2 \approx 0$)
- Switch to Almon (no boundary issues in optimization)
- Or use U-MIDAS if $K$ is manageable

---

## Common Pitfalls

**Pitfall 1: Beta PDF singularity at boundaries.** When $\theta < 1$, the Beta PDF diverges at $x=0$ or $x=1$. Avoid by using midpoint evaluation $(j+0.5)/K$ rather than endpoint evaluation $j/(K-1)$.

**Pitfall 2: Almon overflow.** For large $K$ and $|\theta|$ values, $\exp(\theta_1 j + \theta_2 j^2)$ can overflow. Use log-space computation or clip the argument before exponentiation.

**Pitfall 3: Uniform starting values.** Starting NLS from uniform weights ($\theta_1 = \theta_2 = 1$ for Beta, $\theta_1 = \theta_2 = 0$ for Almon) is often at a saddle point and leads to poor convergence. Use data-informed starting values (correlation profile to pick a declining shape).

**Pitfall 4: Comparing families without accounting for parameterization.** Beta(2,2) and Almon(−0.1, −0.02) may produce similar weight shapes but different flexibility — compare models by information criterion (AIC/BIC), not by which family you prefer a priori.

---

## Connections

- **Builds on:** Guide 01 (MIDAS equation and parameter count)
- **Leads to:** Guide 03 (U-MIDAS), Notebook 02 (weight function comparison), Module 02 (estimation)
- **Related to:** Rational distributed lag (RDL) models, Koyck distributed lag

---

## Practice Problems

1. Compute the Beta polynomial weights for $K=6$, $\theta_1 = 1.0$, $\theta_2 = 3.0$ using the midpoint formula. Verify they sum to 1.

2. The Almon polynomial with $\theta_2 = 0$ produces a monotone weight function. Show that it is equivalent to a geometric lag: $w_j \propto e^{\theta_1 j}$. For what value of $\theta_1$ does this give the same weights as the Beta(1, 5) polynomial (approximately)?

3. A researcher uses a step function with $S=4$ groups of equal size and parameters $\delta = (0.5, 0.3, 0.1, 0.1)$ (unnormalized). Compute the normalized weights and verify they sum to 1.

---

## Further Reading

- Ghysels, E., Sinko, A., & Valkanov, R. (2007). "MIDAS regressions: Further results and new directions." *Econometric Reviews*, 26(1), 53–90. [Weight function taxonomy]
- Chen, X., & Ghysels, E. (2011). "News — good or bad — and its impact on predicting future corporate earnings." *Review of Financial Studies.* [Application with Beta polynomial]
- Andreou, E., Ghysels, E., & Kourtellos, A. (2013). "Should macroeconomic forecasters use daily financial data?" *Journal of Business & Economic Statistics.* [Daily-to-quarterly MIDAS]
