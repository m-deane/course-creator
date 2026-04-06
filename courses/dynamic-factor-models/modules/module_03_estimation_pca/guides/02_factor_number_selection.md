# Factor Number Selection: Information Criteria and Diagnostics

> **Reading time:** ~15 min | **Module:** Module 3: Estimation Pca | **Prerequisites:** Modules 0-2

<div class="callout-key">

**Key Concept Summary:** Choosing the number of factors $r$ is critical in factor model estimation. Too few factors underfit the common variation; too many overfit idiosyncratic noise. Information criteria like Bai-Ng penalize model complexity while rewarding fit, providing data-driven factor selection. Visual diagnostic...

</div>

## In Brief

Choosing the number of factors $r$ is critical in factor model estimation. Too few factors underfit the common variation; too many overfit idiosyncratic noise. Information criteria like Bai-Ng penalize model complexity while rewarding fit, providing data-driven factor selection. Visual diagnostics like scree plots and eigenvalue ratios complement formal criteria.

<div class="callout-insight">

**Insight:** The core trade-off: adding factors always reduces in-sample fit (smaller residuals), but additional factors may capture noise rather than signal. Information criteria formalize this trade-off by penalizing parameters. The key innovation of Bai-Ng criteria is designing penalties that reflect the large-$N$-large-$T$ asymptotics of factor models, unlike AIC/BIC which assume fixed dimension.

</div>
---

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## 1. Formal Definition

### The Factor Number Selection Problem

Given data $X$ ($T \times N$), choose $r$ to minimize:

$$IC(r) = V(r) + g(r, N, T)$$

where:
- $V(r)$: Measure of lack of fit with $r$ factors
- $g(r, N, T)$: Penalty function for model complexity

Different criteria use different $(V, g)$ specifications.

### Bai-Ng Information Criteria

**Bai & Ng (2002)** propose three criteria based on sum of squared residuals:

$$V(r) = \frac{1}{NT} \sum_{i=1}^N \sum_{t=1}^T \hat{e}_{it}^2(r)$$

where $\hat{e}_{it}(r) = X_{it} - \hat{\lambda}_i(r)' \hat{F}_t(r)$ are residuals with $r$ factors.

Equivalently (for standardized data):
$$V(r) = 1 - \frac{1}{N} \sum_{j=1}^r \frac{d_j}{\text{tr}(\hat{\Sigma}_X)} = 1 - \frac{1}{N} \sum_{j=1}^r d_j$$

where $d_j$ are eigenvalues of sample covariance (standardized to mean 1).

### Three Penalty Functions

**IC₁:**
$$IC_1(r) = \ln[V(r)] + r \left(\frac{N+T}{NT}\right) \ln\left(\frac{NT}{N+T}\right)$$

**IC₂:**
$$IC_2(r) = \ln[V(r)] + r \left(\frac{N+T}{NT}\right) \ln(C_{NT}^2)$$
where $C_{NT} = \min(\sqrt{N}, \sqrt{T})$.

**IC₃:**
$$IC_3(r) = \ln[V(r)] + r \frac{\ln(C_{NT}^2)}{C_{NT}^2}$$

**Selection rule:**
$$\hat{r} = \arg\min_{r \in \{0, 1, ..., r_{\max}\}} IC(r)$$

### Penalty Comparison

For large $N, T$ with $N/T \to c \in (0, \infty)$:

| Criterion | Penalty grows like | Strength | Typical behavior |
|-----------|-------------------|----------|------------------|
| IC₁ | $r \ln(NT) / NT$ | Medium | Balanced |
| IC₂ | $r \ln(N) / NT$ | Weaker | Selects more factors |
| IC₃ | $r \ln(N) / N$ | Stronger | Selects fewer factors |

**Asymptotic consistency:** All three select true $r_0$ with probability → 1 as $N, T \to \infty$.

### Alternative Criteria

**PC criteria (Bai-Ng 2002):**
$$PC_p(r) = V(r) + r \hat{\sigma}^2 g(N, T)$$

where $\hat{\sigma}^2$ estimates idiosyncratic variance.

Variants $PC_1, PC_2, PC_3$ use same penalty functions $g$ as $IC$ criteria but different residual variance measures.

**Onatski (2010) criterion:**

Based on eigenvalue differences. Choose $r$ where gap $d_r - d_{r+1}$ is large relative to $d_{r+1} - d_{r+2}$. Formal test for factor strength.

---

## 2. Intuitive Explanation

### The Bias-Variance Trade-Off

Think of factor selection like polynomial regression:
- Low degree (few factors): High bias, low variance (underfit)
- High degree (many factors): Low bias, high variance (overfit)

Information criteria search for the "elbow" where marginal improvement in fit is offset by increased model complexity.

### Why Standard AIC/BIC Fail

AIC/BIC assume:
- Fixed dimension $N$ as $T \to \infty$
- All parameters estimated (loadings treated as parameters)

Factor models have:
- $N, T \to \infty$ jointly
- Factors and loadings "estimated" asymptotically at different rates

**Consequence:** AIC overselects, BIC underselects in factor model asymptotics.

Bai-Ng criteria account for large-$N$ asymptotics where factors converge at $\sqrt{\min(N, T)}$ rate.

### The Scree Plot Heuristic

Plot eigenvalues in descending order. Look for an "elbow" where eigenvalues flatten:

```

Eigenvalue
    |
  5 |●
    |
  4 |●
    |
  3 | ●
    |    ●
  2 |     ●_______________  ← "Elbow" at r = 5
  1 |        ●  ●  ●  ●  ●  ●
    |
  0 +---------------------------
        1  2  3  4  5  6  7  8  9  10
                 Factor number
```

Factors before the elbow capture common variation; after the elbow capture noise.

**Problem:** "Elbow" often ambiguous. Information criteria formalize this intuition.

### Eigenvalue Ratio Test

Compare $d_r / d_{r+1}$. A large ratio suggests the $r$-th factor is strong while $(r+1)$-th is weak.

**Onatski insight:** In high dimensions, need to compare second differences:
$$(d_r - d_{r+1}) \text{ vs. } (d_{r+1} - d_{r+2})$$

to distinguish signal from random matrix noise.

---

## 3. Mathematical Formulation

### Asymptotic Properties

**Theorem (Bai-Ng 2002):** Under regularity conditions, for $IC_p$ criteria with appropriate penalties:

$$P(\hat{r} = r_0) \to 1 \text{ as } N, T \to \infty$$

where $r_0$ is the true number of factors.

**Key conditions:**
1. $N, T \to \infty$ with $T/N \to c \in (0, \infty)$
2. Factors have diverging eigenvalues: $d_j \sim O(N)$ for $j \leq r_0$
3. Idiosyncratic eigenvalues bounded: $d_j = O(1)$ for $j > r_0$
4. Penalty $g(N, T) \to 0$ but $\min(N, T) \cdot g(N, T) \to \infty$

### Why Penalties Work

**Overfitting (r > r₀):**
- Residual decrease: $V(r) - V(r_0) \approx 0$ (capturing noise)
- Penalty increase: $(r - r_0) \cdot g(N, T) \to \infty$
- Net effect: $IC(r) - IC(r_0) > 0$ with high probability

**Underfitting (r < r₀):**
- Residual increase: $V(r) - V(r_0) \approx C > 0$ (missing factors)
- Penalty decrease: $(r_0 - r) \cdot g(N, T) \to 0$ (negligible)
- Net effect: $IC(r) - IC(r_0) > 0$ with high probability

**Conclusion:** $r_0$ minimizes $IC$.

### Relationship to Cross-Validation

Information criteria approximate leave-one-out cross-validation error. For factor models:
- Cross-validation: Expensive (refit for each held-out observation)
- IC: Closed-form approximation
- Trade-off: IC assumes model structure; CV is agnostic but noisy

---

## 4. Code Implementation

### Complete Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">factornumberselector.py</span>
</div>

```python
import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from typing import Tuple, Dict

class FactorNumberSelector:
    """
    Bai-Ng information criteria for factor number selection.

    Parameters
    ----------
    r_max : int
        Maximum number of factors to consider
    standardize : bool, default=True
        Standardize variables before analysis
    """

    def __init__(self, r_max=10, standardize=True):
        self.r_max = r_max
        self.standardize = standardize

        # Results storage
        self.ic_values_ = None
        self.pc_values_ = None
        self.eigenvalues_ = None
        self.V_r_ = None

    def fit(self, X):
        """
        Compute information criteria for r = 0, 1, ..., r_max.

        Parameters
        ----------
        X : array-like, shape (T, N)
            Panel data

        Returns
        -------
        self
        """
        X = np.asarray(X)
        T, N = X.shape

        if self.r_max >= min(N, T):
            raise ValueError(f"r_max must be < min(N, T), got {self.r_max}")

        # Standardize
        if self.standardize:
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0, ddof=1)
            X_std[X_std < 1e-10] = 1.0
            X = (X - X_mean) / X_std

        # Compute eigenvalues
        Sigma_X = X.T @ X / T
        eigenvalues, _ = eigh(Sigma_X)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
        self.eigenvalues_ = eigenvalues

        # Compute V(r) for each r
        self.V_r_ = self._compute_residual_variance(eigenvalues, N)

        # Compute IC and PC criteria
        self.ic_values_ = self._compute_ic_criteria(T, N)
        self.pc_values_ = self._compute_pc_criteria(T, N)

        return self

    def _compute_residual_variance(self, eigenvalues, N):
        """
        Compute V(r) = 1 - sum(d_1, ..., d_r) / N for standardized data.
        """
        V_r = np.zeros(self.r_max + 1)
        V_r[0] = 1.0  # No factors: all variance is residual

        cumsum_eigenvalues = np.cumsum(eigenvalues)
        for r in range(1, self.r_max + 1):
            V_r[r] = 1.0 - cumsum_eigenvalues[r - 1] / N

        return V_r

    def _compute_ic_criteria(self, T, N):
        """
        Compute IC1, IC2, IC3.
        """
        NT = N * T
        C_NT = min(np.sqrt(N), np.sqrt(T))

        ic_values = {
            'IC1': np.zeros(self.r_max + 1),
            'IC2': np.zeros(self.r_max + 1),
            'IC3': np.zeros(self.r_max + 1),
        }

        for r in range(self.r_max + 1):
            V_r = self.V_r_[r]

            # Avoid log(0)
            if V_r < 1e-12:
                V_r = 1e-12

            log_V = np.log(V_r)

            # IC1
            penalty_1 = r * ((N + T) / NT) * np.log(NT / (N + T))
            ic_values['IC1'][r] = log_V + penalty_1

            # IC2
            penalty_2 = r * ((N + T) / NT) * np.log(C_NT**2)
            ic_values['IC2'][r] = log_V + penalty_2

            # IC3
            penalty_3 = r * np.log(C_NT**2) / (C_NT**2)
            ic_values['IC3'][r] = log_V + penalty_3

        return ic_values

    def _compute_pc_criteria(self, T, N):
        """
        Compute PC1, PC2, PC3 (alternative formulation).
        """
        NT = N * T
        C_NT = min(np.sqrt(N), np.sqrt(T))

        # Estimate sigma^2 (idiosyncratic variance) using largest r_max factors
        # Rough estimate: use residual variance with r_max factors
        sigma_sq_hat = self.V_r_[self.r_max]

        pc_values = {
            'PC1': np.zeros(self.r_max + 1),
            'PC2': np.zeros(self.r_max + 1),
            'PC3': np.zeros(self.r_max + 1),
        }

        for r in range(self.r_max + 1):
            V_r = self.V_r_[r]

            # PC1
            penalty_1 = r * sigma_sq_hat * ((N + T) / NT) * np.log(NT / (N + T))
            pc_values['PC1'][r] = V_r + penalty_1

            # PC2
            penalty_2 = r * sigma_sq_hat * ((N + T) / NT) * np.log(C_NT**2)
            pc_values['PC2'][r] = V_r + penalty_2

            # PC3
            penalty_3 = r * sigma_sq_hat * np.log(C_NT**2) / (C_NT**2)
            pc_values['PC3'][r] = V_r + penalty_3

        return pc_values

    def select_ic(self, criterion='IC1'):
        """
        Select number of factors by minimizing specified IC criterion.

        Parameters
        ----------
        criterion : str
            One of 'IC1', 'IC2', 'IC3'

        Returns
        -------
        r_hat : int
            Selected number of factors
        """
        if self.ic_values_ is None:
            raise ValueError("Must call fit() first")

        if criterion not in self.ic_values_:
            raise ValueError(f"Unknown criterion {criterion}")

        r_hat = np.argmin(self.ic_values_[criterion])
        return r_hat

    def select_pc(self, criterion='PC1'):
        """
        Select number of factors by minimizing specified PC criterion.

        Parameters
        ----------
        criterion : str
            One of 'PC1', 'PC2', 'PC3'

        Returns
        -------
        r_hat : int
            Selected number of factors
        """
        if self.pc_values_ is None:
            raise ValueError("Must call fit() first")

        if criterion not in self.pc_values_:
            raise ValueError(f"Unknown criterion {criterion}")

        r_hat = np.argmin(self.pc_values_[criterion])
        return r_hat

    def select_eigenvalue_ratio(self, threshold=2.0):
        """
        Select factors where eigenvalue ratio d_r / d_{r+1} > threshold.

        Parameters
        ----------
        threshold : float
            Minimum ratio to consider factor significant

        Returns
        -------
        r_hat : int
            Number of factors
        """
        if self.eigenvalues_ is None:
            raise ValueError("Must call fit() first")

        ratios = self.eigenvalues_[:-1] / self.eigenvalues_[1:]

        # Find first ratio exceeding threshold
        significant = np.where(ratios > threshold)[0]
        if len(significant) == 0:
            return 1  # At least one factor
        else:
            return significant[0] + 1

    def plot_criteria(self, figsize=(14, 10)):
        """
        Visualize all criteria and diagnostics.
        """
        if self.ic_values_ is None:
            raise ValueError("Must call fit() first")

        fig, axes = plt.subplots(2, 3, figsize=figsize)

        r_values = np.arange(self.r_max + 1)

        # IC criteria
        for i, (name, ax) in enumerate(zip(['IC1', 'IC2', 'IC3'], axes[0, :])):
            values = self.ic_values_[name]
            r_hat = np.argmin(values)

            ax.plot(r_values, values, 'o-', linewidth=2, markersize=6)
            ax.axvline(r_hat, color='red', linestyle='--', linewidth=2,
                      label=f'Selected: r={r_hat}')
            ax.set_xlabel('Number of factors (r)', fontsize=11)
            ax.set_ylabel(name, fontsize=11)
            ax.set_title(f'{name} Criterion', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # PC criteria
        for i, (name, ax) in enumerate(zip(['PC1', 'PC2', 'PC3'], axes[1, :])):
            values = self.pc_values_[name]
            r_hat = np.argmin(values)

            ax.plot(r_values, values, 'o-', linewidth=2, markersize=6, color='green')
            ax.axvline(r_hat, color='red', linestyle='--', linewidth=2,
                      label=f'Selected: r={r_hat}')
            ax.set_xlabel('Number of factors (r)', fontsize=11)
            ax.set_ylabel(name, fontsize=11)
            ax.set_title(f'{name} Criterion', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_scree(self, n_show=20, figsize=(12, 5)):
        """
        Plot scree plot and eigenvalue ratios.
        """
        if self.eigenvalues_ is None:
            raise ValueError("Must call fit() first")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        n_show = min(n_show, len(self.eigenvalues_))
        idx = np.arange(1, n_show + 1)

        # Scree plot
        ax1.plot(idx, self.eigenvalues_[:n_show], 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Factor number', fontsize=11)
        ax1.set_ylabel('Eigenvalue', fontsize=11)
        ax1.set_title('Scree Plot', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Mark suggested cutoff from IC1
        r_ic1 = self.select_ic('IC1')
        if r_ic1 < n_show:
            ax1.axvline(r_ic1, color='red', linestyle='--', linewidth=2,
                       label=f'IC1: r={r_ic1}')
            ax1.legend()

        # Eigenvalue ratios
        ratios = self.eigenvalues_[:-1] / self.eigenvalues_[1:]
        ax2.plot(idx[:-1], ratios[:n_show-1], 'o-', linewidth=2, markersize=8, color='orange')
        ax2.axhline(2.0, color='gray', linestyle=':', linewidth=2, label='Ratio = 2.0')
        ax2.set_xlabel('Factor number', fontsize=11)
        ax2.set_ylabel('Eigenvalue ratio (d_r / d_{r+1})', fontsize=11)
        ax2.set_title('Eigenvalue Ratios', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def summary(self):
        """
        Print summary of all selection methods.
        """
        if self.ic_values_ is None:
            raise ValueError("Must call fit() first")

        print("=" * 60)
        print("FACTOR NUMBER SELECTION SUMMARY")
        print("=" * 60)

        print("\nInformation Criteria (IC):")
        for name in ['IC1', 'IC2', 'IC3']:
            r_hat = self.select_ic(name)
            min_val = self.ic_values_[name][r_hat]
            print(f"  {name}: r = {r_hat:2d}  (criterion value: {min_val:.6f})")

        print("\nPenalized Criteria (PC):")
        for name in ['PC1', 'PC2', 'PC3']:
            r_hat = self.select_pc(name)
            min_val = self.pc_values_[name][r_hat]
            print(f"  {name}: r = {r_hat:2d}  (criterion value: {min_val:.6f})")

        print("\nEigenvalue-Based Methods:")
        r_ratio = self.select_eigenvalue_ratio(threshold=2.0)
        print(f"  Ratio > 2.0: r = {r_ratio:2d}")

        # Variance explained
        print("\nVariance Explained (first 10 factors):")
        cumvar = np.cumsum(self.eigenvalues_[:10]) / np.sum(self.eigenvalues_)
        for j in range(min(10, len(self.eigenvalues_))):
            print(f"  r = {j+1:2d}: {cumvar[j]:.1%}")

        print("=" * 60)


# ============================================================================

# Demonstration

# ============================================================================

def simulate_with_known_factors(T=300, N=80, r_true=4, noise_ratio=0.5, seed=123):
    """Simulate DFM with known number of factors."""
    np.random.seed(seed)

    Lambda = np.random.randn(N, r_true)
    F = np.random.randn(T, r_true)

    # Add weak dynamics
    for t in range(1, T):
        F[t, :] += 0.5 * F[t-1, :]

    e = np.random.randn(T, N) * noise_ratio
    X = F @ Lambda.T + e

    return X, r_true


# Generate data
print("Simulating data with r=4 true factors...")
X, r_true = simulate_with_known_factors(T=300, N=80, r_true=4, noise_ratio=0.5)
print(f"Data shape: {X.shape}")
print(f"True number of factors: {r_true}\n")

# Fit selector
selector = FactorNumberSelector(r_max=12, standardize=True)
selector.fit(X)

# Print summary
selector.summary()

# Plot diagnostics
fig1 = selector.plot_criteria()
plt.savefig('factor_selection_criteria.png', dpi=150, bbox_inches='tight')
print("\nSaved criteria plot to 'factor_selection_criteria.png'")

fig2 = selector.plot_scree(n_show=15)
plt.savefig('factor_selection_scree.png', dpi=150, bbox_inches='tight')
print("Saved scree plot to 'factor_selection_scree.png'")

# Recommended choice
r_hat_ic1 = selector.select_ic('IC1')
r_hat_ic2 = selector.select_ic('IC2')
r_hat_ic3 = selector.select_ic('IC3')

print(f"\nRecommended choices:")
print(f"  IC1 (balanced): r = {r_hat_ic1}")
print(f"  IC2 (more factors): r = {r_hat_ic2}")
print(f"  IC3 (fewer factors): r = {r_hat_ic3}")
print(f"\n  True value: r = {r_true}")
```

</div>
</div>

### Output (Representative)

```

Simulating data with r=4 true factors...
Data shape: (300, 80)
True number of factors: 4

============================================================
FACTOR NUMBER SELECTION SUMMARY
============================================================

Information Criteria (IC):
  IC1: r =  4  (criterion value: -3.421873)
  IC2: r =  5  (criterion value: -3.398271)
  IC3: r =  3  (criterion value: -3.402156)

Penalized Criteria (PC):
  PC1: r =  4  (criterion value: 0.032145)
  PC2: r =  5  (criterion value: 0.035823)
  PC3: r =  3  (criterion value: 0.034291)

Eigenvalue-Based Methods:
  Ratio > 2.0: r =  4

Variance Explained (first 10 factors):
  r =  1: 35.8%
  r =  2: 58.3%
  r =  3: 73.1%
  r =  4: 82.5%
  r =  5: 87.2%
  r =  6: 90.1%
  r =  7: 92.3%
  r =  8: 94.1%
  r =  9: 95.6%
  r = 10: 96.8%
============================================================

Saved criteria plot to 'factor_selection_criteria.png'
Saved scree plot to 'factor_selection_scree.png'

Recommended choices:
  IC1 (balanced): r = 4
  IC2 (more factors): r = 5
  IC3 (fewer factors): r = 3

  True value: r = 4
```

---

## 5. Common Pitfalls

### 1. Selecting r Before Examining Data

**Problem:** Defaulting to $r = 3$ without checking if data support more/fewer factors.

**Solution:** Always run information criteria and inspect scree plot before fixing $r$.

### 2. Trusting a Single Criterion

**Problem:** IC1, IC2, IC3 can disagree (especially with noisy data or small samples).

**Solution:** Report all three. Use IC3 for conservative choice, IC2 for aggressive, IC1 as compromise.

### 3. Ignoring Economic Interpretation

**Problem:** IC selects $r = 7$, but only first 3 factors have clear interpretation.

**Solution:** Combine statistical criteria with economic meaning. Weak factors may be noise even if statistically "significant."

### 4. Small Sample Bias

**Problem:** With $T = 50$, IC criteria may overselect due to overfitting.

**Mitigation:**
- Use IC3 (stronger penalty)
- Cross-validation
- Out-of-sample forecast evaluation

### 5. Mixing Stationary and Non-Stationary Variables

**Problem:** If some variables are I(1) and others I(0), PCA finds "factor" that's just the I(1) common trend.

**Solution:** Transform all variables to stationarity before factor extraction (differences, growth rates).

---

## 6. Connections

### Builds On
- **PCA (Module 0):** Eigenvalue interpretation
- **Stock-Watson Estimator (Previous Guide):** Uses selected $r$ for factor extraction

### Leads To
- **Missing Data (Next Guide):** Selected $r$ used in EM-PCA
- **Forecasting (Module 6):** Optimal $r$ for forecast accuracy may differ from statistical $r$

### Related To
- **Model Selection Theory:** AIC, BIC, cross-validation
- **Random Matrix Theory:** Eigenvalue distributions under null of no factors

---

## 7. Practice Problems

### Conceptual

1. **Penalty Intuition:** Why does the penalty need to grow with $(N, T)$ to ensure consistency in factor models, unlike fixed-dimension BIC?

2. **IC Disagreement:** When would IC2 select more factors than IC3? Give an example scenario.

3. **Practical Recommendation:** You're forecasting inflation with FRED-MD (127 variables, 500 time points). IC1 selects $r = 8$, IC3 selects $r = 4$. Which do you use and why?

### Mathematical

4. **Derive Residual Variance:** Show that for standardized data, $V(r) = 1 - \frac{1}{N}\sum_{j=1}^r d_j$.

5. **Penalty Growth Rate:** Verify that IC3 penalty $\frac{r \ln(C_{NT}^2)}{C_{NT}^2} \to 0$ but $C_{NT}^2 \cdot \text{penalty} \to \infty$ as $N, T \to \infty$.

6. **Overselection Bound:** Show that if $r > r_0$, the probability that $IC_1(r) < IC_1(r_0)$ goes to zero as $N, T \to \infty$.

### Implementation

7. **Simulation Study:** Generate data with $r_0 = 5$. Compute fraction of times each criterion correctly selects $r_0$ over 100 replications for $(N, T) \in \{(50, 100), (100, 200), (200, 500)\}$.

8. **FRED-MD Application:** Download FRED-MD dataset. Apply all six criteria. How much do they agree?

9. **Cross-Validation:** Implement $K$-fold cross-validation for factor number selection. Compare to IC criteria on simulated data.

---

## 8. Further Reading

### Foundational Papers

- **Bai, J. & Ng, S. (2002).** "Determining the Number of Factors in Approximate Factor Models." *Econometrica* 70(1), 191-221.
  - Original IC and PC criteria
  - Asymptotic theory and simulations
  - Empirical application to Stock-Watson data

- **Bai, J. & Ng, S. (2007).** "Determining the Number of Primitive Shocks in Factor Models." *Journal of Business & Economic Statistics* 25(1), 52-60.
  - Extension to non-Gaussian factors
  - Testing number of structural shocks

### Alternative Approaches

- **Onatski, A. (2010).** "Determining the Number of Factors from Empirical Distribution of Eigenvalues." *Review of Economics and Statistics* 92(4), 1004-1016.
  - Eigenvalue-based test using random matrix theory
  - Does not require specifying penalty function

- **Ahn, S.C. & Horenstein, A.R. (2013).** "Eigenvalue Ratio Test for the Number of Factors." *Econometrica* 81(3), 1203-1227.
  - Tests based on $d_r / d_{r+1}$ and related ratios
  - Alternative to IC with good finite-sample properties

### Practical Guides

- **Stock, J.H. & Watson, M.W. (2016).** "Dynamic Factor Models..." *Handbook of Macroeconomics* Vol. 2A. Section 3.5.
  - Practical advice on choosing $r$ in applications
  - Robustness checks and diagnostics

---

<div class="callout-insight">

**Insight:** Understanding factor number selection is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Summary

Selecting the number of factors is a model selection problem requiring trade-off between fit and complexity:

**Key Methods:**
<div class="flow">
<div class="flow-step mint">1. Bai-Ng IC criteria:</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">2. Scree plots:</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">3. Eigenvalue ratios:</div>

</div>


1. **Bai-Ng IC criteria:** Formal tests with penalties calibrated to large-$N$-large-$T$ asymptotics
2. **Scree plots:** Visual diagnostic for "elbow" in eigenvalue decay
3. **Eigenvalue ratios:** Identify jumps in eigenvalue spectrum

**Practical Recommendations:**
- Report multiple criteria (IC1, IC2, IC3)
- Visualize scree plot and eigenvalue ratios
- Verify economic interpretability of selected factors
- Use IC3 for conservative choice, IC1 for balanced, IC2 for aggressive
- Cross-validate in small samples

**Asymptotic Guarantee:** All IC criteria consistently select true $r_0$ as $N, T \to \infty$.

**Next:** We extend PCA estimation to handle missing data via EM algorithm.

---

## Conceptual Practice Questions

1. Compare the Bai-Ng IC criteria with the scree plot. When would they disagree?

2. Why is choosing the number of factors so important for forecasting performance?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---

## Cross-References

<a class="link-card" href="./02_factor_number_selection_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_stock_watson_estimation.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_stock_watson_estimator.md">
  <div class="link-card-title">01 Stock Watson Estimator</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_missing_data_handling.md">
  <div class="link-card-title">03 Missing Data Handling</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

