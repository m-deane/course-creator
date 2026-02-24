# Advanced HMM Variants

## In Brief

Standard HMMs assume all states transition with equal probability regardless of how long the model has been in a state. Advanced variants relax this assumption: Sticky HMMs stay in the current state longer, Input-Output HMMs condition transitions on external signals, and Hierarchical HMMs nest multiple levels of state structure. Each variant exists to model a real failure mode of the basic HMM.

## Key Insight

> 💡 **Every advanced HMM variant solves a specific failure mode of the standard model.** Before reaching for a more complex variant, identify which assumption the standard HMM violates in your data — then choose the extension that fixes exactly that assumption.

## Sticky HMM

Standard HMMs may switch states too frequently. Sticky HMMs increase self-transition probability:

$$a_{ii}^{sticky} = \kappa + (1-\kappa) \cdot a_{ii}$$

where $\kappa \in [0, 1]$ is the "stickiness" parameter.

### Implementation

```python
import numpy as np
from hmmlearn import hmm

class StickyGaussianHMM:
    """Gaussian HMM with sticky transitions."""

    def __init__(self, n_components: int, kappa: float = 0.5):
        self.n_components = n_components
        self.kappa = kappa
        self.model = None

    def fit(self, X: np.ndarray):
        """Fit model with sticky prior on transitions."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Fit standard HMM first
        self.model = hmm.GaussianHMM(
            n_components=self.n_components,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.model.fit(X)

        # Apply stickiness to transition matrix
        A = self.model.transmat_
        A_sticky = np.zeros_like(A)

        for i in range(self.n_components):
            for j in range(self.n_components):
                if i == j:
                    A_sticky[i, j] = self.kappa + (1 - self.kappa) * A[i, j]
                else:
                    A_sticky[i, j] = (1 - self.kappa) * A[i, j]

            # Renormalize row
            A_sticky[i] /= A_sticky[i].sum()

        self.model.transmat_ = A_sticky

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.model.predict_proba(X)

# Example
returns = np.random.randn(500) * 0.02
sticky_hmm = StickyGaussianHMM(n_components=2, kappa=0.7)
sticky_hmm.fit(returns)

print("Sticky transition matrix:")
print(sticky_hmm.model.transmat_)
```

## Markov-Switching Autoregressive Models

Combine regime switching with autoregressive dynamics:

$$y_t = c_{s_t} + \phi_{s_t} y_{t-1} + \sigma_{s_t} \epsilon_t$$

Each regime $s_t$ has its own intercept, AR coefficient, and volatility.

### Implementation with statsmodels

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

def fit_markov_switching_ar(
    y: np.ndarray,
    k_regimes: int = 2,
    order: int = 1,
    switching_variance: bool = True
):
    """
    Fit Markov-Switching AR model.
    """
    # Markov Autoregression
    model = MarkovAutoregression(
        y,
        k_regimes=k_regimes,
        order=order,
        switching_ar=True,
        switching_variance=switching_variance
    )

    results = model.fit()
    return results

# Example with simulated data
np.random.seed(42)
n = 500

# Simulate MS-AR(1)
states = np.zeros(n, dtype=int)
y = np.zeros(n)

state = 0
y[0] = np.random.randn()

for t in range(1, n):
    # State transition
    if state == 0:
        state = 1 if np.random.random() < 0.05 else 0
    else:
        state = 0 if np.random.random() < 0.1 else 1
    states[t] = state

    # Generate observation
    if state == 0:  # Low volatility regime
        y[t] = 0.02 + 0.8 * y[t-1] + 0.5 * np.random.randn()
    else:  # High volatility regime
        y[t] = -0.05 + 0.3 * y[t-1] + 1.5 * np.random.randn()

# Fit model
results = fit_markov_switching_ar(y, k_regimes=2, order=1)
print(results.summary())

# Get smoothed probabilities
smoothed_probs = results.smoothed_marginal_probabilities
```

### Regime-Dependent Forecasting

```python
def regime_forecast(
    results,
    y: np.ndarray,
    horizon: int = 5
) -> dict:
    """
    Generate regime-dependent forecasts.
    """
    # Current regime probability
    current_probs = results.smoothed_marginal_probabilities[-1]

    # Transition matrix
    P = results.regime_transition

    # Regime-specific parameters
    params = results.params

    forecasts = {'point': [], 'by_regime': {}}

    for regime in range(results.k_regimes):
        forecasts['by_regime'][regime] = []

    for h in range(1, horizon + 1):
        # Probability of each regime at horizon h
        future_probs = np.dot(current_probs, np.linalg.matrix_power(P, h))

        # Regime-specific forecasts (simplified - should be iterated AR)
        for regime in range(results.k_regimes):
            if h == 1:
                fc = params[f'const[{regime}]'] + params[f'ar.L1[{regime}]'] * y[-1]
            else:
                fc = params[f'const[{regime}]'] + params[f'ar.L1[{regime}]'] * forecasts['by_regime'][regime][-1]
            forecasts['by_regime'][regime].append(fc)

        # Probability-weighted point forecast
        point_fc = sum(
            future_probs[r] * forecasts['by_regime'][r][-1]
            for r in range(results.k_regimes)
        )
        forecasts['point'].append(point_fc)

    return forecasts
```

## Hierarchical HMM

Hierarchical HMMs have multiple levels of hidden states:

```
Super-states (macro regimes)
      ↓
Sub-states (micro regimes within each macro)
      ↓
Observations
```

### Conceptual Implementation

```python
class HierarchicalHMM:
    """Two-level hierarchical HMM."""

    def __init__(
        self,
        n_super_states: int,
        n_sub_states_per_super: int
    ):
        self.n_super = n_super_states
        self.n_sub = n_sub_states_per_super
        self.n_total = n_super_states * n_sub_states_per_super

        # Super-level transition matrix
        self.A_super = None

        # Sub-level transition matrices (one per super-state)
        self.A_sub = [None] * n_super_states

        # Emission parameters for each sub-state
        self.emission_means = None
        self.emission_vars = None

    def _super_to_sub_idx(self, super_state: int, sub_state: int) -> int:
        """Convert (super, sub) to flat index."""
        return super_state * self.n_sub + sub_state

    def _sub_to_super_idx(self, flat_idx: int) -> tuple:
        """Convert flat index to (super, sub)."""
        return flat_idx // self.n_sub, flat_idx % self.n_sub

    def build_full_transition_matrix(self) -> np.ndarray:
        """
        Build full transition matrix from hierarchical structure.
        """
        A_full = np.zeros((self.n_total, self.n_total))

        for super_i in range(self.n_super):
            for sub_i in range(self.n_sub):
                from_idx = self._super_to_sub_idx(super_i, sub_i)

                for super_j in range(self.n_super):
                    for sub_j in range(self.n_sub):
                        to_idx = self._super_to_sub_idx(super_j, sub_j)

                        if super_i == super_j:
                            # Stay in same super-state: use sub-level transitions
                            A_full[from_idx, to_idx] = (
                                (1 - self.A_super[super_i].sum() + self.A_super[super_i, super_i]) *
                                self.A_sub[super_i][sub_i, sub_j]
                            )
                        else:
                            # Transition to different super-state: reset sub-state
                            A_full[from_idx, to_idx] = (
                                self.A_super[super_i, super_j] / self.n_sub
                            )

        # Normalize rows
        A_full = A_full / A_full.sum(axis=1, keepdims=True)

        return A_full
```

## Input-Output HMM

Include exogenous variables that affect transitions or emissions:

```python
class InputOutputHMM:
    """HMM with exogenous inputs affecting emissions."""

    def __init__(self, n_states: int, n_inputs: int):
        self.n_states = n_states
        self.n_inputs = n_inputs

    def emission_probability(
        self,
        observation: float,
        state: int,
        inputs: np.ndarray
    ) -> float:
        """
        Emission depends on state AND exogenous inputs.

        P(o_t | s_t, x_t) = N(mu_s + W_s @ x_t, sigma_s^2)
        """
        mean = self.means[state] + np.dot(self.input_weights[state], inputs)
        std = self.stds[state]

        from scipy import stats
        return stats.norm.pdf(observation, mean, std)
```

## Duration-Dependent HMM

Standard HMMs have geometric duration distribution. Explicit duration models allow arbitrary distributions:

```python
class ExplicitDurationHMM:
    """HMM with explicit state duration modeling."""

    def __init__(
        self,
        n_states: int,
        max_duration: int = 50
    ):
        self.n_states = n_states
        self.max_duration = max_duration

        # Duration distributions for each state
        self.duration_probs = None  # (n_states, max_duration)

    def set_poisson_durations(self, lambdas: list):
        """Set Poisson duration distributions."""
        from scipy import stats

        self.duration_probs = np.zeros((self.n_states, self.max_duration))

        for state, lam in enumerate(lambdas):
            for d in range(self.max_duration):
                self.duration_probs[state, d] = stats.poisson.pmf(d + 1, lam)

            # Normalize
            self.duration_probs[state] /= self.duration_probs[state].sum()

    def expected_duration(self, state: int) -> float:
        """Expected duration in state."""
        durations = np.arange(1, self.max_duration + 1)
        return np.dot(durations, self.duration_probs[state])
```

## Comparison of HMM Variants

| Variant | Use Case | Complexity |
|---------|----------|------------|
| Standard HMM | Basic regime detection | Low |
| Sticky HMM | Persistent regimes | Low |
| MS-AR | Autocorrelated data | Medium |
| Hierarchical HMM | Multi-scale regimes | High |
| Duration HMM | Non-geometric durations | High |
| Input-Output HMM | Exogenous variables | Medium |

## Key Takeaways

1. **Sticky HMMs** prevent excessive regime switching

2. **MS-AR models** combine regime switching with autoregressive dynamics

3. **Hierarchical HMMs** capture multi-scale structure

4. **Duration models** allow non-geometric state persistence

5. **Choose variant** based on data characteristics and modeling goals
