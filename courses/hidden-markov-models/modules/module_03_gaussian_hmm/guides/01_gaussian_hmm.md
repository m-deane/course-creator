# Gaussian HMMs for Continuous Observations

> **Reading time:** ~5 min | **Module:** Module 3: Gaussian Hmm | **Prerequisites:** Modules 0-2

<div class="callout-key">

**Key Concept Summary:** Financial data (returns, prices, volatility) is continuous. Gaussian HMMs model this directly:

</div>

## From Discrete to Continuous

Financial data (returns, prices, volatility) is continuous. Gaussian HMMs model this directly:

$$o_t | q_t = k \sim \mathcal{N}(\mu_k, \Sigma_k)$$

Each hidden state $k$ has its own mean $\mu_k$ and covariance $\Sigma_k$.

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Gaussian Emission Model

### Univariate Case

$$b_k(o) = \frac{1}{\sqrt{2\pi\sigma_k^2}} \exp\left(-\frac{(o - \mu_k)^2}{2\sigma_k^2}\right)$$

### Multivariate Case

$$b_k(o) = \frac{1}{(2\pi)^{d/2}|\Sigma_k|^{1/2}} \exp\left(-\frac{1}{2}(o - \mu_k)^T\Sigma_k^{-1}(o - \mu_k)\right)$$

## Implementation with hmmlearn

```python
import numpy as np
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt

# Generate synthetic market data
np.random.seed(42)

# True regimes: Bull (0), Bear (1)
n_samples = 1000
true_states = []
current_state = 0

# Simulate regime switching
for _ in range(n_samples):
    true_states.append(current_state)
    if current_state == 0:  # Bull
        if np.random.random() < 0.02:  # 2% chance to switch to Bear
            current_state = 1
    else:  # Bear
        if np.random.random() < 0.05:  # 5% chance to switch to Bull
            current_state = 0

true_states = np.array(true_states)

# Generate returns based on state
returns = np.zeros(n_samples)
returns[true_states == 0] = np.random.normal(0.05, 0.8, (true_states == 0).sum())  # Bull
returns[true_states == 1] = np.random.normal(-0.1, 1.5, (true_states == 1).sum())  # Bear

returns = returns.reshape(-1, 1)

# Fit Gaussian HMM
model = hmm.GaussianHMM(
    n_components=2,
    covariance_type="full",
    n_iter=100,
    random_state=42
)
model.fit(returns)

# Predict states
predicted_states = model.predict(returns)

# Get state probabilities
state_probs = model.predict_proba(returns)

print("Learned parameters:")
print(f"Initial distribution: {model.startprob_}")
print(f"Transition matrix:\n{model.transmat_}")
print(f"Means: {model.means_.flatten()}")
print(f"Variances: {np.sqrt(model.covars_).flatten()}")
```

## Regime Detection Pipeline


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">regimedetector.py</span>

```python
class RegimeDetector:
    """Detect market regimes using Gaussian HMM."""

    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.model = None
        self.regime_labels = None

    def fit(self, returns: np.ndarray, n_iter: int = 100):
        """Fit HMM to return data."""
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=n_iter,
            random_state=42
        )

        self.model.fit(returns)
        self._label_regimes()

        return self

    def _label_regimes(self):
        """Label regimes by mean (higher mean = bull)."""
        means = self.model.means_.flatten()
        self.regime_labels = {
            np.argmax(means): 'bull',
            np.argmin(means): 'bear'
        }

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """Predict regime for each observation."""
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        return self.model.predict(returns)

    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """Get regime probabilities."""
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)
        return self.model.predict_proba(returns)

    def get_regime_stats(self) -> dict:
        """Get statistics for each regime."""
        stats = {}
        for state_idx, label in self.regime_labels.items():
            stats[label] = {
                'mean': self.model.means_[state_idx, 0],
                'std': np.sqrt(self.model.covars_[state_idx, 0, 0]),
                'persistence': self.model.transmat_[state_idx, state_idx]
            }
        return stats

# Usage
detector = RegimeDetector(n_regimes=2)
detector.fit(returns)

print("Regime Statistics:")
for label, stats in detector.get_regime_stats().items():
    print(f"  {label.upper()}:")
    print(f"    Mean: {stats['mean']:.4f}")
    print(f"    Std: {stats['std']:.4f}")
    print(f"    Persistence: {stats['persistence']:.2%}")
```

</div>
</div>

## Model Selection

### Number of States

Use information criteria to select optimal number of states:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">select_n_states.py</span>

```python
def select_n_states(returns: np.ndarray, max_states: int = 5) -> dict:
    """Select optimal number of states using BIC/AIC."""
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)

    results = []

    for n in range(2, max_states + 1):
        model = hmm.GaussianHMM(
            n_components=n,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )

        model.fit(returns)
        log_likelihood = model.score(returns)

        # Number of parameters
        n_params = (
            n - 1 +  # Initial distribution
            n * (n - 1) +  # Transition matrix
            n * returns.shape[1] +  # Means
            n * returns.shape[1] * (returns.shape[1] + 1) // 2  # Covariances
        )

        n_samples = len(returns)
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(n_samples)

        results.append({
            'n_states': n,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'n_params': n_params
        })

    return pd.DataFrame(results)

# Select best model
selection_results = select_n_states(returns, max_states=5)
print(selection_results)

best_n = selection_results.loc[selection_results['bic'].idxmin(), 'n_states']
print(f"\nBest number of states (BIC): {best_n}")
```

</div>
</div>

### Covariance Type


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">compare_covariance_types.py</span>

```python
def compare_covariance_types(returns: np.ndarray, n_states: int = 2) -> dict:
    """Compare different covariance structures."""
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)

    cov_types = ['spherical', 'diag', 'full', 'tied']
    results = []

    for cov_type in cov_types:
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type=cov_type,
            n_iter=100,
            random_state=42
        )

        model.fit(returns)
        log_likelihood = model.score(returns)

        results.append({
            'cov_type': cov_type,
            'log_likelihood': log_likelihood
        })

    return pd.DataFrame(results)
```

</div>
</div>

## Visualization


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">plot_regime_detection.py</span>

```python
def plot_regime_detection(returns, predicted_states, state_probs, dates=None):
    """Visualize regime detection results."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    x = dates if dates is not None else np.arange(len(returns))

    # Returns colored by regime
    ax1 = axes[0]
    colors = ['green' if s == 0 else 'red' for s in predicted_states]
    ax1.bar(x, returns.flatten(), color=colors, alpha=0.7, width=1)
    ax1.set_ylabel('Returns')
    ax1.set_title('Returns by Regime (Green=Bull, Red=Bear)')
    ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)

    # Cumulative returns
    ax2 = axes[1]
    cumulative = np.cumsum(returns.flatten())
    ax2.plot(x, cumulative)
    ax2.fill_between(x, cumulative, 0,
                     where=(predicted_states == 0),
                     color='green', alpha=0.3, label='Bull')
    ax2.fill_between(x, cumulative, 0,
                     where=(predicted_states == 1),
                     color='red', alpha=0.3, label='Bear')
    ax2.set_ylabel('Cumulative Returns')
    ax2.legend()

    # State probabilities
    ax3 = axes[2]
    ax3.fill_between(x, 0, state_probs[:, 0], alpha=0.5, label='P(Bull)')
    ax3.fill_between(x, state_probs[:, 0], 1, alpha=0.5, label='P(Bear)')
    ax3.set_ylabel('State Probability')
    ax3.set_xlabel('Time')
    ax3.legend()

    plt.tight_layout()
    plt.show()

plot_regime_detection(returns, predicted_states, state_probs)
```

</div>
</div>

## Real-World Application


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">regime_analysis.py</span>

```python
import yfinance as yf

def regime_analysis(ticker: str, start: str, end: str, n_states: int = 2):
    """Complete regime analysis for a stock."""

    # Download data
    data = yf.download(ticker, start=start, end=end)
    returns = data['Adj Close'].pct_change().dropna().values.reshape(-1, 1)
    dates = data.index[1:]

    # Fit HMM
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=100,
        random_state=42
    )
    model.fit(returns)

    # Predictions
    states = model.predict(returns)
    probs = model.predict_proba(returns)

    # Statistics by regime
    results = []
    for i in range(n_states):
        mask = states == i
        regime_returns = returns[mask]
        results.append({
            'regime': i,
            'mean_return': regime_returns.mean() * 252,  # Annualized
            'volatility': regime_returns.std() * np.sqrt(252),
            'sharpe': (regime_returns.mean() / regime_returns.std()) * np.sqrt(252),
            'days_in_regime': mask.sum(),
            'pct_time': mask.mean()
        })

    return {
        'model': model,
        'states': states,
        'probs': probs,
        'dates': dates,
        'returns': returns,
        'regime_stats': pd.DataFrame(results)
    }

# Example
result = regime_analysis('SPY', '2020-01-01', '2024-01-01')
print(result['regime_stats'])
```

</div>
</div>

<div class="callout-insight">

**Insight:** Understanding gaussian hmms for continuous observations is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Key Takeaways

1. **Gaussian HMMs** handle continuous observations naturally

2. **Each state has its own** mean and variance (covariance for multivariate)

3. **hmmlearn** provides efficient implementation

4. **Model selection** uses BIC/AIC to choose number of states

5. **Regime labeling** by mean distinguishes bull from bear markets

---

## Conceptual Practice Questions

1. How does a Gaussian HMM differ from a discrete-emission HMM?

2. What financial phenomena can a two-state Gaussian HMM capture?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="../notebooks/01_gaussian_hmm.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_gaussian_emissions.md">
  <div class="link-card-title">01 Gaussian Emissions</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_em_gaussian.md">
  <div class="link-card-title">02 Em Gaussian</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

