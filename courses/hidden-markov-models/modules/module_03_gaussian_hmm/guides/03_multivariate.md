# Multivariate Gaussian HMMs

> **Reading time:** ~7 min | **Module:** Module 3: Gaussian Hmm | **Prerequisites:** Modules 0-2

<div class="callout-key">

**Key Concept Summary:** Financial markets involve multiple correlated variables. Multivariate Gaussian HMMs model joint distributions:

</div>

## Introduction

Financial markets involve multiple correlated variables. Multivariate Gaussian HMMs model joint distributions:

$$\mathbf{o}_t | s_t = k \sim \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

Each state has its own mean vector and covariance matrix.

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Multivariate Emission Distribution

### Parameters

For $d$-dimensional observations:
- $\boldsymbol{\mu}_k \in \mathbb{R}^d$: Mean vector for state $k$
- $\boldsymbol{\Sigma}_k \in \mathbb{R}^{d \times d}$: Covariance matrix for state $k$

### Probability Density

$$b_k(\mathbf{o}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}_k|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{o} - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1}(\mathbf{o} - \boldsymbol{\mu}_k)\right)$$


<span class="filename">multivariategaussianhmm.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from hmmlearn import hmm

class MultivariateGaussianHMM:
    """
    Multivariate Gaussian HMM implementation.
    """

    def __init__(self, n_states, n_features):
        self.n_states = n_states
        self.n_features = n_features

        self.pi = None
        self.A = None
        self.means = None
        self.covars = None

    def set_parameters(self, pi, A, means, covars):
        """
        Set model parameters.
        """
        self.pi = np.array(pi)
        self.A = np.array(A)
        self.means = np.array(means)  # Shape: (n_states, n_features)
        self.covars = np.array(covars)  # Shape: (n_states, n_features, n_features)

        # Validate
        assert self.means.shape == (self.n_states, self.n_features)
        assert self.covars.shape == (self.n_states, self.n_features, self.n_features)

        return self

    def emission_logprob(self, observation, state):
        """
        Log probability of observation given state.
        """
        return stats.multivariate_normal.logpdf(
            observation,
            mean=self.means[state],
            cov=self.covars[state]
        )

    def sample(self, n_samples):
        """
        Generate samples from the model.
        """
        states = []
        observations = []

        # Initial state
        state = np.random.choice(self.n_states, p=self.pi)

        for _ in range(n_samples):
            states.append(state)

            # Emit observation
            obs = np.random.multivariate_normal(
                self.means[state],
                self.covars[state]
            )
            observations.append(obs)

            # Transition
            state = np.random.choice(self.n_states, p=self.A[state])

        return np.array(observations), np.array(states)

# Example: 2-feature model (returns + volatility)
np.random.seed(42)

model = MultivariateGaussianHMM(n_states=2, n_features=2)
model.set_parameters(
    pi=[0.5, 0.5],
    A=[[0.95, 0.05],
       [0.10, 0.90]],
    means=[
        [0.001, 0.01],   # State 0 (Bull): positive return, low vol
        [-0.002, 0.03]   # State 1 (Bear): negative return, high vol
    ],
    covars=[
        [[0.0001, 0.00005],
         [0.00005, 0.0001]],  # State 0
        [[0.0004, 0.0001],
         [0.0001, 0.0004]]    # State 1
    ]
)

# Generate samples
obs, states = model.sample(500)

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Feature 1 over time
ax1 = axes[0, 0]
ax1.plot(obs[:, 0], alpha=0.7)
for t in range(len(states) - 1):
    ax1.axvspan(t, t+1, alpha=0.1, color='green' if states[t] == 0 else 'red')
ax1.set_ylabel('Feature 1 (Return)')
ax1.set_title('Returns Over Time')

# Feature 2 over time
ax2 = axes[0, 1]
ax2.plot(obs[:, 1], alpha=0.7)
for t in range(len(states) - 1):
    ax2.axvspan(t, t+1, alpha=0.1, color='green' if states[t] == 0 else 'red')
ax2.set_ylabel('Feature 2 (Volatility Proxy)')
ax2.set_title('Volatility Proxy Over Time')

# Joint distribution
ax3 = axes[1, 0]
colors = ['green' if s == 0 else 'red' for s in states]
ax3.scatter(obs[:, 0], obs[:, 1], c=colors, alpha=0.5, s=20)
ax3.set_xlabel('Return')
ax3.set_ylabel('Volatility Proxy')
ax3.set_title('Joint Distribution by State')

# State sequence
ax4 = axes[1, 1]
ax4.fill_between(range(len(states)), states, alpha=0.5, step='mid')
ax4.set_xlabel('Time')
ax4.set_ylabel('State')
ax4.set_yticks([0, 1])
ax4.set_yticklabels(['Bull', 'Bear'])
ax4.set_title('Hidden State Sequence')

plt.tight_layout()
plt.show()
```

</div>
</div>

## Covariance Structures

Different covariance structures trade off flexibility vs. parameters:


<span class="filename">compare_covariance_types.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def compare_covariance_types():
    """
    Compare different covariance structures.
    """
    np.random.seed(42)

    # Generate data
    n_samples = 500
    n_features = 3

    # True model has full covariance
    true_means = [
        [0, 0, 0],
        [1, 1, 1]
    ]
    true_covars = [
        [[1, 0.5, 0.3],
         [0.5, 1, 0.4],
         [0.3, 0.4, 1]],
        [[2, 0.2, 0.1],
         [0.2, 2, 0.3],
         [0.1, 0.3, 2]]
    ]

    # Generate samples
    states = np.random.choice(2, size=n_samples, p=[0.6, 0.4])
    obs = np.array([
        np.random.multivariate_normal(true_means[s], true_covars[s])
        for s in states
    ])

    # Fit with different covariance types
    cov_types = ['spherical', 'diag', 'full', 'tied']
    results = {}

    for cov_type in cov_types:
        model = hmm.GaussianHMM(
            n_components=2,
            covariance_type=cov_type,
            n_iter=100,
            random_state=42
        )
        model.fit(obs)
        score = model.score(obs)
        n_params = count_parameters(2, n_features, cov_type)
        bic = -2 * score + n_params * np.log(n_samples)

        results[cov_type] = {
            'score': score,
            'n_params': n_params,
            'bic': bic
        }

    # Print comparison
    print("Covariance Type Comparison:")
    print("=" * 70)
    print(f"{'Type':<12} {'Log-Lik':<15} {'# Params':<12} {'BIC':<15}")
    print("-" * 70)
    for cov_type, res in results.items():
        print(f"{cov_type:<12} {res['score']:<15.2f} {res['n_params']:<12} {res['bic']:<15.2f}")

    return results

def count_parameters(n_states, n_features, cov_type):
    """
    Count number of free parameters.
    """
    # Initial distribution: n_states - 1
    # Transition matrix: n_states * (n_states - 1)
    # Means: n_states * n_features
    base = (n_states - 1) + n_states * (n_states - 1) + n_states * n_features

    # Covariance parameters
    if cov_type == 'spherical':
        cov_params = n_states  # One variance per state
    elif cov_type == 'diag':
        cov_params = n_states * n_features  # Diagonal per state
    elif cov_type == 'full':
        cov_params = n_states * n_features * (n_features + 1) // 2  # Full per state
    elif cov_type == 'tied':
        cov_params = n_features * (n_features + 1) // 2  # One shared

    return base + cov_params

compare_covariance_types()
```

</div>
</div>

## Using hmmlearn for Multivariate Data


<span class="filename">fit_multivariate_hmm_hmmlearn.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def fit_multivariate_hmm_hmmlearn():
    """
    Fit multivariate Gaussian HMM using hmmlearn.
    """
    np.random.seed(42)

    # Generate realistic financial data
    # Features: returns, volume_change, spread
    n_samples = 1000

    # Bull regime
    bull_mean = [0.001, 0.02, -0.001]
    bull_cov = [[0.0002, 0.00005, -0.00001],
                [0.00005, 0.01, 0.0001],
                [-0.00001, 0.0001, 0.00005]]

    # Bear regime
    bear_mean = [-0.002, 0.05, 0.003]
    bear_cov = [[0.0008, 0.0001, 0.00005],
                [0.0001, 0.02, 0.001],
                [0.00005, 0.001, 0.0002]]

    # Generate with regime switching
    states = []
    state = 0
    trans = [[0.95, 0.05], [0.10, 0.90]]

    for _ in range(n_samples):
        states.append(state)
        state = np.random.choice(2, p=trans[state])

    observations = np.array([
        np.random.multivariate_normal(
            bull_mean if s == 0 else bear_mean,
            bull_cov if s == 0 else bear_cov
        )
        for s in states
    ])

    # Fit HMM
    model = hmm.GaussianHMM(
        n_components=2,
        covariance_type='full',
        n_iter=200,
        random_state=42
    )
    model.fit(observations)

    # Predict states
    predicted_states = model.predict(observations)

    # Align state labels (they may be swapped)
    if model.means_[0, 0] < model.means_[1, 0]:
        predicted_states = 1 - predicted_states

    print("Fitted Multivariate HMM:")
    print("=" * 70)

    print("\nLearned means:")
    print(f"  State 0 (Bull): {model.means_[0 if model.means_[0,0] > 0 else 1]}")
    print(f"  State 1 (Bear): {model.means_[1 if model.means_[0,0] > 0 else 0]}")

    print("\nLearned transition matrix:")
    print(model.transmat_.round(3))

    print("\nLearned covariances (diagonal elements):")
    for i in range(2):
        print(f"  State {i}: {np.diag(model.covars_[i]).round(6)}")

    # Accuracy
    accuracy = np.mean(predicted_states == np.array(states))
    print(f"\nState prediction accuracy: {accuracy:.1%}")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    feature_names = ['Return', 'Volume Change', 'Spread']

    # Plot each feature
    for i, (ax, name) in enumerate(zip(axes.flat[:3], feature_names)):
        ax.plot(observations[:, i], alpha=0.7)
        for t in range(n_samples - 1):
            ax.axvspan(t, t+1, alpha=0.1,
                       color='green' if states[t] == 0 else 'red')
        ax.set_ylabel(name)
        ax.set_title(f'{name} with True States')

    # Correlation by state
    ax4 = axes[1, 1]
    state_0_mask = np.array(states) == 0
    ax4.scatter(observations[state_0_mask, 0], observations[state_0_mask, 1],
                alpha=0.3, label='Bull', c='green', s=20)
    ax4.scatter(observations[~state_0_mask, 0], observations[~state_0_mask, 1],
                alpha=0.3, label='Bear', c='red', s=20)
    ax4.set_xlabel('Return')
    ax4.set_ylabel('Volume Change')
    ax4.set_title('Return vs Volume by Regime')
    ax4.legend()

    plt.tight_layout()
    plt.show()

    return model, observations, states

model, obs, true_states = fit_multivariate_hmm_hmmlearn()
```

</div>
</div>

## Feature Engineering for HMMs


<span class="filename">prepare_features_for_hmm.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def prepare_features_for_hmm(prices, window=20):
    """
    Prepare multivariate features for HMM.
    """
    import pandas as pd

    df = pd.DataFrame({'price': prices})

    # Returns
    df['return'] = df['price'].pct_change()

    # Realized volatility
    df['volatility'] = df['return'].rolling(window).std() * np.sqrt(252)

    # Momentum
    df['momentum'] = df['price'] / df['price'].shift(window) - 1

    # Volume-weighted average (if volume available)
    # df['vwap'] = ...

    # Range
    df['high'] = df['price'].rolling(window).max()
    df['low'] = df['price'].rolling(window).min()
    df['range'] = (df['high'] - df['low']) / df['price']

    # Drop NaN
    df = df.dropna()

    # Select features
    features = ['return', 'volatility', 'range']

    # Standardize
    for f in features:
        df[f'{f}_std'] = (df[f] - df[f].mean()) / df[f].std()

    return df[[f'{f}_std' for f in features]].values, df.index

# Example with synthetic data
np.random.seed(42)
prices = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.015, 500)))

features, dates = prepare_features_for_hmm(prices)

print("Feature Matrix Shape:", features.shape)
print("\nFeature Statistics:")
print(f"  Means: {features.mean(axis=0).round(4)}")
print(f"  Stds:  {features.std(axis=0).round(4)}")
```

</div>
</div>

## Model Selection for Multivariate HMMs


<span class="filename">select_multivariate_hmm.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def select_multivariate_hmm(observations, max_states=5):
    """
    Select number of states and covariance type.
    """
    results = []

    for n_states in range(2, max_states + 1):
        for cov_type in ['diag', 'full']:
            try:
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type=cov_type,
                    n_iter=100,
                    random_state=42
                )
                model.fit(observations)

                score = model.score(observations)
                n_params = count_parameters(n_states, observations.shape[1], cov_type)
                n_samples = len(observations)

                aic = -2 * score + 2 * n_params
                bic = -2 * score + n_params * np.log(n_samples)

                results.append({
                    'n_states': n_states,
                    'cov_type': cov_type,
                    'log_lik': score,
                    'n_params': n_params,
                    'aic': aic,
                    'bic': bic
                })
            except:
                pass

    results_df = pd.DataFrame(results)

    print("Model Selection Results:")
    print("=" * 80)
    print(results_df.to_string(index=False))

    best_bic = results_df.loc[results_df['bic'].idxmin()]
    print(f"\nBest by BIC: {best_bic['n_states']:.0f} states, {best_bic['cov_type']} covariance")

    return results_df

import pandas as pd
selection_results = select_multivariate_hmm(obs, max_states=4)
```

</div>
</div>

<div class="callout-insight">

**Insight:** Understanding multivariate gaussian hmms is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Key Takeaways

1. **Multivariate HMMs** model joint distributions of multiple features

2. **Each state has** mean vector $\boldsymbol{\mu}_k$ and covariance $\boldsymbol{\Sigma}_k$

3. **Covariance structure** (spherical, diagonal, full, tied) affects flexibility and parameters

4. **Feature engineering** is critical - standardize and include relevant variables

5. **Model selection** uses BIC to balance fit and complexity

6. **hmmlearn** provides efficient implementation for Gaussian HMMs

---

## Conceptual Practice Questions

1. How does a Gaussian HMM differ from a discrete-emission HMM?

2. What financial phenomena can a two-state Gaussian HMM capture?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./03_multivariate_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_gaussian_hmm.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_gaussian_emissions.md">
  <div class="link-card-title">01 Gaussian Emissions</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_gaussian_hmm.md">
  <div class="link-card-title">01 Gaussian Hmm</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

