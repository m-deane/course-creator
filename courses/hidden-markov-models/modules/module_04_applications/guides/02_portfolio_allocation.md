# Regime-Based Portfolio Allocation

> **Reading time:** ~7 min | **Module:** Module 4: Applications | **Prerequisites:** Modules 0-3

<div class="callout-key">

**Key Concept Summary:** Market regimes have distinct risk-return characteristics. Optimal portfolio allocation should adapt to the current regime.

</div>

## Introduction

Market regimes have distinct risk-return characteristics. Optimal portfolio allocation should adapt to the current regime.

This guide covers:
- Regime-conditional mean-variance optimization
- Dynamic allocation strategies
- Risk management with regime awareness

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Regime-Conditional Parameters

### Estimating Regime-Specific Statistics


<span class="filename">regimeawareportfolio.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from hmmlearn import hmm

class RegimeAwarePortfolio:
    """
    Portfolio optimization with regime awareness.
    """

    def __init__(self, returns_data, n_regimes=2):
        """
        Parameters:
        -----------
        returns_data : DataFrame
            Asset returns (columns = assets)
        n_regimes : int
            Number of market regimes
        """
        self.returns = returns_data
        self.n_regimes = n_regimes
        self.n_assets = returns_data.shape[1]

        self.regime_model = None
        self.regime_probs = None
        self.regime_params = {}

    def fit_regime_model(self, reference_returns=None):
        """
        Fit HMM to detect regimes.
        """
        # Use first asset or reference for regime detection
        if reference_returns is None:
            reference_returns = self.returns.iloc[:, 0].values

        reference_returns = reference_returns.reshape(-1, 1)

        self.regime_model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type='full',
            n_iter=200,
            random_state=42
        )
        self.regime_model.fit(reference_returns)

        # Get regime probabilities
        self.regime_probs = self.regime_model.predict_proba(reference_returns)

        # Identify regimes by volatility (0 = low vol, 1 = high vol)
        regime_vols = np.sqrt(self.regime_model.covars_.flatten())
        self.regime_order = np.argsort(regime_vols)

        return self

    def estimate_regime_parameters(self):
        """
        Estimate mean and covariance for each regime.
        """
        regimes = self.regime_model.predict(
            self.returns.iloc[:, 0].values.reshape(-1, 1)
        )

        for r in range(self.n_regimes):
            mask = regimes == r
            regime_returns = self.returns.iloc[mask]

            self.regime_params[r] = {
                'mean': regime_returns.mean().values * 252,  # Annualize
                'cov': regime_returns.cov().values * 252,
                'n_obs': mask.sum(),
                'volatility': regime_returns.std().values * np.sqrt(252)
            }

        return self

    def mean_variance_optimize(self, expected_return, cov_matrix,
                                risk_free=0.02, target_vol=None):
        """
        Mean-variance optimization.
        """
        n = len(expected_return)

        def portfolio_volatility(weights):
            return np.sqrt(weights @ cov_matrix @ weights)

        def portfolio_return(weights):
            return weights @ expected_return

        def neg_sharpe(weights):
            ret = portfolio_return(weights)
            vol = portfolio_volatility(weights)
            return -(ret - risk_free) / vol

        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        if target_vol is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: portfolio_volatility(w) - target_vol
            })

        # Bounds: long-only
        bounds = [(0, 1) for _ in range(n)]

        # Initial guess
        w0 = np.ones(n) / n

        result = minimize(
            neg_sharpe,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return result.x

    def get_regime_allocations(self, risk_free=0.02):
        """
        Compute optimal allocation for each regime.
        """
        allocations = {}

        for r in range(self.n_regimes):
            params = self.regime_params[r]

            weights = self.mean_variance_optimize(
                params['mean'],
                params['cov'],
                risk_free
            )

            # Portfolio stats
            port_return = weights @ params['mean']
            port_vol = np.sqrt(weights @ params['cov'] @ weights)

            allocations[r] = {
                'weights': weights,
                'expected_return': port_return,
                'volatility': port_vol,
                'sharpe': (port_return - risk_free) / port_vol
            }

        return allocations

# Example
np.random.seed(42)

# Simulate multi-asset returns
n_samples = 1000
n_assets = 4
asset_names = ['Stocks', 'Bonds', 'Gold', 'REITs']

# Bull regime
bull_means = [0.10, 0.04, 0.02, 0.08]
bull_cov = np.array([
    [0.04, 0.01, -0.005, 0.02],
    [0.01, 0.01, 0.002, 0.005],
    [-0.005, 0.002, 0.02, 0.005],
    [0.02, 0.005, 0.005, 0.03]
])

# Bear regime
bear_means = [-0.15, 0.06, 0.12, -0.08]
bear_cov = np.array([
    [0.09, 0.02, -0.01, 0.05],
    [0.02, 0.015, 0.003, 0.01],
    [-0.01, 0.003, 0.03, 0.002],
    [0.05, 0.01, 0.002, 0.07]
])

# Generate regime-switching returns
states = []
state = 0
for _ in range(n_samples):
    states.append(state)
    state = np.random.choice(2, p=[0.95, 0.05] if state == 0 else [0.10, 0.90])

daily_returns = []
for s in states:
    if s == 0:
        ret = np.random.multivariate_normal(
            np.array(bull_means) / 252,
            bull_cov / 252
        )
    else:
        ret = np.random.multivariate_normal(
            np.array(bear_means) / 252,
            bear_cov / 252
        )
    daily_returns.append(ret)

returns_df = pd.DataFrame(daily_returns, columns=asset_names)

# Fit portfolio model
portfolio = RegimeAwarePortfolio(returns_df, n_regimes=2)
portfolio.fit_regime_model()
portfolio.estimate_regime_parameters()

# Get allocations
allocations = portfolio.get_regime_allocations()

print("Regime-Based Portfolio Allocations:")
print("=" * 70)

for r in range(2):
    regime_label = 'Bull' if portfolio.regime_params[r]['mean'][0] > 0 else 'Bear'
    alloc = allocations[r]

    print(f"\n{regime_label} Regime:")
    print(f"  Expected Return: {alloc['expected_return']:.1%}")
    print(f"  Volatility: {alloc['volatility']:.1%}")
    print(f"  Sharpe Ratio: {alloc['sharpe']:.2f}")
    print(f"  Optimal Weights:")
    for asset, weight in zip(asset_names, alloc['weights']):
        print(f"    {asset}: {weight:.1%}")
```

</div>

## Dynamic Allocation Strategy


<span class="filename">dynamicregimeallocator.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
class DynamicRegimeAllocator:
    """
    Dynamically adjust portfolio based on regime probabilities.
    """

    def __init__(self, regime_portfolio):
        self.rp = regime_portfolio
        self.allocations = self.rp.get_regime_allocations()

    def get_current_allocation(self, lookback_returns):
        """
        Get allocation based on current regime probability.
        """
        # Get regime probability for recent data
        probs = self.rp.regime_model.predict_proba(
            lookback_returns.reshape(-1, 1)
        )
        current_prob = probs[-1]  # Latest probabilities

        # Blend allocations by probability
        blended_weights = np.zeros(self.rp.n_assets)
        for r in range(self.rp.n_regimes):
            blended_weights += current_prob[r] * self.allocations[r]['weights']

        return blended_weights, current_prob

    def backtest(self, min_history=100, rebalance_freq=5):
        """
        Backtest dynamic allocation strategy.
        """
        returns = self.rp.returns.values
        n_samples = len(returns)

        # Track portfolio
        portfolio_returns = []
        weights_history = []
        regime_probs_history = []

        current_weights = np.ones(self.rp.n_assets) / self.rp.n_assets

        for t in range(min_history, n_samples):
            # Rebalance periodically
            if (t - min_history) % rebalance_freq == 0:
                lookback = self.rp.returns.iloc[:t, 0].values
                current_weights, current_probs = self.get_current_allocation(
                    lookback
                )
                regime_probs_history.append(current_probs)

            # Calculate portfolio return
            port_ret = current_weights @ returns[t]
            portfolio_returns.append(port_ret)
            weights_history.append(current_weights.copy())

        return np.array(portfolio_returns), np.array(weights_history)

# Backtest
allocator = DynamicRegimeAllocator(portfolio)
port_returns, weights_hist = allocator.backtest(min_history=100, rebalance_freq=5)

# Compare to buy-and-hold
bh_weights = np.ones(4) / 4
bh_returns = returns_df.iloc[100:].values @ bh_weights

# Performance comparison
print("\nBacktest Results:")
print("=" * 60)

strategies = {
    'Dynamic Regime': port_returns,
    'Buy-and-Hold (Equal)': bh_returns
}

for name, rets in strategies.items():
    cum_ret = np.prod(1 + rets) - 1
    ann_ret = (1 + cum_ret) ** (252 / len(rets)) - 1
    ann_vol = np.std(rets) * np.sqrt(252)
    sharpe = ann_ret / ann_vol
    max_dd = np.min(np.cumprod(1 + rets) / np.maximum.accumulate(np.cumprod(1 + rets))) - 1

    print(f"\n{name}:")
    print(f"  Cumulative Return: {cum_ret:.1%}")
    print(f"  Annual Return: {ann_ret:.1%}")
    print(f"  Annual Volatility: {ann_vol:.1%}")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Max Drawdown: {max_dd:.1%}")

# Visualize
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Cumulative returns
ax1 = axes[0]
ax1.plot(np.cumprod(1 + port_returns), label='Dynamic Regime')
ax1.plot(np.cumprod(1 + bh_returns), label='Buy-and-Hold')
ax1.set_ylabel('Cumulative Return')
ax1.set_title('Strategy Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Weights over time
ax2 = axes[1]
for i, asset in enumerate(asset_names):
    ax2.fill_between(range(len(weights_hist)),
                     weights_hist[:, :i].sum(axis=1) if i > 0 else 0,
                     weights_hist[:, :i+1].sum(axis=1),
                     label=asset, alpha=0.7)
ax2.set_ylabel('Weight')
ax2.set_title('Dynamic Asset Allocation')
ax2.legend(loc='upper right')

# Rolling Sharpe
ax3 = axes[2]
window = 60
rolling_sharpe_dynamic = pd.Series(port_returns).rolling(window).mean() / \
                          pd.Series(port_returns).rolling(window).std() * np.sqrt(252)
rolling_sharpe_bh = pd.Series(bh_returns).rolling(window).mean() / \
                     pd.Series(bh_returns).rolling(window).std() * np.sqrt(252)

ax3.plot(rolling_sharpe_dynamic.values, label='Dynamic Regime')
ax3.plot(rolling_sharpe_bh.values, label='Buy-and-Hold')
ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
ax3.set_ylabel('Rolling Sharpe')
ax3.set_xlabel('Time')
ax3.set_title(f'{window}-Day Rolling Sharpe Ratio')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

</div>

## Risk Management with Regimes


<span class="filename">regimeawareriskmanager.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
class RegimeAwareRiskManager:
    """
    Risk management that adapts to market regimes.
    """

    def __init__(self, regime_portfolio):
        self.rp = regime_portfolio

    def regime_conditional_var(self, weights, confidence=0.95):
        """
        Calculate VaR conditional on each regime.
        """
        from scipy import stats

        var_by_regime = {}

        for r in range(self.rp.n_regimes):
            params = self.rp.regime_params[r]

            port_mean = weights @ params['mean'] / 252  # Daily
            port_vol = np.sqrt(weights @ params['cov'] @ weights / 252)

            # Parametric VaR
            var = -(port_mean - stats.norm.ppf(confidence) * port_vol)

            var_by_regime[r] = {
                'var': var,
                'var_pct': var * 100,
                'daily_vol': port_vol
            }

        return var_by_regime

    def expected_var(self, weights, current_probs, confidence=0.95):
        """
        Probability-weighted VaR across regimes.
        """
        regime_var = self.regime_conditional_var(weights, confidence)

        expected_var = sum(
            current_probs[r] * regime_var[r]['var']
            for r in range(self.rp.n_regimes)
        )

        return expected_var

    def stress_test(self, weights, scenario='bear_regime'):
        """
        Stress test portfolio under specific scenarios.
        """
        # Find bear regime (higher volatility)
        vols = [self.rp.regime_params[r]['volatility'][0]
                for r in range(self.rp.n_regimes)]
        bear_regime = np.argmax(vols)

        if scenario == 'bear_regime':
            # Assume we're in bear regime
            params = self.rp.regime_params[bear_regime]

            port_return = weights @ params['mean']
            port_vol = np.sqrt(weights @ params['cov'] @ weights)

            # 1 month return distribution in bear regime
            monthly_mean = params['mean'] / 12
            monthly_cov = params['cov'] / 12

            monthly_port_mean = weights @ monthly_mean
            monthly_port_vol = np.sqrt(weights @ monthly_cov @ weights)

            scenarios = {
                '1-sigma down': monthly_port_mean - monthly_port_vol,
                '2-sigma down': monthly_port_mean - 2 * monthly_port_vol,
                '3-sigma down': monthly_port_mean - 3 * monthly_port_vol,
                'Expected': monthly_port_mean
            }

            return scenarios

    def position_sizing(self, weights, max_var=0.02, confidence=0.95):
        """
        Determine position size based on VaR limit.
        """
        # Calculate VaR at full weight
        from scipy import stats

        # Use highest volatility regime for conservative sizing
        max_vol_regime = np.argmax([
            self.rp.regime_params[r]['volatility'][0]
            for r in range(self.rp.n_regimes)
        ])

        params = self.rp.regime_params[max_vol_regime]
        port_vol = np.sqrt(weights @ params['cov'] @ weights / 252)
        z_score = stats.norm.ppf(confidence)

        # VaR = z * sigma
        # max_var = z * sigma * scale
        # scale = max_var / (z * sigma)
        scale = max_var / (z_score * port_vol)

        return min(scale, 1.0)  # Cap at 100%

# Example
risk_manager = RegimeAwareRiskManager(portfolio)

# Use equal weights for demonstration
test_weights = np.array([0.4, 0.3, 0.2, 0.1])

print("Risk Management Analysis:")
print("=" * 60)

# Regime-conditional VaR
var_by_regime = risk_manager.regime_conditional_var(test_weights)
for r, var_info in var_by_regime.items():
    regime_label = 'Bull' if portfolio.regime_params[r]['mean'][0] > 0 else 'Bear'
    print(f"\n{regime_label} Regime:")
    print(f"  Daily VaR (95%): {var_info['var_pct']:.2f}%")
    print(f"  Daily Volatility: {var_info['daily_vol']*100:.2f}%")

# Stress test
stress = risk_manager.stress_test(test_weights)
print("\nStress Test (Bear Regime, 1 Month):")
for scenario, ret in stress.items():
    print(f"  {scenario}: {ret*100:.2f}%")

# Position sizing
scale = risk_manager.position_sizing(test_weights, max_var=0.02)
print(f"\nPosition Size for 2% Daily VaR limit: {scale*100:.1f}%")
```

</div>

<div class="callout-insight">

**Insight:** Understanding regime-based portfolio allocation is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Key Takeaways

1. **Regime-specific parameters** differ significantly - optimize for each

2. **Dynamic allocation** blends regime-optimal portfolios by probability

3. **Risk management** should condition on current regime

4. **Conservative sizing** uses worst-case regime parameters

5. **Backtest rigorously** to validate regime-switching strategies

6. **Transaction costs** matter for frequent rebalancing

---

## Conceptual Practice Questions

1. How would you adjust portfolio allocation based on HMM regime probabilities?

2. Why is regime-aware allocation potentially better than static mean-variance optimization?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./02_portfolio_allocation_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_market_regimes.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_financial_applications.md">
  <div class="link-card-title">01 Financial Applications</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

