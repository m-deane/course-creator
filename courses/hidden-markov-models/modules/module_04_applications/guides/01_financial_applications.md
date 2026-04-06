# Financial Applications of HMMs

> **Reading time:** ~6 min | **Module:** Module 4: Applications | **Prerequisites:** Modules 0-3

<div class="callout-key">

**Key Concept Summary:** HMMs are widely used in finance for: - Market regime detection - Volatility modeling - Asset allocation - Risk management - Trading signal generation

</div>

## Overview

HMMs are widely used in finance for:
- Market regime detection
- Volatility modeling
- Asset allocation
- Risk management
- Trading signal generation

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Market Regime Detection

### Two-State Bull/Bear Model

```python
import numpy as np
import pandas as pd
from hmmlearn import hmm
import yfinance as yf

class MarketRegimeModel:
    """Two-state market regime model."""

    def __init__(self):
        self.model = None
        self.bull_state = None
        self.bear_state = None

    def fit(self, returns: np.ndarray):
        """Fit 2-state Gaussian HMM."""
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        self.model = hmm.GaussianHMM(
            n_components=2,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        self.model.fit(returns)

        # Identify bull/bear by mean
        self.bull_state = np.argmax(self.model.means_)
        self.bear_state = np.argmin(self.model.means_)

        return self

    def predict_regime(self, returns: np.ndarray) -> np.ndarray:
        """Predict regime (0=bear, 1=bull)."""
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        states = self.model.predict(returns)

        # Remap to consistent labeling
        regime = np.zeros_like(states)
        regime[states == self.bull_state] = 1
        regime[states == self.bear_state] = 0

        return regime

    def regime_probability(self, returns: np.ndarray) -> np.ndarray:
        """Get probability of bull regime."""
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        probs = self.model.predict_proba(returns)
        return probs[:, self.bull_state]

    def get_parameters(self) -> dict:
        """Get regime parameters."""
        return {
            'bull': {
                'mean': self.model.means_[self.bull_state, 0],
                'std': np.sqrt(self.model.covars_[self.bull_state, 0, 0]),
                'persistence': self.model.transmat_[self.bull_state, self.bull_state]
            },
            'bear': {
                'mean': self.model.means_[self.bear_state, 0],
                'std': np.sqrt(self.model.covars_[self.bear_state, 0, 0]),
                'persistence': self.model.transmat_[self.bear_state, self.bear_state]
            }
        }
```

### Regime-Based Trading Strategy

```python
class RegimeStrategy:
    """Trading strategy based on HMM regime detection."""

    def __init__(self, regime_model: MarketRegimeModel):
        self.regime_model = regime_model

    def generate_signals(
        self,
        returns: np.ndarray,
        bull_threshold: float = 0.7,
        bear_threshold: float = 0.3
    ) -> np.ndarray:
        """
        Generate trading signals based on regime probability.

        Returns:
            signals: 1 (long), 0 (neutral), -1 (short)
        """
        bull_prob = self.regime_model.regime_probability(returns)

        signals = np.zeros(len(bull_prob))
        signals[bull_prob > bull_threshold] = 1   # Long in bull
        signals[bull_prob < bear_threshold] = -1  # Short in bear

        return signals

    def backtest(
        self,
        returns: np.ndarray,
        signals: np.ndarray
    ) -> dict:
        """Backtest the strategy."""
        # Shift signals (trade on next day's open)
        signals_shifted = np.roll(signals, 1)
        signals_shifted[0] = 0

        # Strategy returns
        strategy_returns = signals_shifted * returns.flatten()

        # Metrics
        total_return = np.exp(np.sum(np.log1p(strategy_returns))) - 1
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        max_dd = self._max_drawdown(strategy_returns)

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': (strategy_returns > 0).mean()
        }

    def _max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

# Example usage
data = yf.download('SPY', start='2015-01-01', end='2024-01-01')
returns = data['Adj Close'].pct_change().dropna().values

regime_model = MarketRegimeModel()
regime_model.fit(returns)

strategy = RegimeStrategy(regime_model)
signals = strategy.generate_signals(returns)
results = strategy.backtest(returns, signals)

print("Backtest Results:")
for k, v in results.items():
    print(f"  {k}: {v:.4f}")
```

## Volatility Regime Detection

### Multi-State Volatility Model


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">volatilityregimemodel.py</span>

```python
class VolatilityRegimeModel:
    """Detect volatility regimes using HMM on realized volatility."""

    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = None

    def fit(self, returns: np.ndarray, window: int = 21):
        """Fit model on realized volatility."""
        # Calculate rolling volatility
        returns_series = pd.Series(returns.flatten())
        realized_vol = returns_series.rolling(window).std() * np.sqrt(252)
        realized_vol = realized_vol.dropna().values.reshape(-1, 1)

        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        self.model.fit(realized_vol)

        # Sort regimes by mean volatility
        self._sort_regimes()

        return self

    def _sort_regimes(self):
        """Sort regimes from low to high volatility."""
        order = np.argsort(self.model.means_.flatten())
        self.model.means_ = self.model.means_[order]
        self.model.covars_ = self.model.covars_[order]
        self.model.transmat_ = self.model.transmat_[order][:, order]
        self.model.startprob_ = self.model.startprob_[order]

    def predict(self, realized_vol: np.ndarray) -> np.ndarray:
        """Predict volatility regime."""
        if realized_vol.ndim == 1:
            realized_vol = realized_vol.reshape(-1, 1)
        return self.model.predict(realized_vol)

    def get_regime_labels(self) -> dict:
        """Get descriptive labels for regimes."""
        labels = {}
        means = self.model.means_.flatten()

        for i, mean in enumerate(means):
            if i == 0:
                labels[i] = 'low_vol'
            elif i == self.n_regimes - 1:
                labels[i] = 'high_vol'
            else:
                labels[i] = f'medium_vol_{i}'

        return labels
```

</div>
</div>

## Asset Allocation with Regimes


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">regimeawareallocator.py</span>

```python
class RegimeAwareAllocator:
    """Asset allocation based on market regimes."""

    def __init__(
        self,
        regime_model: MarketRegimeModel,
        allocations: dict = None
    ):
        self.regime_model = regime_model
        self.allocations = allocations or {
            'bull': {'stocks': 0.8, 'bonds': 0.2},
            'bear': {'stocks': 0.3, 'bonds': 0.7}
        }

    def get_allocation(self, returns: np.ndarray) -> dict:
        """Get current allocation based on regime."""
        bull_prob = self.regime_model.regime_probability(returns)[-1]

        # Blend allocations by probability
        allocation = {}
        for asset in self.allocations['bull'].keys():
            allocation[asset] = (
                bull_prob * self.allocations['bull'][asset] +
                (1 - bull_prob) * self.allocations['bear'][asset]
            )

        return allocation

    def backtest_allocation(
        self,
        returns_dict: dict,  # {'stocks': returns, 'bonds': returns}
        reference_returns: np.ndarray  # For regime detection
    ) -> dict:
        """Backtest regime-based allocation."""
        n_periods = len(reference_returns)
        portfolio_returns = np.zeros(n_periods)

        for t in range(1, n_periods):
            # Get allocation based on data up to t-1
            allocation = self.get_allocation(reference_returns[:t])

            # Calculate portfolio return at time t
            for asset, weight in allocation.items():
                portfolio_returns[t] += weight * returns_dict[asset][t]

        return {
            'returns': portfolio_returns,
            'total_return': np.exp(np.sum(np.log1p(portfolio_returns))) - 1,
            'sharpe': np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
        }
```

</div>
</div>

## Regime-Switching Risk Models


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">regimeswitchingvar.py</span>

```python
class RegimeSwitchingVaR:
    """Value at Risk with regime switching."""

    def __init__(self, regime_model: MarketRegimeModel):
        self.regime_model = regime_model

    def calculate_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95,
        horizon: int = 1
    ) -> dict:
        """Calculate VaR conditional on current regime."""
        params = self.regime_model.get_parameters()
        bull_prob = self.regime_model.regime_probability(returns)[-1]

        # State-conditional VaR
        from scipy import stats

        bull_var = params['bull']['mean'] - stats.norm.ppf(confidence) * params['bull']['std']
        bear_var = params['bear']['mean'] - stats.norm.ppf(confidence) * params['bear']['std']

        # Probability-weighted VaR
        weighted_var = bull_prob * bull_var + (1 - bull_prob) * bear_var

        # Adjust for horizon
        weighted_var *= np.sqrt(horizon)

        return {
            'bull_var': bull_var * np.sqrt(horizon),
            'bear_var': bear_var * np.sqrt(horizon),
            'weighted_var': weighted_var,
            'bull_prob': bull_prob
        }

    def calculate_expected_shortfall(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> dict:
        """Calculate Expected Shortfall (CVaR)."""
        params = self.regime_model.get_parameters()
        bull_prob = self.regime_model.regime_probability(returns)[-1]

        from scipy import stats

        # ES for normal distribution: -mu + sigma * phi(Phi^-1(alpha)) / (1-alpha)
        z = stats.norm.ppf(confidence)
        phi_z = stats.norm.pdf(z)

        bull_es = -params['bull']['mean'] + params['bull']['std'] * phi_z / (1 - confidence)
        bear_es = -params['bear']['mean'] + params['bear']['std'] * phi_z / (1 - confidence)

        weighted_es = bull_prob * bull_es + (1 - bull_prob) * bear_es

        return {
            'bull_es': bull_es,
            'bear_es': bear_es,
            'weighted_es': weighted_es
        }
```

</div>
</div>

## Practical Considerations

### Look-Ahead Bias


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">expanding_window_regime.py</span>

```python
def expanding_window_regime(
    returns: np.ndarray,
    min_periods: int = 252
) -> np.ndarray:
    """
    Detect regimes using only past data (no look-ahead bias).
    """
    n = len(returns)
    regimes = np.zeros(n)

    for t in range(min_periods, n):
        # Fit on data up to time t
        model = hmm.GaussianHMM(n_components=2, n_iter=100, random_state=42)
        model.fit(returns[:t].reshape(-1, 1))

        # Predict regime at time t
        bull_state = np.argmax(model.means_)
        state = model.predict(returns[:t].reshape(-1, 1))[-1]
        regimes[t] = 1 if state == bull_state else 0

    return regimes
```

</div>
</div>

### Multiple Random Starts


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">fit_with_multiple_starts.py</span>

```python
def fit_with_multiple_starts(
    returns: np.ndarray,
    n_components: int = 2,
    n_starts: int = 10
) -> hmm.GaussianHMM:
    """Fit HMM with multiple random initializations."""
    best_score = -np.inf
    best_model = None

    for seed in range(n_starts):
        model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=1000,
            random_state=seed
        )

        model.fit(returns.reshape(-1, 1))
        score = model.score(returns.reshape(-1, 1))

        if score > best_score:
            best_score = score
            best_model = model

    return best_model
```

</div>
</div>

<div class="callout-insight">

**Insight:** Understanding financial applications of hmms is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Key Takeaways

1. **Market regimes** naturally map to HMM states (bull/bear)

2. **Regime-based strategies** can improve risk-adjusted returns

3. **Avoid look-ahead bias** by using expanding windows

4. **Multiple initializations** improve model robustness

5. **Risk models** should be regime-conditional for accuracy

---

## Conceptual Practice Questions

1. How can an HMM detect market regime changes in equity returns?

2. What are the practical limitations of using HMMs for real-time regime detection?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./01_financial_applications_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_market_regimes.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./02_portfolio_allocation.md">
  <div class="link-card-title">02 Portfolio Allocation</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

