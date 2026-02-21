---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Regime-Based Portfolio Allocation
## Dynamic Optimization with HMMs

### Module 04 — Applications
### Hidden Markov Models Course

<!-- Speaker notes: This section shows how HMM regime detection translates into actionable portfolio decisions. The key idea is that optimal allocations depend on the current market regime, which HMMs estimate probabilistically. -->
---

# Why Regime-Aware Allocation?

Market regimes have **distinct risk-return characteristics**. Static allocation ignores this:

```mermaid
flowchart TD
    subgraph Static[\"Static Allocation\"]
        SA[\"60/40 stocks/bonds<br>Always the same\"]
    end
    subgraph Dynamic[\"Regime-Aware Allocation\"]
        BULL[\"Bull: 80/20<br>Risk-on\"]
        BEAR[\"Bear: 30/70<br>Risk-off\"]
        PROB[\"Blend by P(regime)\"]
        BULL --> PROB
        BEAR --> PROB
    end
    Dynamic -->|\"Better risk-adjusted<br>returns\"| RESULT[\"Improved Sharpe\"]
```

<!-- Speaker notes: Static 60/40 allocation ignores the fact that expected returns and risks change dramatically between regimes. Dynamic allocation adapts to the current regime, improving risk-adjusted returns. -->
---

# RegimeAwarePortfolio — Setup

```python
class RegimeAwarePortfolio:
    def __init__(self, returns_data, n_regimes=2):
        self.returns = returns_data       # DataFrame (columns = assets)
        self.n_regimes = n_regimes
        self.n_assets = returns_data.shape[1]
        self.regime_model = None
        self.regime_params = {}

    def fit_regime_model(self, reference_returns=None):
        if reference_returns is None:
            reference_returns = self.returns.iloc[:, 0].values
        reference_returns = reference_returns.reshape(-1, 1)
        self.regime_model = hmm.GaussianHMM(
            n_components=self.n_regimes, covariance_type='full',
            n_iter=200, random_state=42)
        self.regime_model.fit(reference_returns)
        self.regime_probs = self.regime_model.predict_proba(reference_returns)
        return self
```

<!-- Speaker notes: This class combines regime detection with portfolio optimization. The regime model is fit on a reference asset (typically the equity index), and regime-specific parameters are estimated for all assets. -->
---

# Estimating Regime-Specific Parameters

```python
def estimate_regime_parameters(self):
    regimes = self.regime_model.predict(
        self.returns.iloc[:, 0].values.reshape(-1, 1))

    for r in range(self.n_regimes):
        mask = regimes == r
        regime_returns = self.returns.iloc[mask]
        self.regime_params[r] = {
            'mean': regime_returns.mean().values * 252,   # Annualize
            'cov': regime_returns.cov().values * 252,
            'n_obs': mask.sum(),
            'volatility': regime_returns.std().values * np.sqrt(252)
        }
    return self
```

> Annualize parameters: multiply mean by 252, covariance by 252, std by $\sqrt{252}$.

<!-- Speaker notes: For each regime, we compute annualized means and covariances from the observations assigned to that regime. The annualization factors are 252 for mean and 252 for covariance. -->
---

# Portfolio Optimization Pipeline

```mermaid
flowchart TD
    DATA[\"Multi-Asset Returns\"] --> FIT[\"Fit HMM<br>(reference asset)\"]
    FIT --> REGIMES[\"Identify Regimes\"]
    REGIMES --> PARAMS[\"Regime-Specific<br>mu, Sigma\"]
    PARAMS --> OPT[\"Mean-Variance<br>Optimization\"]
    OPT --> BULL_W[\"Bull Weights\"]
    OPT --> BEAR_W[\"Bear Weights\"]
    BULL_W --> BLEND[\"Probability-Weighted<br>Blend\"]
    BEAR_W --> BLEND
    BLEND --> ALLOC[\"Current Allocation\"]
```

<!-- Speaker notes: This flow diagram shows the complete workflow: fit HMM, identify regimes, estimate per-regime parameters, optimize portfolios per regime, blend by probability, and get the final allocation. -->
---

<!-- _class: lead -->

# Mean-Variance Optimization

<!-- Speaker notes: Mean-variance optimization provides the mathematical framework for computing optimal portfolio weights given regime-specific expected returns and covariances. -->
---

# Optimization Implementation

```python
def mean_variance_optimize(self, expected_return, cov_matrix,
                            risk_free=0.02, target_vol=None):
    n = len(expected_return)

    def neg_sharpe(weights):
        ret = weights @ expected_return
        vol = np.sqrt(weights @ cov_matrix @ weights)
        return -(ret - risk_free) / vol

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    if target_vol is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sqrt(w @ cov_matrix @ w) - target_vol})

    bounds = [(0, 1) for _ in range(n)]  # Long-only
    w0 = np.ones(n) / n

    result = minimize(neg_sharpe, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints)
    return result.x
```

<!-- Speaker notes: The mean-variance optimizer maximizes the Sharpe ratio subject to the budget constraint (weights sum to 1) and optionally a target volatility constraint. Long-only bounds prevent short positions. -->
---

# Regime-Optimal Allocations

```python
def get_regime_allocations(self, risk_free=0.02):
    allocations = {}
    for r in range(self.n_regimes):
        params = self.regime_params[r]
        weights = self.mean_variance_optimize(
            params['mean'], params['cov'], risk_free)
        port_return = weights @ params['mean']
        port_vol = np.sqrt(weights @ params['cov'] @ weights)
        allocations[r] = {
            'weights': weights,
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe': (port_return - risk_free) / port_vol}
    return allocations
```

<!-- Speaker notes: Each regime gets its own optimal portfolio. The bull portfolio tilts toward equities for higher returns, while the bear portfolio tilts toward bonds and gold for capital preservation. -->
---

# Example — Bull vs Bear Allocations

<div class="columns">
<div>

**Bull Regime**
| Asset | Weight |
|----------|----------|
| Stocks | High |
| Bonds | Low |
| Gold | Low |
| REITs | Medium |

<!-- Speaker notes: This side-by-side comparison shows the dramatic difference in optimal allocations between regimes. The practical implication is clear: a static allocation is suboptimal in both regimes. -->

Sharpe maximized with equity tilt.

</div>
<div>

**Bear Regime**
| Asset | Weight |
|----------|----------|


| Stocks | Low |
| Bonds | Medium |
| Gold | High |
| REITs | Low |

Defensive: bonds and gold for protection.

</div>
</div>

---

<!-- _class: lead -->

# Dynamic Allocation Strategy

<!-- Speaker notes: Dynamic allocation blends regime-optimal portfolios by probability, producing smooth transitions rather than abrupt switches. -->
---

# DynamicRegimeAllocator

```python
class DynamicRegimeAllocator:
    def __init__(self, regime_portfolio):
        self.rp = regime_portfolio
        self.allocations = self.rp.get_regime_allocations()

    def get_current_allocation(self, lookback_returns):
        probs = self.rp.regime_model.predict_proba(
            lookback_returns.reshape(-1, 1))
        current_prob = probs[-1]

        # Blend allocations by probability
        blended_weights = np.zeros(self.rp.n_assets)
        for r in range(self.rp.n_regimes):
            blended_weights += current_prob[r] * \
                               self.allocations[r]['weights']
        return blended_weights, current_prob
```

<!-- Speaker notes: The allocator uses the current regime probability to blend the regime-optimal portfolios. This produces smooth allocation transitions that reduce turnover compared to hard regime switching. -->
---

# Dynamic Allocation Flow

```mermaid
flowchart LR
    T0[\"t=0\"] --> FIT[\"Fit HMM\"]
    FIT --> LOOP{\"Each Rebalance<br>Period\"}
    LOOP --> PROB[\"Current Regime<br>Probability\"]
    PROB --> BLEND[\"Blend Weights\"]
    BLEND --> REBAL[\"Rebalance<br>Portfolio\"]
    REBAL --> RET[\"Track Returns\"]
    RET --> LOOP
```

<!-- Speaker notes: The rebalancing loop shows how the strategy operates in real time: at each rebalance period, recompute regime probability, blend weights, and rebalance the portfolio. -->
---

# Backtesting Dynamic Allocation

```python
def backtest(self, min_history=100, rebalance_freq=5):
    returns = self.rp.returns.values
    n_samples = len(returns)
    portfolio_returns = []
    weights_history = []
    current_weights = np.ones(self.rp.n_assets) / self.rp.n_assets

    for t in range(min_history, n_samples):
        if (t - min_history) % rebalance_freq == 0:
            lookback = self.rp.returns.iloc[:t, 0].values
            current_weights, _ = self.get_current_allocation(lookback)

        port_ret = current_weights @ returns[t]
        portfolio_returns.append(port_ret)
        weights_history.append(current_weights.copy())

    return np.array(portfolio_returns), np.array(weights_history)
```

<!-- Speaker notes: The backtest uses expanding windows to avoid look-ahead bias. The rebalance_freq parameter controls how often weights are updated, trading off responsiveness against transaction costs. -->
---

# Strategy Comparison

| Metric | Dynamic Regime | Buy-and-Hold |
|----------|----------|----------|
| Cumulative Return | Typically higher | Baseline |
| Sharpe Ratio | Improved | Lower |
| Max Drawdown | Reduced | Deeper |
| Turnover | Higher (cost) | Minimal |

<!-- Speaker notes: The comparison table shows the typical performance profile: dynamic allocation improves Sharpe ratio and reduces drawdowns at the cost of higher turnover. The net benefit depends on transaction costs. -->

> Dynamic allocation aims for **better risk-adjusted returns** at the cost of higher turnover.

---

<!-- _class: lead -->

# Regime-Aware Risk Management

<!-- Speaker notes: Regime-aware risk management conditions VaR and stress tests on the current regime, providing more accurate risk estimates than unconditional models. -->
---

# Regime-Conditional VaR

```python
class RegimeAwareRiskManager:
    def regime_conditional_var(self, weights, confidence=0.95):
        var_by_regime = {}
        for r in range(self.rp.n_regimes):
            params = self.rp.regime_params[r]
            port_mean = weights @ params['mean'] / 252
            port_vol = np.sqrt(weights @ params['cov'] @ weights / 252)
            var = -(port_mean - stats.norm.ppf(confidence) * port_vol)
            var_by_regime[r] = {
                'var': var, 'var_pct': var * 100,
                'daily_vol': port_vol}
        return var_by_regime
```

<!-- Speaker notes: VaR conditioned on the current regime provides more accurate risk estimates than unconditional VaR. In bear regimes, the VaR is significantly larger, reflecting the higher risk environment. -->
---

# Risk Management Architecture

```mermaid
flowchart TD
    WEIGHTS[\"Portfolio Weights\"] --> RVAR[\"Regime-Conditional VaR\"]
    PROBS[\"Regime Probabilities\"] --> EVAR[\"Expected VaR\"]
    RVAR --> EVAR
    EVAR --> STRESS[\"Stress Testing\"]
    STRESS --> SIZING[\"Position Sizing\"]
    SIZING --> LIMITS[\"Risk Limits\"]

    subgraph Scenarios[\"Stress Scenarios\"]
        S1[\"1-sigma down\"]
        S2[\"2-sigma down\"]
        S3[\"3-sigma down\"]
    end
    STRESS --> Scenarios
```

<!-- Speaker notes: The flow diagram shows the complete risk management pipeline from portfolio weights through regime-conditional VaR to stress testing and position sizing. -->
---

# Stress Testing

```python
def stress_test(self, weights, scenario='bear_regime'):
    vols = [self.rp.regime_params[r]['volatility'][0]
            for r in range(self.rp.n_regimes)]
    bear_regime = np.argmax(vols)

    params = self.rp.regime_params[bear_regime]
    monthly_mean = params['mean'] / 12
    monthly_cov = params['cov'] / 12

    monthly_port_mean = weights @ monthly_mean
    monthly_port_vol = np.sqrt(weights @ monthly_cov @ weights)

    return {
        '1-sigma down': monthly_port_mean - monthly_port_vol,
        '2-sigma down': monthly_port_mean - 2 * monthly_port_vol,
        '3-sigma down': monthly_port_mean - 3 * monthly_port_vol,
        'Expected': monthly_port_mean
    }
```

<!-- Speaker notes: Stress tests use bear regime parameters to simulate worst-case scenarios. The 1, 2, and 3-sigma scenarios correspond to approximately 68, 95, and 99.7 percent of outcomes under the bear regime. -->
---

# Conservative Position Sizing

```python
def position_sizing(self, weights, max_var=0.02, confidence=0.95):
    """Size positions using worst-case regime."""
    max_vol_regime = np.argmax([
        self.rp.regime_params[r]['volatility'][0]
        for r in range(self.rp.n_regimes)])

    params = self.rp.regime_params[max_vol_regime]
    port_vol = np.sqrt(weights @ params['cov'] @ weights / 252)
    z_score = stats.norm.ppf(confidence)

    # scale = max_var / (z * sigma)
    scale = max_var / (z_score * port_vol)
    return min(scale, 1.0)  # Cap at 100%
```

> Use the **highest-volatility regime** for conservative position sizing.

<!-- Speaker notes: Position sizing uses the worst-case (highest-volatility) regime to ensure that VaR limits are respected even during regime transitions. The scale factor caps total position at 100 percent. -->
---

# Key Takeaways

| Takeaway | Detail |
|----------|----------|
| Regime-specific parameters | Mean and covariance differ significantly by regime |
| Mean-variance optimization | Optimize separately for each regime |
| Dynamic allocation | Blend regime-optimal portfolios by probability |
| Rebalancing frequency | Trade off responsiveness vs. transaction costs |
| Risk conditioning | VaR/ES should condition on current regime |
| Conservative sizing | Use worst-case regime for position limits |
| Backtesting | Validate with expanding windows, realistic costs |

<!-- Speaker notes: Regime-based portfolio allocation uses HMM posteriors to dynamically adjust risk exposure. Key practical considerations include transaction cost management through regime probability thresholds, proper backtesting without look-ahead bias, and robustness across different market conditions. -->

---

# Connections

```mermaid
flowchart LR
    RD[\"Regime<br>Detection\"] --> RP[\"Regime<br>Parameters\"]
    RP --> MVO[\"Mean-Variance<br>Optimization\"]
    MVO --> DA[\"Dynamic<br>Allocation\"]
    DA --> RM[\"Risk<br>Management\"]
    RM --> PS[\"Position<br>Sizing\"]
    DA --> BT[\"Backtesting\"]
    BT --> EVAL[\"Performance<br>Evaluation\"]
```

<!-- Speaker notes: This diagram shows portfolio allocation as the practical endpoint of the HMM pipeline: Gaussian HMMs estimate regimes, which inform risk models, which drive allocation decisions. The feedback loop to model validation ensures the system remains calibrated. -->
