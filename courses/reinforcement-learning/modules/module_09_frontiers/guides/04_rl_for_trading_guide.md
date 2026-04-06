# Reinforcement Learning for Trading

> **Reading time:** ~15 min | **Module:** 9 — Frontiers | **Prerequisites:** Modules 5-8

## In Brief

Applying RL to financial markets formalizes portfolio management and execution as a sequential decision problem: at each time step, an agent observes market state, selects a position or trade, and receives a reward based on risk-adjusted returns. Unlike supervised learning approaches that predict prices, RL directly optimizes the objective that matters — portfolio performance — accounting for transaction costs, risk, and sequential dependencies.

<div class="callout-key">

<strong>Key Concept:</strong> Applying RL to financial markets formalizes portfolio management and execution as a sequential decision problem: at each time step, an agent observes market state, selects a position or trade, and receives a reward based on risk-adjusted returns. Unlike supervised learning approaches that predict prices, RL directly optimizes the objective that matters — portfolio performance — accounting for transaction costs, risk, and sequential dependencies.

</div>


## Key Insight

The central design challenge is the reward function: a naive reward of raw P&L encourages excessive risk, while a well-designed reward incorporating the Sharpe ratio or transaction costs shapes the agent toward realistic, deployable behavior. Getting the environment right (state representation, action space, reward signal, episode structure) matters more than algorithm choice.

---


<div class="callout-key">

<strong>Key Point:</strong> The central design challenge is the reward function: a naive reward of raw P&L encourages excessive risk, while a well-designed reward incorporating the Sharpe ratio or transaction costs shapes the ag...

</div>

## Formal Definition: The Trading MDP

The trading problem maps onto an MDP $(\mathcal{S}, \mathcal{A}, \mathcal{P}, R, \gamma)$:

<div class="callout-key">

<strong>Key Point:</strong> The trading problem maps onto an MDP $(\mathcal{S}, \mathcal{A}, \mathcal{P}, R, \gamma)$:

| Component | Trading Interpretation |
|-----------|----------------------|
| State $s \in \mathcal{S}$ | Pr...

</div>


| Component | Trading Interpretation |
|-----------|----------------------|
| State $s \in \mathcal{S}$ | Prices, holdings, technical indicators, macro signals |
| Action $a \in \mathcal{A}$ | Position changes (buy/sell/hold) or portfolio weights |
| Transition $\mathcal{P}(s' \mid s, a)$ | Market dynamics (non-stationary, partially observable) |
| Reward $R(s, a, s')$ | Risk-adjusted return increment (Sharpe, P&L minus costs) |
| Discount $\gamma$ | Time preference for near-term vs long-term returns |

---

## State Space Design

The state must capture everything relevant to the trading decision — without including information unavailable at decision time (look-ahead bias).

<div class="callout-info">

<strong>Info:</strong> The state must capture everything relevant to the trading decision — without including information unavailable at decision time (look-ahead bias).

</div>


### Common State Components

**Price and Volume Features:**

$$s_{\text{price}} = \left[\frac{p_t - p_{t-1}}{p_{t-1}}, \frac{p_t - p_{t-5}}{p_{t-5}}, \frac{v_t}{\bar{v}_{20}}, \sigma_{20}, \text{VWAP}_t\right]$$

**Portfolio State:**

$$s_{\text{portfolio}} = \left[w_1, w_2, \ldots, w_n, \text{cash fraction}, \text{pnl}_{t}\right]$$

**Technical Indicators:**

$$s_{\text{indicators}} = \left[\text{RSI}_{14}, \text{MACD}, \text{BB}_{20}, \text{ATR}_{14}\right]$$

**Macro / Cross-Asset:**

$$s_{\text{macro}} = \left[\text{VIX}, \text{yield spread}, \text{FX rates}, \text{commodity indices}\right]$$


<span class="filename">example.py</span>
</div>
The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
import pandas as pd


def build_state(prices: pd.DataFrame, holdings: np.ndarray, portfolio_value: float,
                lookback: int = 20) -> np.ndarray:
    """
    Construct the state vector for a multi-asset trading environment.

    Parameters
    ----------
    prices          : DataFrame of shape [lookback+1, n_assets] with OHLCV columns
    holdings        : Current portfolio weights (n_assets,), sums to <= 1
    portfolio_value : Current total portfolio value in base currency
    lookback        : Number of historical bars to include

    Returns
    -------
    state : 1D array of normalized features ready for neural network input
    """
    close = prices['close'].values           # shape: [lookback+1]
    volume = prices['volume'].values

    # Returns (normalized, avoiding look-ahead)
    returns = np.diff(close) / close[:-1]   # shape: [lookback]
    ret_1d  = returns[-1]
    ret_5d  = close[-1] / close[-6] - 1 if len(close) > 5 else 0.0
    ret_20d = close[-1] / close[-21] - 1 if len(close) > 20 else 0.0

    # Volatility (realized, 20-day)
    vol_20 = returns[-20:].std() if len(returns) >= 20 else returns.std()

    # Volume ratio
    vol_ratio = volume[-1] / volume[-20:].mean() if len(volume) >= 20 else 1.0

    # RSI (14-period)
    gains   = np.maximum(returns[-14:], 0)
    losses  = np.maximum(-returns[-14:], 0)
    rs      = gains.mean() / (losses.mean() + 1e-8)
    rsi     = 100.0 - 100.0 / (1.0 + rs)

    price_features = np.array([ret_1d, ret_5d, ret_20d, vol_20, vol_ratio, rsi / 100.0])

    # Current portfolio state
    portfolio_features = np.concatenate([
        holdings,                                          # weight in each asset
        [portfolio_value / 1e6],                          # normalized portfolio value
    ])

    return np.concatenate([price_features, portfolio_features])
```

</div>
</div>

---

## Action Space Design

The choice of action space fundamentally shapes what the agent can express and which algorithms apply.

<div class="callout-warning">

<strong>Warning:</strong> The choice of action space fundamentally shapes what the agent can express and which algorithms apply.

</div>


### Discrete Action Space (DQN-compatible)

```
Actions: {Strong Sell, Sell, Hold, Buy, Strong Buy}
         {  -2,        -1,   0,    1,       2     }
```

Each integer maps to a fixed trade size (e.g., 10% of portfolio per unit).

**Pros:** Compatible with DQN and all discrete-action algorithms; easy to interpret.
**Cons:** Cannot express fine-grained position sizes; discretization introduces approximation error.

### Continuous Action Space (SAC/PPO-compatible)

$$a \in [-1, 1]^n \quad \text{(target weight change for each asset)}$$

or

$$a \in \Delta^n = \{w \in \mathbb{R}^n_+ : \sum_i w_i = 1\} \quad \text{(target portfolio weights, long-only)}$$

**Pros:** Can express any position size; natural for portfolio optimization.
**Cons:** Requires continuous-action algorithms (SAC, PPO); exploration is harder.

---

## Reward Function Design

The reward is the most important design decision. Several formulations exist, each with different incentive properties.

### Raw P&L (Naive — Avoid)

$$R_t = \text{PnL}_t = \sum_i w^i_t (p^i_{t+1} - p^i_t)$$

**Problem:** No risk adjustment. Encourages maximal leverage and ignores drawdowns.

### Differential Sharpe Ratio (Recommended)

The differential Sharpe ratio approximates the instantaneous contribution to the Sharpe ratio:

$$R_t = \frac{\bar{R}_{t-1} \Delta R_t - \frac{1}{2} r_t \Delta \sigma^2_{t-1}}{\sigma^2_{t-1}}$$

where $\bar{R}_{t-1}$ and $\sigma^2_{t-1}$ are exponential moving averages of returns and variance, and $\Delta R_t = r_t - \bar{R}_{t-1}$, $\Delta \sigma^2_{t-1} = r_t^2 - \sigma^2_{t-1}$.

**Property:** Maximizing the expected cumulative differential Sharpe reward is equivalent to maximizing the overall Sharpe ratio of the strategy.

### P&L Minus Transaction Costs

$$R_t = \sum_i w^i_t (p^i_{t+1} - p^i_t) - \kappa \sum_i |w^i_t - w^i_{t-1}|$$

where $\kappa$ is the cost per unit of turnover (bid-ask spread + market impact).


<span class="filename">example.py</span>
</div>
The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def trading_reward(
    current_weights: np.ndarray,
    prev_weights: np.ndarray,
    asset_returns: np.ndarray,
    transaction_cost_rate: float = 0.001,
    risk_penalty: float = 0.1,
    window: int = 20,
    return_history: list = None,
) -> float:
    """
    Compute a risk-adjusted reward for the trading step.

    Components:
    - Portfolio return (weighted sum of asset returns)
    - Transaction cost penalty (proportional to turnover)
    - Risk penalty (proportional to realized variance)

    Parameters
    ----------
    current_weights         : Portfolio weights after rebalancing (n_assets,)
    prev_weights            : Portfolio weights before rebalancing (n_assets,)
    asset_returns           : Per-asset returns this step (n_assets,)
    transaction_cost_rate   : Cost per unit absolute weight change
    risk_penalty            : Multiplier on realized variance penalty
    window                  : Window for variance estimation
    return_history          : List of past portfolio returns (for variance)

    Returns
    -------
    reward : float
    """
    # Portfolio return this step
    portfolio_return = float(current_weights @ asset_returns)

    # Transaction costs (proportional to turnover)
    turnover = np.sum(np.abs(current_weights - prev_weights))
    transaction_cost = transaction_cost_rate * turnover

    # Variance penalty (only meaningful with enough history)
    variance_penalty = 0.0
    if return_history and len(return_history) >= window:
        realized_var = np.var(return_history[-window:])
        variance_penalty = risk_penalty * realized_var

    return portfolio_return - transaction_cost - variance_penalty
```

</div>
</div>

---

## Common Algorithmic Approaches

### DQN for Discrete Trading Actions

Use DQN when actions are buy/sell/hold (discrete). The Q-function $Q(s, a)$ estimates the expected discounted return for taking action $a$ in state $s$.

**Practical consideration:** Dueling DQN architecture separates state value from action advantage, which is effective when most actions are equivalent (hold is almost always valid):

$$Q(s, a) = V(s) + A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a')$$

### PPO for Continuous Portfolio Allocation

Use PPO when actions are portfolio weights (continuous). PPO's clipped objective prevents destabilizing updates:

$$\mathcal{L}_{\text{CLIP}} = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

**Advantage:** PPO handles the non-stationary financial environment more robustly than DDPG due to its conservative update step.

### SAC for Risk-Aware Continuous Trading

Soft Actor-Critic adds maximum entropy regularization:

$$\pi^* = \arg\max_\pi \mathbb{E}\left[\sum_t \gamma^t \left(R(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot \mid s_t))\right)\right]$$

The entropy bonus $\alpha \mathcal{H}(\pi)$ encourages the policy to maintain uncertainty across actions, which translates to position diversification. This is a natural match for portfolio allocation.

---

## Backtesting Methodology

Backtesting is the primary evaluation tool for trading strategies. Poor backtesting methodology produces inflated performance estimates.

### Correct Backtesting Protocol


<span class="filename">example.py</span>
</div>
The following implementation builds on the approach above:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def walk_forward_backtest(
    env_class,
    agent,
    prices: pd.DataFrame,
    train_window: int = 252,   # 1 year of daily data
    test_window: int = 63,     # 1 quarter
    retrain: bool = True,
) -> pd.DataFrame:
    """
    Walk-forward backtest: train on rolling window, evaluate on out-of-sample.

    This is the only valid backtesting methodology for adaptive agents.
    The agent is never evaluated on data it was trained on.

    Parameters
    ----------
    prices        : Full price history (dates x assets)
    train_window  : Number of bars in each training window
    test_window   : Number of bars in each out-of-sample evaluation window
    retrain       : Whether to retrain from scratch (True) or fine-tune (False)

    Returns
    -------
    results : DataFrame with columns [date, portfolio_value, sharpe, max_drawdown]
    """
    results = []
    n_bars = len(prices)

    for start in range(train_window, n_bars - test_window, test_window):
        train_data = prices.iloc[start - train_window : start]
        test_data  = prices.iloc[start : start + test_window]

        # Train on historical window
        train_env = env_class(train_data)
        if retrain:
            agent.reset_weights()
        agent.train(train_env, n_steps=100_000)

        # Evaluate on out-of-sample window (no further learning)
        test_env = env_class(test_data)
        episode_result = agent.evaluate(test_env, deterministic=True)

        results.append({
            'start_date':    test_data.index[0],
            'end_date':      test_data.index[-1],
            'sharpe':        episode_result['sharpe'],
            'total_return':  episode_result['total_return'],
            'max_drawdown':  episode_result['max_drawdown'],
        })

    return pd.DataFrame(results)
```

</div>
</div>

---

## Backtesting Pitfalls

### Look-Ahead Bias

Using information at time $t$ that was not available until time $t + k$:

- Using tomorrow's closing price to make today's decision
- Using earnings announcements published after market close for same-day decisions
- Computing indicators over future data (common in vectorized backtests)

**Detection:** Always use `.shift(1)` for any feature derived from the price at which you will trade.

### Survivorship Bias

Backtesting on only companies that exist today, excluding those that were delisted, went bankrupt, or were acquired.

**Effect:** The historical universe looks better than it actually was. A strategy that picks any stock from a "current" Russell 1000 list in 2000 looks great because the list only contains survivors.

**Fix:** Use point-in-time data with full constituent history.

### Transaction Cost Underestimation

- Assuming zero slippage
- Using mid-price instead of executed price
- Ignoring market impact for large orders

**Rule of thumb:** If your strategy does not survive 10 basis points of transaction costs per trade, it is not deployable.

---

## Practical Considerations

### Feature Engineering

**Normalize all features** before feeding to the neural network:


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Rolling Z-score normalization (uses only past data — no look-ahead)
def rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std  = series.rolling(window).std()
    return (series - rolling_mean) / (rolling_std + 1e-8)
```

</div>
</div>

**Use returns, not price levels.** Returns are stationary (approximately); prices are not. Neural networks trained on price levels overfit to the specific price range seen during training.

### Reward Shaping

A well-shaped reward trains faster and produces more realistic behavior:

1. **Normalize rewards** to have unit variance during training (running statistics)
2. **Include transaction costs** in the reward to discourage excessive turnover
3. **Penalize drawdowns** explicitly if drawdown control is a requirement
4. **Do not reward prediction accuracy** — reward portfolio performance directly


<div class="flow">
<div class="flow-step mint">1. Normalize rewards</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Include transaction costs</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Penalize drawdowns</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Do not reward prediction accur...</div>
</div>

### Sim-to-Real Gap

The gap between backtesting performance and live trading performance is the trading analog of sim-to-real in robotics:

| Source of Gap | Description | Mitigation |
|---------------|-------------|-----------|
| Market impact | Your trades move the price | Use market impact models during training |
| Latency | Orders execute at worse prices than expected | Model execution latency in environment |
| Regime change | Market regime in deployment differs from training | Walk-forward training; frequent retraining |
| Partial fills | Not all orders execute at desired size | Model fill probability as part of transition |

---

## Challenges Specific to Financial Markets

### Non-Stationarity

Financial markets are highly non-stationary. Regime changes (bull/bear markets, liquidity crises, rate environments) cause distributional shift that invalidates policies trained on historical data.

**Mitigations:**
- Walk-forward retraining (retrain every quarter on recent data)
- Regime detection to switch between regime-specific policies
- Online learning with decaying weight on old experiences

### Partial Observability

The true state (all market participants' intentions and information) is not observable. The agent observes only prices and public data.

**Mitigation:** LSTM or Transformer architectures to maintain a learned hidden state over the observation history.

### Sparse and Noisy Rewards

Financial returns are highly noisy (Sharpe ratio of 1.0 is excellent, meaning signal-to-noise is approximately 1:1 annually). Many time steps have near-zero rewards regardless of action quality.

**Mitigation:** Differential Sharpe reward (converts episode-level Sharpe into step-level signal); longer episodes; reward smoothing.

---

## Common Pitfalls

<div class="callout-danger">

<strong>Danger:</strong> The pitfalls below are the most common mistakes practitioners make. Each one can silently degrade your results without obvious errors.

</div>

**Pitfall 1 — Optimizing raw P&L without risk adjustment.**
An agent rewarded only on P&L will learn to maximize leverage and take excessive risk. Always include a risk term (variance penalty, Sharpe reward, drawdown constraint). The Sharpe ratio is the standard risk-adjusted performance metric.

<div class="callout-warning">

<strong>Warning:</strong> **Pitfall 1 — Optimizing raw P&L without risk adjustment.**
An agent rewarded only on P&L will learn to maximize leverage and take excessive risk.

</div>

**Pitfall 2 — Look-ahead bias in feature construction.**
Any feature computed using prices or data from after the decision point contaminates your backtest. Common culprits: indicators computed on the full bar that ends after your decision time, labels from future returns used as features. Always verify features are computable from strictly past data.

**Pitfall 3 — Ignoring transaction costs.**
A strategy that trades frequently may look profitable before costs but lose money after costs. Include realistic transaction costs (spread + market impact) in both the reward function and the backtest.

**Pitfall 4 — Evaluating on training data.**
Overfitting to historical price sequences is easy with expressive neural networks. Always use walk-forward evaluation with strict train/test separation.

**Pitfall 5 — Failing to account for market impact.**
Large orders move the market. A strategy that trades 10% of average daily volume will not achieve the simulated execution prices. Use market impact models (e.g., Almgren-Chriss) in the environment when training large-scale execution strategies.

**Pitfall 6 — Treating stationarity as given.**
A policy trained in a bull market may fail catastrophically in a bear market. Regularly retrain and monitor performance metrics in production; implement drawdown-based circuit breakers.

---

## Connections


<div class="callout-info">

<strong>Info:</strong> This section maps how this guide connects to the broader course. Use these links to navigate related material.

</div>

- **Builds on:** Offline RL (Guide 02) — financial data is effectively a fixed dataset; RLHF and Safe RL (Guide 03) — reward shaping and risk constraints apply directly
- **Builds on:** Model-based RL (Module 8) — learned market models for planning
- **Related to:** portfolio optimization (Markowitz), execution algorithms (TWAP, VWAP), high-frequency trading research
- **Related to:** multi-agent RL (Guide 01) — markets as multi-agent systems; MARL for competing execution algorithms

---


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


## Further Reading

- Moody & Saffell (2001). *Learning to Trade via Direct Reinforcement* — early formulation of the trading MDP and differential Sharpe reward
- Jiang et al. (2017). *A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem* — CNN-based portfolio management with continuous action space
- Almgren & Chriss (2001). *Optimal Execution of Portfolio Transactions* — market impact model used in realistic trading environments
- Markowitz (1952). *Portfolio Selection* — the mean-variance optimization framework that RL generalizes
- Sutton & Barto (2018). Chapter 10 — on-policy control for continuous action spaces; foundational for PPO/SAC-based trading agents
- OpenAI. *Spinning Up in Deep RL* — practical SAC and PPO implementations directly applicable to trading environments


---

## Cross-References

<a class="link-card" href="./04_rl_for_trading_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_offline_rl_basics.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
