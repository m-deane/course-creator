# Copy-Paste Recipes

Short, focused code snippets for common bandit problems. Each recipe solves ONE specific problem in < 20 lines.

## How to Use

1. Find the recipe that matches your problem
2. Copy the function
3. Paste into your project
4. Call with your data

No configuration needed - just copy and run.

---

## Recipe Files

### common_patterns.py
**General-purpose bandit algorithms and techniques**

| Recipe | Problem Solved | Lines |
|--------|---------------|-------|
| `thompson_sampling_select()` | Simplest Thompson Sampling | 10 |
| `epsilon_greedy_decay()` | Exploration that decreases over time | 15 |
| `ucb1_custom()` | UCB with adjustable confidence | 12 |
| `retire_poor_arms()` | Drop underperforming options | 18 |
| `sliding_window_reward()` | Non-stationary environments | 15 |
| `normalize_rewards()` | Handle different reward scales | 16 |
| `softmax_select()` | Probabilistic weighted selection | 12 |
| `optimistic_init()` | Encourage initial exploration | 10 |

**Example usage:**
```python
from common_patterns import thompson_sampling_select

arms = {
    "A": {"successes": 10, "failures": 5},
    "B": {"successes": 8, "failures": 3}
}
chosen = thompson_sampling_select(arms)
```

---

### commodity_recipes.py
**Domain-specific patterns for commodity/portfolio allocation**

| Recipe | Problem Solved | Lines |
|--------|---------------|-------|
| `weekly_commodity_tilt()` | Core-satellite allocation | 18 |
| `compute_regime_features()` | Market regime detection | 15 |
| `risk_adjusted_reward()` | Sharpe-like with drawdown | 14 |
| `seasonal_reward_weighting()` | Seasonal commodity emphasis | 16 |
| `correlation_guardrail()` | Prevent correlated concentration | 19 |
| `kelly_position_size()` | Position sizing with Kelly | 12 |
| `volatility_scaled_allocation()` | Risk parity weighting | 10 |

**Example usage:**
```python
from commodity_recipes import risk_adjusted_reward
import pandas as pd

returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
reward = risk_adjusted_reward(returns, risk_free_rate=0.04)
```

---

### evaluation_recipes.py
**Performance monitoring and comparison**

| Recipe | Problem Solved | Lines |
|--------|---------------|-------|
| `plot_cumulative_regret()` | Visualize opportunity cost | 15 |
| `plot_arm_distribution()` | Show exploration patterns | 14 |
| `compare_policies()` | Run multiple algorithms | 20 |
| `inverse_propensity_score()` | Offline policy evaluation | 18 |
| `detect_reward_anomaly()` | Detect performance degradation | 16 |
| `compute_reward_confidence_interval()` | Uncertainty in estimates | 12 |
| `sample_size_for_power()` | Required sample size | 14 |
| `thompson_credible_intervals()` | Bayesian uncertainty | 13 |

**Example usage:**
```python
from evaluation_recipes import plot_cumulative_regret

history = [
    {"arm": "A", "reward": 0.7},
    {"arm": "B", "reward": 0.5},
    {"arm": "A", "reward": 0.8}
]
plot_cumulative_regret(history, best_arm_reward=0.8)
```

---

## Recipe Categories

### Algorithm Implementation
Quick implementations of core algorithms:
- Thompson Sampling (10 lines)
- Epsilon-Greedy with decay (15 lines)
- UCB1 with custom confidence (12 lines)
- Softmax exploration (12 lines)

### Reward Engineering
Transform raw outcomes into effective rewards:
- Risk-adjusted rewards (Sharpe + drawdown)
- Seasonal weighting for time-dependent data
- Sliding window for non-stationary environments
- Reward normalization across scales

### Guardrails & Constraints
Prevent pathological behavior:
- Arm retirement (drop poor performers)
- Correlation guardrails (avoid concentration)
- Position sizing (Kelly criterion)
- Volatility scaling (risk parity)

### Evaluation & Monitoring
Track and compare performance:
- Cumulative regret plots
- Policy comparison across algorithms
- Offline evaluation (IPS)
- Anomaly detection
- Statistical power calculations

### Domain-Specific
Specialized patterns:
- Commodity portfolio allocation
- Market regime features
- Weekly rebalancing logic

---

## Design Principles

Every recipe follows these rules:

1. **One problem, one function** - Focused and reusable
2. **< 20 lines** - Quick to understand and copy
3. **No dependencies beyond numpy/pandas** - Minimal requirements
4. **Input → Output clear** - Obvious what goes in and comes out
5. **Problem statement as comment** - Explains when to use it
6. **Complete code** - No placeholders or TODOs

---

## Common Patterns

### Pattern 1: Stateless Selection
```python
def select_arm(arms: Dict[str, float]) -> str:
    """Takes current state, returns arm to pull"""
    # Selection logic here
    return chosen_arm
```

### Pattern 2: Reward Transformation
```python
def compute_reward(raw_data: pd.Series) -> float:
    """Transform raw outcome into bandit reward [0, 1]"""
    # Computation here
    return normalized_reward
```

### Pattern 3: Constraint Enforcement
```python
def apply_guardrail(allocation: Dict[str, float]) -> Dict[str, float]:
    """Enforce business rules on allocation"""
    # Constraint logic here
    return constrained_allocation
```

### Pattern 4: Visualization
```python
def plot_metric(data: List[Dict], save_path: str = "plot.png"):
    """Generate diagnostic plot"""
    # Plotting logic here
    plt.savefig(save_path)
```

---

## Integration Examples

### Quick Thompson Sampling
```python
from common_patterns import thompson_sampling_select

# Your data
arms = {
    "homepage_v1": {"successes": 100, "failures": 50},
    "homepage_v2": {"successes": 120, "failures": 40}
}

# Each user visit
chosen = thompson_sampling_select(arms)
show_version(chosen)

# After conversion/no-conversion
if user_converted:
    arms[chosen]["successes"] += 1
else:
    arms[chosen]["failures"] += 1
```

### Commodity Allocation
```python
from commodity_recipes import weekly_commodity_tilt, risk_adjusted_reward
import pandas as pd

# Load price data
prices = load_commodity_prices(["GLD", "SLV", "USO"])

# Calculate rewards
for ticker in prices.columns:
    returns = prices[ticker].pct_change().tail(20)
    reward = risk_adjusted_reward(returns)
    update_bandit_stats(ticker, reward)

# Get allocation
allocation = weekly_commodity_tilt(
    prices,
    bandit_stats,
    core_weight=0.6,
    bandit_weight=0.4
)
```

### Policy Evaluation
```python
from evaluation_recipes import plot_cumulative_regret, plot_arm_distribution

# After running bandit
plot_cumulative_regret(history, best_arm_reward=0.8, save_path="regret.png")
plot_arm_distribution(history, save_path="exploration.png")

# Check for anomalies
recent = [h["reward"] for h in history[-50:]]
historical = [h["reward"] for h in history[:-50]]
is_anomaly, z_score = detect_reward_anomaly(recent, historical)

if is_anomaly:
    print(f"Warning: Recent performance anomaly (z={z_score:.2f})")
```

---

## Dependencies

All recipes require:
```bash
pip install numpy pandas
```

For visualization recipes:
```bash
pip install matplotlib
```

For statistical recipes:
```bash
pip install scipy
```

---

## Next Steps

1. **Browse recipes** - Find one that matches your problem
2. **Copy function** - Paste into your project
3. **Test with sample data** - Verify it works
4. **Integrate** - Wire up your real data
5. **Iterate** - Combine multiple recipes as needed

## Related Resources

- **templates/** - Full production systems (100-200 lines)
- **quick-starts/** - Interactive learning notebooks
- **modules/** - Detailed algorithm explanations

---

## Contributing Patterns

If you develop a useful pattern:
1. Keep it < 20 lines
2. Add clear problem statement
3. Include example usage
4. Test with sample data

Good recipes are:
- **Focused** - One problem only
- **Reusable** - Works in multiple contexts
- **Clear** - Obvious inputs and outputs
- **Complete** - No TODOs or placeholders
