# Quick Reference Card

Copy this page for instant access to the most common patterns.

## "I need to..."

### Start a simple bandit experiment
```python
# Use: bandit_engine_template.py
from bandit_engine_template import BanditEngine

engine = BanditEngine(
    arms=["variant_a", "variant_b"],
    policy="thompson_sampling"
)

# Each round
arm = engine.select_arm()
reward = observe_outcome(arm)
engine.record_reward(arm, reward)
```

### Allocate portfolio across commodities
```python
# Use: commodity_allocator_template.py
from commodity_allocator_template import CommodityAllocator

allocator = CommodityAllocator({
    "tickers": ["GLD", "SLV", "USO"],
    "core_weight": 0.6,
    "bandit_weight": 0.4
})

allocator.run_backtest()
```

### Migrate from A/B test to bandit
```python
# Use: ab_migration_template.py
from ab_migration_template import ABtoBanditMigrator

migrator = ABtoBanditMigrator(
    arms=["control", "treatment"],
    config={"burn_in_rounds": 100}
)

# Each round
arm = migrator.select_arm()  # A/B first, then bandit
reward = observe_outcome(arm)
migrator.record_reward(arm, reward)
```

### Personalize based on user features
```python
# Use: contextual_bandit_template.py
from contextual_bandit_template import ContextualBanditEngine

engine = ContextualBanditEngine(
    arms=["option_a", "option_b"],
    feature_dim=3,
    alpha=1.0
)

# Each decision
context = {"age": 35, "location": 8, "engagement": 0.7}
arm = engine.select_arm(context)
reward = observe_outcome(arm, context)
engine.record_reward(arm, context, reward)
```

---

## "I need a quick function for..."

### Thompson Sampling selection
```python
# Use: common_patterns.thompson_sampling_select()
from common_patterns import thompson_sampling_select

arms = {"A": {"successes": 10, "failures": 5}}
chosen = thompson_sampling_select(arms)
```

### Calculate risk-adjusted reward
```python
# Use: commodity_recipes.risk_adjusted_reward()
from commodity_recipes import risk_adjusted_reward

returns = pd.Series([0.01, -0.02, 0.03])
reward = risk_adjusted_reward(returns, risk_free_rate=0.04)
```

### Plot cumulative regret
```python
# Use: evaluation_recipes.plot_cumulative_regret()
from evaluation_recipes import plot_cumulative_regret

history = [{"arm": "A", "reward": 0.7}, ...]
plot_cumulative_regret(history, best_arm_reward=0.8)
```

### Detect performance anomaly
```python
# Use: evaluation_recipes.detect_reward_anomaly()
from evaluation_recipes import detect_reward_anomaly

recent = [0.3, 0.35, 0.32]
historical = [0.5, 0.6, 0.55, 0.58]
is_anomaly, z_score = detect_reward_anomaly(recent, historical)
```

---

## Algorithm Selection Guide

| Use Case | Algorithm | Template/Recipe |
|----------|-----------|-----------------|
| Simple exploration/exploitation | Thompson Sampling | `bandit_engine_template.py` |
| Need deterministic exploration | UCB1 | `bandit_engine_template.py` |
| Legacy A/B test | Epsilon-Greedy | `bandit_engine_template.py` |
| Personalization with features | LinUCB | `contextual_bandit_template.py` |
| Portfolio allocation | Thompson + Guardrails | `commodity_allocator_template.py` |

## Parameter Tuning Cheatsheet

### Epsilon-Greedy
- `epsilon=0.1` → 10% exploration (good default)
- `epsilon=0.01` → 1% exploration (confident in estimates)
- `epsilon=0.3` → 30% exploration (uncertain environment)

### UCB1
- `confidence=2.0` → Standard (good default)
- `confidence=1.0` → More exploitation
- `confidence=3.0` → More exploration

### Thompson Sampling
- No parameters to tune (automatic exploration/exploitation balance)
- Best for: Bernoulli rewards (success/failure)

### LinUCB
- `alpha=1.0` → Standard (good default)
- `alpha=0.5` → Less exploration
- `alpha=2.0` → More exploration

## Common Pitfalls

### Problem: Bandit converges too quickly
**Solution:** Increase exploration
- Epsilon-Greedy: Increase `epsilon` or use `epsilon_greedy_decay()`
- UCB1: Increase `confidence` parameter
- Thompson: Use `min_pulls_per_arm` guardrail

### Problem: Rewards have different scales
**Solution:** Normalize rewards
```python
from common_patterns import normalize_rewards
arms = {"A": [10, 12], "B": [100, 110]}
normalized = normalize_rewards(arms, method="minmax")
```

### Problem: Best arm changes over time
**Solution:** Use sliding window
```python
from common_patterns import sliding_window_reward
recent_mean = sliding_window_reward(all_rewards, window_size=50)
```

### Problem: Arms are correlated
**Solution:** Apply correlation guardrail
```python
from commodity_recipes import correlation_guardrail
adjusted = correlation_guardrail(allocation, prices, max_correlated_allocation=0.6)
```

## File Locations (Absolute Paths)

```
Templates:
/home/user/course-creator/courses/multi-armed-bandits-ab-testing/templates/bandit_engine_template.py
/home/user/course-creator/courses/multi-armed-bandits-ab-testing/templates/commodity_allocator_template.py
/home/user/course-creator/courses/multi-armed-bandits-ab-testing/templates/ab_migration_template.py
/home/user/course-creator/courses/multi-armed-bandits-ab-testing/templates/contextual_bandit_template.py

Recipes:
/home/user/course-creator/courses/multi-armed-bandits-ab-testing/recipes/common_patterns.py
/home/user/course-creator/courses/multi-armed-bandits-ab-testing/recipes/commodity_recipes.py
/home/user/course-creator/courses/multi-armed-bandits-ab-testing/recipes/evaluation_recipes.py
```

## Installation

```bash
# Core dependencies
pip install numpy pandas scipy matplotlib

# Optional (for real commodity data)
pip install yfinance
```

## Testing Your Setup

```bash
# Test templates
cd /home/user/course-creator/courses/multi-armed-bandits-ab-testing/templates
python bandit_engine_template.py

# Test recipes
cd /home/user/course-creator/courses/multi-armed-bandits-ab-testing/recipes
python -c "from common_patterns import thompson_sampling_select; print('OK')"
```

## Next Steps

1. **Copy template** matching your use case
2. **Customize CONFIG** section at top
3. **Run main()** to verify it works
4. **Integrate** with your data source
5. **Deploy** to production

For detailed explanations, see:
- `templates/README.md` - Full template documentation
- `recipes/README.md` - Complete recipe catalog
- Course modules - Algorithm theory and concepts
