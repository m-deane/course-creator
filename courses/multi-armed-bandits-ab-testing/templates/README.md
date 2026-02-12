# Production-Ready Templates

Copy these templates directly into your projects. Each template is fully functional with clear customization points marked by `# TODO: Customize here`.

## Quick Start

```bash
# 1. Copy template to your project
cp bandit_engine_template.py my_project/

# 2. Customize the CONFIG section at the top
# 3. Run it
python my_project/bandit_engine_template.py
```

## Available Templates

### 1. bandit_engine_template.py
**General-purpose multi-armed bandit engine**

**Use cases:**
- Content recommendation
- Ad placement optimization
- Feature variant testing
- Resource allocation

**Features:**
- Multiple policies: epsilon-greedy, UCB1, Thompson Sampling
- Built-in guardrails (min pulls, max allocation)
- Structured logging and reporting
- Production-ready error handling

**Time to working:** 5 minutes

**Key customizations:**
```python
CONFIG = {
    "arms": ["option_a", "option_b", "option_c"],  # Your options
    "policy": "thompson_sampling",                  # Your algorithm
    "epsilon": 0.1,                                 # Exploration rate
    "min_pulls_per_arm": 10,                        # Safety threshold
}
```

---

### 2. commodity_allocator_template.py
**Portfolio allocation with core-satellite strategy**

**Use cases:**
- Commodity portfolio management
- Sector rotation strategies
- Multi-asset allocation
- Dynamic rebalancing

**Features:**
- Core-satellite framework (stable core + adaptive sleeve)
- Multiple reward functions (raw return, Sharpe, stability-weighted)
- Risk guardrails (position limits, volatility caps)
- Automatic data loading from yfinance (with synthetic fallback)
- Weekly rebalancing with full backtest

**Time to working:** 10 minutes

**Key customizations:**
```python
CONFIG = {
    "tickers": ["GLD", "SLV", "DBA", "USO"],  # Your commodities/ETFs
    "core_weight": 0.6,                        # Core allocation (60%)
    "bandit_weight": 0.4,                      # Bandit sleeve (40%)
    "reward_function": "stability_weighted",   # Your reward metric
    "max_allocation": 0.50,                    # Max 50% per commodity
}
```

**Output:**
- Portfolio value over time
- Final allocation percentages
- Total return and performance metrics

---

### 3. ab_migration_template.py
**Gradual migration from A/B testing to bandits**

**Use cases:**
- Transitioning from fixed A/B tests
- Adaptive experimentation
- Multi-stage optimization
- Risk-controlled algorithm adoption

**Features:**
- Burn-in phase with uniform exploration
- Statistical significance testing for phase switch
- Automatic policy transition
- Dual-phase logging and reporting

**Time to working:** 5 minutes

**Key customizations:**
```python
CONFIG = {
    "arms": ["variant_a", "variant_b", "variant_c"],
    "burn_in_rounds": 100,              # A/B test duration
    "min_samples_per_arm": 30,          # Min data per variant
    "significance_threshold": 0.05,     # p-value for switching
    "bandit_policy": "thompson_sampling",
}
```

**How it works:**
1. **Phase 1 (Burn-in):** Run as traditional A/B test with uniform randomization
2. **Statistical test:** Check for significant differences between arms
3. **Phase 2 (Bandit):** Switch to adaptive policy if significance detected
4. **Export:** Full history with phase annotations

---

### 4. contextual_bandit_template.py
**Personalization with contextual features using LinUCB**

**Use cases:**
- Personalized recommendations
- Context-aware ad placement
- Patient treatment selection
- Dynamic content routing

**Features:**
- LinUCB algorithm with online learning
- Feature engineering pipeline
- Uncertainty quantification
- Per-arm weight tracking

**Time to working:** 10 minutes

**Key customizations:**
```python
CONFIG = {
    "arms": ["option_a", "option_b", "option_c"],
    "feature_names": ["age", "location_score", "engagement_score"],
    "feature_dim": 3,              # Must match number of features
    "alpha": 1.0,                  # Exploration strength
}
```

**Usage pattern:**
```python
# Initialize
engine = ContextualBanditEngine(arms, feature_dim, alpha)

# Each decision round
context = {"age": 35, "location_score": 8, "engagement_score": 0.7}
chosen_arm = engine.select_arm(context)

# After observing outcome
reward = get_reward(chosen_arm, context)
engine.record_reward(chosen_arm, context, reward)
```

---

## Template Design Principles

All templates follow these conventions:

1. **CONFIG dict at top** - All customizable parameters in one place
2. **Clear TODO markers** - Obvious customization points
3. **Production patterns** - Logging, error handling, validation
4. **< 200 lines** - Focused and readable
5. **Runnable out-of-box** - Working main() demonstrates usage
6. **Copy-paste ready** - No mocks, stubs, or placeholders

## Installation

Templates require:
```bash
pip install numpy pandas scipy
```

For commodity allocator (optional):
```bash
pip install yfinance
```

## Next Steps

After copying a template:

1. **Customize CONFIG** - Set your arms, policy, and parameters
2. **Integrate data source** - Replace synthetic data with real data
3. **Add business logic** - Customize reward functions for your use case
4. **Deploy** - Add API wrapper, database logging, or monitoring
5. **Iterate** - Tune parameters based on performance

## Related Resources

- **recipes/** - Copy-paste code snippets for specific problems
- **quick-starts/** - Interactive notebooks to learn concepts
- **modules/** - Deep-dive guides on algorithms and theory

## Support

Each template includes:
- Inline comments explaining key logic
- Example usage in main()
- Error messages with clear diagnostics
- Logging for debugging

For questions or issues, refer to the course modules for detailed explanations of algorithms and design patterns.
