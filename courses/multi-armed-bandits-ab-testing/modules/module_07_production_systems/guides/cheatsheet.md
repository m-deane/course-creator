# Production Systems Cheatsheet

## Pre-Deployment Checklist

Before shipping a bandit to production, verify:

1. **Architecture separates concerns** — Policy, guardrails, logging, monitoring are independent modules
2. **Comprehensive logging** — Every decision logs: context, arm scores, selected arm, final arm, metadata, policy version
3. **Monitoring dashboard** — Track cumulative regret, arm distribution entropy, reward moving average, feature drift
4. **Alert conditions defined** — Policy collapse (entropy < 0.5), reward degradation (> 2σ below baseline), feature drift
5. **Offline evaluation passed** — New policy tested on historical data using IPS/DR, shows improvement over baseline
6. **Guardrails implemented** — Position limits, stop-loss, circuit breakers for disabled arms
7. **Rollback plan ready** — Fallback to equal weighting or last known good allocation
8. **Cold start strategy** — Define behavior for new arms with no data (equal probability, conservative UCB, etc.)
9. **Propensities logged** — Store π(a|c) for every decision to enable future offline evaluation
10. **Tested on staging** — Run on paper trading or simulation for minimum 20 decisions before live capital

## System Architecture Components

```
┌─────────────┐   ┌──────────────┐   ┌─────────────┐
│ Arm Registry│──▶│ Policy Engine│──▶│ Guardrails  │
└─────────────┘   └──────────────┘   └─────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   ┌──────────┐   ┌───────────┐  ┌──────────┐
   │  Logger  │   │  Reward   │  │ Monitor  │
   │          │   │  Tracker  │  │          │
   └──────────┘   └───────────┘  └──────────┘
```

**Key Interfaces:**
- `ArmRegistry.get_active_arms()` → List[str]
- `PolicyEngine.select_arm(context, arms)` → str
- `Guardrails.validate(arm, context)` → str (may override)
- `Logger.log_decision(...)` → decision_id
- `Logger.log_reward(decision_id, reward)` → None
- `Monitor.update(arm, reward)` → None
- `Monitor.check_alerts()` → List[Alert]

## What to Log (Per Decision)

```json
{
  "timestamp": "ISO-8601",
  "decision_id": "unique-id",
  "policy_version": "version-string",
  "context": {"feature1": value, "feature2": value},
  "active_arms": ["arm1", "arm2"],
  "policy_scores": {"arm1": 0.7, "arm2": 0.3},
  "selected_arm": "arm1",
  "guardrail_override": false,
  "final_arm": "arm1",
  "metadata": {"any": "extra info"}
}
```

**When reward observed:**
```json
{
  "decision_id": "same-unique-id",
  "reward": 0.023,
  "reward_timestamp": "ISO-8601"
}
```

## Monitoring Metrics

| Metric | Formula | Alert Threshold | Meaning |
|--------|---------|----------------|---------|
| **Cumulative Regret** | $\sum_{t=1}^T (\hat{\mu}^* - \hat{\mu}_{a_t})$ | Growing linearly | Consistently picking suboptimal arms |
| **Selection Entropy** | $-\sum_a p_a \log p_a$ | < 0.5 or > 1.5 | Policy collapse (too low) or no learning (too high) |
| **Reward Moving Avg** | MA(rewards, window=20) | < baseline - 2σ | Performance degradation |
| **Feature Drift** | $D_{KL}(P_{recent} \|\| P_{historical})$ | > threshold | Market regime change |
| **Arm Pull Balance** | max/min pull counts | > 10x | Extreme imbalance |

## Offline Evaluation Formulas

**Inverse Propensity Scoring (IPS):**
$$\hat{V}_{IPS}(\pi_1) = \frac{1}{n} \sum_{i=1}^n \frac{\pi_1(a_i|c_i)}{\pi_0(a_i|c_i)} r_i$$

- **Pro:** Unbiased if propensities correct
- **Con:** High variance if policies very different
- **Use when:** Need unbiased estimate, have good propensity scores

**Doubly Robust (DR):**
$$\hat{V}_{DR}(\pi_1) = \frac{1}{n} \sum_{i=1}^n \left[ \sum_a \pi_1(a|c_i) \hat{r}(c_i,a) + \frac{\pi_1(a_i|c_i)}{\pi_0(a_i|c_i)}(r_i - \hat{r}(c_i,a_i)) \right]$$

- **Pro:** Lower variance, correct if either reward model OR propensities correct
- **Con:** Requires building reward model
- **Use when:** Have reward model, want lower variance

**Replay Method:**
$$\hat{V}_{Replay}(\pi_1) = \text{mean}(r_i : \pi_1(c_i) = a_i)$$

- **Pro:** Simple, no statistical corrections needed
- **Con:** Throws away most data if policies differ
- **Use when:** Policies are similar, have lots of data

## Common Failure Modes & Fixes

| Failure Mode | Symptoms | Root Cause | Fix |
|--------------|----------|------------|-----|
| **Policy Collapse** | Entropy < 0.5, same arm 80%+ | Premature convergence, insufficient exploration | Increase ε in ε-greedy, use optimistic initialization |
| **Reward Degradation** | Avg reward declining over time | Non-stationarity, feature drift, arm quality decline | Retrain, add recency weighting, check feature pipeline |
| **Feature Drift** | KL divergence spike | Market regime shift | Add regime detection, retrain with recent data |
| **Cold Start Failure** | New arms never selected | No exploration of unknown arms | Add optimistic initialization for new arms |
| **Guardrail Override Rate > 50%** | Policy constantly overridden | Policy misaligned with business constraints | Retrain with constraints, adjust guardrail thresholds |

## A/B to Bandit Migration Strategy

**Phase 1: Parallel Logging (Week 1-2)**
- Run existing A/B test
- Simulate bandit decisions offline
- Log everything, change nothing

**Phase 2: Hybrid Mode (Week 3-6)**
- 50% traffic: A/B test (for baseline)
- 50% traffic: Bandit
- Monitor both, ensure statistical power

**Phase 3: Bandit with Burn-In (Week 7-10)**
- 100% bandit
- First K decisions: forced exploration (ε=0.3)
- After burn-in: reduce to ε=0.1

**Phase 4: Full Deployment (Week 11+)**
- 100% bandit with normal exploration
- Continuous monitoring
- Monthly offline evaluation of alternative policies

## Production Code Template

```python
class ProductionBanditSystem:
    def __init__(self, policy, guardrails, logger, monitor):
        self.registry = ArmRegistry()
        self.policy = policy
        self.guardrails = guardrails
        self.logger = logger
        self.monitor = monitor

    def make_decision(self, context):
        # Get active arms
        arms = self.registry.get_active_arms()

        # Policy selection
        selected = self.policy.select_arm(context, arms)

        # Guardrail validation
        final = self.guardrails.validate(selected, context)

        # Log decision
        decision_id = self.logger.log_decision(
            context=context,
            policy_scores=self.policy.get_scores(context, arms),
            selected_arm=selected,
            final_arm=final
        )

        return final, decision_id

    def record_reward(self, decision_id, arm, reward):
        # Log reward
        self.logger.log_reward(decision_id, reward)

        # Update policy
        self.policy.update(arm, reward)

        # Update monitor
        self.monitor.update(arm, reward)

        # Check alerts
        if self.monitor.check_alerts():
            self.send_alerts(self.monitor.get_alerts())
```

## Deployment Environments

| Environment | Purpose | Traffic | Monitoring |
|-------------|---------|---------|------------|
| **Local** | Development, unit tests | Synthetic data | None |
| **Staging** | Integration tests, offline eval | Historical replay | Basic |
| **Paper Trading** | Live data, no real execution | 100% simulated | Full dashboard |
| **Canary** | Small % of live traffic | 1-5% real | Intensive (hourly) |
| **Production** | Full deployment | 100% real | Standard (daily) |

## Quick Reference: When to Use What

**Use epsilon-greedy when:** Simple problem, bounded rewards, need explainability
**Use Thompson Sampling when:** Need probabilistic matching, multi-modal rewards
**Use UCB when:** Need theoretical regret bounds, deterministic exploration
**Use LinUCB when:** Have contextual features, linear reward model
**Use Neural bandits when:** Complex non-linear relationships, lots of data

**Use IPS when:** Policies differ moderately, need unbiased estimate
**Use DR when:** Have good reward model, want lower variance
**Use Replay when:** Policies very similar, abundant data

**Alert immediately when:** Entropy < 0.3, reward < baseline - 3σ, all decisions override
**Investigate within 24h when:** Entropy < 0.5, reward < baseline - 2σ, drift detected
**Review weekly:** All other metrics, policy performance, arm distribution

## Further Reading

- **Papers:** Dudík et al. (2011) "Doubly Robust Policy Evaluation", Li et al. (2010) "Contextual Bandits"
- **Industry:** Netflix, Spotify, Microsoft blog posts on production bandit systems
- **Books:** "Bandit Algorithms" (Lattimore & Szepesvári), "Trustworthy Online Controlled Experiments" (Kohavi et al.)
