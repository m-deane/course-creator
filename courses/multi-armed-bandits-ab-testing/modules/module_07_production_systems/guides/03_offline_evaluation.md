# Offline Evaluation

> **Reading time:** ~15 min | **Module:** 07 — Production Systems | **Prerequisites:** Module 6


## In Brief


<div class="callout-key">

**Key Concept Summary:** Offline evaluation lets you test new bandit policies on historical data without deploying them to production, using techniques like inverse propensity scoring and doubly-robust estimation to correc...

</div>

Offline evaluation lets you test new bandit policies on historical data without deploying them to production, using techniques like inverse propensity scoring and doubly-robust estimation to correct for the bias in logged decisions.

> 💡 **Key Insight:** You can't just replay logged decisions with a new policy and compare rewards — the historical policy chose those arms for a reason (they looked good at the time). This creates selection bias. If your old policy never tried arm X, you have no data about its rewards. Offline evaluation methods mathematically correct for this bias using the probabilities of the original policy's choices.

## Visual Explanation

```
┌────────────────────────────────────────────────────────────────┐
│                  THE OFFLINE EVALUATION PROBLEM                 │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Historical Logged Data (from policy π₀):                      │
│  ┌──────────────────────────────────────────┐                  │
│  │ Context  │ Chosen Arm │ Prob  │ Reward  │                  │
│  ├──────────────────────────────────────────┤                  │
│  │ VIX=20   │ GOLD       │ 0.7   │ 0.02    │ ← π₀ liked GOLD  │
│  │ VIX=25   │ GOLD       │ 0.8   │ 0.01    │                  │
│  │ VIX=18   │ OIL        │ 0.3   │ 0.03    │ ← rare choice    │
│  │ VIX=22   │ GOLD       │ 0.75  │ 0.015   │                  │
│  └──────────────────────────────────────────┘                  │
│                                                                  │
│  New policy π₁ wants to evaluate itself on this data.          │
│  Problem: π₁ might prefer OIL, but we only have 1 OIL sample!  │
│                                                                  │
│  ❌ WRONG: Average the rewards where π₁ agrees with π₀         │
│     → Biased toward π₀'s preferences                            │
│                                                                  │
│  ✓ CORRECT: Weight each sample by π₁(a|c) / π₀(a|c)           │
│     → Inverse Propensity Scoring (IPS)                          │
│                                                                  │
└────────────────────────────────────────────────────────────────┘

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>


┌────────────────────────────────────────────────────────────────┐
│              OFFLINE EVALUATION METHODS                         │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. INVERSE PROPENSITY SCORING (IPS)                           │
│                                                                  │
│     V̂(π₁) = (1/n) Σ [π₁(aᵢ|cᵢ) / π₀(aᵢ|cᵢ)] · rᵢ            │
│                                                                  │
│     Intuition: Up-weight rare decisions that π₁ prefers        │
│     Downside: High variance if π₀(a|c) is very small          │
│                                                                  │
│  2. DOUBLY ROBUST (DR)                                          │
│                                                                  │
│     V̂(π₁) = (1/n) Σ [r̂(cᵢ,a) + (rᵢ - r̂(cᵢ,aᵢ)) ·            │
│                        π₁(aᵢ|cᵢ) / π₀(aᵢ|cᵢ)]                  │
│                                                                  │
│     Intuition: Use reward model r̂ as baseline, correct with IPS│
│     Benefit: Lower variance, works if either r̂ or IPS accurate │
│                                                                  │
│  3. REPLAY METHOD                                               │
│                                                                  │
│     Only use samples where π₁(c) = a (new policy agrees)       │
│     V̂(π₁) = Average reward on matching samples                 │
│                                                                  │
│     Intuition: Conservative, unbiased, but wastes data          │
│     Use when: π₁ is similar to π₀ (lots of agreement)          │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```

## Formal Definition

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


**Inverse Propensity Scoring (IPS):**

Given logged data $\mathcal{D} = \{(c_i, a_i, r_i, \pi_0(a_i|c_i))\}_{i=1}^n$ from policy $\pi_0$, the IPS estimator for a new policy $\pi_1$ is:

$$\hat{V}_{IPS}(\pi_1) = \frac{1}{n} \sum_{i=1}^n \frac{\pi_1(a_i|c_i)}{\pi_0(a_i|c_i)} r_i$$

This is **unbiased**: $\mathbb{E}[\hat{V}_{IPS}(\pi_1)] = V(\pi_1)$ when $\pi_0(a|c) > 0$ for all arms $a$ that $\pi_1$ might choose.

**Doubly Robust (DR) Estimator:**

$$\hat{V}_{DR}(\pi_1) = \frac{1}{n} \sum_{i=1}^n \left[ \sum_{a \in \mathcal{A}} \pi_1(a|c_i) \hat{r}(c_i, a) + \frac{\pi_1(a_i|c_i)}{\pi_0(a_i|c_i)} (r_i - \hat{r}(c_i, a_i)) \right]$$

Where $\hat{r}(c, a)$ is a learned reward model. The DR estimator is unbiased if **either** $\hat{r}$ is correct OR the propensity weights are correct (hence "doubly robust").

**Replay Method:**

$$\hat{V}_{Replay}(\pi_1) = \frac{1}{|\mathcal{D}_{\text{match}}|} \sum_{(c,a,r) \in \mathcal{D}_{\text{match}}} r$$

Where $\mathcal{D}_{\text{match}} = \{(c_i, a_i, r_i) : \pi_1(c_i) = a_i\}$ (new policy agrees with historical decision).

## Intuitive Explanation

Imagine you're a new portfolio manager (policy $\pi_1$) and you want to prove you're better than the previous manager (policy $\pi_0$), but you can't trade yet — you only have their historical trades.

**Naive approach:** "Look at the trades where I would have made the same decision. My average return would have been X%."

**Problem:** The previous manager's decisions were biased toward certain assets. If you prefer different assets, you have less data about them.

**IPS solution:** "I'll use all the historical trades, but weight them by how much I like that decision relative to how much they liked it. If they rarely picked OIL (10% chance) but I would pick it often (50% chance), I'll count that OIL trade 5x more (50%/10%)."

**DR solution:** "I'll build a model to predict returns for all assets. Then I'll use that as a baseline and correct it with IPS only for the differences. If the model is good, I need less data. If the model is wrong, IPS still keeps me unbiased."

**Replay method:** "I'll be conservative and only count trades where we agree. Smaller sample size, but no statistical tricks needed."

## Code Implementation


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
from typing import List, Dict, Callable

class OfflineEvaluator:
    """Offline policy evaluation using logged bandit data."""

    def __init__(self, logged_data: List[Dict]):
        """
        logged_data: list of dicts with keys:
            - 'context': feature vector
            - 'action': chosen arm
            - 'reward': observed reward
            - 'propensity': π₀(action|context)
        """
        self.data = logged_data

    def ips_estimate(self, new_policy: Callable) -> float:
        """Inverse Propensity Scoring estimate."""
        total = 0.0
        for record in self.data:
            context = record['context']
            action = record['action']
            reward = record['reward']
            old_prob = record['propensity']

            # Get new policy's probability for this action
            new_prob = new_policy.get_probability(context, action)

            # IPS weight
            weight = new_prob / (old_prob + 1e-10)
            total += weight * reward

        return total / len(self.data)

    def replay_estimate(self, new_policy: Callable) -> float:
        """Replay method: only use matching decisions."""
        matching_rewards = []

        for record in self.data:
            context = record['context']
            action = record['action']
            reward = record['reward']

            # Check if new policy agrees
            new_action = new_policy.select_arm(context)
            if new_action == action:
                matching_rewards.append(reward)

        if not matching_rewards:
            return np.nan  # No matching samples

        return np.mean(matching_rewards)

    def doubly_robust_estimate(self, new_policy: Callable,
                               reward_model: Callable) -> float:
        """Doubly robust estimator with learned reward model."""
        total = 0.0

        for record in self.data:
            context = record['context']
            action = record['action']
            reward = record['reward']
            old_prob = record['propensity']

            # Direct method component (reward model predictions)
            direct_value = 0.0
            for arm in new_policy.get_arms():
                arm_prob = new_policy.get_probability(context, arm)
                predicted_reward = reward_model.predict(context, arm)
                direct_value += arm_prob * predicted_reward

            # IPS correction
            new_prob = new_policy.get_probability(context, action)
            predicted_reward = reward_model.predict(context, action)
            ips_correction = (new_prob / (old_prob + 1e-10)) * \
                            (reward - predicted_reward)

            total += direct_value + ips_correction

        return total / len(self.data)
```

</div>

**Commodity Application:**

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Evaluate new commodity allocation policy on historical data

# Historical data from Thompson Sampling policy
logged_data = [
    {'context': {'vix': 20, 'momentum': 0.05},
     'action': 'GOLD',
     'reward': 0.02,
     'propensity': 0.7},  # Old policy heavily favored GOLD
    {'context': {'vix': 25, 'momentum': -0.02},
     'action': 'GOLD',
     'reward': 0.01,
     'propensity': 0.8},
    {'context': {'vix': 18, 'momentum': 0.08},
     'action': 'OIL',
     'reward': 0.03,
     'propensity': 0.3},  # Rare OIL selection
    # ... more weeks
]

# New policy: LinUCB contextual bandit
new_policy = LinUCBPolicy()

# Offline evaluation
evaluator = OfflineEvaluator(logged_data)

ips_value = evaluator.ips_estimate(new_policy)
replay_value = evaluator.replay_estimate(new_policy)

print(f"IPS estimate: {ips_value:.4f}")
print(f"Replay estimate: {replay_value:.4f}")

# Doubly robust (requires reward model)
reward_model = LinearRegressionRewardModel()
dr_value = evaluator.doubly_robust_estimate(new_policy, reward_model)
print(f"DR estimate: {dr_value:.4f}")
```

</div>

## Common Pitfalls

**Pitfall 1: Missing propensity scores**
IPS requires knowing $\pi_0(a|c)$ — the probability the old policy assigned to each action. If you didn't log this, you can't do IPS.

**Solution:** Always log policy probabilities/scores with decisions. For deterministic policies, log which arm would have been chosen with probability 1.

**Pitfall 2: Zero propensities**
If $\pi_0(a|c) = 0$ (old policy never picks arm $a$ in context $c$), but $\pi_1(a|c) > 0$, the IPS weight is infinite.

**Solution:** Use epsilon-greedy logging policy that explores all arms with some minimum probability. Or clip propensities to $[\epsilon, 1]$.

**Pitfall 3: High variance with IPS**
When propensities are very different ($\pi_1$ strongly prefers arms $\pi_0$ rarely chose), IPS weights are huge and variance explodes.

**Solution:** Use doubly robust estimator (lower variance) or ensure logging policy explores sufficiently.

**Pitfall 4: Trusting replay method with small overlap**
If new policy is very different from old policy, replay method discards most data and becomes unreliable.

**Solution:** Check sample size. If $|\mathcal{D}_{\text{match}}| < 30$, don't trust replay estimate.

## Connections

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


**Builds on:**
- Module 0: Expected value and regret calculations
- Logging and monitoring (you need logged propensities)
- Contextual bandits (policy probabilities depend on context)

**Leads to:**
- Safe deployment strategies (validate offline before deploying)
- Continuous policy improvement (offline eval → deploy → log → repeat)
- Counterfactual reasoning in decision systems

**Related concepts:**
- Causal inference: propensity score matching
- Off-policy learning in reinforcement learning
- A/B testing: offline eval can replace some A/B tests

## Practice Problems

1. **Conceptual:** You have 1000 logged decisions from an epsilon-greedy policy with ε=0.1 (90% greedy, 10% random). You want to evaluate a new greedy policy. Will IPS work well? Why or why not?

2. **Implementation:** Implement a function that computes 95% confidence intervals for IPS estimates using bootstrap resampling. (Hint: resample logged data with replacement, compute IPS estimate, repeat 1000 times, take percentiles.)

3. **Real Scenario:** Your logged data shows:
   - GOLD: selected 80% of time, avg reward 1.5%
   - OIL: selected 20% of time, avg reward 2.5%

   New policy would select:
   - GOLD: 30% of time
   - OIL: 70% of time

   Compute the IPS estimate of the new policy's expected reward. Is it higher than the old policy's observed 1.7% average?

4. **Design Challenge:** You're building an offline evaluation pipeline for commodity allocation. What data do you need to log during production to enable IPS, doubly robust, and replay methods? Design the schema.

5. **Code Review:** What's wrong with this IPS implementation?

<span class="filename">example.py</span>
</div>

   <div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
   def ips_estimate(data, new_policy):
       total = 0
       for d in data:
           if new_policy.select(d['context']) == d['action']:
               total += d['reward']
       return total / len(data)
   ```

</div>
   Fix it to correctly implement IPS.


---

## Cross-References

<a class="link-card" href="./01_bandit_system_architecture.md">
  <div class="link-card-title">01 Bandit System Architecture</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_bandit_system_architecture.md">
  <div class="link-card-title">01 Bandit System Architecture — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_logging_and_monitoring.md">
  <div class="link-card-title">02 Logging And Monitoring</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_logging_and_monitoring.md">
  <div class="link-card-title">02 Logging And Monitoring — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./cheatsheet.md">
  <div class="link-card-title">Cheatsheet</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./cheatsheet.md">
  <div class="link-card-title">Cheatsheet — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

