# Restless Bandits

> **Reading time:** ~20 min | **Module:** 06 — Advanced Topics | **Prerequisites:** Module 5


## In Brief


<div class="callout-key">

**Key Concept Summary:** Restless bandits are arms that evolve over time *even when you don't select them*. Unlike standard bandits where unselected arms stay frozen, restless arms change on their own — like commodities whose

</div>

Restless bandits are arms that evolve over time *even when you don't select them*. Unlike standard bandits where unselected arms stay frozen, restless arms change on their own — like commodities whose volatility and correlation shift whether you're invested or not.

## Key Insight

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>


**Standard bandits assume passive arms:** If you don't pull an arm, its reward distribution stays the same. But in reality:
- A commodity's volatility changes even if you're not trading it
- An A/B test variant's conversion rate may drift due to external factors
- A trading strategy's performance decays as the market adapts to it

**The challenge:** You can only observe one arm per period, but ALL arms are evolving. Do you keep observing the current best arm, or switch to check on others before they drift too far?

**Restless bandits are PSPACE-hard** (computationally intractable in general), but practical approximations exist for real-world use.

## Visual Explanation

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


```
Standard Bandit (arms freeze when not selected):
           ┌─ Selected ─┐
Arm A:  ───●────●────●────●────●───  (only evolves when selected)
Arm B:  ───○─────────────────────  (frozen - no selection, no change)
Arm C:  ─────────○────○───────○──  (frozen except when selected)

Restless Bandit (all arms evolve continuously):
           ┌─ Selected ─┐
Arm A:  ~~~●~~~~●~~~~●~~~~●~~~~●~~~  (evolves continuously)
Arm B:  ~~~○~~~~~~~~~~~~~~~~~~~~~  (evolves even without selection!)
Arm C:  ~~~~~~~~~○~~~~○~~~~~~~○~~  (evolves continuously)
                 ↑
         All arms are "alive" - they drift, decay, or improve
         independently of your actions
```

**Commodity Example:**

```
Month:        Jan   Feb   Mar   Apr   May   Jun
Portfolio:    [WTI] [WTI] [WTI] [WTI] [WTI] [WTI]  ← You stay in WTI

WTI Vol:       12%   13%   14%   15%   18%   22%   ← Volatility increasing
NATGAS Vol:    25%   24%   23%   20%   18%   16%   ← Volatility decreasing
                                                      (but you don't know,
                                                       you're not watching!)

Problem: By June, NATGAS became less risky than WTI, but you missed it
         because you only observed WTI's evolution.
```

## Formal Definition

### Restless Multi-Armed Bandit (RMAB)

**State-based formulation:**
- Each arm `i` has a hidden state `s_i(t) ∈ S_i` at time `t`
- States evolve via Markov transition: `s_i(t+1) ~ P_i(·|s_i(t), a_t)`
  - `a_t ∈ {1, ..., K}` is the arm selected at time `t`
  - **Key:** Transition happens for ALL arms, not just selected one
- Reward from arm `i`: `r_i(t) ~ R_i(s_i(t))`
- You only observe reward from the selected arm: `r_{a_t}(t)`

**Objective:** Maximize cumulative reward `Σ_t r_{a_t}(t)`

**Constraint:** Select exactly one arm per period (single-action constraint)

### Whittle Index Policy (Practical Approximation)

The Whittle index is a heuristic that assigns each arm a priority score based on its current state. At each time `t`:

```
For each arm i:
    Compute Whittle index W_i(s_i(t))

Select arm: argmax_i W_i(s_i(t))
```

**Intuition:** `W_i(s)` represents the "subsidy" needed to make you indifferent between selecting arm `i` in state `s` vs not selecting it. Higher index = more urgent to select.

**Computing Whittle index** requires solving a dynamic program per arm (see additional readings for details).

### Simplified Practical Approach

For commodity applications, use a greedy approximation:

```
Score_i(t) = E[reward | current belief about arm i]
              - λ · time_since_last_observation_i

Where λ > 0 penalizes arms you haven't checked recently
```

This balances exploitation (high expected reward) with information gathering (check neglected arms).

## Intuitive Explanation

**It's like maintaining friendships that evolve without you.**

Imagine you have 5 friends. You can only hang out with one friend per weekend. Each friend's life changes over time — new jobs, relationships, interests — *whether you see them or not*.

**Standard bandit thinking:** "I'll keep hanging out with my favorite friend. The others will be the same if I ever go back to them."

**Restless bandit reality:** "If I ignore Friend B for 6 months, they might have changed completely. I need to occasionally check in, even if Friend A is more fun right now."

**The trade-off:**
- Stay with the current best? → Risk missing that others improved
- Rotate to check on others? → Give up immediate rewards

Whittle index balances these: "Friend B's life is changing rapidly (high volatility), so I should check on them soon even though Friend A is currently better."

## Code Implementation


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
from collections import defaultdict

class GreedyRestlessBandit:
    """
    Simplified restless bandit using recency-weighted greedy selection.
    Suitable for commodity allocation where volatility/correlation drift.
    """
    def __init__(self, n_arms, recency_penalty=0.01):
        """
        Args:
            n_arms: Number of arms
            recency_penalty: Penalty per period since last observation (λ)
        """
        self.n_arms = n_arms
        self.recency_penalty = recency_penalty

        # Maintain running statistics per arm
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)
        self.last_observed = np.zeros(n_arms)  # Time of last observation
        self.t = 0

    def select_arm(self):
        self.t += 1

        scores = []
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                scores.append(float('inf'))  # Force initial exploration
            else:
                # Expected reward based on history
                expected_reward = self.sums[i] / self.counts[i]

                # Penalty for staleness (haven't observed recently)
                staleness = self.t - self.last_observed[i]
                penalty = self.recency_penalty * staleness

                # Higher score = more urgent to select
                scores.append(expected_reward - penalty)

        return np.argmax(scores)

    def update(self, arm, reward):
        """Update statistics for the OBSERVED arm."""
        self.counts[arm] += 1
        self.sums[arm] += reward
        self.last_observed[arm] = self.t

        # Note: Unobserved arms don't get updated, but their staleness increases

# Example usage
bandit = GreedyRestlessBandit(n_arms=4, recency_penalty=0.01)

for t in range(1000):
    arm = bandit.select_arm()

    # Simulate restless rewards: all arms evolving over time
    # (but you only observe the selected arm)
    true_means = get_time_varying_means(t)  # All arms drift
    reward = np.random.normal(true_means[arm], 0.1)

    bandit.update(arm, reward)
```

</div>
</div>

**Advanced: Discounted Restless Bandit**

Combine recency penalty with exponential discounting:


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
class DiscountedRestlessBandit:
    def __init__(self, n_arms, gamma=0.95, recency_penalty=0.01):
        self.n_arms = n_arms
        self.gamma = gamma
        self.recency_penalty = recency_penalty

        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
        self.last_observed = np.zeros(n_arms)
        self.t = 0

    def select_arm(self):
        self.t += 1

        # Discount all arms (they're all evolving)
        self.alpha *= self.gamma
        self.beta *= self.gamma

        scores = []
        for i in range(self.n_arms):
            # Thompson sampling score
            theta_i = np.random.beta(self.alpha[i], self.beta[i])

            # Recency penalty
            staleness = self.t - self.last_observed[i]
            penalty = self.recency_penalty * staleness

            scores.append(theta_i - penalty)

        return np.argmax(scores)

    def update(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)
        self.last_observed[arm] = self.t
```

</div>
</div>

## Common Pitfalls

### Pitfall 1: Ignoring Restlessness
**What happens:** You stay with the current best arm too long, missing that others improved.

**Example:** In 2020, WTI crude had the best Sharpe ratio pre-COVID. By June 2020, natural gas and agricultural commodities had better risk-adjusted returns, but if you only tracked WTI, you missed the shift.

**Fix:** Periodically sample ALL arms to update beliefs, even if you don't allocate to them. Use recency penalties or forced exploration schedules.

### Pitfall 2: Recency Penalty Too High
**What happens:** You rotate arms too frequently, never exploiting the best one.

**Example:** With λ=0.1 and daily observations, after just 10 days without observing an arm, the penalty outweighs a 1% mean difference. You end up in perpetual exploration.

**Fix:** Calibrate λ to your environment:
- Slow-moving commodities (metals): λ ≈ 0.001-0.01
- Fast-moving commodities (energy during volatility): λ ≈ 0.01-0.05

### Pitfall 3: Treating Restless as Standard Non-Stationary
**What happens:** You use discounted Thompson Sampling, which only updates the selected arm. Unobserved arms keep their old beliefs.

**Example:** Arm A was great 200 periods ago, you switched to Arm B. Arm A has since collapsed, but your discounted beliefs still think it's good (because you haven't observed its collapse).

**Fix:** Apply discounting to ALL arms, not just the selected one (as in DiscountedRestlessBandit above). Or periodically force observation of all arms.

### Pitfall 4: Not Modeling the Restlessness
**What happens:** You apply restless bandit methods when arms are actually passive, wasting effort on unnecessary exploration.

**Example:** A/B testing website variants — the variants don't change unless you manually update them. No need for restless bandits.

**Fix:** Ask: "Do unselected options evolve on their own?"
- YES: Commodity volatility, market conditions → use restless bandits
- NO: Static website variants, fixed drug dosages → use standard bandits

## Connections

### Builds On
- **Non-Stationary Bandits (Module 6.1):** Restless bandits extend non-stationarity to all arms simultaneously
- **Thompson Sampling (Module 2):** Restless variants modify TS to account for unobserved evolution
- **Markov Decision Processes:** Restless bandits are constrained MDPs

### Leads To
- **Multi-Agent Systems:** When multiple learners interact with restless environments
- **Resource Allocation:** Restless bandits model scheduling and allocation under constraints
- **Passive vs Active Arms:** Formalizing when to use restless vs standard bandits

### Related Concepts
- **Kalman Filtering:** Track hidden states of all arms simultaneously
- **Whittle Index Theory:** Optimal indexability conditions (see additional readings)
- **Constrained Optimization:** Restless bandits as Lagrangian relaxation problems

## Practice Problems

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


### Problem 1: When to Use Restless Bandits
For each scenario, decide if restless bandits are appropriate:

a) Allocating between 5 commodity futures (volatility and correlation change daily)
b) A/B testing 3 static website headlines
c) Choosing between advertising channels (channels' effectiveness decays as audience saturates)
d) Selecting crop types to plant (soil quality changes whether you plant or not)

### Problem 2: Calibrating Recency Penalty
You observe that a commodity's volatility regime typically persists for 30 days. You want to re-check unobserved commodities before their regimes are likely to have changed.

What recency penalty λ would force re-observation within ~20 days if the expected reward difference is 0.02?

**Hint:** Set `λ · 20 ≈ 0.02` to make staleness penalty equal to reward difference after 20 days.

### Problem 3: Forced Exploration Schedule
Design a hybrid policy:
- Use greedy selection (highest expected reward) 80% of the time
- Use random exploration (uniform over unobserved arms) 20% of the time

How does this differ from recency-penalized greedy? What are the trade-offs?

### Problem 4: Whittle Index Intuition
Suppose Arm A has high current reward but low volatility (stable), while Arm B has medium current reward but high volatility (changing fast).

Which arm should have a higher Whittle index (higher priority)? Why?

**Hint:** Think about information decay. High volatility = information goes stale fast.

## Commodity Application Example

**Scenario:** Portfolio allocation across 5 commodities: WTI crude, natural gas, gold, copper, corn. Each commodity's volatility and correlation with others evolves daily based on:
- Macroeconomic releases (inflation data → gold volatility spikes)
- Supply shocks (pipeline outages → natural gas volatility spikes)
- Seasonal patterns (harvest season → corn volatility drops)

**Challenge:** You can only hold ONE commodity at a time (single-action constraint). But all commodities' risk profiles are changing, whether you hold them or not.

**Restless Bandit Solution:**


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Initialize with recency penalty
bandit = GreedyRestlessBandit(n_arms=5, recency_penalty=0.02)

commodities = ['WTI', 'NATGAS', 'GOLD', 'COPPER', 'CORN']
returns_data = load_daily_returns(commodities)

for day in trading_days:
    # Select commodity
    arm = bandit.select_arm()
    selected_commodity = commodities[arm]

    # Observe return (only for selected commodity)
    # Normalize to [0, 1] for reward signal
    reward = normalize_sharpe(returns_data[selected_commodity][day])

    bandit.update(arm, reward)

    # Allocate capital
    allocate(selected_commodity, capital)

    # Note: Other commodities' volatilities changed today,
    # but we didn't observe them. Recency penalty will eventually
    # force us to check back.
```

</div>
</div>

**Key insight:** The recency penalty `λ=0.02` means that after ~50 days without observing a commodity, the staleness penalty (0.02 × 50 = 1.0) becomes very large, forcing re-observation even if it was historically bad. This prevents the "ignore and forget" problem where an improving commodity is never reconsidered.

**Backtesting:** Compare restless bandit vs standard Thompson Sampling on historical data with known regime changes (e.g., 2020 COVID crash, 2022 energy crisis). Restless bandit should recover faster after unobserved commodities become attractive.


---

## Cross-References

<a class="link-card" href="./01_non_stationary_bandits.md">
  <div class="link-card-title">01 Non Stationary Bandits</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_non_stationary_bandits.md">
  <div class="link-card-title">01 Non Stationary Bandits — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_adversarial_bandits.md">
  <div class="link-card-title">03 Adversarial Bandits</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_adversarial_bandits.md">
  <div class="link-card-title">03 Adversarial Bandits — Companion Slides</div>
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

