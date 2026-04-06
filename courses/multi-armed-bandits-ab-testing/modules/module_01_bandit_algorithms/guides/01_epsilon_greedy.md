# Epsilon-Greedy Algorithm

> **Reading time:** ~20 min | **Module:** 01 — Bandit Algorithms | **Prerequisites:** Module 0 Foundations


## In Brief
Epsilon-greedy is the simplest bandit algorithm: with probability ε, explore by choosing a random arm; otherwise, exploit by choosing the arm with the highest estimated reward. It balances exploration and exploitation with a single tunable parameter.

<div class="flow">
<div class="flow-step mint">1. Generate Random</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Explore or Exploit?</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Pull Arm</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Observe Reward</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step rose">5. Update Estimates</div>
</div>


<div class="callout-key">

**Key Concept Summary:** This guide covers the core concepts of epsilon-greedy algorithm, with worked examples and practical implementation guidance.

</div>

## Key Insight
The algorithm forces you to make the exploration-exploitation tradeoff explicit through ε. Too high (ε=0.5) and you waste time on bad arms; too low (ε=0.01) and you might never discover the best arm. The sweet spot depends on your problem.

<div class="callout-insight">

**Insight:** Epsilon-greedy is the simplest bandit algorithm, but its constant exploration rate makes it suboptimal in the long run. Decaying epsilon recovers some of this loss, but UCB and Thompson Sampling handle the exploration-exploitation tradeoff more elegantly.

</div>


## Visual Explanation

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


```
                    ┌─────────────────────────┐
                    │   Select Action         │
                    └───────────┬─────────────┘
                                │
                    Generate random u ~ U(0,1)
                                │
                    ┌───────────┴───────────┐
                    │                       │
                u < ε ?                 u ≥ ε ?
                    │                       │
                    ▼                       ▼
            ┌───────────────┐       ┌──────────────┐
            │   EXPLORE     │       │   EXPLOIT    │
            │               │       │              │
            │ Choose random │       │ Choose arm   │
            │ arm uniformly │       │ with max Q̂   │
            └───────┬───────┘       └──────┬───────┘
                    │                      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Pull selected arm   │
                    │  Observe reward r    │
                    │  Update Q̂(a) ← mean │
                    └──────────────────────┘

ε = 0.1 means: 10% exploration, 90% exploitation
```

**Arm Selection Over Time** (5 arms, arm 3 is best):
```
Time:     0─────────────────────────T
Arm 1:    ▓░░░░░░░░░░░░░░░░░░░░░░░░    (explored early, abandoned)
Arm 2:    ░▓░░░░░░░░░░░░░░░░░░░░░░░    (explored early, abandoned)
Arm 3:    ░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓    (best arm, exploited heavily)
Arm 4:    ░░▓░░░░░░░░░░░░░░░░░░░░░░    (explored, abandoned)
Arm 5:    ░░░░▓░░░░░░░░░░░░░░░░░░░░    (explored, abandoned)

Legend: ▓ = pull, ░ = no pull
```

## Formal Definition

**Algorithm:** Epsilon-Greedy Bandit

**Input:**
- K arms (actions)
- ε ∈ [0, 1] (exploration rate)
- T time steps

**Initialization:**
- Q̂(a) ← 0 for all arms a ∈ {1, ..., K}
- N(a) ← 0 for all arms a (pull counts)

**At each time step t:**

1. **Select action:**
   ```
   a_t = { random arm from {1,...,K}     with probability ε
         { argmax_a Q̂(a)                 with probability 1-ε
   ```

2. **Observe reward:** r_t ~ R(a_t)

3. **Update estimates:**
   ```
   N(a_t) ← N(a_t) + 1
   Q̂(a_t) ← Q̂(a_t) + (r_t - Q̂(a_t))/N(a_t)
   ```

**Expected Regret:**

The cumulative regret after T steps is:

```
E[R_T] = O(ε·T + K·Δ²/ε)
```

Where:
- Δ = difference between best arm and second-best arm
- First term (ε·T): cost of random exploration
- Second term (K·Δ²/ε): cost of insufficient exploration

**Optimal ε:** Balancing both terms gives ε* ≈ O((K/T)^(1/3))

This yields regret O(T^(2/3)), which is suboptimal compared to UCB's O(√T log T).

## Intuitive Explanation

Think of epsilon-greedy as a commodity trader with a simple rule:

**Every week, flip a weighted coin:**
- **Heads (10% chance):** Try a random commodity sector, even one you've never traded
- **Tails (90% chance):** Go all-in on whichever sector has made you the most money so far

**Why this works:**
- The 10% random exploration ensures you eventually try every sector
- The 90% exploitation makes sure you're usually betting on your winner
- Over time, you'll have tried everything enough to know what's best

**Why this fails:**
- If the best sector only beats #2 by a tiny margin, you need more exploration
- If you know one sector dominates, 10% random bets is wasteful
- Fixed ε doesn't adapt—you explore the same at t=10 and t=10,000

**Better approach:** Decay ε over time
- Early: ε = 1 (explore everything)
- Middle: ε = 0.1 (mostly exploit, some exploration)
- Late: ε = 1/√t → 0 (almost pure exploitation)

## Code Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, k_arms, epsilon=0.1):
        self.k = k_arms
        self.epsilon = epsilon
        self.q_estimates = np.zeros(k_arms)  # Estimated values
        self.action_counts = np.zeros(k_arms)  # Pull counts

    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)  # Explore
        else:
            return np.argmax(self.q_estimates)  # Exploit

    def update(self, action, reward):
        self.action_counts[action] += 1
        n = self.action_counts[action]
        self.q_estimates[action] += (reward - self.q_estimates[action]) / n

# Usage
bandit = EpsilonGreedyBandit(k_arms=5, epsilon=0.1)
for t in range(1000):
    action = bandit.select_action()
    reward = get_reward(action)  # Your reward function
    bandit.update(action, reward)
```

</div>
</div>

**Decaying ε variant:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
class DecayingEpsilonGreedy(EpsilonGreedyBandit):
    def __init__(self, k_arms, epsilon_fn=lambda t: 1/np.sqrt(t+1)):
        super().__init__(k_arms, epsilon=1.0)
        self.epsilon_fn = epsilon_fn
        self.t = 0

    def select_action(self):
        self.epsilon = self.epsilon_fn(self.t)
        self.t += 1
        return super().select_action()
```

</div>
</div>

## Common Pitfalls

### 1. Fixed ε Doesn't Adapt
**Problem:** Using ε=0.1 for all T means 10% of your pulls are random forever.

**Impact:** Regret grows linearly in T (ε·T term dominates).

**Fix:** Use decaying ε
```python
epsilon = lambda t: min(1.0, 10 / np.sqrt(t + 1))  # Decays as 1/√t
```

**When to use fixed ε:** Non-stationary environments where the best arm changes over time.

### 2. ε Too High → Wasteful Exploration
**Problem:** ε=0.5 means half your pulls are random, even after you've identified the best arm.

**Symptom:** Regret curve never flattens, keeps growing linearly.

**Fix:**
- Start with ε ∈ [0.05, 0.2] for most problems
- Use decaying ε for stationary problems

### 3. ε Too Low → Premature Convergence
**Problem:** ε=0.01 means you only explore 1% of the time. If you get unlucky early (best arm gives low reward in first few pulls), you might never try it again.

**Symptom:** Regret curve flattens quickly but at a high value (you're stuck exploiting a suboptimal arm).

**Fix:**
- Ensure ε·T ≥ K (explore each arm at least once in expectation)
- For K=10 arms and T=1000 steps, ε ≥ 0.01

### 4. Ignoring Ties
**Problem:** If two arms have identical Q̂ values, `argmax` picks the first one (biases towards lower-indexed arms).

**Fix:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def select_action(self):
    if np.random.random() < self.epsilon:
        return np.random.randint(self.k)
    else:
        # Break ties randomly
        max_q = np.max(self.q_estimates)
        max_actions = np.where(self.q_estimates == max_q)[0]
        return np.random.choice(max_actions)
```

</div>
</div>

### 5. Wrong Update Rule
**Problem:** Using exponential moving average instead of sample mean.

**Correct (sample mean):**
```python
q_new = q_old + (reward - q_old) / n
```

**Incorrect (exponential MA with α=0.1):**
```python
q_new = q_old + 0.1 * (reward - q_old)  # Forgets old data
```

**When EMA is better:** Non-stationary rewards (arm distributions change over time).

## Connections

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


### Builds On
- **Basic probability:** Understanding of expectation, variance
- **Law of large numbers:** Why sample mean → true mean
- **Exploration-exploitation tradeoff:** The fundamental tension in sequential decision-making

### Leads To
- **Upper Confidence Bound (UCB):** Removes the need for ε by using confidence intervals
- **Thompson Sampling:** Bayesian alternative that samples from posterior distributions
- **Decaying exploration schedules:** Theory of how to schedule ε(t)
- **A/B testing:** Epsilon-greedy as a baseline for online experiments

### Related Concepts
- **Multi-armed bandits:** The problem class
- **Regret analysis:** How to measure algorithm performance
- **Stationary vs. non-stationary:** When fixed ε outperforms decaying ε

## Practice Problems

### 1. Conceptual Understanding
**Q:** You run ε-greedy with ε=0.1 for T=10,000 steps on 5 arms. Approximately how many pulls does each arm receive?

**A:** Expected pulls per arm:
- Best arm: ~(0.9 × 10,000) + (0.1 × 10,000 / 5) = 9,200
- Other arms: ~(0.1 × 10,000 / 5) = 200 each

**Q:** Why does ε-greedy have O(T^(2/3)) regret instead of O(√T log T) like UCB?

**A:** The ε·T term (cost of random exploration) dominates. Even with optimal ε* ~ T^(-1/3), we get regret ~ T^(2/3). UCB eliminates random exploration by using confidence bounds, achieving √T.

### 2. Implementation Challenge
Implement ε-greedy with optimistic initialization (start Q̂(a) = 10 instead of 0) and compare to standard initialization. What changes?

**Hint:** Optimistic initialization encourages early exploration—every arm looks good initially, so exploitation tries all arms before settling.

### 3. Real-World Scenario
You're allocating capital across 5 commodity sectors (Energy, Metals, Agriculture, Livestock, Softs). You have T=252 trading days (1 year). How would you set ε?

**Suggested approach:**
```python

# Conservative: explore ~10% of the time early, decay to 1% by end
epsilon = lambda t: max(0.01, 0.2 * (1 - t/252))

# Aggressive: pure exploration for 50 days, then pure exploitation
epsilon = lambda t: 1.0 if t < 50 else 0.0

# Balanced: 1/√t decay
epsilon = lambda t: min(0.2, 10 / np.sqrt(t + 1))
```

Test all three and see which achieves lowest regret on your data.

### 4. Debugging Exercise
Your ε-greedy algorithm keeps selecting the same arm over and over, even though another arm has higher average reward. What could be wrong?

**Possible causes:**
- ε too low (not exploring enough)
- Unlucky early samples made the best arm look bad
- Ties in Q̂ values biasing towards first arm (see Pitfall #4)
- Bug in update rule (not incrementing N(a) correctly)

### 5. Extension
Modify ε-greedy to use weighted random exploration instead of uniform random. Arms with higher uncertainty (fewer pulls) should be explored more often.

```python
def select_action(self):
    if np.random.random() < self.epsilon:
        # Weight by 1/sqrt(N(a)+1) — less-pulled arms more likely
        weights = 1 / np.sqrt(self.action_counts + 1)
        probs = weights / weights.sum()
        return np.random.choice(self.k, p=probs)
    else:
        return np.argmax(self.q_estimates)
```

How does this compare to standard ε-greedy and UCB1?


---

## Cross-References

<a class="link-card" href="./02_upper_confidence_bound.md">
  <div class="link-card-title">02 Upper Confidence Bound</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_upper_confidence_bound.md">
  <div class="link-card-title">02 Upper Confidence Bound — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_softmax_boltzmann.md">
  <div class="link-card-title">03 Softmax Boltzmann</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_softmax_boltzmann.md">
  <div class="link-card-title">03 Softmax Boltzmann — Companion Slides</div>
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

