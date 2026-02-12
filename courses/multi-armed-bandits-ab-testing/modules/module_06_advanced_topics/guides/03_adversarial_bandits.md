# Adversarial Bandits

## In Brief

Adversarial bandits assume rewards are chosen by an adversary who knows your algorithm and tries to minimize your payoff. Unlike stochastic bandits (rewards drawn from fixed distributions), adversarial settings require game-theoretic approaches that randomize arm selection to avoid exploitation.

## Key Insight

**Stochastic assumption:** Each arm has a fixed (or slowly drifting) reward distribution. Your job is to learn which is best.

**Adversarial reality:** Rewards can be chosen arbitrarily each round. If your algorithm is predictable, an adversary (or adaptive market) can exploit it.

**Example:** You discover a profitable commodity trading strategy. As you deploy it, other traders notice and front-run you, or liquidity providers widen spreads when they see your pattern. Your "arm" (strategy) becomes worse *because you're selecting it*.

**The solution:** Randomize your actions using the EXP3 algorithm (Exponential-weight algorithm for Exploration and Exploitation). By being unpredictable, you prevent adversaries from adapting against you.

## Visual Explanation

```
Stochastic Bandit (rewards from fixed distributions):
         ┌─ True mean μ=0.6 ─┐
Arm A:   ○─○──○───○─○───○──○  (i.i.d. draws around 0.6)
         ┌─ True mean μ=0.4 ─┐
Arm B:   ○──○─○────○──○─○──○  (i.i.d. draws around 0.4)

         Your algorithm learns: A > B, always select A
         Works great!

Adversarial Bandit (rewards chosen to hurt you):
                    ┌ You selected A recently ┐
Arm A:   ●─●──●───●─○───○──○──○  (adversary makes A bad when you pick it)
Arm B:   ○──○─○────●──●─●──●  (adversary makes B good when you avoid it)

         If you always select A → adversary makes A terrible
         If you switch to B → adversary makes B terrible
         Solution: RANDOMIZE to be unpredictable
```

**Market Impact Example:**

```
Day 1-5:   You buy WTI crude every morning at open
           → Spread is tight: $70.00 bid / $70.05 ask

Day 6-10:  Market makers notice your pattern
           → They front-run: $70.00 bid / $70.20 ask (wider spread)
           → You pay 15 cents more per barrel

Your predictable strategy became exploitable.
EXP3 solution: Randomize buy times, sometimes skip days, mix in other commodities.
```

## Formal Definition

### Adversarial Bandit Setup

**Each round `t = 1, 2, ..., T`:**
1. You select arm `i_t ∈ {1, ..., K}` (possibly randomly)
2. Adversary simultaneously assigns rewards `r_1(t), ..., r_K(t) ∈ [0, 1]` to all arms
   - Adversary can see your past actions and algorithm, but not current random choice
3. You observe reward `r_{i_t}(t)` for the selected arm only
4. You update your strategy

**Objective:** Minimize regret against the best fixed arm in hindsight:

```
Regret(T) = max_i Σ_t r_i(t) - Σ_t r_{i_t}(t)
```

**Key difference from stochastic:** No assumption that `r_i(t)` is drawn from a fixed distribution. Rewards can be arbitrary, even adversarially chosen.

### EXP3 Algorithm (Exponential-weight algorithm for Exploration and Exploitation)

Maintain weights `w_i(t)` for each arm, updated via multiplicative rule:

**Initialization:**
```
w_i(1) = 1  for all i
```

**Each round `t`:**

1. **Compute probabilities:**
   ```
   p_i(t) = (1 - γ) · w_i(t) / Σ_j w_j(t)  +  γ / K

   where γ ∈ (0, 1] is exploration parameter
   ```

2. **Sample arm:** `i_t ~ p(t)` (randomly according to probabilities)

3. **Observe reward:** `r_{i_t}(t)`

4. **Update weights:**
   ```
   Estimated reward: r̂_i(t) = r_i(t) / p_i(t)  if i = i_t, else 0

   w_i(t+1) = w_i(t) · exp(γ · r̂_i(t) / K)
   ```

**Regret bound:** With `γ = sqrt(K ln K / T)`, EXP3 achieves:
```
E[Regret(T)] ≤ 2 sqrt(T K ln K)
```

This is optimal for adversarial bandits (no algorithm can do better in worst case).

## Intuitive Explanation

**It's like rock-paper-scissors against a mind reader.**

If you always choose rock, your opponent will choose paper. If you always alternate rock-scissors-rock-scissors, they'll predict and beat you.

**The solution:** Randomize. Choose each option with probability 1/3. Your opponent can't exploit you because even *you* don't know what you'll choose next.

**EXP3 does this for bandits:**
- Maintain a "weight" for each arm (higher weight = better historical performance)
- Convert weights to probabilities (with forced exploration via γ)
- Randomly sample an arm according to probabilities
- Update weights based on observed reward

**Why inverse probability weighting (`r̂ = r / p`)?**

You only observe the reward of the selected arm. To estimate the expected reward of an arm, you need to account for selection bias:
- If you select an arm with probability `p=0.1`, that single observation carries 10× information
- If you select with `p=0.9`, the observation is less surprising

Dividing by `p` corrects this bias (importance sampling).

## Code Implementation

```python
import numpy as np

class EXP3:
    """
    EXP3 algorithm for adversarial multi-armed bandits.
    Randomizes arm selection to avoid exploitation.
    """
    def __init__(self, n_arms, gamma=None):
        """
        Args:
            n_arms: Number of arms (K)
            gamma: Exploration parameter. If None, uses sqrt(K ln K / T) bound
                   (requires setting T later via set_horizon)
        """
        self.n_arms = n_arms
        self.gamma = gamma
        self.weights = np.ones(n_arms)
        self.t = 0

    def set_horizon(self, T):
        """Set time horizon to compute optimal gamma."""
        if self.gamma is None:
            K = self.n_arms
            self.gamma = min(1.0, np.sqrt(K * np.log(K) / T))

    def get_probabilities(self):
        """Compute selection probabilities from weights."""
        K = self.n_arms
        gamma = self.gamma if self.gamma else 0.1  # Default if not set

        # Normalize weights to probabilities
        total_weight = np.sum(self.weights)
        probs = (1 - gamma) * (self.weights / total_weight) + gamma / K

        return probs

    def select_arm(self):
        """Sample arm according to current probabilities."""
        self.t += 1
        probs = self.get_probabilities()
        arm = np.random.choice(self.n_arms, p=probs)
        return arm

    def update(self, arm, reward):
        """
        Update weights based on observed reward.

        Args:
            arm: Selected arm
            reward: Observed reward in [0, 1]
        """
        probs = self.get_probabilities()

        # Inverse probability weighting (importance sampling)
        estimated_reward = reward / probs[arm]

        # Multiplicative weight update
        K = self.n_arms
        gamma = self.gamma if self.gamma else 0.1
        self.weights[arm] *= np.exp(gamma * estimated_reward / K)

# Example usage
bandit = EXP3(n_arms=3)
bandit.set_horizon(T=1000)  # Set time horizon for optimal gamma

for t in range(1000):
    arm = bandit.select_arm()

    # Adversarial reward function (could depend on your history)
    reward = adversarial_reward(arm, history)

    bandit.update(arm, reward)
```

**EXP3 with Bounded Rewards:**

The above assumes rewards in `[0, 1]`. For arbitrary ranges `[r_min, r_max]`:

```python
def normalize_reward(reward, r_min, r_max):
    """Normalize reward to [0, 1]."""
    return (reward - r_min) / (r_max - r_min)

# In update:
normalized_reward = normalize_reward(reward, r_min, r_max)
bandit.update(arm, normalized_reward)
```

## Common Pitfalls

### Pitfall 1: Using Stochastic Algorithms in Adversarial Settings
**What happens:** Thompson Sampling or UCB get exploited by adaptive adversaries.

**Example:** You use UCB for trading. A competitor reverse-engineers your strategy (UCB is deterministic given observations) and front-runs your predictable moves.

**Why it fails:** Deterministic algorithms are exploitable. Adversary can choose rewards to make your selected arm perform badly.

**Fix:** Use EXP3 or other randomized algorithms when facing adaptive opponents.

### Pitfall 2: Gamma Too Small
**What happens:** No exploration → algorithm gets stuck on initially good arm.

**Example:** `γ=0.01` means only 1% probability is allocated uniformly. If first arm is good by chance, algorithm never tries others.

**Fix:** Use theory-guided `γ = sqrt(K ln K / T)`, or empirically tune on adversarial scenarios. Typical range: `γ ∈ [0.05, 0.3]`.

### Pitfall 3: Gamma Too Large
**What happens:** Too much exploration → almost uniform random selection, no exploitation.

**Example:** `γ=0.9` means 90% of probability is uniform, only 10% based on weights. You're essentially playing randomly.

**Fix:** Don't exceed `γ=0.5` in practice. Theory says `γ = O(sqrt(K/T))`, which decreases with horizon.

### Pitfall 4: Mistaking Market Noise for Adversarial Behavior
**What happens:** You use EXP3 for stochastic problems, getting worse regret than UCB/Thompson Sampling.

**Example:** Commodity returns are noisy but i.i.d. You use EXP3 "to be safe," but `O(sqrt(T))` regret beats `O(log T)` from UCB.

**Fix:** EXP3 is for adversarial settings. Ask:
- **Are rewards actively chosen to hurt me?** (adversarial) → EXP3
- **Are rewards random but independent of my actions?** (stochastic) → UCB/Thompson

**When to worry about adversarial:**
- High-frequency trading (others adapt to your patterns)
- Large positions (your trades move the market against you)
- Public strategies (competitors can reverse-engineer)

**When stochastic is fine:**
- Small positions (no market impact)
- Passive data (rewards don't depend on your actions)
- Natural experiments (A/B tests where variants don't adapt)

### Pitfall 5: Not Handling Market Impact Correctly
**What happens:** You treat market impact as adversarial, but it's actually a function of *size*, not selection.

**Example:** Your trades have market impact proportional to volume. But if you randomize small trades, impact is negligible anyway.

**Fix:**
- **Low market impact:** Use stochastic bandits (better regret bounds)
- **High market impact:** Model impact explicitly (execution cost models) rather than assuming adversarial

## Connections

### Builds On
- **Stochastic Bandits (Module 2):** EXP3 generalizes to worst-case (adversarial) settings
- **Regret Bounds:** Adversarial regret `O(sqrt(T))` vs stochastic `O(log T)`
- **Game Theory:** Randomized strategies for zero-sum games

### Leads To
- **Contextual Adversarial Bandits:** EXP4 algorithm (contexts + adversarial)
- **Online Convex Optimization:** General framework for adversarial online learning
- **Robust Optimization:** Handling worst-case scenarios in portfolio optimization

### Related Concepts
- **Follow-the-Regularized-Leader (FTRL):** Alternative to EXP3 with similar guarantees
- **Multiplicative Weights Update:** General algorithmic technique for online learning
- **Minimax Regret:** Optimal strategy against worst-case adversary

## Practice Problems

### Problem 1: Stochastic vs Adversarial
For each scenario, decide if EXP3 is appropriate or if stochastic bandits are better:

a) Allocating between 5 commodity indices (returns are noisy but i.i.d.)
b) High-frequency market making (your quotes affect the market)
c) A/B testing website layouts (users don't know your algorithm)
d) Bidding in repeated auctions against strategic competitors

### Problem 2: Regret Bound Calculation
With `K=4` arms and horizon `T=10000`, what is the optimal `γ` for EXP3?

What is the expected regret bound `2 sqrt(T K ln K)`?

Compare to UCB regret `O(K log T)` — at what horizon `T` does UCB become better than EXP3 (assuming stochastic setting)?

### Problem 3: Importance Sampling Intuition
EXP3 uses `r̂ = r / p` for inverse probability weighting.

Suppose you select an arm with `p=0.2` and observe `r=0.8`. What is `r̂`? Why is it larger than `r`?

If you selected with `p=0.9` and observed `r=0.8`, what would `r̂` be? Why is it smaller?

### Problem 4: Hybrid Strategy
Design a bandit that:
- Uses Thompson Sampling when trading volume is low (no market impact)
- Switches to EXP3 when volume is high (market impact threshold exceeded)

What signals would trigger the switch? How would you prevent oscillation between modes?

## Commodity Application Example

**Scenario:** You're running a high-frequency commodity arbitrage strategy between 3 exchanges. Your strategy:
1. Detects price discrepancies
2. Buys on cheap exchange, sells on expensive exchange
3. Profits from the spread

**Problem:** As you execute, market makers on all exchanges observe your pattern (you always buy when exchange A is cheapest). They start widening spreads when they see you coming.

**Adversarial nature:**
- Rewards (spreads) depend on your observable behavior
- Liquidity providers adapt to exploit predictable patterns
- Deterministic algorithms leak information

**EXP3 Solution:**

```python
# Initialize EXP3 for 3 exchanges
bandit = EXP3(n_arms=3)
bandit.set_horizon(T=1000)

exchanges = ['CME', 'ICE', 'LME']

for t in range(1000):
    # Randomly select exchange according to EXP3 probabilities
    arm = bandit.select_arm()
    exchange = exchanges[arm]

    # Execute trade on selected exchange
    spread, filled = execute_trade(exchange)

    # Reward: negative of spread (lower spread = better)
    # Normalize to [0, 1]
    reward = 1 - (spread / max_spread)

    bandit.update(arm, reward)

    # Key: Your selection is unpredictable, so market makers
    # can't specifically widen spreads against you
```

**Result:** Randomization prevents market makers from learning your pattern. They must quote competitive spreads on all exchanges, not just the ones you're likely to avoid.

**Trade-off:** EXP3 occasionally selects expensive exchanges (exploration), so you pay higher spreads sometimes. But you gain by preventing systematic exploitation of predictable patterns.

**When to use:**
- Position size > 1% of typical volume (significant market impact)
- Repeated execution in same markets (pattern detectability)
- Presence of strategic adversaries (HFT competitors, informed market makers)

**When NOT to use:**
- Small positions (no market impact → stochastic bandits are better)
- One-time execution (no opportunity for adversary to learn)
- Markets with many passive participants (no strategic adaptation)

**Empirical test:** Backtest both UCB and EXP3 on your execution data. If UCB's regret is similar, the environment is likely stochastic and you don't need EXP3. If EXP3 significantly outperforms, adversarial dynamics are present.
