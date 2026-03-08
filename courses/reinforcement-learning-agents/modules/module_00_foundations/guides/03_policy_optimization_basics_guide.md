# Guide 03: Policy Optimization Basics

## In Brief

A policy is a function that maps states to actions. Policy optimization is the process of adjusting this function to produce better actions. This guide builds the conceptual foundation for GRPO, which Module 01 covers in full detail.

---

## Key Insight

**Policy gradient methods update the policy by asking: "For each action I took, was the outcome better or worse than average? If better, increase the probability of that action. If worse, decrease it."**

This is simple in principle. The challenge is doing it stably and efficiently — which is what GRPO solves.

---

## What is a Policy?

In reinforcement learning, a **policy** $\pi$ is a function that maps observations (states) to actions (or distributions over actions):

$$\pi(a \mid s) = P(\text{action} = a \mid \text{state} = s)$$

For a language model agent:
- **State $s$:** the current context — system prompt, conversation history, tool results so far
- **Action $a$:** the next token (or, at a higher level, the next tool call or response)
- **Policy $\pi_\theta$:** the language model with parameters $\theta$

```python
# Conceptual mapping: LLM as a policy

class LanguageModelPolicy:
    """
    A language model viewed as an RL policy.

    state  = the full context (prompt + history so far)
    action = the next token to generate (or the next tool call)
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_action_distribution(self, state: str) -> dict[str, float]:
        """
        Returns P(action | state) — the probability of each possible next token.

        In RL terms: this IS the policy.
        In LLM terms: this is the output softmax over the vocabulary.
        """
        inputs = self.tokenizer(state, return_tensors="pt")
        logits = self.model(**inputs).logits[0, -1, :]  # last token position
        probs = logits.softmax(dim=-1)

        # Map token ids to probabilities (simplified)
        vocab = self.tokenizer.get_vocab()
        return {token: probs[idx].item() for token, idx in vocab.items()}

    def sample_action(self, state: str) -> str:
        """
        Sample an action from the policy distribution.

        This is what happens at inference time.
        """
        dist = self.get_action_distribution(state)
        tokens = list(dist.keys())
        probs = list(dist.values())

        # Sampling from the distribution
        import random
        return random.choices(tokens, weights=probs, k=1)[0]
```

At training time, we want to adjust the parameters $\theta$ so that the policy produces better sequences.

---

## Policy Gradient Intuition

The core idea of policy gradient methods can be stated in plain English:

1. Run the current policy to produce $G$ complete trajectories (rollouts)
2. Score each trajectory with the reward function $r(\tau)$
3. For trajectories that got high reward: increase the probability of the actions taken
4. For trajectories that got low reward: decrease the probability of the actions taken
5. Repeat

The policy gradient theorem gives us a formula for how to compute the gradient of the expected reward:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot R(\tau) \right]$$

In plain terms: for each action $a_t$ taken in each trajectory, compute the gradient of its log-probability, and scale it by the total reward of that trajectory. Actions that led to high-reward trajectories get a positive gradient (increase their probability); actions in low-reward trajectories get a negative gradient.

```python
import numpy as np


def policy_gradient_update_concept(
    log_probs: list[float],
    rewards: list[float],
    learning_rate: float = 0.01,
) -> list[float]:
    """
    Conceptual implementation of the REINFORCE policy gradient update.

    This illustrates the core idea; real implementations use PyTorch autograd.

    Args:
        log_probs: log P(action | state) for each action in the trajectory
        rewards: reward for the whole trajectory (repeated for each step)
        learning_rate: step size for the gradient update

    Returns:
        list of parameter updates (one per action)
    """
    assert len(log_probs) == len(rewards), "One reward signal per action step"

    # The policy gradient: gradient of log_prob, scaled by reward
    # Positive reward → push log_prob up (increase probability)
    # Negative reward → push log_prob down (decrease probability)
    gradients = [r * lp for r, lp in zip(rewards, log_probs)]

    # Parameter update: move in the direction of the gradient
    updates = [learning_rate * g for g in gradients]

    return updates


# Example: two trajectories, same action, different rewards
action_log_prob = -1.2  # log P(this action) = e^(-1.2) ≈ 0.30

# Trajectory 1: this action led to high reward → reinforce it
update_high_reward = policy_gradient_update_concept(
    log_probs=[action_log_prob],
    rewards=[1.0],
    learning_rate=0.01,
)
print(f"Update for high-reward action: {update_high_reward[0]:.4f}")  # -0.012 (negative = increase log_prob)

# Trajectory 2: this action led to low reward → suppress it
update_low_reward = policy_gradient_update_concept(
    log_probs=[action_log_prob],
    rewards=[0.0],
    learning_rate=0.01,
)
print(f"Update for low-reward action: {update_low_reward[0]:.4f}")  # 0.0 (no update)
```

---

## Why Vanilla Policy Gradient is Noisy and Sample-Inefficient

The basic REINFORCE algorithm has two serious problems in practice:

### Problem 1: High variance

The reward $R(\tau)$ for a complete trajectory is a single noisy number. Different rollouts of the same prompt can produce wildly different rewards due to randomness in sampling. The gradient estimate inherits this noise.

```python
import numpy as np

# Simulate 1000 rollouts of the same policy
# The reward is noisy: same policy, same prompt, different outcomes
np.random.seed(42)
rewards_from_same_policy = np.random.normal(loc=0.6, scale=0.3, size=1000)
rewards_from_same_policy = np.clip(rewards_from_same_policy, 0, 1)

print(f"Mean reward: {rewards_from_same_policy.mean():.3f}")
print(f"Std reward:  {rewards_from_same_policy.std():.3f}")   # High variance!
print(f"Min reward:  {rewards_from_same_policy.min():.3f}")
print(f"Max reward:  {rewards_from_same_policy.max():.3f}")

# The gradient estimate is this noisy value × log_prob
# With high variance in rewards, we need many samples to get a stable gradient
```

### Problem 2: Sample inefficiency

REINFORCE throws away each rollout after one gradient update. This is wasteful: generating a rollout is expensive (requires a full forward pass through the model), and we discard it after a single use.

```python
# REINFORCE: one rollout, one gradient step, discard
def reinforce_step(policy, prompt, reward_fn):
    rollout = policy.generate(prompt)      # expensive: full LLM forward pass
    reward = reward_fn(rollout)            # compute reward
    gradient = compute_gradient(rollout, reward)  # one gradient estimate
    policy.update(gradient)               # one update
    # rollout is now discarded — never used again
    return gradient


# PPO / GRPO: multiple gradient steps per rollout (or group of rollouts)
def grpo_step(policy, prompt, reward_fn, group_size=8, gradient_steps=4):
    rollouts = [policy.generate(prompt) for _ in range(group_size)]  # G rollouts
    rewards = [reward_fn(r) for r in rollouts]
    advantages = compute_advantages(rewards)  # group-relative normalization

    # Multiple gradient steps using the same rollouts
    for _ in range(gradient_steps):
        gradient = compute_grouped_gradient(rollouts, advantages)
        policy.update(gradient)
    # Now we get gradient_steps updates from group_size rollouts
```

---

## Advantage Estimation: "How Much Better Than Average?"

The key fix for high variance is to subtract a **baseline** from the reward. Instead of asking "was this trajectory good?", we ask "was this trajectory better than average?"

$$A_t = R(\tau) - b(s_t)$$

where $b(s_t)$ is the baseline: the expected reward from state $s_t$ under the current policy.

If $A_t > 0$: the trajectory was better than average. Reinforce those actions.
If $A_t < 0$: the trajectory was worse than average. Suppress those actions.
If $A_t = 0$: the trajectory was exactly average. No update.

```python
import numpy as np


def compute_advantages_with_baseline(
    rewards: list[float],
    method: str = "group_mean",
) -> list[float]:
    """
    Compute advantage values for a group of rollouts.

    Args:
        rewards: list of scalar rewards for G rollouts
        method: 'group_mean' uses mean of the group as baseline
                'normalized' uses (r - mean) / std (what GRPO uses)

    Returns:
        list of advantage values
    """
    rewards_arr = np.array(rewards, dtype=float)

    if method == "group_mean":
        baseline = rewards_arr.mean()
        # Advantage: how much better than the group average?
        advantages = rewards_arr - baseline
        return advantages.tolist()

    elif method == "normalized":
        # Normalize to zero mean and unit variance
        # This is what GRPO uses: removes scale dependence
        mean = rewards_arr.mean()
        std = rewards_arr.std()
        if std < 1e-8:
            return [0.0] * len(rewards)
        advantages = (rewards_arr - mean) / std
        return advantages.tolist()

    else:
        raise ValueError(f"Unknown method: {method}")


# Compare the two methods on the same reward group
rewards = [0.2, 0.8, 1.0, 0.6, 0.4, 1.0, 0.0, 0.6]

advantages_mean = compute_advantages_with_baseline(rewards, method="group_mean")
advantages_norm = compute_advantages_with_baseline(rewards, method="normalized")

print(f"{'Reward':>8} {'Advantage (mean)':>18} {'Advantage (norm)':>18}")
print("-" * 46)
for r, a_m, a_n in zip(rewards, advantages_mean, advantages_norm):
    print(f"{r:>8.2f} {a_m:>18.3f} {a_n:>18.3f}")

# Both methods: positive = better than average, negative = worse than average
# Normalized method: scale-invariant (works regardless of reward magnitude)
```

### Why normalization matters

Suppose your reward function returns values in [0, 100] rather than [0, 1]. With raw advantages:
- Better-than-average rollout: advantage = +30
- Worse-than-average rollout: advantage = −30

This large scale causes large gradient updates, which destabilize training. With normalized advantages, the scale is always ±1 standard deviation, independent of the reward scale.

---

## The Update Rule: Reinforce Good, Suppress Bad

Putting it together, the policy gradient update with advantage estimation is:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A_t \right]$$

```python
def compute_policy_gradient(
    log_probs: np.ndarray,     # log P(a_t | s_t) for each step t, shape (T,)
    advantages: np.ndarray,    # A_t for each step t, shape (T,)
) -> float:
    """
    Compute the policy gradient loss (to be minimized by gradient descent).

    Note: we minimize the negative of the objective (since optimizers minimize).
    Maximizing J(π) = minimizing -J(π).

    Args:
        log_probs: log probabilities of the actions taken, shape (T,)
        advantages: advantage values, shape (T,)

    Returns:
        scalar loss value
    """
    # Policy gradient: sum over steps of (log_prob × advantage)
    pg_objective = (log_probs * advantages).sum()

    # We minimize the negative (gradient ASCENT on reward = gradient DESCENT on loss)
    loss = -pg_objective

    return loss


# Example: trajectory where the first action was great, second was bad
example_log_probs = np.array([-0.5, -2.1, -0.8])  # log probabilities of actions taken
example_advantages = np.array([+1.2, -0.8, +0.3])  # advantages at each step

loss = compute_policy_gradient(example_log_probs, example_advantages)
print(f"Policy gradient loss: {loss:.4f}")

# Effect on each action after gradient descent:
for t, (lp, a) in enumerate(zip(example_log_probs, example_advantages)):
    direction = "INCREASE" if a > 0 else "DECREASE"
    print(f"  Step {t}: advantage={a:+.1f} → {direction} probability of this action")
```

---

## How This Sets Up GRPO

Module 01 covers GRPO (Group Relative Policy Optimization) in full detail. This guide has given you the four concepts GRPO builds on:

| Concept | What we covered | How GRPO uses it |
|---------|----------------|------------------|
| Policy | LLM as a state → action mapping | The model being trained |
| Policy gradient | Update rule: reinforce by advantage | The core update formula |
| Vanilla PG problems | Noise, sample inefficiency | What GRPO is designed to fix |
| Advantage estimation | How much better than average? | Group-relative normalization |

GRPO's specific innovations (covered in Module 01):
1. **Group rollouts:** Generate $G$ rollouts per prompt and compute advantages from the group mean/std — this is the "group relative" in the name
2. **Clipped objective:** Prevents catastrophically large updates (borrows from PPO)
3. **KL penalty:** Keeps the updated policy close to the reference policy (prevents mode collapse)

```python
# Preview: what GRPO computes (full derivation in Module 01)
def grpo_objective_preview(
    new_log_probs: np.ndarray,     # log P_new(a_t | s_t)
    old_log_probs: np.ndarray,     # log P_old(a_t | s_t) — the rollout policy
    advantages: np.ndarray,        # group-relative advantages
    clip_epsilon: float = 0.2,     # PPO-style clipping parameter
) -> float:
    """
    Preview of the GRPO objective. Full derivation in Module 01.

    Key ideas:
    - ratio: how much has the policy changed from the rollout policy?
    - clipping: prevents the ratio from getting too large (stability)
    - advantage weighting: same as vanilla policy gradient
    """
    # Importance sampling ratio: how much has the policy changed?
    ratio = np.exp(new_log_probs - old_log_probs)

    # Clipped ratio: don't let the policy change too much in one step
    ratio_clipped = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    # Take the minimum: pessimistic bound on the objective
    per_token_objective = np.minimum(ratio * advantages, ratio_clipped * advantages)

    return per_token_objective.mean()
```

---

## Common Pitfalls

**Pitfall 1: Confusing the policy gradient with supervised gradient.** In SFT, the loss is: "move log_prob of the target tokens up." In policy gradient, the update is: "move log_prob up if advantage is positive, down if advantage is negative." The sign and magnitude depend on the advantage, not a fixed target.

**Pitfall 2: Forgetting that advantages can be zero.** If all rollouts in a group receive the same reward (all correct, or all incorrect), the advantages are all zero and the policy receives no gradient signal. This is the "dead zone" — make sure your reward function produces variation within groups.

**Pitfall 3: Using too small a group size.** With $G = 2$, the baseline (group mean) is estimated from 2 samples — extremely noisy. Use $G \geq 4$, and ideally $G \geq 8$, for stable advantage estimates.

---

## Connections

- **Builds on:** Guide 02 (Reward Signals) — the advantages computed there feed directly into the policy gradient formula here
- **Leads to:** Module 01 (GRPO Algorithm) — GRPO extends vanilla policy gradient with clipping, KL penalty, and group-relative advantages
- **Related:** PPO (Proximal Policy Optimization) is the most widely used policy gradient method; GRPO is a simplified variant designed for LLMs
- **Related:** Actor-Critic methods learn the baseline $b(s_t)$ with a separate neural network (the critic); GRPO uses the group mean instead

---

## Further Reading

- Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (1992) — the original REINFORCE paper; short and worth reading
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017) — PPO, which GRPO simplifies; the standard for RL in practice
- Shao et al., "DeepSeekMath" (2024) — introduces GRPO; Section 3 covers the algorithm derivation in full
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2015) — GAE, the standard advantage estimation technique that GRPO adapts
