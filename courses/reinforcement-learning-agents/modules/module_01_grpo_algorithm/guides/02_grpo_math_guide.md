# Guide 02: GRPO Mathematics — Objective Function, Clipping, and KL Penalty

## Learning Objectives

By the end of this guide you will be able to:

1. State the full GRPO objective function and identify each term's role
2. Explain why the probability ratio $\rho_i = \pi_\theta(o_i|q) / \pi_{old}(o_i|q)$ measures policy change
3. Describe what the clip operation does and why it is necessary for stable training
4. Explain what the KL divergence penalty prevents and what happens when it is removed
5. Implement advantage computation and the clipped surrogate loss in NumPy

---

## The Full GRPO Objective

$$L_{\text{GRPO}} = \mathbb{E}_{q \sim P(Q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\!\left(\frac{\pi_\theta(o_i|q)}{\pi_{old}(o_i|q)} A_i,\ \text{clip}\!\left(\frac{\pi_\theta(o_i|q)}{\pi_{old}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i\right) - \beta\, D_{\text{KL}}(\pi_\theta \| \pi_{ref}) \right]$$

This looks dense. We'll dissect it term by term.

---

## Term 1: The Probability Ratio

$$\rho_i = \frac{\pi_\theta(o_i|q)}{\pi_{old}(o_i|q)}$$

This ratio measures how much the new policy $\pi_\theta$ differs from the policy that *generated* the completion $\pi_{old}$.

- $\rho_i = 1.0$: the new policy assigns the same probability to this completion as the old policy did — no change
- $\rho_i = 1.5$: the new policy is 50% more likely to produce this completion
- $\rho_i = 0.6$: the new policy is 40% less likely to produce this completion

**Why does this matter?** Policy gradient methods work by reweighting samples. If the policy has changed a lot since the samples were collected, the gradient estimate becomes biased. The ratio tracks exactly how much reweighting is needed.

**In practice**, this is computed as:

$$\rho_i = \exp\!\left(\log \pi_\theta(o_i|q) - \log \pi_{old}(o_i|q)\right)$$

Because log-probabilities are what language models actually output (logits → log softmax → sum over tokens).

---

## Term 2: The Unclipped Objective

$$\rho_i \cdot A_i$$

The basic policy gradient update: increase $\rho_i$ (make the completion more likely) when $A_i > 0$, decrease it when $A_i < 0$.

**The problem:** gradient ascent on this term has no upper bound. If the policy finds a lucky completion with high advantage, it will keep increasing that completion's probability without limit. This causes *policy collapse* — the model degenerates to producing only one type of output.

---

## Term 3: The Clipping Mechanism

$$\text{clip}\!\left(\rho_i, 1-\epsilon, 1+\epsilon\right) \cdot A_i$$

The clip operation constrains the probability ratio to the interval $[1-\epsilon, 1+\epsilon]$.

For a typical value of $\epsilon = 0.2$:
- If $\rho_i > 1.2$: treat it as $1.2$ (cap the update)
- If $\rho_i < 0.8$: treat it as $0.8$ (cap the update in the other direction)
- If $0.8 \leq \rho_i \leq 1.2$: use the actual ratio

**Taking the minimum of clipped and unclipped:**

$$\min(\rho_i A_i,\ \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i)$$

This is the conservative step. Consider the two cases:

**When $A_i > 0$ (good completion — want to reinforce):**
- If $\rho_i > 1+\epsilon$: policy already assigns much higher probability to this completion. Additional reinforcement is capped. $\min$ selects the clipped value.
- If $\rho_i \leq 1+\epsilon$: clip is not active. $\min$ selects the unclipped value.

**When $A_i < 0$ (bad completion — want to discourage):**
- If $\rho_i < 1-\epsilon$: policy already assigns much lower probability to this completion. Additional discouragement is capped.
- If $\rho_i \geq 1-\epsilon$: clip is not active.

The result: the policy updates confidently for moderate changes, but cannot make huge jumps in a single step.

---

## Term 4: The KL Divergence Penalty

$$-\beta\, D_{\text{KL}}(\pi_\theta \| \pi_{ref})$$

The KL divergence measures how different the current policy $\pi_\theta$ is from a frozen reference model $\pi_{ref}$ (typically the supervised fine-tuned base model before RL training began).

$$D_{\text{KL}}(\pi_\theta \| \pi_{ref}) = \mathbb{E}_{o \sim \pi_\theta}\!\left[\log \frac{\pi_\theta(o|q)}{\pi_{ref}(o|q)}\right]$$

**What it prevents:** without this term, RL training can *reward-hack* — the model finds degenerate behaviors that score high on the reward function but have completely broken language. Classic examples include reward models that can be fooled by repetition, specific tokens, or adversarial formatting. The KL penalty keeps the model within the "sensible language" region of parameter space.

**$\beta$ controls the tradeoff:**
- $\beta = 0$: pure reward maximization, no constraint → risk of reward hacking
- $\beta$ too large: model barely moves from the reference → no learning
- Typical values: $\beta \in [0.01, 0.1]$

**DeepSeek-R1 removed the KL penalty** in later training phases after the model had demonstrated stable behavior. This is an advanced technique requiring careful monitoring.

---

## Advantage Calculation: Python Implementation

```python
import numpy as np


def compute_group_advantages(rewards: np.ndarray) -> np.ndarray:
    """
    Compute normalized advantages for a group of completions.

    Parameters
    ----------
    rewards : np.ndarray
        Shape (G,) — reward scores for G completions of a single prompt.
        All completions must be for the same prompt.

    Returns
    -------
    np.ndarray
        Shape (G,) — normalized advantages. Mean 0, std 1 (approximately).

    Notes
    -----
    Uses population std (ddof=0) to match the GRPO paper. When G is small
    (e.g., G=4), a small epsilon is added to the denominator to prevent
    division by zero when all rewards are identical.
    """
    mean = np.mean(rewards)
    std = np.std(rewards)  # population std, ddof=0

    # When all rewards are equal, std=0 and the group provides no signal.
    # Return zero advantages to produce zero gradient.
    if std < 1e-8:
        return np.zeros_like(rewards)

    advantages = (rewards - mean) / std
    return advantages


# Verify with the worked example from Guide 01
rewards = np.array([0.9, 0.7, 0.5, 0.3])
advantages = compute_group_advantages(rewards)

print("Rewards:", rewards)
print("Mean:", np.mean(rewards))
print("Std:", np.std(rewards))
print("Advantages:", np.round(advantages, 4))
# Expected: [ 1.3416  0.4472 -0.4472 -1.3416]
```

---

## Clipped Surrogate Loss: Python Implementation

```python
def clipped_surrogate_loss(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    advantages: np.ndarray,
    epsilon: float = 0.2,
) -> float:
    """
    Compute the GRPO clipped surrogate loss for one prompt's group.

    Parameters
    ----------
    log_probs_new : np.ndarray
        Shape (G,) — log probabilities of each completion under the NEW policy.
        Each value is the sum of token log-probs for the full completion.
    log_probs_old : np.ndarray
        Shape (G,) — log probabilities under the OLD policy (used to generate
        the completions). Treated as constant (no gradient flows through it).
    advantages : np.ndarray
        Shape (G,) — group-normalized advantages from compute_group_advantages().
    epsilon : float
        Clip range. Probability ratios are constrained to [1-epsilon, 1+epsilon].
        Typical value: 0.2 (from PPO paper, adopted by GRPO).

    Returns
    -------
    float
        Scalar loss value. We MAXIMIZE this (or equivalently, minimize -loss).
    """
    # Probability ratio: how much more/less likely under new vs old policy
    # Use log difference for numerical stability, then exponentiate
    log_ratio = log_probs_new - log_probs_old
    ratio = np.exp(log_ratio)

    # Unclipped term: standard policy gradient objective
    unclipped = ratio * advantages

    # Clipped term: conservative estimate using bounded ratio
    clipped_ratio = np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
    clipped = clipped_ratio * advantages

    # Take minimum of the two — conservative update
    # For positive advantage: min caps upward updates when ratio is too large
    # For negative advantage: min caps downward updates when ratio is too small
    per_completion_loss = np.minimum(unclipped, clipped)

    # Average over the group (the 1/G term from the GRPO objective)
    return float(np.mean(per_completion_loss))


# Example: four completions, advantages from our worked example
advantages = np.array([1.3416, 0.4472, -0.4472, -1.3416])

# Simulate log-probs: policy has shifted slightly toward better completions
log_probs_old = np.array([-2.1, -2.3, -2.5, -2.8])
log_probs_new = np.array([-2.0, -2.25, -2.55, -2.9])  # better completions up, worse down

loss = clipped_surrogate_loss(log_probs_new, log_probs_old, advantages, epsilon=0.2)
print(f"Surrogate loss: {loss:.4f}")

# Show the components for inspection
ratio = np.exp(log_probs_new - log_probs_old)
print(f"Probability ratios: {np.round(ratio, 4)}")
print(f"Ratios within clip range [0.8, 1.2]:", np.all((ratio >= 0.8) & (ratio <= 1.2)))
```

---

## KL Divergence: Python Implementation

```python
def approximate_kl_divergence(
    log_probs_policy: np.ndarray,
    log_probs_reference: np.ndarray,
) -> float:
    """
    Approximate KL divergence between policy and reference model.

    Uses the unbiased estimator:
        KL(π || π_ref) ≈ mean(log π - log π_ref)

    This is an approximation because we are averaging over samples from π
    rather than computing the true expectation. For GRPO training, this
    approximation is standard and computationally efficient.

    Parameters
    ----------
    log_probs_policy : np.ndarray
        Log probabilities under the current (training) policy.
    log_probs_reference : np.ndarray
        Log probabilities under the frozen reference model.

    Returns
    -------
    float
        Estimated KL divergence. Should be 0 at training start and increase
        as the policy diverges from the reference.
    """
    # KL(π || π_ref) = E_π[log π - log π_ref]
    log_ratio = log_probs_policy - log_probs_reference
    return float(np.mean(log_ratio))


def grpo_loss(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    log_probs_ref: np.ndarray,
    advantages: np.ndarray,
    epsilon: float = 0.2,
    beta: float = 0.04,
) -> dict:
    """
    Full GRPO loss: clipped surrogate + KL penalty.

    Returns a dict with the total loss and each component for logging.
    """
    surrogate = clipped_surrogate_loss(log_probs_new, log_probs_old, advantages, epsilon)
    kl = approximate_kl_divergence(log_probs_new, log_probs_ref)

    # GRPO objective: maximize (surrogate - beta * KL)
    # In practice we minimize the negative of this
    total = surrogate - beta * kl

    return {
        "loss": total,
        "surrogate": surrogate,
        "kl": kl,
        "kl_penalty": beta * kl,
    }


# Full example
log_probs_ref = np.array([-2.1, -2.3, -2.5, -2.8])  # reference = old policy here
result = grpo_loss(
    log_probs_new=np.array([-2.0, -2.25, -2.55, -2.9]),
    log_probs_old=log_probs_ref,
    log_probs_ref=log_probs_ref,
    advantages=advantages,
    epsilon=0.2,
    beta=0.04,
)
print("GRPO loss components:")
for key, value in result.items():
    print(f"  {key}: {value:.4f}")
```

---

## Putting It All Together: One GRPO Update Step

```python
def grpo_update_step(
    rewards: np.ndarray,
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    log_probs_ref: np.ndarray,
    epsilon: float = 0.2,
    beta: float = 0.04,
) -> dict:
    """
    Complete GRPO update for one prompt's group of completions.

    In a real implementation, log_probs_new would be computed from the
    training model (with gradients enabled) and you would call loss.backward().
    Here we compute the scalar loss value for educational purposes.

    Parameters
    ----------
    rewards : np.ndarray
        Shape (G,) — reward scores for each completion.
    log_probs_new : np.ndarray
        Shape (G,) — log-probs under current (differentiable) policy.
    log_probs_old : np.ndarray
        Shape (G,) — log-probs under old policy (no gradient).
    log_probs_ref : np.ndarray
        Shape (G,) — log-probs under frozen reference model (no gradient).
    """
    # Step 1: compute group-normalized advantages
    advantages = compute_group_advantages(rewards)

    # Step 2: compute full GRPO loss
    result = grpo_loss(log_probs_new, log_probs_old, log_probs_ref, advantages, epsilon, beta)

    # In real training: result["loss_tensor"].backward() then optimizer.step()
    return {
        "advantages": advantages,
        **result,
    }


# Run a complete example
step_result = grpo_update_step(
    rewards=np.array([0.9, 0.7, 0.5, 0.3]),
    log_probs_new=np.array([-2.0, -2.25, -2.55, -2.9]),
    log_probs_old=np.array([-2.1, -2.3, -2.5, -2.8]),
    log_probs_ref=np.array([-2.1, -2.3, -2.5, -2.8]),
)

print("One GRPO Update Step:")
print(f"  Advantages: {np.round(step_result['advantages'], 3)}")
print(f"  Surrogate loss: {step_result['surrogate']:.4f}")
print(f"  KL divergence: {step_result['kl']:.4f}")
print(f"  KL penalty: {step_result['kl_penalty']:.4f}")
print(f"  Total loss: {step_result['loss']:.4f}")
```

---

## Hyperparameter Reference

| Parameter | Symbol | Typical Value | Effect |
|-----------|--------|---------------|--------|
| Group size | $G$ | 4–16 | Larger G → more stable baseline, more inference cost |
| Clip range | $\epsilon$ | 0.2 | Smaller → more conservative updates, slower learning |
| KL coefficient | $\beta$ | 0.01–0.1 | Larger → stronger regularization toward reference model |
| Learning rate | $\alpha$ | 1e-5 to 1e-6 | Standard for LLM fine-tuning; GRPO is not especially sensitive |

---

## Summary

The GRPO objective has three components working together:

1. **Probability ratio** $\rho_i$: measures how much the policy has changed since the rollout was collected
2. **Clipped surrogate** $\min(\rho_i A_i, \text{clip}(\rho_i) A_i)$: reinforces good completions and discourages bad ones, with capped step size for stability
3. **KL penalty** $-\beta D_{\text{KL}}$: prevents the model from drifting so far from the reference that it loses coherent language generation

The advantage $A_i = (r_i - \mu)/\sigma$ ties everything back to relative group performance, making the entire system scale-invariant to reward magnitude.

---

## Next

Guide 03 — GRPO vs Alternatives: how GRPO compares to PPO, DPO, and REINFORCE, and when each algorithm is the right choice.
