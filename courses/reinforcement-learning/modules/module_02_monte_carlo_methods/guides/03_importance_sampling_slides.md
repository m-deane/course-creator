---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Off-Policy MC via Importance Sampling

## Learning from a Different Policy's Data

**Module 2 — Monte Carlo Methods**
Reinforcement Learning

<!-- Speaker notes: This guide addresses one of the most practically important ideas in RL: off-policy learning. The core problem — you want to evaluate or optimize policy pi, but your data comes from policy b. This arises constantly in practice: historical data from a previous system, safe exploration policies, human demonstrations. Importance sampling is the mathematical tool that makes this possible. -->

---

## The Off-Policy Setting

<div class="columns">

**On-Policy (Guides 01-02)**
Same policy does both:
- Generates episodes
- Gets evaluated/improved

**Advantage:** Simple, stable
**Disadvantage:** Exploration hurts policy quality; can't reuse old data

**Off-Policy (This Guide)**
Two separate policies:
- **$b$** (behavior): generates episodes
- **$\pi$** (target): gets evaluated/improved

**Advantage:** $b$ explores freely; reuse historical data; train $\pi$ safely

</div>


<div class="callout-insight">
<strong>Insight:</strong> This is a key takeaway from this section that connects to the broader course themes.
</div>

<!-- Speaker notes: Draw the contrast clearly. On-policy is simpler but constraining — the data-generating policy and the learning target are the same thing, so exploring (which is good for learning) contaminates the policy (which you want to be greedy). Off-policy separates these concerns. The behavior policy can be a reckless explorer while the target policy remains pure and greedy. This separation is fundamental to Q-learning and DQN. -->

---

## Why Off-Policy?

Four compelling use cases:

**1. Safe exploration**
Let $b$ explore dangerously (random policy); $\pi$ stays safe (greedy).
*Industrial robots, autonomous vehicles.*

**2. Historical data reuse**
Episodes from old policy $b_\text{old}$ can train new target $\pi$.
*Medical data, logged recommendation system interactions.*

**3. Human demonstrations**
$b$ = human expert behavior; $\pi$ = RL-optimized policy.
*Imitation learning, behavior cloning bootstrapping.*

**4. Multiple targets**
One run of $b$ trains many $\pi_1, \pi_2, \ldots$ simultaneously.
*Hyperparameter search, meta-learning.*


<div class="callout-key">
<strong>Key Point:</strong> Remember this concept — it appears repeatedly in later modules.
</div>

<!-- Speaker notes: Ground each use case in a concrete domain. The medical data example is powerful: you cannot run a random policy on patients. You use historical treatment data (b = past physicians) to evaluate a new treatment policy pi. Safe exploration is increasingly important in robotics. Multiple targets is the basis of "experience replay" in DQN — one replay buffer trains the same target policy repeatedly with different samples. -->

---

## The Coverage Assumption

For off-policy learning to work, the behavior policy must cover the target:

$$\pi(a \mid s) > 0 \implies b(a \mid s) > 0 \quad \forall s, a$$

**Plain English:** Wherever $\pi$ would act, $b$ must be willing to act too.

**Why it's needed:** If $b$ never takes action $a$ in state $s$, we have no data to estimate $Q^\pi(s, a)$.

**Sufficient condition:** $b$ is $\varepsilon$-soft (e.g., uniform random).

**What coverage does NOT require:**
- $b = \pi$ (they can differ arbitrarily in probabilities)
- $b$ being optimal or near-optimal


<div class="callout-warning">
<strong>Warning:</strong> This is a common source of confusion. Pay close attention to the distinction here.
</div>

<!-- Speaker notes: Coverage is a necessary mathematical condition, not just a technical assumption. If b never visits (s, a) pairs that pi would use, we literally have no information about those parts of pi's behavior. A uniform random policy trivially satisfies coverage (assigns positive probability to all actions in all states) and is therefore a common choice for b. Note: coverage can fail in large/continuous spaces where some states are extremely unlikely. -->

---

## The Importance Sampling Ratio

A trajectory from time $t$ to $T-1$ has probability under $\pi$:

$$\Pr_\pi = \prod_{k=t}^{T-1} \pi(A_k \mid S_k)\, \underbrace{p(S_{k+1} \mid S_k, A_k)}_{\text{environment}}$$

And under behavior policy $b$:

$$\Pr_b = \prod_{k=t}^{T-1} b(A_k \mid S_k)\, p(S_{k+1} \mid S_k, A_k)$$

**The ratio** (environment transitions cancel):

$$\boxed{\rho_{t:T-1} = \frac{\Pr_\pi}{\Pr_b} = \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}}$$

Only policy probabilities appear — no model required.


<div class="callout-info">
<strong>Info:</strong> This detail is useful context but not required to memorize.
</div>

<!-- Speaker notes: The cancellation of p(S'|S,A) is crucial and beautiful. We do need the model to generate the environment transitions, but we do NOT need to know the model probabilities to compute the IS ratio. The ratio depends only on how likely each action was under pi vs b. This means we only need to know pi(a|s) and b(a|s) — which we set ourselves — not the environment dynamics. -->

---

## IS Identity: Why It Works

We want $\mathbb{E}_\pi[G_t \mid S_t = s]$ but observe samples from $b$.

**Importance sampling identity:**

$$\mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_b\!\left[\rho_{t:T-1} \cdot G_t \;\middle|\; S_t = s\right]$$

**Intuition:**
- If a trajectory is **more likely** under $\pi$ than $b$: $\rho > 1$, up-weight it
- If a trajectory is **less likely** under $\pi$ than $b$: $\rho < 1$, down-weight it
- If a trajectory is **impossible** under $\pi$: $\rho = 0$, ignore it

The reweighted average recovers the correct expectation under $\pi$.

<!-- Speaker notes: This identity is the mathematical heart of importance sampling. Prove it briefly on the board: E_pi[G] = sum_tau G(tau) * Pr_pi(tau) = sum_tau G(tau) * [Pr_pi(tau)/Pr_b(tau)] * Pr_b(tau) = E_b[rho * G]. The substitution is exact — no approximation. The practical challenge: when rho varies widely across trajectories, the estimator has high variance even though it's unbiased. -->

---

## Ordinary Importance Sampling

Average the IS-weighted returns directly:

$$V(s) = \frac{\displaystyle\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1} \cdot G_t}{\displaystyle|\mathcal{T}(s)|}$$

**Properties:**
- Unbiased: $\mathbb{E}[V(s)] = V^\pi(s)$
- Variance: can be **infinite** for long episodes
- Example: if $\rho$ doubles per step, variance of $\rho^T \cdot G$ is exponential in $T$

**Best for:** Short episodes, $\pi \approx b$, or when unbiasedness is critical.

<!-- Speaker notes: The pathological variance case from Sutton & Barto Example 5.5: a single-state MDP where b takes each of two actions with probability 1/2, but pi takes only one action. The ratio is 2 per step. After T steps, rho = 2^T. The variance of 2^T * G grows without bound as episode length increases. Ordinary IS is theoretically clean but practically dangerous for long episodes. This motivates weighted IS. -->

---

## Weighted Importance Sampling

Normalize weights so they sum to 1:

$$V(s) = \frac{\displaystyle\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1} \cdot G_t}{\displaystyle\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}}$$

**Properties:**
- Biased (but bias $\to 0$ as $N \to \infty$ — **consistent**)
- Variance: **always bounded** (weights are in $[0,1]$ after normalization)
- MSE lower in practice despite the bias

**Intuition:** Weights express *relative importance* of each trajectory. The most $\pi$-likely trajectories contribute most to the estimate.

<!-- Speaker notes: Weighted IS is the standard choice in practice. The normalization ensures that even if individual rho values are enormous, the normalized weights are always between 0 and 1. The bias: if we observe only one trajectory with rho=1000 and one with rho=0.001, weighted IS gives the first trajectory weight ~1 and the second ~0. That estimate is biased toward high-rho trajectories for small N. As N grows, the average smooths out and the estimate converges to the truth. -->

---

## Side-by-Side Comparison

<div class="columns">

**Ordinary IS**
$$V(s) = \frac{\sum_t \rho_t G_t}{N}$$

- Unbiased
- Infinite variance possible
- Unreliable for long episodes
- Good for: theoretical proofs, $\pi \approx b$

**Weighted IS**
$$V(s) = \frac{\sum_t \rho_t G_t}{\sum_t \rho_t}$$

- Biased (consistent)
- Bounded variance
- Reliable for all episode lengths
- Good for: all practical applications

</div>

**MSE comparison (finite samples):**

$$\text{MSE}_\text{WIS} < \text{MSE}_\text{OIS}$$

in virtually all practical scenarios (Sutton & Barto, Ch. 5.6).

<!-- Speaker notes: The MSE comparison is the bottom line. Even though ordinary IS is unbiased, its high variance makes it worse in mean squared error for any finite number of samples. This is the classic bias-variance tradeoff: a small amount of bias from WIS reduces variance enough to lower total MSE. Analogous to ridge regression versus OLS — the biased estimator often wins in practice. -->

---

## Incremental Implementation (Weighted IS)

Maintain running weighted sum efficiently:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# For each episode, backward pass:
G = 0.0
W = 1.0

for (state, action, reward, b_prob) in reversed(episode):
    G = gamma * G + reward

    # Weighted incremental update
    C[state] += W                              # accumulate weight
    V[state] += (W / C[state]) * (G - V[state])  # weighted mean

    # Update IS weight
    if action != target_policy(state):
        break              # pi(a|s) = 0, ratio = 0, stop

    W = W * (1.0 / b_prob)   # pi(a|s)=1 for deterministic pi
```
</div>

`C[state]` is the cumulative weight denominator. The formula `V += (W/C) * (G - V)` is the weighted incremental mean.

<!-- Speaker notes: Walk through this code carefully. The incremental weighted mean formula: if we had weights w_1, w_2, ..., w_n and returns G_1, ..., G_n, the weighted average is sum(w_i * G_i) / sum(w_i). The incremental update maintains this without storing all returns. The break condition: once action != pi(state), the IS ratio for all earlier time steps is zero (because the full product includes this zero factor). Earlier steps do not contribute, so we stop. -->

---

## The Break Condition Visualized

```
Episode (reversed, t goes T-1 → 0):

t=T-1: A_{T-1} == π(S_{T-1}) ✓  W unchanged, update V(S_{T-1})
t=T-2: A_{T-2} == π(S_{T-2}) ✓  W unchanged, update V(S_{T-2})
t=T-3: A_{T-3} ≠  π(S_{T-3}) ✗  BREAK — ratio is 0

Steps t=0 to t=T-4 contribute nothing to this episode.
```

**Effect:** Only the *tail segment* of the episode (where $b$ happened to match $\pi$) contributes to learning. Longer segments = better updates. Shorter segments = wasted data.

**This is the main data-efficiency cost of off-policy MC.** TD methods (Module 3) handle this more efficiently via bootstrapping.

<!-- Speaker notes: The break condition is simultaneously the algorithm's strength and its weakness. Strength: we correctly zero out the contribution of trajectory segments that pi would not have generated. Weakness: most episodes, only the final few steps match pi, so most of the episode is discarded. In the extreme case where b is uniform random and pi is deterministic, an episode of length T only contributes the last step (on average). TD Q-learning solves this by not requiring the tail-segment property. -->

---

## Variance: The Fundamental Challenge

For ordinary IS, variance of $\rho_{t:T-1} \cdot G_t$:

$$\text{Var}[\rho \cdot G] = \mathbb{E}[\rho^2 G^2] - (\mathbb{E}[\rho G])^2$$

For a $T$-step trajectory with ratio $c > 1$ per step:

$$\rho = c^T \implies \text{Var}[\rho \cdot G] \propto c^{2T}$$

**Exponential in episode length.** Doubles for every step where $\pi$ assigns twice the probability as $b$.

**Mitigation strategies:**
1. **Weighted IS** — normalization caps individual weight contribution
2. **Per-decision IS** — separate ratio per reward, not full trajectory
3. **Keep $\pi$ and $b$ close** — small ratios → small variance
4. **Short effective horizon** — use $\gamma < 1$ or truncate

<!-- Speaker notes: The exponential variance growth is why off-policy MC is rarely used for long-horizon problems in practice. TD methods (Q-learning, Retrace) address this by not requiring trajectory-level IS ratios. Per-decision IS (Precup et al., 2000) is a middle ground: use a separate ratio for each reward term, which grows slower than the full trajectory ratio. This is the basis of V-trace and Retrace, used in distributed RL. -->

<div class="flow">
<div class="flow-step mint">Weighted IS</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">Per-decision IS</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">Keep $\pi$ and $b$ close</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">Short effective horizon</div>
</div>

---

## Off-Policy MC in the Module Context

```
Guide 01: MC Prediction (on-policy)
  → V^π(s) from episodes of π

Guide 02: MC Control (on-policy, ε-soft)
  → Q^* approximately, from π's own episodes

Guide 03: MC Control (off-policy, IS)
  → Q^* exactly, from behavior policy b's episodes
  → Target π = greedy; behavior b = ε-soft

Module 3 Preview: TD Methods
  → Q-learning: off-policy without explicit IS ratios
    (bootstrapping makes IS unnecessary per-step)
```

<!-- Speaker notes: Situate IS within the course arc. On-policy MC is simple but requires the data-generating policy to match the learning target. Off-policy MC with IS removes that requirement at the cost of variance. TD Q-learning (Module 3) achieves the same off-policy goal without IS by bootstrapping — the one-step nature of TD updates means the IS ratio is always just one factor, pi(a|s)/b(a|s), rather than a product. This is why Q-learning is so popular. -->

---

## Common Pitfalls

| Pitfall | Effect | Fix |
|---|---|---|
| Ordinary IS with long episodes | Infinite variance, divergence | Use weighted IS |
| Coverage violation ($b(a\|s)=0$) | Division by zero | Use ε-soft $b$ |
| Missing break condition | Spurious updates | Break when $\pi(A_t\|S_t)=0$ |
| $\pi$ and $b$ too different | High variance, slow learning | Keep $b$ reasonably close to $\pi$ |
| Confusing IS with TD off-policy | Wrong algorithm | IS is for MC; Q-learning handles off-policy TD differently |

<!-- Speaker notes: The last row is a common conceptual confusion. Students sometimes try to apply IS ratio correction to TD updates and get confused. TD Q-learning is off-policy by construction through bootstrapping — it doesn't need IS ratios for the standard case. IS ratios appear in TD methods only in advanced settings like multi-step returns (Retrace, V-trace). Keep the MC-IS connection separate from TD. -->

---

<!-- _class: lead -->

## Summary

- **Off-policy MC** separates behavior policy $b$ (data generation) from target policy $\pi$ (learning target)
- **Coverage requirement:** $\pi(a|s) > 0 \Rightarrow b(a|s) > 0$
- **IS ratio:** $\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$ — environment model cancels
- **Ordinary IS:** Unbiased, potentially infinite variance
- **Weighted IS:** Consistent, bounded variance, lower MSE — use this in practice
- **Fundamental tension:** IS corrects the distribution mismatch but amplifies variance

$$V_\text{WIS}(s) = \frac{\sum_t \rho_t G_t}{\sum_t \rho_t}$$

<!-- Speaker notes: Close with the three-part takeaway. (1) The IS ratio formula — students should be able to write it and explain each term. (2) Ordinary vs. weighted IS tradeoff — unbiased but dangerous vs. biased but stable. (3) The practical choice: always use weighted IS unless you have a specific theoretical reason to need unbiasedness. Preview Module 3: TD methods bypass IS entirely for single-step returns, making off-policy much more practical at scale. -->
