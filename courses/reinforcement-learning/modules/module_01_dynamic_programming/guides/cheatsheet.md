# Dynamic Programming Cheatsheet — Module 1

> **Reading time:** ~7 min | **Module:** 1 — Dynamic Programming | **Prerequisites:** Module 0

## Notation Reference

| Symbol | Meaning |
|---|---|
| $s, s'$ | Current state, successor state |
| $a$ | Action |
| $\pi$ | Policy ($\pi(a\|s)$ = probability of action $a$ in state $s$) |
| $V, V^\pi, V^*$ | Value function (general, under $\pi$, optimal) |
| $Q, Q^\pi, Q^*$ | Action-value function (general, under $\pi$, optimal) |
| $\gamma \in [0,1)$ | Discount factor |
| $p(s', r \mid s, a)$ | Environment dynamics (probability of next state $s'$ and reward $r$) |
| $\mathcal{T}^\pi$ | Bellman expectation operator |
| $\mathcal{T}^*$ | Bellman optimality operator |
| $\theta$ | Convergence threshold |

---

## Core Update Rules

### Policy Evaluation (Bellman Expectation)

<div class="callout-key">

<strong>Key Point:</strong> ### Policy Evaluation (Bellman Expectation)

$$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma V_k(s')\bigr]$$

Applied iteratively until $\max_s |V_{k+1}(s) - V_k(s)| < \...

</div>


$$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma V_k(s')\bigr]$$

Applied iteratively until $\max_s |V_{k+1}(s) - V_k(s)| < \theta$.

**Fixed point:** $V^\pi$ — the value function for policy $\pi$.

---

### Policy Improvement (Greedy Step)

$$\pi'(s) = \arg\max_a Q^\pi(s, a)$$

where the action-value function is:

$$Q^\pi(s, a) = \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma V^\pi(s')\bigr]$$

**Guarantee (Policy Improvement Theorem):** $V^{\pi'}(s) \geq V^\pi(s)$ for all $s$.

**Optimality condition:** If $\pi'(s) = \pi(s)$ for all $s$, then $V^\pi = V^*$ and $\pi = \pi^*$.

---

### Value Iteration (Bellman Optimality)

$$V_{k+1}(s) = \max_a \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma V_k(s')\bigr]$$

Applied iteratively until convergence. Extract policy after convergence:

$$\pi^*(s) = \arg\max_a \sum_{s', r} p(s', r \mid s, a)\bigl[r + \gamma V(s')\bigr]$$

**Fixed point:** $V^*$ — the optimal value function.

---

## Policy Iteration vs Value Iteration

| Property | Policy Iteration | Value Iteration |
|---|---|---|
| **Update rule** | $\mathcal{T}^\pi$ (expectation) | $\mathcal{T}^*$ (max) |
| **Phases** | Evaluation + Improvement alternating | Fused into single loop |
| **Evaluation depth** | Full convergence (or $m$ steps) | 1 sweep per outer step |
| **Convergence type** | Finite (exact: reaches $\pi^*$ in finitely many iterations) | Asymptotic (error $\to 0$ but never exactly 0) |
| **Iterations to $\pi^*$** | Few (often $< 20$ for moderate MDPs) | N/A (no finite termination) |
| **Cost per iteration** | High (evaluation dominates) | Low (single sweep) |
| **Total sweeps** | Comparable | Comparable |
| **Preferred when** | $\gamma$ close to 1, small $\|\mathcal{S}\|$ | Large $\|\mathcal{S}\|$, simple implementation |

**Unification:** Both are special cases of modified policy iteration with $m$ evaluation sweeps.
- $m = 1$ sweep $\to$ value iteration
- $m = \infty$ sweeps (full convergence) $\to$ standard policy iteration

---

## Convergence Guarantees

All three algorithms rely on the **contraction mapping theorem** (Banach fixed-point theorem):

<div class="callout-key">

<strong>Key Point:</strong> All three algorithms rely on the **contraction mapping theorem** (Banach fixed-point theorem):

| Operator | Contraction factor | Fixed point |
|---|---|---|
| $\mathcal{T}^\pi$ | $\gamma$ | $V^\pi$ |...

</div>


| Operator | Contraction factor | Fixed point |
|---|---|---|
| $\mathcal{T}^\pi$ | $\gamma$ | $V^\pi$ |
| $\mathcal{T}^*$ | $\gamma$ | $V^*$ |

**Error bound after $k$ sweeps:**

$$\|V_k - V^*\|_\infty \leq \frac{\gamma^k}{1-\gamma}\|V_1 - V_0\|_\infty$$

**Corrected stopping rule for $\epsilon$ accuracy:**

$$\theta = \frac{\epsilon(1-\gamma)}{\gamma}$$

Do not use $\theta = \epsilon$ directly — the actual error is up to $\frac{\gamma}{1-\gamma} \times \delta_k$ larger.

---

## When DP Works vs When It Doesn't

### DP Requirements

<div class="callout-info">

<strong>Info:</strong> ### DP Requirements

| Requirement | Description |
|---|---|
| **Full model** | Must know $p(s', r \mid s, a)$ for all $(s, a, s')$ |
| **Finite state space** | Must enumerate all states $\mathcal{S}$...

</div>


| Requirement | Description |
|---|---|
| **Full model** | Must know $p(s', r \mid s, a)$ for all $(s, a, s')$ |
| **Finite state space** | Must enumerate all states $\mathcal{S}$ |
| **Finite action space** | Must enumerate all actions $\mathcal{A}$ |
| **Discount** $\gamma < 1$ | Required for contraction guarantee (or proper policies with $\gamma = 1$) |

### DP Works Well

- Inventory management, logistics, scheduling
- Board games with complete rules (chess endgame tables)
- Control of finite-state systems with known dynamics
- Any problem where the MDP model is available and the state space fits in memory

### DP Does Not Work (and What to Use Instead)

| Problem | Why DP Fails | Alternative |
|---|---|---|
| Unknown dynamics | Cannot compute expectations without $p(s',r\|s,a)$ | Monte Carlo, TD learning, Q-learning |
| Continuous state space | Cannot enumerate all states | Fitted value iteration, policy gradient |
| Very large state space ($> 10^6$ states) | Memory and compute infeasible | Approximate DP, deep RL |
| Stochastic dynamics with unknown structure | Model must be learned first | Model-based RL |

---

## Quick Reference: Algorithm Pseudocode

### Policy Evaluation

```
V ← 0 for all s
repeat:
    delta ← 0
    for each s:
        v ← sum_a pi(a|s) * sum_{s',r} p(s',r|s,a) * [r + gamma*V(s')]
        delta ← max(delta, |V(s) - v|)
        V(s) ← v
until delta < theta
```

### Policy Iteration

```
pi ← arbitrary
repeat:
    V ← policy_evaluation(pi)          # run above until convergence
    policy_stable ← True
    for each s:
        old_a ← pi(s)
        pi(s) ← argmax_a sum_{s',r} p(s',r|s,a) * [r + gamma*V(s')]
        if old_a ≠ pi(s): policy_stable ← False
until policy_stable
```

### Value Iteration

```
V ← 0 for all s
repeat:
    delta ← 0
    for each s:
        v ← max_a sum_{s',r} p(s',r|s,a) * [r + gamma*V(s')]
        delta ← max(delta, |V(s) - v|)
        V(s) ← v
until delta < theta
pi*(s) ← argmax_a sum_{s',r} p(s',r|s,a) * [r + gamma*V(s')]  # extract once
```

---

## Common Errors at a Glance

| Error | Symptom | Fix |
|---|---|---|
| Missing $\gamma$ in update | Values diverge or grow unboundedly | Always multiply $V(s')$ by $\gamma$ |
| $\theta$ too large | Policy converges to suboptimal | Use $\theta \leq 10^{-8}$ for small MDPs |
| Wrong stopping condition (value vs policy) | Policy iteration stops early | Check $\pi_{k+1}(s) = \pi_k(s)$, not value change |
| Not handling terminal states | Terminal state values drift from 0 | Set $V(\text{terminal}) = 0$; exclude from updates |
| Transposed dynamics array | Values converge to wrong answer (no crash) | Verify: $\text{P}[s, a, s'] = p(s' \mid s, a)$ |
| Extracting policy inside value iteration loop | Suboptimal intermediate policy returned | Extract after convergence only |
| $\gamma = 1$ without terminal states | Infinite loop, no convergence | Require $\gamma < 1$ or proper episodic structure |

---

*Reference: Sutton & Barto (2018), Reinforcement Learning: An Introduction, 2nd ed., Chapter 4*
