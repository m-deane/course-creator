# Module 2 Cheatsheet — Monte Carlo Methods

> **Reading time:** ~8 min | **Module:** 2 — Monte Carlo Methods | **Prerequisites:** Module 1

## Notation

| Symbol | Meaning |
|---|---|
| $s, s'$ | States in $\mathcal{S}$ |
| $a$ | Action in $\mathcal{A}(s)$ |
| $\pi$ | Target policy: $\mathcal{S} \to \Delta(\mathcal{A})$ |
| $b$ | Behavior policy (off-policy) |
| $G_t$ | Return from time $t$: $\sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$ |
| $\gamma$ | Discount factor $\in [0, 1)$ |
| $V^\pi(s)$ | State-value: $\mathbb{E}_\pi[G_t \mid S_t = s]$ |
| $Q^\pi(s, a)$ | Action-value: $\mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$ |
| $N(s)$ | Number of visits to state $s$ |
| $N(s,a)$ | Number of visits to state-action pair $(s,a)$ |
| $\rho_{t:T-1}$ | Importance sampling ratio for steps $t$ to $T-1$ |
| $\varepsilon$ | Exploration probability in $\varepsilon$-greedy |
| $T$ | Episode termination time |

---

## First-Visit vs. Every-Visit MC

| Property | First-Visit | Every-Visit |
|---|---|---|
| Which visits count | First occurrence of $s$ per episode | All occurrences of $s$ |
| Bias | Unbiased | Biased (finite samples) |
| Samples per episode | At most 1 per state | Up to $T$ per state |
| Sample independence | i.i.d. across episodes | Correlated within episode |
| Convergence | Almost sure (S&B Theorem 5.1) | Almost sure asymptotically |
| Preferred for | Theoretical analysis | Data-efficient implementation |
| First-visit check | `if s not in visited_this_episode` | Omit the check |

---


<div class="callout-insight">

<strong>Insight:</strong> across episodes | Correlated within episode |
| Convergence | Almost sure (S&B Theorem 5.1) | Almost sure asymptotically |
| Preferred for | Theoretical analysis | Data-efficient implementation |
| Fi...

</div>
## MC Prediction Algorithm (First-Visit)

**Estimates $V^\pi$ from episodes.**

<div class="callout-key">

<strong>Key Point:</strong> **Estimates $V^\pi$ from episodes.**


**Key equations:**

$$G_t = R_{t+1} + \gamma G_{t+1} \quad \text{(backward recursion)}$$

$$V(s) \leftarrow V(s) + \frac{1}{N(s)}\bigl[G - V(s)\bigr] \quad \tex...

</div>


```
Initialize: V(s) = 0, N(s) = 0 for all s

For each episode:
  Generate S₀, A₀, R₁, S₁, A₁, R₂, ..., S_{T-1}, A_{T-1}, R_T using π

  G = 0
  visited = {}

  For t = T-1, T-2, ..., 0:
    G ← γG + R_{t+1}
    If S_t ∉ visited:
      visited.add(S_t)
      N(S_t) += 1
      V(S_t) += [G − V(S_t)] / N(S_t)     ← incremental mean
```

**Key equations:**

$$G_t = R_{t+1} + \gamma G_{t+1} \quad \text{(backward recursion)}$$

$$V(s) \leftarrow V(s) + \frac{1}{N(s)}\bigl[G - V(s)\bigr] \quad \text{(incremental update)}$$

---

## On-Policy MC Control Algorithm ($\varepsilon$-Greedy)

**Estimates $Q^*$ and improves $\pi$ using the agent's own episodes.**

<div class="callout-info">

<strong>Info:</strong> **Estimates $Q^*$ and improves $\pi$ using the agent's own episodes.**


**$\varepsilon$-greedy policy:**

$$\pi(a \mid s) = \begin{cases} 1 - \varepsilon + \varepsilon/|\mathcal{A}| & a = \arg\max_{...

</div>


```
Initialize: Q(s,a) = 0, N(s,a) = 0, π = uniform ε-soft

For each episode:
  Generate S₀,A₀,R₁,...,S_{T-1},A_{T-1},R_T using π (ε-greedy)

  G = 0
  visited = {}

  For t = T-1, T-2, ..., 0:
    G ← γG + R_{t+1}
    If (S_t, A_t) ∉ visited:
      visited.add((S_t, A_t))
      N(S_t, A_t) += 1
      Q(S_t, A_t) += [G − Q(S_t, A_t)] / N(S_t, A_t)

      A* ← argmax_a Q(S_t, a)
      π(a|S_t) ← ε/|A|  for all a
      π(A*|S_t) += 1 − ε          ← ε-greedy improvement
```

**$\varepsilon$-greedy policy:**

$$\pi(a \mid s) = \begin{cases} 1 - \varepsilon + \varepsilon/|\mathcal{A}| & a = \arg\max_{a'} Q(s,a') \\ \varepsilon / |\mathcal{A}| & \text{otherwise} \end{cases}$$

---

## GLIE Condition

For convergence of on-policy MC control to $Q^*$:

<div class="callout-warning">

<strong>Warning:</strong> For convergence of on-policy MC control to $Q^*$:

**1.

</div>


**1. Infinite exploration:**
$$\lim_{k\to\infty} N_k(s,a) = \infty \quad \forall (s,a)$$

**2. Greedy in the limit:**
$$\lim_{k\to\infty} \pi_k(a \mid s) = \mathbf{1}\!\left[a = \arg\max_{a'} Q_k(s,a')\right]$$

**Standard schedule:** $\varepsilon_k = 1/k$ (theoretical); exponential decay (practical).

---

## Importance Sampling Ratio

For a trajectory segment from step $t$ to $T-1$:

<div class="callout-insight">

<strong>Insight:</strong> For a trajectory segment from step $t$ to $T-1$:

$$\boxed{\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}}$$

- Environment transition probabilities $p(s'|s,a)$ cancel in n...

</div>


$$\boxed{\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}}$$

- Environment transition probabilities $p(s'|s,a)$ cancel in numerator and denominator
- **Coverage requirement:** $\pi(a|s) > 0 \Rightarrow b(a|s) > 0$
- For deterministic $\pi$: $\rho = 0$ if any $A_k \neq \pi(S_k)$; otherwise $\rho = \prod_k 1/b(A_k|S_k)$

---

## Ordinary vs. Weighted Importance Sampling

| Property | Ordinary IS | Weighted IS |
|---|---|---|
| **Formula** | $\dfrac{\sum_t \rho_t G_t}{\|\ \mathcal{T}(s)\|}$ | $\dfrac{\sum_t \rho_t G_t}{\sum_t \rho_t}$ |
| **Bias** | Zero (unbiased) | Non-zero (consistent) |
| **Variance** | Can be infinite | Always bounded |
| **MSE** | Higher in practice | Lower in practice |
| **Use when** | $\pi \approx b$, theory | All practical applications |

**IS identity (why it works):**
$$\mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_b\!\left[\rho_{t:T-1} \cdot G_t \mid S_t = s\right]$$

---

## MC vs. DP Comparison

| Dimension | Dynamic Programming | Monte Carlo |
|---|---|---|
| **Model required?** | Yes — needs $p(s',r\|s,a)$ | No — uses sample episodes |
| **Bootstrapping?** | Yes — uses $V(s')$ in update | No — uses actual return $G_t$ |
| **Episodic only?** | No — handles continuing tasks | Yes — needs episode termination |
| **Update timing** | After each step (DP sweep) | After episode completion |
| **Bias** | Biased (bootstrap error) | Unbiased (actual return) |
| **Variance** | Low (one-step) | High (multi-step sum) |
| **Computational cost** | $O(\|\mathcal{S}\|^2 \|\mathcal{A}\|)$ per sweep | $O(T)$ per episode |
| **Optimal for** | Small MDPs with known model | Large MDPs, simulated environments |
| **Value target** | $\sum_{s',r} p(s',r\|s,a)[r + \gamma V(s')]$ | $G_t = \sum_k \gamma^k R_{t+k+1}$ |

---

## Quick Reference: Which Algorithm to Use

```
Do you have a model of p(s', r | s, a)?
  YES → Use Dynamic Programming (Module 1)
  NO  → Use Monte Carlo or TD (Modules 2-3)

Are episodes episodic (finite T)?
  NO  → Must use TD methods (Module 3)
  YES → MC is applicable

Do you need data from a different policy?
  NO  → On-policy MC control (Guide 02)
  YES → Off-policy MC with weighted IS (Guide 03)

Is variance a problem (long episodes, pi ≠ b)?
  YES → Use weighted IS or switch to TD Q-learning
  NO  → Ordinary or weighted IS both work
```

---

## Common Pitfalls at a Glance

| Pitfall | Guide | Fix |
|---|---|---|
| High variance returns | 01 | More episodes; smaller $\gamma$; weighted IS |
| Missing states (never visited) | 01 | Ensure stochastic policy; use ES |
| Off-by-one in return computation | 01 | `G = gamma * G + reward` (current reward) |
| Estimating $V$ instead of $Q$ for control | 02 | Always collect $(S, A, R)$; estimate $Q$ |
| Deterministic policy misses actions | 02 | Use $\varepsilon$-greedy; exploring starts |
| $\varepsilon$ fixed too large | 02 | Decay $\varepsilon$ with GLIE or exponential schedule |
| Ordinary IS with long episodes | 03 | Switch to weighted IS |
| Coverage violation | 03 | Use $\varepsilon$-soft behavior policy |
| Missing break when $\pi(A_t\|S_t)=0$ | 03 | Break backward loop; set $W=0$ |

---

## Key Equations Summary

$$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1} = R_{t+1} + \gamma G_{t+1}$$

$$V(s) \leftarrow V(s) + \frac{1}{N(s)}\bigl[G_t - V(s)\bigr]$$

$$Q(s,a) \leftarrow Q(s,a) + \frac{1}{N(s,a)}\bigl[G_t - Q(s,a)\bigr]$$

$$\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}$$

$$V_{\text{WIS}}(s) = \frac{\displaystyle\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}\, G_t}{\displaystyle\sum_{t \in \mathcal{T}(s)} \rho_{t:T(t)-1}}$$

All equations consistent with Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd edition, Chapter 5.
