# Module 3: Temporal Difference Learning — Cheatsheet

## Core Update Rules

### TD(0) Prediction

Evaluate a fixed policy $\pi$ by updating state values after every step.

$$V(S_t) \leftarrow V(S_t) + \alpha \bigl[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\bigr]$$

**TD error:** $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$

Compact form: $V(S_t) \leftarrow V(S_t) + \alpha \, \delta_t$

---

### SARSA (On-Policy TD Control)

Update action values using the *actual next action* from the behavior policy.

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \bigl[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\bigr]$$

- $A_{t+1}$ is sampled from the **current $\varepsilon$-greedy policy** (on-policy)
- Converges to the optimal $\varepsilon$-greedy policy

---

### Q-Learning (Off-Policy TD Control)

Update action values using the *greedy best action* regardless of what was actually taken.

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \bigl[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)\bigr]$$

- Uses $\max_a$ in the target (off-policy)
- Converges to $Q^*$ under sufficient exploration (Watkins & Dayan, 1992)

---

### Expected SARSA

Replace the sampled $Q(S_{t+1}, A_{t+1})$ with the expectation over the current policy.

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[R_{t+1} + \gamma \sum_{a} \pi(a \mid S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t)\right]$$

- Reduces variance compared to SARSA
- When $\pi$ is greedy: Expected SARSA $=$ Q-learning

---

### Double Q-Learning

Decouple action selection from action evaluation to remove maximization bias.

$$Q_1(S_t, A_t) \leftarrow Q_1(S_t, A_t) + \alpha \Bigl[R_{t+1} + \gamma Q_2\bigl(S_{t+1}, \arg\max_a Q_1(S_{t+1}, a)\bigr) - Q_1(S_t, A_t)\Bigr]$$

Update $Q_2$ symmetrically (swap $Q_1 \leftrightarrow Q_2$) with probability 0.5.

- $Q_1$ selects the action (argmax)
- $Q_2$ evaluates the selected action
- Independence between $Q_1$ and $Q_2$ eliminates upward bias

---

### n-Step Return

Look $n$ steps ahead before bootstrapping.

$$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V(S_{t+n})$$

Update: $V(S_t) \leftarrow V(S_t) + \alpha \bigl[G_t^{(n)} - V(S_t)\bigr]$

Special cases: $n=1$ → TD(0), $n=\infty$ → Monte Carlo

---

### TD(λ): λ-Return (Forward View)

Exponentially weighted average of all n-step returns.

$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

Special cases: $\lambda=0$ → TD(0), $\lambda=1$ → Monte Carlo

---

### TD(λ): Eligibility Trace (Backward View)

Online implementation of the λ-return via a fading-memory credit vector.

**Trace update (accumulating):**
$$e_t(s) = \gamma \lambda \, e_{t-1}(s) + \mathbf{1}(S_t = s)$$

**Value update (all states):**
$$V(s) \leftarrow V(s) + \alpha \, \delta_t \, e_t(s), \quad \forall s$$

Correct update order per step:
1. Compute $\delta_t$
2. Spike $e_t(S_t) \mathrel{+}= 1$
3. Update all $V(s) \mathrel{+}= \alpha \delta_t e_t(s)$
4. Decay $e_t \mathrel{*}= \gamma\lambda$
5. Reset $e$ to 0 at episode start

---

## Algorithm Comparison: TD vs MC vs DP

| Property | Monte Carlo | TD(0) | Dynamic Programming |
|-----------|-------------|-------|---------------------|
| **Update target** | Full return $G_t$ | $R_{t+1} + \gamma V(S_{t+1})$ | Bellman expectation (exact) |
| **Bootstraps?** | No | Yes | Yes |
| **Model required?** | No | No | Yes |
| **Online updates?** | No (end of episode) | Yes (every step) | No (full sweep) |
| **Episodic only?** | Yes | No | No |
| **Bias** | None | Yes (from bootstrap) | None |
| **Variance** | High | Low | Zero |
| **Data efficient?** | No | Yes | N/A |

---

## On-Policy vs Off-Policy: Quick Reference

| | SARSA | Q-Learning |
|-|-------|------------|
| **Policy type** | On-policy | Off-policy |
| **Behavior policy** | $\varepsilon$-greedy | $\varepsilon$-greedy (any policy works) |
| **Target policy** | $\varepsilon$-greedy (same as behavior) | Greedy $\pi^* = \arg\max_a Q$ |
| **Bootstrap target** | $Q(S', A')$ — actual next action | $\max_a Q(S', a)$ — greedy max |
| **Converges to** | Optimal $\varepsilon$-greedy $Q^\varepsilon$ | Optimal $Q^*$ |
| **Cliff walking** | Safe upper path | Optimal cliff-edge path |
| **Training reward** | Higher (avoids exploration accidents) | Lower (cliff falls during training) |
| **Deployment reward** | Slightly suboptimal | Optimal |
| **Use when** | Exploration is dangerous or costly | Exploration is cheap; want optimal policy |
| **Key reference** | Rummery & Niranjan (1994) | Watkins & Dayan (1992) |

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $S_t, s$ | State at time $t$ |
| $A_t, a$ | Action at time $t$ |
| $R_{t+1}, r$ | Reward received after taking $A_t$ in $S_t$ |
| $\gamma \in [0,1)$ | Discount factor |
| $\alpha \in (0,1]$ | Step size (learning rate) |
| $\varepsilon \in [0,1]$ | Exploration probability |
| $\lambda \in [0,1]$ | TD(λ) parameter |
| $V^\pi(s)$ | State-value function under policy $\pi$ |
| $Q^\pi(s,a)$ | Action-value function under policy $\pi$ |
| $Q^*(s,a)$ | Optimal action-value function |
| $\pi^*(s)$ | Optimal policy: $\arg\max_a Q^*(s,a)$ |
| $\delta_t$ | TD error at time $t$ |
| $e_t(s)$ | Eligibility trace for state $s$ at time $t$ |
| $G_t$ | Return from time $t$: $\sum_{k=0}^\infty \gamma^k R_{t+k+1}$ |
| $G_t^{(n)}$ | n-step return from time $t$ |
| $G_t^\lambda$ | λ-return from time $t$ |

---

## Common Pitfalls at a Glance

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Terminal state bootstrap | V values wrong near episode end | Set `next_value = 0` when `terminated` |
| SARSA vs Q-learning confusion | Wrong path on cliff walking | Check: on-policy uses $Q(S', A')$; off-policy uses $\max_a Q(S',a)$ |
| Carry-forward missing (SARSA) | Updates use re-sampled action | Initialize $A_0$ before loop; carry $A_{t+1}$ to next iteration |
| Traces not reset per episode | TD(λ) assigns cross-episode credit | Initialize `e = zeros` inside episode loop |
| Wrong trace-update order | Current state gets delayed credit | Spike $e[S_t]$, then update $V$, then decay $e$ |
| Large $\alpha$ with TD | Oscillations, no convergence | Use $\alpha \leq 0.1$; for convergence use $\alpha_t = 1/N(S_t)$ |
| Maximization bias | Q-values overestimated | Use Double Q-learning when action space $\geq 10$ actions |
| Fixed $\varepsilon$, expecting $\pi^*$ | Converges to $\varepsilon$-greedy, not optimal | Decay $\varepsilon \to 0$ (GLIE) for guaranteed convergence to $\pi^*$ |

---

## References

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.) — Ch. 6 (TD), Ch. 7 (n-step), Ch. 12 (eligibility traces)
- Watkins & Dayan (1992). *Q-learning.* Machine Learning 8(3-4) — convergence proof for Q-learning
- Rummery & Niranjan (1994). *On-line Q-learning using connectionist systems* — original SARSA paper
- van Hasselt (2010). *Double Q-learning.* NeurIPS — maximization bias and its fix
- Schulman et al. (2016). *High-dimensional continuous control using generalized advantage estimation.* ICLR — TD(λ) in modern deep RL
