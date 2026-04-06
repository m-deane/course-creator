# Module 9 Cheatsheet: Frontiers & Applications

> **Reading time:** ~8 min | **Module:** 9 — Frontiers | **Prerequisites:** Modules 5-8

---

## 1. MARL Taxonomy

| Setting | Reward Structure | Solution Concept | Key Algorithms |
|---------|-----------------|-----------------|----------------|
| **Cooperative** | $R^1 = \cdots = R^n$ (shared) | Social welfare optimum | MAPPO, QMIX, COMA |
| **Competitive** (zero-sum) | $R^1 + R^2 = 0$ | Nash Equilibrium (minimax) | Self-play, NFSP |
| **Mixed** (general-sum) | $R^i$ independent | Nash Equilibrium (no guarantee of efficiency) | MADDPG, PSRO |

### CTDE Pattern (Cooperative MARL)

```
Training:  Centralized critic Q(s, a¹, ..., aⁿ) uses all observations + actions
Execution: Decentralized actor πⁱ(aⁱ | oⁱ) uses only own observation
```

### MARL Challenge Checklist

- [ ] Non-stationarity — other agents are learning simultaneously
- [ ] Credit assignment — who caused the joint reward?
- [ ] Scalability — joint action space is $\prod_i |\mathcal{A}^i|$
- [ ] Partial observability — each agent sees only $o^i \neq s$
- [ ] Communication bandwidth — if agents can talk, at what cost?

---

## 2. Offline RL Key Algorithms

### The Core Problem

<div class="callout-key">

<strong>Key Point:</strong> ### The Core Problem

Behavior policy $\pi_\beta$ induces dataset $\mathcal{D}$.

</div>


Behavior policy $\pi_\beta$ induces dataset $\mathcal{D}$. Learned policy $\pi$ may visit $(s, a) \notin \text{supp}(d^{\pi_\beta})$.

Naive off-policy RL on $\mathcal{D}$ $\Rightarrow$ Q-value overestimation for OOD actions $\Rightarrow$ policy exploits OOD actions $\Rightarrow$ catastrophic deployment failure.

### Algorithm Comparison

| Algorithm | Core Idea | OOD Handling | Stitching |
|-----------|-----------|-------------|-----------|
| **CQL** | Penalize Q for OOD actions; push Q down for $\mu$, up for $\mathcal{D}$ | Pessimistic Q lower bound | Yes |
| **IQL** | Expectile regression for $V(s)$; never query OOD actions in backup | Avoids OOD by construction | Yes |
| **Decision Transformer** | RL as sequence modeling; no Bellman backup at all | No Q-function; no OOD queries | No |

### CQL Loss (simplified)

$$\mathcal{L}_{\text{CQL}} = \alpha \underbrace{(\mathbb{E}_\mu[Q(s,a)] - \mathbb{E}_\mathcal{D}[Q(s,a)])}_{\text{conservative penalty}} + \frac{1}{2}\mathbb{E}_\mathcal{D}\left[(Q - \mathcal{B}^\pi\hat{Q})^2\right]$$

### IQL Value Loss

$$\mathcal{L}_V = \mathbb{E}_{(s,a)\sim\mathcal{D}}\left[\mathcal{L}_2^\tau(Q(s,a) - V(s))\right], \quad \mathcal{L}_2^\tau(u) = |\tau - \mathbf{1}(u<0)|\cdot u^2$$

- $\tau \to 1$: $V(s) \to \max_{a \in \mathcal{D}} Q(s, a)$ (in-dataset max, no OOD)
- Typical $\tau \in [0.6, 0.9]$

### Decision Transformer Sequence

$$(\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \ldots), \quad \hat{R}_t = \sum_{t'=t}^T r_{t'}$$

At inference: condition on desired return $R_{\text{target}}$; Transformer generates actions.

---

## 3. RLHF Three-Step Pipeline

```
Step 1: Supervised Fine-Tuning (SFT)
  Data:  (prompt x, demonstration y) pairs
  Loss:  L_SFT = -Σ log π_θ(y | x)
  Goal:  Initialize policy in good output region

```

<div class="callout-key">

<strong>Key Point:</strong> ### DPO Alternative

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l...

</div>

```
Step 2: Reward Model Training
  Data:  (prompt x, preferred y_w, rejected y_l) triples
  Model: Bradley-Terry  P(y_w ≻ y_l | x) = σ(r_φ(x, y_w) - r_φ(x, y_l))
  Loss:  L_RM = -E[log σ(r_φ(x, y_w) - r_φ(x, y_l))]
  Goal:  Scalar proxy for human preferences

Step 3: PPO Fine-Tuning with KL Penalty
  Objective: max_π E[r_φ(x,y) - β · KL[π(·|x) ‖ π_ref(·|x)]]
  KL penalty prevents: reward hacking, loss of language quality
  β range: 0.01 – 0.5 (higher = more conservative)
```

### DPO Alternative

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

No reward model. No PPO. Single supervised pass on preference pairs. Often competitive with PPO-based RLHF.

---

## 4. Safe RL Constraint Formulation

### Constrained MDP (CMDP)

<div class="callout-info">

<strong>Info:</strong> ### Constrained MDP (CMDP)

$$\max_\pi \; J_R(\pi) \quad \text{subject to} \quad J_{C_k}(\pi) \leq d_k, \quad k = 1, \ldots, K$$

$$J_{C_k}(\pi) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t C_k(S_...

</div>


$$\max_\pi \; J_R(\pi) \quad \text{subject to} \quad J_{C_k}(\pi) \leq d_k, \quad k = 1, \ldots, K$$

$$J_{C_k}(\pi) = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t C_k(S_t, A_t)\right]$$

**Hard constraint** (never violate) vs **soft penalty** $R - \lambda C$ (may violate for high reward):

| Formulation | Constraint guaranteed? |
|-------------|----------------------|
| Soft penalty | No |
| Hard CMDP | Yes (by definition of feasible solution) |

### Lagrangian Primal-Dual

$$\mathcal{L}(\pi, \lambda) = J_R(\pi) - \lambda(J_C(\pi) - d)$$

```
Primal:  π ← argmax_π L(π, λ)          [standard RL with r - λ·cost]
Dual:    λ ← max(0, λ + α_λ(J_C(π) - d)) [increase if violated, decrease if slack]
```

### Risk Measures

| Measure | Formula | Interpretation |
|---------|---------|---------------|
| CVaR$_\alpha$ | $\mathbb{E}[G \mid G \leq \text{VaR}_\alpha(G)]$ | Expected return in worst $\alpha$ fraction |
| Minimax | $\min_{\mathcal{P}' \in \mathcal{U}} \mathbb{E}_{\mathcal{P}'}[G]$ | Worst case over environment uncertainty set |

---

## 5. RL for Trading: Environment Design Checklist

### State Checklist
- [ ] Price returns (not levels) — normalized with rolling Z-score
- [ ] Current portfolio weights and cash fraction
- [ ] Volume indicators (volume ratio, VWAP)
- [ ] Technical indicators (RSI, MACD, Bollinger Bands, ATR)
- [ ] Macro signals (VIX, yield spread, FX rates)
- [ ] No look-ahead — all features use strictly past data (`.shift(1)`)

### Action Space Checklist
- [ ] Discrete (DQN): {strong sell, sell, hold, buy, strong buy} or sized variants
- [ ] Continuous (SAC/PPO): portfolio weights $w \in \Delta^n$, or weight changes $\Delta w$
- [ ] Action constraints: long-only if required, max position limits

### Reward Checklist
- [ ] NOT raw P&L (encourages max leverage)
- [ ] Differential Sharpe OR P&L minus transaction costs OR risk-penalized P&L
- [ ] Transaction costs included ($\kappa \cdot \sum_i |w^i - w^i_{\text{prev}}|$)
- [ ] Reward normalized to unit variance during training

### Backtesting Checklist
- [ ] Walk-forward evaluation (strict train/test separation)
- [ ] No look-ahead in feature computation
- [ ] Point-in-time data (includes delistings, survivorship bias corrected)
- [ ] Realistic transaction costs (spread + market impact)
- [ ] Performance reported per evaluation window (mean $\pm$ std Sharpe)

---

## 6. Research Frontiers Quick Reference

| Topic | Key Question | Leading Approaches (as of 2025) |
|-------|-------------|--------------------------------|
| **MARL** | How do agents coordinate without central control? | CTDE (MAPPO, QMIX), mean field, communication protocols |
| **Offline RL** | Can we learn good policies from fixed data? | IQL, CQL, Decision Transformer, offline policy evaluation |
| **RLHF** | How do we align AI with human values? | PPO + KL, DPO, Constitutional AI, RLAIF |
| **Safe RL** | How do we guarantee constraint satisfaction? | CMDPs, Lagrangian methods, CPO, risk-sensitive RL |
| **RL for Trading** | Can RL outperform supervised strategies? | SAC/PPO with differential Sharpe, transaction cost modeling |
| **Foundation Models + RL** | Can pretrained models accelerate RL? | Decision Transformer, Gato, GROOT, VPT |
| **Sim-to-Real** | How do we transfer RL policies to the real world? | Domain randomization, adversarial training, online adaptation |

---

## 7. Notation Summary (consistent with Module 0)

| Symbol | Meaning |
|--------|---------|
| $s \in \mathcal{S}$ | State |
| $a \in \mathcal{A}$ | Action |
| $\pi(a \mid s)$ | Policy |
| $R(s, a, s')$ | Reward function |
| $G_t = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$ | Return |
| $\gamma \in [0,1)$ | Discount factor |
| $\tau$ | Trajectory $(S_0, A_0, R_1, S_1, \ldots)$ |
| $\boldsymbol{\pi} = (\pi^1, \ldots, \pi^n)$ | Joint policy (MARL) |
| $\mathbf{a} = (a^1, \ldots, a^n)$ | Joint action (MARL) |
| $\pi_\beta$ | Behavior policy (offline RL) |
| $\mathcal{D}$ | Offline dataset |
| $r_\phi(x, y)$ | Reward model (RLHF) |
| $\pi_{\text{ref}}$ | SFT reference policy (RLHF) |
| $J_{C_k}(\pi)$ | Expected cumulative cost $k$ (Safe RL) |
| $d_k$ | Constraint threshold for cost $k$ |
| $\lambda$ | Lagrange multiplier (Safe RL) |
| $w \in \Delta^n$ | Portfolio weight vector (trading) |
| $\kappa$ | Transaction cost rate (trading) |
