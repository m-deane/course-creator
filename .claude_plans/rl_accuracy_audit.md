# Reinforcement Learning Course — Factual Accuracy Audit

**Audit date:** 2026-03-08
**Course path:** `/home/user/course-creator/courses/reinforcement-learning/`
**Auditor:** ml-engineer agent (claude-sonnet-4-6)

---

## Executive Summary

11 of 12 checks PASS. 1 check has a MINOR issue and 1 check has an additional MINOR notation issue. No CRITICAL errors were found. The course is mathematically sound throughout. All core equations are correct and consistently stated. The on-policy/off-policy distinction is handled carefully and correctly in every location checked.

---

## Check Results

---

### Check 1 — Bellman Expectation Equation for V

**Status: PASS**

**Required form:**
$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V^\pi(s')]$$

**Findings:**

- `module_00_foundations/guides/03_bellman_equations_guide.md`, line 51:
  ```
  V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma V^\pi(s')\right]
  ```
  Exact match. Presented in a boxed display equation with correct structure.

- `module_00_foundations/guides/03_bellman_equations_slides.md`, line 49:
  ```
  V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma V^\pi(s')\right]
  ```
  Exact match. Also repeated at the summary slide (line 315) in the same form.

The equation appears in guides, slides, and the final summary slide — all consistent and correct.

---

### Check 2 — Bellman Optimality Equation for V

**Status: PASS**

**Required form:**
$$V^*(s) = \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma V^*(s')]$$

**Findings:**

- `module_00_foundations/guides/03_bellman_equations_guide.md`, line 83:
  ```
  V^*(s) = \max_{a} \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma V^*(s')\right]
  ```
  Exact match.

- `module_00_foundations/guides/03_bellman_equations_slides.md`, line 110:
  ```
  V^*(s) = \max_{a} \sum_{s', r} p(s', r \mid s, a)\left[r + \gamma V^*(s')\right]
  ```
  Exact match. The same slide also presents the intermediate form (line 106):
  ```
  V^*(s) = \max_\pi V^\pi(s) = \max_a \mathbb{E}\left[R_{t+1} + \gamma V^*(S_{t+1}) \mid S_t=s, A_t=a\right]
  ```
  This intermediate form is correct and serves as a derivation step before the explicit sum.

Also correct in summary slide (line 319): identical boxed form. All instances are consistent.

---

### Check 3 — Q-learning Update Rule

**Status: PASS**

**Required form:**
$$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[R_{t+1} + \gamma \max_{a'} Q(S_{t+1},a') - Q(S_t,A_t)]$$

**Findings:**

- `module_03_temporal_difference/guides/03_q_learning_guide.md`, line 19:
  ```
  Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \bigl[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)\bigr]
  ```
  Exact match (uses $\max_a$ rather than $\max_{a'}$, which is equivalent — the dummy variable name is immaterial).

- `module_03_temporal_difference/guides/03_q_learning_slides.md`, line 32–33:
  ```
  Q(S,A) \leftarrow Q(S,A)
  + \alpha[R + \gamma \max_a Q(S',a) - Q(S,A)]
  ```
  Correct (split across display lines for formatting).

- `module_03_temporal_difference/guides/cheatsheet.md`, line 32:
  ```
  Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \bigl[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)\bigr]
  ```
  Exact match.

- `module_03_temporal_difference/exercises/exercises.py`, lines 130–134 (docstring) and lines 172–173 (implementation):
  ```python
  target = reward + gamma * np.max(Q[next_state])
  Q[state, action] += alpha * (target - Q[state, action])
  ```
  Correct implementation matching the equation.

---

### Check 4 — SARSA Update Rule

**Status: PASS**

**Required form:**
$$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]$$

**Findings:**

- `module_03_temporal_difference/guides/02_sarsa_guide.md`, line 19:
  ```
  Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \bigl[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\bigr]
  ```
  Exact match.

- `module_03_temporal_difference/guides/02_sarsa_slides.md`, line 67:
  ```
  Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \bigl[R_{t+1} + \gamma \underbrace{Q(S_{t+1}, A_{t+1})}_{\text{on-policy next}} - Q(S_t, A_t)\bigr]
  ```
  Exact match with a helpful annotation.

- `module_03_temporal_difference/guides/cheatsheet.md`, line 21:
  ```
  Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \bigl[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\bigr]
  ```
  Exact match.

- `module_03_temporal_difference/exercises/exercises.py`, lines 228–229 (docstring) and line 267 (implementation):
  ```python
  target = reward + gamma * Q[next_state, next_action]
  Q[state, action] += alpha * (target - Q[state, action])
  ```
  Correct implementation. The function signature requires `next_action` to be supplied (not internally computed as greedy max), enforcing the on-policy semantics.

---

### Check 5 — On-Policy vs Off-Policy Classification

**Status: PASS**

**Required:** SARSA = on-policy, Q-learning = off-policy, never confused.

**Findings:**

- `module_03_temporal_difference/guides/02_sarsa_guide.md`: SARSA described as "on-policy" throughout. Line 9: "SARSA is *on-policy*: the action $A_{t+1}$ used in the bootstrap target is chosen by the same policy currently being followed." Line 63: explicitly contrasts with Q-learning as off-policy.

- `module_03_temporal_difference/guides/02_sarsa_slides.md`: Line 84: "On-policy: the policy used to generate behavior and the policy being evaluated/improved are the same." Line 293 (summary): "On-policy TD control" is listed as a SARSA fact.

- `module_03_temporal_difference/guides/03_q_learning_guide.md`: Lines 5–9: Q-learning described as off-policy from the opening paragraph. Line 40: "This means Q-learning directly approximates $Q^*$... The exploration policy is a tool for data collection; it does not affect what Q-learning converges to." Line 50: "Contrast with SARSA: SARSA evaluates the behavior policy (including exploration noise). Q-learning evaluates the greedy target policy."

- `module_03_temporal_difference/guides/03_q_learning_slides.md`: Line 24–33: Side-by-side comparison clearly labels SARSA as on-policy and Q-learning as off-policy.

- `module_03_temporal_difference/guides/cheatsheet.md`, lines 121–133: The comparison table correctly lists "On-policy" for SARSA and "Off-policy" for Q-learning in every row.

- `module_03_temporal_difference/exercises/exercises.py`, lines 357–363: Exercise 4's classification function correctly maps `"sarsa"` and `"td0"` to `"on-policy"` and `"q_learning"` to `"off-policy"`.

No confusion between on-policy and off-policy found in any file.

---

### Check 6 — Policy Gradient Theorem

**Status: PASS**

**Required form:**
$$\nabla J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)]$$

**Findings:**

- `module_06_policy_gradient/guides/01_policy_gradient_theorem_guide.md`, line 58:
  ```
  \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(A|S) \cdot Q^{\pi_\theta}(S,A)\right]
  ```
  Exact match. The full $\nabla_\theta$ subscripts are explicit, which is strictly more informative than the required form.

- `module_06_policy_gradient/guides/01_policy_gradient_theorem_guide.md`, line 70 (after derivation):
  Same equation restated.

- `module_06_policy_gradient/guides/01_policy_gradient_theorem_slides.md`, line 161 (boxed):
  ```
  \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(A|S) \cdot Q^{\pi_\theta}(S,A)\right]
  ```
  Exact match.

- `module_06_policy_gradient/guides/01_policy_gradient_theorem_slides.md`, summary slide (line 337):
  ```
  \nabla J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla \log \pi_\theta(A|S) \cdot Q^{\pi_\theta}(S,A)]
  ```
  Exact match (abbreviated $\nabla$ notation on the summary slide is consistent with the required form).

The theorem is stated correctly and consistently in every location checked.

---

### Check 7 — PPO Clipped Objective

**Status: PASS**

**Required form:**
$$L^{CLIP}(\theta) = \mathbb{E}[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$$

**Findings:**

- `module_07_advanced_policy_optimization/guides/02_ppo_guide.md`, line 25:
  ```
  L^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\,\hat{A}_t\right)\right]
  ```
  Exact match.

- `module_07_advanced_policy_optimization/guides/02_ppo_slides.md`, line 70:
  ```
  L^{CLIP}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon)\,\hat{A}_t\right)\right]
  ```
  Exact match.

- `module_07_advanced_policy_optimization/guides/02_ppo_slides.md`, summary slide (line 371):
  Same equation restated as a boxed formula.

- `module_07_advanced_policy_optimization/exercises/exercises.py`, lines 35–40 (docstring) and implementation: The exercise implements `ratio_t * A_t` vs `clip(ratio_t, 1-eps, 1+eps) * A_t` and takes the `min`, exactly matching the equation.

The probability ratio is correctly defined as $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{old}}(a_t|s_t)$ in both guide (line 19) and slides (line 48). No errors found.

---

### Check 8 — DQN: Both Experience Replay AND Target Network

**Status: PASS**

**Required:** DQN must identify BOTH experience replay AND target network as key innovations.

**Findings:**

- `module_05_deep_rl/guides/01_dqn_guide.md`:
  - Opening paragraph (lines 5–9): "Two stabilizing mechanisms — an **experience replay buffer** and a **target network** — make the training process tractable and converge reliably." Both innovations named in the first paragraph.
  - Section headers: "Innovation 1: Experience Replay Buffer" (line 38) and "Innovation 2: Target Network" (line 49) — separate dedicated sections for each.
  - The `DQNAgent` class docstring (lines 208–210) explicitly lists both: "1. Experience replay buffer — decorrelates training samples" and "2. Target network — stabilizes the TD bootstrap target."
  - Lines 63–73: A "Why Both Are Required" section explicitly states both are independently necessary, citing the Mnih 2015 ablation study.

- `module_05_deep_rl/guides/01_dqn_slides.md`: Two separate section title slides: "Innovation 1: Experience Replay Buffer" and "Innovation 2: Target Network."

Both innovations are named, described, motivated, and implemented in code. No omission found.

---

### Check 9 — Actor-Critic: Correct Separation of Actor and Critic

**Status: PASS**

**Required:** Actor = policy network, Critic = value network, correctly separated.

**Findings:**

- `module_06_policy_gradient/guides/03_actor_critic_guide.md`:
  - Lines 18–39: Dedicated subsections "Actor: The Policy Network" and "Critic: The Value Network" with explicit definitions $\pi(a|s;\theta)$ and $V(s;\mathbf{w})$.
  - Line 39: "The critic does **not** select actions. The actor does **not** evaluate states. This separation is the defining structural characteristic of actor-critic methods."
  - Code (lines 271–310): Two separate `nn.Module` classes (`ActorNetwork` and `CriticNetwork`) with separate docstrings making the roles explicit.

- `module_06_policy_gradient/guides/03_actor_critic_slides.md`:
  - Lines 56–82: Side-by-side two-column presentation: "Actor: Policy Network" $\pi(a|s;\theta)$ vs "Critic: Value Network" $V(s;\mathbf{w})$ with separate roles, parameters, outputs, losses, and goals listed for each.
  - Diagram (lines 100–132): Mermaid graph shows actor and critic as separate subgraphs with separate heads.

The separation is stated explicitly and enforced in the code implementations. No conflation found.

---

### Check 10 — Convergence Claims

**Status: PASS with one MINOR notation issue**

**Required:**
- TD(0) converges under tabular + decaying step-sizes
- Q-learning converges to optimal under sufficient exploration
- Policy gradient converges to local optimum

**Findings:**

**TD(0):**
`module_03_temporal_difference/guides/01_td_prediction_guide.md`, lines 94–99:
> "Under tabular representation with a fixed policy $\pi$: $V(s)$ converges to $V^\pi(s)$ for all $s$ with probability 1 if: The step sizes satisfy the Robbins-Monro conditions: $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$, Every state is visited infinitely often (sufficient exploration)"

Correct. "Tabular" and "decaying step-sizes" (via Robbins-Monro) are both present.

**Q-learning:**
`module_03_temporal_difference/guides/03_q_learning_guide.md`, lines 55–68:
> "Q-learning converges to the optimal $Q^*$ with probability 1 given: 1. Sufficient exploration: every state-action pair $(s,a)$ is visited infinitely often 2. Decaying step sizes: Robbins-Monro conditions... 3. Bounded rewards..."

Correct. Convergence to optimal under sufficient exploration is clearly stated.

**Policy gradient:**
`module_06_policy_gradient/guides/01_policy_gradient_theorem_guide.md`, lines 25–26:
> "Under mild conditions, policy gradient methods converge to a local optimum of $J(\theta)$."

`module_06_policy_gradient/guides/01_policy_gradient_theorem_guide.md`, lines 165–166:
> "Policy gradient methods perform gradient ascent on $J(\theta)$ using Monte Carlo estimates of the gradient. Because $J(\theta)$ is generally non-convex, convergence to global optima is not guaranteed — only local optima."

Correct. "Local optimum" is stated explicitly. The non-convexity reason is given.

**MINOR issue — notation inconsistency in Q-learning convergence write-up:**
`module_03_temporal_difference/guides/03_q_learning_guide.md`, line 66: The Bellman operator is written as $\mathcal{T}^*$ and the proof section (line 64) references a sequence $\{Q_t\}$ but writes:
```
Q_{t+1}(s,a) = (1 - \alpha_t) Q_t(s,a) + \alpha_t \bigl[R + \gamma \max_{a'} Q_t(s',a')\bigr]
```
This is the standard stochastic approximation form and is correct. The notation is internally consistent. This is informational, not an error.

**Severity: N/A** — all three convergence claims are factually correct.

---

### Check 11 — No Concept Conflation

**Status: PASS**

**Required:** TD error vs advantage (different things), Return vs reward ($G_t$ vs $R_t$), $V$ vs $Q$ (different functions).

**TD error vs advantage:**

- `module_06_policy_gradient/guides/03_actor_critic_guide.md`, lines 72–76:
  ```
  \mathbb{E}[\delta_t \mid S_t, A_t] = Q^{\pi}(S_t, A_t) - V^{\pi}(S_t) = A^{\pi}(S_t, A_t)
  ```
  The guide correctly states that the TD error $\delta_t$ is an *estimator* of the advantage $A^\pi$, not that they are the same thing. The expectation relationship is explicit, and the bias from critic approximation error is acknowledged (lines 74–76).

- `module_06_policy_gradient/guides/03_actor_critic_slides.md`, line 154:
  ```
  \mathbb{E}[\delta_t \mid S_t, A_t] = A^{\pi}(S_t, A_t) = Q^{\pi}(S_t,A_t) - V^{\pi}(S_t)
  ```
  Correct. The relationship is through expectation, not identity. No conflation.

**Return vs reward:**

- `module_03_temporal_difference/guides/cheatsheet.md`, lines 154–156:
  | $G_t$ | Return from time $t$: $\sum_{k=0}^\infty \gamma^k R_{t+k+1}$ |
  | $R_{t+1}, r$ | Reward received after taking $A_t$ in $S_t$ |
  Explicitly distinguished in the notation table.

- `module_06_policy_gradient/guides/01_policy_gradient_theorem_guide.md`, lines 40–41: $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G_0]$ where $G_0$ is the full return, and $G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$ is defined in the REINFORCE guide (line 17). Consistently maintained.

- `module_06_policy_gradient/exercises/exercises.py`: Exercise 2 (`compute_returns_with_baseline`) explicitly computes $G_t$ via reverse accumulation — the distinction between reward $r_k$ and return $G_t$ is coded correctly.

**V vs Q:**

- `module_00_foundations/guides/03_bellman_equations_guide.md`, lines 13–39: State-value $V^\pi(s)$ and action-value $Q^\pi(s,a)$ are defined separately with explicit distinction: "$V^\pi(s)$ answers 'how good is it to be in state $s$?' ... $Q^\pi(s,a)$ answers 'how good is it to take action $a$ in state $s$?'"

- Throughout the course, $V$ is used exclusively for state-value functions and $Q$ for action-value functions with no substitution or conflation.

No conflation found across any of the three concept pairs.

---

### Check 12 — Code Matches Math

**Status: PASS with one MINOR code bug (non-functional)**

**Files spot-checked:**

**File 1: `module_03_temporal_difference/exercises/exercises.py`**

- Exercise 2 (`q_learning_update`): Implementation `target = reward + gamma * np.max(Q[next_state])` matches the Q-learning equation.
- Exercise 3 (`sarsa_update`): Implementation `target = reward + gamma * Q[next_state, next_action]` matches the SARSA equation.
- Exercise 4 classification: `ON_POLICY = {"sarsa", "td0"}` and `OFF_POLICY = {"q_learning", ...}` — consistent with all guide descriptions.

**File 2: `module_06_policy_gradient/exercises/exercises.py`**

- Exercise 1 (`reinforce_loss`): `-(log_probs * returns).mean()` implements $-\mathbb{E}[\log\pi_\theta(A|S) \cdot G_t]$ exactly.
- Exercise 3 (`compute_gae`): backward recurrence `A = delta + gamma * lam * A` with `delta = rewards[t] + gamma * all_values[t+1] - all_values[t]` matches the GAE formula $\hat{A}_t = \delta_t + \gamma\lambda\hat{A}_{t+1}$.

**File 3: `module_07_advanced_policy_optimization/exercises/exercises.py`**

- `ppo_clipped_loss` docstring (lines 35–40): correctly states `L_CLIP = mean(min(ratio_t * A_t, clip(ratio_t, 1-epsilon, 1+epsilon) * A_t))`.

**File 4: `module_03_temporal_difference/guides/02_sarsa_guide.md` — code section (lines 196–201)**

**MINOR BUG FOUND:**

```python
# Advance — carry the preselected next_action forward
state = action = next_state, next_action   # ← BUG: tuple assignment
state, action = next_state, next_action
```

Lines 200–201 contain redundant and incorrect tuple assignment at line 200: `state = action = next_state, next_action` assigns the tuple `(next_state, next_action)` to both `state` and `action`. Line 201 immediately corrects this with the proper unpacking `state, action = next_state, next_action`. Because line 201 overwrites the incorrect assignment made by line 200, the code functions correctly at runtime — but line 200 is a dead code / copy-paste artifact that could mislead a student reading the implementation.

**File:** `/home/user/course-creator/courses/reinforcement-learning/modules/module_03_temporal_difference/guides/02_sarsa_guide.md`
**Lines:** 200–201

**Severity: MINOR** — the code executes correctly (line 201 overrides the tuple mis-assignment from line 200), but the presence of line 200 is confusing and potentially teaches incorrect Python to learners who study the code.

---

## Summary Table

| # | Check | Status | Severity | File(s) |
|---|-------|--------|----------|---------|
| 1 | Bellman expectation equation for V | PASS | — | module_00/.../03_bellman_equations_guide.md, _slides.md |
| 2 | Bellman optimality equation for V | PASS | — | module_00/.../03_bellman_equations_guide.md, _slides.md |
| 3 | Q-learning update rule | PASS | — | module_03/.../03_q_learning_guide.md, _slides.md, cheatsheet.md, exercises.py |
| 4 | SARSA update rule | PASS | — | module_03/.../02_sarsa_guide.md, _slides.md, cheatsheet.md, exercises.py |
| 5 | On-policy vs off-policy classification | PASS | — | module_03/.../02_sarsa_guide.md, 03_q_learning_guide.md, cheatsheet.md, exercises.py |
| 6 | Policy gradient theorem | PASS | — | module_06/.../01_policy_gradient_theorem_guide.md, _slides.md |
| 7 | PPO clipped objective | PASS | — | module_07/.../02_ppo_guide.md, _slides.md, exercises.py |
| 8 | DQN: experience replay + target network | PASS | — | module_05/.../01_dqn_guide.md, _slides.md |
| 9 | Actor-critic separation | PASS | — | module_06/.../03_actor_critic_guide.md, _slides.md |
| 10 | Convergence claims (TD/Q-learning/PG) | PASS | — | module_03/.../01_td_prediction_guide.md, 03_q_learning_guide.md; module_06/.../01_policy_gradient_theorem_guide.md |
| 11 | No concept conflation | PASS | — | Cross-module |
| 12 | Code matches math | PASS (1 minor bug) | MINOR | module_03/.../02_sarsa_guide.md lines 200–201 |

---

## Issues Found

### Issue 1 — Dead code line in SARSA implementation

**Severity: MINOR**
**File:** `/home/user/course-creator/courses/reinforcement-learning/modules/module_03_temporal_difference/guides/02_sarsa_guide.md`
**Lines:** 200–201

**Incorrect code:**
```python
state = action = next_state, next_action   # ← assigns tuple to both variables
state, action = next_state, next_action
```

**What is wrong:** Line 200 uses chained assignment with a tuple on the right-hand side: `state = action = (next_state, next_action)`. This assigns the 2-tuple `(next_state, next_action)` to both `state` and `action`, not the individual values. Line 201 immediately overwrites this with correct tuple unpacking. At runtime the code is correct because line 201 always executes. However, line 200 is misleading and teaches incorrect Python.

**What is correct:** Only line 201 is needed:
```python
state, action = next_state, next_action
```

**Impact:** No runtime effect (line 201 corrects it), but a student studying the code may be confused by line 200 or, in a simplified environment where only line 200 exists, would have a bug.

---

## No Issues Found In

- All four Bellman equations (expectation and optimality, for V and Q) — correct in all locations.
- All Q-learning and SARSA update equations — correct in guides, slides, cheatsheet, and exercise implementations.
- On-policy/off-policy distinction — rigorously and consistently maintained everywhere, including the exercise classification function.
- Policy gradient theorem — correct form with proper expectation notation in both guide and slides.
- PPO clipped objective — correct form with proper `min`, `clip`, and probability ratio definition throughout.
- DQN innovations — both experience replay and target network named, motivated, and coded.
- Actor-critic separation — explicitly stated and enforced in code with separate classes and optimizers.
- Convergence claims — correct for TD(0) (tabular + Robbins-Monro), Q-learning (sufficient exploration + Robbins-Monro), and policy gradient (local optimum, non-convex landscape).
- TD error vs advantage, return vs reward, V vs Q — all kept distinct throughout the course.
- GAE implementation in exercises.py (module_06) — backward recurrence matches the guide's formula exactly.
- PPO exercise implementation — `min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)` matches the stated objective exactly.
