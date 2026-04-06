# Model-Based Reinforcement Learning: Overview

> **Reading time:** ~10 min | **Module:** 8 — Model-Based RL | **Prerequisites:** Module 5

## In Brief

Model-based reinforcement learning (MBRL) augments the standard agent-environment loop with a **learned model** of the environment. The agent uses this model to simulate experience, plan ahead, or optimize a policy without generating every required data point from real interaction. The primary payoff is **sample efficiency**: fewer real environment steps are needed to reach a given performance level.

<div class="callout-key">

<strong>Key Concept:</strong> Model-based reinforcement learning (MBRL) augments the standard agent-environment loop with a **learned model** of the environment. The agent uses this model to simulate experience, plan ahead, or optimize a policy without generating every required data point from real interaction.

</div>


## Key Insight

Model-free methods are powerful but wasteful — they discard the structural information in each transition $(S_t, A_t, R_{t+1}, S_{t+1})$ after using it once. Model-based methods extract more from each interaction by fitting a compact representation of environment dynamics and then replaying, planning through, or differentiating through that model. The cost is that the model is imperfect, and errors compound as planning looks further ahead.

---


<div class="callout-key">

<strong>Key Point:</strong> Model-free methods are powerful but wasteful — they discard the structural information in each transition $(S_t, A_t, R_{t+1}, S_{t+1})$ after using it once.

</div>

## Intuitive Explanation

Think of a chess grandmaster and a novice both studying to improve. The novice can only improve by playing real games (model-free). The grandmaster additionally runs games in their head — "If I move here, my opponent will likely respond there, then I could…" That internal simulation is planning. A strong mental model of how chess works makes each real game far more educational.

<div class="callout-key">

<strong>Key Point:</strong> Think of a chess grandmaster and a novice both studying to improve.

</div>


In RL terms:
- Each real game = one environment episode (expensive: slow clock, physical wear, financial cost)
- Mental simulation = rollout under the learned model (cheap: pure computation)
- The grandmaster's internal chess model = $\hat{p}_\theta$ and $\hat{r}_\phi$

The model is not perfect, but it is good enough to improve the policy when used carefully.

---


## Formal Definition

### The Environment Model

<div class="callout-info">

<strong>Info:</strong> ### The Environment Model

In a Markov Decision Process $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$, the true dynamics are:

$$p(s' \mid s, a) \quad \text{and} \quad r(s, a) = \mathbb{E}[R_{t+1} \mid S...

</div>


In a Markov Decision Process $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$, the true dynamics are:

$$p(s' \mid s, a) \quad \text{and} \quad r(s, a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]$$

A **learned model** is a parametric approximation:

$$\hat{p}_\theta(s' \mid s, a) \quad \text{and} \quad \hat{r}_\phi(s, a)$$

fitted by supervised learning from the agent's collected experience $\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}$.

### Planning

**Planning** is any computation that uses the model to improve a value function or policy without generating new real-environment transitions. Formally, planning produces a sequence of updates:

$$Q(s, a) \leftarrow \hat{r}(s, a) + \gamma \sum_{s'} \hat{p}(s' \mid s, a) \max_{a'} Q(s', a')$$

or, in sampled form:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \bigl[\hat{r}(s, a) + \gamma \max_{a'} Q(s', a') - Q(s, a)\bigr]$$

where $s' \sim \hat{p}(\cdot \mid s, a)$ is drawn from the model.

---


## What Is a "Model"?

A model has two components:

| Component | Symbol | What it predicts | Common parameterization |
|-----------|--------|-----------------|------------------------|
| Transition model | $\hat{p}_\theta(s' \mid s, a)$ | Next state given current state and action | Neural network, Gaussian process, ensemble |
| Reward model | $\hat{r}_\phi(s, a)$ | Expected reward for a state-action pair | Neural network, linear regression |

**Deterministic vs stochastic models.** Simple environments may be modeled deterministically: $\hat{s}' = f_\theta(s, a)$. Stochastic environments require a distribution — commonly a Gaussian $\hat{p}_\theta(s' \mid s, a) = \mathcal{N}(\mu_\theta(s,a),\, \Sigma_\theta(s,a))$ or a mixture density network.

**Observation vs state models.** When the agent only sees observations $o$ (not full states $s$), the model must operate in observation space or learn a latent state embedding — the approach taken by World Models and MuZero (Guide 03).

---

## Sample Efficiency Advantage

The core benefit of model-based RL is the ratio of planning steps to real steps:

```
Model-free update budget: 1 gradient step per 1 real transition
Model-based update budget: 1 real transition → train model → K planning steps
```

For $K = 50$ planning steps per real step, the effective data usage is multiplied by up to $50\times$. In practice, sample efficiency improvements of $5\times$–$100\times$ versus model-free baselines have been demonstrated on continuous-control benchmarks (MBPO achieves $\approx 30\times$ over SAC on HalfCheetah).

---

## Model Error and Compounding Errors

The fundamental challenge of model-based RL: **every model is wrong**, and errors compound over multi-step rollouts.

### Compounding Error Mechanism

If the one-step prediction error is $\epsilon$, then after $H$ steps the error grows roughly as:

$$\text{Error}_H \approx \epsilon \cdot H + \mathcal{O}(\epsilon^2 H^2)$$

In the worst case (adversarial dynamics), errors compound multiplicatively:

$$\text{Error}_H \lesssim (1 + L\epsilon)^H - 1$$

where $L$ is the Lipschitz constant of the dynamics. For large $H$ or large $\epsilon$ this diverges rapidly.

### Practical Consequence

Long model-based rollouts in an inaccurate model can yield worse performance than model-free methods, because the agent optimizes for a distorted objective. This is called **model exploitation**: the policy finds actions that the model incorrectly thinks are good.

### Mitigations

| Strategy | Description |
|----------|-------------|
| Short rollouts | Use model only for $H \in \{1, 5, 10\}$ steps; return to real data |
| Ensemble models | Train $M$ models; use disagreement as uncertainty signal |
| Pessimistic planning | Penalize states where ensemble disagrees (MOPO, MOReL) |
| Latent-space models | Learn compact representation; errors stay bounded (MuZero) |

---

## Taxonomy of Model-Based RL

Three distinct ways to use a learned model:

### Category 1: Learn Model → Plan

1. Collect real data $\mathcal{D}$
2. Fit $\hat{p}_\theta$, $\hat{r}_\phi$ from $\mathcal{D}$
3. Run a planning algorithm (value iteration, MCTS, trajectory optimization) inside the model
4. Execute the resulting policy in the real environment

**Examples:** Dyna-Q (planning = tabular Q-updates), MCTS in AlphaZero, trajectory optimization in PETS.

### Category 2: Learn Model to Augment Model-Free

1. Run standard model-free algorithm (Q-learning, SAC)
2. Additionally use the model to generate synthetic transitions appended to the replay buffer
3. Model-free algorithm trains on real + synthetic data

**Examples:** Dyna-Q (from the model-free perspective), MBPO, M-SAC.

### Category 3: Optimize Directly Through the Model

1. Backpropagate policy gradients through the differentiable model
2. Compute $\nabla_\theta J(\theta) = \nabla_\theta \mathbb{E}_{\hat{p}}[\sum_t r_t]$ using automatic differentiation

**Examples:** PILCO (Gaussian process model + analytic gradients), SVG(1), Dreamer (latent imagination).

---

## Pipeline Diagram

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

The following implementation builds on the approach above:

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
    subgraph MF["Model-Free Pipeline"]
        direction LR
        ENV1([Real Environment]) -->|"$(s, a, r, s')$"| REPLAY1[(Replay Buffer)]
        REPLAY1 --> MFUPDATE[/"Q / Policy Update"/]
        MFUPDATE --> ENV1
    end

    subgraph MB["Model-Based Pipeline"]
        direction LR
        ENV2([Real Environment]) -->|"$(s, a, r, s')$"| DATA2[(Experience\nBuffer)]
        DATA2 --> MODEL_FIT[/"Fit $\hat{p}_\theta, \hat{r}_\phi$"/]
        MODEL_FIT --> PLAN[/"Planning /\nImagined Rollouts"/]
        PLAN -->|Synthetic $(s, a, r, s')$| REPLAY2[(Augmented\nBuffer)]
        DATA2 --> REPLAY2
        REPLAY2 --> MBUPDATE[/"Q / Policy Update"/]
        MBUPDATE --> ENV2
    end

    style MF fill:#f0f4ff,stroke:#4A90D9
    style MB fill:#fff4f0,stroke:#E8844A
```

</div>

---

## Code Snippet: Minimal Model Training Loop

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

The following implementation builds on the approach above:

```python
import numpy as np
import torch
import torch.nn as nn
from collections import deque


class TransitionModel(nn.Module):
    """
    Deterministic one-step transition model: predicts (s', r) from (s, a).

    This is the simplest possible model — a two-headed MLP.
    In practice, use probabilistic heads or an ensemble.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        in_dim = state_dim + action_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.next_state_head = nn.Linear(hidden, state_dim)
        self.reward_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=-1)
        features = self.trunk(x)
        delta_s = self.next_state_head(features)    # predict residual for stability
        reward = self.reward_head(features).squeeze(-1)
        return state + delta_s, reward              # next_state, reward


def train_model(model, optimizer, buffer, batch_size: int = 256, epochs: int = 5):
    """
    Supervised learning on collected (s, a, r, s') transitions.

    Training the model is standard regression — the RL novelty is
    how the model is then used for planning.
    """
    model.train()
    states, actions, rewards, next_states = buffer.sample(batch_size * epochs)
    dataset = torch.utils.data.TensorDataset(states, actions, rewards, next_states)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for s, a, r, s_next in loader:
        s_pred, r_pred = model(s, a)

        # Regression losses: next-state prediction + reward prediction
        loss_dynamics = nn.functional.mse_loss(s_pred, s_next)
        loss_reward = nn.functional.mse_loss(r_pred, r)
        loss = loss_dynamics + loss_reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

</div>

---

## Common Pitfalls

<div class="callout-danger">

<strong>Danger:</strong> The pitfalls below are the most common mistakes practitioners make. Each one can silently degrade your results without obvious errors.

</div>

**Pitfall 1 — Planning too far into an inaccurate model.**
Long rollouts amplify model errors. A model that is 95% accurate per step has $0.95^{20} \approx 36\%$ accuracy after 20 steps. Restrict rollout horizon to where model accuracy is acceptable, typically $H \in [1, 10]$ for neural network models on complex tasks.

<div class="callout-warning">

<strong>Warning:</strong> **Pitfall 1 — Planning too far into an inaccurate model.**
Long rollouts amplify model errors.

</div>

**Pitfall 2 — Model exploitation (distributional shift).**
The policy optimized inside the model will seek out states where the model is incorrect but appears favorable. This is a form of adversarial attack on your own model. Mitigation: use model uncertainty (ensembles, dropout) to penalize out-of-distribution states during planning.

**Pitfall 3 — Forgetting to retrain the model as the policy changes.**
The data distribution shifts as the policy improves. A model fitted on early exploratory data is systematically wrong in regions the improved policy visits. Retrain the model periodically on the growing buffer.

**Pitfall 4 — Using the model outside its support.**
A model trained on state-action pairs near $\mu_\beta(s,a)$ (the behavior distribution) cannot be trusted for inputs far from that distribution. Monitor reconstruction error on recently collected data as a health signal.

**Pitfall 5 — Neglecting reward model accuracy.**
Practitioners often focus on transition accuracy and underweight the reward model. A 5% error in reward estimates produces systematically biased value estimates even when dynamics are perfect. Evaluate reward prediction MSE separately during training.

---

## Connections


<div class="callout-info">

<strong>Info:</strong> This section maps how this guide connects to the broader course. Use these links to navigate related material.

</div>

- **Builds on:** Bellman equations (Module 0), Q-learning and TD methods (Module 3), function approximation (Module 4)
- **Leads to:** Dyna-Q and MCTS (Guide 02), World Models and MuZero (Guide 03)
- **Related to:** Optimal control and trajectory optimization (classical control theory), model predictive control (MPC)

---


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


## Further Reading

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 8 — foundational treatment of planning and learning with models
- Moerland et al. (2023), "Model-based Reinforcement Learning: A Survey" — comprehensive taxonomy of MBRL approaches
- Janner et al. (2019), "When to Trust Your Model: Model-Based Policy Optimization" (MBPO) — demonstrates 30× sample efficiency improvement with short model rollouts
- Chua et al. (2018), "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models" (PETS) — probabilistic ensemble approach to model uncertainty


---

## Cross-References

<a class="link-card" href="./01_model_based_overview_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_dyna_q.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
