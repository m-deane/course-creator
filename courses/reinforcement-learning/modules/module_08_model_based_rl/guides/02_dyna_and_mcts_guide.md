# Dyna-Q and Monte Carlo Tree Search

> **Reading time:** ~17 min | **Module:** 8 — Model-Based RL | **Prerequisites:** Module 5

## In Brief

Dyna-Q (Sutton, 1991) is the foundational algorithm that unifies learning from real experience and planning with a learned model. Monte Carlo Tree Search (MCTS) is the dominant algorithm for lookahead planning in large discrete spaces and the engine behind AlphaGo and AlphaZero. Both methods exploit a model — one implicitly through replay, one explicitly through simulation.

<div class="callout-key">
<strong>Key Concept:</strong> Dyna-Q (Sutton, 1991) is the foundational algorithm that unifies learning from real experience and planning with a learned model. Monte Carlo Tree Search (MCTS) is the dominant algorithm for lookahead planning in large discrete spaces and the engine behind AlphaGo and AlphaZero.
</div>


## Key Insight

Dyna-Q asks: "After each real step, why not run several imagined steps too?" The imagined steps use a tabular model fitted from real transitions and produce additional Q-learning updates at zero environment cost. MCTS asks: "Before acting, why not simulate many futures and pick the action whose subtree looks best?" Both approaches trade compute for fewer real environment interactions.

---



<div class="callout-key">
<strong>Key Point:</strong> Dyna-Q asks: "After each real step, why not run several imagined steps too?" The imagined steps use a tabular model fitted from real transitions and produce additional Q-learning updates at zero envir...
</div>
## Intuitive Explanation: Dyna-Q

A student preparing for an exam:
- **Real experience** (Step 2): work through new practice problems as they encounter them.
- **Model** (Step 3): maintain a mental model of which problem types map to which solution patterns.
- **Planning** (Step 4): drill from memory — re-solve problems already seen, reinforce the solution pattern without doing new problems.

<div class="callout-key">
<strong>Key Point:</strong> A student preparing for an exam:
- **Real experience** (Step 2): work through new practice problems as they encounter them.
</div>


The $n$ parameter controls the ratio of drilling to new problems. $n = 0$ is pure reactive learning; large $n$ is mostly drilling on past experience. The optimal $n$ depends on model accuracy and the cost of new problems.

---


## Formal Definition: Dyna-Q

### Setting

<div class="callout-info">
<strong>Info:</strong> ### Setting

Tabular MDP $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$ with a deterministic learned model $\hat{p}$.
</div>


Tabular MDP $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$ with a deterministic learned model $\hat{p}$. The agent maintains:

- $Q(s, a)$: action-value estimates (initialized to 0)
- $\text{Model}(s, a)$: a table storing $(r, s')$ for each observed $(s, a)$ pair

### Dyna-Q Algorithm (Sutton & Barto, Ch. 8.2)

```
Initialize Q(s, a) = 0 for all s, a
Initialize Model(s, a) = empty for all s, a

Loop (for each real episode step):
  1. [Act] Observe S; choose A ← ε-greedy(Q, S)

  2. [Real update] Execute A; observe R, S'
     Q(S, A) ← Q(S, A) + α[R + γ max_a Q(S', a) - Q(S, A)]

  3. [Model learning] Model(S, A) ← (R, S')   # store transition

  4. [Planning] Repeat n times:
       S̃ ← random previously observed state
       Ã ← random action previously taken in S̃
       R̃, S̃' ← Model(S̃, Ã)
       Q(S̃, Ã) ← Q(S̃, Ã) + α[R̃ + γ max_a Q(S̃', a) - Q(S̃, Ã)]

  S ← S'
```

Steps 1–3 are standard Q-learning. Step 4 is the **planning loop**: $n$ additional Q-updates using model-generated transitions. The two types of updates are identical in form — the only difference is whether $(S, A, R, S')$ came from real experience or the model.

### Effect of Planning Steps $n$

| $n$ (planning steps) | Real steps needed to solve maze | Speedup vs $n=0$ |
|---------------------|--------------------------------|-----------------|
| 0 (pure Q-learning) | ~1,000 | 1× |
| 5 | ~200 | 5× |
| 50 | ~20 | 50× |

Source: Sutton & Barto Figure 8.2 — the Maze example.

---


## Dyna-Q+ for Changing Environments

Standard Dyna-Q assumes the environment is stationary. If the environment changes (e.g., a wall is removed or added), the model becomes stale and planning reinforces outdated beliefs.

<div class="callout-warning">
<strong>Warning:</strong> Standard Dyna-Q assumes the environment is stationary.
</div>


**Dyna-Q+** (Sutton & Barto, Ch. 8.3) adds an exploration bonus that grows with the time since a state-action pair was last tried in the real environment:

$$\tilde{r}(S, A) = r + \kappa\sqrt{\tau(S, A)}$$

where $\tau(S, A)$ is the number of real time steps since $(S, A)$ was last executed, and $\kappa > 0$ is a small bonus coefficient.

**Effect:** Actions that have not been tried recently receive inflated reward estimates. This drives the agent to re-explore parts of the environment that may have changed, preventing the model from becoming permanently stale.

**Practical setting:** $\kappa = 0.001$ to $0.01$. Too large a value over-explores; too small fails to detect changes.

---

## Python Implementation: Dyna-Q

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

<div class="callout-insight">
<strong>Insight:</strong> example.py


The following implementation builds on the approach above:




---
</div>


The following implementation builds on the approach above:

```python
import numpy as np
from collections import defaultdict


class DynaQ:
    """
    Dyna-Q agent for tabular MDPs.

    Implements Sutton & Barto (2018) Algorithm 8.1.
    States and actions must be hashable.
    """

    def __init__(
        self,
        action_space_size: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        n_planning_steps: int = 10,
    ):
        self.n_actions = action_space_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n_planning_steps

        # Q-table: state -> array of Q-values
        self.Q = defaultdict(lambda: np.zeros(action_space_size))

        # Deterministic model: (state, action) -> (reward, next_state)
        self.model: dict[tuple, tuple] = {}

        # Track which (state, action) pairs have been observed for planning
        self.observed_sa: list[tuple] = []

    def act(self, state) -> int:
        """ε-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state, action: int, reward: float, next_state):
        """
        One Dyna-Q step:
          1. Real Q-update from the actual transition
          2. Model learning: store (state, action) -> (reward, next_state)
          3. n planning Q-updates from randomly sampled model transitions
        """
        # Step 1: Direct RL update (real experience)
        td_target = reward + self.gamma * np.max(self.Q[next_state])
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

        # Step 2: Model learning
        if (state, action) not in self.model:
            self.observed_sa.append((state, action))
        self.model[(state, action)] = (reward, next_state)

        # Step 3: Planning — n additional Q-updates from model
        for _ in range(self.n):
            # Sample a previously observed (state, action) pair uniformly
            idx = np.random.randint(len(self.observed_sa))
            s_sim, a_sim = self.observed_sa[idx]

            # Query the model for the simulated transition
            r_sim, s_next_sim = self.model[(s_sim, a_sim)]

            # Standard Q-learning update on simulated experience
            td_target_sim = r_sim + self.gamma * np.max(self.Q[s_next_sim])
            td_error_sim = td_target_sim - self.Q[s_sim][a_sim]
            self.Q[s_sim][a_sim] += self.alpha * td_error_sim
```
</div>

---

## Formal Definition: Monte Carlo Tree Search (MCTS)

### Setting

MCTS operates from a **current state** $s_0$ and builds a search tree $\mathcal{T}$ by iterating four phases. It does not require a learned model — it requires a **simulator** (the real environment or a learned model) that can be queried with any $(s, a)$ pair.

### The Four Phases

#### Phase 1: Selection

Starting from the root $s_0$, traverse the existing tree by selecting actions using the **UCT** (Upper Confidence bound for Trees) policy:

$$\text{UCT}(s, a) = \bar{Q}(s, a) + c \sqrt{\frac{\ln N(s)}{N(s, a)}}$$

| Symbol | Meaning |
|--------|---------|
| $\bar{Q}(s, a)$ | Mean return observed from $(s, a)$ across all simulations |
| $N(s)$ | Total visit count for state $s$ |
| $N(s, a)$ | Visit count for action $a$ in state $s$ |
| $c > 0$ | Exploration coefficient (typical: $c = \sqrt{2}$ or tuned) |

Continue selection until reaching a **leaf node** (an unexpanded state not yet in the tree).

#### Phase 2: Expansion

Add one or more child nodes to the selected leaf. Typically expand one new state-action pair chosen by the selection policy or uniformly at random.

#### Phase 3: Simulation (Rollout)

From the newly expanded node, run a **rollout** to a terminal state (or depth limit $H$) using a **rollout policy** $\pi_\text{roll}$ — typically a fast, simple policy (uniform random, or a learned heuristic). Collect the total return $G$.

#### Phase 4: Backpropagation

Propagate $G$ back through all nodes visited during selection and expansion:

$$N(s) \leftarrow N(s) + 1, \quad N(s, a) \leftarrow N(s, a) + 1, \quad \bar{Q}(s, a) \leftarrow \bar{Q}(s, a) + \frac{G - \bar{Q}(s, a)}{N(s, a)}$$

The incremental mean update ensures $\bar{Q}(s, a)$ is always the empirical mean of all returns observed from $(s, a)$.

### Action Selection After Search

After $K$ simulations (iterations of the four phases), select the action with the highest visit count from the root:

$$a^* = \arg\max_a N(s_0, a)$$

Visit counts are more robust than Q-values for final action selection because $N$ is less sensitive to outlier rollouts.

---

## UCT Formula: Intuition

The UCT formula balances two objectives:

$$\underbrace{\bar{Q}(s, a)}_{\text{exploitation}} + \underbrace{c \sqrt{\frac{\ln N(s)}{N(s, a)}}}_{\text{exploration}}$$

- **Exploitation term $\bar{Q}(s, a)$:** prefer actions with high estimated return.
- **Exploration term:** the confidence interval width on $\bar{Q}(s, a)$. Actions visited rarely ($N(s,a)$ small relative to $N(s)$) get a large bonus — they could be better than they appear.

The UCT bonus is derived from the UCB1 bandit algorithm applied to the tree. It provides a theoretical guarantee: the number of suboptimal action selections grows only logarithmically.

**Choosing $c$:** $c = \sqrt{2}$ is the theoretical default (Kocsis & Szepesvári, 2006). In practice, $c$ is tuned per domain. Larger $c$ → more exploration, shorter but wider trees. Smaller $c$ → deeper exploitation of promising branches.

---

## MCTS Diagram

```
            ROOT (s_0)
           /    |    \
          a1    a2    a3    ← Selection: UCT picks best unexplored leaf
         /  \
      s_11  s_12           ← Expansion: add new child node
              |
           [ROLLOUT]        ← Simulation: random play to terminal
              |
             G = 0.7        ← Backpropagation: update N, Q̄ on path to root

After K iterations:
  a* = argmax_a N(s_0, a)   ← Action with most visits wins
```

---

## Mermaid Diagram: MCTS Four Phases

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

The following implementation builds on the approach above:

```mermaid
flowchart TD
    START([Current State s₀]) --> SEL

    subgraph SEL["Phase 1: Selection"]
        direction LR
        S1["Traverse tree\nusing UCT"] --> S2{"Leaf\nreached?"}
        S2 -- No --> S1
        S2 -- Yes --> LEAF[Leaf node]
    end

    LEAF --> EXP

    subgraph EXP["Phase 2: Expansion"]
        E1["Add new child\nnode to tree"]
    end

    EXP --> SIM

    subgraph SIM["Phase 3: Simulation"]
        SIM1["Rollout with\nπ_roll to terminal"] --> SIM2["Observe\nreturn G"]
    end

    SIM --> BACK

    subgraph BACK["Phase 4: Backpropagation"]
        B1["Update N(s), N(s,a),\nQ̄(s,a) on path to root"]
    end

    BACK -->|"Repeat K times"| START
    BACK -->|"After K simulations"| ACT["Select a* = argmax N(s₀, a)"]

    style SEL fill:#e8f4ff,stroke:#4A90D9
    style EXP fill:#fff8e8,stroke:#D4A017
    style SIM fill:#f0ffe8,stroke:#4A9D4A
    style BACK fill:#ffe8f4,stroke:#D44A7A
```
</div>

---

## AlphaGo and AlphaZero Connection

Standard MCTS uses a **random rollout policy** for simulation, which is slow and imprecise in complex games like Go (branching factor ≈ 250, game length ≈ 150 moves).

**AlphaGo (Silver et al., 2016)** replaced both the rollout policy and the value estimate with neural networks:

- **Policy network** $p_\sigma(a \mid s)$: guides selection (replaces UCT exploration) and rollouts
- **Value network** $v_\theta(s)$: replaces random rollouts with a direct value estimate at the leaf

The AlphaGo UCT variant:

$$\text{UCT}(s, a) = \bar{Q}(s, a) + c \cdot p_\sigma(a \mid s) \cdot \frac{\sqrt{N(s)}}{1 + N(s, a)}$$

**AlphaZero (Silver et al., 2017)** eliminates separate rollouts entirely — the value network provides all leaf evaluations. The policy and value networks are trained entirely by self-play, with MCTS generating the training data.

**MuZero (Schrittwieser et al., 2020)** further removes the need for a known game simulator — it learns the dynamics model used for MCTS planning. Covered in Guide 03.

---

## Planning vs Learning Trade-Offs

| Dimension | Dyna-Q | MCTS |
|-----------|--------|------|
| **Planning mode** | Background (offline) | Foreground (at decision time) |
| **Model use** | Improve global Q-table | Look ahead from current state |
| **Compute budget** | Fixed $n$ steps/real-step | Flexible: more simulations = better action |
| **Environment type** | Tabular/small continuous | Large discrete, perfect/learned sim |
| **Best suited for** | Sample efficiency during training | High-quality action selection at test time |

The two approaches are complementary. AlphaZero uses both: Dyna-style self-play to improve the value network globally, and MCTS at decision time for high-quality action selection.

---


<div class="compare">
<div class="compare-card">
<div class="header before">Planning</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
<div class="compare-card">
<div class="header after">Learning Trade-Offs</div>
<div class="body">

See detailed comparison in the table above.

</div>
</div>
</div>

## Code Snippet: MCTS Core Loop

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

The following implementation builds on the approach above:

```python
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MCTSNode:
    """
    A node in the MCTS search tree.

    Each node corresponds to a state reached by a specific action from its parent.
    """
    state: object
    parent: Optional["MCTSNode"] = None
    parent_action: Optional[int] = None

    visit_count: int = 0
    value_sum: float = 0.0
    children: dict = field(default_factory=dict)   # action -> MCTSNode

    @property
    def q_value(self) -> float:
        """Mean return from this node (exploitation term)."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def uct_score(self, c: float = math.sqrt(2)) -> float:
        """
        UCT score used by the parent to select this child.

        Balances exploitation (q_value) and exploration (visit-count ratio).
        Returns infinity if unvisited — unvisited nodes are always expanded first.
        """
        if self.visit_count == 0:
            return float("inf")
        parent_visits = self.parent.visit_count
        exploration = c * math.sqrt(math.log(parent_visits) / self.visit_count)
        return self.q_value + exploration


def mcts_search(root_state, env_simulator, n_simulations: int, c: float = math.sqrt(2)):
    """
    Run MCTS from root_state for n_simulations iterations.

    Args:
        root_state:    Initial state to plan from
        env_simulator: Callable (state, action) -> (next_state, reward, done)
                       and env_simulator.action_space_size -> int
        n_simulations: Number of Selection-Expansion-Simulation-Backprop cycles
        c:             UCT exploration coefficient

    Returns:
        best_action: int — action with highest visit count from root
    """
    root = MCTSNode(state=root_state)
    n_actions = env_simulator.action_space_size

    for _ in range(n_simulations):
        # --- Phase 1: Selection ---
        node = root
        while node.children and len(node.children) == n_actions:
            # All actions expanded — select best UCT child
            node = max(node.children.values(), key=lambda c_: c_.uct_score(c))

        # --- Phase 2: Expansion ---
        # Find an untried action and expand it
        tried_actions = set(node.children.keys())
        untried = [a for a in range(n_actions) if a not in tried_actions]
        if untried:
            action = untried[np.random.randint(len(untried))]
            next_state, reward, done = env_simulator(node.state, action)
            child = MCTSNode(state=next_state, parent=node, parent_action=action)
            node.children[action] = child
            node = child

        # --- Phase 3: Simulation (rollout) ---
        sim_state = node.state
        sim_return = 0.0
        discount = 1.0
        max_depth = 50

        for _ in range(max_depth):
            if done:
                break
            a_roll = np.random.randint(n_actions)   # random rollout policy
            sim_state, r, done = env_simulator(sim_state, a_roll)
            sim_return += discount * r
            discount *= 0.99

        # --- Phase 4: Backpropagation ---
        while node is not None:
            node.visit_count += 1
            node.value_sum += sim_return    # same return propagated to all ancestors
            node = node.parent

    # Select action with highest visit count from root (robust to outliers)
    best_action = max(root.children, key=lambda a: root.children[a].visit_count)
    return best_action
```
</div>

---

## Common Pitfalls

<div class="callout-danger">
<strong>Danger:</strong> The pitfalls below are the most common mistakes practitioners make. Each one can silently degrade your results without obvious errors.
</div>

**Pitfall 1 — Using stale model transitions in Dyna-Q after environment change.**
If the environment changes (a wall is added, a reward location moves), Dyna-Q continues to plan with old model transitions that are now wrong. These incorrect planning updates can undo learning from real experience. Use Dyna-Q+ with an exploration bonus, or discard model entries older than a threshold.

<div class="callout-warning">
<strong>Warning:</strong> **Pitfall 1 — Using stale model transitions in Dyna-Q after environment change.**
If the environment changes (a wall is added, a reward location moves), Dyna-Q continues to plan with old model transitions that are now wrong.
</div>

**Pitfall 2 — Insufficient planning steps for the model quality.**
Too many planning steps with an inaccurate model causes model exploitation. Too few planning steps with an accurate model wastes the sample efficiency benefit. A practical heuristic: start with $n = 5$, monitor whether simulated transitions match real transitions, and scale $n$ with model accuracy.

**Pitfall 3 — Wrong UCT exploration coefficient $c$.**
The theoretical $c = \sqrt{2}$ assumes normalized returns $\in [0,1]$. If rewards are not normalized, $c$ must be rescaled. Unnormalized $c$ that is too small causes MCTS to exploit early results, missing better branches. Unnormalized $c$ too large causes random-looking search. Always normalize rewards or tune $c$ per environment.

**Pitfall 4 — Expanding a node before it is a leaf (premature expansion).**
MCTS selection should proceed until reaching a node with unexpanded children. Expanding a non-leaf wastes tree depth and corrupts statistics. Verify that the "is leaf" check is correct: a node is a leaf if it has at least one unvisited action.

**Pitfall 5 — Backpropagating the wrong return.**
The return $G$ backpropagated must reflect cumulative reward from the expanded node to the end of the rollout, not just the rollout rewards. If the node was reached by a real transition with reward $r$, that reward must be included. Track cumulative return starting from the expanded node's parent.

**Pitfall 6 — Conflating Dyna-Q planning and MCTS planning.**
Dyna-Q improves the *global* Q-table by replaying past experience. MCTS improves the *local* decision at the current state by forward simulation. Mixing them incorrectly — e.g., using MCTS rollouts to update Q-table values for states not on the rollout path — violates the statistics of both algorithms.

---

## Connections


<div class="callout-info">
<strong>Info:</strong> This section maps how this guide connects to the broader course. Use these links to navigate related material.
</div>

- **Builds on:** Model-Based Overview (Guide 01), Q-learning and TD updates (Module 3), UCB1 bandit algorithm (Multi-Armed Bandits course)
- **Leads to:** World Models and MuZero (Guide 03) which replace the hand-crafted simulator in MCTS with a learned latent model
- **Related to:** Monte Carlo methods (Module 2) — MCTS uses MC rollouts; prioritized experience replay — a variation of Dyna-Q's planning step sampling

---


## Practice Questions

**Question 1 — Conceptual:** Based on the concepts in this guide, explain in your own words why the core technique matters and when you would choose it over alternatives.

**Question 2 — Application:** Sketch out how you would apply the main concept from this guide to a real-world dataset or problem you have encountered. What would you need to watch out for?


## Further Reading

- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 8.1–8.4 — Dyna-Q, Dyna-Q+, prioritized sweeping
- Kocsis & Szepesvári (2006), "Bandit Based Monte-Carlo Planning" — original UCT paper
- Silver et al. (2016), "Mastering the game of Go with deep neural networks and tree search" (AlphaGo)
- Silver et al. (2017), "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (AlphaZero)
- Browne et al. (2012), "A Survey of Monte Carlo Tree Search Methods" — comprehensive MCTS reference


---

## Cross-References

<a class="link-card" href="./02_dyna_and_mcts_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Interactive slide deck covering the key concepts with visual examples.</div>
</a>

<a class="link-card" href="../notebooks/01_dyna_q.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises and real data.</div>
</a>
