# Project 1: GridWorld Navigator (Beginner)

## What You'll Build

A tabular Q-learning agent that navigates custom gridworld environments you
design yourself. You implement every component from scratch — the grid
environment, the Q-learning agent, the training loop, and a policy
visualization that overlays learned directional arrows on the grid. The
finished project is a self-contained simulation showing how an agent
transitions from random wandering to optimal navigation.

**Tools:** numpy, matplotlib only. No deep learning frameworks.

**Expected time:** 4–6 hours.

---

## Learning Objectives

- Translate an informal navigation problem into a formal MDP (states, actions,
  rewards, transitions, terminal conditions)
- Implement the Q-learning update rule from the Bellman equation without
  reference to a library
- Design an epsilon-greedy exploration schedule and explain the exploration–
  exploitation trade-off in concrete terms
- Diagnose convergence from reward and Q-value plots
- Quantify how learning rate, discount factor, and epsilon decay affect the
  learned policy

---

## The Problem

You control an agent navigating a 2D grid from a start cell to a goal cell.
Some cells are walls (impassable) and some are hazards (large negative
reward). The agent does not know the layout in advance — it must learn by
interacting with the environment.

**State:** `(row, col)` — the agent's current grid position.

**Actions:** Up, Down, Left, Right (4-directional, no diagonal moves).

**Rewards:**
- Reaching the goal: `+10`
- Stepping on a hazard: `-5` (episode continues)
- Every other step: `-0.1` (time penalty to encourage short paths)
- Hitting a wall: no move, `-0.5` penalty

**Termination:** Episode ends when the agent reaches the goal or after 200
steps (timeout).

You will start with a provided 5x5 grid, then modify it or build your own
to demonstrate that your implementation generalizes.

---

## Required Deliverables

1. `environment.py` — `GridWorld` class implementing the MDP
2. `agent.py` — `QLearningAgent` class with Q-table, `select_action()`, and
   `update()` methods
3. `train.py` — training loop that runs N episodes and records episode rewards
4. `visualize.py` — two plots: (a) reward curve over episodes with a rolling
   average, (b) learned policy as directional arrows on the grid
5. `analysis.md` — written responses to the analysis questions in the
   self-assessment section (3–5 sentences each)

---

## Suggested Milestones

### Milestone 1 — Environment (45–60 min)

Implement `GridWorld` with the following interface:

```python
env = GridWorld(grid_map)          # grid_map is a 2D list of chars
state = env.reset()                # returns start state as (row, col)
next_state, reward, done = env.step(action)  # action in {0,1,2,3}
env.render()                       # optional ASCII print
```

Verify it manually: step through the 5x5 grid by hand and confirm rewards,
wall blocking, and terminal detection are correct before writing any agent
code.

**Suggested grid map encoding:**
- `'S'` — start cell
- `'G'` — goal cell
- `'.'` — empty cell
- `'#'` — wall
- `'H'` — hazard

### Milestone 2 — Agent (45–60 min)

Implement `QLearningAgent` with:

```python
agent = QLearningAgent(
    n_states, n_actions,
    alpha=0.1,      # learning rate
    gamma=0.99,     # discount factor
    epsilon=1.0,    # initial exploration rate
    epsilon_min=0.01,
    epsilon_decay=0.995
)

action = agent.select_action(state)
agent.update(state, action, reward, next_state, done)
```

The Q-table is a numpy array of shape `(n_states, n_actions)`. Initialize
all values to zero. The Bellman update is:

```
Q(s, a) ← Q(s, a) + α [r + γ · max_a' Q(s', a') - Q(s, a)]
```

Confirm the update works by running one episode by hand with print statements
before adding the full training loop.

### Milestone 3 — Training loop and convergence (60–90 min)

Write the episode loop, collect per-episode total rewards, and plot the reward
curve. The agent should reach the goal consistently within 500–1000 episodes
on the 5x5 grid with default hyperparameters.

A converged agent is one where:
- The rolling-average reward (window = 50 episodes) flattens
- The agent reliably reaches the goal in near-optimal steps when epsilon is
  set to 0 (pure exploitation)

If the agent is not converging, check: state encoding, Q-table indexing,
reward signs, and epsilon decay rate before tuning hyperparameters.

### Milestone 4 — Visualization and analysis (60–90 min)

Plot the learned policy. For each non-wall, non-goal cell, compute
`argmax_a Q(s, a)` and draw an arrow in that direction on a matplotlib grid.
The arrow map should form a coherent flow toward the goal — not random
directions.

Then run the hyperparameter sensitivity experiments described in the analysis
questions and write up your findings in `analysis.md`.

---

## Self-Assessment Checklist

Use this checklist to evaluate your own work before calling the project
complete. No external grader will check these — be honest with yourself.

- [ ] The agent consistently reaches the goal within 30 steps (5x5 grid) when
  evaluated with epsilon = 0 after training
- [ ] The reward curve shows a clear upward trend with visible convergence
  plateau; the rolling average is plotted alongside raw episode rewards
- [ ] The policy arrow map shows directional coherence — arrows form paths
  that lead toward the goal rather than contradicting each other
- [ ] Walls are correctly blocked: stepping into a wall cell does not move the
  agent and incurs the wall penalty
- [ ] You tested your environment on at least two different grid layouts and
  the agent learned a correct policy on both
- [ ] You ran at least three hyperparameter configurations and documented how
  each parameter (alpha, gamma, epsilon decay) changed convergence speed and
  final policy quality
- [ ] You can explain why gamma < 1 is important for this task and what would
  happen with gamma = 1
- [ ] `analysis.md` answers all five analysis questions with concrete
  observations from your experiments, not general textbook statements

---

## Analysis Questions

Answer these in `analysis.md` after completing the implementation. Each
answer should reference specific observations from your own experiments.

1. **Convergence diagnosis.** How many episodes did your agent need to
   converge on the 5x5 grid? What metric did you use to define convergence?
   How did this change when you increased grid size to 8x8?

2. **Learning rate sensitivity.** Run three experiments with alpha = 0.01,
   0.1, and 0.5. Plot all three reward curves on the same axes. Which
   converged fastest? Which produced the best final policy? Were they the same
   configuration?

3. **Discount factor interpretation.** Set gamma = 0.5 and observe the
   learned policy. Does the agent still find the optimal path? Explain in
   terms of the Bellman equation why a low gamma changes behavior near the
   goal vs far from the goal.

4. **Exploration trade-off.** Compare epsilon-greedy with epsilon_min = 0.01
   vs epsilon_min = 0.2. What does the agent sacrifice by maintaining high
   exploration permanently? What does it sacrifice by annealing to near-zero?

5. **Failure mode identification.** Deliberately break your agent in one of
   the following ways — then show the resulting policy arrow map as evidence:
   (a) set alpha = 1.0, (b) initialize Q-table to large positive values, or
   (c) forget to decay epsilon. What did the broken policy look like and why?

---

## Stretch Goals

These are optional. Do them if the base project felt straightforward and you
want to push further.

**Algorithmic extensions:**
- Implement SARSA alongside Q-learning and compare their policies on a grid
  with a narrow hazard corridor (the classic "cliff edge" experiment)
- Add eligibility traces (Q(lambda)) and measure whether it converges faster
  on a large grid
- Implement Double Q-learning and show on a constructed example where standard
  Q-learning overestimates action values

**Environment extensions:**
- Add stochastic transitions: with probability 0.1, the agent slips to a
  random adjacent cell instead of the intended one; observe the effect on the
  learned policy
- Implement a non-stationary version where the goal location changes every 200
  episodes; compare how quickly the agent re-learns

**Visualization extensions:**
- Animate the training process: show the Q-table heatmap evolving episode by
  episode as a matplotlib animation saved to a GIF
- Plot the value function `V(s) = max_a Q(s, a)` as a heatmap overlaid on the
  grid; verify it decreases with distance from the goal

---

## Getting Started

```bash
pip install numpy matplotlib
```

Recommended build order:
1. Write `GridWorld` and test it with a hand-coded action sequence
2. Write `QLearningAgent` and verify one manual update step with pen and paper
3. Wire together in `train.py` and run 100 episodes; check rewards are not
   all identical
4. Extend to 2000 episodes and add reward curve plotting
5. Add policy visualization
6. Run hyperparameter experiments and write `analysis.md`

Resist the urge to look up Q-learning implementations until after you have
made a genuine attempt. The debugging process is where the learning happens.

---

## Next Steps

After completing this project:

- **Project 2 (CartPole Championship):** Apply deep Q-networks to
  environments where the state space is too large for a Q-table
- **Module 03 (Temporal Difference Learning):** Covers SARSA, Expected SARSA,
  and the on-policy vs off-policy distinction in depth
- **Module 04 (Function Approximation):** The conceptual bridge from tabular
  Q-learning to DQN
