# Project 2: CartPole Championship (Intermediate)

## What You'll Build

Three canonical deep RL algorithms — DQN, REINFORCE, and A2C — implemented
from scratch and benchmarked head-to-head on CartPole-v1 and LunarLander-v2.
You design a structured experiment, track runs across multiple seeds, plot
learning curves with confidence intervals, and write an honest analysis of
each algorithm's failure modes and strengths. The finished project is a
mini benchmark study you can include in a portfolio and reference when
choosing algorithms for new problems.

**Tools:** numpy, matplotlib, gymnasium, torch.

**Expected time:** 8–12 hours.

---

## Learning Objectives

- Implement DQN with experience replay and a target network, understanding
  why each component is necessary
- Implement REINFORCE with a learned baseline and diagnose its variance problem
  empirically
- Implement A2C with shared network weights and explain the role of the entropy
  bonus
- Design a reproducible experiment with multiple random seeds and report
  results with mean and 95% confidence intervals
- Quantify sample efficiency as episodes-to-threshold and articulate concrete
  trade-offs between value-based and policy gradient methods

---

## The Problem

CartPole-v1 and LunarLander-v2 are the canonical benchmarks for introductory
deep RL. They are simple enough to solve in minutes on a laptop but rich
enough to expose real algorithmic differences.

**CartPole-v1:**
- State: 4 continuous values (cart position, velocity, pole angle, angular
  velocity)
- Actions: 2 discrete (push left, push right)
- Reward: +1 for each timestep the pole stays upright
- Solved threshold: average reward >= 475 over 100 consecutive episodes

**LunarLander-v2:**
- State: 8 continuous values (position, velocity, angle, angular velocity, leg
  contact flags)
- Actions: 4 discrete (do nothing, fire left, fire main, fire right)
- Reward: shaped reward for landing on the pad; solved threshold >= 200
- Significantly harder than CartPole — expect longer training and higher
  variance

You will train all three algorithms on both environments, producing six
training runs per algorithm (3 seeds x 2 environments).

---

## Required Deliverables

1. `dqn.py` — DQN implementation with replay buffer and target network
2. `reinforce.py` — REINFORCE with a learned value function baseline
3. `a2c.py` — A2C with shared actor-critic network and entropy regularization
4. `experiment.py` — unified training and evaluation loop that runs all three
   algorithms across seeds and environments, saving results to `results/`
5. `plot_results.py` — plots learning curves (mean ± 95% CI across seeds) for
   each algorithm on each environment, plus a sample efficiency comparison bar
   chart
6. `analysis.md` — written comparison covering the questions in the
   self-assessment section

---

## Algorithm Specifications

### DQN

Architecture: Two hidden layers (128, 128 units, ReLU). Input = state
dimension, output = Q-values for each action.

Required components:
- Replay buffer with capacity 10,000 transitions
- Target network updated every 100 gradient steps (hard copy, not soft update)
- Epsilon-greedy with linear decay from 1.0 to 0.01 over first 10,000 steps
- Batch size 64, Adam optimizer, lr = 1e-3
- Huber loss (smooth L1) for the Bellman target

The Bellman target for non-terminal transitions:
```
y = r + γ · max_a' Q_target(s', a')
```

Training does not start until the replay buffer has at least 1,000 transitions.

### REINFORCE with Baseline

Architecture: Two separate networks — policy network and value network — each
with two hidden layers (64, 64 units, ReLU). Policy network outputs a
softmax distribution over actions; value network outputs a scalar state value.

Required components:
- Monte Carlo returns with gamma = 0.99
- Advantage estimate: `A_t = G_t - V(s_t)` where `V(s_t)` is the learned
  baseline
- Policy gradient loss: `-log π(a_t|s_t) · A_t` averaged over the episode
- Value loss: MSE between predicted values and Monte Carlo returns
- Normalize advantages per episode (subtract mean, divide by std)
- Updates once per episode (not per step)

### A2C

Architecture: Single shared network with two heads — policy head (softmax
over actions) and value head (scalar). Two hidden layers (128, 128 units,
ReLU), shared across both heads.

Required components:
- n-step returns with n = 5 steps
- Generalized Advantage Estimation (GAE) with lambda = 0.95
- Policy gradient loss using advantages from GAE
- Value loss: MSE between n-step bootstrapped returns and value head
- Entropy bonus: `- β · H(π)` added to the loss with β = 0.01
- Single optimizer for the entire shared network
- Update every n steps (not once per episode)

---

## Suggested Milestones

### Milestone 1 — DQN on CartPole (2–3 hrs)

Implement DQN and verify it solves CartPole-v1 (average reward >= 475 over
100 episodes). Plot the learning curve. Confirm your replay buffer and target
network are working by temporarily disabling each and observing degraded
performance.

A DQN that does not use a target network typically diverges or oscillates
dramatically on CartPole — this divergence is itself useful to plot and
understand.

### Milestone 2 — REINFORCE on CartPole (1–2 hrs)

Implement REINFORCE with baseline. It should solve CartPole but with higher
variance than DQN and more episodes required. Plot the learning curve
alongside DQN on the same axes. The high variance of REINFORCE should be
visible in the confidence interval band.

### Milestone 3 — A2C on CartPole (2–3 hrs)

Implement A2C. It should be faster than REINFORCE (lower sample complexity)
and more stable, but possibly less sample-efficient than DQN on simple tasks.
Plot all three curves together.

### Milestone 4 — LunarLander-v2 and full analysis (2–4 hrs)

Transfer all three implementations to LunarLander-v2. You should not need to
change the algorithms — only the network input dimension and the number of
training steps (increase to ~500k steps for DQN, ~3000 episodes for
REINFORCE/A2C). Run 3 seeds each, produce the final plots, and write
`analysis.md`.

---

## Experiment Protocol

Use this protocol exactly so your results are reproducible and comparable.

**Seeds:** Run each algorithm with seeds [42, 123, 456].

**Evaluation:** Every 50 episodes, evaluate the current policy for 10
episodes with epsilon = 0 (DQN) or greedy action selection (policy gradient).
Record the mean evaluation reward. This is what goes on the learning curve.

**Sample efficiency threshold:**
- CartPole-v1: episodes to first reach average evaluation reward >= 400
- LunarLander-v2: episodes to first reach average evaluation reward >= 150

If an algorithm never reaches threshold within the budget, record `inf`.

**Training budget:**
- CartPole-v1: 1000 episodes per run
- LunarLander-v2: 5000 episodes per run

**Saving results:** Save per-seed evaluation histories as numpy arrays in
`results/{env}/{algorithm}/seed_{seed}_eval.npy`. This lets you regenerate
plots without re-training.

---

## Self-Assessment Checklist

- [ ] All three agents solve CartPole-v1 (mean evaluation reward >= 450 over
  the final 100 evaluation checkpoints, averaged across seeds)
- [ ] Learning curves show mean reward with a shaded 95% confidence interval
  band computed across the three seeds — not just a single run
- [ ] The REINFORCE confidence interval is visibly wider than A2C's, showing
  its higher variance empirically
- [ ] You produced a sample efficiency bar chart showing episodes-to-threshold
  for all three algorithms on both environments
- [ ] The replay buffer in DQN is verified: training does not start until the
  buffer has at least 1,000 transitions; disabling the target network causes
  visible instability
- [ ] `analysis.md` explains the DQN target network in your own words — not
  copied from a textbook — and points to a specific event in your training
  curve that illustrates why it matters
- [ ] `analysis.md` contains a concrete recommendation: for a new environment
  with a continuous state space, sparse rewards, and 4 discrete actions, which
  of the three algorithms would you start with and why?
- [ ] All code runs end-to-end with `python experiment.py` without
  modifications

---

## Analysis Questions

Answer these in `analysis.md` using your own experimental results as evidence.

1. **Target network necessity.** Train DQN without a target network (use the
   online network for bootstrap targets). Plot the resulting learning curve
   alongside the standard DQN curve. Describe what you observe and explain
   the instability in terms of the Bellman update.

2. **Variance comparison.** Compare the confidence interval width of REINFORCE
   vs A2C on the same plot. At what point in training is the gap largest? Why
   does the n-step advantage in A2C reduce variance relative to full Monte
   Carlo returns in REINFORCE?

3. **Entropy bonus effect.** Train A2C with β = 0 (no entropy bonus) and
   β = 0.05 (high entropy bonus). Compare final performance and convergence
   speed on LunarLander-v2. What symptom appears when β is too high?

4. **Sample efficiency.** Which algorithm is most sample-efficient on
   CartPole? On LunarLander? Does the answer change between environments?
   Propose a hypothesis for why.

5. **Failure mode post-mortem.** Identify one case where an algorithm clearly
   failed or underperformed across seeds (not just a single bad run). Diagnose
   the likely cause and describe one change to the algorithm or its
   hyperparameters that you would test next.

---

## Stretch Goals

**Algorithmic:**
- Implement Double DQN (decouple action selection and action evaluation) and
  show whether it reduces overestimation on LunarLander compared to standard
  DQN
- Add Prioritized Experience Replay to DQN; plot the priority distribution
  after training to verify high-error transitions are sampled more often
- Implement PPO as a fourth algorithm and include it in the comparison; this
  is excellent preparation for Project 3

**Experimental rigor:**
- Run 5 seeds instead of 3 and compute 95% bootstrap confidence intervals
  rather than assuming Gaussian distributions
- Add a learning curve smoothed with an exponential moving average alongside
  the raw evaluation points
- Produce a performance profile plot (Agarwal et al., 2021 style) across the
  six algorithm-environment combinations

**Environment:**
- Apply your best algorithm (your choice) to a third gymnasium environment
  of your choosing; document what hyperparameter changes were necessary and
  explain why

---

## Getting Started

```bash
pip install numpy matplotlib torch gymnasium
# For LunarLander physics engine:
pip install gymnasium[box2d]
```

Recommended build order:
1. Stand up the training loop with a random agent and verify gymnasium works
2. Implement the replay buffer in isolation and test it with dummy data
3. Build DQN's network, then wire in the buffer and training step
4. Verify DQN solves CartPole before touching REINFORCE or A2C
5. Implement REINFORCE; the training loop is structurally different (per-
   episode rather than per-step) — take care with the return computation
6. Implement A2C; reuse REINFORCE's policy loss, add the value loss and
   entropy term
7. Build the experiment runner and move to LunarLander

Do not start LunarLander until all three algorithms work on CartPole. The
environments are more similar than they appear — algorithm bugs hide behind
environment complexity.

---

## Next Steps

After completing this project:

- **Project 3 (Portfolio Optimization):** Apply PPO to a custom financial
  environment where the real challenge is observation and reward design, not
  the algorithm itself
- **Module 07 (Advanced Policy Optimization):** PPO, TRPO, and the theory
  behind clipped surrogate objectives
- **Module 06 (Policy Gradient Methods):** Derivation of the policy gradient
  theorem and connections between REINFORCE, A2C, and PPO
