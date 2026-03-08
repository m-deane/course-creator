# Reinforcement Learning — Portfolio Projects

Three projects that build on each other from tabular RL to deep RL to a
domain-specific production environment. Each is self-contained: you can do
any one independently, or work through them in order for a coherent arc from
foundations to a deployable system.

---

## Projects at a Glance

| # | Title | Difficulty | Est. Time | Core Algorithms | Tools |
|---|-------|-----------|-----------|----------------|-------|
| 1 | GridWorld Navigator | Beginner | 4–6 hrs | Tabular Q-learning | numpy, matplotlib |
| 2 | CartPole Championship | Intermediate | 8–12 hrs | DQN, REINFORCE, A2C | numpy, torch, gymnasium |
| 3 | Portfolio Optimization with RL | Advanced | 15–20 hrs | PPO | torch, gymnasium, yfinance |

---

## Project 1 — GridWorld Navigator (Beginner)

**Location:** `project_1_beginner/README.md`

**What you build:** A tabular Q-learning agent that solves custom gridworld
environments of your own design. You implement the environment from scratch,
the agent, the training loop, and a policy visualization that shows learned
navigation arrows on the grid.

**Key concepts practiced:**
- Markov Decision Process formulation
- Bellman update and Q-table mechanics
- Epsilon-greedy exploration schedules
- Convergence diagnostics and reward curve plotting
- Hyperparameter sensitivity analysis (learning rate, gamma, epsilon decay)

**Why it matters:** Most deep RL bugs are conceptual errors about MDPs,
rewards, or the Bellman equation — not PyTorch bugs. Getting fluent with
tabular RL in a fully transparent environment eliminates the entire class of
"my agent isn't learning but I don't know why" problems.

---

## Project 2 — CartPole Championship (Intermediate)

**Location:** `project_2_intermediate/README.md`

**What you build:** Implementations of three canonical algorithms — DQN,
REINFORCE, and A2C — benchmarked head-to-head on CartPole-v1 and
LunarLander-v2. You track experiments systematically and write a structured
comparison of each algorithm's trade-offs.

**Key concepts practiced:**
- Value-based vs policy gradient vs actor-critic architectures
- Experience replay and target networks (DQN)
- Monte Carlo returns and variance reduction (REINFORCE with baseline)
- Advantage estimation and shared network heads (A2C)
- Statistical comparison with confidence intervals across seeds
- Sample efficiency analysis

**Why it matters:** Most practitioners pick a default algorithm ("use PPO for
everything") without understanding when each approach fails. This project
builds the judgment to choose the right tool for a given environment.

---

## Project 3 — Portfolio Optimization with RL (Advanced)

**Location:** `project_3_advanced/README.md`

**What you build:** A custom Gymnasium trading environment backed by real
S&P 500 data, a PPO agent that learns a rebalancing policy, and a full
backtesting framework that reports risk-adjusted performance against a
buy-and-hold baseline.

**Key concepts practiced:**
- Custom Gymnasium environment design and API compliance
- Observation and action space engineering for financial data
- PPO clipped surrogate objective and GAE
- Walk-forward backtesting (no data leakage)
- Sharpe ratio, max drawdown, and Calmar ratio
- Hyperparameter sensitivity for financial RL

**Why it matters:** Financial RL separates researchers from practitioners.
The hard parts are not the algorithm — they are observation design, reward
shaping, preventing lookahead bias, and evaluating results without overfitting
to the test period. This project builds all of those skills.

---

## Prerequisites by Project

**Project 1 requires:**
- Python, numpy, matplotlib
- MDP fundamentals (state, action, reward, transition)

**Project 2 requires:**
- Project 1 concepts OR equivalent tabular RL experience
- PyTorch basics (nn.Module, optimizers, autograd)
- gymnasium installation

**Project 3 requires:**
- Project 2 concepts OR equivalent deep RL experience
- pandas, yfinance (or comfort downloading financial CSVs)
- Understanding of financial return metrics

---

## How to Use These Projects

These are portfolio projects, not graded assignments. There are no submission
deadlines and no automated scoring server. The self-assessment checklists in
each README are the only evaluation criteria that matter — use them honestly.

A complete project means: code runs end-to-end, results match the expected
behavior described in the README, and you can explain every design decision
out loud. Reaching that bar is more valuable than completing all three
projects superficially.

Suggested path for a 4-week self-study sprint:
- **Week 1–2:** Project 1 (solidify tabular foundations)
- **Week 3–4:** Project 2 (deep RL implementations)
- **Week 5–6:** Project 3 (domain application and production thinking)
