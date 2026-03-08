# Project 3: Portfolio Optimization with RL (Advanced)

## What You'll Build

A Proximal Policy Optimization (PPO) agent that learns a portfolio
rebalancing policy from historical stock data. You build three tightly
integrated components: a custom Gymnasium environment that turns financial
time series into an RL problem, a PPO agent trained on the resulting
environment, and a walk-forward backtesting framework that honestly evaluates
out-of-sample performance against a buy-and-hold baseline using risk-adjusted
metrics. The finished project is a deployable research artifact and a strong
portfolio centerpiece for ML or quant finance roles.

**Tools:** numpy, pandas, matplotlib, torch, gymnasium, yfinance (or CSV
fallback).

**Expected time:** 15–20 hours.

---

## Learning Objectives

- Design a custom Gymnasium environment that correctly represents a sequential
  financial decision problem without lookahead bias
- Engineer observation and action spaces that give a PPO agent the information
  it needs without reward hacking opportunities
- Implement PPO with Generalized Advantage Estimation (GAE) and the clipped
  surrogate objective from scratch
- Construct a walk-forward backtesting framework that strictly separates the
  training, validation, and test periods
- Compute and interpret Sharpe ratio, maximum drawdown, and Calmar ratio;
  explain what each metric captures that the others miss
- Document hyperparameter sensitivity in a domain where overfitting is not
  always obvious from training curves alone

---

## The Problem

A portfolio manager holds positions in N assets and rebalances at the close of
each trading day. The objective is to maximize risk-adjusted cumulative return
over a multi-year test period, outperforming a passive buy-and-hold strategy
on the same assets.

This is harder than CartPole or LunarLander in every non-algorithmic
dimension: the signal-to-noise ratio is low, rewards are delayed and noisy,
the "optimal" policy is non-stationary, and overfitting to the training period
is always possible and often silent.

**Candidate asset universe (choose one or define your own):**

Option A — Sector ETFs (10 assets, lower noise):
XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE

Option B — Large-cap individual stocks (5 assets, higher noise):
AAPL, MSFT, GOOGL, JPM, XOM

Option C — Your own selection: any 5–15 liquid instruments available in
yfinance with at least 8 years of daily history.

**Data split (strict, do not adjust after choosing):**
- Training: earliest 60% of available daily data
- Validation: next 20%
- Test: final 20% (never touched until final evaluation)

---

## Required Deliverables

1. `environment.py` — `PortfolioEnv` class implementing the full Gymnasium
   API, including `reset()`, `step()`, `observation_space`, and
   `action_space`; passes `check_env()` without errors
2. `ppo.py` — PPO agent with Gaussian policy, value function, GAE advantage
   computation, and clipped surrogate loss
3. `train.py` — training loop that saves the best checkpoint by validation
   Sharpe ratio; logs training return and validation Sharpe every 5 updates
4. `backtest.py` — evaluation script that runs the saved checkpoint on the
   test period and computes all six risk metrics for both agent and baseline
5. `data/prices.csv` — adjusted close prices for your chosen asset universe
6. `results/` — directory containing the training curve plot, cumulative
   return plot, and metrics table for the final evaluation
7. `analysis.md` — written responses to the five analysis questions

---

## Dataset Acquisition

Pull adjusted close prices with yfinance:

```python
import yfinance as yf
import pandas as pd

tickers = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE"]
data = yf.download(tickers, start="2010-01-01", end="2024-12-31", auto_adjust=True)
prices = data["Close"].dropna()
prices.to_csv("data/prices.csv")
```

**CSV fallback:** If yfinance is unavailable, download adjusted close prices
for the same tickers from any free source (Yahoo Finance web UI, Stooq, FRED)
and place the file at `data/prices.csv` with dates as the index and tickers as
columns. The environment code should not care which method was used.

Commit `data/prices.csv` to your repository so results are reproducible.

---

## Environment Specification

Your `PortfolioEnv` must pass `gymnasium.utils.env_checker.check_env(env)`
without errors. This is a hard requirement, not a suggestion.

### Action Space

`gymnasium.spaces.Box(low=0.0, high=1.0, shape=(N,), dtype=np.float32)`

Each action is a vector of target portfolio weights summing to 1.0. Apply
softmax normalization inside the environment to enforce the sum-to-one
constraint regardless of raw agent output.

Do not include cash as a mandatory asset. Optionally, you may add a cash
position as the (N+1)th weight; document clearly if you do.

### Observation Space

`gymnasium.spaces.Box(low=-inf, high=inf, shape=(obs_dim,), dtype=np.float32)`

Minimum required observation features (computed entirely from data available
at the start of the current timestep — no peeking at future prices):

- Log returns for each asset over the past L days (flattened), L = 20
- Current portfolio weights (from previous action)
- Portfolio value normalized by initial value

Optional features that frequently improve performance (include with caution —
each added feature is an overfitting opportunity):
- Rolling volatility per asset (20-day)
- Rolling Sharpe ratio per asset (60-day)
- Day-of-week or month-of-year encoding

### Reward

The reward at each step is the portfolio log return:

```
r_t = log(V_t / V_{t-1})
```

where `V_t` is the portfolio value at the close of day t after rebalancing,
net of transaction costs.

**Transaction cost model:** Apply a proportional cost of 10 basis points
(0.001) on the absolute value of each weight change:

```
cost_t = 0.001 * sum(|w_t - w_{t-1}|)
r_t = log_return_t - cost_t
```

Do not add a shaped reward (e.g., Sharpe-based reward) without documenting
the exact formulation and measuring its effect on behavior. Reward shaping
often produces agents that optimize the shape function, not actual returns.

### Episode Structure

Each episode covers the full training period. Reset returns the observation
at the first valid timestep (after the L-day lookback window). The episode
ends at the last training-period timestep.

Do not use episodic resets mid-training-period. The financial environment has
no natural episode boundary — forcing artificial resets degrades policy
learning.

---

## PPO Specification

Implement PPO from scratch. Do not import stable-baselines3, rllib, or any
other RL library for the agent. You may use them to verify your results after
the fact.

**Network architecture:**

Shared MLP backbone with two heads:
- Input: observation vector
- Shared layers: [256, 256] hidden units, tanh activations
- Policy head: outputs mean of a Gaussian distribution over the action space
  (before softmax normalization); covariance is a learned diagonal matrix
- Value head: scalar output

Use tanh activations (not ReLU) for financial observations. ReLU can produce
dead neurons with normalized return features that frequently hit zero.

**PPO hyperparameters (starting point, tune from here):**

| Parameter | Value |
|-----------|-------|
| Rollout length (T) | 252 (approximately 1 year of trading days) |
| Number of epochs per update | 10 |
| Minibatch size | 64 |
| Clip ratio (epsilon) | 0.2 |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 0.5 |
| GAE lambda | 0.95 |
| Gamma | 0.99 |
| Gradient clip norm | 0.5 |
| Learning rate | 3e-4 |

**Training loop:**

1. Collect T steps of experience using the current policy
2. Compute GAE advantages and normalize them (zero mean, unit variance)
3. Run K epochs of minibatch gradient updates using the clipped surrogate loss
4. Evaluate on the validation set every 5 rollout updates (do not train on
   validation data — evaluation only)
5. Save the checkpoint with the best validation Sharpe ratio for final testing

---

## Backtesting Framework

The backtesting framework evaluates the saved policy checkpoint on the held-out
test period. It must be a completely separate code path from training.

```python
def backtest(env, agent, period="test"):
    obs, _ = env.reset(period=period)
    portfolio_values = [env.initial_value]
    done = False
    while not done:
        action = agent.act_greedy(obs)   # no exploration
        obs, reward, done, _, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
    return portfolio_values
```

Implement `backtest_baseline(prices, period="test")` that computes the
equal-weight buy-and-hold portfolio over the same test period (rebalanced
monthly). This is your primary comparison.

**Metrics to report for both the RL agent and the baseline:**

| Metric | Formula |
|--------|---------|
| Annualized return | `(V_T/V_0)^(252/T) - 1` |
| Annualized volatility | `std(daily log returns) * sqrt(252)` |
| Sharpe ratio | `annualized return / annualized volatility` (assume 0% risk-free rate) |
| Maximum drawdown | `max over t of (peak_to_t - V_t) / peak_to_t` |
| Calmar ratio | `annualized return / abs(max drawdown)` |
| Turnover | `mean daily sum of absolute weight changes` |

Plot: cumulative portfolio value curves for both agent and baseline on the
same axes, with the training/test split clearly marked.

---

## Suggested Milestones

### Milestone 1 — Environment and data pipeline (3–4 hrs)

- Download and clean price data; commit `data/prices.csv`
- Implement `PortfolioEnv` with all required spaces and reward logic
- Pass `check_env(env)` without errors
- Run 5 random-policy episodes and verify: portfolio values are always
  positive, rewards are in a plausible range (-0.05 to +0.05 per day),
  observations contain no NaNs, and the episode length matches the training
  period

Spending extra time here pays compounding returns. A subtle environment bug
(wrong sign on reward, future leak in observations, incorrect weight
normalization) will produce a trained agent that appears to learn but performs
randomly on the test set.

### Milestone 2 — PPO implementation (4–5 hrs)

- Implement the policy and value networks with Gaussian output
- Implement GAE advantage computation
- Implement the clipped surrogate loss, value loss, and entropy bonus
- Verify the training loop runs end-to-end without errors on CartPole-v1
  before applying it to the portfolio environment (use discrete action version
  with softmax output)

This verification step is critical. If PPO solves CartPole, algorithm bugs
are ruled out and any financial environment failures are environment or
hyperparameter problems.

### Milestone 3 — Training and validation (4–5 hrs)

- Train on the training period; track training return and validation Sharpe
  every 5 updates
- Plot: training return over updates, validation Sharpe over updates
- Confirm the agent is learning (validation Sharpe improves from near-zero
  early in training)
- Save the checkpoint with the best validation Sharpe

Common failure mode: the agent learns to hold one asset regardless of market
conditions. Check the action distribution over a validation episode — if
weights collapse to near [1, 0, 0, ...] persistently, add or increase the
entropy bonus.

### Milestone 4 — Backtesting and analysis (4–6 hrs)

- Run `backtest()` with the best checkpoint on the test period
- Compute all six metrics for both agent and baseline
- Produce the cumulative return plot
- Run the three hyperparameter sensitivity experiments in the analysis section
- Write `analysis.md`

---

## Self-Assessment Checklist

- [ ] `check_env(PortfolioEnv())` passes without errors or warnings
- [ ] Observations at every step contain no NaN or infinite values; verified
  with an assertion in `env.step()`
- [ ] The environment uses no future data: the observation at time t is
  constructed from prices at times t-L through t only; verified by reviewing
  the observation construction code line by line
- [ ] The PPO implementation solves CartPole-v1 (mean reward >= 450 over 10
  evaluation episodes) before being applied to the portfolio environment
- [ ] The trained agent achieves a Sharpe ratio > 0 on the test period (the
  agent makes positive risk-adjusted returns, not just positive raw returns)
- [ ] The agent beats the equal-weight buy-and-hold baseline on at least two
  of the following three metrics on the test period: annualized return, Sharpe
  ratio, or max drawdown
- [ ] `analysis.md` documents at least three hyperparameter configurations,
  reports validation and test metrics for each, and draws a conclusion about
  which hyperparameters most affect out-of-sample performance
- [ ] The test period was not touched until final evaluation; you did not
  adjust hyperparameters based on test performance

---

## Analysis Questions

Answer these in `analysis.md`. Evidence must come from your own experiments.

1. **Lookahead bias audit.** Walk through your observation construction code
   and identify every place where future data could theoretically leak into
   the observation. Are you certain none of them do? How did you verify this?

2. **Reward shaping experiment.** Train a second agent with an alternative
   reward: replace log return with `r_t = 252-day rolling Sharpe ratio` (or
   another shaped reward of your choice). Compare test-period performance to
   the log-return agent. Did shaping improve results? What behavior did the
   shaped agent exhibit that the original did not?

3. **Hyperparameter sensitivity.** Choose one hyperparameter (rollout length T,
   entropy coefficient, or learning rate) and train three variants. Plot
   validation Sharpe over training time for all three. Which was most
   sensitive? Did the best validation configuration also produce the best test
   configuration?

4. **Drawdown anatomy.** Identify the two largest drawdown periods in your
   agent's test-period equity curve. For each, describe what was happening in
   the market during that period (draw on your knowledge of the time series,
   not any additional data). Did the agent exit positions or hold through? What
   does this suggest about your reward signal?

5. **Honest evaluation.** If a practitioner were to deploy this agent with
   real capital, what are the three most significant risks that your backtesting
   framework did not capture? What would you add to the framework to address
   each?

---

## Stretch Goals

**Algorithm:**
- Implement a recurrent policy (LSTM actor-critic) and compare it to the MLP
  policy on LunarLander-v2 first, then on the portfolio environment; report
  whether memory improves sample efficiency
- Add a risk penalty term to PPO's objective: subtract a fraction of realized
  portfolio variance from the reward at each update; measure the effect on the
  Sharpe-drawdown trade-off

**Environment:**
- Extend the environment to include short-selling by allowing weights in
  `[-0.5, 1.5]` with a leverage constraint; re-derive the transaction cost
  model and verify environment correctness
- Add a second asset class (e.g., bonds via TLT, or commodities via GLD) and
  measure whether diversification benefits emerge from the trained policy

**Evaluation:**
- Implement a proper walk-forward validation: split the test period into 4
  rolling windows, re-train on expanding data for each window, and report
  average out-of-sample metrics across windows
- Bootstrap the test-period Sharpe ratio (1,000 samples with replacement) and
  report the 95% confidence interval; use this to assess whether the agent's
  outperformance is statistically meaningful

**Comparison:**
- Implement a mean-variance optimization (Markowitz) baseline using rolling
  estimates of expected returns and covariance; compare it to PPO on the same
  test period to contextualize RL's advantage (or lack thereof) vs classical
  finance methods

---

## Getting Started

```bash
pip install numpy pandas matplotlib torch gymnasium yfinance
```

**Critical setup check before writing any agent code:**

```python
from gymnasium.utils.env_checker import check_env
env = PortfolioEnv(prices, period="train")
check_env(env)   # must pass before proceeding
```

Recommended build order:
1. Download data and write the data loading / train-val-test split code
2. Implement the observation construction; verify manually with print statements
3. Implement the reward and step logic; run random-policy episodes
4. Call `check_env()` and fix all errors and warnings
5. Verify PPO on CartPole before connecting it to PortfolioEnv
6. Connect PPO to PortfolioEnv and run 10 updates; confirm the loss decreases
7. Full training run with validation tracking
8. Backtest only after deciding the model is final

The most common mistake in this project is moving from step 4 to step 8 too
quickly. A 20-hour project that produces an agent that secretly uses future
data is worth less than a 20-hour project that produces an agent that provably
does not.

---

## Next Steps

After completing this project:

- **Module 08 (Model-Based RL):** Learn how to learn a model of the
  environment and plan within it; directly applicable to environments with
  expensive simulation like financial backtesting
- **Module 09 (Frontiers):** Offline RL for learning from historical datasets
  without any live environment interaction — the natural next step for
  production financial RL systems
- Research: Jiang et al. (2017) "A Deep Reinforcement Learning Framework for
  the Financial Portfolio Management Problem" — the paper closest to this
  project's architecture; compare their approach to yours
