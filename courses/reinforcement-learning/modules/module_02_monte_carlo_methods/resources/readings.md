# Module 2: Monte Carlo Methods — Recommended Readings

## Essential Reading

- **Reinforcement Learning: An Introduction (2nd ed.), Chapter 5** — Sutton & Barto, 2018. Develops first-visit and every-visit MC prediction, MC control with exploring starts, and importance sampling for off-policy estimation; the blackjack example provides a concrete testbed.

- **Reinforcement Learning with Replacing Eligibility Traces** — Singh & Sutton, 1996. Introduces replacing traces as an alternative to accumulating traces, with empirical comparisons on random-walk tasks that illuminate the bias-variance tradeoff central to MC methods.

## Supplementary Reading

- **Monte Carlo Statistical Methods (2nd ed.), Chapter 3** — Robert & Casella, 2004. The statistical perspective on Monte Carlo estimation, covering variance reduction techniques (control variates, antithetic sampling) that apply directly to value function estimation.

- **Importance Sampling for Reinforcement Learning** — Precup, Sutton & Singh, 2000. Extends importance sampling to long trajectories with eligibility-trace-weighted estimators; the per-decision IS estimator derived here remains the state of the art for off-policy MC.

- **On-Policy vs. Off-Policy Monte Carlo** — Sutton & Barto, 2018 (S&B §5.5–5.7). These three sections within Chapter 5 deserve independent study: they trace the progression from every-visit MC to weighted importance sampling with clear variance analysis.

## Online Resources

- **David Silver's UCL RL Lecture 4: Model-Free Prediction** — Silver, DeepMind/UCL. The first half of Lecture 4 covers MC prediction with visual comparisons against DP backups; search "David Silver RL lecture 4 slides".

- **Spinning Up: Introduction to RL — Part 3: Intro to Policy Optimization** — OpenAI. While policy-gradient focused, the section on Monte Carlo returns clearly explains the high-variance nature of full-trajectory estimates that this module addresses.

- **Rich Sutton's research page** — Sutton, University of Alberta. Original technical reports on MC methods and eligibility traces, including the 1996 Singh & Sutton paper, are available directly from Sutton's faculty page; search "Richard Sutton University of Alberta publications".
