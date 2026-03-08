# Module 6: Policy Gradient Methods — Recommended Readings

## Essential Reading

- **Reinforcement Learning: An Introduction (2nd ed.), Chapter 13** — Sutton & Barto, 2018. Derives the policy gradient theorem from first principles, develops REINFORCE and actor-critic baselines, and analyzes variance reduction; the theoretical spine of this module.

- **Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning** — Williams, 1992. The REINFORCE paper; introduces the likelihood-ratio gradient estimator and the self-critical baseline, foundational to every policy gradient method that follows.

- **Asynchronous Methods for Deep Reinforcement Learning** — Mnih, Badia, Mirza, Graves, Lillicrap, Harley, Silver & Kavukcuoglu, 2016. Introduces A3C, which parallelizes actor-critic training across CPU threads; demonstrated that asynchronous gradient accumulation stabilizes training without experience replay.

- **High-Dimensional Continuous Control Using Generalized Advantage Estimation** — Schulman, Moritz, Levine, Jordan & Abbeel, 2016. Introduces GAE(λ), a bias-variance tradeoff interpolation between TD and MC advantage estimates; GAE is now the default advantage estimator in virtually all modern policy gradient implementations.

## Supplementary Reading

- **Policy Gradient Methods for Reinforcement Learning with Function Approximation** — Sutton, McAllester, Singh & Mansour, 2000. Proves the policy gradient theorem with function approximation and establishes compatible features; the theoretical result that legitimizes gradient ascent on parameterized policies.

- **Actor-Critic Algorithms** — Konda & Tsitsiklis, 2000. Two-timescale analysis of actor-critic methods with convergence guarantees; important for understanding why separate actor and critic learning rates are principled, not just heuristic.

- **Advantage Function** — Schulman et al. (GAE paper §2). The two-page formal definition of the advantage function and its relationship to Q, V, and TD residuals; worth reading in isolation before the full GAE derivation.

## Online Resources

- **David Silver's UCL RL Lecture 7: Policy Gradient Methods** — Silver, DeepMind/UCL. Covers the score function trick, baselines, actor-critic, and natural policy gradient with consistent notation; the clearest single lecture on PG fundamentals. Search "David Silver RL lecture 7 slides".

- **OpenAI Spinning Up: Vanilla Policy Gradient** — OpenAI. Annotated implementation of REINFORCE with a value baseline, with explicit connections between code and the mathematical derivation. Search "OpenAI Spinning Up VPG".

- **Lilian Weng: "Policy Gradient Algorithms"** — Weng, OpenAI, 2018. Comprehensive blog post covering REINFORCE through PPO with consistent notation and implementation notes. Search "lilianweng.github.io policy gradient algorithms".
