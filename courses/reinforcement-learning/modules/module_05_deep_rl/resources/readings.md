# Module 5: Deep Reinforcement Learning — Recommended Readings

## Essential Reading

- **Human-Level Control Through Deep Reinforcement Learning** — Mnih, Kavukcuoglu, Silver et al., 2015 (Nature). The DQN paper; introduces experience replay and target networks as the two architectural innovations that stabilize deep Q-learning on Atari games.

- **Deep Reinforcement Learning with Double Q-Learning** — van Hasselt, Guez & Silver, 2016. Applies the double estimator to DQN and shows it eliminates overestimation bias with minimal implementation cost; the ablation study is a model of empirical RL research.

- **Dueling Network Architectures for Deep Reinforcement Learning** — Wang, Schaul, Hessel, van Hasselt, Lanctot & de Freitas, 2016. Separates value and advantage streams within the Q-network; demonstrates improved data efficiency especially in states where action choice matters little.

- **Prioritized Experience Replay** — Schaul, Quan, Antonoglou & Silver, 2016. Replaces uniform replay sampling with TD-error-based prioritization; shows substantial sample efficiency gains and introduces the importance-sampling correction needed to keep the update unbiased.

## Supplementary Reading

- **Playing Atari with Deep Reinforcement Learning** — Mnih et al., 2013 (NIPS Deep Learning Workshop). The precursor to the 2015 Nature paper, useful for understanding how the architecture evolved across a year of iteration.

- **Rainbow: Combining Improvements in Deep Reinforcement Learning** — Hessel, Modayil, van Hasselt et al., 2018. Integrates Double DQN, Dueling, PER, n-step returns, distributional RL, and noisy nets into a single agent, showing which combinations contribute most.

- **A Distributional Perspective on Reinforcement Learning** — Bellemare, Dabney & Munos, 2017. Introduces C51, which models the full return distribution rather than its expectation; conceptually important and frequently referenced in advanced DRL discussions.

## Online Resources

- **OpenAI Baselines and Stable-Baselines3 DQN implementations** — OpenAI / DLR-RM group. Production-quality implementations of DQN with all Rainbow components; reading the source code alongside the papers is the fastest path to implementation fluency. Search "stable-baselines3 DQN documentation".

- **Lilian Weng: "From DQN to Rainbow"** — Weng, OpenAI, 2018. A single blog post that traces the DQN paper lineage through Double, Dueling, PER, and Rainbow with consistent notation and clear diagrams. Search "lilianweng.github.io from DQN to Rainbow".

- **Atari 57 Benchmark Leaderboard** — Papers with Code. Tracks state-of-the-art scores per game over time, allowing direct comparison of all papers covered in this module. Search "Papers with Code Atari Games benchmark".
