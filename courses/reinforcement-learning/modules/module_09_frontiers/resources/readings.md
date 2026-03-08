# Module 9: Frontiers of Reinforcement Learning — Recommended Readings

## Essential Reading

- **Training Language Models to Follow Instructions with Human Feedback** — Ouyang, Wu, Jiang, Almeida, Wainwright, Mishkin, Zhang, Agarwal, Slama, Ray, Schulman, Hilton, Kelton, Miller, Simens, Askell, Welinder, Christiano, Leike & Lowe, 2022. The InstructGPT paper that operationalized RLHF at scale; describes the reward model training and PPO fine-tuning pipeline that underpins ChatGPT and subsequent instruction-following LLMs.

- **Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems** — Levine, Kumar, Tucker & Fu, 2020. A comprehensive survey of offline (batch) RL covering behavior cloning, conservative Q-learning, and model-based offline methods; the standard entry point for the subfield.

- **A Comprehensive Survey of Multi-Agent Reinforcement Learning** — Busoniu, Babuska & De Schutter, 2008. Surveys cooperative, competitive, and mixed-motive MARL settings with a taxonomy of solution concepts (Nash equilibria, correlated equilibria) and a review of convergent tabular MARL algorithms.

- **Reinforcement Learning for Financial Trading: A Survey** — Fischer, 2018. Surveys applications of Q-learning, policy gradient, and actor-critic methods to algorithmic trading, order execution, and portfolio management; the most cited entry-level survey bridging RL theory and financial practice.

## Supplementary Reading

- **Conservative Q-Learning for Offline Reinforcement Learning** — Kumar, Zhou, Tucker & Levine, 2020. Introduces CQL, which penalizes Q-values on out-of-distribution actions to prevent overestimation without access to an environment; the most widely deployed offline RL algorithm.

- **Constitutional AI: Harmlessness from AI Feedback** — Bai, Jones, Ndousse, Askell, Chen, DasSarma, Drain, Fort, Ganguli, Henighan et al., 2022. Extends RLHF with AI-generated preference labels, reducing human annotation cost; relevant to understanding how RLHF scales beyond the InstructGPT regime.

- **Mastering the Game of Go without Human Knowledge** — Silver, Schrittwieser, Simonyan, Antonoglou, Huang, Guez, Hubert, Baker, Lai, Bolton, Chen, Lillicrap, Hui, Sifre, van den Driessche, Graepel & Hassabis, 2017. AlphaGo Zero demonstrates self-play as a scalable alternative to human-generated training signal, a theme central to modern RLHF and MARL research.

## Online Resources

- **Lilian Weng: "Reinforcement Learning from Human Feedback"** — Weng, 2023. A detailed walkthrough of the RLHF pipeline from preference data collection through reward model training to PPO fine-tuning, with annotated diagrams. Search "lilianweng.github.io RLHF reinforcement learning from human feedback".

- **Offline RL Tutorial at NeurIPS 2020** — Levine, Kumar, Nachum & others. Video and slides covering the motivations, algorithms, and benchmarks for offline RL; the most accessible entry point after reading the Levine et al. 2020 survey. Search "NeurIPS 2020 offline RL tutorial Levine".

- **D4RL: Datasets for Deep Data-Driven Reinforcement Learning** — Fu, Kumar, Nachum, Tucker & Levine, 2020. The standard benchmark suite for offline RL algorithms; understanding the dataset collection methodology is essential for fair comparisons across the papers surveyed in this module. Search "D4RL benchmark GitHub".
