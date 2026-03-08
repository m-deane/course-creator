# Module 7: Advanced Policy Optimization — Recommended Readings

## Essential Reading

- **Trust Region Policy Optimization** — Schulman, Levine, Moritz, Jordan & Abbeel, 2015. Introduces the trust region constraint that bounds the KL divergence between successive policies; proves monotonic improvement and derives the surrogate objective that underlies both TRPO and PPO.

- **Proximal Policy Optimization Algorithms** — Schulman, Wolski, Dhariwal, Radford & Klimov, 2017. Replaces TRPO's hard KL constraint with a clipped surrogate objective; substantially simpler to implement, scales to large networks, and remains the dominant on-policy algorithm in practice.

- **Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor** — Haarnoja, Zhou, Abbeel & Levine, 2018. Introduces the maximum entropy RL framework and derives SAC, which adds entropy regularization to stabilize off-policy actor-critic training; SAC is the standard baseline for continuous control.

## Supplementary Reading

- **Approximately Optimal Approximate Reinforcement Learning** — Kakade & Langford, 2002. Establishes the theoretical lower bound on policy improvement that TRPO's monotone improvement theorem formalizes; important for understanding why conservative updates are necessary.

- **Soft Actor-Critic Algorithms and Applications** — Haarnoja, Zhou, Hartikainen, Tucker, Ha, Tan, Kumar, Zhu, Gupta, Abbeel & Levine, 2019. The extended SAC paper that introduces an automatic entropy temperature tuning scheme and reports results across locomotion, dexterous manipulation, and legged robot hardware.

- **Emergence of Locomotion Behaviours in Rich Environments** — Heess, TB, Sriram, Lemmon, Merel, Wayne, Tassa, Erez, Wang, Eslami, Riedmiller & Silver, 2017. Large-scale TRPO/PPO training producing emergent locomotion gaits in physics simulation; a compelling demonstration of what constrained policy optimization enables at scale.

## Online Resources

- **OpenAI Spinning Up: TRPO, PPO, and SAC** — OpenAI. Annotated implementations with explicit mapping between equations in the papers and lines of code; the PPO implementation note explaining the clip ratio choice is particularly informative. Search "OpenAI Spinning Up PPO implementation".

- **Towards Delivering a Coherent Self-Contained Explanation of Proximal Policy Optimization** — Huang et al., 2022. A pedagogical rewrite of the PPO paper that resolves notational ambiguities and adds ablation experiments missing from the original; search "PPO implementation details Huang 2022".

- **CleanRL: High-quality Single-file Implementations of RL Algorithms** — Huang et al. Single-file PPO and SAC implementations with detailed documentation; the entire algorithm fits in one readable file, making it ideal for studying how theory maps to practice. Search "CleanRL PPO SAC GitHub".
