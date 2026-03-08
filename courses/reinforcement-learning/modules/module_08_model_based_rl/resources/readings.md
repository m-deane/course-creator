# Module 8: Model-Based Reinforcement Learning — Recommended Readings

## Essential Reading

- **Reinforcement Learning: An Introduction (2nd ed.), Chapter 8** — Sutton & Barto, 2018. Introduces Dyna, prioritized sweeping, and trajectory sampling; Chapter 8 is the clearest treatment of how learned models integrate with model-free updates through planning.

- **Dyna, an Integrated Architecture for Learning, Planning, and Reacting** — Sutton, 1991. The original Dyna paper; demonstrates that interleaving real experience with simulated model rollouts accelerates tabular Q-learning, establishing the architecture extended by all subsequent model-based deep RL.

- **World Models** — Ha & Schmidhuber, 2018. Trains a compact world model (VAE + MDN-RNN) entirely in latent space and shows an agent can be trained inside the dream environment; a pivotal paper connecting generative modeling to model-based RL.

- **Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model** — Schrittwieser, Antonoglou, Hubert, Banino, Azar, Grill, Kavukcuoglu & Silver, 2020. Introduces MuZero, which learns a latent dynamics model without a reconstruction loss and achieves superhuman performance by planning with Monte Carlo Tree Search inside the learned model.

## Supplementary Reading

- **Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning** — Nagabandi, Kahn, Fearing & Levine, 2018. An accessible demonstration of learned neural dynamics models for locomotion control with Dyna-style model-free fine-tuning; clear ablations on when the model helps.

- **Dream to Control: Learning Behaviors by Latent Imagination** — Hafner, Lillicrap, Noroutsikis, Pasukonis, Ba & Fischer, 2020. Introduces Dreamer, which back-propagates through differentiable world model rollouts to train the actor; shows that purely imagination-based training matches top model-free algorithms on continuous control.

- **Benchmarking Model-Based Reinforcement Learning** — Wang, Bao, Clavera, Hoang, Wen, Langlois, Zhang, Zhang, Abbeel & Ba, 2019. A systematic empirical comparison of MBRL algorithms that identifies when model bias hurts and how model ensemble size trades off against compute.

## Online Resources

- **David Silver's UCL RL Lecture 8: Integrating Learning and Planning** — Silver, DeepMind/UCL. Covers Dyna, simulation-based search, and MCTS with diagrams that clarify the relationship between model learning and planning; search "David Silver RL lecture 8 slides".

- **Model-Based RL tutorial at ICML 2020** — Levine, Nagabandi & others. A half-day tutorial covering the MBRL landscape from Dyna through latent-space models and MuZero; slides and video available by searching "ICML 2020 model-based RL tutorial Levine".

- **World Models interactive article** — Ha & Schmidhuber, distill.pub style. The authors released an interactive web article with embedded visualizations of the VAE, MDN-RNN, and CMA-ES controller; search "worldmodels.github.io" for the companion site.
