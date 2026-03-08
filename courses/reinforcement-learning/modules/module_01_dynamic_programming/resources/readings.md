# Module 1: Dynamic Programming — Recommended Readings

## Essential Reading

- **Reinforcement Learning: An Introduction (2nd ed.), Chapter 4** — Sutton & Barto, 2018. Covers policy evaluation, policy improvement, policy iteration, and value iteration with worked gridworld examples that directly parallel the module's coding exercises.

- **Dynamic Programming and Optimal Control (Vol. 1), Chapters 1–2** — Bertsekas, 2017. The authoritative mathematical treatment; Chapter 1 formalizes finite-horizon problems and Chapter 2 develops infinite-horizon discounted problems with convergence proofs for value iteration.

## Supplementary Reading

- **Neuro-Dynamic Programming** — Bertsekas & Tsitsiklis, 1996. Chapter 1 provides a rigorous review of exact DP that motivates the approximation methods introduced later in the course; the convergence analysis of policy iteration is particularly thorough.

- **A Survey of Applications of Dynamic Programming to Pattern Recognition Problems** — Bellman & Dreyfus, 1962. A short historical piece that situates the Bellman equation in its original operations-research context, useful for understanding why the principle of optimality is non-trivial.

- **On the Computational Complexity of Stochastic Control** — Papadimitriou & Tsitsiklis, 1987. Establishes why exact DP is intractable for large state spaces, directly motivating approximate and model-free methods introduced in later modules.

## Online Resources

- **David Silver's UCL RL Lecture 2: Markov Decision Processes and Lecture 3: Planning by Dynamic Programming** — Silver, DeepMind/UCL. Lecture 3 is the clearest visual walkthrough of policy iteration and value iteration available; search "David Silver RL lecture 3 slides".

- **OpenAI Gym FrozenLake environment documentation** — OpenAI. The standard benchmark for implementing DP from scratch; the environment's transition and reward matrices make it ideal for verifying policy iteration by hand. Search "OpenAI Gym FrozenLake-v1".

- **Bertsekas DP lecture notes (MIT OCW 6.231)** — Bertsekas, MIT. Free lecture slides that complement the textbook with additional problem sets focused on shortest-path and inventory-control DP formulations.
