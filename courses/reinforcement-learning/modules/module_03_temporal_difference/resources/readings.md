# Module 3: Temporal Difference Learning — Recommended Readings

## Essential Reading

- **Reinforcement Learning: An Introduction (2nd ed.), Chapters 6–7** — Sutton & Barto, 2018. Chapter 6 derives TD(0), SARSA, and Q-learning from first principles; Chapter 7 introduces n-step TD and unifies MC and TD as endpoints of a spectrum.

- **Learning to Predict by the Methods of Temporal Differences** — Sutton, 1988. The foundational paper introducing TD learning; the random-walk experiments and convergence arguments remain essential reading for understanding why TD outperforms supervised prediction in sequential settings.

- **Q-Learning** — Watkins & Dayan, 1992. The paper that introduced Q-learning and proved its convergence under mild conditions; a compact four-page read that every RL practitioner should know.

## Supplementary Reading

- **Technical Note: Q-Learning** — Watkins, 1989 (PhD thesis excerpt). The original derivation with the convergence proof in full; more detailed than the 1992 paper and useful when the tabular convergence analysis is needed for assignments.

- **TD Models: Adaptation in Connectionist Reinforcement Learning** — Tesauro, 1994. The TD-Gammon application, which demonstrated that TD learning with neural networks could reach expert-level backgammon play; a compelling existence proof for the approach covered in this module.

- **Double Q-Learning** — van Hasselt, 2010. Identifies the maximization bias inherent in Q-learning and introduces the double estimator fix; understanding this at the tabular level makes the Deep RL module's Double DQN immediately intuitive.

## Online Resources

- **David Silver's UCL RL Lecture 4: Model-Free Prediction and Lecture 5: Model-Free Control** — Silver, DeepMind/UCL. Lectures 4–5 cover TD prediction and SARSA/Q-learning control with side-by-side comparisons against MC and DP that clarify the tradeoffs; search "David Silver RL lecture 4 5 slides".

- **The Cliff Walking Example Explained** — Multiple blog authors. The cliff-walking gridworld from S&B §6.5 is replicated in dozens of blog posts with visualizations; search "SARSA vs Q-learning cliff walking" for interactive implementations.

- **Sutton 1988 original preprint** — Sutton, GTE Laboratories. The preprint version of the 1988 TD paper is archived on Sutton's website; search "Sutton 1988 temporal differences preprint" to access it directly.
