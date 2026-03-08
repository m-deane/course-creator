# Module 4: Function Approximation — Recommended Readings

## Essential Reading

- **Reinforcement Learning: An Introduction (2nd ed.), Chapters 9–11** — Sutton & Barto, 2018. Chapter 9 covers on-policy prediction with linear and nonlinear approximators; Chapter 10 extends to control; Chapter 11 addresses the deadly triad and off-policy instability that motivates careful architecture choices.

- **An Analysis of Temporal-Difference Learning with Function Approximation** — Tsitsiklis & Van Roy, 1997. Proves convergence of TD(λ) with linear function approximation and characterizes the fixed-point solution; the paper that established the theoretical foundation for this module's core stability results.

## Supplementary Reading

- **Convergence of Stochastic Iterative Dynamic Programming Algorithms** — Jaakkola, Jordan & Singh, 1994. Provides the stochastic approximation convergence framework used to analyze Q-learning and TD; the Robbins-Monro conditions stated here appear repeatedly in later theoretical treatments.

- **Residual Algorithms: Reinforcement Learning with Function Approximation** — Baird, 1995. Introduces residual gradient methods as a stable alternative to semi-gradient TD; valuable for understanding why naive squared-Bellman-error minimization fails in off-policy settings.

- **Tile Coding Software** — Sutton, 2016. A compact technical note accompanying the freely available tile-coding implementation; the note explains why tile coding's linear, binary feature representation is the preferred basis for tabular-to-approximate transitions before neural networks.

- **Feature Engineering for Reinforcement Learning** — Multiple authors (S&B §9.4–9.5). The sections on coarse coding, tile coding, and radial basis functions in Chapter 9 are worth reading as a standalone unit on hand-crafted representation before deep features.

## Online Resources

- **David Silver's UCL RL Lecture 6: Value Function Approximation** — Silver, DeepMind/UCL. Covers SGD, linear VFA, incremental model fitting, and convergence diagrams; the lecture's comparison table of convergence guarantees across methods is uniquely useful. Search "David Silver RL lecture 6 slides".

- **Tsitsiklis & Van Roy 1997 preprint** — Tsitsiklis, MIT LIDS. The full technical report is available through MIT's Laboratory for Information and Decision Systems; search "Tsitsiklis Van Roy 1997 TD function approximation LIDS".

- **Rich Sutton's tile coding software page** — Sutton, University of Alberta. The canonical open-source tile coding implementation in Python and C, with a short usage guide; search "Richard Sutton tile coding software" to find the download page.
