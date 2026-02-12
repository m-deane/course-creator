# Additional Readings: Advanced Topics in Multi-Armed Bandits

Curated research papers, books, and resources for deeper understanding of non-stationary, restless, and adversarial bandits.

## Non-Stationary Bandits

### Foundational Papers

**Garivier, A., & Moulines, E. (2011)**
"On Upper-Confidence Bound Policies for Switching Bandit Problems"
*Algorithmic Learning Theory (ALT)*
[Link](https://arxiv.org/abs/0805.3415)
- Sliding-window UCB and discounted UCB algorithms
- Regret bounds for abruptly changing environments
- Theory for both known and unknown number of changes

**Besbes, O., Gur, Y., & Zeevi, A. (2014)**
"Stochastic Multi-Armed-Bandit Problem with Non-stationary Rewards"
*NeurIPS*
[Link](https://papers.nips.cc/paper/5378-stochastic-multi-armed-bandit-problem-with-non-stationary-rewards)
- General framework for non-stationary rewards
- Total variation budget (bounds on how much rewards can change)
- Minimax optimal algorithms

**Karnin, Z., & Anava, O. (2016)**
"Multi-Armed Bandits with Metric Movement Costs"
*NeurIPS*
- Switching costs between arms (relevant for commodity trading)
- Trade-off between adaptation and transaction costs
- Practical algorithms for cost-aware bandits

### Survey Papers

**Auer, P., Gajane, P., & Ortner, R. (2019)**
"Adaptively Tracking the Best Arm with an Unknown Number of Distribution Changes"
*COLT*
[Link](https://arxiv.org/abs/1902.07346)
- Comprehensive treatment of change-point scenarios
- Algorithms that don't require knowing the number of regime shifts
- Adaptive exploration strategies

**Russac, Y., Cappé, O., & Garivier, A. (2019)**
"Weighted Linear Bandits for Non-Stationary Environments"
*NeurIPS*
- Extends contextual bandits to non-stationary settings
- Weighted ridge regression with discounting
- Applications to recommendation systems

## Change-Point Detection

### CUSUM and Online Detection

**Basseville, M., & Nikiforov, I. V. (1993)**
*Detection of Abrupt Changes: Theory and Application*
Prentice Hall
- Classic reference on CUSUM and other sequential detection methods
- Theoretical foundations and practical implementations
- Applications across multiple domains

**Page, E. S. (1954)**
"Continuous Inspection Schemes"
*Biometrika*
- Original CUSUM paper (seminal work)
- Still highly relevant for sequential monitoring

**Tartakovsky, A., Nikiforov, I., & Basseville, M. (2014)**
*Sequential Analysis: Hypothesis Testing and Changepoint Detection*
Chapman & Hall
- Modern comprehensive treatment
- Bayesian and frequentist approaches
- Multi-stream detection

### Bayesian Change-Point Detection

**Adams, R. P., & MacKay, D. J. (2007)**
"Bayesian Online Changepoint Detection"
*arXiv preprint*
[Link](https://arxiv.org/abs/0710.3742)
- Elegant Bayesian approach to online change detection
- Recursive computation (no need for full history)
- Run-length distributions for regime identification

**Fearnhead, P., & Liu, Z. (2007)**
"On-line Inference for Multiple Changepoint Problems"
*Journal of the Royal Statistical Society*
- Particle filtering for change-point detection
- Handles multiple simultaneous changes
- Applications to time series

## Restless Bandits

### Foundational Theory

**Whittle, P. (1988)**
"Restless Bandits: Activity Allocation in a Changing World"
*Journal of Applied Probability*
- Original restless bandit formulation
- Whittle index policy (heuristic approach)
- Indexability conditions

**Papadimitriou, C. H., & Tsitsiklis, J. N. (1999)**
"The Complexity of Optimal Queueing Network Control"
*Mathematics of Operations Research*
- Proves restless bandits are PSPACE-hard
- Computational intractability results
- Implications for approximation algorithms

### Practical Algorithms

**Liu, K., & Zhao, Q. (2010)**
"Indexability of Restless Bandit Problems and Optimality of Whittle Index for Dynamic Multichannel Access"
*IEEE Transactions on Information Theory*
[Link](https://ieeexplore.ieee.org/document/5447631)
- Conditions for Whittle index optimality
- Applications to wireless networks
- Computational methods

**Meshram, R., Kadota, I., Modiano, E., & Avestimehr, S. (2022)**
"Scheduling for Timely Updates in Multi-Channel Networks with Correlated Sources"
*IEEE Transactions on Communications*
- Modern application of restless bandits
- Handles correlations between arms
- Age-of-information metrics

## Adversarial Bandits

### Core Algorithms

**Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (2002)**
"The Nonstochastic Multiarmed Bandit Problem"
*SIAM Journal on Computing*
[Link](https://epubs.siam.org/doi/10.1137/S0097539701398375)
- EXP3 algorithm (foundational paper)
- Regret bounds: O(√(TK log K))
- Proof of optimality in adversarial setting

**Audibert, J. Y., & Bubeck, S. (2009)**
"Minimax Policies for Adversarial and Stochastic Bandits"
*COLT*
- Algorithms that work in both stochastic and adversarial settings
- Adaptive adversarial algorithms
- Lower bounds

**Bubeck, S., & Cesa-Bianchi, N. (2012)**
"Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems"
*Foundations and Trends in Machine Learning*
[Link](https://arxiv.org/abs/1204.5721)
- Comprehensive survey (100+ pages)
- Covers stochastic, adversarial, and everything in-between
- Essential reading for deep understanding

### Advanced Topics

**Zimmert, J., & Seldin, Y. (2019)**
"An Optimal Algorithm for Stochastic and Adversarial Bandits"
*AISTATS*
- Tsallis-INF algorithm
- Works optimally in both settings without knowing which
- Theoretical breakthrough

**Lykouris, T., Mirrokni, V., & Paes Leme, R. (2018)**
"Stochastic Bandits Robust to Adversarial Corruption"
*STOC*
- Robust bandits (mostly stochastic with some adversarial corruption)
- More realistic than pure adversarial
- Applications to real-world noisy environments

## Applications to Finance and Commodities

### Portfolio Allocation

**Shen, W., Wang, J., Jiang, Y. G., & Zha, H. (2015)**
"Portfolio Choices with Orthant-Based Portfolio Insurance"
*Journal of Financial Economics*
- Online portfolio selection with bandits
- Transaction cost integration
- Practical performance on real markets

**Gao, L., Huang, K. S., & Li, S. (2020)**
"Online Portfolio Selection: A Survey"
*ACM Computing Surveys*
- Comprehensive survey of online portfolio methods
- Bandit-based approaches
- Empirical comparisons

### Regime Detection in Markets

**Ang, A., & Timmermann, A. (2012)**
"Regime Changes and Financial Markets"
*Annual Review of Financial Economics*
- Economic foundations of regime switching
- Empirical evidence in commodity and equity markets
- Statistical methods for regime identification

**Hamilton, J. D. (1989)**
"A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle"
*Econometrica*
- Hidden Markov Models for regime switching
- Foundational paper for regime detection
- Applications to business cycles (relevant for commodity demand)

### High-Frequency Trading

**Cartea, Á., Jaimungal, S., & Penalva, J. (2015)**
*Algorithmic and High-Frequency Trading*
Cambridge University Press
- Market impact and adversarial dynamics
- Optimal execution with adaptive adversaries
- Mathematical treatment of HFT strategies

**Bouchaud, J. P., Farmer, J. D., & Lillo, F. (2009)**
"How Markets Slowly Digest Changes in Supply and Demand"
*Handbook of Financial Markets*
- Market microstructure and impact
- Relevant for understanding when bandits face adversarial rewards
- Empirical evidence from equity and commodity markets

## Online Learning and Optimization

**Hazan, E. (2016)**
"Introduction to Online Convex Optimization"
*Foundations and Trends in Optimization*
[Link](https://arxiv.org/abs/1909.05207)
- General framework including bandits as special case
- Follow-the-Regularized-Leader (FTRL) algorithms
- Connections to game theory

**Cesa-Bianchi, N., & Lugosi, G. (2006)**
*Prediction, Learning, and Games*
Cambridge University Press
- Comprehensive book on online learning
- Covers adversarial and stochastic settings
- Game-theoretic foundations

## Practical Implementation Guides

**Lattimore, T., & Szepesvári, C. (2020)**
*Bandit Algorithms*
Cambridge University Press
[Free online](https://tor-lattimore.com/downloads/book/book.pdf)
- Modern comprehensive textbook
- Includes non-stationary and adversarial chapters
- Practical algorithms with code examples

**Sutton, R. S., & Barto, A. G. (2018)**
*Reinforcement Learning: An Introduction* (2nd Edition)
MIT Press
[Free online](http://incompleteideas.net/book/the-book-2nd.html)
- Contextualizes bandits within RL
- Non-stationary environments chapter
- Practical implementations in Python

## Code Repositories and Tutorials

**Microsoft Research: Contextual Bandits**
[Vowpal Wabbit](https://vowpalwabbit.org/)
- Production-grade bandit library
- Supports non-stationary via warm-start
- Used at Microsoft, Yahoo, etc.

**Facebook Research: ReAgent**
[GitHub: facebookresearch/ReAgent](https://github.com/facebookresearch/ReAgent)
- Applied RL platform including bandits
- Production deployment tools
- Evaluation frameworks

**SMPyBandits: Python Library**
[GitHub: SMPyBandits](https://github.com/SMPyBandits/SMPyBandits)
- Comprehensive Python implementations
- Includes non-stationary algorithms
- Visualization tools

## Domain-Specific Applications

### Commodity Trading

**Geman, H. (2005)**
*Commodities and Commodity Derivatives*
Wiley Finance
- Economic foundations of commodity markets
- Regime characteristics (contango, backwardation)
- Seasonal patterns and structural breaks

**Erb, C. B., & Harvey, C. R. (2006)**
"The Strategic and Tactical Value of Commodity Futures"
*Financial Analysts Journal*
- Empirical evidence of regime changes in commodities
- Risk-return dynamics across market states
- Portfolio allocation strategies

### Energy Markets

**Weron, R. (2014)**
*Electricity Price Forecasting: A Review of the State-of-the-Art with a Look into the Future*
*International Journal of Forecasting*
- Regime switching in energy markets
- Forecasting under non-stationarity
- Relevant for natural gas and electricity allocation

## Mathematical Foundations

**Ross, S. M. (1983)**
*Stochastic Processes* (2nd Edition)
Wiley
- Markov chains and processes (foundations for restless bandits)
- Change detection in stochastic processes
- Renewal theory

**Shiryaev, A. N. (1978)**
*Optimal Stopping Rules*
Springer
- Sequential decision theory
- Change-point detection as optimal stopping
- Foundations for CUSUM and Bayesian methods

## Recent Advances (2020+)

**Filippi, S., Cappe, O., & Garivier, A. (2021)**
"Optimally Tracking a Brownian Target with Costly Discrete Observations"
*Sequential Analysis*
- Modern treatment of restless bandits
- Continuous-state tracking
- Applications to financial markets

**Cheung, W. C., Simchi-Levi, D., & Zhu, R. (2022)**
"Non-Stationary Reinforcement Learning without Prior Knowledge: An Optimal Algorithm for (Un)bounded Dynamic Regret"
*NeurIPS*
- State-of-the-art algorithms for unknown non-stationarity
- No need to tune for regime duration
- Adaptive to changing variation budgets

**Jun, K. S., Bhargava, A., Nowak, R., & Willett, R. (2023)**
"Scalable Generalized Linear Bandits: Online Computation and Hashing"
*NeurIPS*
- Modern contextual bandits at scale
- Relevant for combining contexts with non-stationarity
- Production deployment considerations

## Recommended Reading Order

### For Practitioners
1. Lattimore & Szepesvári (2020) — Chapters 5 (non-stationarity) and 11 (adversarial)
2. Garivier & Moulines (2011) — Sliding-window UCB paper
3. Adams & MacKay (2007) — Bayesian change-point detection
4. Auer et al. (2002) — EXP3 algorithm

### For Researchers
1. Bubeck & Cesa-Bianchi (2012) — Comprehensive survey
2. Besbes et al. (2014) — Non-stationary theory
3. Whittle (1988) — Restless bandits foundations
4. Russac et al. (2019) — Modern non-stationary contextual bandits

### For Commodity Traders
1. Ang & Timmermann (2012) — Regime changes in finance
2. Geman (2005) — Commodity market fundamentals
3. Garivier & Moulines (2011) — Practical non-stationary algorithms
4. Cartea et al. (2015) — High-frequency trading (if relevant)

## Online Courses and Tutorials

**Stanford CS234: Reinforcement Learning**
- Covers bandits and non-stationary environments
- Lecture videos available online

**UCL Course on RL (David Silver)**
- Foundational RL including bandits
- Multi-armed bandit lecture

**Coursera: Practical Reinforcement Learning**
- Hands-on bandit implementations
- Python notebooks

## Academic Conferences to Follow

- **NeurIPS** (Neural Information Processing Systems) — Machine learning track
- **ICML** (International Conference on Machine Learning)
- **COLT** (Conference on Learning Theory) — Theory-focused
- **AISTATS** (Artificial Intelligence and Statistics)
- **KDD** (Knowledge Discovery and Data Mining) — Applications track

## Further Questions?

- **r/MachineLearning** subreddit — Active discussions
- **Cross Validated** (stats.stackexchange.com) — Technical Q&A
- **arXiv cs.LG** — Latest preprints in machine learning
- **JMLR** (Journal of Machine Learning Research) — High-quality publications

---

**Note:** Many papers are freely available on arXiv or authors' websites. Check author homepages for preprints if journal access is paywalled.
