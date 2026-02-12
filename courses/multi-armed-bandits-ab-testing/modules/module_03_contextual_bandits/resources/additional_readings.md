# Additional Readings: Contextual Bandits

## Foundational Papers

### LinUCB and Contextual Bandits

**"A Contextual-Bandit Approach to Personalized News Article Recommendation"**
Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010)
*Proceedings of the 19th International Conference on World Wide Web*

- The seminal LinUCB paper from Yahoo! Research
- Real-world application to news recommendation with 10M+ impressions/day
- Practical considerations: feature engineering, cold-start, computational efficiency
- **Key contribution:** Disjoint and hybrid LinUCB variants with regret bounds
- [Paper link](https://arxiv.org/abs/1003.0146)

**"An Empirical Evaluation of Thompson Sampling"**
Chapelle, O., & Li, L. (2011)
*Advances in Neural Information Processing Systems*

- Compares Thompson Sampling to UCB approaches in contextual setting
- Shows Thompson Sampling often outperforms UCB empirically
- Bayesian linear regression for contextual bandits
- [Paper link](https://papers.nips.cc/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html)

### Theoretical Foundations

**"Regret Bounds for the Adaptive Control of Linear Quadratic Systems"**
Abbasi-Yadkori, Y., & Szepesvári, C. (2011)
*Conference on Learning Theory*

- Theoretical analysis of LinUCB regret bounds
- Proves O(d√T) regret for d-dimensional context
- Self-normalized concentration inequalities
- **For theory enthusiasts:** Deep dive into why LinUCB works

**"Finite-time Analysis of the Multiarmed Bandit Problem"**
Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002)
*Machine Learning*

- The original UCB1 paper (pre-contextual)
- Foundation for understanding LinUCB's exploration strategy
- Optimal regret bounds for standard bandits
- [Paper link](https://link.springer.com/article/10.1023/A:1013689704352)

## Industry Applications

### Technology

**"Personalization at Spotify using Cassowary"**
McInerney, J., et al. (2018)
*RecSys Workshop on Bandit Learning*

- Contextual bandits for music recommendation
- Feature engineering: user taste, listening context, time of day
- Production deployment challenges and solutions
- Scale: billions of impressions
- [Link](https://research.atspotify.com/2018/09/personalization-at-spotify-using-cassowary/)

**"Contextual Bandits for Ranking at LinkedIn"**
Agarwal, D., et al. (2016)
*LinkedIn Engineering Blog*

- Job recommendation using contextual bandits
- Context: user profile, job attributes, economic indicators
- Hybrid approach: combine collaborative filtering with bandits
- [Link](https://engineering.linkedin.com/)

### Finance and Trading

**"Contextual Portfolio Optimization"**
Ban, G.-Y., & Rudin, C. (2019)
*Management Science*

- Application to portfolio allocation with macroeconomic features
- Feature selection for financial contexts
- Comparison with traditional mean-variance optimization
- **Relevance:** Direct application to commodity trading

**"Online Learning in Financial Markets"**
Hazan, E., & Kale, S. (2012)
*Tutorial at ICML*

- Adversarial bandits and portfolio selection
- Dealing with non-stationarity in financial data
- Expert aggregation approaches
- [Tutorial slides](http://www.cs.princeton.edu/~ehazan/)

### E-commerce and Growth

**"Dynamic Pricing with Contextual Bandits"**
Ferreira, K. J., Lee, B. H. A., & Simchi-Levi, D. (2016)
*Management Science*

- Dynamic pricing using contextual information (demand signals, inventory)
- Revenue optimization under uncertainty
- Real retail data experiments
- **Key insight:** 5-15% revenue improvement over fixed pricing

## Feature Engineering for Bandits

**"Feature Engineering for Machine Learning"**
Zheng, A., & Casari, A. (2018)
*O'Reilly Media*

- Chapters 4-6 cover numerical feature scaling, handling temporal features
- Practical recipes for financial time series
- Avoiding leakage in feature construction
- [Book link](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)

**"Advances in Financial Machine Learning"**
López de Prado, M. (2018)
*Wiley*

- Chapter 3: Labeling (defining rewards from financial data)
- Chapter 5: Fractional differentiation (stationary features from prices)
- Chapter 19: Microstructural features (bid-ask, volume)
- **Highly relevant** for commodity feature engineering

## Advanced Topics

### Neural Contextual Bandits

**"Deep Bayesian Bandits Showdown"**
Riquelme, C., Tucker, G., & Snoek, J. (2018)
*International Conference on Learning Representations*

- Neural networks for nonlinear contextual bandits
- Comparison of deep RL approaches (ε-greedy, Thompson, UCB)
- When to use neural vs. linear models
- [Paper link](https://arxiv.org/abs/1802.09127)

**"Neural Contextual Bandits with UCB-based Exploration"**
Zhou, D., Li, L., & Gu, Q. (2020)
*International Conference on Machine Learning*

- NeuralUCB: combines neural networks with UCB exploration
- Theoretical regret analysis
- Outperforms LinUCB when relationships are nonlinear

### Non-Stationary Bandits

**"Taming Non-stationary Bandits: A Bayesian Approach"**
Mellor, J., & Shapiro, J. (2013)
*arXiv preprint*

- Discounted Thompson Sampling for changing environments
- Change-point detection in bandit rewards
- **Relevant** for commodity markets with regime shifts
- [Paper link](https://arxiv.org/abs/1307.4038)

**"Sliding-Window Thompson Sampling for Non-Stationary Settings"**
Trovo, F., Paladino, S., Restelli, M., & Gatti, N. (2016)
*Journal of Artificial Intelligence Research*

- Use sliding windows to adapt to non-stationarity
- Theoretical guarantees for time-varying reward distributions
- Application to online advertising

## Practical Guides and Tutorials

### Online Resources

**"Bandit Algorithms" (Course)**
Lattimore, T., & Szepesvári, C. (2020)
*Cambridge University Press*

- Comprehensive textbook (free online)
- Chapter 19: Contextual bandits
- Chapter 20: LinUCB algorithm and analysis
- [Free PDF](https://tor-lattimore.com/downloads/book/book.pdf)

**"Introduction to Multi-Armed Bandits"**
Slivkins, A. (2019)
*Foundations and Trends in Machine Learning*

- Survey paper covering all major approaches
- Section 4: Contextual bandits
- Includes Python code examples
- [Paper link](https://arxiv.org/abs/1904.07272)

### Code and Implementations

**VowpalWabbit Contextual Bandit Tutorial**
Microsoft Research
- Production-grade contextual bandit implementation
- Used at Microsoft for personalization at scale
- [Tutorial link](https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/python_Contextual_bandits_and_Vowpal_Wabbit.html)

**Bandit Algorithms (Python Library)**
Gimenez, L., & Zou, J.
- Implementation of UCB, Thompson Sampling, LinUCB
- Easy-to-read code for learning
- [GitHub](https://github.com/bgalbraith/bandits)

**PyMC Contextual Bandits Examples**
PyMC Developers
- Bayesian approach to contextual bandits
- Thompson Sampling with Bayesian linear regression
- [Examples](https://www.pymc.io/projects/examples/)

## Domain-Specific Applications

### Commodity Trading

**"Machine Learning for Asset Managers"**
López de Prado, M. (2020)
*Cambridge University Press*

- Chapter 5: Feature importance and selection
- Chapter 6: Portfolio construction under uncertainty
- Online learning for trading strategies

**"Algorithmic and High-Frequency Trading"**
Cartea, Á., Jaimungal, S., & Penalva, J. (2015)
*Cambridge University Press*

- Chapter 9: Statistical arbitrage and online learning
- Regime-switching models
- Feature extraction from order book data

### Personalization and Recommendation

**"Recommender Systems Handbook"**
Ricci, F., Rokach, L., & Shapira, B. (2015)
*Springer*

- Chapter on exploration-exploitation in recommendation
- Cold-start problem solutions
- Context-aware recommendation systems

## Blogs and Industry Posts

**Netflix Tech Blog: "Contextual Bandits"**
- Real-world application to content recommendation
- Feature engineering for user context
- A/B testing vs. bandits comparison
- [Link](https://netflixtechblog.com/)

**Stitch Fix Algorithms Blog**
- Fashion recommendation using contextual bandits
- Handling sparse feedback and delayed rewards
- [Link](https://multithreaded.stitchfix.com/algorithms/)

**Google AI Blog: "Exploration in Recommendation Systems"**
- Balancing exploration and exploitation at scale
- Contextual Thompson Sampling
- Production deployment insights
- [Link](https://ai.googleblog.com/)

## Related Courses and MOOCs

**"Reinforcement Learning" (Coursera)**
David Silver, DeepMind
- Week 2 covers contextual bandits as stateless RL
- Comparison with full RL
- [Course link](https://www.coursera.org/specializations/reinforcement-learning)

**"Machine Learning for Trading" (Udacity)**
Tucker Balch, Georgia Tech
- Module on online learning for trading
- Feature engineering from financial data
- [Course link](https://www.udacity.com/course/machine-learning-for-trading--ud501)

## Research Frontiers (2023-2025)

**Causal Contextual Bandits**
- Using causal inference to improve feature selection
- Identifying which features are causal vs. correlational
- Recent work: "Causal Bandits" (Lattimore et al., 2023)

**Off-Policy Evaluation**
- Evaluating contextual bandit policies from logged data
- Importance sampling and doubly robust estimators
- Key for backtesting bandit systems
- Recent survey: Dudík et al. (2024)

**Fair Contextual Bandits**
- Ensuring fairness in personalization and allocation
- Group fairness constraints in LinUCB
- Applications to hiring, lending, content moderation

## Key Takeaways for Practitioners

1. **Start with LinUCB** — Simple, interpretable, works well for most problems
2. **Feature engineering matters more than algorithm choice** — Good features beat fancy algorithms
3. **Thompson Sampling is a strong alternative** — Often outperforms UCB empirically
4. **Non-stationarity is the norm** — Use discounting or change detection for real-world problems
5. **Off-policy evaluation is critical** — Test bandits offline before deploying

## Recommended Reading Path

**Beginner:**
1. Slivkins survey (2019) — Section 4 on contextual bandits
2. Li et al. (2010) — LinUCB paper
3. VowpalWabbit tutorial — Hands-on implementation

**Intermediate:**
4. Lattimore & Szepesvári book — Chapters 19-20
5. Chapelle & Li (2011) — Thompson Sampling
6. Netflix/Spotify tech blogs — Real-world applications

**Advanced:**
7. Abbasi-Yadkori & Szepesvári (2011) — Theoretical analysis
8. Neural contextual bandits papers (Riquelme et al., Zhou et al.)
9. Non-stationary bandits (Mellor & Shapiro, Trovo et al.)

---

*These resources provide deep dives into contextual bandits from theory to practice. Start with the foundational papers and industry blogs, then explore advanced topics based on your specific application needs.*
