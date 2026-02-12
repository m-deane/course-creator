# Additional Readings: Core Bandit Algorithms

## Foundational Papers

### UCB Algorithm
**Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002)**
"Finite-time Analysis of the Multiarmed Bandit Problem"
*Machine Learning*, 47(2-3), 235-256.
[DOI: 10.1023/A:1013689704352](https://doi.org/10.1023/A:1013689704352)

**Why read this:**
- Original UCB1 paper with complete proofs
- Establishes the O(log T) regret bound
- Foundational for understanding confidence bounds

**Key takeaway:** UCB achieves logarithmic regret by using optimistic upper confidence bounds derived from Hoeffding's inequality.

---

### Epsilon-Greedy and Exploration
**Sutton, R. S., & Barto, A. G. (2018)**
*Reinforcement Learning: An Introduction* (2nd ed.)
Chapter 2: Multi-armed Bandits
[Free online version](http://incompleteideas.net/book/the-book-2nd.html)

**Why read this:**
- Accessible introduction to the bandit problem
- Covers ε-greedy, UCB, gradient bandits
- Excellent intuition and visualizations

**Key takeaway:** The exploration-exploitation tradeoff is fundamental to sequential decision-making, and different algorithms make different tradeoffs.

---

### Softmax/Boltzmann Exploration
**Luce, R. D. (1959)**
*Individual Choice Behavior: A Theoretical Analysis*
Wiley, New York.

**Why read this:**
- Origins of softmax in discrete choice theory
- Connection to Boltzmann distribution from physics
- Foundation for policy gradient methods

**Key takeaway:** Softmax provides a probabilistic framework for action selection that naturally interpolates between exploration and exploitation.

---

## Comprehensive Textbooks

### Bandit Algorithms (Advanced)
**Lattimore, T., & Szepesvári, C. (2020)**
*Bandit Algorithms*
Cambridge University Press.
[Free draft online](https://tor-lattimore.com/downloads/book/book.pdf)

**Coverage:**
- Chapters 1-9: Core algorithms (ε-greedy, UCB, Thompson Sampling)
- Chapter 6: UCB family (UCB1, UCB-V, UCB-Tuned)
- Chapter 15: Regret lower bounds
- Chapter 30+: Advanced topics (contextual, adversarial)

**Difficulty:** Graduate level (requires probability theory background)

**When to read:** After completing this module, for theoretical depth.

---

### Algorithms for Decision Making
**Kochenderfer, M. J., Wheeler, T. A., & Wray, K. H. (2022)**
*Algorithms for Decision Making*
MIT Press.
[Free online version](https://algorithmsbook.com/)

**Coverage:**
- Chapter 8: Multi-armed Bandits
- Focus on practical implementation
- Julia code examples (translates easily to Python)

**Difficulty:** Advanced undergraduate / early graduate

**When to read:** For implementation details and practical considerations.

---

## Survey Papers

### Modern Bandit Algorithms
**Bubeck, S., & Cesa-Bianchi, N. (2012)**
"Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems"
*Foundations and Trends in Machine Learning*, 5(1), 1-122.
[arXiv:1204.5721](https://arxiv.org/abs/1204.5721)

**Why read this:**
- Comprehensive survey of regret bounds
- Covers both stochastic and adversarial bandits
- Excellent for understanding theoretical landscape

**Key sections:**
- Section 2: Stochastic bandits (UCB, ε-greedy)
- Section 3: Regret lower bounds (minimax optimality)

---

### Practical Applications
**Slivkins, A. (2019)**
"Introduction to Multi-Armed Bandits"
*Foundations and Trends in Machine Learning*, 12(1-2), 1-286.
[arXiv:1904.07272](https://arxiv.org/abs/1904.07272)

**Why read this:**
- 280-page tutorial with applications
- Covers contextual bandits, Bayesian methods
- Practical advice for deployment

**Best for:** Practitioners wanting to apply bandits to real problems.

---

## Blog Posts and Tutorials

### Bandit Algorithms for Website Optimization
**White, J. (2012)**
*Bandit Algorithms for Website Optimization*
O'Reilly Media.
[GitHub with code](https://github.com/johnmyleswhite/BanditsBook)

**Why read this:**
- Practical A/B testing applications
- Simple Python implementations
- Focus on web optimization use cases

**Time investment:** 2-3 hours for the full book.

---

### Lil'Log: The Multi-Armed Bandit Problem
**Weng, L. (2018)**
"The Multi-Armed Bandit Problem and Its Solutions"
[Blog post](https://lilianweng.github.io/posts/2018-01-23-multi-armed-bandit/)

**Why read this:**
- Clear visualizations and intuition
- Covers ε-greedy, UCB, Thompson Sampling
- Includes Python code snippets

**Time investment:** 30 minutes

---

### Toward Data Science: UCB Explained
**Kuleshov, V. (2020)**
"Understanding the Upper Confidence Bound Algorithm"
[Medium post](https://towardsdatascience.com/)

**Why read this:**
- Step-by-step derivation of UCB
- Interactive visualizations
- Intuitive explanation of confidence bounds

**Time investment:** 20 minutes

---

## Research Extensions

### UCB Variants

**Audibert, J. Y., Munos, R., & Szepesvári, C. (2009)**
"Exploration-exploitation tradeoff using variance estimates in multi-armed bandits"
*Theoretical Computer Science*, 410(19), 1876-1902.

**Contribution:** UCB-V (variance-aware UCB) for better finite-sample performance.

---

**Garivier, A., & Cappé, O. (2011)**
"The KL-UCB Algorithm for Bounded Stochastic Bandits and Beyond"
*Proceedings of COLT*, 2011.
[arXiv:1102.2490](https://arxiv.org/abs/1102.2490)

**Contribution:** KL-UCB uses Kullback-Leibler divergence for tighter bounds.

---

### Bayesian Approaches

**Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018)**
"A Tutorial on Thompson Sampling"
*Foundations and Trends in Machine Learning*, 11(1), 1-96.
[arXiv:1707.02038](https://arxiv.org/abs/1707.02038)

**Why read this:**
- Thompson Sampling often beats UCB in practice
- Bayesian perspective on exploration
- Preview of Module 3 content

---

## Non-Stationary Bandits

**Garivier, A., & Moulines, E. (2011)**
"On Upper-Confidence Bound Policies for Switching Bandit Problems"
*Algorithmic Learning Theory*, 2011.

**Contribution:** Discounted UCB for environments with regime changes.

---

**Slivkins, A., & Upfal, E. (2008)**
"Adapting to a Changing Environment: The Brownian Restless Bandits"
*Proceedings of COLT*, 2008.

**Contribution:** Algorithms for continuously drifting reward distributions.

---

## Contextual Bandits (Preview)

**Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010)**
"A Contextual-Bandit Approach to Personalized News Article Recommendation"
*Proceedings of WWW*, 2010.
[arXiv:1003.0146](https://arxiv.org/abs/1003.0146)

**Application:** LinUCB algorithm for news recommendation at Yahoo.

**Why read this:** Real-world deployment case study (will be covered in Module 2).

---

## Regret Lower Bounds

**Lai, T. L., & Robbins, H. (1985)**
"Asymptotically Efficient Adaptive Allocation Rules"
*Advances in Applied Mathematics*, 6(1), 4-22.

**Contribution:** Established the fundamental regret lower bound for stochastic bandits.

**Key result:** Any consistent algorithm must have regret Ω(log T / Δ).

---

## Implementation Resources

### Python Libraries

**scikit-learn**: Contextual bandits (MAB not directly supported)
[Documentation](https://scikit-learn.org/)

**Vowpal Wabbit**: Production-scale contextual bandits
[Documentation](https://vowpalwabbit.org/)
[GitHub](https://github.com/VowpalWabbit/vowpal_wabbit)

**striatum**: Python library for bandit algorithms
[GitHub](https://github.com/ntucllab/striatum)

**PyMaBandits**: Research-oriented bandit library
[GitHub](https://github.com/SMPyBandits/SMPyBandits)

---

### Interactive Demos

**Seeing Theory: Bandits**
[Interactive visualization](https://seeing-theory.brown.edu/)

**Bandit Playground**
[Online simulator](https://jmlr.org/papers/v17/14-522.html)

---

## Commodity Trading Applications

### Finance-Specific Papers

**Shen, W., Wang, J., Jiang, Y. G., & Zha, H. (2015)**
"Portfolio Choices with Orthant Ordering of Multivariate Risk Aversion"
*Journal of Financial Economics*, 2015.

**Application:** UCB for dynamic portfolio allocation.

---

**Gasparini, M., & Eisele, J. (2000)**
"A Curve-Free Method for Phase I Clinical Trials"
*Biometrics*, 56(2), 609-615.

**Connection:** Dose-finding in clinical trials uses similar exploration strategies as commodity allocation.

---

## Video Lectures

### Stanford CS234: Reinforcement Learning
**Emma Brunskill (2019)**
Lecture 2: Multi-Armed Bandits
[YouTube](https://www.youtube.com/watch?v=FgzM3zpZ55o)

**Duration:** 1 hour
**Level:** Graduate

---

### DeepMind x UCL: Bandits
**Csaba Szepesvári (2018)**
The Multi-Armed Bandit Problem
[YouTube](https://www.youtube.com/watch?v=eM6IBYVqXEA)

**Duration:** 1.5 hours
**Level:** Advanced graduate

---

## Related Topics

### Reinforcement Learning
If you enjoyed bandits, consider learning full RL:
- **Sutton & Barto (2018):** Chapters 3-13 extend bandits to MDPs
- **Sergey Levine's Deep RL course:** [CS 285](http://rail.eecs.berkeley.edu/deeprlcourse/)

### Online Learning
Bandits are a special case of online learning:
- **Hazan, E. (2016):** "Introduction to Online Convex Optimization"
- **Cesa-Bianchi, N., & Lugosi, G. (2006):** "Prediction, Learning, and Games"

### A/B Testing
Apply bandits to experimentation (covered in Module 4):
- **Kohavi, R., et al. (2020):** *Trustworthy Online Controlled Experiments*
- **VWO Blog:** Practical A/B testing guides

---

## How to Use These Resources

### For Quick Reference (5-10 minutes)
- Blog posts by Lil'Log and Toward Data Science
- Sutton & Barto Chapter 2
- This module's cheatsheet

### For Deep Understanding (2-4 hours)
- Auer et al. (2002) UCB paper
- Sutton & Barto Chapter 2 + exercises
- Lattimore & Szepesvári Chapters 6-8

### For Research / Advanced Work (ongoing)
- Lattimore & Szepesvári full textbook
- Bubeck & Cesa-Bianchi survey
- Recent papers from ICML/NeurIPS/COLT

### For Practical Application (1-2 hours)
- White (2012) Bandit Algorithms book
- VowpalWabbit documentation
- Li et al. (2010) LinUCB case study

---

## Study Plan Recommendation

**Week 1: Core Algorithms**
- ✅ Complete this module (notebooks + exercises)
- Read: Sutton & Barto Chapter 2
- Watch: Emma Brunskill's lecture

**Week 2: Theoretical Depth**
- Read: Auer et al. (2002) UCB paper
- Read: Lattimore & Szepesvári Chapters 6-8
- Implement: UCB variants (UCB-V, KL-UCB)

**Week 3: Extensions**
- Read: Russo et al. (2018) Thompson Sampling tutorial
- Read: Slivkins (2019) survey (Sections 1-5)
- Preview: Contextual bandits (Li et al. 2010)

**Week 4: Applications**
- Read: White (2012) website optimization book
- Build: Your own commodity allocation system
- Deploy: Run live A/B test (Module 4 preview)

---

## Questions for Discussion

After reading, consider:

1. **Theory vs Practice:** Why does Thompson Sampling often beat UCB empirically despite similar theoretical bounds?

2. **Hyperparameter Tuning:** For ε-greedy, how would you choose ε without access to the true reward distributions?

3. **Non-Stationarity:** Design a test to detect when reward distributions have changed (regime shift).

4. **Constraints:** How would you modify UCB for portfolio allocation with budget constraints?

5. **Multiple Objectives:** What if you care about both reward maximization and risk minimization?

---

## Further Questions?

- **Theoretical:** Read Lattimore & Szepesvári or post on [Cross Validated](https://stats.stackexchange.com/)
- **Implementation:** Check [GitHub discussions](https://github.com/) or [Stack Overflow](https://stackoverflow.com/)
- **Applications:** See case studies in Slivkins (2019) or White (2012)

Happy learning!
