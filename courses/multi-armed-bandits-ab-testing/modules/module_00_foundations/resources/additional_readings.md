# Additional Readings and Resources

Curated resources for going deeper into the foundations of multi-armed bandits, A/B testing, and decision theory.

## Essential Textbooks

### Multi-Armed Bandits

**1. Bandit Algorithms (2020)**
- **Authors:** Tor Lattimore & Csaba Szepesvári
- **Link:** https://tor-lattimore.com/downloads/book/book.pdf (free PDF)
- **Level:** Advanced (graduate-level mathematics)
- **Why read it:** The definitive modern reference. Rigorous proofs, comprehensive coverage of all major algorithms.
- **Best chapters for Module 0:**
  - Chapter 1: Introduction (motivation and problem setup)
  - Chapter 2: Algorithms and Regret (formal definitions)
  - Chapter 6: The Explore-Exploit Tradeoff

**2. Introduction to Multi-Armed Bandits (2019)**
- **Author:** Aleksandrs Slivkins
- **Link:** https://arxiv.org/abs/1904.07272 (free monograph)
- **Level:** Intermediate (more accessible than Lattimore & Szepesvári)
- **Why read it:** Excellent intuition, clear explanations, connects to related fields.
- **Best sections for Module 0:**
  - Section 1: Introduction and Motivation
  - Section 2: Stochastic Bandits Basics
  - Section 3.1: Explore-Then-Commit

**3. Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems (2012)**
- **Authors:** Sébastien Bubeck & Nicolo Cesa-Bianchi
- **Link:** https://arxiv.org/abs/1204.5721 (free survey)
- **Level:** Advanced
- **Why read it:** Comprehensive survey of theoretical results, connects stochastic and adversarial settings.
- **Best for:** Understanding regret bounds and algorithm analysis

### Decision Theory and Statistical Testing

**4. Testing Statistical Hypotheses (3rd Edition)**
- **Authors:** E.L. Lehmann & Joseph P. Romano
- **Publisher:** Springer
- **Level:** Advanced
- **Why read it:** Classic reference for hypothesis testing, power analysis, sequential testing.
- **Relevant chapters:**
  - Chapter 9: Multiple Testing
  - Chapter 13: Sequential Analysis

**5. Statistical Decision Theory and Bayesian Analysis (2nd Edition)**
- **Author:** James O. Berger
- **Publisher:** Springer
- **Level:** Advanced
- **Why read it:** Foundations of decision theory, loss functions, Bayesian methods.
- **Key concepts:** Utility theory, minimax decisions, Bayes risk

## Foundational Papers

### A/B Testing and Experimentation

**6. "Controlled experiments on the web: survey and practical guide" (2009)**
- **Authors:** Ron Kohavi, Roger Longbotham, Dan Sommerfield, Randal M. Henne
- **Venue:** Data Mining and Knowledge Discovery, Vol 18
- **Link:** https://ai.stanford.edu/~ronnyk/2009controlledExperimentsOnTheWebSurvey.pdf
- **Why read it:** Industry perspective on A/B testing at scale (Microsoft, Google).
- **Key takeaways:** Common pitfalls, implementation challenges, organizational aspects

**7. "Peaking at A/B Tests" (2015)**
- **Authors:** Ramesh Johari, Leo Pekelis, David J. Walsh
- **Link:** https://arxiv.org/abs/1512.04922
- **Why read it:** Quantifies the harm of "peeking" at results before completion.
- **Key result:** Naive peeking inflates Type I error from 5% to ~28%.

### Multi-Armed Bandits: Classic Papers

**8. "Asymptotically efficient adaptive allocation rules" (1985)**
- **Authors:** T.L. Lai & Herbert Robbins
- **Venue:** Advances in Applied Mathematics, Vol 6
- **Why read it:** Proves lower bounds on regret, introduces UCB-like ideas.
- **Key result:** Any algorithm must have regret Ω(log T) in the worst case.

**9. "Finite-time Analysis of the Multiarmed Bandit Problem" (2002)**
- **Authors:** Peter Auer, Nicolo Cesa-Bianchi, Paul Fischer
- **Venue:** Machine Learning, Vol 47
- **Link:** https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
- **Why read it:** Introduces UCB1 algorithm with finite-time regret bounds.
- **Key contribution:** First practical algorithm with provable O(log T) regret.

**10. "Analysis of Thompson Sampling for the Multi-armed Bandit Problem" (2012)**
- **Authors:** Shipra Agrawal & Navin Goyal
- **Venue:** COLT 2012
- **Link:** https://arxiv.org/abs/1111.1797
- **Why read it:** First rigorous analysis of Thompson Sampling (invented in 1933!).
- **Key result:** Thompson Sampling achieves O(log T) regret.

## Blog Posts and Tutorials

### Practical Guides

**11. "When to Run Bandit Tests Instead of A/B/n Tests" (Optimizely)**
- **Link:** https://www.optimizely.com/optimization-glossary/bandit-tests/
- **Level:** Beginner
- **Why read it:** Industry perspective on when to use bandits vs A/B tests.

**12. "Multi-Armed Bandits and the Stitch Fix Experimentation Platform" (2020)**
- **Author:** Michael Frasco (Stitch Fix)
- **Link:** https://multithreaded.stitchfix.com/blog/2020/08/05/bandits/
- **Level:** Intermediate
- **Why read it:** Real-world implementation at a data-driven company.
- **Key insight:** How bandits integrate with existing experimentation infrastructure.

**13. "Bandit Algorithms for Website Optimization" by John Myles White**
- **Book (O'Reilly, 2013)**
- **GitHub:** https://github.com/johnmyleswhite/BanditsBook
- **Level:** Beginner to Intermediate
- **Why read it:** Code examples in Julia and R, practical focus.

**14. "The Multi-Armed Bandit Problem and Its Solutions" (Li Shenggang, 2023)**
- **Link:** https://towardsdatascience.com/multi-armed-bandits-and-its-applications-8b9ae0a0f158
- **Level:** Beginner
- **Why read it:** Clear visual explanations, motivating examples.
- **Note:** This is the article that inspired this course's practical-first approach.

## Commodity Trading Applications

### Trading Under Uncertainty

**15. "Evidence-Based Technical Analysis" (2006)**
- **Author:** David Aronson
- **Publisher:** Wiley
- **Why read it:** Applies statistical rigor to trading strategy evaluation.
- **Key chapters:**
  - Chapter 9: Multiple Testing Bias
  - Chapter 10: Data Mining Bias
- **Connection to bandits:** Trading strategies as arms, hypothesis testing pitfalls.

**16. "Systematic Trading: A Unique New Method for Designing Trading and Investing Systems" (2015)**
- **Author:** Robert Carver
- **Publisher:** Harriman House
- **Why read it:** Portfolio construction under uncertainty, diversification across strategies.
- **Relevant sections:**
  - Chapter 5: Portfolio Optimization (explores exploit tradeoff for strategy allocation)
  - Chapter 9: Speed and Execution (cost of switching strategies)

**17. "Algorithmic and High-Frequency Trading" (2015)**
- **Authors:** Álvaro Cartea, Sebastian Jaimungal, José Penalva
- **Publisher:** Cambridge University Press
- **Why read it:** Mathematical foundations of trading under uncertainty.
- **Connection to bandits:** Optimal execution as a sequential decision problem.

### Portfolio Allocation

**18. "A Modern Introduction to Online Learning" (2019)**
- **Author:** Francesco Orabona
- **Link:** https://arxiv.org/abs/1912.13213 (free)
- **Level:** Advanced
- **Why read it:** Covers online portfolio selection, expert algorithms.
- **Relevant chapters:**
  - Chapter 7: Online Convex Optimization (portfolio allocation as bandit problem)
  - Chapter 11: Bandits

**19. "Online Convex Optimization in the Bandit Setting: Gradient Descent Without a Gradient" (2005)**
- **Authors:** Abraham D. Flaxman, Adam Tauman Kalai, H. Brendan McMahan
- **Venue:** SODA 2005
- **Why read it:** Extends bandits to continuous action spaces (relevant for portfolio weights).

## Video Lectures and Courses

**20. Stanford CS234: Reinforcement Learning (2024)**
- **Instructor:** Emma Brunskill
- **Link:** https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u
- **Relevant lectures:**
  - Lecture 11: Exploration and Exploitation (bandits in RL context)
- **Level:** Intermediate to Advanced

**21. "Exploration-Exploitation in Reinforcement Learning" (DeepMind x UCL)**
- **Instructor:** Hado van Hasselt
- **Link:** https://www.youtube.com/watch?v=eM6IBYVqXEA
- **Level:** Intermediate
- **Duration:** ~1 hour
- **Why watch:** Clear visual explanations of explore-exploit tradeoff.

**22. "Multi-Armed Bandits" (Microsoft Research)**
- **Instructor:** John Langford
- **Link:** https://www.youtube.com/watch?v=N5x48g_wev8
- **Level:** Intermediate
- **Why watch:** From the creator of contextual bandits, practical insights.

## Research Groups and Resources

**23. Bandits Reading Group (Google Research)**
- **Link:** https://sites.google.com/view/bandit-reading-group
- **Why follow:** Curated papers, active research community.

**24. Microsoft Research: Decision Service**
- **Link:** https://ds.microsoft.com/
- **Why follow:** Production system for contextual bandits at scale.
- **Papers:** Practical implementation challenges and solutions.

**25. Tor Lattimore's Blog**
- **Link:** https://banditalgs.com/
- **Why follow:** Insights from the author of the definitive textbook.
- **Topics:** Algorithm intuition, recent research, open problems.

## Software and Tools

**26. Vowpal Wabbit (Contextual Bandits)**
- **Link:** https://vowpalwabbit.org/
- **GitHub:** https://github.com/VowpalWabbit/vowpal_wabbit
- **Why use it:** Production-grade contextual bandit library (Microsoft Research).

**27. PyMC (Bayesian Modeling for Thompson Sampling)**
- **Link:** https://www.pymc.io/
- **Why use it:** Implement Thompson Sampling with full Bayesian inference.

**28. MABWiser (Python Library)**
- **Link:** https://github.com/fmr-llc/mabwiser
- **Why use it:** Production-ready implementations of standard bandit algorithms.

## Connections to Other Fields

### Reinforcement Learning

**29. "Reinforcement Learning: An Introduction" (2nd Edition, 2018)**
- **Authors:** Richard S. Sutton & Andrew G. Barto
- **Link:** http://incompleteideas.net/book/the-book-2nd.html (free PDF)
- **Relevant chapter:** Chapter 2: Multi-armed Bandits
- **Why read it:** Bandits as stateless RL, connects to full RL.

### Information Theory

**30. "Information Theory, Inference, and Learning Algorithms" (2003)**
- **Author:** David J.C. MacKay
- **Link:** http://www.inference.org.uk/mackay/itila/ (free PDF)
- **Relevant chapters:**
  - Chapter 3: More about Inference (Bayesian decision theory)
  - Chapter 37: Bayesian Inference and Sampling Theory

## Historical Context

**31. "On the Problem of the Most Efficient Tests of Statistical Hypotheses" (1933)**
- **Author:** William R. Thompson
- **Venue:** Biometrika, Vol 25
- **Why read it:** Original Thompson Sampling paper (rediscovered decades later!).

**32. "A Problem in Optimum Allocation" (1952)**
- **Author:** Herbert Robbins
- **Why read it:** First formal treatment of the multi-armed bandit problem.

## Summary: Where to Start

**If you're new to bandits:**
1. Read Slivkins' monograph (Introduction to Multi-Armed Bandits)
2. Watch DeepMind x UCL lecture on exploration-exploitation
3. Work through this course's notebooks

**If you want theoretical depth:**
1. Read Lattimore & Szepesvári textbook
2. Study Auer et al. (2002) for UCB
3. Study Agrawal & Goyal (2012) for Thompson Sampling

**If you're applying to trading:**
1. Read Aronson (Evidence-Based Technical Analysis)
2. Read Carver (Systematic Trading)
3. Implement bandits on real price data (Module 0, Notebook 3)

**If you need production implementation:**
1. Study Vowpal Wabbit documentation
2. Read Microsoft Research Decision Service papers
3. Check out Stitch Fix blog post on real-world deployment

---

## Suggested Reading Path for This Course

**For Module 0 (Foundations):**
- Slivkins, Sections 1-2 (Introduction and Basics)
- Kohavi et al. (A/B testing survey)
- Blog post by Li Shenggang (practical motivation)

**Before Module 1 (Epsilon-Greedy):**
- Sutton & Barto, Chapter 2 (RL introduction to bandits)
- Slivkins, Section 3.1 (Explore-Then-Commit)

**Before Module 2 (UCB):**
- Auer et al. (2002) — UCB1 algorithm
- Lattimore & Szepesvári, Chapter 7 (UCB analysis)

**Before Module 3 (Thompson Sampling):**
- Agrawal & Goyal (2012) — Thompson Sampling analysis
- Russo et al. (2018) — Tutorial on information-directed sampling

**Throughout the course:**
- Keep Lattimore & Szepesvári as a reference
- Browse banditalgs.com for intuition
- Check recent papers on arXiv for advanced topics

---

*Last updated: February 2025*
