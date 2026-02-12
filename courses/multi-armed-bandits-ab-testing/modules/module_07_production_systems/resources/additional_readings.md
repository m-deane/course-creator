# Additional Readings: Production Bandit Systems

## Industry Case Studies

### Netflix

**Contextual Bandits for Artwork Personalization**
- Blog: "Artwork Personalization at Netflix" (2016)
- URL: https://netflixtechblog.com/artwork-personalization-c589f074ad76
- **Key Insights:**
  - Used contextual bandits to select personalized thumbnails for each user
  - Explored vs exploited different artwork variants
  - Production system handled billions of decisions per day
  - Reduced variance compared to A/B testing by 40%

**Recommendation System Architecture**
- Blog: "System Architectures for Personalization and Recommendation" (2013)
- URL: https://netflixtechblog.com/system-architectures-for-personalization-and-recommendation-e081aa94b5d8
- **Key Insights:**
  - Separation of online serving from offline training
  - Real-time feature computation pipeline
  - Monitoring and A/B testing infrastructure
  - Fallback strategies when models fail

### Microsoft

**Contextual Bandits for News Recommendation**
- Paper: "A Contextual-Bandit Approach to Personalized News Article Recommendation" (Li et al., 2010)
- URL: https://arxiv.org/abs/1003.0146
- **Key Insights:**
  - LinUCB algorithm for article selection
  - Handled 20+ million events per day on MSN.com
  - 12.5% CTR improvement over context-free bandits
  - Production deployment lessons learned

**Decision Service for Contextual Bandits**
- Blog: "The Microsoft Decision Service" (2016)
- URL: https://www.microsoft.com/en-us/research/project/decision-service/
- **Key Insights:**
  - Cloud service for deploying contextual bandits
  - Automatic logging of propensities for offline evaluation
  - Built-in exploration strategies
  - A/B testing integration

### Spotify

**Bandits for Playlist Generation**
- Blog: "How We're Using Contextual Bandits at Spotify" (2020)
- URL: https://engineering.atspotify.com/2020/01/16/how-we-are-using-contextual-bandits-in-recommendations-at-spotify/
- **Key Insights:**
  - Personalized playlist recommendations using bandits
  - Cold start problem for new tracks
  - Non-stationary preferences (user tastes change)
  - Production monitoring and alerting

**Experimentation Platform**
- Blog: "Spotify's New Experimentation Platform" (2020)
- URL: https://engineering.atspotify.com/2020/10/29/spotifys-new-experimentation-platform-part-1/
- **Key Insights:**
  - Unified platform for A/B tests and bandits
  - Feature flagging for gradual rollouts
  - Statistical analysis pipeline
  - Lessons from 1000+ experiments per year

### Facebook/Meta

**Practical Lessons from Predicting Clicks on Ads**
- Paper: "Practical Lessons from Predicting Clicks on Ads at Facebook" (He et al., 2014)
- URL: https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/
- **Key Insights:**
  - Model calibration critical for ad delivery
  - Data freshness matters (retrain frequently)
  - Feature engineering for billions of events
  - Production system architecture

### Google

**Exploration in Personalized Recommendation**
- Paper: "Top-K Off-Policy Correction for a REINFORCE Recommender System" (Chen et al., 2019)
- URL: https://arxiv.org/abs/1812.02353
- **Key Insights:**
  - YouTube recommendation system using bandits
  - Off-policy correction for logged data
  - Balancing exploration and user satisfaction
  - Scalability to billions of users

## Academic Papers

### Foundational Theory

**Finite-time Analysis of UCB**
- Auer, Cesa-Bianchi, Fischer (2002)
- "Finite-time Analysis of the Multiarmed Bandit Problem"
- Machine Learning, 47(2-3), 235-256
- **Why read:** Proves UCB regret bounds, foundational theory

**Thompson Sampling**
- Chapelle, Li (2011)
- "An Empirical Evaluation of Thompson Sampling"
- NIPS 2011
- **Why read:** Empirical comparison showing TS often outperforms UCB

**Contextual Bandits**
- Li, Chu, Langford, Schapire (2010)
- "A Contextual-Bandit Approach to Personalized News Article Recommendation"
- WWW 2010
- **Why read:** LinUCB algorithm, real-world deployment, production system design

### Offline Evaluation

**Doubly Robust Estimation**
- Dudík, Langford, Li (2011)
- "Doubly Robust Policy Evaluation and Learning"
- ICML 2011
- **Why read:** Foundational paper on doubly robust estimators for bandits

**Off-Policy Evaluation**
- Li, Munos, Szepesvári (2015)
- "Toward Minimax Off-policy Value Estimation"
- AISTATS 2015
- **Why read:** Theoretical analysis of variance in offline evaluation

**Counterfactual Reasoning**
- Swaminathan, Joachims (2015)
- "The Self-Normalized Estimator for Counterfactual Learning"
- NIPS 2015
- **Why read:** Lower-variance alternative to standard IPS

### Non-Stationarity

**Sliding-Window UCB**
- Garivier, Moulines (2011)
- "On Upper-Confidence Bound Policies for Switching Bandit Problems"
- ALT 2011
- **Why read:** Theory and algorithms for non-stationary bandits

**Change Detection**
- Liu, Lee, Shroff (2018)
- "Change Detection in Multi-armed Bandits with Non-stationary Rewards"
- AAAI 2018
- **Why read:** Detecting regime changes in bandit problems

### Production Systems

**Bandits at Scale**
- Agarwal et al. (2014)
- "Taming the Monster: A Fast and Simple Algorithm for Contextual Bandits"
- ICML 2014
- **Why read:** Vowpal Wabbit implementation, handles millions of features

**Batch Learning from Logged Data**
- Swaminathan, Joachims (2015)
- "Batch Learning from Logged Bandit Feedback through Counterfactual Risk Minimization"
- JMLR 2015
- **Why read:** Learning from logs without online exploration

## Books

### Bandit Algorithms
- Lattimore, Szepesvári (2020)
- "Bandit Algorithms"
- Cambridge University Press
- **Why read:** Comprehensive theoretical treatment, modern algorithms, regret analysis
- **Best for:** Deep understanding of theory and proofs

### Trustworthy Online Controlled Experiments
- Kohavi, Tang, Xu (2020)
- "Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing"
- Cambridge University Press
- **Why read:** Complements bandits with rigorous A/B testing methodology
- **Best for:** Production experimentation platforms, statistical rigor

### Reinforcement Learning: An Introduction
- Sutton, Barto (2018)
- "Reinforcement Learning: An Introduction" (2nd edition)
- MIT Press
- **Why read:** Chapter 2 covers multi-armed bandits, contextual bandits relate to RL
- **Best for:** Understanding bandits in broader RL context

## Open-Source Libraries

### Vowpal Wabbit
- URL: https://vowpalwabbit.org/
- **Features:**
  - Extremely fast contextual bandit implementation
  - Handles millions of features with hashing
  - Command-line and Python API
  - Used in production at Microsoft, Yahoo, etc.
- **Use for:** Large-scale production deployments

### Ray RLlib
- URL: https://docs.ray.io/en/latest/rllib/index.html
- **Features:**
  - Distributed bandit training
  - Integration with Ray for scalability
  - Supports Thompson Sampling, LinUCB, neural bandits
- **Use for:** Scalable multi-agent systems

### PyMC (for Bayesian Bandits)
- URL: https://www.pymc.io/
- **Features:**
  - Probabilistic programming for Thompson Sampling
  - Full Bayesian inference
  - Visualization tools
- **Use for:** Bayesian bandit implementations, custom priors

### scikit-learn (contextual bandits)
- URL: https://scikit-learn.org/
- **Features:**
  - Partial fit for online learning
  - Standard ML models (logistic regression, SGD)
  - Feature preprocessing
- **Use for:** Custom contextual bandit implementations

## Blog Posts & Tutorials

### Engineering Production Systems

**Airbnb: Experimentation Platform**
- "Building an Intelligent Experimentation Platform with Uber's XP"
- URL: https://medium.com/airbnb-engineering/https-medium-com-jonathan-parks-scaling-erf-23fd17c91166
- **Key Topics:** Platform architecture, metric computation, statistical analysis

**Uber: Experimentation Platform**
- "Building Uber's Experimentation Platform"
- URL: https://eng.uber.com/experimentation-platform/
- **Key Topics:** Distributed architecture, real-time metrics, heterogeneous treatment effects

**Stitch Fix: Multiarmed Bandit Testing**
- "Multi-Armed Bandit Tests"
- URL: https://multithreaded.stitchfix.com/blog/2020/08/05/bandits/
- **Key Topics:** Fashion recommendation, Thompson Sampling, production deployment

### MLOps for Bandits

**The ML Test Score: A Rubric for ML Production Readiness**
- Breck et al. (2017), Google
- URL: https://research.google/pubs/pub46555/
- **Key Topics:** Production ML checklist, monitoring, testing

**Rules of Machine Learning**
- Martin Zinkevich, Google
- URL: https://developers.google.com/machine-learning/guides/rules-of-ml
- **Key Topics:** 43 best practices for production ML, applicable to bandits

## Video Lectures

### Stanford CS234: Reinforcement Learning
- URL: https://web.stanford.edu/class/cs234/
- **Lectures 1-3:** Multi-armed bandits, exploration strategies
- **Instructor:** Emma Brunskill

### DeepMind / UCL Reinforcement Learning Course
- URL: https://www.davidsilver.uk/teaching/
- **Lecture 9:** Exploration and Exploitation
- **Instructor:** David Silver

## Advanced Topics

### Neural Bandits
- Riquelme et al. (2018)
- "Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling"
- ICLR 2018
- **Why read:** Neural networks for contextual bandits, uncertainty quantification

### Combinatorial Bandits
- Chen et al. (2013)
- "Combinatorial Multi-Armed Bandit: General Framework and Applications"
- ICML 2013
- **Why read:** Selecting subsets of arms (e.g., ad allocation, resource allocation)

### Fairness in Bandits
- Joseph et al. (2016)
- "Fairness in Learning: Classic and Contextual Bandits"
- NIPS 2016
- **Why read:** Ensuring fairness constraints in bandit algorithms

### Adversarial Bandits
- Auer et al. (2002)
- "The Nonstochastic Multiarmed Bandit Problem"
- SIAM Journal on Computing, 32(1), 48-77
- **Why read:** Exp3 algorithm for worst-case adversarial rewards

## Production Monitoring & Observability

**Datadog: Monitoring Machine Learning Models**
- URL: https://www.datadoghq.com/blog/monitor-machine-learning-models/
- **Topics:** Metrics to track, alerting strategies, dashboards

**Grafana: ML Observability**
- URL: https://grafana.com/docs/grafana/latest/
- **Topics:** Time-series visualization, custom dashboards, alerts

## Deployment Patterns

**Kubernetes for ML**
- URL: https://kubernetes.io/docs/concepts/
- **Topics:** Containerization, orchestration, auto-scaling

**Airflow for ML Pipelines**
- URL: https://airflow.apache.org/
- **Topics:** Workflow orchestration, scheduling, monitoring

**MLflow for Model Management**
- URL: https://mlflow.org/
- **Topics:** Experiment tracking, model registry, deployment

## Commodity Trading Specific

**Quantitative Trading Strategies**
- Chan, Ernest (2013)
- "Algorithmic Trading: Winning Strategies and Their Rationale"
- Wiley
- **Why read:** Context for commodity trading applications

**Bayesian Methods in Finance**
- Rachev et al. (2008)
- "Bayesian Methods in Finance"
- Wiley
- **Why read:** Bayesian approach to portfolio optimization

## Community Resources

### Reddit
- r/MachineLearning
- r/reinforcementlearning
- r/algotrading (for commodity trading applications)

### Stack Overflow
- Tags: `multi-armed-bandit`, `reinforcement-learning`, `online-learning`

### GitHub
- Search: "contextual bandits", "thompson sampling", "bandit algorithms"
- Trending: https://github.com/trending/python?since=monthly

## Conferences

**Key Venues for Bandit Research:**
- **ICML** (International Conference on Machine Learning)
- **NeurIPS** (Neural Information Processing Systems)
- **AISTATS** (Artificial Intelligence and Statistics)
- **KDD** (Knowledge Discovery and Data Mining) - for applications
- **WWW** (The Web Conference) - for web-scale systems

## Recommended Learning Path

### Beginner (Completed this course!)
1. This course (all modules)
2. Sutton & Barto Chapter 2 (Multi-armed Bandits)
3. Li et al. 2010 (LinUCB paper)
4. Netflix/Spotify blog posts

### Intermediate
1. Lattimore & Szepesvári book (first 5 chapters)
2. Microsoft Decision Service blog
3. Dudík et al. 2011 (Doubly Robust)
4. Implement Vowpal Wabbit integration

### Advanced
1. Lattimore & Szepesvári book (complete)
2. Neural bandits papers (Riquelme et al. 2018)
3. Adversarial bandits (Auer et al. 2002)
4. Contribute to open-source bandit libraries

## Where to Go Next

### Build Your Portfolio Project
Apply bandits to your own domain:
- **E-commerce:** Product recommendation
- **Finance:** Portfolio allocation (like this course)
- **Healthcare:** Treatment assignment
- **Marketing:** Campaign optimization
- **Gaming:** Dynamic difficulty adjustment

### Contribute to Open Source
- Vowpal Wabbit: Add new exploration strategies
- Ray RLlib: Improve bandit implementations
- scikit-learn: Contextual bandit module

### Read Cutting-Edge Research
- Follow recent NeurIPS/ICML papers
- Join ML research reading groups
- Implement and benchmark new algorithms

### Get Production Experience
- Deploy the commodity allocator from this course
- Monitor for 3+ months
- Write a case study blog post
- Share learnings with the community

---

**Remember:** The best way to learn is by doing. Pick one additional reading that excites you, implement it, and share your results!
