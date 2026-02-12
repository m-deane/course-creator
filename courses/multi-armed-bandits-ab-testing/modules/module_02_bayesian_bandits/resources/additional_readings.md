# Additional Readings: Bayesian Bandits

## Foundational Papers

### Thompson Sampling Theory

**Thompson, W. R. (1933).** "On the likelihood that one unknown probability exceeds another in view of the evidence of two samples."
- *Biometrika* 25(3/4): 285-294
- **Why read:** The original 1933 paper introducing Thompson Sampling
- **Key insight:** Probability matching as a decision rule
- **Link:** [JSTOR](https://www.jstor.org/stable/2332286)

**Chapelle, O., & Li, L. (2011).** "An empirical evaluation of Thompson Sampling."
- *NIPS 2011*
- **Why read:** First large-scale empirical study showing Thompson Sampling works in practice
- **Key result:** Thompson Sampling matches or beats UCB on real datasets
- **Link:** [PDF](https://papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling.pdf)

**Agrawal, S., & Goyal, N. (2012).** "Analysis of Thompson Sampling for the multi-armed bandit problem."
- *COLT 2012*
- **Why read:** First finite-time regret bound for Thompson Sampling
- **Key result:** Proves O(log T) regret, matching UCB
- **Link:** [arXiv:1111.1797](https://arxiv.org/abs/1111.1797)

**Russo, D., & Van Roy, B. (2014).** "Learning to optimize via information-directed sampling."
- *NIPS 2014*
- **Why read:** Shows Thompson Sampling approximates information-directed sampling (theoretically optimal)
- **Key insight:** Thompson Sampling balances information gain and immediate reward
- **Link:** [arXiv:1403.5556](https://arxiv.org/abs/1403.5556)

**Russo, D., & Van Roy, B. (2018).** "Learning to optimize via Thompson Sampling."
- *Operations Research* 66(5): 1-23
- **Why read:** Comprehensive theoretical analysis and general framework
- **Key contribution:** Thompson Sampling is asymptotically optimal in general environments
- **Link:** [arXiv:1707.02038](https://arxiv.org/abs/1707.02038)

### Bayesian Inference & Conjugate Priors

**Bernardo, J. M., & Smith, A. F. M. (2000).** *Bayesian Theory.*
- **Why read:** Comprehensive reference on Bayesian statistics
- **Relevant chapters:** 5 (Conjugate prior distributions), 8 (Sequential decision problems)
- **Publisher:** Wiley

**Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2013).** *Bayesian Data Analysis* (3rd ed.)
- **Why read:** Modern practical Bayesian statistics
- **Relevant chapters:** 2 (Single-parameter models, Beta-Bernoulli), 3 (Normal models)
- **Publisher:** Chapman & Hall/CRC

**Murphy, K. P. (2012).** *Machine Learning: A Probabilistic Perspective.*
- **Why read:** ML perspective on Bayesian methods
- **Relevant sections:** 5.2 (Conjugate priors), 21.3 (Bandits)
- **Publisher:** MIT Press

## Extensions & Advanced Topics

### Contextual Thompson Sampling

**Agrawal, S., & Goyal, N. (2013).** "Thompson Sampling for contextual bandits with linear payoffs."
- *ICML 2013*
- **Why read:** Extension to contextual bandits (Module 3)
- **Key contribution:** Thompson Sampling for LinUCB-style problems
- **Link:** [arXiv:1209.3352](https://arxiv.org/abs/1209.3352)

**Riquelme, C., Tucker, G., & Snoek, J. (2018).** "Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling."
- *ICLR 2018*
- **Why read:** Neural network approximations for complex contextual bandits
- **Practical value:** When to use deep TS vs linear TS
- **Link:** [arXiv:1802.09127](https://arxiv.org/abs/1802.09127)

### Non-Stationary Bandits

**Garivier, A., & Moulines, E. (2011).** "On upper-confidence bound policies for switching bandit problems."
- *ALT 2011*
- **Why read:** Theory for non-stationary bandits
- **Relevant to:** Commodity markets with regime changes
- **Link:** [arXiv:0805.3415](https://arxiv.org/abs/0805.3415)

**Mellor, J., & Shapiro, J. (2013).** "Thompson Sampling in switching environments with Bayesian online change-point detection."
- *AISTATS 2013*
- **Why read:** Combine Thompson Sampling with change-point detection
- **Commodity application:** Detect regime shifts, reset posteriors
- **Link:** [PDF](http://proceedings.mlr.press/v31/mellor13a.pdf)

### Empirical Applications

**Scott, S. L. (2010).** "A modern Bayesian look at the multi-armed bandit."
- *Applied Stochastic Models in Business and Industry* 26(6): 639-658
- **Why read:** Practitioner perspective on Thompson Sampling at Google
- **Key insight:** Implementation details and practical considerations
- **Link:** [DOI](https://doi.org/10.1002/asmb.874)

**Graepel, T., Candela, J. Q., Borchert, T., & Herbrich, R. (2010).** "Web-scale Bayesian click-through rate prediction for sponsored search advertising in Microsoft's Bing search engine."
- *ICML 2010*
- **Why read:** Large-scale Bayesian bandit deployment
- **Practical lessons:** Scalability, A/B testing migration, production engineering
- **Link:** [PDF](https://www.microsoft.com/en-us/research/publication/web-scale-bayesian-click-through-rate-prediction-for-sponsored-search-advertising-in-microsofts-bing-search-engine/)

## Connection to Bayesian Commodity Forecasting Course

The **Bayesian Commodity Forecasting** course in this repository provides deep background on the Bayesian inference methods underlying Thompson Sampling. Key connections:

### Shared Concepts
- **Posterior updating:** Same Beta-Bernoulli and Normal-Normal conjugacy
- **Conjugate priors:** Gamma-Poisson for count data (inventory, production)
- **Predictive distributions:** Marginalizing over parameter uncertainty
- **Sequential learning:** Yesterday's posterior is today's prior

### Recommended Modules from Bayesian Commodity Forecasting
1. **Module 1: Bayesian Foundations** — Review of Bayes' rule, priors, posteriors
2. **Module 2: Conjugate Priors** — Beta, Normal, Gamma families in depth
3. **Module 4: State-Space Models** — Time-varying parameters (extension to non-stationary bandits)
4. **Module 6: Forecasting Under Uncertainty** — Predictive distributions for decision-making

### How They Complement Each Other
- **Forecasting focus:** Predict future prices/quantities
- **Bandit focus:** Choose actions to maximize reward
- **Integration:** Use commodity forecasts as priors for bandit allocation

**Example workflow:**
1. Use state-space model to forecast commodity returns (Forecasting course)
2. Extract forecast mean and uncertainty
3. Use forecast distribution as informative prior in Thompson Sampling (this course)
4. Thompson Sampling allocates capital based on forecast-informed beliefs

## Tutorials & Blog Posts

**Bayesian Bandits Tutorial by Chris Said**
- Clear visual explanations of Thompson Sampling
- Interactive demos
- **Link:** [https://chris-said.io/2020/01/26/thompson-sampling/](https://chris-said.io/2020/01/26/thompson-sampling/)

**Thompson Sampling for Bernoulli Bandits (Google Research Blog)**
- Practical implementation guide
- A/B testing migration strategies
- **Link:** [Google Research Blog Archive](https://research.google/blog/)

**Bayesian AB Testing with Thompson Sampling (Towards Data Science)**
- Comparison to frequentist A/B testing
- Python implementation
- **Link:** Search "Thompson Sampling Towards Data Science" (multiple good articles)

**PyMC Tutorial: Bayesian Bandits**
- Implement Thompson Sampling with PyMC
- Use MCMC when conjugacy doesn't apply
- **Link:** [PyMC Examples Gallery](https://www.pymc.io/projects/examples/en/latest/)

## Books

**Lattimore, T., & Szepesvári, C. (2020).** *Bandit Algorithms.*
- **Why read:** Comprehensive modern treatment of bandit theory
- **Relevant chapters:** 6 (Bayesian bandits), 14 (Thompson Sampling)
- **Free online:** [https://tor-lattimore.com/downloads/book/book.pdf](https://tor-lattimore.com/downloads/book/book.pdf)

**Russo, D., Van Roy, B., Kazerouni, A., Osband, I., & Wen, Z. (2018).** "A tutorial on Thompson Sampling."
- *Foundations and Trends in Machine Learning* 11(1): 1-96
- **Why read:** Comprehensive tutorial, theory and practice
- **Best for:** Graduate-level understanding
- **Link:** [arXiv:1707.02038](https://arxiv.org/abs/1707.02038)

**Powell, W. B. (2022).** *Reinforcement Learning and Stochastic Optimization: A Unified Framework.*
- **Why read:** Unified view of bandits, RL, and optimization
- **Relevant chapters:** 6 (Multi-armed bandits), 7 (Bayesian learning)
- **Publisher:** Wiley

## Software Libraries

### Python

**scikit-learn (MAB extensions)**
- Not in core sklearn, but community implementations
- **Link:** Search "scikit-learn bandits" on GitHub

**PyMC**
- Probabilistic programming for complex Bayesian bandits
- **Use case:** When conjugacy doesn't apply, use MCMC
- **Link:** [https://www.pymc.io/](https://www.pymc.io/)

**Vowpal Wabbit (VW)**
- Industrial-strength contextual bandit library
- Implements Thompson Sampling and many variants
- **Link:** [https://vowpalwabbit.org/](https://vowpalwabbit.org/)

**MABWiser (FINRA)**
- Python library for multi-armed bandits
- Includes Thompson Sampling, UCB, LinUCB
- **Link:** [https://github.com/fidelity/mabwiser](https://github.com/fidelity/mabwiser)

### R

**contextual**
- R package for contextual bandits
- Comprehensive Thompson Sampling implementations
- **Link:** [https://github.com/Nth-iteration-labs/contextual](https://github.com/Nth-iteration-labs/contextual)

## Industry Applications

**Netflix: Artwork Personalization**
- Contextual Thompson Sampling for image selection
- **Blog post:** "Artwork Personalization at Netflix" (Netflix Tech Blog)

**Google: Content Recommendations**
- Thompson Sampling for news article selection
- **Paper:** Chapelle & Li (2011) - see above

**Meta: Ad Optimization**
- Bayesian bandits for ad auction optimization
- **Resource:** Meta Engineering Blog (search "bandits")

**Spotify: Playlist Recommendations**
- Thompson Sampling for explore-exploit in music recommendations
- **Blog:** Spotify R&D Blog

## Research Groups & Newsletters

**Benjamin Van Roy (Stanford)**
- Leading researcher on Thompson Sampling theory
- **Lab:** [https://web.stanford.edu/~bvr/](https://web.stanford.edu/~bvr/)

**Emma Brunskill (Stanford)**
- Reinforcement learning and bandits
- **Lab:** [https://cs.stanford.edu/people/ebrun/](https://cs.stanford.edu/people/ebrun/)

**Shipra Agrawal (Columbia)**
- Bandit algorithms and online learning
- **Page:** [http://www.columbia.edu/~sa3305/](http://www.columbia.edu/~sa3305/)

**Bandit Algorithms Newsletter**
- Monthly digest of new bandit research
- **Sign up:** Search "bandit algorithms newsletter"

## Datasets for Practice

### Real Commodity Data Sources
**EIA (Energy Information Administration)**
- Oil, gas, coal inventory and price data
- **Link:** [https://www.eia.gov/opendata/](https://www.eia.gov/opendata/)

**USDA (Agriculture)**
- Crop reports, livestock data
- **Link:** [https://www.nass.usda.gov/Data_and_Statistics/](https://www.nass.usda.gov/Data_and_Statistics/)

**LME (Metals)**
- London Metal Exchange data
- **Link:** [https://www.lme.com/](https://www.lme.com/)

**FRED (Macro)**
- Federal Reserve Economic Data
- Includes commodity price indices
- **Link:** [https://fred.stlouisfed.org/](https://fred.stlouisfed.org/)

### Bandit Benchmark Datasets
**OpenML**
- Classification datasets can be framed as contextual bandits
- **Link:** [https://www.openml.org/](https://www.openml.org/)

**UCB1-style Simulations**
- Reproduce experiments from Auer et al. (2002) and Chapelle & Li (2011)
- Available in most bandit libraries

## Next Steps

After mastering this module, continue to:

1. **Module 3: Contextual Bandits** — Extend Thompson Sampling to use features
2. **Bayesian Commodity Forecasting** — Deeper Bayesian methods for price prediction
3. **Russo & Van Roy (2018)** — Theoretical foundations and information-directed sampling
4. **VW Contextual Bandits** — Production-grade implementation

---

**Questions or suggestions?** Open an issue in the course repository or join the discussion forum.
