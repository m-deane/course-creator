# Glossary: Hidden Markov Models for Time Series Analysis

## A

**Absorbing State**
: A state in a Markov chain that, once entered, cannot be exited (transition probability to itself is 1). Rarely used in financial HMMs but important in some applications.

**ACF (Autocorrelation Function)**
: Measures correlation between a time series and its lagged values. Used in model diagnostics.

**AIC (Akaike Information Criterion)**
: Model selection criterion balancing fit and complexity: AIC = 2k - 2ln(L). Lower is better.

**Alpha (Forward Variable)**
: $\alpha_t(i) = P(o_1,...,o_t, q_t=s_i|\lambda)$. The probability of observing the partial sequence up to time t and being in state i.

**ARMA (Autoregressive Moving Average)**
: Time series model combining AR (autoregressive) and MA (moving average) components. Can be emission distribution in HMMs.

---

## B

**Backwardation**
: Market condition where futures prices decline with maturity. Can characterize a market regime in HMM applications.

**Backward Algorithm**
: Computes $\beta_t(i) = P(o_{t+1},...,o_T|q_t=s_i,\lambda)$, the probability of observing the remaining sequence given state i at time t.

**Baum-Welch Algorithm**
: Expectation-Maximization (EM) algorithm for learning HMM parameters from observed sequences. Iteratively improves parameters until convergence.

**Bear Market**
: Sustained period of declining prices. Often modeled as a distinct regime in financial HMMs.

**Beta (Backward Variable)**
: See Backward Algorithm.

**BIC (Bayesian Information Criterion)**
: Model selection criterion with stronger penalty for complexity: BIC = k ln(n) - 2ln(L). More conservative than AIC.

**Bull Market**
: Sustained period of rising prices. Typically a distinct regime in financial HMMs.

---

## C

**Categorical Distribution**
: Discrete probability distribution over K outcomes. Used for discrete emissions in HMMs.

**Convergence**
: When Baum-Welch algorithm parameters stabilize (log-likelihood improvement below threshold). Typically requires 50-200 iterations.

**Covariance Matrix**
: For multivariate Gaussian emissions, describes variance and correlation structure across features.

**Covariance Type**
: In Gaussian HMMs: "full" (general), "diagonal" (independent features), "spherical" (same variance), "tied" (shared across states).

**Cycle**
: Regular patterns in financial markets (business cycles, credit cycles) that can be modeled as regime transitions.

---

## D

**Decoding Problem**
: Finding the most likely state sequence given observations and model parameters. Solved by Viterbi algorithm.

**Diagonal Covariance**
: Assumption that features are independent within each state, simplifying estimation and reducing parameters.

**Discrete HMM**
: HMM where observations come from a discrete set (categorical emissions). Requires discretization of continuous data.

**Duration**
: Expected time spent in a state before transitioning. Related to self-transition probability: $E[\text{duration}] = 1/(1-a_{ii})$.

---

## E

**EM Algorithm (Expectation-Maximization)**
: General algorithm for maximum likelihood with latent variables. Baum-Welch is the HMM-specific instantiation.

**Emission Distribution (Observation Distribution)**
: $b_i(o) = P(o_t | q_t=s_i)$. The probability of observing o when in state i.

**Ergodic**
: A Markov chain where every state is reachable from every other state. Financial regime models are typically ergodic.

**E-step**
: Expectation step of EM. Computes expected state occupancies and transitions using current parameters.

**Evaluation Problem**
: Computing $P(O|\lambda)$, the likelihood of an observation sequence given model parameters. Solved by Forward algorithm.

---

## F

**Forward Algorithm**
: Efficiently computes $P(O|\lambda)$ using dynamic programming. Complexity O(TK²) vs. O(K^T) for naive approach.

**Forward-Backward Algorithm**
: Combines forward and backward passes to compute state probabilities at all time points: $\gamma_t(i) = P(q_t=s_i|O,\lambda)$.

**Full Covariance**
: No independence assumptions; covariance matrix captures all correlations between features. Most flexible but requires more data.

---

## G

**Gamma (State Probability)**
: $\gamma_t(i) = P(q_t=s_i|O,\lambda)$. Posterior probability of being in state i at time t given all observations.

**GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)**
: Model for time-varying volatility. Can be combined with HMMs for regime-switching volatility.

**Gaussian HMM**
: HMM where emissions are Gaussian distributions. Standard for continuous financial data.

**Gaussian Mixture HMM**
: HMM where each state's emission is a mixture of Gaussians. More flexible for complex distributions.

---

## H

**Hidden State (Latent State)**
: The unobserved variable that generates observations. In finance: market regime, volatility state, economic phase.

**HMM (Hidden Markov Model)**
: A statistical model with hidden states that evolve via Markov chain and generate observable outputs via emission distributions.

**hmmlearn**
: Popular Python library for HMM implementation, supporting Gaussian and discrete emissions.

---

## I

**Initial State Distribution**
: $\pi = [\pi_1, \pi_2, ..., \pi_K]$ where $\pi_i = P(q_1=s_i)$. Probability distribution over states at time t=1.

**Initialization**
: Setting initial parameters before Baum-Welch. Critical for convergence. Methods: k-means, random, domain knowledge.

---

## K

**K (Number of States)**
: A key hyperparameter. Too few misses structure; too many overfits. Typically 2-5 for financial applications.

**Kalman Filter**
: Optimal state estimation for linear-Gaussian state space models. Related to but distinct from HMMs.

**K-Means Initialization**
: Using k-means clustering on observations to set initial emission parameters. Effective for Gaussian HMMs.

---

## L

**Lambda (Model Parameters)**
: $\lambda = (A, B, \pi)$ representing transition matrix, emission distributions, and initial probabilities.

**Latent Variable**
: See Hidden State.

**Learning Problem**
: Finding model parameters $\lambda$ that maximize $P(O|\lambda)$. Solved by Baum-Welch algorithm.

**Left-to-Right HMM**
: Restricted topology where states can only transition to themselves or higher-numbered states. Used for sequential processes, not typical in finance.

**Likelihood**
: $P(O|\lambda)$. Probability of observing the data given the model. Maximized during training.

**Log-Likelihood**
: $\log P(O|\lambda)$. More numerically stable than raw likelihood. Monitored for convergence.

---

## M

**Markov Assumption**
: Future state depends only on current state, not history: $P(q_{t+1}|q_1,...,q_t) = P(q_{t+1}|q_t)$.

**Markov Chain**
: Sequence of states where Markov assumption holds. The hidden state sequence in HMMs.

**M-step**
: Maximization step of EM. Updates parameters to maximize expected log-likelihood computed in E-step.

**Multivariate Gaussian HMM**
: HMM where each emission is a multivariate Gaussian, allowing multiple correlated features.

---

## N

**Numerical Underflow**
: Problem in HMM computation where probabilities become too small for floating-point representation. Solved by log-space computation or scaling.

**Number of States**
: See K.

---

## O

**Observation Sequence**
: $O = [o_1, o_2, ..., o_T]$. The visible data used for training or inference.

**Outlier**
: Extreme values that don't fit emission distributions. Can distort parameter estimates; consider robust methods.

---

## P

**Pomegranate**
: Python library for probabilistic models including HMMs, with flexible emission distributions.

**Posterior Decoding**
: Using $\gamma_t(i)$ to assign most likely state at each time point independently. Differs from Viterbi which finds most likely sequence.

**Prediction**
: Forecasting future observations or states. $P(o_{T+1}|O) = \sum_i P(o_{T+1}|q_{T+1}=s_i)P(q_{T+1}=s_i|O)$.

---

## Q

**Q-Function**
: Expected log-likelihood in EM algorithm. Maximized in M-step.

---

## R

**Regime**
: A persistent market state with characteristic behavior. HMM states typically represent regimes.

**Regime Detection**
: Identifying which regime (state) the market is currently in. Primary application of HMMs in finance.

**Regime Switching**
: Transitions between market regimes. Captured by HMM transition matrix.

**Re-estimation**
: Updating parameters in Baum-Welch. Another term for the M-step.

---

## S

**Scaling**
: Numerical technique to prevent underflow by normalizing probabilities at each time step.

**Self-Transition**
: Probability of remaining in the same state: $a_{ii}$. High values indicate regime persistence.

**Sequence**
: See Observation Sequence.

**Simulation**
: Generating synthetic data from an HMM by sampling states from transition matrix and observations from emissions.

**Spherical Covariance**
: All features have same variance, zero correlation. Most restrictive but simplest Gaussian model.

**State**
: See Hidden State.

**State Occupancy**
: Expected time spent in each state. Used in M-step for parameter updates.

**State Space Model**
: General framework for models with latent states. HMMs are discrete-state instances.

**Stationarity (Distribution)**
: Steady-state distribution of Markov chain: $\pi = \pi A$. Long-run state probabilities.

**Stationarity (Time Series)**
: Statistical properties (mean, variance) constant over time. Often violated in financial data, motivating HMMs.

**Structural Break**
: Abrupt change in time series properties. Can be modeled as regime switch in HMM.

**Supervised Learning**
: Training with labeled states. Rare in HMM applications; typically use unsupervised Baum-Welch.

---

## T

**Tied Covariance**
: All states share the same covariance matrix. Reduces parameters while allowing full correlations.

**Topology**
: Structure of allowed transitions. Ergodic (all transitions allowed) is typical for finance.

**Trace**
: Complete sequence of states visited. Not observed but can be inferred via Viterbi.

**Training**
: See Learning Problem.

**Transition Matrix**
: $A = [a_{ij}]$ where $a_{ij} = P(q_{t+1}=s_j|q_t=s_i)$. Row-stochastic (rows sum to 1).

---

## U

**Underflow**
: See Numerical Underflow.

**Univariate HMM**
: HMM with single-dimensional observations (e.g., just returns). Simplest case.

**Unsupervised Learning**
: Training without labeled states. Standard approach for HMMs using Baum-Welch.

---

## V

**Variance**
: Second moment of distribution. For Gaussian emissions, key parameter alongside mean.

**Viterbi Algorithm**
: Dynamic programming algorithm finding most likely state sequence. Complexity O(TK²).

**Viterbi Path**
: The most likely state sequence found by Viterbi algorithm.

**Volatility Clustering**
: Tendency for high volatility to persist. Motivates regime-switching volatility models.

**Volatility Regime**
: Market state characterized by volatility level (low-vol vs high-vol). Common HMM application.

---

## Xi (Transition Probability)

**Xi (ξ)**
: $\xi_t(i,j) = P(q_t=s_i, q_{t+1}=s_j|O,\lambda)$. Joint probability of transition from state i to j at time t given observations.

---

## Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $K$ | Number of hidden states |
| $T$ | Length of observation sequence |
| $S$ | State space $\{s_1, s_2, ..., s_K\}$ |
| $O$ | Observation sequence $[o_1, o_2, ..., o_T]$ |
| $\lambda$ | Model parameters $(A, B, \pi)$ |
| $A$ | Transition matrix $[a_{ij}]$ |
| $a_{ij}$ | $P(q_{t+1}=s_j \| q_t=s_i)$ |
| $B$ | Emission distributions |
| $b_i(o)$ | $P(o_t \| q_t=s_i)$ |
| $\pi$ | Initial state distribution |
| $\pi_i$ | $P(q_1=s_i)$ |
| $\alpha_t(i)$ | Forward variable |
| $\beta_t(i)$ | Backward variable |
| $\gamma_t(i)$ | $P(q_t=s_i \| O, \lambda)$ |
| $\xi_t(i,j)$ | $P(q_t=s_i, q_{t+1}=s_j \| O, \lambda)$ |
| $\mu_i$ | Mean of Gaussian emission for state i |
| $\Sigma_i$ | Covariance of Gaussian emission for state i |

---

## Common Acronyms

| Acronym | Full Name |
|---------|-----------|
| HMM | Hidden Markov Model |
| EM | Expectation-Maximization |
| ACF | Autocorrelation Function |
| PACF | Partial Autocorrelation Function |
| AIC | Akaike Information Criterion |
| BIC | Bayesian Information Criterion |
| GARCH | Generalized AutoRegressive Conditional Heteroskedasticity |
| ARMA | AutoRegressive Moving Average |
| AR | AutoRegressive |
| MA | Moving Average |
| ML | Maximum Likelihood |
| MAP | Maximum A Posteriori |

---

## HMM Topology Types

| Type | Transition Structure | Use Case |
|------|---------------------|----------|
| Ergodic | All states reachable | Financial regimes (typical) |
| Left-to-Right | $i \to j$ only if $j \geq i$ | Sequential processes, speech |
| Absorbing | Some states cannot exit | Termination modeling |

---

## Covariance Types Comparison

| Type | Parameters per State | Assumptions |
|------|---------------------|-------------|
| Spherical | 1 | Equal variance, independent |
| Diagonal | $d$ | Independent features |
| Tied | $d(d+1)/2$ | Same covariance across states |
| Full | $d(d+1)/2$ | General (most flexible) |

Where $d$ is the number of features.

---

## Model Selection Guidelines

| Number of States | Typical Application |
|------------------|---------------------|
| 2 | Bull/bear; high-vol/low-vol |
| 3 | Bull/neutral/bear; low/medium/high vol |
| 4+ | Multiple market conditions; complex regimes |

**Rule of thumb:** Start with 2-3 states. Add states if:
- BIC continues to decrease
- States have clear economic interpretation
- Sufficient data (rule: 100+ observations per state)

---

## Convergence Criteria

| Method | Typical Value |
|--------|---------------|
| Log-likelihood change | < 0.01 or 0.001 |
| Parameter change | $\|\|\theta_{new} - \theta_{old}\|\|_2 < \epsilon$ |
| Maximum iterations | 100-200 |

---

*This glossary covers HMM terminology for time series and financial applications. See module guides for detailed mathematical derivations.*
