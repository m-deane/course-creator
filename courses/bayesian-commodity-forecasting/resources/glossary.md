# Glossary: Bayesian Commodity Forecasting

## Bayesian Concepts

### Bayes' Theorem
The foundational equation of Bayesian inference: P(θ|y) = P(y|θ)P(θ)/P(y). Relates the posterior probability of parameters given data to the likelihood and prior.

### Conjugate Prior
A prior distribution that, when combined with a specific likelihood, yields a posterior in the same distributional family. Enables closed-form solutions.

### Credible Interval
The Bayesian analog to confidence intervals. A 95% credible interval means there is 95% posterior probability that the parameter falls within the interval.

### Evidence (Marginal Likelihood)
P(y) = ∫P(y|θ)P(θ)dθ. The normalizing constant in Bayes' theorem. Used for model comparison.

### Highest Density Interval (HDI)
A credible interval where every point inside has higher posterior density than every point outside. The narrowest possible interval for a given probability.

### Likelihood
P(y|θ). The probability of observing data y given parameters θ. Forms the connection between parameters and data.

### Posterior Distribution
P(θ|y). The probability distribution of parameters after observing data. Represents updated beliefs.

### Posterior Predictive Distribution
P(y_new|y) = ∫P(y_new|θ)P(θ|y)dθ. The distribution of future observations given observed data. Used for forecasting.

### Prior Distribution
P(θ). The probability distribution of parameters before observing data. Encodes domain knowledge.

### Prior Predictive Check
Simulating data from the prior to verify priors are reasonable before fitting to real data.

---

## Time Series Concepts

### Autocorrelation
Correlation of a time series with its own lagged values. ACF(k) measures correlation at lag k.

### ARIMA
AutoRegressive Integrated Moving Average. A class of time series models combining AR, differencing, and MA components.

### Kalman Filter
An algorithm for optimal state estimation in linear-Gaussian state space models. Recursively computes filtered state distributions.

### Mean Reversion
The tendency of a time series to return to a long-term mean. Commodity spreads often exhibit mean reversion.

### Random Walk
A process where each value equals the previous value plus random noise. Yt = Yt-1 + εt.

### Seasonality
Regular, predictable patterns that repeat over fixed periods (weekly, monthly, annually).

### State Space Model
A model with observable outputs (prices) driven by unobservable latent states (trend, volatility).

### Stationarity
A time series property where statistical properties (mean, variance) don't change over time.

### Stochastic Volatility
Models where variance itself is a random process that evolves over time.

---

## Inference Concepts

### Effective Sample Size (ESS)
A measure of the effective number of independent samples after accounting for autocorrelation in MCMC chains.

### Hamiltonian Monte Carlo (HMC)
An MCMC algorithm that uses gradient information to propose efficient moves through parameter space.

### Markov Chain Monte Carlo (MCMC)
A class of algorithms for sampling from probability distributions by constructing Markov chains.

### No-U-Turn Sampler (NUTS)
An extension of HMC that automatically tunes the trajectory length. Default in PyMC and Stan.

### R-hat (Gelman-Rubin Statistic)
A convergence diagnostic comparing between-chain and within-chain variance. Should be < 1.01.

### Trace Plot
A plot of MCMC samples over iterations. Used to diagnose convergence and mixing.

### Variational Inference (VI)
An optimization-based approach to approximate Bayesian inference. Faster but less accurate than MCMC.

---

## Model Evaluation Concepts

### Calibration
The property that probabilistic forecasts match observed frequencies. A 90% interval should contain reality 90% of the time.

### CRPS (Continuous Ranked Probability Score)
A proper scoring rule for probabilistic forecasts that measures both accuracy and calibration.

### Log Score
The log probability of the observed value under the predictive distribution. A proper scoring rule.

### Posterior Predictive Check
Comparing data simulated from the fitted model to observed data to assess model adequacy.

### Proper Scoring Rule
A scoring rule that is optimized when the forecaster reports their true beliefs. CRPS and log score are proper; MSE is not.

### WAIC (Widely Applicable Information Criterion)
A Bayesian model comparison metric that estimates out-of-sample prediction error.

---

## Commodity Market Concepts

### Backwardation
A market condition where futures prices are below spot prices. Indicates supply tightness.

### Carry
The cost of holding a physical commodity (storage, insurance, financing).

### Contango
A market condition where futures prices are above spot prices. Normal when storage costs exceed convenience yield.

### Convenience Yield
The benefit of holding physical commodity rather than a futures contract. High when supply is tight.

### Crack Spread
The price differential between crude oil and refined products (gasoline, heating oil).

### Crush Spread
The price differential between soybeans and soybean products (meal, oil).

### Fundamentals
Supply and demand data: production, consumption, inventories, trade flows.

### Roll Yield
The return (positive or negative) from rolling expiring futures contracts to later months.

### Seasonality
Predictable patterns in commodity prices related to production cycles, weather, or consumption patterns.

### Term Structure
The relationship between futures prices across different expiration dates.

---

## Statistical Distributions

### Beta Distribution
Continuous distribution on [0,1]. Conjugate prior for binomial probability. Beta(α,β).

### Gamma Distribution
Continuous distribution on (0,∞). Conjugate prior for Poisson rate and Normal precision. Gamma(α,β).

### Half-Normal Distribution
A Normal distribution truncated at zero. Common prior for scale parameters.

### Inverse-Gamma Distribution
Conjugate prior for Normal variance. Related to Gamma: if X ~ Gamma(α,β), then 1/X ~ Inv-Gamma(α,β).

### Normal (Gaussian) Distribution
The bell curve. Fundamental to Bayesian inference due to conjugacy and CLT. N(μ,σ²).

### Student-t Distribution
A Normal-like distribution with heavier tails. Robust to outliers. t(ν,μ,σ).

---

## Software Terms

### ArviZ
A Python library for exploratory analysis of Bayesian models. Provides diagnostics, plotting, and model comparison.

### PyMC
A Python library for Bayesian statistical modeling and probabilistic programming.

### NumPyro
A JAX-based probabilistic programming library. Fast due to JIT compilation.

### Stan
A probabilistic programming language and platform for statistical modeling.

### Inference Data (InferenceData)
ArviZ's data structure for storing MCMC results, posterior predictive samples, and model metadata.

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| θ | Parameter(s) of interest |
| y | Observed data |
| α | State vector (state space); shape parameter (distributions) |
| β | Regression coefficients; rate parameter |
| σ | Standard deviation |
| σ² | Variance |
| τ | Precision (1/σ²) |
| μ | Mean |
| ε | Observation noise |
| η | State noise |
| Φ, φ | Autoregressive coefficient |
| ~ | "Distributed as" |
| ∝ | "Proportional to" |
| | | "Given" (conditioning) |
| E[·] | Expectation |
| Var(·) | Variance |
| Cov(·,·) | Covariance |

---

*This glossary covers the primary terminology used throughout the course. Consult specific module guides for deeper explanations.*
