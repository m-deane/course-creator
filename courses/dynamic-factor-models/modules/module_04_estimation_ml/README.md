# Module 4: Estimation II - Likelihood Methods

## Overview

This module covers maximum likelihood and Bayesian estimation of dynamic factor models. You'll learn the EM algorithm for ML estimation, identification in the likelihood framework, Bayesian methods with informative priors, and practical MCMC implementation for posterior inference.

**Estimated Time:** 8-10 hours
**Prerequisites:** Module 2 (Kalman filter), Module 3 (PCA estimation)

## Learning Objectives

By completing this module, you will be able to:

1. **Derive** the likelihood function via prediction error decomposition
2. **Implement** the EM algorithm for maximum likelihood estimation
3. **Explain** identification restrictions in the likelihood framework
4. **Specify** Bayesian priors for DFM parameters
5. **Code** Gibbs sampler for posterior inference
6. **Compare** ML and Bayesian estimates empirically
7. **Compute** standard errors and confidence intervals for parameters

## Module Contents

### Guides
1. `guides/01_mle_via_kalman.md` - Prediction error decomposition and likelihood
2. `guides/02_em_algorithm_dfm.md` - Expectation-maximization derivation
3. `guides/03_bayesian_dfm.md` - Priors and posterior inference

### Notebooks
1. `notebooks/01_em_algorithm_implementation.ipynb` - Complete EM for DFM
2. `notebooks/02_bayesian_dfm.ipynb` - MCMC posterior sampling

### Assessments
- `assessments/quiz_module_04.md` - Conceptual quiz on likelihood methods
- `assessments/mini_project_bayesian.md` - Full Bayesian analysis with diagnostics

## Key Concepts

### Likelihood Function

**State-space form:**
- Measurement: $X_t = Z \alpha_t + \varepsilon_t$, $\varepsilon_t \sim N(0, H)$
- Transition: $\alpha_t = T \alpha_{t-1} + R \eta_t$, $\eta_t \sim N(0, Q)$

**Log-likelihood via prediction error decomposition:**
$$\log L(\theta) = -\frac{TN}{2}\log(2\pi) - \frac{1}{2}\sum_{t=1}^T \left[\log|F_t(\theta)| + v_t(\theta)' F_t(\theta)^{-1} v_t(\theta)\right]$$

where:
- $v_t = X_t - Z \hat{\alpha}_{t|t-1}$ (prediction error)
- $F_t = Z P_{t|t-1} Z' + H$ (prediction error variance)

**Parameters:** $\theta = \{Z, T, Q, H\}$ (in reduced form notation)

### EM Algorithm for DFM

**E-Step:** Compute smoothed factor estimates and covariances given current parameters
$$\hat{\alpha}_{t|T}^{(k)} = E[\alpha_t | X_{1:T}, \theta^{(k)}]$$
$$P_{t|T}^{(k)} = \text{Var}[\alpha_t | X_{1:T}, \theta^{(k)}]$$
$$P_{t,t-1|T}^{(k)} = \text{Cov}[\alpha_t, \alpha_{t-1} | X_{1:T}, \theta^{(k)}]$$

Use **Kalman smoother** to compute these quantities.

**M-Step:** Update parameters to maximize expected complete-data log-likelihood

**Update loadings:**
$$Z^{(k+1)} = \left(\sum_{t=1}^T X_t \hat{\alpha}_{t|T}^{(k)'}\right) \left(\sum_{t=1}^T \hat{\alpha}_{t|T}^{(k)} \hat{\alpha}_{t|T}^{(k)'} + P_{t|T}^{(k)}\right)^{-1}$$

**Update transition matrix:**
$$T^{(k+1)} = \left(\sum_{t=2}^T \hat{\alpha}_{t|T}^{(k)} \hat{\alpha}_{t-1|T}^{(k)'} + P_{t,t-1|T}^{(k)}\right) \left(\sum_{t=2}^T \hat{\alpha}_{t-1|T}^{(k)} \hat{\alpha}_{t-1|T}^{(k)'} + P_{t-1|T}^{(k)}\right)^{-1}$$

**Update covariance matrices:**
$$H^{(k+1)} = \text{diag}\left[\frac{1}{T}\sum_{t=1}^T (X_t - Z^{(k+1)}\hat{\alpha}_{t|T}^{(k)})(X_t - Z^{(k+1)}\hat{\alpha}_{t|T}^{(k)})'\right]$$

$$Q^{(k+1)} = \frac{1}{T}\sum_{t=2}^T [\hat{\alpha}_{t|T}^{(k)} \hat{\alpha}_{t|T}^{(k)'} + P_{t|T}^{(k)} - T^{(k+1)}(\hat{\alpha}_{t|T}^{(k)}\hat{\alpha}_{t-1|T}^{(k)'} + P_{t,t-1|T}^{(k)}) - ...]$$

**Iterate** until convergence: $|\log L^{(k+1)} - \log L^{(k)}| < \epsilon$

### Identification in Likelihood Framework

**Problem:** Likelihood invariant to rotation $H$:
$$L(\Lambda, \Phi, \Sigma_\eta, \Sigma_e) = L(\Lambda H, H^{-1}\Phi H, H^{-1}\Sigma_\eta H^{-1'}, \Sigma_e)$$

**Standard restrictions:**
1. Fix $r(r-1)/2$ elements of $\Lambda$ (lower triangular structure)
2. Constrain $\Sigma_\eta = I_r$ (unit variance factors)
3. Order factors by explained variance

**Alternative:** Fix first $r$ rows of $\Lambda$ to be identity matrix $I_r$ (requires careful variable selection).

### Bayesian Estimation

**Prior distributions:**
- Loadings: $\lambda_i \sim N(\mu_\lambda, \Sigma_\lambda)$
- Transition coefficients: $\text{vec}(\Phi) \sim N(\mu_\Phi, \Sigma_\Phi)$
- Error variances: $\sigma_{e_i}^2 \sim IG(a_e, b_e)$ (inverse gamma)
- Factor innovation variance: $\Sigma_\eta \sim IW(\nu_\eta, S_\eta)$ (inverse Wishart)

**Posterior inference via Gibbs sampler:**

1. Draw factors: $F_{1:T} | X, \Lambda, \Phi, \Sigma_\eta, \Sigma_e$ (simulation smoother)
2. Draw loadings: $\Lambda | X, F, \Sigma_e$ (conditional normal)
3. Draw transition: $\Phi | F, \Sigma_\eta$ (conditional normal)
4. Draw error variances: $\sigma_{e_i}^2 | X, F, \Lambda$ (inverse gamma)
5. Draw factor covariance: $\Sigma_\eta | F, \Phi$ (inverse Wishart)

**Repeat** for 10,000+ iterations, discard first 5,000 (burn-in).

### MCMC Diagnostics

**Convergence checks:**
- Trace plots of parameters
- Geweke diagnostic (beginning vs end of chain)
- Heidelberger-Welch stationarity test
- Effective sample size (ESS)

**Multiple chains:**
- Run 3-4 chains with dispersed starting values
- Gelman-Rubin $\hat{R}$ statistic < 1.1 for all parameters

## Mini-Project: EM Algorithm from Scratch

**Objective:** Implement complete EM algorithm for DFM

**Requirements:**
1. E-step using Kalman smoother
2. M-step parameter updates with identification constraints
3. Log-likelihood computation at each iteration
4. Convergence monitoring
5. Standard errors via observed information matrix
6. Comparison with statsmodels DynamicFactor

**Deliverable:** Working Python class with test cases and visualization

**Evaluation Rubric:**
- Correctness: 40%
- Numerical stability: 25%
- Code quality: 20%
- Documentation: 15%

## Bayesian Project: Complete MCMC Analysis

**Objective:** Conduct full Bayesian inference for DFM

**Requirements:**
1. Specify informative priors based on economic theory
2. Implement Gibbs sampler with all conditional distributions
3. Run multiple chains with convergence diagnostics
4. Posterior predictive checks
5. Compare posterior means to ML estimates
6. Credible intervals for impulse responses

**Deliverable:** Jupyter notebook with complete analysis and interpretation

## Connections to Other Modules

| Module | Connection |
|--------|------------|
| Module 2 | Kalman filter computes likelihood |
| Module 3 | PCA provides initialization for EM |
| Module 5 | Likelihood handles mixed frequencies naturally |
| Module 6 | Estimated parameters used for forecasting |
| Module 8 | Bayesian shrinkage for large N |

## Key Formulas

### Observed Information Matrix

Standard errors from:
$$\text{Var}(\hat{\theta}) \approx \left[-\frac{\partial^2 \log L}{\partial \theta \partial \theta'}\bigg|_{\hat{\theta}}\right]^{-1}$$

Computed numerically or via EM algorithm's outer product approximation.

### Bayesian Posterior

$$p(\theta | X) \propto p(X | \theta) p(\theta)$$

**Posterior mean:**
$$\bar{\theta} = \int \theta \, p(\theta | X) d\theta \approx \frac{1}{M}\sum_{m=1}^M \theta^{(m)}$$

**95% credible interval:**
$$[\theta_{0.025}, \theta_{0.975}]$$ (2.5th and 97.5th percentiles of posterior samples)

### Model Comparison

**Akaike Information Criterion:**
$$AIC = -2\log L(\hat{\theta}) + 2k$$

**Bayesian Information Criterion:**
$$BIC = -2\log L(\hat{\theta}) + k \log(T)$$

where $k$ = number of parameters.

**Deviance Information Criterion (Bayesian):**
$$DIC = -2\log p(X | \bar{\theta}) + 2p_D$$
where $p_D$ = effective number of parameters.

## Reading List

### Required
- Hamilton, J.D. (1994). *Time Series Analysis*, Chapter 13.
- Durbin, J. & Koopman, S.J.M. (2012). *Time Series Analysis by State Space Methods*, 2nd ed., Chapters 7-8.

### Recommended
- Doz, C., Giannone, D., & Reichlin, L. (2012). "A Quasi-Maximum Likelihood Approach for Large, Approximate Dynamic Factor Models." *Review of Economics and Statistics* 94(4), 1014-1024.
- Bańbura, M. & Modugno, M. (2014). "Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data." *Journal of Applied Econometrics* 29(1), 133-160.

### Bayesian Methods
- Kim, C.-J. & Nelson, C.R. (1999). *State-Space Models with Regime Switching*, Chapter 6.
- Carter, C.K. & Kohn, R. (1994). "On Gibbs Sampling for State Space Models." *Biometrika* 81(3), 541-553.

## Practical Applications

After completing this module, you can:

1. **Estimate parameters** with proper uncertainty quantification
2. **Handle complex constraints** (cross-equation, sign restrictions)
3. **Incorporate prior information** from economic theory
4. **Test hypotheses** about factor loadings or dynamics
5. **Forecast with prediction intervals** via posterior predictive distribution
6. **Compare models** using information criteria

## ML vs PCA Comparison

| Aspect | PCA | Maximum Likelihood |
|--------|-----|-------------------|
| Speed | Very fast | Slower (iterative) |
| Starting values | Not needed | Requires initialization |
| Standard errors | Bootstrap required | Analytic (information matrix) |
| Missing data | EM-PCA | Natural in likelihood |
| Small sample | Less efficient | More efficient |
| Constraints | Limited | Flexible |

**Rule of thumb:** Use PCA for T > 200, ML for T < 100.

## Computational Considerations

**EM Algorithm:**
- Convergence can be slow (use PCA initialization)
- Each iteration requires Kalman smoother (O(NT))
- Numerical stability critical (use log-likelihood)
- Monitor condition numbers of covariance matrices

**MCMC:**
- Long burn-in needed (5,000-10,000 iterations)
- Posterior draws correlated (thin by keeping every 10th)
- Simulation smoother is bottleneck
- Parallel chains recommended

**Software:**
- statsmodels: `DynamicFactor` (ML via EM)
- PyMC3/PyMC4: Bayesian estimation
- Custom implementation: Full control but more work

## Common Pitfalls

1. **Identification failure:** EM wanders without constraints
2. **Poor initialization:** Can get stuck in local optimum
3. **Overparameterization:** Too many lags or factors
4. **Convergence criteria too loose:** Premature stopping
5. **MCMC mixing issues:** Highly correlated parameters
6. **Improper priors:** Non-integrable posteriors

## Next Steps

After completing this module:
1. Complete EM algorithm mini-project
2. Implement Bayesian estimation project
3. Compare ML and Bayesian estimates on real data
4. Proceed to Module 5: Mixed Frequency Models

---

*"Maximum likelihood provides a unified framework for estimation, testing, and model comparison. Combined with modern computational methods, it unlocks the full potential of factor models for complex data structures." - Durbin & Koopman*
