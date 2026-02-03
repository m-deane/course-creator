# Mini-Project: Bayesian Dynamic Factor Model Estimation

## Overview

This project implements full Bayesian inference for a dynamic factor model using Markov Chain Monte Carlo (MCMC). You will specify informative priors based on economic theory, code a Gibbs sampler with all conditional distributions, perform convergence diagnostics, and compare Bayesian estimates with maximum likelihood.

**Learning Objectives:**
- Specify economically motivated prior distributions
- Implement Gibbs sampler for posterior inference
- Perform MCMC convergence diagnostics
- Compute credible intervals and posterior predictive checks
- Compare Bayesian and frequentist estimates

**Time Estimate:** 6-8 hours

**Difficulty:** Advanced

---

## Project Specification

### Problem Statement

Estimate a dynamic factor model using Bayesian methods:

**Measurement Equation:**
$$X_t = \Lambda F_t + e_t, \quad e_t \sim N(0, \Sigma_e)$$

**Transition Equation:**
$$F_t = \Phi F_{t-1} + \eta_t, \quad \eta_t \sim N(0, \Sigma_\eta)$$

**Goal:** Obtain posterior distributions for all parameters $\theta = \{\Lambda, \Phi, \Sigma_e, \Sigma_\eta\}$ and latent factors $F_{1:T}$.

---

## Requirements

### Core Requirements (Must Complete)

#### 1. Prior Specification (15 points)

Specify informative priors with economic justification:

```python
class BayesianDFMPriors:
    """
    Prior distributions for Bayesian DFM.
    """

    def __init__(self, N, r, prior_type='informative'):
        """
        Parameters
        ----------
        N : int
            Number of observed variables
        r : int
            Number of factors
        prior_type : str
            'informative', 'weakly_informative', or 'diffuse'
        """
        self.N = N
        self.r = r

        # Specify priors for each parameter
        self.specify_loading_prior()
        self.specify_transition_prior()
        self.specify_error_variance_prior()
        self.specify_factor_variance_prior()

    def specify_loading_prior(self):
        """
        Prior for loadings: λ_i ~ N(μ_λ, Σ_λ)

        Informative: Based on PCA pre-estimates
        Weakly informative: Shrinkage toward zero
        Diffuse: Large variance
        """
        # YOUR CODE HERE
        # Set: self.lambda_mean, self.lambda_cov
        pass

    def specify_transition_prior(self):
        """
        Prior for transition matrix: vec(Φ) ~ N(μ_Φ, Σ_Φ)

        Informative: Persistent but mean-reverting (diagonal elements ~ 0.8)
        Weakly informative: Shrinkage toward identity
        """
        # YOUR CODE HERE
        # Set: self.phi_mean, self.phi_cov
        pass

    def specify_error_variance_prior(self):
        """
        Prior for error variances: σ²_i ~ IG(a_e, b_e)

        Informative: Based on residual variance from PCA
        """
        # YOUR CODE HERE
        # Set: self.sigma_e_shape, self.sigma_e_scale (arrays of length N)
        pass

    def specify_factor_variance_prior(self):
        """
        Prior for factor covariance: Σ_η ~ IW(ν, S)

        Weakly informative: ν = r + 2, S = I_r (unit variance belief)
        """
        # YOUR CODE HERE
        # Set: self.sigma_eta_df, self.sigma_eta_scale
        pass
```

**Prior Justification Document:**

Create a markdown document `prior_justification.md` explaining:
- Economic reasoning for each prior
- How prior parameters were chosen
- Sensitivity analysis plan (which priors matter most?)

**Example Informative Priors:**

For US macroeconomic data with 2 factors:
- **Loadings:** $\lambda_i \sim N(\hat{\lambda}_i^{PCA}, 0.5^2)$ (loosely centered on PCA)
- **Transition:** $\phi_{jj} \sim N(0.7, 0.2^2)$ (persistent but stationary)
- **Error variance:** $\sigma^2_{e_i} \sim IG(3, 2\hat{\sigma}^2_{e_i})$ (mode near PCA residual variance)
- **Factor variance:** $\Sigma_\eta \sim IW(r+3, I_r)$

---

#### 2. Gibbs Sampler Implementation (35 points)

Implement complete Gibbs sampler with all conditional distributions:

```python
class GibbsSamplerDFM:
    """
    Gibbs sampler for Bayesian DFM estimation.
    """

    def __init__(self, priors):
        """
        Parameters
        ----------
        priors : BayesianDFMPriors
            Prior specification object
        """
        self.priors = priors

    def sample_factors(self, X, Lambda, Phi, Sigma_e, Sigma_eta):
        """
        Draw factors from p(F_{1:T} | X, Λ, Φ, Σ_e, Σ_η).

        Uses simulation smoother (Carter-Kohn algorithm):
        1. Run Kalman filter forward
        2. Simulate backward from p(F_T | X_{1:T})

        Parameters
        ----------
        X : ndarray, shape (T, N)
        Lambda : ndarray, shape (N, r)
        Phi : ndarray, shape (r, r)
        Sigma_e : ndarray, shape (N,) - diagonal elements
        Sigma_eta : ndarray, shape (r, r)

        Returns
        -------
        F : ndarray, shape (T, r)
            Simulated factor path
        """
        # YOUR CODE HERE
        # Hint: Use Kalman filter then backward simulation
        pass

    def sample_loadings(self, X, F, Sigma_e):
        """
        Draw loadings from p(Λ | X, F, Σ_e).

        Conditional distribution (variable-by-variable):
        λ_i | ... ~ N(μ_i, V_i)

        where:
        V_i = (F'F / σ²_i + Σ_λ^{-1})^{-1}
        μ_i = V_i (F'X_i / σ²_i + Σ_λ^{-1} μ_λ)

        Parameters
        ----------
        X : ndarray, shape (T, N)
        F : ndarray, shape (T, r)
        Sigma_e : ndarray, shape (N,)

        Returns
        -------
        Lambda : ndarray, shape (N, r)
        """
        # YOUR CODE HERE
        pass

    def sample_transition(self, F, Sigma_eta):
        """
        Draw transition matrix from p(Φ | F, Σ_η).

        Conditional distribution:
        vec(Φ) | ... ~ N(μ_Φ, V_Φ)

        where:
        V_Φ = (Σ_η^{-1} ⊗ Σ_{F,t-1})^{-1} + Σ_Φ)^{-1}
        μ_Φ = V_Φ [(Σ_η^{-1} ⊗ I) vec(Σ_{F_t, F_{t-1}}) + Σ_Φ^{-1} μ_Φ]

        Parameters
        ----------
        F : ndarray, shape (T, r)
        Sigma_eta : ndarray, shape (r, r)

        Returns
        -------
        Phi : ndarray, shape (r, r)
        """
        # YOUR CODE HERE
        # Constraint: Ensure stationarity (eigenvalues < 1)
        pass

    def sample_error_variances(self, X, F, Lambda):
        """
        Draw error variances from p(σ²_i | X, F, Λ).

        Conditional distribution:
        σ²_i | ... ~ IG(a_i^*, b_i^*)

        where:
        a_i^* = a_e + T/2
        b_i^* = b_e + (1/2) Σ_t (X_{it} - λ_i' F_t)²

        Parameters
        ----------
        X : ndarray, shape (T, N)
        F : ndarray, shape (T, r)
        Lambda : ndarray, shape (N, r)

        Returns
        -------
        Sigma_e : ndarray, shape (N,)
        """
        # YOUR CODE HERE
        pass

    def sample_factor_covariance(self, F, Phi):
        """
        Draw factor covariance from p(Σ_η | F, Φ).

        Conditional distribution:
        Σ_η | ... ~ IW(ν^*, S^*)

        where:
        ν^* = ν + T
        S^* = S + Σ_t (F_t - Φ F_{t-1})(F_t - Φ F_{t-1})'

        Parameters
        ----------
        F : ndarray, shape (T, r)
        Phi : ndarray, shape (r, r)

        Returns
        -------
        Sigma_eta : ndarray, shape (r, r)
        """
        # YOUR CODE HERE
        pass

    def run_gibbs(self, X, n_iter=10000, burn_in=5000, thin=5, verbose=True):
        """
        Run Gibbs sampler.

        Parameters
        ----------
        X : ndarray, shape (T, N)
            Observations
        n_iter : int
            Total iterations
        burn_in : int
            Burn-in iterations to discard
        thin : int
            Keep every thin-th draw (reduces autocorrelation)
        verbose : bool
            Print progress

        Returns
        -------
        posterior_samples : dict
            - 'Lambda': ndarray (n_kept, N, r)
            - 'Phi': ndarray (n_kept, r, r)
            - 'Sigma_e': ndarray (n_kept, N)
            - 'Sigma_eta': ndarray (n_kept, r, r)
            - 'F': ndarray (n_kept, T, r)
        diagnostics : dict
            - 'acceptance_rates': dict
            - 'iteration_time': float
        """
        T, N = X.shape
        r = self.priors.r

        # Initialize with PCA estimates
        Lambda, Phi, Sigma_e, Sigma_eta, F = self._initialize(X)

        # Storage for posterior samples
        n_kept = (n_iter - burn_in) // thin
        samples = {
            'Lambda': np.zeros((n_kept, N, r)),
            'Phi': np.zeros((n_kept, r, r)),
            'Sigma_e': np.zeros((n_kept, N)),
            'Sigma_eta': np.zeros((n_kept, r, r)),
            'F': np.zeros((n_kept, T, r))
        }

        # Gibbs iteration
        kept_idx = 0
        for iter in range(n_iter):
            # Draw from conditional distributions
            F = self.sample_factors(X, Lambda, Phi, Sigma_e, Sigma_eta)
            Lambda = self.sample_loadings(X, F, Sigma_e)
            Phi = self.sample_transition(F, Sigma_eta)
            Sigma_e = self.sample_error_variances(X, F, Lambda)
            Sigma_eta = self.sample_factor_covariance(F, Phi)

            # Store samples after burn-in
            if iter >= burn_in and (iter - burn_in) % thin == 0:
                samples['Lambda'][kept_idx] = Lambda
                samples['Phi'][kept_idx] = Phi
                samples['Sigma_e'][kept_idx] = Sigma_e
                samples['Sigma_eta'][kept_idx] = Sigma_eta
                samples['F'][kept_idx] = F
                kept_idx += 1

            if verbose and (iter + 1) % 1000 == 0:
                print(f"Iteration {iter + 1}/{n_iter}")

        return samples, diagnostics
```

**Implementation Checklist:**
- [ ] Carter-Kohn simulation smoother for factors
- [ ] Multivariate normal sampling for loadings and transition
- [ ] Inverse gamma sampling for error variances
- [ ] Inverse Wishart sampling for factor covariance
- [ ] Stationarity constraint on $\Phi$
- [ ] Identification constraint (e.g., $\Lambda[0:r, 0:r]$ lower triangular)

---

#### 3. MCMC Diagnostics (20 points)

Implement comprehensive convergence diagnostics:

```python
class MCMCDiagnostics:
    """
    Convergence diagnostics for MCMC output.
    """

    @staticmethod
    def trace_plots(posterior_samples, params=['Lambda', 'Phi']):
        """
        Plot parameter traces over iterations.

        Check for:
        - Stationarity (no trends)
        - Mixing (rapid movement)
        - No obvious patterns
        """
        pass

    @staticmethod
    def geweke_diagnostic(samples, first=0.1, last=0.5):
        """
        Geweke convergence diagnostic.

        Compares means of first 10% and last 50% of chain.
        Test statistic ~ N(0,1) under convergence.

        Parameters
        ----------
        samples : ndarray, shape (n_iter, ...)
        first, last : float
            Proportions of chain to compare

        Returns
        -------
        z_scores : dict
            Z-score for each parameter (|z| < 2 indicates convergence)
        """
        pass

    @staticmethod
    def effective_sample_size(samples):
        """
        Compute effective sample size (ESS).

        ESS = n / (1 + 2 Σ_k ρ_k)

        where ρ_k is autocorrelation at lag k.

        Returns
        -------
        ess : dict
            Effective sample size for each parameter
        """
        pass

    @staticmethod
    def gelman_rubin(chains):
        """
        Gelman-Rubin diagnostic (requires multiple chains).

        R̂ = √[(W + B/n) / W]

        where:
        - W = within-chain variance
        - B = between-chain variance

        Parameters
        ----------
        chains : list of ndarrays
            Multiple MCMC chains

        Returns
        -------
        r_hat : dict
            R̂ for each parameter (< 1.1 indicates convergence)
        """
        pass

    @staticmethod
    def autocorrelation_plot(samples, max_lag=50):
        """
        Plot autocorrelation function of MCMC samples.

        High autocorrelation indicates slow mixing.
        """
        pass
```

**Diagnostic Report:**

Generate `mcmc_diagnostics_report.md` including:
- Trace plots for 5-10 key parameters
- Geweke z-scores (table)
- ESS for all parameters
- Gelman-Rubin R-hat (if multiple chains)
- Autocorrelation plots
- Interpretation: Did sampler converge? Is thinning sufficient?

---

#### 4. Posterior Analysis (20 points)

Analyze posterior distributions:

```python
def posterior_summary(posterior_samples):
    """
    Compute posterior statistics.

    Returns
    -------
    summary : dict
        For each parameter:
        - 'mean': Posterior mean (point estimate)
        - 'median': Posterior median
        - 'std': Posterior standard deviation
        - 'ci_95': 95% credible interval [2.5%, 97.5%]
        - 'ci_68': 68% credible interval [16%, 84%]
    """
    pass

def posterior_predictive_check(X, posterior_samples, n_rep=100):
    """
    Posterior predictive check.

    Generate replicated datasets:
    X_rep ~ p(X_rep | θ) for θ ~ p(θ | X)

    Compare summary statistics of X_rep to observed X.

    Checks:
    - Mean and variance match
    - Correlation structure preserved
    - No systematic discrepancies
    """
    pass

def impulse_response_function(posterior_samples, horizon=24):
    """
    Compute impulse responses with credible bands.

    IRF(h) = Λ Φ^h

    Parameters
    ----------
    horizon : int
        Forecast horizon

    Returns
    -------
    irf_mean : ndarray, shape (horizon, N)
    irf_lower : ndarray, shape (horizon, N)
    irf_upper : ndarray, shape (horizon, N)
    """
    pass
```

**Deliverable:**
- Posterior summary table (LaTeX or Markdown)
- Posterior density plots for key parameters
- Posterior predictive check visualizations
- Impulse response functions with 68% and 95% credible bands

---

#### 5. Comparison with ML Estimation (10 points)

Compare Bayesian and maximum likelihood estimates:

```python
def compare_bayesian_ml(X, posterior_samples):
    """
    Compare Bayesian (posterior mean) to ML estimates.

    Steps:
    1. Compute posterior mean of parameters
    2. Run MLE using statsmodels DynamicFactor or EM algorithm
    3. Compare estimates
    4. Analyze differences

    Returns
    -------
    comparison_table : DataFrame
        Columns: Parameter, Bayesian, ML, Difference, % Difference
    """
    pass
```

**Analysis Questions:**
1. Which parameters differ most between Bayesian and ML?
2. How do priors affect posterior estimates (prior sensitivity)?
3. Are Bayesian credible intervals wider than ML confidence intervals?
4. Do both methods identify the same factor structure?

---

### Extension Options (Choose 1, 10 points)

#### Option A: Multiple Chains

Run 3-4 chains with dispersed starting values and compute Gelman-Rubin diagnostic:

```python
def run_multiple_chains(X, n_chains=4, **gibbs_kwargs):
    """
    Run multiple MCMC chains in parallel.

    Starting values initialized at:
    - PCA estimates
    - PCA ± 2 standard errors
    - Random draws from prior

    Check: R̂ < 1.1 for all parameters
    """
    pass
```

---

#### Option B: Time-Varying Volatility

Extend model to stochastic volatility:

$$X_{it} = \lambda_i' F_t + e^{h_{it}/2} \varepsilon_{it}$$
$$h_{it} = \mu_i + \phi_i (h_{i,t-1} - \mu_i) + \sigma_i \zeta_{it}$$

Implement additional Gibbs step for log-volatilities.

---

#### Option C: Bayesian Model Comparison

Compare models with different numbers of factors using DIC or marginal likelihood:

```python
def compute_dic(X, posterior_samples):
    """
    Deviance Information Criterion.

    DIC = -2 log p(X | θ̄) + 2 p_D

    where p_D = effective number of parameters.
    """
    pass
```

Test $r = 1, 2, 3, 4$ factors.

---

## Evaluation Rubric

### Functionality (35 points)

| Criterion | Excellent (32-35) | Good (27-31) | Adequate (21-26) | Needs Work (0-20) |
|-----------|-------------------|--------------|------------------|-------------------|
| Prior specification | Well-justified, informative priors | Reasonable priors | Generic priors | Improper or unreasonable priors |
| Gibbs sampler | All conditional distributions correct | Most correct, minor issues | Some incorrect | Major errors |
| Convergence | Proper diagnostics, clear convergence | Most diagnostics, converges | Limited diagnostics | No convergence evidence |
| Posterior analysis | Comprehensive statistics and plots | Good coverage | Basic analysis | Minimal analysis |

**Specific Checks:**
- [ ] Simulation smoother correct (10 pts)
- [ ] All conditional distributions implemented (15 pts)
- [ ] MCMC converges (5 pts)
- [ ] Posterior summaries correct (5 pts)

---

### Statistical Correctness (25 points)

| Criterion | Excellent (23-25) | Good (19-22) | Adequate (15-18) | Needs Work (0-14) |
|-----------|-------------------|--------------|------------------|-------------------|
| Sampling algorithms | Correct distributions, efficient | Mostly correct | Some errors | Major errors |
| Constraints | Identification and stationarity handled | Mostly handled | Partially handled | Not handled |
| Diagnostics | Multiple tests, proper interpretation | Standard diagnostics | Limited diagnostics | No diagnostics |

**Specific Checks:**
- [ ] Inverse Wishart sampling correct (5 pts)
- [ ] Identification constraint enforced (5 pts)
- [ ] Stationarity of Φ ensured (5 pts)
- [ ] ESS and Geweke computed (5 pts)
- [ ] Gelman-Rubin (if multiple chains) (5 pts)

---

### Code Quality (20 points)

| Criterion | Excellent (18-20) | Good (15-17) | Adequate (12-14) | Needs Work (0-11) |
|-----------|-------------------|--------------|------------------|-------------------|
| Organization | Clean classes, modular functions | Mostly organized | Functional but messy | Poor structure |
| Efficiency | Vectorized operations, minimal redundancy | Mostly efficient | Some inefficiencies | Very slow |
| Documentation | Comprehensive docstrings and comments | Good documentation | Basic documentation | Minimal |

**Specific Checks:**
- [ ] NumPy/SciPy used appropriately (5 pts)
- [ ] No unnecessary loops (5 pts)
- [ ] Clear variable names (5 pts)
- [ ] Progress monitoring during MCMC (5 pts)

---

### Interpretation & Insights (20 points)

| Criterion | Excellent (18-20) | Good (15-17) | Adequate (12-14) | Needs Work (0-11) |
|-----------|-------------------|--------------|------------------|-------------------|
| Prior justification | Strong economic reasoning | Reasonable rationale | Generic justification | No justification |
| Posterior interpretation | Deep insights, connects to theory | Good interpretation | Basic comments | No interpretation |
| Bayesian vs ML | Insightful comparison, explains differences | Notes differences | Superficial comparison | No comparison |

**Expected Deliverables:**
- Prior justification document (3-5 pages)
- Interpretation of posterior distributions
- Explanation of when Bayesian approach preferred
- Discussion of prior sensitivity

---

## Submission Instructions

### File Structure

```
mini_project_bayesian/
├── bayesian_dfm.py           # Gibbs sampler implementation
├── priors.py                 # Prior specification
├── diagnostics.py            # MCMC diagnostics
├── analysis.py               # Posterior analysis functions
├── tests/
│   ├── test_sampling.py      # Test conditional distributions
│   └── test_convergence.py   # Test diagnostics
├── notebooks/
│   ├── 01_prior_specification.ipynb
│   ├── 02_gibbs_sampler.ipynb
│   ├── 03_diagnostics.ipynb
│   └── 04_comparison_with_ml.ipynb
├── results/
│   ├── trace_plots.png
│   ├── posterior_densities.png
│   ├── impulse_responses.png
│   └── mcmc_diagnostics_report.md
├── data/
│   └── macro_data.csv
├── prior_justification.md
├── requirements.txt
└── README.md
```

### Submission Checklist

- [ ] All code runs without errors
- [ ] MCMC converges (R̂ < 1.1, ESS > 100)
- [ ] Prior justification document included
- [ ] Diagnostic report shows convergence
- [ ] Comparison with ML estimates included
- [ ] Notebooks have clear narrative
- [ ] All plots properly labeled with titles and legends

### How to Submit

1. Create a private GitHub repository
2. Push all code, notebooks, and results
3. Include `results/` directory with all figures
4. Share repository link with instructor
5. Include PDF export of main analysis notebook

**Deadline:** [To be announced]

**Late Policy:** 10% penalty per day, up to 3 days

---

## Resources

### Required Reading

- Kim & Nelson (1999). *State-Space Models with Regime Switching*, Chapter 6
- Carter & Kohn (1994). "On Gibbs Sampling for State Space Models." *Biometrika* 81(3), 541-553

### Recommended

- Frühwirth-Schnatter (2006). *Finite Mixture and Markov Switching Models*, Chapters 10-11
- Geweke (1992). "Evaluating the accuracy of sampling-based approaches to calculating posterior moments." *Bayesian Statistics* 4, 169-193
- Gelman & Rubin (1992). "Inference from iterative simulation using multiple sequences." *Statistical Science* 7(4), 457-472

### Software

- [PyMC3 documentation](https://docs.pymc.io/)
- [ArviZ (MCMC diagnostics)](https://arviz-devs.github.io/arviz/)
- [NumPy random sampling](https://numpy.org/doc/stable/reference/random/index.html)

---

## Common Pitfalls

1. **Non-convergence:** Insufficient burn-in or poor starting values
2. **High autocorrelation:** Need more thinning or reparameterization
3. **Identification issues:** Factors swap signs/order across iterations (resolve with constraints)
4. **Improper priors:** Check posterior is proper before sampling
5. **Numerical instability:** Covariance matrices not positive definite (use regularization)
6. **Slow mixing:** Poor conditioning of posterior (try reparameterization)
7. **Label switching:** Factors not identified uniquely (impose ordering constraint)

---

## Grading Summary

| Component | Points | Weight |
|-----------|--------|--------|
| Functionality | 35 | 35% |
| Statistical Correctness | 25 | 25% |
| Code Quality | 20 | 20% |
| Interpretation & Insights | 20 | 20% |
| **Total** | **100** | **100%** |

**Minimum to Pass:** 70/100

**Grade Boundaries:**
- A: 90-100
- B: 80-89
- C: 70-79
- F: Below 70

---

## Academic Integrity

- You may discuss high-level Bayesian concepts with classmates
- **All code must be your own**
- You may use PyMC3 for validation, not as your primary sampler
- Cite all external resources
- AI assistance permitted for syntax, not algorithm implementation

**Violations will result in zero credit.**

---

*"Bayesian inference provides a complete probabilistic description of uncertainty. For complex models like DFMs, MCMC makes the intractable tractable."*
