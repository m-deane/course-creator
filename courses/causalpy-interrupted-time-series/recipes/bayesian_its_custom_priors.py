"""
Recipe: Bayesian ITS with Custom Prior Specification
======================================================

Patterns for specifying, diagnosing, and communicating Bayesian priors
in Interrupted Time Series models using CausalPy + PyMC.

Why priors matter in ITS
------------------------
  The default weakly informative priors in CausalPy work well for
  standardised data. When your outcome is on a domain-specific scale
  (e.g., daily hospital admissions, revenue in millions), informative
  priors anchored to domain knowledge improve estimates and communicability.

Patterns covered
----------------
  1. Default CausalPy priors (baseline reference)
  2. Informative priors based on domain knowledge
  3. Prior predictive check — visualise what the priors imply
  4. Posterior predictive check — assess model fit
  5. Prior sensitivity analysis — how much do priors influence estimates?

Dependencies
------------
    pip install causalpy pymc arviz numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import Dict, Optional

warnings.filterwarnings('ignore')

try:
    import pymc as pm
    import arviz as az
    import causalpy as cp
    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    print(
        'PyMC / CausalPy not installed. '
        'Patterns A–E will show prior specification code without running.\n'
        'Install with: pip install causalpy pymc arviz'
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Generate ITS Data
# ─────────────────────────────────────────────────────────────────────────────

def make_its_data(
    n_periods: int = 50,
    intervention_time: int = 30,
    true_level_change: float = 15.0,
    true_slope_change: float = 0.8,
    noise_sd: float = 4.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate ITS data in the format expected by CausalPy.

    The outcome is on a scale of ~50–120 (e.g., weekly admissions),
    so priors anchored to this scale will be informative.
    """
    rng = np.random.default_rng(seed)
    periods = np.arange(1, n_periods + 1)

    y = (
        60.0
        + 0.5 * periods                                    # pre-trend
        + true_level_change * (periods >= intervention_time) # level change
        + true_slope_change * np.maximum(periods - intervention_time, 0)  # slope change
        + rng.normal(0, noise_sd, n_periods)
    )

    df = pd.DataFrame({
        'period':         periods,
        'outcome':        y,
        'post':           (periods >= intervention_time).astype(int),
        'time_since_post': np.maximum(periods - intervention_time, 0),
    })

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pattern A: Default CausalPy Priors (Reference)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PRIOR_CODE = '''
# Pattern A: Default CausalPy priors

import causalpy as cp

result_default = cp.InterruptedTimeSeries(
    data=df,
    treatment_time=30,
    formula="outcome ~ 1 + period + post + time_since_post",
    model=cp.pymc_models.LinearRegression(
        # CausalPy default priors (weakly informative):
        #   beta ~ Normal(0, 50)  for all regression coefficients
        #   sigma ~ HalfNormal(1)  for the noise SD
        sample_kwargs={
            "draws": 2000,
            "tune": 1000,
            "chains": 4,
            "target_accept": 0.9,
        }
    )
)

# Extract level change estimate
tau_samples = result_default.idata.posterior["post"].values.flatten()
print(f"Default priors: τ = {tau_samples.mean():.2f}  HDI: {az.hdi(tau_samples)}")
'''


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pattern B: Informative Priors via Model Subclassing
# ─────────────────────────────────────────────────────────────────────────────

INFORMATIVE_PRIOR_CODE = '''
# Pattern B: Domain-informed priors

import pymc as pm
import causalpy as cp
import numpy as np

class InformativeITSModel(cp.pymc_models.LinearRegression):
    """
    ITS model with priors anchored to domain knowledge.

    Domain knowledge (hypothetical hospital admissions study):
      - Intercept ~ 60 weekly admissions (uncertainty ±20)
      - Pre-trend ~ 0.5 per period (uncertainty ±1.0)
      - Level change ~ 10 admissions (could be up to ±30)
      - Slope change ~ 0.5 per period (uncertainty ±2)
      - Noise SD ~ 5 admissions (not much more than 15)
    """

    def build_model(self, X, y, coords):
        with pm.Model(coords=coords) as self.model:
            # Custom informative priors
            intercept = pm.Normal("Intercept",   mu=60, sigma=20)
            beta_t    = pm.Normal("period",       mu=0.5, sigma=1.0)
            beta_post = pm.Normal("post",         mu=10,  sigma=15)
            beta_slope= pm.Normal("time_since_post", mu=0.5, sigma=2.0)
            sigma     = pm.HalfNormal("sigma",    sigma=10)

            # Linear predictor
            mu = (
                intercept
                + beta_t * X["period"]
                + beta_post * X["post"]
                + beta_slope * X["time_since_post"]
            )

            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        return self.model


result_informative = cp.InterruptedTimeSeries(
    data=df,
    treatment_time=30,
    formula="outcome ~ 1 + period + post + time_since_post",
    model=InformativeITSModel(
        sample_kwargs={
            "draws": 2000,
            "tune": 1000,
            "chains": 4,
            "target_accept": 0.9,
        }
    )
)
'''


# ─────────────────────────────────────────────────────────────────────────────
# 4. Pattern C: Prior Predictive Check
# ─────────────────────────────────────────────────────────────────────────────

PRIOR_PREDICTIVE_CODE = '''
# Pattern C: Prior predictive check
# Before fitting, sample from the prior to visualise what the priors imply.

import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

n_periods = 50
periods = np.arange(1, n_periods + 1)
intervention_time = 30

with pm.Model() as prior_check_model:
    # Same priors as InformativeITSModel
    intercept = pm.Normal("Intercept",       mu=60, sigma=20)
    beta_t    = pm.Normal("period",          mu=0.5, sigma=1.0)
    beta_post = pm.Normal("post",            mu=10,  sigma=15)
    beta_slope= pm.Normal("time_since_post", mu=0.5, sigma=2.0)
    sigma     = pm.HalfNormal("sigma",       sigma=10)

    post        = (periods >= intervention_time).astype(int)
    time_since  = np.maximum(periods - intervention_time, 0)

    mu = (
        intercept
        + beta_t * periods
        + beta_post * post
        + beta_slope * time_since
    )
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, shape=n_periods)

    idata_prior = pm.sample_prior_predictive(samples=200, random_seed=42)

prior_samples = idata_prior.prior_predictive["y_obs"].values.squeeze()

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(periods, prior_samples[:50].T, color="gray", alpha=0.2, lw=0.8)
ax.axvline(intervention_time, color="red", ls="--", lw=1.5)
ax.set_xlabel("Period")
ax.set_ylabel("Outcome")
ax.set_title("Prior Predictive: 50 draws from the prior distribution", fontweight="bold")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# If the prior samples look unreasonable (e.g., negative admissions,
# or values in the thousands), tighten the priors.
'''


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pattern D: Posterior Predictive Check
# ─────────────────────────────────────────────────────────────────────────────

POSTERIOR_PREDICTIVE_CODE = '''
# Pattern D: Posterior predictive check
# After fitting, check whether model-generated data matches observed data.

import arviz as az
import matplotlib.pyplot as plt

# result is a fitted CausalPy InterruptedTimeSeries object
# idata.posterior_predictive contains the model predictions

if hasattr(result, "idata") and "posterior_predictive" in result.idata:
    ppc_samples = result.idata.posterior_predictive["y_obs"].values
    ppc_mean = ppc_samples.mean(axis=(0, 1))
    ppc_lo   = np.percentile(ppc_samples, 3,  axis=(0, 1))
    ppc_hi   = np.percentile(ppc_samples, 97, axis=(0, 1))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.scatter(df["period"], df["outcome"], s=20, color="steelblue", label="Observed")
    ax.plot(df["period"], ppc_mean, "-", color="navy", lw=2, label="PPC mean")
    ax.fill_between(df["period"], ppc_lo, ppc_hi, alpha=0.2, color="steelblue",
                    label="94% PPC interval")
    ax.axvline(30, color="red", ls="--", lw=1.5)
    ax.set_xlabel("Period")
    ax.set_title("Posterior Predictive Check", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    # Generate PPC manually
    result_ppc = result.idata.sample_posterior_predictive()
'''


# ─────────────────────────────────────────────────────────────────────────────
# 6. Pattern E: Prior Sensitivity Analysis (pure Python, no PyMC required)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_bayesian_its_approx(
    df: pd.DataFrame,
    prior_level_change_sd: float,
    outcome: str = 'outcome',
    period_col: str = 'period',
    n_samples: int = 4000,
    seed: int = 42,
) -> np.ndarray:
    """
    Approximate Bayesian ITS using rejection sampling on the level-change prior.

    This is an educational approximation — use PyMC for production.
    It demonstrates how the prior on the level change affects the posterior.

    The likelihood is a Normal with variance estimated from OLS residuals.
    """
    import statsmodels.formula.api as smf

    # OLS fit to get likelihood parameters
    model = smf.ols(
        f'{outcome} ~ {period_col} + post + time_since_post', data=df
    ).fit()

    y_hat    = model.fittedvalues.values
    resid_sd = float(np.std(model.resid))
    ols_est  = float(model.params['post'])

    # Approximate posterior via normal-normal conjugate update
    # Posterior precision = prior precision + likelihood precision
    prior_precision      = 1.0 / (prior_level_change_sd ** 2)
    likelihood_precision = 1.0 / (float(model.bse['post']) ** 2)

    posterior_precision = prior_precision + likelihood_precision
    posterior_sd        = 1.0 / np.sqrt(posterior_precision)
    posterior_mean      = (
        likelihood_precision * ols_est
        + prior_precision * 0.0           # prior centred at 0
    ) / posterior_precision

    rng = np.random.default_rng(seed)
    samples = rng.normal(posterior_mean, posterior_sd, n_samples)
    return samples


def prior_sensitivity_table(
    df: pd.DataFrame,
    prior_sds: List[float] = None,
    outcome: str = 'outcome',
) -> pd.DataFrame:
    """
    Compare posterior estimates under different prior SDs for the level change.

    A wider prior (SD = 50) is nearly non-informative.
    A narrower prior (SD = 5) is informative and will pull the estimate toward zero.

    Returns a table showing how the posterior mean and interval change.
    """
    if prior_sds is None:
        prior_sds = [5.0, 10.0, 20.0, 50.0, 100.0]

    rows = []
    for sd in prior_sds:
        samples = simulate_bayesian_its_approx(df, prior_level_change_sd=sd, outcome=outcome)
        p_pos = float((samples > 0).mean())
        rows.append({
            'Prior SD (level change)': sd,
            'Posterior mean':  float(samples.mean()),
            'Posterior SD':    float(samples.std()),
            '94% HDI lo':      float(np.percentile(samples, 3)),
            '94% HDI hi':      float(np.percentile(samples, 97)),
            'P(τ > 0)':        p_pos,
        })

    return pd.DataFrame(rows)


from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# Demonstration
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    TRUE_LEVEL_CHANGE = 15.0

    df = make_its_data(true_level_change=TRUE_LEVEL_CHANGE)
    print(f'Dataset: {len(df)} periods, intervention at t=30')
    print(f'True level change: {TRUE_LEVEL_CHANGE}')
    print()

    # Prior sensitivity analysis (runs without PyMC)
    print('Prior sensitivity analysis:')
    print('(How much does the prior on the level change affect the posterior?)')
    print()
    sens_df = prior_sensitivity_table(df, outcome='outcome')
    print(sens_df.to_string(index=False))
    print()
    print('Interpretation:')
    print('  - Wide prior (SD=100): posterior dominated by data')
    print('  - Narrow prior (SD=5):  posterior pulled toward zero')
    print('  - Crossover point: where prior starts affecting the conclusion')

    # Show that PyMC patterns are available
    print()
    print('CausalPy / PyMC code patterns available:')
    print('  Pattern A: Default priors            — DEFAULT_PRIOR_CODE')
    print('  Pattern B: Informative priors        — INFORMATIVE_PRIOR_CODE')
    print('  Pattern C: Prior predictive check    — PRIOR_PREDICTIVE_CODE')
    print('  Pattern D: Posterior predictive check — POSTERIOR_PREDICTIVE_CODE')
    print()
    print('Run with PyMC installed for full Bayesian estimation.')
    print()

    if HAS_PYMC:
        print('PyMC detected. You can run the full patterns.')
    else:
        print('Install PyMC: pip install causalpy pymc arviz')
