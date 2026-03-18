# CausalPy-PyMC Internals

## In Brief

CausalPy is a thin wrapper around PyMC that adds causal semantics: it knows what the treatment variable is, when the intervention occurred, and how to compute the counterfactual. Understanding the internals lets you debug, extend, and customize ITS models for cases the default API does not handle.

## Key Insight

CausalPy's `InterruptedTimeSeries` class does four things: (1) builds the design matrix from your formula, (2) passes it to a PyMC model object, (3) runs NUTS sampling, (4) computes counterfactual predictions and stores everything in an ArviZ `InferenceData` object. Each step is inspectable and modifiable.

---

## The CausalPy Architecture in Detail

### Step 1: Design Matrix Construction

CausalPy uses `formulaic` (a Python port of R's formula parser) to build the design matrix from your formula string.

```python
import formulaic
import pandas as pd
import numpy as np

# Example: formula y ~ 1 + t + treated + t_post
df = pd.DataFrame({
    "t": np.arange(10),
    "treated": [0]*5 + [1]*5,
    "t_post": [0]*5 + list(range(5)),
    "y": np.random.normal(100, 10, 10),
})

# This is what CausalPy does internally
y, X = formulaic.model_matrix("y ~ 1 + t + treated + t_post", df)
print("Design matrix X:")
print(X.head())
print("Variable names:", list(X.columns))
```

The variable names in the design matrix become the names of the regression coefficients in the PyMC model (and thus in the ArviZ `InferenceData`).

### Step 2: PyMC Model Construction

The `LinearRegression` model object builds the PyMC model. The actual PyMC code is roughly:

```python
import pymc as pm
import numpy as np

def build_its_pymc_model(X, y):
    """
    This is a simplified version of what CausalPy builds internally.
    The actual CausalPy implementation has additional features, but this
    captures the essential structure.
    """
    n_obs, n_features = X.shape

    with pm.Model() as model:
        # Store data as PyMC Data objects (allows later updating)
        X_data = pm.Data("X", X)
        y_data = pm.Data("y_obs", y)

        # Prior on regression coefficients
        # The default in CausalPy LinearRegression is Normal(0, 20)
        # CausalPy may use different scaling based on data
        beta = pm.Normal(
            "beta",
            mu=0,
            sigma=20,
            shape=n_features,
        )

        # Prior on noise standard deviation
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Linear predictor
        mu = pm.Deterministic("mu", pm.math.dot(X_data, beta))

        # Likelihood
        y_hat = pm.Normal("y_hat", mu=mu, sigma=sigma, observed=y_data)

    return model
```

### Step 3: Sampling

CausalPy calls `pm.sample(**sample_kwargs)` on the constructed model. The `sample_kwargs` you pass to `LinearRegression(sample_kwargs=...)` are forwarded directly.

### Step 4: Counterfactual Computation

After sampling, CausalPy computes the counterfactual predictions by:
1. Setting all treatment-related variables (`treated`, `t_post`) to 0
2. Making predictions under this modified design matrix
3. Storing the counterfactual predictions in the `InferenceData`

```python
# Conceptual version of counterfactual computation
def compute_counterfactual(model_result, X_full, treatment_cols):
    """
    Compute counterfactual predictions by zeroing out treatment variables.
    """
    X_counterfactual = X_full.copy()
    for col in treatment_cols:
        X_counterfactual[col] = 0

    # Predict under counterfactual design matrix
    # Uses posterior samples of beta
    posterior_beta = model_result.idata.posterior["beta"].values  # shape: (chains, draws, p)
    counterfactual_predictions = np.einsum(
        "ijk,lk->ijl",
        posterior_beta,
        X_counterfactual.values,
    )
    return counterfactual_predictions
```

---

## Accessing the Underlying PyMC Model

After fitting, the PyMC model is accessible at `result.model`:

```python
import causalpy as cp
import pandas as pd
import numpy as np

# Fit model
result = cp.InterruptedTimeSeries(
    data=df,
    treatment_time=24,
    formula="y ~ 1 + t + treated + t_post",
    model=cp.pymc_models.LinearRegression(
        sample_kwargs={"draws": 500, "tune": 500, "chains": 2, "random_seed": 42}
    ),
)

# Access the underlying PyMC model
pymc_model = result.model
print(type(pymc_model))  # <class 'pymc.model.Model'>

# List all random variables in the model
print("Random variables:", pymc_model.basic_RVs)

# Access the model graph
# pymc.model_to_graphviz(pymc_model)  # Requires graphviz
```

---

## The InferenceData Object

CausalPy stores all results in an ArviZ `InferenceData` object at `result.idata`. This is a structured container with multiple groups:

```python
# Explore the InferenceData structure
idata = result.idata
print(idata)

# Main groups:
# - posterior: samples from the posterior
# - prior: samples from the prior (if prior predictive was run)
# - posterior_predictive: predictions from the posterior
# - observed_data: the data used to fit the model
# - sample_stats: NUTS diagnostics (divergences, tree depth, etc.)
# - log_likelihood: pointwise log-likelihood for LOO-CV

# Access posterior samples
print("\nPosterior variables:", list(idata.posterior.data_vars))
print("Posterior shape:", idata.posterior["treated"].shape)  # (chains, draws)

# Access observed data
print("\nObserved data variables:", list(idata.observed_data.data_vars))

# Access sample statistics (NUTS diagnostics)
print("\nSample stats:", list(idata.sample_stats.data_vars))
# 'diverging': boolean array marking divergences
# 'energy': NUTS energy diagnostic
# 'tree_depth': how deep each NUTS tree was
```

---

## The Model Graph

PyMC can visualize the model as a Bayesian network:

```python
import pymc as pm

# Visualize the model graph
# Requires: pip install graphviz python-graphviz
# graph = pm.model_to_graphviz(result.model)
# graph.render("its_model_graph", format="png")

# Alternatively, print the model's free variables
with result.model:
    print("Unobserved variables (parameters):")
    for rv in result.model.unobserved_RVs:
        print(f"  {rv.name}: {rv.type}")

    print("\nObserved variables (data):")
    for rv in result.model.observed_RVs:
        print(f"  {rv.name}")
```

---

## Modifying CausalPy Models

The cleanest way to extend CausalPy for custom needs is to subclass `cp.pymc_models.LinearRegression` and override `build_model`:

### Example: Poisson Regression for Count Outcomes

ITS models for count data (hospital admissions, crime counts, accident numbers) often fit better with a Poisson or Negative Binomial likelihood.

```python
import causalpy as cp
import pymc as pm
import numpy as np

class PoissonITSModel(cp.pymc_models.LinearRegression):
    """
    ITS model with Poisson likelihood for count outcomes.
    Uses log link function: E[Y] = exp(X * beta)
    """

    def build_model(self, X, y, coords):
        with pm.Model(coords=coords) as self.model:
            # Data
            X_ = pm.Data("X", X, dims=["obs", "coeffs"])
            y_ = pm.Data("y_obs", y, dims=["obs"])

            # Priors on regression coefficients (on log scale)
            beta = pm.Normal(
                "beta",
                mu=0,
                sigma=1,  # Smaller sigma appropriate for log-scale coefficients
                dims=["coeffs"],
            )

            # Log-linear predictor
            log_mu = pm.Deterministic("log_mu", pm.math.dot(X_, beta))
            mu = pm.Deterministic("mu", pm.math.exp(log_mu))

            # Poisson likelihood
            y_hat = pm.Poisson("y_hat", mu=mu, observed=y_, dims=["obs"])

        return self.model


# Usage
# result = cp.InterruptedTimeSeries(
#     data=df,
#     treatment_time=t_star,
#     formula="count_outcome ~ 1 + t + treated + t_post",
#     model=PoissonITSModel(sample_kwargs={"draws": 1000, "tune": 1000, "chains": 4}),
# )
```

### Example: AR(1) Error Structure

```python
class AR1ITS(cp.pymc_models.LinearRegression):
    """
    ITS model with AR(1) autocorrelated errors.
    Handles temporal autocorrelation explicitly.
    """

    def build_model(self, X, y, coords):
        n = len(y)

        with pm.Model(coords=coords) as self.model:
            X_ = pm.Data("X", X, dims=["obs", "coeffs"])
            y_ = pm.Data("y_obs", y, dims=["obs"])

            # Regression coefficients
            beta = pm.Normal("beta", mu=0, sigma=10, dims=["coeffs"])

            # AR(1) coefficient (-1, 1)
            rho = pm.Uniform("rho", lower=-1, upper=1)

            # Innovation standard deviation
            sigma = pm.HalfNormal("sigma", sigma=5)

            # Linear predictor (mean)
            mu_det = pm.Deterministic("mu_det", pm.math.dot(X_, beta))

            # AR(1) errors
            # ar_errors[0] ~ Normal(0, sigma)
            # ar_errors[t] ~ Normal(rho * ar_errors[t-1], sigma)
            ar_errors = pm.AR("ar_errors", rho=[rho], sigma=sigma, shape=n)

            # Combined mean
            mu = pm.Deterministic("mu", mu_det + ar_errors)

            # Likelihood (with tiny sigma to enforce exact observed values)
            y_hat = pm.Normal("y_hat", mu=mu, sigma=0.01, observed=y_, dims=["obs"])

        return self.model
```

---

## Inspecting the Prior

CausalPy supports prior predictive sampling through the underlying PyMC model:

```python
import pymc as pm

# Sample from the prior predictive distribution
with result.model:
    prior_pred = pm.sample_prior_predictive(
        samples=500,
        random_seed=42,
    )

# The prior predictive is also stored in idata if you run it before sampling
# Access as: prior_pred.prior_predictive["y_hat"]

import arviz as az
import matplotlib.pyplot as plt

# Visualize prior predictive vs observed
az.plot_ppc(
    az.from_dict(
        observed_data={"y_hat": result.idata.observed_data["y_hat"].values},
        prior_predictive={"y_hat": prior_pred.prior_predictive["y_hat"].values},
    ),
    group="prior",
)
plt.title("Prior Predictive Check")
plt.show()
```

---

## CausalPy Source Code Reference

Key classes and methods (CausalPy >= 0.4):

| Location | Class/Method | Purpose |
|----------|-------------|---------|
| `causalpy.pymc_models` | `LinearRegression` | Default Gaussian ITS model |
| `causalpy.experiment_classes` | `InterruptedTimeSeries` | Main ITS interface |
| `InterruptedTimeSeries.plot()` | Method | Two-panel visualization |
| `InterruptedTimeSeries.summary()` | Method | Posterior summary table |
| `InterruptedTimeSeries.model` | Attribute | The underlying PyMC model |
| `InterruptedTimeSeries.idata` | Attribute | ArviZ InferenceData |

The source is available at: https://github.com/pymc-labs/CausalPy

---

## Connections

- **Builds on:** Bayesian ITS motivation (Guide 1)
- **Leads to:** Prior specification (Guide 3), from-scratch PyMC notebook (Notebook 1)
- **Related to:** PyMC documentation, ArviZ documentation, formulaic documentation

## Further Reading

- PyMC documentation: https://docs.pymc.io
- CausalPy source: https://github.com/pymc-labs/CausalPy
- ArviZ documentation: https://python.arviz.org
- Martin, O., Kumar, R., & Lao, J. (2021). *Bayesian Modeling and Computation in Python* — free online, uses PyMC
