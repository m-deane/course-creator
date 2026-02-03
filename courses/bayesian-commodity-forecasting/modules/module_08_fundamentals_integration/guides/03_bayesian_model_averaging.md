# Bayesian Model Averaging for Commodity Forecasting

## In Brief

Bayesian Model Averaging (BMA) combines predictions from multiple models weighted by their posterior model probabilities. Instead of selecting a single "best" model, BMA accounts for model uncertainty by averaging over all candidate models, yielding forecasts that are often more robust and better-calibrated than any individual model.

## Key Insight

Model selection (choosing one model) discards information and understates uncertainty. If Model A has 60% posterior probability and Model B has 40%, why use only Model A? BMA uses both, weighted by their probabilities, capturing both parameter uncertainty (within models) and model uncertainty (across models). For commodity forecasting with competing fundamental specifications, BMA prevents overfitting to any single structural assumption.

## Formal Definition

### The Model Averaging Framework

Given data $\mathcal{D}$ and model space $\mathcal{M} = \{M_1, ..., M_K\}$:

**Posterior model probability:**
$$P(M_k | \mathcal{D}) = \frac{P(\mathcal{D} | M_k) P(M_k)}{\sum_{j=1}^K P(\mathcal{D} | M_j) P(M_j)}$$

Where:
- $P(\mathcal{D} | M_k)$ = Marginal likelihood (evidence) for model $k$
- $P(M_k)$ = Prior model probability

**BMA prediction:**
$$P(y^* | \mathcal{D}) = \sum_{k=1}^K P(y^* | M_k, \mathcal{D}) P(M_k | \mathcal{D})$$

**BMA point forecast:**
$$\hat{y}^* = \sum_{k=1}^K \mathbb{E}[y^* | M_k, \mathcal{D}] \cdot P(M_k | \mathcal{D})$$

**BMA prediction variance:**
$$\text{Var}(y^* | \mathcal{D}) = \underbrace{\sum_{k} \text{Var}(y^* | M_k) P(M_k | \mathcal{D})}_{\text{Within-model}} + \underbrace{\sum_{k} (\mathbb{E}[y^* | M_k] - \hat{y}^*)^2 P(M_k | \mathcal{D})}_{\text{Between-model}}$$

### Marginal Likelihood Computation

**Exact (rare):**
$$P(\mathcal{D} | M_k) = \int P(\mathcal{D} | \boldsymbol{\theta}_k, M_k) P(\boldsymbol{\theta}_k | M_k) d\boldsymbol{\theta}_k$$

**Approximate methods:**
1. **BIC approximation:** $\log P(\mathcal{D} | M_k) \approx -\frac{1}{2}\text{BIC}_k$
2. **WAIC:** Watanabe-Akaike Information Criterion
3. **Bridge sampling:** Monte Carlo estimate from MCMC samples
4. **Laplace approximation:** Gaussian approximation at MAP

### Model Space

**Example: Fundamental commodity models**

- $M_1$: AR(2) (autoregressive, no fundamentals)
- $M_2$: AR(2) + inventory
- $M_3$: AR(2) + inventory + production
- $M_4$: AR(2) + inventory + production + GDP
- $M_5$: GP with fundamentals as covariates

Each model has different complexity and assumptions.

## Intuitive Explanation

Think of weather forecasting:

**Model Selection:**
- Weather Service A: 70% chance rain
- Weather Service B: 30% chance rain
- You believe A is better → Use A → Carry umbrella

**Bayesian Model Averaging:**
- You're 60% confident A is better, 40% confident in B
- BMA forecast: 0.6 × 0.7 + 0.4 × 0.3 = 54% chance rain
- More nuanced decision: Maybe bring umbrella in bag

For commodity forecasting:

**Three competing models:**
1. Pure time series (ARIMA): Predicts oil price = $75
2. Inventory model: Predicts $78
3. Full fundamental model: Predicts $72

**Traditional model selection:**
- Cross-validation suggests inventory model is best
- Use $78 forecast
- Discard other models

**Bayesian Model Averaging:**
- Posterior probabilities: P(M₁) = 0.2, P(M₂) = 0.5, P(M₃) = 0.3
- BMA forecast: 0.2×75 + 0.5×78 + 0.3×72 = $75.60
- BMA uncertainty includes both parameter uncertainty and model uncertainty
- More robust: Doesn't fully commit to inventory model

## Code Implementation

### Basic BMA Implementation

```python
import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from scipy.special import logsumexp

class BayesianModelAveraging:
    """
    Bayesian Model Averaging for commodity forecasting.
    """

    def __init__(self):
        self.models = []
        self.traces = []
        self.log_likelihoods = []
        self.model_names = []

    def add_model(self, model, trace, name):
        """
        Add a fitted model to the ensemble.

        Args:
            model: PyMC model
            trace: MCMC samples (InferenceData)
            name: Model identifier
        """
        self.models.append(model)
        self.traces.append(trace)
        self.model_names.append(name)

        # Compute log pointwise predictive density (WAIC)
        log_lik = az.waic(trace, pointwise=True)
        self.log_likelihoods.append(log_lik.waic)

    def compute_model_weights(self, method='waic'):
        """
        Compute posterior model probabilities.

        Args:
            method: 'waic' | 'loo' | 'bic'

        Returns:
            weights: Posterior model probabilities
        """
        if method == 'waic':
            # Use WAIC for model comparison
            waic_values = []
            for trace in self.traces:
                waic = az.waic(trace)
                waic_values.append(waic.waic)

            waic_values = np.array(waic_values)

            # Convert to log probabilities (smaller WAIC = better)
            log_probs = -0.5 * waic_values
            log_probs -= logsumexp(log_probs)  # Normalize

            weights = np.exp(log_probs)

        elif method == 'loo':
            # Use LOO-CV
            loo_values = []
            for trace in self.traces:
                loo = az.loo(trace)
                loo_values.append(loo.loo)

            loo_values = np.array(loo_values)
            log_probs = -0.5 * loo_values
            log_probs -= logsumexp(log_probs)
            weights = np.exp(log_probs)

        else:
            raise ValueError(f"Unknown method: {method}")

        return weights

    def predict(self, X_new, model_weights=None):
        """
        BMA prediction for new data.

        Args:
            X_new: New covariate values [n_new, d]
            model_weights: Model weights (if None, compute from WAIC)

        Returns:
            predictions: BMA predictions [n_new]
            uncertainty: Total uncertainty [n_new]
        """
        if model_weights is None:
            model_weights = self.compute_model_weights()

        all_predictions = []
        all_uncertainties = []

        # Get predictions from each model
        for i, (model, trace) in enumerate(zip(self.models, self.traces)):
            with model:
                # Posterior predictive samples
                ppc = pm.sample_posterior_predictive(
                    trace,
                    var_names=['y_obs'],
                    predictions=True,
                    extend_inferencedata=False,
                    random_seed=42
                )

                # Extract predictions
                pred_samples = ppc.predictions['y_obs'].values
                pred_mean = pred_samples.mean(axis=(0, 1))
                pred_std = pred_samples.std(axis=(0, 1))

                all_predictions.append(pred_mean)
                all_uncertainties.append(pred_std)

        all_predictions = np.array(all_predictions)  # [K models, n_new]
        all_uncertainties = np.array(all_uncertainties)

        # BMA prediction: weighted average
        bma_predictions = np.average(all_predictions, axis=0, weights=model_weights)

        # BMA uncertainty: within-model + between-model
        within_model_var = np.average(all_uncertainties**2, axis=0, weights=model_weights)
        between_model_var = np.average(
            (all_predictions - bma_predictions)**2,
            axis=0,
            weights=model_weights
        )
        total_variance = within_model_var + between_model_var
        bma_uncertainty = np.sqrt(total_variance)

        return bma_predictions, bma_uncertainty, model_weights

    def summary(self):
        """
        Print BMA summary.
        """
        weights = self.compute_model_weights()

        print("=" * 60)
        print("BAYESIAN MODEL AVERAGING SUMMARY")
        print("=" * 60)

        for i, (name, weight) in enumerate(zip(self.model_names, weights)):
            waic = az.waic(self.traces[i])
            print(f"\nModel {i+1}: {name}")
            print(f"  Posterior probability: {weight:.4f}")
            print(f"  WAIC: {waic.waic:.2f}")
            print(f"  pWAIC: {waic.p_waic:.2f}")

        print("\n" + "=" * 60)
        print(f"Effective number of models: {1 / np.sum(weights**2):.2f}")
        print("=" * 60)


# Generate synthetic data
np.random.seed(42)
n = 200

# Covariates
X1 = np.random.normal(0, 1, n)  # Inventory deviation
X2 = np.random.normal(0, 1, n)  # Production growth
X3 = np.random.normal(0, 1, n)  # GDP growth

# True relationship: price = 50 + 2*inventory - 1*production + noise
y = 50 + 2*X1 - 1*X2 + np.random.normal(0, 2, n)

# Model 1: Intercept only (null model)
with pm.Model() as model1:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=5)

    mu = alpha
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    trace1 = pm.sample(1000, tune=1000, return_inferencedata=True, random_seed=42)

# Model 2: Inventory only
with pm.Model() as model2:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=5)
    sigma = pm.HalfNormal('sigma', sigma=5)

    mu = alpha + beta1 * X1
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    trace2 = pm.sample(1000, tune=1000, return_inferencedata=True, random_seed=42)

# Model 3: Inventory + Production
with pm.Model() as model3:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=5)
    beta2 = pm.Normal('beta2', mu=0, sigma=5)
    sigma = pm.HalfNormal('sigma', sigma=5)

    mu = alpha + beta1 * X1 + beta2 * X2
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    trace3 = pm.sample(1000, tune=1000, return_inferencedata=True, random_seed=42)

# Model 4: All variables (over-parameterized)
with pm.Model() as model4:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=5)
    beta2 = pm.Normal('beta2', mu=0, sigma=5)
    beta3 = pm.Normal('beta3', mu=0, sigma=5)
    sigma = pm.HalfNormal('sigma', sigma=5)

    mu = alpha + beta1 * X1 + beta2 * X2 + beta3 * X3
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    trace4 = pm.sample(1000, tune=1000, return_inferencedata=True, random_seed=42)

# Create BMA ensemble
bma = BayesianModelAveraging()
bma.add_model(model1, trace1, "Null (intercept only)")
bma.add_model(model2, trace2, "Inventory")
bma.add_model(model3, trace3, "Inventory + Production")
bma.add_model(model4, trace4, "Full (all variables)")

# Print summary
bma.summary()

# Visualize model weights
weights = bma.compute_model_weights()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Model weights
ax1.bar(range(len(weights)), weights, alpha=0.7)
ax1.set_xticks(range(len(weights)))
ax1.set_xticklabels(bma.model_names, rotation=45, ha='right')
ax1.set_ylabel('Posterior Model Probability')
ax1.set_title('Bayesian Model Averaging Weights')
ax1.grid(alpha=0.3, axis='y')

# Plot 2: Model comparison (WAIC)
waic_values = [az.waic(trace).waic for trace in bma.traces]
ax2.barh(range(len(waic_values)), waic_values, alpha=0.7)
ax2.set_yticks(range(len(waic_values)))
ax2.set_yticklabels(bma.model_names)
ax2.set_xlabel('WAIC (lower = better)')
ax2.set_title('Model Comparison via WAIC')
ax2.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('bma_weights.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Forecast Comparison: Individual Models vs BMA

```python
def compare_forecast_performance(y_true, X_test, bma):
    """
    Compare BMA forecast to individual model forecasts.

    Args:
        y_true: True test values
        X_test: Test covariates
        bma: BayesianModelAveraging object

    Returns:
        results: Dictionary of performance metrics
    """
    # Get BMA predictions
    bma_pred, bma_unc, weights = bma.predict(X_test)

    # Get individual model predictions
    individual_preds = []
    for model, trace in zip(bma.models, bma.traces):
        with model:
            # Simplified: use posterior mean of parameters
            # In practice, should use posterior predictive
            # This is a placeholder for illustration
            pass

    # Compute metrics
    bma_mae = np.mean(np.abs(y_true - bma_pred))
    bma_rmse = np.sqrt(np.mean((y_true - bma_pred)**2))

    # Coverage: fraction of true values within 95% CI
    ci_lower = bma_pred - 1.96 * bma_unc
    ci_upper = bma_pred + 1.96 * bma_unc
    coverage = np.mean((y_true >= ci_lower) & (y_true <= ci_upper))

    # Continuous ranked probability score (CRPS)
    # Simplified version
    crps = np.mean(np.abs(y_true - bma_pred) - 0.5 * bma_unc)

    results = {
        'mae': bma_mae,
        'rmse': bma_rmse,
        'coverage': coverage,
        'crps': crps
    }

    return results, bma_pred, bma_unc


# Create test set
n_test = 50
X1_test = np.random.normal(0, 1, n_test)
X2_test = np.random.normal(0, 1, n_test)
X3_test = np.random.normal(0, 1, n_test)
y_test = 50 + 2*X1_test - 1*X2_test + np.random.normal(0, 2, n_test)

# Note: Full implementation would require modifying models to accept new X
# This is a conceptual illustration

print("\nForecast Performance:")
print(f"  BMA MAE: {results['mae']:.3f}")
print(f"  BMA RMSE: {results['rmse']:.3f}")
print(f"  BMA Coverage: {results['coverage']:.2%}")
```

### Pseudo-BMA with Stacking

```python
def pseudo_bma_stacking(traces, y_holdout):
    """
    Pseudo-BMA using stacking (Yao et al. 2018).

    Finds optimal weights via cross-validation, not posterior probabilities.
    Often more robust than BIC/WAIC-based BMA.

    Args:
        traces: List of InferenceData traces
        y_holdout: Holdout data for validation

    Returns:
        stacking_weights: Optimal model weights
    """
    from scipy.optimize import minimize

    K = len(traces)
    n = len(y_holdout)

    # Get log predictive densities for each model on holdout
    log_pred_densities = []
    for trace in traces:
        # Extract posterior predictive samples
        # This is simplified; full implementation requires careful alignment
        log_lik = trace.log_likelihood['y_obs'].values
        mean_log_lik = log_lik.mean(axis=(0, 1))
        log_pred_densities.append(mean_log_lik)

    log_pred_densities = np.array(log_pred_densities)  # [K, n]

    # Optimization: find weights that maximize stacked predictive density
    def negative_elpd(weights):
        # Ensure weights sum to 1 and are non-negative
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1):
            return 1e10

        # Expected log predictive density
        stacked_log_pred = logsumexp(log_pred_densities, axis=0, b=weights[:, None])
        return -stacked_log_pred.sum()

    # Optimize
    from scipy.optimize import minimize
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    bounds = [(0, 1) for _ in range(K)]

    result = minimize(
        negative_elpd,
        x0=np.ones(K) / K,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    stacking_weights = result.x

    return stacking_weights


# Example usage
stacking_weights = pseudo_bma_stacking(bma.traces, y[-50:])  # Use last 50 points

print("\nStacking Weights vs BMA Weights:")
bma_weights = bma.compute_model_weights()
for i, name in enumerate(bma.model_names):
    print(f"  {name}:")
    print(f"    BMA: {bma_weights[i]:.4f}")
    print(f"    Stacking: {stacking_weights[i]:.4f}")
```

## Common Pitfalls

**1. Misspecified Prior Model Probabilities**
- **Problem:** Setting P(M_k) = 1/K (uniform) when domain knowledge suggests otherwise
- **Symptom:** BMA gives weight to implausible models
- **Solution:** Use informative prior model probabilities based on theory/experience

**2. Overlapping Model Space**
- **Problem:** Including highly similar models (AR(2) and AR(3))
- **Symptom:** Weight split between similar models, overweighting that model class
- **Solution:** Ensure diverse model space or adjust prior probabilities

**3. Ignoring Computational Cost**
- **Problem:** Averaging over 100 models, each requiring MCMC
- **Symptom:** Prohibitive computation time
- **Solution:** Pre-screen with cheap IC (BIC), then BMA over top 5-10 models

**4. Marginal Likelihood Approximation Error**
- **Problem:** Using BIC approximation when sample size small
- **Symptom:** Incorrect model weights
- **Solution:** Use exact marginal likelihood (bridge sampling) or WAIC/LOO

**5. Over-Interpreting Small Weight Differences**
- **Problem:** Choosing single model because P(M_1) = 0.51 vs P(M_2) = 0.49
- **Symptom:** Not actually doing BMA, still model selecting
- **Solution:** Use BMA when weights are split; model selection only when dominant weight (>0.9)

## Connections

**Builds on:**
- Module 6: MCMC inference (generating traces for each model)
- Module 8.2: Fundamental variables (constructing competing model specifications)
- Model comparison (WAIC, LOO, marginal likelihood)

**Leads to:**
- Ensemble forecasting (combining models for robustness)
- Model selection (BMA as alternative to selecting single model)
- Uncertainty quantification (BMA properly accounts for model uncertainty)

**Related methods:**
- Stacking (Pseudo-BMA via cross-validation)
- Bayesian model selection (choosing single model via posterior odds)
- Bootstrap aggregating (bagging, frequentist averaging)

## Practice Problems

1. **Weight Calculation**
   Three models with WAIC values: 2500, 2520, 2600

   Calculate posterior model probabilities using:
   P(M_k | D) ∝ exp(-WAIC_k / 2)

2. **Prediction Variance Decomposition**
   Two models:
   - M₁: E[y*] = 50, SD[y*] = 5, P(M₁) = 0.6
   - M₂: E[y*] = 55, SD[y*] = 4, P(M₂) = 0.4

   Calculate:
   - BMA prediction
   - Within-model variance
   - Between-model variance
   - Total BMA variance

3. **Effective Number of Models**
   Model weights: [0.5, 0.3, 0.15, 0.05]

   Calculate: N_eff = 1 / Σ w_k²

   Interpretation: How many "effective" models?

4. **When to Use BMA vs Selection**
   Scenario A: Weights = [0.95, 0.03, 0.02]
   Scenario B: Weights = [0.4, 0.35, 0.25]

   Which scenario: (a) Use BMA? (b) Select single model?

5. **Commodity Application**
   Forecasting oil prices with:
   - M₁: Pure time series (P = 0.2)
   - M₂: Inventory model (P = 0.5)
   - M₃: Full fundamentals (P = 0.3)

   Forecasts: M₁→75, M₂→78, M₃→72

   - BMA forecast?
   - If M₂ and M₃ both use inventory, is model space appropriate?

## Further Reading

**Foundational:**
1. **Hoeting et al. (2000)** - "Bayesian Model Averaging: A Tutorial" - Comprehensive introduction
2. **Raftery et al. (1997)** - "Bayesian Model Averaging for Linear Regression Models" - BIC approximation
3. **Madigan & Raftery (1994)** - "Model Selection and Accounting for Model Uncertainty" - Theory

**Model Comparison:**
4. **Vehtari et al. (2017)** - "Practical Bayesian Model Evaluation Using LOO-CV" - LOO for BMA
5. **Yao et al. (2018)** - "Using Stacking to Average Bayesian Predictive Distributions" - Stacking vs BMA

**Applications:**
6. **Wright (2008)** - "Bayesian Model Averaging and Exchange Rate Forecasts" - Economics
7. **Garratt et al. (2009)** - "Forecast Uncertainties in Macroeconometric Modelling" - Policy forecasting

**Computational Methods:**
8. **Gronau et al. (2017)** - "bridgesampling: Bridge Sampling for Marginal Likelihoods" - R package
9. **PyMC Documentation: Model Comparison** - WAIC/LOO implementation

**Commodity-Specific:**
10. **Baumeister & Kilian (2015)** - "Forecasting the Real Price of Oil in a Changing World" - Model uncertainty in oil

---

*"BMA doesn't pick the 'best' model—it uses all models weighted by their plausibility, capturing model uncertainty ignored by selection."*
