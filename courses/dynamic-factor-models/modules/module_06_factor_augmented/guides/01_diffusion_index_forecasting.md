# Diffusion Index Forecasting

> **Reading time:** ~13 min | **Module:** Module 6: Factor Augmented | **Prerequisites:** Modules 0-5

<div class="callout-key">

**Key Concept Summary:** Diffusion index forecasting uses extracted factors from large datasets to predict target variables, leveraging information from hundreds of economic indicators without directly estimating high-dimensional models. This approach, pioneered by Stock and Watson, provides robust forecasts by aggregati...

</div>

## In Brief

Diffusion index forecasting uses extracted factors from large datasets to predict target variables, leveraging information from hundreds of economic indicators without directly estimating high-dimensional models. This approach, pioneered by Stock and Watson, provides robust forecasts by aggregating signals across many variables while quantifying prediction uncertainty.

<div class="callout-insight">

**Insight:** Instead of choosing which predictors to include in a forecasting model, diffusion index methods extract common factors from all available predictors and use these factors as regressors. This transforms a high-dimensional forecasting problem ($N$ predictors) into a low-dimensional one ($r$ factors), combining information optimally while avoiding overfitting.

</div>
---

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## 1. The Diffusion Index Approach

### Motivation

Traditional forecasting faces dimensionality problems:
- With $N$ predictors and $T$ observations, if $N > T$, OLS is infeasible
- Variable selection is unstable and ad-hoc
- Information in unused variables is discarded

**Solution:** Extract $r \ll N$ factors that summarize co-movement across all predictors.

### Core Framework

Given:
- Target variable: $y_t$ (e.g., GDP growth, inflation)
- Predictor panel: $X_t$ ($N \times 1$ vector at time $t$)
- Forecast horizon: $h$ periods ahead

**Step 1: Factor Extraction**

From predictor panel $X$ ($T \times N$):
$$X_t = \Lambda F_t + e_t$$

Estimate factors $\hat{F}_t$ using principal components.

**Step 2: Forecasting Regression**

Use extracted factors as predictors:
$$y_{t+h} = \alpha + \beta' \hat{F}_t + \varepsilon_{t+h}$$

This is the **diffusion index forecast** for horizon $h$.

### Name Origin

"Diffusion index" originally referred to the fraction of indicators increasing/decreasing. Stock-Watson generalized this to continuous factors capturing diffusion of economic conditions across many series.

---

## 2. Mathematical Formulation

### Two-Step Estimation

**Step 1: Extract Factors**

Given standardized data $X$ ($T \times N$), compute:
$$\hat{F} = \sqrt{T} \cdot \text{eigenvectors}_{1:r}(X'X/T)$$

These are the first $r$ principal components, scaled by $\sqrt{T}$.

**Step 2: Forecast Regression**

$$y_{t+h} = \alpha + \sum_{j=1}^r \beta_j \hat{F}_{jt} + \varepsilon_{t+h}$$

Estimate $(\alpha, \beta)$ by OLS of $y_{t+h}$ on $\hat{F}_t$ for $t = 1, ..., T-h$.

### Alternative Specifications

**Direct Forecasting** (above):
$$y_{t+h} = \alpha_h + \beta_h' \hat{F}_t + \varepsilon_{t+h}$$

**Iterated Forecasting** (if factors have dynamics):
$$\hat{F}_{t+1} = A \hat{F}_t + u_{t+1}$$
$$y_{t+1} = \alpha_1 + \beta_1' \hat{F}_t + \varepsilon_{t+1}$$

Then iterate forward to horizon $h$.

**Autoregressive Augmentation**:
$$y_{t+h} = \alpha + \sum_{i=1}^p \phi_i y_{t-i} + \beta' \hat{F}_t + \varepsilon_{t+h}$$

Adds lagged dependent variable for improved accuracy.

---

## 3. Theoretical Properties

### Consistency

Under standard DFM assumptions (Stock-Watson 2002):

$$\sqrt{T}(\hat{\beta} - \beta^*) \xrightarrow{d} N(0, V_\beta)$$

where $\beta^*$ is the pseudo-true parameter (projection coefficient onto estimated factors).

**Key result:** Estimation error in $\hat{F}_t$ does not affect asymptotic distribution when $N, T \to \infty$ with $\sqrt{T}/N \to 0$.

### Forecast Error Decomposition

Forecast error at horizon $h$:
$$y_{T+h} - \hat{y}_{T+h|T} = \underbrace{(y_{T+h} - y_{T+h}^*)}_{\text{Fundamental error}} + \underbrace{(y_{T+h}^* - \hat{y}_{T+h|T})}_{\text{Estimation error}}$$

where $y_{T+h}^*$ is the forecast using true factors.

With large $N$, estimation error vanishes faster than fundamental error.

### Optimal Number of Factors

More factors reduce approximation bias but increase estimation variance.

Trade-off managed by:
- Information criteria (Bai-Ng)
- Cross-validation
- Out-of-sample performance

Typically $r = 3-8$ for macroeconomic panels.

---

## 4. Uncertainty Quantification

### Forecast Standard Errors

Naive approach (ignoring factor estimation uncertainty):
$$\text{se}(\hat{y}_{T+h|T}) = \sqrt{\hat{\sigma}^2_\varepsilon \cdot (1 + \hat{F}_T' (\hat{F}'\hat{F})^{-1} \hat{F}_T)}$$

where $\hat{\sigma}^2_\varepsilon$ is residual variance.

### Bootstrap Procedures

More accurate uncertainty quantification via bootstrap:

**Residual Bootstrap:**
1. Estimate $\hat{F}_t$ and $(\hat{\alpha}, \hat{\beta})$
2. Compute residuals $\hat{\varepsilon}_{t+h} = y_{t+h} - \hat{\alpha} - \hat{\beta}'\hat{F}_t$
3. Resample residuals: $\varepsilon^*_{t+h} \sim \hat{\varepsilon}$
4. Generate pseudo-forecasts: $y^*_{t+h} = \hat{\alpha} + \hat{\beta}'\hat{F}_t + \varepsilon^*_{t+h}$
5. Re-estimate on bootstrap sample
6. Repeat $B$ times, compute quantiles

**Block Bootstrap:**

For dependent data, resample blocks of residuals to preserve serial correlation.

### Prediction Intervals

95% prediction interval:
$$\hat{y}_{T+h|T} \pm 1.96 \cdot \text{se}(\hat{y}_{T+h|T})$$

Or use bootstrap quantiles for non-Gaussian errors.

---

## 5. Intuitive Explanation

### Why Diffusion Indices Work

Imagine forecasting unemployment using:
- Industrial production
- Retail sales
- Housing starts
- Consumer confidence
- Interest rates
- Stock prices
- ... (100 more variables)

**Problem:** Can't include all in a regression (too many parameters).

**Solution:** Notice many move together (co-movement). Extract 3-5 "aggregate signals":
- Factor 1: "Real activity"
- Factor 2: "Financial conditions"
- Factor 3: "Inflation pressures"

These factors capture information from all 100+ variables in just 3-5 numbers.

### Visual Intuition

```
Raw Data (100 variables)          Factors (5 dimensions)       Forecast
    [X₁, X₂, ..., X₁₀₀]     →     [F₁, F₂, F₃, F₄, F₅]    →    ŷₜ₊ₕ
         ↓                              ↓
    High-dimensional              Low-dimensional
    Correlated                    Orthogonal
    Noisy                         Signal extraction
```

### Economic Rationale

Macroeconomic variables respond to common shocks:
- Technology shocks affect output, employment, investment
- Monetary policy shocks affect rates, credit, spending
- Oil price shocks affect inflation, transportation costs

Factors represent these common shocks. Forecasting with factors means forecasting with aggregated shock information.

---

## 6. Code Implementation

### Basic Diffusion Index Forecast

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">diffusionindexforecaster.py</span>
</div>

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats

class DiffusionIndexForecaster:
    """
    Diffusion index forecasting using principal components.

    Parameters
    ----------
    n_factors : int
        Number of factors to extract
    horizon : int
        Forecast horizon (h-step ahead)
    include_ar : bool
        Whether to include autoregressive terms
    ar_lags : int
        Number of AR lags (if include_ar=True)
    """

    def __init__(self, n_factors=5, horizon=1, include_ar=False, ar_lags=1):
        self.n_factors = n_factors
        self.horizon = horizon
        self.include_ar = include_ar
        self.ar_lags = ar_lags

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_factors)
        self.reg = LinearRegression()

        self.factors_ = None
        self.factor_loadings_ = None
        self.forecast_coef_ = None
        self.forecast_intercept_ = None

    def fit(self, X, y):
        """
        Fit diffusion index forecasting model.

        Parameters
        ----------
        X : array-like, shape (T, N)
            Predictor panel data
        y : array-like, shape (T,)
            Target variable
        """
        X = np.asarray(X)
        y = np.asarray(y)
        T, N = X.shape

        # Step 1: Extract factors from full sample
        X_scaled = self.scaler.fit_transform(X)
        self.factors_ = self.pca.fit_transform(X_scaled)
        self.factor_loadings_ = self.pca.components_.T

        # Step 2: Align factors with h-step ahead target
        # For horizon h, regress y_{t+h} on F_t
        T_forecast = T - self.horizon

        F_t = self.factors_[:T_forecast, :]
        y_ahead = y[self.horizon:T_forecast + self.horizon]

        # Add AR terms if requested
        if self.include_ar:
            # Create lagged y matrix
            y_lags = self._create_lags(y, self.ar_lags)
            y_lags = y_lags[:T_forecast, :]

            # Combine factors and AR terms
            predictors = np.column_stack([F_t, y_lags])
        else:
            predictors = F_t

        # Estimate forecast regression
        self.reg.fit(predictors, y_ahead)
        self.forecast_coef_ = self.reg.coef_
        self.forecast_intercept_ = self.reg.intercept_

        return self

    def predict(self, X, y_hist=None):
        """
        Generate h-step ahead forecast.

        Parameters
        ----------
        X : array-like, shape (T_new, N)
            New predictor data
        y_hist : array-like, optional
            Historical y values (needed if include_ar=True)

        Returns
        -------
        forecast : array-like
            h-step ahead forecasts
        """
        X = np.asarray(X)

        # Transform X to factors
        X_scaled = self.scaler.transform(X)
        factors = self.pca.transform(X_scaled)

        # Add AR terms if needed
        if self.include_ar:
            if y_hist is None:
                raise ValueError("y_hist required when include_ar=True")
            y_lags = self._create_lags(y_hist, self.ar_lags)
            predictors = np.column_stack([factors, y_lags])
        else:
            predictors = factors

        # Generate forecast
        forecast = self.reg.predict(predictors)

        return forecast

    def forecast_one_step(self, X_T, y_hist=None):
        """
        Generate single h-step ahead forecast from most recent data.

        Parameters
        ----------
        X_T : array-like, shape (N,)
            Most recent predictor values
        y_hist : array-like, optional
            Recent y values for AR terms

        Returns
        -------
        forecast : float
            h-step ahead point forecast
        """
        X_T = np.asarray(X_T).reshape(1, -1)

        if self.include_ar and y_hist is not None:
            y_hist = np.asarray(y_hist)
            forecast = self.predict(X_T, y_hist[-self.ar_lags:].reshape(1, -1))
        else:
            forecast = self.predict(X_T)

        return forecast[0]

    def compute_prediction_interval(self, X, y_hist=None, confidence=0.95):
        """
        Compute prediction intervals using residual variance.

        Parameters
        ----------
        X : array-like, shape (T_new, N)
            New predictor data
        y_hist : array-like, optional
            Historical y values
        confidence : float
            Confidence level (default 0.95)

        Returns
        -------
        lower : array-like
            Lower bounds of prediction intervals
        upper : array-like
            Upper bounds of prediction intervals
        """
        forecast = self.predict(X, y_hist)

        # Compute residual standard error
        # (Note: this is a simplified approach; proper SE accounts for factor uncertainty)
        se = np.sqrt(np.sum(self.reg.predict(
            self._get_training_predictors(X, y_hist))**2) /
            (len(self.reg.predict(self._get_training_predictors(X, y_hist))) -
             len(self.forecast_coef_) - 1))

        # Critical value
        z = stats.norm.ppf((1 + confidence) / 2)

        lower = forecast - z * se
        upper = forecast + z * se

        return lower, upper

    def explained_variance_ratio(self):
        """Return variance explained by each factor."""
        return self.pca.explained_variance_ratio_

    def _create_lags(self, y, lags):
        """Create lagged matrix from time series."""
        T = len(y)
        Y_lagged = np.zeros((T - lags, lags))

        for i in range(lags):
            Y_lagged[:, i] = y[lags - i - 1:T - i - 1]

        return Y_lagged

    def _get_training_predictors(self, X, y_hist):
        """Helper to reconstruct training predictors."""
        X_scaled = self.scaler.transform(X)
        factors = self.pca.transform(X_scaled)

        if self.include_ar and y_hist is not None:
            y_lags = self._create_lags(y_hist, self.ar_lags)
            predictors = np.column_stack([factors, y_lags])
        else:
            predictors = factors

        return predictors
```

</div>

### Example Application

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# Generate synthetic data
np.random.seed(123)
T, N = 300, 50
r = 3

# True factors
F_true = np.random.randn(T, r)
for i in range(1, r):
    F_true[:, i] = 0.7 * F_true[:, i-1] + np.sqrt(1 - 0.7**2) * np.random.randn(T)

# Factor loadings
Lambda = np.random.randn(N, r) / np.sqrt(r)

# Predictor panel
X = F_true @ Lambda.T + 0.5 * np.random.randn(T, N)

# Target variable (depends on factors with lag)
beta_true = np.array([0.5, 0.3, -0.2])
y = F_true @ beta_true + 0.3 * np.random.randn(T)

# Split into train/test
T_train = 250
X_train, X_test = X[:T_train], X[T_train:]
y_train, y_test = y[:T_train], y[T_train:]

# Fit diffusion index model
di_model = DiffusionIndexForecaster(n_factors=5, horizon=1)
di_model.fit(X_train, y_train)

# Generate forecasts
y_pred = di_model.predict(X_test[:len(y_test)-1])

# Evaluate
forecast_errors = y_test[1:] - y_pred
rmse = np.sqrt(np.mean(forecast_errors**2))
mae = np.mean(np.abs(forecast_errors))

print(f"Out-of-sample RMSE: {rmse:.4f}")
print(f"Out-of-sample MAE: {mae:.4f}")
print(f"\nVariance explained by factors:")
for i, var_ratio in enumerate(di_model.explained_variance_ratio(), 1):
    print(f"  Factor {i}: {var_ratio:.3f}")
```

</div>

### Real Data Example: Forecasting GDP

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">load_fredmd_example.py</span>
</div>

```python
import pandas as pd
import matplotlib.pyplot as plt

# Example with FRED-MD data structure
# (In practice, load from FRED-MD database)
def load_fredmd_example():
    """
    Simulated FRED-MD style data.
    In practice, download from:
    https://research.stlouisfed.org/econ/mccracken/fred-databases/
    """
    np.random.seed(456)
    dates = pd.date_range('2000-01-01', periods=250, freq='MS')

    # Simulate realistic factor structure
    T = len(dates)
    factors = np.zeros((T, 3))
    factors[0] = np.random.randn(3)

    for t in range(1, T):
        factors[t] = 0.9 * factors[t-1] + 0.3 * np.random.randn(3)

    # Create variables with realistic loadings
    N = 30
    loadings = np.random.randn(N, 3)
    X = factors @ loadings.T + 0.5 * np.random.randn(T, N)

    # GDP growth depends on first factor
    gdp_growth = 2.0 + factors[:, 0] * 1.5 + 0.5 * np.random.randn(T)

    return pd.DataFrame(X, index=dates), pd.Series(gdp_growth, index=dates)

# Load data
X_df, gdp_growth = load_fredmd_example()

# Forecasting experiment
forecast_origin = '2018-01-01'
train_mask = X_df.index < forecast_origin
test_mask = X_df.index >= forecast_origin

X_train = X_df.loc[train_mask].values
y_train = gdp_growth.loc[train_mask].values
X_test = X_df.loc[test_mask].values
y_test = gdp_growth.loc[test_mask].values

# Compare models with different factor counts
results = []
for n_factors in range(1, 11):
    model = DiffusionIndexForecaster(n_factors=n_factors, horizon=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test[:len(y_test)-1])
    rmse = np.sqrt(np.mean((y_test[1:] - y_pred)**2))

    results.append({'n_factors': n_factors, 'RMSE': rmse})

results_df = pd.DataFrame(results)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Factor selection
axes[0].plot(results_df['n_factors'], results_df['RMSE'], marker='o')
axes[0].set_xlabel('Number of Factors')
axes[0].set_ylabel('Out-of-Sample RMSE')
axes[0].set_title('Model Selection: Optimal Factor Count')
axes[0].grid(True, alpha=0.3)

# Best model forecasts
best_r = results_df.loc[results_df['RMSE'].idxmin(), 'n_factors']
best_model = DiffusionIndexForecaster(n_factors=int(best_r), horizon=1)
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test[:len(y_test)-1])

axes[1].plot(X_df.index[train_mask][-50:], y_train[-50:],
             label='Training', color='blue', alpha=0.6)
test_dates = X_df.index[test_mask][1:]
axes[1].plot(test_dates, y_test[1:], label='Actual', color='black', marker='o')
axes[1].plot(test_dates, y_pred_best, label='Forecast',
             color='red', linestyle='--', marker='x')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('GDP Growth')
axes[1].set_title(f'GDP Growth Forecasts (r={int(best_r)} factors)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diffusion_index_forecasting.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Optimal number of factors: {int(best_r)}")
print(f"Best out-of-sample RMSE: {results_df['RMSE'].min():.4f}")
```

</div>

---

## 7. Common Pitfalls

### 1. In-Sample Factor Estimation

**Mistake:** Extracting factors from the full dataset (including test period) before forecasting.

**Problem:** This creates look-ahead bias. Factors at time $t$ would incorporate future information.

**Solution:** Use only training data to estimate factors. Transform test data using training-period loadings.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# WRONG
factors_all = PCA().fit_transform(X_all)  # Uses future data!

# CORRECT
pca = PCA().fit(X_train)
factors_train = pca.transform(X_train)
factors_test = pca.transform(X_test)  # Uses only training loadings
```

</div>

### 2. Ignoring Forecast Horizon

**Mistake:** Using the same factors for all horizons without adjustment.

**Problem:** Optimal factors may differ by horizon. Short-horizon forecasts need different information than long-horizon.

**Solution:** Either estimate horizon-specific factors or use a unified framework (FAVAR).

### 3. Overfitting on Factor Count

**Mistake:** Choosing $r$ to maximize in-sample fit.

**Problem:** More factors always improve in-sample fit but may overfit.

**Solution:** Use out-of-sample validation or information criteria (Bai-Ng) to select $r$.

### 4. Neglecting Data Transformations

**Mistake:** Using raw data with different units and stationarity properties.

**Problem:** Non-stationary variables dominate factor extraction. Units affect factor interpretation.

**Solution:** Transform all variables to stationarity (log-differences, differences) and standardize.

---

## 8. Connections

### Builds On
- **Static Factor Models:** Factors extracted via PCA
- **Principal Components:** Mathematical tool for factor estimation
- **Forecasting Theory:** Direct vs. iterated forecasts

### Leads To
- **FAVAR Models:** Combine factor forecasting with VAR dynamics
- **Structural Analysis:** Use factors to identify economic shocks
- **Real-Time Forecasting:** Handling ragged-edge data with factors

### Related Methods
- **Factor-Augmented Regression:** General framework (next guide)
- **Target Factors:** Boivin-Ng approach using target variable information
- **Sparse Factors:** LASSO-type selection of factor predictors

---

## 9. Practice Problems

### Conceptual

1. **Why do diffusion indices avoid overfitting despite using many predictors?**

2. **What is the economic interpretation of using the first principal component to forecast GDP?**

3. **How would you modify the diffusion index approach for real-time forecasting when different variables are released at different times?**

### Mathematical

4. **Show that the diffusion index forecast is equivalent to a restricted high-dimensional regression:**
   $$y_{t+h} = \alpha + \gamma' X_t + \varepsilon_{t+h}$$
   where $\gamma = \Lambda \beta$ (loadings × factor coefficients).

5. **Derive the forecast error variance for a diffusion index forecast, accounting for factor estimation error.**

6. **Prove that when $N, T \to \infty$ with $\sqrt{T}/N \to 0$, factor estimation error is asymptotically negligible.**

### Implementation

7. **Compare diffusion index forecasts with:**
   - Univariate AR model
   - Small VAR with selected predictors
   - Ridge regression with all predictors

   Which performs best for 1-quarter ahead GDP forecasts?

8. **Implement a bootstrap procedure to compute prediction intervals that account for factor estimation uncertainty.**

9. **Create a "factor rotation" experiment: Show that diffusion index forecasts are invariant to orthogonal rotations of factors.**

### Advanced

10. **Implement "targeted" diffusion index forecasting (Bai-Ng 2008) where factors are extracted to maximize correlation with the target variable.**

11. **Extend the implementation to handle mixed-frequency data (e.g., monthly predictors for quarterly GDP).**

---

<div class="callout-insight">

**Insight:** Understanding diffusion index forecasting is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## 10. Further Reading

### Foundational Papers

- **Stock, J.H. & Watson, M.W. (2002).** "Macroeconomic Forecasting Using Diffusion Indexes." *Journal of Business & Economic Statistics*, 20(2), 147-162.
  - Original diffusion index forecasting paper

- **Stock, J.H. & Watson, M.W. (2002).** "Forecasting Using Principal Components from a Large Number of Predictors." *Journal of the American Statistical Association*, 97(460), 1167-1179.
  - Theoretical foundations and empirical results

### Extensions

- **Bai, J. & Ng, S. (2008).** "Forecasting Economic Time Series Using Targeted Predictors." *Journal of Econometrics*, 146(2), 304-317.
  - Improving forecasts by targeting factors to the variable of interest

- **Banbura, M. & Modugno, M. (2014).** "Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data." *Journal of Applied Econometrics*, 29(1), 133-160.
  - Handling ragged-edge data in real-time forecasting

### Applied Studies

- **McCracken, M.W. & Ng, S. (2016).** "FRED-MD: A Monthly Database for Macroeconomic Research." *Journal of Business & Economic Statistics*, 34(4), 574-589.
  - Standard dataset for testing diffusion index methods

- **Stock, J.H. & Watson, M.W. (2012).** "Generalized Shrinkage Methods for Forecasting Using Many Predictors." *Journal of Business & Economic Statistics*, 30(4), 481-493.
  - Comparing factor methods with other high-dimensional approaches

### Books

- **Stock, J.H. & Watson, M.W. (2016).** "Dynamic Factor Models, Factor-Augmented Vector Autoregressions, and Structural Vector Autoregressions in Macroeconomics." *Handbook of Macroeconomics*, Vol 2, 415-525.
  - Comprehensive survey of factor-based forecasting methods

---

## Conceptual Practice Questions

1. What is a diffusion index and how does it relate to factor-augmented forecasting?

2. Why do diffusion index forecasts often outperform individual variable forecasts?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./01_diffusion_index_forecasting_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_diffusion_index_forecasting.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./02_favar_models.md">
  <div class="link-card-title">02 Favar Models</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_structural_identification.md">
  <div class="link-card-title">03 Structural Identification</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

