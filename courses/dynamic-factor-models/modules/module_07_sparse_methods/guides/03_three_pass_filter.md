# Three-Pass Regression Filter

## In Brief

The three-pass regression filter (3PRF), developed by Kelly and Pruitt (2015), is a sophisticated method for forecasting with many predictors that combines variable selection, factor extraction, and forecast aggregation. Unlike standard diffusion indices or targeted predictors, 3PRF explicitly accounts for the relationship between predictors and the target variable at each stage, providing a unified framework that often outperforms alternative methods.

## Key Insight

Standard factor methods extract factors from predictors $X$ without using target information, then regress $y$ on these factors. This two-step approach can miss predictors that matter for $y$ but don't load heavily on dominant factors. The three-pass filter innovates by using $y$ in all three passes: (1) identify predictors correlated with $y$, (2) extract latent factors from these relationships, (3) forecast using both factors and individual predictors, selecting the best combination.

---

## 1. The Three-Pass Philosophy

### Limitations of Existing Methods

**Standard diffusion index:**
- **Pass 1:** Extract factors from $X$ via PCA
- **Pass 2:** Regress $y$ on factors
- **Problem:** Factors chosen without considering $y$

**Targeted predictors (Bai-Ng):**
- **Pass 1:** Select predictors correlated with $y$
- **Pass 2:** Extract factors from selected predictors
- **Pass 3:** Regress $y$ on factors
- **Improvement:** Uses $y$ for selection, but not for factor extraction

### The 3PRF Innovation

**Key insight:** At each stage, explicitly model relationship between $X$ and $y$.

**Philosophy:** Don't just extract common variation in $X$. Extract variation in $X$ that predicts $y$.

**Result:** Factors are "supervised" - designed specifically for forecasting $y$.

---

## 2. The Three Passes: Formal Algorithm

### Pass 1: Univariate Forecasting Regressions

For each predictor $j = 1, ..., N$, run time-series regression:
$$y_{t+h} = \mu_j + \phi_j y_t + \beta_j X_{jt} + \varepsilon_{jt}$$

**Purpose:** Quantify each predictor's individual forecasting power.

**Output:** Coefficient estimates $\hat{\beta}_j$ and fitted values $\hat{y}_{jt} = \hat{\mu}_j + \hat{\phi}_j y_t + \hat{\beta}_j X_{jt}$.

**Interpretation:** $\hat{y}_{jt}$ is predictor $j$'s forecast of $y_{t+h}$.

### Pass 2: Cross-Sectional Factor Extraction

Organize forecasts from Pass 1 into matrix:
$$\hat{Y} = [\hat{y}_1, \hat{y}_2, ..., \hat{y}_N]$$

where $\hat{y}_j$ is $T \times 1$ vector of forecasts from predictor $j$.

**Extract factors from $\hat{Y}$ via PCA:**
$$\hat{Y} = \hat{G} \hat{H}' + \text{residuals}$$

where:
- $\hat{G}$: $T \times L$ matrix of "forecasting factors"
- $\hat{H}$: $N \times L$ matrix of loadings
- $L$: number of factors (typically small, e.g., 2-5)

**Purpose:** Find common patterns across individual forecasts.

**Intuition:** If multiple predictors forecast $y$ similarly, they share an underlying factor.

### Pass 3: Forecast Combination

Use extracted factors $\hat{G}$ as predictors in final forecasting regression:
$$y_{t+h} = \alpha + \gamma' \hat{G}_t + u_t$$

Estimate $\hat{\gamma}$ by OLS.

**Final forecast:**
$$\hat{y}_{T+h|T} = \hat{\alpha} + \hat{\gamma}' \hat{G}_T$$

**Alternative (full model):** Include both factors and original predictors:
$$y_{t+h} = \alpha + \gamma' \hat{G}_t + \sum_{j \in S} \delta_j X_{jt} + u_t$$

where $S$ is selected subset (e.g., via LASSO on residuals).

---

## 3. Mathematical Formulation

### Formal Statement of Algorithm

**Input:**
- Predictors: $X = (X_1, ..., X_N)$, each $X_j$ is $T \times 1$
- Target: $y$ ($T \times 1$)
- Forecast horizon: $h$

**Pass 1:** For $j = 1, ..., N$:
$$\hat{\beta}_j = \arg\min_{\beta_j} \sum_{t=1}^{T-h} (y_{t+h} - \mu_j - \phi_j y_t - \beta_j X_{jt})^2$$

Compute fitted values: $\hat{Y}_{tj} = \hat{\mu}_j + \hat{\phi}_j y_t + \hat{\beta}_j X_{jt}$ for $t = 1, ..., T-h$.

**Pass 2:** Stack forecasts into matrix $\hat{Y}$ ($(T-h) \times N$).

Perform PCA:
$$\hat{G} = \sqrt{T-h} \cdot \text{eigenvectors}_{1:L}(\hat{Y}'\hat{Y}/(T-h))$$

Equivalently: $\hat{G}$ are first $L$ principal components of $\hat{Y}$.

**Pass 3:**
$$(\hat{\alpha}, \hat{\gamma}) = \arg\min_{\alpha, \gamma} \sum_{t=1}^{T-h} (y_{t+h} - \alpha - \gamma' \hat{G}_t)^2$$

### Theoretical Justification

**Approximating factor model:**

Suppose true model is:
$$y_{t+h} = F_t' \beta + u_t$$
$$X_{jt} = F_t' \lambda_j + e_{jt}$$

where $F_t$ are latent factors.

**Pass 1 estimates:**
$$\hat{y}_{jt} \approx E[y_{t+h} | X_{jt}] \approx \lambda_j' F_t$$

**Pass 2 extracts:** Factors from $\hat{Y}$ recover space spanned by $F_t' \Lambda$ where $\Lambda = (\lambda_1, ..., \lambda_N)$.

**Pass 3 projects:** $y_{t+h}$ onto estimated factor space.

**Result:** Consistent estimation of forecast function under regularity conditions.

---

## 4. Intuitive Explanation

### The Three-Pass Journey

**Think of forecasting $y$ with 100 predictors:**

**Pass 1: Individual conversations**
- Ask each predictor separately: "What's your forecast for $y$?"
- Each gives an answer (fitted value from univariate regression)
- Some predictors give similar forecasts, others differ

**Pass 2: Finding consensus groups**
- Notice some predictors cluster in their forecasts
- Group 1: "We all predict recession" (financial indicators)
- Group 2: "We all predict expansion" (employment indicators)
- Group 3: "We see inflation rising" (price indicators)
- Extract these common "narratives" as factors

**Pass 3: Final forecast**
- Combine the group narratives optimally
- Weight each narrative by its historical predictive power

### Why This Beats Alternatives

**vs Standard PCA:**
- Standard PCA: "What varies most in $X$?"
- 3PRF: "What common patterns exist in predictions of $y$?"

**vs Targeted Predictors:**
- Targeted: "Which $X_j$ correlate with $y$? Extract factors from those."
- 3PRF: "What do forecasts from different $X_j$ have in common?"

**Key difference:** 3PRF factors are extracted from forecast space, not predictor space.

### Visual Intuition

```
Pass 1: Individual Forecasts
X₁ → Regression → ŷ₁
X₂ → Regression → ŷ₂
⋮
X_N → Regression → ŷ_N

Pass 2: Factor Extraction
[ŷ₁, ŷ₂, ..., ŷ_N] → PCA → Factors [G₁, G₂, G₃]

Pass 3: Forecast Aggregation
[G₁, G₂, G₃] → Regression → Final forecast ŷ_{T+h}
```

---

## 5. Code Implementation

### Complete Three-Pass Regression Filter

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats

class ThreePassRegressionFilter:
    """
    Three-Pass Regression Filter (Kelly-Pruitt 2015).

    Parameters
    ----------
    n_factors : int
        Number of latent factors to extract (L)
    horizon : int
        Forecast horizon (h periods ahead)
    include_ar : bool
        Include AR term in Pass 1 regressions
    standardize : bool
        Standardize predictors before Pass 1
    """

    def __init__(self, n_factors=3, horizon=1, include_ar=True,
                 standardize=True):
        self.n_factors = n_factors
        self.horizon = horizon
        self.include_ar = include_ar
        self.standardize = standardize

        self.scaler = StandardScaler() if standardize else None
        self.pass1_models_ = []
        self.pass1_forecasts_ = None
        self.pass2_pca_ = None
        self.factors_ = None
        self.pass3_model_ = None
        self.predictor_importance_ = None

    def fit(self, X, y):
        """
        Fit three-pass regression filter.

        Parameters
        ----------
        X : array-like, shape (T, N)
            Predictor panel
        y : array-like, shape (T,)
            Target variable

        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        T, N = X.shape

        # Standardize predictors
        if self.standardize:
            X = self.scaler.fit_transform(X)

        # Effective sample size for h-step ahead forecast
        T_eff = T - self.horizon

        # ============================================================
        # PASS 1: Univariate forecasting regressions
        # ============================================================
        print(f"Pass 1: Running {N} univariate regressions...")

        pass1_forecasts = np.zeros((T_eff, N))
        self.pass1_models_ = []

        for j in range(N):
            # Prepare data for h-step ahead regression
            X_j = X[:T_eff, j].reshape(-1, 1)
            y_ahead = y[self.horizon:T_eff + self.horizon]

            if self.include_ar:
                # Include lagged y
                y_lag = y[:T_eff].reshape(-1, 1)
                X_aug = np.column_stack([y_lag, X_j])
            else:
                X_aug = X_j

            # Fit regression
            model_j = LinearRegression()
            model_j.fit(X_aug, y_ahead)

            # Store model and fitted values
            self.pass1_models_.append(model_j)
            pass1_forecasts[:, j] = model_j.predict(X_aug)

        self.pass1_forecasts_ = pass1_forecasts

        # ============================================================
        # PASS 2: Cross-sectional factor extraction
        # ============================================================
        print(f"Pass 2: Extracting {self.n_factors} latent factors...")

        # PCA on forecast matrix
        self.pass2_pca_ = PCA(n_components=self.n_factors)
        self.factors_ = self.pass2_pca_.fit_transform(pass1_forecasts)

        # ============================================================
        # PASS 3: Forecast combination regression
        # ============================================================
        print(f"Pass 3: Final forecasting regression...")

        # Target for Pass 3 (same as Pass 1)
        y_ahead = y[self.horizon:T_eff + self.horizon]

        # Fit on factors
        self.pass3_model_ = LinearRegression()
        self.pass3_model_.fit(self.factors_, y_ahead)

        # Compute predictor importance (based on Pass 1 coefficients)
        self._compute_importance(X, y)

        print("Three-pass filter fitted successfully.")
        return self

    def predict(self, X, y_hist=None):
        """
        Generate h-step ahead forecasts.

        Parameters
        ----------
        X : array-like, shape (T_new, N)
            New predictor data
        y_hist : array-like, optional
            Historical y values (needed if include_ar=True)

        Returns
        -------
        forecasts : array
            h-step ahead forecasts
        """
        X = np.asarray(X)
        T_new = X.shape[0]

        # Standardize
        if self.standardize:
            X = self.scaler.transform(X)

        # Pass 1: Generate forecasts from each predictor
        pass1_forecasts_new = np.zeros((T_new, len(self.pass1_models_)))

        for j, model_j in enumerate(self.pass1_models_):
            X_j = X[:, j].reshape(-1, 1)

            if self.include_ar:
                if y_hist is None:
                    raise ValueError("y_hist required when include_ar=True")
                # Use provided historical y
                y_lag = y_hist[:T_new].reshape(-1, 1)
                X_aug = np.column_stack([y_lag, X_j])
            else:
                X_aug = X_j

            pass1_forecasts_new[:, j] = model_j.predict(X_aug)

        # Pass 2: Transform forecasts to factors
        factors_new = self.pass2_pca_.transform(pass1_forecasts_new)

        # Pass 3: Generate final forecast
        forecasts = self.pass3_model_.predict(factors_new)

        return forecasts

    def forecast_one_step(self, X_T, y_hist=None):
        """
        Forecast single period from most recent data.

        Parameters
        ----------
        X_T : array-like, shape (N,)
            Most recent predictor values
        y_hist : float or array, optional
            Recent y value(s) for AR term

        Returns
        -------
        forecast : float
            h-step ahead point forecast
        """
        X_T = np.asarray(X_T).reshape(1, -1)

        if self.include_ar and y_hist is not None:
            y_hist = np.atleast_1d(y_hist).reshape(1, -1)

        forecast = self.predict(X_T, y_hist)
        return forecast[0]

    def _compute_importance(self, X, y):
        """Compute predictor importance scores."""
        # Importance based on absolute Pass 1 coefficients
        importance = np.zeros(len(self.pass1_models_))

        for j, model_j in enumerate(self.pass1_models_):
            # Extract coefficient on X_j (last coefficient)
            coef_j = model_j.coef_[-1] if len(model_j.coef_) > 1 else model_j.coef_[0]
            importance[j] = abs(coef_j)

        self.predictor_importance_ = importance / importance.sum()

    def get_factor_summary(self):
        """
        Summary of extracted factors.

        Returns
        -------
        summary : dict
            Factor variance and loadings
        """
        var_explained = self.pass2_pca_.explained_variance_ratio_

        summary = {
            'n_factors': self.n_factors,
            'variance_explained': var_explained,
            'cumulative_variance': var_explained.cumsum(),
            'loadings': self.pass2_pca_.components_
        }

        return summary

    def get_predictor_importance(self, feature_names=None, top_k=10):
        """
        Get most important predictors.

        Parameters
        ----------
        feature_names : list, optional
            Predictor names
        top_k : int
            Number of top predictors to return

        Returns
        -------
        importance_df : DataFrame
            Top predictors by importance
        """
        N = len(self.predictor_importance_)

        if feature_names is None:
            feature_names = [f"X{i}" for i in range(N)]

        importance_df = pd.DataFrame({
            'predictor': feature_names,
            'importance': self.predictor_importance_
        })

        importance_df = importance_df.sort_values('importance', ascending=False)
        return importance_df.head(top_k)

    def plot_results(self, y_true=None, y_pred=None):
        """
        Visualize three-pass filter results.

        Parameters
        ----------
        y_true : array, optional
            True target values
        y_pred : array, optional
            Predicted values
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Factor variance explained
        var_exp = self.pass2_pca_.explained_variance_ratio_
        axes[0, 0].bar(range(1, len(var_exp) + 1), var_exp,
                      alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Factor Number', fontsize=11)
        axes[0, 0].set_ylabel('Variance Explained', fontsize=11)
        axes[0, 0].set_title('Pass 2: Factor Variance Explained',
                            fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # 2. Pass 3 coefficients
        pass3_coef = self.pass3_model_.coef_
        axes[0, 1].bar(range(1, len(pass3_coef) + 1), pass3_coef,
                      alpha=0.7, edgecolor='black')
        axes[0, 1].axhline(0, color='black', linewidth=0.5)
        axes[0, 1].set_xlabel('Factor Number', fontsize=11)
        axes[0, 1].set_ylabel('Forecast Coefficient', fontsize=11)
        axes[0, 1].set_title('Pass 3: Final Regression Coefficients',
                            fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # 3. Predictor importance
        top_10 = self.get_predictor_importance(top_k=10)
        axes[1, 0].barh(range(len(top_10)), top_10['importance'].values,
                       alpha=0.7, edgecolor='black')
        axes[1, 0].set_yticks(range(len(top_10)))
        axes[1, 0].set_yticklabels(top_10['predictor'].values)
        axes[1, 0].set_xlabel('Importance Score', fontsize=11)
        axes[1, 0].set_title('Top 10 Predictors by Importance',
                            fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        axes[1, 0].invert_yaxis()

        # 4. Forecast accuracy (if provided)
        if y_true is not None and y_pred is not None:
            axes[1, 1].scatter(y_true, y_pred, alpha=0.6, edgecolor='black')

            # Add 45-degree line
            lims = [min(y_true.min(), y_pred.min()),
                   max(y_true.max(), y_pred.max())]
            axes[1, 1].plot(lims, lims, 'r--', linewidth=2, label='Perfect forecast')

            # R-squared
            r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2)

            axes[1, 1].set_xlabel('Actual', fontsize=11)
            axes[1, 1].set_ylabel('Predicted', fontsize=11)
            axes[1, 1].set_title(f'Forecast Accuracy (R² = {r2:.3f})',
                                fontsize=12, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Provide y_true and y_pred\nfor accuracy plot',
                           ha='center', va='center', fontsize=12,
                           transform=axes[1, 1].transAxes)
            axes[1, 1].set_xticks([])
            axes[1, 1].set_yticks([])

        plt.tight_layout()
        return fig, axes
```

### Example Application

```python
# Generate realistic data
np.random.seed(456)
T, N = 250, 40
L_true = 3

# True latent factors (persistent)
F_true = np.zeros((T, L_true))
F_true[0] = np.random.randn(L_true)
for t in range(1, T):
    F_true[t] = 0.8 * F_true[t-1] + 0.4 * np.random.randn(L_true)

# Loadings with sparsity (some predictors don't load on some factors)
Lambda = np.random.randn(N, L_true) * np.random.binomial(1, 0.7, (N, L_true))
Lambda = Lambda / np.sqrt(L_true)

# Predictors
X = F_true @ Lambda.T + 0.5 * np.random.randn(T, N)

# Target: depends on factors with different weights
beta_true = np.array([1.0, 0.5, -0.3])
y = F_true @ beta_true + 0.8 * np.random.randn(T)

# Add AR(1) structure to y
for t in range(1, T):
    y[t] += 0.3 * y[t-1]

# Split data
T_train = 200
X_train, X_test = X[:T_train], X[T_train:]
y_train, y_test = y[:T_train], y[T_train:]

# Fit three-pass filter
tprf = ThreePassRegressionFilter(n_factors=5, horizon=1,
                                 include_ar=True, standardize=True)
tprf.fit(X_train, y_train)

# Generate forecasts
# For out-of-sample, need y history for AR term
y_hist_test = y[:T_train + len(X_test) - 1]
y_pred = tprf.predict(X_test[:-1], y_hist=y_hist_test[T_train-1:])

# Evaluation
mse = np.mean((y_test[1:] - y_pred)**2)
mae = np.mean(np.abs(y_test[1:] - y_pred))
r2 = 1 - np.sum((y_test[1:] - y_pred)**2) / np.sum((y_test[1:] - y_test[1:].mean())**2)

print("=" * 60)
print("THREE-PASS REGRESSION FILTER RESULTS")
print("=" * 60)
print(f"Data: T={T}, N={N}, L_true={L_true}")
print(f"Model: L={tprf.n_factors} factors, h={tprf.horizon} horizon")
print()
print("Out-of-Sample Performance:")
print(f"  RMSE: {np.sqrt(mse):.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  R²:   {r2:.4f}")
print()
print("Factor Summary:")
summary = tprf.get_factor_summary()
for i, (var, cum_var) in enumerate(zip(summary['variance_explained'],
                                        summary['cumulative_variance']), 1):
    print(f"  Factor {i}: {var:.3f} variance ({cum_var:.3f} cumulative)")
print()
print("Top 5 Predictors:")
print(tprf.get_predictor_importance(top_k=5))
print("=" * 60)

# Visualization
tprf.plot_results(y_true=y_test[1:], y_pred=y_pred)
plt.savefig('three_pass_filter_results.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Comparison: 3PRF vs Standard Methods

```python
def compare_methods(X_train, y_train, X_test, y_test, n_factors=5, horizon=1):
    """
    Compare three-pass filter with alternative methods.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression, LassoCV

    results = {}

    # Method 1: Three-pass filter
    tprf = ThreePassRegressionFilter(n_factors=n_factors, horizon=horizon,
                                     include_ar=True)
    tprf.fit(X_train, y_train)
    y_hist_test = np.concatenate([y_train, y_test])[:-1]
    y_pred_tprf = tprf.predict(X_test[:-horizon],
                               y_hist=y_hist_test[len(y_train)-1:])
    results['3PRF'] = np.sqrt(np.mean((y_test[horizon:] - y_pred_tprf)**2))

    # Method 2: Standard diffusion index
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=n_factors)
    F_train = pca.fit_transform(X_train_scaled)
    F_test = pca.transform(X_test_scaled)

    # Add AR term
    y_lag_train = y_train[:-horizon].reshape(-1, 1)
    F_train_aug = np.column_stack([y_lag_train, F_train[:-horizon]])

    reg_di = LinearRegression()
    reg_di.fit(F_train_aug, y_train[horizon:])

    y_lag_test = y_test[:-horizon].reshape(-1, 1)
    F_test_aug = np.column_stack([y_lag_test, F_test[:-horizon]])
    y_pred_di = reg_di.predict(F_test_aug)

    results['Diffusion Index'] = np.sqrt(np.mean((y_test[horizon:] - y_pred_di)**2))

    # Method 3: LASSO
    X_train_aug = np.column_stack([y_train[:-horizon].reshape(-1, 1),
                                   X_train_scaled[:-horizon]])
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_train_aug, y_train[horizon:])

    X_test_aug = np.column_stack([y_test[:-horizon].reshape(-1, 1),
                                  X_test_scaled[:-horizon]])
    y_pred_lasso = lasso.predict(X_test_aug)

    results['LASSO'] = np.sqrt(np.mean((y_test[horizon:] - y_pred_lasso)**2))

    # Method 4: Simple AR
    from statsmodels.tsa.ar_model import AutoReg
    ar_model = AutoReg(y_train, lags=1).fit()
    y_pred_ar = ar_model.predict(start=len(y_train),
                                  end=len(y_train) + len(y_test) - 1 - horizon)
    results['AR(1)'] = np.sqrt(np.mean((y_test[horizon:] - y_pred_ar)**2))

    return results

# Run comparison
comparison = compare_methods(X_train, y_train, X_test, y_test,
                            n_factors=5, horizon=1)

print("\nMethod Comparison (RMSE):")
print("-" * 40)
for method, rmse in sorted(comparison.items(), key=lambda x: x[1]):
    print(f"{method:20s}: {rmse:.4f}")
print("-" * 40)

# Relative performance
best_rmse = min(comparison.values())
print("\nRelative Performance:")
for method, rmse in comparison.items():
    rel_perf = (rmse - best_rmse) / best_rmse * 100
    print(f"{method:20s}: +{rel_perf:.1f}% vs best")
```

---

## 6. Common Pitfalls

### 1. Using Too Many Factors

**Mistake:** Setting $L$ (number of factors) very high to "capture more information."

**Problem:** Overfitting. Pass 2 extracts noise from Pass 1 forecasts.

**Solution:** Use cross-validation or information criteria. Typically $L = 2-5$ works well.

### 2. Forgetting the Horizon Alignment

**Mistake:** In Pass 1, regressing $y_t$ on $X_t$ when forecasting $h$ steps ahead.

**Problem:** Model doesn't match forecasting task.

**Solution:** Always regress $y_{t+h}$ on $X_t$ (and $y_t$ if using AR term).

```python
# WRONG for h-step forecast
y_ahead = y[:-h]  # Off by one!

# CORRECT
y_ahead = y[h:T_eff + h]  # Properly aligned
```

### 3. Not Including AR Terms

**Mistake:** Omitting lagged $y$ from Pass 1 regressions.

**Problem:** Miss important autoregressive information, especially for persistent series.

**Solution:** Include AR terms in Pass 1 (set `include_ar=True`).

### 4. Inconsistent Standardization

**Mistake:** Standardizing in Pass 1 but not in prediction.

**Problem:** Scale mismatch, incorrect forecasts.

**Solution:** Apply same transformations in fit and predict.

---

## 7. Connections

### Builds On
- **Diffusion Index Forecasting:** Enhanced version with target information
- **Targeted Predictors:** Shares philosophy of using target info
- **Forecast Combination:** Optimally combines individual forecasts

### Leads To
- **Supervised PCA:** General framework for target-informed dimension reduction
- **Factor-Augmented Forecasting:** Extensions with dynamic factors
- **Ensemble Methods:** Related forecast aggregation approaches

### Related Methods
- **Partial Least Squares:** Also extracts predictive components
- **Boosting:** Sequential version of forecast combination
- **Complete Subset Regressions:** Alternative forecast combination

---

## 8. Practice Problems

### Conceptual

1. **Why does extracting factors from forecasts (Pass 2) differ from extracting factors from predictors?** What information is gained?

2. **Can the three-pass filter perform worse than standard diffusion index?** Under what conditions?

3. **How does 3PRF handle multicollinearity** among predictors?

### Mathematical

4. **Show that Pass 1 fitted values** $\hat{y}_{jt}$ are projections of $y_{t+h}$ onto the space spanned by $(1, y_t, X_{jt})$.

5. **Prove that if all Pass 1 forecasts are identical,** Pass 2 extracts only one factor (rank-one $\hat{Y}$ matrix).

6. **Derive the relationship between Pass 3 coefficients** and the "consensus" among Pass 1 forecasts.

### Implementation

7. **Implement "adaptive 3PRF"** that selects $L$ (number of factors) via cross-validation automatically.

8. **Extend 3PRF to handle missing data** in the predictor panel $X$ during Pass 1.

9. **Create a "sparse 3PRF"** that uses LASSO in Pass 3 instead of OLS to select among factors.

### Advanced

10. **Implement bootstrap prediction intervals** for 3PRF forecasts that account for all three passes of estimation uncertainty.

11. **Compare 3PRF with PLS (partial least squares).** Are they extracting similar factors? Forecast performance?

12. **Extend to multi-horizon forecasting:** Estimate separate 3PRF models for $h=1, 2, ..., 12$ and compare with iterated one-step forecasts.

---

## 9. Further Reading

### Foundational Paper

- **Kelly, B. & Pruitt, S. (2015).** "The Three-Pass Regression Filter: A New Approach to Forecasting Using Many Predictors." *Journal of Econometrics*, 186(2), 294-316.
  - Original three-pass filter paper

- **Kelly, B. & Pruitt, S. (2013).** "Market Expectations in the Cross-Section of Present Values." *Journal of Finance*, 68(5), 1721-1756.
  - Application to asset pricing

### Extensions and Applications

- **Giglio, S., Kelly, B., & Pruitt, S. (2016).** "Systemic Risk and the Macroeconomy: An Empirical Evaluation." *Journal of Financial Economics*, 119(3), 457-471.
  - Application to systemic risk measurement

- **Choi, H. & Xie, Y. (2020).** "Three-Pass Regression Filter for Gross-Output Production Functions and Productivity." *Journal of Applied Econometrics*, 35(4), 508-527.
  - Application to productivity analysis

### Methodology Comparisons

- **Ng, S. (2013).** "Variable Selection in Predictive Regressions." In *Handbook of Economic Forecasting*, Vol 2, 752-789.
  - Survey comparing 3PRF with alternatives

- **Smeekes, S. & Wijler, E. (2021).** "Macroeconomic Forecasting Using Penalized Regression Methods." *International Journal of Forecasting*, 37(4), 1569-1585.
  - Empirical comparison with LASSO, elastic net, etc.

### Theoretical Foundations

- **Bai, J. & Ng, S. (2006).** "Confidence Intervals for Diffusion Index Forecasts and Inference for Factor-Augmented Regressions." *Econometrica*, 74(4), 1133-1150.
  - Theory of factor-based forecasting

- **Stock, J.H. & Watson, M.W. (2016).** "Dynamic Factor Models, Factor-Augmented Vector Autoregressions, and Structural Vector Autoregressions in Macroeconomics." *Handbook of Macroeconomics*, Vol 2, 415-525.
  - Comprehensive survey of factor methods

### Software

- **Python Implementation:** Kelly-Pruitt provide MATLAB code; Python adaptations available in various econometrics packages
- **R Package:** `tfr` implements three-pass regression filter
