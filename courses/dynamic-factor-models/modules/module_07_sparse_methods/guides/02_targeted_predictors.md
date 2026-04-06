# Targeted Predictors: Bai-Ng Factor Selection

> **Reading time:** ~15 min | **Module:** Module 7: Sparse Methods | **Prerequisites:** Modules 0-6

<div class="callout-key">

**Key Concept Summary:** Targeted predictors, developed by Bai and Ng (2008), improve forecasting by extracting factors specifically designed to predict a target variable, rather than merely summarizing variation in the predictor panel. This approach combines principal components analysis with variable selection, using s...

</div>

## In Brief

Targeted predictors, developed by Bai and Ng (2008), improve forecasting by extracting factors specifically designed to predict a target variable, rather than merely summarizing variation in the predictor panel. This approach combines principal components analysis with variable selection, using soft or hard thresholding to identify predictors most relevant for forecasting the target.

<div class="callout-insight">

**Insight:** Standard PCA extracts factors that explain maximum variance in predictors $X$, but this doesn't guarantee good prediction of target $y$. Targeted factor methods first screen predictors based on their individual correlation with $y$, then extract factors from this reduced set. This focuses factor extraction on predictors that matter for the forecasting task, often dramatically improving forecast accuracy.

</div>
---

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## 1. The Targeting Problem

### Motivation: When PCA Factors Fail

**Standard diffusion index approach:**
1. Extract factors $\hat{F}$ from all predictors $X$ via PCA
2. Forecast: $y_{t+h} = \alpha + \beta' \hat{F}_t + \varepsilon_{t+h}$

**Potential problem:**
- PCA finds factors explaining variance in $X$
- High-variance predictors dominate factors
- These predictors may be irrelevant for $y$

**Example:** Forecasting inflation
- 100 predictors: employment, output, prices, surveys, financial vars
- Employment variables highly correlated, high variance
- PCA extracts "employment factor" as Factor 1
- But inflation may depend more on price/financial variables

**Result:** First few PC's may have low predictive power for $y$.

### The Targeting Solution

**Key idea:** Pre-select predictors based on relationship with target $y$ before extracting factors.

**Intuition:** If predictor $X_j$ has weak correlation with $y$, it's unlikely to be useful for forecasting $y$, regardless of its variance or correlation with other predictors.

**Bai-Ng insight:** Combine variable screening with factor extraction to get "targeted factors" optimized for forecasting.

---

## 2. Mathematical Formulation

### Marginal Correlation Screening

For each predictor $j = 1, ..., N$, compute marginal correlation with target:
$$\hat{\rho}_j = \text{Corr}(X_{jt}, y_t)$$

or for forecasting at horizon $h$:
$$\hat{\rho}_j = \text{Corr}(X_{jt}, y_{t+h})$$

**Screening step:** Select predictors with $|\hat{\rho}_j|$ above threshold.

### Hard Thresholding

**Hard thresholding** selects predictors exceeding absolute threshold $c$:
$$\mathcal{S}_{\text{hard}} = \{j : |\hat{\rho}_j| > c\}$$

**Algorithm:**
1. Compute $\hat{\rho}_j$ for all $j$
2. Set threshold $c$ (e.g., via cross-validation)
3. Keep predictors in $\mathcal{S}_{\text{hard}}$
4. Extract $r$ factors from selected predictors via PCA
5. Use these "targeted factors" for forecasting

**Choice of $c$:**
- Too high: Miss important predictors
- Too low: Include too many, lose benefits of targeting
- Typical: Select top 20-50% of predictors

### Soft Thresholding

**Soft thresholding** weights predictors by correlation strength:
$$w_j = \frac{|\hat{\rho}_j|^{\gamma}}{\sum_{k=1}^N |\hat{\rho}_k|^{\gamma}}$$

where $\gamma \geq 0$ controls concentration.

**Weighted PCA:**
- Transform: $\tilde{X}_{jt} = \sqrt{w_j} \cdot X_{jt}$
- Extract factors from $\tilde{X}$ via standard PCA
- Factors emphasize high-correlation predictors

**Special cases:**
- $\gamma = 0$: Equal weights (standard PCA)
- $\gamma = 1$: Linear weighting by correlation
- $\gamma \to \infty$: Approaches hard thresholding

**Advantages over hard thresholding:**
- Smooth function of data (more stable)
- Partially uses all information
- Less sensitive to threshold choice

---

## 3. Theoretical Properties

### Why Targeting Helps

Consider population factor model:
$$X_{jt} = \lambda_j' F_t + e_{jt}$$

And target variable:
$$y_t = \beta' F_t + u_t$$

**If:** Some components of $F_t$ are irrelevant for $y$ (corresponding $\beta$ elements are zero)

**Then:** Predictors loading primarily on irrelevant factors have $\text{Cov}(X_j, y) \approx 0$.

**Targeting removes these predictors** and focuses on factors relevant for $y$.

### Consistency Results (Bai-Ng 2008)

Under regularity conditions:

**1. Factor Estimation:**
Targeted factors $\hat{F}^{\text{target}}$ consistently estimate the space spanned by true factors relevant for $y$.

**2. Forecast Performance:**
With appropriate threshold choice:
$$\text{MSE}_{\text{target}} \leq \text{MSE}_{\text{standard}} + o_p(1)$$

Targeting never hurts asymptotically and often helps substantially.

**3. Optimal Threshold:**
Exists optimal threshold $c^*$ balancing:
- Bias: Too few predictors miss information
- Variance: Too many predictors add noise

For $\gamma$ in soft thresholding:
- $\gamma \in [1, 2]$ typically works well empirically
- Theory suggests $\gamma$ should grow with sample size slowly

---

## 4. Intuitive Explanation

### The Targeting Procedure

**Step-by-step intuition:**

**Step 1: Individual screening**
- Test each predictor separately: Does $X_j$ correlate with $y$?
- Like interviewing candidates individually before group assessment

**Step 2: Selection**
- Keep predictors passing correlation test
- Discard weak predictors (won't help forecast anyway)

**Step 3: Factor extraction**
- Extract factors from selected predictors only
- Factors now focus on relevant variation

**Step 4: Forecasting**
- Use targeted factors exactly like standard diffusion index

### Visual Intuition

```

All Predictors (N=100)
     |
     | Marginal Correlation Screening
     v
Relevant Predictors (N*=30)
     |
     | PCA on Reduced Set
     v
Targeted Factors (r=5)
     |
     | Forecast Regression
     v
  Prediction ŷ_{t+h}
```

Compare to standard approach:
```

All Predictors (N=100)
     |
     | PCA on Full Set
     v
Standard Factors (r=5)
     |
     | Forecast Regression
     v
  Prediction ŷ_{t+h}
```

**Key difference:** Factor extraction focuses on relevant predictors.

### When Does Targeting Help Most?

**Large improvements when:**
- Many predictors unrelated to target
- High-variance predictors are irrelevant
- True factor structure is sparse for target

**Minimal improvement when:**
- All predictors relevant for target
- Factors explaining $X$ variance also predict $y$ well
- Small $N$ (limited scope for selection)

---

## 5. Code Implementation

### Basic Targeted Predictors Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">targetedpredictors.py</span>
</div>

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

class TargetedPredictors:
    """
    Targeted predictor extraction (Bai-Ng 2008).

    Parameters
    ----------
    n_factors : int
        Number of targeted factors to extract
    threshold_type : str
        'hard' or 'soft' thresholding
    threshold : float
        Threshold for hard thresholding (correlation cutoff)
        For soft: None means auto-select gamma
    gamma : float
        Exponent for soft thresholding weights (if soft)
    horizon : int
        Forecast horizon for correlation computation
    standardize : bool
        Whether to standardize predictors
    """

    def __init__(self, n_factors=5, threshold_type='hard', threshold=None,
                 gamma=1.0, horizon=1, standardize=True):
        self.n_factors = n_factors
        self.threshold_type = threshold_type
        self.threshold = threshold
        self.gamma = gamma
        self.horizon = horizon
        self.standardize = standardize

        self.scaler = StandardScaler() if standardize else None
        self.pca = PCA(n_components=n_factors)
        self.reg = LinearRegression()

        self.correlations_ = None
        self.selected_predictors_ = None
        self.weights_ = None
        self.targeted_factors_ = None
        self.forecast_coef_ = None

    def fit(self, X, y):
        """
        Fit targeted predictor model.

        Parameters
        ----------
        X : array-like, shape (T, N)
            Predictor panel
        y : array-like, shape (T,)
            Target variable
        """
        X = np.asarray(X)
        y = np.asarray(y)
        T, N = X.shape

        # Standardize predictors
        if self.standardize:
            X = self.scaler.fit_transform(X)

        # Step 1: Compute marginal correlations with target
        # Account for forecast horizon
        T_corr = T - self.horizon
        self.correlations_ = np.zeros(N)

        for j in range(N):
            # Correlation between X_t and y_{t+h}
            x_vals = X[:T_corr, j]
            y_vals = y[self.horizon:T_corr + self.horizon]

            if len(x_vals) > 1:
                self.correlations_[j], _ = pearsonr(x_vals, y_vals)
            else:
                self.correlations_[j] = 0

        # Handle NaN correlations (constant predictors)
        self.correlations_ = np.nan_to_num(self.correlations_, 0)

        # Step 2: Screening/Weighting
        if self.threshold_type == 'hard':
            # Hard thresholding
            if self.threshold is None:
                # Auto-select threshold as median absolute correlation
                self.threshold = np.median(np.abs(self.correlations_))

            self.selected_predictors_ = np.where(
                np.abs(self.correlations_) >= self.threshold
            )[0]

            if len(self.selected_predictors_) < self.n_factors:
                # Ensure enough predictors for requested factors
                sorted_idx = np.argsort(np.abs(self.correlations_))[::-1]
                self.selected_predictors_ = sorted_idx[:max(self.n_factors, 10)]

            # Extract from selected predictors
            X_selected = X[:, self.selected_predictors_]
            self.weights_ = None

        else:  # soft thresholding
            # Compute weights
            abs_corr = np.abs(self.correlations_)
            weights = abs_corr ** self.gamma
            weights = weights / weights.sum()
            self.weights_ = weights

            # Weight predictors
            X_selected = X * np.sqrt(weights)
            self.selected_predictors_ = np.arange(N)

        # Step 3: Extract targeted factors
        self.targeted_factors_ = self.pca.fit_transform(X_selected)

        # Step 4: Forecast regression with targeted factors
        T_forecast = T - self.horizon
        F_t = self.targeted_factors_[:T_forecast, :]
        y_ahead = y[self.horizon:T_forecast + self.horizon]

        self.reg.fit(F_t, y_ahead)
        self.forecast_coef_ = self.reg.coef_
        self.forecast_intercept_ = self.reg.intercept_

        return self

    def predict(self, X):
        """
        Generate h-step ahead forecasts.

        Parameters
        ----------
        X : array-like, shape (T_new, N)
            New predictor data

        Returns
        -------
        forecasts : array
            h-step ahead forecasts
        """
        X = np.asarray(X)

        # Standardize
        if self.standardize:
            X = self.scaler.transform(X)

        # Apply targeting
        if self.threshold_type == 'hard':
            X_selected = X[:, self.selected_predictors_]
        else:
            X_selected = X * np.sqrt(self.weights_)

        # Transform to factors
        factors = self.pca.transform(X_selected)

        # Generate forecast
        forecasts = self.reg.predict(factors)

        return forecasts

    def get_targeting_summary(self, feature_names=None):
        """
        Summarize targeting results.

        Parameters
        ----------
        feature_names : list, optional
            Names of features

        Returns
        -------
        summary_df : DataFrame
            Targeting summary
        """
        N = len(self.correlations_)

        if feature_names is None:
            feature_names = [f"X{i}" for i in range(N)]

        summary = pd.DataFrame({
            'feature': feature_names,
            'correlation': self.correlations_,
            'abs_correlation': np.abs(self.correlations_)
        })

        if self.threshold_type == 'hard':
            summary['selected'] = np.isin(np.arange(N), self.selected_predictors_)
        else:
            summary['weight'] = self.weights_

        summary = summary.sort_values('abs_correlation', ascending=False)

        return summary

    def plot_targeting_results(self, feature_names=None):
        """
        Visualize targeting results.

        Parameters
        ----------
        feature_names : list, optional
            Feature names for labeling
        """
        summary = self.get_targeting_summary(feature_names)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Correlation distribution
        axes[0, 0].hist(self.correlations_, bins=30, alpha=0.7, edgecolor='black')
        if self.threshold_type == 'hard':
            axes[0, 0].axvline(self.threshold, color='red', linestyle='--',
                              linewidth=2, label=f'Threshold={self.threshold:.3f}')
            axes[0, 0].axvline(-self.threshold, color='red', linestyle='--',
                              linewidth=2)
        axes[0, 0].set_xlabel('Marginal Correlation with Target', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Distribution of Predictor Correlations',
                            fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Absolute correlations (sorted)
        sorted_abs_corr = np.sort(np.abs(self.correlations_))[::-1]
        axes[0, 1].plot(sorted_abs_corr, linewidth=2)
        if self.threshold_type == 'hard':
            n_selected = len(self.selected_predictors_)
            axes[0, 1].axvline(n_selected, color='red', linestyle='--',
                              label=f'{n_selected} selected')
            axes[0, 1].axhline(self.threshold, color='red', linestyle='--',
                              alpha=0.5)
        axes[0, 1].set_xlabel('Predictor Rank', fontsize=11)
        axes[0, 1].set_ylabel('Absolute Correlation', fontsize=11)
        axes[0, 1].set_title('Sorted Absolute Correlations',
                            fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Factor variance explained
        var_explained = self.pca.explained_variance_ratio_
        axes[1, 0].bar(range(1, len(var_explained) + 1), var_explained,
                      alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Factor Number', fontsize=11)
        axes[1, 0].set_ylabel('Variance Explained', fontsize=11)
        axes[1, 0].set_title('Targeted Factors: Variance Explained',
                            fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # 4. Forecast coefficients
        axes[1, 1].bar(range(1, len(self.forecast_coef_) + 1),
                      self.forecast_coef_, alpha=0.7, edgecolor='black')
        axes[1, 1].axhline(0, color='black', linewidth=0.5)
        axes[1, 1].set_xlabel('Factor Number', fontsize=11)
        axes[1, 1].set_ylabel('Forecast Coefficient', fontsize=11)
        axes[1, 1].set_title('Forecast Regression Coefficients',
                            fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig, axes

    def explained_variance_ratio(self):
        """Return variance explained by targeted factors."""
        return self.pca.explained_variance_ratio_
```

</div>
</div>

### Example: Comparing Standard vs Targeted Factors


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Generate data with sparse relevance structure
np.random.seed(789)
T, N = 200, 60
r_true = 3

# True factors
F_true = np.zeros((T, r_true))
F_true[:, 0] = np.random.randn(T)
for i in range(1, r_true):
    F_true[:, i] = 0.7 * F_true[:, i-1] + np.sqrt(0.51) * np.random.randn(T)

# Loadings: only 20 predictors load on factors relevant for y
Lambda = np.zeros((N, r_true))
relevant_predictors = np.random.choice(N, size=20, replace=False)
Lambda[relevant_predictors, :] = np.random.randn(20, r_true) / np.sqrt(r_true)

# Add irrelevant predictors with high variance
irrelevant_idx = np.setdiff1d(np.arange(N), relevant_predictors)
noise_factors = np.random.randn(T, 2)
noise_loadings = np.random.randn(len(irrelevant_idx), 2) * 2  # High loadings!

X = np.zeros((T, N))
X[:, relevant_predictors] = F_true @ Lambda[relevant_predictors, :].T
X[:, irrelevant_idx] = noise_factors @ noise_loadings.T
X += 0.3 * np.random.randn(T, N)

# Target: depends only on first two factors
beta_true = np.array([1.0, 0.5, 0.0])
y = F_true @ beta_true + 0.5 * np.random.randn(T)

# Split
T_train = 150
X_train, X_test = X[:T_train], X[T_train:]
y_train, y_test = y[:T_train], y[T_train:]

# Standard diffusion index
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca_standard = PCA(n_components=5)
F_standard_train = pca_standard.fit_transform(X_train_scaled)
F_standard_test = pca_standard.transform(X_test_scaled)

reg_standard = LinearRegression()
reg_standard.fit(F_standard_train[:-1], y_train[1:])
y_pred_standard = reg_standard.predict(F_standard_test[:-1])

# Targeted predictors (hard thresholding)
target_hard = TargetedPredictors(n_factors=5, threshold_type='hard',
                                 threshold=0.1, horizon=1)
target_hard.fit(X_train, y_train)
y_pred_hard = target_hard.predict(X_test[:-1])

# Targeted predictors (soft thresholding)
target_soft = TargetedPredictors(n_factors=5, threshold_type='soft',
                                 gamma=2.0, horizon=1)
target_soft.fit(X_train, y_train)
y_pred_soft = target_soft.predict(X_test[:-1])

# Evaluate
mse_standard = np.mean((y_test[1:] - y_pred_standard)**2)
mse_hard = np.mean((y_test[1:] - y_pred_hard)**2)
mse_soft = np.mean((y_test[1:] - y_pred_soft)**2)

print("=" * 70)
print("STANDARD vs TARGETED PREDICTORS COMPARISON")
print("=" * 70)
print(f"Data: T={T}, N={N} ({len(relevant_predictors)} relevant predictors)")
print(f"True model: 2 of 3 factors relevant for target")
print()
print("Out-of-Sample MSE:")
print(f"  Standard PCA:           {mse_standard:.4f}")
print(f"  Targeted (Hard):        {mse_hard:.4f}  ({(mse_standard-mse_hard)/mse_standard*100:+.1f}%)")
print(f"  Targeted (Soft, γ=2):   {mse_soft:.4f}  ({(mse_standard-mse_soft)/mse_standard*100:+.1f}%)")
print()
print(f"Hard thresholding selected: {len(target_hard.selected_predictors_)} predictors")
print(f"  True relevant recovered: {len(set(target_hard.selected_predictors_) & set(relevant_predictors))}/{len(relevant_predictors)}")
print("=" * 70)

# Visualization
target_hard.plot_targeting_results()
plt.savefig('targeting_results.png', dpi=300, bbox_inches='tight')
plt.show()
```

</div>
</div>

### Threshold Selection via Cross-Validation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">select_threshold_cv.py</span>
</div>

```python
def select_threshold_cv(X, y, n_factors=5, thresholds=None,
                       cv_folds=5, horizon=1):
    """
    Select optimal threshold via cross-validation.

    Parameters
    ----------
    X : array, shape (T, N)
        Predictors
    y : array, shape (T,)
        Target
    n_factors : int
        Number of factors
    thresholds : array, optional
        Candidate thresholds to try
    cv_folds : int
        Number of CV folds
    horizon : int
        Forecast horizon

    Returns
    -------
    best_threshold : float
        Optimal threshold
    cv_results : DataFrame
        CV results for all thresholds
    """
    T = len(y)

    if thresholds is None:
        # Auto-generate threshold candidates
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        all_corr = []
        for j in range(X.shape[1]):
            corr, _ = pearsonr(X_scaled[:-horizon, j], y[horizon:])
            all_corr.append(abs(corr))
        all_corr = np.array(all_corr)
        thresholds = np.percentile(all_corr, [10, 20, 30, 40, 50, 60, 70, 80, 90])

    # Cross-validation
    fold_size = T // cv_folds
    results = []

    for threshold in thresholds:
        fold_errors = []

        for fold in range(cv_folds):
            # Split into train/val
            val_start = fold * fold_size
            val_end = val_start + fold_size

            train_idx = np.concatenate([
                np.arange(0, val_start),
                np.arange(val_end, T)
            ])
            val_idx = np.arange(val_start, min(val_end, T))

            X_train_cv = X[train_idx]
            y_train_cv = y[train_idx]
            X_val_cv = X[val_idx]
            y_val_cv = y[val_idx]

            # Fit targeted model
            model = TargetedPredictors(n_factors=n_factors,
                                      threshold_type='hard',
                                      threshold=threshold,
                                      horizon=horizon)
            model.fit(X_train_cv, y_train_cv)

            # Predict
            if len(X_val_cv) > horizon:
                y_pred = model.predict(X_val_cv[:-horizon])
                error = np.mean((y_val_cv[horizon:] - y_pred)**2)
                fold_errors.append(error)

        results.append({
            'threshold': threshold,
            'mean_mse': np.mean(fold_errors),
            'std_mse': np.std(fold_errors)
        })

    results_df = pd.DataFrame(results)
    best_threshold = results_df.loc[results_df['mean_mse'].idxmin(), 'threshold']

    return best_threshold, results_df

# Example usage
best_c, cv_results = select_threshold_cv(X_train, y_train, n_factors=5,
                                         cv_folds=5, horizon=1)

print(f"\nOptimal threshold from CV: {best_c:.4f}")
print(f"Expected MSE: {cv_results.loc[cv_results['threshold']==best_c, 'mean_mse'].values[0]:.4f}")

# Plot CV results
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(cv_results['threshold'], cv_results['mean_mse'],
        'b-o', linewidth=2, markersize=8, label='Mean CV MSE')
ax.fill_between(cv_results['threshold'],
                cv_results['mean_mse'] - cv_results['std_mse'],
                cv_results['mean_mse'] + cv_results['std_mse'],
                alpha=0.3, label='±1 Std Dev')
ax.axvline(best_c, color='red', linestyle='--', linewidth=2,
          label=f'Optimal = {best_c:.3f}')
ax.set_xlabel('Threshold (c)', fontsize=12)
ax.set_ylabel('Cross-Validation MSE', fontsize=12)
ax.set_title('Threshold Selection via Cross-Validation',
            fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('threshold_cv.png', dpi=300, bbox_inches='tight')
plt.show()
```

</div>
</div>

---

## 6. Common Pitfalls

### 1. Using Full Sample for Correlation

**Mistake:** Computing correlations using entire dataset including test period.

**Problem:** Look-ahead bias. In real-time forecasting, future values of $y$ are unknown.

**Solution:** Compute correlations only on training data.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# WRONG
correlations = [pearsonr(X[:, j], y)[0] for j in range(N)]

# CORRECT (for out-of-sample evaluation)
correlations = [pearsonr(X_train[:, j], y_train)[0] for j in range(N)]
```

</div>
</div>

### 2. Targeting Without Standardization

**Mistake:** Computing correlations on raw data with different scales.

**Problem:** High-variance variables appear more correlated even if weakly related.

**Solution:** Standardize before computing correlations.

### 3. Over-Aggressive Thresholding

**Mistake:** Setting threshold too high, selecting very few predictors.

**Problem:** May discard useful information, underfit.

**Solution:** Ensure at least $2r$ to $3r$ predictors selected (where $r$ is number of factors).

### 4. Ignoring Forecast Horizon

**Mistake:** Using contemporaneous correlation when forecasting $h$ steps ahead.

**Problem:** $\text{Corr}(X_t, y_t)$ differs from $\text{Corr}(X_t, y_{t+h})$.

**Solution:** Compute correlation with $y_{t+h}$ matching forecast horizon.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# WRONG for h-step forecast
corr = pearsonr(X[:, j], y)[0]

# CORRECT
corr = pearsonr(X[:-h, j], y[h:])[0]
```


---

## 7. Connections

### Builds On
- **Principal Components Analysis:** Factor extraction method
- **Diffusion Index Forecasting:** Standard approach being improved
- **Variable Selection:** Statistical problem of choosing relevant predictors

### Leads To
- **Three-Pass Regression Filter:** More sophisticated targeting approach
- **Sparse Factor Models:** Related but different approach to relevance
- **Boosted Regression Trees:** Alternative adaptive variable selection

### Related Methods
- **LASSO Regression:** Alternative sparse selection method
- **Partial Least Squares:** Also targets predictive directions
- **Forward Stepwise Regression:** Sequential variable selection

---

## 8. Practice Problems

### Conceptual

1. **Why might the first principal component have poor forecasting power** even though it explains the most variance in $X$?

2. **Compare hard vs soft thresholding.** When would you prefer each?

3. **Can targeted predictors hurt forecast performance?** Under what conditions?

### Mathematical

4. **Derive the soft-thresholding weights** for $\gamma = 2$. How do they relate to squared correlations?

5. **Show that soft thresholding with $\gamma \to \infty$** converges to hard thresholding.

6. **Prove that if all predictors are uncorrelated with $y$,** targeting reduces to random selection (no improvement over standard PCA).

### Implementation

7. **Compare standard PCA, hard targeting, and soft targeting** on simulated data where:
   - Half the predictors are relevant for $y$
   - Half are pure noise

   Which method performs best?

8. **Implement "double targeting":** First select predictors correlated with $y$, then among selected predictors, further select those with low correlation with each other (to get diverse factors).

9. **Create adaptive threshold selection** that chooses $c$ to maximize out-of-sample $R^2$ using rolling window validation.

### Advanced

10. **Extend targeting to multivariate forecasting** where you want to predict multiple targets simultaneously. How would you modify the correlation screening?

11. **Implement bootstrap confidence intervals** for the threshold parameter. How stable is the selection?

12. **Compare targeted factors with partial least squares (PLS).** Both aim to find predictive directions. What are the key differences?

---

<div class="callout-insight">

**Insight:** Understanding targeted predictors is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.


## 9. Further Reading

### Foundational Paper

- **Bai, J. & Ng, S. (2008).** "Forecasting Economic Time Series Using Targeted Predictors." *Journal of Econometrics*, 146(2), 304-317.
  - Original targeted predictors paper

- **Bai, J. & Ng, S. (2009).** "Boosting Diffusion Indices." *Journal of Applied Econometrics*, 24(4), 607-629.
  - Extensions and refinements

### Related Methodology

- **Kelly, B. & Pruitt, S. (2015).** "The Three-Pass Regression Filter: A New Approach to Forecasting Using Many Predictors." *Journal of Econometrics*, 186(2), 294-316.
  - Alternative targeting approach (next guide)

- **Ng, S. (2013).** "Variable Selection in Predictive Regressions." In *Handbook of Economic Forecasting*, Vol 2, 752-789.
  - Survey of variable selection methods for forecasting

### Empirical Applications

- **Bulligan, G., Marcellino, M., & Venditti, F. (2015).** "Forecasting Economic Activity with Targeted Predictors." *International Journal of Forecasting*, 31(1), 188-206.
  - Application to European data

- **Smeekes, S. & Wijler, E. (2021).** "Macroeconomic Forecasting Using Penalized Regression Methods." *International Journal of Forecasting*, 37(4), 1569-1585.
  - Comparison with LASSO and other methods

### Theory

- **Fan, J. & Lv, J. (2008).** "Sure Independence Screening for Ultrahigh Dimensional Feature Space." *Journal of the Royal Statistical Society: Series B*, 70(5), 849-911.
  - General theory of correlation-based screening

---

## Conceptual Practice Questions

1. What is the targeted predictor approach and when is it preferable to PCA?

2. How does soft thresholding differ from hard thresholding in predictor selection?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---

## Cross-References

<a class="link-card" href="./02_targeted_predictors_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_lasso_factor_selection.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_high_dimensional_regression.md">
  <div class="link-card-title">01 High Dimensional Regression</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_three_pass_filter.md">
  <div class="link-card-title">03 Three Pass Filter</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

