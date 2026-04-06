# High-Dimensional Regression: LASSO and Elastic Net

> **Reading time:** ~15 min | **Module:** Module 7: Sparse Methods | **Prerequisites:** Modules 0-6

<div class="callout-key">

**Key Concept Summary:** High-dimensional regression addresses the challenge of estimating models when the number of predictors exceeds or approaches the number of observations. LASSO (Least Absolute Shrinkage and Selection Operator) and elastic net use penalized regression to perform simultaneous variable selection and ...

</div>

## In Brief

High-dimensional regression addresses the challenge of estimating models when the number of predictors exceeds or approaches the number of observations. LASSO (Least Absolute Shrinkage and Selection Operator) and elastic net use penalized regression to perform simultaneous variable selection and coefficient estimation, providing sparse solutions that identify the most relevant predictors while avoiding overfitting.

<div class="callout-insight">

**Insight:** When you have more predictors than observations (or comparable numbers), ordinary least squares fails. LASSO adds an L1 penalty that shrinks some coefficients exactly to zero, automatically selecting variables. Elastic net combines L1 and L2 penalties to handle correlated predictors better than LASSO alone, making it ideal for factor models where predictors often move together.

</div>
---

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## 1. The High-Dimensional Regression Problem

### Motivation

In modern econometrics and factor modeling, we routinely face:
- $N$ predictors, $T$ observations, with $N \geq T$ or $N \approx T$
- Strong correlations among predictors
- Unknown which predictors are truly relevant
- Risk of overfitting with standard methods

**Examples:**
- Forecasting GDP using 100+ economic indicators
- Selecting relevant factors from candidate set
- Identifying sparse loadings in factor models

### Why OLS Fails

Standard OLS minimizes:
$$\min_\beta \sum_{t=1}^T (y_t - x_t' \beta)^2$$

**Problems when $N$ is large:**
<div class="flow">
<div class="flow-step mint">1. Non-uniqueness:</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">2. Multicollinearity:</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">3. Overfitting:</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Interpretation:</div>

</div>


1. **Non-uniqueness:** If $N > T$, infinitely many solutions exist
2. **Multicollinearity:** High correlation makes estimates unstable
3. **Overfitting:** Model fits training noise, poor out-of-sample performance
4. **Interpretation:** All $N$ coefficients non-zero, hard to interpret

---

## 2. LASSO Regression

### Formal Definition

The LASSO estimator solves:
$$\hat{\beta}^{\text{LASSO}} = \arg\min_\beta \left\{ \frac{1}{2T} \sum_{t=1}^T (y_t - x_t' \beta)^2 + \lambda \sum_{j=1}^N |\beta_j| \right\}$$

where:
- First term: residual sum of squares (goodness of fit)
- Second term: L1 penalty on coefficient magnitudes
- $\lambda \geq 0$: regularization parameter controlling sparsity

**Alternative formulation (constrained):**
$$\min_\beta \sum_{t=1}^T (y_t - x_t' \beta)^2 \quad \text{subject to} \quad \sum_{j=1}^N |\beta_j| \leq s$$

By Lagrangian duality, these are equivalent for appropriate $\lambda(s)$.

### Why L1 Induces Sparsity

The L1 penalty $|\beta|$ has a "corner" at zero, unlike L2 penalty $\beta^2$.

**Geometric intuition:**
- Constraint region: diamond shape in 2D (L1) vs. circle (L2)
- Objective contours: ellipses
- L1 constraint intersects objective at corners (axes), setting coefficients to exactly zero
- L2 constraint intersects smoothly, shrinking but rarely zeroing

**Mathematical reason:** L1 norm is not differentiable at zero, allowing exact zeros in solution.

### Properties

**1. Variable Selection:**
- As $\lambda$ increases, more coefficients become exactly zero
- Provides automatic variable selection

**2. Bias-Variance Trade-off:**
- Small $\lambda$: low bias, high variance (overfitting)
- Large $\lambda$: high bias, low variance (underfitting)
- Optimal $\lambda$ minimizes prediction error

**3. Path Continuity:**
- Solution path $\hat{\beta}(\lambda)$ is piecewise linear in $\lambda$
- Efficient computation via LARS algorithm

**4. Consistency:**
- Under sparsity assumptions, LASSO can recover true zero/non-zero pattern
- Requires "irrepresentable condition" for exact selection

---

## 3. Elastic Net Regression

### Motivation

LASSO limitations:
1. **Grouped variables:** When predictors are highly correlated, LASSO arbitrarily selects one and ignores others
2. **$N > T$ limit:** LASSO selects at most $T$ variables
3. **Instability:** Small data changes can drastically change selected variables

### Formal Definition

Elastic net combines L1 and L2 penalties:
$$\hat{\beta}^{\text{EN}} = \arg\min_\beta \left\{ \frac{1}{2T} \sum_{t=1}^T (y_t - x_t' \beta)^2 + \lambda \left[ \alpha \|\beta\|_1 + \frac{1-\alpha}{2} \|\beta\|_2^2 \right] \right\}$$

where:
- $\lambda \geq 0$: overall regularization strength
- $\alpha \in [0,1]$: mixing parameter
  - $\alpha = 1$: pure LASSO
  - $\alpha = 0$: pure ridge regression
  - $\alpha \in (0,1)$: elastic net

**Typical choice:** $\alpha = 0.5$ or selected via cross-validation.

### Why Elastic Net Works Better

**Grouping effect:**
When predictors $j$ and $k$ are highly correlated, elastic net assigns similar coefficients:
$$|\hat{\beta}_j - \hat{\beta}_k| \leq \frac{\text{const}}{\lambda(1-\alpha)} \cdot |y' (x_j - x_k)|$$

If $x_j \approx x_k$, then $\hat{\beta}_j \approx \hat{\beta}_k$.

**No variable limit:** Can select more than $T$ variables.

**Stability:** Ridge component ($L2$) stabilizes selection.

---

## 4. Selecting the Regularization Parameter

### Cross-Validation

**K-fold CV procedure:**

1. Split data into $K$ folds (typically $K=5$ or $10$)
2. For each candidate $\lambda$ value:
   - For each fold $k = 1, ..., K$:
     - Train on all folds except $k$
     - Predict on fold $k$
     - Compute prediction error
   - Average errors across folds: $\text{CV}(\lambda) = \frac{1}{K} \sum_{k=1}^K \text{MSE}_k(\lambda)$
3. Select $\lambda^* = \arg\min_\lambda \text{CV}(\lambda)$

**1-SE rule:** Choose largest $\lambda$ within 1 standard error of minimum CV error.
- Provides simpler model with comparable performance
- Guards against overfitting in finite samples

### Information Criteria

Alternative to CV for speed:
$$\text{AIC}_\lambda = T \log(\text{RSS}_\lambda / T) + 2 \cdot \text{df}_\lambda$$
$$\text{BIC}_\lambda = T \log(\text{RSS}_\lambda / T) + \log(T) \cdot \text{df}_\lambda$$

where $\text{df}_\lambda$ = number of non-zero coefficients (effective degrees of freedom).

**BIC typically selects sparser models** than AIC or CV.

---

## 5. Mathematical Formulation

### Matrix Notation

With standardized predictors $X$ ($T \times N$) and response $y$ ($T \times 1$):

**LASSO:**
$$\hat{\beta}^{\text{LASSO}} = \arg\min_\beta \left\{ \frac{1}{2T} \|y - X\beta\|_2^2 + \lambda \|\beta\|_1 \right\}$$

**Elastic Net:**
$$\hat{\beta}^{\text{EN}} = \arg\min_\beta \left\{ \frac{1}{2T} \|y - X\beta\|_2^2 + \lambda \left[ \alpha \|\beta\|_1 + \frac{1-\alpha}{2} \|\beta\|_2^2 \right] \right\}$$

### Coordinate Descent Algorithm

Efficient algorithm for solving LASSO/elastic net:

**Initialize:** $\beta^{(0)} = 0$

**Iterate** until convergence:
- For $j = 1, ..., N$:
  - Compute partial residual: $r_{-j} = y - \sum_{k \neq j} x_k \beta_k$
  - Update coordinate:
  $$\beta_j \leftarrow S\left( \frac{x_j' r_{-j}}{T}, \lambda \alpha \right) / \left( \frac{\|x_j\|^2}{T} + \lambda(1-\alpha) \right)$$

where $S(z, \gamma)$ is the soft-thresholding operator:
$$S(z, \gamma) = \begin{cases}
z - \gamma & \text{if } z > \gamma \\
0 & \text{if } |z| \leq \gamma \\
z + \gamma & \text{if } z < -\gamma
\end{cases}$$

**Convergence:** Typically very fast (10-100 iterations).

### Soft Thresholding

The soft-thresholding operator is key to LASSO's sparsity:
- Shrinks small coefficients exactly to zero
- Shrinks large coefficients toward zero by $\gamma$
- Creates piecewise linear solution paths

---

## 6. Intuitive Explanation

### The Variable Selection Problem

Imagine forecasting inflation using 100 economic indicators:
- Employment, production, interest rates, commodity prices, surveys, etc.
- Only ~50 observations available
- Unknown which indicators truly matter

**Standard approach problems:**
- Can't include all 100 (more variables than data points)
- Manually selecting variables is subjective and unstable
- Stepwise selection is unstable and lacks theoretical guarantees

### LASSO Solution

LASSO says: "Include all variables, but penalize complexity."

**How it works:**
1. Start with all 100 predictors
2. Add penalty for using many variables
3. Algorithm automatically zeros out unimportant ones
4. Final model uses only significant predictors (e.g., 8-15)

**Analogy:** Packing a suitcase with weight limit.
- OLS: Pack everything (overfilled, unstable)
- Subset selection: Pack or leave each item (discrete, unstable)
- LASSO: Include items, but pay penalty per item (automatically prioritizes)

### Ridge vs. LASSO vs. Elastic Net

**Ridge Regression ($L2$ penalty):**
- Shrinks all coefficients smoothly
- Never exactly zeros coefficients
- Best when many small effects

**LASSO ($L1$ penalty):**
- Zeros out irrelevant coefficients
- Automatic variable selection
- Best when few large effects (sparse truth)

**Elastic Net (combination):**
- Combines benefits of both
- Handles correlated predictors well
- Best for factor models (grouped correlations)

---

## 7. Code Implementation

### Basic LASSO Implementation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">highdimensionalregression.py</span>

</div>

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class HighDimensionalRegression:
    """
    High-dimensional regression with LASSO and Elastic Net.

    Parameters
    ----------
    method : str
        Regularization method: 'lasso', 'elasticnet', or 'ridge'
    alpha : float or None
        Regularization parameter (lambda). If None, use CV.
    l1_ratio : float
        Elastic net mixing parameter (alpha in equations)
        Only used if method='elasticnet'
    cv_folds : int
        Number of cross-validation folds
    standardize : bool
        Whether to standardize predictors
    """

    def __init__(self, method='lasso', alpha=None, l1_ratio=0.5,
                 cv_folds=5, standardize=True):
        self.method = method
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.cv_folds = cv_folds
        self.standardize = standardize

        self.scaler = StandardScaler() if standardize else None
        self.model = None
        self.cv_results_ = None
        self.selected_features_ = None

    def fit(self, X, y):
        """
        Fit high-dimensional regression model.

        Parameters
        ----------
        X : array-like, shape (T, N)
            Predictor matrix
        y : array-like, shape (T,)
            Response variable
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Standardize if requested
        if self.standardize:
            X = self.scaler.fit_transform(X)

        # Select model based on method
        if self.alpha is None:
            # Use cross-validation to select alpha
            if self.method == 'lasso':
                self.model = LassoCV(cv=self.cv_folds, random_state=42,
                                     max_iter=10000)
            elif self.method == 'elasticnet':
                self.model = ElasticNetCV(cv=self.cv_folds, l1_ratio=self.l1_ratio,
                                          random_state=42, max_iter=10000)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            self.model.fit(X, y)
            self.alpha = self.model.alpha_
            self.cv_results_ = {
                'alphas': self.model.alphas_,
                'mse_path': self.model.mse_path_,
                'best_alpha': self.alpha
            }
        else:
            # Use specified alpha
            if self.method == 'lasso':
                self.model = Lasso(alpha=self.alpha, max_iter=10000,
                                   random_state=42)
            elif self.method == 'elasticnet':
                self.model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                                        max_iter=10000, random_state=42)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            self.model.fit(X, y)

        # Identify selected features (non-zero coefficients)
        self.selected_features_ = np.where(self.model.coef_ != 0)[0]

        return self

    def predict(self, X):
        """Generate predictions."""
        X = np.asarray(X)

        if self.standardize:
            X = self.scaler.transform(X)

        return self.model.predict(X)

    def get_coefficients(self, feature_names=None):
        """
        Get model coefficients as DataFrame.

        Parameters
        ----------
        feature_names : list, optional
            Names of features

        Returns
        -------
        coef_df : DataFrame
            Coefficients with feature names
        """
        coefs = self.model.coef_

        if feature_names is None:
            feature_names = [f"X{i}" for i in range(len(coefs))]

        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefs,
            'abs_coefficient': np.abs(coefs)
        })

        # Sort by absolute value
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
        coef_df['selected'] = coef_df['coefficient'] != 0

        return coef_df

    def plot_coefficient_path(self, X, y, feature_names=None, n_alphas=100):
        """
        Plot coefficient paths as function of regularization.

        Parameters
        ----------
        X : array-like, shape (T, N)
            Predictor matrix
        y : array-like, shape (T,)
            Response variable
        feature_names : list, optional
            Names of features to label
        n_alphas : int
            Number of alpha values to try
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if self.standardize:
            X = self.scaler.fit_transform(X)

        # Compute regularization path
        if self.method == 'lasso':
            alphas = np.logspace(-4, 1, n_alphas)
            coefs = []

            for alpha in alphas:
                model = Lasso(alpha=alpha, max_iter=10000)
                model.fit(X, y)
                coefs.append(model.coef_)

            coefs = np.array(coefs)
        else:
            raise NotImplementedError("Path plotting only for LASSO")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))

        for i in range(coefs.shape[1]):
            ax.plot(np.log10(alphas), coefs[:, i], linewidth=1.5)

        # Mark selected alpha
        if self.alpha is not None:
            ax.axvline(np.log10(self.alpha), color='red', linestyle='--',
                      label=f'Selected α={self.alpha:.4f}')

        ax.set_xlabel('log₁₀(α)', fontsize=12)
        ax.set_ylabel('Coefficient Value', fontsize=12)
        ax.set_title('LASSO Regularization Path', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        return fig, ax

    def plot_cv_results(self):
        """Plot cross-validation results."""
        if self.cv_results_ is None:
            raise ValueError("No CV results available. Fit with alpha=None first.")

        alphas = self.cv_results_['alphas']
        mse_path = self.cv_results_['mse_path']
        mse_mean = mse_path.mean(axis=1)
        mse_std = mse_path.std(axis=1)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(np.log10(alphas), mse_mean, 'b-', linewidth=2, label='Mean MSE')
        ax.fill_between(np.log10(alphas),
                        mse_mean - mse_std,
                        mse_mean + mse_std,
                        alpha=0.3, label='±1 Std Dev')

        # Mark optimal alpha
        best_idx = np.argmin(mse_mean)
        ax.axvline(np.log10(alphas[best_idx]), color='red', linestyle='--',
                  label=f'Best α={alphas[best_idx]:.4f}')

        # 1-SE rule
        se_threshold = mse_mean[best_idx] + mse_std[best_idx]
        larger_alphas = alphas[mse_mean <= se_threshold]
        if len(larger_alphas) > 0:
            alpha_1se = larger_alphas.max()
            ax.axvline(np.log10(alpha_1se), color='orange', linestyle=':',
                      label=f'1-SE α={alpha_1se:.4f}')

        ax.set_xlabel('log₁₀(α)', fontsize=12)
        ax.set_ylabel('Cross-Validation MSE', fontsize=12)
        ax.set_title(f'{self.method.upper()} Cross-Validation Results',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax

    def feature_importance(self, percentile=80):
        """
        Return most important features.

        Parameters
        ----------
        percentile : float
            Percentile threshold for importance

        Returns
        -------
        important_idx : array
            Indices of important features
        """
        abs_coefs = np.abs(self.model.coef_)
        threshold = np.percentile(abs_coefs[abs_coefs > 0], percentile)
        important_idx = np.where(abs_coefs >= threshold)[0]

        return important_idx
```

</div>

### Example Application

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
# Generate high-dimensional data
np.random.seed(42)
T = 100  # observations
N = 200  # predictors (N > T)

# True sparse model: only 10 predictors are relevant
true_support = np.random.choice(N, size=10, replace=False)
true_beta = np.zeros(N)
true_beta[true_support] = np.random.randn(10) * 2

# Generate correlated predictors
correlation = 0.5
cov_matrix = correlation * np.ones((N, N)) + (1 - correlation) * np.eye(N)
X = np.random.multivariate_normal(np.zeros(N), cov_matrix, size=T)

# Generate response
y = X @ true_beta + np.random.randn(T) * 0.5

# Split data
split = 80
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Fit LASSO with cross-validation
lasso_model = HighDimensionalRegression(method='lasso', alpha=None, cv_folds=5)
lasso_model.fit(X_train, y_train)

# Fit Elastic Net
enet_model = HighDimensionalRegression(method='elasticnet', alpha=None,
                                       l1_ratio=0.5, cv_folds=5)
enet_model.fit(X_train, y_train)

# Predictions
y_pred_lasso = lasso_model.predict(X_test)
y_pred_enet = enet_model.predict(X_test)

# Evaluation
mse_lasso = np.mean((y_test - y_pred_lasso)**2)
mse_enet = np.mean((y_test - y_pred_enet)**2)

print("=" * 60)
print("HIGH-DIMENSIONAL REGRESSION RESULTS")
print("=" * 60)
print(f"Data: T={T} observations, N={N} predictors")
print(f"True model: {len(true_support)} non-zero coefficients")
print()
print("LASSO Results:")
print(f"  Selected α: {lasso_model.alpha:.4f}")
print(f"  Selected features: {len(lasso_model.selected_features_)}")
print(f"  Test MSE: {mse_lasso:.4f}")
print()
print("Elastic Net Results:")
print(f"  Selected α: {enet_model.alpha:.4f}")
print(f"  Selected features: {len(enet_model.selected_features_)}")
print(f"  Test MSE: {mse_enet:.4f}")
print()

# Feature recovery
lasso_recovery = len(set(lasso_model.selected_features_) & set(true_support))
enet_recovery = len(set(enet_model.selected_features_) & set(true_support))

print("Feature Recovery:")
print(f"  LASSO recovered: {lasso_recovery}/{len(true_support)} true features")
print(f"  Elastic Net recovered: {enet_recovery}/{len(true_support)} true features")
print("=" * 60)
```

</div>

### Visualization

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
# Plot CV results
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# LASSO CV
lasso_model.plot_cv_results()
plt.subplot(1, 2, 1)
lasso_model.plot_cv_results()

# Elastic Net CV
plt.subplot(1, 2, 2)
enet_model.plot_cv_results()

plt.tight_layout()
plt.savefig('cv_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Coefficient comparison
coef_lasso = lasso_model.get_coefficients()
coef_enet = enet_model.get_coefficients()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# LASSO coefficients
axes[0].bar(range(N), lasso_model.model.coef_, alpha=0.6, color='blue')
axes[0].axhline(0, color='black', linewidth=0.5)
axes[0].scatter(true_support, true_beta[true_support],
               color='red', s=100, marker='x', linewidth=3,
               label='True non-zero', zorder=5)
axes[0].set_xlabel('Predictor Index', fontsize=12)
axes[0].set_ylabel('Coefficient Value', fontsize=12)
axes[0].set_title(f'LASSO Coefficients ({len(lasso_model.selected_features_)} selected)',
                 fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Elastic Net coefficients
axes[1].bar(range(N), enet_model.model.coef_, alpha=0.6, color='green')
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].scatter(true_support, true_beta[true_support],
               color='red', s=100, marker='x', linewidth=3,
               label='True non-zero', zorder=5)
axes[1].set_xlabel('Predictor Index', fontsize=12)
axes[1].set_ylabel('Coefficient Value', fontsize=12)
axes[1].set_title(f'Elastic Net Coefficients ({len(enet_model.selected_features_)} selected)',
                 fontsize=13, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('coefficient_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

</div>

---

## 8. Common Pitfalls

### 1. Forgetting to Standardize

**Mistake:** Applying LASSO to raw data with different scales.

**Problem:** Penalty affects variables differently based on their scale. Variables with larger scales get penalized less.

**Solution:** Always standardize predictors before applying LASSO/elastic net.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
# WRONG
model = Lasso(alpha=0.1)
model.fit(X_raw, y)  # Variables have different units!

# CORRECT
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
model = Lasso(alpha=0.1)
model.fit(X_scaled, y)
```

</div>

### 2. Not Centering the Response

**Mistake:** Including intercept in penalty.

**Problem:** Intercept should not be penalized (represents mean level).

**Solution:** Center $y$ or use `fit_intercept=True` in sklearn (default).

### 3. Using Training Data Alpha on Test Data

**Mistake:** Selecting $\lambda$ using training data, then evaluating on same data.

**Problem:** Overly optimistic performance estimates.

**Solution:** Use nested cross-validation or separate validation set.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
# WRONG
model = LassoCV(cv=5)
model.fit(X_train, y_train)
score = model.score(X_train, y_train)  # Biased!

# CORRECT
model = LassoCV(cv=5)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)  # Honest estimate
```

</div>

### 4. Interpreting Selected Variables as "Causal"

**Mistake:** Assuming selected variables have causal effect.

**Problem:** LASSO selects for prediction, not causality. Correlated variables may substitute.

**Solution:** Use LASSO for prediction or variable screening, not causal inference.

### 5. Ignoring Correlation Groups

**Mistake:** Using pure LASSO when predictors are highly correlated.

**Problem:** LASSO arbitrarily picks one from group, discards others.

**Solution:** Use elastic net ($\alpha < 1$) to handle grouped variables.

---

## 9. Connections

### Builds On
- **Linear Regression:** Foundation, but extended to high dimensions
- **Ridge Regression:** L2 penalty, complementary to L1
- **Variable Selection:** Statistical problem of choosing relevant predictors

### Leads To
- **Targeted Predictors:** Bai-Ng method using LASSO for factor selection
- **Sparse Factor Models:** LASSO-based factor estimation
- **Three-Pass Filter:** Kelly-Pruitt method combining LASSO with factors

### Related Methods
- **Forward Stepwise Selection:** Discrete alternative, less stable
- **Principal Components:** Dimension reduction without selection
- **Boosting:** Sequential variable selection with shrinkage

---

## 10. Practice Problems

### Conceptual

1. **Why does the L1 penalty produce exact zeros while L2 does not?** Explain geometrically and analytically.

2. **When would you prefer LASSO over elastic net? When would you prefer elastic net?**

3. **Explain why standardization is crucial for LASSO but not for OLS.**

### Mathematical

4. **Derive the soft-thresholding operator** from the LASSO objective for a single coefficient.

5. **Show that the LASSO solution path is piecewise linear in $\lambda$.**

6. **Prove that elastic net with $\alpha = 0$ is equivalent to ridge regression** (up to rescaling of $\lambda$).

### Implementation

7. **Compare LASSO, elastic net, and OLS** on a dataset where:
   - True model has 5 non-zero coefficients
   - You have 50 predictors, 100 observations
   - Predictors have varying correlation levels

   Which method recovers the true support best?

8. **Implement cross-validation from scratch** for LASSO (without using `LassoCV`). Verify your results match sklearn.

9. **Create a simulation study** showing elastic net's advantage when predictors come in highly correlated groups.

### Advanced

10. **Implement the coordinate descent algorithm** for LASSO from scratch. Compare speed with sklearn.

11. **Study LASSO's variable selection consistency.** Generate data where LASSO should select correct variables with high probability as $T \to \infty$. Verify empirically.

12. **Extend the implementation** to handle weighted LASSO where different predictors have different penalties: $\lambda \sum_j w_j |\beta_j|$.

---

<div class="callout-insight">

**Insight:** Understanding high-dimensional regression is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## 11. Further Reading

### Foundational Papers

- **Tibshirani, R. (1996).** "Regression Shrinkage and Selection via the Lasso." *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288.
  - Original LASSO paper

- **Zou, H. & Hastie, T. (2005).** "Regularization and Variable Selection via the Elastic Net." *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320.
  - Introduces elastic net, addresses LASSO limitations

### Theory

- **Bühlmann, P. & van de Geer, S. (2011).** *Statistics for High-Dimensional Data: Methods, Theory and Applications.* Springer.
  - Comprehensive theoretical treatment

- **Hastie, T., Tibshirani, R., & Wainwright, M. (2015).** *Statistical Learning with Sparsity: The Lasso and Generalizations.* CRC Press.
  - Modern perspective on LASSO and extensions

### Algorithms

- **Friedman, J., Hastie, T., & Tibshirani, R. (2010).** "Regularization Paths for Generalized Linear Models via Coordinate Descent." *Journal of Statistical Software*, 33(1), 1-22.
  - Efficient algorithms (basis for sklearn implementation)

### Applications in Economics

- **Belloni, A. & Chernozhukov, V. (2013).** "Least Squares After Model Selection in High-Dimensional Sparse Models." *Bernoulli*, 19(2), 521-547.
  - Post-selection inference

- **Giannone, D., Lenza, M., & Primiceri, G.E. (2021).** "Economic Predictions with Big Data: The Illusion of Sparsity." *Econometrica*, 89(5), 2409-2437.
  - Challenges of sparsity assumption in macroeconomics

### Software Documentation

- **Scikit-learn Documentation:** [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
  - Practical guide to implementation

---

## Conceptual Practice Questions

1. Explain the core idea of high-dimensional regression: lasso and elastic net in your own words to a colleague who has not studied it.

2. What is the most common mistake practitioners make when applying high-dimensional regression, and how would you avoid it?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./01_high_dimensional_regression_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_lasso_factor_selection.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./02_targeted_predictors.md">
  <div class="link-card-title">02 Targeted Predictors</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_three_pass_filter.md">
  <div class="link-card-title">03 Three Pass Filter</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

