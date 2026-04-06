# OLS Review: Matrix Form

> **Reading time:** ~12 min | **Module:** 00 — Foundations | **Prerequisites:** None (entry point)


## In Brief


<div class="callout-key">

**Key Concept Summary:** Ordinary Least Squares (OLS) in matrix notation provides the foundation for understanding panel data estimators. The matrix formulation reveals the geometric interpretation of regression and enable...

</div>

Ordinary Least Squares (OLS) in matrix notation provides the foundation for understanding panel data estimators. The matrix formulation reveals the geometric interpretation of regression and enables efficient computation for large datasets.

> 💡 **Key Insight:** OLS finds the linear combination of predictors that minimizes the sum of squared residuals. In matrix form, this optimization has a closed-form solution: $\hat{\beta} = (X'X)^{-1}X'y$, which represents the projection of $y$ onto the column space of $X$.

## Formal Definition

**OLS Estimator:** Given data $(y, X)$ where $y$ is an $n \times 1$ vector of outcomes and $X$ is an $n \times k$ matrix of predictors, the OLS estimator is:

$$\hat{\beta} = \arg\min_{\beta} (y - X\beta)'(y - X\beta)$$

The solution is:

<div class="callout-insight">

**Insight:** Panel data lets you control for unobservable differences between entities that are constant over time. This is the single most important reason to prefer panel data over repeated cross-sections.

</div>


$$\hat{\beta} = (X'X)^{-1}X'y$$

provided $(X'X)$ is invertible (full column rank).

**Fitted Values:**
$$\hat{y} = X\hat{\beta} = X(X'X)^{-1}X'y = Py$$

where $P = X(X'X)^{-1}X'$ is the projection matrix.

**Residuals:**
$$\hat{\epsilon} = y - \hat{y} = (I - P)y = My$$

where $M = I - P$ is the residual maker matrix.

## Intuitive Explanation

<div class="callout-warning">

**Warning:** Reporting results without appropriate standard errors is a common mistake. In panel data, conventional OLS standard errors are almost always wrong -- use clustered or heteroskedasticity-robust standard errors.

</div>


Think of OLS as finding the "best fitting" hyperplane through a cloud of data points in multi-dimensional space. The matrix $(X'X)^{-1}X'$ acts as a "pseudo-inverse" that solves for the coefficients that minimize vertical distances (residuals) from points to the fitted plane.

**Geometric Interpretation:**
- $X\beta$ represents all possible linear combinations of the columns of $X$
- $\hat{y}$ is the point in this column space closest to $y$
- The residual $\hat{\epsilon}$ is orthogonal to the column space of $X$

**Why Matrix Form Matters:**
- Generalizes to any number of predictors
- Reveals computational structure
- Basis for understanding panel estimators (FE, RE are variations of this)
- Enables theoretical analysis of statistical properties

## Mathematical Formulation

### Derivation of OLS Estimator

Starting from the minimization problem:

$$\min_{\beta} S(\beta) = (y - X\beta)'(y - X\beta)$$

Expand the quadratic form:

$$S(\beta) = y'y - 2\beta'X'y + \beta'X'X\beta$$

Take the derivative with respect to $\beta$:

$$\frac{\partial S}{\partial \beta} = -2X'y + 2X'X\beta$$

Set equal to zero (first-order condition):

$$X'X\beta = X'y$$

These are the **normal equations**. Solving for $\beta$:

$$\hat{\beta} = (X'X)^{-1}X'y$$

### Variance of OLS Estimator

Under the Gauss-Markov assumptions:

$$\text{Var}(\hat{\beta} | X) = \sigma^2(X'X)^{-1}$$

where $\sigma^2 = \text{Var}(\epsilon_i)$.

**Estimator for $\sigma^2$:**

$$\hat{\sigma}^2 = \frac{\hat{\epsilon}'\hat{\epsilon}}{n - k} = \frac{(y - X\hat{\beta})'(y - X\hat{\beta})}{n - k}$$

### Projection Matrices

**Projection Matrix $P$:**
- $P = X(X'X)^{-1}X'$
- Symmetric: $P' = P$
- Idempotent: $P^2 = P$
- Projects onto column space of $X$

**Residual Maker Matrix $M$:**
- $M = I - P$
- Symmetric: $M' = M$
- Idempotent: $M^2 = M$
- Projects onto orthogonal complement of column space of $X$

## Code Implementation


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
from scipy import stats

def ols_estimator(y, X, add_constant=True):
    """
    Compute OLS estimates using matrix algebra.

    Parameters
    ----------
    y : array-like, shape (n,)
        Dependent variable
    X : array-like, shape (n, k)
        Independent variables
    add_constant : bool, default=True
        Whether to add intercept column to X

    Returns
    -------
    results : dict
        Dictionary containing:
        - beta_hat: OLS coefficients
        - y_hat: Fitted values
        - residuals: Residuals
        - sigma2: Residual variance estimate
        - var_beta: Variance-covariance matrix of beta_hat
        - se_beta: Standard errors of beta_hat
        - t_stats: t-statistics for each coefficient
        - p_values: p-values for each coefficient
    """
    # Convert to numpy arrays
    y = np.asarray(y).flatten()
    X = np.asarray(X)

    # Add constant if requested
    if add_constant:
        X = np.column_stack([np.ones(len(y)), X])

    n, k = X.shape

    # Compute OLS estimator: beta_hat = (X'X)^{-1} X'y
    XtX = X.T @ X
    Xty = X.T @ y
    beta_hat = np.linalg.solve(XtX, Xty)  # More stable than direct inverse

    # Fitted values: y_hat = X * beta_hat
    y_hat = X @ beta_hat

    # Residuals: e = y - y_hat
    residuals = y - y_hat

    # Residual variance: sigma^2 = e'e / (n - k)
    sigma2 = (residuals @ residuals) / (n - k)

    # Variance-covariance matrix: Var(beta_hat) = sigma^2 * (X'X)^{-1}
    XtX_inv = np.linalg.inv(XtX)
    var_beta = sigma2 * XtX_inv

    # Standard errors
    se_beta = np.sqrt(np.diag(var_beta))

    # t-statistics: t = beta_hat / se(beta_hat)
    t_stats = beta_hat / se_beta

    # p-values (two-sided test)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-k))

    return {
        'beta_hat': beta_hat,
        'y_hat': y_hat,
        'residuals': residuals,
        'sigma2': sigma2,
        'var_beta': var_beta,
        'se_beta': se_beta,
        't_stats': t_stats,
        'p_values': p_values,
        'n': n,
        'k': k,
        'r_squared': 1 - (residuals @ residuals) / ((y - y.mean()) @ (y - y.mean()))
    }

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n = 100
    X = np.random.randn(n, 2)
    true_beta = np.array([1.5, 2.0, -0.5])  # [intercept, beta1, beta2]
    y = true_beta[0] + X @ true_beta[1:] + np.random.randn(n) * 0.5

    # Estimate OLS
    results = ols_estimator(y, X)

    print("OLS Estimation Results")
    print("=" * 50)
    print(f"Coefficients: {results['beta_hat']}")
    print(f"Std Errors:   {results['se_beta']}")
    print(f"t-statistics: {results['t_stats']}")
    print(f"p-values:     {results['p_values']}")
    print(f"R-squared:    {results['r_squared']:.4f}")
    print(f"Residual std: {np.sqrt(results['sigma2']):.4f}")
```


</div>

## Common Pitfalls

**1. Multicollinearity**
- **Issue:** When $X'X$ is nearly singular (columns of $X$ are nearly linearly dependent), $(X'X)^{-1}$ becomes unstable.
- **Consequence:** Coefficient estimates have very large standard errors; small data changes cause large coefficient changes.
- **Detection:** Check condition number of $X'X$: `np.linalg.cond(X.T @ X)` > 30 indicates problems.
- **Solution:** Remove redundant variables, use regularization (ridge regression), or collect more data.

**2. Forgetting the Intercept**
- **Issue:** Omitting the constant term when it should be included.
- **Consequence:** Biased estimates if the true model has an intercept; forces regression line through origin.
- **Prevention:** Always consider whether the intercept makes theoretical sense; use `add_constant=True` by default.

**3. Confusing Matrix Dimensions**
- **Issue:** Shape mismatches in matrix multiplication (e.g., $X$ is $n \times k$ but $y$ is $k \times 1$).
- **Prevention:** Always verify shapes: $y$ should be $(n,)$ or $(n, 1)$; $X$ should be $(n, k)$; $\beta$ will be $(k,)$ or $(k, 1)$.

**4. Using Matrix Inverse Directly**
- **Issue:** Computing $(X'X)^{-1}$ explicitly is numerically unstable.
- **Better:** Use `np.linalg.solve(X.T @ X, X.T @ y)` instead of `np.linalg.inv(X.T @ X) @ X.T @ y`.
- **Why:** `solve()` uses more stable algorithms (LU decomposition) and is faster.

**5. Assuming Homoskedasticity Without Testing**
- **Issue:** The formula $\text{Var}(\hat{\beta}) = \sigma^2(X'X)^{-1}$ assumes constant variance.
- **Reality:** In panel data, this assumption almost always fails.
- **Solution:** Use robust (heteroskedasticity-consistent) standard errors, or model the variance explicitly.

## Connections

**Builds on:**
- Linear algebra: Matrix multiplication, inverse, rank
- Calculus: Derivatives of quadratic forms
- Probability: Expected values, variance, conditional expectations

**Leads to:**
- **Fixed Effects:** OLS on demeaned data (within transformation)
- **Random Effects:** GLS estimation (weighted OLS)
- **Pooled OLS:** Direct application to panel data (ignoring entity structure)
- **IV/2SLS:** Replacing $X$ with predicted values from first stage

**Related to:**
- **Maximum Likelihood:** OLS is MLE under normality assumption
- **Projection:** OLS is orthogonal projection onto column space
- **Best Linear Unbiased Estimator (BLUE):** Gauss-Markov theorem establishes OLS optimality

## Practice Problems

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.

</div>


### 1. Conceptual: Geometric Interpretation
**Question:** Explain why the residual vector $\hat{\epsilon}$ is orthogonal to every column of $X$. What does this imply about the correlation between residuals and predictors?

<details>
<summary>Solution</summary>

From the normal equations: $X'\hat{\epsilon} = X'(y - X\hat{\beta}) = X'y - X'X\hat{\beta} = 0$

This means each column of $X$ is orthogonal to $\hat{\epsilon}$. Geometrically, $\hat{y}$ is the projection of $y$ onto the column space of $X$, and $\hat{\epsilon}$ is the perpendicular distance from $y$ to this subspace.

Implication: $\text{Cov}(X_j, \hat{\epsilon}) = 0$ for all predictors $X_j$. This is a defining property of OLS - it ensures residuals are uncorrelated with fitted values and all predictors.
</details>

### 2. Implementation: Verify Projection Properties
**Question:** Write code to verify that the projection matrix $P = X(X'X)^{-1}X'$ is symmetric and idempotent. Also verify that $M = I - P$ satisfies the same properties.

<details>
<summary>Solution</summary>


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np

# Generate random data
np.random.seed(42)
n, k = 100, 3
X = np.random.randn(n, k)

# Compute projection matrix
XtX_inv = np.linalg.inv(X.T @ X)
P = X @ XtX_inv @ X.T

# Verify P is symmetric: P = P'
print("P is symmetric:", np.allclose(P, P.T))

# Verify P is idempotent: P^2 = P
print("P is idempotent:", np.allclose(P @ P, P))

# Verify M properties
M = np.eye(n) - P
print("M is symmetric:", np.allclose(M, M.T))
print("M is idempotent:", np.allclose(M @ M, M))

# Verify orthogonality: P @ M = 0
print("P and M are orthogonal:", np.allclose(P @ M, np.zeros((n, n))))
```


</div>
</details>

### 3. Extension: Partitioned Regression
**Question:** Suppose $X = [X_1, X_2]$ where $X_1$ is $n \times k_1$ and $X_2$ is $n \times k_2$. Show that the coefficient on $X_2$ from regressing $y$ on $[X_1, X_2]$ equals the coefficient from regressing $(y - X_1\hat{\beta}_1)$ on $(X_2 - X_1\hat{\gamma})$, where $\hat{\beta}_1$ is from regressing $y$ on $X_1$ and $\hat{\gamma}$ is from regressing $X_2$ on $X_1$.

**Hint:** This is the Frisch-Waugh-Lovell theorem, fundamental for understanding fixed effects estimation.

<details>
<summary>Solution</summary>

This result, called the Frisch-Waugh-Lovell (FWL) theorem, is the basis for the within transformation in fixed effects models.

**Proof sketch:**
1. Let $M_1 = I - X_1(X_1'X_1)^{-1}X_1'$ be the residual maker for $X_1$
2. The coefficient on $X_2$ in the full regression is: $\hat{\beta}_2 = (X_2'M_1X_2)^{-1}X_2'M_1y$
3. $M_1y = y - X_1\hat{\beta}_1$ (residuals from regressing $y$ on $X_1$)
4. $M_1X_2 = X_2 - X_1\hat{\gamma}$ (residuals from regressing $X_2$ on $X_1$)
5. Therefore: $\hat{\beta}_2 = ((M_1X_2)'(M_1X_2))^{-1}(M_1X_2)'(M_1y)$, which is OLS of $M_1y$ on $M_1X_2$

**Implication for Panel Data:** To get the coefficient on a time-varying variable in an FE model, we can:
1. Demean $y$ (residualize with respect to entity dummies)
2. Demean the time-varying variables
3. Run OLS on the demeaned variables

This is exactly the within transformation!
</details>

## Further Reading

**Essential:**
- **Greene, W. H.** (2018). *Econometric Analysis* (8th ed.), Chapter 2: The Classical Multiple Linear Regression Model. *Comprehensive treatment of OLS theory and properties.*

- **Wooldridge, J. M.** (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.), Chapter 4: The Single-Equation Linear Model and OLS Estimation. *Graduate-level treatment with focus on panel data applications.*

**Matrix Algebra Review:**
- **Magnus, J. R., & Neudecker, H.** (2019). *Matrix Differential Calculus with Applications in Statistics and Econometrics* (3rd ed.). *Definitive reference for matrix calculus in econometrics.*

**Computational:**
- **Gentle, J. E.** (2017). *Matrix Algebra: Theory, Computations and Applications in Statistics* (2nd ed.). *Focuses on numerical stability and computational issues in matrix operations.*

**Advanced:**
- **Hayashi, F.** (2000). *Econometrics*, Chapter 1: Finite-Sample Properties of OLS. *Rigorous treatment of OLS properties in finite samples and asymptotic theory.*

**Historical:**
- **Stigler, S. M.** (1981). "Gauss and the Invention of Least Squares." *The Annals of Statistics*, 9(3), 465-474. *Fascinating history of how OLS was developed and its philosophical foundations.*


---

## Cross-References

<a class="link-card" href="./01_panel_data_concepts.md">
  <div class="link-card-title">01 Panel Data Concepts</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_panel_data_concepts.md">
  <div class="link-card-title">01 Panel Data Concepts — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_data_structures.md">
  <div class="link-card-title">02 Data Structures</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_data_structures.md">
  <div class="link-card-title">02 Data Structures — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_data_structures_python.md">
  <div class="link-card-title">02 Data Structures Python</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_data_structures_python.md">
  <div class="link-card-title">02 Data Structures Python — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

