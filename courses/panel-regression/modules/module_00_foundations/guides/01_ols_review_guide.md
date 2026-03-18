# OLS Review: Matrix Form

## In Brief

Ordinary Least Squares (OLS) in matrix notation is the algebraic foundation for every panel estimator. Fixed effects is OLS on demeaned data. Random effects is GLS, which is weighted OLS. Understanding the matrix form — particularly the projection geometry — makes the logic of panel methods transparent rather than mechanical.

## Key Insight

OLS solves the problem of finding $\hat{\beta}$ that minimizes the sum of squared residuals. In matrix form, this has a clean closed-form solution:

$$\hat{\beta} = (X'X)^{-1}X'y$$

This formula has a geometric interpretation: $\hat{y} = X\hat{\beta}$ is the **orthogonal projection** of $y$ onto the column space of $X$. The residual vector $\hat{\epsilon} = y - \hat{y}$ is perpendicular to every column of $X$. This orthogonality — and how it gets altered in panel settings — is the core idea behind fixed effects.

## Formal Definition

Given data $(y, X)$ where $y$ is $n \times 1$ and $X$ is $n \times k$ with full column rank:

**Optimization problem:**

$$\hat{\beta} = \arg\min_{\beta} S(\beta) = (y - X\beta)'(y - X\beta)$$

**Solution (normal equations):**

$$X'X\beta = X'y \implies \hat{\beta} = (X'X)^{-1}X'y$$

**Key objects:**

| Object | Formula | Interpretation |
|--------|---------|----------------|
| Fitted values | $\hat{y} = Py$ | Projection onto col$(X)$ |
| Residuals | $\hat{\epsilon} = My$ | Orthogonal complement |
| Projection matrix | $P = X(X'X)^{-1}X'$ | Symmetric, idempotent |
| Residual maker | $M = I - P$ | Symmetric, idempotent |
| Residual variance | $\hat{\sigma}^2 = \hat{\epsilon}'\hat{\epsilon}/(n-k)$ | Unbiased under Gauss-Markov |
| Coefficient variance | $\widehat{\text{Var}}(\hat{\beta}) = \hat{\sigma}^2(X'X)^{-1}$ | Under homoskedasticity |

**Gauss-Markov Theorem:** Under the classical assumptions (linearity, strict exogeneity $E[\epsilon|X]=0$, spherical errors $\text{Var}(\epsilon|X) = \sigma^2 I$), OLS is the Best Linear Unbiased Estimator (BLUE).

**Derivation of normal equations:**

Expanding $S(\beta) = y'y - 2\beta'X'y + \beta'X'X\beta$, differentiating with respect to $\beta$, and setting to zero:

$$\frac{\partial S}{\partial \beta} = -2X'y + 2X'X\beta = 0 \implies X'X\hat{\beta} = X'y$$

The second-order condition $\partial^2 S / \partial\beta\partial\beta' = 2X'X \succ 0$ (positive definite when $X$ has full column rank) confirms this is a minimum.

## Intuitive Explanation

Imagine the $n$-dimensional outcome vector $y$ floating in space. The columns of $X$ span a $k$-dimensional subspace — all possible linear combinations of the predictors. OLS finds the point in that subspace closest to $y$ (in Euclidean distance). The residual vector points from that closest point back to $y$, and it must be perpendicular to the subspace.

This perpendicularity is the condition $X'\hat{\epsilon} = 0$: residuals are uncorrelated with every predictor. When this holds, there is no information in the predictors that the fitted model has "missed."

**The Frisch-Waugh-Lovell (FWL) Theorem** states that the coefficient on $X_2$ from the regression of $y$ on $[X_1, X_2]$ equals the coefficient from regressing the residuals of $y$ on $X_1$ against the residuals of $X_2$ on $X_1$. This theorem is the algebraic foundation for fixed effects: the within transformation is exactly FWL applied to a matrix of entity dummies.

## Code Implementation

```python
import numpy as np
from scipy import stats
import statsmodels.api as sm

def ols_from_scratch(y, X, add_constant=True):
    """
    Compute OLS estimates using matrix algebra.
    Demonstrates the connection between formula and computation.

    Parameters
    ----------
    y : array-like, shape (n,)
    X : array-like, shape (n, k)
    add_constant : bool
        If True, prepend a column of ones to X.

    Returns
    -------
    dict with keys: beta, se, t_stats, p_values, r_squared, residuals
    """
    y = np.asarray(y, dtype=float).flatten()
    X = np.asarray(X, dtype=float)

    if add_constant:
        X = np.column_stack([np.ones(len(y)), X])

    n, k = X.shape

    # beta = (X'X)^{-1} X'y
    # Use np.linalg.solve rather than inv for numerical stability
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)

    # Fitted values and residuals
    y_hat = X @ beta
    residuals = y - y_hat

    # Unbiased residual variance: sigma^2 = e'e / (n - k)
    sigma2 = (residuals @ residuals) / (n - k)

    # Covariance matrix: Var(beta) = sigma^2 * (X'X)^{-1}
    XtX_inv = np.linalg.inv(XtX)
    var_beta = sigma2 * XtX_inv
    se = np.sqrt(np.diag(var_beta))

    # Inference
    t_stats = beta / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

    # R-squared
    ss_res = residuals @ residuals
    ss_tot = ((y - y.mean()) @ (y - y.mean()))
    r_squared = 1.0 - ss_res / ss_tot

    return {
        'beta': beta,
        'se': se,
        't_stats': t_stats,
        'p_values': p_values,
        'r_squared': r_squared,
        'residuals': residuals,
        'sigma2': sigma2
    }


# --- Verify against statsmodels ---
np.random.seed(42)
n = 500
X = np.random.randn(n, 2)
true_beta = np.array([3.0, 1.5, -0.8])  # intercept, x1, x2
y = true_beta[0] + X @ true_beta[1:] + np.random.randn(n) * 1.2

results_manual = ols_from_scratch(y, X)
results_sm = sm.OLS(y, sm.add_constant(X)).fit()

print("Coefficient comparison:")
print(f"  Manual: {results_manual['beta']}")
print(f"  statsmodels: {results_sm.params.values}")
print(f"\nR-squared: {results_manual['r_squared']:.6f} (manual)")
print(f"R-squared: {results_sm.rsquared:.6f} (statsmodels)")
```

**Projection matrix properties:**

```python
def verify_projection_properties(X):
    """
    Verify that P and M satisfy their algebraic properties.
    These properties are the basis for fixed effects algebra.
    """
    n = len(X)
    P = X @ np.linalg.inv(X.T @ X) @ X.T
    M = np.eye(n) - P

    # Symmetry
    assert np.allclose(P, P.T), "P not symmetric"
    assert np.allclose(M, M.T), "M not symmetric"

    # Idempotency
    assert np.allclose(P @ P, P), "P not idempotent"
    assert np.allclose(M @ M, M), "M not idempotent"

    # Orthogonality
    assert np.allclose(P @ M, np.zeros((n, n))), "P and M not orthogonal"

    # Rank: rank(P) = k, rank(M) = n - k
    k = X.shape[1]
    assert int(np.round(np.trace(P))) == k, "rank(P) != k"
    assert int(np.round(np.trace(M))) == n - k, "rank(M) != n - k"

    print(f"All projection properties verified for n={n}, k={k}")

X_full = sm.add_constant(X)
verify_projection_properties(X_full)
```

**FWL Theorem demonstration — the algebraic basis for within transformation:**

```python
def fwl_demo(y, X1, X2):
    """
    Show that beta_2 from regressing y on [X1, X2] equals
    the coefficient from the FWL (partialled-out) regression.
    This is exactly what the within transformation does.
    """
    # Full regression
    X_full = np.column_stack([X1, X2])
    beta_full = np.linalg.solve(X_full.T @ X_full, X_full.T @ y)
    beta2_full = beta_full[X1.shape[1]:]

    # FWL: residualize y and X2 on X1
    M1 = np.eye(len(y)) - X1 @ np.linalg.inv(X1.T @ X1) @ X1.T
    y_tilde = M1 @ y
    X2_tilde = M1 @ X2

    beta2_fwl = np.linalg.solve(X2_tilde.T @ X2_tilde, X2_tilde.T @ y_tilde)

    print(f"Full regression beta_2: {beta2_full}")
    print(f"FWL beta_2:             {beta2_fwl}")
    print(f"Difference (should be ~0): {np.abs(beta2_full - beta2_fwl).max():.2e}")

# Entity dummies as X1, time-varying regressors as X2
# This is exactly the fixed effects within transformation
N_small, T_small = 10, 5
D_i = np.kron(np.eye(N_small), np.ones((T_small, 1)))  # entity dummies
X2_reg = np.random.randn(N_small * T_small, 2)
y_panel = D_i @ np.random.randn(N_small) + X2_reg @ np.array([1.5, -0.5]) + np.random.randn(N_small * T_small) * 0.5

fwl_demo(y_panel, D_i, X2_reg)
```

## Common Pitfalls

**Using `np.linalg.inv` directly.** The formula $\hat{\beta} = (X'X)^{-1}X'y$ involves a matrix inverse, but computing the inverse explicitly is numerically unstable and slower than solving the linear system. Always use `np.linalg.solve(X.T @ X, X.T @ y)`.

**Forgetting the intercept.** Omitting the constant forces the regression through the origin. The residuals will no longer sum to zero, and $R^2$ can be negative. The default in both `statsmodels` (via `sm.add_constant`) and `linearmodels` is to include a constant — verify this matches your specification.

**Assuming homoskedasticity in panel data.** The formula $\hat{\text{Var}}(\hat{\beta}) = \hat{\sigma}^2(X'X)^{-1}$ assumes constant error variance. In panel data this is almost never true — use robust or clustered standard errors.

**Multicollinearity.** When columns of $X$ are nearly linearly dependent, $(X'X)$ is nearly singular, $\hat{\beta}$ becomes numerically unstable, and standard errors explode. Check `np.linalg.cond(X.T @ X)` — values above 30 indicate potential issues. In panel data, time-invariant variables perfectly collinear with entity dummies cause this.

**Rank deficiency in fixed effects.** LSDV (Least Squares Dummy Variables) for fixed effects drops one dummy to avoid perfect multicollinearity with the intercept. Failing to do this causes $X'X$ to be singular. The within transformation avoids this issue entirely.

## Connections

**Builds on:**
- Linear algebra: matrix multiplication, inverses, rank, eigenvectors
- Calculus: derivatives of quadratic forms, first and second order conditions
- Probability: expected value, variance, the Law of Iterated Expectations

**Leads to:**
- Pooled OLS: apply directly to stacked panel observations
- Fixed effects (within estimator): OLS on entity-demeaned data — $M_{entity}$ applied to both $y$ and $X$
- Random effects: Generalized Least Squares, which is OLS after a weighted quasi-demeaning transformation
- Instrumental Variables (2SLS): replace endogenous $X$ with its projection onto instruments

**Related to:**
- Generalized Least Squares (GLS): OLS on transformed data to address non-spherical errors
- Ridge regression: adds $\lambda I$ to $(X'X)$ — relevant when near-multicollinearity threatens panel estimates
- Frisch-Waugh-Lovell theorem: the algebraic foundation for the within transformation

## Practice Problems

1. **Projection geometry.** Prove algebraically that $\hat{\epsilon} \perp X$ — that is, $X'\hat{\epsilon} = 0$. Then write code to verify this numerically for a regression with $n=200$, $k=3$. What does this orthogonality condition imply about $\text{Cov}(\hat{\epsilon}, \hat{y})$?

2. **Numerical stability.** Generate a design matrix $X$ with near-multicollinearity: $x_2 = x_1 + 0.001 \cdot \text{noise}$. Compare the coefficient estimates and condition numbers when you (a) use `np.linalg.inv(X.T @ X) @ X.T @ y` versus (b) `np.linalg.solve(X.T @ X, X.T @ y)`. Which is more stable?

3. **FWL and fixed effects.** Take the `wage_panel` dataset from `linearmodels.datasets`. Show that regressing log wages on union status after partialling out entity dummies (FWL) gives the same coefficient as running `PanelOLS` with `EntityEffects`. This demonstrates that fixed effects is exactly OLS via FWL.

## Further Reading

- Greene, W. H. (2018). *Econometric Analysis* (8th ed.), Pearson. Chapters 2–4 cover OLS in matrix form with full derivations.
- Davidson, R. & MacKinnon, J. G. (2004). *Econometric Theory and Methods*, Oxford. Chapter 1 develops OLS from the projection perspective.
- Lovell, M. C. (1963). "Seasonal Adjustment of Economic Time Series and Multiple Regression Analysis." *Journal of the American Statistical Association*, 58, 993–1010. The original FWL paper, directly relevant to the within transformation.
- Frisch, R. & Waugh, F. V. (1933). "Partial Time Regressions as Compared with Individual Trends." *Econometrica*, 1(4), 387–401. The original paper establishing the partitioned regression result.
