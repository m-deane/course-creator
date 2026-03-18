# The Static Factor Model: Specification and Assumptions

## In Brief

A static factor model decomposes high-dimensional data into a small number of latent common factors plus variable-specific noise. Rather than modeling $N(N-1)/2$ pairwise correlations, it models $r \ll N$ factors — reducing the macroeconomic dimensionality problem from intractable to tractable.

## Key Insight

When many variables co-move, there is a common cause. Factor models formalize this: all covariance between variables comes from their shared exposure to common factors. The idiosyncratic component is, by assumption, uncorrelated across variables (in the exact model) or weakly correlated (in approximate models). This structure is why factor models work empirically — macroeconomic data genuinely has a low-dimensional factor structure.

## Formal Definition

**Scalar form** for variable $i$ at time $t$:
$$X_{it} = \lambda_{i1}F_{1t} + \lambda_{i2}F_{2t} + \cdots + \lambda_{ir}F_{rt} + e_{it}, \quad i = 1, \ldots, N, \quad t = 1, \ldots, T$$

**Matrix form** (cross-section at time $t$):
$$X_t = \Lambda F_t + e_t, \quad X_t \in \mathbb{R}^N, \quad \Lambda \in \mathbb{R}^{N \times r}, \quad F_t \in \mathbb{R}^r, \quad e_t \in \mathbb{R}^N$$

**Full panel form**:
$$X = F\Lambda' + e, \quad X \in \mathbb{R}^{T \times N}, \quad F \in \mathbb{R}^{T \times r}, \quad e \in \mathbb{R}^{T \times N}$$

**Standard assumptions:**
1. $E[e_t | F_t] = 0$ — idiosyncratic errors uncorrelated with factors
2. $E[F_t] = 0$, $E[F_t F_t'] = I_r$ — factors normalized to identity covariance
3. **Exact model**: $E[e_t e_t'] = \text{diag}(\psi_1^2, \ldots, \psi_N^2) = \Psi$ — cross-sectionally uncorrelated errors
4. **Approximate model**: $E[e_t e_t'] = \Psi$ where $\Psi$ has bounded largest eigenvalue — weak cross-sectional correlation allowed

**Implied covariance structure** (with $\Sigma_F = I_r$):
$$\Sigma_X = \Lambda\Lambda' + \Psi$$

## Intuitive Explanation

Consider 200 macroeconomic series: industrial production, employment, retail sales, housing starts, interest rates, credit spreads, inflation, and so on. These series move together — they are not independent. A recession makes almost all real-activity series fall simultaneously.

The factor model captures this with a few numbers. If there are 3 factors (say, real activity, inflation, and financial conditions), each of the 200 variables loads on these 3 factors. Variable $i$'s behavior is:
- $\lambda_{i1} \times \text{(real activity factor)}$ — how much variable $i$ moves with the economic cycle
- $\lambda_{i2} \times \text{(inflation factor)}$ — how much variable $i$ co-moves with prices
- $\lambda_{i3} \times \text{(financial conditions factor)}$ — sensitivity to credit and financial markets
- $e_{it}$ — a variable-specific shock (a factory fire, a policy change, measurement error)

The loading matrix $\Lambda$ is the "sensitivity matrix" — it says how much each variable responds to each common shock. Two variables with similar loading rows will have high correlation because they respond similarly to the same factors.

The covariance decomposition $\Sigma_X = \Lambda\Lambda' + \Psi$ makes the structure explicit: all pairwise correlations come from the common factors ($\Lambda\Lambda'$ is a low-rank matrix encoding this), while $\Psi$ adds variable-specific noise along the diagonal.

## Code Implementation

```python
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis

# --- 1. Data generating process ---

def simulate_static_factor_model(T, N, r, lambda_scale=1.0,
                                  psi_range=(0.1, 0.5), seed=None):
    """
    Generate data from the exact static factor model:
        X = F @ Lambda' + e
        F ~ N(0, I_r)
        e_i ~ N(0, psi_i^2), independent across i

    Parameters
    ----------
    T : int, number of observations
    N : int, number of variables
    r : int, number of factors
    lambda_scale : float, scale of loadings (controls signal strength)
    psi_range : tuple, (min, max) for idiosyncratic standard deviations
    seed : int or None

    Returns
    -------
    X : ndarray, shape (T, N)
    F_true : ndarray, shape (T, r)
    Lambda_true : ndarray, shape (N, r)
    psi_true : ndarray, shape (N,), idiosyncratic stds
    """
    if seed is not None:
        np.random.seed(seed)

    # True factors: standard normal, shape (T, r)
    F_true = np.random.randn(T, r)

    # True loadings: random, shape (N, r)
    Lambda_true = np.random.randn(N, r) * lambda_scale

    # Idiosyncratic variances: random in specified range
    psi_true = np.random.uniform(psi_range[0], psi_range[1], size=N)

    # Idiosyncratic errors
    e = np.random.randn(T, N) * psi_true  # broadcasting over N

    # Observed data
    X = F_true @ Lambda_true.T + e

    return X, F_true, Lambda_true, psi_true


# --- 2. Model properties ---

def implied_covariance(Lambda, psi):
    """
    Compute the implied covariance matrix: Sigma_X = Lambda @ Lambda' + diag(psi^2).

    This is the population covariance of the factor model.
    The low-rank structure Lambda @ Lambda' carries all cross-variable covariance.
    The diagonal psi^2 adds variable-specific variance.

    Parameters
    ----------
    Lambda : ndarray, shape (N, r)
    psi : ndarray, shape (N,), idiosyncratic standard deviations

    Returns
    -------
    Sigma : ndarray, shape (N, N)
    """
    return Lambda @ Lambda.T + np.diag(psi ** 2)


def communalities(Lambda, psi):
    """
    Compute communality for each variable: fraction of variance from common factors.

    h_i^2 = sum_j(lambda_ij^2) / (sum_j(lambda_ij^2) + psi_i^2)

    High communality: variable mostly driven by common factors
    Low communality: variable mostly idiosyncratic
    """
    common_var = np.sum(Lambda ** 2, axis=1)     # sum over factors
    total_var = common_var + psi ** 2
    return common_var / total_var


def variance_decomposition(Lambda, psi, variable_names=None):
    """
    Decompose variance into common and idiosyncratic components for each variable.

    Returns a DataFrame-like dict for easy inspection.
    """
    N, r = Lambda.shape
    common_var = np.sum(Lambda ** 2, axis=1)
    idio_var = psi ** 2
    total_var = common_var + idio_var
    comms = common_var / total_var

    if variable_names is None:
        variable_names = [f'X_{i+1}' for i in range(N)]

    print("Variable Variance Decomposition:")
    print(f"{'Variable':<12} {'Common':>10} {'Idiosync':>10} {'Total':>10} {'Communality':>12}")
    print("-" * 55)
    for i in range(N):
        print(f"{variable_names[i]:<12} {common_var[i]:>10.3f} {idio_var[i]:>10.3f} "
              f"{total_var[i]:>10.3f} {comms[i]:>12.3f}")

    return {'common': common_var, 'idiosyncratic': idio_var, 'communality': comms}


# --- 3. Parameter counting and identification ---

def count_parameters(N, r):
    """
    Count identified parameters in the static factor model.

    Total parameters (before normalization):
      - Lambda: N*r loadings
      - Psi: N idiosyncratic variances
      - Sigma_F: r*(r+1)/2 factor covariance elements

    Observable (co)variances: N*(N+1)/2

    Identification condition: parameters <= observables
    (After normalization Sigma_F = I_r, remove r*(r+1)/2 and
     add r*(r-1)/2 rotation constraints, net reduction r^2)
    """
    n_loadings = N * r
    n_idiosyncratic = N
    n_factor_cov = r * (r + 1) // 2
    n_observables = N * (N + 1) // 2
    n_rotation = r * (r - 1) // 2  # constraints needed to fix rotation

    total_free = n_loadings + n_idiosyncratic + n_factor_cov - n_rotation
    n_observables_net = n_observables

    print(f"Static Factor Model with N={N}, r={r}:")
    print(f"  Loadings Lambda:       {n_loadings:>6}")
    print(f"  Idiosyncratic vars:    {n_idiosyncratic:>6}")
    print(f"  Factor covariances:    {n_factor_cov:>6}")
    print(f"  Rotation constraints:  {-n_rotation:>6}")
    print(f"  Free parameters:       {total_free:>6}")
    print(f"  Observable variances:  {n_observables_net:>6}")
    print(f"  Over-identified:       {n_observables_net > total_free}")

    return total_free, n_observables_net


# --- 4. Two-factor economic example ---

def economic_factor_example():
    """
    Illustrate a two-factor model with economic interpretation.
    Loadings designed to mimic: Factor 1 = Real Activity, Factor 2 = Nominal/Inflation
    """
    np.random.seed(42)
    T = 300

    # Economic variables: IP, Employment, CPI, Interest Rate
    variable_names = ['Indust. Production', 'Employment', 'CPI Inflation', 'Interest Rate']
    N = len(variable_names)
    r = 2

    # Loadings: real activity vs. nominal factor
    Lambda_true = np.array([
        [0.9, 0.1],   # IP: high on real, low on nominal
        [0.8, 0.2],   # Employment: high on real
        [0.2, 0.9],   # CPI: high on nominal
        [0.5, 0.6],   # Interest rate: responds to both
    ])

    F_true = np.random.randn(T, r)
    psi_true = np.array([0.3, 0.3, 0.4, 0.3])
    e = np.random.randn(T, N) * psi_true
    X = F_true @ Lambda_true.T + e

    print("Economic Two-Factor Model Example")
    print("="*40)

    # Implied covariance
    Sigma = implied_covariance(Lambda_true, psi_true)
    print("\nImplied covariance matrix:")
    print(Sigma.round(3))

    print("\nNote: Off-diagonal elements come entirely from Lambda @ Lambda'")
    print("e.g., Cov(IP, Employment) =", (Lambda_true[0] @ Lambda_true[1]).round(3))
    print("Implied:", Sigma[0, 1].round(3))

    # Variance decomposition
    print()
    variance_decomposition(Lambda_true, psi_true, variable_names)

    return X, F_true, Lambda_true, psi_true


if __name__ == "__main__":
    # Simulate and inspect a 3-factor model
    X, F, Lambda, psi = simulate_static_factor_model(
        T=300, N=30, r=3, lambda_scale=0.9, seed=42
    )
    print(f"Data shape: {X.shape}")
    print(f"Sample covariance shape: {(X - X.mean(0)).T @ (X - X.mean(0)):.0f}...".replace('...', ''))

    # Check parameter identification
    count_parameters(N=30, r=3)

    # Recover factors via PCA
    pca = PCA(n_components=3)
    F_hat = pca.fit_transform(X - X.mean(0))
    Lambda_hat = pca.components_.T

    print(f"\nPCA variance explained: {pca.explained_variance_ratio_.round(3)}")
    print(f"Total: {pca.explained_variance_ratio_.sum():.3f}")

    # Economic example
    X_econ, _, _, _ = economic_factor_example()
```

## Common Pitfalls

**Confusing factors (latent) with principal components (data transforms).** Factors $F_t$ are unobserved latent variables. Principal components are deterministic linear combinations of observed data. Under large-$N$ asymptotics, PCA scores consistently estimate the factor space, but in small samples they differ. The distinction matters for inference.

**Ignoring the identification problem.** The factor model is identified only up to rotation: if $F_t$ satisfies the model, so does $RF_t$ for any orthogonal matrix $R$ (with loadings $\Lambda R'$). Without normalization constraints, you cannot uniquely recover $\Lambda$ or $F$. The standard normalization sets $\Sigma_F = I_r$ and makes $\Lambda$ lower triangular in its first $r$ rows.

**Over-interpreting factor loadings when $N$ is small.** With small $N$, PCA loadings are noisy estimates of true factor loadings. Rotational indeterminacy means a different rotation could give equally valid (and differently interpreted) factors.

**Treating the exact factor model as gospel.** The exact model assumes $\Psi$ is diagonal — zero cross-sectional correlation in errors. In practice, macroeconomic variables have residual correlations (industry effects, regional clustering). The approximate factor model of Chamberlain and Rothschild allows weak cross-sectional correlation, which is both more realistic and still identifiable with large $N$.

## Connections

- **Builds on:** PCA (Module 0, Guide 3), covariance matrix structure (Module 0, Guide 1)
- **Leads to:** Identification constraints (Guide 2 of this module), approximate factor models (Guide 3), dynamic extension via AR factor dynamics (Module 2)
- **Related to:** Capital Asset Pricing Model (market factor = first PC of stock returns); Arbitrage Pricing Theory (factor model for asset pricing); Confirmatory Factor Analysis (restricted loadings)

## Practice Problems

1. For a 2-factor model with $N = 4$ variables and the loading matrix from the economic example, compute the implied correlation matrix by normalizing the covariance matrix. Verify that all off-diagonal correlations come exclusively from $\Lambda\Lambda'$.

2. Generate data from a factor model with 3 true factors but estimate PCA with 2, 3, and 4 components. How does the reconstruction error change? What does the residual covariance matrix look like in each case?

3. Write a function `check_factor_structure(X, r)` that tests whether the data is consistent with an $r$-factor model by examining whether the residual covariance after extracting $r$ PCs is approximately diagonal. Apply it to the simulated economic data.

4. Show algebraically that $\text{Cov}(X_{it}, X_{jt}) = \lambda_i' \lambda_j$ under the exact factor model with $\Sigma_F = I_r$. What does this imply about variables that load on the same factor?

5. FRED-MD has 127 macroeconomic series. Use the parameter counting formula to determine how many factors you could include while keeping the model over-identified.

## Further Reading

- Lawley, D. N. & Maxwell, A. E. (1971). *Factor Analysis as a Statistical Method*, 2nd ed. Classical treatment of the exact factor model.
- Bai, J. & Ng, S. (2008). "Large Dimensional Factor Analysis." *Foundations and Trends in Econometrics* 3(2). Comprehensive modern survey.
- Chamberlain, G. & Rothschild, M. (1983). "Arbitrage, Factor Structure, and Mean-Variance Analysis on Large Asset Markets." *Econometrica*. Introduces the approximate factor model.
- Anderson, T. W. (2003). *An Introduction to Multivariate Statistical Analysis*, 3rd ed., Chapter 14. Rigorous statistical treatment.
