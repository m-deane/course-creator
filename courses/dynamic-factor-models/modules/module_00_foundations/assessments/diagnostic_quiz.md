# Module 0: Diagnostic Assessment

## Instructions

This diagnostic assesses your readiness for the Dynamic Factor Models course. Complete all sections without referring to external resources. Your score will indicate which foundation topics to review.

**Time Estimate:** 30-45 minutes
**Passing Score:** 80% overall, 70% minimum in each section

---

## Section A: Matrix Algebra (25 points)

### Question A1 (5 points)

Given a symmetric matrix $A$ with eigenvalues $\lambda_1 = 4$, $\lambda_2 = 2$, $\lambda_3 = 1$:

a) What is $\text{trace}(A)$?
b) What is $\det(A)$?
c) Is $A$ positive definite? Why or why not?
d) What are the eigenvalues of $A^{-1}$?
e) What are the eigenvalues of $A^2$?

---

### Question A2 (5 points)

For a data matrix $X \in \mathbb{R}^{100 \times 20}$ (100 observations, 20 variables):

a) What are the dimensions of $X'X$?
b) What are the dimensions of $XX'$?
c) If $X = U\Sigma V'$ is the SVD, what is the maximum possible rank of $X$?
d) How do the non-zero eigenvalues of $X'X$ relate to those of $XX'$?
e) What is the relationship between singular values of $X$ and eigenvalues of $X'X$?

---

### Question A3 (5 points)

Let $\Sigma$ be a covariance matrix. Explain why:

a) $\Sigma$ must be symmetric.
b) $\Sigma$ must be positive semi-definite.
c) The diagonal elements of $\Sigma$ must be non-negative.
d) Write the condition for $\Sigma$ to be positive definite (not just semi-definite).

---

### Question A4 (5 points)

Compute the eigendecomposition of:
$$A = \begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$$

Show your work.

---

### Question A5 (5 points)

Given vectors $u = [1, 2, 3]'$ and $v = [4, 5, 6]'$:

a) Compute the outer product $uv'$.
b) What is the rank of $uv'$?
c) Compute the inner product $u'v$.
d) Verify that $\text{trace}(uv') = u'v$.

---

## Section B: Time Series (25 points)

### Question B1 (5 points)

Define weak (covariance) stationarity. List the three conditions that must be satisfied.

---

### Question B2 (5 points)

For the AR(1) process $y_t = 0.8 y_{t-1} + \varepsilon_t$ where $\varepsilon_t \sim WN(0, 1)$:

a) Is this process stationary? Why?
b) What is $E[y_t]$?
c) What is $\text{Var}(y_t)$?
d) What is $\text{Corr}(y_t, y_{t-2})$?
e) Sketch the shape of the autocorrelation function (ACF).

---

### Question B3 (5 points)

For the process $y_t = y_{t-1} + \varepsilon_t$ (random walk):

a) Is this process stationary? Prove your answer.
b) What transformation would make it stationary?
c) What is the ACF of the transformed series?

---

### Question B4 (5 points)

Match each ACF/PACF pattern to the appropriate model:

| Pattern | Model Options |
|---------|---------------|
| ACF decays geometrically, PACF cuts off after lag 2 | |
| ACF cuts off after lag 1, PACF decays | |
| Both ACF and PACF decay gradually | |

Model Options: AR(2), MA(1), ARMA(1,1)

---

### Question B5 (5 points)

Write the AR(2) process $y_t = 0.5y_{t-1} + 0.3y_{t-2} + \varepsilon_t$ in state-space form:

$$\alpha_t = T\alpha_{t-1} + R\eta_t$$
$$y_t = Z\alpha_t$$

Define the state vector $\alpha_t$ and matrices $T$, $R$, $Z$.

---

## Section C: Principal Component Analysis (25 points)

### Question C1 (5 points)

Explain in words what the first principal component represents. Include:
- What objective it optimizes
- What constraint is imposed
- How it relates to the covariance matrix

---

### Question C2 (5 points)

Given a covariance matrix with eigenvalues $\lambda_1 = 10$, $\lambda_2 = 5$, $\lambda_3 = 3$, $\lambda_4 = 2$:

a) What proportion of total variance is explained by the first PC?
b) What proportion is explained by the first two PCs?
c) If you retain components explaining at least 80% of variance, how many do you keep?

---

### Question C3 (5 points)

You have data matrix $X$ (centered) with $T = 50$ observations and $N = 100$ variables.

a) What is the rank of the sample covariance matrix $\frac{1}{T}X'X$?
b) How many non-zero eigenvalues does it have?
c) Would you compute PCA via covariance eigendecomposition or SVD? Why?

---

### Question C4 (5 points)

The loading for variable $j$ on component $k$ is $v_{jk} = 0.8$.

a) If the component score increases by 1 unit, how does variable $j$ change (approximately)?
b) If $v_{jk} = -0.8$, what would this indicate about the relationship?
c) If $v_{jk} \approx 0$, what does this mean?

---

### Question C5 (5 points)

Explain the difference between:
a) PC scores and PC loadings
b) PCA and Factor Analysis (mention at least two differences)

---

## Section D: Programming (25 points)

### Question D1 (5 points)

Write Python code to:
1. Generate a 200x50 matrix of random normal data
2. Center the data (subtract column means)
3. Compute the sample covariance matrix

```python
import numpy as np

# Your code here
```

---

### Question D2 (5 points)

Write Python code to extract the first 3 principal components from a centered data matrix $X$ using SVD.

```python
import numpy as np

def extract_pcs(X, n_components=3):
    """
    Extract principal components using SVD.

    Parameters
    ----------
    X : ndarray, shape (T, N)
        Centered data matrix

    Returns
    -------
    scores : ndarray, shape (T, n_components)
    loadings : ndarray, shape (N, n_components)
    variance_explained : ndarray, shape (n_components,)
    """
    # Your code here
    pass
```

---

### Question D3 (5 points)

Write Python code to simulate an AR(1) process with $\phi = 0.7$, $\sigma = 1$, $T = 200$.

```python
import numpy as np

def simulate_ar1(phi, sigma, T, seed=42):
    """
    Simulate AR(1) process: y_t = phi * y_{t-1} + epsilon_t

    Returns
    -------
    y : ndarray, shape (T,)
    """
    # Your code here
    pass
```

---

### Question D4 (5 points)

Write Python code to compute the sample autocorrelation function up to lag 10.

```python
import numpy as np

def compute_acf(y, max_lag=10):
    """
    Compute sample ACF.

    Returns
    -------
    acf : ndarray, shape (max_lag + 1,)
        ACF values for lags 0, 1, ..., max_lag
    """
    # Your code here
    pass
```

---

### Question D5 (5 points)

Debug the following code that attempts to compute eigenvalues of a matrix:

```python
import numpy as np

def get_sorted_eigenvalues(A):
    """Return eigenvalues sorted in descending order."""
    eigenvalues, eigenvectors = np.linalg.eig(A)
    sorted_eigenvalues = eigenvalues.sort()  # Bug 1
    return sorted_eigenvalues

# Test
A = np.array([[2, 1], [1, 2]])
print(get_sorted_eigenvalues(A))  # Should print [3, 1]
```

Identify the bugs and provide corrected code.

---

## Scoring Guide

| Section | Points | Minimum to Pass |
|---------|--------|-----------------|
| A: Matrix Algebra | 25 | 17.5 (70%) |
| B: Time Series | 25 | 17.5 (70%) |
| C: PCA | 25 | 17.5 (70%) |
| D: Programming | 25 | 17.5 (70%) |
| **Total** | **100** | **80 (80%)** |

---

## After Completing

### Score Interpretation

| Score | Recommendation |
|-------|----------------|
| 90-100 | Ready for Module 1 |
| 80-89 | Review flagged sections, then proceed |
| 70-79 | Complete corresponding review guides |
| Below 70 | Work through all Module 0 materials |

### Review Materials by Section

| Section | If Score < 70%, Review |
|---------|----------------------|
| A | `guides/01_matrix_algebra_review.md` |
| B | `guides/02_time_series_basics.md` |
| C | `guides/03_pca_refresher.md` |
| D | `notebooks/01_foundations_review.ipynb` |

---

## Answer Key

*Available in `solutions/diagnostic_answers.md` (instructor access only)*
