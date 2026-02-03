# Mini-Project: Kalman Filter Implementation

## Overview

This project challenges you to implement the Kalman filter and smoother from scratch for a dynamic factor model in state-space form. You will verify your implementation against statsmodels, ensuring numerical accuracy and proper handling of edge cases.

**Learning Objectives:**
- Implement the prediction and update steps of the Kalman filter
- Compute log-likelihood via prediction error decomposition
- Code the Rauch-Tung-Striebel smoother for full-sample estimates
- Validate against established libraries
- Handle numerical stability issues

**Time Estimate:** 6-8 hours

**Difficulty:** Core

---

## Project Specification

### Problem Statement

Implement a complete Kalman filter and smoother for the following AR(2) dynamic factor model:

**Measurement Equation:**
$$X_t = \Lambda F_t + e_t, \quad e_t \sim N(0, \Sigma_e)$$

**Transition Equation (Factor Dynamics):**
$$F_t = \Phi_1 F_{t-1} + \Phi_2 F_{t-2} + \eta_t, \quad \eta_t \sim N(0, Q)$$

**State-Space Form:**
- **Measurement:** $X_t = Z \alpha_t + \varepsilon_t$, $\varepsilon_t \sim N(0, H)$
- **Transition:** $\alpha_t = T \alpha_{t-1} + R \eta_t$, $\eta_t \sim N(0, Q)$

where the state vector is augmented:
$$\alpha_t = \begin{bmatrix} F_t \\ F_{t-1} \end{bmatrix}$$

---

## Requirements

### Core Requirements (Must Complete)

#### 1. State-Space Representation (10 points)

Implement a function that converts DFM parameters into state-space form:

```python
def construct_state_space(Lambda, Phi1, Phi2, Sigma_e, Q):
    """
    Convert DFM with AR(2) factors to state-space form.

    Parameters
    ----------
    Lambda : ndarray, shape (N, r)
        Factor loadings
    Phi1 : ndarray, shape (r, r)
        First-order transition matrix
    Phi2 : ndarray, shape (r, r)
        Second-order transition matrix
    Sigma_e : ndarray, shape (N, N)
        Idiosyncratic error covariance (diagonal)
    Q : ndarray, shape (r, r)
        Factor innovation covariance

    Returns
    -------
    Z : ndarray, shape (N, 2r)
        Measurement matrix
    T : ndarray, shape (2r, 2r)
        Transition matrix
    R : ndarray, shape (2r, r)
        Selection matrix
    H : ndarray, shape (N, N)
        Measurement error covariance
    Q : ndarray, shape (r, r)
        State innovation covariance
    """
    pass
```

**Specifications:**
- State dimension: $2r$ (current and lagged factors)
- Handle companion form for AR(2) dynamics
- Validate positive definiteness of covariance matrices

---

#### 2. Kalman Filter (30 points)

Implement the complete Kalman filter recursion:

```python
class KalmanFilter:
    """
    Kalman filter for state-space models.
    """

    def __init__(self, Z, T, R, H, Q):
        """
        Initialize Kalman filter with system matrices.

        Parameters
        ----------
        Z : ndarray, shape (N, m)
            Measurement matrix
        T : ndarray, shape (m, m)
            Transition matrix
        R : ndarray, shape (m, r)
            Selection matrix
        H : ndarray, shape (N, N)
            Measurement error covariance
        Q : ndarray, shape (r, r)
            State innovation covariance
        """
        self.Z = Z
        self.T = T
        self.R = R
        self.H = H
        self.Q = Q

        # Precompute RQR' for efficiency
        self.RQR = R @ Q @ R.T

    def filter(self, X, a0=None, P0=None):
        """
        Run Kalman filter on observations X.

        Parameters
        ----------
        X : ndarray, shape (T, N)
            Observations
        a0 : ndarray, shape (m,), optional
            Initial state mean (default: zeros)
        P0 : ndarray, shape (m, m), optional
            Initial state covariance (default: large variance)

        Returns
        -------
        results : dict
            - 'filtered_state': ndarray (T, m) - a_{t|t}
            - 'filtered_cov': ndarray (T, m, m) - P_{t|t}
            - 'predicted_state': ndarray (T, m) - a_{t|t-1}
            - 'predicted_cov': ndarray (T, m, m) - P_{t|t-1}
            - 'prediction_error': ndarray (T, N) - v_t
            - 'prediction_error_cov': ndarray (T, N, N) - F_t
            - 'kalman_gain': ndarray (T, m, N) - K_t
            - 'loglikelihood': float
        """
        T, N = X.shape
        m = self.Z.shape[1]

        # Initialize storage
        a_pred = np.zeros((T, m))
        P_pred = np.zeros((T, m, m))
        a_filt = np.zeros((T, m))
        P_filt = np.zeros((T, m, m))
        v = np.zeros((T, N))
        F = np.zeros((T, N, N))
        K = np.zeros((T, m, N))

        # Initial conditions
        if a0 is None:
            a0 = np.zeros(m)
        if P0 is None:
            # Diffuse initialization: large variance
            P0 = np.eye(m) * 1e6

        loglik = 0.0

        # Kalman filter recursion
        for t in range(T):
            # YOUR CODE HERE: Implement prediction and update steps
            pass

        return {
            'filtered_state': a_filt,
            'filtered_cov': P_filt,
            'predicted_state': a_pred,
            'predicted_cov': P_pred,
            'prediction_error': v,
            'prediction_error_cov': F,
            'kalman_gain': K,
            'loglikelihood': loglik
        }
```

**Implementation Checklist:**
- [ ] Prediction step: Compute $\hat{\alpha}_{t|t-1}$ and $P_{t|t-1}$
- [ ] Prediction error: $v_t = X_t - Z \hat{\alpha}_{t|t-1}$
- [ ] Prediction error covariance: $F_t = Z P_{t|t-1} Z' + H$
- [ ] Kalman gain: $K_t = P_{t|t-1} Z' F_t^{-1}$
- [ ] Update step: $\hat{\alpha}_{t|t} = \hat{\alpha}_{t|t-1} + K_t v_t$
- [ ] Updated covariance: $P_{t|t} = P_{t|t-1} - K_t Z P_{t|t-1}$ (numerically stable form)
- [ ] Log-likelihood accumulation
- [ ] Handle missing observations (optional extension)

**Numerical Stability Tips:**
- Use `scipy.linalg.cho_factor` and `cho_solve` for $F_t^{-1}$
- Compute $P_{t|t}$ using Joseph form: $(I - K_t Z) P_{t|t-1} (I - K_t Z)' + K_t H K_t'$
- Check for singular $F_t$ before inversion

---

#### 3. Kalman Smoother (25 points)

Implement the Rauch-Tung-Striebel backward recursion:

```python
def kalman_smoother(self, filter_results):
    """
    Rauch-Tung-Striebel smoother.

    Parameters
    ----------
    filter_results : dict
        Output from self.filter()

    Returns
    -------
    results : dict
        - 'smoothed_state': ndarray (T, m) - a_{t|T}
        - 'smoothed_cov': ndarray (T, m, m) - P_{t|T}
        - 'smoothed_state_cov': ndarray (T, m, m) - Cov(α_t, α_{t-1} | X_{1:T})
    """
    a_filt = filter_results['filtered_state']
    P_filt = filter_results['filtered_cov']
    a_pred = filter_results['predicted_state']
    P_pred = filter_results['predicted_cov']

    T, m = a_filt.shape

    # Initialize at T
    a_smooth = np.zeros((T, m))
    P_smooth = np.zeros((T, m, m))
    P_smooth_lag = np.zeros((T-1, m, m))

    a_smooth[-1] = a_filt[-1]
    P_smooth[-1] = P_filt[-1]

    # Backward recursion
    for t in range(T-2, -1, -1):
        # YOUR CODE HERE: Implement smoothing recursion
        pass

    return {
        'smoothed_state': a_smooth,
        'smoothed_cov': P_smooth,
        'smoothed_state_cov': P_smooth_lag
    }
```

**Smoothing Recursion:**
$$J_t = P_{t|t} T' P_{t+1|t}^{-1}$$
$$\hat{\alpha}_{t|T} = \hat{\alpha}_{t|t} + J_t (\hat{\alpha}_{t+1|T} - \hat{\alpha}_{t+1|t})$$
$$P_{t|T} = P_{t|t} + J_t (P_{t+1|T} - P_{t+1|t}) J_t'$$

---

#### 4. Validation Against statsmodels (20 points)

Write comprehensive tests comparing your implementation to statsmodels:

```python
import pytest
from statsmodels.tsa.statespace.mlemodel import MLEModel

def test_kalman_filter_accuracy():
    """
    Test filter output matches statsmodels within numerical tolerance.
    """
    # Generate synthetic data
    np.random.seed(42)
    T, N, r = 100, 10, 2

    # True parameters
    Lambda = np.random.randn(N, r)
    Phi1 = np.array([[0.8, 0.1], [0.1, 0.6]])
    Phi2 = np.array([[0.1, 0.0], [0.0, 0.1]])
    Sigma_e = np.diag(np.random.uniform(0.5, 1.5, N))
    Q = np.eye(r)

    # YOUR CODE HERE:
    # 1. Simulate data from DFM
    # 2. Run your Kalman filter
    # 3. Run statsmodels Kalman filter
    # 4. Compare filtered states, log-likelihood

    # Assertions
    assert np.allclose(your_filtered_state, sm_filtered_state, atol=1e-6)
    assert np.allclose(your_loglik, sm_loglik, atol=1e-4)
```

**Test Cases:**
- [ ] Simple AR(1) model (exact comparison)
- [ ] Full AR(2) DFM with multiple factors
- [ ] Edge case: Near-singular covariance matrices
- [ ] Edge case: Very persistent factors (Phi eigenvalues near 1)
- [ ] Log-likelihood values match

**Tolerance Thresholds:**
- Filtered states: $\|a_{your} - a_{sm}\| < 10^{-6}$
- Covariances: $\|P_{your} - P_{sm}\| < 10^{-5}$
- Log-likelihood: $|\log L_{your} - \log L_{sm}| < 10^{-4}$

---

#### 5. Application: Factor Extraction (15 points)

Apply your Kalman filter to extract factors from real data:

```python
def extract_factors_dfm(X, n_factors=2, factor_order=2):
    """
    Extract dynamic factors using Kalman filter.

    Steps:
    1. Initialize parameters with PCA
    2. Run Kalman filter
    3. Run Kalman smoother
    4. Return smoothed factor estimates

    Parameters
    ----------
    X : ndarray, shape (T, N)
        Data matrix (standardized)
    n_factors : int
        Number of factors
    factor_order : int
        AR order for factors

    Returns
    -------
    factors : ndarray, shape (T, n_factors)
        Smoothed factor estimates
    """
    # YOUR CODE HERE
    pass
```

**Deliverable:**
- Load FRED-MD dataset (5 variables, 200 observations)
- Extract 2 dynamic factors with AR(2) dynamics
- Plot filtered vs smoothed factor estimates
- Interpret factor loadings (which variables load strongly?)

---

### Extension Options (Choose 1, 10 points)

#### Option A: Missing Data Handling

Extend your Kalman filter to handle missing observations:

```python
def filter_missing(self, X, missing_mask):
    """
    Kalman filter with missing observations.

    Parameters
    ----------
    missing_mask : ndarray, shape (T, N), dtype=bool
        True where observations are missing
    """
    # Hint: Modify Z and H for each time period
    pass
```

Test on data with 20% randomly missing values.

---

#### Option B: Diffuse Initialization

Implement exact diffuse initialization for non-stationary initial conditions:

```python
def initialize_diffuse(self, X):
    """
    Exact diffuse initialization for unit root factors.

    Uses augmented Kalman filter to handle infinite initial variance.
    """
    pass
```

Reference: Durbin & Koopman (2012), Chapter 5.

---

#### Option C: Univariate Treatment

Implement the univariate approach for high-dimensional $N$:

```python
def filter_univariate(self, X):
    """
    Process observations one at a time for computational efficiency.

    Complexity: O(T * N * m^2) instead of O(T * N^3)
    """
    pass
```

Test speedup on $N = 100$ variables.

---

## Evaluation Rubric

### Functionality (40 points)

| Criterion | Excellent (36-40) | Good (30-35) | Adequate (24-29) | Needs Work (0-23) |
|-----------|-------------------|--------------|------------------|-------------------|
| State-space construction | Perfect companion form, all validations | Correct matrices, minor validation gaps | Mostly correct, some edge cases fail | Major errors in construction |
| Kalman filter | All recursions correct, numerically stable | Core recursions correct, minor stability issues | Basic functionality, some numerical errors | Incorrect implementation |
| Kalman smoother | Perfect backward recursion | Correct algorithm, minor issues | Basic smoothing works | Major errors |
| Validation | Passes all tests, tight tolerances | Passes most tests | Some tests fail | Many failures |

**Specific Checks:**
- [ ] Prediction step formula correct (5 pts)
- [ ] Update step formula correct (5 pts)
- [ ] Log-likelihood computed correctly (5 pts)
- [ ] Smoother recursion correct (5 pts)
- [ ] Matches statsmodels output (10 pts)
- [ ] Edge cases handled (5 pts)
- [ ] Real data application works (5 pts)

---

### Code Quality (25 points)

| Criterion | Excellent (23-25) | Good (19-22) | Adequate (15-18) | Needs Work (0-14) |
|-----------|-------------------|--------------|------------------|-------------------|
| Documentation | Comprehensive docstrings, inline comments | Good docstrings, some comments | Basic documentation | Minimal documentation |
| Structure | Clean classes, logical organization | Mostly organized, some clutter | Functional but messy | Poor organization |
| Efficiency | Optimal algorithms, vectorized | Mostly efficient | Some inefficiencies | Many inefficiencies |
| Error handling | Comprehensive validation | Key validations present | Basic checks | No validation |

**Specific Checks:**
- [ ] NumPy docstring format (5 pts)
- [ ] Type hints for function signatures (3 pts)
- [ ] Descriptive variable names (3 pts)
- [ ] No code duplication (4 pts)
- [ ] Proper matrix operations (no loops where unnecessary) (5 pts)
- [ ] Input validation (dimension checks, positive definiteness) (5 pts)

---

### Numerical Stability (20 points)

| Criterion | Excellent (18-20) | Good (15-17) | Adequate (12-14) | Needs Work (0-11) |
|-----------|-------------------|--------------|------------------|-------------------|
| Matrix operations | Uses Cholesky decomposition, symmetric updates | Mostly stable methods | Some stability issues | Unstable inversions |
| Edge cases | Handles near-singular matrices | Most cases work | Some failures | Frequent numerical errors |
| Conditioning | Monitors condition numbers | Aware of issues | No monitoring | Oblivious to problems |

**Test Cases:**
- [ ] Very persistent factors ($\lambda_{max}(\Phi) = 0.99$)
- [ ] Large observation noise ($H_{ii} = 100$)
- [ ] Near-collinear loadings
- [ ] Long time series ($T = 1000$)

---

### Analysis & Interpretation (15 points)

| Criterion | Excellent (14-15) | Good (11-13) | Adequate (8-10) | Needs Work (0-7) |
|-----------|-------------------|--------------|------------------|-------------------|
| Factor interpretation | Clear economic meaning, loading analysis | Good interpretation | Basic comments | No interpretation |
| Filtered vs smoothed | Explains differences, shows uncertainty bands | Notes differences | Minimal discussion | No comparison |
| Insights | Novel observations, connects to theory | Standard observations | Superficial | None |

**Expected Deliverables:**
- Plot comparing filtered and smoothed estimates
- Discussion of when smoothing helps most (volatile periods?)
- Interpretation of which variables load on each factor
- Connection to economic concepts (business cycle, inflation factor, etc.)

---

## Submission Instructions

### File Structure

```
mini_project_kalman/
├── kalman_filter.py          # Main implementation
├── tests/
│   ├── test_filter.py        # Unit tests
│   ├── test_smoother.py
│   └── test_validation.py    # Comparison with statsmodels
├── notebooks/
│   └── analysis.ipynb        # Application and interpretation
├── data/
│   └── fred_md_subset.csv    # Data for application
├── results/
│   ├── filtered_factors.png
│   └── smoother_comparison.png
├── requirements.txt
└── README.md                 # Setup instructions
```

### Submission Checklist

- [ ] All code runs without errors (run `pytest tests/`)
- [ ] README includes setup instructions
- [ ] requirements.txt lists all dependencies
- [ ] Notebook has clear narrative and interpretations
- [ ] Validation tests pass with tight tolerances
- [ ] Code follows PEP 8 style guidelines
- [ ] No hardcoded paths (use relative paths)

### How to Submit

1. Create a private GitHub repository
2. Push all code, tests, and results
3. Share repository link with instructor
4. Include commit showing tests passing

**Deadline:** [To be announced]

**Late Policy:** 10% penalty per day, up to 3 days

---

## Resources

### Required Reading

- Hamilton (1994). *Time Series Analysis*, Chapter 13 (Kalman Filter)
- Harvey (1989). *Forecasting, Structural Time Series Models and the Kalman Filter*, Chapters 3-4

### Recommended

- Durbin & Koopman (2012). *Time Series Analysis by State Space Methods*, 2nd ed., Chapters 4-5
- Simon, D. (2006). *Optimal State Estimation: Kalman, H∞, and Nonlinear Approaches*, Chapter 5

### Software Documentation

- [statsmodels.tsa.statespace documentation](https://www.statsmodels.org/stable/statespace.html)
- [NumPy linear algebra routines](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [SciPy linear algebra](https://docs.scipy.org/doc/scipy/reference/linalg.html)

### Example Notebooks

See `notebooks/02_kalman_filter_implementation.ipynb` for worked examples.

---

## Common Pitfalls

1. **Matrix dimension errors:** Double-check shapes at each step
2. **Unstable covariance updates:** Use Joseph form for $P_{t|t}$
3. **Incorrect companion form:** Ensure identity matrix in correct position
4. **Forgetting RQR' precomputation:** Slows down filter significantly
5. **Not handling numerical issues:** Check condition numbers of $F_t$
6. **Incorrect log-likelihood:** Remember to include $-\frac{N}{2}\log(2\pi)$ term
7. **Smoother initialization:** Start from $t=T-1$, not $T$
8. **Confusing notation:** $a_{t|t-1}$ is prediction, $a_{t|t}$ is filtered estimate

---

## Grading Summary

| Component | Points | Weight |
|-----------|--------|--------|
| Functionality | 40 | 40% |
| Code Quality | 25 | 25% |
| Numerical Stability | 20 | 20% |
| Analysis & Interpretation | 15 | 15% |
| **Total** | **100** | **100%** |

**Minimum to Pass:** 70/100

**Grade Boundaries:**
- A: 90-100
- B: 80-89
- C: 70-79
- F: Below 70

---

## Academic Integrity

- You may discuss high-level approaches with classmates
- **All code must be your own**
- You may use statsmodels for validation, not as your primary implementation
- Cite any external resources used (StackOverflow, papers, etc.)
- AI tools (Copilot, ChatGPT) permitted for syntax help, not algorithm design

**Violations will result in zero credit for the project.**

---

## Getting Help

- Office hours: [Schedule]
- Discussion forum: [Link]
- Email instructor: [Email]

**Expected response time:** 24 hours on weekdays

---

*Good luck! The Kalman filter is a beautiful algorithm—enjoy implementing it.*
