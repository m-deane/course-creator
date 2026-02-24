# Factor-Augmented Vector Autoregression (FAVAR)

## In Brief

Factor-Augmented VAR (FAVAR) models extend standard VARs by augmenting a small set of observed variables with latent factors extracted from a large information set. This framework captures richer dynamics than pure diffusion index forecasting while remaining computationally tractable, making it ideal for policy analysis and structural identification.

> 💡 **Key Insight:** Traditional VARs use 5-10 variables due to degrees-of-freedom constraints, forcing researchers to choose which information to include. FAVAR solves this by extracting factors from hundreds of variables and modeling joint dynamics of factors plus key observables. This combines VAR's structural modeling with factor models' information aggregation.

---

## 1. Model Specification

### Standard VAR (Review)

For observed variables $Y_t$ ($M \times 1$):
$$Y_t = A_0 + A_1 Y_{t-1} + ... + A_p Y_{t-p} + u_t$$

where $u_t \sim N(0, \Sigma_u)$.

**Limitation:** With $M$ large, parameters explode: $M + M^2 p + M(M+1)/2$ parameters.

### FAVAR Specification

**Observation Equation:**
$$X_t = \Lambda^f F_t + \Lambda^y Y_t + e_t$$

where:
- $X_t$: $N \times 1$ vector of informational variables (large, $N \gg M$)
- $F_t$: $K \times 1$ vector of unobserved factors
- $Y_t$: $M \times 1$ vector of observable variables of interest
- $\Lambda^f$: $N \times K$ loadings on factors
- $\Lambda^y$: $N \times M$ loadings on observables
- $e_t$: $N \times 1$ idiosyncratic errors

**State Equation:**
$$\begin{bmatrix} F_t \\ Y_t \end{bmatrix} = \Phi(L) \begin{bmatrix} F_{t-1} \\ Y_{t-1} \end{bmatrix} + v_t$$

where:
- $\Phi(L) = \Phi_1 + \Phi_2 L + ... + \Phi_p L^{p-1}$ is a lag polynomial
- $v_t \sim N(0, Q)$ is the structural/reduced-form shock

### Compact Form

Define $G_t = [F_t', Y_t']'$ as the $(K+M) \times 1$ state vector:

$$X_t = \Lambda G_t + e_t$$
$$G_t = \Phi(L) G_{t-1} + v_t$$

This is a factor model for $X_t$ combined with a VAR for $(F_t, Y_t)$.

### Relationship to Standard Models

**Special Cases:**
- If $M = 0$: Pure dynamic factor model
- If $K = 0$: Standard VAR on $Y_t$ (no factors)
- If $\Phi$ has no lags: Static factor model

---

## 2. Intuitive Explanation

### Why FAVAR?

Suppose you want to analyze monetary policy:
- **Observable of interest** ($Y_t$): Fed funds rate
- **Want to know:** Effect on economy

**VAR approach:** Include rate + GDP + inflation + unemployment + ...
- Problem: Limited to ~5 variables due to degrees of freedom
- Miss important information (housing, credit, expectations, etc.)

**FAVAR approach:**
1. Extract factors from 100+ economic indicators
2. Model dynamics of factors + Fed funds rate jointly
3. Factors capture information in all variables

Now you can see policy effects on the entire economy through factor responses.

### Visual Representation

```
Information Set (X_t):                    State Vector (G_t):
┌─────────────────────┐                   ┌──────────────┐
│ Industrial Prod.    │                   │   Factor 1   │
│ Retail Sales        │                   │   Factor 2   │
│ Employment          │  ──▶  Extract ──▶ │   Factor 3   │
│ Housing Starts      │       Factors     │      ⋮       │
│ ... (N=127 vars)    │                   │ Fed Funds    │
└─────────────────────┘                   └──────────────┘
                                                 │
                                                 │ VAR Dynamics
                                                 ▼
                                          Next Period States
```

### Example Flow

1. **Observation:** 127 macroeconomic series $X_t$
2. **Extraction:** Compute 5 factors + include Fed funds rate explicitly
3. **Dynamics:** Estimate VAR(2) on $(F_{1t}, ..., F_{5t}, FFR_t)$
4. **Analysis:** Shock FFR, trace effects on all 6 state variables
5. **Interpretation:** Map factor responses back to original 127 series

---

## 3. Mathematical Formulation

### Two-Step Estimation (Bernanke-Boivin-Eliasz 2005)

**Step 1: Extract Factors**

Given slow-moving variables $X_t^{slow}$ (all except policy variable):
$$\tilde{F}_t = \text{First } K \text{ PCs of } X_t^{slow}$$

Normalize: $\tilde{F}_t' \tilde{F}_t / T = I_K$

**Step 2: Estimate VAR**

Form augmented state:
$$G_t = \begin{bmatrix} \tilde{F}_t \\ Y_t \end{bmatrix}$$

Estimate VAR by OLS:
$$G_t = \Phi_1 G_{t-1} + ... + \Phi_p G_{t-p} + v_t$$

### Likelihood-Based Estimation (Doz-Giannone-Reichlin 2012)

Treat as state-space model and use Kalman filter/smoother:

**Measurement Equation:**
$$X_t = \Lambda G_t + e_t, \quad e_t \sim N(0, R)$$

**Transition Equation:**
$$G_t = \Phi G_{t-1} + v_t, \quad v_t \sim N(0, Q)$$

**Estimation:**
1. Initialize parameters $(\Lambda, \Phi, R, Q)$
2. E-step: Run Kalman filter/smoother to get $E[G_t | X_{1:T}]$
3. M-step: Update parameters via maximum likelihood
4. Iterate until convergence

**Advantage:** Jointly estimates factors and dynamics, handles missing data naturally.

### Identification in State Equation

VAR in $(F_t, Y_t)$ faces standard identification issues:
- Factors only identified up to rotation
- Need normalization constraints
- Structural shocks require identification scheme

**Common normalizations:**
1. $Var(F_t) = I_K$ (factors orthonormal)
2. $\Lambda^f$ has specific structure (e.g., upper-triangular block)
3. Particular factor ordering

---

## 4. Impulse Response Analysis

### Reduced-Form Impulses

From VAR:
$$G_t = \Phi_1 G_{t-1} + ... + \Phi_p G_{t-p} + v_t$$

Compute MA representation:
$$G_t = \sum_{s=0}^\infty \Psi_s v_{t-s}$$

Impulse response of $G_t$ to $v_{t-s}$ shock:
$$\text{IR}(s) = \Psi_s$$

Compute recursively:
$$\Psi_0 = I, \quad \Psi_s = \sum_{j=1}^{\min(s,p)} \Phi_j \Psi_{s-j}$$

### Structural Impulses

Identify structural shocks $\varepsilon_t$ via:
$$v_t = B \varepsilon_t, \quad E[\varepsilon_t \varepsilon_t'] = I$$

Standard identification:
- **Cholesky:** $B$ lower-triangular, $BB' = \Sigma_v$
- **Sign restrictions:** Impose economic theory constraints
- **External instruments:** Use outside information

Structural IRF:
$$\text{SIR}(s) = \Psi_s B$$

### Mapping to Observables

Factor IRFs show responses of $F_t$ and $Y_t$.

To get responses of original variables $X_t$:
$$\Delta X_{t+s} = \Lambda^f \Delta F_{t+s} + \Lambda^y \Delta Y_{t+s}$$

where $\Delta F_{t+s}, \Delta Y_{t+s}$ come from structural IRF.

**Example:** Response of industrial production to monetary shock:
$$\text{IR}_{IP}(s) = \lambda_{IP}^f \cdot \text{SIR}_F(s) + \lambda_{IP}^y \cdot \text{SIR}_Y(s)$$

---

## 5. Code Implementation

### FAVAR Class

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import linalg
import warnings

class FAVAR:
    """
    Factor-Augmented Vector Autoregression.

    Parameters
    ----------
    n_factors : int
        Number of latent factors to extract
    n_lags : int
        Number of VAR lags
    observable_indices : array-like, optional
        Indices of variables to include as observables (Y_t)
        If None, no observables (pure dynamic factor model)
    """

    def __init__(self, n_factors=5, n_lags=2, observable_indices=None):
        self.n_factors = n_factors
        self.n_lags = n_lags
        self.observable_indices = observable_indices

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_factors)

        # Estimated components
        self.factors_ = None
        self.observables_ = None
        self.loadings_f_ = None
        self.loadings_y_ = None
        self.Phi_ = None  # VAR coefficients
        self.Sigma_v_ = None  # VAR shock covariance

    def fit(self, X):
        """
        Estimate FAVAR model using two-step procedure.

        Parameters
        ----------
        X : array-like, shape (T, N)
            Panel of informational variables

        Returns
        -------
        self
        """
        X = np.asarray(X)
        T, N = X.shape

        # Step 1: Extract factors from slow-moving variables
        if self.observable_indices is not None:
            # Remove observables from factor extraction
            X_slow = np.delete(X, self.observable_indices, axis=1)
            self.observables_ = X[:, self.observable_indices]
        else:
            X_slow = X
            self.observables_ = None

        # Standardize and extract factors
        X_scaled = self.scaler.fit_transform(X_slow)
        self.factors_ = self.pca.fit_transform(X_scaled)

        # Store loadings
        self.loadings_f_ = self.pca.components_.T

        # Estimate loadings on observables (if present)
        if self.observables_ is not None:
            # Regress X on factors and observables
            G = np.column_stack([self.factors_, self.observables_])
            self.loadings_y_ = self._estimate_loadings(X, G)

        # Step 2: Estimate VAR on (F_t, Y_t)
        if self.observables_ is not None:
            G_t = np.column_stack([self.factors_, self.observables_])
        else:
            G_t = self.factors_

        self.Phi_, self.Sigma_v_ = self._estimate_var(G_t, self.n_lags)

        return self

    def impulse_response(self, horizon=20, shock_index=0, shock_size=1.0,
                        identification='cholesky'):
        """
        Compute impulse response functions.

        Parameters
        ----------
        horizon : int
            Number of periods ahead
        shock_index : int
            Index of variable to shock
        shock_size : float
            Size of shock (in standard deviations)
        identification : str
            'cholesky' or 'none' (reduced-form)

        Returns
        -------
        irf : array-like, shape (horizon, K+M)
            Impulse responses for state variables
        """
        K = self.n_factors
        M = 0 if self.observables_ is None else self.observables_.shape[1]
        dim = K + M

        # Compute MA coefficients Psi_s
        Psi = self._compute_ma_representation(horizon)

        # Identification
        if identification == 'cholesky':
            # Structural shock matrix
            B = linalg.cholesky(self.Sigma_v_, lower=True)
        else:
            # Reduced-form (no identification)
            B = np.eye(dim)

        # Impulse response
        shock_vector = np.zeros(dim)
        shock_vector[shock_index] = shock_size

        irf = np.zeros((horizon, dim))
        for s in range(horizon):
            irf[s, :] = Psi[s] @ B @ shock_vector

        return irf

    def forecast(self, X_history, h=1):
        """
        Forecast h periods ahead.

        Parameters
        ----------
        X_history : array-like, shape (T_hist, N)
            Historical data
        h : int
            Forecast horizon

        Returns
        -------
        forecast : array-like, shape (h, N)
            Forecasted values for all variables
        """
        # Extract factors from history
        X_scaled = self.scaler.transform(X_history)
        F_hist = self.pca.transform(X_scaled)

        if self.observables_ is not None:
            Y_hist = X_history[:, self.observable_indices]
            G_hist = np.column_stack([F_hist, Y_hist])
        else:
            G_hist = F_hist

        # Forecast state vector
        G_forecast = self._forecast_var(G_hist, h)

        # Map back to observables
        if self.observables_ is not None:
            F_forecast = G_forecast[:, :self.n_factors]
            Y_forecast = G_forecast[:, self.n_factors:]

            # Reconstruct X
            Lambda = np.column_stack([self.loadings_f_, self.loadings_y_])
            X_forecast = G_forecast @ Lambda.T
        else:
            X_forecast = G_forecast @ self.loadings_f_.T

        # Inverse transform
        X_forecast = self.scaler.inverse_transform(X_forecast)

        return X_forecast

    def factor_contributions(self, X):
        """
        Decompose each variable into factor components.

        Parameters
        ----------
        X : array-like, shape (T, N)
            Data to decompose

        Returns
        -------
        contributions : dict
            Dictionary with 'factors' and 'idiosyncratic' components
        """
        X_scaled = self.scaler.transform(X)
        F = self.pca.transform(X_scaled)

        # Common component
        X_common = F @ self.loadings_f_.T

        # Idiosyncratic component
        X_idio = X_scaled - X_common

        # Inverse transform to original scale
        X_common_orig = self.scaler.inverse_transform(X_common)
        X_idio_orig = self.scaler.inverse_transform(X_idio)

        return {
            'factors': X_common_orig,
            'idiosyncratic': X_idio_orig,
            'explained_variance': self.pca.explained_variance_ratio_
        }

    def _estimate_loadings(self, X, G):
        """
        Estimate loadings by regressing X on G.

        Parameters
        ----------
        X : array-like, shape (T, N)
        G : array-like, shape (T, K+M)

        Returns
        -------
        Lambda : array-like, shape (N, K+M)
        """
        # Lambda = (G'G)^{-1} G'X
        Lambda = linalg.lstsq(G, X)[0].T
        return Lambda

    def _estimate_var(self, Y, p):
        """
        Estimate VAR(p) by OLS.

        Parameters
        ----------
        Y : array-like, shape (T, M)
            Variables
        p : int
            Lag order

        Returns
        -------
        Phi : list of array-like
            VAR coefficient matrices [Phi_1, ..., Phi_p]
        Sigma : array-like, shape (M, M)
            Residual covariance
        """
        T, M = Y.shape

        # Construct lagged matrix
        Y_lagged = self._create_var_matrix(Y, p)

        # OLS estimation
        Y_dep = Y[p:, :]
        Phi_stacked = linalg.lstsq(Y_lagged, Y_dep)[0]  # Shape: (M*p, M)

        # Reshape into list of matrices
        Phi = [Phi_stacked[i*M:(i+1)*M, :].T for i in range(p)]

        # Residual covariance
        fitted = Y_lagged @ Phi_stacked
        resid = Y_dep - fitted
        Sigma = (resid.T @ resid) / (T - p - M*p)

        return Phi, Sigma

    def _create_var_matrix(self, Y, p):
        """
        Create matrix of lagged variables for VAR estimation.

        Parameters
        ----------
        Y : array-like, shape (T, M)
        p : int
            Number of lags

        Returns
        -------
        Y_lagged : array-like, shape (T-p, M*p)
        """
        T, M = Y.shape
        Y_lagged = np.zeros((T - p, M * p))

        for lag in range(1, p + 1):
            Y_lagged[:, (lag-1)*M:lag*M] = Y[p-lag:-lag, :]

        return Y_lagged

    def _compute_ma_representation(self, horizon):
        """
        Compute MA representation coefficients Psi_s.

        Parameters
        ----------
        horizon : int

        Returns
        -------
        Psi : list of array-like
            MA coefficient matrices [Psi_0, ..., Psi_{horizon-1}]
        """
        M = self.Phi_[0].shape[0]
        Psi = [np.eye(M)]

        for s in range(1, horizon):
            Psi_s = np.zeros((M, M))
            for j in range(min(s, self.n_lags)):
                Psi_s += self.Phi_[j] @ Psi[s - j - 1]
            Psi.append(Psi_s)

        return Psi

    def _forecast_var(self, Y_hist, h):
        """
        Forecast VAR h steps ahead.

        Parameters
        ----------
        Y_hist : array-like, shape (T, M)
            Historical values
        h : int
            Horizon

        Returns
        -------
        forecast : array-like, shape (h, M)
        """
        T, M = Y_hist.shape
        forecast = np.zeros((h, M))

        # Use last p observations as initial conditions
        Y_extended = np.vstack([Y_hist[-self.n_lags:], forecast])

        for t in range(h):
            y_new = np.zeros(M)
            for lag in range(self.n_lags):
                y_new += self.Phi_[lag] @ Y_extended[self.n_lags + t - lag - 1, :]
            Y_extended[self.n_lags + t, :] = y_new
            forecast[t, :] = y_new

        return forecast
```

### Example: Monetary Policy Analysis

```python
import matplotlib.pyplot as plt

# Generate synthetic data mimicking monetary FAVAR
np.random.seed(789)
T, N = 400, 50
K, M = 4, 1  # 4 factors + 1 observable (policy rate)

# True VAR structure for (F_t, Y_t)
Phi1_true = np.array([
    [0.8, 0.0, 0.0, 0.0, -0.2],  # F1: responds negatively to policy
    [0.0, 0.7, 0.0, 0.0, -0.15],  # F2
    [0.0, 0.0, 0.9, 0.0, -0.1],   # F3
    [0.0, 0.0, 0.0, 0.6, -0.05],  # F4
    [0.0, 0.0, 0.0, 0.0, 0.9]     # Policy rate (persistent)
])

# Simulate VAR
G = np.zeros((T, K + M))
G[0] = np.random.randn(K + M) * 0.5

for t in range(1, T):
    G[t] = Phi1_true @ G[t-1] + np.random.randn(K + M) * 0.3

# Extract components
F_true = G[:, :K]
Y_true = G[:, K:]

# Generate observables
Lambda_f_true = np.random.randn(N-M, K) / np.sqrt(K)
Lambda_y_true = np.random.randn(N-M, M) * 0.5

X_from_factors = F_true @ Lambda_f_true.T + Y_true @ Lambda_y_true.T
X_from_factors += np.random.randn(T, N-M) * 0.5

# Combine with policy rate
X = np.column_stack([X_from_factors, Y_true])

# Estimate FAVAR
favar = FAVAR(n_factors=4, n_lags=1, observable_indices=[N-1])
favar.fit(X)

# Analyze monetary policy shock (shock to last variable)
irf = favar.impulse_response(horizon=30, shock_index=4, shock_size=1.0,
                              identification='cholesky')

# Plot impulse responses
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

labels = ['Factor 1 (Real Activity)', 'Factor 2 (Prices)',
          'Factor 3 (Credit)', 'Factor 4 (Expectations)',
          'Policy Rate']

for i in range(5):
    axes[i].plot(irf[:, i], linewidth=2)
    axes[i].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[i].set_title(labels[i])
    axes[i].set_xlabel('Periods')
    axes[i].set_ylabel('Response')
    axes[i].grid(True, alpha=0.3)

# Show explained variance
var_exp = favar.pca.explained_variance_ratio_
axes[5].bar(range(1, len(var_exp)+1), var_exp)
axes[5].set_title('Variance Explained by Factors')
axes[5].set_xlabel('Factor')
axes[5].set_ylabel('Proportion')
axes[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('favar_monetary_policy.png', dpi=300, bbox_inches='tight')
plt.show()

print("FAVAR Estimation Complete")
print(f"Total variance explained: {var_exp.sum():.3f}")
print(f"\nVAR residual covariance:\n{favar.Sigma_v_}")
```

---

## 6. Common Pitfalls

### 1. Factor Identification

**Problem:** Factors are only identified up to rotation. Different estimation runs may yield different factor interpretations.

**Solution:**
- Use consistent normalization (e.g., factors ordered by variance explained)
- Focus on space spanned by factors, not individual factors
- Use structural identification for economic interpretation

### 2. Observable Selection

**Mistake:** Including too many variables as observables defeats the dimensionality reduction purpose.

**Solution:** Include only variables essential for structural analysis (e.g., policy instrument, key targets). Let factors capture other information.

### 3. Lag Length Selection

**Problem:** Too few lags miss dynamics; too many overfit.

**Solution:**
- Information criteria (AIC, BIC) on state-space VAR
- Test lag significance
- Consider forecast performance

### 4. Structural Identification

**Mistake:** Assuming Cholesky ordering identifies economic shocks without justification.

**Solution:**
- Use sign restrictions based on economic theory
- Employ external instruments (high-frequency identification)
- Test robustness to ordering

---

## 7. Connections

### Builds On
- **Diffusion Index Forecasting:** Pure forecasting without dynamics
- **VAR Models:** Multivariate time series dynamics
- **State-Space Models:** Kalman filtering framework

### Leads To
- **Structural FAVAR:** Identification of economic shocks
- **Time-Varying FAVARs:** Allow parameters to drift
- **FAVAR with Stochastic Volatility:** Heteroskedastic shocks

### Related Methods
- **Large Bayesian VARs:** Alternative to handle many variables
- **Global VAR (GVAR):** Multi-country extension
- **Factor-Augmented Error Correction Models (FAVECM):** Cointegration

---

## 8. Practice Problems

### Conceptual

1. **Why include some variables as observables rather than extracting all information through factors?**

2. **In a monetary FAVAR, would you include the policy rate as an observable or let it be captured by factors? Why?**

3. **How does FAVAR address the degrees-of-freedom problem that plagues large VARs?**

### Mathematical

4. **Derive the forecasting equation for $X_{T+h}$ given current state $G_T$.**

5. **Show that the FAVAR likelihood can be written as a product of factor model likelihood and VAR likelihood.**

6. **Prove that impulse responses of $X_t$ to structural shocks are given by:**
   $$\frac{\partial X_{t+s}}{\partial \varepsilon_t} = \Lambda^f \Psi_s^F B^F + \Lambda^y \Psi_s^Y B^Y$$

### Implementation

7. **Compare FAVAR forecasts with:**
   - Small VAR on selected variables
   - Pure diffusion index forecast
   - Large Bayesian VAR

   Which performs best for different horizons?

8. **Implement variance decomposition for FAVAR: What fraction of variance in each $X_i$ is due to each structural shock?**

9. **Add a "fast-moving" observable that is not used for factor extraction (e.g., asset prices). How does this affect impulse responses?**

### Advanced

10. **Implement likelihood-based estimation using the EM algorithm with Kalman filtering.**

11. **Extend FAVAR to allow time-varying parameters using a state-space representation with stochastic coefficients.**

---

## 9. Further Reading

### Foundational Papers

- **Bernanke, B.S., Boivin, J. & Eliasz, P. (2005).** "Measuring the Effects of Monetary Policy: A Factor-Augmented Vector Autoregressive (FAVAR) Approach." *Quarterly Journal of Economics*, 120(1), 387-422.
  - Original FAVAR paper for monetary policy analysis

- **Stock, J.H. & Watson, M.W. (2005).** "Implications of Dynamic Factor Models for VAR Analysis." NBER Working Paper 11467.
  - Theoretical foundations linking factor models and VARs

### Estimation Methods

- **Doz, C., Giannone, D. & Reichlin, L. (2012).** "A Quasi-Maximum Likelihood Approach for Large, Approximate Dynamic Factor Models." *Review of Economics and Statistics*, 94(4), 1014-1024.
  - EM algorithm for FAVAR estimation

- **Banbura, M., Giannone, D. & Reichlin, L. (2010).** "Large Bayesian Vector Auto Regressions." *Journal of Applied Econometrics*, 25(1), 71-92.
  - Bayesian alternative to FAVAR

### Applications

- **Boivin, J., Giannoni, M.P. & Mihov, I. (2009).** "Sticky Prices and Monetary Policy: Evidence from Disaggregated US Data." *American Economic Review*, 99(1), 350-384.
  - Sectoral heterogeneity in monetary transmission via FAVAR

- **Forni, M. & Gambetti, L. (2010).** "The Dynamic Effects of Monetary Policy: A Structural Factor Model Approach." *Journal of Monetary Economics*, 57(2), 203-216.
  - Structural identification in FAVARs

### Surveys

- **Stock, J.H. & Watson, M.W. (2016).** "Dynamic Factor Models, Factor-Augmented Vector Autoregressions, and Structural Vector Autoregressions in Macroeconomics." *Handbook of Macroeconomics*, Vol 2A, Chapter 8.
  - Comprehensive survey of FAVAR methods and applications
