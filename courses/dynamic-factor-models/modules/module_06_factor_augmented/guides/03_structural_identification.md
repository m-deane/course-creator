# Structural Identification in Factor-Augmented Models

> **Reading time:** ~17 min | **Module:** Module 6: Factor Augmented | **Prerequisites:** Modules 0-5

<div class="callout-key">

**Key Concept Summary:** Structural identification in FAVARs recovers economically meaningful shocks from reduced-form innovations by imposing theory-based restrictions. This enables causal analysis of policy interventions, business cycle shocks, and financial disturbances across hundreds of economic variables, revealing...

</div>

## In Brief

Structural identification in FAVARs recovers economically meaningful shocks from reduced-form innovations by imposing theory-based restrictions. This enables causal analysis of policy interventions, business cycle shocks, and financial disturbances across hundreds of economic variables, revealing transmission mechanisms invisible in small VARs.

<div class="callout-insight">

**Insight:** Reduced-form FAVAR residuals are linear combinations of structural shocks. Without restrictions, infinitely many structural interpretations exist. Identification schemes—recursive orderings, sign restrictions, external instruments, or high-frequency surprises—use economic theory to isolate specific shocks and trace their propagation through the entire economy via factor responses.

</div>
---

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## 1. The Identification Problem

### Reduced-Form FAVAR

State equation for $(F_t, Y_t)$:
$$G_t = \Phi(L) G_{t-1} + v_t$$

where $v_t$ are reduced-form innovations with $E[v_t v_t'] = \Sigma_v$.

**Problem:** $v_t$ are forecast errors, not economic shocks.

### Structural Representation

Assume $v_t$ are driven by structural shocks $\varepsilon_t$:
$$v_t = B \varepsilon_t$$

where:
- $\varepsilon_t$: Structural shocks (economically interpretable)
- $E[\varepsilon_t \varepsilon_t'] = I$ (orthogonal, unit variance)
- $B$: Impact matrix linking shocks to variables

**Structural VAR:**
$$G_t = \Phi(L) G_{t-1} + B \varepsilon_t$$

### Identification Challenge

Given estimated $\Sigma_v = B B'$, must recover $B$ from its outer product.

**Indeterminacy:** Any orthogonal rotation works:
$$\Sigma_v = B B' = (BQ)(BQ)' \quad \text{for any orthogonal } Q$$

Need $(K+M)^2$ restrictions to identify $B$. Have $(K+M)(K+M+1)/2$ from $\Sigma_v$.

**Shortfall:** $(K+M)(K+M-1)/2$ additional restrictions required.

---

## 2. Identification Strategies

### 2.1 Recursive Identification (Cholesky)

**Assumption:** Shocks have a causal ordering. Some shocks affect others contemporaneously; reverse is prohibited.

**Implementation:** $B$ lower-triangular via Cholesky decomposition:
$$B = \text{cholesky}(\Sigma_v)$$

**Interpretation:** Variable $i$ responds contemporaneously to its own shock and shocks $1, ..., i-1$ only.

**Example (Monetary FAVAR):**
```
Ordering:  [F_real, F_prices, F_credit, FFR]
Interpretation:
  - Real activity shock affects everything
  - Price shock affects prices, credit, policy
  - Credit shock affects credit, policy
  - Policy shock affects only policy (contemporaneously)
```

**Critique:** Results sensitive to ordering. Economic justification needed.

### 2.2 Sign Restrictions

**Approach:** Impose theory-based inequality constraints on impulse responses.

**Example (Contractionary Monetary Shock):**
- Output falls: $\text{IR}_{output}(s) \leq 0$ for $s = 0, ..., S$
- Prices fall: $\text{IR}_{prices}(s) \leq 0$ for $s = 0, ..., S$
- Policy rate rises: $\text{IR}_{rate}(s) \geq 0$ for $s = 0, ..., S$

**Algorithm:**
1. Draw random orthogonal matrix $Q$
2. Compute candidate $B_Q = B_{\text{chol}} Q$
3. Calculate impulse responses
4. Keep if satisfies sign restrictions
5. Repeat until sufficient accepted draws

**Advantage:** Robust to orderings, transparent about economic assumptions.

**Limitation:** May not fully identify (set identification).

### 2.3 External Instruments

**Approach:** Use outside information correlated with shock of interest but uncorrelated with others.

Let $Z_t$ be external instrument for shock $\varepsilon_{jt}$:
- **Relevance:** $E[Z_t \varepsilon_{jt}] \neq 0$
- **Exogeneity:** $E[Z_t \varepsilon_{kt}] = 0$ for $k \neq j$

**Estimation (Proxy VAR):**
$$E[v_t Z_t'] = E[B \varepsilon_t Z_t'] = b_j E[\varepsilon_{jt} Z_t']$$

Solve for $b_j$ (jth column of $B$):
$$b_j \propto E[v_t Z_t']$$

Normalize and identify remaining columns via orthogonality.

**Monetary Policy Example:**
- Shock: Unexpected change in policy rate
- Instrument: High-frequency rate change in 30-min window around FOMC announcement
- Rationale: Isolates pure policy surprise from other news

### 2.4 High-Frequency Identification

**Context:** Financial markets react instantly to identified shocks.

**Method:**
1. Identify shock at high frequency (e.g., FOMC surprises)
2. Use as instrument in monthly/quarterly FAVAR
3. Trace low-frequency responses

**Example (Gertler-Karadi 2015):**
- Instrument: Fed funds futures change in 30-min window
- FAVAR: Monthly data with Fed funds rate as observable
- Result: Clean monetary shock identification

---

## 3. Mathematical Formulation

### Impulse Response Functions

Structural MA representation:
$$G_t = \sum_{s=0}^\infty \Psi_s B \varepsilon_{t-s}$$

**Structural IRF:** Response of $G_{t+s}$ to one-unit shock $\varepsilon_t$:
$$\text{SIR}(s) = \Psi_s B$$

For specific shock $j$:
$$\text{SIR}_j(s) = \Psi_s b_j$$

where $b_j$ is jth column of $B$.

### Mapping to Observables

Original variables $X_t$ relate to state via:
$$X_t = \Lambda^f F_t + \Lambda^y Y_t + e_t = \Lambda G_t + e_t$$

**Response of $X_i$ to structural shock $j$:**
$$\text{IR}_{X_i, \varepsilon_j}(s) = \lambda_i' \cdot \text{SIR}_j(s)$$

where $\lambda_i$ is $i$th row of $\Lambda = [\Lambda^f, \Lambda^y]$.

**Full response matrix:** $N \times (K+M)$ matrix for each horizon $s$:
$$\text{IR}_X(s) = \Lambda \cdot \Psi_s \cdot B$$

### Forecast Error Variance Decomposition

Variance of $h$-step ahead forecast error for variable $i$:
$$\text{Var}(X_{it+h} - E_t[X_{it+h}]) = \sum_{j=1}^{K+M} \sum_{s=0}^{h-1} (\lambda_i' \Psi_s b_j)^2$$

**Share attributable to shock $j$:**
$$\text{FEV}_{ij}(h) = \frac{\sum_{s=0}^{h-1} (\lambda_i' \Psi_s b_j)^2}{\sum_{j=1}^{K+M} \sum_{s=0}^{h-1} (\lambda_i' \Psi_s b_j)^2}$$

Interpretation: Fraction of $h$-period variance in $X_i$ due to shock $j$.

### Historical Decomposition

Decompose realized path into shock contributions:
$$G_t - E[G_t] = \sum_{j=1}^{K+M} \sum_{s=0}^{\infty} \Psi_s b_j \varepsilon_{j,t-s}$$

Contribution of shock $j$:
$$G_t^{(j)} = \sum_{s=0}^{\infty} \Psi_s b_j \varepsilon_{j,t-s}$$

For observables:
$$X_{it}^{(j)} = \lambda_i' G_t^{(j)}$$

---

## 4. Intuitive Explanation

### Why Identification Matters

**Without identification:**
- Know something moved variables
- Don't know what or why
- Can't predict effects of interventions

**With identification:**
- Isolate specific shocks (policy, technology, demand)
- Trace causal effects
- Quantify importance of different disturbances

### Analogy: Medical Diagnosis

Imagine patient with fever, cough, fatigue:
- **Reduced-form:** Symptoms are correlated
- **Structural:** Identify cause (flu, COVID, pneumonia)
- **Treatment:** Depends on identified cause

Similarly:
- **Reduced-form FAVAR:** Output, inflation, rates move together
- **Structural:** Identify shock (monetary policy, technology, oil price)
- **Policy:** Response depends on identified shock

### Visual: Shock Propagation

```
Structural Shock              State Variables              All Variables
    εₜᵖᵒˡⁱᶜʸ         →         [F₁, F₂, F₃, FFR]    →    [X₁, ..., Xₙ]
       ↓                              ↓                        ↓
  Identified via           VAR dynamics             Factor loadings
  restrictions             propagate shock          map to 100+ vars
```

**Key insight:** Small shock to one variable (policy rate) ripples through factors to affect entire economy.

---

## 5. Code Implementation

### Structural FAVAR Class

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">structuralfavar.py</span>
</div>

```python
import numpy as np
from scipy import linalg, stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class StructuralFAVAR:
    """
    Structural Factor-Augmented VAR with multiple identification schemes.

    Parameters
    ----------
    n_factors : int
        Number of latent factors
    n_lags : int
        VAR lag order
    observable_indices : array-like, optional
        Indices of observable variables
    identification : str
        'cholesky', 'sign_restrictions', or 'external_instrument'
    """

    def __init__(self, n_factors=5, n_lags=2, observable_indices=None,
                 identification='cholesky'):
        self.n_factors = n_factors
        self.n_lags = n_lags
        self.observable_indices = observable_indices
        self.identification = identification

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_factors)

        # Estimated components
        self.factors_ = None
        self.observables_ = None
        self.loadings_ = None
        self.Phi_ = None
        self.Sigma_v_ = None
        self.B_ = None  # Structural impact matrix

    def fit(self, X, instrument=None, sign_restrictions=None):
        """
        Fit structural FAVAR.

        Parameters
        ----------
        X : array-like, shape (T, N)
            Data panel
        instrument : array-like, optional, shape (T,)
            External instrument for shock identification
        sign_restrictions : dict, optional
            Sign restrictions for identification

        Returns
        -------
        self
        """
        X = np.asarray(X)
        T, N = X.shape

        # Step 1: Estimate reduced-form FAVAR
        self._estimate_reduced_form(X)

        # Step 2: Structural identification
        if self.identification == 'cholesky':
            self.B_ = self._identify_cholesky()

        elif self.identification == 'sign_restrictions':
            if sign_restrictions is None:
                raise ValueError("sign_restrictions required for this method")
            self.B_ = self._identify_sign_restrictions(sign_restrictions)

        elif self.identification == 'external_instrument':
            if instrument is None:
                raise ValueError("instrument required for this method")
            self.B_ = self._identify_instrument(instrument)

        else:
            raise ValueError(f"Unknown identification: {self.identification}")

        return self

    def structural_irf(self, horizon=20, shock_index=0, shock_size=1.0):
        """
        Compute structural impulse response function.

        Parameters
        ----------
        horizon : int
            Periods ahead
        shock_index : int
            Which structural shock
        shock_size : float
            Size of shock (in std devs)

        Returns
        -------
        irf_state : array-like, shape (horizon, K+M)
            IRF for state variables (F_t, Y_t)
        irf_observed : array-like, shape (horizon, N)
            IRF for all observed variables X_t
        """
        K = self.n_factors
        M = 0 if self.observables_ is None else self.observables_.shape[1]

        # Compute MA representation
        Psi = self._compute_ma_representation(horizon)

        # Structural shock vector
        shock = np.zeros(K + M)
        shock[shock_index] = shock_size

        # State IRF: Psi_s * B * shock
        irf_state = np.zeros((horizon, K + M))
        for s in range(horizon):
            irf_state[s, :] = Psi[s] @ self.B_ @ shock

        # Map to observables: Lambda * IRF_state
        irf_observed = irf_state @ self.loadings_.T

        return irf_state, irf_observed

    def variance_decomposition(self, horizon=20, variable_indices=None):
        """
        Compute forecast error variance decomposition.

        Parameters
        ----------
        horizon : int
            Forecast horizon
        variable_indices : array-like, optional
            Indices of variables to decompose (default: all)

        Returns
        -------
        fevd : array-like, shape (n_vars, K+M, horizon)
            FEVD for each variable, shock, and horizon
        """
        K = self.n_factors
        M = 0 if self.observables_ is None else self.observables_.shape[1]
        n_shocks = K + M

        if variable_indices is None:
            variable_indices = range(self.loadings_.shape[0])

        n_vars = len(variable_indices)
        fevd = np.zeros((n_vars, n_shocks, horizon))

        # Compute MA representation
        Psi = self._compute_ma_representation(horizon)

        for idx, var_i in enumerate(variable_indices):
            lambda_i = self.loadings_[var_i, :]

            for h in range(horizon):
                # Total variance at horizon h
                total_var = 0
                shock_contributions = np.zeros(n_shocks)

                for s in range(h + 1):
                    # Response to each shock at lag s
                    response = lambda_i @ Psi[s] @ self.B_
                    shock_contributions += response**2
                    total_var += np.sum(response**2)

                # Normalize
                if total_var > 0:
                    fevd[idx, :, h] = shock_contributions / total_var

        return fevd

    def historical_decomposition(self, X, shock_index=0):
        """
        Decompose historical path into shock contributions.

        Parameters
        ----------
        X : array-like, shape (T, N)
            Historical data
        shock_index : int
            Which shock to isolate

        Returns
        -------
        contribution : array-like, shape (T, N)
            Contribution of specified shock to each variable
        """
        T = X.shape[0]

        # Extract factors and state
        X_scaled = self.scaler.transform(X)
        F = self.pca.transform(X_scaled)

        if self.observables_ is not None:
            Y = X[:, self.observable_indices]
            G = np.column_stack([F, Y])
        else:
            G = F

        # Compute structural shocks
        eps = self._recover_structural_shocks(G)

        # Compute MA representation (long horizon for full history)
        Psi = self._compute_ma_representation(T)

        # Contribution of shock j
        K, M = self.n_factors, 0 if self.observables_ is None else self.observables_.shape[1]
        G_contribution = np.zeros((T, K + M))

        for t in range(T):
            for s in range(min(t + 1, len(Psi))):
                G_contribution[t] += Psi[s] @ self.B_[:, shock_index] * eps[t - s, shock_index]

        # Map to observables
        X_contribution = G_contribution @ self.loadings_.T

        return X_contribution

    def _estimate_reduced_form(self, X):
        """Estimate reduced-form FAVAR."""
        T, N = X.shape

        # Extract factors
        if self.observable_indices is not None:
            X_slow = np.delete(X, self.observable_indices, axis=1)
            self.observables_ = X[:, self.observable_indices]
        else:
            X_slow = X
            self.observables_ = None

        X_scaled = self.scaler.fit_transform(X_slow)
        self.factors_ = self.pca.fit_transform(X_scaled)

        # Estimate loadings
        if self.observables_ is not None:
            G = np.column_stack([self.factors_, self.observables_])
        else:
            G = self.factors_

        self.loadings_ = linalg.lstsq(G, X)[0].T

        # Estimate VAR
        self.Phi_, self.Sigma_v_ = self._estimate_var(G)

    def _estimate_var(self, Y):
        """Estimate VAR(p) by OLS."""
        T, M = Y.shape
        p = self.n_lags

        # Construct lagged matrix
        Y_lagged = np.zeros((T - p, M * p))
        for lag in range(1, p + 1):
            Y_lagged[:, (lag-1)*M:lag*M] = Y[p-lag:-lag, :]

        # OLS
        Y_dep = Y[p:, :]
        Phi_stacked = linalg.lstsq(Y_lagged, Y_dep)[0]

        # Reshape
        Phi = [Phi_stacked[i*M:(i+1)*M, :].T for i in range(p)]

        # Residual covariance
        fitted = Y_lagged @ Phi_stacked
        resid = Y_dep - fitted
        Sigma = (resid.T @ resid) / (T - p - M*p)

        return Phi, Sigma

    def _identify_cholesky(self):
        """Cholesky decomposition identification."""
        B = linalg.cholesky(self.Sigma_v_, lower=True)
        return B

    def _identify_sign_restrictions(self, restrictions, n_draws=10000, n_keep=1000):
        """
        Sign restrictions identification.

        Parameters
        ----------
        restrictions : dict
            Format: {shock_index: {variable_index: {horizon_range: sign}}}
            Example: {0: {0: {range(0,4): 1}, 1: {range(0,4): -1}}}
                     Shock 0: variable 0 positive, variable 1 negative for 4 periods

        Returns
        -------
        B : array-like
            Accepted structural impact matrix (median over accepted draws)
        """
        # Cholesky as starting point
        B_chol = linalg.cholesky(self.Sigma_v_, lower=True)
        dim = B_chol.shape[0]

        accepted_B = []

        for _ in range(n_draws):
            # Random orthogonal rotation
            Q = self._random_orthogonal(dim)
            B_candidate = B_chol @ Q

            # Check sign restrictions
            if self._check_sign_restrictions(B_candidate, restrictions):
                accepted_B.append(B_candidate)

            if len(accepted_B) >= n_keep:
                break

        if len(accepted_B) == 0:
            raise ValueError("No draws satisfied sign restrictions")

        # Return median
        B = np.median(accepted_B, axis=0)

        # Ensure BB' = Sigma_v (renormalize)
        B = B @ linalg.sqrtm(linalg.inv(B.T @ B @ linalg.inv(self.Sigma_v_)))

        return B

    def _identify_instrument(self, Z):
        """
        External instrument identification.

        Parameters
        ----------
        Z : array-like, shape (T,)
            External instrument

        Returns
        -------
        B : array-like
            Impact matrix with first column identified via instrument
        """
        # Recover reduced-form residuals
        T = len(Z)
        K = self.n_factors
        M = 0 if self.observables_ is None else self.observables_.shape[1]
        dim = K + M

        # Get VAR residuals
        if self.observables_ is not None:
            G = np.column_stack([self.factors_, self.observables_])
        else:
            G = self.factors_

        Y_lagged = self._create_var_matrix(G, self.n_lags)
        Y_dep = G[self.n_lags:, :]

        fitted = np.zeros_like(Y_dep)
        for i, Phi_i in enumerate(self.Phi_):
            fitted += Y_lagged[:, i*dim:(i+1)*dim] @ Phi_i.T

        v = Y_dep - fitted

        # Align instrument
        Z_aligned = Z[self.n_lags:][:len(v)]

        # Estimate first column of B via projection
        b1 = (v.T @ Z_aligned) / (Z_aligned @ Z_aligned)
        b1 = b1 / np.linalg.norm(b1)  # Normalize

        # Complete B via QR decomposition
        B = np.zeros((dim, dim))
        B[:, 0] = b1

        # Orthogonalize remaining columns
        Q = linalg.qr(np.random.randn(dim, dim))[0]
        Q[:, 0] = b1 / np.linalg.norm(b1)

        # Renormalize to match covariance
        B = Q @ linalg.cholesky(Q.T @ self.Sigma_v_ @ Q, lower=True)

        return B

    def _random_orthogonal(self, dim):
        """Generate random orthogonal matrix via QR decomposition."""
        A = np.random.randn(dim, dim)
        Q, _ = linalg.qr(A)
        return Q

    def _check_sign_restrictions(self, B, restrictions):
        """Check if candidate B satisfies sign restrictions."""
        horizon = 20  # Check horizon for restrictions
        Psi = self._compute_ma_representation(horizon)

        for shock_idx, var_restrictions in restrictions.items():
            for var_idx, horizon_restrictions in var_restrictions.items():
                for h_range, required_sign in horizon_restrictions.items():
                    for h in h_range:
                        # Compute impulse response
                        irf_val = (self.loadings_[var_idx, :] @ Psi[h] @
                                   B[:, shock_idx])

                        # Check sign
                        if required_sign > 0 and irf_val < 0:
                            return False
                        if required_sign < 0 and irf_val > 0:
                            return False

        return True

    def _compute_ma_representation(self, horizon):
        """Compute MA representation Psi_s for s=0,...,horizon-1."""
        M = self.Phi_[0].shape[0]
        Psi = [np.eye(M)]

        for s in range(1, horizon):
            Psi_s = np.zeros((M, M))
            for j in range(min(s, self.n_lags)):
                Psi_s += self.Phi_[j] @ Psi[s - j - 1]
            Psi.append(Psi_s)

        return Psi

    def _create_var_matrix(self, Y, p):
        """Create lagged matrix for VAR."""
        T, M = Y.shape
        Y_lagged = np.zeros((T - p, M * p))
        for lag in range(1, p + 1):
            Y_lagged[:, (lag-1)*M:lag*M] = Y[p-lag:-lag, :]
        return Y_lagged

    def _recover_structural_shocks(self, G):
        """Recover structural shocks from state variables."""
        T, dim = G.shape

        # Get VAR residuals
        Y_lagged = self._create_var_matrix(G, self.n_lags)
        Y_dep = G[self.n_lags:, :]

        fitted = np.zeros_like(Y_dep)
        for i, Phi_i in enumerate(self.Phi_):
            fitted += Y_lagged[:, i*dim:(i+1)*dim] @ Phi_i.T

        v = Y_dep - fitted

        # Structural shocks: epsilon = B^{-1} v
        eps = linalg.solve(self.B_, v.T).T

        return eps
```

</div>

### Example: Monetary Policy Shock

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import matplotlib.pyplot as plt

# Simulate data with known structural shock
np.random.seed(2024)
T, N = 400, 60
K, M = 4, 1

# True structural impact matrix (Cholesky-like)
B_true = np.array([
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.3, 1.0, 0.0, 0.0, 0.0],
    [0.2, 0.2, 1.0, 0.0, 0.0],
    [0.1, 0.1, 0.1, 1.0, 0.0],
    [-0.5, -0.3, -0.2, -0.1, 1.0]  # Policy shock affects all factors negatively
])

# VAR(1) dynamics
Phi_true = np.eye(5) * 0.85

# Simulate structural shocks
eps = np.random.randn(T, 5)

# Simulate state variables
G = np.zeros((T, 5))
for t in range(1, T):
    G[t] = Phi_true @ G[t-1] + B_true @ eps[t]

# Generate observed variables
F = G[:, :K]
Y = G[:, K:]

Lambda = np.random.randn(N-M, K+M) / np.sqrt(K+M)
X_main = G @ Lambda.T + np.random.randn(T, N-M) * 0.5
X = np.column_stack([X_main, Y])

# Estimate structural FAVAR with different identification schemes

# 1. Cholesky
sfavar_chol = StructuralFAVAR(n_factors=4, n_lags=1, observable_indices=[N-1],
                              identification='cholesky')
sfavar_chol.fit(X)

irf_state_chol, irf_obs_chol = sfavar_chol.structural_irf(horizon=30, shock_index=4)

# 2. Sign restrictions
# Contractionary monetary shock: policy rate up, real activity down
sign_restrictions = {
    4: {  # Shock index 4 (policy shock)
        0: {range(0, 6): -1},  # Factor 0 (real activity) negative
        4: {range(0, 6): 1}    # Policy rate positive
    }
}

sfavar_sign = StructuralFAVAR(n_factors=4, n_lags=1, observable_indices=[N-1],
                              identification='sign_restrictions')
sfavar_sign.fit(X, sign_restrictions=sign_restrictions)

irf_state_sign, irf_obs_sign = sfavar_sign.structural_irf(horizon=30, shock_index=4)

# Plot comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

variables = ['Real Activity F1', 'Prices F2', 'Credit F3', 'Expectations F4', 'Policy Rate']

for i, (ax, var_name) in enumerate(zip(axes.flatten()[:5], variables)):
    ax.plot(irf_state_chol[:, i], label='Cholesky', linewidth=2)
    ax.plot(irf_state_sign[:, i], label='Sign Restrictions',
            linewidth=2, linestyle='--')
    ax.axhline(0, color='black', linestyle=':', alpha=0.3)
    ax.set_title(f'{var_name}')
    ax.set_xlabel('Periods')
    ax.set_ylabel('Response')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Variance decomposition for policy shock
fevd = sfavar_chol.variance_decomposition(horizon=20, variable_indices=[0, 1, 2, 3, N-1])

axes[1, 2].plot(fevd[0, 4, :], label='Real Activity', linewidth=2)
axes[1, 2].plot(fevd[1, 4, :], label='Prices', linewidth=2)
axes[1, 2].plot(fevd[4, 4, :], label='Policy Rate', linewidth=2)
axes[1, 2].set_title('FEVD: Policy Shock')
axes[1, 2].set_xlabel('Horizon')
axes[1, 2].set_ylabel('Share of Variance')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('structural_favar_identification.png', dpi=300, bbox_inches='tight')
plt.show()

print("Structural FAVAR Estimation Complete")
print(f"\nEstimated impact matrix (Cholesky):\n{sfavar_chol.B_}")
print(f"\nTrue impact matrix (last column):\n{B_true[:, -1]}")
```

</div>

---

## 6. Common Pitfalls

### 1. Unjustified Recursive Ordering

**Mistake:** Using Cholesky without economic rationale for ordering.

**Problem:** Results are order-dependent. Different orderings give different shocks.

**Solution:** Justify ordering with economic theory or use order-invariant methods (sign restrictions, instruments).

### 2. Weak Instruments

**Mistake:** Using instruments with low relevance (weak first stage).

**Problem:** Identification fails, estimates are imprecise and biased.

**Solution:** Test instrument strength. Use F-statistic > 10 rule of thumb.

### 3. Insufficient Sign Restrictions

**Mistake:** Imposing only one or two sign restrictions.

**Problem:** Set identification remains large. Many structural interpretations consistent with restrictions.

**Solution:** Impose comprehensive restrictions across variables and horizons. Test robustness.

### 4. Mixing Identification Schemes

**Mistake:** Combining incompatible restrictions (e.g., recursiveness + sign restrictions on same shock).

**Problem:** Over-identification or logical inconsistency.

**Solution:** Choose one coherent identification strategy per shock.

---

## 7. Connections

### Builds On
- **FAVAR Models:** Reduced-form framework requiring identification
- **Structural VARs:** Identification theory for VARs
- **Instrumental Variables:** Exogeneity and relevance conditions

### Leads To
- **Narrative Restrictions:** Use historical episodes to identify shocks
- **Time-Varying Structural FAVARs:** Allow shock transmission to change
- **Heterogeneous Agent Models:** Micro-found structural shocks

### Related Methods
- **Local Projections:** Alternative to VAR for impulse responses
- **Proxy SVARs:** Dedicated external instrument framework
- **Bayesian SVARs:** Prior-based identification

---

## 8. Practice Problems

### Conceptual

1. **Why is the structural identification problem worse in high-dimensional VARs?**

2. **In a monetary FAVAR, why might you include the policy rate as an observable rather than extracting it through factors?**

3. **What economic assumptions underlie the use of high-frequency FOMC surprises as instruments for monetary shocks?**

### Mathematical

4. **Prove that any orthogonal rotation of an identified structural impact matrix yields the same reduced-form covariance: $\Sigma_v = B B' = (BQ)(BQ)'$.**

5. **Derive the relationship between structural IRFs and variance decomposition: show that FEVD sums to 1 across shocks.**

6. **Show that with $K$ orthonormal shocks, identifying one shock via external instrument leaves $(K-1)(K-2)/2$ degrees of freedom for remaining shocks.**

### Implementation

7. **Implement narrative sign restrictions: Use known historical episodes (e.g., Volcker disinflation) to constrain shock identification.**

8. **Add confidence bands to structural IRFs using bootstrap:**
   - Resample VAR residuals
   - Re-estimate FAVAR
   - Re-identify structural shocks
   - Compute IRF distributions

9. **Test sensitivity to Cholesky ordering: Try all possible orderings and report range of estimates.**

### Advanced

10. **Implement "penalty function approach" (Mountford-Uhlig 2009): Maximize shock orthogonality subject to sign restrictions via optimization.**

11. **Combine external instruments with sign restrictions: Use instrument for one shock, signs for others.**

---

<div class="callout-insight">

**Insight:** Understanding structural identification in factor-augmented models is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## 9. Further Reading

### Identification Theory

- **Rubio-Ramirez, J.F., Waggoner, D.F. & Zha, T. (2010).** "Structural Vector Autoregressions: Theory of Identification and Algorithms for Inference." *Review of Economic Studies*, 77(2), 665-696.
  - General theory of SVAR identification

- **Kilian, L. & Lutkepohl, H. (2017).** *Structural Vector Autoregressive Analysis*. Cambridge University Press.
  - Comprehensive textbook on SVAR identification methods

### Sign Restrictions

- **Uhlig, H. (2005).** "What are the Effects of Monetary Policy on Output? Results from an Agnostic Identification Procedure." *Journal of Monetary Economics*, 52(2), 381-419.
  - Sign restrictions for monetary policy

- **Arias, J.E., Rubio-Ramirez, J.F. & Waggoner, D.F. (2018).** "Inference Based on Structural Vector Autoregressions Identified with Sign and Zero Restrictions." *Econometrica*, 86(2), 685-720.
  - Bayesian inference with sign restrictions

### External Instruments

- **Mertens, K. & Ravn, M.O. (2013).** "The Dynamic Effects of Personal and Corporate Income Tax Changes in the United States." *American Economic Review*, 103(4), 1212-1247.
  - Narrative approach to fiscal shock identification

- **Gertler, M. & Karadi, P. (2015).** "Monetary Policy Surprises, Credit Costs, and Economic Activity." *American Economic Journal: Macroeconomics*, 7(1), 44-76.
  - High-frequency identification of monetary shocks

### FAVAR Applications

- **Boivin, J., Giannoni, M.P. & Mojon, B. (2009).** "How Has the Euro Changed the Monetary Transmission?" *NBER Macroeconomics Annual*, 23, 77-125.
  - Structural FAVAR for Euro area monetary policy

- **Forni, M., Gambetti, L. & Sala, L. (2014).** "No News in Business Cycles." *Economic Journal*, 124(581), 1168-1191.
  - Identifying news shocks in structural FAVAR

### Surveys

- **Stock, J.H. & Watson, M.W. (2016).** "Dynamic Factor Models, Factor-Augmented Vector Autoregressions, and Structural Vector Autoregressions in Macroeconomics." *Handbook of Macroeconomics*, Vol 2A, 415-525.
  - Section 5 covers structural identification in FAVARs

- **Ramey, V.A. (2016).** "Macroeconomic Shocks and Their Propagation." *Handbook of Macroeconomics*, Vol 2A, 71-162.
  - Survey of identification strategies across methods

---

## Conceptual Practice Questions

1. What happens if you skip the identification step in a factor model? Describe the practical consequence.

2. Compare the PC1 normalization and lower-triangular restrictions. When would you prefer each?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./03_structural_identification_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_diffusion_index_forecasting.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_diffusion_index_forecasting.md">
  <div class="link-card-title">01 Diffusion Index Forecasting</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_favar_models.md">
  <div class="link-card-title">02 Favar Models</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

