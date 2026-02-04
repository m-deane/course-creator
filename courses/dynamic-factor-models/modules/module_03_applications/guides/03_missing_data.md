# Missing Data Handling in Dynamic Factor Models

## In Brief

Missing data is ubiquitous in real-time economic analysis due to publication lags, ragged edges (variables updated at different frequencies), and data revisions. The Kalman filter provides an optimal framework for handling arbitrary patterns of missingness, allowing DFMs to produce nowcasts even when most recent observations are incomplete.

## Key Insight

Traditional methods (forward-fill, interpolation) impose rigid assumptions about missing values and ignore uncertainty. The Kalman filter treats missing observations as latent variables to be integrated out, automatically accounting for uncertainty propagation. This enables principled handling of ragged edges where some series are observed while others are not—the hallmark of real-time nowcasting.

---

## Visual Explanation

```
Ragged Edge Data Matrix (Real-Time Economic Data):
═══════════════════════════════════════════════════════

         Jan    Feb    Mar   │ Apr (nowcasting month)
      ┌───────────────────────┼──────────────────────┐
IP    │  ✓      ✓      ✓     │  ?   (released Apr 15)
EMP   │  ✓      ✓      ✓     │  ✓   (released Apr 5)
Sales │  ✓      ✓      ?     │  ?   (released Apr 12)
PMI   │  ✓      ✓      ✓     │  ✓   (flash estimate)
GDP   │  ─────────  Q1 ──────│  ?   (released Apr 30)
      └───────────────────────┴──────────────────────┘
               ↑                        ↑
          Complete data          Ragged edge!

    ✓ = Observed    ? = Missing    ─ = Quarterly variable


Kalman Filter with Missing Data Flow:
═══════════════════════════════════════════════════════

Time t: Xₜ = [x₁,ₜ, NaN, x₃,ₜ, NaN, x₅,ₜ]  ← Mixed observed/missing
         │
         ▼
    ┌─────────────────────────────────────────────┐
    │  1. Prediction Step (always runs)           │
    │     Fₜ|ₜ₋₁ = Φ Fₜ₋₁|ₜ₋₁                    │
    │     Pₜ|ₜ₋₁ = Φ Pₜ₋₁|ₜ₋₁ Φ' + Q             │
    └─────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────┐
    │  2. Identify Observed Subset                │
    │     obs_idx = [0, 2, 4]  (indices where     │
    │                            not missing)      │
    └─────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────┐
    │  3. Update with Observed Only               │
    │     Forecast error: vₜ = Xₒᵦₛ - Λₒᵦₛ Fₜ|ₜ₋₁│
    │     Kalman gain: Kₜ = Pₜ|ₜ₋₁ Λₒᵦₛ' S⁻¹    │
    │     Updated factor: Fₜ|ₜ = Fₜ|ₜ₋₁ + Kₜvₜ  │
    └─────────────────────────────────────────────┘
         │
         ▼
    Factor estimate accounts for:
    - Observed data (small uncertainty)
    - Missing data (larger uncertainty)
    - Factor dynamics (persistence helps bridge gaps)


Publication Lag Patterns:
═══════════════════════════════════════════════════════

Early Indicators (t+1 to t+5 days):
  PMI surveys, initial jobless claims, financial data
  └─→ High noise, but timely

Mid-Tier (t+10 to t+15 days):
  Employment, retail sales, housing starts
  └─→ Moderate quality, reasonably timely

Late Indicators (t+30 days or more):
  GDP, corporate profits, detailed NIPA accounts
  └─→ High quality, but outdated for nowcasting

DFM Strategy: Use early indicators to infer late ones!
```

---

## Formal Definition

### Missing Data Mechanisms

Let $M_t$ be missingness indicator matrix: $M_{it} = 1$ if $X_{it}$ observed, 0 otherwise.

**Missing Completely at Random (MCAR):**
$$P(M_t | X_t, F_t) = P(M_t)$$
Missingness independent of data. Example: random sensor failure.

**Missing at Random (MAR):**
$$P(M_t | X_t, F_t) = P(M_t | X_t^{\text{obs}})$$
Missingness depends on observed data only. Example: high-volatility series measured more frequently.

**Missing Not at Random (MNAR):**
$$P(M_t | X_t, F_t) = P(M_t | X_t, F_t)$$
Missingness depends on missing values themselves. Example: firms in distress stop reporting.

**Ragged edge in DFMs:** Typically MAR (publication lags are fixed schedules, not data-dependent).

### Kalman Filter with Missing Observations

**Standard Kalman Filter:**
```
Prediction:  Fₜ|ₜ₋₁ = Φ Fₜ₋₁|ₜ₋₁
             Pₜ|ₜ₋₁ = Φ Pₜ₋₁|ₜ₋₁ Φ' + Q

Update:      Kₜ = Pₜ|ₜ₋₁ Λ' (Λ Pₜ|ₜ₋₁ Λ' + Σₑ)⁻¹
             Fₜ|ₜ = Fₜ|ₜ₋₁ + Kₜ(Xₜ - Λ Fₜ|ₜ₋₁)
             Pₜ|ₜ = (I - Kₜ Λ) Pₜ|ₜ₋₁
```

**Modified for Missing Data:**

Let $\mathcal{O}_t$ be set of observed indices at time $t$.

```
Prediction:  (unchanged)

Update:      Define Λₒ = Λ[𝒪ₜ, :], Σₒ = Σₑ[𝒪ₜ, 𝒪ₜ], Xₒ = Xₜ[𝒪ₜ]

             Kₜ = Pₜ|ₜ₋₁ Λₒ' (Λₒ Pₜ|ₜ₋₁ Λₒ' + Σₒ)⁻¹
             Fₜ|ₜ = Fₜ|ₜ₋₁ + Kₜ(Xₒ - Λₒ Fₜ|ₜ₋₁)
             Pₜ|ₜ = (I - Kₜ Λₒ) Pₜ|ₜ₋₁
```

**Key insight:** Only observed dimensions enter the update. Missing dimensions don't contribute to innovation.

**Special case (all missing):** If $\mathcal{O}_t = \emptyset$, skip update entirely:
$$F_{t|t} = F_{t|t-1}, \quad P_{t|t} = P_{t|t-1}$$

Factor evolves via transition equation alone (no new information).

### Implications for Nowcasting

**Nowcast with ragged edge:**

At time $t$ within quarter $\tau$, some March data missing:
$$\hat{y}_{\tau|t} = \beta_0 + \beta_1 \mathbb{E}[\bar{F}_\tau | X_{1:t}^{\text{obs}}]$$

**Uncertainty decomposition:**
$$\text{Var}(\hat{y}_{\tau|t}) = \beta_1^2 \underbrace{\text{Var}(\bar{F}_\tau | X_{1:t}^{\text{obs}})}_{\text{parameter uncertainty}} + \underbrace{\sigma_\epsilon^2}_{\text{shock uncertainty}}$$

- More missing data → larger $\text{Var}(\bar{F}_\tau | X_{1:t}^{\text{obs}})$
- As data arrives, uncertainty shrinks

---

## Intuitive Explanation

### The Nowcasting Detective Problem

You're investigating Q1 GDP. The clues:

- ✓ **January IP:** +0.5% (strong manufacturing)
- ✓ **February IP:** +0.3% (slowing but positive)
- ? **March IP:** Not yet released (it's April 10)
- ✓ **March employment:** +200k jobs (released April 5)

**Question:** What's your best guess for March IP?

**Naive approach:** Carry forward February's +0.3%

**DFM approach:**
1. Extract latent "activity factor" from Jan-Feb IP + all March employment data
2. Factor shows strong momentum (employment surged)
3. Use factor to infer March IP ≈ +0.4% ± 0.2%
4. Explicitly quantify uncertainty (±0.2% wider than if March IP was observed)

### Why Kalman Filter Beats Interpolation

**Linear interpolation:**
```
Feb IP: 105.3
Mar IP: ???
Apr IP: 106.1 (when released in May)

Interpolation: Mar IP ≈ (105.3 + 106.1)/2 = 105.7
```

**Problems:**
- Uses future data (April) not yet available in March!
- Ignores all other indicators (employment, sales, surveys)
- No uncertainty quantification

**Kalman filter:**
- Only uses data available up to nowcast date
- Borrows strength from correlated series (employment helps predict IP)
- Outputs predictive distribution: $\hat{X}_{\text{MarIP}} \sim N(105.6, 0.3^2)$

---

## Code Implementation

### Kalman Filter with Missing Data

```python
import numpy as np
from scipy.linalg import inv

def kalman_filter_missing(X, Lambda, Phi, Sigma_e, Q, F0, P0):
    """
    Kalman filter with arbitrary missing data pattern.

    Parameters
    ----------
    X : array (T, N)
        Data matrix (NaN indicates missing)
    Lambda : array (N, r)
        Factor loadings
    Phi : array (r, r)
        Factor transition matrix
    Sigma_e : array (N, N)
        Idiosyncratic error covariance
    Q : array (r, r)
        Factor innovation covariance
    F0 : array (r,)
        Initial factor estimate
    P0 : array (r, r)
        Initial factor covariance

    Returns
    -------
    F_filtered : array (T, r)
        Filtered factor estimates
    F_predicted : array (T, r)
        One-step-ahead factor predictions
    P_filtered : array (T, r, r)
        Filtered factor covariances
    P_predicted : array (T, r, r)
        Predicted factor covariances
    loglik : float
        Log-likelihood (for missing data)
    """
    T, N = X.shape
    r = Lambda.shape[1]

    # Storage
    F_pred = np.zeros((T, r))
    F_filt = np.zeros((T, r))
    P_pred = np.zeros((T, r, r))
    P_filt = np.zeros((T, r, r))
    loglik = 0.0

    # Initialize
    F_prev = F0
    P_prev = P0

    for t in range(T):
        # ===== PREDICTION STEP =====
        F_pred[t] = Phi @ F_prev
        P_pred[t] = Phi @ P_prev @ Phi.T + Q

        # ===== UPDATE STEP (handle missing) =====
        obs_mask = ~np.isnan(X[t])  # True for observed
        obs_idx = np.where(obs_mask)[0]

        if len(obs_idx) == 0:
            # All missing: no update
            F_filt[t] = F_pred[t]
            P_filt[t] = P_pred[t]
        else:
            # Subset to observed
            X_obs = X[t, obs_idx]
            Lambda_obs = Lambda[obs_idx, :]
            Sigma_obs = Sigma_e[np.ix_(obs_idx, obs_idx)]

            # Innovation
            v = X_obs - Lambda_obs @ F_pred[t]

            # Innovation covariance
            S = Lambda_obs @ P_pred[t] @ Lambda_obs.T + Sigma_obs

            # Kalman gain
            K = P_pred[t] @ Lambda_obs.T @ inv(S)

            # Update
            F_filt[t] = F_pred[t] + K @ v
            P_filt[t] = (np.eye(r) - K @ Lambda_obs) @ P_pred[t]

            # Log-likelihood contribution
            loglik += -0.5 * (len(obs_idx) * np.log(2*np.pi) +
                              np.linalg.slogdet(S)[1] + v.T @ inv(S) @ v)

        # Prepare for next iteration
        F_prev = F_filt[t]
        P_prev = P_filt[t]

    return F_filt, F_pred, P_filt, P_pred, loglik

# Example usage
T, N, r = 100, 10, 2

# Simulate complete data
F_true = np.random.randn(T, r)
Lambda = np.random.randn(N, r)
X_complete = F_true @ Lambda.T + 0.5 * np.random.randn(T, N)

# Impose ragged edge: last 5 periods, 50% missing
X_ragged = X_complete.copy()
X_ragged[-5:, np.random.rand(5, N) < 0.5] = np.nan

# Run Kalman filter
F_filt, F_pred, P_filt, P_pred, ll = kalman_filter_missing(
    X=X_ragged,
    Lambda=Lambda,
    Phi=0.8 * np.eye(r),
    Sigma_e=0.25 * np.eye(N),
    Q=np.eye(r),
    F0=np.zeros(r),
    P0=10 * np.eye(r)
)

print(f"Log-likelihood: {ll:.2f}")
print(f"Factor MSE (last 5 periods): {np.mean((F_filt[-5:] - F_true[-5:])**2):.4f}")
```

### Simulating Publication Lag Patterns

```python
def impose_publication_lags(data, lag_patterns):
    """
    Impose realistic publication lag structure on data.

    Parameters
    ----------
    data : pd.DataFrame (T x N)
        Complete data (e.g., from historical dataset)
    lag_patterns : dict
        Maps column name to publication lag in days
        Example: {'IP': 15, 'Employment': 5, 'GDP': 30}

    Returns
    -------
    data_ragged : pd.DataFrame
        Data with NaNs reflecting publication lags
    """
    data_ragged = data.copy()

    for col, lag_days in lag_patterns.items():
        # Assume data is monthly, convert lag to periods
        lag_months = max(0, lag_days // 30)

        # Set last `lag_months` observations to NaN
        if lag_months > 0:
            data_ragged.iloc[-lag_months:, data_ragged.columns.get_loc(col)] = np.nan

    return data_ragged

# Example
import pandas as pd

data = pd.DataFrame({
    'IP': np.random.randn(100),
    'Employment': np.random.randn(100),
    'Sales': np.random.randn(100),
    'GDP': np.random.randn(100)
}, index=pd.date_range('2000-01-01', periods=100, freq='M'))

lag_patterns = {
    'IP': 15,         # Released mid-month
    'Employment': 5,  # Released early
    'Sales': 12,      # Released mid-month
    'GDP': 45         # Quarterly, long lag
}

data_ragged = impose_publication_lags(data, lag_patterns)
print("Missing data pattern (last 5 rows):")
print(data_ragged.tail().isnull())
```

### Comparing Missing Data Strategies

```python
def compare_imputation_methods(X_true, X_missing, Lambda, Phi, Sigma_e, Q):
    """
    Compare Kalman filter vs naive imputation.

    Parameters
    ----------
    X_true : array (T, N)
        True complete data
    X_missing : array (T, N)
        Data with NaNs
    ... : DFM parameters

    Returns
    -------
    results : dict
        Imputation errors for each method
    """
    # Method 1: Forward-fill
    X_ffill = pd.DataFrame(X_missing).fillna(method='ffill').values

    # Method 2: Linear interpolation
    X_interp = pd.DataFrame(X_missing).interpolate(method='linear').values

    # Method 3: Kalman filter
    F_filt, _, _, _, _ = kalman_filter_missing(
        X_missing, Lambda, Phi, Sigma_e, Q,
        F0=np.zeros(Lambda.shape[1]),
        P0=10*np.eye(Lambda.shape[1])
    )
    X_kalman = F_filt @ Lambda.T

    # Compute errors only for imputed values
    missing_mask = np.isnan(X_missing)

    errors = {
        'Forward Fill': np.sqrt(np.mean((X_ffill[missing_mask] - X_true[missing_mask])**2)),
        'Interpolation': np.sqrt(np.mean((X_interp[missing_mask] - X_true[missing_mask])**2)),
        'Kalman Filter': np.sqrt(np.mean((X_kalman[missing_mask] - X_true[missing_mask])**2))
    }

    return errors

# Example output:
# {'Forward Fill': 1.23, 'Interpolation': 1.05, 'Kalman Filter': 0.78}
```

---

## Common Pitfalls

### 1. Using Future Data to Impute

**Problem:** Interpolating missing March value using April data (not yet available in March).

**Example:**
```python
# WRONG for real-time nowcasting
df['IP'].interpolate(method='linear', inplace=True)  # Uses future!
```

**Solution:** Only use past and current data:
```python
df['IP'].fillna(method='ffill')  # Forward-fill (conservative)
# OR use Kalman filter with factor dynamics
```

### 2. Ignoring Uncertainty from Missingness

**Problem:** Treating imputed values as observed (overconfident nowcasts).

**Impact:** Confidence intervals too narrow during ragged-edge periods.

**Solution:** Propagate uncertainty from Kalman filter covariance $P_{t|t}$.

### 3. Assuming MCAR When Data is MAR/MNAR

**Problem:** High-volatility periods have more missing data (MNAR), but model assumes MCAR.

**Impact:** Biased factor estimates (systematically misses extreme events).

**Solution:**
- Explicitly model missingness mechanism
- Use robust methods (e.g., t-distributed errors instead of Gaussian)
- Expand uncertainty during high-volatility regimes

### 4. Not Validating Imputation Quality

**Problem:** Never checking how well missing values are imputed.

**Solution:**
- **Holdout test:** Artificially impose missingness on complete historical data, measure imputation accuracy
- **Revision analysis:** Compare early imputations to final revised values

### 5. Computational Shortcuts Breaking During Edge Cases

**Problem:** Inverting singular matrices when all variables missing.

**Example:**
```python
S_inv = np.linalg.inv(S)  # Fails if S is singular!
```

**Solution:** Add checks:
```python
if len(obs_idx) == 0:
    # Skip update
else:
    S_inv = np.linalg.inv(S)
```

---

## Connections

### Builds On
- **Kalman Filter** (Module 2): Core algorithm adapted for missing data
- **State-Space Models**: Unified framework for incomplete observations
- **Maximum Likelihood with EM**: Missing data as latent variables

### Leads To
- **Mixed-Frequency Models** (Module 4): Extreme case of structured missingness
- **Nowcasting Applications**: Real-time data flow always has ragged edges
- **Robust Estimation**: Handling outliers and contaminated data

### Related To
- **Multiple Imputation**: Bayesian approach to uncertainty from missingness
- **Online Learning**: Sequential updates as data arrives
- **Sensor Fusion**: Combining asynchronous multi-modal data

---

## Practice Problems

### Conceptual

1. **Missingness Mechanisms**
   - Classify each scenario as MCAR, MAR, or MNAR:
     a) March retail sales not yet published (it's April 5)
     b) Bankrupt firms stop reporting financial data
     c) Weather station offline due to power outage

2. **Information Content**
   - You're nowcasting Q1 GDP on April 15. Which missing data hurts more: March IP or March housing starts? How would you quantify this?

3. **Forecast Uncertainty**
   - Your nowcast uncertainty is 0.5pp with complete data. March IP is missing. Forecast uncertainty rises to 0.8pp. Interpret this increase.

### Implementation

4. **Ragged Edge Simulation**
   ```python
   # Simulate DFM with realistic publication lags:
   # - IP: 15-day lag
   # - Employment: 5-day lag
   # - GDP: quarterly (always missing in monthly nowcasts)
   # Measure nowcast accuracy with vs without Kalman filter
   ```

5. **Holdout Validation**
   ```python
   # Take complete historical data (2000-2020)
   # Randomly set 20% of values to missing
   # Compare imputation RMSE:
   #   a) Forward-fill
   #   b) Kalman filter
   #   c) EM algorithm
   # Which performs best? When does each fail?
   ```

6. **Uncertainty Quantification**
   ```python
   # For your GDP nowcast:
   # - Decompose forecast variance into:
   #   1) Parameter uncertainty (from missing data)
   #   2) Model uncertainty (bridge equation)
   #   3) Shock uncertainty (irreducible)
   # - Visualize as stacked bar chart
   ```

### Extension

7. **Adaptive Missingness**
   - During recessions, data is revised more heavily. Implement time-varying measurement error variance $\Sigma_e(t)$ that increases during NBER recessions.

8. **Multivariate Missingness Patterns**
   - Some variables always missing together (e.g., all survey data released same day). Exploit this structure to improve imputation.

---

## Further Reading

### Essential

- **Durbin, J., & Koopman, S. J. (2012).** *Time Series Analysis by State Space Methods.* 2nd Edition. Oxford University Press. **Chapter 4: Missing Observations.**
  - *Definitive treatment of Kalman filtering with missing data*

- **Bańbura, M., & Modugno, M. (2014).** "Maximum Likelihood Estimation of Factor Models on Datasets with Arbitrary Pattern of Missing Data." *Journal of Applied Econometrics, 29*(1), 133-160.
  - *EM algorithm for DFM with flexible missing patterns*

### Recommended

- **Doz, C., Giannone, D., & Reichlin, L. (2011).** "A Two-Step Estimator for Large Approximate Dynamic Factor Models Based on Kalman Filtering." *Journal of Econometrics, 164*(1), 188-205.
  - *Practical approach: PCA + Kalman smoother for missing data*

- **Little, R. J., & Rubin, D. B. (2019).** *Statistical Analysis with Missing Data.* 3rd Edition. Wiley.
  - *Comprehensive missing data theory (general, not specific to time series)*

### Advanced

- **Mariano, R. S., & Murasawa, Y. (2003).** "A New Coincident Index of Business Cycles Based on Monthly and Quarterly Series." *Journal of Applied Econometrics, 18*(4), 427-443.
  - *Mixed-frequency DFM with missing monthly observations of quarterly variables*

- **Aruoba, S. B., Diebold, F. X., & Scotti, C. (2009).** "Real-Time Measurement of Business Conditions." *Journal of Business & Economic Statistics, 27*(4), 417-427.
  - *High-frequency nowcasting with extreme publication lags and irregularity*

---

**Next Guide:** Cheatsheet - Quick reference for all module concepts
