# Temporal Aggregation in Mixed-Frequency Models

## In Brief

Mixed-frequency data requires understanding how variables aggregate over time—monthly industrial production and quarterly GDP follow different temporal aggregation rules. This guide covers stock versus flow variables and the constraints that preserve consistency across frequencies.

## Key Insight

The relationship between high-frequency and low-frequency observations is not arbitrary: flow variables (GDP, consumption) sum over periods while stock variables (unemployment rate, inventory levels) reflect point-in-time values. Ignoring these aggregation constraints leads to misspecified models and biased forecasts.

---

## 1. Stock Variables vs Flow Variables

### Formal Definition

**Flow Variables**: Quantities measured over a period that can be summed across time.
- Examples: GDP, consumption, investment, sales
- Quarterly value = sum of monthly values
- Mathematical: $Y_t^Q = Y_{3t-2}^M + Y_{3t-1}^M + Y_{3t}^M$

**Stock Variables**: Quantities measured at a point in time.
- Examples: Unemployment rate, inventory levels, money supply, exchange rates
- Quarterly value = value at specific point (usually end of quarter)
- Mathematical: $Y_t^Q = Y_{3t}^M$ (end-of-period sampling)

### Intuitive Explanation

Think of a bathtub analogy:
- **Flow variable**: The water flowing in per minute (can be added up over time)
- **Stock variable**: The water level at specific moments (discrete snapshots)

You can add up water inflow over 3 months to get quarterly inflow, but the water level at month's end is just one measurement—you don't add water levels together.

### Mathematical Framework

For $m$ high-frequency periods in one low-frequency period:

**Flow Aggregation:**
$$Y_t^{(L)} = \sum_{j=1}^{m} Y_{mt-(m-j)}^{(H)}$$

**Stock Aggregation (End-of-Period):**
$$Y_t^{(L)} = Y_{mt}^{(H)}$$

**Stock Aggregation (Average):**
$$Y_t^{(L)} = \frac{1}{m} \sum_{j=1}^{m} Y_{mt-(m-j)}^{(H)}$$

where superscript $(H)$ denotes high frequency, $(L)$ denotes low frequency.

### Code Implementation

```python
import numpy as np
import pandas as pd

def aggregate_flow(monthly_data, freq='Q'):
    """
    Aggregate flow variable from monthly to quarterly.

    Parameters
    ----------
    monthly_data : pd.Series
        Monthly flow data with DatetimeIndex
    freq : str, default 'Q'
        Target frequency ('Q' for quarterly, 'A' for annual)

    Returns
    -------
    quarterly_data : pd.Series
        Aggregated data at lower frequency
    """
    # Flow variables: sum within periods
    return monthly_data.resample(freq).sum()


def aggregate_stock_eop(monthly_data, freq='Q'):
    """
    Aggregate stock variable using end-of-period sampling.

    Parameters
    ----------
    monthly_data : pd.Series
        Monthly stock data with DatetimeIndex
    freq : str
        Target frequency

    Returns
    -------
    quarterly_data : pd.Series
        End-of-period values
    """
    # Stock variables: last value in period
    return monthly_data.resample(freq).last()


def aggregate_stock_average(monthly_data, freq='Q'):
    """
    Aggregate stock variable using period average.

    Parameters
    ----------
    monthly_data : pd.Series
        Monthly stock data with DatetimeIndex
    freq : str
        Target frequency

    Returns
    -------
    quarterly_data : pd.Series
        Average values over period
    """
    # Stock variables: average over period
    return monthly_data.resample(freq).mean()


# Example: GDP (flow) and unemployment (stock)
dates_m = pd.date_range('2020-01', '2023-12', freq='M')
np.random.seed(42)

# Monthly GDP (flow) - sum to quarterly
monthly_gdp = pd.Series(
    100 + np.cumsum(np.random.randn(len(dates_m))),
    index=dates_m
)
quarterly_gdp_flow = aggregate_flow(monthly_gdp)

# Monthly unemployment rate (stock) - end-of-period
monthly_unemp = pd.Series(
    5 + np.random.randn(len(dates_m)) * 0.5,
    index=dates_m
)
quarterly_unemp_eop = aggregate_stock_eop(monthly_unemp)
quarterly_unemp_avg = aggregate_stock_average(monthly_unemp)

print("Flow aggregation (GDP):")
print(quarterly_gdp_flow.head())
print("\nStock aggregation - EOP (Unemployment):")
print(quarterly_unemp_eop.head())
print("\nStock aggregation - Average (Unemployment):")
print(quarterly_unemp_avg.head())
```

---

## 2. Aggregation Constraints in Factor Models

### The Constraint Problem

When modeling mixed-frequency data with factors, aggregation constraints ensure consistency:

$$\Lambda^{(L)} = C \Lambda^{(H)}$$

where $C$ is an aggregation matrix encoding the temporal relationship.

### Flow Variable Constraint Matrix

For monthly-to-quarterly with 3 months per quarter:

$$C_{\text{flow}} = \begin{bmatrix}
1 & 1 & 1 & 0 & 0 & 0 & \cdots \\
0 & 0 & 0 & 1 & 1 & 1 & \cdots \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots
\end{bmatrix}$$

Each row sums 3 consecutive monthly observations.

### Stock Variable Constraint Matrix

For end-of-period sampling:

$$C_{\text{stock}} = \begin{bmatrix}
0 & 0 & 1 & 0 & 0 & 0 & \cdots \\
0 & 0 & 0 & 0 & 0 & 1 & \cdots \\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \ddots
\end{bmatrix}$$

Each row selects the last month of each quarter.

### Code Implementation

```python
def create_aggregation_matrix(n_high, n_low, m, agg_type='flow'):
    """
    Create temporal aggregation constraint matrix.

    Parameters
    ----------
    n_high : int
        Number of high-frequency observations
    n_low : int
        Number of low-frequency observations
    m : int
        Number of high-freq periods in one low-freq period (e.g., 3 for M to Q)
    agg_type : str
        'flow', 'stock_eop', or 'stock_avg'

    Returns
    -------
    C : ndarray, shape (n_low, n_high)
        Aggregation constraint matrix
    """
    C = np.zeros((n_low, n_high))

    if agg_type == 'flow':
        # Sum m consecutive high-frequency observations
        for i in range(n_low):
            start_idx = i * m
            end_idx = min(start_idx + m, n_high)
            C[i, start_idx:end_idx] = 1

    elif agg_type == 'stock_eop':
        # Select last observation in each period
        for i in range(n_low):
            idx = min((i + 1) * m - 1, n_high - 1)
            C[i, idx] = 1

    elif agg_type == 'stock_avg':
        # Average m consecutive observations
        for i in range(n_low):
            start_idx = i * m
            end_idx = min(start_idx + m, n_high)
            n_periods = end_idx - start_idx
            C[i, start_idx:end_idx] = 1 / n_periods

    else:
        raise ValueError(f"Unknown aggregation type: {agg_type}")

    return C


# Example: 12 monthly obs -> 4 quarterly obs
n_monthly = 12
n_quarterly = 4

C_flow = create_aggregation_matrix(n_monthly, n_quarterly, m=3, agg_type='flow')
C_stock = create_aggregation_matrix(n_monthly, n_quarterly, m=3, agg_type='stock_eop')

print("Flow aggregation matrix (first 2 quarters):")
print(C_flow[:2, :9])
print("\nStock aggregation matrix (first 2 quarters):")
print(C_stock[:2, :9])

# Verify on data
monthly_values = np.arange(1, 13)  # [1, 2, ..., 12]
quarterly_flow = C_flow @ monthly_values
quarterly_stock = C_stock @ monthly_values

print(f"\nMonthly values: {monthly_values[:6]}")
print(f"Q1 flow (sum of 1,2,3): {quarterly_flow[0]}")
print(f"Q1 stock (value at 3): {quarterly_stock[0]}")
```

---

## 3. Mixed-Frequency Factor Models with Constraints

### Model Specification

High-frequency observation equation:
$$X_t^{(H)} = \Lambda^{(H)} F_t + e_t^{(H)}$$

Low-frequency observation equation with constraint:
$$X_t^{(L)} = C \Lambda^{(H)} F_t + e_t^{(L)} = \Lambda^{(L)} F_t + e_t^{(L)}$$

where $\Lambda^{(L)} = C \Lambda^{(H)}$ enforces aggregation consistency.

### Why Constraints Matter

Without constraints:
- Low-frequency loadings estimated independently
- Inconsistent with high-frequency structure
- Poor out-of-sample forecast performance

With constraints:
- Parameters tied across frequencies
- Efficient use of all available information
- Theoretically consistent forecasts

### Code Implementation

```python
class ConstrainedMixedFrequencyDFM:
    """
    Dynamic Factor Model with temporal aggregation constraints.

    Enforces consistency between high and low frequency factor loadings.
    """

    def __init__(self, n_factors, agg_type='flow', m=3):
        """
        Parameters
        ----------
        n_factors : int
            Number of latent factors
        agg_type : str
            Aggregation type for low-frequency variables
        m : int
            Aggregation ratio (e.g., 3 for monthly to quarterly)
        """
        self.n_factors = n_factors
        self.agg_type = agg_type
        self.m = m
        self.Lambda_H = None
        self.Lambda_L = None
        self.C = None

    def fit(self, X_high, X_low, n_iter=100):
        """
        Estimate model with aggregation constraints.

        Parameters
        ----------
        X_high : ndarray, shape (T_high, N_high)
            High-frequency data
        X_low : ndarray, shape (T_low, N_low)
            Low-frequency data
        n_iter : int
            Number of EM iterations
        """
        T_high, N_high = X_high.shape
        T_low, N_low = X_low.shape

        # Create aggregation matrix
        self.C = create_aggregation_matrix(
            T_high, T_low, self.m, self.agg_type
        )

        # Initialize loadings (PCA on high-frequency)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=self.n_factors)
        F_init = pca.fit_transform(X_high)
        self.Lambda_H = pca.components_.T  # (N_high, r)

        # Low-frequency loadings from constraint
        self.Lambda_L = self.C @ self.Lambda_H  # NOT estimated separately

        # Simplified: would implement full EM with constraint
        # For illustration, using constrained least squares
        for iteration in range(n_iter):
            # E-step: estimate factors given loadings
            # M-step: update Lambda_H (Lambda_L follows via constraint)
            pass

        return self

    def get_loadings(self):
        """Return both high and low frequency loadings."""
        return {
            'high_freq': self.Lambda_H,
            'low_freq': self.Lambda_L,
            'constraint_satisfied': np.allclose(
                self.Lambda_L, self.C @ self.Lambda_H
            )
        }


# Example usage
np.random.seed(42)
T_m, N_m = 36, 20  # 36 months, 20 variables
T_q, N_q = 12, 5   # 12 quarters, 5 variables

# Generate synthetic data
true_factors = np.random.randn(T_m, 2)
Lambda_true = np.random.randn(N_m, 2)
X_monthly = true_factors @ Lambda_true.T + np.random.randn(T_m, N_m) * 0.5

# Aggregate to quarterly (flow)
C = create_aggregation_matrix(T_m, T_q, m=3, agg_type='flow')
X_quarterly = (C @ X_monthly[:, :N_q])

# Fit constrained model
model = ConstrainedMixedFrequencyDFM(n_factors=2, agg_type='flow', m=3)
model.fit(X_monthly, X_quarterly)

loadings = model.get_loadings()
print(f"Constraint satisfied: {loadings['constraint_satisfied']}")
print(f"High-freq loadings shape: {loadings['high_freq'].shape}")
print(f"Low-freq loadings shape: {loadings['low_freq'].shape}")
```

---

## Common Pitfalls

### 1. Misidentifying Variable Type
- **Mistake**: Treating unemployment rate as flow variable (summing rates)
- **Fix**: Carefully determine if variable measures stock or flow
- **Test**: Does summing values make economic sense?

### 2. Ignoring Skip-Sampled Data
- **Mistake**: Using only quarterly observations when monthly exist
- **Fix**: Incorporate all high-frequency information via constraints
- **Benefit**: More efficient parameter estimates

### 3. Inconsistent Timing Conventions
- **Mistake**: Mixing end-of-quarter and average-of-quarter sampling
- **Fix**: Document and enforce consistent temporal alignment
- **Example**: GDP reported at quarter-end vs average unemployment

### 4. Neglecting Leap Years and Uneven Months
- **Mistake**: Assuming all months/quarters have equal length
- **Fix**: Use date-aware pandas operations
- **Impact**: Small but systematic bias in flow aggregation

---

## Connections

- **Builds on:** Time series basics, data frequency concepts
- **Leads to:** MIDAS regression, mixed-frequency state-space models
- **Related to:** Kalman filtering with missing data, temporal disaggregation

---

## Practice Problems

### Conceptual

1. Why would summing stock variables (like unemployment rates) across months produce meaningless results?

2. A quarterly model forecasts monthly GDP by dividing quarterly forecasts by 3. What aggregation type does this assume? Is it correct?

3. You have daily stock prices and monthly trading volume. How would you aggregate each to quarterly frequency?

### Implementation

4. Implement a function that detects whether aggregation is consistent:
   ```python
   def check_aggregation_consistency(monthly, quarterly, agg_type):
       # Return True if quarterly matches aggregated monthly
       pass
   ```

5. Create a visualization showing how flow vs stock aggregation differs for a seasonal time series.

6. Extend `create_aggregation_matrix` to handle ragged edges (e.g., incomplete final quarter).

### Extension

7. Derive the variance of aggregated measurement error: if $\text{Var}(e_t^{(H)}) = \sigma^2$, what is $\text{Var}(e_t^{(L)})$ under flow aggregation?

8. Research "temporal disaggregation" methods (Chow-Lin, Denton). How do they relate to the aggregation constraints discussed here?

---

## Further Reading

- **Mariano, R.S. & Murasawa, Y.** (2003). "A new coincident index of business cycles based on monthly and quarterly series." *Journal of Applied Econometrics*, 18(4), 427-443.
  - Original application of mixed-frequency factor models with aggregation constraints

- **Banbura, M. & Modugno, M.** (2014). "Maximum likelihood estimation of factor models on datasets with arbitrary pattern of missing data." *Journal of Applied Econometrics*, 29(1), 133-160.
  - Unified framework for mixed-frequency and missing data

- **Ghysels, E., Santa-Clara, P. & Valkanov, R.** (2004). "The MIDAS touch: Mixed data sampling regression models." *Chapel Hill Department of Economics Working Paper*.
  - Alternative approach to mixed-frequency modeling

- **Silvestrini, A. & Veredas, D.** (2008). "Temporal aggregation of univariate and multivariate time series models: A survey." *Journal of Economic Surveys*, 22(3), 458-497.
  - Comprehensive review of aggregation theory

---

## Summary

**Key Takeaways:**
1. Flow variables aggregate via summation; stock variables via sampling or averaging
2. Aggregation constraints $\Lambda^{(L)} = C \Lambda^{(H)}$ ensure cross-frequency consistency
3. Proper aggregation improves parameter efficiency and forecast accuracy
4. Mixed-frequency models exploit all available information while respecting temporal structure

**Next Steps:**
The next guide introduces MIDAS regression, which provides flexible weighting schemes for mixed-frequency relationships without requiring explicit factor structure.
