# Mini-Project: GDP Nowcasting System

## Overview

This project builds a complete real-time nowcasting system for US GDP growth using mixed-frequency data. You will handle ragged-edge publication patterns, estimate a state-space mixed-frequency dynamic factor model, generate nowcasts as data arrives within the quarter, and evaluate out-of-sample forecasting accuracy.

**Learning Objectives:**
- Handle temporal aggregation (flow variables)
- Implement mixed-frequency state-space models
- Process ragged-edge data with publication lags
- Evaluate nowcasts across the information release calendar
- Interpret economic content of real-time forecasts

**Time Estimate:** 7-9 hours

**Difficulty:** Advanced

---

## Project Specification

### Problem Statement

Build a nowcasting model that predicts current quarter US real GDP growth using monthly indicators. The system must:
1. Update nowcasts in real-time as monthly data is released
2. Handle the fact that GDP is published 30 days after quarter-end
3. Quantify how each data release improves the nowcast
4. Evaluate out-of-sample accuracy over 2015-2023

**Real-World Context:**

It's April 15, 2024. Q1 2024 ended on March 31. The advance GDP estimate won't be released until April 30. But monthly indicators for January, February, and March are already available (with varying lags). Your nowcast should predict Q1 GDP growth **today**, using all available information.

---

## Requirements

### Core Requirements (Must Complete)

#### 1. Data Collection & Processing (15 points)

Download and prepare mixed-frequency dataset:

```python
import pandas as pd
from fredapi import Fred

class NowcastingDataLoader:
    """
    Load and process mixed-frequency data for GDP nowcasting.
    """

    def __init__(self, fred_api_key):
        self.fred = Fred(api_key=fred_api_key)

    def load_gdp(self, start_date='2000-01-01'):
        """
        Load quarterly real GDP growth (annualized).

        Series: GDPC1 (Real Gross Domestic Product)
        Transformation: Log-difference, annualized

        Returns
        -------
        gdp : Series, quarterly frequency
        """
        # YOUR CODE HERE
        pass

    def load_monthly_indicators(self, start_date='2000-01-01'):
        """
        Load monthly indicators from FRED.

        Required indicators (at minimum):
        - INDPRO: Industrial Production Index
        - PAYEMS: Total Nonfarm Payrolls
        - RSAFS: Retail Sales
        - UMCSENT: Consumer Sentiment
        - CPIAUCSL: CPI (All Items)
        - TB3MS: 3-Month Treasury Rate
        - GS10: 10-Year Treasury Rate
        - HOUST: Housing Starts

        Apply appropriate transformations:
        - Levels → log-differences
        - Rates → first differences or levels
        - Indices → log-differences

        Returns
        -------
        monthly_data : DataFrame, monthly frequency
            Columns: indicator names (transformed)
        """
        # YOUR CODE HERE
        pass

    def create_publication_calendar(self):
        """
        Define publication lags for each series.

        Examples:
        - Employment (PAYEMS): First Friday of month (30-40 days after reference)
        - Retail sales (RSAFS): Mid-month (~15 days after reference)
        - IP (INDPRO): Mid-month (~15 days after reference)
        - GDP (GDPC1): End of month after quarter (~30 days)

        Returns
        -------
        pub_lags : dict
            {series_name: lag_in_days}
        """
        return {
            'PAYEMS': 35,
            'RSAFS': 15,
            'INDPRO': 15,
            'UMCSENT': 0,  # Released near end of reference month
            'CPIAUCSL': 15,
            'TB3MS': 0,
            'GS10': 0,
            'HOUST': 20,
            'GDPC1': 30
        }

    def construct_ragged_edge_dataset(self, eval_date):
        """
        Construct dataset as it would appear on a specific evaluation date.

        Only include observations that would have been published by eval_date.

        Parameters
        ----------
        eval_date : datetime
            The date at which nowcast is computed

        Returns
        -------
        X_available : DataFrame
            Data with NaN for not-yet-published observations
        """
        # YOUR CODE HERE
        pass
```

**Data Requirements:**
- GDP: 2000-Q1 to 2023-Q4 (96 quarters)
- Monthly indicators: 2000-01 to 2023-12 (288 months)
- At least 8 monthly indicators covering hard and soft data
- Proper transformations to stationarity

**Deliverable:** `data_description.md` documenting:
- Each series: source, transformation, economic interpretation
- Publication lag assumptions
- Summary statistics (mean, std, correlation with GDP)

---

#### 2. Mixed-Frequency State-Space Model (30 points)

Implement mixed-frequency DFM for quarterly flow variables:

```python
class MixedFrequencyDFM:
    """
    Mixed-frequency dynamic factor model in state-space form.

    Model:
    - Monthly indicators: X_t^m = Λ^m F_t + e_t^m
    - Quarterly GDP (flow): X_t^q = Λ^q (F_t + F_{t-1} + F_{t-2}) + e_t^q
    - Factor dynamics: F_t = Φ F_{t-1} + η_t
    """

    def __init__(self, n_factors=2, factor_order=1):
        """
        Parameters
        ----------
        n_factors : int
            Number of latent factors
        factor_order : int
            AR order for factor dynamics
        """
        self.r = n_factors
        self.p = factor_order

    def construct_measurement_equation(self, Lambda_m, Lambda_q, Sigma_e_m, sigma_e_q):
        """
        Build measurement equation for mixed frequencies.

        State vector (for quarterly flow with AR(1) factors):
        α_t = [F_t, F_{t-1}, F_{t-2}]'

        Monthly observation (month t within quarter):
        X_t^m = [Λ^m, 0, 0] α_t + e_t^m

        Quarterly observation (end of quarter):
        X_t^q = [Λ^q, Λ^q, Λ^q] α_t + e_t^q

        Parameters
        ----------
        Lambda_m : ndarray, shape (N_m, r)
            Monthly loadings
        Lambda_q : ndarray, shape (1, r)
            Quarterly loading (single GDP series)
        Sigma_e_m : ndarray, shape (N_m,)
            Monthly error variances
        sigma_e_q : float
            Quarterly error variance

        Returns
        -------
        Z_m : ndarray, shape (N_m, 3*r)
            Monthly measurement matrix
        Z_q : ndarray, shape (1, 3*r)
            Quarterly measurement matrix
        H_m : ndarray, shape (N_m, N_m)
            Monthly error covariance
        H_q : float
            Quarterly error variance
        """
        # YOUR CODE HERE
        pass

    def construct_transition_equation(self, Phi, Q):
        """
        Build transition equation for augmented state.

        For AR(1) with quarterly flow aggregation:
        [F_t    ]   [Φ  0  0] [F_{t-1}  ]   [η_t]
        [F_{t-1}] = [I  0  0] [F_{t-2}  ] + [0  ]
        [F_{t-2}]   [0  I  0] [F_{t-3}  ]   [0  ]

        Parameters
        ----------
        Phi : ndarray, shape (r, r)
            Factor transition matrix
        Q : ndarray, shape (r, r)
            Factor innovation covariance

        Returns
        -------
        T : ndarray, shape (3*r, 3*r)
            Augmented transition matrix
        R : ndarray, shape (3*r, r)
            Selection matrix
        Q : ndarray, shape (r, r)
            Innovation covariance
        """
        # YOUR CODE HERE
        pass

    def estimate(self, X_monthly, X_quarterly, method='em', max_iter=100):
        """
        Estimate model parameters via EM algorithm.

        Initialization:
        - Extract factors from monthly data via PCA
        - Regress GDP on factor averages to get Λ^q
        - Estimate Φ from factor VAR

        EM Algorithm:
        - E-step: Kalman smoother on augmented state
        - M-step: Update parameters

        Parameters
        ----------
        X_monthly : DataFrame, shape (T_m, N_m)
            Monthly indicators
        X_quarterly : Series, shape (T_q,)
            Quarterly GDP growth
        method : str
            'em' or 'pca_twostep'

        Returns
        -------
        params : dict
            Estimated parameters {Λ^m, Λ^q, Φ, Σ_e^m, σ_e^q, Q}
        factors : DataFrame
            Estimated monthly factors
        """
        # YOUR CODE HERE
        pass

    def nowcast(self, X_available, params):
        """
        Generate nowcast given current information set.

        Run Kalman filter with missing observations for not-yet-released data.

        Parameters
        ----------
        X_available : DataFrame
            Data with NaN for unavailable observations
        params : dict
            Estimated parameters

        Returns
        -------
        nowcast_mean : float
            E[GDP_Q | available data]
        nowcast_var : float
            Var[GDP_Q | available data]
        factor_estimates : DataFrame
            Filtered factor estimates
        """
        # YOUR CODE HERE
        pass
```

**Implementation Checklist:**
- [ ] Quarterly flow aggregation (sum of 3 monthly factors)
- [ ] State augmentation for lagged factors
- [ ] Handle mixed observation patterns (monthly and quarterly)
- [ ] Kalman filter with missing observations
- [ ] Parameter estimation via EM or two-step
- [ ] Nowcast mean and variance from filtered state

---

#### 3. Real-Time Nowcast Evolution (25 points)

Track how nowcast evolves as data is released:

```python
class NowcastEvolution:
    """
    Track nowcast updates across information release calendar.
    """

    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    def compute_nowcast_path(self, target_quarter, start_days_before=90):
        """
        Compute nowcast for target quarter at multiple evaluation dates.

        Example: Target Q1 2024 (ends March 31)
        - 90 days before: January 1 (only Dec data available)
        - 60 days before: January 31 (Jan data partially available)
        - 30 days before: March 1 (Jan + Feb available)
        - 0 days: March 31 (full quarter data available)
        - 30 days after: April 30 (GDP release, all data available)

        Parameters
        ----------
        target_quarter : str
            E.g., '2024-Q1'
        start_days_before : int
            How many days before quarter-end to start

        Returns
        -------
        nowcast_path : DataFrame
            Columns: date, nowcast_mean, nowcast_std, available_series
        """
        quarter_end = self._get_quarter_end(target_quarter)
        eval_dates = pd.date_range(
            end=quarter_end,
            periods=start_days_before // 7,  # Weekly evaluations
            freq='W'
        )

        results = []
        for eval_date in eval_dates:
            # Construct dataset as of eval_date
            X_available = self.data_loader.construct_ragged_edge_dataset(eval_date)

            # Generate nowcast
            nowcast_mean, nowcast_var = self.model.nowcast(X_available)

            results.append({
                'date': eval_date,
                'days_to_quarter_end': (quarter_end - eval_date).days,
                'nowcast_mean': nowcast_mean,
                'nowcast_std': np.sqrt(nowcast_var),
                'n_available': X_available.notna().sum().sum()
            })

        return pd.DataFrame(results)

    def decompose_nowcast_revisions(self, nowcast_path):
        """
        Attribute nowcast revisions to specific data releases.

        "News" from each data release = change in nowcast.

        Returns
        -------
        news_decomposition : DataFrame
            Columns: date, series_released, nowcast_revision
        """
        # YOUR CODE HERE
        pass

    def plot_nowcast_evolution(self, nowcast_path, actual_gdp):
        """
        Visualize how nowcast converges to actual value.

        Plot:
        - X-axis: Days relative to quarter-end
        - Y-axis: GDP growth rate
        - Line: Nowcast mean
        - Shaded region: ±2 std (95% interval)
        - Horizontal line: Actual GDP (released 30 days after)
        """
        # YOUR CODE HERE
        pass
```

**Deliverable:**
- Nowcast evolution plots for 4 quarters (e.g., 2023-Q1 to Q4)
- Table showing which data releases had largest impact
- Discussion: When does nowcast accuracy improve most?

---

#### 4. Out-of-Sample Evaluation (20 points)

Evaluate nowcasting performance systematically:

```python
class NowcastEvaluation:
    """
    Out-of-sample evaluation of nowcasting model.
    """

    def __init__(self, model):
        self.model = model

    def pseudo_real_time_backtest(self, start_quarter='2015-Q1', end_quarter='2023-Q4'):
        """
        Pseudo real-time evaluation.

        For each quarter:
        1. Use only data available up to that quarter
        2. Re-estimate model on expanding window
        3. Generate nowcast at quarter-end (before GDP release)
        4. Compare to actual GDP (released 30 days later)

        Parameters
        ----------
        start_quarter, end_quarter : str
            Evaluation period

        Returns
        -------
        backtest_results : DataFrame
            Columns: quarter, nowcast, actual, error, squared_error
        """
        quarters = pd.period_range(start_quarter, end_quarter, freq='Q')
        results = []

        for quarter in quarters:
            # Re-estimate model using data up to quarter-end
            train_data = self._get_data_up_to(quarter)
            params = self.model.estimate(train_data['monthly'], train_data['quarterly'])

            # Generate nowcast at quarter-end
            X_qend = self._get_data_at_quarter_end(quarter)
            nowcast_mean, nowcast_var = self.model.nowcast(X_qend, params)

            # Get actual GDP (released later)
            actual_gdp = self._get_actual_gdp(quarter)

            results.append({
                'quarter': quarter,
                'nowcast': nowcast_mean,
                'actual': actual_gdp,
                'error': nowcast_mean - actual_gdp,
                'squared_error': (nowcast_mean - actual_gdp) ** 2
            })

        return pd.DataFrame(results)

    def compute_metrics(self, backtest_results):
        """
        Compute forecast evaluation metrics.

        Returns
        -------
        metrics : dict
            - RMSE: Root mean squared error
            - MAE: Mean absolute error
            - Bias: Mean error
            - R²: Correlation with actual
            - Directional accuracy: % correct sign
        """
        errors = backtest_results['error']
        actual = backtest_results['actual']
        nowcast = backtest_results['nowcast']

        return {
            'RMSE': np.sqrt(np.mean(errors ** 2)),
            'MAE': np.mean(np.abs(errors)),
            'Bias': np.mean(errors),
            'R2': np.corrcoef(actual, nowcast)[0, 1] ** 2,
            'Directional_Accuracy': np.mean(np.sign(actual) == np.sign(nowcast))
        }

    def benchmark_comparison(self, backtest_results):
        """
        Compare to benchmark models:
        1. Naive: GDP_t = GDP_{t-1}
        2. AR(2): Autoregressive model
        3. Bridge equation: OLS regression on monthly averages

        Returns
        -------
        comparison_table : DataFrame
            Rows: models, Columns: metrics
        """
        # YOUR CODE HERE
        pass
```

**Evaluation Period:** 2015-Q1 to 2023-Q4 (36 quarters)

**Deliverable:**
- Backtest results table with nowcast errors
- Comparison table: MF-DFM vs benchmarks
- Time series plot: Nowcast vs actual GDP
- Analysis: When does model perform best/worst?

---

#### 5. Interpretation & Economic Insights (10 points)

Analyze the economic content of your nowcasting system:

```python
def analyze_factor_interpretation(params, monthly_indicators):
    """
    Interpret extracted factors.

    Questions:
    - Which variables load heavily on each factor?
    - Can factors be labeled (e.g., "real activity", "financial conditions")?
    - How persistent are factors (eigenvalues of Φ)?

    Returns
    -------
    interpretation_report : str (Markdown)
    """
    pass

def analyze_news_content(news_decomposition):
    """
    Which data releases matter most?

    Rank series by:
    - Average absolute nowcast revision
    - Frequency of being "most influential"

    Returns
    -------
    news_ranking : DataFrame
        Columns: series, avg_abs_revision, importance_score
    """
    pass
```

**Deliverable:** `economic_interpretation.md` (2-3 pages) discussing:
- Factor interpretation (what do they represent?)
- Which indicators are most informative for GDP?
- Comparison of hard vs soft data informativeness
- Policy implications: How could central bank use this nowcast?

---

### Extension Options (Choose 1, 10 points)

#### Option A: High-Frequency Financial Data

Incorporate daily financial variables (stock returns, spreads) using daily-to-monthly aggregation:

```python
def add_daily_financial_data():
    """
    Add S&P 500 returns, term spread, credit spread.

    Aggregation: End-of-month value or monthly average.
    """
    pass
```

Test if financial data improves early-quarter nowcasts (when little monthly data available).

---

#### Option B: Regional Disaggregation

Build separate nowcasts for GDP components (consumption, investment, government, net exports):

```python
def nowcast_gdp_components():
    """
    Multi-output nowcasting: Predict all GDP components simultaneously.

    State-space with multiple quarterly series.
    """
    pass
```

Does component-level nowcasting aggregate to better GDP nowcast?

---

#### Option C: Real-Time Data Vintages

Use actual real-time data from ALFRED (FRED's archival database):

```python
def load_real_time_vintages(series, vintage_dates):
    """
    Load data as it was available on specific vintage dates.

    Accounts for:
    - Data revisions
    - Definition changes
    - Benchmark revisions
    """
    pass
```

Compare nowcast accuracy using real-time vs final-revised data.

---

## Evaluation Rubric

### Data Handling & Preparation (15 points)

| Criterion | Excellent (14-15) | Good (11-13) | Adequate (8-10) | Needs Work (0-7) |
|-----------|-------------------|--------------|------------------|-------------------|
| Data collection | Comprehensive, well-documented | Good coverage | Minimal coverage | Insufficient data |
| Transformations | Correct stationarity transformations | Mostly correct | Some errors | Incorrect |
| Publication lags | Realistic, well-researched | Reasonable assumptions | Generic assumptions | Ignored |
| Ragged-edge handling | Correct implementation | Mostly correct | Partially correct | Incorrect |

---

### Model Implementation (30 points)

| Criterion | Excellent (27-30) | Good (23-26) | Adequate (18-22) | Needs Work (0-17) |
|-----------|-------------------|--------------|------------------|-------------------|
| State-space formulation | Perfect quarterly flow aggregation | Correct structure, minor issues | Mostly correct | Major errors |
| Parameter estimation | Converges, stable | Works but some issues | Unstable | Fails |
| Nowcast generation | Correct Kalman filter with missing data | Mostly correct | Basic functionality | Incorrect |

**Specific Checks:**
- [ ] Quarterly flow variable aggregation correct (10 pts)
- [ ] Kalman filter handles missing observations (10 pts)
- [ ] Parameters estimated via EM or two-step (10 pts)

---

### Real-Time Analysis (25 points)

| Criterion | Excellent (23-25) | Good (19-22) | Adequate (15-18) | Needs Work (0-14) |
|-----------|-------------------|--------------|------------------|-------------------|
| Nowcast evolution | Comprehensive tracking, informative plots | Good coverage | Basic tracking | Minimal |
| News decomposition | Clear attribution of revisions | Most revisions explained | Limited decomposition | None |
| Insights | Deep analysis of information flow | Good observations | Superficial | None |

---

### Evaluation & Comparison (20 points)

| Criterion | Excellent (18-20) | Good (15-17) | Adequate (12-14) | Needs Work (0-11) |
|-----------|-------------------|--------------|------------------|-------------------|
| Backtest methodology | Proper pseudo real-time | Mostly correct | Some issues | Incorrect |
| Benchmark comparison | Multiple benchmarks, fair comparison | Standard benchmarks | Limited comparison | None |
| Statistical tests | Significance tests (Diebold-Mariano) | Standard metrics | Basic metrics | None |

---

### Interpretation (10 points)

| Criterion | Excellent (9-10) | Good (7-8) | Adequate (5-6) | Needs Work (0-4) |
|-----------|-------------------|--------------|------------------|-------------------|
| Economic content | Deep insights, policy-relevant | Good interpretation | Basic comments | None |
| Factor interpretation | Clear economic meaning | Reasonable interpretation | Vague | None |

---

## Submission Instructions

### File Structure

```
mini_project_nowcasting/
├── data/
│   ├── raw/                  # Downloaded data
│   ├── processed/            # Transformed data
│   └── data_description.md
├── src/
│   ├── data_loader.py
│   ├── mixed_frequency_dfm.py
│   ├── nowcast_evolution.py
│   ├── evaluation.py
│   └── utils.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_estimation.ipynb
│   ├── 03_nowcast_evolution.ipynb
│   └── 04_evaluation.ipynb
├── results/
│   ├── figures/
│   │   ├── nowcast_evolution_*.png
│   │   ├── backtest_results.png
│   │   └── factor_loadings.png
│   ├── tables/
│   │   ├── backtest_results.csv
│   │   └── benchmark_comparison.csv
│   └── economic_interpretation.md
├── tests/
│   └── test_model.py
├── requirements.txt
├── .env.example             # FRED API key template
└── README.md
```

### Submission Checklist

- [ ] All code runs without errors
- [ ] FRED API key instructions in README
- [ ] Data description document complete
- [ ] Model converges and produces reasonable nowcasts
- [ ] Backtest results on 2015-2023 included
- [ ] Nowcast evolution plots for multiple quarters
- [ ] Economic interpretation document (2-3 pages)
- [ ] Benchmark comparison table included
- [ ] Code well-documented with docstrings

### How to Submit

1. Create a private GitHub repository
2. Push all code, data processing scripts, and results
3. Include `results/` directory with all figures and tables
4. Create PDF exports of all notebooks
5. Share repository link with instructor

**Deadline:** [To be announced]

**Late Policy:** 10% penalty per day, up to 3 days

---

## Resources

### Required Reading

- Mariano & Murasawa (2003). "A New Coincident Index of Business Cycles Based on Monthly and Quarterly Series." *Journal of Applied Econometrics* 18(4), 427-443
- Bańbura et al. (2013). "Now-Casting and the Real-Time Data Flow." *Handbook of Economic Forecasting* Vol. 2, 195-237

### Recommended

- Giannone, Reichlin & Small (2008). "Nowcasting: The Real-Time Informational Content of Macroeconomic Data." *Journal of Monetary Economics* 55(4), 665-676
- Bok et al. (2018). "Macroeconomic Nowcasting and Forecasting with Big Data." *Annual Review of Economics* 10, 615-643

### Data Sources

- [FRED API](https://fred.stlouisfed.org/docs/api/fred/)
- [FRED-MD Database](https://research.stlouisfed.org/econ/mccracken/fred-databases/)
- [BEA GDP Data](https://www.bea.gov/data/gdp)

### Software

- `fredapi`: Python wrapper for FRED API
- `statsmodels.tsa.statespace`: State-space modeling
- `pandas`: Time series manipulation

---

## Common Pitfalls

1. **Wrong aggregation:** Using stock formula for GDP (a flow variable)
2. **Publication lag errors:** Assuming all data available immediately
3. **Look-ahead bias:** Using revised data instead of real-time
4. **Overfit ting:** Too many factors for limited GDP sample
5. **Ignoring estimation uncertainty:** Re-estimate parameters each period
6. **Misaligned frequencies:** Month/quarter mapping errors
7. **Not handling missing data:** Kalman filter must handle NaN properly

---

## Grading Summary

| Component | Points | Weight |
|-----------|--------|--------|
| Data Handling | 15 | 15% |
| Model Implementation | 30 | 30% |
| Real-Time Analysis | 25 | 25% |
| Evaluation & Comparison | 20 | 20% |
| Interpretation | 10 | 10% |
| **Total** | **100** | **100%** |

**Minimum to Pass:** 70/100

**Grade Boundaries:**
- A: 90-100
- B: 80-89
- C: 70-79
- F: Below 70

---

## Academic Integrity

- You may discuss general nowcasting approaches with classmates
- **All code must be your own**
- Properly cite data sources and papers
- AI assistance permitted for data wrangling, not core algorithms

**Violations will result in zero credit.**

---

*"Nowcasting is the art of predicting the present. In a world of delayed official statistics, real-time economic monitoring is essential for policy and business decisions."*
