# Mini-Project: Sparse Factor Methods for Inflation Forecasting

## Overview

This project compares sparse variable selection methods for forecasting inflation using a large panel of macroeconomic predictors. You will implement three modern sparse methods—targeted predictors (Bai-Ng), LASSO factor-augmented regression, and the three-pass regression filter—then evaluate their forecasting performance and interpret which economic variables drive predictions.

**Learning Objectives:**
- Implement sparse variable selection methods for factor models
- Compare sparse and dense factor approaches
- Conduct proper time-series cross-validation
- Interpret selected variables in economic context
- Understand bias-variance tradeoff in high-dimensional forecasting

**Time Estimate:** 6-8 hours

**Difficulty:** Advanced

---

## Project Specification

### Problem Statement

Forecast US CPI inflation at horizons $h = 1, 3, 6, 12$ months using FRED-MD database (124 macroeconomic variables). Compare the forecasting performance of:

1. **Benchmark:** AR(4) model (autoregressive)
2. **Dense factors:** PCA + factor-augmented regression
3. **Targeted predictors:** Bai-Ng pre-screening method
4. **LASSO:** Factor-augmented with L1 penalty
5. **Three-pass filter:** Kelly-Pruitt method

**Evaluation Period:** 2010-2023 with expanding window

**Goal:** Determine whether sparse methods improve forecast accuracy and provide interpretable variable selection.

---

## Requirements

### Core Requirements (Must Complete)

#### 1. Data Preparation (10 points)

Load and preprocess FRED-MD dataset:

```python
import pandas as pd
import numpy as np

class FREDMD_Loader:
    """
    Load and prepare FRED-MD database for inflation forecasting.
    """

    def __init__(self, csv_path='FRED-MD.csv'):
        """
        FRED-MD: 124 monthly US macroeconomic series
        Available at: https://research.stlouisfed.org/econ/mccracken/fred-databases/
        """
        self.data = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    def get_transformation_codes(self):
        """
        FRED-MD transformation codes:
        1: No transformation (levels)
        2: First difference
        3: Second difference
        4: Log
        5: Log difference (growth rate)
        6: Log second difference
        7: Δ(x_t/x_{t-1} - 1)

        Returns
        -------
        transform_codes : dict
            {variable_name: code}
        """
        # First row contains transformation codes
        codes = self.data.iloc[0].to_dict()
        return codes

    def apply_transformations(self):
        """
        Apply FRED-MD transformations to achieve stationarity.

        Returns
        -------
        X_transformed : DataFrame, shape (T, N)
            Transformed variables
        """
        codes = self.get_transformation_codes()
        X = self.data.iloc[1:].astype(float)  # Skip code row
        X_transformed = pd.DataFrame(index=X.index)

        for col in X.columns:
            code = codes[col]
            if code == 1:
                X_transformed[col] = X[col]
            elif code == 2:
                X_transformed[col] = X[col].diff()
            elif code == 4:
                X_transformed[col] = np.log(X[col])
            elif code == 5:
                X_transformed[col] = np.log(X[col]).diff()
            # Add other transformation codes
            # YOUR CODE HERE

        return X_transformed.dropna()

    def get_inflation_target(self, series='CPIAUCSL', horizon=1):
        """
        Construct inflation target variable.

        Inflation: π_t = 1200 * log(CPI_t / CPI_{t-1}) (annualized monthly)
        Forecast target: π_{t+h}

        Parameters
        ----------
        series : str
            'CPIAUCSL' (CPI All Items) or 'CPILFESL' (Core CPI)
        horizon : int
            Forecast horizon in months

        Returns
        -------
        inflation : Series
            Annualized monthly inflation rate
        """
        cpi = self.data[series].iloc[1:].astype(float)
        inflation = 1200 * np.log(cpi / cpi.shift(1))
        return inflation.shift(-horizon)  # Shift for h-step ahead target

    def prepare_forecasting_dataset(self, target_series='CPIAUCSL', horizon=1,
                                   start_date='2000-01', end_date='2023-12'):
        """
        Create aligned dataset: predictors X_t and target y_{t+h}.

        Returns
        -------
        X : DataFrame, shape (T, N)
            Predictors (standardized)
        y : Series, shape (T,)
            Target variable
        """
        X = self.apply_transformations()
        y = self.get_inflation_target(target_series, horizon)

        # Align dates
        common_dates = X.index.intersection(y.index)
        X = X.loc[common_dates]
        y = y.loc[common_dates]

        # Restrict to date range
        mask = (X.index >= start_date) & (X.index <= end_date)
        X = X[mask]
        y = y[mask]

        # Standardize predictors
        X = (X - X.mean()) / X.std()

        return X, y
```

**Data Requirements:**
- Download FRED-MD from [McCracken & Ng website](https://research.stlouisfed.org/econ/mccracken/fred-databases/)
- Sample period: 2000-01 to 2023-12
- Handle missing values (interpolation or removal)
- Standardize all predictors (mean 0, variance 1)

**Deliverable:** `data_summary.md` with:
- Variable categories (real activity, prices, interest rates, etc.)
- Transformation summary
- Correlation of top predictors with inflation
- Missing data treatment

---

#### 2. Benchmark Models (15 points)

Implement baseline forecasting models:

```python
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg

class BenchmarkModels:
    """
    Benchmark inflation forecasting models.
    """

    @staticmethod
    def ar_model(y_train, y_test, p=4):
        """
        Autoregressive AR(p) model.

        y_{t+h} = α + Σ_{j=1}^p φ_j y_{t+1-j} + ε_{t+h}

        Parameters
        ----------
        y_train : Series
            Training data
        y_test : Series
            Test data (for evaluation)
        p : int
            AR order

        Returns
        -------
        forecasts : ndarray
            Out-of-sample forecasts
        """
        model = AutoReg(y_train, lags=p, trend='c')
        fitted = model.fit()

        # Forecast test period
        forecasts = fitted.predict(start=len(y_train),
                                   end=len(y_train) + len(y_test) - 1)
        return forecasts

    @staticmethod
    def pca_factor_model(X_train, y_train, X_test, n_factors=8):
        """
        Dense factor model: PCA + OLS regression.

        Steps:
        1. Extract factors: F = PCA(X_train, n_factors)
        2. Forecast regression: y = β' F + ε
        3. Project X_test onto factor space

        Parameters
        ----------
        X_train, y_train : Training data
        X_test : Test predictors
        n_factors : int
            Number of principal components

        Returns
        -------
        forecasts : ndarray
        factor_model : dict
            {'factors', 'loadings', 'coefficients'}
        """
        from sklearn.decomposition import PCA

        # Extract factors from training data
        pca = PCA(n_components=n_factors)
        F_train = pca.fit_transform(X_train)

        # Forecast regression
        reg = LinearRegression()
        reg.fit(F_train, y_train)

        # Out-of-sample factors and forecasts
        F_test = pca.transform(X_test)
        forecasts = reg.predict(F_test)

        return forecasts, {
            'pca': pca,
            'regression': reg,
            'explained_variance': pca.explained_variance_ratio_
        }
```

**Benchmark Specifications:**
- AR(4): Standard autoregressive model
- PCA-8: 8 principal components + OLS
- Ensure fair comparison (same training data)

---

#### 3. Targeted Predictors Method (20 points)

Implement Bai-Ng (2008) targeted predictor selection:

```python
class TargetedPredictors:
    """
    Targeted predictor method (Bai & Ng, 2008).

    Idea: Extract factors only from variables with predictive power for target.
    """

    def __init__(self, threshold=2.58, n_factors=4):
        """
        Parameters
        ----------
        threshold : float
            t-statistic threshold for variable selection
            (2.58 = 1% significance)
        n_factors : int
            Number of factors to extract from selected variables
        """
        self.threshold = threshold
        self.n_factors = n_factors

    def screen_predictors(self, X_train, y_train):
        """
        Step 1: Screen variables for predictive power.

        For each variable i:
            y_{t+h} = α_i + β_i X_{it} + ε_{it}

        Select variable if |t_i| > threshold, where t_i = β̂_i / SE(β̂_i)

        Parameters
        ----------
        X_train : DataFrame, shape (T, N)
        y_train : Series, shape (T,)

        Returns
        -------
        selected_vars : list
            Names of selected variables
        t_stats : dict
            {variable: t-statistic}
        """
        from scipy import stats

        t_stats = {}
        selected_vars = []

        for col in X_train.columns:
            X_i = X_train[col].values.reshape(-1, 1)

            # Univariate regression
            reg = LinearRegression()
            reg.fit(X_i, y_train)

            # Compute t-statistic
            y_pred = reg.predict(X_i)
            residuals = y_train - y_pred
            mse = np.mean(residuals ** 2)
            se_beta = np.sqrt(mse / np.sum((X_i - X_i.mean()) ** 2))
            t_stat = reg.coef_[0] / se_beta

            t_stats[col] = t_stat

            if np.abs(t_stat) > self.threshold:
                selected_vars.append(col)

        print(f"Selected {len(selected_vars)} / {X_train.shape[1]} variables")
        return selected_vars, t_stats

    def extract_targeted_factors(self, X_train, selected_vars):
        """
        Step 2: Extract factors from selected variables only.

        F^targeted = PCA(X^selected, n_factors)

        Parameters
        ----------
        X_train : DataFrame
        selected_vars : list
            Variables that passed screening

        Returns
        -------
        factors : ndarray, shape (T, n_factors)
        pca : PCA object (fitted)
        """
        from sklearn.decomposition import PCA

        X_selected = X_train[selected_vars]
        pca = PCA(n_components=self.n_factors)
        factors = pca.fit_transform(X_selected)

        return factors, pca

    def forecast(self, X_train, y_train, X_test):
        """
        Complete targeted predictor forecasting workflow.

        Returns
        -------
        forecasts : ndarray
        model_info : dict
            {'selected_vars', 't_stats', 'pca', 'regression'}
        """
        # Screen predictors
        selected_vars, t_stats = self.screen_predictors(X_train, y_train)

        if len(selected_vars) == 0:
            print("Warning: No variables selected, using all variables")
            selected_vars = X_train.columns.tolist()

        # Extract factors
        F_train, pca = self.extract_targeted_factors(X_train, selected_vars)

        # Forecast regression
        reg = LinearRegression()
        reg.fit(F_train, y_train)

        # Out-of-sample forecast
        X_test_selected = X_test[selected_vars]
        F_test = pca.transform(X_test_selected)
        forecasts = reg.predict(F_test)

        return forecasts, {
            'selected_vars': selected_vars,
            't_stats': t_stats,
            'pca': pca,
            'regression': reg
        }
```

**Implementation Checklist:**
- [ ] Univariate predictive regressions for all variables
- [ ] t-statistic computation with correct standard errors
- [ ] Variable selection based on threshold
- [ ] PCA on selected variables
- [ ] Forecast regression on targeted factors

---

#### 4. LASSO Factor-Augmented Model (20 points)

Implement LASSO with factors and individual predictors:

```python
from sklearn.linear_model import LassoCV

class FactorAugmentedLASSO:
    """
    Hybrid factor-LASSO model.

    Model: y_{t+h} = β_F' F_t + β_X' X_t + ε

    where:
    - F_t: factors from PCA (capture common variation)
    - X_t: individual predictors (capture idiosyncratic information)
    - LASSO penalty selects relevant X_t
    """

    def __init__(self, n_factors=5, n_alphas=100, cv_folds=5):
        """
        Parameters
        ----------
        n_factors : int
            Number of PCA factors to include
        n_alphas : int
            Number of lambda values to try in cross-validation
        cv_folds : int
            Time-series cross-validation folds
        """
        self.n_factors = n_factors
        self.n_alphas = n_alphas
        self.cv_folds = cv_folds

    def construct_augmented_features(self, X, pca=None):
        """
        Create feature matrix: [F, X] where F are PCA factors.

        Parameters
        ----------
        X : DataFrame, shape (T, N)
        pca : PCA object (fitted on training data) or None

        Returns
        -------
        Z : ndarray, shape (T, n_factors + N)
            Augmented features [factors, raw predictors]
        pca : PCA object (if newly fitted)
        """
        from sklearn.decomposition import PCA

        if pca is None:
            pca = PCA(n_components=self.n_factors)
            F = pca.fit_transform(X)
        else:
            F = pca.transform(X)

        # Concatenate factors and raw predictors
        Z = np.hstack([F, X.values])
        return Z, pca

    def forecast(self, X_train, y_train, X_test):
        """
        Fit LASSO with cross-validation and forecast.

        Cross-validation: Time-series aware (no future data in folds)

        Returns
        -------
        forecasts : ndarray
        model_info : dict
            {'selected_features', 'coefficients', 'cv_results'}
        """
        from sklearn.model_selection import TimeSeriesSplit

        # Construct augmented features
        Z_train, pca = self.construct_augmented_features(X_train)
        Z_test, _ = self.construct_augmented_features(X_test, pca)

        # LASSO with time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        lasso = LassoCV(
            alphas=None,  # Automatically generate
            cv=tscv,
            n_alphas=self.n_alphas,
            fit_intercept=True,
            max_iter=10000
        )
        lasso.fit(Z_train, y_train)

        # Out-of-sample forecast
        forecasts = lasso.predict(Z_test)

        # Identify selected features
        selected_idx = np.where(lasso.coef_ != 0)[0]
        feature_names = (
            [f'Factor_{i+1}' for i in range(self.n_factors)] +
            X_train.columns.tolist()
        )
        selected_features = [feature_names[i] for i in selected_idx]

        return forecasts, {
            'lasso': lasso,
            'pca': pca,
            'selected_features': selected_features,
            'coefficients': lasso.coef_,
            'best_alpha': lasso.alpha_,
            'cv_mse': lasso.mse_path_.mean(axis=1)
        }
```

**Implementation Checklist:**
- [ ] PCA factor extraction
- [ ] Augmented feature matrix [F, X]
- [ ] LASSO with time-series cross-validation
- [ ] Lambda selection via CV
- [ ] Identification of selected variables

---

#### 5. Three-Pass Regression Filter (20 points)

Implement Kelly-Pruitt (2015) three-pass filter:

```python
class ThreePassFilter:
    """
    Three-pass regression filter (Kelly & Pruitt, 2015).

    Extracts factors that have both:
    1. Explanatory power for target
    2. Common variation across predictors
    """

    def __init__(self, n_factors=3):
        """
        Parameters
        ----------
        n_factors : int
            Number of predictive factors to extract
        """
        self.n_factors = n_factors

    def pass_1_predictive_regressions(self, X_train, y_train):
        """
        Pass 1: Variable-by-variable predictive regressions.

        For each i: y_{t+h} = α_i + γ_i X_{it} + e_{it}

        Store fitted values: Γ̂[i,t] = α̂_i + γ̂_i X_{it}

        Parameters
        ----------
        X_train : DataFrame, shape (T, N)
        y_train : Series, shape (T,)

        Returns
        -------
        Gamma : ndarray, shape (N, T)
            Matrix of fitted values from univariate regressions
        gamma_coefs : dict
            {variable: γ̂_i}
        """
        T, N = X_train.shape
        Gamma = np.zeros((N, T))
        gamma_coefs = {}

        for i, col in enumerate(X_train.columns):
            X_i = X_train[col].values.reshape(-1, 1)
            reg = LinearRegression()
            reg.fit(X_i, y_train)

            # Store fitted values
            Gamma[i, :] = reg.predict(X_i)
            gamma_coefs[col] = reg.coef_[0]

        return Gamma, gamma_coefs

    def pass_2_extract_factors(self, Gamma):
        """
        Pass 2: Extract factors from Γ̂ matrix.

        F̃ = PCA(Γ̂', n_factors)

        These are "predictive factors" that capture common movements
        in variables' predictive content.

        Parameters
        ----------
        Gamma : ndarray, shape (N, T)

        Returns
        -------
        F_tilde : ndarray, shape (T, n_factors)
            Predictive factors
        pca : PCA object
        """
        from sklearn.decomposition import PCA

        # PCA on transposed Gamma (T x N)
        pca = PCA(n_components=self.n_factors)
        F_tilde = pca.fit_transform(Gamma.T)

        return F_tilde, pca

    def pass_3_forecast_regression(self, F_tilde, y_train):
        """
        Pass 3: Forecast regression on predictive factors.

        y_{t+h} = β' F̃_t + ε_{t+h}

        Parameters
        ----------
        F_tilde : ndarray, shape (T, n_factors)
        y_train : Series

        Returns
        -------
        regression : LinearRegression (fitted)
        """
        reg = LinearRegression()
        reg.fit(F_tilde, y_train)
        return reg

    def forecast(self, X_train, y_train, X_test):
        """
        Complete three-pass filter workflow.

        Returns
        -------
        forecasts : ndarray
        model_info : dict
        """
        # Pass 1: Predictive regressions
        Gamma_train, gamma_coefs = self.pass_1_predictive_regressions(X_train, y_train)

        # Pass 2: Extract predictive factors
        F_tilde_train, pca = self.pass_2_extract_factors(Gamma_train)

        # Pass 3: Forecast regression
        reg = self.pass_3_forecast_regression(F_tilde_train, y_train)

        # Out-of-sample forecast
        # Need to project X_test through Pass 1 and Pass 2
        Gamma_test = np.zeros((len(X_train.columns), len(X_test)))
        for i, col in enumerate(X_train.columns):
            # Use Pass 1 coefficients from training
            X_i_test = X_test[col].values.reshape(-1, 1)
            # Reconstruct fitted values (intercept + slope * X_test)
            # Simplified: just use training coefficients
            reg_i = LinearRegression()
            reg_i.fit(X_train[[col]], y_train)
            Gamma_test[i, :] = reg_i.predict(X_test[[col]])

        F_tilde_test = pca.transform(Gamma_test.T)
        forecasts = reg.predict(F_tilde_test)

        return forecasts, {
            'gamma_coefs': gamma_coefs,
            'pca': pca,
            'regression': reg,
            'explained_variance': pca.explained_variance_ratio_
        }
```

**Implementation Checklist:**
- [ ] Pass 1: N univariate regressions
- [ ] Pass 2: PCA on fitted value matrix
- [ ] Pass 3: Forecast regression on predictive factors
- [ ] Out-of-sample projection correct

---

#### 6. Rolling Window Evaluation (15 points)

Conduct proper out-of-sample forecast evaluation:

```python
class RollingWindowEvaluation:
    """
    Time-series cross-validation for forecast comparison.
    """

    def __init__(self, models, initial_window=120, horizons=[1, 3, 6, 12]):
        """
        Parameters
        ----------
        models : dict
            {model_name: model_object}
        initial_window : int
            Initial training window (months)
        horizons : list
            Forecast horizons to evaluate
        """
        self.models = models
        self.initial_window = initial_window
        self.horizons = horizons

    def expanding_window_backtest(self, X, y, horizon):
        """
        Expanding window evaluation.

        For each time t = initial_window, ..., T:
            Train on X[0:t], y[0:t]
            Forecast y[t+horizon]
            Compare to actual

        Parameters
        ----------
        X : DataFrame, shape (T, N)
        y : Series, shape (T,)
        horizon : int

        Returns
        -------
        results : DataFrame
            Columns: date, model_name, forecast, actual, error
        """
        T = len(y)
        results = []

        for t in range(self.initial_window, T - horizon):
            # Training data
            X_train = X.iloc[:t]
            y_train = y.iloc[:t]

            # Test data (single observation)
            X_test = X.iloc[t:t+1]
            y_actual = y.iloc[t + horizon]

            # Forecast with each model
            for model_name, model in self.models.items():
                try:
                    if hasattr(model, 'forecast'):
                        forecast, _ = model.forecast(X_train, y_train, X_test)
                        forecast = forecast[0]
                    else:
                        forecast = model(X_train, y_train, X_test)[0]

                    results.append({
                        'date': y.index[t + horizon],
                        'model': model_name,
                        'forecast': forecast,
                        'actual': y_actual,
                        'error': forecast - y_actual,
                        'squared_error': (forecast - y_actual) ** 2,
                        'abs_error': np.abs(forecast - y_actual)
                    })
                except Exception as e:
                    print(f"Error with {model_name} at t={t}: {e}")

            if (t - self.initial_window) % 12 == 0:
                print(f"Completed {t - self.initial_window} / {T - horizon - self.initial_window} periods")

        return pd.DataFrame(results)

    def compute_metrics(self, results):
        """
        Compute forecast evaluation metrics by model.

        Returns
        -------
        metrics : DataFrame
            Rows: models, Columns: RMSE, MAE, Bias, R²
        """
        metrics = results.groupby('model').apply(lambda df: pd.Series({
            'RMSE': np.sqrt(df['squared_error'].mean()),
            'MAE': df['abs_error'].mean(),
            'Bias': df['error'].mean(),
            'R2': np.corrcoef(df['forecast'], df['actual'])[0, 1] ** 2
        }))
        return metrics

    def diebold_mariano_test(self, results, model1, model2):
        """
        Test if two forecasts have significantly different accuracy.

        H0: E[loss1] = E[loss2]
        Test statistic ~ N(0, 1) asymptotically

        Returns
        -------
        dm_stat : float
        p_value : float
        """
        from scipy import stats

        errors1 = results[results['model'] == model1]['squared_error'].values
        errors2 = results[results['model'] == model2]['squared_error'].values

        d = errors1 - errors2
        d_mean = np.mean(d)
        d_var = np.var(d, ddof=1)
        dm_stat = d_mean / np.sqrt(d_var / len(d))
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

        return dm_stat, p_value
```

**Evaluation Specifications:**
- Initial window: 10 years (120 months)
- Expanding window (not rolling)
- Horizons: h = 1, 3, 6, 12 months
- Metrics: RMSE, MAE, Bias, R²
- Statistical test: Diebold-Mariano

**Deliverable:** Results table with significance stars.

---

## Evaluation Rubric

### Implementation Correctness (40 points)

| Criterion | Excellent (36-40) | Good (30-35) | Adequate (24-29) | Needs Work (0-23) |
|-----------|-------------------|--------------|------------------|-------------------|
| Targeted predictors | Perfect screening and factor extraction | Mostly correct | Some errors | Major errors |
| LASSO | Correct CV, feature selection | Minor issues | Some problems | Incorrect |
| Three-pass filter | All passes correct | Minor errors | Some incorrect | Major errors |
| Cross-validation | Proper time-series CV, no look-ahead | Mostly correct | Some issues | Look-ahead bias |

**Specific Checks:**
- [ ] Bai-Ng t-statistics correct (10 pts)
- [ ] LASSO lambda selection via CV (10 pts)
- [ ] Three-pass filter Gamma matrix (10 pts)
- [ ] Rolling window no look-ahead (10 pts)

---

### Forecast Evaluation (25 points)

| Criterion | Excellent (23-25) | Good (19-22) | Adequate (15-18) | Needs Work (0-14) |
|-----------|-------------------|--------------|------------------|-------------------|
| Out-of-sample testing | Rigorous expanding window | Mostly correct | Some issues | Incorrect |
| Benchmark comparison | Multiple benchmarks, fair | Standard benchmarks | Limited | None |
| Statistical tests | DM test, confidence intervals | Standard metrics | Basic metrics | None |
| Multiple horizons | All 4 horizons evaluated | 2-3 horizons | 1 horizon | None |

---

### Interpretation & Insights (20 points)

| Criterion | Excellent (18-20) | Good (15-17) | Adequate (12-14) | Needs Work (0-11) |
|-----------|-------------------|--------------|------------------|-------------------|
| Variable selection analysis | Deep insights, economic reasoning | Good interpretation | Basic comments | None |
| Sparse vs dense comparison | Thoughtful analysis of bias-variance | Notes differences | Superficial | None |
| Economic interpretation | Policy-relevant insights | Reasonable interpretation | Vague | None |

**Expected Analysis:**
- Which variables selected most frequently?
- Do sparse methods improve accuracy? When/why?
- Economic interpretation: real activity, financial, or inflation expectations?
- Horizon-specific insights (short vs long horizon)

---

### Code Quality & Documentation (15 points)

| Criterion | Excellent (14-15) | Good (11-13) | Adequate (8-10) | Needs Work (0-7) |
|-----------|-------------------|--------------|------------------|-------------------|
| Code organization | Clean classes, modular | Mostly organized | Functional but messy | Poor |
| Documentation | Comprehensive docstrings | Good documentation | Basic | Minimal |
| Reproducibility | Seed setting, clear instructions | Mostly reproducible | Some issues | Not reproducible |

---

## Submission Instructions

### File Structure

```
mini_project_sparse/
├── data/
│   ├── FRED-MD.csv
│   └── data_summary.md
├── src/
│   ├── data_loader.py
│   ├── targeted_predictors.py
│   ├── factor_lasso.py
│   ├── three_pass_filter.py
│   ├── benchmarks.py
│   └── evaluation.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_method_implementations.ipynb
│   ├── 03_forecast_evaluation.ipynb
│   └── 04_interpretation.ipynb
├── results/
│   ├── forecast_comparison_table.csv
│   ├── dm_test_results.csv
│   ├── selected_variables.md
│   ├── figures/
│   │   ├── forecast_errors_by_model.png
│   │   ├── selected_vars_frequency.png
│   │   └── forecast_evolution.png
│   └── economic_interpretation.md
├── tests/
│   └── test_methods.py
├── requirements.txt
└── README.md
```

### Submission Checklist

- [ ] All code runs without errors
- [ ] FRED-MD data included or download script provided
- [ ] All three sparse methods implemented correctly
- [ ] Expanding window evaluation complete (2010-2023)
- [ ] Results for all 4 horizons (h=1,3,6,12)
- [ ] Benchmark comparison table with significance tests
- [ ] Variable selection analysis with economic interpretation
- [ ] Figures properly labeled and saved

### How to Submit

1. Create private GitHub repository
2. Push all code and results
3. Include PDF exports of notebooks
4. Share repository link with instructor

**Deadline:** [To be announced]

**Late Policy:** 10% penalty per day, up to 3 days

---

## Resources

### Required Reading

- Bai & Ng (2008). "Forecasting economic time series using targeted predictors." *Journal of Econometrics* 146(2), 304-317
- Kelly & Pruitt (2015). "The three-pass regression filter: A new approach to forecasting using many predictors." *Journal of Econometrics* 186(2), 294-316

### Recommended

- Tibshirani (1996). "Regression shrinkage and selection via the lasso." *JRSS-B* 58(1), 267-288
- Stock & Watson (2012). "Generalized shrinkage methods for forecasting using many predictors." *JBES* 30(4), 481-493

### Data Source

- [FRED-MD Database](https://research.stlouisfed.org/econ/mccracken/fred-databases/)
- McCracken & Ng (2016). "FRED-MD: A monthly database for macroeconomic research." *JBES* 34(4), 574-589

### Software

- scikit-learn: `LassoCV`, `PCA`, `LinearRegression`
- statsmodels: `AutoReg`
- pandas, numpy, matplotlib

---

## Common Pitfalls

1. **Look-ahead bias:** Using future data in cross-validation
2. **Wrong standardization:** Standardize using training data only
3. **Ignoring time-series structure:** Using standard K-fold CV
4. **Too few CV folds:** Need sufficient folds for stable lambda selection
5. **Forgetting transformations:** Not applying FRED-MD codes
6. **Missing data issues:** Not handling NaN properly
7. **Variable selection interpretation:** Selected ≠ causal, just predictive

---

## Grading Summary

| Component | Points | Weight |
|-----------|--------|--------|
| Implementation Correctness | 40 | 40% |
| Forecast Evaluation | 25 | 25% |
| Interpretation & Insights | 20 | 20% |
| Code Quality | 15 | 15% |
| **Total** | **100** | **100%** |

**Minimum to Pass:** 70/100

**Grade Boundaries:**
- A: 90-100
- B: 80-89
- C: 70-79
- F: Below 70

---

## Academic Integrity

- Discuss general sparse methods with classmates
- **All code must be your own**
- Cite papers and data sources
- AI assistance allowed for debugging, not algorithm design

**Violations = zero credit**

---

*"In high-dimensional forecasting, less is often more. Sparse methods improve accuracy by focusing on relevant predictors and enhancing interpretability."*
