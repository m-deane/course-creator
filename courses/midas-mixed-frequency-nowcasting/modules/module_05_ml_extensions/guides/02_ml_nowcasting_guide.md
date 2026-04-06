# Machine Learning Nowcasting: Tree-Based Methods with Mixed-Frequency Features

> **Reading time:** ~20 min | **Module:** 05 — Ml Extensions | **Prerequisites:** Module 4


## Learning Objectives

<div class="flow">
<div class="flow-step mint">1. Collect Data</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Identify Frequencies</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Align Time Indices</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Build MIDAS Regressors</div>
</div>


<div class="callout-key">

**Key Concept Summary:** Tree-based ensemble methods (random forests, gradient boosting) offer several advantages over linear MIDAS for nowcasting:

</div>

By the end of this guide you will be able to:

1. Explain why tree-based methods require special feature engineering for mixed-frequency data
2. Engineer mixed-frequency features for random forests and gradient boosting
3. Implement XGBoost and LightGBM nowcasting models with proper temporal validation
4. Compare ML methods against MIDAS regression using expanding-window evaluation
5. Apply SHAP values to interpret which high-frequency signals drive forecasts

---

## 1. Why Trees for Nowcasting?

Tree-based ensemble methods (random forests, gradient boosting) offer several advantages over linear MIDAS for nowcasting:

<div class="callout-insight">

**Insight:** Real-time nowcasting is fundamentally different from pseudo out-of-sample backtesting. The ragged-edge data structure means your model sees different information at different points within a quarter.

</div>


1. **Nonlinearity**: Macroeconomic relationships are asymmetric — labour market deterioration during recessions differs from recovery dynamics.
2. **Automatic interactions**: Trees capture interactions between indicators without specifying them explicitly.
3. **Robustness to outliers**: Not affected by extreme observations that can distort linear regression.
4. **No distributional assumptions**: No normality or stationarity requirements (though stationarity still improves performance in practice).

The challenge: trees do not natively handle the temporal structure of mixed-frequency data. Feature engineering is the critical step.

---

## 2. Feature Engineering Approaches

<div class="callout-warning">

**Warning:** Pseudo out-of-sample exercises that do not properly account for the real-time data vintage will overstate nowcast accuracy. Always use the ragged-edge structure that would have been available at each historical nowcast date.

</div>


### 2.1 The Fundamental Challenge

A decision tree splits on individual feature values at a single point in time. Mixed-frequency data presents a stacking problem: monthly data has 1 observation per quarter, daily data has ~65 per quarter. How do we represent this for a tree?

Three approaches, from simplest to most sophisticated:

1. **Flat stacking**: Concatenate all lags as individual features
2. **Statistical summaries**: Aggregate lags into summary statistics
3. **Temporal embeddings**: Learn compact representations of lag windows

### 2.2 Flat Stacking

The simplest approach: for each high-frequency predictor, include lags $x_{t-1}, x_{t-2}, \ldots, x_{t-m}$ as separate features.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
import pandas as pd

def flat_stack_features(df_high, df_low, target_col, m=12):
    """
    Create flat-stacked feature matrix for ML models.

    Parameters
    ----------
    df_high : DataFrame — high-frequency data (monthly), indexed by date
    df_low : DataFrame — low-frequency target (quarterly), indexed by date
    target_col : str — column name in df_low
    m : int — number of high-frequency lags per low-freq period

    Returns
    -------
    X : DataFrame of features
    y : Series of targets
    """
    features = {}
    dates = df_low.index

    for col in df_high.columns:
        for lag in range(1, m + 1):
            feature_name = f"{col}_lag{lag}"
            # Align: for each quarterly date, get m monthly lags
            aligned = []
            for date in dates:
                # Find monthly observations before this quarter-end
                mask = df_high.index <= date
                vals = df_high.loc[mask, col].tail(lag)
                if len(vals) >= lag:
                    aligned.append(vals.iloc[0])
                else:
                    aligned.append(np.nan)
            features[feature_name] = aligned

    X = pd.DataFrame(features, index=dates)
    y = df_low[target_col]

    # Drop rows with any NaN
    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid]
```

</div>

**Drawback**: With 20 monthly series and 12 lags, this produces 240 features. Trees handle this via feature importance, but interpretability suffers.

### 2.3 Statistical Summary Features

Aggregate the lag window into economically meaningful statistics:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def summary_features(series_lags):
    """
    Compute summary statistics over a lag window.

    Parameters
    ----------
    series_lags : array — high-frequency observations in lag window

    Returns
    -------
    dict of summary features
    """
    return {
        'mean': np.mean(series_lags),
        'std': np.std(series_lags),
        'last': series_lags[-1],          # Most recent
        'first': series_lags[0],          # Oldest in window
        'momentum': series_lags[-1] - series_lags[0],  # Change over window
        'acceleration': np.diff(series_lags).mean(),    # Average change
        'max': np.max(series_lags),
        'min': np.min(series_lags),
        'range': np.ptp(series_lags),
        'trend': np.polyfit(range(len(series_lags)), series_lags, 1)[0]
    }
```

</div>

This reduces 12 monthly lags to 10 features per series — a 17× reduction while preserving key temporal information.

### 2.4 Temporal Embeddings

For large daily datasets (e.g., 65 trading days per quarter), embeddings via PCA or autoencoders compress high-dimensional lag windows:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from sklearn.decomposition import PCA

def pca_embedding(lag_matrix, n_components=3):
    """
    PCA embedding of high-frequency lag windows.

    Parameters
    ----------
    lag_matrix : array, shape (T, m) — each row is a lag window
    n_components : int — embedding dimension

    Returns
    -------
    embedding : array, shape (T, n_components)
    pca : fitted PCA object
    """
    pca = PCA(n_components=n_components, random_state=42)
    embedding = pca.fit_transform(lag_matrix)
    explained = pca.explained_variance_ratio_.cumsum()
    print(f"Explained variance with {n_components} components: {explained[-1]:.1%}")
    return embedding, pca
```

</div>

For daily financial data, the first 3 principal components typically capture 80–95% of variance.

---

## 3. Random Forest Nowcasting

### 3.1 Why Random Forests Work

Random forests build many decorrelated trees by:
1. Bootstrap sampling of observations (bagging)
2. Random feature subsetting at each split (feature randomisation)

The ensemble average reduces variance dramatically relative to a single tree. For nowcasting:

- **Nonlinear thresholds**: Capture regime switches (e.g., when PMI falls below 50)
- **Missing data**: Handle unbalanced panels naturally with surrogate splits
- **Feature importance**: Identify which indicators and lags drive the forecast

### 3.2 Implementation

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np

def train_random_forest_nowcast(X_train, y_train, n_estimators=200, max_features='sqrt'):
    """
    Train a Random Forest nowcasting model.

    Parameters
    ----------
    X_train : array — training features
    y_train : array — training targets
    n_estimators : int — number of trees
    max_features : str or int — features per split ('sqrt' or fraction)

    Returns
    -------
    model : fitted RandomForestRegressor
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=3,       # Prevent overfitting on small datasets
        max_depth=None,           # Let trees grow deep (forest controls variance)
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    return rf


def expanding_window_eval(X, y, model_class, model_kwargs, min_train=20):
    """
    Expanding-window out-of-sample evaluation.

    Parameters
    ----------
    X : array — features
    y : array — targets
    model_class : class — sklearn-compatible model
    model_kwargs : dict — model parameters
    min_train : int — minimum training observations

    Returns
    -------
    forecasts : array of out-of-sample predictions
    actuals : array of realised values
    """
    T = len(y)
    forecasts = []
    actuals = []

    for t in range(min_train, T):
        # Train on [0, t), predict at t
        X_tr, y_tr = X[:t], y[:t]
        X_te = X[t:t+1]

        model = model_class(**model_kwargs, random_state=42)
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)[0]

        forecasts.append(y_hat)
        actuals.append(y[t])

    return np.array(forecasts), np.array(actuals)
```

### 3.3 Feature Importance

```python
import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot top-N feature importances from a fitted tree model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importances[indices[::-1]], align='center')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
    ax.set_xlabel('Feature Importance (Mean Decrease Impurity)')
    ax.set_title(f'Top {top_n} Feature Importances')
    plt.tight_layout()
    return fig
```

---

## 4. Gradient Boosting Nowcasting

### 4.1 Gradient Boosting vs Random Forests

| Aspect | Random Forest | Gradient Boosting |
|--------|---------------|-------------------|
| Training | Parallel (bagging) | Sequential (boosting) |
| Bias-variance | Low bias, reduced variance | Lower bias than RF |
| Overfitting risk | Low | Higher (needs regularization) |
| Speed | Fast (parallel) | Slower (sequential) |
| Hyperparameter sensitivity | Low | Higher |
| Asymptotic accuracy | Good | Often better |

Gradient boosting iteratively fits shallow trees to the **residuals** of the current ensemble, greedily minimising a loss function.

### 4.2 XGBoost Implementation

```python
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

def train_xgboost_nowcast(X_train, y_train, X_val=None, y_val=None):
    """
    Train XGBoost nowcasting model with early stopping.

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val : optional validation data for early stopping

    Returns
    -------
    model : fitted XGBRegressor
    """
    model = xgb.XGBRegressor(
        n_estimators=1000,        # Will early-stop
        learning_rate=0.05,       # Small learning rate for better generalisation
        max_depth=4,              # Shallow trees prevent overfit
        min_child_weight=3,       # Min samples in leaf
        subsample=0.8,            # Row subsampling (like bagging)
        colsample_bytree=0.8,     # Feature subsampling per tree
        reg_alpha=0.1,            # L1 regularization
        reg_lambda=1.0,           # L2 regularization
        objective='reg:squarederror',
        eval_metric='rmse',
        random_state=42,
        verbosity=0
    )

    if X_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
    else:
        model.fit(X_train, y_train)

    return model


def xgboost_expanding_window(X, y, min_train=20, val_fraction=0.2):
    """
    XGBoost with expanding window and walk-forward validation.
    Uses last val_fraction of training data as validation for early stopping.
    """
    T = len(y)
    forecasts = []
    actuals = []
    n_rounds = []

    for t in range(min_train, T):
        X_tr, y_tr = X[:t], y[:t]
        X_te = X[t:t+1]

        # Internal validation for early stopping
        val_size = max(5, int(t * val_fraction))
        X_fit, X_val = X_tr[:-val_size], X_tr[-val_size:]
        y_fit, y_val = y_tr[:-val_size], y_tr[-val_size:]

        model = train_xgboost_nowcast(X_fit, y_fit, X_val, y_val)
        y_hat = model.predict(X_te)[0]

        forecasts.append(y_hat)
        actuals.append(y[t])
        if hasattr(model, 'best_iteration'):
            n_rounds.append(model.best_iteration)

    return np.array(forecasts), np.array(actuals), n_rounds
```

### 4.3 LightGBM for Large Datasets

For datasets with many daily series (e.g., 100+ predictors × 252 daily lags), LightGBM is substantially faster than XGBoost due to its histogram-based algorithm:

```python
import lightgbm as lgb

def train_lightgbm_nowcast(X_train, y_train, X_val=None, y_val=None):
    """Train LightGBM for nowcasting — faster than XGBoost for large p."""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,         # Controls complexity (< 2^max_depth)
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]

    if X_val is not None:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        model = lgb.train(params, dtrain, num_boost_round=1000,
                         valid_sets=[dval], callbacks=callbacks)
    else:
        model = lgb.train(params, dtrain, num_boost_round=200)

    return model
```

---

## 5. SHAP Values for Nowcast Interpretation

### 5.1 Why SHAP?

Tree feature importances (mean decrease in impurity) are biased toward high-cardinality features and do not show direction or magnitude of effect. SHAP (SHapley Additive exPlanations) provides:

- **Consistent**: unlike impurity importance, SHAP satisfies theoretical consistency
- **Directional**: positive SHAP = pushed forecast up; negative = pushed down
- **Individual explanations**: "which factors drove this quarter's nowcast"
- **Temporal analysis**: how factor contributions evolved over the vintage horizon

```python
import shap

def explain_nowcast(model, X, feature_names, t_index=-1):
    """
    Compute SHAP values for nowcast explanation.

    Parameters
    ----------
    model : fitted tree model (RF, XGBoost, LightGBM)
    X : array — feature matrix
    feature_names : list — feature names
    t_index : int — index of observation to explain (default: last)

    Returns
    -------
    shap_values : array — SHAP values for each feature at t_index
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # For classifiers

    # Summary plot (all observations)
    shap.summary_plot(shap_values, X,
                     feature_names=feature_names,
                     plot_type='bar', show=False)

    # Waterfall for single forecast
    explanation = shap.Explanation(
        values=shap_values[t_index],
        base_values=explainer.expected_value,
        data=X[t_index],
        feature_names=feature_names
    )
    shap.waterfall_plot(explanation, show=False)

    return shap_values
```

---

## 6. Fair Comparison: ML vs MIDAS

### 6.1 Evaluation Framework

A fair comparison requires:

1. **Same data**: identical training and test periods
2. **Same vintage structure**: both methods see the same ragged-edge information
3. **Same evaluation metric**: RMSFE (Root Mean Square Forecast Error)
4. **Diebold-Mariano test**: formal test of equal predictive accuracy

```python
from scipy import stats

def diebold_mariano_test(e1, e2, h=1):
    """
    Diebold-Mariano test for equal predictive accuracy.

    H0: E[L(e1)] = E[L(e2)]  (equal MSE)
    H1: E[L(e1)] != E[L(e2)]

    Parameters
    ----------
    e1, e2 : arrays — forecast errors from model 1 and model 2
    h : int — forecast horizon (corrects for autocorrelation)

    Returns
    -------
    dm_stat : float — DM test statistic
    p_value : float — two-sided p-value
    """
    d = e1**2 - e2**2  # Loss differential (MSE loss)
    T = len(d)

    # Newey-West variance (accounts for autocorrelation at horizon h)
    gamma0 = np.var(d, ddof=1)
    gammas = [np.cov(d[j:], d[:-j])[0, 1] for j in range(1, h)]
    v = gamma0 + 2 * sum(gammas)

    dm_stat = np.mean(d) / np.sqrt(v / T)
    p_value = 2 * stats.t.sf(np.abs(dm_stat), df=T - 1)

    return dm_stat, p_value


def compare_models(forecasts_dict, actuals):
    """
    Compare multiple nowcasting models.

    Parameters
    ----------
    forecasts_dict : dict of {model_name: forecast_array}
    actuals : array — realised values

    Returns
    -------
    DataFrame — comparison table
    """
    results = []
    benchmark = None

    for name, forecasts in forecasts_dict.items():
        errors = actuals - forecasts
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))
        results.append({'model': name, 'RMSE': rmse, 'MAE': mae})
        if benchmark is None:
            benchmark = errors

    results_df = pd.DataFrame(results).set_index('model')

    # DM tests against first model (benchmark)
    dm_results = {}
    errors_dict = {name: actuals - f for name, f in forecasts_dict.items()}
    names = list(errors_dict.keys())
    e_bench = errors_dict[names[0]]

    for name in names[1:]:
        dm_stat, pval = diebold_mariano_test(errors_dict[name], e_bench)
        dm_results[name] = {'DM_stat': dm_stat, 'DM_pval': pval}

    return results_df, pd.DataFrame(dm_results).T
```

### 6.2 When ML Beats MIDAS

Empirical evidence suggests ML methods outperform MIDAS when:

1. **Nonlinear relationships**: Recession periods with threshold effects
2. **Many indicators**: p >> T setting where group selection helps
3. **Long forecast horizon**: ML handles multicollinearity better
4. **Structural change**: Tree splits adapt to regime shifts

MIDAS tends to outperform when:

1. **Short sample**: ML overfits, MIDAS' smooth weight function helps
2. **Economic theory constraints**: Theoretical sign/shape restrictions improve MIDAS
3. **Real-time data revisions**: MIDAS explicitly models publication lags
4. **Interpretability required**: Smooth weight function has clear economic meaning

---

## 7. Combining ML and MIDAS

### 7.1 Forecast Combination

The simple average of ML and MIDAS forecasts often beats either alone (combination puzzle in forecasting):

```python
def combine_forecasts(forecast_list, weights=None):
    """
    Combine multiple forecasts.

    Parameters
    ----------
    forecast_list : list of arrays — individual model forecasts
    weights : array or None — combination weights (equal if None)

    Returns
    -------
    combined : array — combined forecast
    """
    forecasts_matrix = np.column_stack(forecast_list)
    if weights is None:
        weights = np.ones(len(forecast_list)) / len(forecast_list)
    return forecasts_matrix @ weights


def optimal_combination_weights(forecasts_matrix, actuals):
    """
    Compute OLS-optimal combination weights (Bates-Granger 1969).
    Constrained to sum to 1 with non-negative weights.
    """
    from scipy.optimize import minimize

    n_models = forecasts_matrix.shape[1]

    def loss(w):
        combined = forecasts_matrix @ w
        return np.mean((actuals - combined)**2)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n_models
    w0 = np.ones(n_models) / n_models

    result = minimize(loss, w0, method='SLSQP',
                     constraints=constraints, bounds=bounds)
    return result.x
```

### 7.2 MIDAS Features in ML

An elegant hybrid: use MIDAS-fitted lag weights as features for a gradient boosting model. The MIDAS step compresses the lag window; the boosting step captures nonlinearities:

```python
def midas_features_for_ml(beta_weights, x_high_lags):
    """
    Use MIDAS weight-compressed lags as ML features.
    Combines the dimensionality reduction of MIDAS with ML nonlinearity.
    """
    # MIDAS-weighted sum (single number per indicator per period)
    midas_component = x_high_lags @ beta_weights

    # First difference of MIDAS component (momentum)
    midas_diff = np.diff(midas_component, prepend=midas_component[0])

    return np.column_stack([midas_component, midas_diff])
```

---

## 8. Key References

<div class="callout-danger">

**Danger:** Never use future information when constructing the high-frequency regressor matrix. In a real-time nowcasting context, you only have data up to the current date -- using the full quarter of monthly data when nowcasting mid-quarter is a look-ahead bias that invalidates your results.

</div>


- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*.
- Lundberg, S., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Babii, A., et al. (2021). Machine learning time series regressions with an application to nowcasting. *JBES*, 40(3).
- Diebold, F., & Mariano, R. (1995). Comparing predictive accuracy. *JBES*, 13(3), 253–263.

---

## Summary

Machine learning methods extend MIDAS nowcasting to nonlinear settings:

- **Feature engineering is the key challenge**: flat stacking, summary statistics, PCA embeddings
- **Random forests** are robust, parallel, and naturally handle mixed-frequency features
- **Gradient boosting** (XGBoost, LightGBM) often achieves higher accuracy with proper regularization
- **SHAP values** provide interpretable, theoretically grounded feature attribution
- **Comparison with MIDAS** requires same data, same vintages, formal DM tests
- **Combination forecasts** often outperform individual models

Next: [Lasso MIDAS Notebook](../notebooks/01_lasso_midas.ipynb) — hands-on implementation.


---

## Conceptual Practice Questions

**Practice Question 1:** How does the ragged-edge problem affect the reliability of real-time nowcasts compared to pseudo out-of-sample exercises?

**Practice Question 2:** What is the key difference between direct and iterated multi-step forecasts in a MIDAS context?


---

## Cross-References

<a class="link-card" href="./01_regularized_midas_guide.md">
  <div class="link-card-title">01 Regularized Midas</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_regularized_midas_slides.md">
  <div class="link-card-title">01 Regularized Midas — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

