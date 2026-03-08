# Regime-Aware Feature Selection for Time Series

## In Brief

Financial and economic systems cycle through distinct operating regimes — bull and bear markets, high and low volatility periods, expansions and recessions — in which the statistical relationships between features and targets change substantially. Regime-aware feature selection maintains separate feature sets for each detected regime, adapts selection when regime shifts are detected, and monitors features for drift that signals deteriorating predictive relevance.

## Key Insight

A single global feature set averaged across regimes may be suboptimal in every regime. The optimal set of features for predicting bond returns during a flight-to-quality crisis is unlikely to be the same as during a credit expansion. Regime-aware selection exploits this heterogeneity rather than averaging it away.

---

## 1. Regime Detection Methods

### 1.1 Hidden Markov Models

Hidden Markov Models (HMMs) model the observed time series as emissions from a latent discrete state (regime) process. The latent state $s_t \in \{1, \ldots, K\}$ follows a first-order Markov chain with transition matrix $\mathbf{P}$:

$$P(s_t = j \mid s_{t-1} = i) = p_{ij}$$

At each time step, the observation $y_t$ is drawn from a state-conditional distribution:

$$y_t \mid s_t = k \sim \mathcal{N}(\mu_k, \sigma_k^2) \quad (\text{Gaussian emission})$$

Parameters $\theta = \{\mu_k, \sigma_k, p_{ij}\}$ are estimated with the **Baum-Welch algorithm** (expectation-maximisation for HMMs).

After fitting, regime assignments are decoded with the **Viterbi algorithm** (most probable state sequence) or via posterior probabilities $\gamma_t(k) = P(s_t = k \mid y_{1:T}, \theta)$.

```python
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings('ignore')


def fit_hmm_regimes(
    returns: pd.Series,
    n_states: int = 2,
    n_iter: int = 200,
    random_state: int = 42,
) -> dict:
    """
    Fit a Gaussian HMM to detect market regimes.

    Parameters
    ----------
    returns : pd.Series
        Return series (stationary, e.g., daily log-returns).
    n_states : int
        Number of hidden regimes.
    n_iter : int
        EM iterations.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    dict with:
        'model'        : fitted GaussianHMM
        'states'       : pd.Series of decoded state labels
        'posteriors'   : pd.DataFrame of posterior state probabilities
        'regime_stats' : pd.DataFrame with mean, std, duration per regime
    """
    X = returns.dropna().values.reshape(-1, 1)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type='full',
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-4,
    )
    model.fit(X)

    # Viterbi decoding
    states = model.predict(X)
    posteriors = model.predict_proba(X)

    # Align with original index
    idx = returns.dropna().index
    state_series = pd.Series(states, index=idx, name='regime')
    posterior_df = pd.DataFrame(
        posteriors,
        index=idx,
        columns=[f'P(regime_{k})' for k in range(n_states)],
    )

    # Regime statistics
    regime_stats = []
    for k in range(n_states):
        mask = state_series == k
        regime_returns = returns.dropna()[mask]
        run_lengths = _compute_run_lengths(states, k)

        regime_stats.append({
            'regime': k,
            'mean_return': regime_returns.mean(),
            'volatility': regime_returns.std(),
            'frequency': mask.mean(),
            'avg_duration': np.mean(run_lengths) if run_lengths else 0,
            'label': 'low_vol' if regime_returns.std() < returns.std() else 'high_vol',
        })

    return {
        'model': model,
        'states': state_series,
        'posteriors': posterior_df,
        'regime_stats': pd.DataFrame(regime_stats),
    }


def _compute_run_lengths(states: np.ndarray, k: int) -> list:
    """Compute lengths of consecutive runs of state k."""
    runs = []
    current_run = 0
    for s in states:
        if s == k:
            current_run += 1
        elif current_run > 0:
            runs.append(current_run)
            current_run = 0
    if current_run > 0:
        runs.append(current_run)
    return runs
```

### 1.2 Threshold-Based Regime Detection

Simpler than HMM, threshold methods define regimes by observable indicator thresholds:

```python
def threshold_regime_detection(
    series: pd.Series,
    indicator: pd.Series,
    thresholds: list,
    labels: list = None,
) -> pd.Series:
    """
    Define regimes by thresholds on an observable indicator.

    Examples:
    - VIX > 25: high-volatility regime
    - Yield curve spread < 0: recession warning regime
    - 200-day momentum > 0: trend regime

    Parameters
    ----------
    series : pd.Series
        Main time series (target or returns).
    indicator : pd.Series
        Observable regime indicator.
    thresholds : list of float
        Cutoff values defining regime boundaries.
    labels : list of str
        Regime names (len = len(thresholds) + 1).
    """
    if labels is None:
        labels = [f'regime_{i}' for i in range(len(thresholds) + 1)]

    regime = pd.Series(labels[0], index=indicator.index, name='regime')
    for i, threshold in enumerate(thresholds):
        regime[indicator >= threshold] = labels[i + 1]

    return regime
```

### 1.3 Change Point Detection

Change point detection identifies structural breaks in a time series, useful when regime transitions are abrupt rather than gradual.

```python
import ruptures as rpt  # pip install ruptures


def detect_change_points(
    series: pd.Series,
    model: str = 'rbf',
    n_bkps: int = 5,
    min_size: int = 20,
) -> list:
    """
    Detect structural change points using the ruptures library.

    Parameters
    ----------
    series : pd.Series
        Time series to segment.
    model : str
        Cost model: 'rbf' (variance), 'l1', 'l2', 'normal'.
    n_bkps : int
        Number of breakpoints to detect.
    min_size : int
        Minimum segment length.

    Returns
    -------
    list of int indices where change points occur.
    """
    signal = series.dropna().values
    algo = rpt.Pelt(model=model, min_size=min_size).fit(signal)
    breakpoints = algo.predict(pen=np.log(len(signal)) * signal.var())

    # Alternatively, fix number of breakpoints
    # algo = rpt.Binseg(model=model, min_size=min_size).fit(signal)
    # breakpoints = algo.predict(n_bkps=n_bkps)

    return breakpoints[:-1]  # Remove the last index (end of series)
```

---

## 2. Regime-Conditioned Feature Selection

### 2.1 Separate Feature Sets Per Regime

Once regimes are identified, feature selection runs independently within each regime:

```python
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import grangercausalitytests


def regime_conditioned_feature_selection(
    target: pd.Series,
    features: pd.DataFrame,
    regimes: pd.Series,
    method: str = 'mutual_info',
    top_k: int = 10,
    max_lag: int = 3,
) -> dict:
    """
    Select features separately for each detected regime.

    Parameters
    ----------
    target : pd.Series
        Target variable.
    features : pd.DataFrame
        Candidate features.
    regimes : pd.Series
        Regime labels aligned with target index.
    method : str
        'mutual_info' or 'granger'.
    top_k : int
        Features to select per regime.
    max_lag : int
        Lag order for Granger tests.

    Returns
    -------
    dict mapping regime_label -> list of selected feature names.
    """
    unique_regimes = sorted(regimes.unique())
    regime_features = {}

    for regime in unique_regimes:
        # Filter to this regime
        regime_mask = regimes == regime
        common_idx = target.index.intersection(features.index).intersection(
            regimes[regime_mask].index
        )

        target_regime = target.loc[common_idx].dropna()
        features_regime = features.loc[common_idx].dropna()

        # Align
        common = target_regime.index.intersection(features_regime.index)
        target_r = target_regime.loc[common]
        features_r = features_regime.loc[common]

        n_obs = len(common)
        print(f"Regime '{regime}': {n_obs} observations")

        if n_obs < 30:
            print(f"  Too few observations — skipping.")
            regime_features[regime] = []
            continue

        if method == 'mutual_info':
            scores = mutual_info_regression(
                features_r.fillna(0), target_r.values, random_state=42
            )
            ranked = sorted(
                zip(features.columns, scores),
                key=lambda x: x[1], reverse=True
            )
            regime_features[regime] = [f for f, _ in ranked[:top_k]]

        elif method == 'granger':
            pvalues = {}
            for col in features.columns:
                data = pd.concat(
                    [target_r.rename('y'), features_r[col].rename('x')], axis=1
                ).dropna()
                try:
                    res = grangercausalitytests(data[['y', 'x']],
                                               maxlag=max_lag, verbose=False)
                    pvalues[col] = min(res[l][0]['ssr_ftest'][1] for l in res)
                except Exception:
                    pvalues[col] = 1.0
            ranked = sorted(pvalues, key=pvalues.get)
            regime_features[regime] = ranked[:top_k]

    return regime_features
```

### 2.2 Overlap and Divergence Analysis

After selecting regime-specific feature sets, quantify how much they differ:

```python
def regime_feature_overlap(regime_features: dict) -> pd.DataFrame:
    """
    Compute pairwise Jaccard similarity between regime feature sets.
    """
    regimes = list(regime_features.keys())
    n = len(regimes)
    overlap = pd.DataFrame(np.zeros((n, n)), index=regimes, columns=regimes)

    for i, r1 in enumerate(regimes):
        for j, r2 in enumerate(regimes):
            s1 = set(regime_features[r1])
            s2 = set(regime_features[r2])
            if len(s1 | s2) == 0:
                overlap.loc[r1, r2] = 0.0
            else:
                overlap.loc[r1, r2] = len(s1 & s2) / len(s1 | s2)

    return overlap
```

---

## 3. Markov Switching Regression

### 3.1 Model Specification

Markov switching regression (Hamilton, 1989) models the regression relationship between features and target as switching between $K$ regimes, where the active regime follows a Markov chain:

$$y_t = \mathbf{x}_t^\top \boldsymbol{\beta}_{s_t} + \sigma_{s_t} \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, 1)$$

The feature coefficients $\boldsymbol{\beta}_{s_t}$ and noise level $\sigma_{s_t}$ are regime-dependent. Regime transitions follow:

$$P(s_t = j \mid s_{t-1} = i) = p_{ij}$$

This model simultaneously estimates which features matter in each regime and how regimes transition.

```python
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression


def fit_markov_switching(
    target: pd.Series,
    features: pd.DataFrame,
    n_regimes: int = 2,
    switching_variance: bool = True,
) -> dict:
    """
    Fit Markov switching regression with regime-varying coefficients.

    Parameters
    ----------
    target : pd.Series
        Target variable.
    features : pd.DataFrame
        Feature matrix (should be selected/small set).
    n_regimes : int
        Number of switching regimes.
    switching_variance : bool
        Allow variance to switch across regimes.

    Returns
    -------
    dict with fitted model, regime-specific coefficients, and smoothed probabilities.
    """
    # Align data
    data = pd.concat([target.rename('y'), features], axis=1).dropna()
    y = data['y']
    X = data[features.columns]

    model = MarkovRegression(
        endog=y,
        k_regimes=n_regimes,
        trend='c',
        exog=X,
        switching_variance=switching_variance,
    )
    result = model.fit(disp=False)

    # Extract regime-specific coefficients
    regime_coeffs = {}
    for k in range(n_regimes):
        coeff_names = ['const'] + list(features.columns)
        # MarkovRegression stores params in a specific order
        coeffs = {}
        for j, name in enumerate(coeff_names):
            param_name = f'regime{k}.{name}' if n_regimes > 1 else name
            # Use the result summary to extract coefficients
            coeffs[name] = result.params.get(param_name, np.nan)
        regime_coeffs[k] = coeffs

    return {
        'result': result,
        'regime_coefficients': regime_coeffs,
        'smoothed_probs': result.smoothed_marginal_probabilities,
        'aic': result.aic,
        'bic': result.bic,
    }
```

---

## 4. Feature Drift Detection

### 4.1 Why Drift Matters

Feature drift occurs when the statistical properties of a feature — its distribution, its correlation with the target, or its predictive relevance — change over time. Drift is distinct from regime change: it represents a structural deterioration of the feature rather than a cyclical shift.

### 4.2 Population Stability Index (PSI)

PSI compares the distribution of a feature at two points in time. It was originally developed in credit risk modelling to monitor scorecard stability.

$$\text{PSI} = \sum_{b=1}^{B} \left(p^{(t)}_b - p^{(\text{ref})}_b \right) \ln \frac{p^{(t)}_b}{p^{(\text{ref})}_b}$$

where $p^{(t)}_b$ and $p^{(\text{ref})}_b$ are the fractions of observations in bin $b$ during the current period and reference period respectively.

| PSI | Interpretation |
|---|---|
| < 0.10 | Negligible shift |
| 0.10 – 0.20 | Moderate shift — monitor |
| > 0.20 | Significant shift — investigate |
| > 0.25 | Major shift — consider re-selection |

```python
def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Population Stability Index between two distributions.

    Parameters
    ----------
    reference : np.ndarray
        Feature values in the reference (training) period.
    current : np.ndarray
        Feature values in the current (monitoring) period.
    n_bins : int
        Number of quantile bins.

    Returns
    -------
    float : PSI value.
    """
    # Compute quantile bins from reference
    breakpoints = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    ref_pct = ref_counts / (len(reference) + eps) + eps
    cur_pct = cur_counts / (len(current) + eps) + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)
```

### 4.3 Kolmogorov-Smirnov Test

The two-sample KS test detects distributional shift without requiring binning:

```python
from scipy.stats import ks_2samp


def ks_drift_test(
    reference: np.ndarray,
    current: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Two-sample KS test for feature distribution drift.

    Returns dict with ks_statistic, pvalue, and drift_detected flag.
    """
    ks_stat, p_value = ks_2samp(reference, current)
    return {
        'ks_statistic': ks_stat,
        'pvalue': p_value,
        'drift_detected': p_value < alpha,
    }
```

### 4.4 Wasserstein Distance

The Wasserstein-1 (earth mover's) distance measures the minimum transport cost between two distributions and is more sensitive than KS for detecting shifts in distribution tails:

```python
from scipy.stats import wasserstein_distance


def wasserstein_drift(
    reference: np.ndarray,
    current: np.ndarray,
) -> float:
    """
    Compute Wasserstein-1 distance between reference and current distributions.
    """
    return wasserstein_distance(reference, current)
```

### 4.5 Monitoring Pipeline

```python
def feature_drift_monitor(
    features_reference: pd.DataFrame,
    features_current: pd.DataFrame,
    psi_threshold: float = 0.20,
    ks_alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Monitor all features for distribution drift.

    Returns DataFrame with PSI, KS test results, and Wasserstein
    distances for each feature. Flags features requiring re-selection.
    """
    records = []

    for col in features_reference.columns:
        ref = features_reference[col].dropna().values
        cur = features_current[col].dropna().values

        if len(ref) < 10 or len(cur) < 10:
            continue

        psi = compute_psi(ref, cur)
        ks = ks_drift_test(ref, cur, alpha=ks_alpha)
        w_dist = wasserstein_drift(ref, cur)

        records.append({
            'feature': col,
            'psi': psi,
            'psi_flag': psi > psi_threshold,
            'ks_statistic': ks['ks_statistic'],
            'ks_pvalue': ks['pvalue'],
            'ks_drift': ks['drift_detected'],
            'wasserstein': w_dist,
            'reselect': psi > psi_threshold or ks['drift_detected'],
        })

    return pd.DataFrame(records).sort_values('psi', ascending=False)
```

---

## 5. Adaptive Re-Selection Triggers

### 5.1 When to Re-Run Feature Selection

Re-selection is computationally expensive. Trigger it conditionally on evidence of regime change or feature drift:

```python
class AdaptiveReselectionController:
    """
    Monitors signals and triggers feature re-selection when warranted.

    Trigger conditions:
    - Regime transition detected (HMM posterior shift)
    - Feature drift detected (PSI or KS test exceeds threshold)
    - Predictive performance degradation (rolling validation error spike)
    - Calendar-based forced rebalance (monthly, quarterly)
    """

    def __init__(
        self,
        psi_trigger: float = 0.20,
        regime_prob_trigger: float = 0.80,
        performance_drop_trigger: float = 0.15,
        forced_rebalance_period: int = 63,  # quarterly in business days
    ):
        self.psi_trigger = psi_trigger
        self.regime_prob_trigger = regime_prob_trigger
        self.performance_drop_trigger = performance_drop_trigger
        self.forced_rebalance_period = forced_rebalance_period
        self.last_reselect_date = None
        self.baseline_performance = None

    def should_reselect(
        self,
        current_date,
        current_regime: int,
        previous_regime: int,
        drift_results: pd.DataFrame,
        current_performance: float,
        days_since_last: int,
    ) -> dict:
        """
        Evaluate whether re-selection should be triggered.

        Returns dict with trigger decision and reason.
        """
        triggers = []

        # Regime change trigger
        if current_regime != previous_regime:
            triggers.append('regime_change')

        # Drift trigger
        drifting_features = drift_results[drift_results['reselect']]['feature'].tolist()
        if len(drifting_features) > 0:
            triggers.append(f'feature_drift: {drifting_features}')

        # Performance degradation trigger
        if self.baseline_performance is not None:
            performance_drop = (self.baseline_performance - current_performance) / abs(self.baseline_performance)
            if performance_drop > self.performance_drop_trigger:
                triggers.append(f'performance_drop: {performance_drop:.2%}')

        # Forced calendar trigger
        if days_since_last >= self.forced_rebalance_period:
            triggers.append('calendar_rebalance')

        should = len(triggers) > 0

        return {
            'reselect': should,
            'triggers': triggers,
            'date': current_date,
        }
```

---

## 6. Evolutionary Feature Selection for Time Series

### 6.1 Walk-Forward Fitness

When using genetic algorithms for time series feature selection, the fitness function must respect temporal order. Walk-forward fitness evaluates each chromosome (feature subset) using purged walk-forward cross-validation:

```python
def walk_forward_fitness(
    chromosome: np.ndarray,
    target: pd.Series,
    features: pd.DataFrame,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> float:
    """
    Evaluate fitness of a feature subset using purged walk-forward CV.

    This replaces the standard cross-validated accuracy used in
    non-temporal GA feature selection.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error

    selected = features.columns[chromosome.astype(bool)]
    if len(selected) == 0:
        return -np.inf

    X = features[selected].dropna()
    y = target.loc[X.index].dropna()
    common = X.index.intersection(y.index)
    X, y = X.loc[common], y.loc[common]

    splitter = PurgedWalkForwardSplitter(
        n_splits=n_splits,
        embargo_pct=embargo_pct,
    )

    errors = []
    for train_idx, test_idx in splitter.split(X.values):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        errors.append(mean_squared_error(y_test, preds))

    return -np.mean(errors)  # Negate for maximization
```

### 6.2 Temporal Chromosome Encoding

For regime-aware GA, the chromosome encodes feature selections for each regime separately:

```
Chromosome layout for 2 regimes, 20 features each:
[regime_0_features: 0 1 0 1 0 ... (20 bits)] [regime_1_features: 1 0 0 1 1 ... (20 bits)]
```

This allows the GA to evolve regime-specific feature sets simultaneously, with crossover respecting regime boundaries.

---

## 7. Online Feature Selection for Streaming Data

### 7.1 Incremental Feature Scoring

For streaming data where full retraining is impractical, online feature selection updates scores incrementally as new data arrives:

```python
class OnlineFeatureSelector:
    """
    Incremental feature selection for streaming time series.

    Uses an exponentially-weighted covariance estimator to compute
    rolling mutual information approximations without storing all data.
    """

    def __init__(
        self,
        n_features: int,
        top_k: int = 10,
        decay: float = 0.99,
    ):
        """
        Parameters
        ----------
        n_features : int
            Total number of candidate features.
        top_k : int
            Number of features to select.
        decay : float
            Exponential weight for older observations (0 < decay < 1).
        """
        self.n_features = n_features
        self.top_k = top_k
        self.decay = decay
        self.t = 0

        # Online moments
        self.ew_mean_x = np.zeros(n_features)
        self.ew_mean_y = 0.0
        self.ew_var_x = np.ones(n_features)
        self.ew_var_y = 1.0
        self.ew_cov_xy = np.zeros(n_features)

    def update(
        self,
        x: np.ndarray,
        y: float,
    ) -> list:
        """
        Process one new observation and return updated feature ranking.

        Parameters
        ----------
        x : np.ndarray, shape (n_features,)
            Feature vector at current time step.
        y : float
            Target value at current time step.

        Returns
        -------
        list of top_k feature indices ranked by online correlation.
        """
        self.t += 1
        alpha = 1 - self.decay

        # Update exponentially weighted moments
        self.ew_mean_x = self.decay * self.ew_mean_x + alpha * x
        self.ew_mean_y = self.decay * self.ew_mean_y + alpha * y
        self.ew_var_x = self.decay * self.ew_var_x + alpha * (x - self.ew_mean_x) ** 2
        self.ew_var_y = self.decay * self.ew_var_y + alpha * (y - self.ew_mean_y) ** 2
        self.ew_cov_xy = self.decay * self.ew_cov_xy + alpha * (x - self.ew_mean_x) * (y - self.ew_mean_y)

        # Online Pearson correlation as score
        denom = np.sqrt(self.ew_var_x * self.ew_var_y) + 1e-10
        online_corr = np.abs(self.ew_cov_xy / denom)

        # Return top-k feature indices
        return list(np.argsort(online_corr)[::-1][:self.top_k])

    def selected_features(self) -> list:
        """Return current top-k feature indices."""
        denom = np.sqrt(self.ew_var_x * self.ew_var_y) + 1e-10
        online_corr = np.abs(self.ew_cov_xy / denom)
        return list(np.argsort(online_corr)[::-1][:self.top_k])
```

---

## Common Pitfalls

- **Using future regime labels for selection:** HMM regimes must be decoded with only historical data. Smoothed posteriors use future data; use filtered posteriors ($\gamma_t(k)$ conditioned on data up to $t$) for real-time decisions.
- **Too few observations per regime:** Regime-specific feature selection requires sufficient regime observations. With fewer than ~50–100 observations per regime, selection estimates are unreliable.
- **Number of regimes (K):** Choosing K via BIC from HMM — 2–4 regimes are typical for financial data; more regimes may overfit the regime structure.
- **PSI threshold context-dependence:** PSI > 0.20 is a guideline, not a law. Calibrate against historical drift-performance relationships in your specific domain.
- **Regime label non-stationarity:** If the regime detector itself uses a rolling window, its labels are non-stationary — earlier observations may have been labelled differently as more data arrived.

---

## Connections

- **Builds on:** Module 07 Granger causality (feature selection within regimes); Module 05 genetic algorithms (GA for regime-aware selection)
- **Leads to:** Module 11 production deployment (online selection, monitoring pipelines)
- **Related to:** Hidden Markov Models (entire `hidden-markov-models` course); structural break tests; concept drift in ML

---

## Further Reading

- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2), 357–384. — Markov switching regression.
- Havelock, R. & Madan, D. (2021). "Regime-Aware Asset Allocation." *Journal of Portfolio Management*, 47(5). — Financial application.
- Webb, G.I. et al. (2016). "Characterizing Concept Drift." *Data Mining and Knowledge Discovery*, 30(4), 964–994. — Concept drift taxonomy.
- Gama, J. et al. (2014). "A Survey on Concept Drift Adaptation." *ACM Computing Surveys*, 46(4), 1–37. — Comprehensive drift survey.
- de Prado, M.L. (2018). *Advances in Financial Machine Learning*. Wiley. — Chapter 17: Structural Breaks.
