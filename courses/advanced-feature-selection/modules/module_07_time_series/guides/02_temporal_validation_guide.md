# Temporal Validation for Time Series Feature Selection

## In Brief

Standard k-fold cross-validation assumes iid observations and freely shuffles data between folds. Time series violate this assumption: observations are serially dependent, and future data cannot be used to select features for predicting the past. Temporal validation frameworks — walk-forward validation, purged cross-validation, and embargo periods — enforce a strict temporal boundary between information used for selection and information used for evaluation.

## Key Insight

The central principle of temporal validation is the **information barrier**: no information from time $t$ or later may influence a model evaluated on observations at time $t$. Standard k-fold routinely violates this by placing future observations in the training fold. The cost of this violation is optimistic bias in selection and evaluation — models appear better than they will perform on live data.

---

## 1. Why Standard k-Fold Fails for Time Series

### 1.1 The iid Assumption

Standard $k$-fold cross-validation partitions data by randomly assigning observations to $k$ folds. This is valid only when observations are independently and identically distributed. Time series observations are serially dependent: $y_t$ is correlated with $y_{t-1}, y_{t-2}, \ldots$

### 1.2 The Leakage Problem

When future observations appear in the training fold while past observations appear in the test fold, two forms of leakage occur:

**Look-ahead leakage:** The model is trained on data that includes the period it is being tested on. For feature selection: if features are ranked using data that overlaps with the test period, the ranking is contaminated by future information.

**Autocorrelation leakage:** Even without direct look-ahead, serial correlation between adjacent train and test observations means the model effectively "sees" the test period through correlation. A model trained on $y_{t-1}$ that is correlated with $y_{t+1}$ (which appears in test) has indirect access to test information.

### 1.3 Quantifying the Bias

For a stationary AR(1) process with autocorrelation coefficient $\rho$, the expected optimistic bias in cross-validated accuracy due to ignoring serial dependence is approximately:

$$\text{Bias} \approx \frac{2\rho}{1 - \rho^2} \cdot \text{Var}(y_t)$$

For typical financial return series, $\rho \approx 0.05$–$0.15$, giving a modest but consistently upward-biased estimate of predictive accuracy.

For features constructed with overlapping windows (e.g., a 20-day rolling mean on daily data), the effective autocorrelation is much higher, and the bias is substantially larger.

---

## 2. Walk-Forward Validation

### 2.1 Core Principle

Walk-forward validation (also called rolling-origin validation or time series cross-validation) generates train/test splits that respect temporal order: training always precedes testing.

### 2.2 Expanding Window

In the expanding window variant, the training set grows at each step while the test window remains fixed:

```
Split 1:  Train: [t_0, ..., t_k]      Test: [t_{k+1}, ..., t_{k+h}]
Split 2:  Train: [t_0, ..., t_{k+h}]  Test: [t_{k+h+1}, ..., t_{k+2h}]
Split 3:  Train: [t_0, ..., t_{k+2h}] Test: [t_{k+2h+1}, ..., t_{k+3h}]
```

where $h$ is the forecast horizon (test window size).

**Advantages:** Uses all available historical data; suitable when data are scarce.

**Disadvantages:** Early splits underrepresent recent regime conditions; computational cost grows with each split.

### 2.3 Sliding Window

The sliding window variant keeps both training and test window fixed, sliding forward in time:

```
Split 1:  Train: [t_0,   ..., t_W]          Test: [t_{W+1}, ..., t_{W+h}]
Split 2:  Train: [t_h,   ..., t_{W+h}]      Test: [t_{W+h+1}, ..., t_{W+2h}]
Split 3:  Train: [t_{2h}, ..., t_{W+2h}]   Test: [t_{W+2h+1}, ..., t_{W+3h}]
```

**Advantages:** More representative of recent regime; constant computational cost.

**Disadvantages:** Discards early history; may miss long-run patterns.

### 2.4 Python Implementation

```python
import numpy as np
import pandas as pd
from typing import Iterator, Tuple


class WalkForwardSplitter:
    """
    Walk-forward cross-validation splitter for time series.

    Yields (train_indices, test_indices) pairs in temporal order.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = None,
        min_train_size: int = None,
        window: str = 'expanding',
    ):
        """
        Parameters
        ----------
        n_splits : int
            Number of walk-forward splits.
        test_size : int
            Number of observations per test fold.
        min_train_size : int
            Minimum observations in first training fold.
        window : str
            'expanding' (growing train) or 'sliding' (fixed-size train).
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.window = window

    def split(
        self, X: np.ndarray, y: np.ndarray = None, groups=None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        test_size = self.test_size or max(1, n // (self.n_splits + 1))
        min_train = self.min_train_size or test_size

        # Calculate start of first test fold
        total_test = test_size * self.n_splits
        if n <= total_test + min_train:
            raise ValueError(
                f"Not enough samples ({n}) for {self.n_splits} splits "
                f"with test_size={test_size} and min_train={min_train}"
            )

        first_test_start = n - total_test

        for i in range(self.n_splits):
            test_start = first_test_start + i * test_size
            test_end = test_start + test_size
            test_idx = np.arange(test_start, test_end)

            if self.window == 'expanding':
                train_idx = np.arange(0, test_start)
            else:  # sliding
                train_start = max(0, test_start - (first_test_start))
                train_idx = np.arange(train_start, test_start)

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
```

---

## 3. Purged Cross-Validation (de Prado, 2018)

### 3.1 The Overlap Problem

In many financial applications, features are computed from overlapping windows. For example, a 20-day rolling return at time $t$ uses data from $[t-19, t]$. An observation at $t$ and another at $t+5$ share 15 days of data. When the model is trained on $t$ and tested on $t+5$, the training observation contaminates the test observation through this shared window.

**Formal definition:** Let $[t_i^0, t_i^1]$ be the span of data used to construct observation $i$. Two observations $i$ (train) and $j$ (test) overlap if their spans intersect:

$$[t_i^0, t_i^1] \cap [t_j^0, t_j^1] \neq \emptyset$$

### 3.2 Purging

**Purging** removes from the training set all observations whose label spans overlap with any test observation span.

Let $t^{test}_{start}$ be the start of the earliest test observation. Remove from training any observation $i$ such that:

$$t_i^1 \geq t^{test}_{start}$$

This ensures the training set contains only observations whose entire label period precedes the test period.

### 3.3 Embargo

**Embargo** extends purging by also removing observations immediately *after* the test period, to prevent leakage through serial correlation.

For an embargo period of $e$ observations, remove from training any observation $i$ satisfying:

$$t_i^0 \in [t^{test}_{end} + 1, \; t^{test}_{end} + e]$$

The embargo period $e$ should be set to cover the autocorrelation halflife of the residuals.

### 3.4 Python: Purged Walk-Forward Splitter

```python
import numpy as np
import pandas as pd
from typing import Iterator, Tuple


class PurgedWalkForwardSplitter:
    """
    Purged walk-forward cross-validation (de Prado, 2018, Chapter 7).

    Removes overlapping observations from training folds to prevent
    information leakage through overlapping label windows.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = None,
        embargo_pct: float = 0.01,
    ):
        """
        Parameters
        ----------
        n_splits : int
            Number of walk-forward splits.
        test_size : int
            Test fold size in observations.
        embargo_pct : float
            Embargo size as fraction of total observations.
            Removes this many observations after each test period.
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        groups: np.ndarray = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        groups : array-like of shape (n_samples,), optional
            If provided, groups[i] = t_i^1 = the end of label span for
            observation i (integer index). Used for purging.
            If None, assumes point-in-time labels (no overlap).
        """
        n = len(X)
        test_size = self.test_size or max(1, n // (self.n_splits + 1))
        embargo_size = max(1, int(n * self.embargo_pct))

        first_test_start = n - test_size * self.n_splits

        for fold in range(self.n_splits):
            test_start = first_test_start + fold * test_size
            test_end = min(test_start + test_size, n)
            test_idx = np.arange(test_start, test_end)

            # Purge: remove training obs whose label span overlaps test
            if groups is not None:
                # groups[i] = last index used to form observation i's label
                train_candidates = np.arange(0, test_start)
                # Remove observations where label end >= test_start
                label_ends = groups[train_candidates]
                purged_mask = label_ends < test_start
                train_idx = train_candidates[purged_mask]
            else:
                train_idx = np.arange(0, test_start)

            # Embargo: remove observations immediately after test period
            embargo_start = test_end
            embargo_end = min(test_end + embargo_size, n)
            # These would appear in future training folds — mark to skip
            # For this fold, embargo affects the next fold's training set
            # Here we remove them from the current training set if they leaked
            if groups is not None:
                # Also remove training obs whose span overlaps the embargo zone
                embargo_overlap_mask = groups[train_idx] >= embargo_start
                train_idx = train_idx[~embargo_overlap_mask]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
```

---

## 4. Combinatorial Purged Cross-Validation

### 4.1 Motivation

Standard walk-forward validation produces a single path through the data. Combinatorial purged CV (CPCV) generates multiple train/test paths simultaneously, yielding more robust estimates of feature selection stability and model performance.

### 4.2 Construction

For $N$ total data blocks of equal size, CPCV selects $k$ blocks as test sets in all $\binom{N}{k}$ combinations. For each combination, the training set consists of the remaining $N - k$ blocks, with purging and embargo applied.

The number of paths generated is:

$$\bar{n}(N, k) = \frac{N}{k} \binom{N}{k}$$

For $N = 10$, $k = 2$: $\binom{10}{2} = 45$ test combinations, generating 22.5 paths.

### 4.3 Python: Combinatorial Purged CV

```python
from itertools import combinations


class CombinatorialPurgedCV:
    """
    Combinatorial purged cross-validation (de Prado, 2018, Chapter 12).

    Generates multiple non-overlapping test paths from all combinations
    of N data blocks taken k at a time.
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        embargo_pct: float = 0.01,
    ):
        """
        Parameters
        ----------
        n_splits : int
            N — number of data blocks.
        n_test_splits : int
            k — number of test blocks per combination.
        embargo_pct : float
            Embargo fraction of total observations.
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
        groups: np.ndarray = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        block_size = n // self.n_splits
        embargo_size = max(1, int(n * self.embargo_pct))

        # Block boundaries
        blocks = []
        for i in range(self.n_splits):
            start = i * block_size
            end = (i + 1) * block_size if i < self.n_splits - 1 else n
            blocks.append(np.arange(start, end))

        # Generate all combinations of k test blocks
        for test_combo in combinations(range(self.n_splits), self.n_test_splits):
            test_blocks = set(test_combo)
            train_blocks = [i for i in range(self.n_splits) if i not in test_blocks]

            test_idx = np.concatenate([blocks[i] for i in sorted(test_combo)])
            test_start = test_idx.min()
            test_end = test_idx.max()

            # Build training set from non-test blocks
            train_candidates = np.concatenate([blocks[i] for i in train_blocks])

            # Purge: remove training obs overlapping with test
            if groups is not None:
                label_ends = groups[train_candidates]
                train_candidates = train_candidates[label_ends < test_start]

            # Embargo: remove observations in embargo zone after test
            embargo_zone = np.arange(test_end + 1, test_end + 1 + embargo_size)
            train_idx = train_candidates[~np.isin(train_candidates, embargo_zone)]

            yield train_idx, test_idx
```

---

## 5. Walk-Forward Feature Selection

### 5.1 Rebalancing Feature Sets Over Time

In a walk-forward feature selection framework, features are re-selected at each rebalance point using only data available up to that point. This produces a time series of selected feature sets — revealing which features are consistently selected (stable) versus those that enter and exit the set (unstable or regime-dependent).

```python
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import grangercausalitytests


def walk_forward_feature_selection(
    target: pd.Series,
    features: pd.DataFrame,
    n_splits: int = 5,
    top_k: int = 10,
    method: str = 'granger',
    max_lag: int = 3,
) -> pd.DataFrame:
    """
    Walk-forward feature selection: re-select features at each
    rebalance point using only historically available data.

    Returns
    -------
    pd.DataFrame of shape (n_splits, n_features)
        Boolean selection matrix: True if feature selected at that split.
    """
    n = len(target)
    block_size = n // (n_splits + 1)

    selection_history = []
    split_dates = []

    for fold in range(n_splits):
        train_end = (fold + 1) * block_size

        # Training data up to this rebalance point
        target_train = target.iloc[:train_end]
        features_train = features.iloc[:train_end]

        # Select features using chosen method
        if method == 'granger':
            scores = {}
            for col in features.columns:
                data_pair = pd.concat(
                    [target_train.rename('y'), features_train[col].rename('x')],
                    axis=1
                ).dropna()
                try:
                    res = grangercausalitytests(data_pair[['y', 'x']],
                                               maxlag=max_lag, verbose=False)
                    min_p = min(res[l][0]['ssr_ftest'][1] for l in res)
                    scores[col] = -min_p  # higher = more predictive
                except Exception:
                    scores[col] = -1.0

        elif method == 'mutual_info':
            scores_arr = mutual_info_regression(
                features_train.dropna(), target_train.dropna().loc[features_train.dropna().index]
            )
            scores = dict(zip(features.columns, scores_arr))

        # Select top-k features
        sorted_features = sorted(scores, key=scores.get, reverse=True)
        selected = {col: col in sorted_features[:top_k] for col in features.columns}

        selection_history.append(selected)
        split_dates.append(target.index[train_end - 1] if hasattr(target.index, '__len__') else train_end)

    return pd.DataFrame(selection_history, index=split_dates)
```

---

## 6. Feature Stability Metrics

### 6.1 Why Stability Matters

A feature selected in 8 out of 10 walk-forward folds is more trustworthy than one selected in 2 out of 10, even if it has higher average importance in the 2 folds it is selected. Unstable features are sensitive to specific regime conditions and may not generalise.

### 6.2 Kendall's W (Coefficient of Concordance)

Kendall's $W$ measures agreement in feature rankings across multiple time periods. For $m$ raters (folds) and $n$ features:

$$W = \frac{12 S}{m^2(n^3 - n)}$$

where $S = \sum_{i=1}^{n} \left(R_i - \bar{R}\right)^2$ and $R_i = \sum_{j=1}^{m} r_{ij}$ is the sum of ranks for feature $i$.

$W \in [0, 1]$: $W = 1$ means perfect agreement across all folds; $W = 0$ means no agreement.

### 6.3 Spearman Footrule

For two ranking vectors $\sigma$ and $\tau$ on $n$ features:

$$F(\sigma, \tau) = \sum_{i=1}^{n} |\sigma(i) - \tau(i)|$$

Normalized footrule distance:

$$d_F = \frac{F(\sigma, \tau)}{\lfloor n^2 / 2 \rfloor}$$

$d_F \in [0, 1]$: 0 means identical rankings; 1 means completely reversed.

```python
from scipy.stats import kendalltau, spearmanr
import numpy as np


def feature_stability_metrics(
    selection_matrix: pd.DataFrame,
    ranking_matrix: pd.DataFrame = None,
) -> dict:
    """
    Compute stability metrics for walk-forward feature selection.

    Parameters
    ----------
    selection_matrix : pd.DataFrame, shape (n_folds, n_features)
        Boolean: True if feature selected at each fold.
    ranking_matrix : pd.DataFrame, shape (n_folds, n_features)
        Feature importance scores at each fold (optional).

    Returns
    -------
    dict with:
        'selection_frequency': fraction of folds each feature is selected
        'kendalls_w': concordance across all fold rankings
        'pairwise_spearman': mean pairwise Spearman correlation across folds
    """
    n_folds, n_features = selection_matrix.shape

    # Selection frequency
    sel_freq = selection_matrix.mean(axis=0)

    # Pairwise Spearman correlation of selection scores across folds
    if ranking_matrix is not None:
        spearman_corrs = []
        for i in range(n_folds):
            for j in range(i + 1, n_folds):
                corr, _ = spearmanr(ranking_matrix.iloc[i], ranking_matrix.iloc[j])
                spearman_corrs.append(corr)
        mean_spearman = np.mean(spearman_corrs)
    else:
        mean_spearman = None

    # Kendall's W (if rankings available)
    if ranking_matrix is not None:
        # Rank each fold's scores
        ranks = ranking_matrix.rank(axis=1)
        m = n_folds
        n = n_features
        R = ranks.sum(axis=0)
        R_bar = R.mean()
        S = ((R - R_bar) ** 2).sum()
        W = 12 * S / (m ** 2 * (n ** 3 - n))
    else:
        W = None

    return {
        'selection_frequency': sel_freq,
        'kendalls_w': W,
        'pairwise_spearman': mean_spearman,
    }
```

---

## 7. Temporal Feature Decay

### 7.1 Features That Lose Predictive Power

Some features are predictive only in specific market regimes or over limited time horizons. A feature selected based on 10 years of history may have had its predictive power concentrated in a single regime that no longer exists.

**Temporal decay analysis:**

```python
def temporal_predictive_decay(
    target: pd.Series,
    feature: pd.Series,
    window: int = 252,
    step: int = 21,
) -> pd.DataFrame:
    """
    Track feature predictive power over rolling windows.

    Computes mutual information between feature and target
    over a rolling window, revealing how predictive power
    evolves over time.

    Returns pd.DataFrame with rolling MI and z-scored MI.
    """
    from sklearn.feature_selection import mutual_info_regression

    results = []
    indices = []

    for end in range(window, len(target), step):
        start = end - window
        y_window = target.iloc[start:end].dropna()
        x_window = feature.iloc[start:end].dropna()
        common = y_window.index.intersection(x_window.index)

        if len(common) < window // 2:
            continue

        mi = mutual_info_regression(
            x_window.loc[common].values.reshape(-1, 1),
            y_window.loc[common].values,
            random_state=42
        )[0]

        results.append(mi)
        indices.append(target.index[end - 1])

    mi_series = pd.Series(results, index=indices, name='rolling_mi')
    z_scored = (mi_series - mi_series.mean()) / mi_series.std()

    return pd.DataFrame({'rolling_mi': mi_series, 'z_scored_mi': z_scored})
```

### 7.2 Decay Metrics

| Metric | Definition | Interpretation |
|---|---|---|
| Decay slope | OLS slope of MI vs time | Negative = feature losing power |
| Half-life | Time for MI to drop 50% | Short = unstable feature |
| Minimum window | Smallest window giving MI > 0.05 | Long = requires stable regime |
| Recency bias | Recent-window MI / full-window MI | < 1 = decaying, > 1 = strengthening |

---

## Common Pitfalls

- **Shuffling time series data:** Never shuffle when splitting — this destroys temporal structure.
- **Selecting features on full data before walk-forward validation:** Always select and validate within the same temporal split.
- **Ignoring label overlap:** Overlapping rolling-window labels between train and test create invisible leakage.
- **Setting embargo too short:** A short embargo does not cover the autocorrelation horizon of overlapping features.
- **Reporting single-fold performance:** Walk-forward validation must aggregate across all folds; a single fold result is not representative.

---

## Connections

- **Builds on:** Module 03 wrappers (CV framework); stationarity concepts from Granger guide
- **Leads to:** Module 07 regime-aware selection (time-varying feature sets); Module 11 production deployment
- **Related to:** Backtesting in quantitative finance; time series forecasting evaluation

---

## Further Reading

- de Prado, M.L. (2018). *Advances in Financial Machine Learning*. Wiley. Chapters 7 (purged CV) and 12 (combinatorial purged CV). — Essential reference for financial ML validation.
- Bergmeir, C. & Benítez, J.M. (2012). "On the Use of Cross-Validation for Time Series Predictor Evaluation." *Information Sciences*, 191, 192–213. — Systematic comparison of CV strategies.
- Racine, J. (2000). "Consistent Cross-Validatory Model-Selection for Dependent Data." *Journal of Econometrics*, 99(2), 379–399. — Theoretical foundation for time series CV.
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). — Chapter on cross-validation for time series.
