# Feature Drift Monitoring and Adaptive Re-Selection

## In Brief

Feature drift occurs when the statistical properties of input features change between training time and serving time. Undetected drift silently degrades model performance. This guide covers the primary drift detection methods — PSI, KS test, and Wasserstein distance — and connects them to automated re-selection triggers that keep deployed feature sets current.

## Key Insight

Feature drift and concept drift are distinct problems. Feature drift: $p_{train}(X) \neq p_{serve}(X)$ — the distribution of inputs shifts. Concept drift: $p_{train}(Y|X) \neq p_{serve}(Y|X)$ — the relationship between inputs and target shifts. Both degrade model performance, but they require different responses. Feature drift calls for re-preprocessing or re-selection. Concept drift calls for re-training on updated labels.

---

## Population Stability Index (PSI)

PSI is the industry standard for measuring feature distribution shift between two datasets. It was developed in credit scoring but now applies across all tabular ML domains.

### Definition

For feature $X$ with reference distribution $p$ (training) and current distribution $q$ (serving):

$$\text{PSI}(p, q) = \sum_{i=1}^{B} \left(q_i - p_i\right) \ln \frac{q_i}{p_i}$$

where $B$ is the number of bins and $p_i$, $q_i$ are the proportions of observations falling in bin $i$ under the reference and current distributions respectively.

PSI is the symmetric Kullback-Leibler divergence:

$$\text{PSI}(p, q) = \text{KL}(q \| p) + \text{KL}(p \| q)$$

### Interpretation

| PSI Value | Interpretation | Action |
|-----------|---------------|--------|
| < 0.1 | No significant change | Monitor only |
| 0.1 – 0.2 | Moderate shift | Investigate; consider re-selection |
| > 0.2 | Major shift | Re-select features; possibly retrain |

### Implementation

```python
import numpy as np
import pandas as pd
from typing import Optional


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Compute Population Stability Index between reference and current distributions.

    Parameters
    ----------
    reference : array of shape (n_reference,)
        Training distribution for one feature.
    current : array of shape (n_current,)
        Production distribution for the same feature.
    n_bins : int
        Number of equal-width bins computed on reference distribution.
    eps : float
        Smoothing constant to avoid log(0).

    Returns
    -------
    float
        PSI value. Higher = more drift.
    """
    # Define bins on reference distribution; extend edges to cover current outliers
    bin_edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bin_edges[0]  -= 1e-10  # include minimum
    bin_edges[-1] += 1e-10  # include maximum

    # Count observations per bin
    ref_counts = np.histogram(reference, bins=bin_edges)[0]
    cur_counts = np.histogram(current,   bins=bin_edges)[0]

    # Convert to proportions and smooth
    ref_props = (ref_counts / ref_counts.sum()) + eps
    cur_props = (cur_counts / cur_counts.sum()) + eps

    # Re-normalise after smoothing
    ref_props /= ref_props.sum()
    cur_props /= cur_props.sum()

    psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
    return float(psi)


def compute_psi_dataframe(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute PSI for every feature in a DataFrame."""
    results = []
    for col in reference.columns:
        psi_val = compute_psi(reference[col].values, current[col].values, n_bins)
        flag    = 'critical' if psi_val > 0.2 else ('warning' if psi_val > 0.1 else 'ok')
        results.append({'feature': col, 'psi': psi_val, 'status': flag})
    return pd.DataFrame(results).sort_values('psi', ascending=False)
```

### Why Bins Rather than Density Estimation?

Binned PSI is computationally O(n) and requires no distributional assumption. Kernel density estimation is more statistically principled but orders of magnitude slower and sensitive to bandwidth choice. For monitoring millions of daily predictions, PSI's speed advantage is decisive.

---

## Kolmogorov-Smirnov Test for Distribution Changes

The two-sample KS test checks whether two samples are drawn from the same distribution by comparing their empirical CDFs. It makes no distributional assumption and applies to any continuous feature.

### Definition

For empirical CDFs $F_n$ (reference, $n$ samples) and $G_m$ (current, $m$ samples):

$$D_{n,m} = \sup_x |F_n(x) - G_m(x)|$$

Under $H_0$ (same distribution), the test statistic follows the Kolmogorov distribution. A small p-value rejects $H_0$ — the distributions differ significantly.

### Implementation

```python
from scipy.stats import ks_2samp


def ks_drift_test(
    reference: np.ndarray,
    current: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """
    Two-sample Kolmogorov-Smirnov drift test.

    Returns
    -------
    dict with keys: statistic, pvalue, drifted (bool)
    """
    stat, pvalue = ks_2samp(reference, current)
    return {
        'statistic': float(stat),
        'pvalue':    float(pvalue),
        'drifted':   pvalue < alpha,
    }


def ks_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run KS test for every feature column."""
    rows = []
    for col in reference.columns:
        result = ks_drift_test(reference[col].values, current[col].values, alpha)
        rows.append({'feature': col, **result})
    df = pd.DataFrame(rows).sort_values('statistic', ascending=False)
    n_drifted = df['drifted'].sum()
    print(f"{n_drifted}/{len(df)} features show statistically significant drift "
          f"(alpha={alpha})")
    return df
```

### KS vs. PSI

| Criterion | PSI | KS Test |
|-----------|-----|---------|
| Provides p-value | No | Yes |
| Interpretable threshold | Yes (0.1/0.2) | Depends on alpha |
| Sensitive to tail shifts | Moderate | High |
| Industry standard | Finance | Academic/science |
| Multiple testing correction needed | No | Yes (Bonferroni) |

Use PSI for operational dashboards with categorical thresholds. Use KS for statistical reporting where p-values are required.

---

## Wasserstein Distance for Continuous Drift Quantification

The 1-Wasserstein distance (Earth Mover's Distance) measures the minimum "work" required to transform one distribution into another. Unlike KS, it is a metric that quantifies the magnitude of drift, not just its existence.

### Definition

$$W_1(p, q) = \int_{-\infty}^{\infty} |F_p(x) - F_q(x)| \, dx$$

where $F_p$ and $F_q$ are the CDFs of $p$ and $q$. For empirical samples this equals the L1 distance between their sorted values.

```python
from scipy.stats import wasserstein_distance


def wasserstein_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    normalise: bool = True,
) -> pd.DataFrame:
    """
    Compute Wasserstein distance per feature.

    Parameters
    ----------
    normalise : bool
        If True, divide by the reference standard deviation so distances
        are comparable across features with different scales.
    """
    rows = []
    for col in reference.columns:
        ref_vals = reference[col].values
        cur_vals = current[col].values
        w1 = wasserstein_distance(ref_vals, cur_vals)
        if normalise:
            ref_std = ref_vals.std()
            w1_norm = w1 / ref_std if ref_std > 0 else 0.0
        else:
            w1_norm = w1
        rows.append({
            'feature':    col,
            'wasserstein_raw':        float(w1),
            'wasserstein_normalised': float(w1_norm),
        })
    return pd.DataFrame(rows).sort_values('wasserstein_normalised', ascending=False)
```

Wasserstein is particularly useful when you need to **rank** drifted features by severity — it gives a continuous measure of how far the distribution has moved, not just a binary flag.

---

## Concept Drift vs. Feature Drift: Different Problems, Different Solutions

```
Observed Model Degradation
         │
         ├─── Is input distribution shifting?
         │         │
         │    YES   └──> Feature Drift
         │              ├── Re-select features on recent data
         │              ├── Re-calibrate preprocessing (scaler, encoder)
         │              └── Retrain model on re-preprocessed data
         │
         └─── Is the input-output relationship shifting?
                    │
               YES  └──> Concept Drift
                         ├── Collect new labelled examples
                         ├── Retrain (or fine-tune) model
                         └── May require domain expert review
```

### How to Distinguish Them

1. **With labels available**: Compute feature drift metrics (PSI/KS) AND monitor model metrics (AUC, RMSE). Feature drift without performance degradation is usually benign. Feature drift with performance degradation suggests either pre-processing or model must adapt.

2. **Without labels**: You can only observe feature drift. Monitor PSI/KS; trigger re-selection when thresholds are crossed; flag for human review.

3. **Both simultaneously**: Most common in commodity markets. Regime changes shift both the input distributions (volatility spikes) and the price-signal relationship (momentum reversal). Requires regime detection + conditional re-training.

---

## Adaptive Re-Selection Triggers

A production re-selection trigger monitors drift metrics continuously and fires a re-selection job when conditions are met.

```python
import datetime
from dataclasses import dataclass, field


@dataclass
class ReSelectionTrigger:
    """
    Monitor feature drift and trigger re-selection when thresholds are crossed.

    Parameters
    ----------
    psi_threshold : float
        Trigger if any feature PSI exceeds this value.
    psi_fraction_threshold : float
        Trigger if this fraction of features cross psi_threshold.
    ks_alpha : float
        KS significance level.
    ks_fraction_threshold : float
        Trigger if this fraction of features show significant KS drift.
    max_days_since_selection : int
        Scheduled trigger: re-select after this many calendar days regardless.
    """
    psi_threshold:            float = 0.2
    psi_fraction_threshold:   float = 0.2   # trigger if 20% of features drift
    ks_alpha:                 float = 0.05
    ks_fraction_threshold:    float = 0.3
    max_days_since_selection: int   = 30
    last_selection_date: datetime.date = field(
        default_factory=datetime.date.today
    )

    def should_reselect(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
    ) -> tuple[bool, str]:
        """
        Check all trigger conditions.

        Returns
        -------
        (trigger: bool, reason: str)
        """
        n_features = len(reference.columns)

        # 1. Scheduled trigger
        days_since = (datetime.date.today() - self.last_selection_date).days
        if days_since >= self.max_days_since_selection:
            return True, f"Scheduled: {days_since} days since last selection"

        # 2. PSI trigger
        psi_report = compute_psi_dataframe(reference, current)
        n_psi_critical = (psi_report['psi'] > self.psi_threshold).sum()
        if n_psi_critical / n_features >= self.psi_fraction_threshold:
            worst = psi_report.iloc[0]
            return True, (
                f"PSI trigger: {n_psi_critical}/{n_features} features > "
                f"{self.psi_threshold} (worst: {worst['feature']} PSI={worst['psi']:.3f})"
            )

        # 3. KS trigger
        ks_report = ks_drift_report(reference, current, self.ks_alpha)
        n_ks_drifted = ks_report['drifted'].sum()
        if n_ks_drifted / n_features >= self.ks_fraction_threshold:
            return True, (
                f"KS trigger: {n_ks_drifted}/{n_features} features drifted "
                f"(alpha={self.ks_alpha})"
            )

        return False, "No trigger: all metrics within thresholds"

    def record_selection(self):
        """Call after a re-selection completes."""
        self.last_selection_date = datetime.date.today()
```

### Trigger Strategy Comparison

| Strategy | Pros | Cons |
|----------|------|------|
| Threshold-based (PSI) | Simple, industry-proven | May miss slow drift |
| Statistical (KS) | Principled false-positive control | Multiple testing inflation |
| Scheduled (calendar) | Predictable; easy to operationalise | May trigger unnecessarily |
| Performance-based | Directly tied to model value | Requires labels; lags reality |

**Recommended**: Combine all three. Scheduled trigger as a safety net; PSI/KS for early warning; performance trigger as confirmation.

---

## A/B Testing Feature Sets

When a new feature set is proposed (e.g., after re-selection triggered by drift), deploy it to a fraction of traffic and compare against the incumbent set using a proper statistical test.

```python
from scipy.stats import ttest_ind, mannwhitneyu
import numpy as np


def ab_test_feature_sets(
    scores_control: np.ndarray,
    scores_treatment: np.ndarray,
    metric_name: str = 'AUC',
    alpha: float = 0.05,
    use_mw: bool = True,
) -> dict:
    """
    Compare model performance between two feature sets.

    Parameters
    ----------
    scores_control : array
        Per-sample scores (e.g., predicted probabilities) for incumbent features.
    scores_treatment : array
        Per-sample scores for new features.
    use_mw : bool
        Use Mann-Whitney U (non-parametric) instead of t-test.

    Returns
    -------
    dict with test results and recommendation.
    """
    if use_mw:
        stat, pvalue = mannwhitneyu(
            scores_treatment, scores_control, alternative='greater'
        )
        test_name = 'Mann-Whitney U (one-sided)'
    else:
        stat, pvalue = ttest_ind(scores_treatment, scores_control)
        test_name = 'Welch t-test (two-sided)'

    lift = np.mean(scores_treatment) - np.mean(scores_control)

    result = {
        'test':           test_name,
        'statistic':      float(stat),
        'pvalue':         float(pvalue),
        'lift':           float(lift),
        'control_mean':   float(np.mean(scores_control)),
        'treatment_mean': float(np.mean(scores_treatment)),
        'significant':    pvalue < alpha,
        'recommendation': 'promote' if (pvalue < alpha and lift > 0) else 'keep_control',
    }

    print(f"A/B Test: {metric_name}")
    print(f"  Control:   {result['control_mean']:.4f}")
    print(f"  Treatment: {result['treatment_mean']:.4f}")
    print(f"  Lift:      {lift:+.4f}")
    print(f"  p-value:   {pvalue:.4f} ({test_name})")
    print(f"  Decision:  {result['recommendation'].upper()}")
    return result
```

### Sample Size for Feature Set A/B Test

Before running the test, compute minimum required observations:

$$n = \frac{2(z_{\alpha/2} + z_\beta)^2 \sigma^2}{\delta^2}$$

where $\delta$ is the minimum detectable effect (MDE) and $\sigma^2$ is the variance of the metric.

```python
from scipy.stats import norm as scipy_norm

def ab_sample_size(sigma: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Minimum observations per variant for an A/B test."""
    z_alpha = scipy_norm.ppf(1 - alpha / 2)
    z_beta  = scipy_norm.ppf(power)
    n = 2 * (z_alpha + z_beta)**2 * sigma**2 / mde**2
    return int(np.ceil(n))
```

---

## Online Evolutionary Feature Selection for Streaming Data

When data arrives as a continuous stream, re-running full stability selection is too expensive. An incremental approach maintains a population of candidate feature subsets and updates them as new data arrives.

```python
import collections


class OnlineFeatureSelector:
    """
    Incremental feature selection for streaming data.

    Maintains a sliding window of recent observations and periodically
    re-evaluates feature fitness. Uses exponential moving average to
    track feature importance over time.

    Parameters
    ----------
    n_features_total : int
        Total number of features in the stream.
    k : int
        Target number of selected features.
    window_size : int
        Number of recent observations to use for scoring.
    ema_alpha : float
        Smoothing factor for exponential moving average of importance scores.
    update_every : int
        Re-evaluate feature set every this many new observations.
    """

    def __init__(
        self,
        n_features_total: int,
        k: int = 10,
        window_size: int = 1000,
        ema_alpha: float = 0.1,
        update_every: int = 100,
    ):
        self.n_features_total = n_features_total
        self.k           = k
        self.window_size = window_size
        self.ema_alpha   = ema_alpha
        self.update_every = update_every

        # Circular buffer for the sliding window
        self.buffer_X = collections.deque(maxlen=window_size)
        self.buffer_y = collections.deque(maxlen=window_size)

        # Exponential moving averages of feature importance scores
        self.ema_scores   = np.zeros(n_features_total)
        self.selected_    = np.zeros(n_features_total, dtype=bool)
        self.n_observed   = 0

    def update(self, x: np.ndarray, y: float) -> np.ndarray | None:
        """
        Process one new observation.

        Parameters
        ----------
        x : array of shape (n_features_total,)
        y : scalar target value

        Returns
        -------
        Updated selected feature indices if re-evaluation occurred, else None.
        """
        self.buffer_X.append(x)
        self.buffer_y.append(y)
        self.n_observed += 1

        if self.n_observed % self.update_every == 0 and len(self.buffer_X) >= 50:
            return self._recompute_selection()
        return None

    def _recompute_selection(self) -> np.ndarray:
        """Re-score features on the current window and update EMA."""
        from sklearn.ensemble import RandomForestClassifier

        X_window = np.array(self.buffer_X)
        y_window = np.array(self.buffer_y)

        rf = RandomForestClassifier(n_estimators=30, random_state=42)
        rf.fit(X_window, y_window)
        new_scores = rf.feature_importances_

        # Exponential moving average update
        self.ema_scores = (
            self.ema_alpha * new_scores
            + (1 - self.ema_alpha) * self.ema_scores
        )

        # Select top-k by EMA score
        top_k = np.argsort(self.ema_scores)[-self.k:]
        self.selected_ = np.zeros(self.n_features_total, dtype=bool)
        self.selected_[top_k] = True

        return np.where(self.selected_)[0]

    @property
    def selected_indices(self) -> np.ndarray:
        return np.where(self.selected_)[0]
```

The EMA smoothing ensures that short-term noise does not immediately evict stable features. The window size controls how quickly the selector adapts to regime changes.

---

## Common Pitfalls

- **Binning reference and current separately**: PSI bins must be defined on the reference distribution and applied identically to the current distribution. Defining bins separately makes comparisons meaningless.
- **Not smoothing PSI bins**: Zero observations in a bin causes log(0). Always add a small epsilon before computing PSI.
- **Multiple testing in KS monitoring**: Running KS tests for 100 features at alpha=0.05 expects ~5 false positives. Apply Bonferroni correction: use `alpha_corrected = 0.05 / n_features`.
- **Conflating feature drift with model degradation**: PSI can be high while model performance is unchanged (robust models tolerate some drift). Monitor both independently.
- **Online selector instability**: EMA alpha too high causes thrashing; too low causes sluggish adaptation. Tune empirically on historical regime transitions.

---

## Connections

- **Builds on:** Guide 01 (pipeline serialisation — the pipeline to be monitored)
- **Leads to:** Guide 03 (MLflow logging of drift metrics and re-selection events)
- **Related to:** Module 07 time series (regime-aware selection shares the adaptive re-selection pattern)

---

## Further Reading

- Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley. (PSI origin)
- Kolmogorov, A. N. (1933). Sulla determinazione empirica di una legge di distribuzione. *Giornale dell'Istituto Italiano degli Attuari*, 4, 83–91.
- Villani, C. (2003). *Topics in Optimal Transportation*. AMS. (Wasserstein theory)
- Gama, J., et al. (2014). A Survey on Concept Drift Adaptation. *ACM Computing Surveys*, 46(4).
