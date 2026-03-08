---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->
<!-- Speaker notes: This deck covers one of the most practically impactful topics in time series machine learning: how to evaluate and select features without leaking future information. The concepts here — purged CV, embargo periods — come directly from institutional quantitative finance practice. Emphasise that these aren't theoretical niceties; they prevent strategies that look great in backtest but fail in production. -->

# Temporal Validation for Time Series
## Walk-Forward Validation, Purging, and Embargo

### Module 07 — Feature Selection for Time Series

*Why the way you validate changes everything*

---

<!-- Speaker notes: Start with the fundamental violation. Ask the audience to trace through what happens in standard 5-fold CV on a time series. Pick a specific fold: observations from year 3 are in the training set, observations from year 2 are in the test set. A model trained on year 3 data to predict year 2 outcomes — that is look-ahead bias. -->

## The Problem with Standard k-Fold CV

Standard $k$-fold randomly assigns observations to folds:

```
Data:  [Jan] [Feb] [Mar] [Apr] [May] [Jun] [Jul] [Aug] [Sep] [Oct]
Fold 1:  T     T    TEST   T     T    TEST   T     T    TEST   T
Fold 2:  T    TEST   T     T    TEST   T     T    TEST   T     T
```

**Two forms of leakage:**

1. **Look-ahead bias:** Future observations (e.g., August) appear in training when evaluating past observations (e.g., March)

2. **Autocorrelation leakage:** Adjacent train and test observations share information through serial dependence

**Result:** Cross-validated performance is systematically optimistic — models appear better than they will perform on live data.

---

<!-- Speaker notes: Quantify the bias. For a typical financial return series with autocorrelation 0.10, the bias seems small but it compounds across many features. For overlapping-window features (20-day rolling returns, for example), the effective autocorrelation is much higher and the bias is substantial. -->

## Quantifying the Leakage Bias

For a stationary AR(1) process with coefficient $\rho$:

$$\text{Optimistic Bias} \approx \frac{2\rho}{1 - \rho^2} \cdot \text{Var}(y_t)$$

| Feature type | Typical $\rho$ | Approximate bias |
|---|---|---|
| Raw financial returns | 0.05–0.10 | Small |
| 5-day rolling mean | 0.60–0.80 | Large |
| 20-day rolling mean | 0.85–0.95 | Very large |
| Monthly aggregates from daily | 0.95+ | Extreme |

**Critical rule:** The longer the rolling window relative to the test period, the larger the leakage bias.

---

<!-- Speaker notes: Introduce walk-forward validation as the baseline solution. The expanding vs sliding choice depends on whether early history is relevant — for regime-changing systems, sliding window is preferred because recent data is more representative. For stable systems, expanding window uses more data and is more statistically efficient. -->

## Walk-Forward Validation

Train always precedes test. No exceptions.

**Expanding window** (training set grows):
```
Split 1: [========Train========] [Test1]
Split 2: [==========Train==========] [Test2]
Split 3: [============Train============] [Test3]
```

**Sliding window** (fixed training size):
```
Split 1: [====Train====] [Test1]
Split 2:      [====Train====] [Test2]
Split 3:           [====Train====] [Test3]
```

**Expanding:** use when early history is relevant (stable relationships).

**Sliding:** use when recent regime matters more than history (non-stationary systems).

---

<!-- Speaker notes: This is the de Prado contribution — purging. The key insight is that rolling-window features create label overlap. An observation at t=100 computed from [80, 100] overlaps with a test observation at t=105 computed from [85, 105]. Even though they are in "different" periods, they share 15 days of data. Purging removes training observations that contaminate test. -->

## The Overlap Problem: Why Walk-Forward Isn't Enough

**Rolling-window features create label overlap:**

```
Observation at t=100: uses data from [t=80, t=100]
Observation at t=105: uses data from [t=85, t=105]  (in test set)

Overlap: [t=85, t=100] — 15 shared days
```

Standard walk-forward places $t=100$ in training and $t=105$ in test — but they share 15 days of data!

**Impact on feature selection:** A feature ranked on training data at $t=100$ is partially ranked on information from the test period $[t=85, t=105]$.

This is the key insight from **de Prado (2018), Chapter 7**.

---

<!-- Speaker notes: Walk through the purging definition carefully. The label span concept is crucial. For point-in-time labels (daily returns), the span is just [t, t]. For rolling-window features, the span is [t-window, t]. Purging removes any training observation whose label span overlaps with any test observation's label span. -->

## Purged Cross-Validation

**Purging:** remove training observations whose label spans overlap with test.

Let observation $i$ have label span $[t_i^0, t_i^1]$ (period of data used to form its label).

**Purge criterion:** Remove observation $i$ from training if:

$$t_i^1 \geq t^{\text{test}}_{\text{start}}$$

**Visualisation:**

```
Test period:    [===== test obs 1 =====][===== test obs 2 =====]
                     ^
                     t_test_start

Train obs A:  [=====]          → Keep  (label ends before test start)
Train obs B:      [=====]      → PURGE (label overlaps test start)
Train obs C:          [=====]  → PURGE (label fully inside test)
```

---

<!-- Speaker notes: Embargo extends purging past the test period. Even after the test window ends, the first few observations after the test are highly correlated with the last test observations — through serial dependence in the features. The embargo size should cover the autocorrelation decay time. In practice, 5-10 bars is often sufficient for financial data. -->

## Embargo Periods

**Problem:** Observations immediately *after* the test period are correlated with the last test observations through feature autocorrelation.

**Solution:** Embargo — remove $e$ observations after the test period from the next training fold.

```
Test period:    [======= TEST =======]
Embargo zone:                         [===e===]
Next train:                                    [======= TRAIN =======]
```

**Embargo size $e$:**

$$e = \max\left(1,\; \text{int}\!\left(\frac{\text{autocorrelation halflife}}{2}\right)\right)$$

In practice: set $e = 0.01 \times T$ (1% of total observations) as a conservative default.

de Prado recommends: $e \approx$ rolling window size / 2.

---

<!-- Speaker notes: Combinatorial purged CV is the advanced version. Instead of a single path through the data, it generates all possible test combinations. This gives a better estimate of out-of-sample performance distribution. The number of combinations can be large — for N=10, k=2, there are 45 test sets. The paths concept is useful for computing Sharpe ratios across many out-of-sample periods. -->

## Combinatorial Purged CV (CPCV)

**Limitation of walk-forward:** single path through data → single performance estimate.

**CPCV (de Prado, 2018, Ch. 12):** all $\binom{N}{k}$ combinations of $k$ test blocks from $N$ total blocks:

$$\text{Number of paths} = \bar{n}(N, k) = \frac{N}{k}\binom{N}{k}$$

| $N$ blocks | $k$ test blocks | Combinations | Paths |
|---|---|---|---|
| 6 | 2 | 15 | 45 |
| 8 | 2 | 28 | 112 |
| 10 | 2 | 45 | 225 |
| 10 | 3 | 120 | 400 |

**Benefit:** distributional estimate of out-of-sample performance, not just a point estimate. Enables confidence intervals around feature importance.

---

<!-- Speaker notes: Walk-forward feature selection shows which features are consistently selected across time. A feature selected in 9/10 folds is much more reliable than one selected in 2/10 folds with a high score. This is the temporal stability concept. -->

## Walk-Forward Feature Selection

Re-select features at each rebalance point using only historical data:

```
Rebalance at t=252 (year 1): Select features using data [0, 252]
  → Selected: {X1, X3, X7, X12}

Rebalance at t=504 (year 2): Select features using data [0, 504]
  → Selected: {X1, X3, X8, X12}

Rebalance at t=756 (year 3): Select features using data [0, 756]
  → Selected: {X1, X2, X3, X12}
```

**Feature stability:** Which features appear consistently across rebalances?

- $X_1$: selected 3/3 — stable, reliable
- $X_{12}$: selected 3/3 — stable, reliable
- $X_3$: selected 3/3 — stable, reliable
- $X_7$, $X_8$, $X_2$: selected 1/3 — regime-dependent

---

<!-- Speaker notes: Stability metrics provide a formal way to measure feature reliability across walk-forward folds. Kendall's W is the gold standard for ranking concordance. A feature with W < 0.3 across folds is likely noise. Spearman footrule gives pairwise distance between rankings. -->

## Feature Stability Metrics

**Selection frequency:** Fraction of folds where feature is selected.

$$\text{SF}_j = \frac{1}{K} \sum_{k=1}^{K} \mathbf{1}[\text{feature } j \text{ selected in fold } k]$$

**Kendall's W** (concordance across fold rankings):

$$W = \frac{12 S}{m^2(n^3 - n)}, \quad S = \sum_{i=1}^{n}\left(R_i - \bar{R}\right)^2$$

$W \in [0,1]$: 1 = perfect agreement, 0 = no agreement.

**Spearman footrule distance** between fold rankings $\sigma$, $\tau$:

$$d_F = \frac{\sum_{i=1}^{n} |\sigma(i) - \tau(i)|}{\lfloor n^2 / 2 \rfloor} \in [0, 1]$$

**Rule of thumb:** Features with $\text{SF} > 0.7$ and $W > 0.5$ are considered stable.

---

<!-- Speaker notes: Feature decay is the concept that some features' predictive power degrades over time. This could be because they were effective only in a specific regime, or because market participants have arbitraged away the signal. Rolling MI or rolling Granger p-values reveal this pattern. -->

## Temporal Feature Decay

Some features lose predictive power over time:

- **Regime-specific features:** effective only in bull markets or crisis periods
- **Arbitraged signals:** market participants trade away the alpha
- **Structural breaks:** regulatory change, new market participants alter relationships

**Detection:** rolling mutual information or rolling Granger p-value

```
Feature X rolling MI (252-day window):
  2018: 0.12  (predictive)
  2019: 0.09  (weakening)
  2020: 0.04  (barely predictive)
  2021: 0.02  (noise)
  2022: 0.01  (irrelevant)
```

**Decay metrics:** slope of MI vs time, half-life, recency ratio (recent-window MI / full-window MI).

---

<!-- Speaker notes: Summarise the key pitfalls. The shuffle point is the most common mistake — new practitioners often just use sklearn's cross_val_score with TimeSeriesSplit, which is better than random shuffle but still doesn't handle label overlap. Purging and embargo are the production-grade solution. -->

## Common Pitfalls

| Mistake | Consequence | Fix |
|---|---|---|
| Shuffling time series | Severe look-ahead bias | Never shuffle; use temporal splits |
| Standard k-fold | Moderate-to-severe bias | Use walk-forward or purged CV |
| No embargo | Subtle leakage from overlap | Add embargo = 1% of T |
| Select features on full data | Selection bias | Select within each fold |
| Single-fold reporting | High variance estimate | Average across all folds |
| Ignoring label overlap | Hidden leakage | Track label spans; purge |

---

<!-- Speaker notes: The sklearn TimeSeriesSplit is a useful baseline but does not handle purging or embargo. The PurgedKFold from mlfinlab (de Prado's library) implements the full framework. For production work, implement your own PurgedWalkForwardSplitter that tracks label spans. -->

## Implementation: sklearn Compatibility

The `PurgedWalkForwardSplitter` is designed to be sklearn-compatible:

```python
from sklearn.model_selection import cross_val_score

splitter = PurgedWalkForwardSplitter(
    n_splits=5,
    test_size=252,      # 1 year test windows
    embargo_pct=0.01,   # 1% embargo
)

# Use with cross_val_score
# Note: pass label_spans as groups parameter
scores = cross_val_score(
    estimator=my_model,
    X=X_features,
    y=y_target,
    cv=splitter,
    groups=label_spans,   # end index of each observation's label span
    scoring='neg_mean_squared_error',
)
```

**Key:** Pass `groups=label_spans` to enable purging. Without it, splitter falls back to standard walk-forward (no purging).

---

<!-- Speaker notes: Connect to the notebook. In the walk-forward selection notebook, students will implement the full purged CV pipeline on a real financial dataset, measure the leakage bias by comparing purged vs unpurged selection, and compute feature stability metrics. The key visualisation is the feature selection heatmap over time. -->

## Summary

**Key Takeaways**

1. Standard $k$-fold CV creates **look-ahead bias** for time series — future data trains models evaluated on the past.

2. **Walk-forward validation** is the baseline fix — train always precedes test.

3. **Purging** (de Prado, 2018) removes training observations whose label windows overlap with the test period — essential for rolling-window features.

4. **Embargo** prevents leakage through feature autocorrelation after the test period.

5. **Feature stability** (Kendall's $W$, selection frequency) identifies features that are reliably selected vs regime-dependent.

6. **Temporal decay analysis** reveals features that were once predictive but have lost their signal over time.

---

<!-- Speaker notes: Point to notebook 02, where students implement walk-forward feature selection from scratch and compare the selection results between purged and naive (unpurged) CV. The "aha moment" is seeing how different the selected feature sets are between the two approaches. -->

## Notebook 02: Walk-Forward Feature Selection

**What you will build:**

- Implement `PurgedWalkForwardSplitter` from scratch
- Apply to real financial time series data
- Track feature selection across rebalance points (selection heatmap)
- Compute stability metrics: selection frequency and Kendall's $W$
- **Show the leakage problem:** compare selected features and performance between naive CV and purged CV

**Expected finding:** Naive CV selects different (and often worse) features than purged CV. The performance gap shows the magnitude of leakage bias.

*See `notebooks/02_walk_forward_selection.ipynb`*

---

<!-- Speaker notes: Reference list for this deck. De Prado 2018 is the primary reference — chapters 7 and 12. Bergmeir 2012 is a systematic comparison study. Racine 2000 is the theoretical foundation for dependent data CV. -->

## Further Reading

- de Prado, M.L. (2018). *Advances in Financial Machine Learning*. Wiley.
  - Chapter 7: Cross-Validation in Finance (purged CV)
  - Chapter 12: Backtesting through Combinatorial Purged CV

- Bergmeir, C. & Benítez, J.M. (2012). "On the Use of Cross-Validation for Time Series Predictor Evaluation." *Information Sciences*, 191, 192–213.

- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*, 3rd ed. otexts.com/fpp3 — Chapter on CV for time series.

- Racine, J. (2000). "Consistent Cross-Validatory Model-Selection for Dependent Data." *Journal of Econometrics*, 99(2), 379–399.
