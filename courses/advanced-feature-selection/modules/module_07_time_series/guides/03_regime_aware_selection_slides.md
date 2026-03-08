---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->
<!-- Speaker notes: This is the most practically differentiated deck in Module 7. Regime awareness is what separates serious quantitative practitioners from those who apply standard ML to financial data. The key insight: a single feature set averaged across regimes is suboptimal in every regime. Open with a strong example — VIX is a great predictor of equity returns in high-volatility regimes and nearly useless in calm markets. -->

# Regime-Aware Feature Selection
## Adapting to Market Conditions Over Time

### Module 07 — Feature Selection for Time Series

*Different regimes need different features*

---

<!-- Speaker notes: Motivate with a concrete example. Show returns in two regimes — the predictive features in a crisis (flight-to-quality drivers: VIX, credit spreads, sovereign CDS) are completely different from those in a bull market (momentum, earnings revisions, earnings surprises). A model trained on the full sample mixes these signal environments and dilutes both. -->

## The Regime Problem

**A single feature set averaged across regimes may be suboptimal in every regime.**

Consider equity return prediction:

| Regime | Effective Features | Ineffective Features |
|---|---|---|
| Bull market (low vol) | Momentum, earnings revisions | VIX, credit spreads |
| Crisis (high vol) | VIX, CDS spreads, liquidity | Momentum, earnings |
| Recovery | Value, breadth indicators | Both momentum and vol |

**What happens with one global feature set?**
- Bull-market features dilute crisis signal
- Crisis features add noise in calm regimes
- Model performance is mediocre everywhere

---

<!-- Speaker notes: Introduce Hidden Markov Models as the principled solution to regime detection. The HMM learns regime transitions from data rather than requiring hand-specified thresholds. Walk through the math: latent state s_t is Markov, observations are Gaussian conditional on state. Baum-Welch finds the parameters. Viterbi finds the most likely state sequence. -->

## Hidden Markov Model Regime Detection

**HMM**: latent state $s_t \in \{1, \ldots, K\}$ follows a Markov chain; observations depend on latent state.

$$P(s_t = j \mid s_{t-1} = i) = p_{ij} \qquad \text{(transition matrix)}$$

$$y_t \mid s_t = k \sim \mathcal{N}(\mu_k,\, \sigma_k^2) \qquad \text{(Gaussian emission)}$$

**Algorithm:**
1. **Baum-Welch (EM):** estimate $\{\mu_k, \sigma_k, p_{ij}\}$
2. **Viterbi:** decode most likely state sequence $s_1^*, \ldots, s_T^*$
3. **Forward-backward:** compute posterior $P(s_t = k \mid y_{1:T})$

**Key choice:** $K$ number of regimes. Use BIC across $K \in \{2, 3, 4\}$. Financial data: $K = 2$ (bull/bear) or $K = 3$ (bull/consolidation/bear) are most interpretable.

---

<!-- Speaker notes: Two alternative regime detection approaches. Threshold-based is interpretable and robust — use when you have a clear economic indicator. VIX > 25 is a well-known practitioner threshold. Change point detection is for structural breaks — when the regime shift is abrupt and permanent rather than cyclical. Ruptures is the go-to library. -->

## Alternative Regime Detectors

<div class="columns">

**Threshold-Based (Simple)**
Use an observable indicator:

```python
# VIX-based regime
regime = pd.cut(
    vix,
    bins=[0, 15, 25, np.inf],
    labels=['calm', 'normal', 'crisis']
)
```

Pros: interpretable, economically grounded
Cons: threshold is arbitrary

**Change Point Detection**
Structural breaks via `ruptures`:

```python
import ruptures as rpt
algo = rpt.Pelt(model='rbf').fit(signal)
breakpoints = algo.predict(
    pen=np.log(T) * var
)
```

Pros: detects permanent shifts
Cons: breakpoints are retrospective

</div>

**Choose based on:** cyclical dynamics → HMM; permanent breaks → change points; known indicators → threshold.

---

<!-- Speaker notes: This is the critical "aha" slide. Show the regime-specific feature selection pipeline step by step. Emphasise the ordering: first detect regimes (using historical data only), then select features within each regime period. The key constraint is that regime detection must also respect temporal ordering — no forward-looking regime labels. -->

## Regime-Conditioned Feature Selection Pipeline

```
1. Detect regimes using HMM on returns/volatility
   (Filtered posteriors only — no future data)

2. Label each observation with its regime

3. Split data by regime:
   Regime 0 observations: [t=5, t=7, t=23, ...]
   Regime 1 observations: [t=1, t=3, t=8, ...]

4. Within each regime:
   Run feature selection (Granger, MI, LASSO)
   Apply multiple testing correction
   Record selected feature set

5. Build regime-switching prediction model:
   Detect current regime online
   Apply corresponding feature set
```

---

<!-- Speaker notes: Markov switching regression is the econometric tool that formalises regime-switching in a regression context. Note that this is different from just running two separate regressions — the Markov structure means regime transitions are probabilistic and continuous, not hard-assigned. The coefficients beta_k show which features matter in each regime. -->

## Markov Switching Regression

Regression coefficients switch with the latent regime:

$$y_t = \mathbf{x}_t^\top \boldsymbol{\beta}_{s_t} + \sigma_{s_t} \varepsilon_t, \quad s_t \in \{0, \ldots, K-1\}$$

The transition matrix is estimated jointly with coefficients:

$$\mathbf{P} = \begin{pmatrix} p_{00} & p_{01} \\ p_{10} & p_{11} \end{pmatrix}$$

**Feature selection interpretation:** If $\beta_{k,j} \approx 0$ for regime $k$, feature $j$ is not relevant in that regime.

```python
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

model = MarkovRegression(
    endog=y, k_regimes=2, trend='c',
    exog=X_features, switching_variance=True
)
result = model.fit(disp=False)
```

Examine `result.params` for regime-specific coefficients.

---

<!-- Speaker notes: Population Stability Index is the standard tool from credit risk. It originated in the FICO scorecard world to monitor whether a scoring model's input features were still behaving like they did during model development. PSI > 0.20 means the feature has drifted substantially and the model assumptions may be violated. Walk through the formula. -->

## Feature Drift: Population Stability Index

**PSI measures distribution shift** between reference (training) and current (live) periods:

$$\text{PSI} = \sum_{b=1}^{B} \left(p^{(t)}_b - p^{(\text{ref})}_b \right) \ln \frac{p^{(t)}_b}{p^{(\text{ref})}_b}$$

where $p^{(t)}_b$, $p^{(\text{ref})}_b$ = fraction of observations in bin $b$ at time $t$ and reference period.

| PSI | Interpretation | Action |
|---|---|---|
| < 0.10 | Negligible shift | Monitor |
| 0.10–0.20 | Moderate shift | Investigate |
| 0.20–0.25 | Significant shift | Consider re-selection |
| > 0.25 | Major shift | Re-select now |

**Compute monthly on all selected features.** Flag any feature with PSI > 0.20.

---

<!-- Speaker notes: Two complementary drift statistics. KS test provides a formal p-value for distributional shift — useful for automated alerting. Wasserstein distance is more sensitive to tail shifts which are particularly important in financial data. Wasserstein is sensitive to outliers/tail behavior, KS is more sensitive to central location shifts. Use both. -->

## Feature Drift: KS Test and Wasserstein Distance

<div class="columns">

**Kolmogorov-Smirnov Test:**

$$D_{KS} = \sup_x |F_{\text{ref}}(x) - F_{\text{cur}}(x)|$$

- Formal p-value for distributional shift
- Sensitive to location and scale shifts
- Non-parametric, no binning needed

```python
from scipy.stats import ks_2samp
stat, pval = ks_2samp(ref, current)
drift = pval < 0.05
```

**Wasserstein-1 Distance:**

$$W_1 = \int_{-\infty}^{\infty} |F_{\text{ref}}(x) - F_{\text{cur}}(x)|\, dx$$

- Earth mover's distance
- Sensitive to tail behavior
- No threshold — compare over time

```python
from scipy.stats import wasserstein_distance
w = wasserstein_distance(ref, current)
```

</div>

---

<!-- Speaker notes: The adaptive re-selection controller is the production component. In a real system, you don't want to re-run feature selection at every time step — it's expensive and adds noise. Instead, you monitor trigger signals and re-select only when warranted. The four triggers shown here cover the main cases. Emphasise that calendar-based rebalancing (quarterly) is the baseline even if no other trigger fires. -->

## Adaptive Re-Selection Triggers

**When to re-run feature selection:**

| Trigger | Detection | Threshold |
|---|---|---|
| Regime change | HMM posterior shift | $P(\text{new regime}) > 0.80$ |
| Feature drift | PSI > threshold | PSI > 0.20 for any feature |
| Performance drop | Rolling CV error spike | >15% degradation vs baseline |
| Calendar rebalance | Time elapsed | Quarterly (forced) |

```python
controller = AdaptiveReselectionController(
    psi_trigger=0.20,
    regime_prob_trigger=0.80,
    performance_drop_trigger=0.15,
    forced_rebalance_period=63,  # quarterly in business days
)

if controller.should_reselect(...)['reselect']:
    new_features = run_feature_selection(updated_data)
    model.update_features(new_features)
```

---

<!-- Speaker notes: Walk-forward GA fitness is the connection to Module 5. The key difference from standard GA feature selection: the fitness function uses purged walk-forward CV rather than standard CV. This is essential for time series — standard CV fitness would select features based on leaked future information, making the GA converge to an overfit solution. -->

## GA for Time Series: Walk-Forward Fitness

Standard GA for feature selection uses CV accuracy as fitness. For time series:

**Use purged walk-forward fitness instead:**

```python
def walk_forward_fitness(chromosome, target, features, n_splits=5):
    """Evaluate feature subset with temporal purging."""
    selected = features.columns[chromosome.astype(bool)]
    if len(selected) == 0:
        return -np.inf

    splitter = PurgedWalkForwardSplitter(n_splits=n_splits)
    errors = []
    for train_idx, test_idx in splitter.split(features[selected]):
        model.fit(features[selected].iloc[train_idx], target.iloc[train_idx])
        errors.append(mse(target.iloc[test_idx],
                         model.predict(features[selected].iloc[test_idx])))
    return -np.mean(errors)  # Maximise negative MSE
```

**Temporal chromosome encoding:** encode separate feature subsets per regime.

---

<!-- Speaker notes: Online feature selection is the real-time version. For streaming data where you can't stop and retrain, online methods update feature scores incrementally. The exponentially-weighted covariance approach is simple and effective. The decay parameter controls the "memory" — lower decay = more weight on recent data = faster adaptation to drift. -->

## Online Feature Selection for Streaming Data

When full batch retraining is impractical — e.g., high-frequency data, production systems:

**Exponentially-weighted online correlation:**

$$\hat{c}_j^{(t)} = \lambda \hat{c}_j^{(t-1)} + (1-\lambda)(x_j^{(t)} - \hat{\mu}_j^{(t)})(y^{(t)} - \hat{\mu}_y^{(t)})$$

where $\lambda$ is the forgetting factor (e.g., 0.99 for slow adaptation).

**Properties:**
- $O(p)$ update per time step — constant memory
- $\lambda = 0.99$: effective window $\approx 100$ observations
- $\lambda = 0.999$: effective window $\approx 1000$ observations
- Adapts automatically to drift without explicit detection

**Limitation:** only tracks linear correlations; use more sophisticated online MI estimators for nonlinear relationships.

---

<!-- Speaker notes: Summarise the three key tools covered. The flow from detection to selection to monitoring is the practical workflow. Each step addresses a different aspect of the regime problem. -->

## Summary: Regime-Aware Selection Framework

**Three-layer framework:**

```
Layer 1: Regime Detection
  HMM | Threshold | Change Point
  ↓
  Regime labels for each time period

Layer 2: Regime-Conditioned Selection
  Run feature selection within each regime
  Build regime-specific feature sets
  ↓
  Feature sets: {regime_0: [X1, X3, X7], regime_1: [X2, X5, X9]}

Layer 3: Monitoring and Adaptation
  PSI monitoring for feature drift
  Trigger-based re-selection
  Online update between rebalances
```

**Key rule:** The optimal feature set changes with market conditions. Static selection is a simplification that loses performance in non-stationary environments.

---

<!-- Speaker notes: Point to notebook 3. Students will fit an HMM, run regime-specific Granger feature selection, compare the selected sets across regimes (Jaccard similarity), and then show how using regime-specific features outperforms static features on out-of-sample prediction. The visualisation of regime labels over time is always striking. -->

## Notebook 03: Regime-Aware Feature Selection

**What you will build:**

- Fit a 2-state Gaussian HMM to detect market regimes
- Run Granger feature selection separately within each detected regime
- Compute Jaccard similarity between regime feature sets
- Implement adaptive re-selection triggered by regime change
- Compare regime-aware vs static feature selection out-of-sample performance

**Key expectation:** The Jaccard overlap between feature sets for different regimes will be low (< 0.4), confirming that regimes genuinely require different features.

*See `notebooks/03_regime_aware_features.ipynb`*

---

<!-- Speaker notes: Reference the Hamilton 1989 paper as the foundational Markov switching reference. The de Prado chapter on structural breaks connects this to the financial ML literature. Webb 2016 and Gama 2014 are the concept drift literature for the monitoring piece. -->

## Further Reading

- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series." *Econometrica*, 57(2), 357–384. — Markov switching regression foundation.

- de Prado, M.L. (2018). *Advances in Financial Machine Learning*. Wiley. Chapter 17: Structural Breaks.

- Webb, G.I. et al. (2016). "Characterizing Concept Drift." *Data Mining and Knowledge Discovery*, 30(4), 964–994.

- Gama, J. et al. (2014). "A Survey on Concept Drift Adaptation." *ACM Computing Surveys*, 46(4), 1–37.

- hmmlearn documentation: https://hmmlearn.readthedocs.io/

- ruptures documentation: https://centre-borelli.github.io/ruptures-docs/
