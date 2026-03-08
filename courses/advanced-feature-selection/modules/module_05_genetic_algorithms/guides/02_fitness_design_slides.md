---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->
<!-- Speaker notes: This guide goes deep on the fitness function — the one domain-specific component in the GA. All the operator machinery from Guide 01 is reusable; fitness is what makes this a feature selection problem. The key tension is between accurate fitness estimates (expensive) and fast fitness estimates (cheap but noisy). Today we resolve this with caching, smart CV strategies, and approximation. -->

# Fitness Function Design
## Module 5.2 — Evaluating Feature Subsets

**The only domain-specific component in the GA — getting it right matters**

---

<!-- Speaker notes: Start with why fitness matters so much. The GA is only as good as its fitness function — garbage in, garbage out. A noisy fitness function makes it hard to distinguish good subsets from bad ones. A biased one (training accuracy) leads the GA to overfit. Stress that CV is not optional — it is the minimum viable unbiased estimator. -->

## Why Fitness Function Design Is Critical

The GA is a general optimizer. Its solution quality is **bounded by fitness quality**.

**Common mistakes and their consequences:**

| Mistake | Consequence |
|:---|:---|
| Training accuracy (no CV) | GA finds subsets that overfit |
| Too-small $k$ in CV (k=2) | High-variance fitness → noisy selection |
| No parsimony penalty | All-features solution dominates |
| No edge case handling | Runtime errors, stagnation |
| No caching | 10× slower than needed |
| K-fold on time series | Look-ahead leakage → overstated fitness |

> **Rule**: If your fitness function is wrong, the best possible GA gives you the wrong answer efficiently.

---

<!-- Speaker notes: Walk through the standard fitness formulation. Emphasise that the ratio |s|/p normalises to [0,1], making lambda comparable across different feature set sizes. The range of useful lambda values is 0.001 to 0.1 — anything outside this range either ignores parsimony or destroys all accuracy signal. -->

## The Core Fitness Formula

$$\boxed{\text{fitness}(s) = \underbrace{\text{CV-score}(M, X_s, y)}_{\text{accuracy}} - \underbrace{\lambda \cdot \frac{|s|}{p}}_{\text{parsimony}}}$$

where:
- $s \in \{0,1\}^p$ — binary chromosome (feature mask)
- $X_s$ — data restricted to selected features
- $|s| = \sum_i s_i$ — number of selected features
- $p$ — total features available
- $\lambda$ — parsimony weight (hyperparameter)

<div class="columns">

**$\lambda$ too small**: GA selects all informative features (and some noise)

**$\lambda$ too large**: GA drops genuinely useful features

</div>

**Recommended starting values**: $\lambda \in \{0.005, 0.01, 0.05\}$ — tune on a holdout set.

---

<!-- Speaker notes: Walk through the three CV strategies. Standard k-fold is the baseline. Stratified is always better for classification — very little additional cost. Nested CV is the gold standard but expensive; use with caching to make it feasible. The key question: is your fitness function measuring what you want it to measure? -->

## Cross-Validation Strategies

<div class="columns">

**Standard k-fold** (iid data)

$$\hat{f}(s) = \frac{1}{k}\sum_{j=1}^k \text{score}(M, X_s^{j,\text{train}}, X_s^{j,\text{val}})$$

Default: $k=5$. Use $k=10$ for $n < 500$.

**Stratified k-fold** (imbalanced classes)

Same but folds preserve class ratios. Always preferred over standard for classification.

</div>

**Nested CV** (most unbiased, most expensive):
```
Outer fold j: held-out test
    Inner CV on X_train_j → compute fitness(s)
    fitness = mean inner CV score
```

Use nested CV with fitness caching — outer and inner folds both benefit from cache hits.

---

<!-- Speaker notes: Walk-forward CV is mandatory for time series. The key insight: in time series, the future cannot appear in training data. Standard k-fold shuffles randomly and creates this leakage. The gap parameter is important for financial data where features have overlapping horizons (e.g., 5-day moving averages). Cross-reference Module 7. -->

## Walk-Forward Fitness for Time Series

Standard k-fold **leaks future data** into training for temporal data:

```
K-fold (BAD for time series):
Fold 1: train=[t₃,t₇,t₁₂,...], val=[t₁,t₅,t₉,...]  ← shuffled!

Walk-forward (CORRECT):
Fold 1: train=[t₁..t₁₀₀],  val=[t₁₀₁..t₁₂₀]
Fold 2: train=[t₁..t₁₂₀],  val=[t₁₂₁..t₁₄₀]
```

```python
from sklearn.model_selection import TimeSeriesSplit

def walk_forward_fitness(chromosome, X, y, n_splits=5, gap=0):
    selected = np.where(chromosome == 1)[0]
    if len(selected) == 0: return -1.0
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    model = LogisticRegression(max_iter=500, random_state=0)
    scores = cross_val_score(model, X[:, selected], y,
                             cv=tscv, scoring="accuracy")
    return float(scores.mean())
```

> **Cross-reference**: Module 7 extends this with purged k-fold and embargo periods for financial data.

---

<!-- Speaker notes: Parsimony schemes differ in how aggressively they penalise large subsets. Linear is proportional — 10 features costs 10× more than 1 feature (relative to penalty). Exponential punishes large subsets much more — the 30th feature costs much more than the 3rd. Adaptive (ramp up over generations) is the most sophisticated: let the GA explore freely early, then tighten. -->

## Parsimony Pressure Schemes

**Linear** (standard): $\text{penalty} = \lambda \cdot |s|/p$

Each additional feature adds equal penalty.

**Exponential**: $\text{penalty} = \lambda \cdot (|s|/p)^\gamma,\ \gamma > 1$

$$\gamma = 2:\ \text{adding feature 20 costs more than adding feature 2}$$

**Adaptive** (ramp up over generations):
$$\lambda(t) = \lambda_{\max} \cdot \frac{t}{T}$$

```python
def get_lambda(gen, max_gen, lambda_final=0.05):
    return lambda_final * (gen / max_gen)
```

<div class="columns">

Adaptive: explore freely early ($\lambda \approx 0$)

Adaptive: then tighten as population converges

</div>

> **Recommendation**: use adaptive parsimony — it combines exploration and parsimony naturally.

---

<!-- Speaker notes: The multi-objective formulation extends the basic fitness with a redundancy penalty. The key insight: two features with identical predictive power but high correlation add less value than two features with the same power but zero correlation. The redundancy penalty captures this. Lambda_2 should be smaller than lambda_1 in practice. -->

## Multi-Objective Fitness: Accuracy + Size + Redundancy

$$\text{fitness}(s) = \text{score}(s) - \underbrace{\lambda_1 \cdot \frac{|s|}{p}}_{\text{parsimony}} - \underbrace{\lambda_2 \cdot \bar{\rho}(s)}_{\text{redundancy}}$$

where $\bar{\rho}(s) = \frac{1}{\binom{|s|}{2}} \sum_{i < j \in s} |\rho_{ij}|$ is mean absolute pairwise correlation.

```python
def redundancy_fitness(chromosome, X, y, lambda1=0.01, lambda2=0.05):
    selected = np.where(chromosome == 1)[0]
    if len(selected) == 0: return -1.0
    score = cv_accuracy(chromosome, X, y)
    parsimony = lambda1 * len(selected) / len(chromosome)
    if len(selected) > 1:
        corr = np.corrcoef(X[:, selected], rowvar=False)
        upper = corr[np.triu_indices(len(selected), k=1)]
        redundancy = lambda2 * float(np.mean(np.abs(upper)))
    else:
        redundancy = 0.0
    return score - parsimony - redundancy
```

**Result**: GA prefers diverse, non-redundant feature sets at equal accuracy.

---

<!-- Speaker notes: Edge cases are where GA implementations break in production. Empty chromosomes cause crashes if not handled. All-ones are valid but should be penalised by parsimony. Degenerate populations cause stagnation. NaN/Inf fitness from numerical issues silently corrupt selection. Always wrap fitness in try/except and always check for NaN. -->

## Edge Case Handling

```python
def safe_fitness(chromosome, X, y, **kwargs):
    # 1. Empty chromosome
    if chromosome.sum() == 0:
        return -1.0

    # 2. NaN / numerical failure
    try:
        f = compute_fitness(chromosome, X, y, **kwargs)
        if not np.isfinite(f):
            return -1.0
        return f
    except Exception:
        return -1.0
```

**Degenerate population** (all identical chromosomes):
```python
def is_degenerate(population, threshold=0.01):
    chroms = np.array([ind.chromosome for ind in population])
    # Sample pairwise Hamming distances
    n = min(len(chroms), 20)
    dists = [np.mean(chroms[i] != chroms[j])
             for i in range(n) for j in range(i+1, n)]
    return np.mean(dists) < threshold
```

**Remedy**: inject random immigrants or boost mutation rate for 5–10 generations.

---

<!-- Speaker notes: This is the most important performance optimisation. Explain the cache hit rate analysis: in early generations, most chromosomes are new; near convergence, most chromosomes have been seen before. The 60-90% hit rate near convergence means you get 3-10x speedup for free. The bytes key is perfect because chromosomes are binary arrays. -->

## Fitness Caching — Essential for Performance

**Same chromosome reappears frequently** across generations (especially near convergence):

```python
class CachedFitness:
    def __init__(self, fitness_fn):
        self._fn = fitness_fn
        self._cache: dict[bytes, float] = {}
        self.hits = self.misses = 0

    def __call__(self, chromosome, X, y, **kwargs):
        key = chromosome.tobytes()   # ← perfect hash for binary arrays
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        value = self._fn(chromosome, X, y, **kwargs)
        self._cache[key] = value
        self.misses += 1
        return value
```

| Generation range | Cache hit rate | Speedup |
|:---:|:---:|:---:|
| 1–10 | 5–15% | 1.1× |
| 11–50 | 30–60% | 1.5–2.5× |
| 51–100 | 60–90% | 2.5–10× |

**Total unique evaluations** with caching: ~1,000–3,000 vs 5,000 without.

---

<!-- Speaker notes: Fitness landscape analysis is advanced but very informative. FDC tells you whether the GA can find the optimum at all — low FDC means the landscape is deceptive and the GA is likely to fail. Neutrality tells you whether selection pressure is effective. These metrics help diagnose GA failures before wasting compute. -->

## Fitness Landscape Analysis

**Ruggedness** — fitness-distance correlation (FDC):

$$\text{FDC} = \text{corr}(d(s, s^*),\ f(s))$$

where $d(s, s^*)$ is Hamming distance to best known solution.

| FDC | Landscape type | GA suitability |
|:---:|:---:|:---:|
| $> 0.15$ | Smooth | Excellent |
| $0 \approx 0$ | Neutral | Poor (drift) |
| $< -0.15$ | Deceptive | Very poor |

**Neutrality** — fraction of bit-flip mutations with $|f_\text{after} - f_\text{before}| < \varepsilon$:

High neutrality → CV noise overwhelms signal → GA wanders.

**Remedies for rugged/neutral landscapes**:
- Increase population size ($N \geq 100$)
- Use stronger parsimony to differentiate neutral solutions
- Restart from best-known solution with diversity injection

---

<!-- Speaker notes: Surrogate fitness is the advanced technique for very expensive evaluations — e.g., when the base model is a neural network or XGBoost with many trees. The surrogate predicts fitness from the chromosome without running the full evaluation. The key is to periodically re-train the surrogate with true evaluations to prevent drift. -->

## Surrogate Fitness for Expensive Evaluations

When base model training takes seconds (e.g., XGBoost, neural networks), use a **surrogate model** to predict fitness:

```
1. Sample N_initial chromosomes → true fitness evaluations
2. Train surrogate: RandomForest(chromosome → predicted fitness)
3. Evaluate population with surrogate (fast)
4. Every K generations: re-evaluate top-10 with true fitness
5. Retrain surrogate with accumulated data
```

```python
from sklearn.ensemble import RandomForestRegressor

class SurrogateFitness:
    def __init__(self, true_fn, n_initial=100):
        self._true_fn = true_fn
        self._surrogate = RandomForestRegressor(n_estimators=50)
        self._X_data, self._y_data = [], []
        self._is_trained = False

    def true_evaluate(self, chrom, X, y):
        f = self._true_fn(chrom, X, y)
        self._X_data.append(chrom.copy())
        self._y_data.append(f)
        if len(self._y_data) >= 50:
            self._surrogate.fit(np.array(self._X_data), self._y_data)
            self._is_trained = True
        return f
```

**Speedup**: 10–50× for neural network or XGBoost base models.

---

<!-- Speaker notes: This is the comparison table showing when to use which fitness strategy. Walk through each row. The key takeaway: for most practitioners, stratified CV + linear parsimony + caching is the right choice. Walk-forward is non-negotiable for time series. Nested CV and surrogate are advanced options for specific situations. -->

## Fitness Strategy Selection Guide

| Scenario | CV strategy | Parsimony | Caching |
|:---|:---:|:---:|:---:|
| Balanced classification | StratifiedKFold k=5 | Linear λ=0.01 | Always |
| Imbalanced (>3:1) | StratifiedKFold, AUC | Linear λ=0.01 | Always |
| Regression | KFold k=5, RMSE | Linear λ=0.01 | Always |
| Time series | TimeSeriesSplit | Linear λ=0.01 | Always |
| Small dataset ($n<200$) | StratifiedKFold k=10 | Linear λ=0.01 | Always |
| Correlated features | StratifiedKFold | + redundancy | Always |
| Expensive base model | StratifiedKFold | Linear | Surrogate |
| Publication-quality | Nested CV | Adaptive λ | Always |

> Start with **stratified k-fold + linear parsimony + caching**. This covers 80% of real use cases.

---

<!-- Speaker notes: Parsimony weight tuning is the most common practical question. Lambda=0.01 is the safe default. If you're seeing too many features selected, increase lambda. If you're losing too much accuracy, decrease lambda. The grid search approach is the most principled — but treat lambda as a hyperparameter to tune on the same holdout set used for final model evaluation. -->

## Tuning the Parsimony Weight $\lambda$

**Grid search approach** — evaluate fitness across $\lambda$ values on a holdout set:

```python
import numpy as np
from sklearn.model_selection import train_test_split

lambdas = [0.001, 0.005, 0.01, 0.05, 0.1]
results = {}

for lam in lambdas:
    cfg = GAConfig(parsimony_weight=lam, pop_size=30, n_generations=50)
    selector = GAFeatureSelector(cfg)
    selector.fit(X_train, y_train)

    n_feat = selector.best_individual_.n_selected()
    val_score = evaluate_on_holdout(selector, X_val, y_val)
    results[lam] = {"n_features": n_feat, "val_accuracy": val_score}
    print(f"λ={lam:.3f}: {n_feat} features, val_acc={val_score:.4f}")
```

**Interpretation**:
- Increasing $\lambda$ by 10× roughly halves the expected feature count
- The "knee" of the accuracy-vs-features tradeoff is the optimal $\lambda$
- Visualise as a Pareto front: $\lambda$ traces the front

---

<!-- Speaker notes: Final summary slide. These eight points are the take-home messages from Guide 02. The most important: CV inside GA, parsimony penalty, caching. These three together give you a robust, efficient fitness function. Everything else (redundancy, adaptive parsimony, surrogates) is an enhancement for specific situations. -->

## Module 5.2 Summary

**Eight rules for fitness function design:**

1. Always use **cross-validation** — never training accuracy alone
2. Use **stratified folds** for classification with class imbalance
3. Use **walk-forward CV** for any temporal/time-series data (Module 7)
4. Include a **parsimony penalty** $\lambda \cdot |s|/p$ — tune $\lambda \in \{0.001, 0.01, 0.05\}$
5. Handle **edge cases**: all-zero → $-1.0$, NaN/Inf → $-1.0$, wrap in try/except
6. **Cache** fitness by `chromosome.tobytes()` — gives 2–10× speedup for free
7. Use **adaptive parsimony** (ramp up $\lambda$ over generations) for better exploration
8. Analyse the **fitness landscape** if GA fails to converge — ruggedness and neutrality diagnose why

**Next**:
- **Guide 03**: Convergence detection, diversity preservation, parameter tuning
- **Notebook 01**: Implement the complete GA with all fitness strategies above

---

<!-- Speaker notes: Self-check questions for Guide 02. These cover the most common practical mistakes. Students who can answer all five are ready for Notebook 01. Q3 is the most important — being able to compute the exact penalty and compare two solutions analytically. -->

## Self-Check Questions

1. You use training accuracy (no CV) as your fitness function. The GA selects 5 features with training accuracy 0.98. Test accuracy is 0.71. What went wrong?

2. Your dataset has 1,000 samples and a 10:1 class imbalance. Which CV strategy do you choose and why?

3. With $\lambda = 0.01$ and $p = 50$ features, chromosome $A$ selects 10 features with CV accuracy 0.85. Chromosome $B$ selects 25 features with CV accuracy 0.87. Which has higher fitness? Calculate both.

4. After 80 generations, the cache hit rate is 85%. The GA is running 100 individuals for 100 generations. How many true fitness evaluations (model training calls) did you save compared to no caching?

5. Your GA converges after 20 generations with the best individual selecting all 50 features. Name two likely causes and one fix for each.
