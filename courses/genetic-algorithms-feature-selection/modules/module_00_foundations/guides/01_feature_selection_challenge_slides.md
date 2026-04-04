---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->
<!-- Speaker notes: This is the opening deck for Module 00 -- set the stage by explaining WHY feature selection matters and WHY it is hard. Emphasize the combinatorial explosion early to motivate the need for GAs. -->

# The Feature Selection Challenge
## From Exponential Search Spaces to Intelligent Optimization

### Module 00 — Foundations

Why feature selection is NP-hard and how metaheuristics help

---

<!-- Speaker notes: Ground the problem in practical terms. The key message: feature selection is critical for time series forecasting where observations are limited relative to feature count. -->

## In Brief

Feature selection is the process of identifying the **most relevant subset** of features from a larger feature space to:

- Improve model performance
- Reduce overfitting
- Enhance interpretability

<div class="callout-info">
ℹ️ **Context:** In time series forecasting with many potential features and limited observations, this becomes **critical for generalization**.
</div>

---

<!-- Speaker notes: This is the "wow" moment -- show how quickly the search space grows. Pause on the table and let the numbers sink in. The jump from 20 to 30 features is particularly dramatic. -->

## The Combinatorial Explosion

With $p$ features, there are $2^p$ possible subsets to evaluate.

| Features ($p$) | Possible Subsets ($2^p$) | Evaluation Time* |
|:-:|:-:|:-:|
| 10 | 1,024 | ~1 second |
| 20 | 1,048,576 | ~17 minutes |
| 30 | ~1 billion | ~12 days |
| 50 | ~1 quadrillion | ~35,000 years |
| 100 | ~10^30 | Heat death of universe |

*Assuming 1ms per evaluation

<div class="callout-danger">
🚨 **Warning:** This combinatorial explosion makes exhaustive search infeasible for real problems.
</div>

---

<!-- Speaker notes: Walk through the formal definition slowly. Emphasize that lambda controls the tradeoff between accuracy and simplicity. This is the objective function the GA will optimize. -->

## Formal Definition

**Given:** Feature matrix $X \in \mathbb{R}^{n \times p}$, target $y \in \mathbb{R}^n$, model $M$, metric $L$

**Find:** Binary selection vector $s \in \{0,1\}^p$ that minimizes:

$$s^* = \argmin_{s \in \{0,1\}^p} L(M(X_s), y) + \lambda \cdot ||s||_0$$

Where:
- $X_s$ contains only columns where $s_i = 1$
- $\lambda$ controls the complexity-accuracy tradeoff
- $k_{min} \leq ||s||_0 \leq k_{max}$ (feature count bounds)

---

<!-- Speaker notes: Use the weather expert analogy to build intuition. This helps learners who are less comfortable with math. Ask the audience: "Would you rather have 5 great experts or 100 mediocre ones?" -->

## Intuitive Explanation

Imagine building a **weather forecasting model** with 100 potential features.

Using all 100 is like asking 100 experts and weighing them all equally -- you'd be overwhelmed by irrelevant signals.

Feature selection = assembling the right team:

- **Relevant experts** -- features that actually help predict
- **Diverse perspectives** -- features providing unique information
- **Manageable team** -- not so many that noise dominates

<div class="callout-insight">
💡 **Key Insight:** With 100 features: $2^{100} \approx 10^{30}$ possible teams!
</div>

---

<!-- Speaker notes: Explain the curse of dimensionality using concrete numbers. The p/n ratio of 0.4 is a red flag in practice. This motivates why we MUST reduce features. -->

## The Curse of Dimensionality

For time series with $n$ observations and $p$ features:

$$\text{Overfitting Risk} \propto \frac{p}{n}$$

**Example -- Stock price prediction:**
- $n = 250$ trading days (1 year)
- $p = 100$ technical indicators
- Ratio: $p/n = 0.4$ (HIGH RISK)

<div class="callout-warning">
⚠️ **Warning:** When $p \approx n$ or $p > n$, models learn **noise instead of signal**.
</div>

---

<!-- Speaker notes: Walk through the diagram from left to right. Emphasize that each strategy trades off computational cost against solution quality. The GA path is what this course focuses on. -->

## Search Strategy Overview

![GA Lifecycle](ga_lifecycle.svg)

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
    A[Feature Selection<br/>Search Strategies] --> B[Exhaustive]
    A --> C[Greedy]
    A --> D[Metaheuristic]
    B --> B1["Try all 2^n subsets<br/>Guaranteed optimal<br/>Only feasible p <= 15"]
    C --> C1[Forward Selection]
    C --> C2[Backward Elimination]
    D --> D1[Genetic Algorithms]
    D --> D2[Simulated Annealing]
    D --> D3[Particle Swarm]
```

---

<!-- Speaker notes: This is a key argument for GAs. The XOR problem shows that greedy methods fundamentally cannot find features whose value only emerges in combination. Walk through the correlations and ask: "Would forward selection ever pick A or B?" -->

## Why Greedy Fails: The XOR Problem

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">xor_problem.py</span>
</div>

```python
np.random.seed(42)
n = 1000

A = np.random.randint(0, 2, n)
B = np.random.randint(0, 2, n)
C = np.random.randn(n)  # Noise feature

y = (A ^ B) + 0.1 * np.random.randn(n)
X = np.column_stack([A, B, C])
```

</div>

**The problem:**
- A alone: correlation with y $\approx$ 0
- B alone: correlation with y $\approx$ 0
- {A, B} together: **high predictive power**

<div class="callout-key">
🔑 **Key Point:** Greedy evaluates features individually and misses the interaction!
</div>

---

<!-- Speaker notes: Use this diagram to contrast the single-path nature of greedy search with the parallel exploration of GAs. This is the core motivation for the entire course. -->

## Greedy vs. Population-Based Search

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart LR
    subgraph "Greedy Search"
        G1[Start: empty] --> G2["+best single"] --> G3["+best single"] --> G4["Local optimum"]
    end
    subgraph "GA Search"
        P1["Pop of 50<br/>random subsets"] --> P2["Select best<br/>Crossover"] --> P3["Mutate<br/>Evaluate"] --> P4["Global optimum"]
        P3 -->|"Repeat"| P2
    end
```

<div class="flow">
<div class="flow-step amber">Greedy: Single Path</div>
<div class="flow-arrow">→</div>
<div class="flow-step amber">Local Optimum</div>
<div class="flow-arrow">vs</div>
<div class="flow-step mint">GA: Population</div>
<div class="flow-arrow">→</div>
<div class="flow-step mint">Global Optimum</div>
</div>

---

<!-- Speaker notes: Briefly introduce metaheuristics as a class of optimization methods. The key point is that GAs are one of several options, but are particularly well-suited to feature selection because of binary encoding. -->

## Why GAs for Feature Selection?

1. **Natural encoding**: Binary chromosome = feature mask
   ```
   [1, 0, 1, 1, 0, 0, 1, 0] = features {0, 2, 3, 6} selected
   ```

2. **Population-based**: Explore multiple solutions simultaneously

3. **Crossover**: Combine good feature subsets from different parents

4. **No Free Lunch**: No single algorithm is best for all problems -- GAs are strong for combinatorial binary search

<div class="callout-insight">
💡 **Key Insight:** Binary encoding maps directly to feature masks, making GAs a natural fit for feature selection.
</div>

---

<!-- Speaker notes: The multi-objective formulation is important because in practice we almost always care about both accuracy AND simplicity. Introduce the Pareto frontier concept here; it will be covered in depth in Module 02. -->

## Multi-Objective Formulation

Feature selection often balances competing goals:

$$\min_{s} \begin{cases}
f_1(s) = \text{Prediction Error}(X_s, y) \\
f_2(s) = ||s||_0 = \text{Number of Features}
\end{cases}$$

```
Prediction Error
    |  *  Dominated
    |   *
    |    o---o---o  Pareto Frontier
    |         o--o
    |             o
    +-----------------> # Features
```

<div class="callout-info">
ℹ️ **Info:** Each point on the Pareto frontier is optimal -- improving one objective worsens the other.
</div>

---

<!-- _class: lead -->
<!-- Speaker notes: Transition to the practical pitfalls section. These are the mistakes that even experienced practitioners make. -->

# Common Pitfalls

---

<!-- Speaker notes: Data leakage is the most common and most dangerous pitfall. Show both code snippets side-by-side and emphasize that the WRONG version looks simpler but gives optimistic results. -->

## Pitfall 1: Selection on Full Dataset

<div class="compare">
<div class="compare-card">
<div class="header before">WRONG -- data leakage</div>
<div class="body">

```python
# Uses ALL data for selection
selected = select_features(X, y)
cv_score = cross_val_score(
    model, X[:, selected], y, cv=5
)
```

</div>
</div>
<div class="compare-card">
<div class="header after">RIGHT -- selection inside CV</div>
<div class="body">

```python
tscv = TimeSeriesSplit(n_splits=5)
scores = []
for train_idx, test_idx in tscv.split(X):
    X_train = X[train_idx]
    y_train = y[train_idx]
    selected = select_features(
        X_train, y_train
    )
    model.fit(X_train[:, selected],
              y_train)
    score = model.score(
        X[test_idx][:, selected],
        y[test_idx]
    )
    scores.append(score)
```

</div>
</div>
</div>

---

<!-- Speaker notes: Explain that correlated features waste model capacity. The correlation penalty in the fitness function is a preview of Module 02 content. -->

## Pitfall 2: Ignoring Feature Redundancy

Selecting multiple highly correlated features wastes degrees of freedom.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">fitness_diversity.py</span>
</div>

```python
def fitness_with_diversity(features, X, y):
    X_selected = X[:, features == 1]
    error = cross_val_mse(X_selected, y)

    # Correlation penalty
    if X_selected.shape[1] > 1:
        corr = np.corrcoef(X_selected.T)
        avg_corr = (np.sum(np.abs(corr)) -
                    X_selected.shape[1]) / \
                   (X_selected.shape[1]**2 -
                    X_selected.shape[1])
    else:
        avg_corr = 0

    return -error - 0.1 * avg_corr
```

</div>

<div class="callout-key">
🔑 **Key Point:** Add correlation penalty to promote diverse feature sets.
</div>

---

<!-- Speaker notes: This decision flow is a practical reference the audience can use. Walk through each branch and give examples of when you'd take that path. -->

## Decision Flow for Feature Selection

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
    A[Start: p features] --> B{p > 15?}
    B -->|No| C[Exhaustive Search<br/>Guaranteed optimal]
    B -->|Yes| D{Need interactions?}
    D -->|No| E[Filter Methods<br/>Fast screening]
    D -->|Yes| F{Compute budget?}
    F -->|Limited| G[Greedy Search<br/>Forward/Backward]
    F -->|Generous| H[Genetic Algorithm<br/>Population-based]
    E --> I[Candidate Set]
    G --> I
    H --> I
    C --> I
    I --> J[Validate on<br/>held-out data]
```

---

<!-- Speaker notes: Wrap up with the connections slide. Emphasize that this deck sets the foundation -- the rest of the course builds on these concepts. -->

## Connections & What's Next

<div class="compare">
<div class="compare-card">
<div class="header before">Builds On</div>
<div class="body">

- Linear algebra (feature spaces)
- Optimization theory
- Probability (overfitting)
- Time series fundamentals

</div>
</div>
<div class="compare-card">
<div class="header after">Leads To</div>
<div class="body">

- **Module 1**: GA fundamentals
- **Module 2**: Fitness function design
- **Module 3**: Time series CV strategies
- **Module 5**: Multi-objective optimization

</div>
</div>
</div>

---

<!-- Speaker notes: Use this summary as a quick reference card. Emphasize the two key takeaways: 1) Feature selection is NP-hard, and 2) GAs are a powerful tool for this problem. -->

## Key Takeaways

| Insight | Detail |
|---------|--------|
| **NP-hard** | Exhaustive search infeasible for $p > 15$ |
| **Greedy fails** | Gets stuck in local optima, misses interactions |
| **Curse of dimensionality** | $p/n > 0.3$ leads to overfitting |
| **Metaheuristics** | Explore search space more effectively |
| **GAs well-suited** | Binary encoding maps directly to feature masks |
| **Multi-objective** | Balance accuracy vs. parsimony on Pareto frontier |

<div class="callout-info">
ℹ️ **Next**: Feature selection approaches -- filter, wrapper, and embedded methods.
</div>
