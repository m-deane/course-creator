---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->
<!-- Speaker notes: The curse of dimensionality is the mathematical foundation for why feature selection matters at all. If high-dimensional data behaved like low-dimensional data, we could just use all features and be done with it. This deck shows why we cannot. By the end, students will understand three concrete phenomena: volume concentration, distance breakdown, and the Hughes phenomenon. -->

# The Curse of Dimensionality

## Module 00 — Why High Dimensions Break Your Intuition

*Bellman (1957): the exponential cost of adding dimensions*

---

<!-- Speaker notes: Start with the intuition. In 1D, 10 points cover an interval reasonably. In 2D, you need 100. In 3D, 1000. The exponent keeps growing. For d=50, you would need 10^50 points — more than atoms in the universe — to achieve the same density as 10 points in 1D. This is not a computational trick; it is fundamental geometry. -->

## The Core Problem: Exponential Data Hunger

To cover $[0,1]^d$ with resolution $\epsilon = 0.1$, you need:

$$N(\epsilon, d) = \left\lceil \frac{1}{\epsilon} \right\rceil^d = 10^d \text{ samples}$$

| Dimensions $d$ | Samples needed | Context |
|---|---|---|
| 1 | 10 | Trivial |
| 5 | 100,000 | Large dataset |
| 10 | $10^{10}$ | All human text ever written |
| 20 | $10^{20}$ | More than grains of sand on Earth |
| 50 | $10^{50}$ | More than atoms in universe |

**With $n=10{,}000$ samples, your data is increasingly sparse as $d$ grows.**

---

<!-- Speaker notes: The sphere-in-cube ratio is the cleanest demonstration of volume concentration. At d=2, a circle fills 78% of its bounding square. At d=10, a sphere fills only 0.25% of its bounding hypercube. At d=20, the fraction is essentially zero. The geometric intuition: in high dimensions, the corners of the cube are the dominant region, and the sphere barely touches them. -->

## Volume Concentration: The Sphere Disappears

Volume of a $d$-dimensional sphere of radius $r$:

$$V_d(r) = \frac{\pi^{d/2}}{\Gamma\!\left(\frac{d}{2}+1\right)} r^d$$

Fraction of unit hypercube occupied by inscribed sphere of radius $\frac{1}{2}$:

| $d$ | $V_d(1/2) / V_\text{cube}$ |
|---|---|
| 2 | 0.785 |
| 5 | 0.165 |
| 10 | 0.00249 |
| 20 | $2.5 \times 10^{-8}$ |
| 100 | $\approx 0$ |

**The sphere vanishes. In high $d$, almost all volume is in the corners.**

---

<!-- Speaker notes: This is the more counterintuitive result. For the unit hypersphere itself, essentially all the volume is in a thin shell near the surface. At d=1000, 99% of the sphere's volume lies within 1% of the surface. Sample uniformly from a high-dimensional sphere and you always land near the edge. There is no interior. This kills density-based algorithms — where do you place the density estimate if the interior is empty? -->

## Concentration in a Shell

Fraction of $B_d(1)$ volume in outer shell of thickness $\epsilon$:

$$\frac{V_d(1) - V_d(1-\epsilon)}{V_d(1)} = 1 - (1-\epsilon)^d$$

<div class="columns">

| $\epsilon$ | $d=10$ | $d=100$ | $d=1000$ |
|---|---|---|---|
| 0.01 | 9.6% | 63.4% | ~100% |
| 0.05 | 40.1% | 99.4% | ~100% |
| 0.10 | 65.1% | ~100% | ~100% |

**Interpretation**

At $d=1000$, a shell of thickness 1% of the radius contains virtually 100% of the sphere's volume.

Uniform samples cluster at the surface — there is effectively no interior.

</div>

---

<!-- Speaker notes: Now shift to distances. Beyer et al. (1999) proved this rigorously. The ratio of max to min pairwise distance approaches 1 as d grows. Walk through the coefficient of variation calculation — the key insight is that distance variance grows as sqrt(d) but mean distance grows as sqrt(d), so the CV shrinks as 1/sqrt(d). All points look equally far away from any query point. -->

## Distance Metric Breakdown

For $n$ uniform points in $[0,1]^d$, distances all converge to the same value:

$$\frac{d_\text{max} - d_\text{min}}{d_\text{min}} \xrightarrow{d \to \infty} 0$$

**Why:** Expected squared distance from origin:
$$\mathbb{E}[\|\mathbf{x}\|_2^2] = d \cdot \text{Var}(x_j) = \frac{d}{12}$$

Coefficient of variation of pairwise distances:

$$\text{CV} = \frac{12}{\sqrt{45 \cdot d}} \propto d^{-1/2} \xrightarrow{d \to \infty} 0$$

**All distances converge.** "Nearest neighbour" loses meaning.

---

<!-- Speaker notes: Make this concrete. In 2D, the nearest neighbour is actually nearby. In 50D with 1000 points, the nearest neighbour is 87% of the way across the space. In 100D, virtually all neighbours are at the maximum possible distance. KNN, kernel methods, and any algorithm that relies on the concept of "nearby points" fails in this regime. -->

## When "Nearest" Means Nothing

Expected distance from a query to its nearest neighbour among $n$ uniform points in $[0,1]^d$:

$$r_\text{nn}(n, d) \approx \left(\frac{1}{n}\right)^{1/d}$$

| $n$ | $d=2$ | $d=10$ | $d=50$ | $d=100$ |
|---|---|---|---|---|
| 100 | 0.10 | 0.79 | 0.97 | 0.99 |
| 1,000 | 0.03 | 0.50 | 0.87 | 0.93 |
| 10,000 | 0.01 | 0.25 | 0.75 | 0.86 |

Values are fraction of the diagonal of the unit hypercube.

**With $n=1{,}000$ and $d=50$:** nearest neighbour is 87% of the way to the far corner. Useless as a local estimate.

---

<!-- Speaker notes: The algorithms affected read like a who's who of machine learning. KNN is the most obvious victim. KDE needs bandwidth that grows exponentially. K-means centroids lose meaning when all points are equidistant. RBF kernels with fixed bandwidth become flat (kernel value approximately constant for all pairs). DBSCAN's epsilon parameter has no consistent interpretation. The bottom line: anything based on local structure fails. -->

## Algorithms Broken by High Dimensions

| Algorithm | Failure mechanism |
|---|---|
| **k-Nearest Neighbours** | No meaningful notion of "near" |
| **Kernel Density Estimation** | Bandwidth must grow exponentially with $d$ |
| **K-Means** | All inter-centroid distances converge |
| **Gaussian Processes (RBF kernel)** | Covariance $\approx$ constant; data looks uncorrelated |
| **DBSCAN** | No consistent $\epsilon$ threshold across dimensions |
| **SVM (RBF kernel)** | Kernel matrix approaches identity |
| **Naive Bayes (Gaussian)** | Works — independence assumption helps |
| **Lasso / Ridge** | Works — regularisation controls complexity |

**Feature selection reduces $d$, restoring validity of the first group.**

---

<!-- Speaker notes: Hughes (1968) is the most important counterintuitive result for practitioners. Adding features can hurt. The mathematics: estimation error per parameter is sigma^2/n. With p parameters, total estimation variance is p*sigma^2/n. When this exceeds the information gain from new features, performance drops. The optimal p grows only as sqrt(n * d_0), not as d_0. This is why feature selection can beat using all available features even when those features are informative. -->

## The Hughes Phenomenon

**Counterintuitive result:** Adding features *decreases* accuracy when $p/n$ is too large.

**Why:** Estimation error of $\hat{\boldsymbol{\beta}}$ totals:

$$\sum_{j=1}^p \text{Var}(\hat{\beta}_j) \propto \frac{p \sigma^2}{n}$$

Adding feature $j$:
- **Gain:** reduced bias $\propto \beta_j^2$
- **Cost:** increased variance $\propto \sigma^2/n$

If $\beta_j^2 < \sigma^2/n$, the feature **hurts** out-of-sample performance.

**Optimal feature count for LDA:**

$$p^* \approx \sqrt{n \cdot d_0}$$

where $d_0$ = number of truly informative features.

---

<!-- Speaker notes: Show that Hughes is not theoretical. For n=100 and d_0=10, the optimal p is about 32. You should use 32 features, not 10 (too few) and not 100 or 500 (too many). The optimal is somewhere in between, and it depends on n. This is why "use all your features" is wrong advice in the p/n regime above 0.2. -->

## Hughes Phenomenon: Optimal Feature Count

For Gaussian data with $n$ samples and $d_0$ truly informative features:

$$p^* \approx \sqrt{n \cdot d_0}$$

| $n$ | $d_0=10$ | $d_0=50$ | $d_0=100$ |
|---|---|---|---|
| 100 | **32** | 71 | 100 |
| 500 | **71** | 158 | 224 |
| 1,000 | **100** | 224 | 316 |
| 10,000 | **316** | 707 | 1,000 |

**Key insight:** Even with $d_0=10$ truly informative features and $n=100$ samples, the optimal number of features to use is **32**, not 10 and certainly not all 500 available features.

Including noise features and extra informative features *both* hurt performance when $n$ is small.

---

<!-- Speaker notes: The p/n ratio is the single most useful heuristic for quickly assessing dimensionality risk. Below 0.05, standard methods are fine. Between 0.2 and 1.0, feature selection is not optional — it's required. Above 1.0 (p>n), OLS is undefined and even logistic regression is unstable. Above 10, you need Lasso/Elastic Net as a baseline. This table should be memorised or bookmarked. -->

## The $p/n$ Risk Ratio

$$\text{Risk indicator:} \quad \frac{p}{n}$$

| $p/n$ | Risk | Action required |
|---|---|---|
| $< 0.05$ | Low | Standard methods |
| $0.05$–$0.2$ | Moderate | Add regularisation |
| $0.2$–$1.0$ | High | Feature selection mandatory |
| $1.0$–$10$ | Very high | Lasso/Elastic Net baseline |
| $> 10$ | Extreme | Strong regularisation + selection |

**Rule of thumb for OLS stability:** need $n \geq 10p$

**Rule of thumb for KNN reliability:** need $n \sim r^{-d}$ (exponential in $d$)

---

<!-- Speaker notes: Sample complexity is the theoretical underpinning. PAC learning gives a linear bound in p for linear classifiers — that is manageable. But non-parametric methods (KNN regression, kernel smoothing) have an exponential bound in d. The exponent is (d+2)/2. At d=100, you need n proportional to epsilon^{-51} — astronomically large. This is why non-parametric methods are rarely used in high dimensions and why practitioners default to linear methods with regularisation. -->

## Sample Complexity: Formal Bounds

**PAC bound** for linear classifier with $p$ features:

$$n \geq \frac{1}{\epsilon}\!\left((p+1)\ln\frac{2}{\epsilon} + \ln\frac{1}{\delta}\right) \quad \text{— linear in } p$$

**Non-parametric regression** (KNN, kernel smoothing) to achieve MSE $\leq \epsilon$:

$$n = O\!\left(\epsilon^{-(d+2)/2}\right)$$

| $d$ | Exponent on $\epsilon^{-1}$ |
|---|---|
| 2 | 2 |
| 10 | 6 |
| 20 | 11 |
| 100 | 51 |

**Non-parametric methods become intractable by $d \approx 20$.**

Feature selection must reduce $d$ before non-parametric models are viable.

---

<!-- Speaker notes: The practical table. Most practitioners need a quick rule, not a theorem. The p/n table from slide 10 is one tool. This slide adds the algorithm-specific guidance: how many features you can handle per algorithm, given a typical n. These are rough empirical rules from the literature, not hard mathematical bounds. -->

## Practical Implications for Feature Counts

**Algorithm-specific guidance:**

| Algorithm | Safe $p/n$ ratio | Rule of thumb |
|---|---|---|
| OLS / Linear regression | $< 0.1$ | $n \geq 10p$ |
| Logistic regression | $< 0.1$ | 10 events per predictor |
| Random Forest | $< 5$ | Handles $p > n$ partially |
| Gradient Boosting | $< 2$ | Subsampling helps |
| Lasso / Elastic Net | $< 50$ | Designed for $p \gg n$ |
| KNN | $< 0.01$ | $d < 20$ as strict cap |
| Gaussian Process (RBF) | $< 0.005$ | $d < 10$ as strict cap |

**Bottom line:** Most algorithms need $p/n < 0.1$ for reliable generalisation. Feature selection is how you achieve this.

---

<!-- Speaker notes: Tie the curse back to feature selection. This is the payoff slide. The curse explains why we need feature selection. Reducing d from 500 to 20 is not just about speed — it is about making the problem geometrically well-posed. Distance-based algorithms work again, density estimates are meaningful, and nearest neighbours are actually near. Quantify: from 10 points per dimension needed (rough) to 10^500 points down to 10^20 — still large but much more tractable. -->

## Why Feature Selection Is the Cure

The curse is exponential in $d$. Feature selection reduces $d$.

**Concrete example:** $n=5{,}000$, $p=500$ (only 20 truly informative features)

| Approach | Effective $d$ | $p/n$ ratio | KNN reliability |
|---|---|---|---|
| All features | 500 | 0.1 | Poor |
| Top-100 filter | 100 | 0.02 | Marginal |
| Selected 20 | 20 | 0.004 | Good |

**Distance quality:** With 20 features and 5,000 samples:
$$r_\text{nn} \approx 5000^{-1/20} \approx 0.38 \quad \text{(meaningful locality)}$$

With 500 features and 5,000 samples:
$$r_\text{nn} \approx 5000^{-1/500} \approx 0.98 \quad \text{(all points equidistant)}$$

---

<!-- Speaker notes: Three takeaways. The volume/distance/sample results are all manifestations of the same underlying reality: high-dimensional spaces are mostly empty and all points look similar. The Hughes phenomenon is the empirical consequence. And p/n = 0.1 is the practical threshold to memorise. The next guide will show the computational cost of the methods that fix this. -->

<!-- _class: lead -->

## Three Takeaways

1. **High-dimensional space is mostly empty** — data sparsity grows exponentially with $d$

2. **Adding features can hurt** — the Hughes phenomenon is real and measurable

3. **Target $p/n < 0.1$** for most algorithms; feature selection is how you get there

*Next: Computational Complexity — the cost of finding that good subset*
