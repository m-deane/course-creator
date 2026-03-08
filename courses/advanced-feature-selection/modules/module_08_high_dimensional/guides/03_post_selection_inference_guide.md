# Post-Selection Inference: Valid Inference After Feature Selection

## In Brief

Classical confidence intervals and p-values assume the model was specified before seeing the data. Feature selection breaks this assumption: when we use data to select features and then estimate their effects, the resulting inference is invalid — coverage rates collapse and p-values are systematically too small. Post-selection inference (PoSI) develops methods that provide valid inference conditional on the selection event.

## Key Insight

A Lasso-selected feature is selected precisely because its marginal association with $y$ is large. Reporting a confidence interval for its effect using the same data that selected it produces a biased estimate: we are conditioning on an event (selection) that was correlated with the outcome. This is selection bias at the inferential level, not just the estimation level.

---

## 1. The Post-Selection Inference Problem

### 1.1 Why Naive CIs Fail

Let the true model be $\mathbf{y} = \mathbf{X}_{\mathcal{M}^*} \boldsymbol{\beta}^*_{\mathcal{M}^*} + \boldsymbol{\varepsilon}$ with $\boldsymbol{\varepsilon} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$.

Suppose we run Lasso with penalty $\lambda$ and obtain selected set $\hat{\mathcal{M}}$. We then compute OLS estimates on $\hat{\mathcal{M}}$:

$$\hat{\boldsymbol{\beta}}_{\hat{\mathcal{M}}}^{\text{OLS}} = (\mathbf{X}_{\hat{\mathcal{M}}}^\top \mathbf{X}_{\hat{\mathcal{M}}})^{-1} \mathbf{X}_{\hat{\mathcal{M}}}^\top \mathbf{y}$$

The naive 95% CI is $\hat{\beta}_j \pm 1.96 \cdot \hat{\sigma} / \sqrt{n}$.

**The coverage of this CI is not 95%.** In simulation with $n = 100$, $p = 500$, $s = 10$:

| True coverage of "95%" CI | Empirical coverage |
|---|---|
| Naive OLS on Lasso selected set | 45–60% |
| Debiased Lasso CI | 92–96% |
| Data splitting CI | 93–97% |

The naive interval can be off by a factor of 2 in coverage.

### 1.2 The Selection Event

Define the Lasso selection event at penalty $\lambda$ as:

$$\mathcal{E}_\lambda = \left\{ \hat{\mathcal{M}}(\mathbf{y}, \mathbf{X}, \lambda) = \hat{\mathcal{M}} \right\}$$

The naive CI conditions on $\mathcal{E}_\lambda$ implicitly (we observe which features were selected) but does not account for this conditioning in the interval construction. The result is that:

$$P\!\left(\beta_j^* \in \hat{\beta}_j^{\text{OLS}} \pm z_{0.975} \hat{\sigma}_j \;\middle|\; \mathcal{E}_\lambda\right) \neq 0.95$$

Post-selection inference explicitly accounts for conditioning on $\mathcal{E}_\lambda$.

---

## 2. Debiased Lasso

### 2.1 The Lasso Bias Problem

The Lasso coefficient $\hat{\boldsymbol{\beta}}^{\text{Lasso}}$ satisfies the KKT conditions:

$$\frac{1}{n}\mathbf{X}^\top (\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}) = \lambda \hat{\boldsymbol{z}}$$

where $\hat{z}_j = \text{sign}(\hat{\beta}_j)$ for $\hat{\beta}_j \neq 0$ and $|\hat{z}_j| \leq 1$ for $\hat{\beta}_j = 0$.

The Lasso is biased: $\mathbb{E}[\hat{\beta}_j^{\text{Lasso}}] \neq \beta_j^*$. The bias is $O(\lambda)$ for active features and $O(\lambda)$ for zero features. Standard CIs based on $\hat{\boldsymbol{\beta}}^{\text{Lasso}}$ directly are invalid.

### 2.2 Construction of the Debiased Lasso

Van de Geer et al. (2014) and Zhang & Zhang (2014) independently propose a bias correction. The key identity is:

$$\sqrt{n}(\hat{\boldsymbol{\beta}}^{\text{Lasso}} - \boldsymbol{\beta}^*) = -\frac{1}{\sqrt{n}}\hat{\mathbf{\Theta}} \mathbf{X}^\top \boldsymbol{\varepsilon} + \sqrt{n} \Delta$$

where $\hat{\mathbf{\Theta}}$ is an approximate inverse of $\hat{\boldsymbol{\Sigma}} = \mathbf{X}^\top \mathbf{X}/n$, and $\Delta$ is a remainder term that is $o(1)$ under sparsity conditions.

**Debiased estimator:**

$$\hat{\boldsymbol{\beta}}^{\text{d}} = \hat{\boldsymbol{\beta}}^{\text{Lasso}} + \frac{1}{n} \hat{\mathbf{\Theta}} \mathbf{X}^\top (\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}^{\text{Lasso}})$$

The correction term $\frac{1}{n}\hat{\mathbf{\Theta}} \mathbf{X}^\top (\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}^{\text{Lasso}})$ is the residual from the Lasso, back-projected through the approximate precision matrix $\hat{\mathbf{\Theta}}$.

### 2.3 The Approximate Inverse (Nodewise Lasso)

Computing $\hat{\mathbf{\Theta}} \approx \hat{\boldsymbol{\Sigma}}^{-1}$ when $p > n$ requires $\hat{\boldsymbol{\Sigma}}$ to be invertible — which it is not. Instead, compute an approximate inverse column by column via Lasso regressions:

For each feature $j$:
1. Regress $\mathbf{x}_j$ on all other features using Lasso: $\hat{\boldsymbol{\gamma}}_j = \arg\min_{\boldsymbol{\gamma}: \gamma_j = 0} \frac{1}{2n}\|\mathbf{x}_j - \mathbf{X}_{-j}\boldsymbol{\gamma}\|^2 + \mu_j \|\boldsymbol{\gamma}\|_1$
2. Compute residuals: $\hat{\mathbf{m}}_j = \mathbf{x}_j - \mathbf{X}_{-j}\hat{\boldsymbol{\gamma}}_j$
3. Set: $\hat{\Theta}_{jj} = 1/(\hat{\mathbf{m}}_j^\top \mathbf{x}_j / n)$ and $\hat{\Theta}_{jk} = -\hat{\gamma}_{jk}/(\hat{\mathbf{m}}_j^\top \mathbf{x}_j / n)$

This runs $p$ Lasso regressions — $O(p^2 n)$ total — expensive but parallelisable.

### 2.4 Asymptotic Distribution

**Theorem (van de Geer et al., 2014):** Under sparsity conditions on $\boldsymbol{\beta}^*$ and $\hat{\mathbf{\Theta}}$:

$$\sqrt{n}(\hat{\beta}_j^{\text{d}} - \beta_j^*) / \hat{\sigma}_j \xrightarrow{d} \mathcal{N}(0, 1)$$

where $\hat{\sigma}_j^2 = \hat{\sigma}^2 \hat{\Theta}_{jj}$ is the estimated standard error.

This gives a valid (1-$\alpha$) confidence interval:

$$\hat{\beta}_j^{\text{d}} \pm z_{1-\alpha/2} \cdot \hat{\sigma}_j / \sqrt{n}$$

valid simultaneously for all $j$ with a Bonferroni correction.

---

## 3. Selective Inference

### 3.1 The Polyhedral Lemma

Lee et al. (2016) take a different approach: rather than debiasing the estimator, they derive the exact distribution of $\hat{\boldsymbol{\beta}}_{\hat{\mathcal{M}}}^{\text{OLS}}$ conditional on the Lasso selection event $\mathcal{E}_\lambda$.

**Key insight:** The Lasso selection event $\hat{\mathcal{M}} = \mathcal{M}$ is equivalent to a set of linear constraints on $\mathbf{y}$:

$$\left\{ \hat{\mathcal{M}}(\mathbf{y}) = \mathcal{M} \right\} = \left\{ \mathbf{A}_{\mathcal{M}} \mathbf{y} \leq \mathbf{b}_{\mathcal{M}} \right\}$$

for explicitly computable matrices $\mathbf{A}_{\mathcal{M}}$ and vectors $\mathbf{b}_{\mathcal{M}}$ (polyhedral set in $\mathbf{y}$).

### 3.2 The Truncated Normal Distribution

For any linear contrast $\eta^\top \mathbf{y}$ (e.g., $\eta^\top = \mathbf{e}_j^\top (\mathbf{X}_{\hat{\mathcal{M}}}^\top \mathbf{X}_{\hat{\mathcal{M}}})^{-1} \mathbf{X}_{\hat{\mathcal{M}}}^\top$ gives the OLS estimate of $\beta_j$), the conditional distribution is:

$$\eta^\top \mathbf{y} \;\middle|\; \left(\mathcal{E}_\lambda, \, \mathbf{P}_\eta^\perp \mathbf{y}\right) \sim \text{TruncNorm}(\eta^\top \mathbf{X}\boldsymbol{\beta}^*, \sigma^2 \|\eta\|^2, [\mathcal{V}^-, \mathcal{V}^+])$$

where $\mathcal{V}^-$ and $\mathcal{V}^+$ are the truncation limits derived from the polyhedral constraints, and $\mathbf{P}_\eta^\perp \mathbf{y}$ is the component of $\mathbf{y}$ orthogonal to $\eta$ (conditioned out).

**Inference:** Compute p-values and CIs using the truncated normal survival function — implemented in the `selectiveInference` R package and `pyselective` Python package.

### 3.3 Limitations of Selective Inference

- **Conservative with many selections:** Conditioning on the exact selection event loses power when $|\hat{\mathcal{M}}|$ is large.
- **Requires known $\sigma^2$**: The truncated normal distribution uses $\sigma^2$; unknown $\sigma^2$ requires an additional estimation step.
- **Multiple testing:** Constructing valid CIs for all selected features simultaneously requires further correction.
- **Computationally expensive:** For $p > 10{,}000$, constructing $\mathbf{A}_{\mathcal{M}}$ and $\mathbf{b}_{\mathcal{M}}$ is non-trivial.

---

## 4. Data Splitting for Valid Inference

### 4.1 The Sample Splitting Strategy

The simplest approach to post-selection inference is to use separate data for selection and inference:

```
Full dataset: n observations
        │
        ├── Selection set: n₁ = n/2 observations
        │   → Run Lasso → obtain M_hat
        │
        └── Inference set: n₂ = n/2 observations
            → OLS on X[inference, M_hat]
            → Standard CIs are valid (no selection bias)
```

**Validity:** Because $\hat{\mathcal{M}}$ is computed on the selection set and is independent of the inference set observations, classical OLS theory applies on the inference set — no post-selection adjustment needed.

### 4.2 Power Loss from Data Splitting

The main cost of data splitting is efficiency: both steps use only $n/2$ observations.
- Selection step: Fewer data means Lasso may miss some active features
- Inference step: Standard errors scale as $1/\sqrt{n_2} = 1/\sqrt{n/2}$ — wider CIs by factor $\sqrt{2}$

**Multi-split aggregation (Meinshausen, Meier & Bühlmann, 2009):**

Repeat the split $B$ times, collect $B$ sets of p-values $\{p_{jb}\}_{b=1}^B$, aggregate:

$$\tilde{p}_j = \min\!\left(1, \; Q_\gamma\!\left(\frac{p_{jb} \cdot |\hat{\mathcal{M}}_b|}{\gamma}\right)\right)$$

where $Q_\gamma$ is the $\gamma$-quantile over splits and $|\hat{\mathcal{M}}_b|$ is the size of the selected set in split $b$. This reduces variability from the random split while maintaining validity.

### 4.3 Practical Data Splitting Workflow

```python
import numpy as np
from sklearn.linear_model import LassoCV, LinearRegression
from scipy import stats

def data_splitting_inference(X, y, alpha=0.05, n_splits=100):
    n, p = X.shape
    n1 = n // 2  # selection set size

    pvalues = []
    for _ in range(n_splits):
        # Random 50/50 split
        idx = np.random.permutation(n)
        sel_idx, inf_idx = idx[:n1], idx[n1:]

        # Selection on first half
        lasso = LassoCV(cv=5).fit(X[sel_idx], y[sel_idx])
        selected = np.where(lasso.coef_ != 0)[0]
        if len(selected) == 0:
            continue

        # OLS inference on second half
        X_inf = X[np.ix_(inf_idx, selected)]
        ols = LinearRegression().fit(X_inf, y[inf_idx])

        # Compute p-values from t-statistics
        n2 = len(inf_idx)
        residuals = y[inf_idx] - ols.predict(X_inf)
        sigma_sq = np.sum(residuals**2) / (n2 - len(selected) - 1)
        XtX_inv = np.linalg.pinv(X_inf.T @ X_inf)
        se = np.sqrt(sigma_sq * np.diag(XtX_inv))
        t_stats = ols.coef_ / se
        p_vals = 2 * stats.t.sf(np.abs(t_stats), df=n2 - len(selected) - 1)

        pvalues.append((selected, p_vals))

    return pvalues
```

---

## 5. Knockoff+ for FDR Control in High Dimensions

### 5.1 The FDR Control Objective

Rather than controlling the familywise error rate (FWER) as Bonferroni does, the Knockoff procedure controls the False Discovery Rate:

$$\text{FDR} = \mathbb{E}\!\left[\frac{|\hat{\mathcal{M}} \cap (\mathcal{M}^*)^c|}{|\hat{\mathcal{M}}| \vee 1}\right]$$

### 5.2 Fixed-X Knockoffs (Barber & Candès, 2015)

Construct knockoff features $\tilde{\mathbf{X}}$ satisfying:
1. **Pairwise correlations matched:** $\mathbf{X}^\top \tilde{\mathbf{X}} = \text{diag}(s_1, \ldots, s_p)$ for some $s_j \geq 0$
2. **Self-correlations preserved:** $\tilde{\mathbf{X}}^\top \tilde{\mathbf{X}} = \mathbf{X}^\top \mathbf{X}$

The augmented design matrix $[\mathbf{X}, \tilde{\mathbf{X}}]$ is then fed to Lasso. A feature $j$ is selected if it enters the Lasso path before its knockoff $\tilde{j}$:

$$W_j = Z_j - \tilde{Z}_j$$

where $Z_j$ is the regularisation value at which $j$ first becomes non-zero, and $\tilde{Z}_j$ is the same for the knockoff $\tilde{j}$.

**Knockoff+ threshold:** Select features with $W_j \geq T$ where:

$$T = \min\!\left\{ t > 0 : \frac{|\{j : W_j \leq -t\}| + 1}{|\{j : W_j \geq t\}| \vee 1} \leq q \right\}$$

**Theorem (Barber & Candès, 2015):** The Knockoff+ procedure controls FDR at level $q$ in finite samples under no distributional assumptions on $\boldsymbol{\beta}^*$.

### 5.3 Model-X Knockoffs (Candès et al., 2018)

For $p > n$ (the high-dimensional regime), fixed-X knockoffs cannot be constructed (the augmented matrix $[\mathbf{X}, \tilde{\mathbf{X}}]$ would have $2p > 2n > n$ features and rank at most $n$). Model-X knockoffs relax the construction to require only:

$$(\mathbf{X}, \tilde{\mathbf{X}}) \overset{d}{=} (\mathbf{X}_{\text{swap}(S)}, \tilde{\mathbf{X}}_{\text{swap}(S)}) \quad \forall S \subseteq \{1, \ldots, p\}$$

where $\text{swap}(S)$ swaps features $j$ and $\tilde{j}$ for $j \in S$. This exchangeability condition is sufficient for FDR control and can be achieved when the distribution of $\mathbf{X}$ is known (e.g., Gaussian with known covariance).

---

## 6. Compact GA for Ultra-Wide Data

### 6.1 Population Reduction in High-Dimensional Evolutionary Search

Standard genetic algorithms with population size $N$ and chromosome length $p$ require $O(Np)$ memory and $O(NpT)$ fitness evaluations — infeasible for $p > 10{,}000$.

The Compact Genetic Algorithm (cGA, Harik et al., 1999) maintains only a probability vector $\boldsymbol{\pi} \in [0,1]^p$ rather than an explicit population:

- $\pi_j^{(t)}$ = probability that feature $j$ is selected in generation $t$
- Memory: $O(p)$ rather than $O(Np)$

**cGA update:**

```
For t = 1, 2, ..., until convergence:
    1. Sample two candidate solutions:
       s_a, s_b ~ Bernoulli(pi^(t))
    2. Evaluate fitness: f_a = fitness(s_a), f_b = fitness(s_b)
    3. Winner: w = s_a if f_a > f_b, else s_b
       Loser:  l = s_b if f_a > f_b, else s_a
    4. Update: for j where w_j != l_j:
       pi_j^(t+1) = pi_j^(t) + (w_j - l_j) / N_eff
       where N_eff is the effective population size (hyperparameter)
    5. Clamp: pi_j in [1/N_eff, 1 - 1/N_eff]
```

For $p = 50{,}000$ and $N_{\text{eff}} = 100$: memory = 400KB, vs 4GB for explicit population.

---

## 7. Estimation of Distribution Algorithms for p > 10,000

### 7.1 PBIL: Population-Based Incremental Learning

PBIL (Baluja, 1994) extends cGA to learn structure in the probability vector:

$$\boldsymbol{\pi}^{(t+1)} = (1 - \alpha) \boldsymbol{\pi}^{(t)} + \alpha \cdot \text{best\_solution}^{(t)}$$

where $\alpha$ is the learning rate and $\text{best\_solution}^{(t)}$ is the best feature vector found in generation $t$. The update moves the probability vector toward successful solutions.

### 7.2 UMDA: Univariate Marginal Distribution Algorithm

UMDA (Mühlenbein & Paaß, 1996) estimates the probability vector from the top-$\tau$ fraction of solutions in each generation:

$$\pi_j^{(t+1)} = \frac{1}{|\mathcal{T}^{(t)}|} \sum_{s \in \mathcal{T}^{(t)}} s_j$$

where $\mathcal{T}^{(t)}$ is the set of top-$\tau$ solutions. This is effectively a rolling average of successful feature selections.

For $p > 10{,}000$, UMDA is particularly effective because:
1. Maintains only $O(p)$ memory (the probability vector)
2. Each generation evaluates only $N$ solutions ($N \ll p$ typically 50–200)
3. Converges in $O(p \log p)$ evaluations for separable problems

---

## 8. Feature Selection with Missing Data

### 8.1 The Missing Data Problem

When $\mathbf{X}$ has missing entries (MCAR, MAR, or MNAR), complete-case analysis discards observations with any missing feature — potentially reducing $n$ dramatically for large $p$.

### 8.2 Multiple Imputation + Stability Selection

**Algorithm:**

```
For m = 1, ..., M imputation rounds:
    1. Impute X_m using mice or missForest
    2. Run stability selection on (X_m, y):
       For b = 1, ..., B subsamples:
           Run Lasso/SIS on subsample -> selected_mb
       stability_score_m[j] = mean(selected_mb[j])
    3. Aggregate: stability_score[j] = mean over m of stability_score_m[j]

Final selection: j if stability_score[j] > threshold
```

**Key property:** Features whose stability score is high across imputation rounds are robust to the missing data mechanism — they are selected regardless of how missing values are imputed.

### 8.3 Implementation

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LassoCV
import numpy as np

def mi_stability_selection(X, y, M=10, B=50, threshold=0.7):
    n, p = X.shape
    stability_scores = np.zeros(p)

    for m in range(M):
        # Multiple imputation (MICE)
        imputer = IterativeImputer(max_iter=10, random_state=m)
        X_imp = imputer.fit_transform(X)

        # Stability selection on imputed data
        sub_scores = np.zeros(p)
        for b in range(B):
            subsample = np.random.choice(n, n // 2, replace=False)
            lasso = LassoCV(cv=3).fit(X_imp[subsample], y[subsample])
            sub_scores += (lasso.coef_ != 0)

        stability_scores += sub_scores / B

    stability_scores /= M
    return np.where(stability_scores > threshold)[0]
```

---

## Common Pitfalls

- **Reporting p-values from OLS on Lasso-selected features**: The single most common error in applied high-dimensional work. These p-values are anti-conservative by factors of 2–10.
- **Forgetting that the debiased Lasso requires the nodewise regressions**: The correction term $\hat{\mathbf{\Theta}}$ requires $p$ auxiliary Lasso regressions, which must be run. Using $(\mathbf{X}^\top \mathbf{X}/n)^{-1}$ directly fails when $p > n$.
- **Applying selective inference to the debiased Lasso**: These are two different frameworks. Debiased Lasso provides marginal CIs valid asymptotically. Selective inference provides exact conditional CIs. They should not be combined.
- **Treating Knockoff FDR control as FWER control**: Knockoffs control FDR (expected proportion of false discoveries), not FWER (probability of any false discovery). FDR-controlled procedures will include false positives — this is by design.

---

## The Complete Inference Pipeline

```
Ultra-high dimensional data (p >> n)
         │
    Screen (SIS/ISIS)
         │ d_n ~ n/log(n)
         ▼
    Select (Lasso/Group Lasso)
         │ selected set M_hat
         ▼
    Infer (choose one):
         ├── Debiased Lasso CIs (large n, many features)
         ├── Selective inference (exact, conditional)
         ├── Data splitting (simplest, half power)
         └── Knockoff+ (FDR control, need knockoffs)
         │
         ▼
    Report: coefficients + valid CIs + FDR-controlled p-values
```

---

## Connections

- **Builds on:** Lasso (Module 4), SIS screening (Guide 01), stability selection (Module 10)
- **Leads to:** Causal feature selection (Module 09), production deployment (Module 11)
- **Related to:** Multiple testing correction (Bonferroni, BH), bootstrap inference, conformal prediction

---

## Further Reading

- **Lee, J.D., Sun, D.L., Sun, Y. & Taylor, J.E. (2016).** "Exact post-selection inference, with application to the Lasso." *Annals of Statistics*, 44(3), 907–927. — The selective inference paper with the polyhedral lemma.
- **van de Geer, S., Bühlmann, P., Ritov, Y. & Dezeure, R. (2014).** "On asymptotically optimal confidence regions and tests for high-dimensional models." *Annals of Statistics*, 42(3), 1166–1202. — Debiased Lasso construction and theory.
- **Zhang, C.-H. & Zhang, S.S. (2014).** "Confidence intervals for low dimensional parameters in high dimensional linear models." *Journal of the Royal Statistical Society: Series B*, 76(1), 217–242. — Independent derivation of the debiased Lasso.
- **Barber, R.F. & Candès, E.J. (2015).** "Controlling the false discovery rate via knockoffs." *Annals of Statistics*, 43(5), 2055–2085. — Fixed-X knockoffs with finite-sample FDR guarantee.
- **Candès, E., Fan, Y., Janson, L. & Lv, J. (2018).** "Panning for gold: model-X knockoffs for high dimensional controlled variable selection." *Journal of the Royal Statistical Society: Series B*, 80(3), 551–577. — Model-X knockoffs for $p > n$.
- **Meinshausen, N., Meier, L. & Bühlmann, P. (2009).** "P-values for high-dimensional regression." *Journal of the American Statistical Association*, 104(488), 1671–1681. — Multi-split aggregation.
