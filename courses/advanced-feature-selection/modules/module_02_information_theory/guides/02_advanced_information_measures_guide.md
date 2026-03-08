# Advanced Information Measures for Feature Selection

## In Brief

Standard mutual information assumes well-behaved distributions and symmetric, undirected dependence. Financial and commodity time series violate these assumptions through heavy tails, non-linear tail dependencies, and causal asymmetry. This guide covers four advanced information measures — Rényi entropy, transfer entropy, copula-based MI, and interaction information — and gives precise guidance on when each is needed.

## Key Insight

Standard MI is the right tool when your data looks like a Gaussian mixture. When your data has fat tails, directional causality, or higher-order feature interactions, standard MI will mislead you. The advanced measures in this guide address each failure mode specifically.

---

## 1. Rényi Entropy and $\alpha$-Mutual Information

### Motivation

Shannon entropy $H(X) = -\mathbb{E}[\log p(X)]$ is the expected log-probability. It weights all probability mass equally. For heavy-tailed distributions — common in commodity prices, energy returns, and volatility series — rare but large events dominate the tail. Shannon entropy underweights these events relative to their practical importance.

**Rényi entropy** introduces a parameter $\alpha$ that controls how much weight tail events receive:

$$H_\alpha(X) = \frac{1}{1-\alpha} \log \sum_x p(x)^\alpha, \quad \alpha > 0, \, \alpha \neq 1$$

### Properties of $\alpha$

- $\alpha \to 1$: recovers Shannon entropy $H_1(X) = -\sum_x p(x) \log p(x)$
- $\alpha = 0$: $H_0(X) = \log |\text{supp}(X)|$ — log of the support size, ignores probabilities entirely
- $\alpha = 2$: $H_2(X) = -\log \sum_x p(x)^2$ — the **collision entropy**, related to the probability that two independent draws are equal
- $\alpha \to \infty$: $H_\infty(X) = -\log \max_x p(x)$ — min-entropy, focuses only on the most probable event

**For heavy-tailed data:** $\alpha < 1$ upweights rare events; $\alpha > 1$ downweights them. Using $\alpha \in (0, 1)$ makes the entropy more sensitive to tail events.

### $\alpha$-Mutual Information

Define the Rényi joint entropy:

$$H_\alpha(X, Y) = \frac{1}{1-\alpha} \log \sum_{x,y} p(x,y)^\alpha$$

The **$\alpha$-mutual information** (Verdu, 2015; Sibson, 1969):

$$I_\alpha(X; Y) = \min_{q(y)} D_\alpha(p(x,y) \| p(x) q(y))$$

where $D_\alpha$ is the Rényi divergence:

$$D_\alpha(p \| q) = \frac{1}{\alpha-1} \log \sum_x p(x)^\alpha q(x)^{1-\alpha}$$

For discrete distributions, a simpler empirical formula:

$$I_\alpha^\text{emp}(X; Y) = H_\alpha(X) + H_\alpha(Y) - H_\alpha(X, Y)$$

Note: this additive decomposition is exact for $\alpha = 1$ (Shannon case) and approximate for $\alpha \neq 1$.

### Why $\alpha$-MI for Financial Data

Consider WTI crude oil daily returns. The Student-$t$ distribution with $\nu = 3$ degrees of freedom has:
- Variance finite but excess kurtosis $= 6/(\nu - 4)$ — undefined when $\nu \leq 4$
- Shannon MI based on Gaussian kernel density estimates is biased toward the centre of the distribution
- $\alpha$-MI with $\alpha = 0.5$ amplifies tail events, capturing the dependence structure during market stress

**Practical guidance for $\alpha$ selection:**

| Tail behaviour | Recommended $\alpha$ | Rationale |
|----------------|---------------------|-----------|
| Gaussian ($\kappa = 0$) | $\alpha = 1.0$ | Shannon MI is correct |
| Moderate tails ($\kappa \in [1,5]$) | $\alpha = 0.75$ | Slight tail amplification |
| Heavy tails ($\kappa > 5$) | $\alpha = 0.5$ | Strong tail amplification |
| Extreme tails / crisis data | $\alpha = 0.25$ | Focus on extreme events |

### Implementation

```python
import numpy as np
from scipy.stats import entropy as scipy_entropy

def renyi_entropy(probs, alpha):
    """
    Compute Rényi entropy of order alpha from probability vector.

    Parameters
    ----------
    probs : array
        Probability mass function (must sum to 1)
    alpha : float
        Rényi order. alpha=1 recovers Shannon entropy.

    Returns
    -------
    float : Rényi entropy H_alpha
    """
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]  # Remove zeros to avoid log(0)

    if np.isclose(alpha, 1.0):
        return -np.sum(probs * np.log(probs))  # Shannon

    return (1.0 / (1.0 - alpha)) * np.log(np.sum(probs ** alpha))

def renyi_mi(x_disc, y_disc, alpha):
    """
    Compute alpha-mutual information using the additive approximation.

    Parameters
    ----------
    x_disc : array of int
        Discretised feature values
    y_disc : array of int
        Discretised target values
    alpha : float
        Rényi order

    Returns
    -------
    float : alpha-MI (approximate for alpha != 1)
    """
    n = len(x_disc)
    # Marginal distributions
    _, px = np.unique(x_disc, return_counts=True)
    _, py = np.unique(y_disc, return_counts=True)
    px = px / n
    py = py / n

    # Joint distribution
    # Encode joint as a single integer
    xy = x_disc * (np.max(y_disc) + 1) + y_disc
    _, pxy = np.unique(xy, return_counts=True)
    pxy = pxy / n

    hx = renyi_entropy(px, alpha)
    hy = renyi_entropy(py, alpha)
    hxy = renyi_entropy(pxy, alpha)

    return hx + hy - hxy
```

---

## 2. Transfer Entropy

### Motivation

Mutual information measures **symmetric** statistical dependence: $I(X; Y) = I(Y; X)$. For time series feature selection, we often need to know whether feature $x$ **drives** target $y$ (i.e., whether past values of $x$ predict future values of $y$ over and above $y$'s own past). This is the directed information flow question.

### Definition

**Transfer entropy** from $X$ to $Y$ (Schreiber, 2000) measures the reduction in uncertainty about the next state of $Y$ when we know the past of $X$, beyond what $Y$'s own past already tells us:

$$T_{X \to Y} = I(Y_{t+1}; X_{t-\ell:t} \mid Y_{t-k:t})$$

$$= H(Y_{t+1} \mid Y_{t-k:t}) - H(Y_{t+1} \mid Y_{t-k:t}, X_{t-\ell:t})$$

where:
- $Y_{t-k:t} = (Y_t, Y_{t-1}, \ldots, Y_{t-k})$ — the $k$-step history of $Y$
- $X_{t-\ell:t} = (X_t, X_{t-1}, \ldots, X_{t-\ell})$ — the $\ell$-step history of $X$

### Relationship to Granger Causality

For **linear Gaussian systems**, transfer entropy is proportional to the Granger causality $F_{X \to Y}$:

$$T_{X \to Y} = \frac{1}{2} \log \frac{\text{Var}(Y_{t+1} | Y_{t-k:t})}{\text{Var}(Y_{t+1} | Y_{t-k:t}, X_{t-\ell:t})}$$

Transfer entropy generalises Granger causality to arbitrary distributions and non-linear dynamics. It detects non-linear predictive relationships that Granger causality (which tests linear predictability) misses entirely.

### Directed Information Flow Network

For a set of $p$ features $\{x_1, \ldots, x_p\}$ and target $y$, compute all $p$ transfer entropies $T_{x_i \to y}$ and rank features accordingly. You can also compute the full $p \times p$ matrix of pairwise TEs to understand the causal structure of the feature space.

### Significance Testing

Transfer entropy values must be tested against a null hypothesis of no directed dependence. The standard approach is **time-shifted surrogates**:
1. Randomly shuffle the time series of $X$ (destroying temporal structure but preserving marginal distribution)
2. Compute $T_{X_\text{shuffled} \to Y}$
3. Repeat 200 times to build a null distribution
4. The empirical $T_{X \to Y}$ is significant if it exceeds the 95th percentile of the null

### Implementation

```python
import numpy as np
from sklearn.metrics import mutual_info_score

def embed_time_series(ts, lag):
    """
    Create time-delayed embedding of a time series.

    Parameters
    ----------
    ts : array of shape (T,)
        1D time series
    lag : int
        Number of lag steps

    Returns
    -------
    array of shape (T - lag, lag)
        Each row is [ts[t], ts[t-1], ..., ts[t-lag+1]]
    """
    T = len(ts)
    embedded = np.column_stack([ts[i:T - lag + i] for i in range(lag, 0, -1)])
    return embedded  # shape (T - lag, lag)

def discretise_series(ts, n_bins=10):
    """Quantile-based discretisation of a 1D time series."""
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(ts, quantiles)
    bin_edges[0] -= 1e-10
    bin_edges[-1] += 1e-10
    return np.digitize(ts, bin_edges) - 1

def encode_joint(arrays):
    """
    Encode multiple arrays as a single integer-valued joint variable.

    Parameters
    ----------
    arrays : list of 1D int arrays
        Each array has the same length

    Returns
    -------
    1D array of int
    """
    result = np.zeros(len(arrays[0]), dtype=int)
    multiplier = 1
    for arr in reversed(arrays):
        result += arr * multiplier
        multiplier *= (np.max(arr) + 1)
    return result

def transfer_entropy(x, y, k=1, ell=1, n_bins=10):
    """
    Compute transfer entropy T_{X -> Y}.

    T_{X->Y} = I(Y_{t+1}; X_{t-ell:t} | Y_{t-k:t})
             = H(Y_{t+1} | Y_{t-k:t}) - H(Y_{t+1} | Y_{t-k:t}, X_{t-ell:t})

    Parameters
    ----------
    x : array of shape (T,)
        Source time series (candidate feature)
    y : array of shape (T,)
        Target time series
    k : int
        History length for Y (Markov order of Y)
    ell : int
        History length for X
    n_bins : int
        Number of bins for discretisation

    Returns
    -------
    float : Transfer entropy T_{X->Y} in nats
    """
    x_d = discretise_series(x, n_bins)
    y_d = discretise_series(y, n_bins)

    max_lag = max(k, ell)
    T = len(y_d) - max_lag

    # Y_{t+1}
    y_future = y_d[max_lag:]

    # Y history: k lags
    y_hist_arrays = [y_d[max_lag - lag: -lag if lag > 0 else None]
                     for lag in range(1, k + 1)]
    y_hist = encode_joint(y_hist_arrays)

    # X history: ell lags
    x_hist_arrays = [x_d[max_lag - lag: -lag if lag > 0 else None]
                     for lag in range(1, ell + 1)]
    x_hist = encode_joint(x_hist_arrays)

    # T_{X->Y} = I(Y_{t+1}; X_hist | Y_hist)
    # = I(Y_{t+1}; X_hist, Y_hist) - I(Y_{t+1}; Y_hist)
    xy_hist = encode_joint([x_hist, y_hist])

    cmi = (mutual_info_score(y_future, xy_hist) -
           mutual_info_score(y_future, y_hist))

    return max(0.0, cmi)  # TE >= 0 by definition

def te_significance(x, y, k=1, ell=1, n_bins=10, n_surrogates=200, alpha=0.05):
    """
    Test transfer entropy significance using time-shifted surrogates.

    Returns
    -------
    te : float
        Observed transfer entropy
    p_value : float
        Fraction of surrogates exceeding observed TE
    is_significant : bool
        True if p_value < alpha
    """
    te_obs = transfer_entropy(x, y, k=k, ell=ell, n_bins=n_bins)

    surrogate_tes = []
    rng = np.random.default_rng(42)
    for _ in range(n_surrogates):
        x_shuffled = rng.permutation(x)
        surrogate_tes.append(transfer_entropy(x_shuffled, y, k=k, ell=ell, n_bins=n_bins))

    p_value = np.mean(np.array(surrogate_tes) >= te_obs)
    return te_obs, p_value, p_value < alpha
```

---

## 3. Copula-Based Dependence Measures

### Motivation

Standard MI computed from kernel density estimates or histograms conflates two distinct aspects of dependence:
1. **Marginal distributions** — the shape of each feature's own distribution
2. **Dependence structure** — how features co-move, especially in the tails

For feature selection in financial data, we usually care only about the dependence structure. If two features both have fat-tailed returns, their high MI may be driven by similar marginals rather than genuine co-movement.

**Copulas** separate these concerns via Sklar's theorem: for any joint CDF $F(x, y)$ with marginals $F_X$, $F_Y$, there exists a unique copula $C$ such that:

$$F(x, y) = C(F_X(x), F_Y(y))$$

The copula $C: [0,1]^2 \to [0,1]$ captures the dependence structure independently of the marginals.

### Copula-Based MI

Define the **copula mutual information** (CMI in the copula sense, not conditional MI):

$$I_C(X; Y) = \int_0^1 \int_0^1 c(u, v) \log c(u, v) \, du \, dv$$

where $c(u,v) = \partial^2 C(u,v) / \partial u \partial v$ is the copula density and $u = F_X(x)$, $v = F_Y(y)$ are the probability-integral transforms (PIT).

**Practical computation:**
1. Transform each feature to uniform marginals: $u_i = \hat{F}_{x_i}(x_i)$ using the empirical CDF
2. Discretise $u_i$ to $[0,1]^d$ on a uniform grid
3. Compute MI on the discretised uniform marginals

This removes the effect of marginal distributions and focuses entirely on the dependence structure.

### Tail Dependence Coefficients

For financial applications, the **upper tail dependence coefficient** is often more informative than full copula MI:

$$\lambda_U = \lim_{u \to 1} P(Y > F_Y^{-1}(u) \mid X > F_X^{-1}(u)) = \lim_{u \to 1} \frac{1 - 2u + C(u,u)}{1 - u}$$

$\lambda_U > 0$ indicates tail dependence — features that co-move during extremes (crises, spikes). This is especially important for risk management: a feature that only helps predict normal returns but is independent during crashes is less valuable than one that has tail dependence with the target.

### Implementation

```python
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import mutual_info_score

def probability_integral_transform(x):
    """
    Transform x to uniform [0,1] marginals using empirical CDF.

    Uses rank-based transformation: u_i = rank(x_i) / (n+1)
    """
    ranks = rankdata(x)
    n = len(x)
    return ranks / (n + 1)  # Uniform on (0,1) — avoids 0 and 1

def copula_mi(x, y, n_bins=20):
    """
    Compute copula mutual information.

    Transforms both series to uniform marginals (removes marginal effects),
    then computes MI on the copula domain.

    Parameters
    ----------
    x : array
        Feature time series
    y : array
        Target time series
    n_bins : int
        Bins for discretising the [0,1] copula domain

    Returns
    -------
    float : Copula MI in nats
    """
    u = probability_integral_transform(x)
    v = probability_integral_transform(y)

    # Discretise uniform marginals
    bin_edges = np.linspace(0, 1 + 1e-10, n_bins + 1)
    u_disc = np.digitize(u, bin_edges) - 1
    v_disc = np.digitize(v, bin_edges) - 1

    return mutual_info_score(u_disc, v_disc)

def upper_tail_dependence(x, y, q=0.95):
    """
    Estimate upper tail dependence coefficient lambda_U.

    Estimates: P(Y > Q(q) | X > Q(q)) where Q(q) is the q-th quantile.

    Parameters
    ----------
    x : array
        Feature time series
    y : array
        Target time series
    q : float
        Quantile threshold (0 < q < 1), typically 0.90--0.99

    Returns
    -------
    float : Empirical tail dependence coefficient [0, 1]
    """
    x_thresh = np.quantile(x, q)
    y_thresh = np.quantile(y, q)

    both_above = np.sum((x > x_thresh) & (y > y_thresh))
    x_above = np.sum(x > x_thresh)

    if x_above == 0:
        return 0.0
    return both_above / x_above

def copula_feature_ranking(X, y, n_bins=20, q_tail=0.95):
    """
    Rank features by copula MI and tail dependence.

    Parameters
    ----------
    X : array of shape (n, p)
        Feature matrix
    y : array of shape (n,)
        Target variable
    n_bins : int
        Bins for copula MI
    q_tail : float
        Quantile for tail dependence

    Returns
    -------
    dict with keys 'copula_mi', 'tail_dep', 'combined_rank'
    """
    p = X.shape[1]
    cmi_scores = np.zeros(p)
    tail_scores = np.zeros(p)

    for i in range(p):
        cmi_scores[i] = copula_mi(X[:, i], y, n_bins=n_bins)
        tail_scores[i] = upper_tail_dependence(X[:, i], y, q=q_tail)

    # Combined rank (equal weight)
    from scipy.stats import rankdata
    rank_cmi = rankdata(cmi_scores)
    rank_tail = rankdata(tail_scores)
    combined = rank_cmi + rank_tail

    return {
        'copula_mi': cmi_scores,
        'tail_dep': tail_scores,
        'combined_rank': combined
    }
```

---

## 4. Interaction Information and the Information Lattice

### Motivation

Standard pairwise MI $I(x_k; y)$ measures the dependence of a single feature on the target. Many financial phenomena involve **higher-order interactions**: the target $y$ depends on a specific combination of features, but not on any individual feature. The **interaction information** (also called co-information) quantifies this.

### Definition for Three Variables

The interaction information (McGill, 1954) for three variables $X$, $Y$, $Z$:

$$\text{Int}(X; Y; Z) = I(X; Y) - I(X; Y | Z)$$

Alternative definition via inclusion-exclusion on the information lattice:

$$\text{Int}(X; Y; Z) = I(X; Y) + I(Y; Z) + I(X; Z) - I(X; Y, Z) - I(X; Z, Y) - I(Y; Z, X) + I(X; Y, Z)$$

This simplifies to:

$$\text{Int}(X; Y; Z) = I(X; Y | Z) - I(X; Y)$$

Wait — note the sign. The sign convention varies in the literature. Brown et al. (2012) use $\text{Int}(x_k; x_j; y) = I(x_k; x_j) - I(x_k; x_j | y)$, which is positive for redundancy and negative for synergy.

**Redundant features:** $\text{Int} > 0$ — features $x_k$ and $x_j$ share information about $y$ in the same way. Knowing $y$ reduces their mutual information.

**Synergistic features:** $\text{Int} < 0$ — features $x_k$ and $x_j$ are more informative together than separately. Knowing $y$ increases their mutual information (e.g., XOR patterns).

### The Information Lattice

For $d$ variables, the **information lattice** (also called the partial information lattice) decomposes the total MI $I(\mathbf{x}_S; y)$ into:
- **Unique information**: information that each variable provides exclusively
- **Redundant information**: information shared by multiple variables
- **Synergistic information**: information that only emerges from the combination

For $d = 2$ features $x_1$, $x_2$ and target $y$:

$$I(x_1, x_2; y) = \underbrace{\text{Uniq}_1}_{\text{unique to } x_1} + \underbrace{\text{Uniq}_2}_{\text{unique to } x_2} + \underbrace{\text{Red}_{12}}_{\text{shared}} + \underbrace{\text{Syn}_{12}}_{\text{synergy}}$$

The **partial information decomposition** (PID) of Williams and Beer (2010) provides a framework for computing all four terms.

### Practical Use in Feature Selection

Interaction information can directly guide feature selection:

1. **Screen for synergistic pairs**: Compute $\text{Int}(x_k; x_j; y)$ for all pairs. Pairs with large negative interaction information are synergistic — neither should be excluded even if their individual MI is low.

2. **Screen for redundant pairs**: Pairs with large positive interaction information are redundant — you only need one.

3. **Second-order ITFS**: Instead of greedy selection based on pairwise MI, explicitly optimise for subsets with high synergy and low redundancy.

```python
def interaction_information(x_k, x_j, y):
    """
    Compute interaction information Int(x_k; x_j; y).

    Convention (Brown et al.): Int = I(x_k; x_j) - I(x_k; x_j | y)
    Positive = redundancy, Negative = synergy.

    Parameters
    ----------
    x_k, x_j : array of int
        Discretised feature values
    y : array of int
        Discretised target values

    Returns
    -------
    float : Interaction information
    """
    from sklearn.metrics import mutual_info_score

    mi_xkxj = mutual_info_score(x_k, x_j)

    # I(x_k; x_j | y) via CMI
    # Encode (x_j, y) as joint variable
    n_y = len(np.unique(y))
    xj_y = x_j * n_y + y
    mi_xkxjy = mutual_info_score(x_k, xj_y) - mutual_info_score(x_k, y)

    return mi_xkxj - max(0.0, mi_xkxjy)
```

---

## 5. Partial Information Decomposition (PID)

### Motivation

PID (Williams and Beer, 2010) resolves an ambiguity in interaction information: the total synergy of a pair of features is a single signed number, but we often want to know **how much** redundancy and **how much** synergy exist simultaneously in a triple $(x_k, x_j, y)$.

### The Four PID Atoms

For two sources $x_1$, $x_2$ and a target $y$:

$$I(x_1, x_2; y) = \Pi_\text{red} + \Pi_\text{uniq1} + \Pi_\text{uniq2} + \Pi_\text{syn}$$

where:
- $\Pi_\text{red} \geq 0$: redundant information (shared by both sources)
- $\Pi_\text{uniq1} \geq 0$: information unique to $x_1$
- $\Pi_\text{uniq2} \geq 0$: information unique to $x_2$
- $\Pi_\text{syn} \geq 0$: synergistic information (only in combination)

**Relationship to interaction information:**

$$\text{Int}(x_1; x_2; y) = \Pi_\text{red} - \Pi_\text{syn}$$

PID separates redundancy and synergy into non-negative quantities, which interaction information conflates into a single signed number.

### Computing PID

The minimum mutual information approach (MMI) uses:

$$\Pi_\text{red} = \min(I(x_1; y), I(x_2; y))$$

This is the simplest operationalisation but underestimates redundancy in some cases. More sophisticated approaches (BROJA, SxPID) exist but require specialised libraries.

```python
def pid_mmi(x1, y, x2):
    """
    Partial information decomposition using the minimum MI approximation.

    Returns the four PID atoms: redundancy, unique_x1, unique_x2, synergy.
    Note: MMI underestimates redundancy in some cases.

    Parameters
    ----------
    x1, x2 : array of int
        Discretised feature values
    y : array of int
        Discretised target values

    Returns
    -------
    dict with keys: redundancy, unique_x1, unique_x2, synergy
    """
    from sklearn.metrics import mutual_info_score

    i1 = mutual_info_score(x1, y)
    i2 = mutual_info_score(x2, y)

    # Joint MI I(x1, x2; y)
    n_x2 = len(np.unique(x2))
    x1x2 = x1 * n_x2 + x2
    i12 = mutual_info_score(x1x2, y)

    # MMI approximation
    red = min(i1, i2)
    uniq1 = i1 - red
    uniq2 = i2 - red
    syn = i12 - i1 - i2 + red  # Can be negative in MMI — clip to 0

    return {
        'redundancy': max(0.0, red),
        'unique_x1': max(0.0, uniq1),
        'unique_x2': max(0.0, uniq2),
        'synergy': max(0.0, syn)
    }
```

---

## 6. When Standard MI Fails: A Decision Guide

Use this flowchart to decide which measure to use:

```
Is your data a time series?
├── YES → Do you need directional dependence (does x "cause" y)?
│         ├── YES → Use Transfer Entropy
│         └── NO → Are there heavy tails (kurtosis > 5)?
│                  ├── YES → Use Rényi MI (alpha < 1)
│                  └── NO → Standard MI is fine
└── NO → Are there heavy tails or tail-specific dependence?
          ├── YES → Use Copula MI + Tail Dependence Coefficient
          └── NO → Do you need higher-order interactions?
                   ├── YES → Use Interaction Information or PID
                   └── NO → Standard MI is fine
```

### Practical Thresholds for Financial Data

| Statistic | Threshold | Action |
|-----------|-----------|--------|
| Kurtosis $> 5$ | Heavy tails | Use Rényi MI with $\alpha = 0.5$–$0.75$ |
| Ljung-Box test significant | Autocorrelation in residuals | Use Transfer Entropy |
| Upper tail dep $> 0.3$ | Tail clustering | Use Copula MI as primary score |
| Interaction info $< -0.05$ | Strong synergy | Always include both features |
| Interaction info $> 0.3$ | Strong redundancy | Select only the better marginal feature |

---

## Common Pitfalls

**Pitfall 1: Using transfer entropy without controlling for autocorrelation.**
If $Y$ is autocorrelated, $H(Y_{t+1} | Y_{t-k:t})$ will be small regardless of $X$. Always subtract the conditional entropy given $Y$'s own history, not just $X$'s history alone.

**Pitfall 2: Conflating copula MI with standard MI.**
Copula MI measures only the dependence structure, not the total information shared. It is always $\leq$ standard MI. Do not use copula MI as a drop-in replacement for standard MI in the CLM criterion.

**Pitfall 3: Using PID MMI and interpreting synergy as definitive.**
MMI underestimates redundancy, which inflates apparent synergy. For confirmed synergy, cross-validate with at least two PID methods.

**Pitfall 4: Choosing $\alpha$ for Rényi MI without checking the kurtosis.**
Running multiple $\alpha$ values and selecting post-hoc introduces a model selection problem. Determine $\alpha$ from the kurtosis of the marginal distributions before running feature selection.

---

## Connections

**Builds on:**
- Guide 01: The CLM unified framework and standard MI
- Module 01: Shannon entropy and MI estimation

**Leads to:**
- Notebook 02: Transfer entropy feature selector for financial time series
- Module 07: Time series feature selection — autocorrelation and causality
- Module 09: Causal feature selection — interventional versus observational dependence

**Related to:**
- Granger causality: linear analogue of transfer entropy
- Copula theory: statistical framework for dependence modelling
- Information geometry: geometric view of information measures

---

## Further Reading

- **Schreiber, T. (2000).** "Measuring information transfer." *Physical Review Letters*, 85(2), 461–464. — Original transfer entropy paper. Concise and readable.

- **Rényi, A. (1961).** "On measures of entropy and information." *Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 547–561. — Original Rényi entropy paper.

- **Verdu, S. (2015).** "Alpha mutual information." *Proceedings of the ITW*, 1–5. — Modern treatment of $\alpha$-MI.

- **Williams, P. L. & Beer, R. D. (2010).** "Nonnegative Decomposition of Multivariate Information." *arXiv:1004.2515*. — Introduces PID.

- **Sklar, M. (1959).** "Fonctions de répartition à n dimensions et leurs marges." *Publications de l'Institut de Statistique de l'Université de Paris*, 8, 229–231. — Copula theorem.

- **Kraskov, A., Stögbauer, H., Grassberger, P. (2004).** "Estimating mutual information." *Physical Review E*, 69(6), 066138. — $k$-NN MI estimator for continuous variables. Essential for continuous-valued transfer entropy.
