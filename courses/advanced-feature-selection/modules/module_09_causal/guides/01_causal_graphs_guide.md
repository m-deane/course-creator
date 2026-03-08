# Causal Graphs and the Markov Blanket: The Optimal Feature Set

## In Brief

Causal graphs encode the data-generating mechanism as a directed acyclic graph (DAG). The Markov blanket of a target node — its parents, children, and co-parents — is provably the minimal sufficient feature set for prediction under the true causal model. Pearl's do-calculus and structural causal models provide the theoretical foundation; PC, FCI, and GES are the practical algorithms for recovering these structures from data.

## Key Insight

Every standard feature selection method asks: "which features correlate with the target?" Causal feature selection asks: "which features *causally matter* for the target?" The difference is not philosophical — it is operational. Causal features remain predictive under distribution shift. Spuriously correlated features fail as soon as the environment changes.

---

## 1. Pearl's Structural Causal Models

A structural causal model (SCM) is a tuple $\mathcal{M} = (\mathbf{U}, \mathbf{V}, \mathcal{F}, P_\mathbf{U})$ where:

- $\mathbf{U}$ is a set of exogenous (background) variables with distribution $P_\mathbf{U}$
- $\mathbf{V}$ is a set of endogenous (observed) variables
- $\mathcal{F} = \{f_V\}_{V \in \mathbf{V}}$ is a set of structural equations, one per endogenous variable:

$$V_i := f_{V_i}(\text{Pa}(V_i), U_{V_i})$$

where $\text{Pa}(V_i) \subseteq \mathbf{V} \setminus \{V_i\}$ are the *parents* of $V_i$ in the DAG, and $U_{V_i}$ is an exogenous noise term.

The DAG $\mathcal{G}$ encodes every structural equation: draw a directed edge $V_j \to V_i$ if $V_j \in \text{Pa}(V_i)$.

### The Do-Operator

Standard probability $P(Y \mid X = x)$ is observational: it conditions on seeing $X = x$ naturally. Pearl's do-operator $P(Y \mid do(X = x))$ is interventional: it models *forcing* $X$ to take value $x$ by removing all arrows into $X$ in the DAG.

For feature selection the distinction matters because:

$$P(Y \mid X = x) \neq P(Y \mid do(X = x))$$

when $X$ and $Y$ share a common cause (confounder). Observational feature selection picks up confounders; causal feature selection does not.

### Causal Markov Condition

An SCM implies that every variable $V_i$ is conditionally independent of its non-descendants given its parents:

$$V_i \perp \!\!\! \perp \text{NonDesc}(V_i) \mid \text{Pa}(V_i)$$

This is the *causal Markov condition*. It is the bridge between the DAG structure and the probability distribution.

---

## 2. Causal Markov Blankets: The Optimal Feature Set

### Definition

The **Markov blanket** $\text{MB}(Y)$ of a target node $Y$ in a DAG is the minimal set $S \subseteq \mathbf{V} \setminus \{Y\}$ such that:

$$Y \perp \!\!\! \perp \mathbf{V} \setminus (\{Y\} \cup S) \mid S$$

In a DAG, the Markov blanket consists of exactly three groups of nodes:

1. **Parents** $\text{Pa}(Y)$: direct causes of $Y$
2. **Children** $\text{Ch}(Y)$: direct effects of $Y$
3. **Co-parents** $\text{CoP}(Y)$: other parents of $Y$'s children (spouses)

```
        A          B               Parents of Y: {A, B}
        |          |               Children of Y: {C}
        +----Y-----+               Co-parents of Y: {D}
             |
             C ←── D              MB(Y) = {A, B, C, D}
```

### Why MB(Y) is the Optimal Feature Set

**Theorem (Pearl, 2009).** Given the causal Markov condition and faithfulness, the Markov blanket $\text{MB}(Y)$ is the unique minimal sufficient statistic for predicting $Y$:

$$Y \perp \!\!\! \perp \mathbf{V} \setminus (\{Y\} \cup \text{MB}(Y)) \mid \text{MB}(Y)$$

This means:
- Adding any feature outside $\text{MB}(Y)$ provides zero additional information about $Y$
- Removing any feature from $\text{MB}(Y)$ loses information about $Y$
- $\text{MB}(Y)$ is both necessary and sufficient

The optimality holds under the *true* causal model. In practice, estimated Markov blankets from finite data are approximations, but the theoretical guarantee motivates the approach.

### Practical Implications

| Feature Type | In MB? | Reason |
|---|---|---|
| Direct cause of $Y$ | Yes (parent) | Structurally affects $Y$ |
| Direct effect of $Y$ | Yes (child) | Contains information about $Y$'s state |
| Cause of $Y$'s child | Yes (co-parent) | Needed to explain child given $Y$ |
| Cause of a cause of $Y$ | No | Screened off by parent |
| Spurious correlate | No | No causal connection |
| Common effect of $Y$ and noise | No | Collider — conditioning introduces bias |

---

## 3. The PC Algorithm: Constraint-Based Causal Discovery

The PC algorithm (Spirtes, Glymour, Scheines, 1993) recovers the Markov equivalence class of the causal DAG using only conditional independence tests. It runs in two phases.

### Phase 1: Skeleton Discovery

Start with a complete undirected graph over all variables. Remove edges using the following rule:

For each pair $(X, Y)$ and increasing conditioning set size $k = 0, 1, 2, \ldots$:

1. Find any set $S \subseteq \text{Adj}(X) \setminus \{Y\}$ with $|S| = k$
2. Test: $X \perp \!\!\! \perp Y \mid S$
3. If independent: remove edge $X - Y$, record $\text{Sep}(X, Y) = S$
4. Stop increasing $k$ when $|\text{Adj}(X)| - 1 < k$ for all edges

The result is an undirected skeleton plus a separation set for each missing edge.

### Phase 2: Orientation (V-Structures)

For each unshielded triple $X - Z - Y$ (where $X - Y$ is absent):

- If $Z \notin \text{Sep}(X, Y)$: orient as $X \to Z \leftarrow Y$ (v-structure / collider)
- Otherwise: leave unoriented

Apply Meek's rules to propagate additional orientations while avoiding cycles and new v-structures.

### Conditional Independence Tests

For continuous Gaussian data, the partial correlation test is standard:

$$H_0: X \perp \!\!\! \perp Y \mid S \iff \rho_{XY \cdot S} = 0$$

Test statistic: $z = \frac{1}{2} \ln \frac{1 + \hat{\rho}_{XY \cdot S}}{1 - \hat{\rho}_{XY \cdot S}} \sim \mathcal{N}(0, (n - |S| - 3)^{-1})$

For discrete or nonlinear data, alternatives include:
- $G^2$ test (chi-squared) for discrete variables
- Kernel-based tests (HSIC, KCI) for nonlinear dependencies
- Permutation tests for distribution-free testing

### PC Algorithm Complexity

- Worst case: $O(p^k)$ CI tests where $k$ is the maximum degree in the graph
- Sparse graphs (small $k$): polynomial in $p$
- Key assumption: *causal sufficiency* (no latent confounders)

```python
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

# PC algorithm with Fisher-z conditional independence test
cg = pc(
    data=X_array,           # numpy array, shape (n_samples, n_features)
    alpha=0.05,             # significance level for CI tests
    indep_test=fisherz,     # Fisher z-transform test for Gaussian data
    stable=True,            # stable PC variant (consistent ordering)
    uc_rule=0,              # v-structure orientation rule
    uc_priority=-1          # priority for existing edges
)
```

---

## 4. FCI: Handling Latent Confounders

The PC algorithm assumes *causal sufficiency*: all common causes are observed. When latent confounders exist, PC produces spurious edges. The **Fast Causal Inference (FCI)** algorithm relaxes this assumption.

### FCI's Key Additions

FCI uses a richer representation: a **partial ancestral graph (PAG)** with three edge marks:

- `→` arrowhead at a node (the node is not an ancestor of the other)
- `—` tail at a node (the node is an ancestor of the other)
- `o` circle at a node (unknown: could be either)

Edge types in the output PAG:

| Edge | Meaning |
|---|---|
| $X \to Y$ | $X$ is a cause of $Y$, no hidden common cause |
| $X \leftrightarrow Y$ | Hidden common cause exists between $X$ and $Y$ |
| $X \circ\!\!\to Y$ | $X$ might be a cause of $Y$ or a common cause |
| $X \circ\!\!-\!\!\circ Y$ | Relationship undetermined |

### FCI Algorithm Steps

1. Run PC skeleton discovery (same as PC Phase 1)
2. Add *possible-D-sep* orientations (accounts for paths through latent variables)
3. Re-run independence tests conditioning on larger sets
4. Apply FCI orientation rules (10 rules vs PC's 4)

### FCI for Feature Selection

From a PAG, the Markov blanket analog is the **Markov boundary in the presence of latents**:

- Features with $X \to Y$ edges: likely direct causes
- Features with $X \leftrightarrow Y$ edges: may share latent confounder — include for prediction, but cannot be leveraged for intervention

```python
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz

G, edges = fci(
    dataset=X_array,
    independence_test_method=fisherz,
    alpha=0.05,
    verbose=False
)
```

---

## 5. GES: Score-Based Causal Discovery

**Greedy Equivalence Search (GES)** takes a different approach: instead of testing independence, it optimises a score function over the space of DAGs via a greedy search over Markov equivalence classes (CPDAGs).

### GES Score Function

For Gaussian data, the BIC score is:

$$\text{BIC}(\mathcal{G}, \mathcal{D}) = -2 \ln \hat{L}(\mathcal{G}, \mathcal{D}) + k \ln n$$

where $\hat{L}$ is the maximum likelihood, $k$ is the number of free parameters, and $n$ is the sample size. BIC penalises complexity, encoding Occam's razor.

### GES Three Phases

**Phase 1: Forward (Insert)**
- Start from empty graph (no edges)
- Greedily add edges that maximally increase the score
- Stop when no addition improves score

**Phase 2: Backward (Delete)**
- Greedily remove edges that increase score
- Handles overfitting from Phase 1
- Stop when no deletion improves score

**Phase 3: Turning (optional)**
- Orient additional edges by turning undirected edges

### GES vs PC

| Aspect | PC | GES |
|---|---|---|
| Approach | Constraint-based | Score-based |
| Assumption | CI tests valid | Gaussian / parametric |
| Sample efficiency | Lower | Higher (uses all data) |
| Handling of latents | No (use FCI) | No (use RFCI) |
| Output | CPDAG | CPDAG |
| Consistency | Yes (faithful) | Yes (faithful) |

```python
from causallearn.search.ScoreBased.GES import ges

record = ges(
    X=X_array,
    score_func='local_score_BIC',  # BIC score for Gaussian data
    maxP=None,                      # max parents per node (None = unlimited)
    parameters=None
)
cpdag = record['G']
```

---

## 6. From Causal Graph to Feature Set

### Reading Off the Markov Blanket

Given a discovered CPDAG or PAG and a target variable $Y$:

1. **Identify parents of $Y$**: nodes with directed edges $X \to Y$
2. **Identify children of $Y$**: nodes with directed edges $Y \to Z$
3. **Identify co-parents**: for each child $Z$, find all other nodes $W$ with edge $W \to Z$
4. **Union**: $\text{MB}(Y) = \text{Pa}(Y) \cup \text{Ch}(Y) \cup \text{CoP}(Y)$

For undirected edges (ambiguous orientation in CPDAG):
- Include the adjacent node conservatively (false negative is worse than false positive for feature selection)

### Practical Feature Extraction

```python
import networkx as nx

def extract_markov_blanket(cpdag_adjacency: np.ndarray,
                           target_idx: int,
                           feature_names: list) -> list:
    """
    Extract Markov blanket features from a discovered CPDAG.

    Parameters
    ----------
    cpdag_adjacency : np.ndarray, shape (p, p)
        Adjacency matrix where cpdag[i,j]=1 means i -> j.
        cpdag[i,j]=1 and cpdag[j,i]=1 means undirected edge i -- j.
    target_idx : int
        Index of the target variable Y.
    feature_names : list
        Names of all variables.

    Returns
    -------
    mb_features : list
        Names of Markov blanket features.
    """
    p = len(feature_names)
    parents = []
    children = []
    co_parents = []

    for i in range(p):
        if i == target_idx:
            continue
        # Parent: i -> target (directed)
        if cpdag_adjacency[i, target_idx] == 1 and cpdag_adjacency[target_idx, i] == 0:
            parents.append(i)
        # Child: target -> i (directed)
        if cpdag_adjacency[target_idx, i] == 1 and cpdag_adjacency[i, target_idx] == 0:
            children.append(i)
        # Undirected: include conservatively
        if cpdag_adjacency[i, target_idx] == 1 and cpdag_adjacency[target_idx, i] == 1:
            parents.append(i)  # treat as possible parent

    # Co-parents: other parents of each child
    for child in children:
        for j in range(p):
            if j == target_idx or j == child:
                continue
            if cpdag_adjacency[j, child] == 1 and cpdag_adjacency[child, j] == 0:
                co_parents.append(j)

    mb_indices = list(set(parents + children + co_parents))
    return [feature_names[i] for i in mb_indices]
```

### Feature Set Comparison Framework

After recovering the Markov blanket, compare with other selection methods:

```python
def compare_feature_sets(mb_features, lasso_features, boruta_features, shap_features):
    """Compute overlap statistics between feature selection methods."""
    all_sets = {
        'Markov Blanket (PC)': set(mb_features),
        'Lasso': set(lasso_features),
        'Boruta': set(boruta_features),
        'SHAP': set(shap_features),
    }
    results = {}
    for name_a, set_a in all_sets.items():
        for name_b, set_b in all_sets.items():
            if name_a >= name_b:
                continue
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            jaccard = intersection / union if union > 0 else 0
            results[(name_a, name_b)] = {
                'intersection': intersection,
                'jaccard': jaccard,
            }
    return results
```

---

## Common Pitfalls

- **Faithfulness violations**: Real data may violate faithfulness (e.g., cancelling paths). PC and GES assume faithfulness — violations cause missed edges.
- **Finite sample errors**: CI tests have limited power with small samples. Use `alpha=0.01` or lower for small $n$.
- **Non-Gaussian data**: Fisher-z CI test assumes Gaussian residuals. Use kernel-based tests for heavy-tailed or nonlinear data.
- **Causal sufficiency for PC**: If latent confounders exist, use FCI instead of PC.
- **CPDAG vs DAG**: PC and GES return a CPDAG (equivalence class), not a unique DAG. Some edges remain undirected — handle conservatively.
- **Collider bias**: Never condition on a common effect of two causes. Conditioning on a collider opens a spurious path.

---

## Connections

- **Builds on:** Conditional independence testing (Module 1 — the filter methods' independence tests underpin PC's skeleton discovery), information-theoretic feature selection (Module 2 — conditional MI is the information-theoretic equivalent of the partial correlation tests used in PC), Granger causality (Module 7 — Granger extends to structural causality when the VAR assumptions hold and there are no latent confounders)
- **Leads to:** Invariant Causal Prediction (Guide 02), Causal ML methods (Guide 03)
- **Related to:** Bayesian network structure learning, graphical models, d-separation

---

## Cross-Module Connections

**From predictive to causal selection:** Modules 1–8 all select features that correlate with the target. Causal selection is strictly stronger: it asks which features remain predictive under intervention. The practical implication is distribution shift robustness — a model built on causal features generalises across environments; a model built on spurious correlates does not.

**Granger vs. structural causality (Module 7):** Granger causality (Module 7) is a weak form of causal selection: it identifies predictive precedence in time series. PC, FCI, and GES identify structural causes — edges in the underlying DAG that persist under interventions. A practical workflow for time series: Granger screen to reduce the feature space (computationally cheap), then apply ICP or FCI on the survivors (computationally expensive but causally rigorous).

**Markov blanket vs. information-theoretic selection (Module 2):** The MI-based Markov blanket approximation in Module 1 (greedy forward CMI search) and Module 2 (CMIM criterion) estimate the same object as the graph-theoretic Markov blanket from PC/GES, but via different paths. Under large samples and faithfulness, both converge to the same set. In finite samples, the graph-based approach (PC) tends to select smaller, sparser sets; the MI approach selects more conservatively (lower false negative rate).

**Causal selector in production (Module 11):** The ICP and double ML selectors developed in Guides 02 and 03 of this module wrap directly into the `SelectorTransformer` pattern from Module 11, deploying as a drop-in pipeline step with correct cross-validation handling.

---

## Further Reading

- Pearl, J. (2009). *Causality: Models, Reasoning and Inference* (2nd ed.). Cambridge University Press. — The definitive reference for SCMs and do-calculus.
- Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press. — PC and FCI algorithms in full detail.
- Chickering, D. M. (2002). Optimal structure identification with greedy search. *JMLR*, 3, 507–554. — GES consistency proof.
- Tsamardinos, I., Brown, L. E., & Aliferis, C. F. (2006). The max-min hill-climbing Bayesian network structure learning algorithm. *Machine Learning*, 65(1), 31–78. — MMHC algorithm: fast hybrid approach.
- `causal-learn` library: https://causal-learn.readthedocs.io — Python implementations of PC, FCI, GES, and more.
