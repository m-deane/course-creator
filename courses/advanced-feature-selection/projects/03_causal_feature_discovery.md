# Project 3: Causal Feature Discovery

Identify truly causal features versus spurious predictors in observational data. Apply both predictive and causal selection methods to the same dataset, quantify their overlap and divergence, and demonstrate empirically that causal features maintain predictive performance under distribution shift while predictive-only features degrade. Build a blended selection strategy that combines causal stability with predictive power.

**Estimated time:** 15–20 hours
**Modules drawn from:** 4, 6, 9, 10
**Primary tools:** `DoWhy`, `causal-learn`, `scikit-learn`, `DEAP`, `shap`, `networkx`, `matplotlib`

---

## Motivation

Predictive and causal relevance are not the same thing. A feature can have high mutual information with the outcome, high SHAP importance, survive Boruta, and be selected by an optimised genetic algorithm — and still be a spurious predictor that collapses under distribution shift. Consider a healthcare dataset where zip code predicts patient outcomes with high accuracy because it is correlated with income, which drives both treatment adherence and health outcomes. A model trained to maximise predictive accuracy will select zip code. A model trained to identify the causal mechanism will not — income, treatment adherence, or the specific pathway mediating their effect on outcomes is what you actually want to measure and intervene on.

The stakes are asymmetric: a spurious predictor discovered during training on one time period or geography may perform well on the test set (which comes from the same distribution) but fail catastrophically when the model is deployed in a new hospital, a new country, or a different economic regime. Causal features — those that are part of the causal mechanism generating the outcome — are stable across environments precisely because the mechanism itself does not change, even when the marginal distribution of features does.

This project creates a controlled empirical comparison. You will select a dataset with known or plausible causal structure, apply both predictive methods (SHAP-ranked importance, Boruta, genetic algorithm) and causal methods (PC algorithm, Invariant Causal Prediction), and then test both feature sets under distribution shift. The core empirical claim to evaluate is: causal features generalise better to out-of-distribution environments than predictive features selected without causal constraints. Your analysis should confirm, qualify, or complicate this claim using your chosen data.

---

## Core Requirements

1. **Dataset with known or plausible causal structure and at least two distinct environments.** The two environments are essential — without them, Invariant Causal Prediction cannot be applied and you cannot test distribution-shift robustness. Valid dataset choices include:
   - **Sachs et al. (2005) protein signalling data** (gold standard): 11 proteins, 9 experimental conditions (environments), 7,466 observations, ground-truth causal DAG published. This is the canonical benchmark for causal discovery algorithms.
   - **IHDP (Infant Health and Development Program)**: observational study of the effect of specialist home visits on cognitive test scores. Two natural environments are rural vs. urban sites.
   - **abcde dataset (Peters et al. 2016 ICP paper supplement)**: synthetic data with known DAG and multiple environments, used in the original ICP paper.
   - **Financial panel data across economic regimes**: use the FRED-MD monthly macroeconomic dataset, splitting into pre-2008 and post-2008 environments (or bear vs. bull market periods). Features are macro indicators; target is one-month-ahead industrial production growth.
   - **Causeme benchmark** (https://causeme.uv.es/): a repository of causal discovery benchmarks from climate science, ecology, and economics with ground truth causal graphs.

   Justify your dataset choice. Explicitly state what you treat as "environments" and why the environments plausibly differ in their feature distributions while preserving the underlying causal mechanism.

2. **Three predictive selection methods applied to the pooled training data.** Use the combined data from all environments (minus the held-out test environments) as the training set. Apply:
   - SHAP-ranked importance: fit LightGBM on the full feature set, compute SHAP TreeExplainer values, rank features by mean absolute SHAP, select the top-k features where k is chosen by cross-validated performance on a validation split
   - Boruta: run to convergence with shadow features; collect all "confirmed" features
   - Genetic algorithm (DEAP binary GA): optimise cross-validated AUC or RMSE, select the Pareto-knee solution from a brief NSGA-II run (accuracy vs. cardinality)

   All three methods receive the same training data split. Record the selected feature set for each method.

3. **Two causal discovery methods.** Apply both of the following:
   - **PC algorithm** via `causal-learn`: run the PC algorithm with the Fisher-Z conditional independence test (continuous data) or G-squared test (discrete). Use significance level 0.05. Extract the Markov blanket of the target variable from the resulting CPDAG — this is the causal feature set under the PC algorithm's assumptions.
   - **Invariant Causal Prediction (ICP)** via the `InvariantCausalPrediction` implementation in `causal-learn` or a direct implementation following Peters, Mooij, Janzing & Scholkopf (2016): for each candidate subset S of features, test whether the conditional distribution of Y given X_S is invariant across environments (using a linear model with F-test or a nonparametric test). The accepted set is the intersection of all invariant subsets. Features not in any accepted set are excluded.

   For ICP, you need explicit environment labels. Use the environments defined in Requirement 1.

4. **Overlap and divergence quantification.** For all five methods (three predictive, two causal), compute and report:
   - The selected feature set for each method (name the features, not just the count)
   - Pairwise Jaccard similarity matrix (5 × 5 grid) — visualise as a heatmap
   - Features in the predictive consensus (selected by at least two of the three predictive methods) but not in any causal set — call this the "spurious candidates" list
   - Features in the causal consensus (PC Markov blanket ∩ ICP accepted set) but not in any predictive set — call this the "hidden causal" list
   - Features in both the predictive consensus and the causal consensus — call this the "stable core"

5. **Causal graph visualisation.** Produce a visualisation of the CPDAG output by the PC algorithm. Highlight which nodes are in the target's Markov blanket. Overlay the ICP-accepted features with a distinct colour. Overlay the predictive-consensus features with a third colour. This single figure should show at a glance the degree of agreement between causal and predictive methods.

   Use `networkx` for graph layout. The figure must be readable (node labels visible, edge directions shown, legend present).

6. **Distribution shift robustness test.** This is the central empirical test of the project. For each of the five feature sets (SHAP, Boruta, GA, PC Markov blanket, ICP accepted set), plus the stable core and spurious candidates as additional sets:
   - Fit a LightGBM model on each feature set using the in-distribution training data
   - Evaluate on the in-distribution test set (same-environment held-out data)
   - Evaluate on the out-of-distribution test set (different environment, not used in any part of training or selection)
   - Report: in-distribution performance, out-of-distribution performance, and the gap between them (the "generalisation gap")

   The hypothesis is that causal feature sets will have a smaller generalisation gap than predictive feature sets. Plot a bar chart of in-distribution vs. out-of-distribution performance for each feature set, sorted by generalisation gap.

7. **Blended selection strategy.** Design a combined selector that uses both causal stability and predictive power to score features. One approach:
   - Assign each feature a causal score: 1 if in PC Markov blanket, 1 if in ICP accepted set, 0.5 if in one but not both, 0 otherwise
   - Assign each feature a predictive score: mean SHAP importance normalised to [0, 1], or selection probability from Boruta
   - Blend: `final_score = alpha * causal_score + (1 - alpha) * predictive_score`
   - Sweep alpha ∈ {0, 0.25, 0.5, 0.75, 1.0} and evaluate each blended feature set on the out-of-distribution test set
   - Plot out-of-distribution performance versus alpha

   Implement this as a class with a `fit(X, y, environments)` method and a `select(alpha)` method.

---

## Suggested Approach

**Step 1 — Dataset acquisition and environment setup (2–3 hours)**

If using Sachs et al. (2005): download the data from the supplementary materials of the original paper (Science 2005) or from https://www.bnlearn.com/research/sachs06/. Each of the nine experimental conditions is an environment. You will need to decide which environments to use as training and which as out-of-distribution test. A natural split: conditions 1–7 as training environments (these are observational and general stimulation), conditions 8–9 as OOD test (these involve specific interventions — PKC and PMA — that alter the causal structure in known ways).

Perform exploratory data analysis: distribution plots per environment, correlation heatmaps, evidence that the environments differ in feature distributions (required for ICP). If the environments do not differ, ICP cannot find invariant subsets and the project's empirical premise collapses. Document that the environments differ.

**Step 2 — Predictive selection (3–4 hours)**

Apply SHAP, Boruta, and GA to the pooled training data. This is the most computationally expensive step. Run Boruta and the GA overnight if needed — document wall-clock times.

For the GA, define fitness as 5-fold cross-validated AUC (binary classification) or RMSE (regression) with a parsimony penalty. Use population size 100 and at least 100 generations. Log the fitness trajectory and convergence.

For SHAP, fit LightGBM with careful hyperparameter tuning (at least a brief Optuna search), then extract SHAP values with `shap.TreeExplainer`. Select the top-k features by mean absolute SHAP where k is determined by cross-validated performance drop (stop when dropping the next feature costs more than 0.01 AUC or 0.01 RMSE).

**Step 3 — PC algorithm (2–3 hours)**

```python
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

# Pool training environments
X_train_pooled = pd.concat([env_data[e] for e in train_environments])
data_array = X_train_pooled[feature_cols + [target_col]].values

cg = pc(data_array, alpha=0.05, indep_test=fisherz)
# cg.G is a GeneralGraph; cg.nx_graph is a networkx DiGraph
```

Extract the Markov blanket of the target node (index = last column). The Markov blanket in a DAG is the set of parents, children, and co-parents (other parents of the target's children). In a CPDAG, extract the adjacency of the target node as an approximation.

Handle the PC algorithm's sensitivity to conditioning set size: with n < 500 and p > 10, higher-order conditional independence tests will be unreliable. Cap the maximum conditioning set size at 3 using the `max_cond_vars` parameter.

**Step 4 — ICP (2–3 hours)**

ICP requires explicit environment labels. Implement ICP using the linear invariance test:

```python
# ICP test: for a candidate subset S, fit OLS(y ~ X_S) per environment.
# Test whether residual variance is equal across environments (Levene or
# Bartlett test) AND coefficients are equal (Chow test).
# A subset S "passes" if p-value > alpha for both tests.

def test_invariance(X, y, env_labels, subset_indices, alpha=0.05):
    """
    Returns True if the conditional distribution of y | X[:, subset_indices]
    is invariant across environments at significance level alpha.
    """
    ...
```

For p features, the search space is $2^p$ subsets — infeasible for p > 20. Use the following practical approach from Peters et al. (2016, Section 3.2): start with the empty set, test all singletons, then pairs, then triples, pruning branches where the subset fails the invariance test. The ICP accepted set is the intersection of all subsets that pass. If no subset passes at alpha = 0.05, relax to alpha = 0.10.

**Step 5 — Overlap analysis and visualisation (2 hours)**

Compute all pairwise Jaccard similarities and render the heatmap. Identify the spurious candidates, hidden causal features, and stable core. Render the causal graph with feature-set overlays using networkx. Spend time on figure quality — this is the centrepiece of your portfolio presentation.

**Step 6 — Distribution shift experiment (1–2 hours)**

Fit LightGBM on each feature set and evaluate on both in-distribution and out-of-distribution test sets. Use a fixed LightGBM configuration across all feature sets (do not tune per feature set, as this would confound the comparison). Report results in a table and a bar chart.

**Step 7 — Blended selector and alpha sweep (1–2 hours)**

Implement `CausalPredictiveBlender` with the API described in Requirement 7. Sweep alpha, fit a model at each alpha, evaluate on OOD test set. Plot the performance curve and identify the optimal alpha. Discuss whether the optimal alpha is closer to 0 (pure predictive) or 1 (pure causal) and what that implies about the data.

---

## Data Sources and Setup

**Sachs et al. (2005) protein signalling (recommended)**

```python
import pandas as pd
# Direct download from bnlearn (pre-processed CSV versions)
import urllib.request
url = "https://www.bnlearn.com/research/sachs06/sachs.data.txt"
urllib.request.urlretrieve(url, "data/sachs.txt")
sachs = pd.read_csv("data/sachs.txt", sep="\t")
# 11 columns: Raf, Mek, PLCg, PIP2, PIP3, Erk, Akt, PKA, PKC, P38, Jnk
# Target for a regression formulation: Erk (extracellular signal-regulated kinase)
# Ground truth DAG from Sachs et al. 2005, Fig. 3A
```

The nine experimental conditions must be downloaded separately as individual CSV files from the paper's supplementary data.

**FRED-MD monthly macroeconomic data (alternative, finance focus)**

```python
import pandas as pd
fred_md = pd.read_csv(
    "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv"
)
# 128 monthly macro variables from 1959 to present
# Target: INDPRO (industrial production, one-month-ahead growth)
# Environments: pre-2008 recession, post-2008 expansion, COVID period
```

**Peters et al. (2016) synthetic data (for ICP validation)**

The original ICP paper (Peters, Mooij, Janzing & Scholkopf, 2016) provides synthetic datasets in the paper's supplementary materials and the companion R package `InvariantCausalPrediction`. Reproduce the paper's Example 1 to validate your ICP implementation before applying it to real data.

**Causeme benchmark (alternative)**

Download benchmark datasets from https://causeme.uv.es/ — the platform provides ground truth graphs and pre-split training/test environments for climate science and ecology datasets.

**Environment setup**

```bash
pip install causal-learn dowhy shap lightgbm deap networkx \
    scikit-learn pandas numpy scipy matplotlib seaborn optuna
```

```python
# Verify causal-learn installation
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.FCIBased.FCI import fci
print("causal-learn installed correctly")
```

---

## Expected Deliverables

**1. Jupyter notebook (`causal_feature_discovery.ipynb`)**

A narrative notebook structured as a controlled experiment, not a tutorial. Each section poses a question and answers it with code and evidence:
- Section 1: What does the data look like across environments? (EDA, environment differences)
- Section 2: What do predictive methods select? (SHAP, Boruta, GA results)
- Section 3: What do causal methods select? (PC CPDAG, ICP accepted set, Markov blanket)
- Section 4: Where do they agree and disagree? (overlap analysis, Jaccard heatmap)
- Section 5: Which features survive distribution shift? (OOD robustness experiment)
- Section 6: Can we do better by blending? (alpha sweep, optimal blend)
- Section 7: What did we learn?

**2. Causal graph visualisation (`causal_graph.png` and `causal_graph_annotated.pdf`)**

A standalone, publication-quality figure of the PC algorithm CPDAG with:
- Node size proportional to the target variable's SHAP importance for each feature
- Node colour indicating feature-set membership (spurious-only, causal-only, stable core, neither selected)
- Edge direction and edge style distinguishing directed from undirected edges in the CPDAG
- Legend and figure caption

**3. Robustness analysis report (`robustness_analysis.md`)**

A 1500–3000 word written analysis structured as a scientific report:
- Background: why causal features should generalise better
- Methods: brief description of each selection method and the distribution shift test design
- Results: the main table (in-distribution vs. OOD performance by feature set), the generalisation gap ranking, the alpha sweep results
- Discussion: does your data support the causal-generalisation hypothesis? Under what conditions did predictive features fail? Were there causal features that also failed? What might explain any anomalies?
- Limitations: what assumptions did each causal method require, and how confident are you that those assumptions hold in your data?

---

## Extension Ideas

**Causal discovery under latent confounders (FCI).** Replace the PC algorithm with the Fast Causal Inference (FCI) algorithm, which outputs a Partial Ancestral Graph (PAG) instead of a CPDAG. PAGs explicitly represent the possibility of latent common causes between variables. Extract the Markov blanket under FCI (using the Markov boundary definition for PAGs from Richardson & Spirtes 2002). Compare the FCI-derived feature set to the PC-derived set.

**Sensitivity analysis for hidden confounders.** For the ICP-accepted feature set, perform a sensitivity analysis: how much unmeasured confounding (parametrised as the partial R-squared of a hypothetical confounder with both the selected features and the target) would be required to overturn the invariance conclusion? Use Cinelli & Hazlett's (2020) omitted variable bias framework or Rosenbaum's sensitivity analysis for observational studies.

**Double ML for causal coefficients.** Apply Double Machine Learning (Chernozhukov et al. 2018) to estimate the causal effect of each feature on the target, partialling out the influence of all other features using cross-fitted nuisance models. Compare the double-ML coefficient estimates to OLS estimates — features with large OLS coefficients but small double-ML estimates are likely confounded. Add a double-ML filter to the blended selection strategy.

**Instrumental variable selection.** If your dataset has or can be augmented with instruments (variables that affect the target only through a specific feature, not directly), implement 2SLS-based causal feature selection. An instrument for feature X is a variable Z that correlates with X (relevance condition) and is independent of the outcome conditional on X (exclusion restriction). In the FRED-MD context, oil price shocks can instrument for energy-related macro variables.

**Online ICP with streaming environments.** ICP as described by Peters et al. (2016) assumes all environment data is available simultaneously. Design an online variant that updates the set of invariant subsets incrementally as data from a new environment arrives, without re-running the full procedure from scratch. This matters in practice where new market regimes or new geographies become available over time.

---

## Key References

**Causal feature selection and ICP**

- Peters, J., Mooij, J. M., Janzing, D. & Scholkopf, B. (2016). Causal discovery with continuous additive noise models. *Journal of Machine Learning Research*, 15, 2009–2053.
- Peters, J., Buhlmann, P. & Meinshausen, N. (2016). Causal inference by using invariant prediction: identification and confidence intervals. *Journal of the Royal Statistical Society B*, 78(5), 947–1012. — The ICP paper. Read Section 2 (the invariance condition) and Section 3 (the algorithm) carefully.
- Arjovsky, M., Bottou, L., Gulrajani, I. & Lopez-Paz, D. (2019). Invariant risk minimization. arXiv:1907.02893. — A neural network approach to the same invariance idea.

**PC algorithm and causal discovery**

- Spirtes, P., Glymour, C. & Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press. — The canonical reference for constraint-based causal discovery. Chapter 5 covers the PC algorithm.
- Colombo, D. & Maathuis, M. H. (2014). Order-independent constraint-based causal structure learning. *Journal of Machine Learning Research*, 15, 3741–3782. — Addresses the ordering dependence problem in the PC algorithm.
- Richardson, T. & Spirtes, P. (2002). Ancestral graph Markov models. *Annals of Statistics*, 30(4), 962–1030. — Foundation for FCI and PAGs (extension).

**Benchmark dataset**

- Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A. & Nolan, G. P. (2005). Causal protein-signaling networks derived from multiparameter single-cell data. *Science*, 308(5721), 523–529. — The dataset and its ground-truth DAG.

**Distribution shift and generalisation**

- Scholkopf, B. et al. (2021). Toward causal representation learning. *Proceedings of the IEEE*, 109(5), 612–634. — Conceptual foundation for why causal representations generalise.
- Koh, P. W. et al. (2021). WILDS: a benchmark of in-the-wild distribution shifts. *ICML 2021*. — Empirical evidence on distribution shift failure modes.
- Subbaswamy, A. & Saria, S. (2020). From development to deployment: dataset shift, causality, and shift-stable models in health AI. *Biostatistics*, 21(2), 345–352.

**Double ML (extension)**

- Chernozhukov, V. et al. (2018). Double/debiased machine learning for treatment and structural parameters. *Econometrics Journal*, 21(1), C1–C68.
- Cinelli, C. & Hazlett, C. (2020). Making sense of sensitivity: extending omitted variable bias. *Journal of the Royal Statistical Society B*, 82(1), 39–67.

**Tools**

- causal-learn documentation: https://causal-learn.readthedocs.io/
- DoWhy documentation: https://www.pywhy.org/dowhy/
- Causeme benchmark platform: https://causeme.uv.es/
- bnlearn Sachs data: https://www.bnlearn.com/research/sachs06/
