# Project 2: High-Frequency Feature Screener

Build an ultra-fast feature screening library for high-dimensional, low-sample-count data. The system must reduce a p > 10,000 feature space to a compact, predictive subset in sub-second wall-clock time on new data batches, using a cascaded pipeline that combines sure independence screening, stability selection, and a compact genetic algorithm.

**Estimated time:** 15–20 hours
**Modules drawn from:** 1, 2, 4, 5, 8, 10
**Primary tools:** `scikit-learn`, `numpy`, `scipy`, `DEAP`, `pandas`, `pytest`, `matplotlib`

---

## Motivation

The "wide data" regime — many more features than samples — is the rule, not the exception, in several high-value domains: genomics (20,000 gene expression probes, 200 patients), text-based financial signals (TF-IDF or sentence embedding matrices spanning tens of thousands of tokens per document), alternative data (satellite imagery features, credit card transaction aggregates, web traffic signals). The challenge is not only statistical — which features are actually informative — but computational: when p = 50,000 and n = 300, fitting a Lasso takes minutes, Boruta takes hours, and a naive GA is infeasible.

The practitioner's answer is a cascade: use a cheap, statistically principled filter to collapse p from 50,000 to 500 in milliseconds, then apply a more expensive but precise method to that reduced space. Sure Independence Screening (SIS) provides the theoretical guarantee for the first stage: under regularity conditions, the true predictors survive the marginal correlation screen with probability approaching 1 as n grows (Fan & Lv 2008). Iterative SIS (ISIS) conditions on already-selected features to recover predictors with weak marginal signal. Stability selection on the screened set provides FDR control. A compact GA or Estimation of Distribution Algorithm (EDA) then searches the residual space efficiently.

The additional challenge in deployed systems is that features are not static: new data sources arrive, old ones become unavailable, and the system must handle missing data gracefully without retraining from scratch. This project builds not just a one-shot screener but a library that a data engineer can call on a stream of new feature batches.

---

## Core Requirements

1. **High-dimensional dataset with p > 10,000 and n in hundreds.** Use one or more of the following:
   - A real gene expression dataset: GEO (Gene Expression Omnibus) provides hundreds of processed datasets. Suitable examples include GSE2034 (breast cancer recurrence, ~22,000 probes, 286 patients) or any TCGA processed expression matrix.
   - A wide financial panel: construct a TF-IDF matrix from a public earnings call transcript corpus (e.g., StreetEvents via WRDS, or a scraped public corpus), producing a document-term matrix with p >> n.
   - A synthetic-but-calibrated benchmark: generate data following the SIS paper's design (Fan & Lv 2008, model 3 or 4) with known true predictors, p = 20,000, n = 400, and correlation structure, so you can measure exact recovery rates.

   Justify your dataset choice. If using the synthetic benchmark for benchmarking, pair it with a real dataset for the applied analysis.

2. **SIS first-stage screen.** Implement Sure Independence Screening from scratch (not from a library). The implementation must:
   - Compute marginal Pearson or Spearman correlations (continuous target) or marginal log-likelihood ratios (binary target) between each of the p features and the target
   - Reduce to a candidate set of size $d = \lfloor n / \log n \rfloor$ (Fan & Lv's recommended size), or to a configurable top-k
   - Run in under 2 seconds for p = 20,000 and n = 500 on a standard laptop CPU
   - Be vectorised using numpy broadcasting — no Python loops over features

3. **Iterative SIS (ISIS) second stage.** Implement ISIS with at least three iterations. At each iteration, condition on the currently selected set by projecting out its linear influence from both features and target (orthogonalisation), then re-run the marginal screen on the residuals. Stop when the selected set stabilises or a maximum iteration count is reached. Compare the recovery rate of ISIS versus plain SIS on the synthetic benchmark.

4. **Stability selection on the screened set.** Apply subsampling Lasso (Meinshausen & Buhlmann 2010) to the ISIS output set (which should now be of size O(n)). Parameters: 100 subsampling resamples, subsample fraction 0.6, regularisation path from maximum lambda to lambda/100 in 50 steps. Compute per-feature selection probabilities. Apply the threshold to control the expected number of false positives (PFER) at a target of 1.0.

5. **Compact GA or EDA final stage.** On the stability-selected candidate set (which should be tens to low hundreds of features), run either:
   - A compact genetic algorithm (cGA): maintains only a probability vector over feature inclusion, not an explicit population; update the probability vector based on a tournament between two sampled solutions. Show that cGA converges to a similar solution as standard GA in dramatically fewer fitness evaluations.
   - An Estimation of Distribution Algorithm (PBIL or UMDA): maintain a probability vector, sample a population, update the vector toward the best solutions. Use population size 50 and 100 generations.

   The GA stage must complete in under 30 seconds for the candidate set produced by stability selection. This is the sub-second constraint applied to the full pipeline on new data (after the model is fitted): the fitted pipeline applies pre-computed thresholds and inclusion masks, so scoring a new batch requires only matrix multiplications and threshold comparisons.

6. **Sub-second scoring on new data batches.** After fitting the full cascade on training data, the fitted pipeline must score a new batch of m rows × p columns in under 1 second for p = 20,000 and m = 1 to m = 1,000. Demonstrate this with `time.perf_counter` measurements and plot latency versus batch size.

7. **Wall-clock benchmark: full cascade versus direct methods.** On the same dataset and the same train/test split, measure total wall-clock time and held-out prediction performance for:
   - Full SIS → ISIS → stability selection → compact GA cascade (your system)
   - Direct Lasso on the full p = 20,000 feature set
   - Boruta on the full feature set (or on the SIS-screened set if Boruta on 20,000 features is infeasible in reasonable time — document this if so)

   Report: wall-clock time to fit, number of features selected, held-out RMSE or AUC, and selected set overlap (Jaccard) between methods.

8. **Missing data and feature arrival/departure handling.** Implement the following robustness mechanisms:
   - Missing values: the screener must operate correctly when up to 30% of feature values are missing (NaN). Use a fast imputation strategy (column median) that is computed on training data and applied at score time without refitting.
   - Feature arrival: when the scoring batch contains new columns not present in the training feature matrix, the fitted pipeline must silently ignore them (new features are unscreened and excluded from predictions until the next refit).
   - Feature departure: when the scoring batch is missing columns that were in the training feature matrix, the pipeline must substitute the column median imputed at training time and flag the departing feature in a warning log.

---

## Suggested Approach

**Step 1 — Data acquisition and preparation (2–3 hours)**

If using a genomics dataset: download the GEO Series Matrix file for your chosen accession number using the `GEOparse` library or direct FTP. Parse the expression matrix, log2-transform if not already done, and align to clinical outcome labels. Apply variance filtering to remove probes with near-zero variance (these cannot be selected by any method and only slow computation).

If using the synthetic benchmark: implement the data generator following Fan & Lv (2008, Section 4, Model 3). This gives you ground-truth knowledge of which features are true predictors, enabling exact recovery rate measurement.

```python
def generate_fan_lv_model3(n=400, p=20_000, rho=0.5, snr=3.0, seed=0):
    """
    Fan & Lv (2008) Model 3: true support of size s=6, AR(1) correlation
    among features with parameter rho, signal-to-noise ratio snr.
    Returns X (n, p), y (n,), true_support (list of int indices).
    """
    ...
```

**Step 2 — SIS implementation and validation (2–3 hours)**

Write `sis_screen(X, y, d=None)` as a pure numpy function. Validate correctness against a naive reference implementation on a small problem (p = 100, n = 50). Profile with `cProfile` or `line_profiler` to confirm the vectorised version is at least 100x faster than the naive loop. Test that the sure-screening property holds empirically on 100 synthetic replicates: in what fraction of replicates does SIS retain all true predictors in the candidate set?

**Step 3 — ISIS implementation (1–2 hours)**

Write `isis_screen(X, y, max_iter=5, d=None)` that wraps SIS with orthogonalisation. The orthogonalisation step (projecting out selected features) should use `numpy.linalg.lstsq` on the selected columns to obtain residuals. Measure how many additional true predictors ISIS recovers versus plain SIS on the synthetic benchmark.

**Step 4 — Stability selection (1–2 hours)**

Adapt the course's `stability_selection_template.py` to accept a pre-screened feature matrix (the ISIS output). The key modification is that the input is now O(n) dimensional, so the subsampling Lasso runs in seconds rather than minutes. Implement the PFER bound: given threshold $\pi_{thr}$ and number of regularisation steps $\Lambda$, the expected number of false positives is bounded by $\frac{1}{2\pi_{thr} - 1} \cdot \frac{p^2}{\Lambda}$. Let the user specify a PFER target and back out $\pi_{thr}$.

**Step 5 — Compact GA or EDA (2–3 hours)**

Implement cGA or PBIL on the stability-selected candidate set. The fitness function is cross-validated prediction performance on the training data. Since the candidate set is small (tens to low hundreds), each fitness evaluation is cheap — a 5-fold LightGBM fit on O(100) features with n = 400 runs in under a second. Profile total convergence time and plot the fitness trajectory.

**Step 6 — Library packaging (2–3 hours)**

Assemble the cascade into a Python library with a clean public API. The library should feel like a scikit-learn transformer: users call `fit(X_train, y_train)` and then `transform(X_new)` or `predict(X_new)` (if you include a final estimator). Handle all robustness requirements (missing data, feature arrival/departure) inside `transform`.

```python
# Target public API
from hf_screener import HighFrequencyScreener

screener = HighFrequencyScreener(
    sis_d="auto",          # floor(n / log(n))
    isis_max_iter=3,
    stability_pfer=1.0,
    final_stage="cga",     # "cga" | "eda" | "none"
    cga_generations=200,
    random_state=42,
)
screener.fit(X_train, y_train)
X_screened = screener.transform(X_test)      # (m, n_selected) — sub-second
print(screener.selected_features_)           # list of column names / indices
print(screener.selection_probabilities_)     # from stability selection stage
```

**Step 7 — Benchmarking and scaling analysis (2–3 hours)**

Design a benchmarking suite that measures:
- Wall-clock fit time versus p at fixed n (p ∈ {1,000; 5,000; 10,000; 20,000; 50,000})
- Wall-clock fit time versus n at fixed p (n ∈ {100, 200, 400, 800})
- Scoring latency versus batch size m (m ∈ {1, 10, 100, 500, 1000})

Plot each relationship on a log-log scale and fit a power law to determine empirical scaling exponents. Compare to theoretical expectations (SIS is O(np), ISIS is O(k × np) for k iterations).

---

## Data Sources and Setup

**Gene expression data (recommended primary dataset)**

```bash
pip install GEOparse
```

```python
import GEOparse
gse = GEOparse.get_GEO("GSE2034", destdir="./data/")
# Expression matrix is in gse.gsms — iterate to build (n_samples, n_probes) matrix
# Outcome label (distant metastasis free survival) is in sample metadata
```

Alternative genomics datasets with similar structure:
- GSE7390 (breast cancer, ~22,000 probes, 198 patients)
- GSE25066 (breast cancer, neoadjuvant treatment response, ~22,000 probes, 508 patients)
- TCGA processed data via `TCGAbiolinks` (R) or the GDC Data Portal (direct download)

**Wide financial text data (alternative)**

Earnings call transcripts from the SEC EDGAR full-text search API (https://efts.sec.gov/LATEST/search-index?q=...&dateRange=custom). Parse 8-K filings for Item 2.02 (results of operations) to extract call transcripts. Apply `sklearn.feature_extraction.text.TfidfVectorizer` with `max_features=50000` and `sublinear_tf=True` to produce the feature matrix.

**Synthetic benchmark**

No download required — implement the generator following Fan & Lv (2008) as shown in Step 1. Use this for the recovery rate experiments and scaling benchmarks where ground truth is needed.

**Environment setup**

```bash
pip install scikit-learn numpy scipy pandas lightgbm deap \
    matplotlib seaborn pytest GEOparse
```

Run the test suite after implementing each component:

```bash
pytest tests/test_sis.py tests/test_isis.py tests/test_stability.py tests/test_pipeline.py -v
```

---

## Expected Deliverables

**1. Python library (`hf_screener/`)**

A properly structured Python package with:
- `screening.py` — `HighFrequencyScreener` class (sklearn-compatible)
- `sis.py` — `sis_screen` and `isis_screen` functions
- `stability.py` — subsampling Lasso stability selection with PFER control
- `compact_ga.py` (or `eda.py`) — compact GA or PBIL implementation
- `robustness.py` — missing data imputer, feature arrival/departure handler
- `benchmarks.py` — timing utilities for the scaling analysis
- `__init__.py` with clean public exports

**2. Benchmarking notebook (`benchmarks.ipynb`)**

A standalone notebook that:
- Runs the full benchmark suite (fit time vs. p, fit time vs. n, scoring latency vs. m)
- Produces publication-quality log-log scaling plots with fitted power laws
- Compares the cascade against direct Lasso and Boruta on the chosen real dataset
- Documents hardware (CPU, RAM, number of cores) so results are reproducible

**3. Scaling analysis report (`scaling_analysis.md`)**

A 1000–2000 word written analysis covering:
- Empirical scaling exponents and how they compare to theoretical predictions
- Where the cascade breaks down (at what p or n does sub-second scoring become infeasible on commodity hardware?)
- The statistical cost of the cascade — does the SIS screen cause the overall system to miss true predictors that direct Lasso would have found?
- Recommendations for when to use each stage of the cascade and when to skip stages

---

## Extension Ideas

These directions extend the project into research territory. None are required.

**Online learning integration.** Replace the batch-retrain paradigm with an online update that modifies the SIS marginal correlation estimates incrementally as each new sample arrives. Use a running mean and variance (Welford's algorithm) to update marginal correlations in O(p) per new sample. Measure how quickly the selected set converges to the batch solution as samples accumulate.

**Distributed screening.** Parallelise the SIS screen across multiple CPU cores using `multiprocessing.Pool` or `joblib.Parallel`. Measure the speedup curve versus core count. Then design a distributed version using `dask` that could handle p = 1,000,000 features partitioned across machines.

**GPU acceleration.** Implement the SIS marginal correlation computation using `cupy` (GPU numpy). Measure the GPU speedup on a CUDA-capable machine. The correlation computation is a matrix-vector product — this is the ideal GPU workload. Report the crossover point: at what p does GPU computation outweigh the data transfer overhead?

**Knockoff filter integration.** Replace stability selection (the second stage) with the model-X knockoff filter (Candes et al. 2018), which provides exact FDR control rather than PFER control. Compare the two approaches: knockoffs are statistically stronger but require knowledge (or estimation) of the feature distribution. Quantify the additional computation cost.

**False discovery rate on the synthetic benchmark.** Using the synthetic benchmark where the true support is known, sweep the PFER threshold and measure empirical false discovery rate versus empirical power. Compare the theoretical PFER bound to the empirical FDR. At what threshold does the bound become loose?

---

## Key References

**Sure independence screening**

- Fan, J. & Lv, J. (2008). Sure independence screening for ultrahigh dimensional feature space. *Journal of the Royal Statistical Society B*, 70(5), 849–911. — The foundational SIS paper. Read the sure-screening theorem and the iterative extension.
- Fan, J., Samworth, R. & Wu, Y. (2009). Ultrahigh dimensional feature selection: beyond the linear model. *Electronic Journal of Statistics*, 3, 1295–1321. — Extension to nonlinear models.
- Fan, J. & Song, R. (2010). Sure independence screening in generalized linear models with NP-dimensionality. *Annals of Statistics*, 38(6), 3567–3604. — Extends SIS to GLMs.

**Stability selection**

- Meinshausen, N. & Buhlmann, P. (2010). Stability selection. *Journal of the Royal Statistical Society B*, 72(4), 417–473. — The PFER bound derivation.
- Shah, R. D. & Samworth, R. J. (2013). Variable selection with error control: another look at stability selection. *Journal of the Royal Statistical Society B*, 75(1), 55–80. — Strengthens the FDR control.

**Compact and estimation-of-distribution algorithms**

- Harik, G. R., Lobo, F. G. & Goldberg, D. E. (1999). The compact genetic algorithm. *IEEE Transactions on Evolutionary Computation*, 3(4), 287–297. — Original cGA paper.
- Baluja, S. (1994). Population-based incremental learning. Carnegie Mellon University, Technical Report CMU-CS-94-163. — Original PBIL paper.
- Hauschild, M. & Pelikan, M. (2011). An introduction and survey of estimation of distribution algorithms. *Swarm and Evolutionary Computation*, 1(3), 111–128.

**High-dimensional feature selection surveys**

- Chandrashekar, G. & Sahin, F. (2014). A survey on feature selection methods. *Computers & Electrical Engineering*, 40(1), 16–28.
- Bommert, A., Sun, X., Bischl, B., Rahnenführer, J. & Lang, M. (2020). Benchmark for filter methods for feature selection in high-dimensional classification data. *Computational Statistics & Data Analysis*, 143, 106839.

**Knockoff filter (extension)**

- Candes, E., Fan, Y., Janson, L. & Lv, J. (2018). Panning for gold: model-X knockoffs for high dimensional controlled variable selection. *Journal of the Royal Statistical Society B*, 80(3), 551–577.

**Data resources**

- NCBI GEO (Gene Expression Omnibus): https://www.ncbi.nlm.nih.gov/geo/
- GEOparse Python library: https://geoparse.readthedocs.io/
- SEC EDGAR full-text search: https://efts.sec.gov/LATEST/search-index
- Fan & Lv (2008) simulation designs: Appendix of the paper, all parameters specified
