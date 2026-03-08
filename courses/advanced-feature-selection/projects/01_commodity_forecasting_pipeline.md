# Project 1: Commodity Forecasting Pipeline

Build a production-grade, regime-aware feature selection system for multi-commodity price prediction. This project integrates the full arc of the course — filter screening, evolutionary search, stability analysis, drift monitoring, and MLflow-tracked deployment — into a single coherent system operating on real market data.

**Estimated time:** 15–20 hours
**Modules drawn from:** 1, 2, 3, 4, 5, 6, 7, 10, 11
**Primary tools:** `scikit-learn`, `DEAP`, `MLflow`, `yfinance`, `pandas-datareader`, `LightGBM`, `statsmodels`

---

## Motivation

Commodity price forecasting sits at the intersection of macro-economics, microstructure, and supply-demand physics. The feature space that a practitioner assembles — lags of price returns, rolling volatility, cross-commodity spreads, CFTC positioning data, weather indices, inventory reports, term structure metrics — routinely exceeds 500 variables for a single commodity. Standard approaches (drop correlated features, rank by tree importance, keep top-20) fail silently: they ignore regime shifts, ignore the temporal structure of the data, and ignore the instability of the selected subset across time.

The cost of getting feature selection wrong in this domain is asymmetric. A spurious macro variable that appears important during a low-volatility regime poisons the model when volatility spikes. A spread that Granger-causes the target during backwardation becomes irrelevant in contango. Regime-aware, stability-tested selection is not a refinement — it is the baseline for production credibility.

This project asks you to build what a quant trading desk would actually deploy: a walk-forward selection pipeline that re-selects features at each retraining window, tracks which features survive across windows, monitors for distribution shift in production, and wraps everything in a reproducible MLflow experiment.

---

## Core Requirements

1. **Feature engineering at scale.** Construct a feature matrix with at least 500 columns for at least two commodities (e.g., WTI crude, Henry Hub natural gas, copper, or corn). Feature categories must include:
   - Price lags: 1–60 trading days
   - Rolling statistics: mean, standard deviation, skewness, kurtosis at windows of 5, 10, 20, 60 days
   - Technical indicators: RSI, MACD signal line, Bollinger Band width, ATR, OBV momentum, commodity channel index
   - Cross-commodity spreads and ratios (e.g., crack spread, spark spread, brent-WTI spread)
   - Macro variables from FRED: industrial production, CPI, USD index, VIX, yield curve slope (10Y–2Y)
   - Seasonal and calendar features: month, week-of-year, days to contract expiry, seasonal decomposition residuals

2. **Regime identification.** Fit a two- or three-state hidden Markov model (or rolling volatility quantile classifier) on returns to label each trading day with a regime (e.g., trending/mean-reverting, high/low volatility). Regime labels are used to condition feature selection — run a separate selection for each regime in addition to the unconditional selection.

3. **Walk-forward selection with purged cross-validation.** Implement a walk-forward loop with expanding or sliding training windows. Within each window, apply purged cross-validation with an embargo gap of at least five trading days to prevent leakage. Select features independently within each window. The outer test set is the window immediately following the training period and is never touched during selection.

4. **Five selection methods, same data, comparable conditions.** Implement and compare all five of the following on each walk-forward window:
   - mRMR (minimum redundancy maximum relevance) using KSG mutual information estimates
   - Lasso with stability selection (subsampling fraction 0.6, 100 resamples, FDR target 0.1)
   - Boruta with shadow features (perc=100, alpha=0.05, max_iter=200)
   - Binary genetic algorithm via DEAP (population 100, tournament selection, 100 generations, parsimony pressure)
   - NSGA-II via DEAP optimising simultaneously for validation error and feature count (population 150, 150 generations)

   All five methods must receive identical training folds and be evaluated on the same held-out fold within each window.

5. **Feature stability tracking.** For each method and each walk-forward window, record the selected feature set as a binary inclusion vector. Compute and plot:
   - Selection frequency per feature across windows (heatmap, features × windows)
   - Kuncheva's consistency index (KCI) between consecutive windows for each method
   - Jaccard similarity between pairs of methods within the same window
   - Rank a "stable core" of features selected in at least 70% of windows by at least three methods

6. **Drift monitor with re-selection trigger.** After fitting on the full historical window, simulate a production deployment over the most recent 12 months of data (held out entirely during development). At each monthly checkpoint:
   - Compute Population Stability Index (PSI) for each feature in the selected set
   - Compute the Kolmogorov-Smirnov statistic between training and production distributions
   - Trigger re-selection if PSI > 0.25 for any feature in the top-10 by importance, or if the KS test rejects at 1% for more than 20% of selected features
   - Log all drift events and re-selection triggers to MLflow

7. **Production sklearn Pipeline with MLflow tracking.** Wrap the final selected feature set (from the last walk-forward window, stable core variant) into a sklearn-compatible Pipeline:
   - Stage 1: Feature engineering transformer (custom `BaseEstimator`/`TransformerMixin`)
   - Stage 2: Feature selection using the winning method (by held-out Sharpe or RMSE)
   - Stage 3: LightGBM regressor
   - Log to MLflow: selected features, selection method, walk-forward window parameters, validation metrics, all selection run artifacts
   - Register the pipeline as an MLflow model with input/output schema

---

## Suggested Approach

**Step 1 — Data acquisition and feature engineering (3–4 hours)**

Pull daily price data using `yfinance` for your chosen commodities. Pull macro data using `pandas-datareader` or direct FRED API calls. Write a single `FeatureEngineer` class that accepts a raw OHLCV DataFrame and returns the full 500+ column feature matrix. Validate that there is no lookahead: every feature value at time $t$ must be computable from data available at close of day $t$. Document each feature group in a data dictionary.

```python
# Suggested structure — not a template, adapt as needed
class CommodityFeatureEngineer:
    def __init__(self, lags, windows, macro_tickers):
        ...
    def fit(self, df_raw):
        ...  # compute rolling stats using only past data
    def transform(self, df_raw) -> pd.DataFrame:
        ...  # return (n_days, 500+) DataFrame, no lookahead
```

**Step 2 — Regime labelling (1–2 hours)**

Fit an HMM from `hmmlearn` on log-returns, or use a rolling 60-day volatility quantile to assign regime labels. Validate that regime transitions look economically meaningful. Plot price with regime overlaid.

**Step 3 — Walk-forward harness (2–3 hours)**

Implement a `WalkForwardSelector` that:
- Splits data into N windows (suggest N = 8–12 non-overlapping test periods)
- For each window, instantiates purged CV splits using `mlfinance.PurgedKFold` or a custom implementation
- Calls each of the five selection methods with a consistent API
- Records selected features, validation metrics, and wall-clock time
- Returns a results dictionary keyed by (window_index, method_name)

Purged CV is mandatory. If you use `sklearn.model_selection.TimeSeriesSplit`, you must add purging and embargo manually.

**Step 4 — Five selection methods (4–5 hours)**

Implement each method as a class conforming to the sklearn transformer API (`fit`, `transform`, `get_feature_names_out`). Use the course templates as starting points but adapt for the walk-forward context. Pay particular attention to:
- mRMR: KSG estimator is expensive at 500 features — profile and parallelise
- Stability selection: track selection probabilities, not just binary outputs
- Boruta: use `BorutaPy` or implement shadow features from scratch; respect the temporal split
- GA: checkpoint every 25 generations; log fitness trajectory to MLflow
- NSGA-II: log the full Pareto front per window, not just the knee-point solution

**Step 5 — Stability analysis (1–2 hours)**

Compute and visualise the full stability analysis. The heatmap of selection frequency (features × windows) is the centrepiece output. Rank features by stability-weighted importance (selection frequency × mean SHAP value when selected). Define the stable core.

**Step 6 — Drift monitor (2 hours)**

Implement `DriftMonitor` using PSI and KS statistics. Simulate the production deployment period. Log every check to MLflow as a run nested under the main training run. Visualise drift over time for the top-10 features.

**Step 7 — Pipeline assembly and MLflow registry (1–2 hours)**

Assemble the final Pipeline. Run a final end-to-end experiment and register the model. Write a brief model card documenting: training window, selected features, selection method, drift thresholds, and re-selection schedule.

---

## Data Sources and Setup

**Commodity price data**

```python
import yfinance as yf

tickers = {
    "wti_crude": "CL=F",
    "natural_gas": "NG=F",
    "copper": "HG=F",
    "corn": "ZC=F",
    "brent": "BZ=F",
}
# Pull 10+ years of daily OHLCV
data = {name: yf.download(ticker, start="2014-01-01", end="2024-12-31")
        for name, ticker in tickers.items()}
```

**FRED macro data**

```python
import pandas_datareader as pdr

fred_series = {
    "industrial_production": "INDPRO",
    "cpi":                   "CPIAUCSL",
    "usd_index":             "DTWEXBGS",
    "vix":                   "VIXCLS",
    "yield_slope":           "T10Y2Y",
    "us_crude_inventory":    "WCESTUS1",  # weekly, forward-fill to daily
    "natural_gas_storage":   "NGSC",
}
macro = pdr.get_data_fred(list(fred_series.values()), start="2014-01-01")
macro.columns = list(fred_series.keys())
```

**CFTC Commitments of Traders (optional but recommended)**

The CFTC publishes weekly COT data at `https://www.cftc.gov/MarketReports/CommitmentsofTraders/`. The net speculative positioning variable has documented predictive value in energy markets. Parsing the CSV is straightforward.

**Environment setup**

```bash
pip install yfinance pandas-datareader fredapi hmmlearn deap lightgbm \
    mlflow scikit-learn statsmodels scipy plotly seaborn boruta
```

Set `MLFLOW_TRACKING_URI` to a local directory:

```bash
export MLFLOW_TRACKING_URI=./mlruns
mlflow ui  # inspect runs at http://localhost:5000
```

---

## Expected Deliverables

**1. Jupyter notebook (`commodity_forecasting_pipeline.ipynb`)**

A single narrative notebook that walks through the entire pipeline end-to-end. It should be reproducible from top to bottom with a single "Run All" — any reader should be able to clone the repo, install dependencies, and reproduce every figure and metric. Structure:
- Section 1: Data acquisition and feature engineering (show feature count, null audit, lookahead validation)
- Section 2: Regime identification (HMM fit, regime plot, transition matrix)
- Section 3: Walk-forward selection harness (describe CV architecture, show timeline diagram)
- Section 4: Five methods compared (selection frequency table, validation RMSE or Sharpe by method)
- Section 5: Stability analysis (KCI table, Jaccard heatmap, stable core identification)
- Section 6: Drift monitor (PSI and KS time series, trigger events)
- Section 7: Production pipeline (MLflow run link, model card)

**2. Python module (`feature_selection_system/`)**

A proper Python package (not a script) containing:
- `features.py` — `CommodityFeatureEngineer` class
- `regime.py` — regime labelling utilities
- `walk_forward.py` — `WalkForwardSelector` harness
- `selectors.py` — five selector classes, all with sklearn API
- `stability.py` — KCI, Jaccard, stable core computation
- `drift.py` — `DriftMonitor` with PSI and KS
- `pipeline.py` — final sklearn Pipeline assembly
- `tracking.py` — MLflow logging helpers

**3. Analysis report (`analysis_report.md`)**

A 1500–3000 word written analysis covering:
- Which selection method produced the most stable feature sets, and why you think that is
- Whether regime-conditional selection improved held-out performance versus unconditional selection
- The drift events that occurred during the simulated production period and what drove them
- Limitations of the approach and what you would do differently with more time or data

---

## Extension Ideas

These are directions to take the project further. None are required.

**Regime-switching selection.** Instead of fitting separate selectors per regime, implement a soft regime-switching selector that weights training samples by their posterior regime probability. Features that rank highly only in one regime receive a stability penalty.

**Multi-commodity transfer learning.** Train the selector on one commodity (e.g., WTI crude) and evaluate whether the selected feature set transfers to another (e.g., Brent crude or RBOB gasoline). Use Kuncheva's index to quantify transfer overlap. Discuss the economics of why transfer would or would not hold.

**Online GA with incremental updates.** Replace the batch-retrain GA with an online variant that updates the chromosome population incrementally as new data arrives, rather than rerunning from scratch at each window. Compare convergence speed and final subset quality against batch GA.

**Causal audit of the stable core.** Apply the PC algorithm or ICP (from Module 9) to the stable core features. Do any survive the causal screen? Do the causally-identified features produce better out-of-sample performance during the drift period than the predictive-only stable core?

---

## Key References

**Feature selection methodology**

- Ding, C. & Peng, H. (2005). Minimum redundancy feature selection from microarray gene expression data. *Journal of Bioinformatics and Computational Biology*, 3(2), 185–205. — Original mRMR paper.
- Meinshausen, N. & Buhlmann, P. (2010). Stability selection. *Journal of the Royal Statistical Society B*, 72(4), 417–473. — Stability selection with FDR control.
- Kursa, M. B. & Rudnicki, W. R. (2010). Feature selection with the Boruta package. *Journal of Statistical Software*, 36(11). — Boruta algorithm.
- Deb, K., Pratap, A., Agarwal, S. & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

**Time series and financial feature selection**

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. — Chapters 7 (cross-validation), 8 (feature importance), 17 (feature engineering for finance). The purged CV and embargo methodology is from this book.
- Bailey, N., Lopez de Prado, M. & Taylor, J. (2014). Pseudo-mathematics and financial charlatanism: the effects of backtest overfitting on out-of-sample performance. *Notices of the American Mathematical Society*, 61(5).

**Drift monitoring**

- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M. & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4), 1–37.
- Population Stability Index (PSI): standard industry metric in credit risk modelling; see any credit risk model validation reference (e.g., Siddiqi 2006, *Credit Risk Scorecards*).

**Production ML pipelines**

- MLflow documentation: https://mlflow.org/docs/latest/index.html
- Sculley, D. et al. (2015). Hidden technical debt in machine learning systems. *NeurIPS 2015*. — Read this before designing your pipeline architecture.

**Data sources**

- Yahoo Finance via `yfinance`: https://github.com/ranaroussi/yfinance
- FRED (Federal Reserve Economic Data): https://fred.stlouisfed.org
- CFTC Commitments of Traders: https://www.cftc.gov/MarketReports/CommitmentsofTraders/
- Kenneth French Data Library (factor data, useful for extension): https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
