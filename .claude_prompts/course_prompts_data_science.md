# Course Creation Prompts — Three Data Science Courses

Use each prompt below with `/create-course` or directly as a task specification. Each prompt is self-contained and aligned with this repository's course architecture (guides + companion `_slides.md` decks, 15-min Jupyter micro-notebooks, exercises, templates, recipes, quick-starts, and portfolio projects).

---

## 1. CausalPy & Interrupted Time Series

```
/create-course --name causalpy-interrupted-time-series

Create a fully featured course: "Causal Inference with CausalPy — Interrupted Time Series and Beyond"

COURSE SLUG: causalpy-interrupted-time-series
TARGET AUDIENCE: Data scientists and applied economists who can write Python and know basic regression, but have no prior causal inference training.
PREREQUISITE KNOWLEDGE: Linear regression, pandas/numpy fluency, basic probability (distributions, conditional expectation).

MODULE STRUCTURE (8 modules):

Module 00 — Foundations & Setup
- Causal vs predictive mindset: why correlation ≠ causation matters in practice
- The Rubin causal model (potential outcomes) in plain language
- Directed Acyclic Graphs (DAGs) — when and why they help
- Environment setup: install CausalPy, ArviZ, PyMC, matplotlib
- Quick-start notebook: run a pre-built interrupted time series in <2 minutes
- Guide + companion slide deck

Module 01 — Interrupted Time Series (ITS) Fundamentals
- What ITS is, when to use it, and when NOT to use it
- Segmented regression: level change vs slope change vs both
- Autocorrelation in time series residuals — why OLS fails and what to do
- CausalPy's `InterruptedTimeSeries` class end-to-end
- Notebook: ITS on a real public-health policy dataset (e.g., smoking ban hospitalisations)
- Notebook: Diagnosing model fit — residual plots, posterior predictive checks
- Guide + companion slide deck for ITS theory
- Guide + companion slide deck for CausalPy ITS API

Module 02 — Bayesian ITS with PyMC Under the Hood
- Why Bayesian? Uncertainty quantification, priors as domain knowledge
- How CausalPy wraps PyMC — model graph, priors, sampling
- Customising priors: informative vs weakly informative
- Posterior predictive checks and counterfactual distributions
- Notebook: Build an ITS model from scratch in PyMC, then replicate with CausalPy one-liner
- Notebook: Prior sensitivity analysis — how prior choice changes the causal estimate
- Guide + companion slide deck

Module 03 — Synthetic Control Methods
- The synthetic control idea: build a "fake" control from donor units
- When synthetic control beats ITS (multiple units, no randomisation)
- CausalPy's `SyntheticControl` class
- Permutation-based inference and placebo tests
- Notebook: Synthetic control on economic policy data (e.g., California tobacco program)
- Notebook: Placebo tests and inference
- Guide + companion slide deck

Module 04 — Difference-in-Differences (DiD)
- The parallel trends assumption — what it means, how to test it
- Two-period DiD, then generalised staggered adoption DiD
- CausalPy's `DifferenceInDifferences` class
- Event study plots and pre-trend diagnostics
- Notebook: DiD on a labour economics dataset
- Notebook: Staggered DiD with multiple treatment cohorts
- Guide + companion slide deck

Module 05 — Regression Discontinuity Designs (RDD)
- Sharp vs fuzzy RDD
- Bandwidth selection and polynomial order
- CausalPy's `RegressionDiscontinuity` class
- Notebook: Sharp RDD on an education policy dataset
- Notebook: Sensitivity to bandwidth and polynomial choice
- Guide + companion slide deck

Module 06 — Instrumental Variables & Advanced Designs
- IV intuition: finding exogenous variation
- Two-stage least squares (2SLS) and weak instruments
- CausalPy's `InstrumentalVariable` class (if available) or manual PyMC IV
- Combining designs: ITS + DiD, RDD + IV
- Notebook: IV estimation on a classic economics instrument
- Guide + companion slide deck

Module 07 — Production Causal Inference Pipelines
- Model selection workflow: which design for which question?
- Decision flowchart (Mermaid diagram)
- Reporting causal estimates: effect sizes, credible intervals, sensitivity
- Deploying causal models in production (monitoring, retraining triggers)
- Template: end-to-end causal pipeline (data → model → report)
- Recipe: quick CausalPy patterns for common scenarios
- Portfolio project: full causal study on a dataset of the learner's choice — problem statement, design justification, model, diagnostics, written interpretation
- Guide + companion slide deck

CROSS-CUTTING REQUIREMENTS:
- Every notebook ≤15 minutes, uses REAL data (no mocks)
- Every guide has a companion _slides.md using Marp with theme: course
- Every slide has <!-- Speaker notes: ... --> on every slide
- Include quick-starts/ with a 2-minute "hello world" for each CausalPy model type
- Include templates/ with production-ready Python scripts
- Include recipes/ with copy-paste code patterns (e.g., "ITS with seasonal dummies", "DiD with covariates")
- Include projects/ with 2 portfolio projects (one guided, one open-ended)
- Exercises are self-check only — no quizzes, rubrics, or grading
- Use ArviZ for all Bayesian diagnostics and plots
- Mermaid diagrams for DAGs and decision flowcharts in slide decks
- All code must run end-to-end without errors
```

---

## 2. MIDAS Mixed-Frequency Models & Nowcasting

```
/create-course --name midas-mixed-frequency-nowcasting

Create a fully featured course: "Mixed-Frequency Models — MIDAS Regression and Nowcasting"

COURSE SLUG: midas-mixed-frequency-nowcasting
TARGET AUDIENCE: Economists, macro strategists, and data scientists working with mixed-frequency economic or financial data (e.g., daily prices + quarterly GDP).
PREREQUISITE KNOWLEDGE: OLS regression, time series basics (stationarity, autocorrelation, ARIMA intuition), pandas/numpy, basic matrix algebra.

MODULE STRUCTURE (9 modules):

Module 00 — Foundations & the Mixed-Frequency Problem
- Why mixed frequencies matter: quarterly GDP, monthly employment, daily markets
- The aggregation problem: temporal aggregation destroys information
- Traditional solutions (interpolation, bridge equations) and their limits
- Key datasets used throughout the course (FRED API, Yahoo Finance)
- Environment setup: install midasml (or statsmodels MIDAS extensions), fredapi, arch
- Quick-start notebook: forecast quarterly GDP growth using monthly indicators in <2 minutes
- Guide + companion slide deck

Module 01 — MIDAS Regression Fundamentals
- The MIDAS equation: how high-frequency regressors enter a low-frequency model
- Weighting schemes: Almon polynomial, Beta polynomial, exponential Almon, step functions
- Why unrestricted MIDAS (U-MIDAS) sometimes wins
- Lag polynomial visualisation — what the weights look like
- Notebook: Estimate a basic MIDAS regression (quarterly GDP ~ monthly industrial production)
- Notebook: Compare Almon vs Beta vs U-MIDAS on the same data
- Guide + companion slide deck for MIDAS theory
- Guide + companion slide deck for weight functions

Module 02 — Estimation & Inference
- NLS estimation for restricted MIDAS
- OLS estimation for U-MIDAS
- Choosing the number of high-frequency lags (information criteria, cross-validation)
- Hypothesis testing: are mixed-frequency regressors significant?
- Heteroskedasticity and serial correlation robust standard errors
- Notebook: Step-by-step NLS optimisation of MIDAS weights
- Notebook: Model selection — AIC/BIC vs expanding-window cross-validation
- Guide + companion slide deck

Module 03 — Nowcasting with MIDAS
- The nowcasting problem: real-time estimation of the current quarter
- Ragged-edge data and the publication calendar
- Direct vs iterated multi-step forecasting
- Real-time vintage data vs revised data — why it matters
- Notebook: Build a GDP nowcasting model updated daily as new data arrives
- Notebook: Simulate the "ragged edge" — how forecasts improve as the quarter progresses
- Guide + companion slide deck

Module 04 — Dynamic Factor Models for Nowcasting
- Factor models: extracting common signal from many indicators
- DFM with mixed frequencies (Mariano-Murasawa approach)
- Kalman filter/smoother for state extraction
- Combining DFM with MIDAS (factor-augmented MIDAS)
- Notebook: Estimate a small DFM (3–5 indicators) with the Kalman filter
- Notebook: Factor-augmented MIDAS vs plain MIDAS forecast comparison
- Guide + companion slide deck

Module 05 — Machine Learning Extensions
- MIDAS with regularisation: Lasso-MIDAS, Ridge-MIDAS, Elastic Net
- midasml package: group Lasso for MIDAS
- Random forests and gradient boosting with mixed-frequency features
- Feature engineering for ML: flattening vs embedding high-frequency data
- Notebook: Lasso-MIDAS with many monthly/daily predictors
- Notebook: XGBoost nowcasting vs MIDAS regression — fair comparison
- Guide + companion slide deck

Module 06 — Financial Applications
- Realised volatility forecasting: MIDAS-RV (Ghysels-Santa Clara-Valkanov)
- Mixed-frequency risk models: daily VaR using monthly macro indicators
- Term structure nowcasting: yield curve factors at daily frequency
- Commodity price nowcasting with mixed-frequency fundamentals (inventories, shipping, weather)
- Notebook: MIDAS-RV for S&P 500 realised volatility
- Notebook: Commodity price nowcasting (crude oil or copper)
- Guide + companion slide deck

Module 07 — Macroeconomic Applications
- GDP nowcasting in practice (Fed, ECB, central bank approaches)
- Inflation nowcasting with daily price data
- Labour market nowcasting (weekly claims → monthly payrolls)
- Multi-country nowcasting and cross-country spillovers
- Notebook: Replicate a simplified NY Fed GDP Nowcast
- Notebook: Inflation nowcasting with daily commodity prices
- Guide + companion slide deck

Module 08 — Production Nowcasting Systems
- Architecture of a real-time nowcasting pipeline
- Data acquisition: APIs, publication calendars, revision handling
- Model monitoring: when to re-estimate, detecting structural breaks
- Reporting and dashboards: automated nowcast reports
- Template: end-to-end nowcasting pipeline (ingest → estimate → publish)
- Recipe: common MIDAS patterns (adding a new indicator, switching weight functions, forecast combination)
- Decision flowchart: which model for which nowcasting problem (Mermaid diagram)
- Portfolio project: build a live nowcasting dashboard for a macro variable of choice
- Guide + companion slide deck

CROSS-CUTTING REQUIREMENTS:
- Every notebook ≤15 minutes, uses REAL data (FRED, Yahoo Finance, or included CSV fallbacks for offline use)
- Every guide has a companion _slides.md using Marp with theme: course
- Every slide has <!-- Speaker notes: ... --> on every slide
- Include quick-starts/ with a 2-minute "first nowcast" notebook
- Include templates/ with production-ready Python scripts
- Include recipes/ with copy-paste patterns (e.g., "add a new monthly indicator", "switch from Beta to Almon weights", "expanding-window backtest")
- Include projects/ with 2 portfolio projects (one macro, one financial)
- Exercises are self-check only — no quizzes, rubrics, or grading
- Show real-time forecast evolution plots (how the nowcast changes as data arrives)
- Mermaid diagrams for pipeline architecture and decision flowcharts in slide decks
- All code must run end-to-end without errors
- Include fallback CSV data files so notebooks work offline without API keys
```

---

## 3. Captum for Neural Network Interpretability

```
/create-course --name captum-neural-network-interpretability

Create a fully featured course: "Neural Network Interpretability with Captum"

COURSE SLUG: captum-neural-network-interpretability
TARGET AUDIENCE: ML engineers and data scientists who build PyTorch models and need to explain predictions to stakeholders, debug models, or meet regulatory requirements.
PREREQUISITE KNOWLEDGE: PyTorch basics (nn.Module, DataLoader, training loop), neural network fundamentals (forward/backprop, CNNs, basic NLP), numpy/matplotlib.

MODULE STRUCTURE (9 modules):

Module 00 — Foundations & the Interpretability Landscape
- Why interpretability matters: debugging, trust, regulation (EU AI Act, SR 11-7)
- Taxonomy: intrinsic vs post-hoc, local vs global, model-specific vs model-agnostic
- The Captum library: architecture, philosophy, and how it fits the ecosystem
- Comparison: Captum vs SHAP vs LIME vs Integrated Gradients from scratch
- Environment setup: install captum, torch, torchvision, transformers
- Quick-start notebook: explain an ImageNet prediction in <2 minutes
- Guide + companion slide deck

Module 01 — Gradient-Based Attribution Methods
- Vanilla gradients (Saliency maps)
- Input × Gradient
- Guided Backpropagation and Guided GradCAM
- Deconvolution
- Captum API: `Saliency`, `InputXGradient`, `GuidedBackprop`, `Deconvolution`
- Notebook: Compare all four gradient methods on a CNN image classifier
- Notebook: Gradient-based attribution on a tabular neural network
- Guide + companion slide deck for gradient theory
- Guide + companion slide deck for Captum gradient API

Module 02 — Integrated Gradients & Path Methods
- The axiomatic approach: sensitivity and implementation invariance
- Integrated Gradients algorithm — step-by-step derivation
- Choosing baselines: zero, random, blurred, domain-specific
- Convergence diagnostics: number of interpolation steps
- Captum API: `IntegratedGradients`, `NoiseTunnel` (SmoothGrad on top of IG)
- Notebook: IG on image classification with baseline comparison
- Notebook: IG on a text classification model (token-level attributions)
- Guide + companion slide deck

Module 03 — Layer & Neuron Attribution
- GradCAM, Guided GradCAM, and LayerGradCAM
- Layer Conductance — attributing importance to internal layers
- Neuron Conductance — which neurons matter for a specific prediction?
- Internal Influence
- Captum API: `LayerGradCam`, `LayerConductance`, `NeuronConductance`, `InternalInfluence`
- Notebook: GradCAM heatmaps on a ResNet — visualise which regions drive classification
- Notebook: Layer conductance across all layers — find the "decision-making" layer
- Guide + companion slide deck

Module 04 — Perturbation-Based Methods
- Feature Ablation — systematically mask features and measure impact
- Occlusion — sliding window ablation for images
- Shapley Value Sampling — approximating Shapley values via sampling
- Feature Permutation — permutation importance within Captum
- Captum API: `FeatureAblation`, `Occlusion`, `ShapleyValueSampling`, `FeaturePermutation`
- Notebook: Occlusion maps on medical images (skin lesion classifier)
- Notebook: Feature ablation on structured/tabular data
- Guide + companion slide deck

Module 05 — SHAP and KernelSHAP in Captum
- SHAP theory: Shapley values from cooperative game theory
- KernelSHAP: model-agnostic approximation
- GradientSHAP: combining Integrated Gradients with SHAP
- DeepLift and DeepLiftSHAP — propagation-based attribution
- Captum API: `KernelShap`, `GradientShap`, `DeepLift`, `DeepLiftShap`
- Notebook: KernelSHAP vs GradientSHAP vs DeepLiftSHAP — accuracy vs speed trade-off
- Notebook: SHAP summary plots and dependence plots using Captum + matplotlib
- Guide + companion slide deck

Module 06 — Concept-Based & Example-Based Explanations
- Testing with Concept Activation Vectors (TCAV)
- Concept-based explanations: "does this model use the concept of 'stripes'?"
- Influence Functions — which training examples most influence a prediction?
- TracIn — tracing training influence efficiently
- Captum API: `TCAV`, `TracInCP`, `TracInCPFast`, `SimilarityInfluence`
- Notebook: TCAV on an image classifier — test for texture vs shape bias
- Notebook: TracIn — find the most influential training examples for misclassified inputs
- Guide + companion slide deck

Module 07 — NLP & Transformer Interpretability
- Token-level attributions for text classification and NQA
- Attention is not explanation — why attention weights can mislead
- Layer-wise attribution in Transformers (BERT, GPT-2)
- Sequence-to-sequence attribution challenges
- Captum API: `LayerIntegratedGradients`, `TokenReferenceBase`, `configure_interpretable_embedding_layer`
- Notebook: Explain a BERT sentiment classifier — token-level IG with visualisation
- Notebook: Compare attention weights vs Integrated Gradients on the same model
- Guide + companion slide deck

Module 08 — Production Interpretability Pipelines
- Captum Insights: interactive web-based visualisation tool
- Building an interpretability API endpoint (FastAPI + Captum)
- Batch attribution for model monitoring and drift detection
- Regulatory reporting: generating explanation reports for auditors
- Model debugging workflow: using attributions to find spurious correlations
- Template: interpretability-as-a-service (FastAPI + Captum + model registry)
- Recipe: common Captum patterns (custom baselines, multi-output models, batched attributions, attribution caching)
- Decision flowchart: which attribution method for which model type and use case (Mermaid diagram)
- Portfolio project: full model interpretability audit — pick a model, apply 3+ methods, compare, write stakeholder report
- Guide + companion slide deck

CROSS-CUTTING REQUIREMENTS:
- Every notebook ≤15 minutes, uses REAL models and data (pretrained torchvision/HuggingFace models, public datasets)
- Every guide has a companion _slides.md using Marp with theme: course
- Every slide has <!-- Speaker notes: ... --> on every slide
- Include quick-starts/ with 2-minute notebooks for: image attribution, text attribution, tabular attribution
- Include templates/ with production-ready Python scripts
- Include recipes/ with copy-paste patterns (e.g., "IG on a custom model", "GradCAM for any CNN", "token attributions for any HuggingFace model")
- Include projects/ with 2 portfolio projects (one CV-focused, one NLP-focused)
- Exercises are self-check only — no quizzes, rubrics, or grading
- Visualisation-heavy: every attribution method produces a visual output
- Side-by-side comparison plots throughout (method A vs method B on same input)
- Mermaid diagrams for method taxonomy and decision flowcharts in slide decks
- All code must run end-to-end without errors
- Models should use pretrained weights — no training from scratch in the course (except small demos)
```

---

## Usage

Run any of these prompts with:
```bash
# Full course
/create-course --name <course-slug>

# Single module
/create-course --name <course-slug> --module 01

# Just notebooks for a module
/create-course --name <course-slug> --module 03 --notebook

# Just guides + slides for a module
/create-course --name <course-slug> --module 02 --guide
```
