# Module 03 — Sample Paths: The Correct Uncertainty Framework

This is the core module of the course. It introduces sample paths as the mathematically correct framework for multi-step forecast uncertainty, explains the Gaussian Copula method that NeuralForecast uses to generate them, and demonstrates their superiority over marginal quantile forecasts for real business decisions.

## Learning Objectives

By the end of this module you will be able to:

1. Explain why marginal quantiles fail for multi-period decisions
2. Describe the Gaussian Copula method for generating sample paths
3. Use neuralforecast's `.simulate()` to generate sample paths from trained models
4. Answer business questions (inventory stocking, reorder timing) using the Monte Carlo framework
5. Compare sample path bounds vs marginal quantile bounds

## Why This Module Matters

Module 02 showed that quantile forecasts fail for any question involving more than one forecast step. This module provides the solution: sample paths drawn from the joint forecast distribution. The Monte Carlo framework — simulate paths, apply a function, compute statistics — answers any business question correctly.

## Structure

```
module_03_sample_paths/
├── guides/
│   ├── 01_sample_paths_theory.md        # What sample paths are; Monte Carlo framework
│   ├── 01_sample_paths_theory_slides.md  # Companion Marp slide deck (16 slides)
│   ├── 02_gaussian_copula.md            # Six-step Gaussian Copula method
│   └── 02_gaussian_copula_slides.md     # Companion Marp slide deck (16 slides)
├── notebooks/
│   ├── 01_generating_sample_paths.ipynb  # Load data, train, generate, visualise
│   └── 02_business_decisions.ipynb       # Three decision problems with Monte Carlo
├── exercises/
│   └── 01_sample_path_exercises.py       # Self-check with assertions
└── README.md
```

## Learning Path

1. Read **Guide 01** for the conceptual framework: what sample paths are, why marginal quantiles fail for multi-step decisions, and the Monte Carlo template.
2. Read **Guide 02** for the Gaussian Copula implementation: six steps from marginal quantiles to correlated paths, with mathematics and code.
3. Run **Notebook 01** to see the full pipeline on French Bakery data: train NHITS with MQLoss, call `.simulate()`, visualise 100 paths.
4. Run **Notebook 02** to answer three business questions: weekly total, worst-case day, and reorder timing — with explicit comparison to the wrong marginal quantile answers.
5. Run **Exercises** to confirm your understanding with assertion-based self-checks.

## Key Concepts

### Sample Paths

A sample path $\omega^{(s)} = (y_1^{(s)}, \ldots, y_H^{(s)})$ is one draw from the joint forecast distribution $F_{1:H}$. It is a complete, internally consistent demand trajectory for the forecast horizon. Adjacent days in the same path co-vary as they do in historical data.

### The Monte Carlo Framework

```
SIMULATE → generate S paths (n_paths, H)
APPLY    → compute f(path) for each row
AGGREGATE → np.quantile(results, service_level)
```

Any function `f` works: sums, maxima, threshold crossings, reorder points.

### The Gaussian Copula Method

Six steps that power `.simulate()` internally:

| Step | Operation | Output |
|------|-----------|--------|
| 1 | Train NHITS with MQLoss | Marginal quantile forecasts |
| 2 | AR(1) on differenced data | Autocorrelation $\phi$ |
| 3 | Toeplitz matrix $\Sigma_{ij} = \phi^{\|i-j\|}$ | Correlation matrix |
| 4 | Cholesky $\Sigma = LL^T$, draw $z = L\varepsilon$ | Correlated normals |
| 5 | $u = \Phi(z)$ | Correlated uniforms in $[0,1]$ |
| 6 | $y_t = F_t^{-1}(u_t)$ | Sample paths in forecast scale |

### The Central Inequality

The 80th percentile of a weekly total is not the sum of daily 80th percentiles:

$$Q_{0.8}\!\left(\sum_{t=1}^H y_t\right) \neq \sum_{t=1}^H Q_{0.8}(y_t)$$

This inequality drives all three business decision examples in Notebook 02.

## Data

**French Bakery Daily Sales** — baguette item. Loaded directly from the Nixtla open dataset repository. No local file needed; the notebooks fetch it via URL.

## Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `h` | 7 | One full week forecast horizon |
| `input_size` | 28 | Four weeks of look-back |
| `MQLoss(level=[80, 90])` | 80th and 90th percentile intervals | Standard retail service levels |
| `scaler_type="robust"` | Median/IQR normalisation | Robust to daily demand spikes |
| `n_paths` | 100 (notebooks) / 500 (exercises) | 100 for speed; 500 for stable probability estimates |

## Running the Exercises

```bash
cd exercises/
python 01_sample_path_exercises.py
```

All five assertions should pass. Each failure message explains what went wrong and how to fix it.

## Rendering the Slides

```bash
npx @marp-team/marp-cli --html \
  --theme-set /path/to/resources/themes/course-theme.css \
  -- guides/01_sample_paths_theory_slides.md \
     guides/02_gaussian_copula_slides.md
```

## Connection to Other Modules

- **Module 02** established why quantile forecasts fail for multi-step decisions. This module provides the solution.
- **Module 04** (Explainability) uses sample paths to attribute forecast uncertainty to input features.
- **Module 06** (Production Patterns) shows how to serve sample paths in a production API.
