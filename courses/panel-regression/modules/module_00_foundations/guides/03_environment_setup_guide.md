# Environment Setup for Panel Data Analysis

## In Brief

Panel data econometrics requires a specific set of Python libraries beyond the standard data science stack. The core tool is `linearmodels`, which provides `PanelOLS`, `RandomEffects`, `BetweenOLS`, and IV estimators designed specifically for panel data. A reproducible, isolated environment using `venv` or `conda` ensures consistent results across machines and over time.

## Key Insight

The `linearmodels` package is the primary Python tool for panel regression. It expects `pandas` DataFrames with a two-level MultiIndex (`[entity, time]`) and returns result objects with panel-aware standard error options (clustered, robust, kernel-weighted). `statsmodels` handles diagnostic tests and some auxiliary regressions. Both are required for a complete panel workflow.

## Formal Definition

The software stack for panel data analysis has the following dependency structure:

```
numpy >= 1.22
  └── pandas >= 1.4
        └── statsmodels >= 0.13
              └── linearmodels >= 5.0
  └── scipy >= 1.8
        └── statsmodels
matplotlib >= 3.5
  └── seaborn >= 0.12
```

**Required packages and their roles:**

| Package | Version | Role in Panel Analysis |
|---------|---------|------------------------|
| `numpy` | ≥ 1.22 | Array operations, matrix algebra |
| `pandas` | ≥ 1.4 | MultiIndex DataFrames, data manipulation |
| `scipy` | ≥ 1.8 | Statistical distributions, test statistics |
| `statsmodels` | ≥ 0.13 | OLS, diagnostic tests, Breusch-Pagan, Wooldridge |
| `linearmodels` | ≥ 5.0 | `PanelOLS`, `RandomEffects`, `BetweenOLS`, `IV2SLS` |
| `matplotlib` | ≥ 3.5 | Visualization |
| `seaborn` | ≥ 0.12 | Statistical visualizations |
| `jupyter` | ≥ 1.0 | Interactive analysis notebooks |

## Intuitive Explanation

Think of the software stack as a set of specialized tools in a workshop. `numpy` is the raw material (arrays). `pandas` is the workbench (data organization). `statsmodels` is the standard toolkit (OLS, tests). `linearmodels` is the specialized panel data equipment that `statsmodels` lacks. Without `linearmodels`, you would have to implement entity demeaning, GLS quasi-demeaning, and clustered standard errors manually — which is error-prone and time-consuming.

## Code Implementation

**Step 1: Create an isolated environment**

```bash
# Option A: Python venv (recommended for reproducibility)
python3 -m venv panel_env

# Activate (macOS/Linux)
source panel_env/bin/activate

# Activate (Windows)
panel_env\Scripts\activate

# Option B: conda
conda create -n panel_env python=3.11
conda activate panel_env
```

**Step 2: Install packages**

```bash
# Upgrade pip first
pip install --upgrade pip

# Core scientific stack
pip install "numpy>=1.22,<2.0" "pandas>=1.4,<3.0" "scipy>=1.8,<2.0"

# Statistical modeling
pip install "statsmodels>=0.13,<0.16"

# Panel data econometrics (the key package)
pip install "linearmodels>=5.0,<6.0"

# Visualization
pip install "matplotlib>=3.5" "seaborn>=0.12"

# Jupyter
pip install jupyter jupyterlab ipykernel

# Register kernel for Jupyter
python -m ipykernel install --user \
    --name=panel_env \
    --display-name="Python (Panel Data)"
```

**Step 3: Verification script**

```python
def verify_panel_environment():
    """
    Run this script to confirm the environment is correctly configured.
    All imports and a basic model fit should succeed.
    """
    # Core imports
    import numpy as np
    import pandas as pd
    from scipy import stats
    import statsmodels.api as sm
    print(f"numpy:       {np.__version__}")
    print(f"pandas:      {pd.__version__}")
    print(f"statsmodels: {sm.__version__}")

    # Panel-specific imports
    from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
    from linearmodels.datasets import wage_panel
    import linearmodels
    print(f"linearmodels: {linearmodels.__version__}")

    # Functional test: fit a fixed effects model
    data = wage_panel.load()
    data = data.set_index(['nr', 'year'])

    fe_model = PanelOLS.from_formula(
        'lwage ~ expersq + union + married + EntityEffects',
        data=data
    )
    result = fe_model.fit(cov_type='clustered', cluster_entity=True)

    print(f"\nFixed Effects model fit successfully")
    print(f"  R-squared (within): {result.rsquared:.4f}")
    print(f"  N entities:         {result.entity_info['total']}")
    print(f"  N observations:     {result.nobs}")
    print("\nEnvironment verified successfully.")

verify_panel_environment()
```

**requirements.txt for reproducible environments:**

```
numpy>=1.22.0,<2.0.0
pandas>=1.4.0,<3.0.0
scipy>=1.8.0,<2.0.0
statsmodels>=0.13.0,<0.16.0
linearmodels>=5.0,<6.0
jupyter>=1.0.0
jupyterlab>=3.0.0
ipykernel>=6.0.0
matplotlib>=3.5.0,<4.0.0
seaborn>=0.12.0
```

```bash
# Recreate environment from requirements
pip install -r requirements.txt

# Export current environment
pip freeze > requirements_pinned.txt

# conda alternative
conda env export > environment.yml
conda env create -f environment.yml
```

**R setup (optional but useful for cross-validation):**

```r
# Panel data
install.packages("plm")

# Robust inference
install.packages("sandwich")
install.packages("lmtest")
install.packages("clubSandwich")  # Multi-way clustering

# Econometrics
install.packages("AER")

# Verification
library(plm)
data("Grunfeld", package = "plm")
fe <- plm(inv ~ value + capital,
          data = Grunfeld,
          index = c("firm", "year"),
          model = "within")
summary(fe)
```

**Recommended project structure:**

```
panel_analysis_project/
├── data/
│   ├── raw/            # Original data files (never modify)
│   ├── processed/      # Cleaned, restructured panel data
│   └── README.md       # Data sources and transformation log
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_fixed_effects.ipynb
│   └── 04_model_selection.ipynb
├── src/
│   ├── data_utils.py   # Panel loading and restructuring functions
│   ├── panel_utils.py  # Variation decomposition, diagnostics
│   └── visualization.py
├── tests/
│   └── test_panel_utils.py
├── results/
│   ├── tables/
│   └── figures/
├── requirements.txt
├── environment.yml
└── README.md
```

## Common Pitfalls

**`linearmodels` installation fails with compiler error.** On some systems, `linearmodels` requires a C compiler for optional Cython extensions. Use `conda install -c conda-forge linearmodels` instead of `pip` to get pre-compiled binaries.

**Wrong Jupyter kernel.** After creating a virtual environment, the default Jupyter kernel may still be the system Python. Always run `python -m ipykernel install --user --name=panel_env` and select the correct kernel in the notebook.

**Version conflicts.** `linearmodels` versions above 5.0 changed the API for several result attributes. Pin versions in `requirements.txt` to avoid breaking changes when the environment is rebuilt.

**`conda` and `pip` mixing.** Mixing `conda` and `pip` installs within the same environment can produce dependency conflicts. In a `conda` environment, install as much as possible from `conda` channels before using `pip`.

**Missing `statsmodels` for diagnostics.** `linearmodels` handles estimation but delegates some diagnostic tests (Breusch-Pagan, Wooldridge serial correlation) to `statsmodels`. Both must be installed.

## Connections

**Builds on:**
- Python package management fundamentals (`pip`, `conda`, `venv`)
- Scientific Python ecosystem (`numpy`, `pandas`, `scipy`)

**Leads to:**
- All estimation in this course uses `linearmodels.panel` or `statsmodels`
- Diagnostic testing workflow requires `statsmodels.stats` and `scipy.stats`

**Related to:**
- R's `plm` package: equivalent functionality, useful for cross-validating Python results
- Stata's `xtreg` command: the industry standard for panel regression, useful reference for syntax and defaults

## Practice Problems

1. **Environment audit.** Clone this course repository and run `verify_panel_environment()`. If anything fails, diagnose the issue using the troubleshooting table and fix it. Document the steps you took.

2. **Version pinning.** Create a `requirements.txt` that pins all package versions to their exact installed values. Then create a fresh virtual environment, install from the requirements file, and verify the environment passes `verify_panel_environment()`.

3. **R cross-validation.** Using the Grunfeld investment dataset, estimate a fixed effects model in both Python (`linearmodels`) and R (`plm`). Confirm that the coefficients on `value` and `capital` are identical to at least 4 decimal places. Document any differences in default settings between the two implementations.

## Further Reading

- `linearmodels` documentation: https://bashtage.github.io/linearmodels/ — the complete API reference with examples for every estimator.
- `statsmodels` documentation: https://www.statsmodels.org/ — covers OLS, GLS, and the diagnostic tests used in this course.
- "The Python Data Science Handbook" by Jake VanderPlas — covers `numpy`, `pandas`, and `matplotlib` as prerequisites.
- Croissant, Y. & Millo, G. (2008). "Panel Data Econometrics in R: The plm Package." *Journal of Statistical Software*, 27(2). Useful for understanding the R equivalent of every Python panel operation.
