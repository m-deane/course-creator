# Environment Setup for Panel Data Analysis

> **Reading time:** ~20 min | **Module:** 00 — Foundations | **Prerequisites:** None (entry point)


## In Brief


<div class="callout-key">

**Key Concept Summary:** Proper environment setup ensures reproducible panel data analysis with the right tools and versions. This guide covers installation of Python packages (linearmodels, statsmodels) and R packages (pl...

</div>

Proper environment setup ensures reproducible panel data analysis with the right tools and versions. This guide covers installation of Python packages (linearmodels, statsmodels) and R packages (plm, lmtest) needed for panel regression.

> 💡 **Key Insight:** Panel data econometrics requires specialized libraries beyond standard data science tools. While pandas handles data manipulation, you need **linearmodels** (Python) or **plm** (R) for proper panel regression estimation. Setting up a dedicated virtual environment prevents version conflicts and ensures reproducibility.

## Prerequisites

<div class="callout-insight">

**Insight:** Panel data lets you control for unobservable differences between entities that are constant over time. This is the single most important reason to prefer panel data over repeated cross-sections.

</div>


- **Python:** Version 3.8 or higher
- **R:** Version 4.0 or higher (if using R)
- **Package manager:** pip (Python) or conda, plus R package installer
- **IDE:** Jupyter Lab/Notebook, VS Code, or RStudio
- **Basic knowledge:** Command line operations, virtual environments

## Python Setup

<div class="callout-warning">

**Warning:** Reporting results without appropriate standard errors is a common mistake. In panel data, conventional OLS standard errors are almost always wrong -- use clustered or heteroskedasticity-robust standard errors.

</div>


### Step 1: Create Virtual Environment

Using `venv` (built-in):

```bash
# Navigate to your project directory
cd /path/to/your/project

# Create virtual environment
python3 -m venv panel_env

# Activate environment
# On macOS/Linux:
source panel_env/bin/activate
# On Windows:
panel_env\Scripts\activate
```

Using `conda` (alternative):

```bash
# Create environment with Python 3.10
conda create -n panel_env python=3.10

# Activate environment
conda activate panel_env
```

### Step 2: Install Core Packages

```bash
# Upgrade pip
pip install --upgrade pip

# Install core data science stack
pip install numpy pandas matplotlib seaborn jupyter

# Install statistical packages
pip install scipy statsmodels

# Install panel data econometrics library
pip install linearmodels

# Optional: Install additional useful packages
pip install scikit-learn plotly

# Install development tools (for running tests, formatting)
pip install pytest black isort mypy
```

### Step 3: Verify Installation

Create a test script `test_installation.py`:


<span class="filename">test_installation.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
"""Test that all required packages are installed and working."""

def test_imports():
    """Test that all packages can be imported."""
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import statsmodels.api as sm
        from linearmodels.panel import PanelOLS, RandomEffects
        from linearmodels.datasets import wage_panel

        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_basic_functionality():
    """Test basic panel regression functionality."""
    try:
        from linearmodels.datasets import wage_panel
        from linearmodels.panel import PanelOLS

        # Load example data
        data = wage_panel.load()

        # Set panel structure
        data = data.set_index(['nr', 'year'])

        # Run basic FE model
        model = PanelOLS.from_formula('lwage ~ expersq + union + married + EntityEffects',
                                      data=data)
        result = model.fit()

        print("✅ Basic panel regression successful!")
        print(f"   R-squared: {result.rsquared:.4f}")
        return True
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False


def check_versions():
    """Print versions of key packages."""
    import numpy as np
    import pandas as pd
    import statsmodels
    import linearmodels

    print("\n" + "="*50)
    print("Package Versions:")
    print("="*50)
    print(f"NumPy:        {np.__version__}")
    print(f"Pandas:       {pd.__version__}")
    print(f"Statsmodels:  {statsmodels.__version__}")
    print(f"Linearmodels: {linearmodels.__version__}")
    print("="*50)


if __name__ == "__main__":
    print("Testing Panel Data Analysis Environment Setup\n")

    # Test imports
    imports_ok = test_imports()

    # Test functionality
    if imports_ok:
        functionality_ok = test_basic_functionality()

    # Show versions
    check_versions()

    if imports_ok and functionality_ok:
        print("\n🎉 Environment setup complete and verified!")
    else:
        print("\n⚠️  Some tests failed. Please check error messages above.")
```


</div>

Run the test:

```bash
python test_installation.py
```

Expected output:
```
Testing Panel Data Analysis Environment Setup

✅ All imports successful!
✅ Basic panel regression successful!
   R-squared: 0.1834

==================================================
Package Versions:
==================================================
NumPy:        1.24.3
Pandas:       2.0.2
Statsmodels:  0.14.0
Linearmodels: 5.3
==================================================

🎉 Environment setup complete and verified!
```

### Step 4: Jupyter Setup

Install and configure Jupyter:

```bash
# Install Jupyter
pip install jupyter jupyterlab

# Create IPython kernel for this environment
python -m ipykernel install --user --name=panel_env --display-name="Python (Panel Data)"

# Launch Jupyter Lab
jupyter lab
```

In Jupyter, verify the kernel:
- Create a new notebook
- Select kernel: "Python (Panel Data)"
- Run test cell:


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import sys
print(f"Python path: {sys.executable}")

import linearmodels
print(f"Linearmodels version: {linearmodels.__version__}")
```


</div>

## R Setup

### Step 1: Install R Packages

Open R or RStudio and run:

```r
# Install core panel data package
install.packages("plm")

# Install testing and diagnostics packages
install.packages("lmtest")
install.packages("sandwich")

# Install data manipulation and visualization
install.packages("tidyverse")
install.packages("haven")  # For reading Stata/SPSS files

# Install table formatting
install.packages("stargazer")
install.packages("texreg")

# Optional: Install additional econometrics packages
install.packages("AER")       # Applied Econometrics with R
install.packages("clubSandwich")  # Cluster-robust inference
```

### Step 2: Verify Installation

Create test script `test_installation.R`:

```r
# Test R Environment Setup for Panel Data Analysis

test_imports <- function() {
  cat("Testing package imports...\n")

  required_packages <- c("plm", "lmtest", "sandwich", "stargazer")

  for (pkg in required_packages) {
    if (require(pkg, character.only = TRUE, quietly = TRUE)) {
      cat(sprintf("  ✓ %s loaded successfully\n", pkg))
    } else {
      cat(sprintf("  ✗ Failed to load %s\n", pkg))
      return(FALSE)
    }
  }

  cat("✅ All packages loaded successfully!\n")
  return(TRUE)
}

test_basic_functionality <- function() {
  cat("\nTesting basic panel regression...\n")

  tryCatch({
    library(plm)

    # Load example data (Grunfeld investment data)
    data("Grunfeld", package = "plm")

    # Estimate fixed effects model
    fe_model <- plm(inv ~ value + capital,
                    data = Grunfeld,
                    index = c("firm", "year"),
                    model = "within")

    cat("✅ Basic panel regression successful!\n")
    cat(sprintf("   R-squared: %.4f\n", summary(fe_model)$r.squared[1]))

    return(TRUE)
  }, error = function(e) {
    cat(sprintf("❌ Error: %s\n", e$message))
    return(FALSE)
  })
}

check_versions <- function() {
  cat("\n")
  cat(paste(rep("=", 50), collapse = ""), "\n")
  cat("Package Versions:\n")
  cat(paste(rep("=", 50), collapse = ""), "\n")

  cat(sprintf("R version:    %s\n", R.version.string))
  cat(sprintf("plm:          %s\n", packageVersion("plm")))
  cat(sprintf("lmtest:       %s\n", packageVersion("lmtest")))
  cat(sprintf("sandwich:     %s\n", packageVersion("sandwich")))

  cat(paste(rep("=", 50), collapse = ""), "\n")
}

# Run tests
cat("Testing R Environment for Panel Data Analysis\n\n")

imports_ok <- test_imports()

if (imports_ok) {
  functionality_ok <- test_basic_functionality()
} else {
  functionality_ok <- FALSE
}

check_versions()

if (imports_ok && functionality_ok) {
  cat("\n🎉 R environment setup complete and verified!\n")
} else {
  cat("\n⚠️  Some tests failed. Please check error messages above.\n")
}
```

Run in R:

```r
source("test_installation.R")
```

## Common Installation Issues

### Issue 1: Compiler Required for linearmodels

**Symptom:**
```
error: Microsoft Visual C++ 14.0 or greater is required
```
or
```
error: command 'gcc' failed with exit status 1
```

**Solution (Windows):**
1. Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Or install full Visual Studio with C++ support

**Solution (macOS):**
```bash
# Install Xcode Command Line Tools
xcode-select --install
```

**Solution (Linux):**
```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# CentOS/RHEL
sudo yum install gcc gcc-c++ python3-devel
```

### Issue 2: linearmodels Installation Fails

**Alternative using conda:**

```bash
conda install -c conda-forge linearmodels
```

This installs pre-compiled binaries, avoiding compiler issues.

### Issue 3: R Package Dependencies Missing

**Symptom:**
```
ERROR: dependencies 'X', 'Y' are not available for package 'plm'
```

**Solution:**
Install dependencies individually:

```r
install.packages("bdsmatrix")
install.packages("zoo")
install.packages("nlme")

# Then retry
install.packages("plm")
```

### Issue 4: Jupyter Kernel Not Found

**Symptom:** The "Python (Panel Data)" kernel doesn't appear in Jupyter.

**Solution:**

```bash
# Activate your environment
source panel_env/bin/activate  # or conda activate panel_env

# Reinstall kernel
python -m ipykernel install --user --name=panel_env --display-name="Python (Panel Data)" --force

# List available kernels
jupyter kernelspec list

# Restart Jupyter
```

### Issue 5: Version Conflicts

**Symptom:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages...
```

**Solution 1: Install with --no-deps and resolve manually:**

```bash
pip install --no-deps linearmodels
pip install scipy statsmodels pandas numpy  # Install dependencies
```

**Solution 2: Use specific versions:**

Create `requirements.txt`:

```
numpy>=1.22.0,<2.0.0
pandas>=1.4.0,<3.0.0
scipy>=1.8.0,<2.0.0
statsmodels>=0.13.0,<0.15.0
linearmodels>=5.0,<6.0
jupyter>=1.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
```

Install:

```bash
pip install -r requirements.txt
```

## Recommended Project Structure

```
panel_regression_project/
│
├── panel_env/                 # Virtual environment (gitignored)
│
├── data/
│   ├── raw/                   # Original data files
│   ├── processed/             # Cleaned panel data
│   └── README.md             # Data documentation
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_fixed_effects.ipynb
│   └── 04_model_selection.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py          # Data loading/cleaning functions
│   ├── panel_utils.py         # Panel-specific utilities
│   └── visualization.py       # Plotting functions
│
├── tests/
│   ├── test_data_utils.py
│   └── test_panel_utils.py
│
├── results/
│   ├── figures/
│   └── tables/
│
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment (alternative)
├── .gitignore
└── README.md
```

Create `.gitignore`:

```
# Virtual environments
panel_env/
venv/
env/

# Jupyter
.ipynb_checkpoints/
*/.ipynb_checkpoints/*

# Python
__pycache__/
*.py[cod]
*$py.class
.pytest_cache/

# Data (if large or sensitive)
data/raw/*.csv
data/raw/*.dta

# Results
*.log

# IDE
.vscode/
.idea/
.Rproj.user/
*.Rproj
```

## Verification Checklist

Before starting the course, verify:

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All packages installed (`pip list` shows linearmodels, statsmodels, pandas)
- [ ] Test script runs successfully
- [ ] Jupyter kernel configured and accessible
- [ ] Can import linearmodels in Jupyter notebook
- [ ] (If using R) plm package installed and working
- [ ] Project directory structure created

## Quick Reference Commands

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.

</div>


**Activate environment:**
```bash
# venv
source panel_env/bin/activate  # macOS/Linux
panel_env\Scripts\activate     # Windows

# conda
conda activate panel_env
```

**Deactivate environment:**
```bash
deactivate  # venv
conda deactivate  # conda
```

**Update packages:**
```bash
pip install --upgrade linearmodels statsmodels pandas
```

**Export environment:**
```bash
# For pip
pip freeze > requirements.txt

# For conda
conda env export > environment.yml
```

**Recreate environment on another machine:**
```bash
# From requirements.txt
pip install -r requirements.txt

# From environment.yml
conda env create -f environment.yml
```

## Further Reading

**Python Environment Management:**
- [Python Virtual Environments: A Primer](https://realpython.com/python-virtual-environments-a-primer/)
- [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html)

**linearmodels Documentation:**
- [linearmodels Official Documentation](https://bashtage.github.io/linearmodels/)
- [Panel Data Examples](https://bashtage.github.io/linearmodels/panel/examples/examples.html)

**plm (R) Documentation:**
- [plm: Linear Models for Panel Data](https://cran.r-project.org/web/packages/plm/vignettes/A_plmPackage.html)
- [Panel Data Econometrics in R: The plm Package](https://www.jstatsoft.org/article/view/v027i02)

**Best Practices:**
- [Reproducible Research in Computational Science](https://doi.org/10.1126/science.1213847)
- [Good Enough Practices in Scientific Computing](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510)


---

## Conceptual Practice Questions

**Practice Question 1:** What problem does this approach solve that simpler methods cannot?

**Practice Question 2:** What are the key assumptions, and how would you test them in practice?


---

## Cross-References

<a class="link-card" href="./01_ols_review.md">
  <div class="link-card-title">01 Ols Review</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_ols_review.md">
  <div class="link-card-title">01 Ols Review — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./01_panel_data_concepts.md">
  <div class="link-card-title">01 Panel Data Concepts</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_panel_data_concepts.md">
  <div class="link-card-title">01 Panel Data Concepts — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_data_structures.md">
  <div class="link-card-title">02 Data Structures</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_data_structures.md">
  <div class="link-card-title">02 Data Structures — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

