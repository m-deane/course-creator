# Environment Setup Guide

## Overview

This guide walks you through setting up your development environment for the Dynamic Factor Models course. We use Python with scientific computing libraries, state-space modeling tools, and data access APIs.

---

## Step 1: Install Conda

If you don't have Anaconda or Miniconda installed:

### Option A: Miniconda (Recommended - Lightweight)
```bash
# macOS
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh

# Linux
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Windows: Download installer from https://docs.conda.io/en/latest/miniconda.html
```

### Option B: Anaconda (Full Distribution)
Download from https://www.anaconda.com/download

---

## Step 2: Create Course Environment

```bash
# Create new environment with Python 3.11
conda create -n dfm-course python=3.11 -y

# Activate environment
conda activate dfm-course
```

---

## Step 3: Install Core Packages

### Scientific Computing Stack
```bash
pip install numpy scipy pandas matplotlib seaborn
```

### Econometrics & Statistics
```bash
pip install statsmodels linearmodels arch
pip install scikit-learn
```

### Bayesian & State-Space
```bash
pip install pymc arviz
pip install numpyro jax jaxlib
```

### Data Access
```bash
pip install pandas-datareader fredapi yfinance
```

### Jupyter & Visualization
```bash
pip install jupyterlab ipywidgets plotly
```

### Development Tools
```bash
pip install pytest black isort
```

---

## Step 4: Verify Installation

Create a test file `test_environment.py`:

```python
"""Environment verification script for DFM course."""

def test_imports():
    """Test all required packages import successfully."""
    # Core
    import numpy as np
    import scipy
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Econometrics
    import statsmodels.api as sm
    from statsmodels.tsa.statespace.mlemodel import MLEModel
    import sklearn

    # Bayesian
    import pymc as pm
    import arviz as az

    # Data
    import pandas_datareader as pdr
    from fredapi import Fred

    print("All imports successful!")

def test_numpy_operations():
    """Test basic NumPy linear algebra."""
    import numpy as np

    # Create random matrix
    X = np.random.randn(100, 10)

    # SVD
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # Eigendecomposition of covariance
    cov = X.T @ X / 100
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    print(f"SVD: {U.shape}, {s.shape}, {Vt.shape}")
    print(f"Eigenvalues: {eigenvalues[-3:]}")
    print("NumPy operations successful!")

def test_statsmodels_statespace():
    """Test state-space model functionality."""
    import numpy as np
    import statsmodels.api as sm

    # Generate simple AR(1) data
    np.random.seed(42)
    T = 100
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.8 * y[t-1] + np.random.randn()

    # Fit using state-space framework
    mod = sm.tsa.UnobservedComponents(y, level='local level')
    res = mod.fit(disp=False)

    print(f"State-space model fitted, AIC: {res.aic:.2f}")
    print("Statsmodels state-space successful!")

def test_pymc():
    """Test PyMC installation."""
    import pymc as pm
    import numpy as np

    # Simple Bayesian model
    np.random.seed(42)
    y = np.random.randn(50) + 2

    with pm.Model() as model:
        mu = pm.Normal('mu', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=1)
        likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)

        # Just compile, don't sample (too slow for test)
        prior = pm.sample_prior_predictive(samples=10)

    print("PyMC model compiled successfully!")

def test_fred_api():
    """Test FRED data access (requires API key)."""
    import os
    from fredapi import Fred

    api_key = os.environ.get('FRED_API_KEY')
    if api_key:
        fred = Fred(api_key=api_key)
        gdp = fred.get_series('GDP', observation_start='2020-01-01')
        print(f"FRED API working, retrieved {len(gdp)} GDP observations")
    else:
        print("FRED_API_KEY not set - skipping API test")
        print("Get your free key at: https://fred.stlouisfed.org/docs/api/api_key.html")

if __name__ == "__main__":
    print("=" * 50)
    print("DFM Course Environment Verification")
    print("=" * 50)

    test_imports()
    print()

    test_numpy_operations()
    print()

    test_statsmodels_statespace()
    print()

    test_pymc()
    print()

    test_fred_api()
    print()

    print("=" * 50)
    print("Environment setup complete!")
    print("=" * 50)
```

Run verification:
```bash
python test_environment.py
```

---

## Step 5: Configure FRED API Access

### Get API Key
1. Go to https://fred.stlouisfed.org/docs/api/api_key.html
2. Create a free account
3. Request an API key

### Set Environment Variable

**macOS/Linux:**
```bash
# Add to ~/.bashrc or ~/.zshrc
export FRED_API_KEY="your_api_key_here"

# Reload shell
source ~/.bashrc
```

**Windows:**
```cmd
setx FRED_API_KEY "your_api_key_here"
```

**In Python (temporary):**
```python
import os
os.environ['FRED_API_KEY'] = 'your_api_key_here'
```

---

## Step 6: Launch Jupyter Lab

```bash
# Activate environment
conda activate dfm-course

# Start Jupyter Lab
jupyter lab
```

Navigate to course notebooks in your browser.

---

## Troubleshooting

### JAX Installation Issues (Apple Silicon)
```bash
# For M1/M2 Macs, use metal backend
pip install jax-metal
```

### PyMC Compilation Errors
```bash
# Install C compiler on macOS
xcode-select --install

# On Linux, ensure gcc is installed
sudo apt-get install build-essential
```

### Memory Issues with Large Datasets
```python
# Use chunked reading for FRED-MD
import pandas as pd
df = pd.read_csv('fred-md.csv', chunksize=1000)
```

### Package Conflicts
```bash
# Create fresh environment if conflicts occur
conda deactivate
conda remove -n dfm-course --all
conda create -n dfm-course python=3.11 -y
# Reinstall packages
```

---

## Optional: GPU Acceleration

For faster Bayesian inference (MCMC):

### NVIDIA GPU
```bash
pip install numpyro[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Apple Silicon (M1/M2)
```bash
pip install jax-metal
```

---

## Package Versions

For reproducibility, here are tested package versions:

```
numpy==1.26.0
scipy==1.11.3
pandas==2.1.1
matplotlib==3.8.0
seaborn==0.13.0
statsmodels==0.14.0
scikit-learn==1.3.1
pymc==5.10.0
arviz==0.17.0
numpyro==0.13.2
jax==0.4.20
pandas-datareader==0.10.0
fredapi==0.5.1
jupyterlab==4.0.7
```

Save to `requirements.txt` and install:
```bash
pip install -r requirements.txt
```

---

## Next Steps

After environment setup:

1. Run `test_environment.py` to verify installation
2. Complete Module 0 diagnostic assessment
3. Review prerequisite materials if needed
4. Begin Module 1 notebooks

---

## Support

If you encounter issues:

1. Check the course FAQ in `resources/faq.md`
2. Search/post in course discussion forum
3. Attend office hours for complex debugging
