# Setup Guide: Multi-Armed Bandits & A/B Testing Course

## Environment Setup

### Option 1: Conda (Recommended)

```bash
# Create environment
conda create -n bandits python=3.11
conda activate bandits

# Core packages
conda install numpy pandas matplotlib seaborn scipy

# Scientific computing
conda install scikit-learn statsmodels

# Bayesian tools
conda install -c conda-forge pymc arviz

# Data sources
pip install yfinance fredapi

# Notebooks
conda install jupyterlab ipywidgets plotly

# Verify installation
python -c "import numpy, pandas, scipy, pymc; print('✅ All packages installed')"
```

### Option 2: pip + venv

```bash
# Create virtual environment
python3.11 -m venv bandits-env
source bandits-env/bin/activate  # On Windows: bandits-env\Scripts\activate

# Install packages
pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scipy
pip install scikit-learn statsmodels
pip install pymc arviz
pip install yfinance fredapi
pip install jupyterlab ipywidgets plotly

# Verify
python -c "import numpy, pandas, scipy, pymc; print('✅ Setup complete')"
```

## Package Versions (Tested)

```
numpy==1.26.0
pandas==2.1.0
matplotlib==3.8.0
seaborn==0.13.0
scipy==1.11.0
scikit-learn==1.3.0
statsmodels==0.14.0
pymc==5.9.0
arviz==0.16.0
yfinance==0.2.28
fredapi==0.5.1
jupyterlab==4.0.6
ipywidgets==8.1.1
plotly==5.17.0
```

## Data Source Setup

### Yahoo Finance (yfinance)

No API key needed for basic usage:

```python
import yfinance as yf

# Test commodity data access
wti = yf.download('CL=F', start='2023-01-01', end='2024-01-01')
print(f"✅ Loaded {len(wti)} days of WTI data")
```

**Commodity Tickers:**
- `CL=F` - WTI Crude Oil
- `GC=F` - Gold
- `HG=F` - Copper
- `NG=F` - Natural Gas
- `ZC=F` - Corn
- `ZS=F` - Soybeans
- `KC=F` - Coffee
- `SI=F` - Silver

### FRED API (Optional)

For macro indicators:

1. Get API key: https://fred.stlouisfed.org/docs/api/api_key.html
2. Set environment variable:

```bash
export FRED_API_KEY='your_api_key_here'
```

3. Test:

```python
from fredapi import Fred
import os

fred = Fred(api_key=os.getenv('FRED_API_KEY'))
vix = fred.get_series('VIXCLS', observation_start='2023-01-01')
print(f"✅ Loaded {len(vix)} days of VIX data")
```

**Useful Series:**
- `VIXCLS` - VIX (volatility index)
- `DTWEXBGS` - Dollar index
- `EFFR` - Federal funds rate
- `T10Y2Y` - 10Y-2Y yield spread
- `DCOILWTICO` - WTI spot price

## Jupyter Setup

### Launch Jupyter Lab

```bash
conda activate bandits
jupyter lab
```

### Enable Interactive Widgets

For interactive visualizations:

```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### Test in Notebook

```python
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact

# Test interactive plot
@interact(n=(10, 1000, 10))
def plot_bandit_regret(n):
    regret = np.log(np.arange(1, n+1))
    plt.plot(regret)
    plt.xlabel('Rounds')
    plt.ylabel('Regret')
    plt.title(f'Logarithmic Regret (n={n})')
    plt.show()
```

## Verification Script

Run this to verify your setup:

```python
#!/usr/bin/env python
"""
Course Setup Verification Script
Run this to check that all dependencies are correctly installed.
"""

import sys

def check_package(name, import_name=None):
    """Check if a package is installed and importable."""
    if import_name is None:
        import_name = name

    try:
        __import__(import_name)
        print(f"✅ {name:20s} installed")
        return True
    except ImportError:
        print(f"❌ {name:20s} MISSING")
        return False

def check_data_source(name, test_func):
    """Check if a data source is accessible."""
    try:
        test_func()
        print(f"✅ {name:20s} accessible")
        return True
    except Exception as e:
        print(f"⚠️  {name:20s} issue: {str(e)[:50]}")
        return False

def main():
    print("="*60)
    print("MULTI-ARMED BANDITS COURSE - SETUP VERIFICATION")
    print("="*60 + "\n")

    # Core packages
    print("Core Packages:")
    core = [
        ('NumPy', 'numpy'),
        ('Pandas', 'pandas'),
        ('Matplotlib', 'matplotlib'),
        ('SciPy', 'scipy'),
        ('Seaborn', 'seaborn'),
    ]
    core_ok = all(check_package(name, imp) for name, imp in core)

    # Scientific computing
    print("\nScientific Computing:")
    sci = [
        ('scikit-learn', 'sklearn'),
        ('statsmodels', 'statsmodels'),
    ]
    sci_ok = all(check_package(name, imp) for name, imp in sci)

    # Bayesian
    print("\nBayesian Tools:")
    bayes = [
        ('PyMC', 'pymc'),
        ('ArviZ', 'arviz'),
    ]
    bayes_ok = all(check_package(name, imp) for name, imp in bayes)

    # Data sources
    print("\nData Sources:")
    data = [
        ('yfinance', 'yfinance'),
        ('fredapi', 'fredapi'),
    ]
    data_ok = all(check_package(name, imp) for name, imp in data)

    # Notebooks
    print("\nNotebook Tools:")
    nb = [
        ('JupyterLab', 'jupyterlab'),
        ('ipywidgets', 'ipywidgets'),
        ('Plotly', 'plotly'),
    ]
    nb_ok = all(check_package(name, imp) for name, imp in nb)

    # Test data access
    print("\nData Access Tests:")
    def test_yfinance():
        import yfinance as yf
        data = yf.download('CL=F', start='2024-01-01', end='2024-01-07', progress=False)
        if len(data) == 0:
            raise ValueError("No data returned")

    def test_fred():
        import os
        if not os.getenv('FRED_API_KEY'):
            raise ValueError("FRED_API_KEY not set (optional)")

    check_data_source('Yahoo Finance', test_yfinance)
    check_data_source('FRED API', test_fred)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_ok = core_ok and sci_ok and bayes_ok and data_ok and nb_ok

    if all_ok:
        print("✅ All required packages installed correctly!")
        print("\nYou're ready to start the course.")
        print("Run: jupyter lab")
        return 0
    else:
        print("❌ Some packages are missing.")
        print("\nInstall missing packages:")
        print("  conda install <package>")
        print("  or")
        print("  pip install <package>")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

Save as `verify_setup.py` and run:

```bash
python verify_setup.py
```

## Troubleshooting

### Issue: PyMC installation fails

```bash
# Try installing from conda-forge
conda install -c conda-forge pymc

# Or install minimal version
pip install pymc --no-deps
pip install arviz pytensor
```

### Issue: yfinance returns no data

```python
# Common fixes:
import yfinance as yf

# 1. Update yfinance
pip install --upgrade yfinance

# 2. Use different date range
data = yf.download('CL=F', period='1mo')

# 3. Check ticker validity
ticker = yf.Ticker('CL=F')
print(ticker.info)
```

### Issue: Jupyter widgets not displaying

```bash
# Enable widget extension
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

### Issue: ImportError for specific modules

```python
# Check Python path
import sys
print('\n'.join(sys.path))

# Check package location
import numpy
print(numpy.__file__)

# Reinstall if needed
pip uninstall numpy
pip install numpy
```

## Alternative: Google Colab

If local setup fails, use Google Colab (free):

1. Go to https://colab.research.google.com
2. Upload course notebooks
3. Install packages in notebook:

```python
# First cell of every notebook
!pip install yfinance pymc arviz
```

All course materials work in Colab.

## Next Steps

Once setup is complete:

1. Clone course repository (or download materials)
2. Navigate to `courses/multi-armed-bandits-ab-testing/`
3. Start with `quick-starts/00_your_first_bandit.ipynb`
4. Or follow structured path: `modules/module_00_foundations/`

## Getting Help

If you encounter issues:

1. Check package versions: `pip list | grep <package>`
2. Try in a fresh environment: `conda create -n test python=3.11`
3. Search error messages: Most issues documented on Stack Overflow
4. Use synthetic data: Course provides fallbacks for data issues

---

**You're ready to start learning bandits!** 🎰
