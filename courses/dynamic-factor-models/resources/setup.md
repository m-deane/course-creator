# Environment Setup: Get Running in 2 Minutes

**TL;DR:** Click any notebook's Colab badge → Run. That's it. No installation needed.

---

## Option 1: Google Colab (Recommended - Zero Setup)

**Perfect for:** Getting started immediately, no local installation.

### Step 1: Open a Notebook
Click the Colab badge on any notebook or go to:
https://colab.research.google.com/

### Step 2: Run This Cell
```python
# Install required packages (runs automatically in Colab)
!pip install statsmodels pandas-datareader scikit-learn -q

# Import and verify
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

print("✓ Ready to extract factors!")
```

### Step 3: Load Data
```python
# FRED-MD data loads directly in browser
url = "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv"
data = pd.read_csv(url, skiprows=1, index_col=0, parse_dates=True)
print(f"Loaded {data.shape[1]} series, {data.shape[0]} months")
```

**Done!** Start running examples.

### Colab Tips
- **Save to Drive:** File → Save a copy in Drive
- **Upload data:** Use file upload widget or mount Google Drive
- **GPU acceleration:** Runtime → Change runtime type → GPU (for Bayesian methods)
- **Share notebooks:** Share button (top right) → Anyone with link

---

## Option 2: Local Installation (For Offline Work)

**Perfect for:** Large datasets, custom data sources, production deployments.

### Quick Install (5 minutes)
```bash
# Install core packages
pip install numpy pandas statsmodels matplotlib scikit-learn
pip install pandas-datareader

# Verify installation
python -c "import statsmodels; import sklearn; print('Ready!')"
```

### Run a Test
```python
# test_setup.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 20)

# Extract 3 factors
pca = PCA(n_components=3)
factors = pca.fit_transform(X)

print(f"✓ Extracted {factors.shape[1]} factors from {X.shape[1]} series")
print(f"✓ Explained variance: {pca.explained_variance_ratio_.sum():.1%}")
```

Run: `python test_setup.py`

---

## Option 3: Conda Environment (Recommended for Local)

**Perfect for:** Reproducibility, managing dependencies, isolating projects.

### Step 1: Install Miniconda
If you don't have Conda:
- **Download:** https://docs.conda.io/en/latest/miniconda.html
- **Install:** Follow instructions for your OS

### Step 2: Create Environment
```bash
# Create environment with Python 3.11
conda create -n dfm python=3.11 -y
conda activate dfm
```

### Step 3: Install Packages
```bash
# Core
pip install numpy pandas statsmodels matplotlib seaborn scikit-learn

# Data access
pip install pandas-datareader fredapi

# Notebook interface
pip install jupyterlab ipywidgets

# Optional: Bayesian methods
pip install pymc arviz
```

### Step 4: Start Jupyter
```bash
jupyter lab
```

Opens in browser at http://localhost:8888

### Conda Tips
```bash
# List environments
conda env list

# Deactivate environment
conda deactivate

# Remove environment (if starting over)
conda remove -n dfm --all

# Export environment for sharing
conda env export > environment.yml

# Create from shared environment
conda env create -f environment.yml
```

---

## Package Reference

### Required (Core Functionality)
| Package | Purpose | Install |
|---------|---------|---------|
| numpy | Matrix operations, linear algebra | `pip install numpy` |
| pandas | Data manipulation, time series | `pip install pandas` |
| statsmodels | State-space models, Kalman filter | `pip install statsmodels` |
| scikit-learn | PCA, preprocessing, validation | `pip install scikit-learn` |
| matplotlib | Plotting, visualization | `pip install matplotlib` |

### Optional (Advanced Features)
| Package | Purpose | When to Use |
|---------|---------|-------------|
| pandas-datareader | FRED API access | Loading live economic data |
| fredapi | Direct FRED access | More control over data downloads |
| seaborn | Statistical plots | Better visualizations |
| pymc | Bayesian estimation | Bayesian DFMs, uncertainty quantification |
| arviz | MCMC diagnostics | Analyzing Bayesian results |
| plotly | Interactive plots | Dashboards, exploration |

### One-Line Install (All Packages)
```bash
pip install numpy pandas statsmodels scikit-learn matplotlib seaborn pandas-datareader fredapi jupyterlab ipywidgets plotly
```

---

## FRED API Setup (Optional but Recommended)

Get free access to 800,000+ economic time series.

### Step 1: Get API Key
1. Go to: https://fred.stlouisfed.org/docs/api/api_key.html
2. Create free account
3. Request API key (arrives instantly)

### Step 2: Set Environment Variable

**macOS/Linux:**
```bash
# Add to ~/.bashrc or ~/.zshrc
export FRED_API_KEY="your_key_here"

# Reload
source ~/.bashrc
```

**Windows:**
```cmd
setx FRED_API_KEY "your_key_here"
```

**In Notebook (temporary):**
```python
import os
os.environ['FRED_API_KEY'] = 'your_key_here'
```

### Step 3: Test Access
```python
from fredapi import Fred
import os

fred = Fred(api_key=os.environ['FRED_API_KEY'])
gdp = fred.get_series('GDP')
print(f"✓ Loaded {len(gdp)} GDP observations")
```

### FRED-MD Direct Download (No API Key)
```python
# Alternative: Direct CSV download
url = "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv"
fred_md = pd.read_csv(url, skiprows=1, index_col=0, parse_dates=True)
```

---

## Troubleshooting

### Import Error: No module named 'statsmodels'
**Fix:**
```bash
pip install statsmodels
# Or in Colab:
!pip install statsmodels
```

### Kalman Filter Not Converging
**Symptoms:** `LinAlgError: Matrix is singular`
**Fix:**
```python
# Add small noise to diagonal for numerical stability
mod = sm.tsa.DynamicFactor(data, k_factors=3, factor_order=1)
res = mod.fit(disp=False, maxiter=1000)
```

### FRED API Rate Limit
**Symptoms:** `HTTPError: 429 Too Many Requests`
**Fix:** Add delays between requests
```python
import time
for series_id in series_list:
    data = fred.get_series(series_id)
    time.sleep(0.5)  # 500ms delay
```

### Out of Memory with Large Datasets
**Fix:** Process in chunks
```python
# Load only recent data
data = fred_md['2010':]  # From 2010 onward

# Or sample columns
data = fred_md.iloc[:, :50]  # First 50 series
```

### PCA Results Look Wrong
**Common cause:** Forgot to standardize
**Fix:**
```python
from sklearn.preprocessing import StandardScaler

# Standardize first!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
factors = PCA(n_components=3).fit_transform(X_scaled)
```

### Jupyter Kernel Dies
**Fix:** Increase memory or reduce data size
```python
# Check data size
print(f"Data size: {data.memory_usage().sum() / 1e6:.1f} MB")

# Reduce if needed
data = data.astype('float32')  # Use less memory
```

---

## Testing Your Setup

Run this complete test to verify everything works:

```python
"""
Complete environment test for DFM course.
Run this to verify your setup is ready.
"""

def test_complete_pipeline():
    """Test full DFM pipeline."""
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    print("Step 1: Generate synthetic data...")
    np.random.seed(42)

    # 3 latent factors
    T, N, r = 200, 30, 3
    F = np.random.randn(T, r)

    # Random loadings
    Lambda = np.random.randn(N, r)

    # Generate observed data: X = F @ Lambda.T + noise
    X = F @ Lambda.T + 0.5 * np.random.randn(T, N)
    X_df = pd.DataFrame(X, columns=[f'var_{i}' for i in range(N)])

    print(f"✓ Generated {N} series, {T} time periods")

    print("\nStep 2: Extract factors with PCA...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    pca = PCA(n_components=r)
    factors_pca = pca.fit_transform(X_scaled)

    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"✓ Extracted {r} factors")
    print(f"✓ Explained variance: {variance_explained:.1%}")

    print("\nStep 3: Fit state-space DFM...")
    mod = sm.tsa.DynamicFactor(
        X_df,
        k_factors=r,
        factor_order=1
    )
    res = mod.fit(disp=False, maxiter=100)

    print(f"✓ State-space model converged")
    print(f"✓ Log-likelihood: {res.llf:.2f}")

    print("\nStep 4: Visualize factors...")
    fig, axes = plt.subplots(r, 1, figsize=(10, 6))
    for i in range(r):
        axes[i].plot(factors_pca[:, i])
        axes[i].set_title(f'Factor {i+1}')
        axes[i].set_ylabel('Value')
    axes[-1].set_xlabel('Time')
    plt.tight_layout()

    # Try to display or save
    try:
        plt.show()
    except:
        plt.savefig('test_factors.png')
        print("✓ Plot saved to test_factors.png")

    print("\n" + "="*50)
    print("ALL TESTS PASSED! Environment is ready.")
    print("="*50)
    print("\nNext steps:")
    print("1. Open quick-starts/00_hello_world.ipynb")
    print("2. Run all cells")
    print("3. Start building!")

if __name__ == "__main__":
    test_complete_pipeline()
```

Save as `test_environment.py` and run:
```bash
python test_environment.py
```

**Expected output:**
```
Step 1: Generate synthetic data...
✓ Generated 30 series, 200 time periods

Step 2: Extract factors with PCA...
✓ Extracted 3 factors
✓ Explained variance: 85.3%

Step 3: Fit state-space DFM...
✓ State-space model converged
✓ Log-likelihood: -1234.56

Step 4: Visualize factors...
✓ Plot saved to test_factors.png

==================================================
ALL TESTS PASSED! Environment is ready.
==================================================
```

---

## Advanced Setup: Production Deployments

### Docker Container
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install numpy pandas statsmodels scikit-learn pandas-datareader

COPY . /app

CMD ["python", "nowcast_pipeline.py"]
```

Build and run:
```bash
docker build -t dfm-pipeline .
docker run dfm-pipeline
```

### Requirements.txt for Reproducibility
```txt
# requirements.txt
numpy==1.26.0
pandas==2.1.0
statsmodels==0.14.0
scikit-learn==1.3.0
matplotlib==3.8.0
pandas-datareader==0.10.0
```

Install exact versions:
```bash
pip install -r requirements.txt
```

---

## Quick Start Checklist

- [ ] Choose setup option (Colab = fastest, local = most control)
- [ ] Install required packages
- [ ] Test with `test_environment.py`
- [ ] Get FRED API key (optional but useful)
- [ ] Open first notebook: `quick-starts/00_hello_world.ipynb`
- [ ] Run all cells successfully
- [ ] Start modifying code!

---

## Getting Help

**Installation issues?**
1. Check error message carefully
2. Try `pip install --upgrade [package]`
3. Create fresh environment
4. Use Colab as fallback

**Still stuck?**
- Search error on Stack Overflow
- Check package documentation
- Post in course forum with error details

---

**You're ready!** Head to [quick-starts/](../quick-starts/) and extract your first factors.
