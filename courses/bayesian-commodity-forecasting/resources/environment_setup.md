# Environment Setup Guide

## Quick Setup

### Option 1: Conda (Recommended)

```bash
# Create environment
conda create -n bayes-commodity python=3.11 -y
conda activate bayes-commodity

# Install PyMC (handles complex dependencies)
conda install -c conda-forge pymc arviz -y

# Additional packages
pip install yfinance pandas-datareader seaborn plotly
pip install jupyterlab ipywidgets nbformat

# Optional: NumPyro backend (faster on some systems)
pip install numpyro jax jaxlib
```

### Option 2: pip/venv

```bash
# Create virtual environment
python -m venv bayes-commodity-env
source bayes-commodity-env/bin/activate  # Linux/Mac
# bayes-commodity-env\Scripts\activate  # Windows

# Install packages
pip install --upgrade pip
pip install pymc arviz numpy pandas scipy matplotlib seaborn
pip install yfinance fredapi
pip install jupyterlab ipywidgets

# Register Jupyter kernel
python -m ipykernel install --user --name=bayes-commodity
```

---

## Package Versions (Tested)

```
python>=3.11
pymc>=5.10
arviz>=0.17
numpy>=1.24
pandas>=2.0
scipy>=1.11
matplotlib>=3.7
seaborn>=0.12
yfinance>=0.2
jupyterlab>=4.0
```

---

## Verification

Run this in Python to verify installation:

```python
import sys
print(f"Python: {sys.version}")

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"PyMC: {pm.__version__}")
print(f"ArviZ: {az.__version__}")

# Quick test
with pm.Model():
    x = pm.Normal('x', 0, 1)
    trace = pm.sample(100, tune=50, cores=1, progressbar=False)

print("PyMC sampling: OK")
print("\n✅ Environment ready!")
```

---

## Common Issues

### Issue: PyMC installation fails

**Solution:** Use conda instead of pip:
```bash
conda install -c conda-forge pymc
```

### Issue: Theano/PyTensor errors

**Solution:** PyMC 5.x uses PyTensor. Ensure you don't have old Theano installed:
```bash
pip uninstall theano theano-pymc
pip install --upgrade pymc
```

### Issue: Slow sampling

**Solutions:**
1. Reduce `cores` parameter: `pm.sample(cores=1)`
2. Try NumPyro backend:
   ```python
   import pymc as pm
   pm.sample(nuts_sampler="numpyro")
   ```

### Issue: Jupyter kernel not found

**Solution:**
```bash
python -m ipykernel install --user --name=bayes-commodity
```

### Issue: yfinance rate limiting

**Solution:** Add delays between requests:
```python
import time
time.sleep(1)  # Wait 1 second between API calls
```

---

## Data API Keys (Optional)

### FRED API
1. Register at https://fred.stlouisfed.org/docs/api/api_key.html
2. Set environment variable:
   ```bash
   export FRED_API_KEY="your_key_here"
   ```

### Quandl
1. Register at https://www.quandl.com/sign-up
2. Set in code:
   ```python
   import quandl
   quandl.ApiConfig.api_key = "your_key_here"
   ```

---

## IDE Recommendations

### VS Code
- Install Python extension
- Install Jupyter extension
- Enable notebook outline for navigation

### JupyterLab
- Install TOC extension for navigation
- Enable variable inspector

---

## GPU Acceleration (Optional)

For faster inference on large models:

### JAX with GPU
```bash
# Check CUDA version first
nvidia-smi

# Install appropriate JAX version
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Verify GPU
```python
import jax
print(jax.devices())  # Should show GPU
```

---

## Docker Setup (Advanced)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN pip install pymc arviz numpy pandas matplotlib seaborn jupyterlab yfinance

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

Build and run:
```bash
docker build -t bayes-commodity .
docker run -p 8888:8888 -v $(pwd):/app bayes-commodity
```

---

## Support

If you encounter issues:
1. Check this guide's troubleshooting section
2. Search the course forum
3. Post your error message with:
   - Operating system
   - Python version
   - Package versions (`pip freeze`)
   - Full error traceback
