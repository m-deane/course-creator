# Figures and Visual Assets - Module 04

This directory contains visual assets supporting Module 04 (Advanced Extensions).

## Available Figures

Place generated figures here with descriptive names:

### Time-Varying Parameters
- `tvp_loading_evolution.png` - How loadings change over time
- `rolling_window_comparison.png` - Constant vs rolling window forecasts
- `structural_break_detection.png` - Pre/post break parameter estimates
- `forgetting_factor_weights.png` - Exponential decay visualization

### Mixed-Frequency
- `midas_lag_structure.png` - MIDAS weight function visualization
- `skip_sampling_pattern.png` - Monthly-quarterly observation pattern
- `temporal_aggregation.png` - Flow vs stock aggregation diagram
- `mixed_freq_nowcast_evolution.png` - Nowcast updates as monthly data arrives

### Large Datasets
- `scree_plot.png` - Eigenvalue decay for factor number selection
- `sparse_loadings_heatmap.png` - Sparse vs dense loading matrix
- `computational_comparison.png` - Two-step vs full MLE timing
- `fred_md_factors.png` - Extracted factors from FRED-MD

## Creating Figures

Standard settings for consistency:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (12, 6)

# Create visualization
fig, ax = plt.subplots()
# ... plotting code ...

# Save
plt.savefig('figures/my_figure.png', bbox_inches='tight', dpi=300)
plt.savefig('figures/my_figure.pdf', bbox_inches='tight')
```

## ASCII Diagrams (From Guides)

### Time-Varying Parameters Evolution
```
Constant vs Time-Varying:

CONSTANT:
   λ_{IP} = 0.8  (always)
   2000  2008  2015  2020  2024
   ──────────────────────────────→
    0.8   0.8   0.8   0.8   0.8

TIME-VARYING:
   λ_{IP,t} evolves over time
   2000  2008  2015  2020  2024
   ──────────────────────────────→
    0.8   0.6   0.7   0.3   0.5
           ↓            ↓
    Financial     COVID-19
     Crisis      Shock
```

### Mixed-Frequency Skip-Sampling
```
Monthly GDP observations:
t:    1   2   3 | 4   5   6 | 7   8   9
GDP:  ✗   ✗   ✓ | ✗   ✗   ✓ | ✗   ✗   ✓
IP:   ✓   ✓   ✓ | ✓   ✓   ✓ | ✓   ✓   ✓
```

### Two-Step Estimation Flow
```
DATA: X (T x N)
  ↓
[STEP 1: PCA]
  ├→ Compute C = X'X / T
  ├→ Eigendecomp: C = Λ̂·Λ̂'
  └→ F̂ = X·Λ̂ / √T
  ↓
[STEP 2: Kalman Smoother]
  ├→ Estimate VAR on F̂
  ├→ Kalman smooth
  └→ Refined F̃
  ↓
OUTPUT: Factors F̃, Loadings Λ̂
```

## Example: Time-Varying Loading Plot

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Simulate rolling window loadings
T = 200
dates = pd.date_range('2005-01-01', periods=T, freq='M')
loadings_over_time = []

for t in range(60, T):
    # True loading has structural break at t=100
    loading = 0.8 if t < 100 else 0.4
    # Add estimation noise
    loading_est = loading + 0.1 * np.random.randn()
    loadings_over_time.append(loading_est)

# Plot
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(dates[60:], loadings_over_time, linewidth=2, label='Estimated Loading')
ax.axvline(dates[100], color='red', linestyle='--', linewidth=2, label='Structural Break')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Factor Loading', fontsize=12)
ax.set_title('Time-Varying Factor Loading: Rolling Window Estimation',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/tvp_loading_evolution.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Example: Scree Plot for Factor Selection

```python
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Generate data
T, N = 200, 50
X = np.random.randn(T, N)

# PCA
pca = PCA()
pca.fit(X)

# Scree plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, 21), pca.explained_variance_ratio_[:20],
        'o-', linewidth=2, markersize=8)
ax.axvline(5, color='red', linestyle='--', alpha=0.7, label='Selected: 5 factors')
ax.set_xlabel('Factor Number', fontsize=12)
ax.set_ylabel('Proportion of Variance Explained', fontsize=12)
ax.set_title('Scree Plot: Factor Number Selection', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/scree_plot.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Example: Sparse vs Dense Loading Heatmap

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate loadings
N, r = 20, 3

# Dense loadings
dense_loadings = np.random.randn(N, r) * 0.5

# Sparse loadings (many zeros)
sparse_loadings = dense_loadings.copy()
sparse_loadings[np.abs(sparse_loadings) < 0.3] = 0

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

sns.heatmap(dense_loadings, cmap='RdBu_r', center=0,
            ax=axes[0], cbar_kws={'label': 'Loading'})
axes[0].set_title('Dense Loadings', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Factor', fontsize=11)
axes[0].set_ylabel('Variable', fontsize=11)

sns.heatmap(sparse_loadings, cmap='RdBu_r', center=0,
            ax=axes[1], cbar_kws={'label': 'Loading'})
axes[1].set_title('Sparse Loadings (LASSO)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Factor', fontsize=11)
axes[1].set_ylabel('Variable', fontsize=11)

plt.tight_layout()
plt.savefig('figures/sparse_loadings_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

sparsity = (sparse_loadings == 0).sum() / sparse_loadings.size * 100
print(f"Sparsity: {sparsity:.1f}% of loadings are zero")
```

## Figure Naming Convention

Use descriptive names with module prefix:

- `mod04_tvp_*.png` - Time-varying parameter figures
- `mod04_mf_*.png` - Mixed-frequency figures
- `mod04_large_*.png` - Large dataset figures
- `mod04_comparison_*.png` - Comparison figures

## Requested Figures

If you need specific visualizations created, please:
1. Open an issue with description
2. Provide example data or simulation code
3. Specify intended use (lecture, paper, etc.)

## License

All figures in this directory are released under the same license as the course materials (typically CC-BY-4.0 or similar). Include attribution when reusing.

---

*Note: This directory will be populated as course materials are developed. Students are encouraged to contribute high-quality figures from their exercises and projects.*
