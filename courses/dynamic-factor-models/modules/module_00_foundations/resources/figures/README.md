# Module 00: Figures and Visual Assets

This directory contains diagrams and visual aids for understanding state space models and the Kalman filter.

## Available Figures

All visual explanations are currently embedded as ASCII diagrams in the guides and notebooks for maximum portability. If you need high-resolution figures for presentations, generate them using the code provided in the notebooks.

## Generate Your Own Figures

### State Space Model Diagram
Use the visualization code in `notebooks/01_state_space_intro.ipynb` to create publication-quality plots showing:
- Hidden state evolution
- Noisy observations
- State vs observation separation

### Kalman Filter Cycle
Use `notebooks/02_kalman_filter_visual.ipynb` to generate:
- Predict-update cycle visualization
- Kalman gain convergence
- Innovation diagnostics
- Missing data handling

### Creating Custom Figures

```python
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2

# Your visualization code here
# ...

# Save high-resolution
plt.savefig('figure_name.png', dpi=300, bbox_inches='tight')
plt.savefig('figure_name.pdf', bbox_inches='tight')  # Vector format
```

## Recommended Additions

Students are encouraged to create and contribute:

1. **State Space Architecture Diagram**
   - Show T, Z, R, Q, H matrices visually
   - Illustrate dimensions and data flow

2. **Kalman Gain Intuition**
   - Plot K vs signal-to-noise ratio
   - Show optimal blending of model and data

3. **Comparison Plots**
   - Filtered vs smoothed estimates
   - Different initialization strategies
   - Missing data interpolation methods

4. **Innovation Diagnostics**
   - ACF/PACF plots
   - Q-Q plots for normality
   - Histogram of standardized innovations

## Figure Naming Convention

Use descriptive names:
- `state_space_structure.png` - Not `figure1.png`
- `kalman_gain_convergence.png` - Not `plot.png`
- `innovation_acf.png` - Not `diag.png`

## Attribution

When using figures in presentations or papers:
- Figures generated from course notebooks: No attribution needed
- Modified versions: Note "Adapted from [course name]"
- Original diagrams from textbooks: Cite original source

## Tools for Creating Diagrams

**Recommended:**
- **Matplotlib/Seaborn**: Code-generated plots (reproducible)
- **tikz/pgfplots**: LaTeX integration for papers
- **draw.io**: Quick conceptual diagrams
- **Inkscape**: Vector graphics editing

**For presentations:**
- Export as high-res PNG (300 DPI) or PDF
- Use consistent color schemes
- Ensure text is readable at small sizes

## Color Schemes

Suggested colors for consistency:

```python
COLORS = {
    'true_state': 'black',
    'observations': 'gray',
    'filtered': 'red',
    'predicted': 'blue',
    'smoothed': 'green',
    'confidence': 'red' (with alpha=0.2)
}
```

This ensures visual consistency across all module figures.
