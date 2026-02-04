# Figures and Visual Assets

This directory contains visual assets supporting Module 03 (Applications).

## Available Figures

When creating visualizations for this module, place them here with descriptive names:

### Nowcasting Diagrams
- `nowcasting_pipeline.png` - Full nowcasting workflow diagram
- `ragged_edge_timeline.png` - Publication lag timeline visualization
- `news_decomposition.png` - News vs uncertainty decomposition chart

### Forecast Evaluation
- `forecast_comparison.png` - DFM vs benchmarks over time
- `error_distribution.png` - Forecast error histograms
- `dm_test_results.png` - Diebold-Mariano test visualization

### Missing Data
- `kalman_missing_data_flow.png` - Kalman filter update with missing obs
- `uncertainty_evolution.png` - Factor uncertainty during ragged edge
- `imputation_comparison.png` - Comparing imputation methods

## Creating Figures

All figures should:
- Be high resolution (300 DPI minimum)
- Use consistent color scheme (matplotlib default or seaborn)
- Include clear axis labels and legends
- Be saved in both PNG (for notebooks) and PDF (for papers/presentations)

## Example Code to Generate Figures

```python
import matplotlib.pyplot as plt

# Standard figure settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (10, 6)

# Create your visualization
fig, ax = plt.subplots()
# ... plotting code ...

# Save with consistent naming
plt.savefig('figures/my_figure.png', bbox_inches='tight', dpi=300)
plt.savefig('figures/my_figure.pdf', bbox_inches='tight')
```

## ASCII Diagrams (For Guides)

ASCII diagrams in the guides (see `guides/*.md`) can be copied here for reference:

### Nowcasting Timeline
```
Timeline of Information Flow:
════════════════════════════════════════════════════════════════

Q1 Ends              GDP Advance      GDP Final
(March 31)           Estimate         Estimate
    │                    │                │
    │<---4-6 weeks------>│<--2 months---->│
    │                    │                │
    ▼                    ▼                ▼
┌───────────────────────────────────────────────────────────┐
│  Jan   Feb   Mar  │ Apr   May   Jun  │ Jul   Aug   Sep   │
│   ▲     ▲     ▲    │                  │                   │
│   │     │     │    │                  │                   │
│  IP₁   IP₂   IP₃  │ We wait here --> │ Finally get GDP!  │
│  EMP₁  EMP₂  EMP₃ │ with no data     │                   │
│  PMI₁  PMI₂  PMI₃ │                  │                   │
└───────────────────────────────────────────────────────────┘
         │
         └──> DFM Nowcast: "Q1 GDP ≈ 2.3% ± 0.8%"
              Available: April 15 (2 weeks before official!)
```

### Ragged Edge Data Matrix
```
         Jan    Feb    Mar   │ Apr (nowcasting month)
      ┌───────────────────────┼──────────────────────┐
IP    │  ✓      ✓      ✓     │  ?   (released Apr 15)
EMP   │  ✓      ✓      ✓     │  ✓   (released Apr 5)
Sales │  ✓      ✓      ?     │  ?   (released Apr 12)
PMI   │  ✓      ✓      ✓     │  ✓   (flash estimate)
GDP   │  ─────────  Q1 ──────│  ?   (released Apr 30)
      └───────────────────────┴──────────────────────┘

    ✓ = Observed    ? = Missing    ─ = Quarterly variable
```

## Figure Requests

If you need specific visualizations created, please:
1. Open an issue describing the figure
2. Provide example data or code
3. Specify intended use (lecture, paper, etc.)

---

*Note: This directory will be populated as course materials are developed. Students are encouraged to contribute figures from their exercises and projects.*
