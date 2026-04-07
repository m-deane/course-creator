# Module 01: Point Forecasting with Neural Models

## Learning Objectives

By the end of this module you will be able to:

1. Describe the architectural differences between MLP, N-BEATS, NHITS, PatchTST, and DLinear
2. Explain how NHITS uses multi-rate signal sampling and hierarchical interpolation to handle multiple frequency bands simultaneously
3. Train NHITS on real time series data using the NeuralForecast library
4. Select appropriate values for `input_size`, `scaler_type`, and `loss` for a given dataset
5. Evaluate forecast quality using MAE and MSE from `utilsforecast`
6. Compare models fairly using rolling window cross-validation

## Prerequisites

- Python 3.9+
- Basic pandas and matplotlib
- Familiarity with supervised learning (train/test splits, loss functions)

Install dependencies:
```bash
pip install neuralforecast utilsforecast matplotlib pandas
```

## Module Contents

```
module_01_point_forecasting/
├── guides/
│   ├── 01_neural_architectures.md        # Architecture overview + NHITS deep dive
│   ├── 01_neural_architectures_slides.md  # Marp slide deck (13 slides)
│   ├── 02_hyperparameter_tuning.md        # input_size, scaler_type, loss, cross-val
│   └── 02_hyperparameter_tuning_slides.md # Marp slide deck (13 slides)
├── notebooks/
│   ├── 01_training_nhits.ipynb            # Train NHITS, evaluate, visualize
│   └── 02_cross_validation.ipynb          # Cross-validation, DLinear comparison
├── exercises/
│   └── 01_point_forecasting_exercises.py  # Self-check exercises with asserts
└── README.md                              # This file
```

## Suggested Learning Path

1. Read **Guide 01** — start with the code block, then read the architectural explanations
2. Run **Notebook 01** — train NHITS and see real MAE/MSE numbers on bakery data
3. Read **Guide 02** — understand why `input_size` matters most
4. Run **Notebook 02** — compare NHITS vs DLinear with cross-validation
5. Complete **Exercise 01** — verify your understanding with assert-based checks

Total time: approximately 2–3 hours.

## Dataset

All code uses the **French Bakery** dataset: daily sales for multiple bakery products (Baguette, Pain au Chocolat, Croissant, etc.) from a real boulangerie in France, covering approximately 3 years.

The dataset exhibits:
- Strong weekly seasonality (weekend peaks)
- Mild annual seasonality (summer slowdown, holiday spikes)
- Multiple interacting series (products sold in the same shop)

Data source: Kaggle (matthieugimbert/french-bakery-daily-sales), hosted by Nixtla.

## Key Concepts

### Nixtla Long Format

NeuralForecast requires data as a pandas DataFrame with columns:
- `unique_id` — series identifier (one per product)
- `ds` — date (datetime64)
- `y` — target value (sales count)

### NHITS in One Paragraph

NHITS (Neural Hierarchical Interpolation for Time Series) uses a stack of MLP blocks where each block applies a MaxPool downsampling to the input before its MLP. The pooling kernel size decreases from stack to stack (heavy → light → none), so early stacks see only coarse, low-frequency summaries of the input while later stacks process the full detail. Each stack also produces its forecast by predicting a coarser output grid and upsampling via interpolation — enforcing smoothness on low-frequency components. The stacks connect via doubly residual links: each block subtracts its "backcast" from the input before passing the residual to the next block. The final forecast is the sum of all stacks' contributions. This design makes NHITS 50x faster than Transformer-based models while achieving 20%+ accuracy improvements on long-horizon benchmarks.

### Hyperparameter Priorities

| Priority | Parameter | Rule |
|---|---|---|
| 1 | `input_size` | 2–4× horizon; try 8× for annual patterns |
| 2 | `scaler_type` | `"robust"` for sales data with outliers |
| 3 | `loss` | `MAE()` for sales; `MSE()` when large errors are costly |
| 4 | `max_steps` | 500 for exploration; 1000–2000 for production |

## Connection to Other Modules

- **Module 00 (Foundations)**: NeuralForecast API, data format, environment setup
- **Module 02 (Probabilistic Forecasting)**: Replace point forecasts with prediction intervals using `MQLoss()`
- **Module 03 (Sample Paths)**: Generate Monte Carlo trajectories from neural models
- **Module 05 (DLinear)**: When linear models outperform neural models and why
