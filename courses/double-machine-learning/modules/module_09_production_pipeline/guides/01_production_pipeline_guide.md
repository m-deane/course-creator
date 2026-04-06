# Production DML Pipeline

> **Reading time:** ~5 min | **Module:** 9 — Production Pipeline | **Prerequisites:** Modules 0-8

## In Brief

You will learn how to design an end-to-end DML pipeline for production use, including data validation, nuisance model selection with cross-validation, sensitivity analysis, and automated reporting. This module takes DML from research notebooks to deployable code.

<div class="callout-insight">

<strong>Key Insight:</strong> A production DML pipeline needs four layers beyond the core estimator: data validation (catch problems before they corrupt results), model selection (systematic comparison of nuisance models), sensitivity analysis (how robust is the result?), and reporting (communicate results to stakeholders who do not understand DML internals).

</div>

<div class="callout-key">

<strong>Key Concept:</strong> You will learn how to design an end-to-end DML pipeline for production use, including data validation, nuisance model selection with cross-validation, sensitivity analysis, and automated reporting. This module takes DML from research notebooks to deployable code.

</div>

## Visual Explanation

```

PRODUCTION DML PIPELINE

Data → Validate → Select Models → Estimate → Sensitivity → Report
  │        │           │             │            │           │
  │   Check missing,   │     Cross-  │    Vary    │    CI,
  │   overlap, p/n,    │     fitted  │    specs   │    forest
  │   treatment var    │     R² for  │    and ML  │    plot,
  │                    │     nuisance│    models  │    GATES
  └────────────────────┴─────────────┴────────────┴──────────┘
```

## How to Build a Production Pipeline Class

The pipeline below is designed for commodity market causal analysis — estimating effects like carbon price changes on generation mix, sanctions on freight rates, or inventory surprises on futures spreads. It handles the full workflow from raw data to stakeholder-ready reports.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               RandomForestClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
import warnings


@dataclass
class DMLResult:
    """Container for DML estimation results."""
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    pval: float
    model_type: str
    nuisance_model: str
    n_obs: int
    n_controls: int
    r2_outcome: float
    r2_treatment: float


class ProductionDMLPipeline:
    """
    End-to-end DML pipeline with validation, model selection, and diagnostics.

    Usage:
        pipeline = ProductionDMLPipeline(df, y_col='outcome',
                                          d_cols='treatment', x_cols=controls)
        pipeline.validate()
        pipeline.select_nuisance_models()
        results = pipeline.estimate()
        pipeline.sensitivity_analysis()
        pipeline.report()
    """

    def __init__(self, df: pd.DataFrame, y_col: str, d_cols: str,
                 x_cols: List[str], z_cols: Optional[str] = None,
                 n_folds: int = 5, random_state: int = 42):
        self.df = df.copy()
        self.y_col = y_col
        self.d_cols = d_cols
        self.x_cols = x_cols
        self.z_cols = z_cols
        self.n_folds = n_folds
        self.random_state = random_state
        self.results: List[DMLResult] = []
        self.best_nuisance = None

    def validate(self) -> Dict:
        """Run data validation checks."""
        checks = {}

        # Missing values
        n_missing = self.df[self.x_cols + [self.y_col, self.d_cols]].isnull().sum().sum()
        checks['missing_values'] = n_missing
        if n_missing > 0:
            warnings.warn(f"Found {n_missing} missing values. Dropping rows.")
            self.df = self.df.dropna(subset=self.x_cols + [self.y_col, self.d_cols])

        # Dimension check
        n, p = len(self.df), len(self.x_cols)
        checks['n'] = n
        checks['p'] = p
        checks['p_over_n'] = p / n
        if p / n > 0.5:
            warnings.warn(f"p/n = {p/n:.2f} is high. Consider dimensionality reduction.")

        # Treatment variation
        d = self.df[self.d_cols]
        checks['treatment_std'] = d.std()
        checks['treatment_mean'] = d.mean()
        if d.std() < 1e-6:
            raise ValueError("Treatment has no variation.")

        # Binary treatment check
        unique_d = d.nunique()
        checks['is_binary'] = unique_d == 2
        if unique_d == 2:
            checks['treatment_prevalence'] = d.mean()
            if d.mean() < 0.05 or d.mean() > 0.95:
                warnings.warn("Treatment prevalence is extreme. Consider IRM with trimming.")

        print("Validation Results:")
        for k, v in checks.items():
            print(f"  {k}: {v}")

        return checks

    def select_nuisance_models(self) -> str:
        """Compare nuisance models using cross-validated R²."""
        X = self.df[self.x_cols].values
        Y = self.df[self.y_col].values
        D = self.df[self.d_cols].values

        candidates = {
            'lasso': LassoCV(cv=3, random_state=self.random_state),
            'rf': RandomForestRegressor(100, max_depth=10, random_state=self.random_state),
            'gbm': GradientBoostingRegressor(100, max_depth=5, random_state=self.random_state),
        }

        scores_y = {}
        scores_d = {}
        for name, model in candidates.items():
            r2_y = cross_val_score(model, X, Y, cv=3, scoring='r2').mean()
            r2_d = cross_val_score(model, X, D, cv=3, scoring='r2').mean()
            scores_y[name] = r2_y
            scores_d[name] = r2_d
            print(f"  {name:<10} R²(Y|X)={r2_y:.3f}  R²(D|X)={r2_d:.3f}")

        # Select best by average R²
        avg_scores = {k: (scores_y[k] + scores_d[k]) / 2 for k in candidates}
        self.best_nuisance = max(avg_scores, key=avg_scores.get)
        print(f"\nSelected: {self.best_nuisance}")
        return self.best_nuisance

    def estimate(self, nuisance: Optional[str] = None) -> DMLResult:
        """Run DML estimation with selected nuisance model."""
        nuisance = nuisance or self.best_nuisance or 'gbm'

        model_map = {
            'lasso': (LassoCV(cv=3), LassoCV(cv=3)),
            'rf': (RandomForestRegressor(200, random_state=self.random_state),
                   RandomForestRegressor(200, random_state=self.random_state)),
            'gbm': (GradientBoostingRegressor(200, max_depth=5, random_state=self.random_state),
                    GradientBoostingRegressor(200, max_depth=5, random_state=self.random_state)),
        }

        ml_l, ml_m = model_map[nuisance]
        dml_data = DoubleMLData(self.df, y_col=self.y_col,
                                 d_cols=self.d_cols, x_cols=self.x_cols)

        dml = DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m,
                           n_folds=self.n_folds)
        dml.fit()

        ci = dml.confint()
        result = DMLResult(
            estimate=dml.coef[0],
            se=dml.se[0],
            ci_low=ci.iloc[0, 0],
            ci_high=ci.iloc[0, 1],
            pval=dml.pval[0],
            model_type='PLR',
            nuisance_model=nuisance,
            n_obs=len(self.df),
            n_controls=len(self.x_cols),
            r2_outcome=0.0,
            r2_treatment=0.0
        )
        self.results.append(result)
        return result

    def sensitivity_analysis(self) -> pd.DataFrame:
        """Run estimation with all nuisance models to check robustness."""
        rows = []
        for nuisance in ['lasso', 'rf', 'gbm']:
            result = self.estimate(nuisance=nuisance)
            rows.append({
                'model': nuisance,
                'estimate': result.estimate,
                'se': result.se,
                'ci_low': result.ci_low,
                'ci_high': result.ci_high,
                'pval': result.pval,
            })

        sa_df = pd.DataFrame(rows)
        print("\nSensitivity Analysis:")
        print(sa_df.to_string(index=False))

        # Check robustness
        estimates = sa_df['estimate']
        spread = estimates.max() - estimates.min()
        mean_se = sa_df['se'].mean()
        if spread < mean_se:
            print("\nResult is ROBUST: estimate spread < average SE.")
        else:
            print("\nResult is SENSITIVE: estimate varies across specifications.")

        return sa_df

    def report(self) -> str:
        """Generate text report of main results."""
        if not self.results:
            return "No results yet. Call estimate() first."

        best = self.results[-1]
        sig = "statistically significant" if best.pval < 0.05 else "not statistically significant"

        report = f"""
DML Estimation Report
=====================
Model:     {best.model_type} with {best.nuisance_model} nuisance
Data:      n={best.n_obs}, p={best.n_controls}
Folds:     {self.n_folds}

Results:
  Estimate:  {best.estimate:.4f}
  Std Error: {best.se:.4f}
  95% CI:    [{best.ci_low:.4f}, {best.ci_high:.4f}]
  P-value:   {best.pval:.4f}
  Status:    {sig} at 5% level
"""
        print(report)
        return report
```

</div>
</div>

This pipeline class encapsulates the full DML workflow. Each method handles one phase: validation, model selection, estimation, sensitivity analysis, and reporting.

<div class="callout-warning">

<strong>Warning:</strong> Production pipelines should log all decisions (model selection, trimming, dropped observations) for reproducibility. Include the random seed, library versions, and data hash in every report.

</div>

## Connections

<div class="callout-info">

<strong>How this connects to the rest of the course:</strong>

</div>

**Builds on:**
- All prior modules (00-08)

**Leads to:**
- Deployment in trading systems
- Automated causal inference reporting

## Practice Problems

### Implementation

**1. Extend the Pipeline:**
Add IRM estimation alongside PLR. The pipeline should automatically detect binary treatment and switch to IRM.

**2. Repeated Cross-Fitting:**
Add a method that runs DML 10 times with different random seeds and reports the median estimate with interquartile range.


## Resources

<a class="link-card" href="../notebooks/01_production_pipeline_notebook.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
