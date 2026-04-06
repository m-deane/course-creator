# Deploying Causal Models in Production

> **Reading time:** ~7 min | **Module:** 7 — Production Pipelines | **Prerequisites:** Modules 1-6

## Learning Objectives

By the end of this guide, you will be able to:
1. Structure a causal inference pipeline as deployable Python code
2. Implement data validation checks before model fitting
3. Monitor causal model outputs for drift and assumption violations
4. Define retraining triggers and re-estimation workflows
5. Log and version causal analysis results for reproducibility

---

## 1. Why Production Causal Pipelines Are Different

Causal inference models differ from predictive models in important ways for deployment:

| Dimension | Predictive Model | Causal Model |
|-----------|-----------------|--------------|
| Primary goal | Minimise prediction error | Identify a causal effect |
| Key validation | Hold-out set RMSE | Assumption diagnostics |
| Monitoring | Prediction drift | Assumption stability |
| Retraining trigger | Accuracy degrades | Design context changes |
| Output | Probability/value | Causal effect estimate + uncertainty |

Causal models don't get "better" by seeing more data in the predictive ML sense. They get more precise. The design assumptions must remain valid.

---

## 2. Pipeline Architecture

A production causal pipeline has these stages:

```

Stage 1: Data Ingestion
  ├── Load raw data
  ├── Validate schema and types
  └── Check for missing values and outliers

Stage 2: Data Preparation
  ├── Merge and reshape to required format
  ├── Create treatment indicators
  └── Create time variables

Stage 3: Assumption Checks
  ├── Pre-treatment trend test (DiD)
  ├── Density test at cutoff (RDD)
  ├── First stage F-statistic (IV)
  └── Flag violations for human review

Stage 4: Estimation
  ├── Fit causal model
  ├── Extract treatment effect estimate
  └── Compute uncertainty bounds

Stage 5: Diagnostics
  ├── Convergence checks (Bayesian)
  ├── Robustness checks
  └── Sensitivity analysis

Stage 6: Output and Logging
  ├── Write results to store
  ├── Log run metadata
  └── Alert if assumptions flagged
```

---

## 3. Production Pipeline Implementation


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
"""
Causal Inference Production Pipeline
=====================================

A fully automated pipeline for running and monitoring causal analyses.
"""
import logging
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)


class CausalPipelineResult:
    """Container for causal pipeline outputs."""

    def __init__(self):
        self.estimate: Optional[float] = None
        self.se: Optional[float] = None
        self.ci_lo: Optional[float] = None
        self.ci_hi: Optional[float] = None
        self.n_obs: Optional[int] = None
        self.assumptions_passed: bool = False
        self.assumption_warnings: list = []
        self.diagnostics: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def to_dict(self) -> Dict:
        return {
            'estimate': self.estimate,
            'se': self.se,
            'ci_lo': self.ci_lo,
            'ci_hi': self.ci_hi,
            'n_obs': self.n_obs,
            'assumptions_passed': self.assumptions_passed,
            'assumption_warnings': self.assumption_warnings,
            'diagnostics': self.diagnostics,
            'metadata': self.metadata,
            'timestamp': datetime.utcnow().isoformat(),
        }

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class DiDPipeline:
    """Production DiD estimation pipeline with automatic diagnostics."""

    def __init__(
        self,
        outcome: str,
        group_col: str,
        time_col: str,
        unit_col: str,
        treatment_time: int,
        pre_trend_alpha: float = 0.05,
    ):
        self.outcome = outcome
        self.group_col = group_col
        self.time_col = time_col
        self.unit_col = unit_col
        self.treatment_time = treatment_time
        self.pre_trend_alpha = pre_trend_alpha

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """Validate input data before estimation."""
        errors = []

        required_cols = [self.outcome, self.group_col, self.time_col, self.unit_col]
        for col in required_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")

        if df[self.outcome].isna().any():
            n_missing = df[self.outcome].isna().sum()
            errors.append(f"Missing values in outcome: {n_missing} rows")

        if df[self.group_col].nunique() < 2:
            errors.append("Need at least 2 groups (treated + control)")

        if df[self.time_col].nunique() < 2:
            errors.append("Need at least 2 time periods")

        return len(errors) == 0, errors

    def check_parallel_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test for parallel pre-treatment trends."""
        pre_df = df[df[self.time_col] < self.treatment_time].copy()

        if len(pre_df) < 10:
            return {'passed': None, 'reason': 'Insufficient pre-treatment periods'}

        # Create event-time relative to treatment
        treated_units = df[df[self.group_col] == 1][self.unit_col].unique()
        pre_df['treated'] = pre_df[self.unit_col].isin(treated_units).astype(int)
        pre_df['rel_time'] = pre_df[self.time_col] - self.treatment_time

        # Group-time means
        group_means = pre_df.groupby([self.time_col, 'treated'])[self.outcome].mean().unstack()

        if group_means.shape[1] < 2:
            return {'passed': None, 'reason': 'Only one group in pre-period'}

        # Test for differential trends: interaction of time × treated
        try:
            model = smf.ols(
                f'{self.outcome} ~ treated * {self.time_col} + C({self.unit_col})',
                data=pre_df
            ).fit()

            interaction_key = f'treated:{self.time_col}'
            if interaction_key in model.params:
                interaction_coef = model.params[interaction_key]
                interaction_pval = model.pvalues[interaction_key]
                passed = interaction_pval > self.pre_trend_alpha
            else:
                passed = True
                interaction_coef = 0
                interaction_pval = 1.0

            return {
                'passed': passed,
                'interaction_coef': float(interaction_coef),
                'p_value': float(interaction_pval),
                'alpha': self.pre_trend_alpha,
                'warning': (not passed)
            }
        except Exception as e:
            return {'passed': None, 'reason': str(e)}

    def estimate(self, df: pd.DataFrame) -> CausalPipelineResult:
        """Run the full DiD pipeline."""
        result = CausalPipelineResult()
        result.metadata = {
            'design': 'DiD',
            'outcome': self.outcome,
            'treatment_time': self.treatment_time,
            'n_total': len(df),
            'n_treated_units': int((df[self.group_col] == 1).sum() / df[self.time_col].nunique()),
        }

        # 1. Data validation
        valid, errors = self.validate_data(df)
        if not valid:
            result.assumption_warnings.extend(errors)
            logger.error("Data validation failed: %s", errors)
            return result

        # 2. Pre-trend check
        pt_result = self.check_parallel_trends(df)
        result.diagnostics['parallel_trends'] = pt_result
        if pt_result.get('warning'):
            result.assumption_warnings.append(
                f"Pre-trend violation detected (p={pt_result['p_value']:.4f} < {self.pre_trend_alpha})"
            )
            logger.warning("Parallel trends test failed.")

        # 3. Estimation
        df_model = df.copy()
        df_model['post'] = (df_model[self.time_col] >= self.treatment_time).astype(int)
        df_model['post_treated'] = df_model['post'] * df_model[self.group_col]

        formula = (f'{self.outcome} ~ post_treated + '
                   f'C({self.unit_col}) + C({self.time_col})')

        model = smf.ols(formula, data=df_model).fit(
            cov_type='cluster', cov_kwds={'groups': df_model[self.unit_col]}
        )

        result.estimate = float(model.params['post_treated'])
        result.se = float(model.bse['post_treated'])
        result.ci_lo = float(model.conf_int().loc['post_treated', 0])
        result.ci_hi = float(model.conf_int().loc['post_treated', 1])
        result.n_obs = int(model.nobs)

        result.assumptions_passed = len(result.assumption_warnings) == 0

        logger.info(
            "DiD estimate: τ = %.4f (SE = %.4f), n = %d, assumptions_passed = %s",
            result.estimate, result.se, result.n_obs, result.assumptions_passed
        )

        return result
```

</div>
</div>

---

## 4. Monitoring and Drift Detection

### What to Monitor

Once a causal model is deployed and producing regular estimates, monitor:

1. **Data distribution:** Is the running variable still continuous near the cutoff? Is the group composition stable?
2. **Assumption stability:** Are pre-trends still approximately parallel? Is the density test still passing?
3. **Estimate stability:** Is the treatment effect changing over time? (Effect modification vs. true change)
4. **Sample composition:** Are there changes in the types of units entering the sample?


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
class CausalMonitor:
    """Monitor a deployed causal pipeline for drift."""

    def __init__(self, baseline_result: CausalPipelineResult, tolerance: float = 0.1):
        self.baseline_estimate = baseline_result.estimate
        self.baseline_se = baseline_result.se
        self.tolerance = tolerance  # fraction of baseline SE for drift alert

    def check_estimate_stability(self, new_result: CausalPipelineResult) -> Dict:
        """Check if new estimate is within tolerance of baseline."""
        if new_result.estimate is None:
            return {'stable': False, 'reason': 'New estimate is None'}

        # Normalised difference
        z_score = abs(new_result.estimate - self.baseline_estimate) / self.baseline_se

        stable = z_score < 2.0  # within 2 baseline SEs

        return {
            'stable': stable,
            'z_score': z_score,
            'baseline': self.baseline_estimate,
            'new': new_result.estimate,
            'change': new_result.estimate - self.baseline_estimate,
            'alert': not stable
        }
```

</div>
</div>

---

## 5. Retraining Triggers

Unlike predictive models, causal models don't "degrade" in accuracy — they can only become invalid due to design context changes.

### When to Re-estimate

| Trigger | Action |
|---------|--------|
| New data covers a longer time window | Re-run with extended time series |
| Treatment policy changes (e.g., new cutoff) | Re-specify the design |
| Parallel trends test fails on new data | Investigate — may need different controls |
| Running variable distribution shifts (RDD) | Re-check density test and bandwidth |
| New comparison units become available | Consider adding to control group |
| Published critique of your design | Respond with robustness checks |

### Automated Re-estimation Workflow


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def run_monthly_causal_update(data_path: Path, results_dir: Path):
    """
    Monthly causal estimate update with monitoring.
    """
    # Load and validate new data
    df = pd.read_parquet(data_path)

    # Run pipeline
    pipeline = DiDPipeline(
        outcome='employment_rate',
        group_col='treated',
        time_col='month',
        unit_col='county_id',
        treatment_time=202401,
    )

    result = pipeline.estimate(df)

    # Save result
    run_id = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    result_path = results_dir / f'did_result_{run_id}.json'
    result.save(result_path)

    # Alert if assumptions failed
    if not result.assumptions_passed:
        send_alert(
            subject="Causal Pipeline: Assumption Warning",
            body=f"Warnings: {result.assumption_warnings}\nEstimate: {result.estimate:.4f}"
        )

    return result
```

</div>
</div>

---

## 6. Reproducibility and Versioning

Every causal analysis run should be reproducible. Best practices:


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import hashlib
import json

def create_run_manifest(df: pd.DataFrame, config: Dict, result: CausalPipelineResult) -> Dict:
    """Create a reproducibility manifest for a causal analysis run."""

    # Hash the input data
    data_hash = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

    manifest = {
        'run_id': datetime.utcnow().isoformat(),
        'data_hash': data_hash,
        'n_rows': len(df),
        'columns': list(df.columns),
        'config': config,
        'result': result.to_dict(),
        'code_version': '1.0.0',  # tag with your repo version
    }

    return manifest
```

</div>
</div>

---

## 7. Summary: Production Checklist

Before deploying a causal pipeline:

- [ ] Data validation with descriptive error messages
- [ ] All assumption checks automated and logged
- [ ] Results saved with run metadata and data hash
- [ ] Monitoring for estimate drift (compare to baseline)
- [ ] Clear retraining triggers documented
- [ ] Alert system for assumption failures
- [ ] Code versioned with results
- [ ] Human review step before acting on results

---

**Previous:** [02 — Reporting Guide](02_reporting_guide.md)
**Next:** [Module 07 Notebooks](../notebooks/)

<div class="callout-key">

<strong>Key Concept:</strong> **Previous:** [02 — Reporting Guide](02_reporting_guide.md)
**Next:** [Module 07 Notebooks](../notebooks/)

</div>



## Resources

<a class="link-card" href="../notebooks/01_model_selection_workflow.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
