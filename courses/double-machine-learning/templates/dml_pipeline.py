"""
Production DML Pipeline - Copy and customize for your use case
Works with: Any observational data with treatment, outcome, and controls
Time to working: 10 minutes

Example use cases:
- Policy impact estimation (carbon tax, sanctions, regulations)
- Treatment effect analysis (marketing interventions, operational changes)
- Causal inference with many controls (commodity markets, healthcare, economics)
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from doubleml import DoubleMLData, DoubleMLPLR, DoubleMLIRM
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               RandomForestClassifier, GradientBoostingClassifier)
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score

# ============================================================================
# CUSTOMIZE THESE
# ============================================================================
CONFIG = {
    # Column names in your DataFrame
    "outcome_col": "outcome",         # TODO: Your outcome variable
    "treatment_col": "treatment",     # TODO: Your treatment variable
    "control_cols": None,             # TODO: List of control column names (None = auto-detect)
    "instrument_col": None,           # Optional: instrument for IV estimation

    # DML settings
    "n_folds": 5,                     # Cross-fitting folds (5 is standard)
    "score": "partialling out",       # "partialling out" for PLR
    "dml_procedure": "dml2",          # "dml1" or "dml2" (dml2 recommended)

    # Model selection
    "nuisance_models": ["lasso", "rf", "gbm"],  # Models to compare
    "n_estimators": 200,              # Trees for RF/GBM
    "random_state": 42,

    # Binary treatment settings
    "trimming_threshold": 0.05,       # Propensity score trimming
}


# ============================================================================
# RESULT CONTAINER
# ============================================================================
@dataclass
class DMLResult:
    """Container for a single DML estimation result."""
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    pval: float
    model_type: str
    nuisance_model: str
    n_obs: int
    n_controls: int
    significant: bool

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# PRODUCTION PIPELINE
# ============================================================================
class DMLPipeline:
    """
    Production-ready DML estimation pipeline.

    Usage:
        pipeline = DMLPipeline(df, config=CONFIG)
        pipeline.validate()
        pipeline.select_nuisance_models()
        result = pipeline.estimate()
        sensitivity = pipeline.sensitivity_analysis()
        pipeline.report()
    """

    def __init__(self, df: pd.DataFrame, config: Dict = None):
        self.config = config or CONFIG
        self.df = df.copy()
        self.results: List[DMLResult] = []
        self.best_nuisance: Optional[str] = None
        self.is_binary: bool = False
        self._validated = False

        # Parse columns
        self.y_col = self.config["outcome_col"]
        self.d_col = self.config["treatment_col"]
        self.x_cols = self.config.get("control_cols")
        if self.x_cols is None:
            self.x_cols = [c for c in df.columns
                          if c not in [self.y_col, self.d_col,
                                      self.config.get("instrument_col")]]

    def validate(self) -> Dict:
        """Run data validation checks. Call before estimation."""
        checks = {}

        # Missing values
        cols = self.x_cols + [self.y_col, self.d_col]
        n_missing = self.df[cols].isnull().sum().sum()
        checks["missing_values"] = int(n_missing)
        if n_missing > 0:
            warnings.warn(f"Dropping {n_missing} missing values.")
            self.df = self.df.dropna(subset=cols)

        # Dimensions
        n = len(self.df)
        p = len(self.x_cols)
        checks["n_obs"] = n
        checks["n_controls"] = p
        checks["p_over_n"] = round(p / n, 4)

        if p / n > 0.5:
            warnings.warn(f"p/n = {p/n:.2f}. Consider reducing controls.")
        if n < 500:
            warnings.warn(f"n = {n}. DML works best with n >= 1000.")

        # Treatment variation
        d = self.df[self.d_col]
        checks["treatment_std"] = round(d.std(), 4)

        if d.std() < 1e-6:
            raise ValueError("Treatment has no variation.")

        # Binary detection
        self.is_binary = d.nunique() == 2
        checks["is_binary"] = self.is_binary
        if self.is_binary:
            checks["treatment_prevalence"] = round(d.mean(), 4)
            if d.mean() < 0.05 or d.mean() > 0.95:
                warnings.warn("Extreme treatment prevalence. Use trimming.")

        self._validated = True
        print("Validation: PASS")
        for k, v in checks.items():
            print(f"  {k}: {v}")
        return checks

    def select_nuisance_models(self) -> str:
        """Compare nuisance models using cross-validated R-squared."""
        if not self._validated:
            self.validate()

        X = self.df[self.x_cols].values
        Y = self.df[self.y_col].values
        D = self.df[self.d_col].values
        rs = self.config["random_state"]

        model_factories = {
            "lasso": lambda: LassoCV(cv=3, random_state=rs),
            "rf": lambda: RandomForestRegressor(
                self.config["n_estimators"], max_depth=10, random_state=rs),
            "gbm": lambda: GradientBoostingRegressor(
                self.config["n_estimators"], max_depth=5, random_state=rs),
        }

        print("\nNuisance Model Selection:")
        best_name, best_score = None, -np.inf
        for name in self.config["nuisance_models"]:
            model = model_factories[name]()
            r2_y = cross_val_score(model, X, Y, cv=3, scoring="r2").mean()
            r2_d = cross_val_score(model, X, D, cv=3, scoring="r2").mean()
            avg = (r2_y + r2_d) / 2
            print(f"  {name:<8} R2(Y)={r2_y:.3f}  R2(D)={r2_d:.3f}  avg={avg:.3f}")
            if avg > best_score:
                best_name, best_score = name, avg

        self.best_nuisance = best_name
        print(f"  Selected: {best_name}")
        return best_name

    def _get_ml_models(self, name: str):
        """Create ML model instances for given name."""
        rs = self.config["random_state"]
        n_est = self.config["n_estimators"]

        if name == "lasso":
            return LassoCV(cv=3, random_state=rs), LassoCV(cv=3, random_state=rs)
        elif name == "rf":
            return (RandomForestRegressor(n_est, max_depth=10, random_state=rs),
                    RandomForestRegressor(n_est, max_depth=10, random_state=rs))
        elif name == "gbm":
            return (GradientBoostingRegressor(n_est, max_depth=5, random_state=rs),
                    GradientBoostingRegressor(n_est, max_depth=5, random_state=rs))
        else:
            raise ValueError(f"Unknown model: {name}")

    def estimate(self, nuisance: Optional[str] = None) -> DMLResult:
        """Run DML estimation."""
        if not self._validated:
            self.validate()

        nuisance = nuisance or self.best_nuisance or "gbm"
        ml_l, ml_m = self._get_ml_models(nuisance)

        dml_data = DoubleMLData(self.df, y_col=self.y_col,
                                 d_cols=self.d_col, x_cols=self.x_cols)

        if self.is_binary:
            # Use IRM for binary treatment
            if nuisance == "lasso":
                ml_m_cls = RandomForestClassifier(100, random_state=self.config["random_state"])
            elif nuisance == "rf":
                ml_m_cls = RandomForestClassifier(self.config["n_estimators"],
                                                   random_state=self.config["random_state"])
            else:
                ml_m_cls = GradientBoostingClassifier(self.config["n_estimators"],
                                                       random_state=self.config["random_state"])

            dml = DoubleMLIRM(dml_data, ml_g=ml_l, ml_m=ml_m_cls,
                               score="ATE", n_folds=self.config["n_folds"],
                               trimming_threshold=self.config["trimming_threshold"])
            model_type = "IRM"
        else:
            dml = DoubleMLPLR(dml_data, ml_l=ml_l, ml_m=ml_m,
                               n_folds=self.config["n_folds"])
            model_type = "PLR"

        dml.fit()
        ci = dml.confint()

        result = DMLResult(
            estimate=float(dml.coef[0]),
            se=float(dml.se[0]),
            ci_low=float(ci.iloc[0, 0]),
            ci_high=float(ci.iloc[0, 1]),
            pval=float(dml.pval[0]),
            model_type=model_type,
            nuisance_model=nuisance,
            n_obs=len(self.df),
            n_controls=len(self.x_cols),
            significant=float(dml.pval[0]) < 0.05
        )
        self.results.append(result)
        return result

    def sensitivity_analysis(self) -> pd.DataFrame:
        """Run estimation with all nuisance models."""
        rows = []
        for name in self.config["nuisance_models"]:
            result = self.estimate(nuisance=name)
            rows.append(result.to_dict())

        sa_df = pd.DataFrame(rows)
        spread = sa_df["estimate"].max() - sa_df["estimate"].min()
        mean_se = sa_df["se"].mean()

        print("\nSensitivity Analysis:")
        print(sa_df[["nuisance_model", "estimate", "se", "ci_low", "ci_high"]].to_string(index=False))
        print(f"\nSpread: {spread:.4f}, Mean SE: {mean_se:.4f}")
        print(f"Robust: {'YES' if spread < mean_se else 'NO'}")

        return sa_df

    def report(self) -> str:
        """Generate text report."""
        if not self.results:
            return "No results. Call estimate() first."

        r = self.results[-1]
        sig = "significant" if r.significant else "not significant"

        text = f"""
DML Estimation Report
=====================
Model:     {r.model_type} with {r.nuisance_model}
Data:      n={r.n_obs}, p={r.n_controls}
Estimate:  {r.estimate:.4f}
SE:        {r.se:.4f}
95% CI:    [{r.ci_low:.4f}, {r.ci_high:.4f}]
P-value:   {r.pval:.4f}
Status:    {sig} at 5% level
"""
        print(text)
        return text


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    # Simulate data
    np.random.seed(42)
    n, p = 2000, 50
    X = np.random.randn(n, p)
    cols = [f"X{i}" for i in range(p)]
    D = 0.5 * np.sin(X[:, 0]) + 0.3 * X[:, 1]**2 + np.random.randn(n) * 0.5
    Y = -0.5 * D + np.exp(0.2 * X[:, 0]) + 0.3 * X[:, 2] + np.random.randn(n) * 0.3

    df = pd.DataFrame(X, columns=cols)
    df["outcome"] = Y
    df["treatment"] = D

    # Run pipeline
    config = {**CONFIG, "outcome_col": "outcome", "treatment_col": "treatment",
              "control_cols": cols}
    pipeline = DMLPipeline(df, config=config)
    pipeline.validate()
    pipeline.select_nuisance_models()
    result = pipeline.estimate()
    pipeline.report()
