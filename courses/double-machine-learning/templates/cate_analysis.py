"""
CATE Analysis Template - Estimate heterogeneous treatment effects
Works with: Observational data where effects may vary across subgroups
Time to working: 10 minutes

Example use cases:
- Which commodity sectors respond most to inventory surprises?
- Which customer segments benefit most from a pricing intervention?
- Where does a policy have the largest impact?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from econml.dml import CausalForestDML, LinearDML
from sklearn.ensemble import GradientBoostingRegressor

# ============================================================================
# CUSTOMIZE THESE
# ============================================================================
CONFIG = {
    # Data columns
    "outcome_col": "outcome",
    "treatment_col": "treatment",
    "effect_modifiers": None,    # Columns that moderate the effect (X in econml)
    "controls": None,            # Additional controls (W in econml)

    # Model settings
    "n_estimators": 200,
    "max_depth": 5,
    "random_state": 42,

    # GATES settings
    "n_gates_groups": 5,
}


# ============================================================================
# CATE ANALYSIS CLASS
# ============================================================================
class CATEAnalysis:
    """
    CATE estimation with BLP, GATES, and visualization.

    Usage:
        analysis = CATEAnalysis(df, config=CONFIG)
        analysis.fit()
        cate = analysis.predict_cate(X_new)
        gates = analysis.gates_analysis()
        analysis.plot_results()
    """

    def __init__(self, df: pd.DataFrame, config: Dict = None):
        self.config = config or CONFIG
        self.df = df.copy()
        self.model = None
        self.cate_hat = None

    def fit(self, method: str = "causal_forest"):
        """Fit CATE model."""
        Y = self.df[self.config["outcome_col"]].values
        D = self.df[self.config["treatment_col"]].values

        x_cols = self.config.get("effect_modifiers")
        w_cols = self.config.get("controls")

        X = self.df[x_cols].values if x_cols else None
        W = self.df[w_cols].values if w_cols else None

        rs = self.config["random_state"]
        n_est = self.config["n_estimators"]
        md = self.config["max_depth"]

        if method == "causal_forest":
            self.model = CausalForestDML(
                model_y=GradientBoostingRegressor(n_est, max_depth=md, random_state=rs),
                model_t=GradientBoostingRegressor(n_est, max_depth=md, random_state=rs),
                n_estimators=n_est,
                random_state=rs
            )
        elif method == "linear":
            self.model = LinearDML(
                model_y=GradientBoostingRegressor(n_est, max_depth=md, random_state=rs),
                model_t=GradientBoostingRegressor(n_est, max_depth=md, random_state=rs),
                random_state=rs
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        self.model.fit(Y, D, X=X, W=W)
        if X is not None:
            self.cate_hat = self.model.effect(X)
        print(f"CATE model fitted: {method}")
        print(f"  ATE: {np.mean(self.cate_hat):.4f}" if self.cate_hat is not None else "")

    def predict_cate(self, X_new: np.ndarray) -> np.ndarray:
        """Predict CATE for new observations."""
        return self.model.effect(X_new)

    def predict_cate_interval(self, X_new: np.ndarray,
                               alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Predict CATE with confidence intervals."""
        return self.model.effect_interval(X_new, alpha=alpha)

    def gates_analysis(self, cate: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Group Average Treatment Effects (GATES) analysis."""
        if cate is None:
            cate = self.cate_hat
        if cate is None:
            raise ValueError("No CATE predictions available. Call fit() first.")

        n_groups = self.config["n_gates_groups"]
        sorted_idx = np.argsort(cate)
        group_size = len(cate) // n_groups

        gates = []
        for g in range(n_groups):
            start = g * group_size
            end = (g + 1) * group_size if g < n_groups - 1 else len(cate)
            idx = sorted_idx[start:end]
            gates.append({
                "group": g + 1,
                "mean_cate": np.mean(cate[idx]),
                "std_cate": np.std(cate[idx]),
                "n": len(idx),
                "min_cate": np.min(cate[idx]),
                "max_cate": np.max(cate[idx]),
            })

        return pd.DataFrame(gates)

    def subgroup_analysis(self, group_col: str) -> pd.DataFrame:
        """Compute CATE by subgroup."""
        if self.cate_hat is None:
            raise ValueError("No CATE predictions. Call fit() first.")

        groups = self.df[group_col].unique()
        rows = []
        for g in sorted(groups):
            mask = self.df[group_col] == g
            rows.append({
                "group": g,
                "mean_cate": np.mean(self.cate_hat[mask]),
                "std_cate": np.std(self.cate_hat[mask]),
                "n": mask.sum(),
            })
        return pd.DataFrame(rows)

    def plot_results(self, save_path: Optional[str] = None):
        """Plot CATE distribution and GATES."""
        if self.cate_hat is None:
            raise ValueError("No CATE predictions. Call fit() first.")

        gates_df = self.gates_analysis()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # CATE distribution
        axes[0].hist(self.cate_hat, bins=40, color="steelblue",
                     alpha=0.7, edgecolor="black", linewidth=0.5)
        axes[0].axvline(x=np.mean(self.cate_hat), color="red",
                        linestyle="--", linewidth=2,
                        label=f"ATE = {np.mean(self.cate_hat):.3f}")
        axes[0].set_xlabel("CATE", fontsize=12)
        axes[0].set_ylabel("Count", fontsize=12)
        axes[0].set_title("CATE Distribution", fontsize=14, fontweight="bold")
        axes[0].legend(fontsize=11)
        axes[0].grid(alpha=0.3)

        # GATES
        axes[1].bar(gates_df["group"], gates_df["mean_cate"],
                    color="steelblue", alpha=0.7, edgecolor="black")
        axes[1].set_xlabel("CATE Quintile", fontsize=12)
        axes[1].set_ylabel("Group Average Treatment Effect", fontsize=12)
        axes[1].set_title("GATES: Sorted Group Effects", fontsize=14, fontweight="bold")
        axes[1].grid(alpha=0.3, axis="y")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    n, p = 5000, 20

    X = np.random.randn(n, p)
    sector = np.random.choice([0, 1, 2], n, p=[0.4, 0.3, 0.3])
    D = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n) * 0.5
    tau = np.where(sector == 0, 1.2, np.where(sector == 1, 0.3, -0.1))
    Y = tau * D + 0.5 * X[:, 0] + np.random.randn(n) * 0.5

    x_cols = [f"X{i}" for i in range(p)]
    df = pd.DataFrame(X, columns=x_cols)
    df["outcome"] = Y
    df["treatment"] = D
    df["sector"] = sector

    config = {**CONFIG,
              "effect_modifiers": x_cols,
              "controls": None}

    analysis = CATEAnalysis(df, config=config)
    analysis.fit()

    gates = analysis.gates_analysis()
    print("\nGATES:")
    print(gates.to_string(index=False))

    subgroups = analysis.subgroup_analysis("sector")
    print("\nSubgroup Analysis:")
    print(subgroups.to_string(index=False))

    analysis.plot_results()
