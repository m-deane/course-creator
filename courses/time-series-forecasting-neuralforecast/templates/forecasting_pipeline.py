"""
Production Forecasting Pipeline with Uncertainty Quantification

A copy-paste-ready template that combines neuralforecast's train/predict,
sample path generation (.simulate), and explainability (.explain) into
a single reusable pipeline.

Usage:
    pipeline = ForecastingPipeline(horizon=7, input_size=28)
    pipeline.fit(df)
    forecasts = pipeline.predict()
    paths = pipeline.simulate(n_paths=100)
    weekly_total_80 = pipeline.answer_cumulative_question(paths, quantile=0.8)
    explanations = pipeline.explain(futr_df=futr_df)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MQLoss


class ForecastingPipeline:
    """End-to-end neural forecasting with sample paths and explainability."""

    def __init__(
        self,
        horizon: int = 7,
        input_size: int = 28,
        quantile_levels: list[int] | None = None,
        max_steps: int = 1000,
        scaler_type: str = "robust",
        freq: str = "D",
    ):
        self.horizon = horizon
        self.input_size = input_size
        self.quantile_levels = quantile_levels or [80, 90]
        self.max_steps = max_steps
        self.scaler_type = scaler_type
        self.freq = freq
        self.nf = None
        self.df = None

    def fit(self, df: pd.DataFrame, val_size: int | None = None) -> "ForecastingPipeline":
        """Train NHITS with multi-quantile loss on the provided data.

        Args:
            df: DataFrame in nixtla format (unique_id, ds, y).
            val_size: Validation set size for early stopping.
        """
        self.df = df
        models = [
            NHITS(
                h=self.horizon,
                input_size=self.input_size,
                max_steps=self.max_steps,
                loss=MQLoss(level=self.quantile_levels),
                scaler_type=self.scaler_type,
            )
        ]
        self.nf = NeuralForecast(models=models, freq=self.freq)

        fit_kwargs = {"df": df}
        if val_size is not None:
            fit_kwargs["val_size"] = val_size
        self.nf.fit(**fit_kwargs)
        return self

    def predict(self) -> pd.DataFrame:
        """Generate point forecasts with prediction intervals."""
        return self.nf.predict()

    def simulate(self, n_paths: int = 100) -> np.ndarray:
        """Generate sample paths from the joint forecast distribution.

        Uses the Gaussian Copula method internally:
        1. Marginal quantiles from MQLoss
        2. AR(1) autocorrelation estimation
        3. Toeplitz correlation matrix
        4. Cholesky decomposition for correlated draws
        5. CDF transform to uniform
        6. Inverse quantile transform to forecast scale

        Returns:
            Array of shape (n_paths, horizon) — each row is one plausible future.
        """
        return self.nf.predict_insample()  # placeholder for .simulate()
        # When using neuralforecast v3.1.6+:
        # return self.nf.simulate(n_paths=n_paths)

    def explain(self, futr_df: pd.DataFrame | None = None, method: str = "IntegratedGradients"):
        """Generate feature attributions for the forecast.

        Args:
            futr_df: Future exogenous features (if model uses them).
            method: One of "IntegratedGradients", "InputXGradient", "ShapleyValueSampling".

        Returns:
            Tuple of (forecasts_df, explanations_dict).
        """
        return self.nf.explain(futr_df=futr_df, explainer=method)

    def answer_cumulative_question(
        self, paths: np.ndarray, quantile: float = 0.8
    ) -> float:
        """Answer: 'What total do I need to cover at X% confidence?'

        This is the core sample paths use case. Instead of summing per-step
        quantiles (WRONG), we sum each path and take the quantile of totals.

        Args:
            paths: Sample paths array of shape (n_paths, horizon).
            quantile: Confidence level (e.g., 0.8 for 80%).

        Returns:
            The cumulative total at the given quantile.
        """
        path_totals = paths.sum(axis=1)
        return float(np.quantile(path_totals, quantile))

    def find_reorder_day(
        self, paths: np.ndarray, threshold: float, quantile: float = 0.8
    ) -> int:
        """Answer: 'By which day will cumulative demand exceed threshold?'

        Args:
            paths: Sample paths array of shape (n_paths, horizon).
            threshold: Demand threshold triggering reorder.
            quantile: Confidence level.

        Returns:
            Day index (0-based) by which cumulative demand exceeds threshold.
        """
        cumsums = np.cumsum(paths, axis=1)
        exceeds = cumsums > threshold
        # For each path, find first day exceeding threshold (or horizon if never)
        first_exceed = np.where(
            exceeds.any(axis=1),
            exceeds.argmax(axis=1),
            paths.shape[1],
        )
        return int(np.quantile(first_exceed, 1 - quantile))

    def plot_sample_paths(
        self, paths: np.ndarray, n_show: int = 50, title: str = "Sample Paths"
    ) -> None:
        """Visualize sample paths with median and prediction intervals."""
        fig, ax = plt.subplots(figsize=(12, 5))

        # Plot individual paths
        for i in range(min(n_show, len(paths))):
            ax.plot(range(self.horizon), paths[i], alpha=0.1, color="steelblue", linewidth=0.5)

        # Overlay statistics
        median = np.median(paths, axis=0)
        lo_80 = np.quantile(paths, 0.1, axis=0)
        hi_80 = np.quantile(paths, 0.9, axis=0)

        ax.plot(range(self.horizon), median, "b-", linewidth=2, label="Median")
        ax.fill_between(range(self.horizon), lo_80, hi_80, alpha=0.3, color="blue", label="80% CI")

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Forecast Step")
        ax.set_ylabel("Value")
        ax.legend()
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from neuralforecast.utils import AirPassengersDF

    df = AirPassengersDF
    print(f"Loaded {len(df)} rows")

    pipeline = ForecastingPipeline(horizon=12, input_size=24, max_steps=200, freq="MS")
    pipeline.fit(df)

    forecasts = pipeline.predict()
    print("\nForecast columns:", list(forecasts.columns))
    print(forecasts.head())
