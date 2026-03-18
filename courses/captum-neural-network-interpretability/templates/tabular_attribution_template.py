"""
Captum Neural Network Interpretability
Production tabular attribution pipeline template.

Supports MLP and feedforward models on structured/tabular data.
Covers: DeepLIFT, DeepLIFTSHAP, GradientSHAP, KernelSHAP,
feature importance summary, and SHAP beeswarm/waterfall visualization.

Usage:
    from tabular_attribution_template import TabularAttributionPipeline

    pipeline = TabularAttributionPipeline(
        model=my_mlp,
        feature_names=["age", "income", "education", ...],
        class_names=["declined", "approved"],
    )
    # Record background for baseline
    pipeline.record_background(X_train[:200])

    # Attribute a single example
    result = pipeline.attribute(x_single, target_class=1)
    pipeline.plot_waterfall(result)

    # Global importance across dataset
    summary = pipeline.global_importance(X_test[:100], target_class=1)
    pipeline.plot_summary_bar(summary)
"""

from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from captum.attr import (
    DeepLift,
    DeepLiftShap,
    GradientShap,
    IntegratedGradients,
    KernelShap,
)


# ─────────────────────────────────────────────────────────────────────────────
# TABULAR ATTRIBUTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────


class TabularAttributionPipeline:
    """
    Attribution pipeline for tabular/structured data models (MLP, feedforward).

    Attributes:
        model:         PyTorch model in eval mode
        feature_names: List of feature name strings (one per input dimension)
        class_names:   List of class label strings
        background:    Background tensor for multi-baseline methods
        device:        Computation device string
    """

    SUPPORTED_METHODS = Literal[
        "deeplift",
        "deeplift_shap",
        "gradient_shap",
        "integrated_gradients",
        "kernel_shap",
    ]

    def __init__(
        self,
        model: torch.nn.Module,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.feature_names = feature_names
        self.class_names = class_names
        self.background: Optional[torch.Tensor] = None
        self.background_mean: Optional[torch.Tensor] = None
        self.device = device

    def record_background(
        self, X: torch.Tensor, n_samples: Optional[int] = None
    ) -> None:
        """
        Record a background sample for multi-baseline methods.

        Args:
            X:         (N, n_features) tensor of background examples
            n_samples: If set, randomly subsample to this many examples
        """
        X = X.to(self.device)
        if n_samples is not None and n_samples < X.shape[0]:
            idx = torch.randperm(X.shape[0])[:n_samples]
            X = X[idx]
        self.background = X
        self.background_mean = X.mean(dim=0, keepdim=True)

    def build_baseline(
        self,
        inputs: torch.Tensor,
        strategy: Literal["zero", "mean", "median"] = "mean",
    ) -> torch.Tensor:
        """
        Build a single baseline tensor for DeepLIFT / IG.

        Args:
            inputs:   (1, n_features) tensor
            strategy: 'zero', 'mean' (over background), or 'median' (over background)

        Returns:
            (1, n_features) baseline tensor
        """
        if strategy == "zero":
            return torch.zeros_like(inputs)

        elif strategy == "mean":
            if self.background_mean is not None:
                return self.background_mean.expand_as(inputs)
            return torch.zeros_like(inputs)

        elif strategy == "median":
            if self.background is not None:
                median = self.background.median(dim=0).values.unsqueeze(0)
                return median.expand_as(inputs)
            return torch.zeros_like(inputs)

        else:
            raise ValueError(f"Unknown baseline strategy: {strategy!r}")

    def predict(self, inputs: torch.Tensor) -> Dict:
        """
        Return prediction details for inputs.

        Returns dict with: class_index, class_name, confidence, probs
        """
        with torch.no_grad():
            logits = self.model(inputs)
            probs = torch.softmax(logits, dim=1)[0]
            pred_class = int(probs.argmax().item())

        class_name = (
            self.class_names[pred_class]
            if self.class_names and pred_class < len(self.class_names)
            else str(pred_class)
        )
        return {
            "class_index": pred_class,
            "class_name": class_name,
            "confidence": float(probs[pred_class].item()),
            "probs": probs.cpu(),
        }

    def attribute(
        self,
        inputs: torch.Tensor,
        target_class: Optional[int] = None,
        method: str = "deeplift_shap",
        baseline_strategy: str = "mean",
        n_steps: int = 50,
        n_samples: int = 50,
    ) -> Dict:
        """
        Compute feature attributions for a single tabular example.

        Args:
            inputs:           (1, n_features) tensor
            target_class:     Class to attribute toward. None → predicted class.
            method:           Attribution method name
            baseline_strategy: Single-baseline strategy for deeplift/ig
            n_steps:          IG integration steps
            n_samples:        Number of baseline samples for SHAP methods

        Returns:
            Dict with: attributions (n_features,), prediction, delta, method
        """
        inputs = inputs.to(self.device)
        prediction = self.predict(inputs)
        if target_class is None:
            target_class = prediction["class_index"]

        def forward(x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

        baseline = self.build_baseline(inputs, baseline_strategy)

        delta_val: Optional[float] = None

        if method == "deeplift":
            dl = DeepLift(forward)
            attrs, delta = dl.attribute(
                inputs, baseline, target=target_class,
                return_convergence_delta=True,
            )
            delta_val = float(delta.item())

        elif method == "deeplift_shap":
            if self.background is None:
                raise RuntimeError(
                    "Call record_background() before using deeplift_shap"
                )
            dl_shap = DeepLiftShap(forward)
            bg = self.background[:n_samples] if n_samples < len(self.background) else self.background
            attrs, delta = dl_shap.attribute(
                inputs, bg, target=target_class,
                return_convergence_delta=True,
            )
            delta_val = float(delta.item())

        elif method == "gradient_shap":
            if self.background is None:
                raise RuntimeError(
                    "Call record_background() before using gradient_shap"
                )
            gs = GradientShap(forward)
            bg = self.background[:n_samples] if n_samples < len(self.background) else self.background
            attrs, delta = gs.attribute(
                inputs, bg, n_samples=n_samples, target=target_class,
                return_convergence_delta=True,
            )
            delta_val = float(delta.item())

        elif method == "integrated_gradients":
            ig = IntegratedGradients(forward)
            attrs, delta = ig.attribute(
                inputs, baseline, target=target_class,
                n_steps=n_steps,
                return_convergence_delta=True,
            )
            delta_val = float(delta.item())

        elif method == "kernel_shap":
            ks = KernelShap(forward)
            bg = baseline.expand(min(n_samples, 64), -1)
            attrs = ks.attribute(
                inputs, baselines=bg, target=target_class,
                n_samples=n_samples, perturbations_per_eval=16,
            )

        else:
            raise ValueError(f"Unknown method: {method!r}")

        attributions = attrs.squeeze(0).detach().cpu().numpy()
        feature_names = self.feature_names or [f"f{i}" for i in range(len(attributions))]

        return {
            "inputs": inputs.squeeze(0).cpu().numpy(),
            "attributions": attributions,
            "feature_names": feature_names,
            "prediction": prediction,
            "target_class": target_class,
            "delta": delta_val,
            "method": method,
        }

    def global_importance(
        self,
        X: torch.Tensor,
        target_class: int,
        method: str = "deeplift_shap",
        n_examples: int = 100,
        n_samples: int = 30,
    ) -> Dict:
        """
        Compute mean |attribution| across a dataset for global feature importance.

        Args:
            X:            (N, n_features) tensor
            target_class: Class to attribute toward
            method:       Attribution method
            n_examples:   Max examples to process
            n_samples:    Background samples per attribution

        Returns:
            Dict with: mean_abs_attribution, feature_names, all_attributions
        """
        X = X.to(self.device)
        n_examples = min(n_examples, X.shape[0])
        all_attrs: List[np.ndarray] = []

        for i in range(n_examples):
            inp = X[i].unsqueeze(0)
            result = self.attribute(inp, target_class=target_class,
                                    method=method, n_samples=n_samples)
            all_attrs.append(result["attributions"])

        all_attrs_np = np.array(all_attrs)  # (n_examples, n_features)
        mean_abs = np.abs(all_attrs_np).mean(axis=0)
        feature_names = self.feature_names or [f"f{i}" for i in range(mean_abs.shape[0])]

        return {
            "mean_abs_attribution": mean_abs,
            "feature_names": feature_names,
            "all_attributions": all_attrs_np,
            "n_examples": n_examples,
            "method": method,
            "target_class": target_class,
        }

    def plot_waterfall(
        self,
        result: Dict,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 7),
        max_features: int = 20,
    ) -> plt.Figure:
        """
        Waterfall plot: cumulative attribution from baseline to prediction.

        Shows top `max_features` features sorted by |attribution|.
        """
        attrs = result["attributions"]
        names = result["feature_names"]
        values = result["inputs"]
        prediction = result["prediction"]

        # Sort by |attribution|, take top k
        ranked = sorted(
            zip(names, attrs, values),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:max_features]
        names_r, attrs_r, values_r = zip(*ranked)

        colors = ["#2ca02c" if a >= 0 else "#d62728" for a in attrs_r]

        fig, ax = plt.subplots(figsize=figsize)
        y = np.arange(len(names_r))
        ax.barh(y, attrs_r, color=colors, alpha=0.85, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{n} = {v:.2f}" for n, v in zip(names_r, values_r)], fontsize=9
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Attribution Score", fontsize=11)
        ax.set_title(
            f"Feature Attribution Waterfall\n"
            f"Prediction: {prediction['class_name']} ({prediction['confidence']:.1%})  "
            f"Method: {result['method']}",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_summary_bar(
        self,
        summary: Dict,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
        max_features: int = 20,
    ) -> plt.Figure:
        """
        Global feature importance bar chart.

        Shows mean |attribution| per feature, ranked descending.
        """
        mean_abs = summary["mean_abs_attribution"]
        names = summary["feature_names"]

        ranked = sorted(zip(names, mean_abs), key=lambda x: x[1], reverse=True)[
            :max_features
        ]
        names_r, scores_r = zip(*ranked)

        fig, ax = plt.subplots(figsize=figsize)
        y = np.arange(len(names_r))
        ax.barh(y, scores_r, color="#1f77b4", alpha=0.85, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(names_r, fontsize=9)
        ax.set_xlabel("Mean |Attribution Score|", fontsize=11)
        ax.set_title(
            f"Global Feature Importance (n={summary['n_examples']} examples)\n"
            f"Method: {summary['method']}  |  "
            f"Class: {self.class_names[summary['target_class']] if self.class_names else summary['target_class']}",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Minimal MLP for demonstration
    class SimpleTabularMLP(torch.nn.Module):
        def __init__(self, in_features: int, n_classes: int) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_features, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    N_FEATURES = 10
    N_CLASSES = 2
    feature_names = [f"feature_{i}" for i in range(N_FEATURES)]

    model = SimpleTabularMLP(N_FEATURES, N_CLASSES)
    X_bg = torch.randn(200, N_FEATURES)
    X_test = torch.randn(50, N_FEATURES)

    pipeline = TabularAttributionPipeline(
        model=model,
        feature_names=feature_names,
        class_names=["Class 0", "Class 1"],
    )
    pipeline.record_background(X_bg)

    # Single example attribution
    result = pipeline.attribute(X_test[0:1], method="deeplift_shap")
    print(f"Prediction: {result['prediction']['class_name']} ({result['prediction']['confidence']:.1%})")
    print(f"Top features:")
    ranked = sorted(zip(result["feature_names"], result["attributions"]),
                    key=lambda x: abs(x[1]), reverse=True)
    for name, attr in ranked[:5]:
        print(f"  {name}: {attr:+.4f}")

    print("\nTabularAttributionPipeline ready.")
    print("Methods: deeplift | deeplift_shap | gradient_shap | integrated_gradients | kernel_shap")
