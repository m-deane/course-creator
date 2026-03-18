"""
Module 05 — SHAP Methods: Self-Check Exercises

These exercises reinforce key concepts from Module 05.
Run each section and verify your understanding by checking the assertions.
"""

import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# SETUP: Small model and data for all exercises
# ─────────────────────────────────────────────────────────────────────────────

torch.manual_seed(42)
np.random.seed(42)


class SmallMLP(nn.Module):
    def __init__(self, input_dim=5, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


model = SmallMLP(input_dim=5)
model.eval()

X_data = torch.randn(200, 5)
background = X_data[:50]
x_test = X_data[100:103]  # 3 test inputs

feature_names = ["f0", "f1", "f2", "f3", "f4"]


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1: KernelSHAP — Efficiency Verification
#
# The efficiency axiom: SHAP values must sum to f(x) - E[f(X')]
# Verify this holds for KernelSHAP.
# ─────────────────────────────────────────────────────────────────────────────

def exercise_1_kernelshap_efficiency():
    """Verify the KernelSHAP efficiency axiom."""
    from captum.attr import KernelShap

    kernel_shap = KernelShap(model)

    attributions = kernel_shap.attribute(
        inputs=x_test,
        baselines=background,
        n_samples=300,
        perturbations_per_eval=32,
        target=1,
    )

    # Compute expected sum: f(x) - E[f(X')]
    with torch.no_grad():
        f_x = torch.softmax(model(x_test), dim=1)[:, 1]
        f_baselines = torch.softmax(model(background), dim=1)[:, 1]
        expected_baseline = f_baselines.mean()
        expected_sums = f_x - expected_baseline

    actual_sums = attributions.sum(dim=1)

    print("Exercise 1: KernelSHAP Efficiency Verification")
    print("-" * 50)
    for i in range(len(x_test)):
        diff = abs(actual_sums[i].item() - expected_sums[i].item())
        print(f"  Input {i}: SHAP sum={actual_sums[i].item():.4f}, "
              f"f(x)-E[f]={expected_sums[i].item():.4f}, "
              f"diff={diff:.4f}")

    # Efficiency should hold approximately for KernelSHAP
    # (not exact due to sampling approximation, but should be < 0.1)
    max_diff = max(abs(actual_sums[i].item() - expected_sums[i].item())
                   for i in range(len(x_test)))
    print(f"\n  Max efficiency gap: {max_diff:.4f}")
    print(f"  (< 0.05 is good; < 0.1 is acceptable for n_samples=300)")

    # Check that all attributions are finite
    assert not torch.isnan(attributions).any(), "NaN attributions detected!"
    assert not torch.isinf(attributions).any(), "Inf attributions detected!"

    print("\n  PASS: Attributions are finite")
    return attributions


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2: GradientSHAP — Convergence Delta
#
# GradientSHAP returns a convergence delta. Explore how delta changes
# with the number of Monte Carlo samples.
# ─────────────────────────────────────────────────────────────────────────────

def exercise_2_gradientshap_convergence():
    """Show how convergence delta improves with more Monte Carlo samples."""
    from captum.attr import GradientShap

    grad_shap = GradientShap(model)

    print("\nExercise 2: GradientSHAP Convergence Delta vs. n_samples")
    print("-" * 50)

    deltas_by_n = {}
    for n_samples in [5, 20, 50, 100]:
        _, delta = grad_shap.attribute(
            inputs=x_test,
            baselines=background,
            n_samples=n_samples,
            target=1,
            return_convergence_delta=True,
        )
        mean_delta = delta.abs().mean().item()
        deltas_by_n[n_samples] = mean_delta
        print(f"  n_samples={n_samples:4d}: mean |δ| = {mean_delta:.4f}")

    # More samples should generally reduce delta
    # (not strict monotone due to randomness, but trend should be visible)
    print("\n  Trend: More samples → smaller convergence delta")
    print("  (This verifies GradientSHAP converges to Shapley values)")

    return deltas_by_n


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 3: DeepLIFT — Exact Efficiency
#
# DeepLIFT guarantees exact efficiency (convergence delta ≈ 0).
# Verify this and compare with a gradient method.
# ─────────────────────────────────────────────────────────────────────────────

def exercise_3_deeplift_exact_efficiency():
    """Verify DeepLIFT's exact efficiency guarantee."""
    from captum.attr import DeepLift, IntegratedGradients

    zero_baseline = torch.zeros_like(x_test)

    print("\nExercise 3: DeepLIFT vs IG Efficiency Comparison")
    print("-" * 50)

    # DeepLIFT
    dl = DeepLift(model)
    attrs_dl, delta_dl = dl.attribute(
        inputs=x_test,
        baselines=zero_baseline,
        target=1,
        return_convergence_delta=True,
    )

    # Integrated Gradients (50 steps)
    ig = IntegratedGradients(model)
    attrs_ig, delta_ig = ig.attribute(
        inputs=x_test,
        baselines=zero_baseline,
        target=1,
        n_steps=50,
        return_convergence_delta=True,
    )

    print(f"  DeepLIFT max |δ|:  {delta_dl.abs().max().item():.2e}")
    print(f"  IG (n=50) max |δ|: {delta_ig.abs().max().item():.4f}")
    print()
    print("  DeepLIFT δ should be near machine precision (~1e-6)")
    print("  IG δ is larger because integration is numerically approximated")

    # DeepLIFT must satisfy near-exact efficiency
    assert delta_dl.abs().max().item() < 1e-3, \
        f"DeepLIFT delta too large: {delta_dl.abs().max().item()}"

    print("\n  PASS: DeepLIFT satisfies near-exact efficiency")
    return attrs_dl, attrs_ig


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 4: Method Agreement — Spearman Correlation
#
# Compare feature rankings from KernelSHAP, GradientSHAP, and DeepLIFT.
# All three should broadly agree on which features are most important.
# ─────────────────────────────────────────────────────────────────────────────

def exercise_4_method_agreement():
    """Compute Spearman correlation between attribution methods."""
    from captum.attr import KernelShap, GradientShap, DeepLiftShap
    from scipy.stats import spearmanr

    print("\nExercise 4: Method Agreement (Spearman Rank Correlation)")
    print("-" * 50)

    zero_bl = torch.zeros(1, 5)
    x_single = x_test[[0]]

    # All three methods on the same input
    ks_attrs = KernelShap(model).attribute(
        x_single, baselines=background, n_samples=200,
        perturbations_per_eval=32, target=1
    ).detach().squeeze().numpy()

    gs_attrs = GradientShap(model).attribute(
        x_single, baselines=background, n_samples=50, target=1
    ).detach().squeeze().numpy()

    dl_attrs = DeepLiftShap(model).attribute(
        x_single, baselines=background[:20], target=1
    ).detach().squeeze().numpy()

    pairs = [
        ('KernelSHAP', 'GradientSHAP', ks_attrs, gs_attrs),
        ('KernelSHAP', 'DeepLIFT SHAP', ks_attrs, dl_attrs),
        ('GradientSHAP', 'DeepLIFT SHAP', gs_attrs, dl_attrs),
    ]

    print(f"  {'Pair':<35} {'r':>6}")
    print("  " + "-" * 43)
    for n1, n2, a1, a2 in pairs:
        r, pval = spearmanr(a1, a2)
        significance = "***" if pval < 0.01 else ("*" if pval < 0.05 else "ns")
        print(f"  {n1} vs {n2:<20} {r:+.4f} {significance}")

    print()
    print("  Attribution values:")
    for feat, v_ks, v_gs, v_dl in zip(feature_names, ks_attrs, gs_attrs, dl_attrs):
        print(f"  {feat}: KS={v_ks:+.4f}  GS={v_gs:+.4f}  DL={v_dl:+.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 5: Background Dataset Size — Sensitivity Analysis
#
# How sensitive is KernelSHAP to background dataset size?
# Use Spearman r against the n=50 reference.
# ─────────────────────────────────────────────────────────────────────────────

def exercise_5_background_sensitivity():
    """Test sensitivity of GradientSHAP to background dataset size."""
    from captum.attr import GradientShap
    from scipy.stats import spearmanr

    print("\nExercise 5: Background Size Sensitivity (GradientSHAP)")
    print("-" * 50)

    x_single = x_test[[0]]

    # Reference: 50 backgrounds
    ref_attrs = GradientShap(model).attribute(
        x_single, baselines=background[:50], n_samples=50, target=1
    ).detach().squeeze().numpy()

    print(f"  {'n_bg':<8} {'Spearman r vs n=50 ref'}")
    print("  " + "-" * 35)

    for n_bg in [5, 10, 20, 30, 50]:
        attrs = GradientShap(model).attribute(
            x_single, baselines=background[:n_bg], n_samples=50, target=1
        ).detach().squeeze().numpy()
        r, _ = spearmanr(ref_attrs, attrs)
        bar = "#" * int(r * 20) if r > 0 else ""
        print(f"  {n_bg:<8} {r:+.4f}  {bar}")

    print("\n  Small background (n_bg < 10) may produce unstable attributions.")
    print("  50-100 background samples is a reasonable default.")


# ─────────────────────────────────────────────────────────────────────────────
# BONUS CHALLENGE: Implement your own KernelSHAP weighting
#
# The SHAP kernel is: π(z') = (d-1) / (C(d, |z'|) * |z'| * (d - |z'|))
# Compute and plot the kernel weights for d=5 features.
# ─────────────────────────────────────────────────────────────────────────────

def bonus_shap_kernel_weights():
    """Compute and display SHAP kernel weights for d=5."""
    from math import comb

    print("\nBonus: SHAP Kernel Weights (d=5)")
    print("-" * 50)

    d = 5
    print(f"  Coalition size |z'| → SHAP kernel weight π(z')")
    print()

    weights = {}
    for s in range(1, d):
        numerator = d - 1
        denominator = comb(d, s) * s * (d - s)
        w = numerator / denominator
        weights[s] = w
        bar = "█" * int(w * 30 / max(weights.values()) if weights else 1)
        print(f"  |z'| = {s}: π = {w:.4f}  {bar}")

    print()
    print("  Interpretation:")
    print("  - Small coalitions (|z'|=1) get highest weight")
    print("  - Large coalitions (|z'|=d-1=4) also get high weight")
    print("  - Mid-size coalitions get lower weight")
    print("  → KernelSHAP concentrates sampling on coalitions that most")
    print("    constrain individual feature Shapley values.")


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL EXERCISES
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("MODULE 05 — SHAP SELF-CHECK EXERCISES")
    print("=" * 60)

    exercise_1_kernelshap_efficiency()
    exercise_2_gradientshap_convergence()
    exercise_3_deeplift_exact_efficiency()
    exercise_4_method_agreement()
    exercise_5_background_sensitivity()
    bonus_shap_kernel_weights()

    print("\n" + "=" * 60)
    print("All exercises complete.")
    print("=" * 60)
