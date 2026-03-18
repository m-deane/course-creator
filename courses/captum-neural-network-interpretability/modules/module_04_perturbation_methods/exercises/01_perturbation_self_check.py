"""
Module 04 — Exercise 01: Perturbation Methods Self-Check

Self-check covering Occlusion, Feature Ablation, Shapley Value Sampling,
the four Shapley axioms, and API usage in Captum.

Run with: python 01_perturbation_self_check.py
"""

# ============================================================
# PART 1: Occlusion Theory
# ============================================================

# Q1: Occlusion computes attribution as:
#     attr(u,v) = f(x) - f(x with window at (u,v))
#
# If occluding a window at position (u,v) INCREASES the model's
# confidence, what is the sign of the attribution at (u,v)?
occlusion_negative_attribution = None  # "positive", "negative", or "zero"

# Q2: For a 224×224 image with window=15×15 and stride=8,
#     approximately how many forward passes are needed?
#     (formula: ceil(224/8) × ceil(224/8))
occlusion_n_passes = None  # Numeric answer

# Q3: In Captum's Occlusion, the correct `strides` parameter for
#     occluding all 3 color channels together with spatial stride 8 is:
occlusion_strides = None  # (3, 8, 8), (1, 8, 8), (8, 8, 8), or (3, 3, 3)

# Q4: The `sliding_window_shapes` parameter for occluding a 15×15
#     window across all 3 color channels is:
occlusion_window_shape = None  # (3, 15, 15), (1, 15, 15), (15, 15), or (3, 15)

# Q5: Occlusion differs from IG in that Occlusion:
occlusion_vs_ig_difference = None
# Options:
# "a" = Requires model gradients
# "b" = Uses only forward passes (model-agnostic)
# "c" = Always produces higher quality attributions
# "d" = Works only for image models

# Q6: Negative Occlusion attribution (occluding a region HELPS the prediction)
#     is a sign of:
negative_occlusion_meaning = None
# Options:
# "a" = The model is perfectly calibrated
# "b" = The model is being confused or misled by that region (spurious correlation)
# "c" = The occlusion window is too small
# "d" = The baseline value is incorrect

# ============================================================
# PART 2: Feature Ablation
# ============================================================

# Q7: Feature Ablation generalizes Occlusion by supporting:
feature_ablation_generalization = None
# Options:
# "a" = Only rectangular sliding windows on images
# "b" = Arbitrary feature groupings (superpixels, tabular features, etc.)
# "c" = Gradient-based attribution for faster computation
# "d" = Only binary (present/absent) features

# Q8: For Feature Ablation with a superpixel mask containing 50 superpixels,
#     how many forward passes are required?
fa_superpixel_passes = None  # 50, 500, 2500, or 50 × n_steps

# Q9: In Captum's FeatureAblation, the `feature_mask` parameter:
feature_mask_meaning = None
# Options:
# "a" = A binary mask indicating which pixels to always occlude
# "b" = An integer tensor where pixels with the same value form one feature group
# "c" = A floating-point tensor of pre-computed feature importances
# "d" = The baseline values for each feature group

# Q10: For tabular data with 11 features and no feature_mask,
#      how many forward passes does FeatureAblation require?
fa_tabular_passes = None  # Numeric answer

# Q11: Which aggregation of color channels is standard when converting
#      Occlusion's (1, 3, 224, 224) output to a 2D heatmap?
occlusion_aggregation = None
# Options:
# "a" = Take only the first channel: attr[0, 0, :, :]
# "b" = Mean across color channels: attr.mean(dim=1)
# "c" = Sum across batch: attr.sum(dim=0)
# "d" = Max across channels: attr.max(dim=1).values

# ============================================================
# PART 3: Shapley Value Theory
# ============================================================

# Q12: The Efficiency axiom (also called Completeness) states:
shapley_efficiency = None
# Options:
# "a" = sum_i(phi_i) = f(x)
# "b" = sum_i(phi_i) = f(x) - f(baseline)
# "c" = sum_i(phi_i) = 0
# "d" = max_i(phi_i) = f(x) - f(baseline)

# Q13: The Shapley value is defined as:
#   phi_i = SUM over subsets S that don't contain i of:
#           [|S|! × (n - |S| - 1)!] / n! × [v(S ∪ {i}) - v(S)]
#
# What is v(S) in the attribution context?
shapley_v_S_meaning = None
# Options:
# "a" = The Shapley value of the coalition S
# "b" = The model's prediction when only features in S are present
#       (others replaced by baseline)
# "c" = The gradient of the output with respect to features in S
# "d" = The number of features in S

# Q14: For n=11 tabular features, exact Shapley computation requires
#      evaluating how many subsets?
exact_shapley_subsets = None  # 11, 121, 1024, or 2048

# Q15: ShapleyValueSampling with n_samples=100 on an 11-feature model
#      requires approximately how many forward passes?
svs_passes = None  # Numeric answer: n_features × n_samples

# Q16: Shapley values capture feature INTERACTIONS, while IG does not.
#      For the function f(A, B) = A × B with A=3, B=2, baseline=(0,0),
#      what Shapley value should feature A receive?
#      (Hint: phi_A = 0.5×[f({A})-f({})] + 0.5×[f({A,B})-f({B})]
#              = 0.5×[0-0] + 0.5×[6-0] = 3.0)
shapley_interaction_phi_A = None  # Numeric answer

# Q17: The SYMMETRY axiom states that if features i and j have the same
#      marginal contribution in every coalition, then phi_i = phi_j.
#      For f(A, B) = A × B with A=3, B=2, baseline=(0,0),
#      what should phi_B equal?
shapley_interaction_phi_B = None  # Numeric answer (same as phi_A by symmetry)

# ============================================================
# PART 4: Shapley Axioms
# ============================================================

# Q18: Which axiom states: if a feature contributes nothing in any coalition,
#      it receives zero Shapley value?
shapley_dummy_axiom = None  # "Efficiency", "Symmetry", "Dummy", or "Additivity"

# Q19: Shapley values are the UNIQUE attribution method satisfying:
shapley_unique_count = None  # 2, 3, or 4 axioms simultaneously

# Q20: Integrated Gradients satisfies Efficiency (Completeness) and Sensitivity.
#      Which Shapley axiom does IG NOT satisfy?
ig_fails_shapley_axiom = None  # "Efficiency", "Dummy", "Symmetry", or "Additivity"

# ============================================================
# PART 5: API and Practical Usage
# ============================================================

# Q21: The correct Captum call for ShapleyValueSampling is:
correct_svs_call = None
# Options:
# "a" = svs.attribute(input, baselines=bl, target=cls, n_samples=200)
# "b" = svs.attribute(input, baseline=bl, class=cls, samples=200)
# "c" = svs.shapley(input, baseline=bl, target=cls, iterations=200)
# "d" = svs.attribute(input, n_samples=200, return_variance=True)

# Q22: You run ShapleyValueSampling with n_samples=25 and get:
#      phi = [0.12, -0.05, 0.03, 0.08, ...]
#      You run it again (same model, same input) with n_samples=25 and get:
#      phi = [0.09, -0.07, 0.04, 0.11, ...]
#      What should you do?
svs_variance_action = None
# Options:
# "a" = The model is broken — the results should be identical
# "b" = Increase n_samples to reduce Monte Carlo variance
# "c" = Use a different baseline value
# "d" = ShapleyValueSampling is not appropriate for this model

# Q23: Which Captum class is the global (dataset-level) counterpart to
#      ShapleyValueSampling?
global_importance_class = None
# Options:
# "a" = FeatureAblation (global mode)
# "b" = PermutationFeatureImportance
# "c" = GlobalShapley
# "d" = DatasetIntegratedGradients

# Q24: You compare Feature Ablation and IG on a tabular model.
#      Feature Ablation gives: alcohol=0.15, sulphates=0.08, volatile_acidity=-0.12
#      IG gives:               alcohol=0.09, sulphates=0.11, volatile_acidity=-0.06
#
#      Both methods agree on direction (sign). The most likely cause of
#      the magnitude difference is:
fa_vs_ig_magnitude_difference = None
# Options:
# "a" = IG has a bug and should be discarded
# "b" = Feature Ablation measures the full effect of removing a feature;
#       IG measures local gradient sensitivity — they answer slightly different questions
# "c" = The training mean is not the correct baseline
# "d" = n_steps for IG is too low


# ============================================================
# SELF-CHECK ENGINE
# ============================================================

def check(name, got, expected, hint):
    if got is None:
        print(f"[ TODO ] {name}")
        return False
    if isinstance(expected, list):
        passed = got in expected
    elif isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        passed = abs(got - expected) < 1e-6
    elif isinstance(expected, tuple) and isinstance(got, tuple):
        passed = got == expected
    else:
        passed = got == expected
    status = "[ PASS ]" if passed else "[ FAIL ]"
    print(f"{status} {name}")
    if not passed:
        print(f"         Got:      {got!r}")
        print(f"         Expected: {expected!r}")
        print(f"         Hint: {hint}")
    return passed


def run():
    results = []
    print("=" * 65)
    print("PART 1: Occlusion Theory")
    print("=" * 65)

    results.append(check(
        "Occlusion: negative attribution sign",
        occlusion_negative_attribution, "negative",
        "attr(u,v) = f(x) - f(occluded). If occluding INCREASES confidence, "
        "f(occluded) > f(x), so attr = f(x) - f(occluded) < 0 → negative."
    ))
    results.append(check(
        "Occlusion forward passes (window=15, stride=8)",
        occlusion_n_passes, 784,
        "ceil(224/8) = 28. Total passes = 28 × 28 = 784."
    ))
    results.append(check(
        "Occlusion strides parameter",
        occlusion_strides, (3, 8, 8),
        "strides=(3, 8, 8): 3 means cover all 3 color channels, "
        "8 is the spatial stride. (1, 8, 8) would stride per channel."
    ))
    results.append(check(
        "Occlusion sliding_window_shapes",
        occlusion_window_shape, (3, 15, 15),
        "sliding_window_shapes=(3, 15, 15): cover all 3 channels, "
        "15×15 spatial window."
    ))
    results.append(check(
        "Occlusion vs IG key difference",
        occlusion_vs_ig_difference, "b",
        "Occlusion uses only forward passes — no gradients needed. "
        "This makes it model-agnostic (works on non-differentiable models)."
    ))
    results.append(check(
        "Negative occlusion attribution meaning",
        negative_occlusion_meaning, "b",
        "Negative attribution means removing the region helps the prediction. "
        "The model is being confused by that region — a sign of spurious correlation."
    ))

    print("\n" + "=" * 65)
    print("PART 2: Feature Ablation")
    print("=" * 65)

    results.append(check(
        "Feature Ablation generalization",
        feature_ablation_generalization, "b",
        "FeatureAblation supports arbitrary feature groups via feature_mask. "
        "Superpixels for images, individual features for tabular, etc."
    ))
    results.append(check(
        "FA with 50 superpixels: forward passes",
        fa_superpixel_passes, 50,
        "One forward pass per superpixel group = 50 passes for 50 superpixels. "
        "Much cheaper than pixel-level occlusion."
    ))
    results.append(check(
        "feature_mask meaning",
        feature_mask_meaning, "b",
        "feature_mask is an integer tensor where pixels with the same integer "
        "value belong to the same feature group and are ablated together."
    ))
    results.append(check(
        "FA tabular forward passes (11 features)",
        fa_tabular_passes, 11,
        "One forward pass per feature (each ablated independently) = 11 passes."
    ))
    results.append(check(
        "Occlusion color channel aggregation",
        occlusion_aggregation, "b",
        "attr.mean(dim=1) averages across the 3 color channels to produce "
        "a single 2D (H, W) heatmap. abs+mean is also common."
    ))

    print("\n" + "=" * 65)
    print("PART 3: Shapley Value Theory")
    print("=" * 65)

    results.append(check(
        "Efficiency axiom statement",
        shapley_efficiency, "b",
        "Efficiency: sum_i(phi_i) = f(x) - f(baseline). "
        "Attributions sum to the prediction difference."
    ))
    results.append(check(
        "v(S) meaning in Shapley formula",
        shapley_v_S_meaning, "b",
        "v(S) = model prediction when only features in S are present, "
        "with all other features replaced by their baseline values."
    ))
    results.append(check(
        "Exact Shapley: number of subsets for n=11",
        exact_shapley_subsets, 2048,
        "2^11 = 2048 subsets. For n=20: 2^20 ≈ 1M. Exponential growth."
    ))
    results.append(check(
        "SVS forward passes (n=11 features, n_samples=100)",
        svs_passes, 1100,
        "Cost = n_features × n_samples = 11 × 100 = 1100 forward passes."
    ))
    results.append(check(
        "Shapley phi_A for f(A,B)=A*B, A=3, B=2, baseline=(0,0)",
        shapley_interaction_phi_A, 3.0,
        "phi_A = 0.5*[f({A})-f({})] + 0.5*[f({A,B})-f({B})] "
        "= 0.5*[0-0] + 0.5*[6-0] = 3.0."
    ))
    results.append(check(
        "Shapley phi_B for f(A,B)=A*B (symmetry check)",
        shapley_interaction_phi_B, 3.0,
        "By Symmetry axiom: f(A,B)=A*B is symmetric in A and B "
        "(neither is 'more important'), so phi_A = phi_B = 3.0."
    ))

    print("\n" + "=" * 65)
    print("PART 4: Shapley Axioms")
    print("=" * 65)

    results.append(check(
        "Axiom: irrelevant feature gets zero attribution",
        shapley_dummy_axiom, "Dummy",
        "The Dummy axiom: if v(S∪{i})=v(S) for all S, then phi_i=0. "
        "Features that never change the output get zero attribution."
    ))
    results.append(check(
        "Number of axioms Shapley values uniquely satisfy",
        shapley_unique_count, 4,
        "Shapley values are the UNIQUE method satisfying all 4 axioms: "
        "Efficiency, Symmetry, Dummy, and Additivity."
    ))
    results.append(check(
        "Shapley axiom IG fails",
        ig_fails_shapley_axiom, "Symmetry",
        "IG is path-dependent: the attribution depends on the direction of "
        "the integration path. Two features with identical effects on "
        "every coalition may receive different IG attributions. "
        "Shapley values (by Symmetry axiom) always give them equal attribution."
    ))

    print("\n" + "=" * 65)
    print("PART 5: API and Practical Usage")
    print("=" * 65)

    results.append(check(
        "Correct SVS call",
        correct_svs_call, "a",
        "svs.attribute(input, baselines=bl, target=cls, n_samples=200) "
        "is the correct signature."
    ))
    results.append(check(
        "SVS variance action",
        svs_variance_action, "b",
        "Variance between runs with n_samples=25 is expected — it's a "
        "Monte Carlo approximation. Increase n_samples to reduce variance."
    ))
    results.append(check(
        "Global attribution Captum class",
        global_importance_class, "b",
        "PermutationFeatureImportance is the dataset-level method: "
        "it shuffles features across samples to measure global importance."
    ))
    results.append(check(
        "FA vs IG magnitude difference cause",
        fa_vs_ig_magnitude_difference, "b",
        "FA measures the total effect of removing a feature (replacing with baseline). "
        "IG measures local gradient sensitivity integrated along a path. "
        "Both are valid but answer slightly different attribution questions."
    ))

    print("\n" + "=" * 65)
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"RESULTS: {passed}/{total} checks passed")
    if passed == total:
        print("All checks passed! You have mastered perturbation methods.")
    else:
        print(f"{total - passed} checks need review.")
    print("=" * 65)


if __name__ == "__main__":
    run()
