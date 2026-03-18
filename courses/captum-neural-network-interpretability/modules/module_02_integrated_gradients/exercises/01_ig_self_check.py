"""
Module 02 — Exercise 01: Integrated Gradients Self-Check

Self-check covering IG theory, the completeness property,
baseline selection, and the LayerIntegratedGradients API.

Run with: python 01_ig_self_check.py
"""

# ============================================================
# PART 1: Theoretical Foundation
# ============================================================

# Q1: IG uses which mathematical theorem to guarantee that
#     sum(attributions) == f(input) - f(baseline)?
ig_mathematical_theorem = None  # "Mean Value Theorem", "Fundamental Theorem of Calculus",
                                  # or "Chain Rule"

# Q2: In IG, what does the parameter alpha represent?
ig_alpha_meaning = None
# Options:
# "a" = the learning rate during backpropagation
# "b" = the interpolation parameter from baseline (0) to input (1)
# "c" = the regularization strength on attributions
# "d" = the number of integration steps

# Q3: The IG formula is:
#   IG_i(x) = (x_i - x'_i) * integral from 0 to 1 of [∂f/∂x_i at x(alpha)] dalpha
# At alpha=0, x(alpha) = ?
ig_alpha_zero = None  # "input x", "baseline x'", or "midpoint (x + x') / 2"

# Q4: At alpha=1, x(alpha) = ?
ig_alpha_one = None  # "input x", "baseline x'", or "midpoint (x + x') / 2"

# Q5: The completeness property states:
#   sum_i(IG_i(x)) = f(x) - f(x')
#
# If f(input) = 0.85 and f(baseline) = 0.10, what should the
# sum of IG attributions equal?
ig_completeness_value = None  # Numeric answer

# Q6: IG satisfies the sensitivity axiom. Saliency does not.
#     The key reason: what does IG do that saliency does not?
ig_vs_saliency = None
# Options:
# "a" = IG takes absolute values; saliency uses signed values
# "b" = IG integrates gradients along a path; saliency evaluates at one point
# "c" = IG uses a non-zero baseline; saliency always uses zero baseline
# "d" = IG is faster than saliency for complex models

# Q7: True or False: IG and Guided Backpropagation both satisfy
#     the implementation invariance axiom.
ig_and_gbp_impl_invariance = None  # "true" or "false"

# ============================================================
# PART 2: Convergence and Validation
# ============================================================

# Q8: The convergence delta is defined as:
#   delta = sum_i(IG_i^approx) - (f(x) - f(x'))
#
# If delta = 0.15 and you are using n_steps=50, what should you do?
convergence_action = None
# Options:
# "a" = Reduce the learning rate
# "b" = Increase n_steps (e.g., to 100 or 300)
# "c" = Change the model architecture
# "d" = Use a larger baseline image

# Q9: In Captum, which parameter enables the convergence delta output?
captum_delta_param = None
# Options:
# "n_steps", "return_convergence_delta", "compute_delta", "enable_validation"

# Q10: If delta=0.001 with n_steps=50, should you increase n_steps?
delta_increase_question = None  # "yes" or "no"

# ============================================================
# PART 3: Baseline Selection
# ============================================================

# Q11: For an image classification model, which baseline represents
#      "compared to having no image at all"?
image_no_info_baseline = None  # "zero (black image)", "mean of ImageNet", "random noise", "blurred"

# Q12: For a tabular neural network predicting loan default,
#      which baseline represents a "typical" loan application?
tabular_baseline = None  # "zero vector", "mean of training set", "maximum feature values", "median"

# Q13: For a text classification model using a transformer,
#      what is the standard baseline input?
text_baseline = None  # "zero token IDs (PAD tokens)", "average embedding vector",
                       # "random token IDs", "empty string"

# Q14: You compute IG with a zero baseline for an image classifier.
#      The convergence delta is 0.001 (excellent).
#      But the attribution heatmap shows the BACKGROUND is heavily attributed.
#      What does this likely indicate?
attribution_background_issue = None
# Options:
# "a" = The model correctly identifies background as important
# "b" = The model may be using spurious background features to classify
# "c" = The convergence delta is too small and should be larger
# "d" = The baseline should be changed to a higher value

# Q15: You run IG with three different baselines:
#   - Zero baseline: top-10 attributed regions = [ears, snout, fur]
#   - Blurred baseline: top-10 attributed regions = [eyes, nose, fur texture]
#   - Random baseline: top-10 attributed regions = [ears, eyes, nose]
#
# Which feature is most likely to be robustly important?
robust_feature = None  # "ears", "eyes", "snout", "fur", or "fur texture"

# ============================================================
# PART 4: LayerIntegratedGradients for Text
# ============================================================

# Q16: Why do we use LayerIntegratedGradients (not IntegratedGradients)
#      for transformer models?
lig_reason = None
# Options:
# "a" = LayerIG is faster for transformer models
# "b" = Standard IG cannot compute gradients w.r.t. integer token IDs
# "c" = LayerIG provides better attribution accuracy for BERT
# "d" = Standard IG does not work on GPU for transformer models

# Q17: After computing LayerIG for a text model,
#      attributions have shape (1, seq_len, embed_dim).
#      To get per-token importance scores, what operation is applied?
lig_aggregation = None
# Options:
# "sum along embed_dim", "max along embed_dim",
# "mean along seq_len", "flatten"

# Q18: In LayerIntegratedGradients for sentiment classification,
#      the token "terrible" should have what sign of attribution
#      when explaining the POSITIVE (good review) class prediction?
token_terrible_sign_positive = None  # "positive", "negative", or "zero"

# Q19: What is the standard baseline for text (token ID) inputs?
text_token_baseline = None  # "zeros_like(input_ids)", "ones_like(input_ids)",
                              # "random_like(input_ids)", "input_ids themselves"

# ============================================================
# PART 5: SmoothGrad and NoiseTunnel
# ============================================================

# Q20: In Captum, NoiseTunnel wraps an existing method.
#      Which parameter sets the averaging mode (SmoothGrad vs VarGrad)?
nt_type_param = None  # "nt_type", "smooth_type", "averaging_mode", "variance_mode"

# Q21: SmoothGrad computes the MEAN of attributions over noisy samples.
#      VarGrad computes the VARIANCE of attributions.
#      For identifying regions that are CONSISTENTLY important,
#      which is more appropriate?
consistent_importance_method = None  # "SmoothGrad" or "VarGrad"

# Q22: You want SmoothGrad-IG with 20 samples and 50 integration steps.
#      How many total forward+backward passes will this require?
total_passes = None  # Numeric answer


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
    print("PART 1: Theoretical Foundation")
    print("=" * 65)

    results.append(check(
        "IG mathematical theorem",
        ig_mathematical_theorem, "Fundamental Theorem of Calculus",
        "FTC: integral of f' from 0 to 1 equals f(1) - f(0)."
    ))
    results.append(check(
        "Alpha meaning",
        ig_alpha_meaning, "b",
        "Alpha interpolates: x(alpha) = x' + alpha*(x - x'). At 0: baseline. At 1: input."
    ))
    results.append(check(
        "x(alpha) at alpha=0",
        ig_alpha_zero, "baseline x'",
        "At alpha=0: x(0) = x' + 0*(x - x') = x' (the baseline)."
    ))
    results.append(check(
        "x(alpha) at alpha=1",
        ig_alpha_one, "input x",
        "At alpha=1: x(1) = x' + 1*(x - x') = x (the input)."
    ))
    results.append(check(
        "Completeness value",
        ig_completeness_value, 0.75,
        "sum(IG) = f(x) - f(x') = 0.85 - 0.10 = 0.75."
    ))
    results.append(check(
        "IG vs Saliency key difference",
        ig_vs_saliency, "b",
        "IG integrates gradient along the path from baseline to input. "
        "Saliency evaluates gradient only at the input point."
    ))
    results.append(check(
        "IG and GBP both satisfy impl. invariance?",
        ig_and_gbp_impl_invariance, "false",
        "GBP fails implementation invariance — it depends on architecture, not weights."
    ))

    print("\n" + "=" * 65)
    print("PART 2: Convergence and Validation")
    print("=" * 65)

    results.append(check(
        "Convergence action for delta=0.15",
        convergence_action, "b",
        "Increase n_steps to reduce approximation error."
    ))
    results.append(check(
        "Captum delta parameter name",
        captum_delta_param, "return_convergence_delta",
        "ig.attribute(..., return_convergence_delta=True) returns (attributions, delta)."
    ))
    results.append(check(
        "Increase n_steps for delta=0.001?",
        delta_increase_question, "no",
        "delta=0.001 is excellent — no need for more steps."
    ))

    print("\n" + "=" * 65)
    print("PART 3: Baseline Selection")
    print("=" * 65)

    results.append(check(
        "Image 'no information' baseline",
        image_no_info_baseline, "zero (black image)",
        "Zero baseline = black image = no pixel information."
    ))
    results.append(check(
        "Tabular 'typical loan' baseline",
        tabular_baseline, ["mean of training set", "median"],
        "Training mean represents a typical application in the training distribution."
    ))
    results.append(check(
        "Text baseline",
        text_baseline, "zero token IDs (PAD tokens)",
        "PAD tokens represent 'no token' — the standard text baseline."
    ))
    results.append(check(
        "Background attribution issue",
        attribution_background_issue, "b",
        "If background is heavily attributed, the model may be using background style "
        "rather than object features — a spurious correlation problem."
    ))
    results.append(check(
        "Robust feature across baselines",
        robust_feature, ["fur", "fur texture"],
        "Features appearing in all three baseline results are most robust. "
        "Fur/fur texture appears in all three."
    ))

    print("\n" + "=" * 65)
    print("PART 4: LayerIntegratedGradients for Text")
    print("=" * 65)

    results.append(check(
        "Why LayerIG for transformers",
        lig_reason, "b",
        "Token IDs are integers — gradients cannot be computed w.r.t. integers. "
        "LayerIG attributes to the continuous embedding layer output."
    ))
    results.append(check(
        "LayerIG aggregation for per-token scores",
        lig_aggregation, "sum along embed_dim",
        "Sum across embedding dimension collapses (seq_len, embed_dim) to (seq_len,)."
    ))
    results.append(check(
        "Attribution sign for 'terrible' explaining POSITIVE class",
        token_terrible_sign_positive, "negative",
        "'terrible' decreases the POSITIVE class score — negative attribution."
    ))
    results.append(check(
        "Text token baseline",
        text_token_baseline, "zeros_like(input_ids)",
        "Zero IDs correspond to [PAD] tokens — the standard no-information baseline."
    ))

    print("\n" + "=" * 65)
    print("PART 5: SmoothGrad and NoiseTunnel")
    print("=" * 65)

    results.append(check(
        "NoiseTunnel type parameter",
        nt_type_param, "nt_type",
        "nt.attribute(..., nt_type='smoothgrad') or nt_type='vargrad'"
    ))
    results.append(check(
        "Consistent importance: SmoothGrad or VarGrad?",
        consistent_importance_method, "VarGrad",
        "LOW variance in VarGrad = consistently important across perturbations. "
        "Note: captum returns variance, so high VarGrad = inconsistent, low VarGrad = consistent."
    ))
    results.append(check(
        "Total passes for 20 samples × 50 steps",
        total_passes, 1000,
        "20 noise samples × 50 integration steps = 1000 total forward+backward passes."
    ))

    print("\n" + "=" * 65)
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"RESULTS: {passed}/{total} checks passed")
    if passed == total:
        print("All checks passed! You have mastered Integrated Gradients.")
    else:
        print(f"{total - passed} checks need review.")
    print("=" * 65)


if __name__ == "__main__":
    run()
