"""
Module 01 — Exercise 01: Gradient Methods Self-Check

Self-check exercise covering gradient attribution theory,
axiom compliance, the sanity check, and practical API usage.

Run with: python 01_gradient_self_check.py
"""

# ============================================================
# PART 1: Method Properties
# ============================================================
# Fill in the correct value for each property.

# Saliency
saliency_formula = None    # "abs(gradient)", "input * gradient", or "modified_gradient"
saliency_baseline_needed = None   # "yes" or "no"
saliency_passes = None     # approximate number of forward+backward passes per attribution: 1, 50, or 1000

# Input × Gradient
ixg_formula = None         # "abs(gradient)", "input * gradient", or "modified_gradient"
ixg_signed = None          # "yes" or "no" — does it return signed values?
ixg_taylor_order = None    # "first" or "second" — which order Taylor approximation?

# Guided Backpropagation
gbp_passes = None          # 1, 50, or 1000
gbp_modifies_backward = None  # "yes" or "no"
gbp_architecture_dependent = None  # "yes" or "no" — does it depend on architecture not weights?
gbp_satisfies_impl_invariance = None  # "yes" or "no"

# Deconvolution
deconv_condition = None    # "gradient_positive", "activation_positive", or "both"
# (what condition must be met for gradient flow in deconvolution backward pass)

# ============================================================
# PART 2: The Saturation Problem
# ============================================================
# Consider: f(x) = ReLU(3x - 6)
# For this function:

# At x = 1.0: pre-ReLU = 3*1 - 6 = -3, so f(1.0) = 0
# At x = 2.5: pre-ReLU = 3*2.5 - 6 = +1.5, so f(2.5) = 1.5

# Q1: What is the gradient df/dx at x = 1.0?
gradient_at_saturation = None  # numeric value: 0, 3, or None

# Q2: What is the saliency attribution at x = 1.0?
saliency_at_saturation = None  # numeric value: 0, 3, or None

# Q3: Is x = 1.0 relevant to the output? (does changing x affect output for x near 2.0?)
is_x_relevant = None  # "yes" or "no"

# Q4: Does saliency correctly identify this?
saliency_identifies_relevance = None  # "yes" or "no"

# Q5: What method correctly gives non-zero attribution even for x = 1.0?
# Hint: it integrates gradients from a baseline
method_resolves_saturation = None  # "Saliency", "InputXGradient", or "IntegratedGradients"

# ============================================================
# PART 3: The Sanity Check (Adebayo et al., 2018)
# ============================================================

# The randomization sanity check compares attribution maps between
# a TRAINED model and a RANDOMLY INITIALIZED model (same architecture).

# Q1: For a faithful attribution method, what should the correlation be
#     between trained-model and random-model attributions?
expected_correlation_faithful = None  # "near 0", "near 0.5", or "near 1.0"

# Q2: For Guided Backpropagation, what does the sanity check reveal?
sanity_check_gbp_finding = None
# Options:
# "a" = low correlation (maps differ for trained vs random — faithful)
# "b" = high correlation (maps are similar for trained vs random — architecture dependent)
# "c" = correlation exactly 0.0 (perfectly decorrelated)

# Q3: Why does Guided Backpropagation fail the sanity check?
# Write a brief explanation (fill in the string)
gbp_sanity_check_reason = None
# Example: "because [explanation of what GBP does that causes this]"

# Q4: Which of these methods passes the sanity check?
# Options: "Saliency", "GuidedBackprop", "Deconvolution", "IntegratedGradients"
# Choose ALL that pass (list of strings)
methods_that_pass_sanity_check = None  # e.g., ["Saliency", "IntegratedGradients"]

# ============================================================
# PART 4: SmoothGrad and NoiseTunnel
# ============================================================

# Q1: What does SmoothGrad compute?
# Options:
# "a" = gradient at a single perturbed input
# "b" = mean gradient over many noise-perturbed inputs
# "c" = variance of gradients over noise-perturbed inputs
smoothgrad_computation = None  # "a", "b", or "c"

# Q2: What does VarGrad compute?
vargrad_computation = None  # "a", "b", or "c"

# Q3: In Captum, which parameter controls the number of noise samples in NoiseTunnel?
noise_tunnel_samples_param = None  # "nt_samples", "n_steps", "n_perturbations", or "num_samples"

# Q4: Does SmoothGrad fix the gradient saturation problem?
smoothgrad_fixes_saturation = None  # "yes" or "no"
# (Hint: SmoothGrad averages gradients over noisy inputs.
#  If the base input is saturated, are noisy copies necessarily not saturated?)

# Q5: Does SmoothGrad reduce gradient noise?
smoothgrad_reduces_noise = None  # "yes" or "no"

# ============================================================
# PART 5: Captum API Debugging
# ============================================================
# Each code snippet below contains ONE bug. Identify the bug.

# Snippet A:
snippet_a_bug = None
# Code:
#   model = resnet50(weights='IMAGENET1K_V1')
#   sal = Saliency(model)
#   input_tensor = preprocess(image).unsqueeze(0)
#   attrs = sal.attribute(input_tensor, target=0)
#
# Bug options:
# "a" = model not in eval mode
# "b" = input tensor missing requires_grad
# "c" = target should be float, not int
# "d" = should call model.forward() explicitly

# Snippet B:
snippet_b_bug = None
# Code:
#   model.eval()
#   gbp = GuidedBackprop(model)
#   inp = preprocess(image).unsqueeze(0).requires_grad_(True)
#   attrs = gbp.attribute(inp, target="golden_retriever")
#
# Bug options:
# "a" = GuidedBackprop does not work in eval mode
# "b" = target should be an integer class index, not a string
# "c" = requires_grad should be False for GuidedBackprop
# "d" = should use Saliency, not GuidedBackprop, for pretrained models

# Snippet C:
snippet_c_bug = None
# Code:
#   model.eval()
#   ixg = InputXGradient(model)
#   inp = preprocess(image).unsqueeze(0).requires_grad_(True)
#   attrs = ixg.attribute(inp, target=5)
#   print(attrs.shape)  # (3, 224, 224) — missing batch dim?
#   attrs_np = attrs.numpy()  # error!
#
# Bug options:
# "a" = missing .detach() before .numpy()
# "b" = target=5 is wrong — should be 0-indexed differently
# "c" = InputXGradient requires a baseline
# "d" = should call model.zero_grad() before attribution


# ============================================================
# SELF-CHECK ENGINE
# ============================================================

def check(name, got, expected, explanation):
    if got is None:
        print(f"[ TODO ] {name}: not answered")
        return False
    if isinstance(expected, list):
        if isinstance(got, list):
            passed = sorted(got) == sorted(expected)
        else:
            passed = got in expected
    else:
        passed = got == expected
    status = "[ PASS ]" if passed else "[ FAIL ]"
    print(f"{status} {name}")
    if not passed:
        print(f"         Got:      {got!r}")
        print(f"         Expected: {expected!r}")
        print(f"         Hint: {explanation}")
    return passed


def run():
    results = []
    print("=" * 65)
    print("PART 1: Method Properties")
    print("=" * 65)

    results.append(check(
        "Saliency formula",
        saliency_formula, "abs(gradient)",
        "Saliency = |∂f/∂x| — absolute value of the gradient."
    ))
    results.append(check(
        "Saliency: baseline needed?",
        saliency_baseline_needed, "no",
        "Saliency only uses the gradient at the input point — no baseline."
    ))
    results.append(check(
        "Saliency: forward+backward passes",
        saliency_passes, 1,
        "Saliency requires 1 forward pass + 1 backward pass."
    ))
    results.append(check(
        "Input×Gradient formula",
        ixg_formula, "input * gradient",
        "Input×Gradient = x_i * ∂f/∂x_i."
    ))
    results.append(check(
        "Input×Gradient: signed?",
        ixg_signed, "yes",
        "Input×Gradient returns signed values: positive=supports prediction, negative=opposes."
    ))
    results.append(check(
        "Input×Gradient: Taylor order",
        ixg_taylor_order, "first",
        "Input×Gradient is a first-order Taylor approximation: f(x) - f(0) ≈ sum(x_i * ∂f/∂x_i)."
    ))
    results.append(check(
        "GuidedBackprop: passes",
        gbp_passes, 1,
        "GuidedBackprop is a single modified backward pass."
    ))
    results.append(check(
        "GuidedBackprop: modifies backward?",
        gbp_modifies_backward, "yes",
        "GuidedBackprop zeroes out negative gradients in the backward pass through ReLUs."
    ))
    results.append(check(
        "GuidedBackprop: architecture dependent?",
        gbp_architecture_dependent, "yes",
        "Adebayo et al. (2018) showed GBP produces similar maps for trained and random models."
    ))
    results.append(check(
        "GuidedBackprop: satisfies impl. invariance?",
        gbp_satisfies_impl_invariance, "no",
        "GBP fails implementation invariance: different weight initializations get different attributions."
    ))
    results.append(check(
        "Deconvolution backward condition",
        deconv_condition, "gradient_positive",
        "Deconvolution zeroes gradient flow where the gradient itself is negative (not based on activation)."
    ))

    print("\n" + "=" * 65)
    print("PART 2: The Saturation Problem")
    print("=" * 65)

    results.append(check(
        "Gradient at saturation (x=1.0)",
        gradient_at_saturation, 0,
        "ReLU(3*1 - 6) = ReLU(-3) = 0. Gradient of inactive ReLU = 0."
    ))
    results.append(check(
        "Saliency at saturation",
        saliency_at_saturation, 0,
        "Saliency = |gradient| = |0| = 0."
    ))
    results.append(check(
        "Is x relevant?",
        is_x_relevant, "yes",
        "Moving from x=1 to x=2.5 changes output from 0 to 1.5 — x IS relevant."
    ))
    results.append(check(
        "Saliency correctly identifies relevance?",
        saliency_identifies_relevance, "no",
        "Saliency gives 0 attribution but the feature is relevant — saturation failure."
    ))
    results.append(check(
        "Method that resolves saturation",
        method_resolves_saturation, "IntegratedGradients",
        "IG integrates gradients from baseline (x=0) to input — crosses the activation threshold."
    ))

    print("\n" + "=" * 65)
    print("PART 3: The Sanity Check")
    print("=" * 65)

    results.append(check(
        "Expected correlation for faithful method",
        expected_correlation_faithful, "near 0",
        "A faithful method produces very different maps for trained vs random models."
    ))
    results.append(check(
        "Sanity check finding for GBP",
        sanity_check_gbp_finding, "b",
        "GBP shows high correlation — maps look similar for trained and random models."
    ))
    results.append(check(
        "GBP sanity check reason (answered?)",
        gbp_sanity_check_reason is not None and len(str(gbp_sanity_check_reason)) > 20,
        True,
        "Explain: GBP filters gradients based on ReLU sign, which depends on architecture, not weights."
    ))
    results.append(check(
        "Methods that pass sanity check",
        methods_that_pass_sanity_check,
        ["Saliency", "IntegratedGradients"],
        "Saliency and IG pass: their maps change dramatically for random models."
    ))

    print("\n" + "=" * 65)
    print("PART 4: SmoothGrad and NoiseTunnel")
    print("=" * 65)

    results.append(check(
        "SmoothGrad computation",
        smoothgrad_computation, "b",
        "SmoothGrad = mean gradient over many noise-perturbed inputs."
    ))
    results.append(check(
        "VarGrad computation",
        vargrad_computation, "c",
        "VarGrad = variance of gradients over noise-perturbed inputs."
    ))
    results.append(check(
        "NoiseTunnel samples param",
        noise_tunnel_samples_param, "nt_samples",
        "In Captum NoiseTunnel, 'nt_samples' controls the number of noise samples."
    ))
    results.append(check(
        "SmoothGrad fixes saturation?",
        smoothgrad_fixes_saturation, "no",
        "SmoothGrad averages over nearby points. If all nearby points are also saturated, "
        "the average gradient is still 0. Saturation requires a baseline path (IG), not noise."
    ))
    results.append(check(
        "SmoothGrad reduces noise?",
        smoothgrad_reduces_noise, "yes",
        "Yes — averaging over many noisy samples reduces the variance of the attribution estimate."
    ))

    print("\n" + "=" * 65)
    print("PART 5: API Debugging")
    print("=" * 65)

    results.append(check(
        "Snippet A bug",
        snippet_a_bug, ["a", "b"],
        "Two bugs: model not in eval() mode AND input missing requires_grad_(True)."
    ))
    results.append(check(
        "Snippet B bug",
        snippet_b_bug, "b",
        "target should be an integer class index (e.g., 207), not a string."
    ))
    results.append(check(
        "Snippet C bug",
        snippet_c_bug, "a",
        "Missing .detach() before .numpy() — cannot convert gradient-tracking tensor to numpy."
    ))

    print("\n" + "=" * 65)
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"RESULTS: {passed}/{total} checks passed")
    if passed == total:
        print("All checks passed! You understand gradient attribution methods.")
    else:
        print(f"{total - passed} checks need attention. Review the relevant guide sections.")
    print("=" * 65)


if __name__ == "__main__":
    run()
