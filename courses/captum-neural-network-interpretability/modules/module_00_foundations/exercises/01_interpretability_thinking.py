"""
Module 00 — Exercise 01: Interpretability Thinking

Self-check exercise: classify interpretability methods by taxonomy,
identify the right method for each scenario, and reason about trade-offs.

Run this file directly with: python 01_interpretability_thinking.py
Each check() call will print PASS or FAIL with an explanation.
"""

# ============================================================
# PART 1: Taxonomy Classification
# ============================================================
# For each method, fill in its properties.
# Values are strings: "yes" or "no", or from a fixed set.

# Instructions: Fill in the blanks (replace None with the correct value)

# --- Method: Saliency (vanilla gradient) ---
# Intrinsic or post-hoc?
saliency_posthoc = None          # "intrinsic" or "post-hoc"
# Local or global?
saliency_scope = None            # "local" or "global"
# Model-specific or model-agnostic?
saliency_specificity = None      # "model-specific" or "model-agnostic"
# Attribution type?
saliency_attr_type = None        # "input", "layer", or "neuron"
# Satisfies implementation invariance axiom?
saliency_impl_invariance = None  # "yes" or "no"
# Satisfies sensitivity axiom?
saliency_sensitivity = None      # "yes" or "no"

# --- Method: GradCAM ---
# Intrinsic or post-hoc?
gradcam_posthoc = None
# Local or global?
gradcam_scope = None
# Model-specific or model-agnostic?
gradcam_specificity = None
# Attribution type?
gradcam_attr_type = None         # "input", "layer", or "neuron"
# Requires differentiability (gradients)?
gradcam_needs_gradients = None   # "yes" or "no"

# --- Method: LIME ---
# Intrinsic or post-hoc?
lime_posthoc = None
# Local or global?
lime_scope = None
# Model-specific or model-agnostic?
lime_specificity = None
# Faster or slower than Integrated Gradients for a single prediction?
lime_speed_vs_ig = None          # "faster" or "slower"

# --- Method: Integrated Gradients ---
# Intrinsic or post-hoc?
ig_posthoc = None
# Satisfies BOTH sensitivity AND implementation invariance?
ig_satisfies_both_axioms = None  # "yes" or "no"
# Does it require a baseline?
ig_needs_baseline = None         # "yes" or "no"
# Approximately how many forward+backward passes for n_steps=50?
ig_approx_passes = None          # "1", "50", "500", or "5000"

# --- Method: Decision Tree ---
# Intrinsic or post-hoc?
dt_posthoc = None                # "intrinsic" or "post-hoc"
# Model-specific or model-agnostic?
dt_specificity = None            # "model-specific" or "model-agnostic"


# ============================================================
# PART 2: Scenario Matching
# ============================================================
# For each scenario, choose the best method from the list:
# Options: "Saliency", "IntegratedGradients", "GradCAM",
#          "FeatureAblation", "SHAP", "LIME", "LayerConductance"

# Scenario 1:
# You need to explain a PyTorch CNN's prediction for a medical image.
# A regulator requires the explanation to be provably faithful
# (satisfies completeness property). Speed is not a constraint.
scenario_1_method = None

# Scenario 2:
# You need a quick visual heatmap showing which region of an image
# caused the classification. Clean, spatial localization is more
# important than pixel-precise faithfulness. Single forward pass budget.
scenario_2_method = None

# Scenario 3:
# You need to explain predictions from a decision tree ensemble
# (XGBoost) on tabular financial data. The model is NOT PyTorch-based.
# You cannot compute gradients.
scenario_3_method = None

# Scenario 4:
# You want to understand which of 50 input features a tabular neural
# network ignores vs. relies on. You have 1 hour of compute budget.
# Each feature can be set to its mean value to "remove" it.
scenario_4_method = None

# Scenario 5:
# You need to understand which neurons in layer 3 of a CNN are
# most important for classifying a specific image.
# You need activation-level (not input-level) explanation.
scenario_5_method = None

# Scenario 6:
# A junior ML engineer says "our model's accuracy is 96% on the test
# set so it must be learning the right features." How would you challenge
# this assumption? What method would you use?
# Write your answer as a string below.
scenario_6_answer = None  # e.g., "I would use X because..."


# ============================================================
# PART 3: Baseline Reasoning
# ============================================================
# For Integrated Gradients, the baseline choice matters.
# Choose the most appropriate baseline for each use case:
# Options: "zero (black image)", "blurred image", "mean of training set",
#          "random noise (averaged)", "domain-specific (e.g., [MASK] token)"

# Use case A: Explaining a sentiment classification model for the sentence
# "The product quality is excellent." You want to know which words matter
# relative to a "neutral" reference.
baseline_nlp = None

# Use case B: Explaining an image classifier for a chest X-ray.
# You want attributions relative to "no clinical signal."
baseline_xray = None

# Use case C: Explaining a tabular credit scoring model.
# The baseline should represent a "typical" application.
baseline_tabular = None


# ============================================================
# SELF-CHECK ENGINE
# ============================================================

def check(name, got, expected, explanation):
    """Report pass/fail for each answer."""
    if got is None:
        print(f"[ TODO ] {name}: not answered yet")
        return False
    if isinstance(expected, list):
        passed = got in expected
    else:
        passed = got == expected
    status = "[ PASS ]" if passed else "[ FAIL ]"
    print(f"{status} {name}")
    if not passed:
        print(f"         Your answer: {got!r}")
        print(f"         Expected:    {expected!r}")
        print(f"         Hint: {explanation}")
    return passed


def run_all_checks():
    results = []
    print("=" * 60)
    print("PART 1: Taxonomy Classification")
    print("=" * 60)

    results.append(check(
        "Saliency: post-hoc?",
        saliency_posthoc, "post-hoc",
        "Saliency is computed after training; it is not an intrinsic property of the model."
    ))
    results.append(check(
        "Saliency: scope",
        saliency_scope, "local",
        "Saliency explains one prediction at a time — it varies per input."
    ))
    results.append(check(
        "Saliency: specificity",
        saliency_specificity, "model-specific",
        "Saliency requires gradient computation — only works on differentiable models."
    ))
    results.append(check(
        "Saliency: attribution type",
        saliency_attr_type, "input",
        "Saliency produces a gradient w.r.t. the input — attribution is in input space."
    ))
    results.append(check(
        "Saliency: implementation invariance",
        saliency_impl_invariance, "yes",
        "Saliency uses standard gradients which are implementation-invariant."
    ))
    results.append(check(
        "Saliency: sensitivity axiom",
        saliency_sensitivity, "no",
        "Saliency fails sensitivity: gradients can be zero even for relevant features (saturation)."
    ))
    results.append(check(
        "GradCAM: post-hoc?",
        gradcam_posthoc, "post-hoc",
        "GradCAM is a post-hoc explanation method."
    ))
    results.append(check(
        "GradCAM: scope",
        gradcam_scope, "local",
        "GradCAM explains one prediction for one input — it is local."
    ))
    results.append(check(
        "GradCAM: specificity",
        gradcam_specificity, "model-specific",
        "GradCAM requires convolutional layers with spatial feature maps."
    ))
    results.append(check(
        "GradCAM: attribution type",
        gradcam_attr_type, "layer",
        "GradCAM produces a heatmap at a specific convolutional layer — layer attribution."
    ))
    results.append(check(
        "GradCAM: needs gradients",
        gradcam_needs_gradients, "yes",
        "GradCAM uses gradients w.r.t. the target layer's activations."
    ))
    results.append(check(
        "LIME: post-hoc?",
        lime_posthoc, "post-hoc",
        "LIME builds a surrogate explanation after training."
    ))
    results.append(check(
        "LIME: scope",
        lime_scope, "local",
        "LIME builds a LOCAL surrogate — Linear Interpretable Model-Agnostic Explanations."
    ))
    results.append(check(
        "LIME: specificity",
        lime_specificity, "model-agnostic",
        "LIME only uses input-output pairs — it works on any model type."
    ))
    results.append(check(
        "LIME: speed vs IG",
        lime_speed_vs_ig, "slower",
        "LIME requires 1000+ model forward passes for a single explanation. IG uses ~50."
    ))
    results.append(check(
        "IG: post-hoc?",
        ig_posthoc, "post-hoc",
        "Integrated Gradients is computed post-training."
    ))
    results.append(check(
        "IG: satisfies both axioms",
        ig_satisfies_both_axioms, "yes",
        "IG is the unique gradient-based method satisfying both sensitivity and implementation invariance."
    ))
    results.append(check(
        "IG: needs baseline",
        ig_needs_baseline, "yes",
        "IG measures attribution relative to a baseline — the baseline choice matters."
    ))
    results.append(check(
        "IG: approx passes for n_steps=50",
        ig_approx_passes, "50",
        "IG with n_steps=50 does 50 forward+backward passes along the interpolation path."
    ))
    results.append(check(
        "Decision Tree: post-hoc?",
        dt_posthoc, "intrinsic",
        "A decision tree IS interpretable by design — no post-hoc method needed."
    ))
    results.append(check(
        "Decision Tree: specificity",
        dt_specificity, "model-specific",
        "Decision tree paths are specific to the tree structure — not applicable to other models."
    ))

    print("\n" + "=" * 60)
    print("PART 2: Scenario Matching")
    print("=" * 60)

    results.append(check(
        "Scenario 1 (faithful, PyTorch CNN)",
        scenario_1_method, "IntegratedGradients",
        "IG satisfies the completeness property (faithfulness) and works natively with PyTorch."
    ))
    results.append(check(
        "Scenario 2 (quick spatial heatmap)",
        scenario_2_method, "GradCAM",
        "GradCAM produces spatially coherent heatmaps in one forward pass."
    ))
    results.append(check(
        "Scenario 3 (XGBoost, no gradients)",
        scenario_3_method, ["SHAP", "LIME"],
        "SHAP (TreeSHAP specifically) is purpose-built for tree ensembles. LIME also works."
    ))
    results.append(check(
        "Scenario 4 (tabular, feature importance, 1hr budget)",
        scenario_4_method, ["FeatureAblation", "SHAP"],
        "FeatureAblation systematically removes each feature — natural for tabular data."
    ))
    results.append(check(
        "Scenario 5 (neuron importance in layer 3)",
        scenario_5_method, "LayerConductance",
        "LayerConductance provides neuron-level importance scores within a specified layer."
    ))
    results.append(check(
        "Scenario 6: answered?",
        scenario_6_answer is not None and len(str(scenario_6_answer)) > 20,
        True,
        "Provide a written response explaining which method you would use and why."
    ))

    print("\n" + "=" * 60)
    print("PART 3: Baseline Reasoning")
    print("=" * 60)

    results.append(check(
        "NLP baseline (sentiment, neutral reference)",
        baseline_nlp,
        ["domain-specific (e.g., [MASK] token)", "domain-specific"],
        "For text, [MASK] tokens represent 'no information' in the vocabulary. "
        "Zero embeddings are less meaningful."
    ))
    results.append(check(
        "Chest X-ray baseline",
        baseline_xray,
        ["zero (black image)", "blurred image"],
        "A black image or blurred image represents 'no diagnostic signal' in medical imaging."
    ))
    results.append(check(
        "Tabular baseline (typical application)",
        baseline_tabular,
        ["mean of training set", "mean of training set"],
        "For tabular models, the feature mean represents a 'typical' or 'average' input."
    ))

    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"RESULTS: {passed}/{total} checks passed")
    if passed == total:
        print("All checks passed! You have mastered the interpretability taxonomy.")
    else:
        print(f"{total - passed} checks need attention.")
        print("Review the taxonomy guide and re-attempt the failed items.")
    print("=" * 60)


if __name__ == "__main__":
    run_all_checks()
