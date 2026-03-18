"""
Recipe: Integrated Gradients on any custom PyTorch model.

Pattern: wrap any model with a forward function that returns logits,
then apply IG with appropriate baseline and target.

Copy-paste ready. Requires: captum, torch
"""

import torch
from captum.attr import IntegratedGradients


# ─────────────────────────────────────────────────────────────────────────────
# MINIMAL RECIPE
# ─────────────────────────────────────────────────────────────────────────────

def attribute_ig(model, inputs, target_class, baseline=None, n_steps=50):
    """
    Compute Integrated Gradients for any model.

    Args:
        model:        torch.nn.Module in eval mode
        inputs:       (1, *input_shape) tensor
        target_class: int — output class index to attribute toward
        baseline:     Optional baseline tensor. Defaults to zeros.
        n_steps:      Integration steps. 50 is standard; 200 for compliance.

    Returns:
        attrs (same shape as inputs), convergence_delta (float)

    Usage:
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        attrs, delta = attribute_ig(model, x, target_class=281)
        print(f"Convergence delta: {delta:.5f}")
    """
    model.eval()

    if baseline is None:
        baseline = torch.zeros_like(inputs)

    # The forward_func must return raw logits (not softmax)
    # If your model returns something else, wrap it here
    ig = IntegratedGradients(lambda x: model(x))

    attrs, delta = ig.attribute(
        inputs,
        baseline,
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    return attrs, float(delta.item())


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-OUTPUT MODELS (e.g., bounding box + classification)
# ─────────────────────────────────────────────────────────────────────────────

def attribute_ig_multi_output(model, inputs, output_index, class_index,
                               baseline=None, n_steps=50):
    """
    IG for models that return a tuple/list of outputs.

    Args:
        model:        Model returning (output_0, output_1, ...) tuple
        output_index: Which output to attribute (e.g., 0 for classification head)
        class_index:  Class index within that output

    Usage:
        # Model returns (classification_logits, bbox_coords)
        attrs, delta = attribute_ig_multi_output(
            model, inputs, output_index=0, class_index=5
        )
    """
    def forward_func(x):
        outputs = model(x)
        if isinstance(outputs, (list, tuple)):
            return outputs[output_index]
        return outputs

    ig = IntegratedGradients(forward_func)
    if baseline is None:
        baseline = torch.zeros_like(inputs)

    attrs, delta = ig.attribute(
        inputs, baseline, target=class_index,
        n_steps=n_steps, return_convergence_delta=True,
    )
    return attrs, float(delta.item())


# ─────────────────────────────────────────────────────────────────────────────
# MODELS WITH ADDITIONAL INPUTS (e.g., BERT with attention mask)
# ─────────────────────────────────────────────────────────────────────────────

def attribute_ig_with_additional_args(model, input_ids, baseline_ids,
                                       attention_mask, token_type_ids,
                                       target_class, n_steps=50):
    """
    IG for models that require additional non-attributed inputs.

    Uses additional_forward_args to pass through attention_mask and token_type_ids
    without computing gradients with respect to them.

    Usage:
        # BERT-style model
        attrs, delta = attribute_ig_with_additional_args(
            model, input_ids, baseline_ids, attention_mask,
            token_type_ids, target_class=1
        )
    """
    def forward_func(input_ids_, attention_mask_=None, token_type_ids_=None):
        return model(
            input_ids=input_ids_,
            attention_mask=attention_mask_,
            token_type_ids=token_type_ids_,
        ).logits

    ig = IntegratedGradients(forward_func)
    attrs, delta = ig.attribute(
        input_ids,
        baseline_ids,
        additional_forward_args=(attention_mask, token_type_ids),
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    return attrs, float(delta.item())


# ─────────────────────────────────────────────────────────────────────────────
# CONVERGENCE CHECK
# ─────────────────────────────────────────────────────────────────────────────

def verify_completeness(model, inputs, attrs, target_class, baseline=None):
    """
    Verify the completeness axiom: sum(attrs) ≈ f(inputs) - f(baseline).

    Returns the absolute gap. Should be < 0.01 for n_steps=50.

    Usage:
        attrs, delta = attribute_ig(model, inputs, target_class=3)
        gap = verify_completeness(model, inputs, attrs, target_class=3)
        print(f"Completeness gap: {gap:.5f}")
    """
    if baseline is None:
        baseline = torch.zeros_like(inputs)

    with torch.no_grad():
        f_x = torch.softmax(model(inputs), dim=1)[0, target_class].item()
        f_b = torch.softmax(model(baseline), dim=1)[0, target_class].item()

    attr_sum = attrs.sum().item()
    expected = f_x - f_b
    return abs(attr_sum - expected)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torchvision.models import resnet18, ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    x = torch.randn(1, 3, 224, 224)

    attrs, delta = attribute_ig(model, x, target_class=0, n_steps=50)
    gap = verify_completeness(model, x, attrs, target_class=0)

    print(f"Input shape:       {x.shape}")
    print(f"Attribution shape: {attrs.shape}")
    print(f"Convergence delta: {delta:.5f}")
    print(f"Completeness gap:  {gap:.5f}  {'PASS' if gap < 0.01 else 'WARN'}")
