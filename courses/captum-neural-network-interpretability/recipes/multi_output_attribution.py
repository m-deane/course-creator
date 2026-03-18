"""
Recipe: Attribution for models with multiple outputs.

Patterns:
1. Multi-head: model returns (logits_head_0, logits_head_1, ...)
2. Multi-task: model returns dict {"cls": ..., "seg": ...}
3. Multi-label: model returns independent sigmoid outputs
4. Object detection: attribute toward a specific bounding box score

Copy-paste ready. Requires: captum, torch
"""

import torch
from captum.attr import IntegratedGradients, Saliency


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 1: TUPLE OUTPUT — SELECT BY INDEX
# ─────────────────────────────────────────────────────────────────────────────

def attribute_tuple_output(model, inputs, output_index, class_index,
                            baseline=None, n_steps=50):
    """
    Attribute toward a specific output in a tuple-returning model.

    Args:
        model:        Model returning (out_0, out_1, ...) — any length tuple
        output_index: Which element of the tuple to attribute
        class_index:  Class within that output's logit vector

    Usage:
        # Multi-task model: (classification_logits, segmentation_logits)
        attrs, delta = attribute_tuple_output(
            model, inputs, output_index=0, class_index=3
        )
    """
    def forward_func(x):
        out = model(x)
        return out[output_index]

    ig = IntegratedGradients(forward_func)
    if baseline is None:
        baseline = torch.zeros_like(inputs)

    attrs, delta = ig.attribute(
        inputs, baseline, target=class_index,
        n_steps=n_steps, return_convergence_delta=True,
    )
    return attrs, float(delta.item())


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 2: DICT OUTPUT — SELECT BY KEY
# ─────────────────────────────────────────────────────────────────────────────

def attribute_dict_output(model, inputs, output_key, class_index,
                           baseline=None, n_steps=50):
    """
    Attribute toward a specific output in a dict-returning model.

    Args:
        output_key: String key selecting the output tensor

    Usage:
        # Model returns {"class_logits": ..., "bbox_offsets": ...}
        attrs, delta = attribute_dict_output(
            model, inputs, output_key="class_logits", class_index=7
        )
    """
    def forward_func(x):
        out = model(x)
        return out[output_key]

    ig = IntegratedGradients(forward_func)
    if baseline is None:
        baseline = torch.zeros_like(inputs)

    attrs, delta = ig.attribute(
        inputs, baseline, target=class_index,
        n_steps=n_steps, return_convergence_delta=True,
    )
    return attrs, float(delta.item())


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 3: MULTI-LABEL — SIGMOID OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def attribute_multilabel(model, inputs, label_index, baseline=None, n_steps=50):
    """
    Attribute toward a single sigmoid output in a multi-label classifier.

    Multi-label models output independent probabilities per label (no softmax).
    We attribute toward the probability of one specific label.

    Args:
        label_index: Index of the label to attribute toward

    Usage:
        # Binary model with 10 independent labels
        # Attribute toward "presence of cat" (label 3)
        attrs, delta = attribute_multilabel(model, inputs, label_index=3)
    """
    def forward_func(x):
        # Return the scalar probability for label_index
        # Shape: (batch,) — required when target=None
        return torch.sigmoid(model(x))[:, label_index:label_index+1]

    ig = IntegratedGradients(forward_func)
    if baseline is None:
        baseline = torch.zeros_like(inputs)

    # target=0 because we sliced to a single output column
    attrs, delta = ig.attribute(
        inputs, baseline, target=0,
        n_steps=n_steps, return_convergence_delta=True,
    )
    return attrs, float(delta.item())


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 4: REGRESSION OUTPUT — ATTRIBUTE TOWARD A SCALAR
# ─────────────────────────────────────────────────────────────────────────────

def attribute_regression(model, inputs, output_index=0, baseline=None, n_steps=50):
    """
    Attribute toward a scalar regression output.

    For regression, there is no class index — target is a specific output neuron.
    Use output_index=0 for single-output regression.

    Usage:
        # Model predicts house price as a scalar
        attrs, delta = attribute_regression(model, inputs)
    """
    def forward_func(x):
        # Ensure output is (batch, n_outputs) shape
        out = model(x)
        if out.dim() == 1:
            out = out.unsqueeze(1)
        return out

    ig = IntegratedGradients(forward_func)
    if baseline is None:
        baseline = torch.zeros_like(inputs)

    attrs, delta = ig.attribute(
        inputs, baseline, target=output_index,
        n_steps=n_steps, return_convergence_delta=True,
    )
    return attrs, float(delta.item())


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 5: SEQUENCE-TO-SEQUENCE — ATTRIBUTE EACH OUTPUT TOKEN
# ─────────────────────────────────────────────────────────────────────────────

def attribute_seq2seq_step(model_encoder, model_decoder, encoder_outputs,
                            decoder_input_ids, output_step, target_vocab_id,
                            embedding_layer, n_steps=30):
    """
    Attribute toward a specific output token in a seq2seq model.

    This covers the encoder input attribution for a single decoder output step.
    Run once per output token for full output attribution.

    Args:
        model_encoder:    Encoder module
        model_decoder:    Decoder module
        encoder_outputs:  Precomputed encoder hidden states
        decoder_input_ids: Decoder input up to current step
        output_step:      Which decoder step to attribute (0-indexed)
        target_vocab_id:  Vocabulary ID of the target output token
        embedding_layer:  Encoder embedding layer for LayerIG

    Note: This is a simplified pattern. Real seq2seq attribution is complex
    due to auto-regressive decoding. Consider using LayerIntegratedGradients
    on the encoder embeddings for each output step independently.
    """
    from captum.attr import LayerIntegratedGradients

    def forward_func(enc_input_ids):
        enc_out = model_encoder(enc_input_ids)
        dec_out = model_decoder(
            decoder_input_ids=decoder_input_ids,
            encoder_hidden_states=enc_out.last_hidden_state,
        )
        # Return logits for the specific output step
        return dec_out.logits[:, output_step, :]

    baseline_ids = torch.zeros_like(encoder_outputs)
    lig = LayerIntegratedGradients(forward_func, embedding_layer)

    attrs, delta = lig.attribute(
        encoder_outputs, baseline_ids,
        target=target_vocab_id,
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    return attrs, float(delta.item())


# ─────────────────────────────────────────────────────────────────────────────
# ATTRIBUTE ACROSS ALL CLASSES
# ─────────────────────────────────────────────────────────────────────────────

def attribute_all_classes(model, inputs, n_classes, method="ig",
                           baseline=None, n_steps=30):
    """
    Compute attribution toward every output class.

    Returns a dict mapping class_index → attribution_map.
    Useful for comparing what features matter for different classes.

    Usage:
        all_attrs = attribute_all_classes(model, inputs, n_classes=10)
        for cls_idx, attr_map in all_attrs.items():
            print(f"Class {cls_idx}: max attr = {attr_map.abs().max():.4f}")
    """
    if baseline is None:
        baseline = torch.zeros_like(inputs)

    def forward_func(x):
        return model(x)

    ig = IntegratedGradients(forward_func)
    results = {}

    for cls_idx in range(n_classes):
        attrs, delta = ig.attribute(
            inputs, baseline, target=cls_idx,
            n_steps=n_steps, return_convergence_delta=True,
        )
        results[cls_idx] = attrs.detach()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch.nn as nn

    # Multi-head model example
    class MultiHeadModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Sequential(nn.Linear(10, 32), nn.ReLU())
            self.head_a = nn.Linear(32, 5)   # 5-class classification
            self.head_b = nn.Linear(32, 3)   # 3-class auxiliary task

        def forward(self, x):
            h = self.shared(x)
            return self.head_a(h), self.head_b(h)

    model = MultiHeadModel().eval()
    inputs = torch.randn(1, 10)

    # Attribute toward head_a, class 2
    attrs, delta = attribute_tuple_output(model, inputs, output_index=0, class_index=2)
    print(f"Multi-head attribution: shape={attrs.shape}, delta={delta:.5f}")

    # Attribute toward all classes (single-head)
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 4))

        def forward(self, x):
            return self.net(x)

    simple_model = SimpleModel().eval()
    all_attrs = attribute_all_classes(simple_model, inputs, n_classes=4, n_steps=30)
    for cls_idx, attr_map in all_attrs.items():
        print(f"Class {cls_idx}: max |attr| = {attr_map.abs().max():.4f}")
