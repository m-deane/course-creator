"""
Module 07 — NLP & Transformer Interpretability: Self-Check Exercises

Exercises covering token attribution, attention vs. IG, and layer analysis.
"""

import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from captum.attr import LayerIntegratedGradients, LayerConductance

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

print("Loading BERT sentiment model...")
model_name = "textattack/bert-base-uncased-SST-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()
LABELS = ["NEGATIVE", "POSITIVE"]
print("Model ready.")


def tokenize(text):
    inp = tokenizer(text, return_tensors='pt', max_length=128, truncation=True)
    return inp['input_ids'], inp.get('attention_mask'), inp.get('token_type_ids')

def make_baseline(ids):
    return torch.full_like(ids, tokenizer.pad_token_id)

def forward_func(input_ids, attention_mask=None, token_type_ids=None):
    return model(input_ids=input_ids, attention_mask=attention_mask,
                 token_type_ids=token_type_ids).logits

def predict(text):
    ids, mask, ttype = tokenize(text)
    with torch.no_grad():
        logits = forward_func(ids, mask, ttype)
    probs = torch.softmax(logits, dim=1)[0]
    pred = probs.argmax().item()
    return pred, probs[pred].item(), LABELS[pred]


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1: Baseline Quality Check
#
# A good baseline should produce near-neutral (≈0.5) class probabilities.
# Test the PAD baseline and compare to a MASK baseline.
# ─────────────────────────────────────────────────────────────────────────────

def exercise_1_baseline_quality():
    """Compare PAD and MASK baselines for BERT sentiment."""
    print("\nExercise 1: Baseline Quality Check")
    print("-" * 50)

    test_text = "The movie was incredibly boring and disappointing."
    ids, mask, ttype = tokenize(test_text)

    # PAD baseline
    pad_baseline = make_baseline(ids)

    # MASK baseline (replace content tokens with [MASK])
    mask_baseline = ids.clone()
    special_ids = {tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id}
    for i in range(ids.shape[1]):
        if ids[0, i].item() not in special_ids:
            mask_baseline[0, i] = tokenizer.mask_token_id

    # Evaluate both baselines
    with torch.no_grad():
        for name, bl in [("PAD baseline", pad_baseline), ("MASK baseline", mask_baseline)]:
            logits = forward_func(bl, mask, ttype)
            probs = torch.softmax(logits, dim=1)[0]
            print(f"  {name}: NEG={probs[0]:.3f}, POS={probs[1]:.3f}")
            print(f"    → Ideal: close to [0.5, 0.5]")

    # Normal input prediction
    cls, conf, label = predict(test_text)
    print(f"\n  Normal text prediction: {label} ({conf:.1%})")
    print(f"  Good baseline: shifts prediction from {label} toward neutral")


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2: Efficiency Verification for LayerIG
#
# The convergence delta for LayerIG should satisfy:
# sum(attrs) ≈ f(input) - f(baseline)
# ─────────────────────────────────────────────────────────────────────────────

def exercise_2_efficiency_verification():
    """Verify LayerIG efficiency: attribution sum ≈ f(x) - f(x')."""
    print("\nExercise 2: LayerIG Efficiency Verification")
    print("-" * 50)

    text = "An absolutely stunning masterpiece of modern cinema."
    ids, mask, ttype = tokenize(text)
    baseline = make_baseline(ids)

    with torch.no_grad():
        f_x = torch.softmax(forward_func(ids, mask, ttype), dim=1)[0, 1].item()
        f_b = torch.softmax(forward_func(baseline, mask, ttype), dim=1)[0, 1].item()

    expected_sum = f_x - f_b
    pred_class = 1  # POSITIVE

    lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)
    attrs, delta = lig.attribute(
        ids, baseline, additional_forward_args=(mask, ttype),
        target=pred_class, n_steps=50, return_convergence_delta=True,
    )
    actual_sum = attrs.sum(dim=-1).squeeze(0).sum().item()

    print(f"  f(text):     {f_x:.4f}")
    print(f"  f(baseline): {f_b:.4f}")
    print(f"  Expected attribution sum: {expected_sum:.4f}")
    print(f"  Actual attribution sum:   {actual_sum:.4f}")
    print(f"  Convergence delta:        {delta.item():.6f}")
    print(f"  Efficiency gap:           {abs(actual_sum - expected_sum):.6f}")

    efficiency_gap = abs(actual_sum - expected_sum)
    print(f"\n  {'PASS' if efficiency_gap < 0.01 else 'WARN'}: Gap < 0.01 expected for n_steps=50")


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 3: Negation Attribution
#
# For "not boring", "NOT" should receive positive attribution.
# For "boring" alone, "boring" should receive high negative attribution.
# Compare the two.
# ─────────────────────────────────────────────────────────────────────────────

def exercise_3_negation_attribution():
    """Show that IG correctly attributes negation modifiers."""
    print("\nExercise 3: Negation Attribution")
    print("-" * 50)

    texts = [
        "The movie was boring and dull.",
        "The movie was NOT boring at all.",
    ]

    for text in texts:
        ids, mask, ttype = tokenize(text)
        baseline = make_baseline(ids)
        tokens = tokenizer.convert_ids_to_tokens(ids[0].tolist())

        with torch.no_grad():
            pred = forward_func(ids, mask, ttype).argmax(dim=1).item()

        lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)
        attrs, _ = lig.attribute(
            ids, baseline, additional_forward_args=(mask, ttype),
            target=pred, n_steps=40,
        )
        token_scores = attrs.sum(dim=-1).squeeze(0).detach().numpy()

        print(f"\n  Text: '{text}'")
        print(f"  Prediction: {LABELS[pred]}")
        print("  Token attributions:")
        for tok, score in zip(tokens, token_scores):
            if tok in {'[CLS]', '[SEP]', '[PAD]'}:
                continue
            bar = '▓' * max(0, int(abs(score) / (np.abs(token_scores).max() + 1e-8) * 15))
            sign = '+' if score > 0 else '-'
            print(f"    {tok:<15} {sign}{abs(score):.4f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 4: Layer Importance Profile
#
# Check which layers are most important for a sentiment prediction.
# Expected: layers 10-11 dominate for sentiment.
# ─────────────────────────────────────────────────────────────────────────────

def exercise_4_layer_profile():
    """Compute and display layer importance profile."""
    print("\nExercise 4: Layer Importance Profile")
    print("-" * 50)

    text = "A masterfully crafted and profoundly moving film experience."
    ids, mask, ttype = tokenize(text)
    baseline = make_baseline(ids)

    with torch.no_grad():
        pred = forward_func(ids, mask, ttype).argmax(dim=1).item()

    layer_scores = []
    print(f"  Text: '{text}'")
    print(f"  Prediction: {LABELS[pred]}")
    print()

    for i, enc_layer in enumerate(model.bert.encoder.layer):
        lc = LayerConductance(forward_func, enc_layer)
        cond = lc.attribute(ids, baseline, additional_forward_args=(mask, ttype),
                            target=pred, n_steps=10)
        score = cond.abs().sum().item()
        layer_scores.append(score)

    # Normalize and display
    max_score = max(layer_scores)
    print("  Layer importance profile (normalized):")
    for i, score in enumerate(layer_scores):
        normalized = score / max_score
        bar = '█' * int(normalized * 25)
        print(f"    Layer {i:2d}: {normalized:.3f}  {bar}")

    top3 = sorted(range(len(layer_scores)), key=lambda i: -layer_scores[i])[:3]
    print(f"\n  Top 3 most important layers: {top3}")
    print(f"  (Expected for sentiment: layers 9, 10, 11)")


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 5: Attention vs. IG Agreement
#
# For a clear positive sentence, measure Spearman correlation between
# attention weights and IG attributions.
# ─────────────────────────────────────────────────────────────────────────────

def exercise_5_attention_ig_agreement():
    """Compare attention and IG agreement quantitatively."""
    from scipy.stats import spearmanr

    print("\nExercise 5: Attention vs. IG Agreement")
    print("-" * 50)

    sentences = [
        "The film is absolutely brilliant.",
        "The movie was not bad, actually quite good.",  # negation case
        "A terribly disappointing waste of time.",
    ]

    for text in sentences:
        ids, mask, ttype = tokenize(text)
        baseline = make_baseline(ids)
        tokens = tokenizer.convert_ids_to_tokens(ids[0].tolist())

        # IG attribution
        with torch.no_grad():
            pred = forward_func(ids, mask, ttype).argmax(dim=1).item()
        lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)
        ig_attrs = lig.attribute(ids, baseline, additional_forward_args=(mask, ttype),
                                  target=pred, n_steps=30
                                 ).sum(dim=-1).squeeze(0).detach().numpy()

        # Attention (last layer, mean head, from CLS)
        with torch.no_grad():
            outputs = model(ids, attention_mask=mask, output_attentions=True)
        attn_last = outputs.attentions[-1].mean(dim=1).squeeze(0)[0].detach().numpy()

        # Trim to non-special tokens
        non_special = [i for i, t in enumerate(tokens) if t not in {'[CLS]', '[SEP]', '[PAD]'}]
        ig_trim = ig_attrs[non_special]
        at_trim = attn_last[non_special]

        r, pval = spearmanr(np.abs(ig_trim), at_trim)
        print(f"\n  '{text}'")
        print(f"  Prediction: {LABELS[pred]}")
        print(f"  Spearman r(|IG|, Attn): {r:+.4f}  (p={pval:.4f})")
        print(f"  {'High agreement' if r > 0.7 else 'Low/no agreement' if r < 0.3 else 'Moderate agreement'}")


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL EXERCISES
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("MODULE 07 — NLP INTERPRETABILITY: SELF-CHECK")
    print("=" * 60)

    exercise_1_baseline_quality()
    exercise_2_efficiency_verification()
    exercise_3_negation_attribution()
    exercise_4_layer_profile()
    exercise_5_attention_ig_agreement()

    print("\n" + "=" * 60)
    print("All exercises complete.")
    print("=" * 60)
