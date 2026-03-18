"""
Recipe: Token attribution for any HuggingFace sequence classification model.

Pattern: LayerIntegratedGradients on the embedding layer,
subword aggregation, colored HTML visualization.

Copy-paste ready. Requires: captum, torch, transformers
"""

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from captum.attr import LayerIntegratedGradients


# ─────────────────────────────────────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name, device="cpu"):
    """
    Load a HuggingFace sequence classifier and tokenizer.

    Usage:
        model, tokenizer = load_model("textattack/bert-base-uncased-SST-2")
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device).eval()
    return model, tokenizer


def get_embedding_layer(model):
    """
    Auto-detect the embedding layer for BERT, RoBERTa, DistilBERT, ALBERT.

    Usage:
        emb_layer = get_embedding_layer(model)
        lig = LayerIntegratedGradients(forward_func, emb_layer)
    """
    if hasattr(model, "bert"):
        return model.bert.embeddings
    if hasattr(model, "roberta"):
        return model.roberta.embeddings
    if hasattr(model, "distilbert"):
        return model.distilbert.embeddings
    if hasattr(model, "albert"):
        return model.albert.embeddings
    raise AttributeError(
        "Cannot auto-detect embedding layer. "
        "Pass the layer manually: model.your_model.embeddings"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZATION AND BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text, tokenizer, max_length=128, device="cpu"):
    """
    Tokenize text and return tensors on device.

    Returns:
        input_ids, attention_mask, token_type_ids (may be None for GPT-2 etc.)
    """
    enc = tokenizer(
        text, return_tensors="pt",
        max_length=max_length, truncation=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    token_type_ids = enc.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
    return input_ids, attention_mask, token_type_ids


def build_pad_baseline(input_ids, tokenizer):
    """
    Build PAD token baseline for BERT-family models.

    Replaces all content tokens with [PAD], keeps [CLS] and [SEP].
    This produces near-neutral predictions — a well-defined reference point.
    """
    special_ids = {
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
    }
    baseline = torch.full_like(input_ids, tokenizer.pad_token_id)
    for pos in range(input_ids.shape[1]):
        if input_ids[0, pos].item() in special_ids:
            baseline[0, pos] = input_ids[0, pos]
    return baseline


# ─────────────────────────────────────────────────────────────────────────────
# ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def compute_token_attribution(
    model, tokenizer, text, target_class=None,
    n_steps=50, device="cpu",
):
    """
    Compute LayerIntegratedGradients token attribution for a text string.

    Args:
        model:        HuggingFace classification model in eval mode
        tokenizer:    Matching tokenizer
        text:         Input string
        target_class: Class to attribute toward. None → predicted class.
        n_steps:      IG integration steps
        device:       Computation device

    Returns:
        dict with keys:
            tokens (list of str), attributions (np.ndarray, shape: seq_len),
            prediction (dict), delta (float), text (str)

    Usage:
        model, tokenizer = load_model("textattack/bert-base-uncased-SST-2")
        result = compute_token_attribution(model, tokenizer, "Great movie!")
        print_colored_attributions(result)
    """
    input_ids, attention_mask, token_type_ids = tokenize(text, tokenizer, device=device)

    def forward_func(ids, mask=None, ttype=None):
        return model(input_ids=ids, attention_mask=mask,
                     token_type_ids=ttype).logits

    # Prediction
    with torch.no_grad():
        logits = forward_func(input_ids, attention_mask, token_type_ids)
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = int(probs.argmax().item())
        confidence = float(probs[pred_class].item())

    if target_class is None:
        target_class = pred_class

    # Build PAD baseline
    baseline_ids = build_pad_baseline(input_ids, tokenizer)

    # LayerIG
    embedding_layer = get_embedding_layer(model)
    lig = LayerIntegratedGradients(forward_func, embedding_layer)

    attrs, delta = lig.attribute(
        input_ids,
        baseline_ids,
        additional_forward_args=(attention_mask, token_type_ids),
        target=target_class,
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    # Sum over embedding dimension → one score per token
    token_attributions = attrs.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

    return {
        "text": text,
        "tokens": tokens,
        "attributions": token_attributions,
        "prediction": {
            "class_index": pred_class,
            "confidence": confidence,
        },
        "target_class": target_class,
        "delta": float(delta.item()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUBWORD AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_subwords(tokens, attributions, skip_special=True):
    """
    Merge WordPiece ## subword tokens into whole-word scores by summing.

    Usage:
        words, word_attrs = aggregate_subwords(result["tokens"], result["attributions"])
    """
    special = {"[CLS]", "[SEP]", "[PAD]"}
    words, word_attrs = [], []
    cur_word, cur_attr = None, 0.0

    for tok, attr in zip(tokens, attributions):
        if skip_special and tok in special:
            continue
        if tok.startswith("##"):
            cur_word = (cur_word or "") + tok[2:]
            cur_attr += float(attr)
        else:
            if cur_word is not None:
                words.append(cur_word)
                word_attrs.append(cur_attr)
            cur_word, cur_attr = tok, float(attr)

    if cur_word is not None:
        words.append(cur_word)
        word_attrs.append(cur_attr)

    return words, np.array(word_attrs)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def print_colored_attributions(result, labels=None):
    """
    Print token attributions with ANSI terminal colors.

    green = positive attribution, red = negative, white = neutral

    Usage:
        result = compute_token_attribution(model, tokenizer, "Brilliant film!")
        print_colored_attributions(result, labels=["NEG", "POS"])
    """
    tokens = result["tokens"]
    attrs = result["attributions"]
    max_attr = np.abs(attrs).max() + 1e-8

    pred_cls = result["prediction"]["class_index"]
    label_str = labels[pred_cls] if labels else str(pred_cls)
    print(f"\n  {result['text']!r}")
    print(f"  → {label_str} ({result['prediction']['confidence']:.1%})  "
          f"δ={result['delta']:.5f}")
    print()

    line = "  "
    for tok, attr in zip(tokens, attrs):
        if tok in {"[CLS]", "[SEP]", "[PAD]"}:
            continue
        norm = attr / max_attr
        if norm > 0.15:
            color = "\033[92m"   # green
        elif norm < -0.15:
            color = "\033[91m"   # red
        else:
            color = "\033[97m"   # white
        line += f"{color}{tok}\033[0m "
    print(line)
    print()


def attribution_to_html(result, labels=None):
    """
    Convert token attributions to an HTML string with background colors.

    Positive attribution → green background; negative → red background.

    Returns:
        HTML string renderable in a Jupyter notebook or browser.

    Usage:
        html = attribution_to_html(result, labels=["NEG", "POS"])
        from IPython.display import HTML, display
        display(HTML(html))
    """
    tokens = result["tokens"]
    attrs = result["attributions"]
    max_attr = np.abs(attrs).max() + 1e-8

    pred_cls = result["prediction"]["class_index"]
    label_str = labels[pred_cls] if labels else str(pred_cls)

    html = [f"<div style='font-family:monospace; margin:10px'>"]
    html.append(f"<b>{result['text']}</b> → <em>{label_str} "
                f"({result['prediction']['confidence']:.1%})</em><br><br>")

    for tok, attr in zip(tokens, attrs):
        if tok in {"[CLS]", "[SEP]", "[PAD]"}:
            continue
        norm = attr / max_attr
        if norm > 0:
            r, g, b = int(255 * (1 - norm * 0.6)), 255, int(255 * (1 - norm * 0.6))
        else:
            r, g, b = 255, int(255 * (1 + norm * 0.6)), int(255 * (1 + norm * 0.6))
        bg = f"rgb({r},{g},{b})"
        html.append(f"<span style='background:{bg}; padding:2px 4px; "
                    f"border-radius:3px; margin:1px'>{tok}</span> ")

    html.append("</div>")
    return "".join(html)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MODEL_NAME = "textattack/bert-base-uncased-SST-2"
    LABELS = ["NEGATIVE", "POSITIVE"]

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = load_model(MODEL_NAME)

    texts = [
        "The film was absolutely brilliant and deeply moving.",
        "A terrible waste of time with no redeeming qualities.",
        "The movie was NOT boring at all.",
    ]

    for text in texts:
        result = compute_token_attribution(model, tokenizer, text, n_steps=50)
        print_colored_attributions(result, labels=LABELS)

        words, word_attrs = aggregate_subwords(result["tokens"], result["attributions"])
        top_idx = np.argmax(np.abs(word_attrs))
        print(f"  Most important word: '{words[top_idx]}' (attr={word_attrs[top_idx]:+.4f})")
        print()
