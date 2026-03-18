"""
Captum Neural Network Interpretability
Production text attribution pipeline template.

Supports any HuggingFace AutoModelForSequenceClassification model.
Covers: LayerIntegratedGradients, baseline selection, subword aggregation,
and colored HTML visualization.

Usage:
    from text_attribution_template import TextAttributionPipeline

    pipeline = TextAttributionPipeline(
        model_name="textattack/bert-base-uncased-SST-2",
        labels=["NEGATIVE", "POSITIVE"],
    )
    result = pipeline.attribute("This film was absolutely brilliant.")
    pipeline.print_colored(result)
    pipeline.plot_bar(result, save_path="token_attribution.png")
"""

from typing import Dict, List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from captum.attr import LayerConductance, LayerIntegratedGradients


# ─────────────────────────────────────────────────────────────────────────────
# SUBWORD AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────


def aggregate_subwords(
    tokens: List[str],
    attributions: np.ndarray,
    skip_special: bool = True,
) -> Tuple[List[str], np.ndarray]:
    """
    Merge WordPiece subword tokens into whole-word attribution scores.

    Tokens with '##' prefix are merged with the preceding token by summing
    their attribution scores.

    Args:
        tokens:        List of token strings (e.g., ['[CLS]', 'un', '##be', 'liev', '##able'])
        attributions:  Float array of shape (seq_len,) — one score per token
        skip_special:  If True, exclude [CLS], [SEP], [PAD] tokens

    Returns:
        (word_list, word_attributions) with one entry per word
    """
    special = {"[CLS]", "[SEP]", "[PAD]"}
    word_list: List[str] = []
    word_attrs: List[float] = []
    cur_word: Optional[str] = None
    cur_attr: float = 0.0

    for tok, attr in zip(tokens, attributions):
        if skip_special and tok in special:
            continue
        if tok.startswith("##"):
            cur_word = (cur_word or "") + tok[2:]
            cur_attr += float(attr)
        else:
            if cur_word is not None:
                word_list.append(cur_word)
                word_attrs.append(cur_attr)
            cur_word = tok
            cur_attr = float(attr)

    if cur_word is not None:
        word_list.append(cur_word)
        word_attrs.append(cur_attr)

    return word_list, np.array(word_attrs)


# ─────────────────────────────────────────────────────────────────────────────
# TEXT ATTRIBUTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────


class TextAttributionPipeline:
    """
    Production text attribution pipeline for HuggingFace sequence classifiers.

    Attributes:
        model:      AutoModelForSequenceClassification in eval mode
        tokenizer:  AutoTokenizer matching the model
        labels:     List of class label strings
        max_length: Maximum tokenization length
    """

    # Special token IDs treated as padding in baselines
    BASELINE_STRATEGIES = Literal["pad", "mask", "zero_embedding"]

    def __init__(
        self,
        model_name: str,
        labels: Optional[List[str]] = None,
        max_length: int = 128,
        device: str = "cpu",
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(device).eval()
        self.labels = labels or [f"class_{i}" for i in range(self.model.config.num_labels)]
        self.max_length = max_length
        self.device = device

    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize a text string.

        Returns dict with: input_ids, attention_mask, token_type_ids
        """
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded.get("attention_mask", None),
            "token_type_ids": encoded.get("token_type_ids", None),
        }

    def build_baseline(
        self,
        input_ids: torch.Tensor,
        strategy: str = "pad",
    ) -> torch.Tensor:
        """
        Build a baseline input_ids tensor.

        Strategies:
            'pad':  Replace all non-special tokens with [PAD] — recommended for BERT
            'mask': Replace all non-special tokens with [MASK]
        """
        special_ids = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        }

        if strategy == "pad":
            baseline = torch.full_like(input_ids, self.tokenizer.pad_token_id)
            # Keep CLS and SEP at their original positions
            for pos in range(input_ids.shape[1]):
                if input_ids[0, pos].item() in special_ids:
                    baseline[0, pos] = input_ids[0, pos]
            return baseline

        elif strategy == "mask":
            baseline = input_ids.clone()
            mask_id = getattr(self.tokenizer, "mask_token_id", self.tokenizer.pad_token_id)
            for pos in range(input_ids.shape[1]):
                if input_ids[0, pos].item() not in special_ids:
                    baseline[0, pos] = mask_id
            return baseline

        else:
            raise ValueError(f"Unknown baseline strategy: {strategy!r}")

    def forward_func(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits

    def predict(self, text: str) -> Dict:
        """
        Predict class and confidence for a text string.

        Returns dict with: class_index, class_name, confidence, probs
        """
        enc = self.tokenize(text)
        with torch.no_grad():
            logits = self.forward_func(**enc)
            probs = torch.softmax(logits, dim=1)[0]
            pred_class = int(probs.argmax().item())

        return {
            "class_index": pred_class,
            "class_name": self.labels[pred_class],
            "confidence": float(probs[pred_class].item()),
            "probs": probs.cpu(),
        }

    def attribute(
        self,
        text: str,
        target_class: Optional[int] = None,
        baseline_strategy: str = "pad",
        n_steps: int = 50,
        return_delta: bool = True,
        aggregate_subwords_: bool = True,
    ) -> Dict:
        """
        Compute LayerIntegratedGradients attribution for a text input.

        Args:
            text:                Input text string
            target_class:        Class to attribute toward. If None, uses predicted class.
            baseline_strategy:   Baseline type: 'pad' or 'mask'
            n_steps:             IG integration steps
            return_delta:        Whether to include convergence delta
            aggregate_subwords_: Whether to merge ## subword tokens into words

        Returns:
            Dict with: text, tokens, words, word_attributions, token_attributions,
                       prediction, delta, method
        """
        enc = self.tokenize(text)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        token_type_ids = enc.get("token_type_ids")

        # Predict
        with torch.no_grad():
            logits = self.forward_func(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(logits, dim=1)[0]
            pred_class = int(probs.argmax().item())

        if target_class is None:
            target_class = pred_class

        # Build baseline
        baseline_ids = self.build_baseline(input_ids, baseline_strategy)

        # LayerIG targets the embedding layer
        embedding_layer = self._get_embedding_layer()
        lig = LayerIntegratedGradients(self.forward_func, embedding_layer)

        attrs, delta = lig.attribute(
            input_ids,
            baseline_ids,
            additional_forward_args=(attention_mask, token_type_ids),
            target=target_class,
            n_steps=n_steps,
            return_convergence_delta=True,
        )

        # Aggregate over embedding dimension → one score per token
        token_attributions = attrs.sum(dim=-1).squeeze(0).detach().cpu().numpy()

        # Convert token IDs to token strings
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

        # Optionally aggregate subwords → word-level scores
        if aggregate_subwords_:
            words, word_attributions = aggregate_subwords(tokens, token_attributions)
        else:
            # Filter specials only
            non_special = {"[CLS]", "[SEP]", "[PAD]"}
            words = [t for t in tokens if t not in non_special]
            word_attributions = np.array([
                a for t, a in zip(tokens, token_attributions)
                if t not in non_special
            ])

        return {
            "text": text,
            "tokens": tokens,
            "token_attributions": token_attributions,
            "words": words,
            "word_attributions": word_attributions,
            "prediction": {
                "class_index": pred_class,
                "class_name": self.labels[pred_class],
                "confidence": float(probs[pred_class].item()),
            },
            "target_class": target_class,
            "delta": float(delta.item()),
            "method": "layer_integrated_gradients",
            "baseline_strategy": baseline_strategy,
            "n_steps": n_steps,
        }

    def attribute_all_layers(
        self,
        text: str,
        target_class: Optional[int] = None,
        n_steps: int = 15,
    ) -> Dict:
        """
        Compute LayerConductance for all encoder layers.

        Returns:
            Dict with: layer_scores (list of float, one per layer),
                       layer_names, text, prediction
        """
        enc = self.tokenize(text)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
        token_type_ids = enc.get("token_type_ids")

        with torch.no_grad():
            logits = self.forward_func(input_ids, attention_mask, token_type_ids)
            pred_class = int(logits.argmax(dim=1).item())

        if target_class is None:
            target_class = pred_class

        baseline_ids = self.build_baseline(input_ids)
        encoder_layers = self._get_encoder_layers()

        layer_scores: List[float] = []
        layer_names: List[str] = []

        for i, layer in enumerate(encoder_layers):
            lc = LayerConductance(self.forward_func, layer)
            cond = lc.attribute(
                input_ids,
                baseline_ids,
                additional_forward_args=(attention_mask, token_type_ids),
                target=target_class,
                n_steps=n_steps,
            )
            layer_scores.append(float(cond.abs().sum().item()))
            layer_names.append(f"L{i}")

        return {
            "text": text,
            "layer_scores": layer_scores,
            "layer_names": layer_names,
            "prediction": {
                "class_index": pred_class,
                "class_name": self.labels[pred_class],
            },
        }

    def _get_embedding_layer(self) -> torch.nn.Module:
        """Resolve the embedding layer based on model architecture."""
        # BERT-family
        if hasattr(self.model, "bert"):
            return self.model.bert.embeddings
        # RoBERTa-family
        if hasattr(self.model, "roberta"):
            return self.model.roberta.embeddings
        # DistilBERT
        if hasattr(self.model, "distilbert"):
            return self.model.distilbert.embeddings
        # ALBERT
        if hasattr(self.model, "albert"):
            return self.model.albert.embeddings
        raise AttributeError(
            "Cannot auto-detect embedding layer. "
            "Pass embedding_layer= explicitly to LayerIntegratedGradients."
        )

    def _get_encoder_layers(self) -> List[torch.nn.Module]:
        """Return list of transformer encoder layers."""
        if hasattr(self.model, "bert"):
            return list(self.model.bert.encoder.layer)
        if hasattr(self.model, "roberta"):
            return list(self.model.roberta.encoder.layer)
        if hasattr(self.model, "distilbert"):
            return list(self.model.distilbert.transformer.layer)
        raise AttributeError("Cannot auto-detect encoder layers.")

    def print_colored(self, result: Dict, width: int = 80) -> None:
        """
        Print token attributions with ANSI color coding to the terminal.

        Positive attribution → green; negative → red; neutral → white.
        """
        words = result["words"]
        attrs = result["word_attributions"]
        max_attr = np.abs(attrs).max() + 1e-8

        print(f"\n  Text: {result['text']!r}")
        print(f"  Prediction: {result['prediction']['class_name']} "
              f"({result['prediction']['confidence']:.1%})")
        print(f"  Target class: {result['target_class']}")
        print(f"  Convergence delta: {result['delta']:.5f}")
        print()

        line = "  "
        for word, attr in zip(words, attrs):
            norm = attr / max_attr
            if norm > 0.1:
                color = "\033[92m"  # green
            elif norm < -0.1:
                color = "\033[91m"  # red
            else:
                color = "\033[97m"  # white
            reset = "\033[0m"
            token_str = f"{color}{word}{reset} "
            if len(line) + len(word) + 1 > width:
                print(line)
                line = "  "
            line += token_str
        if line.strip():
            print(line)
        print()

    def plot_bar(
        self,
        result: Dict,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 4),
    ) -> plt.Figure:
        """
        Plot word attribution scores as a horizontal bar chart.

        Positive attributions → green; negative → red.
        """
        words = result["words"]
        attrs = result["word_attributions"]

        colors = ["#2ca02c" if a >= 0 else "#d62728" for a in attrs]

        fig, ax = plt.subplots(figsize=figsize)
        y = np.arange(len(words))
        ax.barh(y, attrs, color=colors, alpha=0.85, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(words, fontsize=10)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Attribution Score", fontsize=11)
        ax.set_title(
            f"Token Attribution — {result['prediction']['class_name']} "
            f"({result['prediction']['confidence']:.1%})\n"
            f"\"{result['text'][:60]}{'...' if len(result['text']) > 60 else ''}\"",
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
    pipeline = TextAttributionPipeline(
        model_name="textattack/bert-base-uncased-SST-2",
        labels=["NEGATIVE", "POSITIVE"],
    )

    texts = [
        "The film was absolutely brilliant and deeply moving.",
        "A terrible waste of time with no redeeming qualities.",
        "The movie was not bad at all, actually quite good.",
    ]

    for text in texts:
        result = pipeline.attribute(text, n_steps=50)
        pipeline.print_colored(result)

    # Layer profile
    layer_result = pipeline.attribute_all_layers(texts[0])
    top3 = sorted(range(len(layer_result["layer_scores"])),
                  key=lambda i: -layer_result["layer_scores"][i])[:3]
    print(f"Top-3 layers for '{texts[0][:30]}...': {top3}")
