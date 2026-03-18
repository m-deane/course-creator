"""
Recipe: Custom baseline strategies for attribution methods.

The baseline defines "absence of information" — it is the reference point
from which Integrated Gradients measures the importance of each feature.
Baseline choice affects attribution values significantly.

Patterns:
1. Image baselines: black, white, blur, mean, noise, inpaint
2. Text baselines: PAD, MASK, uniform, zero-embedding
3. Tabular baselines: zero, mean, median, distribution sample
4. Evaluating baseline quality (near-neutral prediction check)
5. Multiple baselines with GradientSHAP / DeepLIFTSHAP

Copy-paste ready. Requires: captum, torch, transformers
"""

from typing import List, Optional

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE BASELINES
# ─────────────────────────────────────────────────────────────────────────────

def black_image_baseline(inputs: torch.Tensor) -> torch.Tensor:
    """
    All-zeros baseline (black image in normalized space).

    Standard for ImageNet models. Represents "no image" or complete darkness.
    Corresponds to a mean-centered image at the lower bound.
    """
    return torch.zeros_like(inputs)


def white_image_baseline(inputs: torch.Tensor) -> torch.Tensor:
    """
    All-ones baseline (white image in unnormalized space).

    In normalized space this is a large positive value. Use only when
    the domain has a natural white background (e.g., document images).
    """
    return torch.ones_like(inputs)


def mean_image_baseline(inputs: torch.Tensor,
                          background_samples: torch.Tensor) -> torch.Tensor:
    """
    Mean over a background dataset.

    Represents a "typical example" rather than an absence of signal.
    Produces more stable attributions when the model is sensitive to image statistics.

    Args:
        background_samples: (N, C, H, W) tensor of training/background examples

    Usage:
        bg = torch.stack([dataset[i][0] for i in range(200)])
        baseline = mean_image_baseline(inputs, bg)
    """
    mean = background_samples.mean(dim=0, keepdim=True)
    return mean.expand_as(inputs)


def gaussian_blur_baseline(inputs: torch.Tensor, kernel_size: int = 31,
                             sigma: float = 15.0) -> torch.Tensor:
    """
    Heavily blurred version of the input image.

    Retains low-frequency structure (average color, brightness) while
    removing high-frequency detail. Useful when black creates
    attribution artifacts at object edges.

    Args:
        kernel_size: Size of Gaussian kernel (must be odd)
        sigma:       Blur standard deviation

    Usage:
        baseline = gaussian_blur_baseline(inputs, sigma=25.0)
    """
    # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Build separable Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)

    C = inputs.shape[1]
    kernel_2d = kernel_2d.repeat(C, 1, 1, 1)  # (C, 1, k, k)

    pad = kernel_size // 2
    blurred = torch.nn.functional.conv2d(
        inputs, kernel_2d, padding=pad, groups=C
    )
    return blurred.detach()


def random_noise_baseline(inputs: torch.Tensor, sigma: float = 0.1,
                            n_samples: int = 1) -> torch.Tensor:
    """
    Gaussian noise baseline.

    Returns n_samples baseline tensors (shape: (n_samples, C, H, W)).
    Use with GradientSHAP by passing as background.

    Args:
        sigma:     Standard deviation of Gaussian noise
        n_samples: Number of noise samples

    Usage:
        baseline = random_noise_baseline(inputs, sigma=0.1, n_samples=50)
        # Use as background for GradientSHAP
    """
    if n_samples == 1:
        return torch.randn_like(inputs) * sigma
    return torch.randn(n_samples, *inputs.shape[1:]) * sigma


def uniform_color_baseline(inputs: torch.Tensor,
                             color: List[float] = None) -> torch.Tensor:
    """
    Solid color baseline (same color across all spatial positions).

    Args:
        color: [R, G, B] values in normalized space. Defaults to gray (0.5, 0.5, 0.5)
               in unnormalized space, converted to normalized space.

    Usage:
        # Gray baseline in ImageNet normalized space
        baseline = uniform_color_baseline(inputs)
    """
    if color is None:
        # ImageNet normalization: (0.5 - mean) / std for each channel
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
        imagenet_std  = torch.tensor([0.229, 0.224, 0.225])
        color_normalized = (torch.tensor([0.5, 0.5, 0.5]) - imagenet_mean) / imagenet_std
        color = color_normalized.tolist()

    baseline = torch.zeros_like(inputs)
    for c, val in enumerate(color):
        baseline[:, c, :, :] = val
    return baseline


# ─────────────────────────────────────────────────────────────────────────────
# TEXT BASELINES
# ─────────────────────────────────────────────────────────────────────────────

def pad_token_baseline(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Replace all non-special tokens with [PAD].

    This is the standard baseline for BERT-family models. PAD tokens produce
    near-neutral predictions, providing a well-defined attribution reference.

    Usage:
        baseline = pad_token_baseline(input_ids, tokenizer)
        # baseline[0] = [CLS_ID, PAD, PAD, ..., PAD, SEP_ID, PAD, ...]
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


def mask_token_baseline(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Replace all content tokens with [MASK].

    Slightly different semantics than PAD: [MASK] means "present but unknown",
    while [PAD] means "absent". For some models, MASK produces predictions
    closer to 0.5 than PAD.

    Usage:
        baseline = mask_token_baseline(input_ids, tokenizer)
    """
    special_ids = {
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
    }
    mask_id = getattr(tokenizer, "mask_token_id", tokenizer.pad_token_id)
    baseline = input_ids.clone()
    for pos in range(input_ids.shape[1]):
        if input_ids[0, pos].item() not in special_ids:
            baseline[0, pos] = mask_id
    return baseline


def uniform_token_baseline(input_ids: torch.Tensor,
                             vocab_size: int) -> torch.Tensor:
    """
    Uniformly random token baseline.

    Each position filled with a random vocabulary token.
    Not recommended as a primary baseline (noisy results) but useful
    for sensitivity analysis when combined with multiple samples.

    Usage:
        baseline = uniform_token_baseline(input_ids, tokenizer.vocab_size)
    """
    return torch.randint_like(input_ids, 0, vocab_size)


def zero_embedding_baseline_func(input_embeds: torch.Tensor) -> torch.Tensor:
    """
    Zero-embedding baseline for use with LayerIntegratedGradients when
    targeting the embedding layer directly (e.g., for GPT-2 where no PAD exists).

    Usage:
        lig = LayerIntegratedGradients(forward_func, model.transformer.wte)
        # Baseline is passed as embedding, not as input_ids
        baseline_func = lambda emb: torch.zeros_like(emb)
    """
    return torch.zeros_like(input_embeds)


# ─────────────────────────────────────────────────────────────────────────────
# TABULAR BASELINES
# ─────────────────────────────────────────────────────────────────────────────

def zero_baseline(inputs: torch.Tensor) -> torch.Tensor:
    """
    All-zeros baseline.

    Use when features are normalized (zero-centered) so that zero
    represents the absence/mean of each feature.
    """
    return torch.zeros_like(inputs)


def training_mean_baseline(inputs: torch.Tensor,
                             training_data: torch.Tensor) -> torch.Tensor:
    """
    Training set mean as baseline.

    Represents "a typical sample" — the average input over the training distribution.
    Attribution then measures deviation from average behavior.

    Args:
        training_data: (N, n_features) tensor of training examples

    Usage:
        baseline = training_mean_baseline(inputs, X_train)
    """
    mean = training_data.mean(dim=0, keepdim=True)
    return mean.expand_as(inputs)


def training_median_baseline(inputs: torch.Tensor,
                               training_data: torch.Tensor) -> torch.Tensor:
    """
    Training set median as baseline.

    More robust to outliers than the mean. Recommended when features
    have skewed distributions.
    """
    median = training_data.median(dim=0).values.unsqueeze(0)
    return median.expand_as(inputs)


def random_background_samples(training_data: torch.Tensor,
                                n_samples: int = 100) -> torch.Tensor:
    """
    Random sample from training data as multi-baseline for SHAP methods.

    Returns (n_samples, n_features) tensor for use with
    GradientSHAP and DeepLIFTSHAP.

    Usage:
        background = random_background_samples(X_train, n_samples=100)
        grad_shap = GradientShap(model)
        attrs = grad_shap.attribute(inputs, background, target=1)
    """
    idx = torch.randperm(training_data.shape[0])[:n_samples]
    return training_data[idx]


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE QUALITY EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_baseline_neutrality(model, baseline: torch.Tensor,
                                   target_class: int) -> dict:
    """
    Evaluate how "neutral" a baseline is by checking its prediction.

    A good baseline should produce predictions close to uniform (1/n_classes)
    for each class. For binary classification, ideal is ~0.5.

    Args:
        model:        torch.nn.Module in eval mode
        baseline:     Baseline tensor to evaluate
        target_class: Class index to check

    Returns:
        dict with: p_target (probability of target class), deviation_from_uniform,
                   diagnosis ('good' if deviation < 0.1, else 'poor')

    Usage:
        baseline = black_image_baseline(inputs)
        quality = evaluate_baseline_neutrality(model, baseline, target_class=0)
        print(f"Baseline quality: {quality['diagnosis']}")
    """
    with torch.no_grad():
        logits = model(baseline)
        probs = torch.softmax(logits, dim=1)[0]
        n_classes = probs.shape[0]
        p_target = float(probs[target_class].item())
        uniform_prob = 1.0 / n_classes
        deviation = abs(p_target - uniform_prob)

    return {
        "p_target": round(p_target, 4),
        "uniform_probability": round(uniform_prob, 4),
        "deviation_from_uniform": round(deviation, 4),
        "diagnosis": "good" if deviation < 0.15 else "poor",
    }


def compare_baselines(model, inputs: torch.Tensor, baseline_dict: dict,
                       target_class: int) -> dict:
    """
    Compare multiple baselines by their attribution properties.

    Args:
        baseline_dict: {'name': baseline_tensor} mapping

    Returns:
        dict mapping baseline name to quality evaluation

    Usage:
        baselines = {
            'black':  black_image_baseline(inputs),
            'blur':   gaussian_blur_baseline(inputs),
            'mean':   mean_image_baseline(inputs, bg_samples),
        }
        results = compare_baselines(model, inputs, baselines, target_class=3)
        for name, quality in results.items():
            print(f"{name}: {quality['diagnosis']} (p={quality['p_target']:.3f})")
    """
    results = {}
    for name, baseline in baseline_dict.items():
        results[name] = evaluate_baseline_neutrality(model, baseline, target_class)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torchvision.models import resnet18, ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    inputs = torch.randn(1, 3, 224, 224)
    bg_samples = torch.randn(100, 3, 224, 224)  # mock background

    target_class = 0

    baselines = {
        "black":  black_image_baseline(inputs),
        "white":  white_image_baseline(inputs),
        "blur":   gaussian_blur_baseline(inputs, sigma=20.0),
        "mean":   mean_image_baseline(inputs, bg_samples),
        "gray":   uniform_color_baseline(inputs),
    }

    print("Baseline quality comparison (p_target should be ≈ 0.001 for 1000-class model):")
    print()
    results = compare_baselines(model, inputs, baselines, target_class)
    for name, quality in results.items():
        print(f"  {name:<8}: p(target)={quality['p_target']:.5f}  "
              f"deviation={quality['deviation_from_uniform']:.5f}  "
              f"[{quality['diagnosis']}]")
