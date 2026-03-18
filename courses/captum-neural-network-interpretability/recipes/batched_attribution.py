"""
Recipe: Batched attribution for large datasets.

Patterns:
1. Native Captum batch attribution (single forward/backward over batch)
2. Iterative batch attribution with progress tracking
3. Streaming attribution (generator, no memory accumulation)
4. Parallel attribution with concurrent.futures
5. Attribution with DataLoader integration

Copy-paste ready. Requires: captum, torch, torchvision
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator, Iterator, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from captum.attr import IntegratedGradients, Saliency


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 1: NATIVE CAPTUM BATCH ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def attribute_batch_native(model, images, targets, method="saliency", n_steps=30):
    """
    Compute attribution for an entire batch in a single call.

    Captum methods accept (batch_size, *input_shape) tensors natively.
    This is the most efficient pattern for GPU computation.

    Args:
        model:   torch.nn.Module in eval mode
        images:  (N, C, H, W) tensor
        targets: (N,) tensor of class indices
        method:  'saliency' or 'integrated_gradients'
        n_steps: IG steps (unused for saliency)

    Returns:
        attrs: (N, C, H, W) attribution tensor

    Usage:
        images, labels = next(iter(dataloader))
        attrs = attribute_batch_native(model, images, labels)
        # attrs[i] is the attribution for images[i]
    """
    def forward_func(x):
        return model(x)

    if method == "saliency":
        sal = Saliency(forward_func)
        # targets must be a list or tensor of length N
        attrs = sal.attribute(images.requires_grad_(True), target=targets, abs=True)

    elif method == "integrated_gradients":
        ig = IntegratedGradients(forward_func)
        baseline = torch.zeros_like(images)
        attrs = ig.attribute(images, baseline, target=targets, n_steps=n_steps)

    else:
        raise ValueError(f"Unknown method: {method!r}")

    return attrs.detach()


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 2: ITERATIVE DATALOADER ATTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def attribute_dataloader(
    model,
    dataloader: DataLoader,
    method: str = "saliency",
    n_steps: int = 30,
    max_batches: Optional[int] = None,
    verbose: bool = True,
) -> List[dict]:
    """
    Compute attributions for all examples in a DataLoader.

    Args:
        model:       torch.nn.Module in eval mode
        dataloader:  DataLoader yielding (inputs, labels) batches
        method:      Attribution method
        n_steps:     IG steps
        max_batches: Optional limit on number of batches
        verbose:     Print progress every 10 batches

    Returns:
        List of dicts, one per example: {attrs, pred_class, true_label, correct}

    Usage:
        results = attribute_dataloader(model, val_loader, method='saliency')
        n_correct = sum(r['correct'] for r in results)
        print(f"Accuracy: {n_correct}/{len(results)}")
    """
    results = []
    t0 = time.perf_counter()

    for batch_idx, (images, labels) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Predictions for target determination
        with torch.no_grad():
            logits = model(images)
            pred_classes = logits.argmax(dim=1)

        # Batch attribution
        attrs = attribute_batch_native(model, images, pred_classes, method, n_steps)

        # Collect per-example results
        for i in range(images.shape[0]):
            results.append({
                "attrs": attrs[i].sum(dim=0),           # (H, W) spatial map
                "attrs_full": attrs[i],                  # (C, H, W)
                "pred_class": int(pred_classes[i].item()),
                "true_label": int(labels[i].item()),
                "correct": pred_classes[i].item() == labels[i].item(),
            })

        if verbose and (batch_idx + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            n = (batch_idx + 1) * dataloader.batch_size
            print(f"  [{batch_idx+1:4d} batches / {n:5d} examples]  "
                  f"{n/elapsed:.1f} ex/s")

    elapsed = time.perf_counter() - t0
    if verbose:
        n_correct = sum(r["correct"] for r in results)
        print(f"Done: {len(results)} examples in {elapsed:.1f}s  "
              f"({len(results)/elapsed:.1f} ex/s)  "
              f"Accuracy: {n_correct}/{len(results)} = {n_correct/len(results):.1%}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 3: STREAMING ATTRIBUTION GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def stream_attributions(
    model,
    dataloader: DataLoader,
    method: str = "saliency",
    n_steps: int = 30,
    max_examples: int = 10000,
) -> Generator[dict, None, None]:
    """
    Generator that yields attribution results one example at a time.

    Does not accumulate results in memory — suitable for writing to disk
    incrementally or for very large datasets.

    Usage:
        for result in stream_attributions(model, dataloader):
            save_to_disk(result)  # write immediately, don't accumulate
    """
    count = 0

    for images, labels in dataloader:
        if count >= max_examples:
            break

        with torch.no_grad():
            pred_classes = model(images).argmax(dim=1)

        attrs = attribute_batch_native(model, images, pred_classes, method, n_steps)

        for i in range(images.shape[0]):
            if count >= max_examples:
                return
            yield {
                "attrs": attrs[i].sum(dim=0),
                "pred_class": int(pred_classes[i].item()),
                "true_label": int(labels[i].item()),
            }
            count += 1


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 4: PARALLEL ATTRIBUTION WITH THREAD POOL
# ─────────────────────────────────────────────────────────────────────────────

def attribute_parallel(
    model,
    example_list: List[torch.Tensor],
    target_list: List[int],
    method: str = "saliency",
    n_workers: int = 4,
    n_steps: int = 30,
) -> List[torch.Tensor]:
    """
    Compute attributions for a list of individual examples in parallel
    using a ThreadPoolExecutor.

    Note: CPU-bound attribution does not benefit from Python threads (GIL).
    This pattern is most useful for I/O-bound workloads (loading images,
    writing results to disk) where attribution itself is fast (Saliency).

    Args:
        example_list: List of (1, C, H, W) tensors
        target_list:  List of target class indices (same length)
        n_workers:    Number of threads

    Returns:
        List of attribution tensors in the same order as input

    Usage:
        examples = [img.unsqueeze(0) for img in dataset[:100]]
        targets = [model(ex).argmax(1).item() for ex in examples]
        attrs = attribute_parallel(model, examples, targets, n_workers=4)
    """
    def _attribute_single(args):
        idx, inp, tgt = args
        attrs = attribute_batch_native(model, inp, torch.tensor([tgt]), method, n_steps)
        return idx, attrs[0]

    results = [None] * len(example_list)
    tasks = [(i, ex, tgt) for i, (ex, tgt) in enumerate(zip(example_list, target_list))]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_attribute_single, task): task[0] for task in tasks}
        for future in as_completed(futures):
            idx, attrs = future.result()
            results[idx] = attrs

    return results


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 5: ATTRIBUTION WITH MEMORY BUDGET
# ─────────────────────────────────────────────────────────────────────────────

def attribute_with_memory_budget(
    model,
    images: torch.Tensor,
    targets: torch.Tensor,
    method: str = "saliency",
    max_batch_size: int = 8,
    n_steps: int = 30,
) -> torch.Tensor:
    """
    Compute attribution for a large batch by splitting into sub-batches.

    Prevents OOM errors when the full batch is too large for GPU memory.

    Args:
        images:        (N, C, H, W) tensor
        targets:       (N,) target tensor
        max_batch_size: Max examples per sub-batch

    Returns:
        (N, C, H, W) attribution tensor

    Usage:
        # Safe batch attribution even for N=1000 on GPU
        attrs = attribute_with_memory_budget(model, all_images, all_targets,
                                              max_batch_size=16)
    """
    N = images.shape[0]
    all_attrs = []

    for start in range(0, N, max_batch_size):
        end = min(start + max_batch_size, N)
        batch_images = images[start:end]
        batch_targets = targets[start:end]

        batch_attrs = attribute_batch_native(
            model, batch_images, batch_targets, method, n_steps
        )
        all_attrs.append(batch_attrs)

    return torch.cat(all_attrs, dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    from torchvision.models import resnet18, ResNet18_Weights
    import torchvision.transforms as T
    from torch.utils.data import Subset

    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval()

    transform = T.Compose([
        T.Resize(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = CIFAR10(root="/tmp/cifar10", train=False, download=True, transform=transform)
    subset = Subset(dataset, list(range(40)))
    loader = DataLoader(subset, batch_size=8, shuffle=False)

    print("=== Native batch attribution ===")
    images, labels = next(iter(loader))
    with torch.no_grad():
        preds = model(images).argmax(dim=1)
    attrs = attribute_batch_native(model, images, preds, method="saliency")
    print(f"Batch shape: {images.shape} → Attribution shape: {attrs.shape}")

    print("\n=== DataLoader attribution ===")
    results = attribute_dataloader(model, loader, method="saliency", max_batches=5, verbose=True)
    print(f"Total results: {len(results)}")

    print("\n=== Memory-budget attribution ===")
    all_images = torch.stack([dataset[i][0] for i in range(20)])
    all_targets = torch.tensor([model(all_images[i:i+1]).argmax(1).item() for i in range(20)])
    attrs_safe = attribute_with_memory_budget(model, all_images, all_targets,
                                               max_batch_size=4)
    print(f"Memory-safe attribution: {attrs_safe.shape}")
