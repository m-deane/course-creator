"""
Recipe: Attribution caching strategies.

Patterns:
1. In-memory LRU cache (fast, volatile)
2. Disk-based cache (persistent across runs)
3. Decorator-based caching for attribution functions
4. Cache with invalidation based on model version
5. Pre-computing and storing attributions for a full dataset

Copy-paste ready. Requires: captum, torch, numpy
"""

import functools
import hashlib
import json
import os
import pickle
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from captum.attr import IntegratedGradients, Saliency


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 1: IN-MEMORY LRU CACHE
# ─────────────────────────────────────────────────────────────────────────────

class InMemoryAttributionCache:
    """
    Thread-safe in-memory LRU attribution cache.

    Key: MD5 of (inputs_bytes, method, target, n_steps)
    Value: attribution tensor (CPU)

    Usage:
        cache = InMemoryAttributionCache(max_size=512)
        attrs = cache.get_or_compute(model, inputs, method='saliency', target=3)
    """

    def __init__(self, max_size: int = 512) -> None:
        self._store: Dict[str, torch.Tensor] = {}
        self._order: list = []
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _key(self, inputs: torch.Tensor, method: str,
              target: int, n_steps: int) -> str:
        raw = inputs.cpu().numpy().tobytes()
        return hashlib.md5(
            raw + method.encode() + str(target).encode() + str(n_steps).encode()
        ).hexdigest()

    def get(self, inputs: torch.Tensor, method: str,
             target: int, n_steps: int = 50) -> Optional[torch.Tensor]:
        key = self._key(inputs, method, target, n_steps)
        with self._lock:
            if key in self._store:
                self._hits += 1
                return self._store[key]
            self._misses += 1
            return None

    def put(self, inputs: torch.Tensor, method: str, target: int,
             n_steps: int, value: torch.Tensor) -> None:
        key = self._key(inputs, method, target, n_steps)
        with self._lock:
            if len(self._order) >= self._max_size:
                oldest = self._order.pop(0)
                self._store.pop(oldest, None)
            self._store[key] = value.cpu()
            self._order.append(key)

    def get_or_compute(
        self, model, inputs: torch.Tensor, method: str = "saliency",
        target: Optional[int] = None, n_steps: int = 50
    ) -> torch.Tensor:
        """Return cached attribution or compute and cache it."""
        if target is None:
            with torch.no_grad():
                target = int(model(inputs).argmax(dim=1).item())

        cached = self.get(inputs, method, target, n_steps)
        if cached is not None:
            return cached

        attrs = _run_attribution(model, inputs, method, target, n_steps)
        self.put(inputs, method, target, n_steps, attrs)
        return attrs

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "size": len(self._store),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self.hit_rate:.1%}",
        }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._order.clear()


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 2: DISK-BASED PERSISTENT CACHE
# ─────────────────────────────────────────────────────────────────────────────

class DiskAttributionCache:
    """
    Persistent disk-based attribution cache using numpy files.

    Attributions are stored as .npy files keyed by SHA256 hash.
    Survives Python restarts and can be shared across processes.

    Usage:
        cache = DiskAttributionCache(cache_dir="./attribution_cache")
        attrs = cache.get_or_compute(model, inputs, method='ig', target=5)
    """

    def __init__(self, cache_dir: str = "./attribution_cache",
                  model_version: str = "v1") -> None:
        self.cache_dir = Path(cache_dir) / model_version
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _key(self, inputs: torch.Tensor, method: str,
              target: int, n_steps: int) -> str:
        raw = inputs.cpu().numpy().tobytes()
        payload = raw + method.encode() + str(target).encode() + str(n_steps).encode()
        return hashlib.sha256(payload).hexdigest()[:20]

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npy"

    def get(self, inputs: torch.Tensor, method: str,
             target: int, n_steps: int) -> Optional[torch.Tensor]:
        key = self._key(inputs, method, target, n_steps)
        path = self._path(key)
        if path.exists():
            self._hits += 1
            return torch.from_numpy(np.load(path))
        self._misses += 1
        return None

    def put(self, inputs: torch.Tensor, method: str, target: int,
             n_steps: int, value: torch.Tensor) -> None:
        key = self._key(inputs, method, target, n_steps)
        path = self._path(key)
        np.save(path, value.cpu().numpy())

    def get_or_compute(
        self, model, inputs: torch.Tensor, method: str = "integrated_gradients",
        target: Optional[int] = None, n_steps: int = 50
    ) -> torch.Tensor:
        if target is None:
            with torch.no_grad():
                target = int(model(inputs).argmax(dim=1).item())

        cached = self.get(inputs, method, target, n_steps)
        if cached is not None:
            return cached

        attrs = _run_attribution(model, inputs, method, target, n_steps)
        self.put(inputs, method, target, n_steps, attrs)
        return attrs

    def size_on_disk(self) -> str:
        """Return total cache size as a human-readable string."""
        total = sum(f.stat().st_size for f in self.cache_dir.glob("*.npy"))
        if total < 1024:
            return f"{total} B"
        elif total < 1024**2:
            return f"{total/1024:.1f} KB"
        else:
            return f"{total/1024**2:.1f} MB"

    def stats(self) -> dict:
        n_files = len(list(self.cache_dir.glob("*.npy")))
        return {
            "n_entries": n_files,
            "cache_dir": str(self.cache_dir),
            "disk_size": self.size_on_disk(),
            "hits": self._hits,
            "misses": self._misses,
        }

    def clear(self) -> None:
        for f in self.cache_dir.glob("*.npy"):
            f.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 3: DECORATOR-BASED CACHING
# ─────────────────────────────────────────────────────────────────────────────

_default_cache = InMemoryAttributionCache(max_size=256)


def cached_attribution(method: str = "saliency", n_steps: int = 50,
                        cache: Optional[InMemoryAttributionCache] = None):
    """
    Decorator: wrap any attribution function with caching.

    The wrapped function must accept (model, inputs, target) as positional args.

    Usage:
        @cached_attribution(method='integrated_gradients', n_steps=50)
        def my_attr_func(model, inputs, target):
            ig = IntegratedGradients(lambda x: model(x))
            return ig.attribute(inputs, torch.zeros_like(inputs), target=target)

        # First call computes; second call returns cached result
        attrs = my_attr_func(model, inputs, target=3)
        attrs = my_attr_func(model, inputs, target=3)  # cache hit
    """
    c = cache or _default_cache

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(model, inputs, target=None, **kwargs):
            if target is None:
                with torch.no_grad():
                    target = int(model(inputs).argmax(dim=1).item())

            cached = c.get(inputs, method, target, n_steps)
            if cached is not None:
                return cached

            result = func(model, inputs, target, **kwargs)
            if isinstance(result, tuple):
                attrs = result[0]
            else:
                attrs = result
            c.put(inputs, method, target, n_steps, attrs.detach())
            return result

        return wrapper

    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN 4: PRE-COMPUTE AND STORE FULL DATASET
# ─────────────────────────────────────────────────────────────────────────────

def precompute_attribution_store(
    model,
    dataloader,
    save_path: str,
    method: str = "saliency",
    n_steps: int = 30,
    max_examples: int = 10000,
    verbose: bool = True,
) -> None:
    """
    Pre-compute attributions for an entire dataset and save to a single .npz file.

    Useful for downstream analysis without rerunning attribution.
    Saves: attributions (N, H, W), pred_classes (N,), true_labels (N,)

    Args:
        save_path: Path to output .npz file

    Usage:
        precompute_attribution_store(model, val_loader, "val_attributions.npz")
        data = np.load("val_attributions.npz")
        attributions = data["attributions"]  # (N, H, W)
    """
    all_attrs, all_preds, all_labels = [], [], []
    count = 0

    for images, labels in dataloader:
        if count >= max_examples:
            break

        with torch.no_grad():
            preds = model(images).argmax(dim=1)

        attrs = _run_attribution(model, images, method, preds, n_steps)
        # Aggregate over channels → (N, H, W)
        attrs_2d = attrs.abs().sum(dim=1)

        all_attrs.append(attrs_2d.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        count += images.shape[0]

        if verbose and count % 100 == 0:
            print(f"  Precomputed {count}/{max_examples} examples")

    np.savez_compressed(
        save_path,
        attributions=np.concatenate(all_attrs, axis=0),
        pred_classes=np.concatenate(all_preds, axis=0),
        true_labels=np.concatenate(all_labels, axis=0),
    )

    if verbose:
        data = np.load(save_path)
        print(f"  Saved {len(data['attributions'])} attributions to {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SHARED ATTRIBUTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _run_attribution(
    model, inputs: torch.Tensor, method: str,
    target, n_steps: int
) -> torch.Tensor:
    """Internal: run the specified attribution method."""
    if isinstance(target, int):
        target_t = target
    else:
        target_t = target  # already tensor or list

    def forward(x):
        return model(x)

    if method == "saliency":
        sal = Saliency(forward)
        return sal.attribute(inputs.requires_grad_(True), target=target_t, abs=True).detach()

    elif method in ("ig", "integrated_gradients"):
        ig = IntegratedGradients(forward)
        baseline = torch.zeros_like(inputs)
        return ig.attribute(inputs, baseline, target=target_t, n_steps=n_steps).detach()

    else:
        raise ValueError(f"Unknown method in cache helper: {method!r}")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    from torchvision.models import resnet18, ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).eval()
    inputs = torch.randn(1, 3, 224, 224)

    # --- In-memory cache ---
    mem_cache = InMemoryAttributionCache(max_size=64)

    t0 = time.perf_counter()
    attrs1 = mem_cache.get_or_compute(model, inputs, method="saliency", target=0)
    cold_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    attrs2 = mem_cache.get_or_compute(model, inputs, method="saliency", target=0)
    warm_time = time.perf_counter() - t0

    print(f"In-memory cache:")
    print(f"  Cold time: {cold_time*1000:.2f} ms")
    print(f"  Warm time: {warm_time*1000:.4f} ms")
    print(f"  Speedup:   {cold_time/warm_time:.0f}x")
    print(f"  Stats:     {mem_cache.stats()}")

    # --- Disk cache ---
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        disk_cache = DiskAttributionCache(cache_dir=tmpdir, model_version="resnet18_v1")

        _ = disk_cache.get_or_compute(model, inputs, method="ig", target=5, n_steps=20)
        _ = disk_cache.get_or_compute(model, inputs, method="ig", target=5, n_steps=20)

        print(f"\nDisk cache stats: {disk_cache.stats()}")
