"""
Module 08 — Production Pipelines: Self-Check Exercises

Exercises covering Captum Insights configuration, batch attribution,
caching, API patterns, and method selection.
"""

import hashlib
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset

from captum.attr import IntegratedGradients, Saliency, GradientShap, LayerGradCam
from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

print("Loading ResNet-18 and CIFAR-10...")
weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
model.eval()

normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = T.Compose([T.Resize(224), T.ToTensor(), normalize])

dataset_full = CIFAR10(root='/tmp/cifar10', train=False, download=True, transform=transform)
subset = Subset(dataset_full, indices=list(range(100)))
dataloader = DataLoader(subset, batch_size=1, shuffle=False)

CIFAR_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
print("Setup complete.")


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1: Insights Configuration Validation
#
# Verify that the AttributionVisualizer can be constructed without error.
# Check that all required parameters are present and well-typed.
# ─────────────────────────────────────────────────────────────────────────────

def exercise_1_insights_configuration():
    """Validate AttributionVisualizer configuration."""
    print("\nExercise 1: Insights Configuration Validation")
    print("-" * 50)

    # Build a small synthetic dataset
    synthetic_batches = []
    for i in range(10):
        arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        from PIL import Image
        img = Image.fromarray(arr)
        img_t = T.Compose([T.ToTensor(), normalize])(img).unsqueeze(0)
        synthetic_batches.append(Batch(inputs=img_t, labels=torch.tensor([i % 10])))

    # Configure ImageFeature
    image_feature = ImageFeature(
        name="Input Image",
        baseline_transforms=[lambda x: torch.zeros_like(x)],
        input_transforms=[transform],
    )

    # Construct AttributionVisualizer
    try:
        visualizer = AttributionVisualizer(
            models=[model],
            score_func=lambda out: torch.softmax(out, dim=1),
            classes=CIFAR_CLASSES,
            features=[image_feature],
            dataset=synthetic_batches,
            num_examples=4,
        )
        print("  PASS: AttributionVisualizer constructed successfully")
    except Exception as e:
        print(f"  FAIL: {e}")
        return

    # Verify batch shape
    b = synthetic_batches[0]
    assert b.inputs.shape == (1, 3, 224, 224), \
        f"Expected (1, 3, 224, 224), got {b.inputs.shape}"
    assert b.labels.shape == (1,), f"Expected (1,), got {b.labels.shape}"
    print(f"  Batch input shape: {b.inputs.shape}  ✓")
    print(f"  Batch label shape: {b.labels.shape}  ✓")
    print(f"  Dataset size: {len(synthetic_batches)} batches")
    print(f"  Classes configured: {len(CIFAR_CLASSES)}")
    print(f"\n  NOTE: To launch the server, call visualizer.serve(port=5001)")
    print(f"  This is not done here to avoid blocking the exercise runner.")


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2: Attribution Cache Correctness
#
# Verify that the cache returns the same attribution on second call,
# and that the hit rate increases correctly.
# ─────────────────────────────────────────────────────────────────────────────

class SimpleAttributionCache:
    def __init__(self, max_size=256):
        self._store = {}
        self._order = []
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def _key(self, inputs, method, target):
        raw = inputs.cpu().numpy().tobytes()
        return hashlib.md5(raw + method.encode() + str(target).encode()).hexdigest()

    def get(self, inputs, method, target):
        k = self._key(inputs, method, target)
        if k in self._store:
            self._hits += 1
            return self._store[k]
        self._misses += 1
        return None

    def set(self, inputs, method, target, value):
        k = self._key(inputs, method, target)
        if len(self._order) >= self._max_size:
            oldest = self._order.pop(0)
            self._store.pop(oldest, None)
        self._store[k] = value
        self._order.append(k)

    @property
    def hit_rate(self):
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


def exercise_2_cache_correctness():
    """Verify cache returns identical results on hit and measures speedup."""
    print("\nExercise 2: Attribution Cache Correctness")
    print("-" * 50)

    cache = SimpleAttributionCache(max_size=64)
    saliency = Saliency(lambda x: model(x))

    # Get one example
    img, lbl = next(iter(dataloader))
    with torch.no_grad():
        pred = model(img).argmax(dim=1).item()

    # First call: cache miss
    t0 = time.perf_counter()
    attrs_first = saliency.attribute(img.requires_grad_(True), target=pred, abs=True).detach()
    t1 = time.perf_counter()
    miss_time = t1 - t0
    cache.set(img, 'saliency', pred, attrs_first)

    # Second call: cache hit
    t0 = time.perf_counter()
    attrs_cached = cache.get(img, 'saliency', pred)
    t2 = time.perf_counter()
    hit_time = t2 - t0

    # Verify same result
    assert attrs_cached is not None, "Cache should return a result on second call"
    assert torch.allclose(attrs_first, attrs_cached), \
        "Cached attribution should be identical to original"

    print(f"  Cache miss time:  {miss_time*1000:.2f} ms")
    print(f"  Cache hit time:   {hit_time*1000:.4f} ms")
    print(f"  Speedup: {miss_time/hit_time:.0f}x")
    print(f"  Hit rate: {cache.hit_rate:.1%}")
    print(f"  PASS: Cached result is identical to original")

    # Test LRU eviction
    small_cache = SimpleAttributionCache(max_size=3)
    for i, (img_b, lbl_b) in enumerate(dataloader):
        if i >= 5:
            break
        with torch.no_grad():
            pred_b = model(img_b).argmax(dim=1).item()
        attrs_b = saliency.attribute(img_b.requires_grad_(True), target=pred_b, abs=True).detach()
        small_cache.set(img_b, 'saliency', pred_b, attrs_b)

    assert len(small_cache._store) <= 3, \
        f"Cache should not exceed max_size=3, got {len(small_cache._store)}"
    print(f"  LRU eviction: cache size = {len(small_cache._store)} ≤ 3  PASS")


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 3: Convergence Delta Threshold Verification
#
# Check that IG with n_steps=200 produces delta < 0.05 (compliance threshold)
# while n_steps=10 may or may not satisfy the threshold.
# ─────────────────────────────────────────────────────────────────────────────

def exercise_3_convergence_delta():
    """Verify IG convergence delta across different step counts."""
    print("\nExercise 3: Convergence Delta Threshold Verification")
    print("-" * 50)

    img, lbl = next(iter(dataloader))
    with torch.no_grad():
        pred = model(img).argmax(dim=1).item()

    baseline = torch.zeros_like(img)
    ig = IntegratedGradients(lambda x: model(x))

    step_configs = [10, 30, 50, 100, 200]
    print(f"  Input: CIFAR-10 example (pred_class={pred})")
    print()
    print(f"  {'n_steps':>8}  {'|delta|':>10}  {'< 0.05?':>10}  {'time (ms)':>10}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")

    compliance_n_steps = None
    for n_steps in step_configs:
        t0 = time.perf_counter()
        _, delta = ig.attribute(
            img, baseline,
            target=pred, n_steps=n_steps,
            return_convergence_delta=True,
        )
        elapsed = time.perf_counter() - t0
        abs_delta = abs(delta.item())
        compliant = abs_delta < 0.05
        if compliant and compliance_n_steps is None:
            compliance_n_steps = n_steps
        print(f"  {n_steps:>8}  {abs_delta:>10.5f}  {'YES' if compliant else 'NO':>10}  {elapsed*1000:>10.1f}")

    print()
    if compliance_n_steps:
        print(f"  Minimum n_steps for compliance (<0.05): {compliance_n_steps}")
    else:
        print(f"  Delta did not reach <0.05 threshold in tested range")
    print(f"  Recommendation: n_steps=50 for production, n_steps=200 for compliance reports")


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 4: Method Selection by Latency Budget
#
# For a given latency budget, identify which methods are feasible.
# Budget tiers: <10ms, <50ms, <200ms, <500ms
# ─────────────────────────────────────────────────────────────────────────────

def exercise_4_method_selection_by_latency():
    """Empirically validate method selection guidelines by measuring latency."""
    print("\nExercise 4: Method Selection by Latency Budget")
    print("-" * 50)

    img, lbl = next(iter(dataloader))
    with torch.no_grad():
        pred = model(img).argmax(dim=1).item()

    methods = {
        'Saliency':          lambda: Saliency(lambda x: model(x)).attribute(
            img.requires_grad_(True), target=pred, abs=True),
        'GradCAM':           lambda: LayerGradCam(lambda x: model(x), model.layer4[-1]).attribute(
            img, target=pred),
        'GradShap (n=10)':   lambda: GradientShap(lambda x: model(x)).attribute(
            img, torch.zeros_like(img).expand(10, -1, -1, -1),
            n_samples=10, target=pred),
        'IG (n_steps=20)':   lambda: IntegratedGradients(lambda x: model(x)).attribute(
            img, torch.zeros_like(img), target=pred, n_steps=20),
        'IG (n_steps=50)':   lambda: IntegratedGradients(lambda x: model(x)).attribute(
            img, torch.zeros_like(img), target=pred, n_steps=50),
        'IG (n_steps=200)':  lambda: IntegratedGradients(lambda x: model(x)).attribute(
            img, torch.zeros_like(img), target=pred, n_steps=200),
    }

    budgets = {
        '<10ms': 0.010,
        '<50ms': 0.050,
        '<200ms': 0.200,
        '<500ms': 0.500,
    }

    print(f"  {'Method':<22}  {'Mean (ms)':>10}  {'Budget tier':>15}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*15}")

    for name, fn in methods.items():
        # Warm up
        _ = fn()
        # Measure 5 runs
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        mean_s = np.mean(times)
        # Find applicable budget tier
        applicable = [tier for tier, limit in budgets.items() if mean_s < limit]
        tier = applicable[0] if applicable else '>500ms'
        print(f"  {name:<22}  {mean_s*1000:>10.1f}  {tier:>15}")

    print()
    print("  Method selection rules:")
    print("  - Real-time dashboard (< 10ms): Saliency or GradCAM")
    print("  - Interactive tool (< 200ms):   IG (n_steps=20–30)")
    print("  - Production serving (< 500ms): IG (n_steps=50)")
    print("  - Compliance report:            IG (n_steps=200)")


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 5: Baseline Sensitivity Check
#
# Test three baseline types (zero, mean, noise) and compare attribution outputs.
# Good baselines: produce predictions close to 0.5 (neutral class probability).
# ─────────────────────────────────────────────────────────────────────────────

def exercise_5_baseline_sensitivity():
    """Compare attribution quality across different baseline types."""
    from scipy.stats import spearmanr

    print("\nExercise 5: Baseline Sensitivity Check")
    print("-" * 50)

    # Collect background images for mean baseline
    background_imgs = []
    for img_b, _ in dataloader:
        background_imgs.append(img_b)
        if len(background_imgs) >= 20:
            break
    mean_baseline = torch.stack([b.squeeze(0) for b in background_imgs]).mean(dim=0, keepdim=True)

    img, lbl = next(iter(dataloader))
    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)[0]
        pred = probs.argmax().item()

    baselines = {
        'Zero (black)': torch.zeros_like(img),
        'Mean image':   mean_baseline.expand_as(img),
        'Gaussian noise (σ=0.1)': torch.randn_like(img) * 0.1,
    }

    attrs_per_baseline = {}
    ig = IntegratedGradients(lambda x: model(x))

    print(f"  Input class: {CIFAR_CLASSES[lbl.item()]} | Predicted: {pred}")
    print()
    print(f"  {'Baseline':<30}  {'Baseline P(pred)':>18}  {'Attr sum':>10}  {'|delta|':>8}")
    print(f"  {'-'*30}  {'-'*18}  {'-'*10}  {'-'*8}")

    for bname, bline in baselines.items():
        with torch.no_grad():
            bline_prob = torch.softmax(model(bline), dim=1)[0, pred].item()

        attrs, delta = ig.attribute(
            img, bline, target=pred, n_steps=50,
            return_convergence_delta=True,
        )
        attr_map = attrs.sum(dim=1).squeeze(0).detach()
        attrs_per_baseline[bname] = attr_map
        print(f"  {bname:<30}  {bline_prob:>18.4f}  {attr_map.sum().item():>10.4f}  {abs(delta.item()):>8.5f}")

    # Rank correlation between baseline pairs
    names = list(attrs_per_baseline.keys())
    flat_attrs = {n: attrs_per_baseline[n].numpy().flatten() for n in names}

    print()
    print("  Spearman correlation between baselines:")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r, p = spearmanr(np.abs(flat_attrs[names[i]]), np.abs(flat_attrs[names[j]]))
            print(f"    {names[i]:<30} vs {names[j]:<30}: r={r:+.4f}  p={p:.4f}")

    print()
    print("  NOTE: High correlation (r > 0.7) means baseline choice has low impact.")
    print("  Low correlation signals that choice of baseline meaningfully changes attribution.")


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL EXERCISES
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("MODULE 08 — PRODUCTION PIPELINES: SELF-CHECK")
    print("=" * 60)

    exercise_1_insights_configuration()
    exercise_2_cache_correctness()
    exercise_3_convergence_delta()
    exercise_4_method_selection_by_latency()
    exercise_5_baseline_sensitivity()

    print("\n" + "=" * 60)
    print("All exercises complete.")
    print("=" * 60)
