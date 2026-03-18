"""
Module 06 — Concept & Example-Based Explanations: Self-Check Exercises

Exercises covering TCAV, TracIn, and SimilarityInfluence concepts.
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('concept_exercises_cache', exist_ok=True)
os.makedirs('concept_exercises_data', exist_ok=True)

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
model.eval()

transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

print("Loading CIFAR-10 test set (small subset)...")
test_data = CIFAR10(root='concept_exercises_data', train=False,
                    download=True, transform=transform)
np.random.seed(42)
subset_idx = np.random.choice(len(test_data), 200, replace=False)
test_subset = Subset(test_data, subset_idx)


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1: Manual CAV Training
#
# Train a linear probe (CAV) on ResNet layer4 activations to distinguish
# "bird" images from "car" images. This is the core TCAV operation.
# ─────────────────────────────────────────────────────────────────────────────

def exercise_1_manual_cav():
    """Train a CAV to separate two CIFAR-10 classes in ResNet representation space."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    print("Exercise 1: Manual CAV Training (Bird vs. Car)")
    print("-" * 50)

    # Collect activations for "bird" (class 2) and "automobile" (class 1)
    def get_activations(dataset, target_class, n_samples=50):
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        activations = []
        collected = 0
        with torch.no_grad():
            for imgs, labels in loader:
                bird_mask = (labels == target_class)
                if bird_mask.sum() == 0:
                    continue
                # Hook to capture avgpool output
                pool_output = []
                hook = model.avgpool.register_forward_hook(
                    lambda m, i, o: pool_output.append(o.detach().flatten(1))
                )
                _ = model(imgs[bird_mask])
                hook.remove()
                activations.append(pool_output[0])
                collected += bird_mask.sum().item()
                if collected >= n_samples:
                    break
        if not activations:
            return torch.zeros(0, 512)
        return torch.cat(activations, dim=0)[:n_samples]

    bird_acts = get_activations(test_subset, target_class=2, n_samples=40)
    car_acts  = get_activations(test_subset, target_class=1, n_samples=40)

    print(f"  Bird activations: {bird_acts.shape}")
    print(f"  Car activations:  {car_acts.shape}")

    if len(bird_acts) < 5 or len(car_acts) < 5:
        print("  Not enough samples in subset. Try with full test set.")
        return

    # Combine and train logistic probe
    X = torch.cat([bird_acts, car_acts], dim=0).numpy()
    y = np.array([1] * len(bird_acts) + [0] * len(car_acts))

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))

    print(f"  CAV probe accuracy: {acc:.1%}")
    print(f"  CAV vector shape:   {clf.coef_.shape}")
    print(f"  Separability: {'good (>70%)' if acc > 0.7 else 'low (<70%) — concept not linear here'}")

    # The CAV is clf.coef_[0]
    cav_vector = clf.coef_[0]
    print(f"  CAV L2 norm: {np.linalg.norm(cav_vector):.4f}")

    print()
    print("  Interpretation: A high CAV probe accuracy (>80%) means the model")
    print("  has learned to linearly separate bird from car in this layer's space.")
    print("  This is a prerequisite for TCAV to produce meaningful results.")


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2: Directional Derivative Computation
#
# Compute the TCAV directional derivative: how much does moving activations
# in the concept direction change the model's output?
# ─────────────────────────────────────────────────────────────────────────────

def exercise_2_directional_derivative():
    """Compute directional derivative in concept direction manually."""
    from sklearn.linear_model import LogisticRegression

    print("\nExercise 2: Directional Derivative Computation")
    print("-" * 50)

    # Use a simple 2D example for clear demonstration
    # Model: f(x) = x1^2 + x2^2 (output depends on both dimensions)
    def simple_model(x):
        return (x ** 2).sum(dim=1, keepdim=True)

    # Concept direction: diagonal [1/√2, 1/√2] (equal weight both dims)
    concept_direction = torch.tensor([1.0, 1.0]) / np.sqrt(2)

    # Test inputs
    test_points = torch.tensor([
        [1.0, 0.0],   # on x1 axis
        [0.0, 1.0],   # on x2 axis
        [1.0, 1.0],   # diagonal
        [-1.0, -1.0], # negative diagonal
    ], requires_grad=False)

    print(f"  Concept direction: {concept_direction.numpy()}")
    print()

    for i, x in enumerate(test_points):
        x_var = x.unsqueeze(0).clone().requires_grad_(True)
        output = simple_model(x_var)
        grad = torch.autograd.grad(output.sum(), x_var)[0]
        directional_deriv = (grad.squeeze() * concept_direction).sum().item()
        print(f"  Point {x.numpy()}: f={output.item():.2f}, "
              f"grad={grad.squeeze().numpy()}, "
              f"S_concept={directional_deriv:+.4f}")

    print()
    print("  TCAV score = fraction of points with S_concept > 0")
    pos_count = sum(1 for x in test_points
                    if (torch.autograd.grad(
                        simple_model(x.unsqueeze(0).requires_grad_(True)).sum(),
                        x.unsqueeze(0).requires_grad_(True)
                    )[0].squeeze() * concept_direction).sum().item() > 0)
    print(f"  TCAV score: {pos_count}/{len(test_points)} = {pos_count/len(test_points):.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 3: SimilarityInfluence vs. Pixel Similarity
#
# Compare class match rates for pixel-space vs. representation-space
# nearest neighbors.
# ─────────────────────────────────────────────────────────────────────────────

def exercise_3_similarity_comparison():
    """Compare pixel and representation similarity for nearest neighbor retrieval."""
    from scipy.spatial.distance import cdist

    print("\nExercise 3: Pixel vs. Representation Space Similarity")
    print("-" * 50)

    # Extract activations for a small subset
    def extract_feats(subset, n=100):
        imgs = torch.stack([subset[i][0] for i in range(min(n, len(subset)))])
        lbls = np.array([subset[i][1] for i in range(min(n, len(subset)))])
        feats = []
        with torch.no_grad():
            for i in range(0, len(imgs), 32):
                batch = imgs[i:i+32]
                pool_out = []
                hook = model.avgpool.register_forward_hook(
                    lambda m, inp, out: pool_out.append(out.detach().flatten(1))
                )
                _ = model(batch)
                hook.remove()
                feats.append(pool_out[0])
        return torch.cat(feats, dim=0).numpy(), lbls

    print("  Extracting representations (takes ~30s)...")
    reprs, labels = extract_feats(test_subset, n=100)

    # Split into query (first 20) and gallery (remaining 80)
    query_reprs = reprs[:20]
    gallery_reprs = reprs[20:]
    query_labels = labels[:20]
    gallery_labels = labels[20:]

    query_flat   = torch.stack([test_subset[i][0] for i in range(20)]).flatten(1).numpy()
    gallery_flat = torch.stack([test_subset[i][0] for i in range(20, 100)]).flatten(1).numpy()

    # Representation-space kNN (k=5)
    repr_dists = cdist(query_reprs, gallery_reprs, metric='euclidean')
    pixel_dists = cdist(query_flat, gallery_flat, metric='euclidean')

    k = 5
    repr_match = pixel_match = 0
    for i in range(len(query_labels)):
        repr_nn = gallery_labels[repr_dists[i].argsort()[:k]]
        pixel_nn = gallery_labels[pixel_dists[i].argsort()[:k]]
        repr_match  += (repr_nn == query_labels[i]).sum()
        pixel_match += (pixel_nn == query_labels[i]).sum()

    total = len(query_labels) * k
    print(f"  k-NN same-class match rate (k={k}):")
    print(f"    Pixel space:          {pixel_match}/{total} = {pixel_match/total:.1%}")
    print(f"    Representation space: {repr_match}/{total} = {repr_match/total:.1%}")
    print()
    improvement = repr_match / max(pixel_match, 1)
    print(f"  Representation space is {improvement:.1f}x better at finding same-class examples.")
    print("  This validates SimilarityInfluence's semantic relevance.")


# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 4: TCAV Score Interpretation
#
# Given TCAV scores, interpret what they mean and flag unreliable results.
# ─────────────────────────────────────────────────────────────────────────────

def exercise_4_tcav_interpretation():
    """Interpret a set of TCAV scores and identify statistically significant ones."""
    from scipy.stats import ttest_1samp

    print("\nExercise 4: TCAV Score Interpretation")
    print("-" * 50)

    # Simulated TCAV scores across multiple random runs
    # (as would be computed by Captum's TCAV with multiple experimental sets)
    tcav_data = {
        ('zebra',     'layer4', 'striped'):  [0.82, 0.85, 0.81, 0.84, 0.83],
        ('zebra',     'layer4', 'dotted'):   [0.52, 0.48, 0.51, 0.55, 0.49],
        ('zebra',     'layer4', 'random_0'): [0.53, 0.47, 0.50, 0.52, 0.48],
        ('sports_car','layer4', 'striped'):  [0.51, 0.53, 0.48, 0.52, 0.50],
        ('sports_car','layer4', 'dotted'):   [0.55, 0.58, 0.52, 0.56, 0.54],
        ('cat',       'layer4', 'striped'):  [0.62, 0.65, 0.60, 0.63, 0.61],
        ('cat',       'layer1', 'striped'):  [0.52, 0.50, 0.53, 0.51, 0.49],
    }

    print(f"  {'Class':<15} {'Layer':<8} {'Concept':<12} {'Mean':>6} {'Std':>6} {'p-value':>10} {'Significant'}")
    print("  " + "-" * 65)

    for (cls, layer, concept), scores in tcav_data.items():
        scores_arr = np.array(scores)
        mean = scores_arr.mean()
        std  = scores_arr.std()
        _, p = ttest_1samp(scores_arr, 0.5)
        sig = "*** YES" if p < 0.01 else ("* YES" if p < 0.05 else "NO")
        print(f"  {cls:<15} {layer:<8} {concept:<12} {mean:>6.3f} {std:>6.3f} {p:>10.4f} {sig}")

    print()
    print("  Key observations:")
    print("  1. Zebra + striped @ layer4: strongly significant (model uses stripes for zebra)")
    print("  2. Zebra + dotted @ layer4: not significant (dots not used for zebra)")
    print("  3. Cat + striped @ layer4: marginally significant (cat has stripes in some breeds)")
    print("  4. Cat + striped @ layer1: not significant (concept not yet encoded in early layers)")
    print("  5. Random concept: never significant (validates null hypothesis)")


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL EXERCISES
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("MODULE 06 — CONCEPT & EXAMPLE EXPLANATIONS: SELF-CHECK")
    print("=" * 60)

    exercise_1_manual_cav()
    exercise_2_directional_derivative()
    exercise_3_similarity_comparison()
    exercise_4_tcav_interpretation()

    print("\n" + "=" * 60)
    print("All exercises complete.")
    print("=" * 60)
