"""
Module 01: Dynamic Factor Model Theory
Self-Check Exercises (Ungraded)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def exercise_1():
    """Compare static PCA vs dynamic factors."""
    print("="*70)
    print("EXERCISE 1: Static vs Dynamic Factor Extraction")
    print("="*70)

    # Generate dynamic factors
    T, N, r = 200, 30, 3
    factors = np.zeros((T, r))
    phi = np.array([0.9, 0.7, 0.5])

    for t in range(1, T):
        factors[t] = phi * factors[t-1] + np.random.randn(r) * 0.5

    # Generate observations
    Lambda = np.random.randn(N, r)
    X = factors @ Lambda.T + np.random.randn(T, N) * 0.3

    # PCA extraction
    pca = PCA(n_components=r)
    factors_pca = pca.fit_transform(X)

    # Compare
    corr = [np.abs(np.corrcoef(factors[:, i], factors_pca[:, i])[0, 1])
            for i in range(r)]

    print(f"\nCorrelations between true and PCA factors:")
    for i, c in enumerate(corr):
        print(f"  Factor {i+1}: {c:.3f}")

    if all(c > 0.8 for c in corr):
        print("\n✓ PCA recovers factors well (high persistence case)")
    else:
        print("\n⚠ PCA struggles (need dynamic model)")

    return factors, factors_pca

def exercise_2():
    """Build state space matrices for DFM."""
    print("\n" + "="*70)
    print("EXERCISE 2: DFM to State Space Conversion")
    print("="*70)

    N, r, p, q = 10, 2, 1, 1
    print(f"\nSpecification: N={N}, r={r}, p={p}, q={q}")

    # Your solution here
    Lambda_0 = np.random.randn(N, r)
    Lambda_1 = np.random.randn(N, r) * 0.3
    Phi_1 = np.diag([0.8, 0.6])

    s = max(p + 1, q)
    Z = np.zeros((N, r * s))
    Z[:, :r] = Lambda_0
    Z[:, r:2*r] = Lambda_1

    T = np.zeros((r * s, r * s))
    T[:r, :r] = Phi_1
    T[r:, :r] = np.eye(r)

    R = np.zeros((r * s, r))
    R[:r, :] = np.eye(r)

    # Check
    print(f"\nState dimension: {T.shape[0]} (should be {r*s})")
    print(f"Z shape: {Z.shape}")
    print(f"T shape: {T.shape}")

    # Verify stationarity
    eigenvalues = np.linalg.eigvals(T)
    max_eig = np.max(np.abs(eigenvalues))
    print(f"\nMax eigenvalue of T: {max_eig:.3f}")

    if max_eig < 1:
        print("✓ Stationary dynamics")
    else:
        print("✗ Non-stationary!")

    return Z, T, R

def exercise_3():
    """Identification restrictions."""
    print("\n" + "="*70)
    print("EXERCISE 3: Imposing Identification")
    print("="*70)

    # Unidentified loadings
    Lambda = np.random.randn(20, 3)
    print(f"\nOriginal loadings shape: {Lambda.shape}")

    # QR decomposition for triangular identification
    Q, R_tri = np.linalg.qr(Lambda)
    signs = np.sign(np.diag(R_tri))
    Lambda_id = Q * signs

    print("\nFirst 3×3 block (should be triangular):")
    print(Lambda_id[:3, :])

    # Verify identification
    is_lower_triangular = np.allclose(
        np.triu(Lambda_id[:3, :], k=1), 0
    )

    if is_lower_triangular:
        print("\n✓ Identification correctly imposed")
    else:
        print("\n✗ Not properly identified")

    return Lambda_id

if __name__ == "__main__":
    print("\n" + "🎓"*35)
    print("MODULE 01: DYNAMIC FACTOR MODEL THEORY")
    print("Self-Check Exercises")
    print("🎓"*35)

    factors, factors_pca = exercise_1()
    Z, T, R = exercise_2()
    Lambda_id = exercise_3()

    print("\n" + "="*70)
    print("✓ All exercises complete!")
    print("="*70)
