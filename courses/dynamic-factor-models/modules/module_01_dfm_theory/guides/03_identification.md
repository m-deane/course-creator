# Identification in Dynamic Factor Models

## In Brief

Factor models are rotation-invariant: if (F, Λ) explains the data, so does (F·R, Λ·R⁻¹) for any invertible R. This creates infinitely many equivalent representations. Identification restrictions pin down a unique solution by imposing structure on Λ or normalizing F.

## Key Insight

**The rotation problem:** Factors are unobserved, so we can only identify them up to rotation. Without restrictions, estimation algorithms can converge to any rotation.

**The solution:** Impose r(r-1)/2 restrictions to uniquely identify r factors. Common choices:
1. Triangular loading matrix (lower triangular with unit diagonal)
2. Normalize factor variance and set some loadings to zero
3. Orthogonal factors + ordered variance

## Formal Definition

**Identification Problem:**
Given X(t) = Λ·F(t) + ε(t), for any invertible R ∈ ℝ^(r×r):
```
X(t) = Λ·F(t) + ε(t) = Λ·R·(R⁻¹·F(t)) + ε(t) = Λ*·F*(t) + ε(t)
```
where Λ* = Λ·R and F* = R⁻¹·F. Both representations are observationally equivalent.

**Required Restrictions:**
Need r² normalizations total:
- r restrictions for scale (factor variance or loading normalization)
- r(r-1)/2 restrictions for rotation (zero or sign restrictions)

**Common Schemes:**

1. **Triangular (Stock-Watson):**
   ```
   Λ = [1   0   0 ]
       [λ₂₁ 1   0 ]
       [λ₃₁ λ₃₂ 1 ]
       [... ... ...]
   ```

2. **PC normalization:**
   ```
   Λ'Λ = I (orthogonal loadings)
   Var(F) = Σ_F diagonal (ordered variance)
   ```

## Code Implementation

```python
import numpy as np

def impose_identification(Lambda, method='triangular'):
    """
    Impose identification restrictions on loading matrix.

    Parameters:
    Lambda: (N, r) loading matrix (unidentified)
    method: 'triangular', 'pc', or 'manual'

    Returns: Lambda_identified, rotation matrix R
    """
    N, r = Lambda.shape

    if method == 'triangular':
        # QR decomposition gives triangular loadings
        Q, R_mat = np.linalg.qr(Lambda)
        # Ensure positive diagonal
        signs = np.sign(np.diag(R_mat))
        Lambda_id = Q * signs
        rotation = R_mat * signs[:, np.newaxis]
        return Lambda_id, rotation

    elif method == 'pc':
        # PCA normalization: orthogonal loadings
        U, s, Vt = np.svd(Lambda, full_matrices=False)
        Lambda_id = U * np.sqrt(N)  # Scale so Λ'Λ/N = I
        rotation = np.diag(s) @ Vt / np.sqrt(N)
        return Lambda_id, rotation

    else:
        raise ValueError("Method must be 'triangular' or 'pc'")

# Example
Lambda_unidentified = np.random.randn(20, 3)
Lambda_id, R = impose_identification(Lambda_unidentified, method='triangular')

print("First 5 rows of identified loadings:")
print(Lambda_id[:5])
print("\nTop-left block should be triangular:")
print(Lambda_id[:3, :3])
```

## Common Pitfalls

### 1. Forgetting to Identify
**Problem:** Estimation converges to arbitrary rotation.
**Solution:** Always check that r² restrictions are imposed.

### 2. Over-Identification
**Problem:** Imposing too many restrictions (> r²).
**Solution:** Count restrictions carefully.

### 3. Interpretation After Rotation
**Problem:** Rotating factors changes economic interpretation.
**Solution:** Use economically meaningful restrictions (e.g., monetary/real factors).

## Practice Problems

1. Count restrictions for r=4 factors
2. Verify QR decomposition gives identified loadings
3. Compare PC vs triangular identification on simulated data

## Further Reading

- Stock & Watson (2002): Section 2.3 on identification
- Bai & Ng (2013): "Principal Components Estimation and Identification"
