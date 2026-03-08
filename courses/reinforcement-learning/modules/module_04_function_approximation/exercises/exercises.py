"""
Module 04 - Function Approximation: Self-Check Exercises

Covers:
- Polynomial feature vector construction for continuous states
- Semi-gradient TD update for linear function approximation
- Tile coding for 2D continuous state spaces

Run with: python exercises.py
Dependencies: numpy
"""

import numpy as np


# ---------------------------------------------------------------------------
# Exercise 1: Polynomial Feature Vector
# ---------------------------------------------------------------------------

def polynomial_features(state: np.ndarray, degree: int) -> np.ndarray:
    """
    Construct a polynomial feature vector from a continuous state.

    Problem
    -------
    Given a state vector s in R^d and a polynomial degree n, return a
    feature vector phi(s) containing all monomials up to total degree n.

    For a 1D state s = [x], degree 2 gives: [1, x, x^2].
    For a 2D state s = [x, y], degree 2 gives:
        [1, x, y, x^2, xy, y^2]   (all monomials with sum of exponents <= 2)

    Implementation shortcut:
    - For degree 1: [1, s_0, s_1, ..., s_{d-1}]            (d+1 features)
    - For degree 2: extend with all pairwise products and squares
    - Use itertools.combinations_with_replacement to enumerate exponent tuples.

    Hints
    -----
    - from itertools import combinations_with_replacement
    - Each exponent tuple (i, j, ...) maps to product(s[k]**e for k,e in ...)
    - The constant 1 corresponds to the empty exponent tuple (degree 0).

    Parameters
    ----------
    state : np.ndarray, shape (d,)
        Continuous state vector.
    degree : int
        Maximum polynomial degree (>= 0).

    Returns
    -------
    np.ndarray, shape (K,)
        Feature vector, where K = C(d + degree, degree).

    Examples
    --------
    >>> polynomial_features(np.array([2.0]), degree=2)
    array([1., 2., 4.])   # [1, x, x^2]

    >>> polynomial_features(np.array([1.0, 2.0]), degree=1)
    array([1., 1., 2.])   # [1, x, y]
    """
    # SOLUTION
    from itertools import combinations_with_replacement
    d = len(state)
    features = []
    for deg in range(degree + 1):
        for exponents in combinations_with_replacement(range(d), deg):
            val = 1.0
            for idx in exponents:
                val *= state[idx]
            features.append(val)
    return np.array(features)


def test_exercise_1() -> None:
    # Degree 0: constant feature [1]
    phi = polynomial_features(np.array([3.0, 4.0]), degree=0)
    assert phi.shape == (1,), f"degree=0 must give 1 feature, got {phi.shape}."
    assert np.isclose(phi[0], 1.0), f"degree=0 feature must be 1.0, got {phi[0]}."

    # 1D, degree 2: [1, x, x^2]
    phi = polynomial_features(np.array([2.0]), degree=2)
    assert phi.shape == (3,), f"1D degree-2 must give 3 features, got {phi.shape}."
    assert np.allclose(phi, [1.0, 2.0, 4.0]), \
        f"Expected [1, 2, 4], got {phi}."

    # 2D, degree 1: [1, x, y]
    phi = polynomial_features(np.array([1.0, 2.0]), degree=1)
    assert phi.shape == (3,), f"2D degree-1 must give 3 features, got {phi.shape}."
    assert np.allclose(phi, [1.0, 1.0, 2.0]), \
        f"Expected [1, 1, 2], got {phi}."

    # 2D, degree 2: 6 features [1, x, y, x^2, xy, y^2]
    phi = polynomial_features(np.array([2.0, 3.0]), degree=2)
    assert phi.shape == (6,), f"2D degree-2 must give 6 features, got {phi.shape}."
    expected = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 9.0])
    assert np.allclose(phi, expected), \
        f"Expected {expected}, got {phi}."

    print("Exercise 1 PASSED")


# ---------------------------------------------------------------------------
# Exercise 2: Semi-Gradient TD Update for Linear FA
# ---------------------------------------------------------------------------

def semi_gradient_td_update(
    weights: np.ndarray,
    phi_s: np.ndarray,
    phi_s_next: np.ndarray,
    reward: float,
    alpha: float,
    gamma: float,
    done: bool = False,
) -> np.ndarray:
    """
    Apply a semi-gradient TD(0) weight update for linear function approximation.

    Problem
    -------
    With linear value approximation V(s; w) = w^T phi(s), the semi-gradient
    TD(0) update treats the TD target as fixed (stops gradient through it):

        delta = r + gamma * w^T phi(s') - w^T phi(s)   (if not done)
        delta = r - w^T phi(s)                           (if done)
        w <- w + alpha * delta * phi(s)

    "Semi-gradient" means we do NOT differentiate through the bootstrap target
    w^T phi(s'). This is the standard approach in linear TD.

    Hints
    -----
    - Compute V_s = weights @ phi_s and V_next = weights @ phi_s_next.
    - delta uses V_next only if not done.
    - The gradient of V(s;w) w.r.t. w is phi(s).

    Parameters
    ----------
    weights : np.ndarray, shape (K,)
        Current weight vector (modified in-place and returned).
    phi_s : np.ndarray, shape (K,)
        Feature vector for current state phi(s).
    phi_s_next : np.ndarray, shape (K,)
        Feature vector for next state phi(s').
    reward : float
        Observed reward.
    alpha : float
        Learning rate.
    gamma : float
        Discount factor.
    done : bool
        Whether the episode ended.

    Returns
    -------
    np.ndarray, shape (K,)
        Updated weight vector.
    """
    # SOLUTION
    V_s = weights @ phi_s
    if done:
        delta = reward - V_s
    else:
        V_next = weights @ phi_s_next
        delta = reward + gamma * V_next - V_s
    weights += alpha * delta * phi_s
    return weights


def test_exercise_2() -> None:
    # Simple 2-feature case: phi=[1, s], weights=[0, 0]
    # state s=1: phi=[1,1]; next state s=2: phi=[1,2]
    # r=1, gamma=0.9, alpha=0.1
    weights = np.zeros(2)
    phi_s = np.array([1.0, 1.0])
    phi_s_next = np.array([1.0, 2.0])

    weights = semi_gradient_td_update(weights, phi_s, phi_s_next,
                                       reward=1.0, alpha=0.1, gamma=0.9)
    # V_s = 0, V_next = 0; delta = 1+0-0 = 1
    # w += 0.1 * 1 * [1,1] = [0.1, 0.1]
    assert np.allclose(weights, [0.1, 0.1]), \
        f"Expected [0.1, 0.1], got {weights}."

    # Done=True: target = r only
    weights2 = np.zeros(2)
    phi_s2 = np.array([1.0, 0.0])
    semi_gradient_td_update(weights2, phi_s2, phi_s2,
                             reward=5.0, alpha=0.5, gamma=0.9, done=True)
    # V_s=0, delta=5; w += 0.5*5*[1,0] = [2.5, 0]
    assert np.allclose(weights2, [2.5, 0.0]), \
        f"Expected [2.5, 0.0], got {weights2}."

    # With non-zero weights
    weights3 = np.array([1.0, 0.5])
    phi_s3 = np.array([1.0, 2.0])   # V_s = 1+1 = 2
    phi_n3 = np.array([1.0, 3.0])   # V_next = 1+1.5 = 2.5
    semi_gradient_td_update(weights3, phi_s3, phi_n3,
                             reward=0.0, alpha=0.1, gamma=1.0)
    # delta = 0 + 1*2.5 - 2 = 0.5; w += 0.1*0.5*[1,2] = [0.05, 0.1]
    expected = np.array([1.05, 0.6])
    assert np.allclose(weights3, expected), \
        f"Expected {expected}, got {weights3}."

    print("Exercise 2 PASSED")


# ---------------------------------------------------------------------------
# Exercise 3: Tile Coding for 2D Continuous State
# ---------------------------------------------------------------------------

def tile_coding_features(
    state: np.ndarray,
    bounds: list[tuple[float, float]],
    n_tilings: int,
    n_tiles: int,
) -> np.ndarray:
    """
    Construct a binary feature vector using tile coding for a 2D state.

    Problem
    -------
    Tile coding creates n_tilings overlapping grids (tilings) over the
    state space. Each tiling divides each dimension into n_tiles bins.
    For each tiling t, the state falls into exactly one tile; the
    corresponding feature bit is set to 1.

    Total features: n_tilings * n_tiles^d (d=2 here).

    Each tiling is offset by (t / n_tilings) * tile_width in each dimension,
    creating n_tilings distinct but overlapping tilings.

    Steps:
      1. Compute tile_width[dim] = (high - low) / n_tiles for each dim.
      2. For tiling t, offset state by (t / n_tilings) * tile_width.
      3. Compute tile index: floor((state_offset - low) / tile_width).
      4. Clip tile index to [0, n_tiles - 1].
      5. Map (tiling, tile_i, tile_j) to a flat binary feature index.

    Hints
    -----
    - Use np.floor and np.clip for tile index computation.
    - Flat index = t * n_tiles^2 + tile_i * n_tiles + tile_j.
    - The returned vector has exactly n_tilings nonzero entries (one per tiling).

    Parameters
    ----------
    state : np.ndarray, shape (2,)
        2D continuous state.
    bounds : list of (float, float)
        [(low_0, high_0), (low_1, high_1)] — state space bounds per dimension.
    n_tilings : int
        Number of overlapping tilings.
    n_tiles : int
        Number of tiles per dimension per tiling.

    Returns
    -------
    np.ndarray, shape (n_tilings * n_tiles^2,), dtype float
        Binary feature vector with exactly n_tilings ones.

    Examples
    --------
    >>> state = np.array([0.5, 0.5])
    >>> bounds = [(0.0, 1.0), (0.0, 1.0)]
    >>> phi = tile_coding_features(state, bounds, n_tilings=1, n_tiles=4)
    >>> phi.sum()
    1.0   # exactly one tile active per tiling
    """
    # SOLUTION
    d = len(state)
    total_features = n_tilings * (n_tiles ** d)
    features = np.zeros(total_features)

    tile_widths = np.array([(hi - lo) / n_tiles for lo, hi in bounds])

    for t in range(n_tilings):
        offset = (t / n_tilings) * tile_widths
        indices = []
        for dim in range(d):
            lo = bounds[dim][0]
            s_offset = state[dim] - lo + offset[dim]
            tile_idx = int(np.clip(np.floor(s_offset / tile_widths[dim]),
                                   0, n_tiles - 1))
            indices.append(tile_idx)

        # Flat index for this tiling
        flat = t * (n_tiles ** d)
        for idx in indices:
            flat = flat * n_tiles + idx
        # Recompute cleanly
        flat_idx = t * (n_tiles ** d)
        stride = n_tiles ** (d - 1)
        for i, idx in enumerate(indices):
            flat_idx += idx * (n_tiles ** (d - 1 - i))
        features[flat_idx] = 1.0

    return features


def test_exercise_3() -> None:
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    # 1 tiling, 4x4 tiles: exactly 1 active feature
    phi = tile_coding_features(np.array([0.5, 0.5]), bounds,
                                n_tilings=1, n_tiles=4)
    assert phi.shape == (16,), f"Shape must be (16,), got {phi.shape}."
    assert phi.sum() == 1.0, f"Exactly 1 active tile, got {phi.sum()}."

    # 4 tilings: exactly 4 active features
    phi4 = tile_coding_features(np.array([0.5, 0.5]), bounds,
                                 n_tilings=4, n_tiles=4)
    assert phi4.shape == (64,), f"Shape must be (64,), got {phi4.shape}."
    assert phi4.sum() == 4.0, \
        f"Exactly 4 active tiles (one per tiling), got {phi4.sum()}."

    # States at corners of [0,1]^2 must land in different tiles
    phi_corner1 = tile_coding_features(np.array([0.0, 0.0]), bounds,
                                        n_tilings=1, n_tiles=4)
    phi_corner2 = tile_coding_features(np.array([0.99, 0.99]), bounds,
                                        n_tilings=1, n_tiles=4)
    assert not np.array_equal(phi_corner1, phi_corner2), \
        "Opposite corners must activate different tiles."

    # Binary: all entries are 0 or 1
    assert set(np.unique(phi4)).issubset({0.0, 1.0}), \
        "Feature vector must be binary (0s and 1s only)."

    print("Exercise 3 PASSED")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("Exercise 1: Polynomial Feature Vector", test_exercise_1),
        ("Exercise 2: Semi-Gradient TD Update (Linear FA)", test_exercise_2),
        ("Exercise 3: Tile Coding for 2D State", test_exercise_3),
    ]
    for name, fn in tests:
        try:
            fn()
        except AssertionError as exc:
            print(f"FAILED  {name}: {exc}")
        except Exception as exc:
            print(f"ERROR   {name}: {type(exc).__name__}: {exc}")
