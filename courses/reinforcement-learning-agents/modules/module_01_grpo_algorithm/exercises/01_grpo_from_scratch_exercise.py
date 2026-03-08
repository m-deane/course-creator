"""
Exercise 01: GRPO From Scratch
==============================
Module 01 — Reinforcement Learning for AI Agents

Implement the core components of Group Relative Policy Optimization using
only NumPy. This exercise mirrors the math in Guide 02.

Work through each section in order. Each function has a docstring describing
exactly what to implement. Run the file to check your work — all assertions
must pass before you move to Module 02.

Prerequisites:
    pip install numpy

Run:
    python 01_grpo_from_scratch_exercise.py
"""

import numpy as np

# ---------------------------------------------------------------------------
# Part 1: Advantage Calculation
# ---------------------------------------------------------------------------


def compute_group_advantages(rewards: np.ndarray) -> np.ndarray:
    """
    Compute group-normalized advantages from a vector of reward scores.

    Given G rewards for G completions of the same prompt, normalize them
    so that the group has mean 0 and std 1 (approximately).

    Formula:
        A_i = (r_i - mean(r)) / std(r)

    Edge case: if all rewards are identical, std = 0. Return zero advantages
    (no gradient signal — the model is already consistent on this prompt).

    Parameters
    ----------
    rewards : np.ndarray
        Shape (G,). Reward scores for G completions of a single prompt.
        All completions must be for the same prompt.

    Returns
    -------
    np.ndarray
        Shape (G,). Normalized advantages. Sum is approximately 0.
        Mean is exactly 0 (when std > 0). Std is exactly 1 (when std > 0).

    Implementation notes:
    - Use np.mean() and np.std() (population std, ddof=0 is the default)
    - Check if std < 1e-8 before dividing to handle the zero-variance case
    - Do NOT use scipy or sklearn — only numpy
    """
    # YOUR IMPLEMENTATION HERE
    raise NotImplementedError("Implement compute_group_advantages")


def test_compute_group_advantages():
    """Verify advantage calculation against the Guide 01 worked example."""

    # Test 1: The worked example from Guide 01
    rewards = np.array([0.9, 0.7, 0.5, 0.3])
    advantages = compute_group_advantages(rewards)

    expected = np.array([1.3416, 0.4472, -0.4472, -1.3416])
    assert advantages.shape == (4,), f"Expected shape (4,), got {advantages.shape}"
    assert np.allclose(advantages, expected, atol=1e-3), (
        f"Advantages mismatch.\nExpected: {expected}\nGot: {advantages}"
    )

    # Test 2: Mean of advantages should be 0 (by construction)
    assert abs(np.mean(advantages)) < 1e-10, (
        f"Advantages should have mean 0, got mean={np.mean(advantages):.6f}"
    )

    # Test 3: Std of advantages should be 1 (by construction)
    assert abs(np.std(advantages) - 1.0) < 1e-10, (
        f"Advantages should have std 1, got std={np.std(advantages):.6f}"
    )

    # Test 4: Identical rewards → zero advantages (no gradient)
    uniform_rewards = np.array([0.7, 0.7, 0.7, 0.7])
    zero_advantages = compute_group_advantages(uniform_rewards)
    assert np.allclose(zero_advantages, 0.0), (
        f"Identical rewards should produce zero advantages, got {zero_advantages}"
    )

    # Test 5: Order is preserved (highest reward → highest advantage)
    rewards_ordered = np.array([0.1, 0.5, 0.8, 1.0])
    adv_ordered = compute_group_advantages(rewards_ordered)
    assert np.all(np.diff(adv_ordered) > 0), (
        f"Advantages should be monotonically increasing with rewards, got {adv_ordered}"
    )

    # Test 6: Scale invariance — multiplying rewards by a constant changes std but
    # normalization removes it; advantages should be identical.
    rewards_scaled = rewards * 10.0
    advantages_scaled = compute_group_advantages(rewards_scaled)
    assert np.allclose(advantages_scaled, expected, atol=1e-3), (
        "Advantages should be scale-invariant (multiplying rewards by 10 should "
        f"not change advantages).\nExpected: {expected}\nGot: {advantages_scaled}"
    )

    print("Part 1 PASSED: compute_group_advantages")


# ---------------------------------------------------------------------------
# Part 2: Probability Ratio
# ---------------------------------------------------------------------------


def compute_probability_ratio(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
) -> np.ndarray:
    """
    Compute the probability ratio rho_i = pi_theta(o_i|q) / pi_old(o_i|q).

    Do NOT compute exp(log_probs_new) / exp(log_probs_old) directly —
    this will overflow for large log-prob magnitudes. Instead, use the
    log-space equivalent:

        rho_i = exp(log_probs_new - log_probs_old)

    Parameters
    ----------
    log_probs_new : np.ndarray
        Shape (G,). Log-probabilities of each completion under the current
        (training) policy. These are the values gradients flow through.
    log_probs_old : np.ndarray
        Shape (G,). Log-probabilities under the old (rollout) policy.
        These are constants — no gradient flows through them.

    Returns
    -------
    np.ndarray
        Shape (G,). Probability ratios rho_i. Should equal 1.0 when
        log_probs_new == log_probs_old (policy hasn't changed).
    """
    # YOUR IMPLEMENTATION HERE
    raise NotImplementedError("Implement compute_probability_ratio")


def test_compute_probability_ratio():
    """Verify probability ratio computation."""

    # Test 1: No change → all ratios = 1.0
    log_probs = np.array([-2.1, -2.3, -2.5, -2.8])
    ratios = compute_probability_ratio(log_probs, log_probs)
    assert np.allclose(ratios, 1.0), (
        f"When new == old, all ratios should be 1.0. Got {ratios}"
    )

    # Test 2: New policy more likely → ratio > 1.0
    log_probs_old = np.array([-2.5])
    log_probs_new = np.array([-2.0])  # higher log-prob = more likely
    ratio = compute_probability_ratio(log_probs_new, log_probs_old)
    assert ratio[0] > 1.0, f"More likely completion should have ratio > 1, got {ratio[0]}"
    assert np.isclose(ratio[0], np.exp(-2.0 - (-2.5))), (
        f"Expected exp(0.5) = {np.exp(0.5):.4f}, got {ratio[0]:.4f}"
    )

    # Test 3: New policy less likely → ratio < 1.0
    log_probs_old = np.array([-2.0])
    log_probs_new = np.array([-2.5])  # lower log-prob = less likely
    ratio = compute_probability_ratio(log_probs_new, log_probs_old)
    assert ratio[0] < 1.0, f"Less likely completion should have ratio < 1, got {ratio[0]}"

    # Test 4: Numerical stability — large magnitudes should not overflow
    log_probs_large_old = np.array([-1000.0, -1000.5])
    log_probs_large_new = np.array([-999.0, -1001.0])
    ratios_large = compute_probability_ratio(log_probs_large_new, log_probs_large_old)
    assert np.isfinite(ratios_large).all(), (
        f"Ratios should be finite even for large log-probs. Got {ratios_large}"
    )

    print("Part 2 PASSED: compute_probability_ratio")


# ---------------------------------------------------------------------------
# Part 3: Clipped Surrogate Loss
# ---------------------------------------------------------------------------


def clipped_surrogate_loss(
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    advantages: np.ndarray,
    epsilon: float = 0.2,
) -> float:
    """
    Compute the GRPO clipped surrogate loss for one prompt's group.

    Formula:
        L = (1/G) * sum_i [ min(rho_i * A_i, clip(rho_i, 1-eps, 1+eps) * A_i) ]

    Parameters
    ----------
    log_probs_new : np.ndarray
        Shape (G,). Log-probs under current policy (differentiable in practice).
    log_probs_old : np.ndarray
        Shape (G,). Log-probs under rollout policy (constant).
    advantages : np.ndarray
        Shape (G,). Group-normalized advantages from compute_group_advantages().
    epsilon : float
        Clip range. Ratios are constrained to [1 - epsilon, 1 + epsilon].

    Returns
    -------
    float
        Scalar loss. We MAXIMIZE this value (gradient ascent).
        In practice: compute -loss and minimize it (standard gradient descent).

    Implementation steps:
    1. Compute probability ratios using compute_probability_ratio()
    2. Compute the unclipped term: ratio * advantage (element-wise)
    3. Compute the clipped ratio using np.clip()
    4. Compute the clipped term: clipped_ratio * advantage (element-wise)
    5. Take element-wise minimum of unclipped and clipped terms
    6. Return the mean over the group

    Important: np.minimum() is element-wise (what you want).
               np.min() reduces to a scalar (NOT what you want).
    """
    # YOUR IMPLEMENTATION HERE
    raise NotImplementedError("Implement clipped_surrogate_loss")


def test_clipped_surrogate_loss():
    """Verify the clipped surrogate loss behavior."""

    advantages = np.array([1.3416, 0.4472, -0.4472, -1.3416])
    log_probs_old = np.array([-2.1, -2.3, -2.5, -2.8])

    # Test 1: No policy change → ratios all 1.0 → loss = mean(1.0 * advantages)
    loss_no_change = clipped_surrogate_loss(log_probs_old, log_probs_old, advantages)
    expected_no_change = float(np.mean(advantages))  # all ratios = 1.0
    assert np.isclose(loss_no_change, expected_no_change, atol=1e-6), (
        f"No-change loss should equal mean(advantages)={expected_no_change:.4f}, "
        f"got {loss_no_change:.4f}"
    )

    # Test 2: Clip activates for large ratio with positive advantage
    # ratio = 2.0 >> 1.2 (for epsilon=0.2) → should be clipped to 1.2
    log_probs_new_high = np.array([-2.1 + np.log(2.0)])  # ratio = 2.0 for first completion
    log_probs_old_single = np.array([-2.1])
    adv_positive = np.array([1.0])
    loss_clipped = clipped_surrogate_loss(
        log_probs_new_high, log_probs_old_single, adv_positive, epsilon=0.2
    )
    # With ratio=2.0 and A=1.0: unclipped=2.0, clipped=1.2, min=1.2
    assert np.isclose(loss_clipped, 1.2, atol=1e-4), (
        f"Clipped loss for ratio=2.0, A=1.0, eps=0.2 should be 1.2, got {loss_clipped:.4f}"
    )

    # Test 3: Clip activates for small ratio with negative advantage
    # ratio = 0.5 << 0.8 (for epsilon=0.2) → should be clipped to 0.8
    log_probs_new_low = np.array([-2.1 + np.log(0.5)])  # ratio = 0.5 for first completion
    adv_negative = np.array([-1.0])
    loss_clipped_neg = clipped_surrogate_loss(
        log_probs_new_low, log_probs_old_single, adv_negative, epsilon=0.2
    )
    # With ratio=0.5 and A=-1.0: unclipped=0.5*(-1)=-0.5, clipped=0.8*(-1)=-0.8
    # min(-0.5, -0.8) = -0.8
    assert np.isclose(loss_clipped_neg, -0.8, atol=1e-4), (
        f"Clipped loss for ratio=0.5, A=-1.0, eps=0.2 should be -0.8, got {loss_clipped_neg:.4f}"
    )

    # Test 4: Policy moving in the right direction → positive loss
    # Better completions become more likely, worse become less likely
    log_probs_new = np.array([-2.0, -2.25, -2.55, -2.9])  # shifted toward good
    loss_good_update = clipped_surrogate_loss(log_probs_new, log_probs_old, advantages)
    assert loss_good_update > 0, (
        f"Loss should be positive when policy moves toward better completions, "
        f"got {loss_good_update:.4f}"
    )

    # Test 5: Return type is float (scalar), not array
    assert isinstance(loss_good_update, float), (
        f"clipped_surrogate_loss should return a float, got {type(loss_good_update)}"
    )

    print("Part 3 PASSED: clipped_surrogate_loss")


# ---------------------------------------------------------------------------
# Part 4: KL Divergence Approximation
# ---------------------------------------------------------------------------


def approximate_kl_divergence(
    log_probs_policy: np.ndarray,
    log_probs_reference: np.ndarray,
) -> float:
    """
    Approximate KL divergence between current policy and reference model.

    KL(pi_theta || pi_ref) ≈ mean(log pi_theta - log pi_ref)

    This is an unbiased estimator using samples from pi_theta.
    The true KL requires integrating over all sequences — this approximation
    averages over the available batch.

    Parameters
    ----------
    log_probs_policy : np.ndarray
        Shape (G,). Log-probs under the current training policy.
    log_probs_reference : np.ndarray
        Shape (G,). Log-probs under the frozen reference model.

    Returns
    -------
    float
        Estimated KL divergence. Should be:
        - 0.0 when policy == reference
        - Positive when policy has moved away from reference
        - Negative values indicate numerical issues (KL is non-negative in theory)
    """
    # YOUR IMPLEMENTATION HERE
    raise NotImplementedError("Implement approximate_kl_divergence")


def test_approximate_kl_divergence():
    """Verify KL divergence approximation."""

    # Test 1: KL of policy with itself = 0
    log_probs = np.array([-2.1, -2.3, -2.5, -2.8])
    kl_self = approximate_kl_divergence(log_probs, log_probs)
    assert np.isclose(kl_self, 0.0, atol=1e-10), (
        f"KL(pi || pi) should be 0, got {kl_self}"
    )

    # Test 2: Policy more concentrated than reference → positive KL
    log_probs_ref = np.array([-3.0, -3.0, -3.0, -3.0])     # uniform
    log_probs_pol = np.array([-1.0, -1.0, -5.0, -5.0])     # more concentrated
    # mean(pol - ref) = mean([-1-(-3), -1-(-3), -5-(-3), -5-(-3)]) = mean([2, 2, -2, -2]) = 0
    kl_val = approximate_kl_divergence(log_probs_pol, log_probs_ref)
    assert isinstance(kl_val, float), f"KL should return float, got {type(kl_val)}"

    # Test 3: Policy shifted uniformly higher than reference → positive KL
    log_probs_ref2 = np.array([-2.0, -2.0, -2.0])
    log_probs_pol2 = np.array([-1.5, -1.5, -1.5])  # all shifted by +0.5
    kl_shifted = approximate_kl_divergence(log_probs_pol2, log_probs_ref2)
    assert np.isclose(kl_shifted, 0.5, atol=1e-10), (
        f"Uniform +0.5 shift → KL should be 0.5, got {kl_shifted}"
    )

    print("Part 4 PASSED: approximate_kl_divergence")


# ---------------------------------------------------------------------------
# Part 5: Full GRPO Loss
# ---------------------------------------------------------------------------


def grpo_loss(
    rewards: np.ndarray,
    log_probs_new: np.ndarray,
    log_probs_old: np.ndarray,
    log_probs_ref: np.ndarray,
    epsilon: float = 0.2,
    beta: float = 0.04,
) -> dict:
    """
    Compute the complete GRPO loss for one prompt's group of completions.

    Full objective:
        L_GRPO = ClippedSurrogate(rho, A) - beta * KL(pi_theta || pi_ref)

    This function should:
    1. Call compute_group_advantages(rewards) to get advantages
    2. Call clipped_surrogate_loss(...) to get the surrogate term
    3. Call approximate_kl_divergence(...) to get the KL term
    4. Combine them: total = surrogate - beta * kl

    Parameters
    ----------
    rewards : np.ndarray
        Shape (G,). Reward scores for each completion.
    log_probs_new : np.ndarray
        Shape (G,). Log-probs under current (training) policy.
    log_probs_old : np.ndarray
        Shape (G,). Log-probs under rollout policy (constant).
    log_probs_ref : np.ndarray
        Shape (G,). Log-probs under frozen reference model (constant).
    epsilon : float
        Clip range for surrogate loss. Default 0.2.
    beta : float
        KL penalty coefficient. Default 0.04.

    Returns
    -------
    dict with keys:
        "loss"      : float — total GRPO loss (maximize this)
        "surrogate" : float — clipped surrogate term
        "kl"        : float — KL divergence estimate
        "kl_penalty": float — beta * kl
        "advantages": np.ndarray — group advantages
    """
    # YOUR IMPLEMENTATION HERE
    raise NotImplementedError("Implement grpo_loss")


def test_grpo_loss():
    """Verify the complete GRPO loss combines components correctly."""

    rewards = np.array([0.9, 0.7, 0.5, 0.3])
    log_probs_old = np.array([-2.1, -2.3, -2.5, -2.8])
    log_probs_ref = log_probs_old.copy()
    log_probs_new = np.array([-2.0, -2.25, -2.55, -2.9])

    result = grpo_loss(
        rewards=rewards,
        log_probs_new=log_probs_new,
        log_probs_old=log_probs_old,
        log_probs_ref=log_probs_ref,
        epsilon=0.2,
        beta=0.04,
    )

    # Test 1: All required keys present
    required_keys = {"loss", "surrogate", "kl", "kl_penalty", "advantages"}
    assert required_keys == set(result.keys()), (
        f"Missing keys. Expected {required_keys}, got {set(result.keys())}"
    )

    # Test 2: Advantages have correct shape
    assert result["advantages"].shape == (4,), (
        f"Advantages should have shape (4,), got {result['advantages'].shape}"
    )

    # Test 3: KL penalty = beta * kl
    assert np.isclose(result["kl_penalty"], 0.04 * result["kl"], atol=1e-10), (
        f"kl_penalty should be beta * kl = 0.04 * {result['kl']:.4f} = "
        f"{0.04 * result['kl']:.4f}, got {result['kl_penalty']:.4f}"
    )

    # Test 4: Total loss = surrogate - kl_penalty
    assert np.isclose(result["loss"], result["surrogate"] - result["kl_penalty"], atol=1e-10), (
        f"loss should be surrogate - kl_penalty = "
        f"{result['surrogate']:.4f} - {result['kl_penalty']:.4f} = "
        f"{result['surrogate'] - result['kl_penalty']:.4f}, got {result['loss']:.4f}"
    )

    # Test 5: When policy == old policy and policy == ref, loss = mean(advantages) = 0
    result_zero = grpo_loss(
        rewards=rewards,
        log_probs_new=log_probs_old,
        log_probs_old=log_probs_old,
        log_probs_ref=log_probs_old,
        epsilon=0.2,
        beta=0.04,
    )
    assert np.isclose(result_zero["kl"], 0.0, atol=1e-10), (
        f"KL should be 0 when policy == reference, got {result_zero['kl']}"
    )
    # surrogate = mean(1.0 * advantages) = mean(advantages) ≈ 0 (numerically)
    assert np.isclose(result_zero["surrogate"], 0.0, atol=1e-10), (
        f"Surrogate should be 0 when ratios=1 and advantages sum to 0, "
        f"got {result_zero['surrogate']}"
    )

    print("Part 5 PASSED: grpo_loss")


# ---------------------------------------------------------------------------
# Part 6: Mini GRPO Training Loop
# ---------------------------------------------------------------------------


def mini_grpo_loop(
    prompts_rewards: list[tuple[str, np.ndarray]],
    initial_log_probs: np.ndarray,
    learning_rate: float = 0.1,
    epsilon: float = 0.2,
    beta: float = 0.04,
    n_epochs: int = 3,
) -> dict:
    """
    Run a simplified GRPO training loop on synthetic data.

    In real training, log_probs would come from a neural network and you'd
    use autograd. Here we simulate the update with a simple additive step
    to demonstrate that the loss drives the policy in the right direction.

    The simulation:
    - We parameterize the "policy" as a vector of log-probs (one per completion)
    - We update: log_probs += learning_rate * gradient_direction
    - Gradient direction: increase log-probs for positive-advantage completions,
      decrease for negative-advantage completions
    - This is a simplified (not exact) policy gradient update for demonstration

    Parameters
    ----------
    prompts_rewards : list of (prompt_name, rewards_array)
        Each element represents one prompt with G reward scores.
    initial_log_probs : np.ndarray
        Shape (G,). Starting log-probabilities for all prompts (shared for simplicity).
    learning_rate : float
        Step size for log-prob updates.
    epsilon : float
        GRPO clip range.
    beta : float
        KL penalty coefficient.
    n_epochs : int
        Number of full passes through the prompt list.

    Returns
    -------
    dict with keys:
        "loss_history"      : list of float — total loss per step
        "surrogate_history" : list of float — surrogate term per step
        "kl_history"        : list of float — KL per step
        "final_log_probs"   : np.ndarray — log-probs after training
    """
    loss_history = []
    surrogate_history = []
    kl_history = []

    # Reference policy: frozen at initialization (before any RL training)
    log_probs_ref = initial_log_probs.copy()

    # Current policy parameters (these get updated)
    log_probs_current = initial_log_probs.copy()

    for epoch in range(n_epochs):
        for prompt_name, rewards in prompts_rewards:
            # Snapshot old policy before this update step
            log_probs_old = log_probs_current.copy()

            # Compute GRPO loss
            result = grpo_loss(
                rewards=rewards,
                log_probs_new=log_probs_current,
                log_probs_old=log_probs_old,
                log_probs_ref=log_probs_ref,
                epsilon=epsilon,
                beta=beta,
            )

            loss_history.append(result["loss"])
            surrogate_history.append(result["surrogate"])
            kl_history.append(result["kl"])

            # Simplified gradient step: move toward positive-advantage completions
            # In real training: call loss.backward() and optimizer.step()
            advantages = result["advantages"]
            log_probs_current = log_probs_current + learning_rate * advantages

    return {
        "loss_history": loss_history,
        "surrogate_history": surrogate_history,
        "kl_history": kl_history,
        "final_log_probs": log_probs_current,
    }


def test_mini_grpo_loop():
    """
    Verify that the mini GRPO loop converges in the right direction.

    After training, the completion with the highest reward should have
    the highest log-probability (most likely to be generated).
    """
    # Two prompts, each with 4 completions, clear reward ordering
    prompts_rewards = [
        ("prompt_A", np.array([0.9, 0.7, 0.4, 0.1])),
        ("prompt_B", np.array([0.8, 0.6, 0.3, 0.2])),
    ]

    # Start with equal log-probs (uniform policy)
    initial_log_probs = np.array([-2.0, -2.0, -2.0, -2.0])

    result = mini_grpo_loop(
        prompts_rewards=prompts_rewards,
        initial_log_probs=initial_log_probs,
        learning_rate=0.05,
        n_epochs=5,
    )

    final_log_probs = result["final_log_probs"]

    # Test 1: Result has expected keys
    required_keys = {"loss_history", "surrogate_history", "kl_history", "final_log_probs"}
    assert required_keys == set(result.keys()), (
        f"Missing keys. Expected {required_keys}, got {set(result.keys())}"
    )

    # Test 2: Loss history is not empty
    expected_steps = 2 * 5  # 2 prompts × 5 epochs
    assert len(result["loss_history"]) == expected_steps, (
        f"Expected {expected_steps} loss values, got {len(result['loss_history'])}"
    )

    # Test 3: Best completion (index 0, highest reward) should have highest log-prob
    assert final_log_probs[0] == np.max(final_log_probs), (
        f"Completion with highest reward should have highest log-prob after training. "
        f"Final log-probs: {final_log_probs}"
    )

    # Test 4: Worst completion (index 3, lowest reward) should have lowest log-prob
    assert final_log_probs[3] == np.min(final_log_probs), (
        f"Completion with lowest reward should have lowest log-prob after training. "
        f"Final log-probs: {final_log_probs}"
    )

    # Test 5: Log-probs are monotonically decreasing with reward rank
    assert np.all(np.diff(final_log_probs) < 0), (
        f"Log-probs should decrease with reward rank after training. "
        f"Final log-probs: {final_log_probs}"
    )

    print("Part 6 PASSED: mini_grpo_loop")


# ---------------------------------------------------------------------------
# Main: Run all tests
# ---------------------------------------------------------------------------


def run_all_tests():
    """Run all exercise tests in order. All must pass."""
    print("=" * 60)
    print("GRPO From Scratch — Exercise Checks")
    print("=" * 60)

    tests = [
        ("Part 1: Advantage Calculation", test_compute_group_advantages),
        ("Part 2: Probability Ratio", test_compute_probability_ratio),
        ("Part 3: Clipped Surrogate Loss", test_clipped_surrogate_loss),
        ("Part 4: KL Divergence", test_approximate_kl_divergence),
        ("Part 5: Full GRPO Loss", test_grpo_loss),
        ("Part 6: Mini Training Loop", test_mini_grpo_loop),
    ]

    passed = 0
    failed = []

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except NotImplementedError:
            print(f"  {name}: NOT IMPLEMENTED YET")
            failed.append(name)
        except AssertionError as e:
            print(f"  {name}: FAILED")
            print(f"    {e}")
            failed.append(name)
        except Exception as e:
            print(f"  {name}: ERROR — {type(e).__name__}: {e}")
            failed.append(name)

    print()
    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} passed")
    if failed:
        print(f"Failed: {failed}")
        print("Fix the above before moving to Module 02.")
    else:
        print("All tests passed. You understand GRPO.")
        print("Next: Module 02 — ART Framework")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
