"""
Module 2: Bayesian Bandits - Self-Check Exercises

These exercises help you practice implementing Thompson Sampling variants
and understanding posterior updating. All exercises use commodity trading
themes for context.

Run this file with: python exercises.py
"""

import numpy as np
from scipy.stats import beta, gamma, norm
import matplotlib.pyplot as plt


# ============================================================================
# Exercise 1: Poisson Thompson Sampling for Trade Arrival Rates
# ============================================================================

def exercise_1():
    """
    Implement Thompson Sampling for a Poisson bandit.

    Context: You have 3 trading signals. Each generates trade opportunities
    at different rates (Poisson process). You want to identify which signal
    produces the most opportunities.

    Arms: 3 signals with true arrival rates λ = [2.0, 3.5, 2.8] trades/day
    Prior: Gamma(1, 1) (weakly informative)

    Task: Implement Poisson Thompson Sampling for 200 days.
    """
    print("\n" + "="*70)
    print("EXERCISE 1: Poisson Thompson Sampling")
    print("="*70)

    # True arrival rates (trades per day)
    true_rates = np.array([2.0, 3.5, 2.8])
    signal_names = ['Momentum', 'Mean-Reversion', 'Carry']
    T = 200  # days

    # Initialize Gamma posteriors: Gamma(α, β)
    # For Poisson likelihood, conjugate prior is Gamma
    # After observing count x in time period t:
    #   α_new = α_old + x
    #   β_new = β_old + t

    # YOUR CODE HERE
    # Initialize alpha and beta for 3 arms
    alpha = np.ones(3)
    beta_param = np.ones(3)

    # Track selections
    selections = []

    for day in range(T):
        # Sample λ from Gamma posterior for each signal
        samples = np.random.gamma(alpha, 1/beta_param)

        # Select signal with highest sampled rate
        signal = np.argmax(samples)
        selections.append(signal)

        # Observe count (number of trades that day)
        count = np.random.poisson(true_rates[signal])

        # Update posterior
        # Gamma(α, β) + Poisson(λ) with count x in 1 day
        # → Gamma(α + x, β + 1)
        alpha[signal] += count
        beta_param[signal] += 1

    # Results
    print("\nAfter 200 days:")
    print(f"{'Signal':<20s} {'True Rate':>10s} {'Post Mean':>10s} {'Selections':>11s}")
    print("-" * 53)
    for i, name in enumerate(signal_names):
        posterior_mean = alpha[i] / beta_param[i]
        n_selections = selections.count(i)
        print(f"{name:<20s} {true_rates[i]:>10.2f} {posterior_mean:>10.2f} {n_selections:>11d}")

    # Self-check
    best_signal = np.argmax(alpha / beta_param)
    assert best_signal == 1, "Should identify Mean-Reversion (signal 1) as best"
    assert selections.count(1) > selections.count(0), "Best signal should be selected most"

    print("\n✓ Exercise 1 passed: Poisson Thompson Sampling correctly identifies best signal")

    return alpha, beta_param, selections


# ============================================================================
# Exercise 2: Compare Prior Strengths
# ============================================================================

def exercise_2():
    """
    Compare Thompson Sampling convergence with different prior strengths.

    Context: You're testing 4 commodity trading strategies. You want to see
    how prior strength affects learning speed.

    Task: Run Thompson Sampling with weak, moderate, and strong priors.
          Measure how many rounds until posterior means converge.
    """
    print("\n" + "="*70)
    print("EXERCISE 2: Prior Strength Comparison")
    print("="*70)

    # True win rates
    true_probs = np.array([0.45, 0.52, 0.48, 0.50])
    T = 1000

    # Three different prior strengths
    prior_configs = [
        (1, 1, "Weak: Beta(1,1)"),
        (5, 5, "Moderate: Beta(5,5)"),
        (20, 20, "Strong: Beta(20,20)")
    ]

    convergence_times = []

    for alpha_0, beta_0, label in prior_configs:
        # Initialize
        alpha = np.ones(4) * alpha_0
        beta_param = np.ones(4) * beta_0

        # Track posterior means
        posterior_means = np.zeros((T, 4))

        for t in range(T):
            # Sample and select
            samples = np.random.beta(alpha, beta_param)
            arm = np.argmax(samples)

            # Observe
            reward = np.random.binomial(1, true_probs[arm])

            # Update
            if reward == 1:
                alpha[arm] += 1
            else:
                beta_param[arm] += 1

            # Track
            posterior_means[t] = alpha / (alpha + beta_param)

        # Find convergence time (when max deviation < 0.05 from true)
        deviations = np.abs(posterior_means - true_probs).max(axis=1)
        converged = np.where(deviations < 0.05)[0]

        if len(converged) > 0:
            convergence_time = converged[0]
        else:
            convergence_time = T

        convergence_times.append(convergence_time)

        print(f"\n{label}:")
        print(f"  Converged by round: {convergence_time}")
        print(f"  Final posterior means: {posterior_means[-1]}")

    # Self-check: Weak prior should converge fastest
    assert convergence_times[0] < convergence_times[2], \
        "Weak prior should converge faster than strong prior"

    print("\n✓ Exercise 2 passed: Weak priors converge faster than strong priors")

    return convergence_times


# ============================================================================
# Exercise 3: Batched Thompson Sampling
# ============================================================================

def exercise_3():
    """
    Implement batched Thompson Sampling where updates happen every N pulls.

    Context: In commodity trading, you often rebalance weekly (batch of 5 days)
    rather than daily. Implement Thompson Sampling that updates posteriors
    every 5 rounds instead of every round.

    Task: Compare batched (update every 5 rounds) vs sequential (update every round).
    """
    print("\n" + "="*70)
    print("EXERCISE 3: Batched Thompson Sampling")
    print("="*70)

    true_probs = np.array([0.45, 0.55, 0.50])
    T = 500
    batch_size = 5

    # Sequential (standard Thompson Sampling)
    alpha_seq = np.ones(3)
    beta_seq = np.ones(3)
    regret_seq = []
    optimal = true_probs.max()

    for t in range(T):
        samples = np.random.beta(alpha_seq, beta_seq)
        arm = np.argmax(samples)
        reward = np.random.binomial(1, true_probs[arm])

        if reward == 1:
            alpha_seq[arm] += 1
        else:
            beta_seq[arm] += 1

        regret_seq.append(optimal - true_probs[arm])

    # Batched (update every 5 rounds)
    alpha_batch = np.ones(3)
    beta_batch = np.ones(3)
    regret_batch = []

    # Buffer for batched updates
    batch_buffer = {i: {'successes': 0, 'failures': 0} for i in range(3)}

    for t in range(T):
        # Sample from CURRENT posterior (not updated yet)
        samples = np.random.beta(alpha_batch, beta_batch)
        arm = np.argmax(samples)
        reward = np.random.binomial(1, true_probs[arm])

        # Add to buffer
        if reward == 1:
            batch_buffer[arm]['successes'] += 1
        else:
            batch_buffer[arm]['failures'] += 1

        regret_batch.append(optimal - true_probs[arm])

        # Update every batch_size rounds
        if (t + 1) % batch_size == 0:
            for i in range(3):
                alpha_batch[i] += batch_buffer[i]['successes']
                beta_batch[i] += batch_buffer[i]['failures']
                batch_buffer[i] = {'successes': 0, 'failures': 0}

    # Compare cumulative regret
    cumulative_seq = np.cumsum(regret_seq)
    cumulative_batch = np.cumsum(regret_batch)

    print(f"\nAfter {T} rounds:")
    print(f"  Sequential cumulative regret: {cumulative_seq[-1]:.2f}")
    print(f"  Batched cumulative regret:    {cumulative_batch[-1]:.2f}")
    print(f"  Difference:                   {cumulative_batch[-1] - cumulative_seq[-1]:.2f}")

    # Batched should have slightly higher regret (delayed learning)
    # but should still be reasonable
    assert cumulative_batch[-1] < T * 0.2, "Batched regret should still be reasonable"

    print("\n✓ Exercise 3 passed: Batched Thompson Sampling implemented correctly")
    print("  (Batched has slightly higher regret due to delayed updates, as expected)")

    return cumulative_seq, cumulative_batch


# ============================================================================
# Exercise 4: Thompson Sampling with Discounting (Non-Stationary)
# ============================================================================

def exercise_4():
    """
    Implement Thompson Sampling with exponential discounting for non-stationary rewards.

    Context: Commodity markets are non-stationary. A strategy that worked last year
    may fail this year. Implement discounted Thompson Sampling that forgets old data.

    Task: Compare standard TS vs discounted TS on a non-stationary problem where
          true probabilities shift halfway through.
    """
    print("\n" + "="*70)
    print("EXERCISE 4: Discounted Thompson Sampling (Non-Stationary)")
    print("="*70)

    T = 1000
    shift_point = 500

    # True probabilities shift at t=500
    # Before: [0.6, 0.4, 0.5]
    # After:  [0.4, 0.6, 0.5]  (arms 0 and 1 swap)

    def get_true_prob(t):
        if t < shift_point:
            return np.array([0.6, 0.4, 0.5])
        else:
            return np.array([0.4, 0.6, 0.5])

    # Standard Thompson Sampling (no discounting)
    alpha_std = np.ones(3)
    beta_std = np.ones(3)
    regret_std = []

    # Discounted Thompson Sampling (γ = 0.99 per round)
    alpha_disc = np.ones(3)
    beta_disc = np.ones(3)
    regret_disc = []
    gamma = 0.99

    for t in range(T):
        true_probs = get_true_prob(t)
        optimal = true_probs.max()

        # Standard TS
        samples_std = np.random.beta(alpha_std, beta_std)
        arm_std = np.argmax(samples_std)
        reward_std = np.random.binomial(1, true_probs[arm_std])
        if reward_std == 1:
            alpha_std[arm_std] += 1
        else:
            beta_std[arm_std] += 1
        regret_std.append(optimal - true_probs[arm_std])

        # Discounted TS (discount before update)
        alpha_disc *= gamma
        beta_disc *= gamma

        samples_disc = np.random.beta(alpha_disc, beta_disc)
        arm_disc = np.argmax(samples_disc)
        reward_disc = np.random.binomial(1, true_probs[arm_disc])
        if reward_disc == 1:
            alpha_disc[arm_disc] += 1
        else:
            beta_disc[arm_disc] += 1
        regret_disc.append(optimal - true_probs[arm_disc])

    # Compare regret after the shift
    regret_after_shift_std = np.sum(regret_std[shift_point:])
    regret_after_shift_disc = np.sum(regret_disc[shift_point:])

    print(f"\nRegret after shift (rounds {shift_point}-{T}):")
    print(f"  Standard TS:   {regret_after_shift_std:.2f}")
    print(f"  Discounted TS: {regret_after_shift_disc:.2f}")
    print(f"  Improvement:   {regret_after_shift_std - regret_after_shift_disc:.2f}")

    # Discounted should have lower regret after shift (adapts faster)
    assert regret_after_shift_disc < regret_after_shift_std, \
        "Discounted TS should adapt faster to regime change"

    print("\n✓ Exercise 4 passed: Discounted TS adapts faster to non-stationarity")

    return regret_std, regret_disc


# ============================================================================
# Main: Run All Exercises
# ============================================================================

def main():
    """Run all self-check exercises."""
    print("\n" + "="*70)
    print("MODULE 2: BAYESIAN BANDITS - SELF-CHECK EXERCISES")
    print("="*70)

    np.random.seed(42)

    # Run exercises
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()

    print("\n" + "="*70)
    print("ALL EXERCISES COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Thompson Sampling extends to any conjugate prior-likelihood pair")
    print("2. Weak priors converge faster (less prior evidence to overcome)")
    print("3. Batched updates work fine (slight regret increase, but natural for real trading)")
    print("4. Discounting helps in non-stationary environments (commodity markets!)")
    print("\nYou're ready for Module 3: Contextual Bandits!")


if __name__ == "__main__":
    main()
