"""
Module 00: State Space Models & Kalman Filter
Self-Check Exercises (Ungraded)

Run these exercises to test your understanding. Each exercise has instant feedback.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov


# ============================================================================
# EXERCISE 1: Build State Space Model for AR(2)
# ============================================================================

def exercise_1():
    """
    Convert AR(2) model to state space form.

    Given: y(t) = 0.7*y(t-1) + 0.2*y(t-2) + ε(t), ε ~ N(0, 1)

    Task: Specify T, Z, R, Q, H matrices
    """
    print("="*70)
    print("EXERCISE 1: AR(2) to State Space")
    print("="*70)
    print("\nModel: y(t) = 0.7*y(t-1) + 0.2*y(t-2) + ε(t)")
    print("\nYour task: Fill in the matrices below")

    # YOUR SOLUTION HERE
    # Hint: State vector is α(t) = [y(t), y(t-1)]'

    T = np.array([
        [0.7, 0.2],
        [1.0, 0.0]
    ])

    Z = np.array([[1.0, 0.0]])

    R = np.array([[1.0], [0.0]])

    Q = np.array([[1.0]])

    H = np.array([[0.0]])

    # TEST YOUR SOLUTION
    print("\n" + "-"*70)
    print("CHECKING YOUR SOLUTION...")
    print("-"*70)

    # Check dimensions
    correct = True
    if T.shape != (2, 2):
        print("❌ T should be (2, 2)")
        correct = False
    else:
        print("✓ T dimensions correct")

    if Z.shape != (1, 2):
        print("❌ Z should be (1, 2)")
        correct = False
    else:
        print("✓ Z dimensions correct")

    if R.shape != (2, 1):
        print("❌ R should be (2, 1)")
        correct = False
    else:
        print("✓ R dimensions correct")

    # Check values
    if np.allclose(T[0, 0], 0.7) and np.allclose(T[0, 1], 0.2):
        print("✓ T transition coefficients correct")
    else:
        print("❌ Check T matrix values")
        correct = False

    if np.allclose(Z, [[1.0, 0.0]]):
        print("✓ Z observation matrix correct")
    else:
        print("❌ Z should select first state variable")
        correct = False

    # Simulate and check
    np.random.seed(42)
    n = 500
    states = np.zeros((n, 2))

    for t in range(1, n):
        eta = np.random.randn()
        states[t] = T @ states[t-1] + R.flatten() * eta

    y = (Z @ states.T).flatten()

    # Check autocorrelation
    from statsmodels.tsa.stattools import acf
    acf_y = acf(y, nlags=2)

    print(f"\nSimulation check:")
    print(f"  ACF(1): {acf_y[1]:.3f} (should be ≈ 0.7)")
    print(f"  ACF(2): {acf_y[2]:.3f} (should be ≈ 0.2)")

    if correct:
        print("\n" + "="*70)
        print("🎉 EXCELLENT! Your state space representation is correct!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("💡 TIP: Review the companion form in the guide")
        print("="*70)

    return T, Z, R, Q, H


# ============================================================================
# EXERCISE 2: Implement One Step of Kalman Filter
# ============================================================================

def exercise_2():
    """
    Implement one iteration of the Kalman filter by hand.
    """
    print("\n" + "="*70)
    print("EXERCISE 2: One Kalman Filter Step")
    print("="*70)

    # Given
    T = np.array([[1.0]])
    Z = np.array([[1.0]])
    R = np.array([[1.0]])
    Q = np.array([[0.25]])
    H = np.array([[1.0]])

    # Previous filtered state
    a_prev = np.array([2.0])
    P_prev = np.array([[0.5]])

    # New observation
    y = np.array([2.5])

    print("\nGiven:")
    print(f"  Previous filtered state: a(t-1|t-1) = {a_prev[0]:.2f}")
    print(f"  Previous covariance: P(t-1|t-1) = {P_prev[0,0]:.2f}")
    print(f"  New observation: y(t) = {y[0]:.2f}")
    print(f"  Model: Local level (T=1, Z=1, Q=0.25, H=1.0)")

    print("\nYour task: Compute a(t|t) and P(t|t)")

    # YOUR SOLUTION HERE
    # Step 1: Prediction
    a_pred = T @ a_prev
    P_pred = T @ P_prev @ T.T + R @ Q @ R.T

    # Step 2: Innovation
    v = y - Z @ a_pred
    F = Z @ P_pred @ Z.T + H

    # Step 3: Update
    K = P_pred @ Z.T / F  # Kalman gain
    a_filt = a_pred + K * v
    P_filt = P_pred - K @ F @ K.T

    # SOLUTION (for checking)
    a_true = 2.208
    P_true = 0.436

    print("\n" + "-"*70)
    print("YOUR SOLUTION:")
    print("-"*70)
    print(f"  a(t|t) = {a_filt[0]:.3f}")
    print(f"  P(t|t) = {P_filt[0,0]:.3f}")
    print(f"  Kalman gain K = {K[0,0]:.3f}")

    print("\n" + "-"*70)
    print("CHECKING...")
    print("-"*70)

    if np.abs(a_filt[0] - a_true) < 0.01:
        print("✓ Filtered state a(t|t) is correct!")
    else:
        print(f"❌ Expected a(t|t) ≈ {a_true:.3f}")

    if np.abs(P_filt[0,0] - P_true) < 0.01:
        print("✓ Filtered covariance P(t|t) is correct!")
    else:
        print(f"❌ Expected P(t|t) ≈ {P_true:.3f}")

    # Explanation
    print("\n" + "-"*70)
    print("INTUITION:")
    print("-"*70)
    print(f"  Prediction: a(t|t-1) = {a_pred[0]:.3f} (model says stay at 2.0)")
    print(f"  Observation: y(t) = {y[0]:.2f} (data says it's 2.5)")
    print(f"  Kalman gain: K = {K[0,0]:.3f} (weight on data)")
    print(f"  Update: a(t|t) = 2.0 + 0.42*(2.5-2.0) = {a_filt[0]:.3f}")
    print(f"  → Filter blends prediction and observation!")

    return a_filt, P_filt, K


# ============================================================================
# EXERCISE 3: Effect of Noise on Kalman Gain
# ============================================================================

def exercise_3():
    """
    Explore how Q and H affect steady-state Kalman gain.
    """
    print("\n" + "="*70)
    print("EXERCISE 3: Noise and Kalman Gain")
    print("="*70)

    print("\nExperiment: How does Kalman gain depend on Q and H?")

    # Test different noise levels
    scenarios = [
        ("Low Q, High H", 0.1, 2.0),
        ("Equal noise", 1.0, 1.0),
        ("High Q, Low H", 2.0, 0.1)
    ]

    results = []

    for name, q, h in scenarios:
        T = np.array([[1.0]])
        Q_mat = np.array([[q]])
        R = np.array([[1.0]])
        H_mat = np.array([[h]])

        # Steady-state P
        P_inf = solve_discrete_lyapunov(T, R @ Q_mat @ R.T)

        # Steady-state K
        Z = np.array([[1.0]])
        F_inf = Z @ P_inf @ Z.T + H_mat
        K_inf = P_inf @ Z.T / F_inf

        results.append((name, q, h, K_inf[0, 0]))
        print(f"\n{name}:")
        print(f"  Q = {q:.1f}, H = {h:.1f}")
        print(f"  Steady-state K = {K_inf[0,0]:.4f}")

    print("\n" + "-"*70)
    print("INTERPRETATION:")
    print("-"*70)

    for name, q, h, k in results:
        if k < 0.3:
            verdict = "Trust MODEL more (smooth filtering)"
        elif k > 0.7:
            verdict = "Trust DATA more (track observations)"
        else:
            verdict = "Balance model and data"
        print(f"  {name}: K={k:.3f} → {verdict}")

    print("\n" + "-"*70)
    print("KEY INSIGHT:")
    print("-"*70)
    print("  • Large Q/H ratio → Large K → Follow observations closely")
    print("  • Small Q/H ratio → Small K → Smooth heavily")
    print("  • Kalman gain is the signal-to-noise ratio!")

    # Visualization
    plt.figure(figsize=(10, 5))

    q_values = np.logspace(-2, 1, 50)
    h = 1.0
    k_values = []

    for q in q_values:
        Q_mat = np.array([[q]])
        P_inf = solve_discrete_lyapunov(T, R @ Q_mat @ R.T)
        F_inf = Z @ P_inf @ Z.T + H_mat
        K_inf = P_inf @ Z.T / F_inf
        k_values.append(K_inf[0, 0])

    plt.semilogx(q_values, k_values, linewidth=2)
    plt.xlabel('State Noise Q (log scale)')
    plt.ylabel('Steady-State Kalman Gain K')
    plt.title('How State Noise Affects Kalman Gain (H = 1.0 fixed)')
    plt.grid(True, alpha=0.3)
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='K=0.5 (equal weight)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return results


# ============================================================================
# EXERCISE 4: Check Innovation Properties
# ============================================================================

def exercise_4():
    """
    Verify that innovations are white noise for correctly specified model.
    """
    print("\n" + "="*70)
    print("EXERCISE 4: Innovation Diagnostics")
    print("="*70)

    print("\nSimulate data and check if innovations are white noise...")

    # Simulate true model
    np.random.seed(42)
    n = 500
    T = np.array([[1.0]])
    Z = np.array([[1.0]])
    R = np.array([[1.0]])
    Q = np.array([[0.5]])
    H = np.array([[1.0]])

    # Simulate
    states = np.zeros(n)
    obs = np.zeros(n)

    for t in range(1, n):
        states[t] = states[t-1] + np.random.randn() * np.sqrt(0.5)
        obs[t] = states[t] + np.random.randn()

    # Run filter with CORRECT model
    a_filt, v_correct, K = [], [], []
    a, P = 0.0, 10.0

    for t in range(n):
        # Predict
        a_pred = a
        P_pred = P + 0.5

        # Update
        v_t = obs[t] - a_pred
        F = P_pred + 1.0
        K_t = P_pred / F
        a = a_pred + K_t * v_t
        P = P_pred - K_t * F * K_t

        v_correct.append(v_t)
        a_filt.append(a)

    v_correct = np.array(v_correct)

    # Now run filter with WRONG model (Q too small)
    v_wrong = []
    a, P = 0.0, 10.0

    for t in range(n):
        a_pred = a
        P_pred = P + 0.1  # WRONG: Q=0.1 instead of 0.5

        v_t = obs[t] - a_pred
        F = P_pred + 1.0
        K_t = P_pred / F
        a = a_pred + K_t * v_t
        P = P_pred - K_t * F * K_t

        v_wrong.append(v_t)

    v_wrong = np.array(v_wrong)

    # Diagnostics
    from scipy.stats import jarque_bera
    from statsmodels.stats.diagnostic import acorr_ljungbox

    # Correct model
    jb_correct = jarque_bera(v_correct)
    lb_correct = acorr_ljungbox(v_correct, lags=10, return_df=False)

    # Wrong model
    jb_wrong = jarque_bera(v_wrong)
    lb_wrong = acorr_ljungbox(v_wrong, lags=10, return_df=False)

    print("\n" + "-"*70)
    print("CORRECT MODEL (Q = 0.5, true value):")
    print("-"*70)
    print(f"  Innovation mean: {v_correct.mean():.4f} (should be ≈ 0)")
    print(f"  Innovation std: {v_correct.std():.4f}")
    print(f"  Jarque-Bera p-value: {jb_correct[1]:.4f} (> 0.05 is good)")
    print(f"  Ljung-Box p-value: {lb_correct[1][-1]:.4f} (> 0.05 is good)")

    if jb_correct[1] > 0.05 and lb_correct[1][-1] > 0.05:
        print("  ✓ Innovations pass tests → Model is well-specified!")

    print("\n" + "-"*70)
    print("WRONG MODEL (Q = 0.1, misspecified):")
    print("-"*70)
    print(f"  Innovation mean: {v_wrong.mean():.4f}")
    print(f"  Innovation std: {v_wrong.std():.4f}")
    print(f"  Jarque-Bera p-value: {jb_wrong[1]:.4f}")
    print(f"  Ljung-Box p-value: {lb_wrong[1][-1]:.4f}")

    if lb_wrong[1][-1] < 0.05:
        print("  ❌ Innovations are serially correlated → Model is misspecified!")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Correct model innovations
    axes[0, 0].plot(v_correct, alpha=0.7)
    axes[0, 0].axhline(0, color='red', linestyle='--')
    axes[0, 0].set_title('Correct Model: Innovations')
    axes[0, 0].grid(True, alpha=0.3)

    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(v_correct, lags=20, ax=axes[0, 1])
    axes[0, 1].set_title('Correct Model: ACF')

    # Wrong model innovations
    axes[1, 0].plot(v_wrong, alpha=0.7)
    axes[1, 0].axhline(0, color='red', linestyle='--')
    axes[1, 0].set_title('Wrong Model: Innovations')
    axes[1, 0].grid(True, alpha=0.3)

    plot_acf(v_wrong, lags=20, ax=axes[1, 1])
    axes[1, 1].set_title('Wrong Model: ACF (Note Significant Lags!)')

    plt.tight_layout()
    plt.show()

    print("\n" + "="*70)
    print("KEY TAKEAWAY:")
    print("="*70)
    print("Innovation diagnostics reveal model misspecification!")
    print("Always check: (1) zero mean, (2) no autocorrelation, (3) normality")

    return v_correct, v_wrong


# ============================================================================
# EXERCISE 5: Missing Data Challenge
# ============================================================================

def exercise_5():
    """
    Compare different approaches to handling missing data.
    """
    print("\n" + "="*70)
    print("EXERCISE 5: Missing Data Challenge")
    print("="*70)

    print("\nCompare: Kalman filter vs simple imputation")

    # Generate data
    np.random.seed(123)
    n = 100
    states = np.cumsum(np.random.randn(n) * 0.5)
    obs = states + np.random.randn(n)

    # Create missing data (30%)
    missing = np.random.rand(n) < 0.3
    obs_missing = obs.copy()
    obs_missing[missing] = np.nan

    print(f"\nGenerated {n} observations with {missing.sum()} missing ({missing.mean()*100:.1f}%)")

    # Method 1: Forward fill
    obs_ffill = obs_missing.copy()
    last_valid = 0
    for t in range(n):
        if np.isnan(obs_ffill[t]):
            obs_ffill[t] = obs_ffill[t-1] if t > 0 else 0

    # Method 2: Linear interpolation
    from scipy.interpolate import interp1d
    valid_idx = ~np.isnan(obs_missing)
    interp_func = interp1d(np.where(valid_idx)[0], obs_missing[valid_idx],
                          kind='linear', fill_value='extrapolate')
    obs_interp = interp_func(np.arange(n))

    # Method 3: Kalman filter
    T = np.array([[1.0]])
    Z = np.array([[1.0]])
    R = np.array([[1.0]])
    Q = np.array([[0.25]])
    H = np.array([[1.0]])

    a_filt = []
    a, P = 0.0, 10.0

    for t in range(n):
        a_pred = a
        P_pred = P + 0.25

        if not np.isnan(obs_missing[t]):
            v = obs_missing[t] - a_pred
            F = P_pred + 1.0
            K = P_pred / F
            a = a_pred + K * v
            P = P_pred - K * F * K
        else:
            a = a_pred
            P = P_pred

        a_filt.append(a)

    a_filt = np.array(a_filt)

    # Compute errors
    mse_ffill = np.mean((obs_ffill - states)**2)
    mse_interp = np.mean((obs_interp - states)**2)
    mse_kalman = np.mean((a_filt - states)**2)

    print("\n" + "-"*70)
    print("RESULTS:")
    print("-"*70)
    print(f"  MSE (Forward Fill): {mse_ffill:.4f}")
    print(f"  MSE (Interpolation): {mse_interp:.4f}")
    print(f"  MSE (Kalman Filter): {mse_kalman:.4f}")

    best_method = min([("Forward Fill", mse_ffill),
                      ("Interpolation", mse_interp),
                      ("Kalman Filter", mse_kalman)],
                     key=lambda x: x[1])

    print(f"\n  🏆 Winner: {best_method[0]}")

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Data with gaps
    axes[0].plot(states, 'k-', linewidth=2, label='True State', alpha=0.7)
    axes[0].plot(obs_missing, 'o', color='gray', markersize=4, label='Observations')
    axes[0].legend()
    axes[0].set_title('Original Data with Missing Values')
    axes[0].grid(True, alpha=0.3)

    # Forward fill
    axes[1].plot(states, 'k-', linewidth=2, alpha=0.4)
    axes[1].plot(obs_ffill, 'b-', linewidth=2, label=f'Forward Fill (MSE={mse_ffill:.3f})')
    axes[1].legend()
    axes[1].set_title('Method 1: Forward Fill')
    axes[1].grid(True, alpha=0.3)

    # Interpolation
    axes[2].plot(states, 'k-', linewidth=2, alpha=0.4)
    axes[2].plot(obs_interp, 'g-', linewidth=2, label=f'Interpolation (MSE={mse_interp:.3f})')
    axes[2].legend()
    axes[2].set_title('Method 2: Linear Interpolation')
    axes[2].grid(True, alpha=0.3)

    # Kalman filter
    axes[3].plot(states, 'k-', linewidth=2, alpha=0.4)
    axes[3].plot(a_filt, 'r-', linewidth=2, label=f'Kalman Filter (MSE={mse_kalman:.3f})')
    axes[3].legend()
    axes[3].set_title('Method 3: Kalman Filter')
    axes[3].set_xlabel('Time')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "="*70)
    print("CONCLUSION:")
    print("="*70)
    print("Kalman filter uses the model dynamics to optimally interpolate!")
    print("Simple methods ignore the underlying process structure.")

    return mse_ffill, mse_interp, mse_kalman


# ============================================================================
# RUN ALL EXERCISES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🎓 "*35)
    print("MODULE 00: STATE SPACE MODELS & KALMAN FILTER")
    print("Self-Check Exercises")
    print("🎓 "*35)

    # Exercise 1
    T, Z, R, Q, H = exercise_1()

    # Exercise 2
    a_filt, P_filt, K = exercise_2()

    # Exercise 3
    results = exercise_3()

    # Exercise 4
    v_correct, v_wrong = exercise_4()

    # Exercise 5
    mse_ffill, mse_interp, mse_kalman = exercise_5()

    print("\n" + "="*70)
    print("🎉 ALL EXERCISES COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review any exercises where you had trouble")
    print("  2. Check the guides for deeper explanations")
    print("  3. Move on to Module 01: Dynamic Factor Model Theory")
    print("="*70 + "\n")
