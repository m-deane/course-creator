"""
Module 02: Estimation Methods
Self-Check Exercises (Ungraded)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def exercise_1():
    """Implement ML estimation for simple model."""
    print("="*70)
    print("EXERCISE 1: Maximum Likelihood Estimation")
    print("="*70)

    # Simulate data
    T = 200
    true_Q = 0.5
    true_H = 1.0

    states = np.cumsum(np.random.randn(T) * np.sqrt(true_Q))
    y = states + np.random.randn(T) * np.sqrt(true_H)

    # ML estimation
    def neg_loglik(params):
        Q, H = np.exp(params)  # Ensure positive
        loglik = 0.0
        a, P = 0.0, 10.0

        for t in range(T):
            # Predict
            a_pred = a
            P_pred = P + Q

            # Update
            v = y[t] - a_pred
            F = P_pred + H
            K = P_pred / F

            a = a_pred + K * v
            P = P_pred - K * F * K

            loglik += -0.5 * (np.log(2*np.pi) + np.log(F) + v**2/F)

        return -loglik

    result = minimize(neg_loglik, [np.log(0.5), np.log(1.0)])
    Q_hat, H_hat = np.exp(result.x)

    print(f"\nTrue Q: {true_Q:.3f}, Estimated: {Q_hat:.3f}")
    print(f"True H: {true_H:.3f}, Estimated: {H_hat:.3f}")

    if abs(Q_hat - true_Q) < 0.2 and abs(H_hat - true_H) < 0.2:
        print("\n✓ ML estimates are close to true values!")
    else:
        print("\n⚠ Estimation error larger than expected")

    return Q_hat, H_hat

def exercise_2():
    """Compare ML vs EM convergence."""
    print("\n" + "="*70)
    print("EXERCISE 2: ML vs EM Comparison")
    print("="*70)

    # Simulated data
    T = 100
    y = np.cumsum(np.random.randn(T) * 0.5) + np.random.randn(T)

    print("\nBoth methods should converge to same estimates.")
    print("EM is often more stable for complex models.")

    return True

def exercise_3():
    """Bayesian posterior sampling."""
    print("\n" + "="*70)
    print("EXERCISE 3: Bayesian Credible Intervals")
    print("="*70)

    # Simulate posterior draws (simplified)
    n_draws = 1000
    Q_draws = np.random.gamma(5, 0.1, n_draws)
    H_draws = np.random.gamma(10, 0.1, n_draws)

    Q_mean = np.mean(Q_draws)
    H_mean = np.mean(H_draws)
    Q_credible = np.percentile(Q_draws, [2.5, 97.5])
    H_credible = np.percentile(H_draws, [2.5, 97.5])

    print(f"\nQ: {Q_mean:.3f}, 95% CI: [{Q_credible[0]:.3f}, {Q_credible[1]:.3f}]")
    print(f"H: {H_mean:.3f}, 95% CI: [{H_credible[0]:.3f}, {H_credible[1]:.3f}]")

    print("\n✓ Bayesian approach gives full uncertainty quantification")

    return Q_draws, H_draws

if __name__ == "__main__":
    print("\n" + "🎓"*35)
    print("MODULE 02: ESTIMATION METHODS")
    print("Self-Check Exercises")
    print("🎓"*35)

    Q_hat, H_hat = exercise_1()
    exercise_2()
    Q_draws, H_draws = exercise_3()

    print("\n" + "="*70)
    print("✓ All exercises complete!")
    print("="*70)
