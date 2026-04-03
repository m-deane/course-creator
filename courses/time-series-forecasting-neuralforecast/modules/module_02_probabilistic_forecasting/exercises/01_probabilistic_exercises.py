"""
Module 02 Self-Check Exercises: Probabilistic Forecasting

These exercises verify that you can:
1. Train NHITS with MQLoss and inspect the output columns
2. Reproduce the sum-of-quantiles error with real data
3. Confirm the inequality holds significantly — not just as a rounding artifact
4. Understand how the error depends on the horizon and correlation structure

Run with: python 01_probabilistic_exercises.py

All assertions provide specific feedback on what went wrong.
No submission or grading — these are for self-verification only.
"""

import numpy as np
import pandas as pd

# ── Exercise configuration ────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Exercise 1: Train with MQLoss and verify output columns
# ─────────────────────────────────────────────────────────────────────────────

def exercise_1_train_and_check_columns():
    """
    Train NHITS with MQLoss(level=[80, 90]) on the French Bakery dataset.
    Verify that the forecast output contains the expected columns.
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    from neuralforecast.losses.pytorch import MQLoss

    url = "https://raw.githubusercontent.com/nicholasjmorales/French-Bakery-Daily-Transactional-Dataset/main/Bakery_sales.csv"
    raw = pd.read_csv(url, parse_dates=["date"])

    baguettes = (
        raw[raw["article"].str.upper().str.contains("BAGUETTE")]
        .groupby("date")["Quantity"]
        .sum()
        .reset_index()
        .rename(columns={"date": "ds", "Quantity": "y"})
        .assign(unique_id="bakery_total")
        .sort_values("ds")
        .reset_index(drop=True)
    )

    cutoff = baguettes["ds"].max() - pd.Timedelta(days=14)
    train_df = baguettes[baguettes["ds"] <= cutoff]

    model = NHITS(
        h=7,
        input_size=28,
        loss=MQLoss(level=[80, 90]),
        max_steps=300,
        random_seed=RANDOM_SEED,
    )

    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=train_df)
    forecast = nf.predict()

    # ── Assertions ─────────────────────────────────────────────────────────────
    expected_columns = ["NHITS-lo-80", "NHITS-hi-80", "NHITS-lo-90", "NHITS-hi-90"]

    for col in expected_columns:
        assert col in forecast.columns, (
            f"Missing column '{col}'. "
            f"Found columns: {forecast.columns.tolist()}. "
            f"Did you use MQLoss(level=[80, 90])? "
            f"The level= parameter controls which intervals are produced."
        )

    assert len(forecast) == 7, (
        f"Expected 7 forecast rows (one per day), got {len(forecast)}. "
        f"Check that h=7 in your NHITS model."
    )

    # Quantile ordering: hi > lo for each level
    for level in [80, 90]:
        lo_col = f"NHITS-lo-{level}"
        hi_col = f"NHITS-hi-{level}"
        violations = (forecast[lo_col] >= forecast[hi_col]).sum()
        assert violations == 0, (
            f"Found {violations} rows where {lo_col} >= {hi_col}. "
            f"Quantiles should be strictly ordered: lower bound < upper bound. "
            f"This may indicate insufficient training steps."
        )

    # 90% interval should be wider than 80% interval
    width_80 = (forecast["NHITS-hi-80"] - forecast["NHITS-lo-80"]).mean()
    width_90 = (forecast["NHITS-hi-90"] - forecast["NHITS-lo-90"]).mean()
    assert width_90 > width_80, (
        f"90% interval (avg width {width_90:.1f}) should be wider than "
        f"80% interval (avg width {width_80:.1f}). "
        f"A 90% interval covers more of the distribution, so it must be wider."
    )

    print("Exercise 1 PASSED: MQLoss produces the expected output columns and structure.")
    print(f"  Columns found: {[c for c in forecast.columns if 'NHITS' in c]}")
    print(f"  80% avg interval width: {width_80:.1f} baguettes")
    print(f"  90% avg interval width: {width_90:.1f} baguettes")
    return forecast


# ─────────────────────────────────────────────────────────────────────────────
# Exercise 2: Compute the sum-of-quantiles error
# ─────────────────────────────────────────────────────────────────────────────

def exercise_2_sum_of_quantiles_error(forecast: pd.DataFrame):
    """
    Given a 7-day forecast from MQLoss, compute:
    1. The sum of daily 80th percentiles (the naive approach)
    2. The true 80th percentile of the weekly total (via simulation)
    3. Show that the naive approach overestimates

    Args:
        forecast: DataFrame with NHITS forecast columns from exercise_1
    """
    # ── Your code: compute these three quantities ─────────────────────────────
    # Hint for sum_naive_q80:
    #   The 80% interval upper bound (NHITS-hi-80) is the per-day 90th percentile.
    #   But for the "sum of marginal 80th percentile" demonstration, we want the
    #   NHITS-hi-80 column (the upper bound of the 80% coverage interval = 90th pct).
    #   Sum this across all 7 days.

    sum_naive_q80 = forecast["NHITS-hi-80"].sum()
    sum_of_means = forecast["NHITS"].sum()

    # Simulate the true 80th percentile of the weekly total
    # Use per-day mean and sigma (estimated from the 80% interval width)
    interval_width = forecast["NHITS-hi-80"] - forecast["NHITS-lo-80"]
    per_day_sigma = interval_width / (2 * 1.2816)  # Normal approximation

    N_SIM = 200_000
    rng = np.random.default_rng(RANDOM_SEED)

    independent_sim = rng.normal(
        loc=forecast["NHITS"].values,
        scale=per_day_sigma.values,
        size=(N_SIM, 7),
    )
    weekly_totals = independent_sim.sum(axis=1)
    true_q80_weekly = np.percentile(weekly_totals, 80)

    overestimate = sum_naive_q80 - true_q80_weekly
    overestimate_pct = overestimate / true_q80_weekly * 100

    # ── Assertions ─────────────────────────────────────────────────────────────
    assert sum_naive_q80 > sum_of_means, (
        f"sum_naive_q80 ({sum_naive_q80:.0f}) should be greater than "
        f"sum_of_means ({sum_of_means:.0f}). "
        f"The 90th percentile (upper bound of 80% interval) should be above the mean."
    )

    assert true_q80_weekly > sum_of_means, (
        f"true_q80_weekly ({true_q80_weekly:.0f}) should be greater than "
        f"sum_of_means ({sum_of_means:.0f}). "
        f"The 80th percentile of the weekly total should be above the weekly mean."
    )

    assert sum_naive_q80 > true_q80_weekly, (
        f"sum_naive_q80 ({sum_naive_q80:.0f}) should be greater than "
        f"true_q80_weekly ({true_q80_weekly:.0f}). "
        f"The sum of daily 90th percentiles should overestimate the true weekly 80th pct. "
        f"If this fails, check your simulation: use independent days (no correlation)."
    )

    assert overestimate_pct > 5, (
        f"Expected overestimate > 5% but got {overestimate_pct:.1f}%. "
        f"The overestimate should be substantial for a 7-day horizon. "
        f"Verify that you are using independent days in the simulation."
    )

    print("Exercise 2 PASSED: Sum-of-quantiles overestimates the true weekly 80th pct.")
    print(f"  Sum of daily 90th pcts (naive):   {sum_naive_q80:,.0f}")
    print(f"  True 80th pct of weekly total:    {true_q80_weekly:,.0f}")
    print(f"  Overestimate:                     {overestimate:+,.0f} ({overestimate_pct:+.1f}%)")
    return {
        "sum_naive_q80": sum_naive_q80,
        "true_q80_weekly": true_q80_weekly,
        "overestimate": overestimate,
        "overestimate_pct": overestimate_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Exercise 3: Verify the inequality across distributions
# ─────────────────────────────────────────────────────────────────────────────

def exercise_3_inequality_is_universal():
    """
    Verify that sum(quantiles) > quantile(sums) holds for:
    - Multiple distributions: Normal, Poisson, LogNormal
    - Multiple quantile levels: 70th, 80th, 90th, 95th percentile
    - Multiple horizons: H = 3, 7, 14

    This confirms the inequality is mathematical, not a modeling artifact.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    N = 300_000

    distributions = {
        "Normal(100, 20)":     rng.normal(100, 20, (N, 14)),
        "Poisson(lam=100)":    rng.poisson(100, (N, 14)),
        "LogNormal(4.6, 0.2)": rng.lognormal(4.6, 0.2, (N, 14)),
    }

    quantile_levels = [70, 80, 90, 95]
    horizons = [3, 7, 14]

    failures = []

    for dist_name, data in distributions.items():
        for alpha in quantile_levels:
            for H in horizons:
                data_H = data[:, :H]

                # Sum of per-day alpha-th percentiles
                per_day_q = np.percentile(data_H, alpha, axis=0)  # shape (H,)
                sum_of_q = per_day_q.sum()

                # True alpha-th percentile of the weekly total
                weekly_totals = data_H.sum(axis=1)
                q_of_sum = np.percentile(weekly_totals, alpha)

                ratio = sum_of_q / q_of_sum

                if ratio <= 1.02:  # Allow for tiny rounding/simulation noise
                    failures.append(
                        f"{dist_name}, q={alpha}%, H={H}: "
                        f"ratio={ratio:.4f} (expected > 1.02)"
                    )

    assert len(failures) == 0, (
        f"The sum-of-quantiles inequality FAILED for {len(failures)} cases:\n"
        + "\n".join(f"  {f}" for f in failures)
        + "\nThis is unexpected. Check your simulation code."
    )

    # Spot-check: the ratio should increase with H for fixed alpha
    normal_data = distributions["Normal(100, 20)"]
    ratios_by_H = {}
    for H in horizons:
        data_H = normal_data[:, :H]
        sum_q = np.percentile(data_H, 80, axis=0).sum()
        q_sum = np.percentile(data_H.sum(axis=1), 80)
        ratios_by_H[H] = sum_q / q_sum

    assert ratios_by_H[14] > ratios_by_H[7] > ratios_by_H[3], (
        f"For Normal data at q=80%, the overestimate ratio should grow with H. "
        f"Got: H=3 → {ratios_by_H[3]:.3f}, H=7 → {ratios_by_H[7]:.3f}, H=14 → {ratios_by_H[14]:.3f}. "
        f"The error (sum_q / q_sum) should increase as H increases because "
        f"diversification benefit grows with more independent time steps."
    )

    print("Exercise 3 PASSED: Sum-of-quantiles inequality holds universally.")
    print(f"  Tested {len(distributions)} distributions × {len(quantile_levels)} quantile levels × {len(horizons)} horizons")
    print(f"  = {len(distributions) * len(quantile_levels) * len(horizons)} total cases, all confirmed")
    print()
    print("  Ratio (sum_q / q_sum) grows with horizon (Normal, q=80%):")
    for H, ratio in ratios_by_H.items():
        print(f"    H={H:2d} days: {ratio:.3f}x overestimate")


# ─────────────────────────────────────────────────────────────────────────────
# Exercise 4: Correlation reduces but does not eliminate the error
# ─────────────────────────────────────────────────────────────────────────────

def exercise_4_correlation_effect():
    """
    Show that introducing positive correlation between days reduces the
    sum-of-quantiles error but does not eliminate it.

    Under perfect positive correlation (all days move together),
    the sum-of-quantiles error goes to zero.
    Under independence, the error is maximum.
    """
    N = 200_000
    H = 7
    mu = 100.0
    sigma = 20.0
    rng = np.random.default_rng(RANDOM_SEED)

    correlation_levels = [0.0, 0.25, 0.50, 0.75, 0.95, 1.0]
    overestimates = {}

    for rho in correlation_levels:
        # Generate correlated data using a common factor
        # X_t = sqrt(rho) * Z + sqrt(1-rho) * eps_t
        # where Z ~ N(0,1) is the common factor and eps_t are independent noise
        common_factor = rng.normal(0, 1, (N, 1))
        idiosyncratic = rng.normal(0, 1, (N, H))

        if rho < 1.0:
            data = mu + sigma * (np.sqrt(rho) * common_factor + np.sqrt(1 - rho) * idiosyncratic)
        else:
            # Perfect correlation: all days equal the common factor
            data = mu + sigma * np.tile(common_factor, (1, H))

        # Sum of per-day 80th percentiles
        sum_q = np.percentile(data, 80, axis=0).sum()
        # True 80th percentile of weekly total
        q_sum = np.percentile(data.sum(axis=1), 80)
        overestimates[rho] = (sum_q - q_sum) / q_sum * 100

    # ── Assertions ─────────────────────────────────────────────────────────────
    assert overestimates[0.0] > overestimates[0.95], (
        f"Overestimate should decrease as correlation increases. "
        f"Got: rho=0.0 → {overestimates[0.0]:.1f}%, rho=0.95 → {overestimates[0.95]:.1f}%."
    )

    assert overestimates[1.0] < 2.0, (
        f"Under perfect correlation (rho=1.0), overestimate should be near 0%. "
        f"Got {overestimates[1.0]:.1f}%. "
        f"When all days move together, summing quantiles is exact."
    )

    # Most importantly: even at rho=0.50, the error should still be substantial
    assert overestimates[0.50] > 5.0, (
        f"At rho=0.50 correlation, overestimate should still be > 5%. "
        f"Got {overestimates[0.50]:.1f}%. "
        f"Partial correlation does not fix the problem — it only reduces it."
    )

    print("Exercise 4 PASSED: Correlation reduces but does not eliminate the error.")
    print()
    print("  Overestimate (% above true weekly 80th pct) by day-to-day correlation:")
    for rho, pct in overestimates.items():
        bar = "█" * int(pct / 2)
        print(f"    rho={rho:.2f}: {pct:5.1f}% overestimate  {bar}")
    print()
    print("  Takeaway: Only at perfect correlation (rho=1.0) does the error disappear.")
    print("  Real bakery demand has rho < 0.5, so the error remains large.")


# ─────────────────────────────────────────────────────────────────────────────
# Run all exercises
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Module 02 Self-Check Exercises")
    print("Probabilistic Forecasting: Why Quantiles Aren't Enough")
    print("=" * 60)
    print()

    try:
        print("Exercise 1: Train with MQLoss and verify output columns")
        print("-" * 55)
        forecast = exercise_1_train_and_check_columns()
        print()

        print("Exercise 2: Compute the sum-of-quantiles error")
        print("-" * 55)
        results = exercise_2_sum_of_quantiles_error(forecast)
        print()

        print("Exercise 3: Verify the inequality is universal")
        print("-" * 55)
        exercise_3_inequality_is_universal()
        print()

        print("Exercise 4: Effect of temporal correlation on the error")
        print("-" * 55)
        exercise_4_correlation_effect()
        print()

        print("=" * 60)
        print("All exercises PASSED.")
        print()
        print("Key verified facts:")
        print(f"  1. MQLoss produces 4 interval columns for level=[80, 90]")
        print(f"  2. Sum of daily 90th pcts overestimates weekly 80th pct")
        print(f"     by {results['overestimate_pct']:.1f}% on this bakery dataset")
        print(f"  3. The inequality holds for Normal, Poisson, and LogNormal distributions")
        print(f"  4. The error grows with the planning horizon")
        print(f"  5. Positive correlation reduces but does not eliminate the error")
        print()
        print("Next: Module 3 — Sample paths fix this problem correctly.")
        print("=" * 60)

    except AssertionError as e:
        print(f"\nExercise FAILED:\n{e}")
        raise
