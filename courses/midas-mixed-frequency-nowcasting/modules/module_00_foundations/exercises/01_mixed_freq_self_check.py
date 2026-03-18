"""
Module 00: Mixed-Frequency Foundations — Self-Check Exercises

These exercises test your understanding of the mixed-frequency problem,
temporal aggregation, and the MIDAS data structure.

Run this file directly: python 01_mixed_freq_self_check.py
All checks print PASS or FAIL with explanations.
"""

import os
import sys
import numpy as np
import pandas as pd

# Locate resources directory (works when run from exercises/ or module root)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'resources')
if not os.path.exists(RESOURCES_DIR):
    RESOURCES_DIR = os.path.join(SCRIPT_DIR, '..', 'resources')


def load_csv(filename):
    """Load a CSV from resources directory."""
    path = os.path.join(RESOURCES_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV not found: {path}\n"
            f"Run from the module root or ensure resources/ directory exists."
        )
    return pd.read_csv(path, index_col='date', parse_dates=True).squeeze()


def check(condition, description, explanation=""):
    """Print a pass/fail check with optional explanation."""
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {description}")
    if not condition and explanation:
        print(f"         {explanation}")
    return condition


# ============================================================
# Exercise 1: Frequency Ratios
# ============================================================

def exercise_1_frequency_ratios():
    """
    Question: For each pair of frequencies, compute the frequency ratio m
    (how many high-frequency periods per low-frequency period).

    Fill in the correct values for each ratio.
    """
    print("\nExercise 1: Frequency Ratios")
    print("-" * 45)

    # YOUR ANSWERS: Fill in the correct integer frequency ratios
    monthly_to_quarterly = 3       # How many months per quarter?
    weekly_to_monthly = 4          # How many weeks per month (approximate)?
    daily_to_quarterly = 65        # How many trading days per quarter (approximate)?
    daily_to_monthly = 22          # How many trading days per month (approximate)?

    all_pass = True
    all_pass &= check(monthly_to_quarterly == 3,
                      "Monthly-to-quarterly ratio = 3",
                      "A quarter has exactly 3 months.")

    all_pass &= check(3 <= weekly_to_monthly <= 5,
                      f"Weekly-to-monthly ratio in [3,5]: got {weekly_to_monthly}",
                      "Most months have 4-5 weeks. 4 is the standard approximation.")

    all_pass &= check(60 <= daily_to_quarterly <= 70,
                      f"Daily-to-quarterly trading days in [60,70]: got {daily_to_quarterly}",
                      "~65 trading days per quarter (252 trading days / 4 quarters).")

    all_pass &= check(18 <= daily_to_monthly <= 25,
                      f"Daily-to-monthly trading days in [18,25]: got {daily_to_monthly}",
                      "~22 trading days per month (252 trading days / 12 months).")

    # Derived check: daily_to_quarterly ≈ 3 × daily_to_monthly
    ratio_consistency = abs(daily_to_quarterly - 3 * daily_to_monthly) <= 5
    all_pass &= check(ratio_consistency,
                      f"Ratios consistent: daily/quarterly ≈ 3 × daily/monthly",
                      f"{daily_to_quarterly} ≈ 3 × {daily_to_monthly} = {3*daily_to_monthly}")

    return all_pass


# ============================================================
# Exercise 2: Information Loss from Aggregation
# ============================================================

def exercise_2_information_loss():
    """
    Question: Load monthly IP growth from CSV. For each quarter in 2008,
    compute the within-quarter variance of the 3 monthly IP observations.
    Identify the quarter with the highest within-quarter variance.
    """
    print("\nExercise 2: Information Loss from Aggregation")
    print("-" * 45)

    all_pass = True

    try:
        ip = load_csv('industrial_production_monthly.csv')
    except FileNotFoundError as e:
        print(f"  [SKIP] Cannot load data: {e}")
        return False

    # Convert to Period index
    ip.index = pd.PeriodIndex(ip.index, freq='M')

    # YOUR CODE: For each quarter, compute within-quarter variance
    # (variance of the 3 monthly observations in that quarter)
    ip_df = pd.DataFrame({'ip': ip.values}, index=ip.index)
    ip_df['quarter'] = ip_df.index.to_timestamp().to_period('Q')

    # Compute within-quarter standard deviation for each quarter
    within_q_std = ip_df.groupby('quarter')['ip'].std()

    # The maximum within-quarter variance should be during the financial crisis or COVID
    max_q = within_q_std.idxmax()
    max_std = within_q_std.max()

    all_pass &= check(
        str(max_q.year) in ['2008', '2009', '2020'],
        f"Maximum within-quarter IP std in crisis year: {max_q} (std={max_std:.3f})",
        "The GFC (2008-2009) or COVID (2020) should produce the highest within-quarter variance."
    )

    # Average within-quarter std
    avg_std = within_q_std.mean()
    total_std = ip.std()

    all_pass &= check(
        avg_std > 0,
        f"Within-quarter std > 0 (avg = {avg_std:.3f}%)",
        "There must be some within-quarter variation for MIDAS to improve over aggregation."
    )

    all_pass &= check(
        avg_std < total_std,
        f"Within-quarter std ({avg_std:.3f}) < total std ({total_std:.3f})",
        "Within-quarter variance must be less than total variance by construction."
    )

    # Quantify information loss: what fraction of total variance is within-quarter?
    total_var = ip.var()
    mean_within_var = ip_df.groupby('quarter')['ip'].var().mean()
    within_fraction = mean_within_var / total_var

    print(f"  Within-quarter fraction of total variance: {within_fraction:.1%}")
    all_pass &= check(
        0.05 < within_fraction < 0.95,
        f"Within-quarter variance fraction in reasonable range (5%–95%): {within_fraction:.1%}",
        "Should be a meaningful fraction but not all of total variance."
    )

    return all_pass


# ============================================================
# Exercise 3: Aggregation Strategies
# ============================================================

def exercise_3_aggregation():
    """
    Question: For quarterly GDP growth on aggregated IP growth,
    which aggregation strategy (last-period vs. equal-weight) gives
    higher R-squared? Is the difference economically meaningful?
    """
    print("\nExercise 3: Aggregation Strategy Comparison")
    print("-" * 45)

    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    all_pass = True

    try:
        gdp = load_csv('gdp_quarterly.csv')
        ip = load_csv('industrial_production_monthly.csv')
    except FileNotFoundError as e:
        print(f"  [SKIP] Cannot load data: {e}")
        return False

    # Convert to Period index
    gdp.index = pd.PeriodIndex(gdp.index, freq='Q')
    ip.index = pd.PeriodIndex(ip.index, freq='M')

    # Aggregate IP to quarterly using two methods
    ip_last = ip.resample('Q').last()    # Last-period sampling
    ip_mean = ip.resample('Q').mean()    # Equal-weight average

    # Align with GDP
    common = gdp.index.intersection(ip_last.index)
    y = gdp.loc[common].values

    x_last = ip_last.loc[common].values.reshape(-1, 1)
    x_mean = ip_mean.loc[common].values.reshape(-1, 1)

    # Fit OLS regressions
    r2_last = r2_score(y, LinearRegression().fit(x_last, y).predict(x_last))
    r2_mean = r2_score(y, LinearRegression().fit(x_mean, y).predict(x_mean))

    print(f"  R² (last-period):   {r2_last:.4f}")
    print(f"  R² (equal-weight):  {r2_mean:.4f}")
    print(f"  Difference:         {r2_last - r2_mean:.4f}")

    # Last-period should have higher R² (last month most informative)
    all_pass &= check(
        r2_last > r2_mean,
        "Last-period R² > equal-weight R²",
        "The last month of the quarter is most recent and carries the most current-state information."
    )

    # Both should be positive
    all_pass &= check(
        r2_last > 0.05 and r2_mean > 0.05,
        f"Both R² values positive and meaningful (>5%)",
        "IP and GDP are positively correlated; both regressions should have some explanatory power."
    )

    # Difference should be non-trivial but not huge
    diff = r2_last - r2_mean
    all_pass &= check(
        diff >= 0,
        f"Last-period advantage: {diff:.4f} R² points",
        "Even a small R² improvement has forecasting implications."
    )

    return all_pass


# ============================================================
# Exercise 4: MIDAS Matrix Structure
# ============================================================

def exercise_4_midas_matrix():
    """
    Question: Build a small MIDAS data matrix by hand.
    Verify the lag indexing is correct.
    """
    print("\nExercise 4: MIDAS Data Matrix Construction")
    print("-" * 45)

    all_pass = True

    # Simulate 6 monthly IP observations (2 quarters)
    # Quarter 1: months 1, 2, 3 → values: 0.5, 0.3, 0.8
    # Quarter 2: months 4, 5, 6 → values: 0.2, 0.6, 0.4
    x_monthly = np.array([0.5, 0.3, 0.8, 0.2, 0.6, 0.4])

    # For Quarter 2 (t=1, 0-indexed), with 6 lags:
    # Lag 0 (most recent): Month 6 = 0.4
    # Lag 1: Month 5 = 0.6
    # Lag 2: Month 4 = 0.2
    # Lag 3: Month 3 = 0.8
    # Lag 4: Month 2 = 0.3
    # Lag 5: Month 1 = 0.5

    n_lags = 6
    freq_ratio = 3
    t = 1  # Quarter index (0-indexed), i.e., second quarter

    # YOUR CODE: Compute the MIDAS row for quarter t
    midas_row = np.array([
        x_monthly[(t + 1) * freq_ratio - 1 - j]
        for j in range(n_lags)
    ])

    expected_row = np.array([0.4, 0.6, 0.2, 0.8, 0.3, 0.5])

    all_pass &= check(
        np.allclose(midas_row, expected_row),
        f"MIDAS row for Quarter 2 is correct",
        f"Expected {expected_row}, got {midas_row}"
    )

    # Verify lag 0 is the most recent observation (Month 6)
    all_pass &= check(
        midas_row[0] == 0.4,
        f"Lag 0 (most recent) = 0.4 (Month 6)",
        "Lag 0 should be the last month of the current quarter."
    )

    # Verify lag 2 is the first month of current quarter (Month 4)
    all_pass &= check(
        midas_row[2] == 0.2,
        f"Lag 2 (first of current quarter) = 0.2 (Month 4)",
        "Lag 2 should be the first month of the current quarter."
    )

    # Verify lag 3 crosses into previous quarter (Month 3)
    all_pass &= check(
        midas_row[3] == 0.8,
        f"Lag 3 (last of previous quarter) = 0.8 (Month 3)",
        "Lag 3 should be the last month of the previous quarter."
    )

    return all_pass


# ============================================================
# Exercise 5: Publication Calendar
# ============================================================

def exercise_5_publication_calendar():
    """
    Question: Reason about the information available at different
    points within a quarter.
    """
    print("\nExercise 5: Real-Time Information Availability")
    print("-" * 45)

    all_pass = True

    # Scenario: It is Week 6 of 2024Q1 (mid-February 2024).
    # Answer True/False for each data availability question.

    # What data is available for nowcasting 2024Q1 GDP?

    # GDP for 2023Q4: Released late January 2024 — available? YES
    gdp_2023q4_available = True

    # GDP for 2024Q1: Not released until late April 2024 — available? NO
    gdp_2024q1_available = False

    # IP for January 2024: Released ~Feb 15 — borderline, but typically YES by week 6
    ip_jan_2024_available = True

    # IP for February 2024: Released ~Mar 15 — NOT yet available in mid-Feb
    ip_feb_2024_available = False

    # IP for March 2024: Released ~Apr 15 — NOT available in mid-Feb
    ip_mar_2024_available = False

    # S&P 500 returns through yesterday: ALWAYS available
    sp500_through_yesterday = True

    # Payrolls for January 2024: Released first Friday of February 2024 ≈ Feb 2 — YES
    payrolls_jan_2024 = True

    all_pass &= check(
        gdp_2023q4_available is True,
        "GDP 2023Q4 available in mid-Feb 2024",
        "Advance estimate released ~Jan 30."
    )
    all_pass &= check(
        gdp_2024q1_available is False,
        "GDP 2024Q1 NOT available in mid-Feb 2024",
        "Won't be released until late April 2024."
    )
    all_pass &= check(
        ip_jan_2024_available is True,
        "IP January 2024 available by week 6 of February",
        "Released around Feb 15 — borderline but typically yes."
    )
    all_pass &= check(
        ip_feb_2024_available is False,
        "IP February 2024 NOT available in mid-February",
        "Released ~March 15 — won't be out until next month."
    )
    all_pass &= check(
        ip_mar_2024_available is False,
        "IP March 2024 NOT available in mid-February",
        "March data won't be released until April."
    )
    all_pass &= check(
        sp500_through_yesterday is True,
        "S&P 500 returns through yesterday: always available",
        "Financial market data is available in real time."
    )

    # The ragged edge for 2024Q1 in mid-February:
    # Monthly data: 1 out of 3 months available (January only)
    # Daily data: ~30 trading days available out of ~65 total
    months_available_q1 = 1  # January only
    all_pass &= check(
        months_available_q1 == 1,
        f"In mid-February, only {months_available_q1}/3 monthly obs available for 2024Q1",
        "This is the ragged edge problem: data arrives month by month within the quarter."
    )

    return all_pass


# ============================================================
# Main runner
# ============================================================

def main():
    print("=" * 55)
    print("Module 00 Self-Check: Mixed-Frequency Foundations")
    print("=" * 55)

    results = []
    results.append(exercise_1_frequency_ratios())
    results.append(exercise_2_information_loss())
    results.append(exercise_3_aggregation())
    results.append(exercise_4_midas_matrix())
    results.append(exercise_5_publication_calendar())

    n_passed = sum(results)
    n_total = len(results)

    print("\n" + "=" * 55)
    print(f"Overall: {n_passed}/{n_total} exercises passed")

    if n_passed == n_total:
        print("All exercises passed. Ready to proceed to Module 01.")
    else:
        failed = [i + 1 for i, r in enumerate(results) if not r]
        print(f"Exercises {failed} need review.")
        print("Re-read the corresponding sections in the guide before continuing.")

    print("=" * 55)
    return 0 if n_passed == n_total else 1


if __name__ == '__main__':
    sys.exit(main())
