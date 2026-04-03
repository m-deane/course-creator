"""
Module 04: Explainability for Neural Forecasting Models
Self-Check Exercises

These exercises verify that you can run the .explain() API, parse the
attribution tensors, and extract meaningful insights from the results.

Run with: python 01_explainability_exercises.py

All exercises use the same synthetic blog traffic dataset from the notebooks.
No network calls or external downloads required.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# Data and model setup (shared by all exercises)
# ============================================================================

def build_dataset_and_model():
    """
    Build the synthetic blog traffic dataset and train NHITS.

    Returns
    -------
    tuple
        (nf, futr_df, train) — trained NeuralForecast instance plus data frames
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    from neuralforecast.losses.pytorch import MSE, MAE

    rng = np.random.default_rng(42)
    n_days = 1000
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")
    day_of_week_multiplier = np.array([0.80, 0.90, 1.00, 1.05, 1.10, 0.95, 0.85])
    seasonal_component = day_of_week_multiplier[dates.dayofweek]
    trend = 400 + np.linspace(0, 200, n_days)

    published = np.zeros(n_days, dtype=np.float32)
    for i in range(0, n_days, 7):
        n_articles = rng.integers(2, 4)
        weekdays = [j for j in range(i, min(i + 5, n_days)) if dates[j].dayofweek < 5]
        if len(weekdays) >= n_articles:
            publish_days = rng.choice(weekdays, size=n_articles, replace=False)
            published[publish_days] = 1.0

    holidays = {
        "2020-01-01", "2020-07-04", "2020-11-26", "2020-12-25",
        "2021-01-01", "2021-07-04", "2021-11-25", "2021-12-25",
        "2022-01-01", "2022-07-04"
    }
    is_holiday = np.array(
        [1.0 if str(d.date()) in holidays else 0.0 for d in dates],
        dtype=np.float32
    )

    publishing_lift = np.zeros(n_days)
    publishing_lift += published * 600.0
    publishing_lift[1:] += published[:-1] * 200.0
    holiday_reduction = is_holiday * (-0.15 * trend)
    noise = rng.normal(0, 40, n_days)
    visitors = trend * seasonal_component + publishing_lift + holiday_reduction + noise
    visitors = np.clip(visitors, 50, None)

    df = pd.DataFrame({
        "unique_id": "blog",
        "ds": dates,
        "y": visitors.astype(np.float32),
        "published": published,
        "is_holiday": is_holiday
    })

    horizon = 28
    input_size = 56
    train = df.iloc[:-horizon].copy()
    test  = df.iloc[-horizon:].copy()
    futr_df = test[["unique_id", "ds", "published", "is_holiday"]].copy()

    models = [NHITS(
        h=horizon,
        input_size=input_size,
        futr_exog_list=["published", "is_holiday"],
        max_steps=300,
        loss=MSE(),
        valid_loss=MAE()
    )]
    nf = NeuralForecast(models=models, freq="D")
    nf.fit(df=train, val_size=horizon)

    return nf, futr_df, train


# ============================================================================
# Exercise 1: Run .explain() and verify output dict keys
# ============================================================================

def exercise_1(nf, futr_df):
    """
    Run .explain() with IntegratedGradients and verify the returned dict
    contains the three expected keys: 'insample', 'futr_exog',
    'baseline_predictions'.

    Task:
        1. Call nf.explain(futr_df=futr_df, explainer="IntegratedGradients")
        2. Extract the explanations dict
        3. Return the sorted list of dict keys

    Expected output: ['baseline_predictions', 'futr_exog', 'insample']
    """
    print("\n" + "=" * 70)
    print("EXERCISE 1: Run .explain() and verify dict keys")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    # Call nf.explain() and return sorted(explanations.keys())
    # -------------------------------------------------------------------------
    fcsts_df, explanations = nf.explain(futr_df=futr_df, explainer="IntegratedGradients")
    result_keys = sorted(explanations.keys())
    # -------------------------------------------------------------------------

    expected_keys = ["baseline_predictions", "futr_exog", "insample"]
    assert result_keys == expected_keys, (
        f"Expected keys {expected_keys}, got {result_keys}. "
        "Did you call nf.explain() and sort the keys?"
    )

    print(f"Keys returned: {result_keys}")
    print("PASS: .explain() returns the three expected keys.")
    return explanations


# ============================================================================
# Exercise 2: Extract insample attributions and verify shape
# ============================================================================

def exercise_2(explanations):
    """
    Extract the insample attribution tensor and verify its shape matches
    the expected dimensions for this model configuration:
      [batch=1, horizon=28, series=1, output=1, input_size=56, 2]

    Task:
        1. Extract explanations["insample"]
        2. Return the tensor shape as a tuple
        3. Verify it equals (1, 28, 1, 1, 56, 2)

    Remember:
        - Last dimension is always 2: [lag_value, attribution_score]
        - Index 1 of last dimension gives the attribution scores
    """
    print("\n" + "=" * 70)
    print("EXERCISE 2: Extract insample tensor and verify shape")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    # Extract insample tensor and return its shape as a tuple
    # -------------------------------------------------------------------------
    insample = explanations["insample"]
    result_shape = tuple(insample.shape)
    # -------------------------------------------------------------------------

    expected_shape = (1, 28, 1, 1, 56, 2)
    assert result_shape == expected_shape, (
        f"Expected shape {expected_shape}, got {result_shape}.\n"
        f"Shape interpretation: [batch, horizon, series, output, input_size, 2]\n"
        f"  batch=1 (single forecast), horizon=28, series=1, output=1 (point forecast),\n"
        f"  input_size=56 (lags), 2 = [value, attribution]"
    )

    print(f"insample shape: {result_shape}")
    print("Shape breakdown:")
    print(f"  batch      = {result_shape[0]}  (single forecast pass)")
    print(f"  horizon    = {result_shape[1]}  (28 forecast days)")
    print(f"  series     = {result_shape[2]}  (one blog series)")
    print(f"  output     = {result_shape[3]}  (point forecast)")
    print(f"  input_size = {result_shape[4]}  (56 past lags)")
    print(f"  2          = {result_shape[5]}  ([lag_value, attribution_score])")
    print("PASS: insample tensor has correct shape.")
    return insample


# ============================================================================
# Exercise 3: Find the most important lag position
# ============================================================================

def exercise_3(insample):
    """
    Find the lag position that receives the highest mean absolute attribution
    across all 28 forecast steps.

    Task:
        1. Extract attribution scores from insample (index 1 of last dimension)
        2. Compute mean absolute attribution per lag position, averaged across
           all 28 forecast steps
        3. Return the lag index with the highest mean absolute attribution

    Expected output: lag 0 (the most recent observation before forecast start)

    Hint:
        - insample[0, :, 0, 0, :, 1] gives you the attribution matrix
          with shape (28, 56)
        - np.abs(...).mean(axis=0) averages across forecast steps
        - np.argmax(...) finds the dominant lag
    """
    print("\n" + "=" * 70)
    print("EXERCISE 3: Find the most important lag position")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # YOUR CODE HERE
    # Return the integer lag index with highest mean absolute attribution
    # -------------------------------------------------------------------------
    attr_matrix = insample[0, :, 0, 0, :, 1]   # shape (28, 56)
    mean_abs_per_lag = np.abs(attr_matrix).mean(axis=0)  # shape (56,)
    dominant_lag = int(np.argmax(mean_abs_per_lag))
    # -------------------------------------------------------------------------

    assert isinstance(dominant_lag, int), (
        f"Expected an integer, got {type(dominant_lag)}. "
        "Use int(np.argmax(...)) to convert."
    )
    assert 0 <= dominant_lag < 56, (
        f"Lag index {dominant_lag} is out of range [0, 55]."
    )

    print(f"Mean absolute attribution per lag (first 10): {mean_abs_per_lag[:10].round(2)}")
    print(f"Dominant lag: {dominant_lag} (attribution = {mean_abs_per_lag[dominant_lag]:.2f})")

    # Provide context on the finding
    if dominant_lag == 0:
        print("Lag 0 is the most recent observation — this is the expected pattern for")
        print("a daily series where recent history is the strongest predictor.")
    elif dominant_lag <= 7:
        print(f"Lag {dominant_lag} dominates. This may indicate a within-week seasonality pattern.")
    else:
        print(f"Lag {dominant_lag} dominates. Investigate: distant lags dominating recent ones")
        print("can signal data leakage or a poorly fitted model.")

    # Print top 5 lags
    top5_idx = np.argsort(mean_abs_per_lag)[::-1][:5]
    print("\nTop 5 most important lags:")
    print(f"{'Rank':<6} {'Lag':<6} {'Mean |Attribution|'}")
    print("-" * 35)
    for rank, idx in enumerate(top5_idx, 1):
        print(f"{rank:<6} {idx:<6} {mean_abs_per_lag[idx]:.2f}")

    print("\nPASS: Most important lag identified.")
    return dominant_lag


# ============================================================================
# Main
# ============================================================================

def main():
    print("Module 04 Explainability Self-Check Exercises")
    print("=" * 70)
    print("Building dataset and training NHITS (takes ~30-60 seconds)...")
    print()

    nf, futr_df, train = build_dataset_and_model()
    print("Model ready.")

    # Run exercises
    explanations = exercise_1(nf, futr_df)
    insample     = exercise_2(explanations)
    dominant_lag = exercise_3(insample)

    print("\n" + "=" * 70)
    print("All exercises passed.")
    print("=" * 70)
    print()
    print("Summary of findings:")
    print(f"  .explain() returns 3 keys: insample, futr_exog, baseline_predictions")
    print(f"  insample shape: (1, 28, 1, 1, 56, 2)")
    print(f"  Most important lag: lag {dominant_lag}")
    print()
    print("Next step: open notebooks/02_attribution_visualization.ipynb to")
    print("compare all three methods and build the stakeholder narrative.")


if __name__ == "__main__":
    main()
