# Module 3 Quiz: Time Series Specific Techniques

**Course:** Genetic Algorithms for Feature Selection
**Module:** 3 - Time Series Specific Techniques
**Total Points:** 100
**Estimated Time:** 30-35 minutes
**Attempts Allowed:** 2

## Instructions

This quiz assesses your understanding of time series feature selection challenges including walk-forward validation, lag feature engineering, stationarity concerns, and temporal dependencies. Questions focus on practical applications to forecasting problems.

---

## Section 1: Walk-Forward Validation (25 points)

### Question 1 (10 points)

Consider the following time series dataset with 100 monthly observations:

```
Time: 2015-01 to 2023-04 (100 months)
Target: Monthly stock returns
```

You're using 5-fold TimeSeriesSplit for fitness evaluation. Which of the following BEST describes how the data is split?

A) 5 random 80/20 train/test splits
B) 5 contiguous blocks, each with training on past data and testing on immediate future
C) 5 expanding windows where training set grows and test set is always the most recent 20%
D) 5 sliding windows of equal size moving through the time series

**Answer:** ___________

**Detailed explanation of the split structure:**

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

---

### Question 2 (15 points)

Complete the following walk-forward validation implementation:

```python
import numpy as np
from sklearn.metrics import mean_squared_error

def walk_forward_validate(X, y, model, chromosome, n_splits=5):
    """
    Walk-forward validation with expanding window.

    Parameters:
    - X: Feature matrix (n_samples, n_features)
    - y: Target vector (n_samples,)
    - model: Estimator with fit/predict methods
    - chromosome: Binary array indicating selected features
    - n_splits: Number of forward validation splits

    Returns:
    - scores: List of RMSE scores for each split
    """
    selected_features = X[:, chromosome == 1]
    n_samples = len(y)
    split_size = n_samples // (n_splits + 1)

    scores = []

    for i in range(n_splits):
        # Define train/test indices for expanding window
        train_end = ________________  # Fill in
        test_start = ________________  # Fill in
        test_end = ________________  # Fill in

        # Split data
        X_train = selected_features[________________]  # Fill in
        X_test = selected_features[________________]  # Fill in
        y_train = y[________________]  # Fill in
        y_test = y[________________]  # Fill in

        # Train and evaluate
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = ________________  # Fill in: calculate RMSE

        scores.append(rmse)

    return scores
```

**Fill in the 9 blanks above.**

---

## Section 2: Lag Feature Engineering (25 points)

### Question 3 (12 points)

You're predicting next day's stock return using past returns as features:

```python
import pandas as pd

# Original time series
dates = pd.date_range('2020-01-01', periods=100, freq='D')
returns = pd.Series([...], index=dates)  # Daily returns

# Create lag features
for lag in [1, 2, 3, 5, 10]:
    df[f'return_lag_{lag}'] = returns.shift(lag)
```

**Part A (4 points):** After creating these lag features and dropping NaN values, how many valid samples remain for training?

**Answer:** ___________

**Explanation:** ___________________________________________________________

**Part B (8 points):** Why is it crucial that the GA's fitness function does NOT use future information when evaluating these lag features? Provide a specific example of what could go wrong.

**Explanation:**

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

**Specific Example:**

___________________________________________________________________________

___________________________________________________________________________

---

### Question 4 (13 points)

Consider the following lag feature creation code:

```python
def create_lag_features(data, target_col, max_lag=10):
    """Create lag features from time series data."""
    df = data.copy()

    for lag in range(1, max_lag + 1):
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

    # Drop rows with NaN
    df = df.dropna()

    return df
```

You use this to create 10 lag features from a dataset with 500 samples. You then run a GA with fitness evaluation using 5-fold TimeSeriesSplit.

**Part A (5 points):** What problem might occur during fitness evaluation, especially in early folds?

**Answer:**

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

**Part B (8 points):** Modify the code or suggest a strategy to handle this problem:

**Solution:**

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

---

## Section 3: Stationarity and Preprocessing (25 points)

### Question 5 (10 points)

Match each stationarity transformation with its appropriate use case:

| Transformation | Use Case |
|----------------|----------|
| 1. First differencing | A. Multiplicative trend/seasonality |
| 2. Log transformation | B. Linear trend removal |
| 3. Seasonal differencing | C. Repeating seasonal patterns |
| 4. Detrending (subtract MA) | D. Non-constant variance |

**Answers:**
- First differencing: ___________
- Log transformation: ___________
- Seasonal differencing: ___________
- Detrending: ___________

---

### Question 6 (15 points)

Consider a time series of monthly sales with an upward trend. You want to select features to predict future sales.

**Option A:** Use raw sales values as target
**Option B:** Use first-differenced sales (difference from previous month) as target
**Option C:** Use log-transformed sales as target

```python
# Raw
y_raw = sales

# Differenced
y_diff = sales.diff().dropna()

# Log-transformed
y_log = np.log(sales)
```

**Part A (5 points):** Which option is most likely to produce stationary target values? Why?

**Answer:** ___________

**Explanation:**

___________________________________________________________________________

___________________________________________________________________________

**Part B (5 points):** If you train a model on differenced sales (Option B), how do you convert predictions back to actual sales forecasts?

**Method:**

___________________________________________________________________________

___________________________________________________________________________

**Part C (5 points):** What implication does this choice have for feature engineering? Should you transform features the same way as the target?

**Answer:**

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

---

## Section 4: Temporal Dependencies (15 points)

### Question 7 (8 points)

True or False with detailed justification:

**Statement:** When selecting features for time series forecasting, you should test for stationarity of each candidate feature before including it in the GA optimization, and exclude any non-stationary features.

**Answer (T/F):** ___________

**Justification (3-4 sentences):**

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

---

### Question 8 (7 points)

You're using a GA to select from 30 technical indicators for predicting crypto prices. Some indicators are highly autocorrelated (e.g., 10-day SMA, 20-day SMA, 50-day SMA).

**Part A (4 points):** Why might selecting all three SMAs be suboptimal even if each individually correlates with the target?

**Answer:**

___________________________________________________________________________

___________________________________________________________________________

**Part B (3 points):** How could you modify the fitness function to discourage selecting redundant, correlated features?

**Answer:**

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

---

## Section 5: Data Leakage Prevention (10 points)

### Question 9 (10 points)

Identify ALL instances of data leakage in the following code:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

def leaky_fitness(chromosome, X, y, model):
    # Select features
    selected = X[:, chromosome == 1]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(selected)  # LINE A

    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)  # LINE B
        score = model.score(X_test, y_test)  # LINE C
        scores.append(score)

    return np.mean(scores),
```

**List ALL data leakage issues:**

**Issue 1:**

___________________________________________________________________________

___________________________________________________________________________

**Issue 2 (if any):**

___________________________________________________________________________

___________________________________________________________________________

**Corrected code (at least fix Issue 1):**

```python
def fixed_fitness(chromosome, X, y, model):
    selected = X[:, chromosome == 1]

    # YOUR CORRECTED CODE HERE
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for train_idx, test_idx in tscv.split(selected):










    return np.mean(scores),
```

---

## Section 6: Practical Application (Bonus: +10 points)

### Question 10 (10 points BONUS)

You're building a forex trading system that predicts EUR/USD returns 1 hour ahead. Your candidate features include:

- Lag returns: 1h, 2h, 4h, 8h, 24h ago
- Moving averages: 5h, 10h, 20h, 50h
- Volatility measures: 5h, 10h, 20h rolling std
- Time-of-day indicators: hour, day-of-week (encoded)
- Correlation with other pairs: EUR/GBP, GBP/USD

You have 2 years of hourly data (17,520 samples) and want to use a GA to select the optimal feature subset.

**Part A (4 points):** Describe the train/validation/test split strategy you would use. Be specific about:
- How you split the data
- What each split is used for
- How you prevent data leakage

**Strategy:**

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

**Part B (3 points):** The GA converges to a solution that selects only the 1h lag return and achieves excellent validation performance (R² = 0.85). Why should you be suspicious of this result?

**Answer:**

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

**Part C (3 points):** How would you detect whether the high performance is due to data leakage or look-ahead bias versus genuine predictive power?

**Answer:**

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

---

# Answer Key

## Section 1: Walk-Forward Validation

### Question 1 (10 points)

**Correct Answer:** B) 5 contiguous blocks, each with training on past data and testing on immediate future

**Strong Explanation:**
"TimeSeriesSplit creates 5 folds where each fold uses an expanding window of past data for training and the next contiguous block for testing. For 100 samples with 5 splits, it creates splits like: Fold 1 trains on samples 0-19, tests on 20-35; Fold 2 trains on 0-35, tests on 36-51; continuing with expanding training sets and contiguous future test sets. This mimics realistic forecasting where you train on all historical data and predict the immediate future."

**Key Points:**
1. Expanding/growing training window
2. Contiguous blocks (not random)
3. Testing always on immediate future after training
4. Simulates real forecasting scenario

**Grading:**
- B with excellent explanation (all key points): 10 points
- B with good explanation (3 key points): 8 points
- B with basic explanation (2 key points): 6 points
- B with weak explanation: 4 points
- Other answers: 0 points

---

### Question 2 (15 points)

**Correct Answers:**

1. `train_end`: `(i + 1) * split_size` OR `split_size * (i + 1)`

2. `test_start`: `(i + 1) * split_size` OR `train_end`

3. `test_end`: `(i + 2) * split_size` OR `test_start + split_size`

4. `X_train`: `:train_end` OR `0:train_end`

5. `X_test`: `test_start:test_end`

6. `y_train`: `:train_end` OR `0:train_end`

7. `y_test`: `test_start:test_end`

8. RMSE calculation: `np.sqrt(mean_squared_error(y_test, predictions))`
   OR `mean_squared_error(y_test, predictions, squared=False)`

**Grading:**
- All correct (within reasonable variation): 15 points
- 7-8 correct: 12 points
- 5-6 correct: 9 points
- 3-4 correct: 6 points
- 1-2 correct: 3 points
- Minor syntax issues: -1 point

---

## Section 2: Lag Feature Engineering

### Question 3 (12 points)

**Part A (4 points):**

**Answer:** 90 samples

**Explanation:** With maximum lag of 10, the first 10 rows will have NaN values for the lag_10 feature (since there's no data 10 days before). After dropping NaN, we have 100 - 10 = 90 valid samples.

**Grading:**
- 90 with correct explanation: 4 points
- 90 without explanation: 3 points
- Wrong answer but reasonable logic: 1 point

**Part B (8 points):**

**Strong Explanation:**
"It's crucial because using future information in fitness evaluation creates look-ahead bias, causing the GA to select features that appear predictive but actually exploit information not available at prediction time. This leads to severely overfit models that fail in production."

**Specific Example:**
"For example, if the fitness function uses TimeSeriesSplit incorrectly and doesn't respect temporal order, the GA might select lag_1 because the model can learn the relationship between day t+1 returns (test set) and day t returns (included in training set that extends into future). The model appears to predict well but is actually using future data to 'predict' the past. In real trading, you don't have tomorrow's return to predict today, so this feature selection fails catastrophically."

**Alternative Example:**
"If we accidentally include lag_1 from the original un-shifted series (same as current value), the GA will definitely select it since it perfectly predicts the target - but this is the target itself! The fitness function would show perfect accuracy, but in production, we don't have the current day's return to predict the current day's return."

**Grading:**
- Strong explanation (3-4 sentences) + specific example: 8 points
- Good explanation + good example: 6 points
- Basic explanation + example: 4 points
- Vague or missing example: 2 points

---

### Question 4 (13 points)

**Part A (5 points):**

**Strong Answer:**
"In early folds of TimeSeriesSplit, the training set is small (e.g., first fold might have only 100 samples), but we've already lost 10 samples to NaN from lag features. With small training sets and relatively high-dimensional feature space (10 lag features plus originals), the model might overfit severely. Additionally, some lags (like lag_10) need at least 10 samples to be meaningful, so very early folds may not have enough data to learn lag relationships properly."

**Key Issues to Mention:**
- Small training sets in early folds
- Further reduced by NaN removal
- Risk of overfitting
- Insufficient data for longer lags

**Grading:**
- Identifies small training set + NaN problem: 5 points
- Identifies only one issue clearly: 3 points
- Vague answer: 1 point

**Part B (8 points):**

**Strong Solutions:**

**Option 1: Minimum training set size**
```python
def create_lag_features(data, target_col, max_lag=10, min_train_size=50):
    df = data.copy()

    for lag in range(1, max_lag + 1):
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

    # Drop rows with NaN
    df = df.dropna()

    # Only use TimeSeriesSplit if we have enough samples
    # Or use fewer splits
    recommended_splits = max(3, len(df) // min_train_size)

    return df, recommended_splits
```

**Option 2: Forward-fill or adaptive lags**
```python
# Don't use lags longer than available data in each fold
# Or use forward-filling for missing early values (with caution)
```

**Option 3: Purging/embargo periods**
```python
# Add a gap between train and test to ensure lag features are valid
# Use TimeSeriesSplit with gap parameter
```

**Grading:**
- Complete, working solution with explanation: 8 points
- Good strategy with some implementation: 6 points
- Reasonable idea but incomplete: 4 points
- Vague suggestion: 2 points

---

## Section 3: Stationarity and Preprocessing

### Question 5 (10 points)

**Correct Answers:**
- First differencing: B (Linear trend removal)
- Log transformation: D (Non-constant variance) [Also acceptable: A]
- Seasonal differencing: C (Repeating seasonal patterns)
- Detrending: B (Linear trend removal) [Also acceptable: B]

**Note:** Some ambiguity exists - log can help with multiplicative trends too.

**Grading:**
- All correct (with reasonable interpretation): 10 points
- 3 correct: 7 points
- 2 correct: 4 points
- 1 correct: 2 points

---

### Question 6 (15 points)

**Part A (5 points):**

**Answer:** Option B (First-differenced)

**Explanation:**
"First differencing removes trends, which are the primary source of non-stationarity in trending data. Differencing transforms the series from modeling absolute levels to modeling changes, which typically have constant mean and variance. Raw sales have increasing mean (non-stationary), and log-transformed sales still have a trend (non-stationary), but differenced sales show stable statistical properties."

**Grading:**
- B with strong explanation: 5 points
- B with basic explanation: 3 points
- B with no/wrong explanation: 1 point
- Other answers: 0 points

**Part B (5 points):**

**Strong Answer:**
"To convert back, use cumulative sum starting from the last known actual value. If you predict differences [d1, d2, d3, ...], and last actual sales value is S_t, then forecasts are: S_{t+1} = S_t + d1, S_{t+2} = S_{t+1} + d2 = S_t + d1 + d2, etc. Each forecast is the previous value plus the predicted difference."

**Code Example:**
```python
# predictions_diff contains predicted differences
last_actual_value = sales.iloc[-1]
forecasts = [last_actual_value + predictions_diff[0]]
for diff in predictions_diff[1:]:
    forecasts.append(forecasts[-1] + diff)
```

**Grading:**
- Correct method with clear explanation: 5 points
- Correct idea but unclear: 3 points
- Partially correct: 2 points
- Wrong: 0 points

**Part C (5 points):**

**Strong Answer:**
"Not necessarily the same transformation. The target should be differenced to achieve stationarity, but features may need different treatments. Some features (like moving averages) are already stationary or less trended. Others might benefit from differencing. The GA can select from both raw and differenced features - include both versions and let evolution choose. The key is ensuring features don't use future information when computing differences."

**Key Points:**
- Don't need to transform all features the same way
- Can include multiple versions (raw, differenced, etc.)
- Let GA select which works best
- Still avoid look-ahead bias in transformations

**Grading:**
- Comprehensive answer with key points: 5 points
- Correct direction but incomplete: 3 points
- Misunderstanding (says must transform same way): 1 point

---

## Section 4: Temporal Dependencies

### Question 7 (8 points)

**Correct Answer:** False

**Strong Justification:**
"False. While stationarity is important for many time series models, you shouldn't pre-filter features based on stationarity tests before GA optimization. First, the relationship between features and target matters more than feature stationarity alone - a non-stationary feature can still be predictive if it co-integrates with the target. Second, stationarity depends on the transformation and time window - you might difference features during preprocessing. Third, the GA's fitness function (using validation) will naturally select features that generalize, which implicitly handles stationarity concerns. Pre-filtering removes potentially useful features and reduces the GA's search space unnecessarily. Instead, include both raw and transformed (differenced) versions and let the GA decide."

**Key Points:**
1. Relationship with target matters more than individual stationarity
2. Co-integration possible between non-stationary variables
3. Can transform features (difference them)
4. Fitness function implicitly handles this through validation
5. Pre-filtering removes useful features unnecessarily

**Grading:**
- False with strong justification (3+ key points): 8 points
- False with good justification (2 key points): 6 points
- False with basic justification: 4 points
- False with weak justification: 2 points
- True: 0 points

---

### Question 8 (7 points)

**Part A (4 points):**

**Strong Answer:**
"Selecting all three SMAs is suboptimal due to multicollinearity - they're highly correlated since they're all calculated from the same underlying price series, just with different windows. This causes redundancy (no additional information), increases model complexity unnecessarily, can harm model training (especially for linear models), and wastes computational resources evaluating correlated features."

**Key Points:**
- Multicollinearity/high correlation
- Redundant information
- Unnecessary complexity
- May harm model performance

**Grading:**
- Clear explanation with key points: 4 points
- Mentions correlation/redundancy: 3 points
- Vague understanding: 1 point

**Part B (3 points):**

**Strong Answers:**

**Option 1: Add correlation penalty**
```python
penalty = lambda_corr * mean_pairwise_correlation(selected_features)
fitness = accuracy - penalty
```

**Option 2: Add diversity objective**
```python
# In multi-objective: maximize accuracy, minimize feature correlation
diversity = 1 - mean_correlation(selected_features)
return accuracy, diversity
```

**Option 3: Cluster features and penalize multiple selections from same cluster**

**Grading:**
- Specific, implementable method: 3 points
- General idea but not specific: 2 points
- Vague: 1 point

---

## Section 5: Data Leakage Prevention

### Question 9 (10 points)

**Issue 1 (PRIMARY):**
"Data leakage at LINE A: The scaler is fit on the ENTIRE dataset (including test sets) before cross-validation. This means test set statistics (mean, std) leak into training, causing overfit fitness estimates. The scaler learns global statistics that include future information, which won't be available when predicting truly unseen data."

**Issue 2 (MINOR, if noted):**
"Depending on the model, LINE B might have issues if the model has internal data-dependent preprocessing or if it's not properly reset between folds, but this is less critical than the scaling issue."

**Corrected Code:**

```python
def fixed_fitness(chromosome, X, y, model):
    selected = X[:, chromosome == 1]

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for train_idx, test_idx in tscv.split(selected):
        X_train, X_test = selected[train_idx], selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit scaler ONLY on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        # Transform test data using training statistics
        X_test_scaled = scaler.transform(X_test)

        model.fit(X_train_scaled, y_train)
        score = model.score(X_test_scaled, y_test)
        scores.append(score)

    return np.mean(scores),
```

**Grading:**
- Identifies scaling leakage + correct fix: 10 points
- Identifies scaling leakage + partial fix: 7 points
- Identifies scaling leakage, no fix: 5 points
- Vague answer or wrong issue identified: 2 points
- Completely wrong: 0 points

---

## Section 6: Practical Application (Bonus)

### Question 10 (10 points BONUS)

**Part A (4 points):**

**Strong Strategy:**
"Use a three-way temporal split: Training (first 60%, ~12 months), Validation (next 20%, ~6 months), Test (final 20%, ~6 months). Training set is used to train models during GA fitness evaluation. Validation set is used as the GA's fitness evaluation set (what the GA optimizes) - run walk-forward validation on this set. Test set is held out completely and only used AFTER GA completes to assess final generalization. This prevents the GA from overfitting to the validation set. Within validation, use walk-forward with expanding window (e.g., 5 folds) to prevent data leakage. All preprocessing (scaling, feature engineering) is fit only on training data available at each point."

**Key Elements:**
1. Three splits: train/validation/test
2. GA optimizes on validation
3. Test is hold-out
4. Walk-forward within validation
5. Proper preprocessing order

**Grading:**
- Comprehensive strategy with all elements: 4 points
- Good strategy missing 1-2 elements: 3 points
- Basic understanding: 2 points
- Vague: 1 point

**Part B (3 points):**

**Strong Answer:**
"This result is highly suspicious because 1h lag return has an R² of 0.85 suggests near-perfect autocorrelation at 1 hour, which is unrealistic for forex markets (they're closer to random walks). This likely indicates data leakage - perhaps the 1h lag is actually 0h lag (using current return to predict current return), or the train/test split isn't properly temporal, or the target variable was accidentally included in features. True forex 1h autocorrelation is typically very small (<0.05 R²)."

**Key Points:**
- Unrealistically high R² for forex
- Markets are near random walks
- Suggests data leakage
- Likely same-time data included

**Grading:**
- Identifies unrealistic performance + suggests leakage: 3 points
- Notes suspicion but vague reason: 2 points
- Wrong reasoning: 0 points

**Part C (3 points):**

**Strong Answer:**
"To detect leakage: (1) Manually inspect the lag_1 feature creation to ensure it's truly shifted by 1 hour and not 0 hours. (2) Check train/test temporal separation - plot predictions vs actuals across time boundaries to see if performance drops at validation boundaries. (3) Test on hold-out test set from a completely different time period - genuine predictive power should transfer to new data. (4) Calculate the theoretical maximum R² from ACF analysis - if model exceeds this, it's using leaked information. (5) Compare performance to a naive baseline (previous value) - genuine signal should substantially beat this."

**Grading:**
- Multiple concrete detection methods: 3 points
- 1-2 reasonable methods: 2 points
- Vague suggestion: 1 point

---

## Score Interpretation

| Score Range | Performance Level | Recommendation |
|-------------|------------------|----------------|
| 95-110 (with bonus) | Exceptional | Ready for Module 4 |
| 85-94 | Strong | Ready for Module 4 |
| 75-84 | Good | Review walk-forward validation, proceed |
| 65-74 | Adequate | Review time series concepts carefully |
| Below 65 | Needs Improvement | Re-study module, especially validation |

## Common Misconceptions to Address

1. **TimeSeriesSplit vs KFold:** Students often use random CV for time series
2. **Lag Feature Creation:** Confusion about which data is "available" at prediction time
3. **Stationarity Requirements:** Over-emphasis on requiring stationary features
4. **Scaling Leakage:** Fitting scalers on entire dataset before CV
5. **Look-Ahead Bias:** Not recognizing subtle forms of using future information
6. **Differencing Predictions:** Forgetting to transform predictions back to original scale
