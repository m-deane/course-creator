# Module 2 Quiz: Fitness Function Design

**Course:** Genetic Algorithms for Feature Selection
**Module:** 2 - Fitness Function Design
**Total Points:** 100
**Estimated Time:** 30-35 minutes
**Attempts Allowed:** 2

## Instructions

This quiz evaluates your understanding of fitness function design, validation strategies, overfitting prevention, and multi-objective optimization for feature selection. Questions include both conceptual understanding and practical implementation scenarios.

---

## Section 1: Fitness Function Fundamentals (20 points)

### Question 1 (10 points)

Which of the following is the MOST critical consideration when designing a fitness function for time series feature selection?

A) Maximizing the number of selected features to capture all information
B) Using training set performance to enable fast fitness evaluation
C) Preventing data leakage by using proper temporal validation
D) Minimizing computational cost by using simple filter metrics

**Answer:** ___________

**Explanation (3 sentences):**

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

---

### Question 2 (10 points)

Consider the following fitness function:

```python
def fitness(chromosome, X_train, y_train, model):
    selected_features = X_train[:, chromosome == 1]

    if selected_features.shape[1] == 0:
        return -1e10,

    model.fit(selected_features, y_train)
    train_score = model.score(selected_features, y_train)

    return train_score,
```

**Part A (5 points):** Identify at least TWO serious problems with this fitness function.

**Problem 1:**

___________________________________________________________________________

**Problem 2:**

___________________________________________________________________________

**Part B (5 points):** What is the likely consequence of using this fitness function in a GA?

**Answer:**

___________________________________________________________________________

___________________________________________________________________________

---

## Section 2: Cross-Validation Strategies (30 points)

### Question 3 (12 points)

For time series data, which cross-validation strategy is appropriate and why?

```python
from sklearn.model_selection import KFold, TimeSeriesSplit

# Dataset: Monthly stock returns from 2010-2020 (120 samples)

# Strategy A: KFold
cv_a = KFold(n_splits=5, shuffle=True, random_state=42)

# Strategy B: KFold without shuffle
cv_b = KFold(n_splits=5, shuffle=False)

# Strategy C: TimeSeriesSplit
cv_c = TimeSeriesSplit(n_splits=5)
```

**Correct Strategy:** ___________

**Explanation (why this one and why not the others?):**

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

---

### Question 4 (10 points)

You're implementing walk-forward validation for feature selection. Complete the following code:

```python
from sklearn.model_selection import TimeSeriesSplit

def walk_forward_fitness(chromosome, X, y, model, n_splits=5):
    """
    Evaluate chromosome using walk-forward validation.

    Returns: Average test score across folds
    """
    selected_idx = np.where(chromosome == 1)[0]

    if len(selected_idx) == 0:
        return -np.inf,

    X_selected = X[:, selected_idx]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    scores = []
    for train_idx, test_idx in tscv.split(X_selected):
        # Split data
        X_train, X_test = ________________  # Fill in
        y_train, y_test = ________________  # Fill in

        # Train and evaluate
        model.fit(X_train, y_train)
        score = ________________  # Fill in (use MSE)

        scores.append(score)

    return ________________,  # Fill in (return average)
```

**Fill in the blanks:**

1. X_train, X_test: ___________________________________________________________

2. y_train, y_test: ___________________________________________________________

3. score calculation: _________________________________________________________

4. return statement: __________________________________________________________

---

### Question 5 (8 points)

What is the computational complexity of evaluating a single chromosome's fitness using k-fold cross-validation?

**Part A (4 points):** Express the complexity in terms of:
- n = number of samples
- k = number of CV folds
- m = number of selected features
- T(n,m) = time to train model on n samples with m features

**Complexity:** ___________________________________________________________

**Part B (4 points):** For a GA with population size 100, running for 50 generations, with 5-fold CV, approximately how many model training operations occur?

**Answer:** ___________

**Calculation:**

___________________________________________________________________________

---

## Section 3: Overfitting Prevention (25 points)

### Question 6 (12 points)

Implement a fitness function with parsimony pressure:

```python
def fitness_with_parsimony(chromosome, X, y, model, lambda_penalty=0.01):
    """
    Fitness function with parsimony pressure.

    Fitness = CV_Score - lambda * (num_selected / total_features)

    Returns: Tuple with single fitness value
    """
    from sklearn.model_selection import cross_val_score

    selected_idx = np.where(chromosome == 1)[0]

    # Handle edge case: no features selected
    if ________________:  # Fill in condition
        return ________________,  # Fill in return value

    X_selected = X[:, selected_idx]

    # Cross-validation score (negative MSE)
    cv_scores = cross_val_score(
        model, X_selected, y,
        cv=5,
        scoring='neg_mean_squared_error'
    )
    mean_score = np.mean(cv_scores)

    # Calculate parsimony penalty
    num_selected = ________________  # Fill in
    total_features = ________________  # Fill in
    penalty = lambda_penalty * (________________)  # Fill in

    # Final fitness
    fitness_value = ________________  # Fill in

    return fitness_value,
```

**Fill in the 6 blanks above.**

---

### Question 7 (8 points)

Consider two chromosomes evaluated on the same dataset:

| Chromosome | CV Score (neg_MSE) | Features Selected | Features Total |
|------------|-------------------|------------------|----------------|
| A | -0.045 | 8 | 20 |
| B | -0.050 | 15 | 20 |

Using the fitness function: `Fitness = CV_Score - 0.02 * (num_selected / total)`

**Part A (4 points):** Calculate the fitness for both chromosomes.

**Fitness A:** ___________

**Fitness B:** ___________

**Part B (4 points):** Which chromosome would be selected by the GA? Explain why this is beneficial.

**Selected:** ___________

**Explanation:**

___________________________________________________________________________

___________________________________________________________________________

---

### Question 8 (5 points)

True or False with justification:

**Statement:** Increasing the parsimony penalty parameter (λ) will always improve generalization performance by preventing overfitting.

**Answer (T/F):** ___________

**Justification:**

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

---

## Section 4: Multi-Objective Optimization (25 points)

### Question 9 (10 points)

In multi-objective feature selection, we optimize both accuracy and complexity. Which of the following statements are TRUE? Select ALL that apply.

- [ ] A. A solution dominates another if it is better in at least one objective and not worse in any objective
- [ ] B. The Pareto front contains all non-dominated solutions
- [ ] C. All solutions on the Pareto front have the same fitness value
- [ ] D. Multi-objective GAs require different selection operators than single-objective GAs
- [ ] E. A solution with 10 features and 90% accuracy can dominate a solution with 15 features and 89% accuracy
- [ ] F. You must choose a single best solution from the Pareto front before deployment

**Selected Answers:** ___________

---

### Question 10 (15 points)

Complete the multi-objective fitness function:

```python
def multi_objective_fitness(chromosome, X, y, model):
    """
    Returns: (accuracy_objective, complexity_objective)
    Both objectives should be MAXIMIZED.
    """
    from sklearn.model_selection import cross_val_score

    selected_idx = np.where(chromosome == 1)[0]

    # Handle no features selected
    if len(selected_idx) == 0:
        return ________________, ________________  # Fill in

    X_selected = X[:, selected_idx]

    # Objective 1: Accuracy (as negative MSE, to be maximized)
    cv_scores = cross_val_score(
        model, X_selected, y,
        cv=5,
        scoring='neg_mean_squared_error'
    )
    accuracy_obj = ________________  # Fill in

    # Objective 2: Simplicity (inverse of complexity, to be maximized)
    num_selected = len(selected_idx)
    total_features = len(chromosome)
    simplicity_obj = ________________  # Fill in

    return ________________, ________________  # Fill in
```

**Part A (10 points):** Fill in the 5 blanks above.

**Part B (5 points):** Why is it important that both objectives be formulated as maximization problems for NSGA-II?

**Answer:**

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

---

## Section 5: Practical Implementation (Bonus: +10 points)

### Question 11 (10 points BONUS)

You notice that your GA for feature selection is taking too long to run. The bottleneck is fitness evaluation. Your current setup:

- Population size: 100
- Generations: 100
- CV folds: 10
- Features: 50
- Model: Random Forest with 100 trees

**Part A (5 points):** Estimate the total number of Random Forest models that need to be trained during the entire GA run. Show your calculation.

**Calculation:**

___________________________________________________________________________

___________________________________________________________________________

**Total Models:** ___________

**Part B (5 points):** Suggest THREE practical strategies to speed up fitness evaluation WITHOUT significantly reducing solution quality.

**Strategy 1:**

___________________________________________________________________________

**Strategy 2:**

___________________________________________________________________________

**Strategy 3:**

___________________________________________________________________________

---

# Answer Key

## Section 1: Fitness Function Fundamentals

### Question 1 (10 points)

**Correct Answer:** C) Preventing data leakage by using proper temporal validation

**Strong Explanation Examples:**
- "Temporal validation is critical because time series data has autocorrelation and temporal dependencies. Using future data to predict the past (data leakage) leads to unrealistically optimistic fitness scores. This causes the GA to select features that appear good in training but fail in actual forward prediction. Proper walk-forward validation ensures features are selected based on realistic forecasting performance."

- "Time series requires respecting temporal order during validation to prevent look-ahead bias. If we use random CV or training set performance, the GA will select features that exploit future information, leading to overfitting. The fitness function must use walk-forward or expanding window validation to simulate real forecasting conditions where only past data is available."

**Grading:**
- C with strong explanation covering data leakage and temporal order: 10 points
- C with moderate explanation: 7 points
- C with weak explanation: 5 points
- Other answers: 0 points

---

### Question 2 (10 points)

**Part A (5 points):**

**Critical Problems:**

1. **No validation set - trains and tests on same data**
   - Using `train_score` on training data guarantees overfitting
   - Fitness will select features that memorize training data

2. **No cross-validation**
   - Single train/test split provides unreliable fitness estimate
   - High variance in fitness across random splits

3. **For time series: No temporal validation**
   - If data is time series, this violates temporal ordering
   - Causes look-ahead bias

**Grading Part A:**
- 2 distinct, serious problems identified: 5 points
- 1 serious problem: 3 points
- Minor issues only: 1 point

**Part B (5 points):**

**Strong Answers:**
- "The GA will quickly converge to chromosomes that select many features, achieving near-perfect training accuracy but performing poorly on unseen data. The evolved feature sets will be overfit to training data noise rather than capturing genuine predictive patterns."

- "Severe overfitting. The fitness function rewards features that memorize training data, so the GA will evolve increasingly complex feature sets with excellent training scores but poor generalization. This defeats the purpose of feature selection."

**Grading Part B:**
- Clearly explains overfitting consequence: 5 points
- Mentions overfitting but vague: 3 points
- Wrong/unclear: 0 points

---

## Section 2: Cross-Validation Strategies

### Question 3 (12 points)

**Correct Strategy:** C (TimeSeriesSplit)

**Strong Explanation:**
"TimeSeriesSplit is correct because it respects temporal order, training on past data and testing on future data, which matches real forecasting scenarios. Strategy A (KFold with shuffle) is wrong because shuffling destroys temporal structure and creates look-ahead bias - the model trains on future data to predict the past. Strategy B (KFold without shuffle) is also incorrect because even without shuffling, KFold creates non-contiguous training sets that can include future observations, violating the causal structure of time series."

**Key Points:**
1. Identifies temporal ordering requirement
2. Explains why shuffling is problematic (look-ahead bias)
3. Understands that even non-shuffled KFold violates temporal structure
4. Recognizes TimeSeriesSplit matches forecasting reality

**Grading:**
- C with strong explanation hitting all key points: 12 points
- C with good explanation (3 key points): 9 points
- C with basic explanation (2 key points): 6 points
- C with weak explanation: 3 points
- Wrong strategy: 0 points

---

### Question 4 (10 points)

**Correct Answers:**

1. `X_selected[train_idx], X_selected[test_idx]`

2. `y[train_idx], y[test_idx]`

3. `-mean_squared_error(y_test, model.predict(X_test))`
   OR `model.score(X_test, y_test)` (if using R² score)
   OR `mean_squared_error(y_test, model.predict(X_test))` (depends on whether negative or not)

4. `np.mean(scores)` OR `sum(scores) / len(scores)`

**Note:** For #3, accept any valid MSE calculation. The sign depends on interpretation.

**Grading:**
- All 4 correct: 10 points (2.5 per blank)
- 3 correct: 7 points
- 2 correct: 4 points
- 1 correct: 2 points
- Minor syntax errors with correct logic: -1 point per error

---

### Question 5 (8 points)

**Part A (4 points):**

**Correct Answer:** O(k × T(n, m)) or O(k × n × m) for linear models

**Explanation:**
- k-fold CV requires k training operations
- Each training operation takes T(n,m) time
- Total is k × T(n,m)

**Grading:**
- Correct expression with k and T(n,m): 4 points
- Shows k multiplier but unclear on T(n,m): 2 points
- Wrong: 0 points

**Part B (4 points):**

**Correct Answer:** Approximately 25,000 model training operations

**Calculation:**
- Population size: 100
- Generations: 50
- Total individual evaluations: 100 × 50 = 5,000
- CV folds per evaluation: 5
- Total training operations: 5,000 × 5 = 25,000

**Note:** Accept 20,000-30,000 range accounting for elitism, crossover probability, etc.

**Grading:**
- Correct calculation (24,000-26,000): 4 points
- Reasonable range with some consideration: 3 points
- Shows understanding but calculation error: 2 points
- Wrong approach: 0 points

---

## Section 3: Overfitting Prevention

### Question 6 (12 points)

**Correct Answers:**

1. Condition: `len(selected_idx) == 0`

2. Return value: `-np.inf` OR `-1e10` OR very large negative number

3. num_selected: `len(selected_idx)` OR `np.sum(chromosome)` OR `sum(chromosome)`

4. total_features: `len(chromosome)`

5. penalty calculation: `num_selected / total_features`

6. fitness_value: `mean_score - penalty`

**Grading:**
- All 6 correct: 12 points (2 per blank)
- 5 correct: 10 points
- 4 correct: 8 points
- 3 correct: 5 points
- Fewer: proportional (1-2 points)

---

### Question 7 (8 points)

**Part A (4 points):**

**Fitness A Calculation:**
- CV_Score: -0.045
- Penalty: 0.02 × (8/20) = 0.02 × 0.4 = 0.008
- Fitness A = -0.045 - 0.008 = **-0.053**

**Fitness B Calculation:**
- CV_Score: -0.050
- Penalty: 0.02 × (15/20) = 0.02 × 0.75 = 0.015
- Fitness B = -0.050 - 0.015 = **-0.065**

**Grading:**
- Both correct: 4 points
- One correct: 2 points
- Shows work but calculation error: 1 point

**Part B (4 points):**

**Selected:** Chromosome A (higher fitness: -0.053 > -0.065)

**Strong Explanation:**
- "Chromosome A is selected because it has higher fitness despite slightly worse CV score. This is beneficial because A achieves nearly the same accuracy (-0.045 vs -0.050) with 47% fewer features (8 vs 15), resulting in a simpler, more interpretable model that's less prone to overfitting and faster to deploy."

**Grading:**
- A with strong explanation: 4 points
- A with basic explanation: 3 points
- A with no/wrong explanation: 1 point
- B selected: 0 points

---

### Question 8 (5 points)

**Correct Answer:** False

**Strong Justification:**
- "False. While parsimony pressure helps prevent overfitting by favoring simpler models, excessive λ values can cause underfitting by penalizing useful features too heavily. The GA may converge to very small feature sets with poor predictive performance. The optimal λ balances model complexity and accuracy, and must be tuned for each problem."

- "False. Very high λ values lead to underfitting, where the GA selects too few features to capture the underlying patterns. There's an optimal λ that balances complexity and accuracy - too low causes overfitting, too high causes underfitting. Always improving generalization would require perfect λ tuning, which isn't guaranteed by simply increasing λ."

**Grading:**
- False with strong justification mentioning underfitting: 5 points
- False with reasonable justification: 3 points
- False with weak justification: 2 points
- True: 0 points

---

## Section 4: Multi-Objective Optimization

### Question 9 (10 points)

**Correct Answers:** A, B, D, E

**Explanation:**
- **A: TRUE** - Dominance requires better in ≥1 objective, not worse in any
- **B: TRUE** - Pareto front is by definition all non-dominated solutions
- **C: FALSE** - Pareto front solutions have different trade-offs, different fitness vectors
- **D: TRUE** - Multi-objective uses specialized selection (NSGA-II uses crowding distance)
- **E: TRUE** - Better in both objectives (10<15 features, 90%>89% accuracy)
- **F: FALSE** - You choose based on deployment constraints, but don't have to pick one

**Grading:**
- All 4 correct (A,B,D,E), no incorrect selections: 10 points
- 3 correct, no incorrect: 7 points
- 4 correct but 1 incorrect selected: 6 points
- 2 correct, no incorrect: 4 points
- Other combinations: 0-3 points proportional

---

### Question 10 (15 points)

**Part A (10 points):**

**Correct Answers:**

1. Return for no features: `-np.inf, -np.inf` OR `(-float('inf'), -float('inf'))`

2. accuracy_obj: `np.mean(cv_scores)` (already negative MSE, higher is better)

3. simplicity_obj: `1 - (num_selected / total_features)`
   OR `(total_features - num_selected) / total_features`

4. Final return: `accuracy_obj, simplicity_obj`

**Alternative for #3:** Some might use `total_features / num_selected` but this has issues when num_selected is small.

**Grading Part A:**
- All 5 correct: 10 points (2 per blank)
- 4 correct: 8 points
- 3 correct: 6 points
- 2 correct: 4 points
- 1 correct: 2 points

**Part B (5 points):**

**Strong Answer:**
"NSGA-II uses Pareto dominance ranking which requires consistent optimization direction across all objectives. If objectives have mixed directions (some minimize, some maximize), the dominance comparison becomes ambiguous. By formulating all objectives as maximization (or all as minimization), the algorithm can correctly identify which solutions dominate others: solution A dominates B if A is better in all objectives."

**Key Points:**
- Consistency in optimization direction
- Required for dominance comparison
- Prevents ambiguity in Pareto ranking

**Grading Part B:**
- Strong answer covering key points: 5 points
- Mentions consistency/dominance: 3 points
- Vague but correct direction: 2 points
- Wrong: 0 points

---

## Section 5: Practical Implementation (Bonus)

### Question 11 (10 points BONUS)

**Part A (5 points):**

**Calculation:**
- Population size: 100 individuals
- Generations: 100
- Assuming each generation evaluates all 100 individuals: 100 × 100 = 10,000 individual evaluations
- CV folds: 10
- Total model trainings: 10,000 × 10 = **100,000 Random Forest models**

**Notes:**
- Could be slightly less with elitism (elit individuals not re-evaluated)
- Initial population adds 100 evaluations (100 × 10 = 1,000 models)
- Total: approximately 100,000 models

**Grading:**
- Correct calculation (95,000-110,000): 5 points
- Shows understanding but arithmetic error: 3 points
- Wrong approach: 0 points

**Part B (5 points):**

**Strong Strategies:**

1. **Reduce CV folds**
   - "Use 5-fold CV instead of 10-fold. Research shows 5-fold provides reliable estimates with half the computation. For time series, use fewer expanding window folds."

2. **Reduce Random Forest trees**
   - "Use 50 trees instead of 100. Performance typically plateaus around 50-100 trees, and fitness evaluation is relative ranking, not absolute accuracy, so fewer trees suffice."

3. **Parallelize fitness evaluation**
   - "Use joblib or multiprocessing to evaluate multiple individuals in parallel. GA populations are embarrassingly parallel - each individual's fitness is independent."

4. **Reduce population size or generations**
   - "Use population size of 50 with 150 generations. Smaller populations with more generations often find similar solutions with fewer total evaluations."

5. **Use surrogate models**
   - "Train a surrogate model (like Gaussian Process) on evaluated chromosomes to predict fitness for new ones, only doing full evaluation for promising candidates."

6. **Smart initialization**
   - "Initialize population with filter-method results or domain knowledge rather than random, reducing generations needed for convergence."

**Grading (any 3 strategies):**
- 3 practical, well-explained strategies: 5 points
- 3 reasonable strategies, brief explanation: 4 points
- 2 good strategies: 3 points
- 1 good strategy: 1 point
- Strategies that harm quality significantly: 0 points

---

## Score Interpretation

| Score Range | Performance Level | Recommendation |
|-------------|------------------|----------------|
| 95-110 (with bonus) | Exceptional | Ready for Module 3 |
| 85-94 | Strong | Ready for Module 3 |
| 75-84 | Good | Review validation strategies, proceed |
| 65-74 | Adequate | Review fitness design and CV concepts |
| Below 65 | Needs Improvement | Re-study module, especially CV and multi-objective |

## Common Misconceptions to Address

1. **Training Set Fitness:** Using training performance instead of validation
2. **Random CV for Time Series:** Not respecting temporal order
3. **Parsimony Always Helps:** Excessive penalties cause underfitting
4. **Multi-Objective is Weighted Sum:** NSGA-II uses Pareto dominance, not scalarization
5. **CV Complexity:** Underestimating computational cost of k-fold CV
6. **Empty Feature Sets:** Not handling edge case of zero selected features
