# Module 0 Quiz: Foundations of Feature Selection

**Course:** Genetic Algorithms for Feature Selection
**Module:** 0 - Foundations
**Total Points:** 100
**Estimated Time:** 25-30 minutes
**Attempts Allowed:** 2

## Instructions

This quiz assesses your understanding of the feature selection problem, search space complexity, and optimization fundamentals. Answer all questions to the best of your ability. Partial credit may be awarded for coding questions that demonstrate correct understanding.

---

## Section 1: Feature Selection Problem (30 points)

### Question 1 (8 points)

You have a dataset with 25 candidate features. How many possible feature subsets exist?

A) 625
B) 33,554,432
C) 25!
D) 1,000,000

**Answer:** ___________

---

### Question 2 (10 points)

Which of the following BEST describes the "curse of dimensionality" in feature selection?

A) As the number of features increases linearly, the search space grows exponentially, making exhaustive search computationally infeasible
B) Adding more features always improves model performance because more information is available
C) High-dimensional data requires more memory to store, causing storage problems
D) Feature selection is impossible when you have more than 100 features

**Answer:** ___________

---

### Question 3 (12 points)

Consider a time series forecasting problem with 30 candidate lag features. You want to select the optimal subset using exhaustive search, evaluating each subset with 5-fold cross-validation. If each cross-validation run takes 0.01 seconds:

**Part A (6 points):** How many total feature subsets need to be evaluated?

**Answer:** ___________

**Part B (6 points):** Approximately how long would this exhaustive search take in hours? Show your calculation.

**Answer:** ___________

---

## Section 2: Selection Approaches (25 points)

### Question 4 (10 points)

Match each feature selection approach with its primary characteristic:

| Approach | Characteristic |
|----------|---------------|
| 1. Filter | A. Uses model performance during training |
| 2. Wrapper | B. Statistical correlation with target |
| 3. Embedded | C. Evaluates feature subsets using full model |
| 4. Genetic Algorithm | D. Evolutionary search through subset space |

**Answers:**
- Filter: ___________
- Wrapper: ___________
- Embedded: ___________
- Genetic Algorithm: ___________

---

### Question 5 (8 points)

You need to select features for a random forest model predicting stock returns. Your dataset has 100 candidate features and 5000 samples. Which approach would you recommend and why?

A) Filter methods - fastest and most efficient for large feature sets
B) Exhaustive wrapper search - guarantees optimal solution
C) Genetic algorithm wrapper - balances search quality and computational cost
D) No feature selection - use all features to maximize information

**Answer:** ___________

**Justification (2 sentences):**

___________________________________________________________________________

___________________________________________________________________________

---

### Question 6 (7 points)

Rank the following feature selection approaches from LOWEST to HIGHEST computational cost (1 = lowest, 4 = highest):

- _____ Filter methods (e.g., correlation-based)
- _____ Embedded methods (e.g., LASSO regularization)
- _____ Wrapper with exhaustive search
- _____ Wrapper with genetic algorithm

---

## Section 3: Optimization Fundamentals (25 points)

### Question 7 (10 points)

Consider the following binary encoding for a feature subset:

```
Features: [lag1, lag2, lag3, sma5, sma10, rsi, macd, volume]
Chromosome: [1, 0, 1, 1, 0, 0, 1, 1]
```

**Part A (5 points):** Which features are SELECTED in this chromosome?

**Answer:** ___________

**Part B (5 points):** What is the feature subset size (number of selected features)?

**Answer:** ___________

---

### Question 8 (8 points)

In the context of feature selection optimization, what does a "fitness landscape" represent?

A) A 2D visualization of feature importance scores
B) The mapping from all possible feature subsets to their performance metrics
C) The correlation matrix between features
D) The training curve showing accuracy over epochs

**Answer:** ___________

---

### Question 9 (7 points)

True or False: Justify your answer with one sentence.

**Statement:** For a feature selection problem with 50 features, the search space is continuous because feature importance can take any value between 0 and 1.

**Answer (T/F):** ___________

**Justification:**

___________________________________________________________________________

___________________________________________________________________________

---

## Section 4: Why Genetic Algorithms? (20 points)

### Question 10 (10 points)

Which of the following are advantages of using genetic algorithms for feature selection? Select ALL that apply.

- [ ] A. Do not require gradient information
- [ ] B. Always find the global optimal solution
- [ ] C. Can handle large, discrete search spaces efficiently
- [ ] D. Naturally support parallelization
- [ ] E. Complete search in polynomial time
- [ ] F. Can optimize custom, non-differentiable fitness functions
- [ ] G. Require less computation than filter methods

**Selected answers:** ___________

---

### Question 11 (10 points)

Consider three optimization approaches for selecting features from a pool of 40 candidates:

1. **Greedy forward selection:** Start with no features, iteratively add the feature that most improves performance
2. **Exhaustive search:** Evaluate all 2^40 possible subsets
3. **Genetic algorithm:** Evolve a population of 100 chromosomes for 50 generations

**Part A (5 points):** Which approach is MOST likely to find the global optimal solution?

**Answer:** ___________

**Part B (5 points):** Which approach offers the best balance of solution quality and computational feasibility for this problem? Explain in 1-2 sentences.

**Answer:** ___________

**Explanation:**

___________________________________________________________________________

___________________________________________________________________________

---

## Section 5: Coding Comprehension (Bonus: +5 points)

### Question 12 (5 points BONUS)

Analyze the following Python code for feature selection:

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

def evaluate_features(chromosome, X, y):
    selected_idx = np.where(chromosome == 1)[0]

    if len(selected_idx) == 0:
        return -np.inf

    X_selected = X[:, selected_idx]
    model = Ridge(alpha=1.0)
    scores = cross_val_score(model, X_selected, y, cv=5, scoring='neg_mean_squared_error')

    return np.mean(scores)
```

What happens if `chromosome` is all zeros (no features selected)? Why is this important for genetic algorithms?

**Answer:**

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

---

# Answer Key

## Section 1: Feature Selection Problem

### Question 1 (8 points)
**Correct Answer:** B) 33,554,432

**Explanation:** With 25 features, there are 2^25 = 33,554,432 possible subsets (including the empty set and the full set). Each feature can either be selected (1) or not selected (0), giving 2 choices per feature.

**Grading:**
- Correct answer: 8 points
- Shows calculation 2^25 but wrong final number: 6 points
- Other answers: 0 points

---

### Question 2 (10 points)
**Correct Answer:** A) As the number of features increases linearly, the search space grows exponentially, making exhaustive search computationally infeasible

**Explanation:** The curse of dimensionality in feature selection specifically refers to the exponential growth of the search space (2^p subsets for p features). While high dimensionality can affect model performance and memory, the question asks about the context of feature selection optimization.

**Grading:**
- A: 10 points (correct)
- B: 0 points (misconception - more features often hurt due to overfitting)
- C: 3 points (true but not the primary issue for feature selection)
- D: 0 points (incorrect - feature selection is possible but challenging)

---

### Question 3 (12 points)
**Part A Correct Answer:** 1,073,741,824 (or 2^30)

**Explanation:** With 30 features, there are 2^30 = 1,073,741,824 possible subsets.

**Grading:**
- Correct answer: 6 points
- Shows 2^30 but calculation error: 4 points
- Uses wrong formula: 0 points

**Part B Correct Answer:** Approximately 14,930 hours (or about 621 days or 1.7 years)

**Calculation:**
- Subsets: 2^30 = 1,073,741,824
- CV folds: 5
- Total evaluations: 1,073,741,824 × 5 = 5,368,709,120
- Time per evaluation: 0.01 seconds
- Total seconds: 53,687,091.2 seconds
- Total hours: 53,687,091.2 / 3600 ≈ 14,913 hours

**Grading:**
- Correct calculation (14,000-15,000 hours): 6 points
- Shows correct approach but arithmetic error: 4 points
- Forgets to multiply by CV folds: 3 points
- Wrong approach: 0 points

---

## Section 2: Selection Approaches

### Question 4 (10 points)
**Correct Answers:**
- Filter: B (Statistical correlation with target)
- Wrapper: C (Evaluates feature subsets using full model)
- Embedded: A (Uses model performance during training)
- Genetic Algorithm: D (Evolutionary search through subset space)

**Grading:**
- All 4 correct: 10 points
- 3 correct: 7 points
- 2 correct: 4 points
- 1 correct: 2 points
- 0 correct: 0 points

---

### Question 5 (8 points)
**Correct Answer:** C) Genetic algorithm wrapper - balances search quality and computational cost

**Strong Justification Examples:**
- "With 100 features, exhaustive search requires 2^100 evaluations (impossible). Filter methods don't account for feature interactions. GAs can explore the space efficiently while using model performance for evaluation."
- "GAs can find near-optimal solutions in reasonable time by intelligently searching the 2^100 subset space, unlike exhaustive search which is infeasible and filter methods which may miss feature interactions."

**Grading:**
- C with strong justification: 8 points
- C with weak justification: 6 points
- A with reasonable justification about speed: 4 points
- B (shows misunderstanding of scale): 0 points
- D (shows misunderstanding of feature selection purpose): 0 points

---

### Question 6 (7 points)
**Correct Ranking:**
1. Filter methods (lowest)
2. Embedded methods
3. Wrapper with genetic algorithm
4. Wrapper with exhaustive search (highest)

**Grading:**
- All correct: 7 points
- 1 adjacent swap: 5 points
- 2 adjacent swaps: 3 points
- More errors: 0 points

---

## Section 3: Optimization Fundamentals

### Question 7 (10 points)
**Part A Correct Answer:** lag1, lag3, sma5, macd, volume

**Explanation:** Features with chromosome value 1 are selected. Positions 0, 2, 3, 6, 7 have value 1.

**Grading:**
- All 5 features correct: 5 points
- 4 correct: 3 points
- 3 or fewer: 0 points

**Part B Correct Answer:** 5

**Grading:**
- Correct: 5 points
- Incorrect: 0 points

---

### Question 8 (8 points)
**Correct Answer:** B) The mapping from all possible feature subsets to their performance metrics

**Explanation:** A fitness landscape maps every point in the search space (each possible feature subset) to its fitness value (performance metric). This creates a high-dimensional "landscape" that the GA must navigate.

**Grading:**
- B: 8 points
- A: 2 points (confuses visualization with concept)
- C: 0 points (unrelated)
- D: 0 points (unrelated)

---

### Question 9 (7 points)
**Correct Answer:** False

**Strong Justification Examples:**
- "False. The search space is discrete because each feature is either selected (1) or not selected (0), giving 2^50 distinct possibilities."
- "False. Binary feature selection creates a discrete search space with exactly 2^50 possible subsets, not a continuous range."

**Grading:**
- False with correct justification: 7 points
- False with weak/incorrect justification: 3 points
- True: 0 points

---

## Section 4: Why Genetic Algorithms?

### Question 10 (10 points)
**Correct Answers:** A, C, D, F

**Explanation:**
- A: TRUE - GAs use fitness evaluation, not gradients
- B: FALSE - GAs find good solutions but don't guarantee global optimum
- C: TRUE - GAs efficiently explore large discrete spaces
- D: TRUE - Population-based nature enables parallelization
- E: FALSE - GAs are heuristic with no polynomial time guarantee
- F: TRUE - Any fitness function can be used
- G: FALSE - GAs require more computation than simple filters

**Grading:**
- All 4 correct, no incorrect selections: 10 points
- 3 correct, no incorrect selections: 7 points
- 4 correct but 1 incorrect selected: 6 points
- 2 correct, no incorrect selections: 4 points
- Other combinations: 0-3 points proportional to accuracy

---

### Question 11 (10 points)
**Part A Correct Answer:** 2. Exhaustive search

**Grading:**
- Exhaustive search: 5 points
- Other answers: 0 points

**Part B Correct Answer:** 3. Genetic algorithm

**Strong Explanation Examples:**
- "GA offers the best balance because exhaustive search is computationally impossible (2^40 ≈ 1 trillion evaluations) while greedy search gets stuck in local optima. GAs efficiently explore promising regions without evaluating every possibility."
- "Genetic algorithm balances quality and feasibility by intelligently sampling the search space over generations, avoiding both the impossibility of exhaustive search and the limited exploration of greedy methods."

**Grading:**
- GA with strong explanation: 5 points
- GA with weak explanation: 3 points
- Greedy with reasonable trade-off argument: 2 points
- Exhaustive (contradicts feasibility): 0 points

---

## Section 5: Coding Comprehension (Bonus)

### Question 12 (5 points BONUS)
**Strong Answer Example:**

"When chromosome is all zeros, `selected_idx` will be an empty array, causing `len(selected_idx) == 0` to be True. The function returns `-np.inf` (negative infinity) as the fitness. This is important for GAs because it prevents the evolution of invalid solutions with no features - such chromosomes receive the worst possible fitness and are eliminated through selection, ensuring the population maintains at least one feature in viable solutions."

**Key Points for Full Credit:**
1. Recognizes the empty array condition (1 point)
2. Identifies return of -np.inf (1 point)
3. Explains this gives worst possible fitness (1 point)
4. Connects to GA selection pressure eliminating these solutions (2 points)

**Grading:**
- All key points covered: 5 points
- 3-4 key points: 3 points
- 1-2 key points: 1 point
- Wrong understanding: 0 points

---

## Score Interpretation

| Score Range | Performance Level | Recommendation |
|-------------|------------------|----------------|
| 95-105 (with bonus) | Exceptional | Ready for Module 1 |
| 85-94 | Strong | Ready for Module 1 |
| 75-84 | Good | Review weak areas, proceed to Module 1 |
| 65-74 | Adequate | Review module materials before continuing |
| Below 65 | Needs Improvement | Re-study module, retake quiz |

## Common Misconceptions to Address

1. **Search Space Size:** Many students underestimate exponential growth
2. **GA Guarantees:** GAs find good solutions but not guaranteed global optima
3. **Filter vs Wrapper:** Filters don't use model performance in selection
4. **Continuous vs Discrete:** Feature selection is discrete optimization
5. **Computational Cost:** Understanding the trade-offs between methods
