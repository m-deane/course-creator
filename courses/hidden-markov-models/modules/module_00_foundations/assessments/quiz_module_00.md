# Quiz: Module 0 - Foundations of Markov Chains

**Course:** Hidden Markov Models
**Module:** 0 - Foundations
**Total Points:** 100
**Estimated Time:** 25-30 minutes
**Attempts Allowed:** 2

---

## Instructions

This quiz assesses your understanding of Markov chain fundamentals, transition matrices, and probability theory foundations needed for Hidden Markov Models. Show all work for mathematical problems. Select the best answer for multiple choice questions.

---

## Section 1: Markov Chain Fundamentals (30 points)

### Question 1 (8 points)

Which property best defines the Markov property for a stochastic process {X_t}?

A) The future state depends on all past states with equal weight
B) P(X_{t+1} | X_t, X_{t-1}, ..., X_0) = P(X_{t+1} | X_t)
C) The process must eventually return to its starting state
D) All states must be reachable from every other state

**Answer: B**

**Explanation:**
The Markov property (memorylessness) states that the conditional probability of the next state depends only on the current state, not on the sequence of states that preceded it. This is formally written as P(X_{t+1} | X_t, X_{t-1}, ..., X_0) = P(X_{t+1} | X_t).

- A is incorrect: Past states have no influence, not equal influence
- C describes recurrence, not the Markov property
- D describes irreducibility, a separate property

---

### Question 2 (10 points)

Consider a Markov chain with three states: {Bull, Neutral, Bear}. The transition matrix is:

```
       Bull   Neutral  Bear
Bull   [0.6    0.3     0.1  ]
Neutral[0.2    0.5     0.3  ]
Bear   [0.1    0.3     0.6  ]
```

If the market is currently in the Bull state, what is the probability that it will be in the Bear state after exactly 2 time steps?

A) 0.10
B) 0.16
C) 0.19
D) 0.25

**Answer: C (0.19)**

**Explanation:**
We need to compute P(X_2 = Bear | X_0 = Bull) = (P^2)[Bull, Bear]

First, identify paths from Bull to Bear in 2 steps:
- Bull → Bull → Bear: 0.6 × 0.1 = 0.06
- Bull → Neutral → Bear: 0.3 × 0.3 = 0.09
- Bull → Bear → Bear: 0.1 × 0.6 = 0.06

Total: 0.06 + 0.09 + 0.06 = 0.19

Alternatively, compute P^2:
```
P^2 = P × P
    = [0.6  0.3  0.1]   [0.6  0.3  0.1]
      [0.2  0.5  0.3] × [0.2  0.5  0.3]
      [0.1  0.3  0.6]   [0.1  0.3  0.6]
```

The (0,2) element (Bull to Bear) = 0.6×0.1 + 0.3×0.3 + 0.1×0.6 = 0.19

---

### Question 3 (12 points)

A Markov chain is said to be **irreducible** if:

A) Every state can be reached from every other state
B) The transition matrix has all positive entries
C) The chain has a unique stationary distribution
D) All states have period 1

**Answer: A**

**Part B:** Which of the following transition matrices represents an irreducible chain?

Matrix 1:
```
[0.5  0.5  0  ]
[0.3  0.4  0.3]
[0    0.5  0.5]
```

Matrix 2:
```
[0.4  0.3  0.3]
[0.2  0.5  0.3]
[0.3  0.3  0.4]
```

**Answer: Matrix 2**

**Explanation:**
Irreducibility means that for any pair of states (i, j), there exists some n such that P^n(i,j) > 0. In other words, you can reach any state from any other state in a finite number of steps.

Matrix 1 is NOT irreducible because state 0 cannot reach state 2 (note the 0 in position (0,2)), and state 2 cannot reach state 0 (note the 0 in position (2,0)). The chain has two communicating classes: {0,1} and {2}.

Matrix 2 IS irreducible because all entries are positive, meaning you can reach any state from any other state in exactly one step.

---

## Section 2: Transition Matrices (25 points)

### Question 4 (8 points)

What mathematical property must ALL transition matrices satisfy?

A) Symmetry: P_{ij} = P_{ji}
B) Row sums equal 1: Σ_j P_{ij} = 1
C) Determinant equals 1
D) All eigenvalues are real and positive

**Answer: B**

**Explanation:**
A transition matrix is a stochastic matrix where each row represents a probability distribution over the next states. Therefore, each row must sum to 1 (the probabilities of all possible next states must sum to 1).

- A is false: Most transition matrices are not symmetric
- C is false: The determinant can be any value
- D is false: While all eigenvalues have magnitude ≤ 1 and at least one equals 1, they need not all be real and positive

---

### Question 5 (10 points)

Given the transition matrix:
```
P = [0.7  0.3]
    [0.4  0.6]
```

Compute the stationary distribution π = [π_1, π_2] such that πP = π.

A) π = [0.5, 0.5]
B) π = [0.571, 0.429]
C) π = [0.4, 0.6]
D) π = [0.7, 0.3]

**Answer: B**

**Explanation:**
The stationary distribution satisfies πP = π and π_1 + π_2 = 1.

Setting up the equations:
```
[π_1, π_2][0.7  0.3] = [π_1, π_2]
          [0.4  0.6]
```

This gives:
- 0.7π_1 + 0.4π_2 = π_1  →  -0.3π_1 + 0.4π_2 = 0
- 0.3π_1 + 0.6π_2 = π_2  →  0.3π_1 - 0.4π_2 = 0

From the first equation: π_2 = (0.3/0.4)π_1 = 0.75π_1

Using π_1 + π_2 = 1:
- π_1 + 0.75π_1 = 1
- 1.75π_1 = 1
- π_1 = 4/7 ≈ 0.571

Therefore: π_2 = 3/7 ≈ 0.429

The stationary distribution is π = [4/7, 3/7] ≈ [0.571, 0.429]

---

### Question 6 (7 points)

For a finite, irreducible, aperiodic Markov chain, which statement is TRUE?

A) The stationary distribution always exists but may not be unique
B) The chain converges to the stationary distribution regardless of initial state
C) P^n approaches the zero matrix as n → ∞
D) The chain must return to the starting state with probability 1

**Answer: B**

**Explanation:**
For a finite, irreducible, aperiodic Markov chain:
- A unique stationary distribution π exists
- For any initial distribution μ, μP^n → π as n → ∞
- The chain converges to the stationary distribution from any starting state

Option A is false because the stationary distribution is unique for irreducible chains.
Option C is false; P^n approaches a matrix where each row equals π.
Option D is true but is a consequence of positive recurrence, not the most distinctive property.

---

## Section 3: Probability Review (20 points)

### Question 7 (6 points)

Given two discrete random variables X and Y with joint distribution P(X,Y), which equation correctly expresses the law of total probability for P(Y)?

A) P(Y) = Σ_x P(X|Y)
B) P(Y) = Σ_x P(X,Y)
C) P(Y) = Σ_x P(Y|X)P(X)
D) P(Y) = P(X)P(Y|X)

**Answer: C**

**Explanation:**
The law of total probability states that we can compute the marginal probability P(Y) by summing over all possible values of X:

P(Y) = Σ_x P(Y|X=x)P(X=x) = Σ_x P(X=x, Y)

This is fundamental to HMM inference where we sum over hidden states.

Option B is also correct (marginalization), but C explicitly shows the law of total probability form that's most relevant to HMMs.

---

### Question 8 (8 points)

Consider a system with hidden states {S_1, S_2} and observations {O_1, O_2, O_3}. If we observe O_1, which formula correctly computes P(S_1 | O_1)?

A) P(S_1 | O_1) = P(O_1 | S_1) / P(O_1)
B) P(S_1 | O_1) = P(S_1) P(O_1 | S_1) / P(O_1)
C) P(S_1 | O_1) = P(S_1) P(O_1)
D) P(S_1 | O_1) = P(O_1 | S_1) P(S_1)

**Answer: B**

**Explanation:**
This is Bayes' theorem:

P(S_1 | O_1) = P(O_1 | S_1) P(S_1) / P(O_1)

Where:
- P(O_1 | S_1) is the emission probability (likelihood)
- P(S_1) is the prior probability of state S_1
- P(O_1) is the marginal probability of observation O_1 (normalization constant)

This is the foundation of HMM inference algorithms.

---

### Question 9 (6 points)

The joint probability of a sequence of observations O_1, O_2, O_3 and hidden states S_1, S_2, S_3 in an HMM can be decomposed as:

A) P(O_{1:3}, S_{1:3}) = P(S_1)P(O_1|S_1)P(S_2)P(O_2|S_2)P(S_3)P(O_3|S_3)
B) P(O_{1:3}, S_{1:3}) = P(S_1)P(O_1|S_1)P(S_2|S_1)P(O_2|S_2)P(S_3|S_2)P(O_3|S_3)
C) P(O_{1:3}, S_{1:3}) = P(S_1|O_1)P(S_2|O_2)P(S_3|O_3)
D) P(O_{1:3}, S_{1:3}) = P(S_1)P(S_2|S_1)P(S_3|S_2)P(O_1)P(O_2)P(O_3)

**Answer: B**

**Explanation:**
Using the chain rule and Markov/HMM assumptions:

P(O_{1:3}, S_{1:3}) = P(S_1) × P(O_1|S_1) × P(S_2|S_1) × P(O_2|S_2) × P(S_3|S_2) × P(O_3|S_3)

This decomposes into:
- Initial state: P(S_1)
- Emissions: P(O_t | S_t) for each time step
- Transitions: P(S_{t+1} | S_t) for each transition

This factorization is fundamental to HMM algorithms.

---

## Section 4: Applied Concepts (25 points)

### Question 10 (10 points)

A financial analyst models market regimes as a 3-state Markov chain: {Bull, Neutral, Bear}. Historical data shows:
- Bull markets last an average of 10 quarters before transitioning
- When leaving Bull, the market goes to Neutral 75% of the time
- Neutral markets last an average of 5 quarters

What is the transition probability P(Bull → Bull)?

A) 0.80
B) 0.85
C) 0.90
D) 0.95

**Answer: C (0.90)**

**Explanation:**
If a state lasts an average of n periods, the self-transition probability is (n-1)/n, or equivalently, the exit probability is 1/n.

For Bull markets lasting average 10 quarters:
- P(leave Bull) = 1/10 = 0.10
- P(Bull → Bull) = 1 - 0.10 = 0.90

The information about Neutral states and transition destinations helps complete the full matrix but doesn't affect this calculation.

---

### Question 11 (8 points)

Consider the following transition matrix for a two-state volatility model:

```
          Low    High
Low     [0.95   0.05]
High    [0.10   0.90]
```

This matrix exhibits:

A) High persistence in both states
B) Rapid switching between states
C) Equal probability of being in each state
D) Non-stationarity

**Answer: A**

**Explanation:**
Both states have high self-transition probabilities (0.95 and 0.90), indicating persistence - once in a state, the system tends to stay there for multiple periods.

- Low volatility persists with 95% probability
- High volatility persists with 90% probability

This is characteristic of regime-switching models where states represent distinct market conditions that persist over time. The stationary distribution can be calculated as approximately [0.67, 0.33], showing the system spends more time in low volatility.

---

### Question 12 (7 points)

Why is the concept of a stationary distribution important for Hidden Markov Models in financial applications?

A) It guarantees profit in trading strategies
B) It represents long-run frequencies of being in each regime
C) It ensures all states are equally likely
D) It allows perfect prediction of future states

**Answer: B**

**Explanation:**
The stationary distribution π represents the long-run proportion of time spent in each state. For financial HMMs:

- If π_bull = 0.40, π_neutral = 0.35, π_bear = 0.25, the market spends approximately 40% of time in bull regimes over the long run
- This informs portfolio allocation and risk management
- It provides a baseline for regime probabilities when no recent information is available

Options A and D are false - HMMs don't guarantee profits or perfect prediction.
Option C is false - states typically have different long-run probabilities.

---

## Section 5: Mathematical Foundations (10 points)

### Question 13 (10 points)

Given the transition matrix:
```
P = [0.8  0.1  0.1]
    [0.2  0.6  0.2]
    [0.3  0.3  0.4]
```

Starting from state 0 with probability 1, compute the probability distribution after 1 step: π^(1) = π^(0)P.

**Answer:**

π^(0) = [1, 0, 0]

π^(1) = π^(0)P = [1, 0, 0] × [0.8  0.1  0.1] = [0.8, 0.1, 0.1]
                              [0.2  0.6  0.2]
                              [0.3  0.3  0.4]

**Result: π^(1) = [0.8, 0.1, 0.1]**

**Explanation:**
When starting with certainty in state 0 (π^(0) = [1, 0, 0]), the distribution after one step is simply the first row of the transition matrix. This is because:

π^(1)[j] = Σ_i π^(0)[i] × P[i,j] = 1 × P[0,j] = P[0,j]

So we have 80% probability of remaining in state 0, 10% of transitioning to state 1, and 10% of transitioning to state 2.

---

## Bonus Question (5 points)

### Question 14 (5 points)

Consider a periodic Markov chain where states alternate deterministically: 0 → 1 → 0 → 1 → ...

The transition matrix is:
```
P = [0  1]
    [1  0]
```

Does a stationary distribution exist? If so, what is it?

**Answer: Yes, π = [0.5, 0.5]**

**Explanation:**
Even though the chain is periodic (period = 2), a stationary distribution still exists. Setting πP = π:

[π_0, π_1][0  1] = [π_0, π_1]
          [1  0]

This gives:
- π_1 = π_0
- π_0 = π_1

Combined with π_0 + π_1 = 1, we get π_0 = π_1 = 0.5.

The stationary distribution represents the long-run proportion of time in each state (50% in each), but P^n does NOT converge to a matrix with identical rows due to periodicity. Instead:
- P^(even) = I (identity matrix)
- P^(odd) = P

This demonstrates that stationarity and convergence are distinct properties.

---

## Answer Key Summary

1. B (8 pts)
2. C - 0.19 (10 pts)
3. A, Matrix 2 (12 pts)
4. B (8 pts)
5. B - [0.571, 0.429] (10 pts)
6. B (7 pts)
7. C (6 pts)
8. B (8 pts)
9. B (6 pts)
10. C - 0.90 (10 pts)
11. A (8 pts)
12. B (7 pts)
13. [0.8, 0.1, 0.1] (10 pts)
14. π = [0.5, 0.5] (5 pts - Bonus)

**Total: 100 points (105 with bonus)**

---

## Grading Rubric

- **90-100 points:** Excellent - Strong grasp of Markov chain fundamentals
- **80-89 points:** Good - Solid understanding with minor gaps
- **70-79 points:** Satisfactory - Adequate knowledge, needs review of some concepts
- **60-69 points:** Needs Improvement - Review Markov property and stationary distributions
- **Below 60:** Incomplete Understanding - Revisit all module materials

---

## Learning Objectives Assessed

- [ ] Understand and apply the Markov property
- [ ] Compute multi-step transition probabilities
- [ ] Identify irreducible and aperiodic chains
- [ ] Calculate stationary distributions
- [ ] Apply Bayes' theorem to hidden state inference
- [ ] Decompose joint probabilities in sequential models
- [ ] Interpret transition matrices in financial contexts
