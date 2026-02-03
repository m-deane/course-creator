# Quiz: Module 2 - HMM Algorithms

**Course:** Hidden Markov Models
**Module:** 2 - Core Algorithms (Forward-Backward, Viterbi, Baum-Welch)
**Total Points:** 100
**Estimated Time:** 30-35 minutes
**Attempts Allowed:** 2

---

## Instructions

This quiz assesses your understanding of the three core HMM algorithms: Forward-Backward (evaluation), Viterbi (decoding), and Baum-Welch (learning). Many questions require calculations - show all work for partial credit.

---

## Section 1: Forward Algorithm (20 points)

### Question 1 (10 points)

Consider an HMM with N=2 states {s_1, s_2} and observations {v_1, v_2}.

Parameters:
```
A = [0.7  0.3]
    [0.4  0.6]

B = [0.9  0.1]
    [0.2  0.8]

π = [0.6, 0.4]
```

Compute the forward variable α_2(s_2) for the observation sequence O = [v_1, v_1].

**Solution:**

**Step 1: Initialize α_1**

α_1(s_1) = π_1 × b_1(v_1) = 0.6 × 0.9 = 0.54
α_1(s_2) = π_2 × b_2(v_1) = 0.4 × 0.2 = 0.08

**Step 2: Recursion for α_2(s_2)**

α_2(s_2) = b_2(v_1) × [α_1(s_1) × a_{1,2} + α_1(s_2) × a_{2,2}]

α_2(s_2) = 0.2 × [0.54 × 0.3 + 0.08 × 0.6]
         = 0.2 × [0.162 + 0.048]
         = 0.2 × 0.210
         = 0.042

**Answer: α_2(s_2) = 0.042**

**Explanation:**
The forward variable α_t(j) represents:
- P(O_1, ..., O_t, S_t = j | λ)
- Probability of observing the sequence up to time t AND being in state j at time t

The recursion:
α_{t+1}(j) = b_j(o_{t+1}) × Σ_i α_t(i) × a_{ij}

This sums over all possible previous states, weighted by transition probabilities.

---

### Question 2 (10 points)

Using the forward variables from Question 1, compute P(O_{1:2} | λ) for the sequence O = [v_1, v_1].

**Solution:**

First, compute α_2(s_1):

α_2(s_1) = b_1(v_1) × [α_1(s_1) × a_{1,1} + α_1(s_2) × a_{2,1}]
         = 0.9 × [0.54 × 0.7 + 0.08 × 0.4]
         = 0.9 × [0.378 + 0.032]
         = 0.9 × 0.410
         = 0.369

Now sum over all final states:

P(O_{1:2} | λ) = Σ_j α_2(j) = α_2(s_1) + α_2(s_2)
                = 0.369 + 0.042
                = 0.411

**Answer: P(O_{1:2} | λ) = 0.411 or 41.1%**

**Explanation:**
The total observation likelihood is the sum of forward variables at the final time step:

P(O_{1:T} | λ) = Σ_j α_T(j)

This works because α_T(j) gives the probability of the observation sequence AND ending in state j, so summing over all final states gives the total probability of the observation sequence.

This is the solution to the **evaluation problem** (Problem 1).

---

## Section 2: Backward Algorithm (15 points)

### Question 3 (8 points)

What does the backward variable β_t(i) represent?

A) P(O_1, ..., O_t | S_t = i, λ)
B) P(O_{t+1}, ..., O_T | S_t = i, λ)
C) P(S_t = i | O_{1:T}, λ)
D) P(O_{1:T} | λ)

**Answer: B**

**Explanation:**
The backward variable β_t(i) represents:
- P(O_{t+1}, O_{t+2}, ..., O_T | S_t = i, λ)
- Probability of observing the **future** observations given that we are in state i at time t

Key properties:
- Computed recursively **backward** from t=T to t=1
- β_T(i) = 1 for all i (no future observations)
- Recursion: β_t(i) = Σ_j a_{ij} × b_j(o_{t+1}) × β_{t+1}(j)

Combined with forward variables:
α_t(i) × β_t(i) ∝ P(S_t = i, O_{1:T} | λ) ∝ P(S_t = i | O_{1:T}, λ)

This is how we compute the smoothing distribution.

---

### Question 4 (7 points)

For the HMM in Question 1 with observation sequence O = [v_1, v_1], compute β_1(s_1).

**Solution:**

**Step 1: Initialize β_2**

β_2(s_1) = 1
β_2(s_2) = 1

**Step 2: Recursion for β_1(s_1)**

β_1(s_1) = Σ_j a_{1,j} × b_j(o_2) × β_2(j)
         = a_{1,1} × b_1(v_1) × β_2(s_1) + a_{1,2} × b_2(v_1) × β_2(s_2)
         = 0.7 × 0.9 × 1 + 0.3 × 0.2 × 1
         = 0.63 + 0.06
         = 0.69

**Answer: β_1(s_1) = 0.69**

**Explanation:**
The backward recursion moves from the end of the sequence toward the beginning:

β_t(i) = Σ_j a_{ij} × b_j(o_{t+1}) × β_{t+1}(j)

This sums over all possible next states j, weighting by:
- Transition probability from i to j
- Emission probability of observing o_{t+1} in state j
- Backward variable from the next time step

---

## Section 3: Viterbi Algorithm (25 points)

### Question 5 (12 points)

For the HMM in Question 1 with observation sequence O = [v_1, v_1], find the most likely state sequence using the Viterbi algorithm.

**Solution:**

**Step 1: Initialize δ_1**

δ_1(s_1) = π_1 × b_1(v_1) = 0.6 × 0.9 = 0.54
δ_1(s_2) = π_2 × b_2(v_1) = 0.4 × 0.2 = 0.08

ψ_1(s_1) = 0
ψ_1(s_2) = 0

**Step 2: Recursion for δ_2**

For state s_1:
δ_2(s_1) = b_1(v_1) × max[δ_1(s_1) × a_{1,1}, δ_1(s_2) × a_{2,1}]
         = 0.9 × max[0.54 × 0.7, 0.08 × 0.4]
         = 0.9 × max[0.378, 0.032]
         = 0.9 × 0.378
         = 0.3402

ψ_2(s_1) = argmax[0.54 × 0.7, 0.08 × 0.4] = s_1

For state s_2:
δ_2(s_2) = b_2(v_1) × max[δ_1(s_1) × a_{1,2}, δ_1(s_2) × a_{2,2}]
         = 0.2 × max[0.54 × 0.3, 0.08 × 0.6]
         = 0.2 × max[0.162, 0.048]
         = 0.2 × 0.162
         = 0.0324

ψ_2(s_2) = argmax[0.54 × 0.3, 0.08 × 0.6] = s_1

**Step 3: Termination and backtracking**

Best final state: S*_2 = argmax[δ_2(s_1), δ_2(s_2)] = argmax[0.3402, 0.0324] = s_1

Backtrack: S*_1 = ψ_2(s_1) = s_1

**Answer: Most likely state sequence = [s_1, s_1]**

Maximum probability: P* = 0.3402

**Explanation:**
The Viterbi algorithm finds the single most likely state sequence by:
1. Computing max probability of reaching each state at each time
2. Recording which previous state achieved this maximum
3. Backtracking from the best final state

Key difference from forward algorithm: MAX instead of SUM.

---

### Question 6 (8 points)

What is the key difference between the forward algorithm and the Viterbi algorithm?

A) Forward uses multiplication, Viterbi uses addition
B) Forward sums over all paths, Viterbi finds the maximum path
C) Forward is faster to compute
D) Viterbi works only for discrete observations

**Answer: B**

**Explanation:**

**Forward Algorithm:**
- Sums over all possible state sequences
- α_{t+1}(j) = b_j(o_{t+1}) × **Σ_i** α_t(i) × a_{ij}
- Computes P(O_{1:T} | λ) = marginal likelihood
- Considers all paths, weighted by probability

**Viterbi Algorithm:**
- Finds the single best state sequence
- δ_{t+1}(j) = b_j(o_{t+1}) × **max_i** δ_t(i) × a_{ij}
- Computes max_{S_{1:T}} P(S_{1:T}, O_{1:T} | λ)
- Selects only the maximum probability path

**Analogy:**
- Forward: "What is the average height of all paths through a maze?"
- Viterbi: "What is the height of the tallest path through the maze?"

Both have O(N²T) complexity, but solve different problems.

---

### Question 7 (5 points)

In financial applications, why might the Viterbi state sequence differ from the sequence of individually most likely states at each time point?

A) The Viterbi algorithm has numerical errors
B) Viterbi enforces transition probability constraints
C) Individual marginals ignore model parameters
D) Viterbi is only an approximation

**Answer: B**

**Explanation:**

**Individual marginals:** max_i P(S_t = i | O_{1:T}, λ) at each t independently

This can produce impossible sequences:
- Time 1: Bull most likely (70%)
- Time 2: Bear most likely (65%)
- But if P(Bull → Bear) = 0.01, this transition is extremely unlikely

**Viterbi sequence:** max_{S_{1:T}} P(S_{1:T} | O_{1:T}, λ)

This finds the globally most probable sequence that:
- Respects transition probabilities
- May have lower marginal probabilities at individual time points
- Guarantees a valid, coherent path

**Example:**
Viterbi might give: [Bull, Neutral, Bear] (respects typical transitions)
Instead of: [Bull, Bear, Bull] (marginals high but transitions unlikely)

This is crucial for regime detection where state persistence matters.

---

## Section 4: Baum-Welch Algorithm (25 points)

### Question 8 (10 points)

The Baum-Welch algorithm is an instance of which general optimization framework?

A) Gradient Descent
B) Expectation-Maximization (EM)
C) Newton's Method
D) Simulated Annealing

**Answer: B (Expectation-Maximization)**

**Part B:** Describe the two steps of Baum-Welch:

**E-Step (Expectation):**
Compute expected sufficient statistics using current parameters λ^{old}:
- γ_t(i) = P(S_t = i | O_{1:T}, λ^{old})
- ξ_t(i,j) = P(S_t = i, S_{t+1} = j | O_{1:T}, λ^{old})

These are computed using forward-backward variables:
γ_t(i) = α_t(i) × β_t(i) / P(O_{1:T} | λ^{old})
ξ_t(i,j) = α_t(i) × a_{ij} × b_j(o_{t+1}) × β_{t+1}(j) / P(O_{1:T} | λ^{old})

**M-Step (Maximization):**
Update parameters to maximize expected log-likelihood:

π_i^{new} = γ_1(i)

a_{ij}^{new} = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
(expected transitions from i to j / expected time in i)

b_i(v_k)^{new} = Σ_{t: o_t=v_k} γ_t(i) / Σ_t γ_t(i)
(expected emissions of v_k in state i / expected time in i)

**Explanation:**
Baum-Welch guarantees:
- Monotonic improvement: P(O | λ^{new}) ≥ P(O | λ^{old})
- Convergence to local maximum
- Requires multiple sequences for good estimates

---

### Question 9 (8 points)

What does γ_t(i) represent in the Baum-Welch algorithm?

A) P(S_t = i | λ)
B) P(O_t | S_t = i, λ)
C) P(S_t = i | O_{1:T}, λ)
D) P(O_{1:T} | S_t = i, λ)

**Answer: C**

**Explanation:**
γ_t(i) is the **smoothing distribution**:

γ_t(i) = P(S_t = i | O_{1:T}, λ)

This represents the probability of being in state i at time t, given:
- All observations (past, present, future)
- Current model parameters λ

**Computation using forward-backward:**

γ_t(i) = α_t(i) × β_t(i) / P(O_{1:T} | λ)

Where:
- α_t(i): probability of past observations and being in state i
- β_t(i): probability of future observations given state i
- Denominator: normalization constant

**In Baum-Welch:**
- γ_t(i) is the expected "weight" of state i at time t
- Used to compute expected state occupancy counts
- Drives parameter re-estimation

**Types of inference:**
- **Filtering:** P(S_t | O_{1:t}, λ) (only past observations)
- **Smoothing:** P(S_t | O_{1:T}, λ) (all observations) ← γ_t(i)
- **Prediction:** P(S_t | O_{1:t-1}, λ) (before observing O_t)

---

### Question 10 (7 points)

In the M-step of Baum-Welch, the transition probability update is:

a_{ij}^{new} = Σ_{t=1}^{T-1} ξ_t(i,j) / Σ_{t=1}^{T-1} γ_t(i)

Interpret this formula in plain English.

**Answer:**

**Numerator: Σ_{t=1}^{T-1} ξ_t(i,j)**
- Expected number of transitions from state i to state j
- Sum over all time steps of P(S_t = i, S_{t+1} = j | O, λ)

**Denominator: Σ_{t=1}^{T-1} γ_t(i)**
- Expected number of transitions out of state i
- Sum over all time steps of P(S_t = i | O, λ)

**Ratio:**
- Expected transitions from i to j / Expected transitions from i
- Fraction of times we leave state i and go to state j
- Maximum likelihood estimate based on expected counts

**Intuition:**
"Of all the expected times we were in state i and needed to transition somewhere, what fraction of the time did we go to state j?"

This is exactly how you would estimate transition probabilities from observed data, but using soft counts (expectations) instead of hard counts.

---

## Section 5: Algorithm Comparison (15 points)

### Question 11 (7 points)

Match each algorithm to its computational complexity and purpose:

**Algorithms:**
1. Forward algorithm
2. Viterbi algorithm
3. Baum-Welch (one iteration)

**Complexity:**
a) O(N²T)
b) O(N²T²)
c) O(KN²T) where K = number of iterations

**Purpose:**
i) Compute P(O_{1:T} | λ)
ii) Find argmax_{S_{1:T}} P(S_{1:T} | O_{1:T}, λ)
iii) Find argmax_λ P(O_{1:T} | λ)

**Correct Matching:**

1. Forward: Complexity = a (O(N²T)), Purpose = i (likelihood)
2. Viterbi: Complexity = a (O(N²T)), Purpose = ii (best path)
3. Baum-Welch: Complexity = c (O(KN²T)), Purpose = iii (learn parameters)

**Explanation:**
All three core algorithms use dynamic programming with O(N²T) per pass:
- N² states at each time step (previous × current)
- T time steps

Baum-Welch requires:
- Forward pass: O(N²T)
- Backward pass: O(N²T)
- Compute γ and ξ: O(N²T)
- Update parameters: O(N²T)
- K iterations until convergence

Total: O(KN²T)

---

### Question 12 (8 points)

A financial analyst has 500 days of stock returns and wants to fit a 3-state HMM. Which statements are TRUE?

A) They should run forward algorithm to estimate parameters
B) They should use Baum-Welch to estimate A, B, π from data
C) They should use Viterbi to find the most likely regime on each day after fitting
D) They need multiple sequences to reliably estimate parameters

**Select all that apply: B, C, D**

**Explanation:**

**A - FALSE:** Forward algorithm computes likelihood P(O | λ) given known parameters. It doesn't estimate parameters.

**B - TRUE:** Baum-Welch (EM algorithm) is the standard method for estimating HMM parameters from observed data:
- Input: observation sequence(s)
- Output: estimated λ = (A, B, π)

**C - TRUE:** After fitting the model with Baum-Welch, Viterbi can be used to decode the most likely regime sequence:
- Helps interpret which regime the market was in historically
- Useful for backtesting regime-based strategies

**D - TRUE (in practice):** While Baum-Welch can work with a single sequence:
- Multiple sequences provide more reliable estimates
- Reduces overfitting to a single realization
- For financial data: could use multiple assets, or split long sequence into sub-sequences
- 500 days is marginal for 3 states with continuous emissions

**Better approach:**
- Use multiple stocks or multiple time periods
- Apply regularization or informative priors (Bayesian HMM)
- Validate on out-of-sample data

---

## Section 6: Applied Problem Solving (15 points)

### Question 13 (15 points)

A quant researcher models VIX (volatility index) states with a 2-state HMM:

```
States: {Low Vol (s_1), High Vol (s_2)}

Estimated parameters:
A = [0.98  0.02]
    [0.05  0.95]

Emission: Gaussian
Low Vol: N(μ=15, σ=3)
High Vol: N(μ=30, σ=8)

π = [0.71, 0.29] (stationary distribution)
```

Given the VIX observations: [16, 18, 28, 32, 30]

**Part A (7 points):** Qualitatively, what regime sequence would you expect Viterbi to find? Explain your reasoning without full calculation.

**Expected Answer:**

Likely sequence: [s_1, s_1, s_2, s_2, s_2]

**Reasoning:**

**Days 1-2 (VIX = 16, 18):**
- Both close to Low Vol mean (15) and far from High Vol mean (30)
- Within 1 standard deviation of Low Vol
- About 1.5 standard deviations below High Vol
- Strong evidence for Low Vol state

**Day 3 (VIX = 28):**
- Close to High Vol mean (30)
- Far from Low Vol mean (15) - about 4.3 standard deviations above
- Likely transition to High Vol
- High self-transition probabilities (0.95) make regime switches rare but persistent

**Days 4-5 (VIX = 32, 30):**
- Very close to High Vol mean
- Far from Low Vol mean
- High Vol state very persistent (0.95 stay probability)
- Clear High Vol regime

**Transition timing:**
The jump from 18 → 28 is the most likely transition point because:
- Large shift in observations
- High state persistence means once we transition, we stay
- Transition probability (0.02) is low, so we transition when evidence is overwhelming

**Part B (8 points):** After observing VIX = 35 on day 6, the forward algorithm gives:

α_6(s_1) = 0.0001
α_6(s_2) = 0.3200

Compute:
1. P(O_{1:6} | λ)
2. P(S_6 = s_2 | O_{1:6}, λ) (filtering distribution)

**Solution:**

**1. Observation likelihood:**

P(O_{1:6} | λ) = Σ_i α_6(i) = α_6(s_1) + α_6(s_2)
                = 0.0001 + 0.3200
                = 0.3201

**2. Filtering distribution:**

P(S_6 = s_2 | O_{1:6}, λ) = α_6(s_2) / P(O_{1:6} | λ)
                            = 0.3200 / 0.3201
                            = 0.9997
                            ≈ 99.97%

**Answers:**
1. P(O_{1:6} | λ) = 0.3201
2. P(S_6 = s_2 | O_{1:6}, λ) ≈ 0.9997 or 99.97%

**Explanation:**
VIX = 35 is:
- Far from Low Vol mean (15): z = (35-15)/3 = 6.7 sigma (probability ≈ 0)
- Close to High Vol mean (30): z = (35-30)/8 = 0.625 sigma (probability ≈ 0.53)

Combined with the preceding high VIX values, we are almost certain (99.97%) to be in the High Vol state. The filtering distribution shows extreme confidence due to:
- Strong emission probability favoring High Vol
- Previous observations consistent with High Vol
- High state persistence

---

## Bonus Question (5 points)

### Question 14 (5 points)

Why does the Baum-Welch algorithm only guarantee convergence to a local (not global) maximum?

**Answer:**

The log-likelihood function L(λ) = log P(O_{1:T} | λ) is **not convex** in the parameters λ = (A, B, π).

**Reasons for non-convexity:**

1. **Label switching:** Multiple parameter configurations can have identical likelihood
   - Swapping all labels (state 1 ↔ state 2) gives same likelihood
   - Creates symmetric local maxima

2. **Mixture models:** HMMs are mixture models with hidden variables
   - Likelihood surface has multiple modes
   - Different state interpretations can fit data similarly

3. **EM property:** EM guarantees:
   - Monotonic improvement: L(λ^{t+1}) ≥ L(λ^t)
   - Convergence to a stationary point
   - But NOT convergence to global maximum

**Practical implications:**

1. **Multiple restarts:** Run Baum-Welch with different random initializations
2. **Best result:** Select λ with highest final likelihood
3. **Informative initialization:** Use domain knowledge or k-means on observations
4. **Validation:** Check if learned states make sense (interpretability)

**Example:**
For 3-state market regime model:
- One run might converge to: {Bull, Neutral, Bear}
- Another run might find: {Bull, Mild Bull, Extreme Bull}
- Both are local maxima, but one may be more interpretable/useful

This is why careful initialization and validation are crucial in HMM applications.

---

## Answer Key Summary

1. α_2(s_2) = 0.042 (10 pts)
2. P(O_{1:2} | λ) = 0.411 (10 pts)
3. B (8 pts)
4. β_1(s_1) = 0.69 (7 pts)
5. [s_1, s_1], P* = 0.3402 (12 pts)
6. B (8 pts)
7. B (5 pts)
8. B, plus description of E-step and M-step (10 pts)
9. C (8 pts)
10. Interpretation of ratio formula (7 pts)
11. All correct matchings (7 pts)
12. B, C, D (8 pts)
13. Part A: [s_1, s_1, s_2, s_2, s_2] with reasoning (7 pts)
    Part B: P(O)=0.3201, P(S_6=s_2|O)≈0.9997 (8 pts)
14. Non-convexity explanation (5 pts - Bonus)

**Total: 100 points (105 with bonus)**

---

## Grading Rubric

- **90-100 points:** Excellent - Mastery of all three core algorithms
- **80-89 points:** Good - Strong understanding with minor computational errors
- **70-79 points:** Satisfactory - Adequate knowledge, review forward-backward
- **60-69 points:** Needs Improvement - Review algorithm mechanics and applications
- **Below 60:** Incomplete Understanding - Revisit all algorithms with worked examples

---

## Learning Objectives Assessed

- [ ] Compute forward variables α_t(i) using recursion
- [ ] Compute backward variables β_t(i) using recursion
- [ ] Calculate observation likelihood P(O_{1:T} | λ)
- [ ] Apply Viterbi algorithm to find most likely state sequence
- [ ] Distinguish forward (sum) from Viterbi (max) operations
- [ ] Understand E-step and M-step of Baum-Welch
- [ ] Interpret γ_t(i) and ξ_t(i,j) in parameter estimation
- [ ] Apply algorithms to financial regime detection problems
- [ ] Analyze computational complexity of HMM algorithms
