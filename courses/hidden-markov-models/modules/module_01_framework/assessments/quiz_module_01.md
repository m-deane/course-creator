# Quiz: Module 1 - Hidden Markov Model Framework

**Course:** Hidden Markov Models
**Module:** 1 - HMM Framework
**Total Points:** 100
**Estimated Time:** 25-30 minutes
**Attempts Allowed:** 2

---

## Instructions

This quiz assesses your understanding of HMM structure, components, parameters, and the three fundamental problems. Show all work for calculations. Select the best answer for multiple choice questions.

---

## Section 1: HMM Definition and Structure (25 points)

### Question 1 (8 points)

What distinguishes a Hidden Markov Model from a standard Markov Chain?

A) HMMs have more than two states
B) The states are not directly observable; we only observe emissions
C) HMMs require continuous observations
D) The transition probabilities change over time

**Answer: B**

**Explanation:**
The key distinction is that in an HMM, the underlying states are **hidden** (latent/unobserved). We only observe emissions that are probabilistically related to the hidden states.

In a standard Markov chain, we directly observe which state the system is in at each time step. In an HMM:
- Hidden layer: States S_t following Markov dynamics
- Observable layer: Observations O_t emitted from states

This two-layer structure is what makes HMMs powerful for modeling systems where the true state is uncertain or unmeasurable.

---

### Question 2 (10 points)

An HMM is fully specified by the parameter set λ = (A, B, π). Match each component to its definition:

**Components:**
1. A (Transition matrix)
2. B (Emission/Observation matrix)
3. π (Initial state distribution)

**Definitions:**
a) P(O_t | S_t) - probability of observing O_t when in state S_t
b) P(S_1) - probability of starting in each state
c) P(S_{t+1} | S_t) - probability of transitioning between states

**Correct Matching:**

1 - c: A contains transition probabilities P(S_{t+1} = j | S_t = i) = a_{ij}

2 - a: B contains emission probabilities P(O_t = v_k | S_t = j) = b_j(v_k)

3 - b: π contains initial state probabilities P(S_1 = i) = π_i

**Explanation:**
The complete parameter set λ = (A, B, π) consists of:

**A (Transition Matrix):** N×N matrix where a_{ij} = P(S_{t+1} = j | S_t = i)
- Rows must sum to 1 (stochastic matrix)
- Governs hidden state dynamics

**B (Emission Matrix):** N×M matrix where b_j(v_k) = P(O_t = v_k | S_t = j)
- For discrete observations: rows sum to 1
- For continuous: typically Gaussian densities

**π (Initial Distribution):** N-vector where π_i = P(S_1 = i)
- Must sum to 1
- Represents prior beliefs about starting state

---

### Question 3 (7 points)

Which independence assumption is central to HMMs?

A) Observations at different times are independent
B) Observations are independent given the hidden states
C) All hidden states are independent of each other
D) The emission probabilities are the same for all states

**Answer: B**

**Explanation:**
The key conditional independence assumption in HMMs is:

**P(O_t | S_{1:T}, O_{1:t-1}, O_{t+1:T}) = P(O_t | S_t)**

This means observations are conditionally independent given the hidden state sequence. Each observation O_t depends ONLY on the current hidden state S_t, not on:
- Other observations
- Past states
- Future states

This assumption allows efficient inference algorithms. Without it, we would need to model complex dependencies between all observations, making computation intractable.

Note: The hidden states themselves are NOT independent - they follow Markov dynamics where S_{t+1} depends on S_t.

---

## Section 2: The Three Fundamental Problems (30 points)

### Question 4 (12 points)

The three fundamental problems in HMM theory are:

**Problem 1 (Evaluation):** Given model λ and observations O_{1:T}, compute P(O_{1:T} | λ)

**Problem 2 (Decoding):** Given model λ and observations O_{1:T}, find the most likely state sequence S*_{1:T}

**Problem 3 (Learning):** Given observations O_{1:T}, find the model parameters λ* that maximize P(O_{1:T} | λ)

Match each problem to its primary algorithm:

a) Viterbi algorithm
b) Baum-Welch algorithm (EM)
c) Forward-Backward algorithm

**Correct Matching:**

Problem 1 (Evaluation) - c (Forward-Backward algorithm)
- Computes P(O_{1:T} | λ) efficiently using dynamic programming
- Also computes state probabilities P(S_t | O_{1:T}, λ)

Problem 2 (Decoding) - a (Viterbi algorithm)
- Finds most likely state sequence: argmax_{S_{1:T}} P(S_{1:T} | O_{1:T}, λ)
- Uses dynamic programming with max instead of sum

Problem 3 (Learning) - b (Baum-Welch algorithm)
- Expectation-Maximization algorithm
- Iteratively updates parameters to maximize likelihood
- Uses forward-backward in E-step

**Explanation:**
These three problems cover all major tasks in HMM applications:
- Evaluation: How well does the model explain the data?
- Decoding: What is the hidden state sequence?
- Learning: How do we estimate the model from data?

---

### Question 5 (10 points)

For a financial HMM with market regimes {Bull, Bear, Neutral} as hidden states and daily returns as observations, which problem are you solving when you ask:

"Given 20 days of returns, what is the most likely sequence of market regimes?"

A) Evaluation problem - computing likelihood
B) Decoding problem - finding state sequence
C) Learning problem - estimating parameters
D) Prediction problem - forecasting future states

**Answer: B**

**Explanation:**
This is the **decoding problem** (Problem 2). Given:
- Model parameters λ (transition probabilities, emission distributions, initial state)
- Observations O_{1:20} (the 20 days of returns)

Find:
- The most likely state sequence S*_{1:20}

This is solved using the **Viterbi algorithm**, which finds:

S*_{1:20} = argmax_{S_{1:20}} P(S_{1:20} | O_{1:20}, λ)

In financial applications, this tells us which regime the market was most likely in on each day, given all the observed returns. This is crucial for:
- Backtesting trading strategies
- Understanding market history
- Validating regime identification

---

### Question 6 (8 points)

Why is the naive approach to computing P(O_{1:T} | λ) intractable for large T?

The naive approach: P(O_{1:T} | λ) = Σ_{all S_{1:T}} P(O_{1:T}, S_{1:T} | λ)

A) The observations are continuous
B) The number of possible state sequences grows exponentially with T
C) The transition matrix is too large
D) The emission probabilities are unknown

**Answer: B**

**Explanation:**
For N hidden states and T time steps:
- Number of possible state sequences = N^T
- For N=3, T=100: 3^100 ≈ 5×10^47 sequences

Computing P(O_{1:T} | λ) by summing over all possible sequences requires:
- Evaluating N^T joint probabilities
- Computational complexity: O(N^T × T) - exponential!

**Example:**
- 3 states, 100 observations: 3^100 evaluations (impossible)
- 5 states, 50 observations: 5^50 evaluations (impossible)

The **forward algorithm** solves this efficiently using dynamic programming:
- Complexity: O(N^2 × T) - polynomial!
- For 3 states, 100 obs: 900 operations instead of 10^47

This is why dynamic programming algorithms are essential for HMMs.

---

## Section 3: HMM Parameters and Notation (20 points)

### Question 7 (10 points)

Consider an HMM with N=2 states and M=3 possible discrete observations.

Given:
```
A = [0.7  0.3]    (Transition matrix)
    [0.4  0.6]

B = [0.5  0.3  0.2]    (Emission matrix)
    [0.1  0.6  0.3]

π = [0.6, 0.4]    (Initial distribution)
```

Compute the probability of the observation sequence O = [o_1, o_2] = [v_1, v_3] and hidden state sequence S = [s_1, s_2] = [1, 2].

**Solution:**

P(O_{1:2}, S_{1:2} | λ) = π_{s_1} × b_{s_1}(o_1) × a_{s_1,s_2} × b_{s_2}(o_2)

Breaking it down:
- π_{s_1} = π_1 = 0.6 (start in state 1)
- b_{s_1}(o_1) = b_1(v_1) = 0.5 (emit v_1 from state 1)
- a_{s_1,s_2} = a_{1,2} = 0.3 (transition from state 1 to state 2)
- b_{s_2}(o_2) = b_2(v_3) = 0.3 (emit v_3 from state 2)

P(O_{1:2}, S_{1:2} | λ) = 0.6 × 0.5 × 0.3 × 0.3 = 0.027

**Answer: 0.027 or 2.7%**

**Explanation:**
The joint probability factorizes according to HMM structure:

P(O_{1:T}, S_{1:T} | λ) = π_{s_1} × ∏_{t=1}^{T} b_{s_t}(o_t) × ∏_{t=1}^{T-1} a_{s_t,s_{t+1}}

This factorization reflects:
- Initial state probability
- Emission at each time step
- Transition between consecutive states

---

### Question 8 (10 points)

For an HMM with N states, what are the constraints on the parameters?

**Part A (5 points):** Which constraints must the transition matrix A satisfy?

Select all that apply:
1. a_{ij} ≥ 0 for all i,j
2. Σ_j a_{ij} = 1 for all i
3. a_{ij} = a_{ji} (symmetry)
4. det(A) = 1

**Answer: 1 and 2**

**Explanation:**
The transition matrix must be a **stochastic matrix**:

1. **Non-negativity:** a_{ij} ≥ 0 (probabilities cannot be negative)
2. **Row sum:** Σ_j a_{ij} = 1 (from any state, must transition somewhere with total probability 1)

Symmetry (3) is NOT required - transitions can be asymmetric.
The determinant (4) is NOT constrained to equal 1.

**Part B (5 points):** For discrete observations with M possible values, what constraint applies to the emission matrix B?

A) Column sums equal 1
B) Row sums equal 1
C) All entries must be equal
D) Determinant must equal 1

**Answer: B**

**Explanation:**
For each state j, the emission probabilities over all possible observations must sum to 1:

Σ_k b_j(v_k) = 1 for all states j

This is because b_j represents a probability distribution over observations when in state j. When in a state, you must emit some observation with total probability 1.

Note: For continuous emissions (e.g., Gaussian), B represents parameters of probability density functions, not a matrix with sum constraints.

---

## Section 4: Financial Applications (15 points)

### Question 9 (8 points)

A hedge fund models market volatility using a 2-state HMM:
- Hidden states: {Low Vol, High Vol}
- Observations: Daily returns (continuous)

They estimate:
```
A = [0.95  0.05]
    [0.10  0.90]

Low Vol: μ = 0.001, σ = 0.01
High Vol: μ = 0.000, σ = 0.03
```

On a given day, the observed return is -4%. Which state is more likely to have generated this observation?

A) Low Vol (definitely)
B) High Vol (definitely)
C) Cannot determine without prior probabilities
D) Equally likely

**Answer: B (High Vol)**

**Explanation:**
We need to compare emission probabilities:

For a return of -0.04:

**Low Vol likelihood:**
z = (x - μ)/σ = (-0.04 - 0.001)/0.01 = -4.1
This is -4.1 standard deviations from the mean - extremely unlikely
P(x | Low Vol) ∝ exp(-4.1²/2) ≈ 10^-8

**High Vol likelihood:**
z = (x - μ)/σ = (-0.04 - 0.000)/0.03 = -1.33
This is -1.33 standard deviations from the mean - unusual but plausible
P(x | High Vol) ∝ exp(-1.33²/2) ≈ 0.38

The return of -4% is much more consistent with the high volatility regime. A -4% return is:
- 4.1 sigma event in low vol (probability ≈ 0.000002)
- 1.3 sigma event in high vol (probability ≈ 0.18)

While prior probabilities matter for the full posterior, the likelihood ratio is so extreme that High Vol is far more likely.

---

### Question 10 (7 points)

Why might an analyst prefer a 3-state HMM {Bull, Neutral, Bear} over a 2-state model {Bull, Bear} for equity regime detection?

A) Three states always have higher likelihood
B) It captures an intermediate regime with distinct return/volatility characteristics
C) Three states are required for the forward algorithm
D) It reduces computational complexity

**Answer: B**

**Explanation:**
Market behavior often exhibits three distinct regimes:

**Bull:**
- Positive mean returns
- Moderate volatility
- High state persistence

**Neutral/Sideways:**
- Near-zero mean returns
- Low to moderate volatility
- Can persist for extended periods

**Bear:**
- Negative mean returns
- High volatility
- Often shorter duration than bull markets

A 2-state model forces neutral periods to be classified as either bull or bear, potentially:
- Mixing different return distributions
- Generating false signals
- Missing regime transitions

The 3-state model better captures empirical market behavior, though it requires more data for reliable parameter estimation and adds complexity.

Model selection should use:
- Information criteria (AIC, BIC)
- Out-of-sample performance
- Interpretability of regimes

---

## Section 5: Conceptual Understanding (10 points)

### Question 11 (5 points)

In an HMM, why do we use the term "forward" for the forward algorithm?

A) It processes observations from past to future (time 1 to T)
B) It predicts future observations
C) It moves forward through the state space
D) It computes only future probabilities

**Answer: A**

**Explanation:**
The **forward algorithm** is called "forward" because it processes data in chronological order, computing probabilities from t=1 to t=T.

**Forward variable:** α_t(i) = P(O_1, O_2, ..., O_t, S_t = i | λ)
- Probability of partial observation sequence up to time t AND being in state i at time t
- Computed recursively: α_{t+1}(j) depends on α_t(i)
- Processes data forward in time

**Backward algorithm:** β_t(i) = P(O_{t+1}, O_{t+2}, ..., O_T | S_t = i, λ)
- Probability of future observations given current state
- Computed recursively backward: β_t(i) depends on β_{t+1}(j)
- Processes data backward in time

Together, they enable efficient computation of P(S_t | O_{1:T}, λ) at all time points.

---

### Question 12 (5 points)

What is the primary advantage of using the Viterbi algorithm over computing P(S_t | O_{1:T}) independently at each time step t?

A) Viterbi is faster to compute
B) Viterbi finds a globally optimal state sequence; independent marginals may be inconsistent
C) Viterbi works with continuous observations
D) Viterbi doesn't require the transition matrix

**Answer: B**

**Explanation:**
Computing marginal state probabilities P(S_t | O_{1:T}) at each time step independently can yield inconsistent results where the "most likely" state at each time point forms an impossible sequence.

**Example:**
Consider impossible transitions with marginals:
- P(S_1=Bull | O_{1:T}) = 0.6 (most likely at t=1)
- P(S_2=Bear | O_{1:T}) = 0.7 (most likely at t=2)

But if P(Bull → Bear) = 0 in the transition matrix, this sequence is impossible!

**Viterbi algorithm** solves:
S*_{1:T} = argmax_{S_{1:T}} P(S_{1:T} | O_{1:T}, λ)

This finds the **globally most likely path** that respects transition constraints. The resulting sequence:
- Is always valid (non-zero probability)
- Maximizes joint probability
- Respects state dynamics

This is crucial for regime detection where state persistence matters.

---

## Bonus Section: Advanced Concepts (10 points)

### Question 13 (5 points)

For an ergodic HMM (irreducible and aperiodic), what happens to P(S_t = i | λ) as t → ∞?

A) It approaches 0
B) It approaches 1/N (uniform)
C) It approaches the stationary distribution π_i^∞
D) It becomes undefined

**Answer: C**

**Explanation:**
For an ergodic HMM, the hidden state distribution converges to the stationary distribution of the transition matrix A:

lim_{t→∞} P(S_t = i | λ) = π_i^∞

where π^∞ satisfies π^∞ A = π^∞.

This means:
- Long-run state probabilities are independent of initial conditions
- The model has a stable equilibrium distribution
- π^∞ can be used to initialize the model for long sequences

In financial applications:
- π^∞ represents long-run regime frequencies
- Useful for portfolio allocation in absence of recent data
- Provides model validation: do empirical frequencies match π^∞?

---

### Question 14 (5 points)

An HMM has the property that P(O_{1:T} | λ) can be computed in O(N²T) time using the forward algorithm. If we instead used the naive summation over all state sequences, what would the computational complexity be?

A) O(NT)
B) O(N²T)
C) O(N^T)
D) O(T^N)

**Answer: C**

**Explanation:**
Naive approach:

P(O_{1:T} | λ) = Σ_{all sequences S_{1:T}} P(O_{1:T}, S_{1:T} | λ)

- Number of sequences: N^T (N choices at each of T time steps)
- Cost per sequence: O(T) (compute T emissions and T-1 transitions)
- Total complexity: O(N^T × T) ≈ O(N^T)

**Forward algorithm complexity:** O(N²T)
- At each time step t: compute α_t(j) for all j
- For each j: sum over all i (N operations)
- Cost per time step: O(N²)
- Total: O(N²T)

**Efficiency gain:**

For N=3, T=100:
- Naive: 3^100 ≈ 10^47 operations (impossible)
- Forward: 3² × 100 = 900 operations (milliseconds)

This exponential-to-polynomial reduction is why HMMs are practical for real-world applications.

---

## Answer Key Summary

1. B (8 pts)
2. 1-c, 2-a, 3-b (10 pts)
3. B (7 pts)
4. Problem 1-c, Problem 2-a, Problem 3-b (12 pts)
5. B (10 pts)
6. B (8 pts)
7. 0.027 (10 pts)
8. Part A: 1,2; Part B: B (10 pts)
9. B - High Vol (8 pts)
10. B (7 pts)
11. A (5 pts)
12. B (5 pts)
13. C (5 pts - Bonus)
14. C (5 pts - Bonus)

**Total: 100 points (110 with bonus)**

---

## Grading Rubric

- **90-100 points:** Excellent - Strong understanding of HMM framework
- **80-89 points:** Good - Solid grasp with minor conceptual gaps
- **70-79 points:** Satisfactory - Adequate knowledge, review parameters and problems
- **60-69 points:** Needs Improvement - Review HMM structure and the three fundamental problems
- **Below 60:** Incomplete Understanding - Revisit all module materials

---

## Learning Objectives Assessed

- [ ] Define HMM components (A, B, π) and their constraints
- [ ] Distinguish HMMs from standard Markov chains
- [ ] Identify the three fundamental problems and their algorithms
- [ ] Compute joint probabilities P(O_{1:T}, S_{1:T} | λ)
- [ ] Apply HMM concepts to financial regime detection
- [ ] Understand computational complexity of HMM algorithms
- [ ] Interpret emission and transition probabilities in applications
