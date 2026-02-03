# Module 1 Quiz: GA Fundamentals

**Course:** Genetic Algorithms for Feature Selection
**Module:** 1 - GA Fundamentals
**Total Points:** 100
**Estimated Time:** 25-30 minutes
**Attempts Allowed:** 2

## Instructions

This quiz assesses your understanding of genetic algorithm components: chromosome encoding, selection operators, crossover, mutation, and the basic GA workflow. Code-based questions require understanding of implementation details.

---

## Section 1: Chromosome Encoding (20 points)

### Question 1 (8 points)

Given the following feature set and binary chromosome:

```
Features:    [return_1d, return_5d, volume, volatility, rsi, macd, sma_20]
Chromosome:  [   1,         0,         1,       0,        1,    0,     1   ]
```

Write Python code to extract the selected feature names from the full feature list:

```python
features = ['return_1d', 'return_5d', 'volume', 'volatility', 'rsi', 'macd', 'sma_20']
chromosome = [1, 0, 1, 0, 1, 0, 1]

selected_features = # YOUR CODE HERE
```

**Answer:**

```python
selected_features = ___________________________________________________________
```

---

### Question 2 (12 points)

Consider two chromosome representations for the same feature selection problem:

**Representation A (Binary):**
```python
chromosome = [1, 0, 1, 1, 0, 0, 1]  # 7 bits for 7 features
```

**Representation B (Integer list):**
```python
chromosome = [0, 2, 3, 6]  # List of selected feature indices
```

**Part A (6 points):** What are the advantages and disadvantages of each representation for genetic algorithms? Name at least one of each.

**Binary Advantages:**

___________________________________________________________________________

**Binary Disadvantages:**

___________________________________________________________________________

**Integer List Advantages:**

___________________________________________________________________________

**Integer List Disadvantages:**

___________________________________________________________________________

**Part B (6 points):** Which representation is standard in DEAP and most GA frameworks for feature selection? Why?

**Answer:** ___________

**Reasoning:**

___________________________________________________________________________

___________________________________________________________________________

---

## Section 2: Selection Operators (25 points)

### Question 3 (10 points)

Match each selection operator with its selection mechanism:

| Operator | Mechanism |
|----------|-----------|
| 1. Tournament Selection | A. Probability proportional to fitness rank |
| 2. Roulette Wheel | B. Choose best from k random individuals |
| 3. Rank Selection | C. Probability proportional to fitness value |
| 4. Elitism | D. Deterministically keep top n individuals |

**Answers:**
- Tournament: ___________
- Roulette Wheel: ___________
- Rank Selection: ___________
- Elitism: ___________

---

### Question 4 (10 points)

Consider a population of 6 individuals with the following fitness values (higher is better):

| Individual | Chromosome | Fitness |
|------------|------------|---------|
| A | [1,0,1,0,1] | 0.85 |
| B | [0,1,1,0,0] | 0.72 |
| C | [1,1,0,1,0] | 0.91 |
| D | [0,0,1,1,1] | 0.68 |
| E | [1,0,0,1,1] | 0.79 |
| F | [0,1,0,0,1] | 0.55 |

**Part A (5 points):** Using tournament selection with k=3, you randomly draw individuals B, D, and F. Which individual is selected? What is the probability of this individual being selected from this tournament?

**Selected Individual:** ___________

**Selection Probability:** ___________

**Part B (5 points):** If using elitism with n=2, which two individuals are guaranteed to survive to the next generation?

**Individuals:** ___________ and ___________

---

### Question 5 (5 points)

True or False with justification:

**Statement:** Increasing tournament size (k) in tournament selection increases selection pressure, making the algorithm more likely to converge quickly but potentially to local optima.

**Answer (T/F):** ___________

**Justification (2 sentences):**

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

---

## Section 3: Crossover Operators (25 points)

### Question 6 (12 points)

Perform single-point crossover at position 3 (after the 3rd bit):

```
Parent 1: [1, 0, 1, | 1, 0, 1, 0]
Parent 2: [0, 1, 0, | 0, 1, 0, 1]
```

**Child 1:** [ ___, ___, ___, ___, ___, ___, ___ ]

**Child 2:** [ ___, ___, ___, ___, ___, ___, ___ ]

---

### Question 7 (13 points)

Consider the following uniform crossover implementation:

```python
import random

def uniform_crossover(parent1, parent2, prob=0.5):
    child1 = []
    child2 = []

    for bit1, bit2 in zip(parent1, parent2):
        if random.random() < prob:
            child1.append(bit1)
            child2.append(bit2)
        else:
            child1.append(bit2)
            child2.append(bit1)

    return child1, child2
```

**Part A (5 points):** Given parents `[1,1,0,0]` and `[0,0,1,1]`, if the random values generated are `[0.3, 0.6, 0.4, 0.7]`, what are the two children?

**Child 1:** [ ___, ___, ___, ___ ]

**Child 2:** [ ___, ___, ___, ___ ]

**Part B (8 points):** What is the expected number of bits that will come from Parent 1 in Child 1 for a chromosome of length n? Explain your reasoning.

**Expected Number:** ___________

**Explanation:**

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

---

## Section 4: Mutation Operators (20 points)

### Question 8 (8 points)

For binary chromosomes in feature selection, what is the typical recommended range for mutation probability per bit?

A) 0.5 to 1.0 (high mutation)
B) 0.1 to 0.3 (medium mutation)
C) 1/n to 0.01 (low mutation, where n is chromosome length)
D) Mutation is not used in feature selection GAs

**Answer:** ___________

**Explanation (why this range?):**

___________________________________________________________________________

___________________________________________________________________________

---

### Question 9 (12 points)

Analyze the following mutation operator:

```python
def mutate(chromosome, indpb=0.05):
    """Flip each bit independently with probability indpb."""
    for i in range(len(chromosome)):
        if random.random() < indpb:
            chromosome[i] = 1 - chromosome[i]
    return chromosome,  # Note: returns tuple
```

**Part A (4 points):** Apply this mutation to chromosome `[1, 0, 1, 1, 0]` with `indpb=0.05`. If the random values are `[0.02, 0.08, 0.03, 0.12, 0.01]`, what is the mutated chromosome?

**Mutated Chromosome:** [ ___, ___, ___, ___, ___ ]

**Part B (4 points):** For a chromosome of length 20 with `indpb=0.05`, what is the expected number of bits that will be flipped?

**Expected Number:** ___________

**Calculation:**

___________________________________________________________________________

**Part C (4 points):** Why does this function return a tuple `(chromosome,)` instead of just `chromosome`?

**Answer:**

___________________________________________________________________________

___________________________________________________________________________

---

## Section 5: GA Workflow (10 points)

### Question 10 (10 points)

Order the following steps in a standard genetic algorithm iteration (1 = first, 6 = last):

- _____ Apply crossover to selected parents
- _____ Evaluate fitness of offspring
- _____ Select parents using selection operator
- _____ Apply mutation to offspring
- _____ Replace old population with new population
- _____ Evaluate fitness of current population

---

## Section 6: Coding Challenge (Bonus: +10 points)

### Question 11 (10 points BONUS)

Complete the following function that implements a basic generational genetic algorithm:

```python
import random

def simple_ga(population, fitness_fn, n_generations, cx_prob=0.7, mut_prob=0.01):
    """
    Run a simple GA.

    Parameters:
    - population: list of chromosomes (binary lists)
    - fitness_fn: function that takes a chromosome and returns fitness
    - n_generations: number of generations to run
    - cx_prob: probability of crossover
    - mut_prob: probability of mutation per bit

    Returns:
    - best_individual: chromosome with highest fitness
    - best_fitness: fitness of best individual
    """

    for gen in range(n_generations):
        # Evaluate fitness
        fitness_values = [________________ for ind in population]

        # Selection (tournament with k=3)
        selected = []
        for _ in range(len(population)):
            tournament = random.sample(________________, 3)
            winner = max(tournament, key=________________)
            selected.append(winner)

        # Crossover and Mutation
        offspring = []
        for i in range(0, len(selected), 2):
            parent1, parent2 = selected[i], selected[i+1]

            if random.random() < cx_prob:
                # Single-point crossover at midpoint
                point = len(parent1) // 2
                child1 = ________________
                child2 = ________________
            else:
                child1, child2 = parent1[:], parent2[:]

            offspring.extend([child1, child2])

        # Mutation
        for individual in offspring:
            for i in range(len(individual)):
                if random.random() < mut_prob:
                    ________________

        population = offspring

    # Return best from final population
    fitness_values = [fitness_fn(ind) for ind in population]
    best_idx = ________________
    return population[best_idx], fitness_values[best_idx]
```

**Fill in the blanks (numbered 1-8):**

1. ___________________________________________________________________________

2. ___________________________________________________________________________

3. ___________________________________________________________________________

4. ___________________________________________________________________________

5. ___________________________________________________________________________

6. ___________________________________________________________________________

7. ___________________________________________________________________________

8. ___________________________________________________________________________

---

# Answer Key

## Section 1: Chromosome Encoding

### Question 1 (8 points)
**Correct Answers (any of these):**

```python
# Option 1: List comprehension with enumerate
selected_features = [features[i] for i, bit in enumerate(chromosome) if bit == 1]

# Option 2: Numpy style
selected_features = [f for f, c in zip(features, chromosome) if c == 1]

# Option 3: Using numpy
import numpy as np
selected_features = list(np.array(features)[np.array(chromosome) == 1])
```

**Grading:**
- Correct working code: 8 points
- Correct logic but syntax errors: 6 points
- Shows understanding but significant errors: 3 points
- Wrong approach: 0 points

---

### Question 2 (12 points)

**Part A (6 points):**

**Strong Answers:**

Binary Advantages:
- "Fixed-length representation makes crossover and mutation operators straightforward"
- "Standard operators (single-point crossover, bit-flip mutation) work directly"
- "Each gene position has consistent meaning across population"

Binary Disadvantages:
- "Can represent invalid solutions (zero or all features selected)"
- "Redundant representation for small feature subsets"
- "May select too many features without parsimony pressure"

Integer List Advantages:
- "Compact representation for small feature subsets"
- "Cannot represent empty feature set"
- "More memory efficient for sparse selections"

Integer List Disadvantages:
- "Variable-length chromosomes complicate crossover operations"
- "Need specialized operators to maintain validity (no duplicates)"
- "Harder to implement standard GA operators"

**Grading:**
- 1 advantage + 1 disadvantage for each: 6 points
- Only advantages or only disadvantages: 3 points
- Generic/incorrect answers: 0 points

**Part B (6 points):**

**Correct Answer:** Binary representation

**Strong Reasoning:**
- "Binary is standard because it works with established GA operators (single-point crossover, uniform crossover, bit-flip mutation) without modification, and DEAP's built-in operators expect fixed-length binary chromosomes."
- "Binary encoding is preferred in DEAP because it enables use of standard genetic operators and maintains population diversity more easily than variable-length representations."

**Grading:**
- Binary with strong reasoning: 6 points
- Binary with weak reasoning: 4 points
- Integer list: 0 points

---

## Section 2: Selection Operators

### Question 3 (10 points)
**Correct Answers:**
- Tournament: B (Choose best from k random individuals)
- Roulette Wheel: C (Probability proportional to fitness value)
- Rank Selection: A (Probability proportional to fitness rank)
- Elitism: D (Deterministically keep top n individuals)

**Grading:**
- All 4 correct: 10 points
- 3 correct: 7 points
- 2 correct: 4 points
- 1 correct: 2 points

---

### Question 4 (10 points)

**Part A (5 points):**

**Selected Individual:** B

**Selection Probability:** 100% (or 1.0)

**Explanation:** In tournament selection, the individual with highest fitness from the tournament is always selected. B has fitness 0.72, D has 0.68, and F has 0.55, so B wins deterministically.

**Grading:**
- B with 100%: 5 points
- B with wrong probability: 3 points
- Wrong individual: 0 points

**Part B (5 points):**

**Correct Answer:** C and A (or A and C)

**Explanation:** Elitism keeps the n best individuals. C has highest fitness (0.91) and A has second highest (0.85).

**Grading:**
- Both correct: 5 points
- One correct: 2 points
- Neither correct: 0 points

---

### Question 5 (5 points)

**Correct Answer:** True

**Strong Justification Examples:**
- "True. Larger tournament size means fitter individuals are more likely to be in any tournament and thus selected, increasing selection pressure. This speeds convergence but may cause premature convergence to local optima by reducing diversity."
- "True. With larger k, the probability that the best individuals are selected increases, intensifying selection pressure. This can lead to faster convergence but risks losing population diversity and getting stuck in local optima."

**Grading:**
- True with solid justification: 5 points
- True with weak justification: 3 points
- False: 0 points

---

## Section 3: Crossover Operators

### Question 6 (12 points)

**Correct Answers:**

**Child 1:** [1, 0, 1, 0, 1, 0, 1]

**Child 2:** [0, 1, 0, 1, 0, 1, 0]

**Explanation:** Child 1 takes first 3 bits from Parent 1 [1,0,1] and last 4 bits from Parent 2 [0,1,0,1]. Child 2 does the opposite.

**Grading:**
- Both children correct: 12 points
- One child correct: 6 points
- Shows understanding but swapped: 8 points
- Both incorrect: 0 points

---

### Question 7 (13 points)

**Part A (5 points):**

**Logic:** When `random.random() < 0.5`, take from corresponding parent; otherwise swap.

- Position 0: 0.3 < 0.5 → Child1 gets parent1[0]=1, Child2 gets parent2[0]=0
- Position 1: 0.6 >= 0.5 → Child1 gets parent2[1]=0, Child2 gets parent1[1]=1
- Position 2: 0.4 < 0.5 → Child1 gets parent1[2]=0, Child2 gets parent2[2]=1
- Position 3: 0.7 >= 0.5 → Child1 gets parent2[3]=1, Child2 gets parent1[3]=0

**Child 1:** [1, 0, 0, 1]

**Child 2:** [0, 1, 1, 0]

**Grading:**
- Both correct: 5 points
- One correct: 2 points
- Wrong but shows understanding: 1 point

**Part B (8 points):**

**Expected Number:** n/2 (or 0.5n, or half the chromosome length)

**Strong Explanation:**
- "At each position, there's a 0.5 probability the bit comes from Parent 1, so expected value is n × 0.5 = n/2 by linearity of expectation."
- "Since each bit independently has 50% chance of coming from Parent 1, the expected number is 0.5 × n = n/2."

**Grading:**
- n/2 with correct explanation: 8 points
- n/2 with weak explanation: 5 points
- Wrong answer but shows probability understanding: 2 points
- Wrong: 0 points

---

## Section 4: Mutation Operators

### Question 8 (8 points)

**Correct Answer:** C) 1/n to 0.01 (low mutation, where n is chromosome length)

**Strong Explanation Examples:**
- "Low mutation rates preserve good solutions while providing enough variation for exploration. 1/n means on average one bit flips per chromosome, maintaining diversity without disrupting good solutions."
- "This range balances exploration and exploitation. Too high turns the GA into random search; too low causes stagnation. The 1/n guideline ensures approximately one mutation per chromosome."

**Grading:**
- C with good explanation: 8 points
- C with weak explanation: 5 points
- B with reasonable argument: 3 points
- Other answers: 0 points

---

### Question 9 (12 points)

**Part A (4 points):**

**Logic:** Flip bit when random value < 0.05

- Position 0: 0.02 < 0.05 → flip: 1 → 0
- Position 1: 0.08 >= 0.05 → no flip: 0
- Position 2: 0.03 < 0.05 → flip: 1 → 0
- Position 3: 0.12 >= 0.05 → no flip: 1
- Position 4: 0.01 < 0.05 → flip: 0 → 1

**Mutated Chromosome:** [0, 0, 0, 1, 1]

**Grading:**
- Correct: 4 points
- 1-2 errors: 2 points
- More errors: 0 points

**Part B (4 points):**

**Expected Number:** 1 (or 1.0)

**Calculation:** Expected value = n × indpb = 20 × 0.05 = 1

**Grading:**
- Correct answer with calculation: 4 points
- Correct answer without calculation: 3 points
- Wrong but shows E[X] = np reasoning: 1 point

**Part C (4 points):**

**Strong Answers:**
- "DEAP requires mutation (and crossover) operators to return tuples because the framework expects this signature for all genetic operators, allowing consistent handling of multi-objective problems where operators may return multiple values."
- "Returns a tuple for consistency with DEAP's operator interface. All variation operators return tuples, even if containing a single individual, to support uniform handling in the toolbox."

**Grading:**
- Mentions DEAP convention/requirement: 4 points
- Generic answer about Python: 2 points
- Wrong: 0 points

---

## Section 5: GA Workflow

### Question 10 (10 points)

**Correct Order:**
1. Evaluate fitness of current population (step 6 in list)
2. Select parents using selection operator (step 3)
3. Apply crossover to selected parents (step 1)
4. Apply mutation to offspring (step 4)
5. Evaluate fitness of offspring (step 2)
6. Replace old population with new population (step 5)

**Alternative Correct Order (without offspring evaluation):**
1. Evaluate fitness of current population
2. Select parents using selection operator
3. Apply crossover to selected parents
4. Apply mutation to offspring
5. Replace old population with new population
6. Evaluate fitness of offspring (now becomes current population for next gen)

**Grading:**
- Perfect sequence: 10 points
- 1 adjacent swap: 7 points
- 2 adjacent swaps: 4 points
- Shows general understanding: 2 points
- Completely wrong: 0 points

**Note:** Both orders are acceptable as evaluation can occur before or after replacement.

---

## Section 6: Coding Challenge (Bonus)

### Question 11 (10 points BONUS)

**Correct Answers:**

1. `fitness_fn(ind)` or `fitness_fn(individual)`

2. `population` or `list(zip(population, fitness_values))`

3. `fitness_fn` or `lambda x: fitness_fn(x)` or `lambda ind: fitness_values[population.index(ind)]`

4. `parent1[:point] + parent2[point:]`

5. `parent2[:point] + parent1[point:]`

6. `individual[i] = 1 - individual[i]` or `individual[i] = 0 if individual[i] == 1 else 1`

7. `fitness_values.index(max(fitness_values))` or `max(range(len(fitness_values)), key=lambda i: fitness_values[i])`

**Note:** For #2 and #3, if using the zip approach:
2. `list(zip(population, fitness_values))`
3. `lambda x: x[1]`

**Grading:**
- All correct (7 blanks): 10 points
- 6 correct: 8 points
- 5 correct: 6 points
- 4 correct: 4 points
- 3 correct: 2 points
- Fewer than 3: 0 points

**Partial credit for showing correct understanding even with minor syntax errors.**

---

## Score Interpretation

| Score Range | Performance Level | Recommendation |
|-------------|------------------|----------------|
| 95-110 (with bonus) | Exceptional | Ready for Module 2 |
| 85-94 | Strong | Ready for Module 2 |
| 75-84 | Good | Review operator details, proceed to Module 2 |
| 65-74 | Adequate | Review module, especially operators |
| Below 65 | Needs Improvement | Re-study module, practice coding |

## Common Misconceptions to Address

1. **Selection Pressure:** Confusion about how tournament size affects convergence
2. **Crossover Mechanics:** Not understanding how genetic material is exchanged
3. **Mutation Rate:** Using too high mutation rates, turning GA into random search
4. **Operator Returns:** Forgetting DEAP's tuple return requirement
5. **Fitness Evaluation:** Unclear when fitness is evaluated in the GA loop
6. **Binary Operations:** Difficulty with bit-flip operation (1-bit vs XOR)
