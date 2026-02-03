# Module 4 Quiz: DEAP Implementation

**Course:** Genetic Algorithms for Feature Selection
**Module:** 4 - DEAP Implementation
**Total Points:** 100
**Estimated Time:** 30-35 minutes
**Attempts Allowed:** 2

## Instructions

This quiz tests your understanding of the DEAP framework for implementing genetic algorithms, including toolbox setup, custom operators, algorithm configuration, and performance optimization. Practical coding questions require working knowledge of DEAP's API.

---

## Section 1: DEAP Fundamentals (20 points)

### Question 1 (10 points)

Complete the following DEAP setup for binary feature selection:

```python
from deap import base, creator, tools
import random

# Define fitness and individual
creator.create("FitnessMax", base.Fitness, weights=________________)  # 1
creator.create("Individual", ________________, fitness=creator.FitnessMax)  # 2

# Initialize toolbox
toolbox = base.Toolbox()

# Attribute generator: binary values 0 or 1
toolbox.register("attr_bool", ________________)  # 3

# Individual generator: list of n_features binary values
toolbox.register("individual", tools.________________,  # 4
                 creator.Individual, toolbox.attr_bool, n=n_features)

# Population generator
toolbox.register("population", tools.________________,  # 5
                 list, toolbox.individual)
```

**Fill in the 5 blanks:**

1. ___________________________________________________________________________

2. ___________________________________________________________________________

3. ___________________________________________________________________________

4. ___________________________________________________________________________

5. ___________________________________________________________________________

---

### Question 2 (10 points)

Consider the following fitness function registration:

```python
def evaluate_features(individual, X, y, model):
    """Fitness function for feature selection."""
    selected = [i for i, bit in enumerate(individual) if bit == 1]

    if len(selected) == 0:
        return 0.0,

    X_selected = X[:, selected]
    scores = cross_val_score(model, X_selected, y, cv=5)
    return np.mean(scores),

# Register fitness function
toolbox.register("evaluate", evaluate_features, X=X_train, y=y_train, model=model)
```

**Part A (5 points):** Why does the fitness function return a tuple `(value,)` instead of just `value`?

**Answer:**

___________________________________________________________________________

___________________________________________________________________________

**Part B (5 points):** What is the purpose of using partial application (binding X, y, model arguments) when registering the evaluation function?

**Answer:**

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

---

## Section 2: Genetic Operators in DEAP (25 points)

### Question 3 (12 points)

Match each DEAP operator with its correct function signature and purpose:

| Function | Purpose |
|----------|---------|
| 1. `tools.cxTwoPoint` | A. Tournament selection with parameter k |
| 2. `tools.mutFlipBit` | B. Two-point crossover for sequences |
| 3. `tools.selTournament` | C. Bit-flip mutation with parameter indpb |
| 4. `tools.selBest` | D. Select n best individuals |

**Answers:**
- `tools.cxTwoPoint`: ___________
- `tools.mutFlipBit`: ___________
- `tools.selTournament`: ___________
- `tools.selBest`: ___________

**For each operator, provide the required parameters (besides the individual(s)):**

- `cxTwoPoint`: ___________________________________________________________

- `mutFlipBit`: ___________________________________________________________

- `selTournament`: ________________________________________________________

- `selBest`: ______________________________________________________________

---

### Question 4 (13 points)

Complete the operator registration for a feature selection GA:

```python
# Crossover: two-point crossover
toolbox.register("mate", ________________)  # 1

# Mutation: flip each bit with probability indpb
toolbox.register("mutate", tools.mutFlipBit, indpb=________________)  # 2

# Selection: tournament with size 3
toolbox.register("select", tools.________________,  # 3
                 tournsize=________________)  # 4

# Run the algorithm
population = toolbox.population(n=100)

# Evaluate initial population
fitnesses = map(________________, population)  # 5
for ind, fit in zip(population, fitnesses):
    ________________  # 6

# Evolution loop
for gen in range(n_generations):
    # Select parents
    offspring = toolbox.select(population, len(population))
    offspring = list(map(________________, offspring))  # 7

    # Crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < crossover_prob:
            toolbox.mate(________________, ________________)  # 8
            del child1.fitness.values  # 9: Why delete fitness?
            del child2.fitness.values

    # Mutation
    for mutant in offspring:
        if random.random() < mutation_prob:
            toolbox.________________(mutant)  # 10
            del mutant.fitness.values
```

**Fill in the 10 blanks.**

---

## Section 3: Custom Operators (20 points)

### Question 5 (10 points)

Implement a custom mutation operator that ensures at least `min_features` are selected:

```python
def mutate_with_minimum(individual, indpb, min_features=1):
    """
    Flip bits with probability indpb, but ensure at least
    min_features remain selected.

    Returns: Tuple containing the individual
    """
    # Standard bit-flip mutation
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = ________________  # 1: flip bit

    # Check if we have enough features
    n_selected = ________________  # 2: count selected features

    # If too few, randomly select until we have min_features
    if n_selected < min_features:
        unselected = ________________  # 3: find unselected indices
        to_select = ________________  # 4: how many more to select?
        selected_indices = random.sample(________________, to_select)  # 5
        for idx in selected_indices:
            ________________  # 6: set these indices to 1

    return ________________,  # 7: return tuple
```

**Fill in the 7 blanks.**

---

### Question 6 (10 points)

Create a custom crossover operator that exchanges selected features between parents:

```python
def crossover_features(ind1, ind2):
    """
    Crossover that exchanges randomly selected features between individuals.
    Each parent's selected features are pooled, then randomly redistributed.

    Returns: Tuple containing both individuals
    """
    # Get selected features from each parent
    selected1 = ________________  # 1
    selected2 = ________________  # 2

    # Pool all selected features
    all_selected = list(set(________________))  # 3: combine and deduplicate

    if len(all_selected) == 0:
        return ind1, ind2  # No features to exchange

    # Randomly redistribute
    random.shuffle(all_selected)
    split_point = len(all_selected) // 2

    # Clear both individuals
    for i in range(len(ind1)):
        ind1[i] = 0
        ind2[i] = 0

    # Assign features to each child
    for idx in all_selected[:split_point]:
        ________________  # 4

    for idx in all_selected[split_point:]:
        ________________  # 5

    return ________________  # 6
```

**Fill in the 6 blanks.**

---

## Section 4: Algorithm Configuration (20 points)

### Question 7 (12 points)

You're configuring a GA for time series feature selection with 40 candidate features:

**Part A (4 points):** You set the mutation probability `indpb = 0.05`. What is the expected number of bits flipped per individual? Is this appropriate?

**Expected Flips:** ___________

**Appropriateness:**

___________________________________________________________________________

___________________________________________________________________________

**Part B (4 points):** You set population size to 50 and run for 100 generations. Approximately how many unique feature subsets will be evaluated (assuming 70% crossover rate and no duplicate evaluations)?

**Calculation:**

___________________________________________________________________________

**Answer:** ___________

**Part C (4 points):** Tournament selection with `tournsize=2` provides low selection pressure. What `tournsize` would you recommend for this problem and why?

**Recommended:** ___________

**Reasoning:**

___________________________________________________________________________

___________________________________________________________________________

---

### Question 8 (8 points)

Compare two algorithm configurations:

**Config A:**
- Population: 200
- Generations: 50
- Crossover: 0.9
- Mutation: 0.01

**Config B:**
- Population: 50
- Generations: 200
- Crossover: 0.7
- Mutation: 0.05

**Part A (4 points):** Which configuration is likely to converge faster? Why?

**Answer:** ___________

**Explanation:**

___________________________________________________________________________

___________________________________________________________________________

**Part B (4 points):** Which configuration is likely to find more diverse solutions? Why?

**Answer:** ___________

**Explanation:**

___________________________________________________________________________

___________________________________________________________________________

---

## Section 5: Performance Optimization (15 points)

### Question 9 (8 points)

The following code evaluates fitness sequentially:

```python
# Sequential evaluation
for ind in population:
    ind.fitness.values = toolbox.evaluate(ind)
```

Modify it to use parallel evaluation with multiprocessing:

```python
from multiprocessing import Pool

# Parallel evaluation
pool = ________________  # 1: create pool with 4 processes

fitnesses = ________________  # 2: map evaluate function over population

for ind, fit in ________________:  # 3: assign fitness values
    ind.fitness.values = fit

pool.close()
```

**Fill in the 3 blanks.**

---

### Question 10 (7 points)

True or False with justification:

**Statement:** Using `tools.HallOfFame(n=10)` to track the best 10 individuals throughout evolution increases computational cost significantly because it requires re-evaluating these individuals each generation.

**Answer (T/F):** ___________

**Justification:**

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________

---

## Section 6: Debugging and Best Practices (Bonus: +10 points)

### Question 11 (10 points BONUS)

Debug the following code which has multiple issues:

```python
from deap import base, creator, tools
import random

# Setup
creator.create("Fitness", base.Fitness, weights=(1.0))  # Issue 1?
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_bool, n=20)
toolbox.register("population", tools.initRepeat,
                 list, toolbox.individual)

def evaluate(individual):
    return sum(individual)  # Issue 2?

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run
pop = toolbox.population(n=50)

for ind in pop:
    ind.fitness.values = toolbox.evaluate(ind)

for gen in range(10):
    offspring = toolbox.select(pop, 50)

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        toolbox.mate(child1, child2)
        del child1.fitness.values
        del child2.fitness.values  # Issue 3?

    for mutant in offspring:
        toolbox.mutate(mutant)
        del mutant.fitness.values  # Issue 4?

    for ind in offspring:
        ind.fitness.values = toolbox.evaluate(ind)

    pop = offspring

best = tools.selBest(pop, 1)[0]
print(f"Best fitness: {best.fitness.values}")
```

**Identify and explain at least 4 issues. Provide corrections.**

**Issue 1:**

___________________________________________________________________________

**Correction:** _________________________________________________________________

**Issue 2:**

___________________________________________________________________________

**Correction:** _________________________________________________________________

**Issue 3:**

___________________________________________________________________________

**Correction:** _________________________________________________________________

**Issue 4:**

___________________________________________________________________________

**Correction:** _________________________________________________________________

**Additional issues (if any):**

___________________________________________________________________________

___________________________________________________________________________

---

# Answer Key

## Section 1: DEAP Fundamentals

### Question 1 (10 points)

**Correct Answers:**

1. `(1.0,)` - Tuple with single value for maximization

2. `list` - Individual is a list

3. `random.randint, 0, 1` - Generates random 0 or 1

4. `initRepeat` - Repeats the attribute generator n times

5. `initRepeat` - Repeats the individual generator

**Complete code:**
```python
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_bool, n=n_features)
toolbox.register("population", tools.initRepeat,
                 list, toolbox.individual)
```

**Grading:**
- All 5 correct: 10 points (2 per blank)
- 4 correct: 8 points
- 3 correct: 6 points
- 2 correct: 4 points
- 1 correct: 2 points

---

### Question 2 (10 points)

**Part A (5 points):**

**Strong Answer:**
"DEAP's fitness system requires fitness values to be tuples because it supports multi-objective optimization. Even for single-objective problems, the fitness is stored as a tuple to maintain consistency across the framework. The tuple format allows the same code to handle both single and multi-objective cases, where multi-objective would have tuples like (obj1, obj2, obj3)."

**Key Points:**
- DEAP requirement for consistency
- Supports multi-objective optimization
- Single value in tuple for single-objective

**Grading:**
- Complete explanation with key points: 5 points
- Mentions DEAP requirement and multi-objective: 4 points
- Basic understanding: 2 points
- Wrong: 0 points

**Part B (5 points):**

**Strong Answer:**
"Partial application binds the dataset and model arguments at registration time, so during evolution we only need to pass the individual. This is necessary because DEAP's algorithms expect the evaluation function to take only the individual as an argument. It also makes the code cleaner and avoids passing the same data repeatedly. The toolbox.evaluate function becomes a single-argument function that internally has access to X, y, and model."

**Key Points:**
- DEAP algorithms expect single-argument evaluate function
- Binds data at registration time
- Cleaner code, no repeated argument passing
- Creates closure over training data

**Grading:**
- Complete explanation: 5 points
- Explains DEAP requirement: 3 points
- Vague understanding: 1 point

---

## Section 2: Genetic Operators in DEAP

### Question 3 (12 points)

**Correct Matches:**
- `tools.cxTwoPoint`: B
- `tools.mutFlipBit`: C
- `tools.selTournament`: A
- `tools.selBest`: D

**Required Parameters:**
- `cxTwoPoint`: None (just the two individuals)
- `mutFlipBit`: `indpb` (probability per bit)
- `selTournament`: `individuals, k, tournsize`
- `selBest`: `individuals, k` (number to select)

**Grading:**
- All matches + all parameters: 12 points
- All matches, missing some parameters: 9 points
- 3 matches correct: 6 points
- 2 matches correct: 3 points

---

### Question 4 (13 points)

**Correct Answers:**

1. `tools.cxTwoPoint`

2. `0.01` to `0.05` (accept 1/n_features where n_features is chromosome length)

3. `selTournament`

4. `3`

5. `toolbox.evaluate`

6. `ind.fitness.values = fit`

7. `toolbox.clone` (or `copy.deepcopy`)

8. `child1, child2` (the two individuals to crossover)

9. This is a comment/explanation, not a fill-in. Accept: "Fitness is no longer valid after modification"

10. `mutate`

**Grading:**
- All correct: 13 points
- 8-9 correct: 10 points
- 6-7 correct: 7 points
- 4-5 correct: 5 points
- Fewer: proportional

---

## Section 3: Custom Operators

### Question 5 (10 points)

**Correct Answers:**

1. `1 - individual[i]` OR `0 if individual[i] == 1 else 1`

2. `sum(individual)`

3. `[i for i in range(len(individual)) if individual[i] == 0]`

4. `min_features - n_selected`

5. `unselected, to_select`

6. `individual[idx] = 1`

7. `individual,` OR `(individual,)`

**Complete function:**
```python
def mutate_with_minimum(individual, indpb, min_features=1):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = 1 - individual[i]

    n_selected = sum(individual)

    if n_selected < min_features:
        unselected = [i for i in range(len(individual)) if individual[i] == 0]
        to_select = min_features - n_selected
        selected_indices = random.sample(unselected, to_select)
        for idx in selected_indices:
            individual[idx] = 1

    return individual,
```

**Grading:**
- All 7 correct: 10 points
- 6 correct: 8 points
- 5 correct: 6 points
- 4 correct: 4 points
- Fewer: proportional

---

### Question 6 (10 points)

**Correct Answers:**

1. `[i for i in range(len(ind1)) if ind1[i] == 1]`
   OR `[i for i, bit in enumerate(ind1) if bit == 1]`

2. `[i for i in range(len(ind2)) if ind2[i] == 1]`

3. `selected1 + selected2`

4. `ind1[idx] = 1`

5. `ind2[idx] = 1`

6. `ind1, ind2` OR `(ind1, ind2)`

**Grading:**
- All 6 correct: 10 points
- 5 correct: 8 points
- 4 correct: 6 points
- 3 correct: 4 points
- Fewer: proportional

---

## Section 4: Algorithm Configuration

### Question 7 (12 points)

**Part A (4 points):**

**Expected Flips:** 2 (40 features × 0.05 = 2)

**Appropriateness:**
"This is appropriate. With indpb=0.05, we expect about 2 bits to flip per mutation, which follows the 1/n guideline (1/40=0.025). This provides enough variation for exploration while preserving most of the chromosome structure, balancing exploration and exploitation."

**Grading:**
- Correct calculation + good explanation: 4 points
- Correct calculation + basic explanation: 3 points
- Calculation only: 2 points

**Part B (4 points):**

**Calculation:**
- Initial population: 50
- Generations: 100
- Each generation creates approximately 50 new individuals (offspring)
- With 70% crossover: 35 individuals created via crossover + mutation
- With 30% mutation-only: 15 individuals
- Total new evaluations: ~50 per generation × 100 = ~5000
- Plus initial: 5000 + 50 = ~5050

**Answer:** Approximately 5,000-5,500 unique evaluations

**Note:** Exact answer depends on elitism, duplicate prevention, etc.

**Grading:**
- Reasonable calculation and answer (4500-6000): 4 points
- Shows understanding but error: 2 points
- Wrong approach: 0 points

**Part C (4 points):**

**Recommended:** 3-5 (accept this range)

**Strong Reasoning:**
"Tournament size of 3-4 provides moderate selection pressure appropriate for this problem size. It's strong enough to drive convergence but not so strong (like 7+) that it causes premature convergence and loss of diversity in the population. With 40 features and a 50-generation run, we need balanced pressure to explore the space effectively."

**Grading:**
- Reasonable size (3-5) with good reasoning: 4 points
- Reasonable size with weak reasoning: 2 points
- Extreme values: 0 points

---

### Question 8 (8 points)

**Part A (4 points):**

**Answer:** Config A

**Explanation:**
"Config A converges faster because larger population size (200 vs 50) means more parallel exploration per generation, and higher crossover rate (0.9 vs 0.7) means more genetic recombination. With 200 individuals, good solutions are found and propagated more quickly across the population, leading to faster convergence."

**Grading:**
- Config A with solid explanation: 4 points
- Config A with weak explanation: 2 points
- Wrong answer: 0 points

**Part B (4 points):**

**Answer:** Config B

**Explanation:**
"Config B maintains more diversity because higher mutation rate (0.05 vs 0.01) introduces more variation each generation, and longer run (200 gens) allows continued exploration. Smaller population with higher mutation preserves diversity longer, whereas Config A's large population and high crossover lead to rapid convergence and homogeneity."

**Grading:**
- Config B with solid explanation: 4 points
- Config B with weak explanation: 2 points
- Wrong: 0 points

---

## Section 5: Performance Optimization

### Question 9 (8 points)

**Correct Answers:**

1. `Pool(processes=4)` OR `Pool(4)`

2. `pool.map(toolbox.evaluate, population)`

3. `zip(population, fitnesses)`

**Complete code:**
```python
from multiprocessing import Pool

pool = Pool(processes=4)
fitnesses = pool.map(toolbox.evaluate, population)
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit
pool.close()
```

**Grading:**
- All 3 correct: 8 points
- 2 correct: 5 points
- 1 correct: 3 points
- Shows understanding but syntax errors: 4 points

---

### Question 10 (7 points)

**Correct Answer:** False

**Strong Justification:**
"False. HallOfFame does not require re-evaluation. It tracks individuals and their fitness values throughout evolution by maintaining references to the best individuals already evaluated. When an individual is evaluated and enters the HallOfFame, its fitness is stored. The HallOfFame simply compares new individuals' fitness to existing members - no re-evaluation occurs. The computational overhead is minimal, just comparison and sorting operations."

**Key Points:**
- No re-evaluation needed
- Stores references to individuals with their fitness
- Only comparison overhead
- Minimal computational cost

**Grading:**
- False with complete explanation: 7 points
- False with basic explanation: 5 points
- False with minimal explanation: 3 points
- True: 0 points

---

## Section 6: Debugging and Best Practices (Bonus)

### Question 11 (10 points BONUS)

**Issues and Corrections:**

**Issue 1:**
"weights should be a tuple: `weights=(1.0,)` not `weights=(1.0)`. Without the comma, Python treats (1.0) as a float, not a tuple."

**Correction:** `weights=(1.0,)`

**Issue 2:**
"Evaluate function returns scalar instead of tuple. Should return `(sum(individual),)` for DEAP compatibility."

**Correction:** `return (sum(individual),)` or `return sum(individual),`

**Issue 3:**
"Deleting fitness unconditionally after crossover, but crossover might not modify both children (e.g., if crossover_prob check were added). However, in this code, mate is always called, so this is actually not wrong - it's appropriate."

**Note:** This is actually correct behavior. The issue is missing crossover probability check, but fitness deletion is appropriate.

**Issue 4:**
"Similar to Issue 3 - mutation is always applied without probability check, making fitness deletion appropriate. But typically you'd check mutation_prob first."

**Correction:** Should check mutation probability:
```python
for mutant in offspring:
    if random.random() < mutation_prob:
        toolbox.mutate(mutant)
        del mutant.fitness.values
```

**Issue 5 (MAIN ISSUE):**
"No clone/copy of selected individuals. `offspring = toolbox.select(pop, 50)` returns references to existing individuals, so modifying offspring modifies the original population. Need `offspring = list(map(toolbox.clone, offspring))`."

**Correction:**
```python
offspring = toolbox.select(pop, 50)
offspring = list(map(toolbox.clone, offspring))
```

**Issue 6:**
"No crossover or mutation probability checks. All individuals are crossed over and mutated every time."

**Issue 7:**
"No elitism - best solutions can be lost. Should use `tools.selBest` to preserve top individuals or use `pop = tools.selBest(pop + offspring, len(pop))`."

**Grading:**
- Identifies 4+ issues with corrections: 10 points
- Identifies 3 issues with corrections: 7 points
- Identifies 2 issues with corrections: 5 points
- Identifies issues but weak corrections: 3 points
- Missing major issues (especially cloning): -3 points

**Most Critical Issues:**
1. Weights tuple syntax
2. Return tuple from evaluate
3. Missing clone operation (MAJOR)
4. Missing probability checks
5. No elitism

---

## Score Interpretation

| Score Range | Performance Level | Recommendation |
|-------------|------------------|----------------|
| 95-110 (with bonus) | Exceptional | Ready for Module 5 |
| 85-94 | Strong | Ready for Module 5 |
| 75-84 | Good | Review DEAP operators, proceed |
| 65-74 | Adequate | Practice DEAP coding, review documentation |
| Below 65 | Needs Improvement | Re-study module, complete more coding exercises |

## Common Misconceptions to Address

1. **Tuple Returns:** Forgetting fitness must be returned as tuple
2. **Cloning:** Not cloning selected individuals before modification
3. **Fitness Deletion:** Not understanding when to delete fitness.values
4. **Parallel Evaluation:** Assuming it's automatic or very complex
5. **Weights Syntax:** Using (1.0) instead of (1.0,) for single-objective
6. **Partial Application:** Not understanding how registration with arguments works
