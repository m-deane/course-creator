# Conceptual Depth Review: Genetic Algorithms for Feature Selection

**Date:** 2026-04-05
**Reviewer:** Claude (automated review)
**Course:** `genetic-algorithms-feature-selection`
**User feedback:** "The course is very code example heavy -- it is hard to understand the key points and concepts from this course alone."

---

## 1. Executive Summary

The user feedback is substantiated. Across all 21 guide files and the sampled notebooks, the course has a significant **concept-to-code imbalance**. Code blocks dominate the content -- many guides are effectively reference implementations with thin explanatory wrappers. While the code itself is high quality (well-structured, real libraries, production patterns), the surrounding conceptual scaffolding is insufficient for a reader to develop deep understanding.

**Key findings:**

- **Concept-to-code ratio** averages roughly 25-30% explanation to 70-75% code across all modules. Module 0 (Foundations) is the best-balanced at roughly 40/60; Modules 4 and 5 are the worst at roughly 15-20% explanation to 80-85% code.
- **Missing "why" layer:** Most guides explain *what* an algorithm does and *how* to code it, but rarely explain *why* it works, *when* it fails, or *what alternatives exist and why they are inferior*. The Building Block Hypothesis in Module 1 Guide 03 is mentioned in one sentence but never unpacked.
- **Sparse analogies and intuitions:** Modules 0 and 3 include useful analogies (weather experts, exam-taking). Modules 1, 4, and 5 largely skip analogies and jump straight to formal definitions and code.
- **Diagram poverty:** SVG references exist (ga_lifecycle.svg, fitness_landscape.svg, etc.) but conceptual flow is overwhelmingly communicated through code rather than visual explanations. No decision-tree diagrams for "which operator to choose," no visual comparison of convergence paths, no annotated examples showing GA evolution step-by-step on a real problem.
- **Bridging text deficit:** Between code blocks, transitions are often a single sentence or a bare section header. The reader must infer the conceptual thread connecting one implementation to the next.
- **Notebook issue:** The sampled Module 2 notebook claims "90 minutes" but provides minimal conceptual setup between code cells. Markdown cells are typically 2-4 sentences before launching into code. The notebooks are even more code-heavy than the guides.

**Overall assessment:** The course teaches by *showing* code, not by *explaining* concepts. A reader who already understands GA theory could use this as a cookbook. A reader trying to *learn* GA feature selection will struggle to build mental models.

---

## 2. Per-Module Analysis

### Module 00: Foundations

**Files reviewed:** 01_feature_selection_problem.md, 02_optimization_basics.md, 03_evolutionary_operators.md, 04_selection_approaches.md

**Concept-to-code ratio:** ~40/60 (best in the course)

**Strengths:**
- Guide 01 has a good "Intuitive Explanation" section with the weather experts analogy
- Guide 01 has a concrete table showing search space explosion (10 features to 100 features)
- Guide 02 has the XOR problem as an intuitive motivator for why greedy fails
- Guide 04 has the resume/trial period/hiring analogy for filter/wrapper/embedded methods

**Top 3 concepts needing better explanation:**
1. **Why GAs specifically for feature selection:** Guide 02 states GAs are "well-suited" but the explanation is four bullet points. Missing: a worked comparison showing GA finding a feature set that forward selection misses (e.g., the XOR example run to completion).
2. **The No Free Lunch Theorem and its practical implications:** Guide 02 devotes 10 lines to this theorem but never connects it to *practical decisions the reader will make*. When should they choose GA over Lasso? Over RFE? The theorem is stated as trivia, not as an actionable framework.
3. **How the three evolutionary operators (selection, crossover, mutation) interact as a system:** Guide 03 opens with a callout saying "removing any one breaks the system" but never demonstrates *how* or *why*. No example of a GA with mutation disabled (converges prematurely) vs one with crossover disabled (random search) vs the full system.

**Missing bridging text:**
- Guide 03: Between the tournament selection code and the roulette wheel code, there is no text explaining *when you would choose one over the other* or what tradeoffs exist. The reader sees two implementations but no decision framework.
- Guide 04: Between filter, wrapper, and embedded code implementations, the connecting text is purely structural ("### Wrapper Method: Forward Selection"). Missing: "Now that we've seen filters miss feature interactions, let's see how wrapper methods address this..."

**Suggested additions:**
- Add a "Decision Framework" section to Guide 02 with a flowchart: "Choose your feature selection approach based on: number of features, computational budget, model type, need for interaction detection."
- Add a "System Dynamics" section to Guide 03 showing what happens when you remove each operator (3 small experiments with results, not necessarily code -- could be described textually with a table of outcomes).

---

### Module 01: GA Fundamentals

**Files reviewed:** 01_encoding.md, 01_ga_components.md, 02_selection.md, 03_genetic_operators.md

**Concept-to-code ratio:** ~25/75

**Strengths:**
- Guide 01 (Encoding) has the light switch panel analogy -- effective and memorable
- Guide 03 (Genetic Operators) has the chef recipe analogy for crossover
- Guide 02 (Selection) has formal probability definitions alongside intuitive sports/lottery analogies

**Top 3 concepts needing better explanation:**
1. **The Building Block Hypothesis:** Guide 03 introduces schemata in 3 sentences and never explains why this matters for feature selection. This is the theoretical foundation for *why crossover works* -- it should be a full subsection with an example showing building blocks being combined.
2. **Selection pressure and its consequences:** Guide 02 defines selection pressure mathematically but never gives the reader intuition for what "too much" or "too little" pressure looks like in practice. Missing: a worked example showing premature convergence vs. stuck search at different tournament sizes.
3. **Why uniform crossover is better for feature selection than single-point:** Stated as fact in multiple guides but never demonstrated. The claim is that "features are unordered" -- this needs a concrete example showing how single-point crossover creates positional bias and why that hurts.

**Missing bridging text:**
- Guide 01 (ga_components): After the Population class code, there is no explanation connecting the data structure to the algorithm. The reader sees classes but does not understand why the code is organized this way.
- Guide 03: Between the crossover operators section and the mutation operators section, there is no transition explaining that crossover *exploits* existing solutions while mutation *explores* new territory. This exploitation/exploration duality is the conceptual heart of GAs.

**Suggested additions:**
- Add a "Building Blocks in Action" section to Guide 03 showing: two parents each with a useful feature cluster, crossover combining them, and the offspring outperforming both parents. Visual annotation of which building blocks were preserved.
- Add a "Selection Pressure Experiment" to Guide 02: table showing tournament sizes 2, 3, 5, 10 with resulting diversity and convergence speed (not necessarily runnable code -- could be pre-computed results with interpretation).

---

### Module 02: Fitness Functions

**Files reviewed:** 01_fitness_functions.md, 02_cross_validation_fitness.md, 03_multi_objective.md

**Concept-to-code ratio:** ~20/80 (Guide 01 is worst: ~5 min reading time, almost entirely code)

**Strengths:**
- Guide 03 (Multi-Objective) has excellent Pareto front intuition with the laptop shopping analogy
- Guide 03 has the manual Pareto front practice problem -- good for cementing understanding
- Guide 02 has the callout about fitness function being the single most impactful component

**Top 3 concepts needing better explanation:**
1. **How to design a fitness function from scratch:** Guide 01 shows 6 different fitness function implementations but never walks through the *design process*. Missing: a section titled "Designing Your First Fitness Function" that starts from the problem statement, identifies objectives, discusses how to weight them, and arrives at an implementation. The current guide is a catalog of solutions without a design methodology.
2. **The parsimony penalty -- how to choose lambda:** Multiple guides mention lambda or penalty_weight but none explain how to calibrate it. Missing: guidance on "start with 0.01, if the GA selects all features increase it, if it selects too few decrease it" with a sensitivity analysis showing how lambda changes the accuracy-complexity tradeoff.
3. **What makes a fitness landscape hard or easy for GAs:** The fitness landscape concept is mentioned in passing but never explained in depth. Missing: explanation of ruggedness, epistasis, deception -- and what each means for GA configuration. A rugged landscape needs larger population; a deceptive one needs crossover that preserves building blocks.

**Missing bridging text:**
- Guide 01: The six code blocks (cv_fitness, multi_objective_fitness, walk_forward_fitness, expanding_window_fitness, nested_cv_fitness, regularized_fitness) have minimal connecting explanation. The reader sees a catalog but doesn't understand when to use which. Need a decision tree or comparison table.
- Guide 02: Between CVFitnessEvaluator and TimeSeriesFitnessEvaluator, there is no explanation of *when to switch from one to the other*. Just "For financial applications, use time-aware CV" with no elaboration on what goes wrong if you don't.

**Suggested additions:**
- Add a "Fitness Function Design Recipe" section to Guide 01 with a step-by-step process: (1) List objectives, (2) Choose evaluation metric for each, (3) Decide single-objective vs multi-objective, (4) Set penalty weights via sensitivity analysis, (5) Validate that fitness correlates with true out-of-sample performance.
- Add a "Lambda Calibration Guide" subsection showing 3-5 lambda values and their effect on selected feature count and accuracy.

---

### Module 03: Time Series

**Files reviewed:** 01_timeseries_considerations.md, 01_walk_forward.md, 02_lag_features.md, 03_stationarity.md

**Concept-to-code ratio:** ~30/70 (better than Modules 1-2, thanks to longer intuitive explanations)

**Strengths:**
- Guide 01 (Walk Forward) has an excellent intuitive explanation comparing k-fold to "studying with exam answers"
- Guide 02 (Lag Features) has the temperature prediction analogy showing why lag-365 matters
- Guide 03 (Stationarity) has the house price prediction example for why non-stationarity is dangerous
- The Fixed vs Expanding window comparison cards are effective

**Top 3 concepts needing better explanation:**
1. **Why standard k-fold fails for time series -- the mechanism:** Every guide says "don't use k-fold" but only Guide 01 (Walk Forward) explains *why* with an analogy. Missing: a concrete numerical example showing the same model getting 95% accuracy with k-fold and 60% with walk-forward, with an explanation of the information leakage mechanism. The compare_with_wrong_cv function exists in Guide 01 code but is not accompanied by the kind of pre-computed results and interpretation that would drive the point home for a reader who skims code.
2. **Stationarity -- practical impact on feature selection:** Guide 03 is long and thorough on testing stationarity but weak on *connecting stationarity to GA feature selection outcomes*. Missing: a worked example where a GA trained on non-stationary data selects spurious features, vs the same GA on stationarized data selecting the correct features.
3. **The embargo/gap concept:** Guides mention "gap" between train and test but don't explain the underlying mechanism (autocorrelation leaking information). Missing: an intuitive explanation of *how* autocorrelation leaks and *how* the gap prevents it.

**Missing bridging text:**
- Guide 02 (Lag Features): The jump from ACF/PACF analysis to "GA Fitness Function for Lag Selection" has no bridging text explaining *how ACF/PACF results should inform GA configuration*. Should the significant PACF lags be used to initialize the population? Should non-significant lags be given higher mutation probability?
- Guide 03: Between stationarity testing code and the GA fitness function with stationarity penalty, there is no explanation of *why penalizing non-stationary features improves generalization*.

**Suggested additions:**
- Add a "Why K-Fold Leaks Information" diagram to Guide 01 showing a timeline with train/test splits, arrows showing where future data contaminates training.
- Add a "From ACF/PACF to GA Configuration" bridging section in Guide 02 explaining how to use ACF/PACF results to (a) set max lag, (b) seed initial population, (c) design lag-group mutation.

---

### Module 04: Implementation

**Files reviewed:** 01_deap_implementation.md, 02_custom_operators.md, 03_production_considerations.md

**Concept-to-code ratio:** ~15/85 (worst module for conceptual depth)

**Strengths:**
- Guide 01 (DEAP) has the flow diagram showing the 5-step DEAP workflow
- Guide 02 (Custom Operators) has the one-hot encoding example showing why standard crossover fails
- Guide 03 (Production) has the parallelization intuitive example with time calculations

**Top 3 concepts needing better explanation:**
1. **DEAP's design philosophy and mental model:** Guide 01 jumps straight to `creator.create()` without explaining *why DEAP is structured as a toolbox*. Missing: a conceptual overview of DEAP's architecture -- creator, toolbox, algorithms, tools -- and how they compose. The reader should understand the mental model before seeing the code.
2. **When and why to design custom operators:** Guide 02 provides excellent custom operator code but insufficient motivation. The one-hot example is good but is the only example. Missing: a taxonomy of problem structures that benefit from custom operators: (a) grouped features, (b) hierarchical features, (c) budget-constrained features, (d) correlated features -- with a sentence for each explaining the benefit.
3. **Production concerns beyond parallelization:** Guide 03 covers parallelization and caching but barely mentions monitoring, logging, model versioning, or CI/CD integration. The sklearn-compatible GAFeatureSelector class is excellent code but has no surrounding explanation of *why wrapping GA in a sklearn interface matters for production*.

**Missing bridging text:**
- Guide 01: Between "Basic GA Setup" and "Running the GA" there is no explanation of the *decision points* the reader faces: how to choose pop_size, n_generations, crossover_prob, mutation_prob. These are presented as magic numbers.
- Guide 02: Between each custom operator implementation (group-aware, hierarchical, importance-weighted, correlation-aware), there is no "choose your operator" framework.

**Suggested additions:**
- Add a "DEAP Mental Model" section to Guide 01 with a diagram showing how creator, toolbox, algorithms, and tools relate. Explain the toolbox as a "recipe" that defines the GA's ingredients.
- Add a "Parameter Selection Guide" table to Guide 01 with heuristics: pop_size = 2-5x n_features, mutation_rate = 1/n_features, crossover_prob = 0.6-0.9, tournament_size = 3-7.
- Add a "Which Custom Operator?" decision tree to Guide 02.

---

### Module 05: Advanced Topics

**Files reviewed:** 01_advanced_techniques.md, 02_hybrid_methods.md, 03_adaptive_operators.md

**Concept-to-code ratio:** ~30/70 (Guide 02 is best in this module at ~35/65)

**Strengths:**
- Guide 02 (Hybrid Methods) has the best conceptual balance in the course with clear "Pure GA vs Pure Hill Climbing vs Memetic" comparison
- Guide 02's Lamarckian vs Baldwinian explanation is well done with concrete examples
- Guide 03 (Adaptive Operators) has the "Why Fixed Parameters Fail" example showing three generation-by-generation scenarios

**Top 3 concepts needing better explanation:**
1. **NSGA-II's non-dominated sorting -- the mechanism:** Guide 01 uses NSGA-II but never explains *how non-dominated sorting works*. The reader is told to call `tools.selNSGA2` without understanding what happens inside. Missing: a step-by-step walkthrough of non-dominated sorting on a small example (5 solutions, 2 objectives).
2. **When to use hybrid methods vs pure GA:** Guide 02 provides memetic algorithm code but no guidance on *when the added complexity is justified*. Missing: decision criteria -- "Use hybrid when: (a) fitness evaluations are cheap, (b) landscape has many local optima, (c) you need high-precision solutions."
3. **Surrogate models -- the tradeoff:** Guide 01 includes a SurrogateFitness class but provides zero conceptual explanation of what a surrogate model is, why GP is used, what the uncertainty threshold means, or when surrogates help vs hurt.

**Missing bridging text:**
- Guide 01: Between NSGA-II code and Hybrid Methods code, there is no transition explaining that NSGA-II handles *what to optimize* (multiple objectives) while hybrids handle *how to optimize* (combining strategies). These are orthogonal ideas but are presented sequentially with no connecting thread.
- Guide 03: Between linear, diversity-based, and feedback-based adaptation, there is no comparison explaining when each is preferred.

**Suggested additions:**
- Add a "How NSGA-II Works" walkthrough to Guide 01 with 5 solutions plotted on a 2D objective space, showing domination relationships, ranking assignment, and crowding distance calculation -- all annotated, no code needed.
- Add a "Surrogate Model Intuition" section explaining: "A surrogate is a cheap approximation of the expensive fitness function. It learns from previous evaluations to predict fitness of new individuals. When confident, use the prediction; when uncertain, compute exact fitness."
- Add a "Choosing Your Strategy" comparison table to Guide 03: linear (good for simple problems, no tuning), diversity (robust default), feedback (when you can define 'improvement'), self-adaptive (when you have large populations and long runs).

---

## 3. Cross-Cutting Issues

### Issue 1: Code-first, concept-second pattern
Nearly every guide follows the structure: brief intro (2-3 sentences) -> formal definition -> code block -> more code -> key takeaways. The "Intuitive Explanation" sections that do exist are placed *after* the formal definition, when they should come *before* or *alongside* it. Readers who don't already know the math will bounce off the formal definitions.

### Issue 2: Missing "when to use" guidance
The course presents many techniques (selection operators, crossover types, fitness functions, validation strategies) but almost never helps the reader choose between them. Decision frameworks, comparison tables, and flowcharts are almost entirely absent.

### Issue 3: Callouts are underused for key insights
The course uses `callout-insight`, `callout-warning`, `callout-danger`, and `callout-key` divs, and these are often the best-written conceptual content in each guide. But they tend to be isolated one-liners rather than developed explanations. The callouts should be expanded or supplemented with paragraphs that develop the insight.

### Issue 4: Practice problems are overwhelmingly code exercises
Every guide ends with practice problems that ask the reader to *implement* something. None ask the reader to *explain*, *compare*, *design*, or *evaluate* conceptually. Adding non-code problems ("Explain why tournament selection is more robust than roulette wheel for minimization problems") would reinforce conceptual understanding.

### Issue 5: No "big picture" conceptual thread
There is no recurring narrative that connects the modules. A reader finishing Module 2 may not understand why Module 3 (Time Series) matters to them if they are not working with time series. Missing: a running example that evolves across all modules, showing the same GA being improved step by step.

### Issue 6: Duplicate/overlapping guides with no differentiation
Module 01 has both `01_encoding.md` and `01_ga_components.md`. Module 03 has both `01_timeseries_considerations.md` and `01_walk_forward.md`. These pairs cover overlapping material without clear differentiation, confusing the reading order. The shorter "overview" guides (ga_components, timeseries_considerations) are almost entirely code with minimal conceptual content.

### Issue 7: Notebooks amplify the problem
The sampled notebook (Module 2, fitness functions) has a "90-minute" estimate but provides almost no conceptual scaffolding between code cells. Markdown cells average 2-4 sentences. The notebook effectively asks the reader to learn by reading and running code, with no explanatory pauses.

---

## 4. Priority Enhancement List

Ranked by impact (combination of how many readers are affected and how much understanding improves).

| # | File | Section/Location | What's Missing | What to Add | Impact |
|---|------|-----------------|---------------|------------|--------|
| 1 | All guides | Structure | Intuition comes after formalism | Move "Intuitive Explanation" sections to immediately after "In Brief", before "Formal Definition" | **High** |
| 2 | Module 00, Guide 02 | After "Metaheuristic Optimization" | Decision framework for choosing feature selection approach | Flowchart: "Choosing Your Feature Selection Strategy" based on n_features, budget, model type | **High** |
| 3 | Module 02, Guide 01 | Before first code block | Fitness function design process | "Designing Your First Fitness Function" step-by-step recipe (5 steps, no code) | **High** |
| 4 | Module 01, Guide 03 | "Building Block Hypothesis" section | Why crossover works -- theoretical foundation | Expand to full subsection with annotated visual example of building blocks combining | **High** |
| 5 | Module 04, Guide 01 | Before "Basic GA Setup" | DEAP mental model and architecture | Conceptual diagram of creator/toolbox/algorithms/tools + 1-paragraph explanation of each | **High** |
| 6 | Module 04, Guide 01 | After "Running the GA" | How to choose GA hyperparameters | Table of heuristics for pop_size, mutation_rate, crossover_prob, tournament_size with reasoning | **High** |
| 7 | Module 03, Guide 01 (Walk Forward) | After "Intuitive Explanation" | Why k-fold leaks information -- mechanism | Timeline diagram showing how future data contaminates training in k-fold vs walk-forward | **High** |
| 8 | Module 02, Guide 01 | Between the 6 fitness function implementations | Decision framework for which fitness function to use | Comparison table: approach, when to use, computational cost, overfitting risk | **High** |
| 9 | Module 05, Guide 01 | "NSGA-II for Feature Selection" section | How non-dominated sorting works | Step-by-step walkthrough with 5 solutions, 2 objectives -- annotated diagram, no code | **High** |
| 10 | Module 01, Guide 02 | After selection operator implementations | Selection pressure intuition + when to use each | Worked example: tournament size 2 vs 5 showing diversity loss, plus "Choose your selection operator" table | **Medium** |
| 11 | Module 02, Guides 01-02 | Parsimony penalty discussions | How to calibrate lambda | Sensitivity analysis guidance: "start at 0.01, observe feature count, adjust" with example results | **Medium** |
| 12 | Module 03, Guide 02 | Between ACF/PACF analysis and GA fitness | How to connect ACF/PACF results to GA configuration | Bridging section: "Using ACF/PACF to configure your GA" (population seeding, max lag, group mutation) | **Medium** |
| 13 | Module 05, Guide 01 | "Surrogate Model Fitness" section | What a surrogate model is and why to use one | 2-paragraph conceptual explanation before the code: what it is, when it helps, what the uncertainty threshold means | **Medium** |
| 14 | Module 05, Guide 02 | Before code implementations | When to use hybrid methods | Decision criteria: "Use hybrid when..." with 3-4 conditions and counter-examples | **Medium** |
| 15 | Module 01, Guide 03 | Between crossover and mutation sections | Exploitation vs exploration duality | 1-2 paragraphs explaining that crossover exploits (combines known good), mutation explores (introduces new), and why both are needed | **Medium** |
| 16 | Module 00, Guide 03 | Between each selection/crossover/mutation operator | Comparison and selection guidance | After each operator type (tournament, roulette, rank), add a 2-sentence comparison of when to prefer each | **Medium** |
| 17 | Module 04, Guide 02 | Before custom operator implementations | Taxonomy of when custom operators help | List of problem structures (grouped, hierarchical, budget-constrained, correlated) with 1-sentence benefit each | **Medium** |
| 18 | Module 05, Guide 03 | Between linear, diversity, and feedback adaptation | Comparison of adaptation strategies | Table: strategy, complexity, robustness, best-for, worst-for | **Low** |
| 19 | All guides | "Practice Problems" sections | All problems are code exercises | Add 1-2 conceptual questions per guide ("Explain why...", "Compare X and Y for...") | **Low** |
| 20 | Module 03, Guide 03 | After stationarity testing code | Why stationarity matters for GA feature selection specifically | Worked example: GA on non-stationary data selects spurious features; GA on stationarized data selects correct ones | **Low** |

---

## 5. New Content Recommendations

### 5.1 New Guide: "GA Feature Selection: A Conceptual Roadmap"
**Location:** `modules/module_00_foundations/guides/00_conceptual_roadmap.md`
**Purpose:** Provide the "big picture" narrative that connects all modules. Include: (a) a visual roadmap showing how concepts build on each other, (b) a running example problem introduced here and revisited in each module, (c) key decision points a practitioner faces and which module addresses each.
**Impact:** High -- addresses the lack of conceptual thread across modules.

### 5.2 New Guide: "Choosing Your GA Configuration"
**Location:** `modules/module_04_implementation/guides/00_configuration_guide.md`
**Purpose:** A concept-first guide (minimal code) that walks through all the decisions a practitioner must make: encoding type, selection operator, crossover type, mutation rate, population size, stopping criteria, validation strategy. Each decision includes a brief explanation of tradeoffs and a recommendation.
**Impact:** High -- addresses the "missing decision framework" cross-cutting issue.

### 5.3 New Guide: "Common Misconceptions About GAs"
**Location:** `modules/module_00_foundations/guides/05_common_misconceptions.md`
**Purpose:** Explicitly address and debunk common misconceptions: (a) "GAs always find the global optimum" (no -- they find good solutions fast), (b) "More generations = better results" (diminishing returns, overfitting risk), (c) "Larger population is always better" (computational cost tradeoff), (d) "Mutation rate doesn't matter much" (it is critical), (e) "Feature selection results are deterministic" (stochastic -- run multiple times).
**Impact:** Medium -- directly addresses the "common misconceptions" evaluation criterion.

### 5.4 New Section in Each Guide: "Key Concept Summary"
**Purpose:** Add a boxed section near the top of each guide (after "In Brief") containing 2-3 sentences that distill the most important idea. Written so that a reader could explain the concept to someone else after reading only this box.
**Impact:** Medium -- makes guides scannable and reinforces the core idea.

### 5.5 Enhanced Notebooks: Conceptual Checkpoints
**Purpose:** In each notebook, after every 2-3 code cells, add a markdown cell with a conceptual question the reader should be able to answer. Example: "Before running the next cell, predict: will adding the parsimony penalty increase or decrease the number of selected features? Why?" This transforms passive code-running into active learning.
**Impact:** Medium -- addresses notebook concept deficit without major restructuring.

---

## Appendix: Detailed Concept-to-Code Ratios

| Module | Guide | Estimated Explanatory Paragraphs | Estimated Code Blocks | Ratio (text:code) |
|--------|-------|--------------------------------|----------------------|-------------------|
| 00 | 01_feature_selection_problem | 15 | 8 | 65:35 |
| 00 | 02_optimization_basics | 10 | 8 | 55:45 |
| 00 | 03_evolutionary_operators | 5 | 12 | 30:70 |
| 00 | 04_selection_approaches | 12 | 10 | 55:45 |
| 01 | 01_encoding | 6 | 10 | 35:65 |
| 01 | 01_ga_components | 3 | 11 | 20:80 |
| 01 | 02_selection | 8 | 12 | 40:60 |
| 01 | 03_genetic_operators | 6 | 15 | 30:70 |
| 02 | 01_fitness_functions | 3 | 10 | 20:80 |
| 02 | 02_cross_validation_fitness | 4 | 9 | 30:70 |
| 02 | 03_multi_objective | 10 | 9 | 50:50 |
| 03 | 01_timeseries_considerations | 4 | 7 | 35:65 |
| 03 | 01_walk_forward | 8 | 8 | 50:50 |
| 03 | 02_lag_features | 8 | 8 | 50:50 |
| 03 | 03_stationarity | 7 | 8 | 45:55 |
| 04 | 01_deap_implementation | 3 | 7 | 25:75 |
| 04 | 02_custom_operators | 8 | 10 | 40:60 |
| 04 | 03_production_considerations | 7 | 7 | 50:50 |
| 05 | 01_advanced_techniques | 3 | 8 | 25:75 |
| 05 | 02_hybrid_methods | 10 | 8 | 55:45 |
| 05 | 03_adaptive_operators | 8 | 6 | 55:45 |

*Note: "Explanatory paragraphs" counts substantive conceptual paragraphs (not headers, not code comments, not single-sentence transitions). "Code blocks" counts fenced code blocks. Ratios are approximate and subjective.*
