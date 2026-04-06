# GA Feature Selection: A Conceptual Roadmap

> **Reading time:** ~10 min | **Module:** 0 — Foundations | **Prerequisites:** None

## In Brief

This roadmap provides the big-picture narrative connecting all six modules. It introduces a running example -- selecting features for commodity price forecasting -- that reappears throughout the course, and maps the key decisions a practitioner faces to the module that addresses each.

<div class="callout-key">
<strong>Key Concept:</strong> Genetic algorithms solve feature selection by treating it as an evolutionary search problem. A population of candidate feature subsets competes, combines, and mutates over generations until a high-performing, parsimonious subset emerges. Every module in this course adds one essential piece to making that process work reliably on real data.
</div>

## The Running Example: Commodity Price Forecasting

Throughout this course, we follow a single problem:

**Predict next-week crude oil price direction** using up to **50 candidate features** drawn from:

| Category | Example Features | Count |
|----------|-----------------|-------|
| Price lags | 1-day, 5-day, 10-day, 20-day returns | 8 |
| Technical indicators | RSI, MACD, Bollinger Band width, ATR | 10 |
| Fundamental data | Inventory levels, rig counts, OPEC production | 8 |
| Macro indicators | USD index, 10Y yield, VIX, PMI | 8 |
| Seasonal/calendar | Month, day-of-week, quarter, holiday flag | 6 |
| Cross-commodity | Natural gas, gold, copper prices and spreads | 10 |

With 50 features, there are $2^{50} \approx 10^{15}$ possible subsets -- exhaustive search would take roughly 35,000 years at 1 ms per evaluation. A genetic algorithm can find a strong subset in minutes.

But building a GA that works reliably on this problem requires answering a series of questions. Each question maps to a module.

## The Decision Roadmap

```
                    ┌─────────────────────────────┐
                    │  START: 50 candidate features│
                    │  Goal: predict oil prices    │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  MODULE 0: FOUNDATIONS       │
                    │  "Why can't I just try all   │
                    │   subsets or use Lasso?"      │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  MODULE 1: GA FUNDAMENTALS   │
                    │  "How do I encode, select,   │
                    │   cross, and mutate feature  │
                    │   subsets?"                   │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  MODULE 2: FITNESS FUNCTIONS │
                    │  "How do I score a feature   │
                    │   subset fairly?"             │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  MODULE 3: TIME SERIES       │
                    │  "How do I avoid leaking     │
                    │   future data into fitness?"  │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  MODULE 4: IMPLEMENTATION    │
                    │  "How do I build this in     │
                    │   production code with DEAP?" │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  MODULE 5: ADVANCED TOPICS   │
                    │  "How do I handle multiple   │
                    │   objectives and scale up?"   │
                    └──────────────────────────────┘
```

## How Concepts Build on Each Other

### Module 0: Foundations -- The Problem Landscape

**Central question:** *Why do I need a genetic algorithm at all?*

You learn that with 50 features, exhaustive search is impossible. Greedy methods like forward selection are fast but miss feature interactions -- for instance, RSI and Bollinger Band width might be worthless individually but powerful together. Filter methods (correlation, mutual information) evaluate features in isolation, so they miss these interactions too. Embedded methods (Lasso) handle interactions within their own model structure but impose assumptions that may not match your final model.

This module establishes the motivation: you need a search strategy that explores multiple feature subsets simultaneously, discovers non-obvious interactions, and escapes local optima. That is what GAs do.

**Running example checkpoint:** You try forward selection on the 50 commodity features and find it selects 12 features with 58% accuracy. You suspect there are better subsets it missed because it cannot evaluate feature interactions.

### Module 1: GA Fundamentals -- The Machinery

**Central question:** *How does a GA search through feature subsets?*

Each candidate subset is encoded as a binary chromosome: `[1,0,0,1,1,0,...,1]` where 1 means "include this feature." A population of 100 such chromosomes evolves:

1. **Selection** picks the better-performing subsets as parents
2. **Crossover** combines features from two parents into offspring
3. **Mutation** randomly flips features on/off to explore new territory

The key insight is the **exploitation-exploration balance**: crossover exploits what the population already knows (combining proven feature groups), while mutation explores what it does not (trying features no one has tested). Remove crossover and you get random search. Remove mutation and you get premature convergence.

**Running example checkpoint:** You encode the 50 features as a binary chromosome and run a basic GA for 50 generations with population size 100. It finds a 9-feature subset with 63% accuracy -- better than forward selection, and it discovered that the MACD-VIX interaction is valuable.

### Module 2: Fitness Functions -- The Scorecard

**Central question:** *How do I evaluate whether a feature subset is good?*

The fitness function is the single most consequential design decision. A naive fitness function (train accuracy) leads the GA to select all 50 features. A well-designed one balances prediction accuracy against feature count, uses cross-validation to prevent overfitting, and penalizes complexity.

You also face a multi-objective design choice: do you combine accuracy and parsimony into a single score (weighted sum), or do you optimize both simultaneously and choose from the Pareto frontier?

**Running example checkpoint:** You switch from simple train MSE to 5-fold cross-validated MSE with a parsimony penalty. The GA now selects 7 features instead of 15, and out-of-sample accuracy improves from 61% to 65% because overfitting is reduced.

### Module 3: Time Series -- The Temporal Trap

**Central question:** *Why does standard cross-validation lie about feature quality in time series?*

Standard k-fold CV shuffles data randomly, so future price information leaks into training folds. A feature subset that "works" under k-fold may fail catastrophically on truly unseen future data. Walk-forward validation, expanding windows, and embargo gaps address this.

This module also covers stationarity: non-stationary features (raw price levels) create spurious correlations that fool the GA into selecting them. Differencing and stationarity testing prevent this.

**Running example checkpoint:** You replace k-fold with walk-forward validation and add stationarity checks. The GA drops raw price level features (non-stationary, spuriously correlated) and selects differenced returns and stationary indicators instead. Out-of-sample accuracy improves to 67%.

### Module 4: Implementation -- Production Code

**Central question:** *How do I build this properly with DEAP and make it production-ready?*

Moving from prototype to production means using the DEAP library (or similar) for efficient GA execution, implementing custom operators for domain-specific constraints (e.g., grouped features, budget limits), adding caching and parallelization for expensive fitness evaluations, and wrapping everything in a scikit-learn-compatible interface.

**Running example checkpoint:** You implement the commodity feature selector in DEAP with parallel fitness evaluation (8 cores), reducing runtime from 45 minutes to 6 minutes. You wrap it as a scikit-learn transformer for use in your production pipeline.

### Module 5: Advanced Topics -- Scaling and Refinement

**Central question:** *How do I handle multiple objectives, adaptive parameters, and hybrid strategies?*

NSGA-II enables true multi-objective optimization (accuracy vs. complexity vs. computational cost). Adaptive operators automatically tune mutation and crossover rates as the search progresses. Hybrid methods combine GA search with local optimization for faster convergence.

**Running example checkpoint:** You apply NSGA-II to simultaneously optimize accuracy and feature count, producing a Pareto frontier of 12 non-dominated solutions ranging from 3 features (62% accuracy) to 11 features (69% accuracy). You select the "knee" solution with 7 features and 67% accuracy as the best balance.

## Practitioner Decision Map

When working on your own feature selection problem, you face a series of decisions. This table maps each decision to the module that covers it.

| Decision | Options | Guidance | Module |
|----------|---------|----------|--------|
| Do I need a GA at all? | Exhaustive, greedy, filter, GA | If n_features < 20: try exhaustive. If 20-100: GA shines. If >100: filter first, then GA. | 0 |
| Binary or integer encoding? | Binary, integer | Binary is the default. Integer only when selecting very few from very many (e.g., 5 from 10,000). | 1 |
| Which selection operator? | Tournament, roulette, rank | Tournament (size 3-5) is the robust default. | 1 |
| Which crossover operator? | Single-point, uniform, scattered | Uniform for feature selection (no positional structure). | 1 |
| Single or multi-objective? | Weighted sum, Pareto (NSGA-II) | Single objective for fast prototyping. Multi-objective when you need to explore the accuracy-complexity tradeoff. | 2, 5 |
| How to validate fitness? | k-fold, walk-forward, expanding window | Walk-forward for time series. k-fold for cross-sectional data. | 3 |
| Library choice? | DEAP, custom, sklearn-genetic | DEAP for flexibility. sklearn-genetic for quick integration. | 4 |
| Fixed or adaptive parameters? | Fixed, linear decay, feedback-based | Start with fixed. Switch to adaptive if convergence stalls. | 5 |

## How to Use This Course

**If you are new to GAs:** Read modules 0 through 3 in order. Each builds on the previous. Then read Module 4 to implement. Return to Module 5 when your basic GA is working and you want to improve it.

**If you already know GA basics:** Skim Module 0-1, focus on Module 2 (fitness design) and Module 3 (time series validation) -- these are where most real-world GA feature selection projects fail. Module 4-5 provides production patterns.

**If you have a specific problem:** Use the Practitioner Decision Map above to identify which modules address your current decisions, and start there.

---

**Next:** [The Feature Selection Problem](./01_feature_selection_problem.md)
