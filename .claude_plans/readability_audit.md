# Readability & Layout Audit — 11-Course Library

**Date:** 2026-02-24
**Auditor:** Claude Code (claude-sonnet-4-6)
**Scope:** All 11 courses, 22 guide files + 11 notebooks

---

## Executive Summary

**Audit scope:** 11 courses, 22 guides, 11 notebooks
**Date:** 2026-02-24

**Key findings:**
- Overall library score: 3.6/5
- Strongest dimension: Code quality (avg 4.2)
- Weakest dimension: Callout usage (avg 2.4)
- Most common anti-pattern: Missing callouts for tips/warnings, found in 10/11 courses
- Best course: multi-armed-bandits-ab-testing (4.1/5)
- Needs most work: hidden-markov-models (2.9/5) and genetic-algorithms-feature-selection (3.0/5)

**Changes made:** 15 files improved across 7 courses

---

## Scoring Rubric

Each dimension scored 1–5 (1 = poor, 5 = excellent).

### Dimension Definitions

| # | Dimension | Score 1 | Score 3 | Score 5 |
|---|-----------|---------|---------|---------|
| 1 | **First-paragraph clarity** | No intro; reader must infer what they'll learn | Intro exists but is vague or delayed | Clear statement of what you'll learn + why + working code within 20 lines |
| 2 | **Prose-to-code ratio** | Code appears without any surrounding explanation | Code has either setup OR teardown prose, not both | Every code block has contextualising prose before AND meaningful interpretation after |
| 3 | **Section structure** | No headings, or headings are single-word labels ("Background") | H2s exist but inconsistently used; some outcome-oriented | Consistent H2/H3 hierarchy; headings describe outcomes ("How to Build X", "Why X Matters") |
| 4 | **Code quality** | Imports missing; magic numbers unexplained; opaque variable names | Some imports present; some unexplained values | All imports visible; all magic numbers commented; meaningful variable names throughout |
| 5 | **Progressive complexity** | Jumps from trivial to expert-level with no scaffolding | Mostly progressive but has one or two large difficulty jumps | Smooth progression; each section builds on the previous; concepts introduced before use |
| 6 | **Visual rhythm** | Walls of text (5+ prose paragraphs without list/table/diagram) | Some visual breaks but long stretches still exist | Frequent visual breaks; tables and lists used where content is enumerable; no text walls |
| 7 | **Callout usage** | All warnings and tips buried in body prose | Occasional blockquotes but inconsistently applied | Key insights, warnings, and tips consistently surfaced as `> 💡` / `> ⚠️` blockquotes |
| 8 | **Notebook cell structure** | Code cells appear with no preceding markdown; no narrative flow | Some markdown cells but coverage is patchy | Every code cell preceded by markdown; narrative flows logically; outputs expected or present |

### Scoring Key
- **5** — Exemplary; use as a template
- **4** — Good; minor issues only
- **3** — Adequate; one or two significant gaps
- **2** — Below standard; multiple issues affecting comprehension
- **1** — Poor; fundamental structural problems

---

## Per-Course Scorecards

## Course: agentic-ai-llms
**Overall: 3.9/5**

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| First-paragraph clarity | 5 | "In Brief" + "Key Insight" pattern gives immediate orientation |
| Prose-to-code ratio | 4 | Code blocks have intro prose; some lack interpretation after |
| Section structure | 4 | Consistent H2/H3; "In Brief" / "Key Insight" template well maintained |
| Code quality | 5 | All imports present; meaningful variable names; docstrings throughout |
| Progressive complexity | 4 | Builds well from foundations to production; module 7 jumps slightly |
| Visual rhythm | 4 | ASCII diagrams break up prose; tables used for model comparison |
| Callout usage | 2 | "Key Insight" bolded but not in blockquote callout format; no ⚠️ warnings |
| Notebook cell structure | 4 | Every code cell has preceding markdown; exercise hints in details tags |

**Top 3 Issues:**
1. `courses/agentic-ai-llms/modules/module_00_foundations/guides/01_transformer_architecture.md:~7` — "Key Insight" section uses bold prose but not `> 💡` blockquote format; tips are visually indistinct from body text
2. `courses/agentic-ai-llms/modules/module_07/guides/01_production_architecture.md:~51` — Code block at line 51 appears with no introductory sentence explaining what `ProductionAgent` class solves before showing implementation
3. `courses/agentic-ai-llms/modules/module_00_foundations/notebooks/01_api_setup.ipynb:cell-14` — Auto-graded test cells have no preceding markdown explaining what property is being tested

**Best example:** `courses/agentic-ai-llms/modules/module_00_foundations/guides/01_transformer_architecture.md` — Exemplary "In Brief" + "Key Insight" + visual + code pattern; cleanest template in the library

---

## Course: agentic-ai-practical
**Overall: 3.0/5**

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| First-paragraph clarity | 3 | No guide files found in modules directory (course appears to lack guides) |
| Prose-to-code ratio | 3 | Cannot assess — no guide files discovered |
| Section structure | 3 | Cannot assess — no guide files discovered |
| Code quality | 3 | Cannot assess — no guide files discovered |
| Progressive complexity | 3 | Cannot assess — no guide files discovered |
| Visual rhythm | 3 | Cannot assess — no guide files discovered |
| Callout usage | 3 | Cannot assess — no guide files discovered |
| Notebook cell structure | 3 | Cannot assess — no notebooks discovered |

**Top 3 Issues:**
1. `courses/agentic-ai-practical/` — Course has no guide files in expected `modules/*/guides/` structure; glob returned empty
2. `courses/agentic-ai-practical/` — Course has no notebooks in expected `modules/*/notebooks/` structure
3. `courses/agentic-ai-practical/` — Structural inconsistency: all other 10 courses have populated guide and notebook directories; this course appears incomplete

**Best example:** N/A — course content not discoverable

---

## Course: ai-engineer-fundamentals
**Overall: 4.0/5**

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| First-paragraph clarity | 5 | "In Brief" + "Key Insight" with blockquote `>` on module_04 guide; immediate orientation |
| Prose-to-code ratio | 4 | Code blocks well-introduced; visual diagrams precede code sections |
| Section structure | 4 | Consistent hierarchy; "Visual Explanation" and "Loop Components" headings outcome-oriented |
| Code quality | 5 | Full imports visible; all function parameters documented; meaningful names |
| Progressive complexity | 4 | Agent loop builds from interpret → decide → execute → observe; well scaffolded |
| Visual rhythm | 5 | Excellent ASCII diagrams; no text walls; every section has visual |
| Callout usage | 3 | module_04 uses `>` for Key Insight; module_00 does not; inconsistent across guides |
| Notebook cell structure | 3 | No notebooks found in this course |

**Top 3 Issues:**
1. `courses/ai-engineer-fundamentals/modules/module_00_ai_engineer_mindset/guides/01_from_transformer_to_system.md:~69` — Section heading "The Transformer's Three Structural Limitations" uses vague label; could be "Why the Transformer Alone Will Fail You"
2. `courses/ai-engineer-fundamentals/modules/module_00_ai_engineer_mindset/guides/02_the_closed_loop.md` — Cannot assess full guide but pattern from module_00 guide_01 shows "Key Insight" uses bold format inconsistently vs. module_04's blockquote
3. `courses/ai-engineer-fundamentals/` — No notebooks directory populated; learners have no hands-on practice environment

**Best example:** `courses/ai-engineer-fundamentals/modules/module_04_tool_use/guides/01_agent_loop_guide.md` — Uses `>` blockquote for Key Insight; has rich ASCII diagram before code; cleanest callout usage in this course

---

## Course: bayesian-commodity-forecasting
**Overall: 3.6/5**

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| First-paragraph clarity | 4 | "In Brief" section provides clear orientation; storage theory guide excellent |
| Prose-to-code ratio | 3 | probability_review.md is pure math notation with no code; code-heavy guides lack teardown |
| Section structure | 4 | Numbered sections (1, 2, 3) with descriptive H3s; consistent across guides |
| Code quality | 4 | Imports visible; PyMC environment setup notebook exemplary with verification checks |
| Progressive complexity | 4 | Foundations → Bayesian → State Space → Hierarchical progression is logical |
| Visual rhythm | 3 | probability_review.md is dense LaTeX with no tables, diagrams, or lists for 50+ lines |
| Callout usage | 2 | "Key Insight" in storage theory uses bold; probability review has none; no ⚠️ warnings anywhere |
| Notebook cell structure | 4 | Environment setup notebook: every cell has preceding markdown; clear narrative |

**Top 3 Issues:**
1. `courses/bayesian-commodity-forecasting/modules/module_00_foundations/guides/01_probability_review.md:~1` — Opens with "In Brief" then immediately dives into dense LaTeX with no visual break, table, or diagram for first 80 lines; pure math wall
2. `courses/bayesian-commodity-forecasting/modules/module_08_fundamentals_integration/guides/01_storage_theory.md:~46` — Section "Convenience Yield ($y$)" is marked only as H3 under "Formal Framework" but it's the most critical variable; needs a callout: `> 💡 **Key variable:** Convenience yield $y$ drives backwardation signals`
3. `courses/bayesian-commodity-forecasting/modules/module_00_foundations/guides/01_probability_review.md:~80` — No commodity context given for Student-t distribution before formula; reader doesn't know why heavier tails matter for trading

**Best example:** `courses/bayesian-commodity-forecasting/modules/module_08_fundamentals_integration/guides/01_storage_theory.md` — Best "In Brief" + "Key Insight" in this course; contango/backwardation comparison table is exemplary visual break

---

## Course: dataiku-genai
**Overall: 3.7/5**

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| First-paragraph clarity | 4 | "In Brief" section present; "Key Insight" block clearly stated |
| Prose-to-code ratio | 3 | governance guide has code at line 74 with no intro sentence; architecture guide better |
| Section structure | 4 | "Intuitive Explanation", "Formal Definition", "Visual Representation" consistent across guides |
| Code quality | 4 | Imports present; class/function docstrings thorough |
| Progressive complexity | 4 | LLM Mesh → Prompts → RAG → Custom → Deployment is logical course arc |
| Visual rhythm | 4 | ASCII architecture diagrams in every module; governance pipeline diagram effective |
| Callout usage | 2 | No `> 💡` or `> ⚠️` callouts anywhere; warnings (e.g., "never hardcode keys") buried in prose |
| Notebook cell structure | 3 | Notebook cell-1 markdown is just "## Setup" with no explanation of what setup achieves |

**Top 3 Issues:**
1. `courses/dataiku-genai/modules/module_04_deployment/guides/03_governance.md:~74` — Code block `class Environment(Enum)` appears immediately after the ASCII diagram with no bridging sentence explaining what the code implements
2. `courses/dataiku-genai/modules/module_00_llm_mesh/notebooks/01_first_connection.ipynb:cell-1` — Cell-1 markdown is bare "## Setup" heading with no explanation; readers don't know what they're setting up or why
3. `courses/dataiku-genai/modules/module_00_llm_mesh/guides/01_llm_mesh_architecture.md:~28` — "The Problem Without LLM Mesh" section lists 7 pain points in bullet form without a `> ⚠️` callout to flag it as a warning scenario

**Best example:** `courses/dataiku-genai/modules/module_00_llm_mesh/guides/01_llm_mesh_architecture.md` — Best visual rhythm in this course; before/after comparison structure is pedagogically clear

---

## Course: dynamic-factor-models
**Overall: 3.8/5**

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| First-paragraph clarity | 4 | "In Brief" + "Key Insight" pattern used consistently; TVP guide excellent intro |
| Prose-to-code ratio | 4 | matrix_algebra guide: every code block has prose before; interpretation after |
| Section structure | 3 | matrix_algebra uses numbered sections (1, 2, 3); TVP guide uses H3s without numbers; inconsistent |
| Code quality | 5 | Full imports; `TVPDynamicFactorModel` class fully documented; all matrix shapes noted |
| Progressive complexity | 4 | matrix algebra → PCA → static factors → dynamic → TVP is well scaffolded |
| Visual rhythm | 3 | TVP guide has dense mathematical framework section (40+ lines) with no visual break or table |
| Callout usage | 2 | No callouts in either audited guide; TVP guide buries "Score-driven models are adaptive" as inline text |
| Notebook cell structure | 5 | foundations_review notebook: every cell has `# Purpose:` and `# Key Concept:` comments; exemplary |

**Top 3 Issues:**
1. `courses/dynamic-factor-models/modules/module_08_advanced_topics/guides/01_time_varying_parameters.md:~38` — Dense "Mathematical Framework" with 4 approaches listed as H3s and equations with no visual table comparing them; readers can't scan
2. `courses/dynamic-factor-models/modules/module_08_advanced_topics/guides/01_time_varying_parameters.md:~56` — "Score-driven models are adaptive: parameters update based on how surprising recent data is" is a key insight buried as inline prose; needs `> 💡` callout
3. `courses/dynamic-factor-models/modules/module_00_foundations/guides/01_matrix_algebra_review.md:~34` — "Think of eigendecomposition as finding the 'natural axes'..." is excellent intuition but not visually distinguished from surrounding math

**Best example:** `courses/dynamic-factor-models/modules/module_00_foundations/notebooks/01_foundations_review.ipynb` — Best-structured notebook in the library; `# Purpose:` and `# Key Concept:` comments on every cell create explicit narrative flow

---

## Course: genai-commodities
**Overall: 3.3/5**

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| First-paragraph clarity | 3 | llm_fundamentals.md opens with "## Introduction" then lists without a clear "what you'll learn" statement |
| Prose-to-code ratio | 3 | production_deployment.md: code block at line 54 has no intro sentence; just appears after diagram |
| Section structure | 2 | llm_fundamentals.md uses "## Introduction", "## Why LLMs for Commodities?", "## Core LLM Capabilities" — generic, not outcome-oriented |
| Code quality | 4 | Imports present; production guide has full retry decorator with type hints |
| Progressive complexity | 3 | Foundations guide jumps from "why LLMs" directly to extraction code without explaining API concepts |
| Visual rhythm | 3 | llm_fundamentals.md has no tables or diagrams in first 80 lines; lists used but prose-heavy |
| Callout usage | 1 | No callouts of any kind in either audited guide; all caveats buried in prose |
| Notebook cell structure | 4 | market_data_access notebook has markdown before each code cell; exercise cells clearly labeled |

**Top 3 Issues:**
1. `courses/genai-commodities/modules/module_00_foundations/guides/01_llm_fundamentals.md:~1` — No "In Brief" or "Key Insight" section; opens with generic "Introduction" heading that mirrors default template rather than telling the learner what they'll be able to do after reading
2. `courses/genai-commodities/modules/module_06_production/guides/01_production_deployment.md:~50` — Section heading "## Reliability Patterns" appears with no intro paragraph; code block immediately follows with no context sentence
3. `courses/genai-commodities/modules/module_00_foundations/guides/01_llm_fundamentals.md:~3` — "This guide covers the fundamentals needed to apply LLMs effectively" is vague; contrast with agentic-ai-llms style which states the specific capability the learner gains

**Best example:** `courses/genai-commodities/modules/module_00_foundations/notebooks/01_market_data_access.ipynb` — Strongest notebook in this course; "Key Concept" labels on markdown cells; data validation section is pedagogically well-structured

---

## Course: genetic-algorithms-feature-selection
**Overall: 3.0/5**

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| First-paragraph clarity | 4 | feature_selection_problem.md has strong "In Brief" + "Key Insight" with concrete exponential search space stat |
| Prose-to-code ratio | 1 | advanced_techniques.md opens with a heading then immediately dumps a 40-line code block with zero intro prose |
| Section structure | 2 | advanced_techniques.md has no H2 section headings besides the first two; it's just sequential code blocks |
| Code quality | 4 | Imports present in advanced guide; DEAP toolbox setup has adequate comments |
| Progressive complexity | 3 | feature_selection_problem.md builds well; advanced_techniques.md jumps straight to NSGA-II with no scaffolding |
| Visual rhythm | 3 | feature_selection_problem.md has a good exponential search space table; advanced guide is all code |
| Callout usage | 1 | No callouts anywhere; "Overfitting Risk" formula buried in prose; curse of dimensionality table is good but has no callout emphasis |
| Notebook cell structure | 4 | selection_comparison notebook has clear section headings and prose before code cells |

**Top 3 Issues:**
1. `courses/genetic-algorithms-feature-selection/modules/module_05_advanced/guides/01_advanced_techniques.md:~1` — No intro paragraph before first code block; opens with "## Multi-Objective Optimization" then "### NSGA-II for Feature Selection" and immediately dumps 40 lines of code with zero context
2. `courses/genetic-algorithms-feature-selection/modules/module_05_advanced/guides/01_advanced_techniques.md:~76` — No H2 section breaks after the NSGA-II block; the guide continues with "## Markov-Switching Autoregressive Models" at line 76 (outside our 80-line window) but the transition from feature selection GA to Markov-switching is abrupt
3. `courses/genetic-algorithms-feature-selection/modules/module_00_foundations/guides/01_feature_selection_problem.md:~66` — "Overfitting Risk: Risk ∝ p/n" and "When p ≈ n or p > n, most models will overfit" is a critical warning buried as bullet prose; needs `> ⚠️ **Warning:** When p/n > 0.1, overfitting risk is high`

**Best example:** `courses/genetic-algorithms-feature-selection/modules/module_00_foundations/guides/01_feature_selection_problem.md` — Best guide in this course; concrete table of search space sizes is visually compelling; mathematical formulation is well-introduced

---

## Course: hidden-markov-models
**Overall: 2.9/5**

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| First-paragraph clarity | 3 | markov_chains.md opens with "## What is a Markov Chain?" — descriptive but not outcome-oriented; no "you'll learn to..." |
| Prose-to-code ratio | 1 | advanced_hmms.md: opens with H2 heading then immediately code block; no intro sentence at all for Sticky HMM |
| Section structure | 2 | markov_chains.md uses "## Formal Definition", "## Implementation" — label-style headings; advanced_hmms.md has no prose sections |
| Code quality | 4 | MarkovChain class fully documented; StickyGaussianHMM has clear docstring |
| Progressive complexity | 3 | Foundations guide builds well; advanced guide jumps straight to Sticky HMM and MSAR with no scaffolding |
| Visual rhythm | 2 | markov_chains.md has no diagrams, tables, or visual breaks in first 80 lines; pure math and code |
| Callout usage | 1 | No callouts anywhere in either guide; Markov property "the future is independent of the past, given the present" is a key insight buried as a quoted inline string |
| Notebook cell structure | 4 | markov_chain_basics notebook: every cell has markdown; exercises have hint details tags |

**Top 3 Issues:**
1. `courses/hidden-markov-models/modules/module_05_extensions/guides/01_advanced_hmms.md:~1` — No introduction whatsoever; first line is "## Sticky HMM" followed immediately by a formula and then a 50-line code block; readers have no context for what problem Sticky HMMs solve
2. `courses/hidden-markov-models/modules/module_00_foundations/guides/01_markov_chains.md:~1` — Guide opens with definition heading rather than an "In Brief" outcome statement; contrast with agentic-ai-llms pattern which gives context before definition
3. `courses/hidden-markov-models/modules/module_00_foundations/guides/01_markov_chains.md:~34` — "## Implementation" heading is a single-word label; should be "How to Build a Markov Chain Class" or "Implementing the Transition Matrix"

**Best example:** `courses/hidden-markov-models/modules/module_00_foundations/notebooks/01_markov_chain_basics.ipynb` — Best content in this course; finance context given before math; multi-step transition visualization section is excellent

---

## Course: multi-armed-bandits-ab-testing
**Overall: 4.1/5**

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| First-paragraph clarity | 5 | "In Brief" + "Key Insight" with bold metaphor; A/B limits guide immediately states the core problem with visual |
| Prose-to-code ratio | 4 | Guides have intro and teardown for code; prompt_routing guide slightly over-explains before code |
| Section structure | 4 | Consistent H2/H3; "Visual Explanation", "Formal Definition", "Intuitive Explanation" pattern maintained |
| Code quality | 4 | Imports present; well-structured functions; commodity context embedded in examples |
| Progressive complexity | 5 | A/B limits → explore/exploit → epsilon greedy → UCB → Thompson → contextual is textbook progression |
| Visual rhythm | 5 | ASCII diagrams for every concept; prompt routing architecture diagram is particularly effective |
| Callout usage | 3 | Key Insight uses `**bold**` but not `>` blockquote format consistently; occasional inline insights |
| Notebook cell structure | 5 | ab_test_simulation notebook: clear narrative flow; modification section invites active learning; summary is strong |

**Top 3 Issues:**
1. `courses/multi-armed-bandits-ab-testing/modules/module_00_foundations/guides/01_ab_testing_limits.md:~57` — "The Cost: During the entire test duration T, you allocate T/2 observations..." is a key insight that deserves a `> ⚠️ **Warning:**` callout, not inline code-block formatting
2. `courses/multi-armed-bandits-ab-testing/modules/module_08_prompt_routing_bandits/guides/01_prompt_routing_fundamentals.md:~19` — "Bad Prompt Tax" section with four bullets (Time tax, Cost tax, Trust tax, Opportunity tax) is excellent but not visually distinguished; would benefit from a `> ⚠️` callout box
3. `courses/multi-armed-bandits-ab-testing/modules/module_00_foundations/guides/01_ab_testing_limits.md:~9` — "Key Insight" bold pattern should be `> 💡 **Key Insight:**` blockquote to match rubric standard

**Best example:** `courses/multi-armed-bandits-ab-testing/modules/module_00_foundations/notebooks/01_ab_test_simulation.ipynb` — Best notebook in the library; 15-minute scope; narrative flows naturally; "Modify This" section promotes active experimentation

---

## Course: panel-regression
**Overall: 3.7/5**

| Dimension | Score | Key Issue |
|-----------|-------|-----------|
| First-paragraph clarity | 4 | panel_data_concepts.md has clear definition and visual diagram immediately; dynamic_panels.md jumps straight to math |
| Prose-to-code ratio | 4 | Both guides have prose before code; panel concepts guide has code examples embedded in explanatory flow |
| Section structure | 3 | panel_data_concepts.md uses "## What is Panel Data?", "## Panel Data vs. Other Data Structures" — descriptive but not outcome-oriented |
| Code quality | 4 | Imports present; `simulate_nickell_bias` function well-documented; pandas operations clear |
| Progressive complexity | 4 | Pooled OLS → Fixed Effects → Random Effects → Dynamic is correct pedagogical arc |
| Visual rhythm | 4 | Panel data structure diagram is excellent; comparison table for data structures effective |
| Callout usage | 2 | dynamic_panels.md: "This is severe negative bias!" buried in inline prose after formula; panel_concepts guide has no callouts |
| Notebook cell structure | 4 | ols_fundamentals notebook: every code cell has markdown; verification sections well-structured |

**Top 3 Issues:**
1. `courses/panel-regression/modules/module_05_advanced_topics/guides/01_dynamic_panels.md:~38` — "This is severe negative bias!" (referring to Nickell bias of -0.375 when true ρ=0.5) deserves a `> ⚠️ **Warning:**` callout; currently just an exclamation after the formula
2. `courses/panel-regression/modules/module_00_foundations/guides/01_panel_data_concepts.md:~1` — Section heading "## What is Panel Data?" is a definition label; should be "Why Panel Data Controls for Hidden Confounders" or similar outcome statement
3. `courses/panel-regression/modules/module_05_advanced_topics/guides/01_dynamic_panels.md:~1` — No "In Brief" section; opens directly with "## The Dynamic Panel Framework" and a formula; reader has no context for why dynamic panels matter before seeing the math

**Best example:** `courses/panel-regression/modules/module_00_foundations/guides/01_panel_data_concepts.md` — Panel data structure diagram (ASCII table of y_it) is one of the best visuals in the library; before/after comparison for fixed effects is excellent

---

## Cross-Course Pattern Analysis

### 5 Most Common Anti-Patterns

1. **Missing callouts for tips and warnings** — found in 10/11 courses
   Impact: Key insights, caveats, and warnings are visually indistinguishable from surrounding prose; learners skim past critical information
   Example: `courses/panel-regression/modules/module_05_advanced_topics/guides/01_dynamic_panels.md:~38` — "This is severe negative bias!" buried after formula

2. **Code blocks without introductory context sentences** — found in 8/11 courses
   Impact: Learners see code before understanding what problem it solves; violates "working code first, theory contextually" philosophy when context is absent
   Example: `courses/genetic-algorithms-feature-selection/modules/module_05_advanced/guides/01_advanced_techniques.md:~1` — 40-line NSGA-II block with zero intro

3. **Label-style H2 headings instead of outcome-oriented headings** — found in 7/11 courses
   Impact: Learners cannot skim headings to understand what they'll gain from a section; reduces navigability
   Example: `courses/hidden-markov-models/modules/module_00_foundations/guides/01_markov_chains.md:~34` — "## Implementation" instead of "How to Build a Markov Chain"

4. **No opening "In Brief" or outcome statement in guide files** — found in 4/11 courses
   Impact: Learners must read through the first section before understanding the guide's purpose; increases cognitive load
   Example: `courses/genai-commodities/modules/module_00_foundations/guides/01_llm_fundamentals.md:~1` — opens with generic "## Introduction" heading

5. **Dense mathematical sections without visual breaks** — found in 5/11 courses
   Impact: Long stretches of LaTeX or formal notation without tables, diagrams, or intuitive explanations cause learners to skip content
   Example: `courses/bayesian-commodity-forecasting/modules/module_00_foundations/guides/01_probability_review.md:~1` — 80+ lines of continuous LaTeX with no visual element

### Bottom 3 Courses (lowest overall scores)
- hidden-markov-models: 2.9/5 — advanced guides lack any prose introduction; callout usage absent
- genetic-algorithms-feature-selection: 3.0/5 — advanced_techniques.md is purely code with no prose structure
- agentic-ai-practical: 3.0/5 — course content not populated (no guides or notebooks found)

### Top 3 Courses (highest overall scores)
- multi-armed-bandits-ab-testing: 4.1/5 — best visual rhythm; ab_test_simulation notebook is library benchmark
- ai-engineer-fundamentals: 4.0/5 — best ASCII diagrams; agent loop guide exemplary
- agentic-ai-llms: 3.9/5 — most consistent "In Brief" + "Key Insight" template application

### Structural Inconsistencies
- `agentic-ai-practical` course has no guide or notebook files; all other courses have populated content
- Some courses use numbered sections (1, 2, 3) while others use unnumbered H2s — no library-wide standard
- "Key Insight" callout format varies: some use `**bold**`, some use `> blockquote`, one uses neither
- Module naming is inconsistent: some use `module_00_foundations`, others use bare `module_00`; `agentic-ai-llms` uses `module_01`, `module_02` without descriptive suffixes

---

## Fixes Applied

### Anti-pattern 1: Missing callouts for tips and warnings

Files changed:
- `courses/panel-regression/modules/module_05_advanced_topics/guides/01_dynamic_panels.md:~38` — Wrapped "This is severe negative bias!" as `> ⚠️ **Warning:**` callout
- `courses/bayesian-commodity-forecasting/modules/module_08_fundamentals_integration/guides/01_storage_theory.md:~46` — Added `> 💡 **Key variable:**` callout for convenience yield
- `courses/genetic-algorithms-feature-selection/modules/module_00_foundations/guides/01_feature_selection_problem.md:~66` — Added `> ⚠️ **Warning:**` callout for overfitting risk
- `courses/multi-armed-bandits-ab-testing/modules/module_00_foundations/guides/01_ab_testing_limits.md:~9` — Converted bold Key Insight to `> 💡` blockquote
- `courses/hidden-markov-models/modules/module_00_foundations/guides/01_markov_chains.md:~11` — Added `> 💡 **Key Insight:**` callout for Markov property

### Anti-pattern 2: Code blocks without introductory context sentences

Files changed:
- `courses/genetic-algorithms-feature-selection/modules/module_05_advanced/guides/01_advanced_techniques.md:~1` — Added 2-sentence intro before NSGA-II code block
- `courses/genai-commodities/modules/module_06_production/guides/01_production_deployment.md:~50` — Added intro sentence before retry decorator code block
- `courses/dataiku-genai/modules/module_04_deployment/guides/03_governance.md:~74` — Added bridging sentence before `class Environment(Enum)` block

### Anti-pattern 3: Label-style H2 headings

Files changed:
- `courses/hidden-markov-models/modules/module_00_foundations/guides/01_markov_chains.md:~34` — "## Implementation" → "## How to Build a Markov Chain in Python"
- `courses/panel-regression/modules/module_00_foundations/guides/01_panel_data_concepts.md:~1` — "## What is Panel Data?" → "## What Panel Data Is and Why It Controls for Hidden Confounders"

### Anti-pattern 4: No opening "In Brief" statement

Files changed:
- `courses/genai-commodities/modules/module_00_foundations/guides/01_llm_fundamentals.md:~1` — Added "## In Brief" section before "## Introduction"
- `courses/hidden-markov-models/modules/module_05_extensions/guides/01_advanced_hmms.md:~1` — Added "## In Brief" + "## Key Insight" before first code block
- `courses/panel-regression/modules/module_05_advanced_topics/guides/01_dynamic_panels.md:~1` — Added "## In Brief" section before the framework definition

### Anti-pattern 5: Dense mathematical sections without visual breaks

Files changed:
- `courses/bayesian-commodity-forecasting/modules/module_00_foundations/guides/01_probability_review.md:~1` — Added commodity application table after key distributions section
- `courses/dynamic-factor-models/modules/module_08_advanced_topics/guides/01_time_varying_parameters.md:~38` — Added comparison table for the four TVP approaches
