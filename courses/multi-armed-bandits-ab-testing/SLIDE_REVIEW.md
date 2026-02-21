# Slide Deck Review: Multi-Armed Bandits for A/B Testing & Commodity Trading

**Reviewer:** Course Quality Reviewer (Automated)
**Date:** 2026-02-20
**Course:** `courses/multi-armed-bandits-ab-testing/`
**Total Decks Reviewed:** 38 across 9 modules

---

## Executive Summary

**Overall Score: 82/100 -- Strong**

This is a well-executed slide deck collection that demonstrates consistent structural discipline and strong domain integration. The course successfully bridges mathematical theory (bandit algorithms) with two practical domains (content optimization and commodity trading) and caps with an innovative prompt-routing application. The Marp framework is used competently throughout, with valid front matter, Mermaid diagrams, LaTeX math, and two-column layouts.

**Strengths:**
- Exceptionally consistent structure across all 38 decks (title, In Brief, Key Insight, Visual, Formal Definition, Code, Pitfalls, Connections, Visual Summary)
- Strong commodity trading domain integration -- not generic ML examples but real-world commodity context throughout
- Every deck has multiple Mermaid diagrams providing genuine visual value
- Code examples are practical, well-sized (under 15 lines per block), and copy-paste ready
- Pitfall sections are genuinely useful, not filler -- they reflect real practitioner mistakes
- Module 08 (Prompt Routing Bandits) is the standout module -- novel, practical, and well-argued

**Weaknesses:**
- Several decks are structurally identical to their companion guides with minimal adaptation for slide format
- Some Mermaid diagrams are too dense for slide projection (5+ nested subgraphs)
- Formal definition slides tend to be text-heavy walls of LaTeX without sufficient visual scaffolding
- Practice problems across modules are repetitive in format (always "implement X", "compare Y vs Z")
- Cheatsheet decks are inconsistent in depth -- some are comprehensive reference cards, others are thin summaries
- No speaker notes anywhere -- limits presentation utility

**Best Decks:** Module 08 Guide 04 (Commodity Research Assistant), Module 05 Guide 01 (Accumulator Bandit Playbook), Module 06 Guide 03 (Adversarial Bandits)
**Weakest Decks:** Module 00 Cheatsheet, Module 02 Guide 02 (Posterior Updating), Module 04 Guide 02 (Conversion Optimization)

---

## Dimension Scores

| Dimension | Weight | Score | Weighted |
|-----------|--------|-------|----------|
| Design & Visual Quality | 20% | 80/100 | 16.0 |
| Narrative & Story Flow | 20% | 84/100 | 16.8 |
| Comprehensiveness | 20% | 85/100 | 17.0 |
| Technical Accuracy | 20% | 82/100 | 16.4 |
| Added Visual Value | 10% | 78/100 | 7.8 |
| Production Readiness | 10% | 76/100 | 7.6 |
| **TOTAL** | **100%** | | **81.6** |

---

## Dimension Analysis

### 1. Design & Visual Quality (80/100)

**What works:**
- Consistent Marp front matter (`marp: true`, `theme: default`, `paginate: true`, `math: mathjax`) across all 38 decks
- Standard CSS for `.columns` grid layout reused everywhere
- Clean separation between lead slides (`<!-- _class: lead -->`) and content slides
- Two-column layouts used effectively for comparisons (Thompson vs UCB, YES vs NO lists)
- Tables are well-formatted with clear headers

**What does not work:**
- Font size fixed at 24px globally -- too small for large lecture halls, too large for personal screens
- No responsive design consideration -- decks are designed for one screen size only
- Color palette in Mermaid diagrams uses only 3 colors (#36f, #6f6, #f66) across all 38 decks -- visually monotonous
- No custom theme -- relies entirely on Marp default theme, making all decks look identical to any other Marp deck
- No images or photographs -- 100% text, code, tables, and Mermaid diagrams
- Some slides have too much content for effective projection (especially formal definition slides)

**Specific issues:**
- Module 05, Guide 01 (Accumulator Bandit Playbook): The TwoWalletBandit code slide has ~20 lines -- borderline for slide readability
- Module 07, Guide 01 (System Architecture): The 6-component Mermaid diagram is too dense for a single slide
- Module 03, Guide 02 (LinUCB): The code slide contains a 15-line class that would be difficult to read in presentation

### 2. Narrative & Story Flow (84/100)

**What works:**
- Every deck follows a predictable and effective arc: motivating problem -> intuitive analogy -> formal definition -> code -> pitfalls -> connections
- "In Brief" sections on slide 2 consistently set context in 2-3 sentences
- Analogies are creative and memorable: restaurant dilemma (explore-exploit), newsroom editor (contextual routing), casino floor (Thompson Sampling), two-wallet investor (commodity bandits)
- Module progression is logical: foundations -> algorithms -> Bayesian -> contextual -> domain applications -> advanced -> production -> capstone
- Cross-module connections are explicit in every deck's "Connections" section

**What does not work:**
- The "In Brief -> Key Insight -> Visual -> Formal -> Code -> Pitfalls -> Connections -> Visual Summary" template is followed so rigidly that decks feel formulaic by Module 03
- No narrative variation -- no case-study-first decks, no debate-format decks, no "what went wrong" failure stories (except Module 08 Guide 04)
- Module 04 (Content Growth Optimization) feels disconnected from the commodity trading theme -- it serves content creators, not traders
- Transitions between slides are implicit -- no "here is where we are" progress indicators within decks
- Some "Key Insight" quotes are generic rather than insightful (e.g., "Different algorithms suit different situations")

**Specific issues:**
- Module 00, Guide 03 (Decision Theory): Jumps from utility functions to Sharpe ratio to VPI without adequate motivation for each transition
- Module 02, Guide 02 (Posterior Updating): The narrative is essentially a math lecture -- no compelling story or problem to anchor the updates
- Module 04, Guide 02 (Conversion Optimization): Reads like a marketing guide; the bandit angle feels forced

### 3. Comprehensiveness (85/100)

**What works:**
- All fundamental bandit algorithms are covered: epsilon-greedy, UCB1, softmax/Boltzmann, Thompson Sampling
- Bayesian foundations are thorough: Beta-Bernoulli, Normal-Normal conjugate pairs, prior calibration
- Contextual bandits covered with full LinUCB derivation and feature engineering
- Advanced topics genuinely advanced: non-stationary, restless, adversarial bandits
- Production systems coverage is practical: architecture, logging, monitoring, offline evaluation
- Module 08 (Prompt Routing) adds genuine novelty not found in standard bandit textbooks
- Each module has a cheatsheet deck that (mostly) serves as a useful quick reference

**What does not work:**
- No coverage of combinatorial bandits or cascading bandits -- relevant for portfolio construction
- Offline evaluation (Module 07, Guide 03) covers IPS and doubly robust but omits direct method and MAGIC estimator
- No multi-objective bandit coverage (optimizing reward + cost + latency simultaneously)
- Module 05 has 5 slides decks (one more than other modules) but Module 00 only has 4 -- uneven depth
- No explicit "prerequisites" slide at the start of each module (what you need to know before starting)
- Glossary/notation slide missing -- each deck introduces notation independently, leading to minor inconsistencies

**Specific issues:**
- Module 01, Guide 03 (Softmax): Missing derivation of the gradient relationship to policy gradient methods -- a natural connection
- Module 06, Guide 02 (Restless Bandits): Whittle index is introduced but not derived; indexability condition mentioned without proof sketch
- Module 07, Guide 03 (Offline Evaluation): Propensity clipping discussed but optimal clipping thresholds not addressed
- Module 08: No coverage of A/B testing the prompt router itself (meta-evaluation)

### 4. Technical Accuracy (82/100)

**What works:**
- Core algorithms (epsilon-greedy, UCB1, Thompson Sampling, LinUCB, EXP3) are correctly implemented
- Mathematical notation is consistent within individual decks
- Regret bounds cited correctly: O(sqrt(KT log T)) for UCB1, O(sqrt(KT log K)) for EXP3
- Beta-Bernoulli conjugate update rules are correct
- LinUCB update equations match the Li et al. (2010) formulation
- Code examples are syntactically correct Python (verified by inspection)

**What does not work:**
- Module 01, Guide 02 (UCB1): The Hoeffding bound slide references "optimism in the face of uncertainty" but does not clearly distinguish UCB1 from UCB2 or KL-UCB -- could mislead readers into thinking UCB1 is the only UCB variant
- Module 02, Guide 01 (Thompson Sampling): The claim "Thompson Sampling achieves optimal regret" is stated without the important caveat that this is for specific reward distributions (Bernoulli) and the general case requires more nuance
- Module 05, Guide 02 (Reward Design): States "raw returns = NEVER use" as a universal rule -- this is too strong; risk-neutral settings exist where raw returns are appropriate
- Module 06, Guide 01 (Non-Stationary): Discounted Thompson Sampling implementation uses a fixed gamma without discussing how to detect non-stationarity to trigger the discount
- Module 06, Guide 03 (Adversarial Bandits): The importance sampling correction r_hat = r/p is shown without discussing the variance issue and practical need for mixing with uniform exploration

**Specific issues:**
- Module 03, Guide 02 (LinUCB): The alpha parameter is described as "exploration parameter" but its relationship to confidence level (1-delta) is not made explicit
- Module 05, Guide 04 (Regime-Aware): K-means for regime detection is presented without caveats about its sensitivity to initialization and cluster count
- Module 07, Guide 03 (Offline Evaluation): IPS estimator shown without discussion of the requirement for logging policy to have full support over action space

### 5. Added Visual Value (78/100)

**What works:**
- Mermaid flowcharts effectively illustrate algorithmic decision processes (Module 01 cheatsheet decision tree is excellent)
- Comparison tables add genuine value for algorithm selection (Thompson vs UCB, stationary vs non-stationary)
- The "Visual Summary" slide at the end of every deck provides a useful mental model
- Module 08 uses sequence diagrams effectively for the delayed reward pattern
- Two-column YES/NO layouts are immediately scannable

**What does not work:**
- No actual data visualizations -- no plots of regret curves, posterior distributions, or convergence behavior. These are described in text but never shown as actual charts
- Mermaid diagrams are limited to flowcharts and sequence diagrams -- no class diagrams, state diagrams, or Gantt charts that could add variety
- The 3-color palette (#36f blue, #6f6 green, #f66 red) is overused and semantically inconsistent -- green sometimes means "good", sometimes means "output", sometimes means "selected"
- Several diagrams are essentially reformatted bullet points (boxes connected with arrows where the arrows add no semantic meaning)
- No animated or progressive reveal diagrams -- everything is static
- Mathematical equations are rendered but never accompanied by geometric/visual interpretations

**Specific issues:**
- Module 02, Guide 02 (Posterior Updating): A posterior distribution visualization (Beta PDF evolving with observations) would be far more effective than the text description
- Module 06, Guide 01 (Non-Stationary): No visualization of how discounting affects the posterior -- this is a missed opportunity for a powerful visual
- Module 01, Guide 02 (UCB1): The confidence bound concept begs for a number-line visualization with intervals narrowing over time

### 6. Production Readiness (76/100)

**What works:**
- Marp front matter is valid and consistent -- all 38 decks would render correctly
- LaTeX via MathJax is properly configured
- CSS `.columns` class works for two-column layouts
- Pagination is enabled on all decks
- File naming convention is consistent (`NN_descriptive_name_slides.md`)

**What does not work:**
- No speaker notes in any deck -- a presenter would need the companion guide open alongside
- No build/export instructions -- how to generate PDF/PPTX from these Marp files is not documented
- No estimated presentation duration per deck
- Several code blocks exceed what can be read on a projected slide (15+ lines)
- Mermaid diagram rendering depends on Marp CLI version and Mermaid plugin availability -- no version pinning
- No accessibility considerations (alt text for diagrams, high-contrast options)
- No print-friendly formatting -- some Mermaid diagrams may render poorly in PDF export
- Lead slides (`<!-- _class: lead -->`) are used only for title and section breaks -- underutilized

**Specific issues:**
- Module 05, Guide 01: TwoWalletBandit class code spans ~20 lines -- needs to be split across 2 slides or simplified for presentation
- Module 07, Guide 01: The 6-component architecture diagram should be built progressively across multiple slides
- Module 08, Guide 03: ContextualPromptRouter code appears nearly identically in both Guide 03 and the Cheatsheet -- unnecessary duplication

---

## Per-Deck Reviews

### Module 00: Foundations (4 decks)

#### 01_ab_testing_limits_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 88 |
| Comprehensiveness | 85 |
| Technical Accuracy | 85 |
| Added Visual Value | 80 |
| Production Readiness | 78 |

**Strengths:** Strong opening deck. The "linear regret" concept is well-motivated with a $50K/month cost example. Four pitfalls (peeking, Simpson's paradox, fixed horizon, non-stationarity) are well-chosen and concise. The sample-size formula slide is clean.

**Issues:**
- Slide 3 (Fixed-Horizon Trap): The Mermaid flowchart is simple but effective; however, the p-value discussion could use a visual showing the "peeking" problem as a timeline
- The z-test formula is presented without visual context (where does each number come from?)
- Connections section could link forward to Module 01 more explicitly

**Fixes:** Add a "why this matters" cost visualization; add speaker notes for the z-test formula slide.

---

#### 02_explore_exploit_tradeoff_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 80 |
| Narrative & Story Flow | 90 |
| Comprehensiveness | 85 |
| Technical Accuracy | 80 |
| Added Visual Value | 78 |
| Production Readiness | 75 |

**Strengths:** The restaurant dilemma analogy is the best analogy in the entire course. The explore-exploit spectrum diagram is immediately intuitive. Regret bounds comparison table is clean and useful.

**Issues:**
- The companion guide has a rich "commodity trading version" of the dilemma that is absent from the slides -- a missed opportunity
- The regret bound expressions are presented without visual regret curve plots
- "Gittins Index" is mentioned in passing without adequate context

**Fixes:** Port the commodity trading analogy from the guide; add a simple regret curve sketch; either explain Gittins properly or remove the mention.

---

#### 03_decision_theory_basics_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 78 |
| Narrative & Story Flow | 75 |
| Comprehensiveness | 88 |
| Technical Accuracy | 82 |
| Added Visual Value | 72 |
| Production Readiness | 75 |

**Strengths:** Comprehensive coverage of expected value, utility functions, Sharpe ratio, VPI, and the Bayesian vs frequentist distinction. The decision framework table is a useful reference.

**Issues:**
- This is the densest deck in the course -- tries to cover too much for a single slide deck
- Narrative flow suffers from topic-hopping: utility -> Sharpe -> VPI -> Bayesian vs frequentist feels like four mini-lectures
- The formal definition slides are walls of LaTeX with insufficient visual scaffolding
- VPI (Value of Perfect Information) deserves its own diagram showing the decision tree

**Fixes:** Consider splitting into two decks (decision theory fundamentals + Bayesian foundations); add decision tree visualization for VPI; reduce LaTeX density with interleaved intuition.

---

#### cheatsheet_slides.md (Module 00)
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 75 |
| Narrative & Story Flow | 70 |
| Comprehensiveness | 72 |
| Technical Accuracy | 80 |
| Added Visual Value | 68 |
| Production Readiness | 72 |

**Strengths:** Provides a single-deck summary of Module 00 concepts.

**Issues:**
- This is the weakest cheatsheet in the course -- thin on content compared to later module cheatsheets
- No decision flowchart (which later cheatsheets include)
- Missing code snippets (which later cheatsheets include)
- Feels more like an agenda slide than a reference card

**Fixes:** Add a decision flowchart for "which concept applies when?"; add minimal code examples; model after the Module 05 or Module 08 cheatsheets which are much stronger.

---

### Module 01: Bandit Algorithms (4 decks)

#### 01_epsilon_greedy_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 85 |
| Comprehensiveness | 88 |
| Technical Accuracy | 85 |
| Added Visual Value | 80 |
| Production Readiness | 78 |

**Strengths:** Clean algorithm presentation. The decay schedule variants (1/t, 1/sqrt(t), stepped) are practical and well-compared. Five pitfalls are specific and actionable.

**Issues:**
- No visualization of exploration rate decaying over time
- The companion guide has more practice problems than the slides -- slides should have at least one worked example
- Code is correct but the `if random.random() < epsilon` pattern could show the full select-update loop

**Fixes:** Add an exploration decay curve; include one worked numerical example.

---

#### 02_upper_confidence_bound_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 80 |
| Narrative & Story Flow | 82 |
| Comprehensiveness | 85 |
| Technical Accuracy | 80 |
| Added Visual Value | 75 |
| Production Readiness | 78 |

**Strengths:** Clear derivation from Hoeffding's inequality. The confidence bounds table is useful. Formal definition is precise.

**Issues:**
- The confidence interval concept demands a visual (number line with narrowing bounds) that is not provided
- Does not distinguish UCB1 from other UCB variants (UCB2, KL-UCB, MOSS)
- The "optimism in the face of uncertainty" principle deserves a dedicated visual

**Fixes:** Add confidence interval visualization; add a brief note on UCB variant landscape; add speaker notes.

---

#### 03_softmax_boltzmann_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 80 |
| Comprehensiveness | 82 |
| Technical Accuracy | 85 |
| Added Visual Value | 78 |
| Production Readiness | 78 |

**Strengths:** Temperature parameter explanation is excellent. The log-sum-exp numerical stability trick is a practical gem. Three-way comparison table (epsilon-greedy vs UCB vs softmax) is one of the best comparison tables in the course.

**Issues:**
- The connection to policy gradient methods is not mentioned (natural extension)
- No visualization of how temperature affects the action probability distribution
- The comparison table could include a "when to use" row

**Fixes:** Add a probability distribution visualization at different temperatures; add "when to use" guidance.

---

#### cheatsheet_slides.md (Module 01)
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 78 |
| Comprehensiveness | 85 |
| Technical Accuracy | 82 |
| Added Visual Value | 82 |
| Production Readiness | 80 |

**Strengths:** The algorithm decision flowchart is excellent -- one of the best single slides in the course. Minimal code templates for each algorithm are well-chosen.

**Issues:**
- Could include a regret comparison summary
- The "when to use" section is present but brief

**Fixes:** Minor -- add regret bound comparison row to the summary table.

---

### Module 02: Bayesian Bandits (4 decks)

#### 01_thompson_sampling_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 85 |
| Comprehensiveness | 88 |
| Technical Accuracy | 82 |
| Added Visual Value | 78 |
| Production Readiness | 78 |

**Strengths:** The 15-line Thompson Sampling implementation is clean and complete. Casino floor analogy works well. Posterior evolution is described clearly.

**Issues:**
- Claims "optimal regret" without sufficient caveats about distribution assumptions
- No posterior distribution plots (the single most impactful visualization for this topic is missing)
- The Beta distribution is introduced but not visually shown at different (alpha, beta) values

**Fixes:** Add caveat about optimality conditions; add Beta distribution plots at key (alpha, beta) values; this is a high-priority visual improvement.

---

#### 02_posterior_updating_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 78 |
| Narrative & Story Flow | 72 |
| Comprehensiveness | 85 |
| Technical Accuracy | 85 |
| Added Visual Value | 68 |
| Production Readiness | 72 |

**Strengths:** Conjugate prior table is comprehensive (Beta-Bernoulli, Normal-Normal, Gamma-Poisson). The update rules are correct. Prior strength discussion is important and well-placed.

**Issues:**
- This is the weakest narrative in the course -- reads like a reference manual, not a story
- No motivating problem at the start ("why do we need posterior updating?")
- The visual value is the lowest of any guide deck -- LaTeX-heavy with no distribution plots
- Slide progression feels like a textbook page, not a presentation
- No worked numerical example showing a posterior actually being updated step by step

**Fixes:** Add a motivating scenario (e.g., "your boss asks which supplier is more reliable after 10 orders"); add posterior evolution plots; add a worked numerical example with specific numbers; restructure to story-first.

---

#### 03_thompson_vs_ucb_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 85 |
| Comprehensiveness | 85 |
| Technical Accuracy | 82 |
| Added Visual Value | 82 |
| Production Readiness | 78 |

**Strengths:** The comparison table is one of the most useful in the course. Delayed feedback discussion is practical. The commodity sector rotation example ties theory to practice effectively. The "probability matching" property explanation is clear.

**Issues:**
- Could include a side-by-side regret curve comparison (even a schematic one)
- The "when to use which" recommendation could be stronger -- currently balanced to a fault

**Fixes:** Add a comparative regret sketch; add a strong recommendation with clear criteria.

---

#### cheatsheet_slides.md (Module 02)
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 80 |
| Narrative & Story Flow | 78 |
| Comprehensiveness | 82 |
| Technical Accuracy | 82 |
| Added Visual Value | 78 |
| Production Readiness | 78 |

**Strengths:** Conjugate pair table and quick code snippets are useful. Non-stationary adaptation section bridges to Module 06.

**Issues:**
- No decision flowchart (unlike Module 01 cheatsheet which has an excellent one)
- Could be more concise

**Fixes:** Add a decision flowchart for "which Bayesian approach?"; tighten content.

---

### Module 03: Contextual Bandits (4 decks)

#### 01_contextual_bandit_framework_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 85 |
| Comprehensiveness | 82 |
| Technical Accuracy | 82 |
| Added Visual Value | 80 |
| Production Readiness | 78 |

**Strengths:** Standard vs contextual comparison is immediately clear. The context-to-action mapping diagram is effective. Formal definition is concise.

**Issues:**
- The motivating example could be stronger -- the commodity trading context should be front and center
- No example of what a context vector actually looks like with real numbers

**Fixes:** Add a concrete commodity trading example with real feature values; add speaker notes.

---

#### 02_linucb_algorithm_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 80 |
| Narrative & Story Flow | 82 |
| Comprehensiveness | 88 |
| Technical Accuracy | 80 |
| Added Visual Value | 75 |
| Production Readiness | 75 |

**Strengths:** Full LinUCB derivation with ridge regression explanation. The A-matrix update is clearly shown. Regret bound O(d*sqrt(T log T)) is correctly stated.

**Issues:**
- The confidence ellipsoid concept deserves a 2D visualization that is absent
- Code block is 15 lines -- borderline for slide readability
- The alpha parameter's relationship to confidence level is not made explicit
- Dense mathematical content without adequate visual scaffolding

**Fixes:** Add 2D confidence ellipsoid visualization; split code across two slides; clarify alpha-delta relationship.

---

#### 03_feature_engineering_bandits_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 82 |
| Comprehensiveness | 85 |
| Technical Accuracy | 82 |
| Added Visual Value | 78 |
| Production Readiness | 78 |

**Strengths:** Tier 1/2/3 feature categorization is practical and memorable. The 5-feature rule is a good heuristic. Commodity context pipeline diagram is domain-relevant.

**Issues:**
- Feature engineering for bandits is under-discussed in the literature, so this deck adds real value -- but could go deeper on feature selection methods
- No mention of feature interaction effects

**Fixes:** Add a brief discussion of feature interactions; consider a worked feature selection example.

---

#### cheatsheet_slides.md (Module 03)
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 78 |
| Comprehensiveness | 85 |
| Technical Accuracy | 82 |
| Added Visual Value | 80 |
| Production Readiness | 78 |

**Strengths:** LinUCB code template is complete and correct. Parameter selection guidance is practical. Troubleshooting section adds genuine value.

**Issues:** Minor -- could include a feature engineering checklist.

**Fixes:** Add feature engineering quick checklist.

---

### Module 04: Content Growth Optimization (4 decks)

#### 01_creator_bandit_playbook_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 85 |
| Comprehensiveness | 82 |
| Technical Accuracy | 80 |
| Added Visual Value | 80 |
| Production Readiness | 78 |

**Strengths:** The "Topic x Format" arm design is creative. Read ratio reward function is well-motivated. Three creator archetypes (explorer, optimizer, diversifier) are memorable. Quarterly lifecycle model is practical.

**Issues:**
- Domain shift from commodity trading to content creation is jarring -- needs better framing of why this module exists in a commodity trading course
- "Read ratio" as reward is domain-specific and may not generalize

**Fixes:** Add a framing slide explaining how content optimization mirrors commodity allocation decisions.

---

#### 02_conversion_optimization_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 78 |
| Narrative & Story Flow | 75 |
| Comprehensiveness | 78 |
| Technical Accuracy | 80 |
| Added Visual Value | 72 |
| Production Readiness | 75 |

**Strengths:** A/B test vs Thompson Sampling comparison is useful. Multi-step funnel concept is practical.

**Issues:**
- Weakest deck in Module 04 -- reads like a generic marketing guide with bandit terminology sprinkled in
- The bandit angle feels forced; conversion optimization is well-served by standard A/B testing for many use cases
- No posterior evolution visualization
- The multi-step funnel is described but not diagrammed effectively

**Fixes:** Strengthen the bandit-specific advantage over A/B testing; add a funnel diagram; add posterior evolution plot; consider whether this deck is necessary or could be merged with Guide 01.

---

#### 03_arm_management_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 82 |
| Comprehensiveness | 85 |
| Technical Accuracy | 82 |
| Added Visual Value | 80 |
| Production Readiness | 78 |

**Strengths:** Arm retirement criteria (UCB < LCB_best) is a practical and well-defined rule. Introduction protocol with burn-in period is actionable. Windowed estimate approach for non-stationarity is practical.

**Issues:**
- The retirement criteria should include a visual showing two arms' confidence intervals diverging
- No discussion of how to handle retired arms that might become relevant again

**Fixes:** Add confidence interval divergence visualization; discuss arm resurrection scenarios.

---

#### cheatsheet_slides.md (Module 04)
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 80 |
| Narrative & Story Flow | 78 |
| Comprehensiveness | 82 |
| Technical Accuracy | 80 |
| Added Visual Value | 78 |
| Production Readiness | 78 |

**Strengths:** 6-step framework is a clean summary. Algorithm selection flowchart is useful.

**Issues:** Minor -- consistent with other cheatsheets.

**Fixes:** None critical.

---

### Module 05: Commodity Trading Bandits (5 decks)

#### 01_accumulator_bandit_playbook_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 85 |
| Narrative & Story Flow | 90 |
| Comprehensiveness | 90 |
| Technical Accuracy | 85 |
| Added Visual Value | 85 |
| Production Readiness | 78 |

**Strengths:** **Top 3 deck in the course.** The two-wallet framework (80% core, 20% bandit sleeve) is immediately practical and memorable. The 6-step playbook is actionable. The TwoWalletBandit class is a complete, usable implementation. Risk management is integrated throughout, not an afterthought.

**Issues:**
- The TwoWalletBandit code is ~20 lines -- should be split across two slides for presentation readability
- The 80/20 split is presented as a recommendation without discussing how to calibrate this ratio

**Fixes:** Split code across two slides; add guidance on calibrating the core/bandit split ratio.

---

#### 02_reward_design_commodities_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 88 |
| Comprehensiveness | 88 |
| Technical Accuracy | 78 |
| Added Visual Value | 82 |
| Production Readiness | 78 |

**Strengths:** The "raw returns = NEVER" messaging is bold and attention-grabbing. The progression from bad rewards to good rewards to commodity-specific rewards is well-structured. Contango/backwardation reward considerations are domain-specific and valuable.

**Issues:**
- "Raw returns = NEVER" is too absolute -- there are valid risk-neutral settings where raw returns are appropriate
- The seasonality reward adjustment is described but not formalized
- No worked example showing reward calculation with real numbers

**Fixes:** Soften the "NEVER" claim to "almost never in risk-managed settings"; add a worked reward calculation example with real commodity data.

---

#### 03_guardrails_and_safety_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 85 |
| Comprehensiveness | 88 |
| Technical Accuracy | 85 |
| Added Visual Value | 80 |
| Production Readiness | 80 |

**Strengths:** Five guardrails are well-chosen and practical: position limits, minimum allocation, tilt speed, core protection, volatility dampening. Each guardrail has a clear parameter and rationale. The progression from safety to speed to stability is logical.

**Issues:**
- No discussion of guardrail parameter calibration methodology
- Could include a "guardrail violation" example showing what happens when guardrails are missing

**Fixes:** Add a calibration discussion; add a failure scenario.

---

#### 04_regime_aware_allocation_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 85 |
| Comprehensiveness | 85 |
| Technical Accuracy | 78 |
| Added Visual Value | 80 |
| Production Readiness | 78 |

**Strengths:** Six regime features are well-chosen. Classification strategies (rule-based, K-means, HMM) cover the practical spectrum. RegimeAwareBandit class is complete and usable.

**Issues:**
- K-means for regime detection is presented without caveats (sensitivity to initialization, cluster count)
- HMM is mentioned but not implemented -- inconsistent depth
- No backtesting results or worked example

**Fixes:** Add K-means caveats; either add HMM implementation or clearly label it as "advanced extension"; add a backtesting example.

---

#### cheatsheet_slides.md (Module 05)
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 85 |
| Narrative & Story Flow | 82 |
| Comprehensiveness | 88 |
| Technical Accuracy | 82 |
| Added Visual Value | 85 |
| Production Readiness | 80 |

**Strengths:** **Best cheatsheet in the course.** Decision flowchart is excellent. Guardrail parameters table (conservative/moderate/aggressive) is immediately actionable. Debugging guide adds practical value not found elsewhere.

**Issues:** Minor -- the deck is quite long for a cheatsheet (12+ slides).

**Fixes:** Consider whether some content could be condensed; but this is a minor concern given the quality.

---

### Module 06: Advanced Topics (4 decks)

#### 01_non_stationary_bandits_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 85 |
| Comprehensiveness | 85 |
| Technical Accuracy | 80 |
| Added Visual Value | 78 |
| Production Readiness | 78 |

**Strengths:** Discounted Thompson Sampling and sliding-window UCB are both well-presented. The COVID crash example is vivid and practical. Gamma tuning guidance is actionable.

**Issues:**
- No visualization of discounting effect on posterior
- Fixed gamma without change-point detection discussion
- No comparison of discounted TS vs sliding-window UCB on the same problem

**Fixes:** Add discounting visualization; discuss change-point detection; add a head-to-head comparison.

---

#### 02_restless_bandits_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 80 |
| Narrative & Story Flow | 82 |
| Comprehensiveness | 80 |
| Technical Accuracy | 78 |
| Added Visual Value | 78 |
| Production Readiness | 75 |

**Strengths:** The concept that arms evolve when not selected is well-explained. Recency penalty calibration is practical. The Whittle index introduction is accessible.

**Issues:**
- Whittle index is introduced conceptually but the indexability condition is hand-waved
- No proof sketch or even intuition for why the index policy is approximately optimal
- Recency penalty could be more rigorously defined

**Fixes:** Add a brief intuition for indexability; formalize the recency penalty definition; this is acceptable as an introduction but should signal the depth that is missing.

---

#### 03_adversarial_bandits_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 88 |
| Comprehensiveness | 85 |
| Technical Accuracy | 80 |
| Added Visual Value | 82 |
| Production Readiness | 78 |

**Strengths:** **Top 3 deck in the course.** The rock-paper-scissors analogy is excellent for adversarial settings. EXP3 algorithm is correctly presented. Importance sampling explanation is clear. Market impact as adversarial feedback is a creative domain connection.

**Issues:**
- Importance sampling variance issue (r_hat = r/p can have huge variance when p is small) is mentioned but not sufficiently emphasized
- No discussion of EXP3.P (high-probability variant)
- The mixing parameter gamma in EXP3 is not clearly connected to the theoretical regret bound

**Fixes:** Emphasize variance issue more strongly; mention EXP3.P as extension; clarify gamma's role.

---

#### cheatsheet_slides.md (Module 06)
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 80 |
| Comprehensiveness | 85 |
| Technical Accuracy | 82 |
| Added Visual Value | 82 |
| Production Readiness | 78 |

**Strengths:** Algorithm selection decision tree is excellent. Hyperparameter tuning guide is practical. Commodity-specific guidelines bridge theory and application.

**Issues:** Minor -- consistent quality.

**Fixes:** None critical.

---

### Module 07: Production Systems (4 decks)

#### 01_bandit_system_architecture_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 80 |
| Narrative & Story Flow | 82 |
| Comprehensiveness | 85 |
| Technical Accuracy | 82 |
| Added Visual Value | 78 |
| Production Readiness | 80 |

**Strengths:** The 6-component architecture (Registry, Context, Policy, Guardrails, Logger, Monitor) is clean and practical. Separation of concerns is well-motivated. Each component has a clear interface.

**Issues:**
- The architecture diagram is too dense for a single slide -- 6 components with interconnections
- No discussion of technology choices (what language/framework, database, message queue)
- No deployment topology (single machine vs distributed)

**Fixes:** Split architecture diagram across 2-3 slides (build progressively); add technology recommendations.

---

#### 02_logging_and_monitoring_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 82 |
| Comprehensiveness | 85 |
| Technical Accuracy | 85 |
| Added Visual Value | 80 |
| Production Readiness | 82 |

**Strengths:** JSONL logging format is practical and immediately usable. Five monitoring metrics are well-chosen. Five actionable alerts are specific (not generic "monitor performance"). BanditLogger and BanditMonitor classes are production-quality.

**Issues:**
- No mention of log rotation, retention policies, or storage considerations
- Alert thresholds are presented as fixed values without calibration guidance
- No integration with common monitoring tools (Prometheus, Grafana, DataDog)

**Fixes:** Add log management considerations; add calibration guidance for alert thresholds; mention monitoring tool integration patterns.

---

#### 03_offline_evaluation_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 80 |
| Narrative & Story Flow | 82 |
| Comprehensiveness | 80 |
| Technical Accuracy | 78 |
| Added Visual Value | 75 |
| Production Readiness | 78 |

**Strengths:** IPS and doubly robust estimators are correctly presented. Propensity scoring discussion is practical. OfflineEvaluator class is usable.

**Issues:**
- Missing direct method and MAGIC estimator coverage
- IPS estimator shown without the critical requirement that logging policy must have full support
- Propensity clipping discussed but optimal thresholds not addressed
- No visualization of how offline evaluation accuracy improves with more data

**Fixes:** Add logging policy support requirement caveat; discuss MAGIC estimator; add convergence visualization.

---

#### cheatsheet_slides.md (Module 07)
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 80 |
| Comprehensiveness | 85 |
| Technical Accuracy | 82 |
| Added Visual Value | 80 |
| Production Readiness | 82 |

**Strengths:** Pre-deployment checklist is immediately actionable. A/B to bandit migration phases are practical. Deployment environment comparison is useful.

**Issues:** Minor -- solid cheatsheet.

**Fixes:** None critical.

---

### Module 08: Prompt Routing Bandits (5 decks)

#### 01_prompt_routing_fundamentals_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 88 |
| Comprehensiveness | 85 |
| Technical Accuracy | 85 |
| Added Visual Value | 82 |
| Production Readiness | 78 |

**Strengths:** Novel topic not found in standard bandit textbooks. Six prompt arm types are well-defined. PromptRouter class is clean. The "bad prompt tax" concept is memorable. Analyst analogy works well.

**Issues:**
- Could include a comparison to prompt selection approaches outside bandits (chain-of-thought routing, LLM-as-judge)
- The 6 prompt types feel somewhat arbitrary -- no methodology for discovering what prompts to test

**Fixes:** Add brief comparison to alternative prompt selection methods; add guidance on prompt discovery methodology.

---

#### 02_reward_design_llm_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 85 |
| Comprehensiveness | 88 |
| Technical Accuracy | 85 |
| Added Visual Value | 80 |
| Production Readiness | 78 |

**Strengths:** Composite reward function (primary + guardrail penalties) is well-structured. Hallucination detection approach is practical. Four primary metrics are well-chosen. The penalty severity discussion is important and well-placed.

**Issues:**
- Reward normalization across different metric scales is not discussed
- No discussion of reward delay (LLM responses may be evaluated later)
- Hallucination detection is described as a binary check -- real-world detection is probabilistic

**Fixes:** Add reward normalization discussion; add delayed reward handling; acknowledge probabilistic nature of hallucination detection.

---

#### 03_contextual_prompt_routing_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 82 |
| Narrative & Story Flow | 88 |
| Comprehensiveness | 90 |
| Technical Accuracy | 85 |
| Added Visual Value | 82 |
| Production Readiness | 78 |

**Strengths:** Five context features are well-chosen and practical. Feature extraction pipeline diagram is clear. The LinUCB application to prompt routing is well-connected to Module 03. The "What the Router Learns" results table is compelling. The newsroom analogy is excellent.

**Issues:**
- The ContextualPromptRouter code is nearly identical to the Module 03 LinUCB code -- could acknowledge this more explicitly rather than presenting it as new
- 15-dimensional context vector may be too large for cold-start -- no discussion of dimensionality reduction
- The "What the Router Learns" table presents results without discussing sample sizes needed

**Fixes:** Explicitly note code reuse from Module 03; discuss cold-start mitigation for high-dimensional contexts; add sample size requirements.

---

#### 04_commodity_research_assistant_slides.md
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 85 |
| Narrative & Story Flow | 92 |
| Comprehensiveness | 90 |
| Technical Accuracy | 85 |
| Added Visual Value | 85 |
| Production Readiness | 82 |

**Strengths:** **Best deck in the entire course.** Three case studies are concrete, specific, and compelling. Results are quantified with before/after metrics. The hallucination story (35% to 2%) is dramatic and practical. The EIA report processor case study shows real-world self-adaptation. The delayed reward discussion (5-day lag for trading signals) addresses a practical concern. The "When to Use / When Not to Use" decision framework is honest and balanced.

**Issues:**
- Case study results may be too optimistic -- no error bars or confidence intervals
- The "Sharpe ratio 0.9 to 1.4" improvement claim is extraordinary and should be caveated
- No discussion of the cost of running the bandit system itself (compute, latency overhead)

**Fixes:** Add uncertainty ranges to results; caveat the Sharpe improvement claim; discuss system overhead costs.

---

#### cheatsheet_slides.md (Module 08)
| Dimension | Score |
|-----------|-------|
| Design & Visual Quality | 85 |
| Narrative & Story Flow | 82 |
| Comprehensiveness | 88 |
| Technical Accuracy | 85 |
| Added Visual Value | 82 |
| Production Readiness | 80 |

**Strengths:** Second-best cheatsheet (after Module 05). Algorithm selection flowchart is clear. Both Thompson Sampling and LinUCB code templates provided. Failure mode table is practical. Production deployment checklist is actionable. Key metrics table with targets and red flags is immediately useful.

**Issues:**
- The ContextualPromptRouter code is duplicated from Guide 03 -- unnecessary repetition
- The cheatsheet is quite long (13+ slides)

**Fixes:** Reference Guide 03 code instead of duplicating; consider condensing.

---

## Comparison to Best-in-Class

### Stanford CS229 Machine Learning Slides
- **CS229 advantage:** Cleaner mathematical typesetting, more whitespace per slide, progressive notation introduction, proof sketches included
- **This course advantage:** Far more practical code examples, domain-specific applications, production considerations
- **Gap:** This course should adopt CS229's practice of introducing notation on a dedicated slide before using it; should add more whitespace to formal definition slides

### fast.ai Practical Deep Learning Slides
- **fast.ai advantage:** Real data visualizations (actual plots, not Mermaid flowcharts), progressive code building (one line at a time), Jeremy Howard's "show the result first" pedagogy
- **This course advantage:** More formal mathematical foundations, better structured reference materials (cheatsheets), explicit connection sections between topics
- **Gap:** This course critically needs real data visualizations -- regret curves, posterior distributions, convergence plots. The absence of actual charts is the single biggest visual gap.

### DataCamp Slide Decks
- **DataCamp advantage:** Consistent branding, professional visual design, tight 3-5 slide format per concept, every slide has exactly one takeaway
- **This course advantage:** Deeper technical content, more comprehensive coverage, better domain integration, superior code examples
- **Gap:** This course's slides are sometimes too dense -- should adopt DataCamp's "one idea per slide" discipline more rigorously

---

## Priority Fix List

### Critical (Must Fix)

1. **Add real data visualizations throughout the course.** Every module should have at least 2-3 actual plots (regret curves, posterior distributions, convergence). Mermaid diagrams alone are insufficient for a quantitative course. This is the single highest-impact improvement.

2. **Add speaker notes to all 38 decks.** Without speaker notes, these decks are presentation outlines, not presentation-ready materials. Each slide should have 2-4 sentences of presenter guidance.

3. **Fix "optimal regret" claim in Module 02 Guide 01.** Add caveats about distribution assumptions. Currently misleading.

4. **Fix "raw returns = NEVER" in Module 05 Guide 02.** Soften to "rarely appropriate in risk-managed settings." The absolute claim is technically incorrect.

### High Priority (Should Fix)

5. **Split oversized code blocks across multiple slides.** Affected decks: Module 05 Guide 01 (TwoWalletBandit), Module 03 Guide 02 (LinUCB class), Module 07 Guide 01 (architecture). Any code block over 15 lines should be split.

6. **Add posterior distribution visualizations to Module 02.** Thompson Sampling and posterior updating without distribution plots is like teaching geometry without diagrams.

7. **Restructure Module 02 Guide 02 (Posterior Updating)** from reference-manual format to story-first format with a motivating scenario and worked numerical example.

8. **Add confidence interval visualization to Module 01 Guide 02 (UCB).** The confidence bound concept demands a visual.

9. **Resolve Module 04 framing issue.** Content Growth Optimization needs an explicit framing slide explaining why it belongs in a commodity trading course.

10. **Add IPS full-support requirement caveat to Module 07 Guide 03.** Missing theoretical requirement for offline evaluation validity.

### Medium Priority (Nice to Fix)

11. **Diversify Mermaid color palette.** The 3-color scheme (#36f, #6f6, #f66) is monotonous across 38 decks. Add 2-3 additional colors with consistent semantic meaning.

12. **Add build/export instructions.** Document how to render these Marp decks to PDF/PPTX/HTML.

13. **Add prerequisites slide to each module's first deck.**

14. **Remove ContextualPromptRouter code duplication between Module 08 Guide 03 and Cheatsheet.**

15. **Standardize cheatsheet depth.** Module 00 cheatsheet is too thin; model all cheatsheets after Module 05's cheatsheet (the best one).

16. **Add estimated presentation duration to each deck's front matter.**

17. **Add a course-wide notation/glossary slide** to prevent inconsistent notation introduction across decks.

### Low Priority (Consider)

18. **Add practice problem variety.** Current practice problems are repetitive ("implement X", "compare Y vs Z"). Add "debug this code", "what went wrong", and "design a system" style problems.

19. **Consider whether Module 04 Guide 02 (Conversion Optimization) should be merged with Guide 01** or dropped entirely -- it is the weakest deck.

20. **Add EXP3.P mention to Module 06 Guide 03** for completeness.

21. **Consider adding combinatorial bandits coverage** for portfolio construction use cases.

22. **Add K-means initialization caveats to Module 05 Guide 04.**

---

## Appendix: Score Summary Table

| Module | Deck | Design | Narrative | Comprehensive | Accuracy | Visual | Production | Weighted |
|--------|------|--------|-----------|---------------|----------|--------|------------|----------|
| M00 | 01 AB Testing Limits | 82 | 88 | 85 | 85 | 80 | 78 | 84 |
| M00 | 02 Explore-Exploit | 80 | 90 | 85 | 80 | 78 | 75 | 83 |
| M00 | 03 Decision Theory | 78 | 75 | 88 | 82 | 72 | 75 | 79 |
| M00 | Cheatsheet | 75 | 70 | 72 | 80 | 68 | 72 | 73 |
| M01 | 01 Epsilon-Greedy | 82 | 85 | 88 | 85 | 80 | 78 | 84 |
| M01 | 02 UCB | 80 | 82 | 85 | 80 | 75 | 78 | 81 |
| M01 | 03 Softmax | 82 | 80 | 82 | 85 | 78 | 78 | 81 |
| M01 | Cheatsheet | 82 | 78 | 85 | 82 | 82 | 80 | 82 |
| M02 | 01 Thompson Sampling | 82 | 85 | 88 | 82 | 78 | 78 | 83 |
| M02 | 02 Posterior Updating | 78 | 72 | 85 | 85 | 68 | 72 | 78 |
| M02 | 03 Thompson vs UCB | 82 | 85 | 85 | 82 | 82 | 78 | 83 |
| M02 | Cheatsheet | 80 | 78 | 82 | 82 | 78 | 78 | 80 |
| M03 | 01 Contextual Framework | 82 | 85 | 82 | 82 | 80 | 78 | 82 |
| M03 | 02 LinUCB | 80 | 82 | 88 | 80 | 75 | 75 | 81 |
| M03 | 03 Feature Engineering | 82 | 82 | 85 | 82 | 78 | 78 | 82 |
| M03 | Cheatsheet | 82 | 78 | 85 | 82 | 80 | 78 | 81 |
| M04 | 01 Creator Playbook | 82 | 85 | 82 | 80 | 80 | 78 | 82 |
| M04 | 02 Conversion Opt | 78 | 75 | 78 | 80 | 72 | 75 | 77 |
| M04 | 03 Arm Management | 82 | 82 | 85 | 82 | 80 | 78 | 82 |
| M04 | Cheatsheet | 80 | 78 | 82 | 80 | 78 | 78 | 80 |
| M05 | 01 Accumulator Playbook | 85 | 90 | 90 | 85 | 85 | 78 | **87** |
| M05 | 02 Reward Design | 82 | 88 | 88 | 78 | 82 | 78 | 84 |
| M05 | 03 Guardrails | 82 | 85 | 88 | 85 | 80 | 80 | 84 |
| M05 | 04 Regime-Aware | 82 | 85 | 85 | 78 | 80 | 78 | 82 |
| M05 | Cheatsheet | 85 | 82 | 88 | 82 | 85 | 80 | **84** |
| M06 | 01 Non-Stationary | 82 | 85 | 85 | 80 | 78 | 78 | 82 |
| M06 | 02 Restless | 80 | 82 | 80 | 78 | 78 | 75 | 79 |
| M06 | 03 Adversarial | 82 | 88 | 85 | 80 | 82 | 78 | 83 |
| M06 | Cheatsheet | 82 | 80 | 85 | 82 | 82 | 78 | 82 |
| M07 | 01 Architecture | 80 | 82 | 85 | 82 | 78 | 80 | 82 |
| M07 | 02 Logging | 82 | 82 | 85 | 85 | 80 | 82 | 83 |
| M07 | 03 Offline Eval | 80 | 82 | 80 | 78 | 75 | 78 | 79 |
| M07 | Cheatsheet | 82 | 80 | 85 | 82 | 80 | 82 | 82 |
| M08 | 01 Prompt Fundamentals | 82 | 88 | 85 | 85 | 82 | 78 | 84 |
| M08 | 02 Reward Design LLM | 82 | 85 | 88 | 85 | 80 | 78 | 84 |
| M08 | 03 Contextual Routing | 82 | 88 | 90 | 85 | 82 | 78 | 85 |
| M08 | 04 Research Assistant | 85 | 92 | 90 | 85 | 85 | 82 | **88** |
| M08 | Cheatsheet | 85 | 82 | 88 | 85 | 82 | 80 | **84** |

**Top 5 Decks:** M08-04 (88), M05-01 (87), M08-03 (85), M05-Cheat (84), M08-Cheat (84)
**Bottom 5 Decks:** M00-Cheat (73), M04-02 (77), M02-02 (78), M00-03 (79), M06-02 (79)

---

*Review completed 2026-02-20. All 38 decks in 9 modules reviewed across 6 dimensions.*
