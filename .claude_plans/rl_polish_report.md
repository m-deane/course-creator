# RL Course Polish Report

**Date:** 2026-03-08
**Course path:** `/home/user/course-creator/courses/reinforcement-learning/`
**Auditor:** Claude Code (claude-sonnet-4-6)
**Scope:** 10 modules (00–09), 32 slide decks, 20 notebooks, 10 exercise files, 3 template files, 2 recipe files

---

## Executive Summary

5 of 7 checks PASS. 2 checks FAIL. The course has one critical notation inconsistency (exploration rate symbol) that appears across 9 files in modules 02–03, and three incorrect cross-module references that will confuse learners following the linear curriculum. No rendering failures, no code style violations, and speaker notes are substantive throughout. The course is **conditionally ready** for learners starting at Module 0 — critical issues should be resolved before Module 2 and Module 4 are encountered.

---

## Check 1 — Notation Consistency

**Status: FAIL**

### Issue 1.1 — Exploration rate: `\varepsilon` vs `\epsilon`

**Severity: CRITICAL**

The Module 0 cheatsheet defines the exploration rate as `$\epsilon$` (`\epsilon`). Modules 02 and 03 consistently use `$\varepsilon$` (`\varepsilon`) for the same concept. These render as visually distinct symbols (ε vs ϵ). A learner progressing linearly will encounter a different symbol for the same concept with no explanation.

Affected files (9 total):

- `/home/user/course-creator/courses/reinforcement-learning/modules/module_02_monte_carlo_methods/guides/02_monte_carlo_control_guide.md`
- `/home/user/course-creator/courses/reinforcement-learning/modules/module_02_monte_carlo_methods/guides/02_monte_carlo_control_slides.md`
- `/home/user/course-creator/courses/reinforcement-learning/modules/module_02_monte_carlo_methods/guides/03_importance_sampling_slides.md`
- `/home/user/course-creator/courses/reinforcement-learning/modules/module_02_monte_carlo_methods/guides/cheatsheet.md` — symbol table entry: `| $\varepsilon$ | Exploration probability in $\varepsilon$-greedy |`
- `/home/user/course-creator/courses/reinforcement-learning/modules/module_03_temporal_difference/guides/02_sarsa_guide.md`
- `/home/user/course-creator/courses/reinforcement-learning/modules/module_03_temporal_difference/guides/02_sarsa_slides.md`
- `/home/user/course-creator/courses/reinforcement-learning/modules/module_03_temporal_difference/guides/03_q_learning_guide.md`
- `/home/user/course-creator/courses/reinforcement-learning/modules/module_03_temporal_difference/guides/03_q_learning_slides.md`
- `/home/user/course-creator/courses/reinforcement-learning/modules/module_03_temporal_difference/guides/cheatsheet.md` — symbol table entry: `| $\varepsilon$ | Exploration rate`

**Fix:** Replace all `\varepsilon` instances used as the exploration rate symbol with `\epsilon` in the above files, consistent with the Module 0 cheatsheet definition. Note: the single instance in `module_05_deep_rl/guides/02_dqn_improvements_slides.md` uses `\varepsilon` as a noise variable (not exploration rate) — this is acceptable and should not be changed.

---

### Issue 1.2 — `\alpha` symbol collision in Module 07 (SAC)

**Severity: MINOR**

The Module 0 cheatsheet defines `$\alpha$` as the learning rate. Module 07's SAC guide and cheatsheet repurpose `$\alpha$` as the **temperature parameter** (entropy coefficient) without acknowledging the collision with the course-wide definition.

Affected files:

- `/home/user/course-creator/courses/reinforcement-learning/modules/module_07_advanced_policy_optimization/guides/03_sac_guide.md` — uses `$\alpha$` for temperature throughout
- `/home/user/course-creator/courses/reinforcement-learning/modules/module_07_advanced_policy_optimization/guides/cheatsheet.md` — line 70: `| $\alpha$ | Temperature (entropy coefficient) in SAC |`

Additionally, the module 07 cheatsheet uses `$\alpha$` for the natural gradient step size (line 17), the SAC temperature (line 70), and implicitly the optimizer learning rate — three different meanings in the same file.

**Fix:** Add a note in the SAC guide and cheatsheet explaining that `$\alpha$` in the SAC context is the temperature/entropy coefficient, which differs from the course-wide use of `$\alpha$` as a gradient step size. Consider using `$\alpha_{temp}$` or `$\tau_{ent}$` for the SAC temperature to avoid confusion, or explicitly acknowledge the overload at first use.

---

### Issue 1.3 — `$r_t(\theta)$` importance ratio reuses reward symbol `$r$`

**Severity: MINOR**

Module 07 PPO content uses `$r_t(\theta)$` for the importance ratio (policy probability ratio), which clashes with the global symbol `$r$` for reward. Within Module 07's cheatsheet, line 47 uses `$r_t$` for reward and line 66 defines `$r_t(\theta)$` as the importance ratio. The `$(\theta)$` argument disambiguates in formal expressions, but in running text where the `$(\theta)$` is sometimes dropped (e.g., "when $r_t > 1+\epsilon$"), learners may be confused.

Affected files:

- `/home/user/course-creator/courses/reinforcement-learning/modules/module_07_advanced_policy_optimization/guides/02_ppo_guide.md`
- `/home/user/course-creator/courses/reinforcement-learning/modules/module_07_advanced_policy_optimization/guides/02_ppo_slides.md`
- `/home/user/course-creator/courses/reinforcement-learning/modules/module_07_advanced_policy_optimization/guides/cheatsheet.md`

**Fix:** This is standard PPO literature notation (Schulman et al. 2017 use `$r_t(\theta)$`). Add a one-sentence clarification at first use: "Here $r_t(\theta)$ denotes the policy probability ratio, not the reward — we follow the notation of Schulman et al. (2017)."

---

## Check 2 — Cross-References

**Status: FAIL**

Spot-checked 5 files containing cross-module references. Found 3 incorrect references.

### Issue 2.1 — Module 04 Guide 01 misattributes TD as Module 02

**Severity: CRITICAL**

File: `/home/user/course-creator/courses/reinforcement-learning/modules/module_04_function_approximation/guides/01_why_function_approximation_guide.md`, line 287:

```
- **Builds on:** Module 02 (tabular TD methods), Module 03 (policy gradient concepts), ...
```

Both attributions are wrong:
- Tabular TD methods are **Module 03** (Temporal Difference), not Module 02 (Monte Carlo)
- Policy gradient concepts are **Module 06**, not Module 03 (which is Temporal Difference)

A learner reading this "Builds on" section and trying to review prerequisites will be directed to the wrong content.

**Fix:** Change to: `Builds on: Module 03 (tabular TD methods), Module 06 (policy gradient concepts — for the value function gradient derivation), Module 00 (MDP notation and $V^\pi$, $Q^\pi$ definitions)`

---

### Issue 2.2 — Module 04 Guide 02 misattributes TD as Module 02

**Severity: CRITICAL**

File: `/home/user/course-creator/courses/reinforcement-learning/modules/module_04_function_approximation/guides/02_linear_methods_guide.md`, line 527:

```
- **Builds on:** Guide 01 (function approximation motivation and feature vectors), Module 02 (tabular TD(0) — linear methods are the same algorithm with generalization added)
```

Tabular TD(0) is **Module 03**, not Module 02 (Monte Carlo).

**Fix:** Change to: `Module 03 (tabular TD(0) — linear methods are the same algorithm with generalization added)`

---

### Issue 2.3 — Module 06 slide summary mischaracterizes Module 03 content

**Severity: MINOR**

File: `/home/user/course-creator/courses/reinforcement-learning/modules/module_06_policy_gradient/guides/01_policy_gradient_theorem_slides.md`, line 19 (speaker note):

```
- Modules 01-03 covered dynamic programming and Monte Carlo, Module 04-05 covered value function approximation
```

This drops Module 03 (Temporal Difference) from the summary entirely, and groups Module 05 (Deep RL / DQN) as "value function approximation" when it is better described as deep reinforcement learning.

**Fix:** Change to: `Modules 01–03 covered dynamic programming, Monte Carlo, and temporal-difference methods; Modules 04–05 covered function approximation and deep RL (DQN)`

---

### Issue 2.4 — SAC guide cross-reference points to wrong modules

**Severity: MINOR**

File: `/home/user/course-creator/courses/reinforcement-learning/modules/module_07_advanced_policy_optimization/guides/03_sac_guide.md`, line 415:

```
- **Builds on:** Q-learning and actor-critic architectures (Module 4), importance of exploration (Module 1)
```

Two errors:
- Actor-critic architectures are in **Module 06** (Policy Gradient), not Module 04 (Function Approximation)
- "Importance of exploration" is not the subject of Module 01 (Dynamic Programming); exploration is introduced in **Module 02** and **Module 03**

**Fix:** Change to: `Builds on: Q-learning and neural function approximation (Module 05), actor-critic architectures (Module 06), exploration trade-offs (Modules 02–03)`

---

## Check 3 — Content Duplication

**Status: PASS with one WARNING**

Guides are generally theory-focused and notebooks implementation-focused. The following was verified:

- Modules 00–03: Guides contain instructional code snippets (typically 30–45% code by line count) that illustrate theory inline. Companion notebooks provide the runnable environment. No verbatim duplication detected.
- Modules 05–07: Guides contain more substantial code (45–54% code by line count) as algorithmic walkthroughs; companion notebooks expand and run these implementations.

### Warning 3.1 — SAC has no companion notebook; guide contains complete implementation

**Severity: MINOR**

File: `/home/user/course-creator/courses/reinforcement-learning/modules/module_07_advanced_policy_optimization/guides/03_sac_guide.md`

The SAC guide contains a complete PyTorch implementation of SAC (54% code by line count, including `QNetwork`, `GaussianPolicy`, `ReplayBuffer`, and `SACAgent` classes, totalling ~232 lines of code in one block). The Module 07 notebooks are:
- `01_ppo_from_scratch.ipynb` (PPO)
- `02_algorithm_showdown.ipynb` (REINFORCE vs A2C vs PPO comparison)

There is no SAC-specific notebook. The full implementation in the guide is the only runnable artifact for SAC. This inverts the course philosophy ("working code first, theory contextually") by burying the implementation in a guide. Learners cannot run the SAC code without manually extracting it from the guide.

**Recommendation:** Either create `03_sac_from_scratch.ipynb` and reduce the guide's code to a concise conceptual excerpt, or add a clear callout box in the guide noting that the code block is a standalone copy-paste implementation.

---

## Check 4 — Speaker Notes Quality

**Status: PASS**

Spot-checked 3 slide decks: `module_00/guides/01_rl_landscape_slides.md`, `module_05/guides/01_dqn_slides.md`, and `module_07/guides/02_ppo_slides.md`.

Results:

| File | Note count | Avg length | Min length | Format |
|------|-----------|------------|------------|--------|
| `module_00/01_rl_landscape_slides.md` | 16/16 slides | 410 chars | 329 chars | Inline paragraph |
| `module_05/01_dqn_slides.md` | 18/18 slides | 383 chars | 224 chars | Header + bullets |
| `module_07/02_ppo_slides.md` | 14/14 slides | 418 chars | 361 chars | Inline paragraph |

All notes are substantive and go beyond restating bullet points. Notes in modules 00–04 use inline paragraph format (`<!-- Speaker notes: ... -->`); modules 05–07 use a multi-line bullet format (`<!--\nSpeaker notes: Key talking points\n- bullet\n-->`). Both formats render correctly in Marp.

A full sweep of all 32 slide decks confirms no slides are missing speaker notes. Minimum note length across the entire course is 206 characters (found in `module_03/02_sarsa_slides.md`), which is still substantive.

---

## Check 5 — Code Style

**Status: PASS**

Spot-checked 3 exercise files (`module_00`, `module_03`, `module_06`) and all 5 code files in `templates/` and `recipes/`.

Results:

| File | Classes | Functions | Issues |
|------|---------|-----------|--------|
| `module_00/exercises/exercises.py` | none | 8 snake_case | none |
| `module_03/exercises/exercises.py` | none | 8 snake_case | none |
| `module_06/exercises/exercises.py` | none | 6 snake_case | none |
| `templates/rl_agent_template.py` | 5 PascalCase | snake_case methods | none |
| `templates/experiment_tracker_template.py` | 3 PascalCase | snake_case methods | none |
| `templates/environment_wrapper_template.py` | 4 PascalCase | snake_case methods | none |
| `recipes/common_patterns.py` | `_Transition`, `_Batch` | snake_case | See note |
| `recipes/evaluation_recipes.py` | `_Agent`, `_Env` | snake_case | See note |

**Note on private Protocol classes:** `recipes/common_patterns.py` and `recipes/evaluation_recipes.py` define private stub classes prefixed with `_` (`_Transition`, `_Batch`, `_Agent`, `_Env`). These use lowercase after the underscore, which is a common Python convention for private internal types. Since these are `NamedTuple` or `Protocol` definitions used only internally, this is acceptable — the underscore prefix signals private scope. Not a violation.

All 10 exercise files were checked for the full suite: no snake_case violations in function names, no PascalCase violations in class names.

---

## Check 6 — Progressive Difficulty

**Status: PASS with one WARNING**

Actual dependencies measured from notebook import statements across all modules:

| Module | Expected (per checklist) | Actual |
|--------|--------------------------|--------|
| 00 — Foundations | numpy only | numpy, **matplotlib** |
| 01 — Dynamic Programming | numpy + matplotlib | numpy, matplotlib |
| 02 — Monte Carlo | numpy + matplotlib | numpy, matplotlib |
| 03 — Temporal Difference | numpy + matplotlib | numpy, matplotlib |
| 04 — Function Approximation | numpy + matplotlib + gymnasium | numpy, matplotlib, gymnasium |
| 05 — Deep RL | numpy + matplotlib + gymnasium + torch | numpy, matplotlib, gymnasium, torch |
| 06 — Policy Gradient | numpy + matplotlib + gymnasium + torch | numpy, matplotlib, gymnasium, torch |
| 07 — Advanced Policy Opt. | numpy + matplotlib + gymnasium + torch | numpy, matplotlib, gymnasium, torch |
| 08 — Model-Based RL | (expected torch) | numpy, matplotlib only |
| 09 — Frontiers | (assumed all prior) | numpy, matplotlib only |

### Warning 6.1 — Module 00 uses matplotlib (violates "numpy only" requirement)

**Severity: MINOR**

All three Module 00 notebooks import `matplotlib`:
- `01_agent_environment_loop.ipynb`
- `02_mdp_builder.ipynb`
- `03_bellman_equations_lab.ipynb`

The checklist specifies Module 0 should require only numpy. Matplotlib is used for environment visualizations and value function heatmaps. In practice, matplotlib is ubiquitous and this will not block learners, but it contradicts the stated dependency level.

**Options:** (a) Accept this and update the checklist documentation, or (b) replace matplotlib visualizations with text-based representations (ASCII grids) in Module 00 notebooks.

### Info 6.2 — Modules 08–09 notebooks use only numpy despite covering deep-RL topics in guides

**Severity: INFO**

Module 08 (Model-Based RL) guide 03 (`world_models_guide.md`) contains a full PyTorch implementation of a World Models VAE+RNN, but both Module 08 notebooks (`01_dyna_q.ipynb`, `02_mcts_tic_tac_toe.ipynb`) use only numpy. Module 09 notebooks (`01_offline_rl_basics.ipynb`, `02_rl_trading_environment.ipynb`) also use only numpy despite covering algorithms that in practice require deep RL.

This is a deliberate simplification (tabular Dyna-Q and pure-numpy MCTS are pedagogically sound) but creates an inconsistency with the guides, which use PyTorch. No action required unless course goals demand runnable deep implementations for modules 08–09.

---

## Check 7 — Slide Deck Rendering

**Status: PASS**

Checked all 32 slide decks for:

| Check | Result |
|-------|--------|
| Frontmatter: `marp: true` present | PASS — all 32 decks |
| Frontmatter: `theme: course` (not `default`) | PASS — all 32 decks |
| Frontmatter: `math: mathjax` present | PASS — all 32 decks |
| HTML `<div>` / `</div>` balance | PASS — all balanced |
| `$$` math block balance (even count) | PASS — all balanced |
| Mermaid blocks properly fenced with ` ```mermaid ` | PASS — no bare graph directives found |

No rendering issues detected. The initial scan flagged `$ var` patterns as potential math-delimiter errors, but manual inspection confirmed these are all cases where math ends with `$` followed by English text (e.g., "reward `$r$` is received") — not mis-formed math expressions.

---

## Summary: Ready for Learners?

The course is **conditionally ready** for learners to start at Module 0 and progress linearly, with the following qualifications:

**Must fix before publishing:**

1. **Check 1.1 / CRITICAL:** Replace `\varepsilon` with `\epsilon` for exploration rate in 9 files across modules 02 and 03. Learners will encounter a different symbol for the same concept without warning.

2. **Check 2.1 / CRITICAL:** Fix `module_04/guides/01_why_function_approximation_guide.md` line 287 — "Module 02 (tabular TD methods)" should be "Module 03 (tabular TD methods)" and "Module 03 (policy gradient concepts)" should be "Module 06 (policy gradient concepts)".

3. **Check 2.2 / CRITICAL:** Fix `module_04/guides/02_linear_methods_guide.md` line 527 — "Module 02 (tabular TD(0))" should be "Module 03 (tabular TD(0))".

**Should fix before publishing:**

4. **Check 2.3 / MINOR:** Fix Module 06 slides line 19 speaker note — summary of Modules 01–03 drops TD.

5. **Check 2.4 / MINOR:** Fix SAC guide "Builds on" section — actor-critic references Module 04 instead of Module 06.

6. **Check 1.2 / MINOR:** Add clarification note in SAC guide that `$\alpha$` means temperature, not learning rate.

7. **Check 3.1 / MINOR:** SAC has no companion notebook. Add callout in guide or create `03_sac_from_scratch.ipynb`.

**Consider fixing:**

8. **Check 6.1 / MINOR:** Module 00 notebooks use matplotlib despite "numpy only" specification. Either update documentation or remove matplotlib from the three Module 00 notebooks.

9. **Check 1.3 / MINOR:** Add a clarifying note in PPO content that `$r_t(\theta)$` denotes the policy ratio, not reward.
