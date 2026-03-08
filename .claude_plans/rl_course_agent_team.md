# Reinforcement Learning Course — Agent Team Orchestration Prompt

## Master Prompt

Copy and paste the prompt below into a new Claude Code session to launch the full agent team.

---

```
Create a comprehensive university-level course on **Reinforcement Learning** at path `courses/reinforcement-learning/`. Follow the structure, conventions, and philosophy defined in CLAUDE.md and `.claude_prompts/course_creator.md`. Use the `multi-armed-bandits-ab-testing` course as the structural reference.

## Course Scope

**Target audience:** Graduate students and ML practitioners
**Prerequisites:** Python, probability, linear algebra, basic ML (supervised learning)
**Theme:** Working code first, commodity/finance application flavor where natural, general RL otherwise

### Module Plan (10 modules)

| Module | Topic | Key Algorithms / Concepts |
|--------|-------|--------------------------|
| `module_00_foundations` | RL Foundations & MDP Formalism | Agent-environment loop, states, actions, rewards, discount factor, Markov property, Bellman equations |
| `module_01_dynamic_programming` | Dynamic Programming | Policy evaluation, policy iteration, value iteration, contraction mapping |
| `module_02_monte_carlo_methods` | Monte Carlo Methods | First-visit MC, every-visit MC, MC control, importance sampling, off-policy MC |
| `module_03_temporal_difference` | Temporal Difference Learning | TD(0), SARSA, Q-Learning, Expected SARSA, double Q-learning, TD(λ) |
| `module_04_function_approximation` | Function Approximation | Linear FA, tile coding, neural network FA, semi-gradient methods, deadly triad |
| `module_05_deep_rl` | Deep Reinforcement Learning | DQN, experience replay, target networks, Double DQN, Dueling DQN, Rainbow |
| `module_06_policy_gradient` | Policy Gradient Methods | REINFORCE, baseline subtraction, actor-critic, A2C, A3C, GAE |
| `module_07_advanced_policy_optimization` | Advanced Policy Optimization | PPO, TRPO, natural policy gradient, SAC, entropy regularization |
| `module_08_model_based_rl` | Model-Based RL | Dyna-Q, MCTS, world models, MuZero, planning vs learning trade-offs |
| `module_09_frontiers` | Frontiers & Applications | Multi-agent RL, offline RL, RLHF, safe RL, RL for trading/portfolio optimization |

---

## Phase 1 — Architecture & Quick-Starts

**Agent:** `course-developer`
**Task:** Create the full directory scaffold and foundational content.

### Deliverables
1. Directory tree: `modules/module_00` through `module_09`, each with `guides/`, `notebooks/`, `exercises/`, `resources/` subdirectories
2. `quick-starts/` — 4 entry-point notebooks:
   - `00_your_first_rl_agent.ipynb` (gridworld, <2 min)
   - `01_q_learning_cartpole.ipynb` (classic control)
   - `02_policy_gradient_starter.ipynb` (REINFORCE on simple env)
   - `03_deep_rl_playground.ipynb` (DQN on Atari-lite)
3. `templates/` — 3 production scaffolds:
   - `rl_agent_template.py` (base agent class with train/act/evaluate)
   - `environment_wrapper_template.py` (Gymnasium wrapper)
   - `experiment_tracker_template.py` (logging, checkpointing, plotting)
4. `recipes/` — `common_patterns.py`, `evaluation_recipes.py`
5. `projects/` — 3 portfolio projects (beginner/intermediate/advanced) with `PROJECT_SUMMARY.md`

### Success Criteria
- [ ] All 10 module directories exist with correct subdirectory structure
- [ ] Every quick-start notebook runs end-to-end with `gymnasium` (no external API keys)
- [ ] Templates contain zero TODOs/stubs — fully working code
- [ ] Each project spec has clear problem statement, dataset source, expected deliverables, and self-assessment checklist
- [ ] File naming follows `snake_case` convention; no orphan files

---

## Phase 2 — Guides & Slide Decks (per module, parallelizable)

**Agent:** `course-developer` (structure) + `notebook-author` (content)
**Task:** For each of the 10 modules, create guides and companion slide decks.

### Deliverables per module
- 2–4 concept guides (`guides/01_concept_guide.md`, `guides/02_theory_guide.md`, …)
- Matching slide decks (`guides/01_concept_slides.md`, `guides/02_theory_slides.md`, …)
- `guides/cheatsheet.md` — one-page reference

### Success Criteria
- [ ] Every guide file has a companion `_slides.md` file (1:1 mapping)
- [ ] All slide decks have correct Marp frontmatter: `marp: true`, `theme: course`, `paginate: true`, `math: mathjax`
- [ ] Every slide has `<!-- Speaker notes: ... -->` with presenter notes
- [ ] Slide decks render without errors: `npx @marp-team/marp-cli --html --theme-set resources/themes/course-theme.css -- "courses/reinforcement-learning/**/*_slides.md"`
- [ ] Guides include: formal definition, intuitive explanation, code snippet, at least one diagram (Mermaid or described), and "Common Pitfalls" section
- [ ] Math notation is consistent across all modules (same symbols for state $s$, action $a$, reward $r$, policy $\pi$, value $V$, action-value $Q$)
- [ ] No factual errors in Bellman equations, update rules, or convergence conditions
- [ ] Cheatsheet per module fits on one printed page

---

## Phase 3 — Jupyter Notebooks (per module, parallelizable)

**Agent:** `notebook-author` + `ml-engineer`
**Task:** Create 2–3 interactive notebooks per module (15 min each).

### Deliverables per module
- `notebooks/01_*.ipynb` — concept introduction with visualization
- `notebooks/02_*.ipynb` — from-scratch implementation
- `notebooks/03_*.ipynb` (optional) — comparison/lab exercise

### Success Criteria
- [ ] Every notebook states learning objectives, prerequisites, and estimated time (≤15 min)
- [ ] Every code cell has a preceding markdown cell explaining what comes next
- [ ] All notebooks execute top-to-bottom without errors in a fresh Python 3.10+ environment with `gymnasium`, `numpy`, `matplotlib`, `torch` (for modules 5–7)
- [ ] Implementations are from scratch where pedagogically valuable (not just library calls)
- [ ] Each notebook has ≥2 interactive exercises with `# YOUR CODE HERE` blocks and solution cells
- [ ] Visualizations: every module has at least one reward curve, one policy heatmap or state-value plot
- [ ] No mock data — use real Gymnasium environments or real financial datasets
- [ ] Notebooks use `np.random.seed(42)` or equivalent for reproducibility

---

## Phase 4 — Exercises & Self-Checks

**Agent:** `python-pro`
**Task:** Create `exercises/exercises.py` for each module.

### Deliverables per module
- `exercises/exercises.py` — 3–5 self-check exercises with assert-based validation

### Success Criteria
- [ ] Each exercise has: docstring with problem statement, hints, and expected behavior
- [ ] Exercises progress from basic recall to implementation to extension
- [ ] All exercises are runnable standalone (`python exercises.py` prints pass/fail)
- [ ] No grading rubrics, quizzes, or submission infrastructure (self-check only)
- [ ] Solutions are inline (not in a separate `solutions/` directory) with clear `# SOLUTION` markers

---

## Phase 5 — Review Iteration 1: Completeness Audit

**Agent:** `code-reviewer` + `Explore`
**Task:** Audit every file against the success criteria above.

### Checklist
- [ ] Count all guide/slides pairs — verify 1:1 mapping, ≥20 pairs total
- [ ] Count all notebooks — verify ≥20 total, all execute cleanly
- [ ] Verify every module has: guides/, notebooks/, exercises/, resources/ populated
- [ ] Verify quick-starts, templates, recipes, projects all exist and are non-empty
- [ ] Run `npx @marp-team/marp-cli --html --theme-set resources/themes/course-theme.css -- "courses/reinforcement-learning/**/*_slides.md"` and confirm zero errors
- [ ] Check no files use `theme: default` (must be `theme: course`)
- [ ] Check no orphan files outside the directory structure
- [ ] Produce a gap report in `.claude_plans/rl_completeness_audit.md` listing every missing or incomplete item

### Success Criteria for this phase
- [ ] Gap report exists and is actionable
- [ ] All critical gaps (missing files, broken notebooks, wrong frontmatter) are identified
- [ ] Each gap has a specific remediation action listed

---

## Phase 6 — Review Iteration 2: Factual Accuracy

**Agent:** `ml-engineer` + `code-reviewer`
**Task:** Verify technical correctness of all RL content.

### Checklist
- [ ] Bellman expectation equation: $V^\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V^\pi(s')]$
- [ ] Bellman optimality equation: $V^*(s) = \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma V^*(s')]$
- [ ] Q-learning update: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$
- [ ] SARSA update: $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s',a') - Q(s,a)]$
- [ ] Policy gradient theorem: $\nabla J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)]$
- [ ] PPO clipped objective: $L^{CLIP}(\theta) = \mathbb{E}[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$
- [ ] Convergence claims: TD(0) converges under tabular + decaying step-sizes; Q-learning converges to optimal under sufficient exploration; policy gradient converges to local optimum
- [ ] No confusion between on-policy (SARSA) and off-policy (Q-learning) descriptions
- [ ] DQN correctly describes experience replay AND target network (both required)
- [ ] Actor-critic correctly separates policy network (actor) from value network (critic)
- [ ] All code implementations match their mathematical descriptions
- [ ] Produce accuracy report in `.claude_plans/rl_accuracy_audit.md`

### Success Criteria for this phase
- [ ] Every equation in guides/slides matches Sutton & Barto (2nd ed) or original papers
- [ ] Every algorithm implementation matches its pseudocode description
- [ ] No conflation of similar but distinct concepts (e.g., TD error vs advantage, return vs reward)
- [ ] All convergence/divergence claims are correctly qualified

---

## Phase 7 — Review Iteration 3: Polish & Consistency

**Agent:** `code-reviewer`
**Task:** Final quality pass.

### Checklist
- [ ] Consistent notation across all 10 modules (symbol table in module_00 propagated)
- [ ] All cross-references between modules are correct ("As we saw in Module 3…" actually refers to Module 3 content)
- [ ] No duplicate content between guides and notebooks (guides = theory, notebooks = implementation)
- [ ] Speaker notes on every slide are substantive (not just restating bullet points)
- [ ] Code style: all Python follows snake_case, PascalCase classes, SCREAMING_SNAKE constants
- [ ] Every `resources/` directory has at least a `readings.md` or relevant figure
- [ ] Progressive difficulty: Module 0 requires only NumPy; Module 9 assumes all prior modules
- [ ] Produce final report in `.claude_plans/rl_polish_report.md`

### Success Criteria for this phase
- [ ] Zero consistency issues across modules
- [ ] All slide decks render cleanly
- [ ] Course is ready for a learner to start at Module 0 and progress linearly without gaps

---

## Execution Strategy

Run phases in this order, parallelizing where noted:

```
Phase 1 (sequential — creates structure)
    ↓
Phase 2 + Phase 3 + Phase 4 (parallel — all per-module, independent)
    ↓
Phase 5 — Completeness audit → fix gaps
    ↓
Phase 6 — Accuracy audit → fix errors
    ↓
Phase 7 — Polish audit → fix consistency
    ↓
Commit & push to branch
```

Within Phases 2–4, parallelize across modules using multiple agents:
- Launch `course-developer` agents for modules 0–4 guides/slides simultaneously
- Launch `course-developer` agents for modules 5–9 guides/slides simultaneously
- Launch `notebook-author` + `ml-engineer` agents for notebooks in parallel batches
- Launch `python-pro` agents for exercises in parallel

After each review phase (5, 6, 7), fix all identified issues before proceeding to the next review.

## Final Commit

After all 7 phases complete:
```bash
git add courses/reinforcement-learning/
git commit -m "Add reinforcement learning course: 10 modules, guides, slides, notebooks, exercises, quick-starts, templates, recipes, projects"
git push -u origin claude/rl-course-agent-team-AtCdu
```
```

---

## How to Use This Prompt

1. Copy the entire block between the ``` markers above
2. Paste into a fresh Claude Code session (or this one)
3. Claude will orchestrate the agent team across all 7 phases
4. Review the three audit reports in `.claude_plans/` after completion:
   - `rl_completeness_audit.md`
   - `rl_accuracy_audit.md`
   - `rl_polish_report.md`
