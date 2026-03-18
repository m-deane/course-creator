# Course Creator

A practical-first course creation system producing professional-grade educational materials across **17 courses** spanning ML, GenAI, econometrics, causal inference, reinforcement learning, and trading systems.

## Content Inventory

| Metric | Count |
|--------|-------|
| Courses | 17 |
| Slide decks (Marp) | 357 |
| Jupyter notebooks | 293 |
| Concept guides | 132 |
| Python templates | 25 |
| Recipes | 37 |
| Portfolio projects | 47 |
| Quick-starts | 27 |
| Exercises | 56 |

## Courses

| Course | Decks | Notebooks | Topic |
|--------|-------|-----------|-------|
| `agentic-ai-llms` | 32 | 20 | LLM agents and multi-agent systems |
| `agentic-ai-practical` | — | 3 | Practical agentic AI |
| `ai-engineer-fundamentals` | 11 | 1 | AI engineering mindset and systems |
| `bayesian-commodity-forecasting` | 27 | 24 | Bayesian time series for trading |
| `captum-neural-network-interpretability` | 22 | 28 | Neural network interpretability with Captum |
| `causalpy-interrupted-time-series` | 21 | 25 | Causal inference and interrupted time series |
| `dataiku-genai` | 14 | 10 | GenAI on Dataiku platform |
| `double-machine-learning` | 10 | 11 | Double/debiased ML for causal inference |
| `dynamic-factor-models` | 28 | 15 | Econometric factor models |
| `genai-commodities` | 21 | 17 | GenAI for commodity trading |
| `genetic-algorithms-feature-selection` | 20 | 12 | GA for feature selection |
| `hidden-markov-models` | 16 | 14 | HMM theory and financial applications |
| `midas-mixed-frequency-nowcasting` | 23 | 28 | Mixed-frequency nowcasting models |
| `multi-armed-bandits-ab-testing` | 38 | 33 | Bandits for optimization |
| `panel-regression` | 22 | 13 | Panel data econometrics |
| `power-automate` | 20 | 13 | Microsoft Power Automate workflows |
| `reinforcement-learning` | 32 | 26 | RL theory and applications |

## Quick Start

### Render Slide Decks

```bash
# Render a single deck
npx @marp-team/marp-cli --html --theme-set resources/themes/course-theme.css -- path/to/slides.md

# Render all decks for one course
npx @marp-team/marp-cli --html --theme-set resources/themes/course-theme.css -- "courses/COURSE_NAME/**/*_slides.md"

# Render all 357 decks
npx @marp-team/marp-cli --html --theme-set resources/themes/course-theme.css -- "courses/**/*_slides.md"
```

### Create New Content

```bash
# Use Claude Code with the /create-course slash command
claude

# Supports: --module, --notebook, --guide, --quick-start, --template, --recipe, --project, --visual
```

## Project Structure

```
courses/                           # 17 courses
├── {course-name}/
│   ├── modules/
│   │   └── module_NN_{topic}/
│   │       ├── guides/            # Concept guides + companion *_slides.md decks
│   │       ├── notebooks/         # 15-min Jupyter micro-notebooks
│   │       ├── exercises/         # Self-check Python exercises (ungraded)
│   │       └── resources/         # Readings, figures
│   ├── quick-starts/              # Entry-point notebooks (run in <2 min)
│   ├── templates/                 # Production-ready Python scaffolds
│   ├── recipes/                   # Copy-paste code patterns
│   └── projects/                  # Portfolio projects (not graded)
resources/themes/course-theme.css  # Marp CSS theme for all slide decks
.claude_prompts/                   # Course creation framework and prompts
.claude_plans/                     # Project plans and progress tracking
```

## Content Philosophy

- **Working code first** — every concept starts with runnable code
- **Visual-first** — diagram before text, always
- **15-minute max** — no notebook exceeds 15 minutes
- **Copy-paste ready** — all code works in the learner's own projects
- **Portfolio over grades** — no quizzes, exams, or grading rubrics
- **No mocks or stubs** — complete working implementations only

## Technology Stack

- **Content**: Python, Markdown, Jupyter notebooks
- **Slides**: [Marp](https://marp.app/) with custom `course` theme
- **Math**: MathJax (LaTeX syntax)
- **Diagrams**: Mermaid fenced blocks in slides

## License

[Your License Here]
