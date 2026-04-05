# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **practical-first course creation system** producing professional-grade educational materials across 19 courses. Content includes Marp slide decks, Jupyter notebooks, Python templates, and markdown guides for topics spanning ML, GenAI, econometrics, and trading systems.

**Technology Stack**: Python (course content), Marp/Node.js (slide rendering), Jupyter notebooks, Streamlit (course browser)

## Build & Run Commands

### Streamlit Course Browser
```bash
pip install -r requirements.txt          # streamlit>=1.30.0, pandas
streamlit run app.py                     # Local dev server
```
The app (`app.py`) is a course content browser with navigation across all courses, modules, slides, notebooks, and guides. Theme config lives in `.streamlit/config.toml`.

### Slide Decks (Marp)
```bash
# Render a single slide deck to HTML
npx @marp-team/marp-cli --html --theme-set resources/themes/course-theme.css -- path/to/slides.md

# Render all slides for one course
npx @marp-team/marp-cli --html --theme-set resources/themes/course-theme.css -- "courses/COURSE_NAME/**/*_slides.md"

# Render all decks across all courses
npx @marp-team/marp-cli --html --theme-set resources/themes/course-theme.css -- "courses/**/*_slides.md"
```

The custom theme at `resources/themes/course-theme.css` MUST be passed via `--theme-set` on every render. All slide decks use `theme: course` in their frontmatter.

### Slide Deck Frontmatter (required)
```yaml
---
marp: true
theme: course
paginate: true
math: mathjax
---
```

## Architecture

### Content Structure
```
courses/                            # 19 courses
├── {course-name}/
│   ├── modules/
│   │   └── module_NN_{topic}/
│   │       ├── guides/             # Markdown concept guides + *_slides.md companion decks
│   │       ├── notebooks/          # 15-min Jupyter micro-notebooks
│   │       ├── exercises/          # Self-check Python exercises (ungraded)
│   │       └── resources/          # Readings, figures
│   ├── quick-starts/               # Entry-point notebooks (run in <2 min)
│   ├── templates/                  # Production-ready Python scaffolds
│   ├── recipes/                    # Copy-paste code patterns
│   └── projects/                   # Portfolio projects (not graded)
app.py                              # Streamlit course browser
resources/themes/course-theme.css   # Marp CSS theme for all slide decks
.claude_prompts/course_creator.md   # Full course creation framework
.claude_plans/                      # Project plans and progress tracking
```

### Slide Deck Convention
Every guide markdown (`guides/01_concept_guide.md`) has a companion slide deck (`guides/01_concept_slides.md`). The `_slides.md` suffix distinguishes slide decks from regular guides.

Slide decks use:
- `<!-- _class: lead -->` for title/section slides
- `<!-- Speaker notes: ... -->` for presenter notes (every slide should have them)
- `<div class="columns">` for two-column layouts
- Mermaid diagrams via ` ```mermaid ` fenced blocks
- LaTeX math via `$...$` and `$$...$$` (MathJax)

### Courses (19 total)
- `agentic-ai-llms` - LLM agents and multi-agent systems
- `agentic-ai-practical` - Practical agentic AI
- `ai-engineer-fundamentals` - AI engineering mindset and systems
- `bayesian-commodity-forecasting` - Bayesian time series for trading
- `bayesian-prompt-engineering` - Bayesian prompt engineering
- `captum-neural-network-interpretability` - Neural network interpretability with Captum
- `causalpy-interrupted-time-series` - Causal inference with interrupted time series
- `dataiku-genai` - GenAI on Dataiku platform
- `double-machine-learning` - Double/debiased machine learning
- `dynamic-factor-models` - Econometric factor models
- `genai-commodities` - GenAI for commodity trading
- `genetic-algorithms-feature-selection` - GA for feature selection
- `hidden-markov-models` - HMM theory and financial applications
- `midas-mixed-frequency-nowcasting` - Mixed-frequency nowcasting with MIDAS
- `multi-armed-bandits-ab-testing` - Bandits for optimization
- `panel-regression` - Panel data econometrics
- `power-automate` - Microsoft Power Automate
- `reinforcement-learning` - Reinforcement learning
- `time-series-forecasting-neuralforecast` - Time series with NeuralForecast

## Content Creation Philosophy

- **Working code first, theory contextually** - Every concept starts with runnable code
- **Visual-first** - Diagram before text, always
- **15-minute max** for any single notebook
- **Copy-paste ready** - All code works in learner's own projects
- **Portfolio over grades** - No quizzes, exams, or grading rubrics
- **No mocks or stubs** - Complete working implementations with real datasets only

## Key Slash Command

`/create-course` is the primary command for content generation. It supports flags: `--module`, `--notebook`, `--guide`, `--quick-start`, `--template`, `--recipe`, `--project`, `--visual`.

## Key Agents

| Agent | Use For |
|-------|---------|
| `course-developer` | Course architecture, module structure, learning paths |
| `notebook-author` | Jupyter notebooks, interactive content |
| `python-pro` | Production templates, code patterns |
| `ml-engineer` | ML/DS content, model templates |

## Anti-Patterns

- No `quiz.md`, `grading_rubric.md`, `final_exam.md`, or `assignment_submission.md`
- No 90-minute notebooks (break into 15-min pieces)
- No theory-first content (always start with working code)
- No synthetic/mock data (use real datasets)
- Never use `theme: default` in slide decks (always `theme: course`)
- Use `assessment-designer` agent sparingly (self-check exercises only, no formal assessments)
