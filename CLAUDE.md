# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **practical-first course creation system** producing professional-grade educational materials across 17 courses. Content includes Marp slide decks, Jupyter notebooks, Python templates, and markdown guides for topics spanning ML, GenAI, econometrics, reinforcement learning, causal inference, and trading systems.

**Technology Stack**: Python (course content), Marp/Node.js (slide rendering), Jupyter notebooks
**Primary Language**: Markdown + Python

## Build & Render Commands

### Slide Decks (Marp)
```bash
# Render a single slide deck to HTML
npx @marp-team/marp-cli --html --theme-set resources/themes/course-theme.css -- path/to/slides.md

# Render all slides for one course
npx @marp-team/marp-cli --html --theme-set resources/themes/course-theme.css -- "courses/COURSE_NAME/**/*_slides.md"

# Render all 357 decks across all courses (run per-course in parallel)
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
courses/                         # 17 courses
├── {course-name}/
│   ├── modules/
│   │   └── module_NN_{topic}/
│   │       ├── guides/          # Markdown concept guides + *_slides.md companion decks
│   │       ├── notebooks/       # 15-min Jupyter micro-notebooks
│   │       ├── exercises/       # Self-check Python exercises (ungraded)
│   │       └── resources/       # Readings, figures
│   ├── quick-starts/            # Entry-point notebooks (run in <2 min)
│   ├── templates/               # Production-ready Python scaffolds
│   ├── recipes/                 # Copy-paste code patterns
│   └── projects/                # Portfolio projects (not graded)
resources/themes/course-theme.css  # Marp CSS theme for all slide decks
.claude_prompts/course_creator.md  # Full course creation framework
.claude_plans/                     # Project plans and progress tracking
```

### Slide Deck Convention
Every guide markdown (`guides/01_concept_guide.md`) has a companion slide deck (`guides/01_concept_slides.md`). The `_slides.md` suffix distinguishes slide decks from regular guides. There are currently 357 slide decks and 259 rendered HTML files (98 pending render).

Slide decks use:
- `<!-- _class: lead -->` for title/section slides
- `<!-- Speaker notes: ... -->` for presenter notes (every slide should have them)
- `<div class="columns">` for two-column layouts
- Mermaid diagrams via ` ```mermaid ` fenced blocks
- LaTeX math via `$...$` and `$$...$$` (MathJax)

### Courses (17 total)
- `agentic-ai-llms` (32 decks, 20 notebooks) - LLM agents and multi-agent systems
- `agentic-ai-practical` (3 notebooks, 2 templates, 5 projects) - Practical agentic AI
- `ai-engineer-fundamentals` (11 decks, 1 notebook, 7 projects) - AI engineering mindset and systems
- `bayesian-commodity-forecasting` (27 decks, 24 notebooks) - Bayesian time series for trading
- `captum-neural-network-interpretability` (22 decks, 28 notebooks, 22 guides) - Neural network interpretability with Captum
- `causalpy-interrupted-time-series` (21 decks, 25 notebooks, 21 guides) - Causal inference and interrupted time series
- `dataiku-genai` (14 decks, 10 notebooks) - GenAI on Dataiku platform
- `double-machine-learning` (10 decks, 11 notebooks, 10 guides) - Double/debiased ML for causal inference
- `dynamic-factor-models` (28 decks, 15 notebooks) - Econometric factor models
- `genai-commodities` (21 decks, 17 notebooks) - GenAI for commodity trading
- `genetic-algorithms-feature-selection` (20 decks, 12 notebooks) - GA for feature selection
- `hidden-markov-models` (16 decks, 14 notebooks) - HMM theory and financial applications
- `midas-mixed-frequency-nowcasting` (23 decks, 28 notebooks, 23 guides) - Mixed-frequency nowcasting models
- `multi-armed-bandits-ab-testing` (38 decks, 33 notebooks, 14 projects) - Bandits for optimization
- `panel-regression` (22 decks, 13 notebooks) - Panel data econometrics
- `power-automate` (20 decks, 13 notebooks, 20 guides) - Microsoft Power Automate workflows
- `reinforcement-learning` (32 decks, 26 notebooks, 32 guides) - RL theory and applications

## Content Creation Philosophy

- **Working code first, theory contextually** - Every concept starts with runnable code
- **Visual-first** - Diagram before text, always
- **15-minute max** for any single notebook
- **Copy-paste ready** - All code works in learner's own projects
- **Portfolio over grades** - No quizzes, exams, or grading rubrics
- **No mocks or stubs** - Complete working implementations only

## Key Slash Command

`/create-course` is the primary command for content generation. It supports flags: `--module`, `--notebook`, `--guide`, `--quick-start`, `--template`, `--recipe`, `--project`, `--visual`.

## Key Agents

| Agent | Use For |
|-------|---------|
| `course-developer` | Course architecture, module structure, learning paths |
| `notebook-author` | Jupyter notebooks, interactive content |
| `python-pro` | Production templates, code patterns |
| `ml-engineer` | ML/DS content, model templates |

## Workflow Conventions

- Plans go in `.claude_plans/`
- Tests go in `tests/`
- Reference `.claude_prompts/course_creator.md` for the full creation framework
- No orphan files in root - everything in proper directory structure
- Use `assessment-designer` agent sparingly (self-check exercises only, no formal assessments)

## Anti-Patterns

- No `quiz.md`, `grading_rubric.md`, `final_exam.md`, or `assignment_submission.md`
- No 90-minute notebooks (break into 15-min pieces)
- No theory-first content (always start with working code)
- No synthetic/mock data (use real datasets)
- Never use `theme: default` in slide decks (always `theme: course`)
