# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **practical-first course creation system** producing professional-grade educational materials across 11 courses. Content includes Marp slide decks, Jupyter notebooks, Python templates, and markdown guides for topics spanning ML, GenAI, econometrics, and trading systems.

**Technology Stack**: Python (course content), Marp/Node.js (slide rendering), Jupyter notebooks
**Primary Language**: Markdown + Python

## Build & Render Commands

### Slide Decks (Marp)
```bash
# Render a single slide deck to HTML
npx @marp-team/marp-cli --html --theme-set resources/themes/course-theme.css -- path/to/slides.md

# Render all slides for one course
npx @marp-team/marp-cli --html --theme-set resources/themes/course-theme.css -- "courses/COURSE_NAME/**/*_slides.md"

# Render all 229 decks across all courses (run per-course in parallel)
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
courses/                         # 11 courses
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
Every guide markdown (`guides/01_concept_guide.md`) has a companion slide deck (`guides/01_concept_slides.md`). The `_slides.md` suffix distinguishes slide decks from regular guides. There are currently 229 slide decks and 229 rendered HTML files.

Slide decks use:
- `<!-- _class: lead -->` for title/section slides
- `<!-- Speaker notes: ... -->` for presenter notes (every slide should have them)
- `<div class="columns">` for two-column layouts
- Mermaid diagrams via ` ```mermaid ` fenced blocks
- LaTeX math via `$...$` and `$$...$$` (MathJax)

### Courses (11 total)
- `agentic-ai-llms` (32 decks) - LLM agents and multi-agent systems
- `agentic-ai-practical` - Practical agentic AI
- `ai-engineer-fundamentals` (11 decks) - AI engineering mindset and systems
- `bayesian-commodity-forecasting` (27 decks) - Bayesian time series for trading
- `dataiku-genai` (14 decks) - GenAI on Dataiku platform
- `dynamic-factor-models` (28 decks) - Econometric factor models
- `genai-commodities` (21 decks) - GenAI for commodity trading
- `genetic-algorithms-feature-selection` (20 decks) - GA for feature selection
- `hidden-markov-models` (16 decks) - HMM theory and financial applications
- `multi-armed-bandits-ab-testing` (38 decks) - Bandits for optimization
- `panel-regression` (22 decks) - Panel data econometrics

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
