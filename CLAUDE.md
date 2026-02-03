# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **university-level course creation system** for producing advanced educational materials. Contains 7 courses in development with comprehensive modules, guides, notebooks, and assessments.

**Primary Purpose:** Generate rigorous yet accessible course content including written guides, interactive Jupyter notebooks, auto-graded exercises, and assessment materials.

## Repository Structure

```
courses/                    # All course content (7 courses)
├── bayesian-commodity-forecasting/   # Bayesian time series for trading
├── agentic-ai-llms/                  # LLM agents and multi-agent systems
├── dataiku-genai/                    # GenAI on Dataiku platform
├── genai-commodities/                # GenAI for commodity trading
├── genetic-algorithms-feature-selection/
├── hidden-markov-models/
└── panel-regression/

.claude/                    # Claude Code configuration
├── commands/               # 15 slash commands (create-course, etc.)
└── agents/                 # 19 specialized agents

.claude_prompts/            # Workflow templates
├── course_creator.md       # Full course creation framework
└── CLAUDE.md               # Standard workflow guide
```

## Course Structure Convention

Each course follows this structure:
```
course-name/
├── README.md               # Course overview
├── syllabus/               # Syllabus, learning objectives, schedule
├── modules/
│   ├── module_00_foundations/  # Prerequisites and setup
│   │   ├── README.md
│   │   ├── guides/         # Written conceptual explanations
│   │   ├── notebooks/      # Interactive Jupyter notebooks
│   │   └── assessments/    # Quizzes, exercises
│   └── module_N_[topic]/
├── capstone/               # Final project specification
└── resources/              # Glossary, environment setup, bibliography
```

## Key Slash Commands

| Command | Usage |
|---------|-------|
| `/create-course [topic]` | Initialize new course or module |
| `/create-course --module [name]` | Create single module |
| `/create-course --notebook [topic]` | Create Jupyter notebook |
| `/create-course --assessment [type]` | Create quiz/project/rubric |

## Key Agents for Course Development

| Agent | Use For |
|-------|---------|
| `course-developer` | Educational content, course architecture |
| `notebook-author` | Jupyter notebooks, interactive exercises |
| `assessment-designer` | Quizzes, rubrics, capstone projects |
| `python-pro` | Python code in notebooks |
| `ml-engineer` | ML/DS course content |

## Content Creation Guidelines

### Written Guides
Every concept guide must include:
- **In Brief** - 1-2 sentence summary
- **Key Insight** - Core idea in plain language
- **Formal Definition** - Precise technical definition
- **Intuitive Explanation** - Analogy or visual approach
- **Code Implementation** - Minimal working example
- **Common Pitfalls** - What to avoid
- **Practice Problems** - Progressive difficulty

### Jupyter Notebooks
- Learning objectives stated upfront
- Markdown explanation before every code cell
- Comments explain "why" not just "what"
- Exercises require modification (not just run)
- Auto-graded tests with helpful error messages
- Visual outputs for complex concepts

### Quality Standards
- **No mocks or stubs** - Complete working implementations only
- **Multiple explanations** - Mathematical, intuitive, visual
- **Real datasets** - Every exercise uses real data
- **Progressive complexity** - Foundation → Core → Extension

## Workflow Guidelines

1. **Planning** - Write plans to `.claude_plans/projectplan.md`
2. **Tests** - Store all tests in `tests/` directory
3. **Reference** - Check `.claude_prompts/course_creator.md` for full framework
4. **No orphan files** - Everything in appropriate folder location

## File Naming Conventions

- Modules: `module_NN_descriptive_name/`
- Guides: `01_concept_name.md`, `02_next_concept.md`
- Notebooks: `01_topic_intro.ipynb`, `02_implementation.ipynb`
- Assessments: `quiz.md`, `coding_exercises.py`, `peer_review_rubric.md`
