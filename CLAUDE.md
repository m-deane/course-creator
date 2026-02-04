# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **practical-first course creation system** for producing professional-grade educational materials that learners can immediately apply. Contains 8 courses with hands-on notebooks, reusable templates, and portfolio projects.

**Primary Purpose:** Generate practical, immediately-usable course content following the "working code first, theory contextually" philosophy inspired by fast.ai and DataCamp.

## Core Philosophy

**The Shift:**
- FROM: University course with grades → TO: Professional toolkit with portfolio projects
- FROM: "Learn theory, then apply" → TO: "Get it working, then understand why"
- FROM: "Complete all modules in order" → TO: "Grab what you need, when you need it"
- FROM: "Assessment measures learning" → TO: "Building something real demonstrates mastery"

**Key Principles:**
1. **Working code in 2 minutes** - Every concept starts with runnable code
2. **Visual-first explanations** - Diagram before text, always
3. **Copy-paste ready** - All code should work in learner's own projects
4. **15-minute max** - No notebook longer than 15 minutes
5. **Portfolio over grades** - Build real things, not pass tests

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
├── README.md                    # Quick overview + "Start Here" link
├── quick-starts/                # 5-10 min working examples (START HERE)
│   ├── 01_hello_world.ipynb     # Works in 2 minutes, no setup
│   ├── 02_your_first_model.ipynb
│   └── 03_real_data_example.ipynb
├── templates/                   # Production-ready scaffolds (COPY THESE)
│   ├── pipeline_template.py     # Data pipeline scaffold
│   ├── model_template.py        # ML model scaffold
│   ├── api_template.py          # API endpoint scaffold
│   └── dashboard_template.py    # Visualization scaffold
├── recipes/                     # Copy-paste code patterns
│   ├── common_patterns.py       # Frequently-used snippets
│   ├── data_loading.py          # Data ingestion patterns
│   └── troubleshooting.md       # Common errors + fixes
├── concepts/                    # Visual guides + optional deep-dives
│   ├── visual_guides/           # 1-page visual summaries per concept
│   └── deep_dives/              # Optional theory (for curious learners)
├── modules/                     # Structured learning path (REQUIRED for full courses)
│   ├── module_00_foundations/
│   │   ├── README.md            # Module overview + learning objectives
│   │   ├── guides/              # Detailed concept guides (REQUIRED)
│   │   │   ├── 01_concept_guide.md
│   │   │   ├── 02_theory_deep_dive.md
│   │   │   └── cheatsheet.md
│   │   ├── notebooks/           # 15-min micro-notebooks
│   │   ├── exercises/           # Self-check exercises (ungraded)
│   │   └── resources/           # Per-module supporting materials
│   │       ├── additional_readings.md
│   │       └── figures/
│   └── module_N_[topic]/
├── projects/                    # Portfolio projects (NOT graded capstones)
│   ├── project_1_beginner/      # Build something real
│   ├── project_2_intermediate/
│   └── project_3_advanced/
└── resources/
    ├── cheat_sheet.pdf          # 1-page reference card
    ├── setup.md                 # Environment setup
    └── glossary.md              # Key terms
```

**Entry Points (in order of priority):**
1. `quick-starts/` - New learners start here
2. `templates/` - Practitioners grab production code
3. `recipes/` - Developers copy specific patterns
4. `modules/` - Structured learners follow the path (with full guides and resources)

**Full Course Requirements:**
When creating a complete course, ALL of the following are REQUIRED:
- `quick-starts/` with 3-5 working examples
- `modules/` with guides, notebooks, exercises, and resources per module
- `concepts/` with visual guides and optional deep-dives
- `projects/` with at least one portfolio project
- `resources/` with glossary, setup, and cheat sheet

## Key Slash Commands

| Command | Usage |
|---------|-------|
| `/create-course [topic]` | Initialize new course with full structure (modules + guides + resources) |
| `/create-course --quick-start [topic]` | Create a 5-min quick-start notebook |
| `/create-course --template [type]` | Create production-ready template |
| `/create-course --recipe [pattern]` | Create copy-paste code recipe |
| `/create-course --module [name]` | Create single module with guides, notebooks, exercises, resources |
| `/create-course --guide [concept]` | Create detailed concept guide for a module |
| `/create-course --notebook [topic]` | Create 15-min micro-notebook |
| `/create-course --project [level]` | Create portfolio project (beginner/intermediate/advanced) |
| `/create-course --visual [concept]` | Create 60-second visual concept card |

## Key Agents for Course Development

| Agent | Use For |
|-------|---------|
| `course-developer` | Course architecture, learning paths |
| `notebook-author` | Quick-starts, micro-notebooks, interactive content |
| `python-pro` | Production templates, clean code patterns |
| `ml-engineer` | ML/DS content, model templates |
| `technical-researcher` | Industry best practices, API documentation |

**Note:** The `assessment-designer` agent should be used sparingly - only for self-check exercises and portfolio project scaffolds, NOT for formal quizzes or graded assessments.

## Content Creation Guidelines

### Quick-Start Notebooks (HIGHEST PRIORITY)
Every course MUST have quick-starts that:
- Run successfully in < 2 minutes with zero setup
- Use Colab/Binder links (no local installation required)
- Show a complete working example in the first cell
- Include "Modify This" sections for experimentation
- End with "What's Next?" pointers

### Visual Concept Cards (60-Second Format)
Every concept gets a 1-page visual guide:
```
┌─────────────────────────────────────┐
│ CONCEPT NAME                        │
├─────────────────────────────────────┤
│ [Visual: Diagram/Flowchart/Graph]   │
├─────────────────────────────────────┤
│ TL;DR: One sentence explanation     │
├─────────────────────────────────────┤
│ Code (< 15 lines):                  │
│   result = do_the_thing(data)       │
├─────────────────────────────────────┤
│ Common Pitfall: What breaks         │
└─────────────────────────────────────┘
```

### Production Templates ("Steal This Code")
Every template MUST:
- Work out-of-the-box with `python template.py`
- Have clear `# TODO: Customize here` markers
- Use real API endpoints and data sources
- Include production patterns (error handling, logging, config)
- Have a README with deployment instructions

**Template Structure:**
```python
"""
Template Name - Copy and customize for your use case
Works with: [list compatible tools/APIs]
Time to working: X minutes
"""

# CUSTOMIZE THESE
CONFIG_VAR = "your_value"

# COPY THIS ENTIRE BLOCK (production-ready)
def main_function():
    ...  # 20-30 lines max

# RUN IT
if __name__ == "__main__":
    main_function()
```

### Recipes (Copy-Paste Patterns)
Short, focused code snippets for common tasks:
- Each recipe solves ONE specific problem
- Include the problem statement as a comment
- Show input → output clearly
- Max 20 lines per recipe

### Micro-Notebooks (15-Minute Max)
Structure for learning notebooks:
```
├── Goal (1 sentence)
├── Quick Win (working code in < 2 minutes)
├── How It Works (with interactive widget if possible)
├── Modify This (parameter playground)
├── Copy-Paste Template (production-ready)
└── Go Deeper (optional theory links)
```

**Rules:**
- Break 90-min content into 15-min micro-notebooks
- One concept per notebook
- Visual BEFORE text explanation
- Code cells < 20 lines each
- Include `ipywidgets` for parameter exploration where useful

### Deep-Dive Guides (OPTIONAL - for curious learners)
Only create when learners explicitly want theory:
- **TL;DR** - 1-2 sentence summary at top
- **Visual Explanation** - Diagram first
- **Intuitive Analogy** - "It's like..." explanation
- **Formal Definition** - For reference (collapsible)
- **When to Use** - Decision criteria
- **When NOT to Use** - Anti-patterns

### Module Guides (REQUIRED for each module)
Every module MUST include a `guides/` directory with detailed concept explanations:

**Structure:**
```
modules/module_NN_topic/
├── guides/
│   ├── 01_[concept]_guide.md      # Main concept explanation
│   ├── 02_[concept]_deep_dive.md  # Theory and formal details
│   ├── 03_[concept]_guide.md      # Additional concepts as needed
│   └── cheatsheet.md              # Quick reference for module
└── resources/
    ├── additional_readings.md     # Papers, books, links
    └── figures/                   # Diagrams and visual assets
```

**Guide Template (each concept guide MUST include):**
```markdown
# [Concept Name]

## In Brief
[1-2 sentence summary - what it is and why it matters]

## Key Insight
[The core idea in plain language]

## Visual Explanation
[Diagram or flowchart - ALWAYS include]

## Formal Definition
[Precise technical definition with notation]

## Intuitive Explanation
[Analogy for intuition building: "It's like..."]

## Code Implementation
```python
# Minimal working example (< 20 lines)
```

## Common Pitfalls
- [Pitfall 1]: Why it happens and how to avoid
- [Pitfall 2]: Explanation

## Connections
- **Builds on:** [Prerequisite concepts]
- **Leads to:** [Advanced concepts this enables]

## Practice Problems
1. [Conceptual question]
2. [Implementation challenge]
```

**Per-Module Resources (REQUIRED):**
- `additional_readings.md` - Curated papers, books, and external resources
- `figures/` - All diagrams and visual assets for the module

### Quality Standards
- **No mocks or stubs** - Complete working implementations only
- **Visual-first** - Every concept has a diagram
- **Real datasets** - Every example uses real data (Kaggle, UCI, APIs)
- **Copy-paste ready** - All code works in other projects
- **< 20 lines** - Code examples stay short and focused
- **Browser-first** - Colab/Binder links on every notebook

## GenAI Course Requirements (2025+)

All GenAI/LLM courses MUST include these practical components:

### Week 1: Immediate Hands-On
- **API calls in first 10 minutes** - Working Claude/OpenAI call
- **Prompt templates library** - Copy-paste prompts for common tasks
- **Few-shot learning patterns** - Consistent output formatting

### Week 2: RAG Implementation
- **Vector database setup** - Chroma, Pinecone, or pgvector
- **Document chunking strategies** - With trade-off explanations
- **Retrieval evaluation** - Metrics that matter

### Week 3: Agents & Tools
- **Tool-using agents** - Working examples with real APIs
- **Multi-agent orchestration** - Patterns for agent coordination
- **Error handling** - Production-grade retry logic

### Week 4: Production & Evaluation
- **Deployment patterns** - Modal, Railway, or cloud deployment
- **Evaluation frameworks** - RAGAS, LangSmith integration
- **Cost optimization** - Token counting, caching strategies
- **When NOT to use LLMs** - Decision framework

### Required GenAI Notebooks
Every GenAI course must include:
```
quick-starts/
├── 00_your_first_api_call.ipynb      # Works in 2 minutes
├── 01_prompt_templates.ipynb          # Copy-paste ready
├── 02_rag_starter_kit.ipynb           # Production scaffold
├── 03_agent_blueprint.ipynb           # Multi-agent template
└── 04_evaluation_dashboard.ipynb      # Monitor your apps
```

## Portfolio Projects (NOT Assessments)

**Philosophy:** Building something real demonstrates mastery better than passing tests.

### What to Create
- **Portfolio Projects** - Real, deployable applications
- **Self-Check Exercises** - Instant feedback, no grades
- **"Modify This" Challenges** - Extend working code

### What NOT to Create
- Formal quizzes with point values
- Grading rubrics with percentages
- Timed assessments
- Academic-style capstone reports

### Project Structure
```
projects/
├── project_1_beginner/
│   ├── README.md           # What you'll build + learning goals
│   ├── starter_code.py     # Working foundation to extend
│   ├── solution.py         # Reference implementation
│   └── deploy.md           # How to deploy it live
└── project_2_intermediate/
    └── ...
```

### Self-Check Exercises (Ungraded)
```python
# Exercise: Modify the chunk_size and observe the difference
# Try: 500, 1000, 2000 tokens

chunk_size = 1000  # TODO: Change this value

# Run this cell to see the impact
results = evaluate_chunking(chunk_size)
print(f"Retrieval accuracy: {results['accuracy']:.2%}")

# Did you notice? Smaller chunks = more precise but slower
# Larger chunks = faster but might miss details
```

## Workflow Guidelines

1. **Planning** - Write plans to `.claude_plans/projectplan.md`
2. **Tests** - Store all tests in `tests/` directory
3. **Reference** - Check `.claude_prompts/course_creator.md` for full framework
4. **No orphan files** - Everything in appropriate folder location

## File Naming Conventions

### Quick-Starts
- `00_hello_world.ipynb` - Always start with 00
- `01_first_real_example.ipynb`
- `02_your_own_data.ipynb`

### Templates
- `pipeline_template.py` - Descriptive name + `_template`
- `model_template.py`
- `api_template.py`

### Recipes
- `load_data_from_api.py` - Action-oriented names
- `preprocess_text.py`
- `common_patterns.py`

### Concepts
- `visual_guides/bayesian_updating.md` - Concept name (no numbers)
- `deep_dives/01_foundations.md` - Numbered for sequence

### Modules (when used)
- `module_NN_descriptive_name/`
- Notebooks: `01_topic.ipynb` (15-min max)
- Exercises: `exercises.py` (self-check, ungraded)

### Projects
- `project_1_beginner/`
- `project_2_intermediate/`
- `project_3_advanced/`

## Anti-Patterns to Avoid

**DO NOT create:**
- `quiz.md` - No formal quizzes
- `grading_rubric.md` - No grading rubrics
- `assignment_submission.md` - No assignment submissions
- `final_exam.md` - No exams
- 90-minute notebooks - Break into 15-min pieces
- Theory-first content - Always start with working code
- Synthetic/mock data - Always use real datasets
