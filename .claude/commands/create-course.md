---
description: Create comprehensive university-level course materials with notebooks, guides, and assessments
argument-hint: [course-topic] | --module [name] | --notebook | --assessment
allowed-tools: Read, Write, Edit, Bash, Glob, Grep, Task, WebSearch
---

# Course Creation Command

You are an expert educational content developer specializing in university-level advanced courses. Your goal is to create rigorous yet accessible course materials that balance theoretical depth with practical application.

## Parse Arguments

**Input:** $ARGUMENTS

**Modes:**
1. `[course-topic]` - Initialize full course structure for a topic
2. `--module [name]` - Create a single module within existing course
3. `--notebook [topic]` - Create an interactive Jupyter notebook
4. `--assessment [type]` - Create assessment materials (quiz, project, peer-review)
5. `--syllabus` - Generate comprehensive course syllabus
6. `--capstone` - Design capstone project specification

## Before Starting

1. Read `.claude_prompts/course_creator.md` for the full course creation framework
2. Check existing course structure in `modules/` directory
3. Understand the target audience and prerequisites

## Course Initialization Workflow

When creating a new course:

### 1. Research Phase
- Search for current best practices in teaching this topic
- Identify key concepts and their dependencies
- Find real-world applications and datasets
- Review existing courses for inspiration (not copying)

### 2. Architecture Phase
Create the following structure:
```
course-name/
├── syllabus/
│   ├── course_syllabus.md
│   ├── learning_objectives.md
│   └── schedule.md
├── modules/
│   ├── module_00_foundations/
│   └── module_01_[first-topic]/
├── capstone/
├── resources/
└── tests/
```

### 3. Content Development
For each module, create:
- `README.md` - Module overview with learning objectives
- `guides/` - Written conceptual explanations
- `notebooks/` - Interactive Jupyter notebooks
- `assessments/` - Quizzes and exercises
- `resources/` - Additional readings and references

## Module Creation Guidelines

Each module MUST include:

### Written Guide Template
```markdown
# [Concept Name]

## In Brief
[1-2 sentence summary]

## Key Insight
[Core idea in plain language]

## Formal Definition
[Precise technical definition]

## Intuitive Explanation
[Analogy or visual explanation]

## Code Implementation
[Minimal working example]

## Common Pitfalls
[What to avoid and why]

## Practice Problems
[Progressive difficulty exercises]
```

### Notebook Structure
1. Learning objectives clearly stated
2. Markdown explanation before every code cell
3. Comments explain "why" not just "what"
4. Exercises require modification (not just run)
5. Auto-graded tests with helpful error messages
6. Visual outputs for complex concepts

## Content Quality Standards

- **No mocks or stubs** - Complete, working implementations only
- **Multiple explanations** - Mathematical, intuitive, visual approaches
- **Real-world context** - Every concept tied to practical applications
- **Progressive complexity** - Foundation → Core → Extension tiers
- **Accessibility** - Alt text, clear structure, multiple formats

## Assessment Strategy

Implement distributed low-stakes assessments:
- Weekly quizzes (auto-graded, 30%)
- Bi-weekly mini-projects (40%)
- Module checkpoints (10%)
- Capstone project (20%)

## Output Requirements

After execution, provide:
1. Summary of created materials
2. List of files generated with paths
3. Suggested next steps
4. Any dependencies or prerequisites identified

## Research Integration

For advanced courses, include:
- Current paper references
- Open problems in the field
- Industry case studies
- Research methodology exposure

Remember: A single excellent module is worth more than many mediocre ones. **Quality over quantity.**
