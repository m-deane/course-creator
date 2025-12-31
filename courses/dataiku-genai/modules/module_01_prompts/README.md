# Module 1: Prompt Design with Prompt Studios

## Overview

Design, test, and iterate on prompts using Dataiku's visual Prompt Studios. Learn template variables, few-shot examples, and prompt versioning.

**Time Estimate:** 6-8 hours

## Learning Objectives

By completing this module, you will:
1. Create prompts in Prompt Studios
2. Use template variables for dynamic content
3. Add few-shot examples effectively
4. Version and test prompts systematically

## Contents

### Guides
- `01_prompt_studio_basics.md` - The visual interface
- `02_template_variables.md` - Dynamic prompt content
- `03_testing_iteration.md` - Systematic prompt improvement

### Notebooks
- `01_prompt_creation.ipynb` - Building prompts
- `02_prompt_testing.ipynb` - Evaluation workflows

## Key Concepts

### Prompt Studios Interface

```
┌─────────────────────────────────────────────────────────┐
│  Prompt Studio: Customer Sentiment                      │
├─────────────────────────────────────────────────────────┤
│  System Prompt:                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ You are a sentiment analyst...                   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  User Template:                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Analyze: {{customer_feedback}}                   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Variables:        Test Data:        Output:           │
│  - customer_feedback  [Sample 1]    [Preview]          │
└─────────────────────────────────────────────────────────┘
```

### Template Variables

| Variable Type | Syntax | Use Case |
|---------------|--------|----------|
| Simple | `{{variable}}` | Direct substitution |
| Loop | `{{#items}}...{{/items}}` | Multiple items |
| Conditional | `{{#if condition}}` | Optional content |

### Best Practices

1. Clear system prompts
2. Structured output formats
3. Example-driven design
4. Version control prompts
5. A/B test variations

## Prerequisites

- Module 0 completed
- LLM Mesh configured
