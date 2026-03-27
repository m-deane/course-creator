# Module 4: Conditional Trees — When One Answer Is Wrong

## Overview

Some questions look like they have one right answer. They don't. They have a decision tree wearing the costume of a simple question.

"Should I incorporate in Canada or the U.S.?" is not a question with an answer. It is a tree: if your customer base is primarily Canadian, if your funding is from Canadian investors, if your founders are Canadian residents — then Canada. If not, probably the U.S. But the model won't tell you this unless you prompt for it.

This module teaches the core skill of recognizing hidden decision trees inside flat questions — and prompting the model to surface them instead of collapsing them into a single verdict that's wrong for your conditions.

---

## Learning Objectives

By the end of this module, you will be able to:

1. Identify questions that secretly contain conditional structure (decision trees in disguise)
2. Distinguish between "the most common answer" and "the answer given your conditions"
3. Write prompts that produce conditional tree responses instead of single verdicts
4. Use the meta-prompt technique to extract the conditions that would change an answer
5. Design multi-branch prompts where each branch addresses a specific condition set
6. Recognize when a model should say "I need more information" and how to prompt for it

---

## Module Contents

### Guides

| File | Topic | Time |
|------|-------|------|
| `guides/01_conditional_trees_guide.md` | Why single-answer prompts fail; hidden conditional structure; decision trees vs verdicts; prompting for branches | 20 min |
| `guides/01_conditional_trees_slides.md` | Slide deck companion (15–18 slides) | Presentation |
| `guides/02_uncertainty_prompting_guide.md` | Prompting for uncertainty acknowledgment; the meta-prompt; structured if-then responses; getting models to ask the right questions | 15 min |
| `guides/02_uncertainty_prompting_slides.md` | Slide deck companion (12–15 slides) | Presentation |

### Notebooks

| File | Topic | Time |
|------|-------|------|
| `notebooks/01_decision_tree_prompts.ipynb` | Build prompts that produce decision trees instead of flat answers; 3 worked examples using Claude API | 15 min |

### Exercises

| File | Topic |
|------|-------|
| `exercises/01_convert_to_conditional.md` | Convert 5 single-answer prompts into conditional tree prompts |

---

## Prerequisites

- Module 2: Switch Variables (understanding conditions that flip branches)
- Module 3: The Condition Stack Framework (how conditions combine)
- Basic Python (for the notebook)

---

## Core Idea

Every question exists inside an implicit decision tree. The branches are the conditions that would change the correct answer. When you ask for a single answer, the model picks the branch most common in training — which is often not your branch.

The fix is structural. Instead of:

> "What's the best database for my app?"

Prompt for the tree:

> "Before answering, list the conditions that would lead to different database choices. Then answer for each set of conditions."

The first prompt produces *a* answer. The second produces *your* answer — once you know which branch you're on.

**The craft is prompting for structure, not just content.**

---

## What's Next

Module 5 applies conditional tree reasoning to agents and multi-step workflows — where the decision tree is executed dynamically, not just described in text.
