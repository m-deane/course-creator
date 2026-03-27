# Exercise 1: Rank Switch Variables

## Overview

For each of the 10 prompts below, identify the switch variables and rank them by their likely information gain — that is, how much each condition would change the correct answer if it were different.

**No API calls required.** This is a paper exercise. The goal is to build the habit of thinking in branches before writing any prompt.

## Framework

Apply this process to each prompt:

**1. List candidate conditions** — what might a practitioner in this domain ask before giving advice?

**2. Classify each by category:**
- `J` = Jurisdiction / Scope
- `T` = Timing / Posture
- `S` = Status / Role
- `C` = Constraint
- `O` = Objective function

**3. Rank by information gain using the 100% / 50% / 10% test:**
- **100%** — if this variable were different, the correct answer would be categorically different (different recommendation, different legal outcome, different architecture)
- **50%** — the answer would shift substantially but stay in the same category
- **10%** — the answer would be refined but fundamentally the same

**4. Select your top 3** — the conditions you would add before submitting this prompt.

---

## The Prompts

### Prompt 1 — Law

> "Can I fire this employee immediately?"

**Domain:** Employment law

**Your analysis:**

| Condition | Category | Impact (100/50/10%) | Why |
|-----------|----------|---------------------|-----|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

**Top 3 conditions you would add (in order):**

1.
2.
3.

**Reflection:** What would the answer look like without these conditions? What would it look like with all three?

---

### Prompt 2 — Medicine

> "What antibiotic should I prescribe for this respiratory infection?"

**Domain:** Clinical medicine

**Your analysis:**

| Condition | Category | Impact (100/50/10%) | Why |
|-----------|----------|---------------------|-----|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

**Top 3 conditions you would add (in order):**

1.
2.
3.

**Reflection:** Identify one condition that looks like a switch variable but is actually a low-gain detail for this specific question.

---

### Prompt 3 — Software Architecture

> "Should I use SQL or NoSQL for this project?"

**Domain:** Software engineering

**Your analysis:**

| Condition | Category | Impact (100/50/10%) | Why |
|-----------|----------|---------------------|-----|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

**Top 3 conditions you would add (in order):**

1.
2.
3.

**Reflection:** What is the objective function here? Is "best database" optimizing for latency, cost, team familiarity, consistency guarantees, or something else? How does the answer change if the objective shifts?

---

### Prompt 4 — Finance

> "Should I hold or sell this position?"

**Domain:** Portfolio management / trading

**Your analysis:**

| Condition | Category | Impact (100/50/10%) | Why |
|-----------|----------|---------------------|-----|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

**Top 3 conditions you would add (in order):**

1.
2.
3.

**Reflection:** The objective function is the most commonly omitted condition in finance. Name three distinct objectives that would call for opposite actions on the same position.

---

### Prompt 5 — Business Strategy

> "Should we expand into international markets?"

**Domain:** Business strategy

**Your analysis:**

| Condition | Category | Impact (100/50/10%) | Why |
|-----------|----------|---------------------|-----|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

**Top 3 conditions you would add (in order):**

1.
2.
3.

**Reflection:** This prompt has a hidden timing variable. What is it, and why does it produce categorically different strategic advice?

---

### Prompt 6 — AI / Prompt Engineering

> "How should I structure my system prompt for this agent?"

**Domain:** LLM prompt engineering

**Your analysis:**

| Condition | Category | Impact (100/50/10%) | Why |
|-----------|----------|---------------------|-----|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

**Top 3 conditions you would add (in order):**

1.
2.
3.

**Reflection:** The deployment target (single-turn chat vs. multi-step agent vs. RAG pipeline) is a switch variable for prompt structure. Why does this variable change the system prompt so fundamentally?

---

### Prompt 7 — Tax / Accounting

> "Can I deduct this expense?"

**Domain:** US tax law

**Your analysis:**

| Condition | Category | Impact (100/50/10%) | Why |
|-----------|----------|---------------------|-----|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

**Top 3 conditions you would add (in order):**

1.
2.
3.

**Reflection:** Identify the single highest-gain switch variable for this question and explain why it's higher gain than any other condition you listed.

---

### Prompt 8 — Security Engineering

> "How do I protect this API endpoint?"

**Domain:** Application security

**Your analysis:**

| Condition | Category | Impact (100/50/10%) | Why |
|-----------|----------|---------------------|-----|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

**Top 3 conditions you would add (in order):**

1.
2.
3.

**Reflection:** The compliance requirement (HIPAA, PCI-DSS, SOC2, none) is a jurisdiction-category switch variable. How does its presence or absence change the security recommendations from "optional best practices" to "mandatory controls"?

---

### Prompt 9 — HR / People Management

> "How should I handle this performance issue?"

**Domain:** People management

**Your analysis:**

| Condition | Category | Impact (100/50/10%) | Why |
|-----------|----------|---------------------|-----|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

**Top 3 conditions you would add (in order):**

1.
2.
3.

**Reflection:** There are two distinct objective functions hidden in this prompt: "improve the employee's performance" and "build a paper trail for eventual termination." These call for opposite approaches. Identify what other conditions interact with the objective function to produce different advice.

---

### Prompt 10 — Data Science / ML

> "What model should I use for this prediction problem?"

**Domain:** Machine learning

**Your analysis:**

| Condition | Category | Impact (100/50/10%) | Why |
|-----------|----------|---------------------|-----|
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

**Top 3 conditions you would add (in order):**

1.
2.
3.

**Reflection:** The interpretability requirement is a constraint-category switch variable that eliminates entire model classes. At what point does the constraint "the model must be interpretable to regulators" eliminate gradient boosting, neural networks, and ensemble methods from consideration?

---

## Cross-Prompt Patterns

After completing all 10, answer these synthesis questions:

**1. Which category appears most often as the highest-gain switch variable across these 10 prompts?**

Your answer:

**2. The objective function appeared as a switch variable in several prompts. Which three prompts had the most dramatic answer change when the objective was different?**

Your answer:

**3. Identify one prompt where a number (with a threshold) was a genuine switch variable, not just a descriptive detail. What threshold was it, and what category change did it cause?**

Your answer:

**4. For any two prompts from different domains, compare the top 3 switch variables. What patterns appear across domains, and what is unique to each domain?**

Your answer:

---

## Reference: Five-Category Checklist

Use this as a checklist when you're stuck:

| Category | The question to ask |
|----------|---------------------|
| **Jurisdiction** | Which rule set governs this? (legal, regulatory, technical standard) |
| **Timing** | Where in the process are we? Is there a deadline or state transition that matters? |
| **Status** | Who is the actor? What classification applies to them? |
| **Constraints** | What eliminates entire solution classes? (budget, allergy, legal prohibition, hard limit) |
| **Objective** | What are we optimizing for? Is the goal explicitly stated or assumed? |

The single most reliable question for any prompt:

> **"What are the top 5 conditions that would make the correct answer completely different?"**
