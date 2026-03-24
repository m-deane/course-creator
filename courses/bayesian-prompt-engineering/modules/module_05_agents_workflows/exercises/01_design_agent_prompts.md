# Exercise: Design Condition-Aware System Prompts for Agent Roles

## Overview

You will design condition-aware system prompts for three different agent roles. For each agent, you will:

1. Identify the switch variables the agent needs to do its job precisely
2. Write a system prompt that establishes Layer 0 through Layer 4 conditions
3. Specify how the agent should handle missing conditions — and how it should ask for them

This exercise applies everything from Module 5 to real agent design problems.

---

## How to Use This Exercise

For each agent role:
1. Read the role description and example query
2. Complete the switch variable analysis (the template is provided)
3. Write the system prompt
4. Write the missing conditions handler
5. Test your system prompt by running it against the example query with and without the switch variables you identified

---

## Agent Role 1: Legal Research Agent

### Role Description

This agent assists legal professionals (attorneys and paralegals) with researching relevant case law, statutes, and regulatory guidance. It returns structured research findings with citations.

### Example Query

"Find cases that support our position on employee non-compete clause enforceability."

### Why This Query Is Difficult Without Conditions

The answer to "are non-compete clauses enforceable?" ranges from "always, with reasonable limitations" (Georgia) to "almost never" (California) to "never for most employees" (Minnesota, starting 2023) depending on jurisdiction. The right answer also depends on whether the client is seeking to enforce or escape the clause, the employee's role, and the timeline of the dispute.

---

### Your Task: Switch Variable Analysis

Complete this analysis before writing the system prompt.

**Question: What are the switch variables for this agent's work?**

Think about: which conditions, if unknown, would cause this agent to research the wrong jurisdiction, the wrong legal standard, or the wrong precedent?

| Switch Variable | Possible Values | How It Changes the Research |
|----------------|-----------------|----------------------------|
| Jurisdiction | [fill in] | [fill in — think: 3-4 different state standards] |
| Client position | [fill in] | [fill in — enforcing vs. escaping changes which cases to find] |
| Employee type | [fill in] | [fill in — executives vs. general employees have different standards] |
| Dispute timeline | [fill in] | [fill in — pending vs. active vs. anticipatory] |

Add any additional switch variables you identify.

---

### Your Task: Write the System Prompt

Write a system prompt that:
- Establishes the agent's role (Layer 0)
- Specifies standing constraints on research quality and citation standards (Layer 4)
- Leaves Layer 5 (facts) and Layer 6 (output format) for per-query injection

**Starter template (fill in the [...] sections):**

```
You are a legal research specialist assisting [who?] with [what purpose?].

JURISDICTION: [How should the agent handle unknown jurisdiction?]

RESEARCH STANDARDS:
- [Standard 1 — e.g., citation requirements]
- [Standard 2 — e.g., recency requirements]
- [Standard 3 — e.g., what to do when precedents conflict]

STANDING CONSTRAINTS:
- [Constraint 1 — what you will never do]
- [Constraint 2 — when you will flag a result as unreliable]
- [Constraint 3 — how to handle jurisdictional gaps]

MISSING CONDITIONS PROTOCOL:
If jurisdiction is not specified: [what should the agent do?]
If client position is not specified: [what should the agent do?]
```

---

### Your Task: Missing Conditions Handler

Write the questions this agent should ask when switch variables are missing. The questions should:
- Be specific (not "what is your jurisdiction?" but something that helps a non-lawyer specify correctly)
- Be ordered by importance (most important switch variable first)
- Offer examples or options where helpful

**Questions to ask when jurisdiction is missing:**

```
[Write the question here]
```

**Questions to ask when client position is missing:**

```
[Write the question here]
```

---

### Test Your Design

Run this prompt against the example query with:
1. No additional context — observe what the agent does
2. Jurisdiction: California — observe how the answer changes
3. Jurisdiction: Georgia + client is enforcing — observe the change again

The answer should change substantially between these three conditions. If it does not, your system prompt may not be setting the right Layer 0 conditions.

---
---

## Agent Role 2: Code Review Agent

### Role Description

This agent reviews code diffs and produces structured feedback. It integrates with pull request workflows and provides comments on quality, security, performance, and correctness.

### Example Query

"Review this Python function for issues."

```python
def process_user_data(user_id, db_connection):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db_connection.execute(query)
    return result.fetchall()
```

### Why This Query Is Difficult Without Conditions

What constitutes a "review issue" depends heavily on context:
- Is this a prototype or production code?
- What is the security threat model?
- What are the performance requirements?
- Does the codebase have existing patterns this code should match?
- Is the reviewer looking for blockers or all issues?

The SQL injection vulnerability is critical in production. In a local development prototype with trusted inputs, it may be a lower-priority note.

---

### Your Task: Switch Variable Analysis

| Switch Variable | Possible Values | How It Changes the Review |
|----------------|-----------------|--------------------------|
| Code environment | [fill in] | [fill in] |
| Review scope | [fill in] | [fill in] |
| Security threat model | [fill in] | [fill in] |
| Codebase standards | [fill in] | [fill in] |

---

### Your Task: Write the System Prompt

The code review agent needs a system prompt that produces consistent, actionable reviews. Key design decisions:

- What severity levels should it use? (blocker / warning / suggestion / nitpick — or something else?)
- What categories should it review? (security, performance, readability, correctness, maintainability)
- What should it do when it cannot determine environment context?
- How should it handle the tension between "thorough" and "not overwhelming the developer"?

```
You are a code review specialist operating under these standing conditions:

REVIEW SCOPE: [Default scope when not specified]

SEVERITY LEVELS:
- [Level 1]: [Definition and when to use]
- [Level 2]: [Definition and when to use]
- [Level 3]: [Definition and when to use]

REVIEW CATEGORIES (apply in this order):
1. [Category 1 — highest priority]
2. [Category 2]
3. [Category 3]

STANDING CONSTRAINTS:
- [Constraint on number of issues to surface]
- [Constraint on what NOT to flag unless specifically requested]
- [Constraint on how to handle ambiguous cases]

MISSING CONDITIONS:
If the code environment is unspecified: [what to assume and what to ask]
```

---

### Your Task: Missing Conditions Handler

For a code review agent, some conditions can be partially inferred from the code itself (language, patterns), while others genuinely need to be asked.

Which switch variables can be inferred from the code? Which must be asked?

**Inferred from code:**
- [List conditions you can detect from the code diff/function itself]

**Must be asked:**
- [List conditions that genuinely need to be asked, with the questions to ask]

---

### Test: Environment Switch Variable

Write the same function review prompt three times with different environment conditions:
1. "This is a production API endpoint handling external user input"
2. "This is a data science notebook for exploratory analysis, not deployed"
3. "This is an internal tool used only by admins with read-only database access"

The review of the SQL injection issue should differ substantially across these three conditions. Write what you would expect the review to say for each.

---
---

## Agent Role 3: Data Analysis Agent

### Role Description

This agent analyzes datasets and produces insights, summaries, and recommendations. It works with business analysts who provide data and questions.

### Example Query

"Analyze the sales data and tell me what's driving the revenue decline."

### Why This Query Is Difficult Without Conditions

"Revenue decline" could mean:
- Total revenue vs. revenue per customer vs. revenue per product
- Month-over-month vs. year-over-year vs. compared to plan
- Is the decline happening everywhere or in a specific segment?

"Driving" implies a causal analysis, but:
- Do you want correlation or causation?
- What is the business context that would make a cause actionable?
- Are there known external factors that should be excluded?

Without conditions, the agent will produce an analysis of average generality that is unlikely to match what the analyst actually needs.

---

### Your Task: Switch Variable Analysis

| Switch Variable | Possible Values | How It Changes the Analysis |
|----------------|-----------------|---------------------------|
| Metric definition | [fill in] | [fill in] |
| Time comparison | [fill in] | [fill in] |
| Analysis depth | [fill in] | [fill in] |
| Actionability constraint | [fill in] | [fill in] |

---

### Your Task: Write the System Prompt

The data analysis agent needs special handling for:
- Ambiguous metric definitions (what counts as "revenue"?)
- The distinction between correlation and causation
- Unknown context (why does the business care about this?)
- How to handle incomplete data

```
You are a data analysis specialist operating under these standing conditions:

ANALYSIS OBJECTIVE: [Default objective when not specified]

METRIC STANDARDS:
- [How to handle ambiguous metric definitions]
- [Default statistical methods]
- [How to report confidence / uncertainty]

CAUSATION POLICY:
- [When to claim causation vs. correlation]
- [How to flag confounders]

STANDING CONSTRAINTS:
- [What this agent will NOT conclude without sufficient evidence]
- [How to handle missing data]
- [When to ask for clarification vs. proceeding with assumptions]

MISSING CONDITIONS PROTOCOL:
If the metric definition is ambiguous: [what to do]
If the time period for comparison is unspecified: [what to do]
```

---

### Your Task: Missing Conditions Handler

For a data analysis agent, clarification questions need to be specific and non-technical (since analysts may not know statistical terminology). Write clarification questions that:
- Are business-language, not statistical language
- Offer concrete examples to help the user specify correctly
- Are prioritized (most important first)

**Questions to ask before starting analysis:**

1. **About the metric:** [Write the question — offer 2-3 examples of how revenue could be defined]

2. **About the comparison:** [Write the question — offer examples of comparison periods]

3. **About the desired outcome:** [Write the question — what decision will this analysis inform?]

---

### Advanced: Layer 0 vs. Layer 5 Separation

For the data analysis agent, there is a design tension: the agent needs some context that changes per analysis (the data, the question), but other context that should be stable (how to handle statistical uncertainty, when to claim causation).

**Map each condition to its layer:**

| Condition | Layer | Reason |
|-----------|-------|--------|
| "Always report confidence intervals for trend claims" | [Layer 0 or 5?] | [Why?] |
| "This dataset covers January–June 2024" | [Layer 0 or 5?] | [Why?] |
| "The company's objective is to maximize customer lifetime value" | [Layer 0 or 5?] | [Why?] |
| "Revenue = invoiced amount, not cash received" | [Layer 0 or 5?] | [Why?] |
| "Do not attribute changes to seasonality without explicit seasonal adjustment" | [Layer 0 or 5?] | [Why?] |

---

## Reflection Questions

After completing all three agents, answer these questions. They connect back to the core concepts from Module 5.

**1. Condition decay risk**

Which of the three agents faces the highest condition decay risk in a multi-step pipeline? Why?

*Hint: Think about which agent's conditions would most change the analysis if they were dropped midway through a pipeline.*

**2. Minimal viable condition set**

For each agent, which single switch variable, if unspecified, would cause the most harm? This is the variable that must survive every handoff — your minimum viable condition.

| Agent | Most critical switch variable | Why |
|-------|------------------------------|-----|
| Legal Research | | |
| Code Review | | |
| Data Analysis | | |

**3. System prompt as Layer 0**

Look at the system prompts you wrote. Which conditions did you put in the system prompt? Why those and not others?

If you put task-specific conditions in the system prompt (conditions that change per query), move them to the user message template instead. System prompts should contain only stable conditions.

**4. The missing conditions protocol**

Compare the three missing conditions protocols you wrote. What is the pattern? When is it better to ask vs. proceed with an assumption?

Write a rule: "Ask when ____. Proceed with assumption when ____."

---

## Extension: Build One

Pick one of the three agent roles above. Implement it as a working Claude agent using the Anthropic Python SDK. Your implementation should:

1. Use your system prompt as the `system` parameter
2. Accept a user question plus an optional `known_conditions` dict
3. Use forced `tool_choice` to extract needed conditions that aren't present
4. Ask for missing conditions (or list what it would ask, if not interactive)
5. Generate a structured answer using your condition stack

Reference: `notebooks/01_condition_aware_agent.ipynb` for the implementation pattern.
