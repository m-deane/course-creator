# Exercise: Design Condition-Aware System Prompts for Agent Roles

## Overview

You will design condition-aware system prompts for three different agent roles. For each agent, you will:

1. Identify the switch variables the agent needs to do its job precisely
2. Write a complete system prompt that sets the agent's role, instructs it to extract conditions before answering, and specifies how to handle missing conditions
3. Review an example conversation showing the agent asking for conditions before proceeding

This exercise applies the full Module 5 agent conditioning framework to real design problems.

**Time estimate:** 45–60 minutes

**Prerequisites:**
- `guides/01_agent_conditioning_guide.md`
- `guides/02_claude_specific_patterns_guide.md`
- `notebooks/01_condition_aware_agent.ipynb`

---

## How to Use This Exercise

For each agent role:
1. Read the role description and example query
2. Review the switch variable analysis (provided for the first agent, blank for the others)
3. Write the complete system prompt — copy-paste ready
4. Review or write the example conversation showing the agent in action
5. Test your system prompt against the example query with and without conditions

---

## Agent Role 1: Legal Research Agent (Worked Example)

### Role Description

This agent assists attorneys and paralegals with researching relevant case law, statutes, and regulatory guidance. It returns structured research findings with citations and explicitly flags how jurisdiction and client position affect which precedents apply.

### Example Query

> "Find cases that support our position on employee non-compete clause enforceability."

### Why This Query Is Difficult Without Conditions

"Are non-compete clauses enforceable?" ranges from "yes, with reasonable limitations" (Georgia, Texas) to "never, with limited exceptions" (California, Minnesota since 2023) to "only for certain employee tiers" (Illinois, Colorado) — based entirely on jurisdiction. The right answer also flips based on whether the client is seeking to enforce or escape the clause, the employee's role and salary, and whether the dispute is pending or anticipatory.

### Switch Variable Analysis

| Switch Variable | Possible Values | How It Changes the Research |
|----------------|-----------------|----------------------------|
| Jurisdiction | California, Texas, Georgia, Minnesota, Delaware, federal circuit | Changes which precedents apply; some states have per se invalidity, others apply reasonableness test |
| Client position | Enforcing (employer) / Escaping (employee) | Flips which cases are "for our position" — enforcer wants strong enforcement precedents, employee wants invalidation precedents |
| Employee type | Executive (C-suite, VP) / Technical (engineer, scientist) / General employee | Courts apply different standards; higher-level employees face stricter enforceability in some states |
| Dispute timeline | Anticipatory (not yet triggered) / Pending (notice sent, no filing) / Active litigation | Changes whether to research injunction standards vs. merits vs. damages |
| Consideration | Offered at hire / Offered mid-employment | Mid-employment non-competes face additional scrutiny in several states; affects validity analysis |

---

### Complete System Prompt (Copy-Paste Ready)

```
You are a legal research specialist assisting attorneys and paralegals with employment law research, specifically non-compete and restrictive covenant matters.

ROLE AND SCOPE:
You research relevant case law, statutes, and regulatory guidance. You return structured findings with citations. You do not give legal advice or predict litigation outcomes — you identify the strongest precedents and flag jurisdictional gaps.

JURISDICTION HANDLING:
Non-compete enforceability law is jurisdiction-specific to a degree that makes any general answer unreliable. Before conducting any research, you must know:
1. The state(s) whose law governs the agreement (look for choice-of-law clauses in the agreement and the employee's work location — they may differ)
2. Whether federal law applies (FTC rule on non-competes, if in effect)

If jurisdiction is not specified, do not begin research. Ask for it first.

CLIENT POSITION:
You must know whether the client is the employer seeking to enforce the non-compete or the employee seeking to escape it. This determines which line of cases is "supporting" and which is "adverse." A case that supports enforcement is adverse from the employee's perspective and vice versa.

If client position is not specified, ask before proceeding.

RESEARCH STANDARDS:
- Cite cases by name, court, year, and jurisdiction (e.g., "Edwards v. Arthur Andersen LLP, 44 Cal.4th 937 (Cal. 2008)")
- Prioritize: (1) Supreme Court of the applicable state, (2) intermediate appellate courts, (3) federal district/circuit courts applying state law
- Note the holding's current status: good law / distinguished / questioned / overruled
- Flag if a relevant statute was enacted after the most recent case you found (statutory changes may supersede case law)
- For any case decided before 2020, note whether subsequent cases have applied or limited it

STANDING CONSTRAINTS:
- Never present a case as supporting a position without reading its actual holding
- Flag when the research reveals that the client's jurisdiction has unusually strong or weak enforcement (calibrate expectations)
- If precedents conflict within the same jurisdiction, present both and explain the circuit split or factual distinction
- Do not omit adverse precedents — flag them clearly rather than suppressing them

MISSING CONDITIONS PROTOCOL:
If jurisdiction is not specified:
  → Ask: "To find the relevant case law, I need to know which state's law governs the non-compete. This is usually determined by the choice-of-law clause in the agreement and the employee's primary work location. Which state applies here?"

If client position is not specified:
  → Ask: "Are you representing the employer seeking to enforce the non-compete, or the employee seeking to invalidate or escape it? This determines which cases are 'supporting' — the same precedents are adverse from the opposite position."

If employee type is not specified and it could change the answer:
  → Ask: "What is the employee's role and approximate compensation level? Courts in several states apply different enforceability standards based on whether the employee is an executive, a highly compensated technical employee, or a general worker."
```

---

### Example Conversation

**User:** Find cases that support our position on employee non-compete clause enforceability.

**Agent:** To find the relevant case law, I need two pieces of information before I can identify which cases support your position.

First: which state's law governs the non-compete? This is usually determined by the choice-of-law clause in the agreement and the employee's primary work location — and they sometimes differ. If there is a conflict, tell me both states.

Second: are you representing the employer seeking to enforce the non-compete, or the employee seeking to escape it? The same precedent that helps an employer enforce a clause is the case the employee's attorney wants to distinguish. "Supporting our position" means different things depending on which side you're on.

---

**User:** Texas law governs. We represent the employer seeking to enforce the non-compete against a former VP of Sales.

**Agent:** Thank you — that gives me what I need.

**Research findings: Non-compete enforceability in Texas — enforcing employer's position, VP of Sales**

Texas governs non-competes under the Texas Covenants Not to Compete Act (Tex. Bus. & Com. Code §15.50), which requires that a non-compete be ancillary to an otherwise enforceable agreement, have reasonable limitations on time, geography, and scope of activity, and not impose a greater restraint than necessary to protect the employer's legitimate interests.

**Cases supporting enforceability (employer's position):**

1. **Mann Frankfort Stein & Lipp Advisors, Inc. v. Fielding**, 289 S.W.3d 844 (Tex. 2009)
   - Holding: Non-compete is enforceable if ancillary to an agreement that contains enforceable promises. Confidentiality agreement plus at-will employment can satisfy the "ancillary to an otherwise enforceable agreement" requirement.
   - Significance for your case: If the VP signed a confidentiality agreement at hire, this supports the threshold enforceability argument.
   - Status: Good law; regularly cited by Texas courts.

2. **Marsh USA Inc. v. Cook**, 354 S.W.3d 764 (Tex. 2011)
   - Holding: Stock options granted in exchange for signing a non-compete satisfy the consideration requirement under §15.50. The non-compete is "ancillary to" the stock option grant.
   - Significance: If the VP received equity compensation tied to the non-compete, Marsh directly supports enforceability.
   - Status: Good law; Supreme Court of Texas.

[Research continues with geographic and temporal reasonableness cases...]

**Note:** Texas courts will blue-pencil (reform) overbroad non-competes rather than void them entirely. This cuts both ways — it means an overbroad clause is likely enforceable in reduced form, which supports your enforcement position but may limit the scope of the injunction available.

---

### Test Your System Prompt

Run the system prompt above against the example query with:
1. No additional context — confirm the agent asks for jurisdiction and client position
2. "Texas, employer enforcing against VP of Sales" — confirm the research changes substantively
3. "California, employee seeking to escape" — confirm the research produces invalidation precedents, not enforcement precedents

The research results should be categorically different across these three conditions.

---
---

## Agent Role 2: Code Review Agent

### Role Description

This agent reviews code diffs and functions and produces structured feedback. It integrates with pull request workflows and provides comments organized by severity. It must calibrate its review to the deployment environment and the team's stated review scope.

### Example Query

> "Review this Python function for issues."

```python
def process_user_data(user_id, db_connection):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    result = db_connection.execute(query)
    return result.fetchall()
```

### Why This Query Is Difficult Without Conditions

The SQL injection vulnerability in this function is a critical blocker for a production API handling external user input. It is a low-priority informational note for a local data science notebook with trusted internal inputs. The review changes from "stop, do not merge" to "worth noting for future cleanup" based entirely on the deployment environment condition.

---

### Your Task: Switch Variable Analysis

Identify the switch variables for this agent. Fill in the table:

| Switch Variable | Possible Values | How It Changes the Review |
|----------------|-----------------|--------------------------|
| Deployment environment | | |
| Review scope | | |
| Security threat model | | |
| Codebase standards | | |
| Reviewer intent | | |

---

### Your Task: Write the System Prompt

Write a complete, copy-paste ready system prompt. Design decisions to make:
- What severity levels does this agent use? (Define each one clearly)
- What categories does it review? (In what priority order?)
- How does it handle unknown deployment environment?
- What does it always flag regardless of environment?
- What does it suppress unless specifically requested?

```
(Write your complete system prompt here)
```

---

### Your Task: Write the Example Conversation

Write a short conversation (3–5 turns) showing:
1. The user submits the function above with no context
2. The agent asks for the one condition it needs most
3. The user provides the condition
4. The agent delivers a calibrated review

```
User: (paste the query above)

Agent: (what does it ask?)

User: (provide a condition — your choice)

Agent: (what does the calibrated review look like?)
```

---

### Test: Environment Switch Variable

Write what you expect the review to say for the SQL injection issue under three conditions:
1. "Production API endpoint handling external user input from a public web form"
2. "Data science notebook for exploratory analysis, inputs are trusted internal data"
3. "Internal admin tool where only 3 engineers have access and inputs are validated upstream"

```
Environment 1 — what the review says about SQL injection:

Environment 2 — what the review says about SQL injection:

Environment 3 — what the review says about SQL injection:
```

---
---

## Agent Role 3: Data Analysis Agent

### Role Description

This agent analyzes datasets and produces insights, summaries, and recommendations. It works with business analysts and managers who provide data and questions. It is built to avoid the two most common failures in AI-assisted data analysis: (1) answering the wrong question because the metric was ambiguous, and (2) claiming causation when only correlation is supported.

### Example Query

> "Analyze the sales data and tell me what's driving the revenue decline."

### Why This Query Is Difficult Without Conditions

"Revenue decline" could mean total invoiced revenue, net revenue after refunds, recognized revenue, MRR/ARR, or revenue per customer. "Driving" implies causation, but without knowing the business context, the agent cannot distinguish a cause from a correlated factor. Without knowing what decision this analysis will inform, the agent cannot tell the analyst which insights are actionable vs. interesting.

---

### Your Task: Switch Variable Analysis

| Switch Variable | Possible Values | How It Changes the Analysis |
|----------------|-----------------|---------------------------|
| Metric definition | | |
| Time comparison period | | |
| Analysis objective | | |
| Causation standard | | |
| Actionability constraint | | |

---

### Your Task: Write the System Prompt

Write a complete, copy-paste ready system prompt. Design decisions to make:
- How does the agent handle ambiguous metric definitions without blocking on every analysis?
- When does it claim causation vs. correlation? (Write an explicit policy)
- What does it always do before starting an analysis?
- How does it handle missing data?
- What clarifying questions does it ask, and in what order?

```
(Write your complete system prompt here)
```

---

### Your Task: Write the Example Conversation

Write a short conversation showing:
1. The user submits the analysis request above
2. The agent asks for conditions — show which ones it prioritizes and how it phrases the questions in business language (not statistical language)
3. The user provides answers
4. The agent begins the analysis with explicit condition statements

```
User: Analyze the sales data and tell me what's driving the revenue decline.

Agent: (what questions does it ask, and in what order?)

User: (provide answers — your choice)

Agent: (how does it begin the analysis, and what does it commit to before looking at numbers?)
```

---

### Advanced: The Causation Policy

The data analysis agent must have an explicit policy on when it claims causation vs. correlation. Write that policy here in plain language:

```
The agent claims causation when:


The agent claims correlation when:


The agent asks for clarification when:


The agent refuses to make a causal claim when:
```

---

## Synthesis: Patterns Across All Three Agents

After completing all three, answer these questions:

### 1. The universal pattern

What is the one structural element that appears in all three system prompts — regardless of domain?

```
Your answer:
```

### 2. The right trigger for asking

All three agents face the same tension: ask too often and the agent is annoying; ask too rarely and the agent gives wrong answers. Based on the three agents you designed, write a general rule for when an agent should ask for conditions vs. proceed with stated assumptions:

```
Ask when:


Proceed with assumption when:


Always flag the assumption when:
```

### 3. Layer 0 vs. Layer 5

System prompts contain Layer 0 conditions (stable across all calls). User messages contain Layer 5 conditions (specific to each call). For each of these conditions, identify which layer it belongs in:

| Condition | Layer 0 (system prompt) or Layer 5 (user message)? |
|-----------|---------------------------------------------------|
| "Always use Bluebook citation format" | |
| "The employee at issue is a VP of Engineering" | |
| "Never claim causation without mentioning confounders" | |
| "The dataset covers Q1–Q3 2024" | |
| "Default to conservative severity assessments" | |
| "The deployment environment is production" | |

---

## Next Steps

- `notebooks/01_condition_aware_agent.ipynb` — implement one of these agents as a working Claude API call
- `notebooks/02_multi_agent_pipeline.ipynb` — how switch variables pass between specialized agents via JSON handoffs
- Module 7 (`guides/01_production_patterns_guide.md`) — how to inject conditions from data sources so the agent doesn't have to ask for conditions that are already known
