# Capstone Project: Build a Complete Bayesian Prompt System

> *This is a portfolio piece. Build something you would actually use.*

## What You Are Building

A complete Bayesian prompt engineering system for a real-world domain of your choice. By the end, you will have a working Python agent, a reusable prompt library, and documented evidence that condition stacking produces measurably better outputs than raw prompting.

This project synthesizes every concept from the course: switch variables (Module 2), condition stacks (Module 3), conditional trees (Module 4), and agent workflows (Module 5). The result is a professional artifact you can show to colleagues, clients, or a hiring committee.

---

## Choose Your Domain

Pick one domain you know or work in. Depth matters more than breadth — a shallow system across five domains is worthless; a sharp system for one domain is genuinely useful.

**High-value domain examples:**

| Domain | Why It Works Well |
|--------|------------------|
| Legal research | High switch variable density (jurisdiction, posture, objective) |
| Medical triage | Branch conditions are life-critical — the stakes make the difference visible |
| Financial analysis | Regulatory jurisdiction, time horizon, and risk appetite flip entire recommendations |
| Software architecture | Stack, scale, team size, and deployment model are classic switch variables |
| Tax planning | Jurisdiction, entity type, and filing posture create entirely different answer spaces |
| Education curriculum | Grade level, learning objective type, and accessibility requirements branch sharply |
| Engineering design | Regulatory standard, material constraints, and failure mode priority drive divergent designs |
| Real estate investment | Market, financing structure, and hold period change every number |
| Supply chain optimization | Demand pattern, lead time distribution, and inventory policy assumptions matter entirely |
| Clinical trial design | Phase, endpoint type, regulatory pathway, and population selection branch at every step |

You are not limited to this list. Any domain where expert answers depend heavily on context is a good fit.

---

## Project Deliverables

### Deliverable 1: Domain Analysis

**What to produce:** A switch variable catalog for your domain — a structured reference of the conditions that most change expert answers.

**Specification:**

Identify the top 10 switch variables for your domain. For each variable, document:

```
Variable name: [clear, short label]
Type: [categorical | ordinal | boolean | continuous]
Leverage: [high | medium | low]
Why it matters: [one sentence — what changes when this variable changes]
Example values: [3–5 concrete values]
Default assumption: [what a model assumes if you don't specify this]
```

The catalog should be a markdown table or structured list that you could hand to a new team member and have them immediately understand why each variable matters.

**Quality bar:** Each switch variable should be independently testable — you should be able to prompt the model with and without the variable and see a measurably different answer.

---

### Deliverable 2: Condition Stack Templates

**What to produce:** 5 condition stack templates for common question types in your domain. Each template covers all 6 layers and is parameterized for reuse.

**The 6 layers (from Module 3):**

| Layer | Name | What It Specifies |
|-------|------|------------------|
| 1 | Jurisdiction + Rule Set | Which legal, regulatory, or domain universe applies |
| 2 | Time + Procedural Posture | When this is happening and what stage of the process |
| 3 | Objective Function | What "good" means — minimize, maximize, certify, speed, safety |
| 4 | Constraints | What is off the table regardless of other conditions |
| 5 | Facts | The specific numbers, timeline, documents, and details |
| 6 | Output Specification | What form the answer should take |

**Template format:**

```
Template name: [short descriptor]
Question type: [the class of questions this handles]
Required switch variables: [list from your catalog]

Layer 1 — Jurisdiction + Rule Set:
  [Parameterized text with {placeholders} for variable injection]

Layer 2 — Time + Procedural Posture:
  [Parameterized text]

Layer 3 — Objective Function:
  [Parameterized text]

Layer 4 — Constraints:
  [Parameterized text]

Layer 5 — Facts:
  [Parameterized text]

Layer 6 — Output Specification:
  [Parameterized text specifying structure, length, format]
```

All 5 templates must be tested — run each one through the Claude API and include a sample output showing the template produced a correctly-conditioned, domain-appropriate answer.

---

### Deliverable 3: Decision Tree Prompts

**What to produce:** For 3 common questions in your domain, create conditional tree prompts that produce branching answers instead of single verdicts.

**What this means in practice (from Module 4):** Instead of prompting for "the answer," you prompt for the structure: "List the conditions that would lead to different answers, then answer for each set of conditions."

For each of your 3 questions, deliver:

1. The raw prompt (what most people would write)
2. The conditional tree prompt (structured to elicit branches)
3. The raw output (model's single-verdict answer)
4. The conditional tree output (branching answer)
5. A brief analysis: which branches are relevant to your domain and why the single verdict was misleading

The three questions should span different question types — not three variations of the same question. At minimum: one policy/recommendation question, one analysis question, and one planning/design question.

---

### Deliverable 4: Working Agent

**What to produce:** A complete, runnable Python agent that applies your Bayesian prompt system to incoming questions.

**Agent behavior specification:**

The agent receives a question in your domain and executes the following pipeline:

1. **Parse the question** — extract any switch variables already present in the question
2. **Identify missing conditions** — compare present conditions against required conditions from your switch variable catalog
3. **Ask for missing conditions** — for each missing high-leverage switch variable, ask the user one clarifying question at a time (not a list dump)
4. **Assemble the condition stack** — once all high-leverage conditions are specified, build the full 6-layer prompt using the appropriate template from Deliverable 2
5. **Generate the conditional tree answer** — use the conditional tree technique from Deliverable 3 to produce a branching answer

**Technical requirements:**

- Use the `anthropic` SDK: `import anthropic` with `client = anthropic.Anthropic()`
- Model: `claude-opus-4-5` (or the current recommended Claude model at time of submission)
- The agent must run from the command line: `python agent.py`
- Interactive mode: the agent waits for user input between clarifying questions
- Batch mode: the agent accepts a JSON file of pre-specified conditions and runs non-interactively

**What the agent is not:** A wrapper that just injects your templates. The agent must reason about what is missing and ask intelligently. The condition identification step uses Claude to parse the question and compare against your catalog — it is not a regex or keyword match.

---

### Deliverable 5: Testing Suite

**What to produce:** Documented evidence that condition stacking produces more consistent and more useful outputs than raw prompting.

**Test protocol:**

Run each of 3 representative prompts through two conditions:

- **Condition A:** Raw prompt (no condition stack)
- **Condition B:** Full condition stack prompt (your template from Deliverable 2)

Each prompt × condition combination runs 5 times (30 total API calls).

**Measure and report:**

1. **Consistency score** — for each set of 5 outputs, how similar are they to each other? Use a simple rubric: did the outputs give the same recommendation/conclusion? (0–5 scale per run set)
2. **Specificity score** — does the output reference specific conditions from your domain (jurisdiction, timing, constraints), or is it generic? (0–5 scale per run set)
3. **Branch quality** — for tree prompts: how many distinct, correct branches did the output produce? (count)
4. **Summary table** — aggregate scores across all 30 runs, with mean and range for each metric by condition

**Report format:** A markdown table summarizing results, plus 2–3 paragraphs of analysis. What pattern did you observe? Where did condition stacking help most? Were there cases where it did not help?

---

### Deliverable 6: Documentation

**What to produce:** A README that explains your system clearly enough that someone unfamiliar with the course can use it.

**README must cover:**

1. **Domain summary** — what domain you chose and why the Bayesian approach matters for it specifically
2. **Switch variable catalog summary** — the 3–5 most important variables and the intuition for why they matter
3. **How to use the agent** — setup, installation, running interactively and in batch mode
4. **Example session** — a complete transcript of an agent interaction from question to conditional tree answer
5. **What you learned** — 3–5 specific insights about prompt engineering in your domain that you did not know before building this system. Not general observations — specific, domain-relevant insights.

---

## Suggested Timeline (2 Weeks)

### Milestone 1 — Domain Setup (Days 1–3)

- Choose your domain
- Research: what are the major decision branches practitioners face?
- Draft the switch variable catalog (all 10 variables)
- Validate by testing: prompt Claude with and without each variable; confirm the output changes
- Deliverable 1 complete

### Milestone 2 — Templates and Trees (Days 4–7)

- Build all 5 condition stack templates
- Run each template; keep only the ones that produce domain-correct output
- Build the 3 conditional tree prompts
- Compare raw vs. tree output for each; keep only the ones where the difference is clear and meaningful
- Deliverables 2 and 3 complete

### Milestone 3 — Agent Implementation (Days 8–11)

- Build the `BayesianAgent` class (see starter code)
- Implement condition parsing using Claude
- Implement the clarifying question loop
- Implement condition stack assembly and conditional tree generation
- Test interactively with real questions from your domain
- Deliverable 4 complete

### Milestone 4 — Testing and Documentation (Days 12–14)

- Run the full testing suite (30 API calls)
- Compile and analyze results
- Write the README
- Review the self-assessment checklist
- Deliverables 5 and 6 complete

---

## Example Domain Walkthrough: Tax Planning

This walkthrough illustrates what Deliverable 1 looks like for a well-chosen domain. Use it as a reference, not a template to copy.

**Domain:** U.S. Federal and State Tax Planning for Small Business Owners

**Why it works:** Tax advice without conditions is not just imprecise — it is actively misleading. The correct answer to "can I deduct this?" changes entirely based on entity type (sole proprietor vs. S-corp vs. C-corp), state of incorporation, whether the taxpayer is on a cash or accrual basis, and the current tax year. These are not edge cases — they are the core variables every tax practitioner evaluates before saying anything.

**Sample switch variables (3 of 10):**

```
Variable name: entity_type
Type: categorical
Leverage: high
Why it matters: S-corp, C-corp, sole proprietorship, and partnership are four
  different tax universes. A strategy that is optimal for an S-corp shareholder
  may be prohibited or tax-inefficient for a C-corp.
Example values: sole_proprietor, s_corp, c_corp, llc_disregarded,
  llc_partnership, general_partnership
Default assumption (without specification): sole proprietor or "typical small business"
  — which is wrong for approximately 60% of real cases

Variable name: tax_year
Type: ordinal
Leverage: high
Why it matters: Tax law changes frequently. A deduction that existed in 2022
  may be phased out or restructured in 2024. Depreciation schedules, bonus
  depreciation percentages, and SALT cap rules all vary by year.
Example values: 2022, 2023, 2024, 2025
Default assumption: current year — but "current year" in training data is ambiguous

Variable name: accounting_method
Type: categorical
Leverage: medium
Why it matters: Cash-basis and accrual-basis taxpayers recognize income and
  expenses at different times, which changes year-end tax planning strategies completely.
Example values: cash_basis, accrual_basis
Default assumption: cash basis — which is wrong for businesses with >$27M revenue
```

**Sample condition stack template (1 of 5):**

```
Template name: deductibility_analysis
Question type: "Is [expense] deductible?"

Layer 1 — Jurisdiction + Rule Set:
  Analyze under {federal_or_state} tax law for tax year {tax_year}.
  Apply {applicable_code_sections} if relevant.

Layer 2 — Time + Procedural Posture:
  The taxpayer is in {planning | filing | audit | appeals} posture.
  The transaction {occurred_date}. The deduction {has_been_taken | is_being_considered}.

Layer 3 — Objective Function:
  The objective is to {minimize_current_year_liability | minimize_audit_risk |
  maximize_long_term_tax_efficiency}. Prioritize {objective} over alternatives
  where they conflict.

Layer 4 — Constraints:
  The taxpayer cannot: {constraint_list}. Amended returns are {available | not_available}.

Layer 5 — Facts:
  Entity type: {entity_type}. Accounting method: {accounting_method}.
  Expense: {expense_description}. Amount: {amount}.
  Business purpose: {stated_business_purpose}. Documentation: {documentation_status}.

Layer 6 — Output Specification:
  Produce a conditional analysis structured as:
  (1) deductibility verdict for each plausible interpretation of the facts,
  (2) the conditions that change the verdict,
  (3) documentation requirements,
  (4) audit risk assessment on a 1–5 scale with rationale.
```

---

## Example Domain Walkthrough: Software Architecture

**Domain:** Backend system architecture for production web applications

**Why it works:** "What database should I use?" has no answer without knowing: expected read/write ratio, consistency requirements, team familiarity, deployment environment, and anticipated query patterns. Every architectural recommendation branches on these conditions. The gap between the model's default answer and the correct answer for a specific team is often one entirely wrong technology choice.

**Sample switch variables (3 of 10):**

```
Variable name: consistency_requirement
Type: categorical
Leverage: high
Why it matters: Strong consistency (ACID) requirements rule out most distributed
  NoSQL options. Eventual consistency allows much higher scalability but changes
  the application logic required.
Example values: strong_consistency, eventual_consistency, causal_consistency
Default assumption: strong consistency — which rules out the most scalable options prematurely

Variable name: team_size
Type: ordinal
Leverage: medium
Why it matters: A 3-person team that adopts Kubernetes and a service mesh will
  spend more time on operations than product. A 50-person team that stays on a
  monolith will hit coordination costs. Scale decisions depend on team size.
Example values: 1-3, 4-10, 11-30, 31+
Default assumption: "typical startup" — no standard size implied

Variable name: traffic_pattern
Type: categorical
Leverage: high
Why it matters: Steady-state high traffic and bursty/spiky traffic require different
  scaling architectures. Bursty workloads benefit from auto-scaling; steady-state
  high traffic benefits from provisioned capacity with predictable latency.
Example values: steady_high, bursty_spiky, low_steady, batch_periodic
Default assumption: steady growth — incorrect for consumer apps with viral potential
```

---

## Starter Code

The file `starter_code.py` in this directory provides complete, working implementations of the core classes. Use it as the foundation for your agent.

**Classes provided:**

- `ConditionStack` — all 6 layers, with validation and prompt assembly
- `SwitchVariableCatalog` — manages your domain's switch variable definitions and tracks which are present
- `BayesianAgent` — the core agent loop using the Anthropic SDK
- `PromptTester` — runs consistency and specificity tests

See the starter code for full implementation details and the `main()` demonstration.

---

## Self-Assessment Checklist

Use this before considering the project complete. This is not a grading rubric — it is a quality filter to ensure you built something real.

### Domain Analysis
- [ ] Each switch variable has been empirically tested — prompting with and without it produces measurably different output
- [ ] The catalog covers both high-leverage and medium-leverage variables (not just the obvious ones)
- [ ] Default assumption is documented for each variable — you know what the model assumes in the absence of specification
- [ ] A colleague unfamiliar with the domain could read the catalog and understand why each variable matters

### Condition Stack Templates
- [ ] All 5 templates have been run through the API and produced correct domain output
- [ ] Each template covers all 6 layers — no layer is empty or trivially filled
- [ ] Templates are genuinely parameterized — placeholders are replaced with real conditions at runtime, not hardcoded
- [ ] The 5 templates cover meaningfully different question types (not 5 variations of one question)

### Decision Tree Prompts
- [ ] Raw prompt vs. conditional tree prompt comparison is present for all 3 questions
- [ ] The conditional tree output contains real branches (not just "it depends")
- [ ] The branches are domain-correct — an expert in your domain would recognize them as the right conditions
- [ ] The analysis section explains why the single verdict was misleading, not just that it was

### Working Agent
- [ ] The agent runs from the command line without modification
- [ ] The condition identification step uses Claude — it is not a keyword match
- [ ] The clarifying question loop works correctly — it asks one question at a time and stops when conditions are sufficient
- [ ] Batch mode works with a JSON input file
- [ ] The agent produces a conditional tree answer, not a single verdict

### Testing Suite
- [ ] All 30 API calls have been run (3 prompts × 2 conditions × 5 runs)
- [ ] Results are in a summary table with numerical scores
- [ ] Analysis identifies specific patterns, not just "condition stacking helped"
- [ ] At least one honest observation about where condition stacking did not help or helped less than expected

### Documentation
- [ ] A colleague could set up and run the agent from the README alone
- [ ] The example session is a real interaction, not fabricated
- [ ] "What you learned" section contains domain-specific insights (not restatements of course concepts)
- [ ] The README is honest about limitations of the system
