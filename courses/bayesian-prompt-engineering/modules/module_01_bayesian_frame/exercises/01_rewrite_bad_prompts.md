# Exercise 01: Rewrite Bad Prompts

## Overview

Five prompts below each produce a prior-dominated answer — the model's statistical average across all users who have ever asked something similar. Your task is to diagnose why and rewrite each one with the conditions needed to collapse the posterior onto a specific, useful answer.

The skill is not padding. It is identifying the three or four conditions that, if different, would produce a categorically different answer — and stating only those.

**Time estimate:** 30–40 minutes

**Prerequisites:** Read `guides/01_prompts_as_evidence_guide.md` and `guides/02_evidence_vs_information_guide.md` before starting.

---

## How to Work Through Each Prompt

For each prompt:

1. **Identify the default world.** What situation does the model assume when no other context is given? Consider jurisdiction, entity type, timing, objective, and constraints.

2. **Find the gaps.** Which missing conditions are most likely to be wrong for a real professional use case? These are the ones that shift the posterior.

3. **Rewrite the prompt.** Add only the conditions that change which answer is correct. Do not pad — specific and concise.

4. **State the expected improvement.** What specifically changes in the model's output when your conditions are present?

---

## Worked Example 1: Retirement Account

### The Bad Prompt

> "Should I contribute to a traditional IRA or a Roth IRA?"

### What the Model Assumes by Default

The model assumes a US-based individual in their working years, with no unusual tax situation, making a decision for the first time. It produces a balanced comparison: traditional IRA gives a current-year tax deduction; Roth IRA grows tax-free. It probably recommends Roth for younger people, traditional for those in high brackets now. The answer covers the average earner at the average life stage — nobody's actual situation.

### What Conditions Are Missing

1. **Current marginal tax rate vs. expected retirement rate.** This is the highest-leverage switch variable. If your current rate is 32% and your expected retirement rate is 22%, traditional wins cleanly. If the rates are expected to flip, Roth wins. Without this, the model cannot distinguish.

2. **Whether you are eligible to contribute at all.** Roth eligibility phases out at higher incomes. Traditional deductibility depends on whether you have a workplace plan. The model will not check this without being told which situation applies.

3. **Current age and time horizon.** The compounding benefit of Roth grows with time. The deduction benefit of traditional matters most when marginal rates are high today. These interact with age in ways the model cannot determine without both numbers.

4. **Existing account balances and tax diversification.** Someone with no Roth assets and $800k in traditional accounts has a different answer than someone starting from zero.

### Rewritten Prompt

> "I am deciding between a traditional IRA and Roth IRA contribution for 2024.
>
> My conditions:
> - Current marginal federal tax rate: 24% (married filing jointly, $165k household income)
> - Expected retirement marginal rate: 15–22% (estimate based on projected withdrawals)
> - Age: 38, target retirement at 65
> - No existing Roth IRA balance; $120k in traditional 401(k) from previous employer
> - I have a workplace 401(k), so the traditional IRA deduction may be limited
>
> Which account type should I contribute to this year, and is the traditional IRA deduction available to me given my income and workplace plan?"

### Expected Improvement

The rewritten prompt forces the model to reason about: (1) the rate arbitrage (24% now vs. 15–22% later — modest traditional advantage), (2) whether the traditional IRA deduction is actually available at this income with a workplace plan (it phases out — the model can now give the exact threshold), and (3) the specific portfolio context. The output changes from a generic "it depends on your tax rate" explanation to a specific answer for this income level and situation.

---

## Worked Example 2: Code Generation — API Endpoint

### The Bad Prompt

> "Write a Python function to call an external API and return the data."

### What the Model Assumes by Default

The model assumes a simple synchronous call using `requests`, no authentication, no error handling beyond a basic `try/except`, JSON response parsed with `.json()`. It will produce a 15-line function that works for a demo but is wrong for almost every production situation.

### What Conditions Are Missing

1. **Authentication mechanism.** Bearer token, API key header, OAuth 2.0, HMAC signature — these produce categorically different code. The model defaults to "no auth" unless told otherwise.

2. **Sync vs. async runtime.** If this function is called inside a FastAPI route or an asyncio event loop, synchronous `requests` will block the event loop and cause subtle performance bugs. The model needs to know the runtime context.

3. **Error handling requirements.** "Handle errors" means different things: retry on 429, raise exceptions on 4xx, return a typed result object, log and swallow, or write to a dead letter queue. Without this condition, the model produces minimal generic error handling.

4. **Response shape and typing.** Whether to return raw dict, a Pydantic model, or a dataclass changes the code substantially. The model defaults to raw dict.

### Rewritten Prompt

> "Write a Python function to fetch data from an external REST API.
>
> Runtime: async (this will be called from an async FastAPI route)
> Auth: Bearer token passed as an argument — do not hardcode
> Error handling: raise `httpx.HTTPStatusError` on 4xx/5xx; retry up to 3 times on 429 with exponential backoff
> Response: parse JSON and return as a plain dict; caller handles typing
> Library constraint: use `httpx` (not `requests`) — it is already in the project dependencies
>
> Function signature: `async def fetch_api_data(url: str, token: str, params: dict = None) -> dict`"

### Expected Improvement

The rewritten prompt produces a function that: uses `httpx.AsyncClient` (not `requests`), implements proper exponential backoff for rate limits, passes the token as a header (not hardcoded), and compiles with the correct async signature. The generic version would fail in production within hours of deployment. The conditioned version works for the described runtime.

---

## Your Turn: 5 Prompts to Diagnose and Rewrite

---

## Prompt 1: Tax Planning

### The Bad Prompt

> "How can I reduce my tax bill this year?"

### What you observe when sent to a model

The model produces a list of common deductions: maximize 401(k) contributions, consider an HSA, take the home office deduction if self-employed, harvest tax losses, donate to charity. The advice is not wrong, but it applies to the statistical average of all people who want to reduce taxes. Several suggestions will be inapplicable or already exhausted depending on the actual situation.

### Your Diagnosis

*What world does this prompt assume by default? (Write your answer here)*

---

*What conditions are missing? List at least four — focus on conditions that would change which strategies are actually available:*

1.
2.
3.
4.

---

**Your rewritten prompt:**

> (Write your condition-specified version here)

---

*Expected improvement — what specifically changes in the output when your conditions are present?*

---

---

## Prompt 2: Medical Symptom Analysis

### The Bad Prompt

> "I've been having headaches every day for two weeks. What could be causing them?"

### What you observe when sent to a model

The model produces a list of common causes: tension headaches, dehydration, poor sleep, caffeine withdrawal, eye strain from screens. It probably mentions that persistent daily headaches should be evaluated by a doctor. This is medically accurate for the typical presentation of daily headache in an otherwise healthy adult — and completely uninformative for any specific patient situation.

### Your Diagnosis

*What world does this prompt assume by default?*

---

*What conditions are missing? List at least four — focus on conditions that would change which diagnoses are most likely and which red flags apply:*

1.
2.
3.
4.

---

**Your rewritten prompt:**

> (Write your condition-specified version here)

---

*Expected improvement — how does the differential diagnosis shift when your conditions are present?*

---

---

## Prompt 3: Code Generation — Authentication System

### The Bad Prompt

> "How do I add authentication to my web app?"

### What you observe when sent to a model

The model recommends username/password with hashed passwords (bcrypt), JWT for session management, possibly OAuth for social login. It will mention libraries like Passport.js or Auth0. This is the median implementation for a new web app with no constraints. It will be wrong for a regulated industry, wrong if there is an existing identity provider, wrong if the team lacks the security expertise to implement JWT correctly, and wrong for an internal tool where SSO is the right answer.

### Your Diagnosis

*What world does this prompt assume by default?*

---

*What conditions are missing? List at least four — focus on conditions that flip which authentication approach is correct:*

1.
2.
3.
4.

---

**Your rewritten prompt:**

> (Write your condition-specified version here)

---

*Expected improvement — how does the implementation recommendation change when your conditions are present?*

---

---

## Prompt 4: Business Strategy — Market Entry

### The Bad Prompt

> "Should we expand our product into the European market?"

### What you observe when sent to a model

The model gives a framework answer: assess market size, evaluate regulatory requirements (GDPR), consider localization costs, analyze competition, validate product-market fit in the target market. It may mention the EU's regulatory environment as a complexity. This is the standard strategic checklist, which applies to every company equally and helps none of them specifically.

### Your Diagnosis

*What world does this prompt assume by default?*

---

*What conditions are missing? List at least four — focus on conditions that would produce a specific "yes/no/not yet" recommendation instead of a framework:*

1.
2.
3.
4.

---

**Your rewritten prompt:**

> (Write your condition-specified version here)

---

*Expected improvement — what does "yes, proceed" vs. "not yet" require in terms of specific conditions that your rewrite supplies?*

---

---

## Prompt 5: Data Analysis

### The Bad Prompt

> "Analyze this sales data and tell me why revenue dropped last month."

### What you observe when sent to a model

The model asks clarifying questions or proceeds with a generic framework: segment by product, by region, by customer type; compare to prior period and prior year; look for outliers. If data is provided, the model will run generic aggregations and point to the largest deltas. It will not know which metric definition to use, which comparison period is relevant to the business, or what a "real" drop looks like vs. normal variance — unless those conditions are specified.

### Your Diagnosis

*What world does this prompt assume by default?*

---

*What conditions are missing? List at least four — focus on conditions that determine what counts as an explanation vs. noise:*

1.
2.
3.
4.

---

**Your rewritten prompt:**

> (Write your condition-specified version here)

---

*Expected improvement — how does the analysis change when the model knows which metric, which comparison period, and what threshold matters?*

---

---

## Verification

After completing all five rewrites, test each against a model and compare the original and rewritten outputs.

For each prompt, check:

- [ ] Does the rewritten response reason about your specific situation rather than the average case?
- [ ] Does it avoid recommendations that only apply to conditions you did not specify?
- [ ] Did the output change in the direction you predicted?
- [ ] Is there anything in the rewritten output that would not have appeared in the original?

---

## Connection to the Bayesian Frame

When you rewrote each prompt, you did not just "add context." You supplied evidence:

$$P(\text{answer} \mid \text{your conditions}) \gg P(\text{answer} \mid \text{generic prompt})$$

Each condition you added is a likelihood term that makes some answers far more probable than others. The model's prior (the average case) is only the starting point. Your conditions are what move the posterior toward the answer that is correct for your actual situation.

**The prompts that failed were not missing context. They were missing discriminating evidence — conditions that change which answer is correct.**

---

## Domain Reference: High-Impact Conditions by Domain

| Domain | Switch variables with the highest leverage |
|--------|-------------------------------------------|
| Tax planning | Jurisdiction, entity type (W-2 vs. 1099 vs. business), income level relative to phase-out thresholds, tax year and filing stage, primary objective (minimize current tax vs. reduce audit risk) |
| Medical | Patient demographics (age, sex, weight), presenting symptoms with duration and severity, current medications and allergies, relevant comorbidities, clinical setting (outpatient vs. ED vs. specialist) |
| Code generation | Language version, framework and its version, runtime (sync/async), auth and security model, error handling requirements, existing codebase constraints |
| Business strategy | Company stage and revenue, current market position, funding/runway, objective function (growth vs. profitability vs. exit), competitive landscape |
| Data analysis | Metric definition (what counts), comparison period and baseline, what constitutes a meaningful vs. noise-level change, decision that the analysis will inform |

---

## Next Steps

- `notebooks/01_posterior_shift_simulator.ipynb` — run your rewritten prompts against the original and measure the output distribution shift
- Module 2 (`guides/01_switch_variables_guide.md`) — systematic framework for identifying which conditions have the most leverage in any domain
- Module 3 (`guides/01_condition_stack_guide.md`) — the 6-layer protocol that turns condition identification into a repeatable method
