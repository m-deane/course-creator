# Prior Dominance: Why LLMs Produce Coherent but Wrong Answers

> **Reading time:** ~10 min | **Module:** 0 — Foundations | **Prerequisites:** Basic probability concepts


## In Brief

When a language model generates an answer that is confident, fluent, and wrong for your situation, the cause is almost always the same: the model has defaulted to the most common context in its training data rather than the specific context you are in. This is the prior dominance problem, and it has a precise solution.


<div class="callout-key">

<strong>Key Concept Summary:</strong> When a language model generates an answer that is confident, fluent, and wrong for your situation, the cause is almost always the same: the model has defaulted to the most common context in its training data rather than the specific context you are in.

</div>

---

## Key Insight

Language models do not respond to what you meant — they respond to the most probable completion of the token sequence you provided, where "probable" is defined by patterns in training data. When your prompt leaves conditions unspecified, the model fills those gaps with the most statistically common scenario from billions of training examples. That scenario is almost never your scenario.

---

## 1. What a Prior Is (and Why LLMs Have One)

In Bayesian inference, a **prior** is the probability distribution over outcomes before you observe any evidence. When you observe evidence, you update the prior to get a posterior.

A language model's training process instils a prior over all possible contexts. For any token sequence $t_1, \ldots, t_k$, the model has learned:

$$P_{\text{training}}(t_{k+1} \mid t_1, \ldots, t_k)$$

This distribution reflects what followed that token sequence in training data — which reflects the statistical properties of internet text, books, and code, not the properties of your specific situation.

**When your prompt fully specifies your context**, the model can condition on that context and produce an appropriate posterior output.

**When your prompt leaves gaps**, the model fills those gaps with the most typical values from its prior — the "most typical world" in training data.

---

## 2. The Most Typical World Problem

Consider the prompt: `"What are the tax implications of filing late?"`
<div class="callout-warning">

<strong>Warning:</strong> Consider the prompt: `"What are the tax implications of filing late?"`

</div>


In the training corpus, this question was probably asked most often by:
- US taxpayers
- Filing a personal (not business) return
- For the current or most recent tax year
- As a one-time oversight, not a multi-year pattern
- Without specific circumstances like fraud, disaster relief, or amended returns

The model's prior is shaped by that distribution. Its answer will be appropriate for the statistical average of those contexts — which is fine if you are in the most typical situation.

But suppose you are:
- A UK taxpayer
- Filing a corporation tax return
- For a year that is now two years old
- Because the company was in administration

Every one of those conditions shifts the correct answer dramatically. None of them were specified in the prompt. The model will produce a fluent, confident answer about the most typical world — which is wrong for your world.

---

## 3. The Accountant Example

This example illustrates prior dominance at its most concrete.
<div class="callout-key">

<strong>Key Point:</strong> This example illustrates prior dominance at its most concrete.

</div>


### Prompt A (underspecified)

> "My client filed their tax return late. What penalties should they expect?"

**What the model's prior fills in:**
- Jurisdiction: US (most common in training data)
- Return type: personal income tax
- Tax year: the most recently completed year
- Late filing: a few weeks or months overdue
- Reason: simple oversight

**Typical model output (paraphrased):**
> "For a late-filed federal tax return, the failure-to-file penalty is 5% of unpaid taxes per month, up to 25%. If you are due a refund, there is no penalty for late filing."

This is accurate for the most typical scenario. It may be completely wrong for your client.

---

### Prompt B (conditions specified)

> "My client is a UK limited company. Their 2024 corporation tax return has not been filed and is now being submitted in 2026, two years after the deadline. The company was not dissolved but was dormant during the intervening period. What penalties should they expect from HMRC?"

**What changed:** UK jurisdiction, corporation tax (not personal), 2-year delay, dormant company — four conditions added.

**Model output (paraphrased):**
> "HMRC penalties for a late corporation tax return escalate with time. For a return more than 12 months late, HMRC can issue a 100% tax-geared penalty in addition to the fixed penalties. For a dormant company, if the taxable profit is nil, the fixed penalties still apply but the tax-geared element would be zero. You should file the return, pay any fixed penalties, and consider writing to HMRC to explain the dormancy period..."

This is a completely different — and correct — answer for that specific situation.

The conditions you added were not context-setting niceties. They were the evidence that shifted the model's probability distribution from "most typical world" to "your actual world."

---

## 4. Prior Dominance Is Not a Bug

It is tempting to frame prior dominance as a flaw that should be fixed. It is not. It is the correct behaviour of a probabilistic model operating with incomplete information.
<div class="callout-insight">

<strong>Insight:</strong> It is tempting to frame prior dominance as a flaw that should be fixed. It is not. It is the correct behaviour of a probabilistic model operating with incomplete information.

</div>


If you ask a doctor "what is the recommended treatment?" without saying what condition, age, comorbidities, or contraindications apply, a good doctor will give you the most typical treatment for the most common presentation. That is the right response to incomplete information.

The model does the same. The problem is not the model's response strategy — it is that users often do not realise they have provided incomplete information.

**Prior dominance becomes a problem when:**
1. The user believes they specified enough conditions (they did not)
2. The most typical world differs significantly from the user's actual situation
3. The model's output sounds specific and confident even when it is generic

This third point is the dangerous one. Unlike a doctor, a language model does not say "I need more information before I can answer this." It gives you the best answer for the most typical scenario, phrased with equal confidence regardless of whether your scenario is typical.

---

## 5. The Condition Specification Technique

The fix for prior dominance is precise: supply the conditions that distinguish your situation from the most typical situation.
<div class="callout-warning">

<strong>Warning:</strong> The fix for prior dominance is precise: supply the conditions that distinguish your situation from the most typical situation.

</div>


### Step 1: Identify the prior assumption

Ask yourself: "What world does this prompt assume by default?"

For each dimension of ambiguity, write down what the model will assume:
- Who is involved? (role, expertise level, demographics)
- What is the context? (domain, platform, time period)
- What are the constraints? (budget, time, regulations)
- What is already known or decided? (prior choices, existing state)

### Step 2: Identify the delta

For each assumption, ask: "Does my actual situation match this assumption?"

When the answer is no, you have identified a missing condition.

### Step 3: Add conditions as explicit evidence

Rewrite the prompt with each missing condition stated explicitly. You are not adding padding — you are providing the evidence that shifts the probability distribution.

---

## 6. Mapping Conditions to Probability Shifts

The relationship between conditions and probability shifts is not linear. Some conditions shift the distribution dramatically; others barely move it. The conditions that matter most are those that distinguish your situation from the typical situation the model assumes.

| Condition Type | Effect on Distribution |
|----------------|----------------------|
| Role/expertise ("you are a senior cardiologist") | High shift — changes baseline knowledge and register |
| Jurisdiction ("in the UK") | High shift for legal/tax/regulatory questions |
| Time period ("in Q4 2026") | Medium shift — relevant when training data is dated |
| Constraints ("the budget is fixed at $50k") | Medium shift — eliminates solutions outside constraint |
| Existing state ("the codebase uses TypeScript 4.9") | High shift for technical questions |
| Tone ("formal, no bullet points") | Low shift on substance, high shift on format |

---

## 7. Prior Dominance in Different Domains

**Medical:** The model's prior assumes an adult patient with no unusual comorbidities in a high-income country. Age, weight, kidney function, concurrent medications, and country of practice are all conditions that can reverse a recommendation.
<div class="callout-insight">

<strong>Insight:</strong> **Medical:** The model's prior assumes an adult patient with no unusual comorbidities in a high-income country. Age, weight, kidney function, concurrent medications, and country of practice are all conditions that can reverse a recommendation.

</div>


**Legal:** The model's prior defaults to US law. Jurisdiction, date of the incident, and specific statute matter enormously and are often unspecified.

**Code:** The model's prior defaults to the most popular library versions, a greenfield project, and a web-based deployment. Framework version, legacy constraints, and deployment environment can all change the correct answer.

**Finance:** The model's prior assumes retail investment, a personal account, and long-term horizons. Institutional context, derivatives, short positions, and regulatory regime shift the correct answer substantially.

**Writing:** The model's prior assumes a general audience at roughly a 10th-grade reading level. Audience expertise, publication context, and document purpose are regularly unspecified.

In each case, the fix is the same: supply the conditions that distinguish your situation from the default.

---

## 8. Common Pitfalls

**Pitfall 1: Mistaking fluency for accuracy**
A prior-dominated answer is often perfectly fluent and well-structured. Fluency is a property of the generation process, not a signal that the answer is correct for your situation. Always check whether the model's assumed context matches yours.
<div class="callout-warning">

<strong>Warning:</strong> **Pitfall 1: Mistaking fluency for accuracy**

</div>


**Pitfall 2: Adding context but missing the critical condition**
You can add five sentences of context and still miss the one condition that most shifts the distribution. Focus on conditions where your situation differs from the typical case, not on conditions where you match it.

**Pitfall 3: Assuming the model will ask for clarification**
Unlike a human expert, the model will not say "I need more information." It will answer with whatever context was provided, filling gaps from its prior without flagging that it has done so.

**Pitfall 4: Over-conditioning**
Adding many conditions that are obvious or that match the default wastes prompt space and can dilute the signal of conditions that actually matter. Be selective: specify what makes your situation unusual.

---

## Connections

- **Builds on:** Guide 01 — Autoregressive Generation (the mechanism that produces prior-dominated outputs)
- **Leads to:** Module 1 — Prompts as Bayesian Evidence (systematic condition specification)
- **Related to:** Bayesian inference (prior, likelihood, posterior), information theory (prompt as evidence)

---

## Practice Problems

1. **Diagnosis:** Take this prompt — "How should I invest my savings?" — and list five conditions that the model will fill from its prior. For each, write what the default assumption is and why it might be wrong.

2. **Rewrite:** Rewrite the prompt above for a specific hypothetical person: a 58-year-old self-employed contractor in Australia with €200k in savings, 7 years to retirement, and a high risk tolerance. Identify which conditions you added and why each one shifts the distribution.

3. **Domain analysis:** Pick a domain you work in. Describe the "most typical world" the model assumes for questions in that domain. What conditions would you always include in prompts for that domain?

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Key Insight" and why it matters in practice.

2. Given a real-world scenario involving prior dominance: why llms produce coherent but wrong answers, what would be your first three steps to apply the techniques from this guide?

</div>

## Further Reading

- Min et al. (2022) "Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?" — examines how context shifts model behaviour, with evidence that even incorrect labels matter less than format and distribution
- Zhao et al. (2021) "Calibrate Before Use: Improving Few-Shot Performance of Language Models" — demonstrates how surface-level prompt features produce systematic biases from prior distributions

---

## Cross-References

<a class="link-card" href="../notebooks/01_token_probability.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
