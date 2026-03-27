# Exercise 01: Identify Missing Conditions

## Overview

The five prompts below all produce generic, prior-dominated answers when sent to a language model. Your task is to diagnose why — which conditions are missing — and rewrite each prompt to supply those conditions.

This exercise is the core skill of the course in its purest form. There is no code to write; only conditions to identify and specify.

**Time estimate:** 20–30 minutes

---

## How to Work Through Each Prompt

For each prompt:

1. **Identify the default world.** What situation does the model assume when no other context is given? Consider: jurisdiction, entity type, audience expertise, time period, constraints, existing state.

2. **Find the gaps.** Which assumptions are most likely wrong for a real professional use case? These are the missing conditions.

3. **Rewrite the prompt.** Add only the conditions that shift the distribution. Do not pad — be specific and concise.

4. **State your prediction.** Before testing your rewrite against a model, write one sentence predicting how the output will differ.

---

## Prompt 1: Investment Advice

**Original prompt:**

> "Should I invest in index funds or individual stocks?"

**What you observe when sent to a model:**
The model produces a balanced generic comparison: index funds offer diversification and lower fees, individual stocks offer potential for higher returns but require research. It probably recommends index funds for most people and mentions risk tolerance. The advice applies to nobody's actual situation in particular.

**Your diagnosis:**

*What world does this prompt assume by default? (Write your answer here)*

---

*Which specific conditions are missing? List at least four:*

1.
2.
3.
4.

---

**Your rewritten prompt:**

> (Write your condition-specified version here)

---

*Predicted change in model output:*

---

---

## Prompt 2: HR Policy Question

**Original prompt:**

> "What is the legal notice period for terminating an employee?"

**What you observe when sent to a model:**
The model gives a general answer about at-will employment (US default), mentions that notice periods vary by state and contract, and suggests consulting an employment lawyer. It is technically correct for US at-will employment but useless for most real HR situations outside the US or for employees with contracts.

**Your diagnosis:**

*What world does this prompt assume by default?*

---

*Which specific conditions are missing? List at least four:*

1.
2.
3.
4.

---

**Your rewritten prompt:**

> (Write your condition-specified version here)

---

*Predicted change in model output:*

---

---

## Prompt 3: API Architecture

**Original prompt:**

> "What's the best way to design an API for our mobile app?"

**What you observe when sent to a model:**
The model recommends REST with JSON, possibly mentions GraphQL as an alternative, talks about authentication with JWT, versioning strategies, and rate limiting. It is a competent general answer for a new mobile app with no constraints. It will be wrong for most real situations.

**Your diagnosis:**

*What world does this prompt assume by default?*

---

*Which specific conditions are missing? List at least four:*

1.
2.
3.
4.

---

**Your rewritten prompt:**

> (Write your condition-specified version here)

---

*Predicted change in model output:*

---

---

## Prompt 4: Medical Documentation

**Original prompt:**

> "Help me write a patient discharge summary."

**What you observe when sent to a model:**
The model produces a template discharge summary with generic sections: admission date, diagnosis, treatment, medications, follow-up instructions. It is structurally correct and useful as a template. It contains no actual clinical information and assumes a standard adult inpatient admission in a generic healthcare system.

**Your diagnosis:**

*What world does this prompt assume by default?*

---

*Which specific conditions are missing? List at least four:*

1.
2.
3.
4.

---

**Your rewritten prompt:**

> (Write your condition-specified version here)

---

*Predicted change in model output:*

---

---

## Prompt 5: Content Strategy

**Original prompt:**

> "How should we grow our social media following?"

**What you observe when sent to a model:**
The model recommends posting consistently, using relevant hashtags, engaging with your audience, posting at optimal times, and using a content calendar. It mentions Instagram, TikTok, and LinkedIn as options. This advice applies to the average small business or individual creator. It is generic to the point of uselessness for any specific organisation.

**Your diagnosis:**

*What world does this prompt assume by default?*

---

*Which specific conditions are missing? List at least four:*

1.
2.
3.
4.

---

**Your rewritten prompt:**

> (Write your condition-specified version here)

---

*Predicted change in model output:*

---

---

## Verification

After completing all five rewrites, test each against a model (you can use the `ask()` function from Notebook 01) and compare your rewritten prompts against the originals.

For each prompt, check:

- [ ] Does the rewritten response mention your specified jurisdiction, entity type, or domain context?
- [ ] Does it avoid advice that only applies to the "most typical world"?
- [ ] Is the rewritten response something an expert in your specified context would actually give?
- [ ] Did the output change in the direction you predicted?

---

## Reference: What Conditions Matter Most by Domain

Use this as a checklist when writing your rewrites:

| Domain | High-impact conditions to always specify |
|--------|------------------------------------------|
| Finance / investment | Country, account type (retail/institutional), tax wrapper, time horizon, existing positions, risk tolerance |
| Legal / HR | Jurisdiction (country and region), entity type, employment contract type, date of event |
| Software / architecture | Language version, framework version, deployment target, existing infrastructure, team size |
| Medical / clinical | Patient age and weight, comorbidities, current medications, contraindications, country of practice |
| Marketing / growth | Platform, current audience size, content type, B2B vs B2C, regulatory restrictions on advertising |

---

## Key Principle

You are not adding context to be polite or to give the model more to work with. You are supplying the conditions that distinguish your situation from the most typical scenario in the model's training data.

Each condition you add is a likelihood term in a Bayesian update: it shifts $P(\text{output} \mid \text{your context})$ away from the generic prior and toward the answer that is correct for your actual situation.

Specificity is the mechanism. Everything else is decoration.
