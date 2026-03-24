# Exercise 1: Convert Single-Answer Prompts to Conditional Trees

## Overview

You will take 5 "single-answer" prompts — the kind that seem to invite one clear response — and transform each into a conditional tree prompt. For each prompt, you will:

1. Identify the hidden uncertainty (what conditions does the prompt not specify?)
2. List the conditions that create branches (the variables that change the correct answer)
3. Rewrite the prompt as a conditional tree prompt
4. Evaluate the output quality difference

**Format:** Work through each prompt in sequence. You can use Claude or any other LLM to test your rewritten prompts.

**Time estimate:** 30–45 minutes

**Prerequisite:** Complete `notebooks/01_decision_tree_prompts.ipynb` before this exercise.

---

## How to Evaluate Your Work

For each prompt, compare the output of your rewritten prompt against the original. A good conditional tree prompt produces output that:

- Names 3–5 distinct branches
- Gives a specific recommendation for each branch (not just "it depends on X")
- Includes conditions that are concrete and checkable (not "it depends on your situation")
- Surfaces at least one condition you hadn't thought of yourself

---

## Prompt 1: The Pricing Decision

**Original prompt:**
> "What pricing model should I use for my SaaS product?"

---

### Step 1: Identify the Hidden Uncertainty

Read the original prompt. What does it not tell the model? List every piece of information that the model would need to give you a specific answer:

```
Information missing from this prompt:
- _______________________________________________
- _______________________________________________
- _______________________________________________
- _______________________________________________
- _______________________________________________
```

### Step 2: List the Conditions That Create Branches

From your list above, identify which pieces of missing information would actually *change* the correct answer (not just refine it). These are your branch conditions:

```
Branch conditions (variables that flip the answer):
1. Condition: _______________________________________________
   If true → answer leans toward: ___________________________
   If false → answer leans toward: __________________________

2. Condition: _______________________________________________
   If true → answer leans toward: ___________________________
   If false → answer leans toward: __________________________

3. Condition: _______________________________________________
   If true → answer leans toward: ___________________________
   If false → answer leans toward: __________________________
```

### Step 3: Rewrite as a Conditional Tree Prompt

Using Technique 1 (Explicit Branch Request) or Technique 2 (Meta-Prompt) from Guide 1:

```
Your rewritten prompt:

_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________
```

### Step 4: Compare Output Quality

Run both the original and rewritten prompts. Fill in the comparison:

| | Original Prompt Output | Rewritten Prompt Output |
|-|------------------------|------------------------|
| Number of distinct recommendations | | |
| Mentions of "it depends" without specifying what | | |
| Conditions I can actually check about my situation | | |
| Could I identify which branch I'm on? | Yes / No | Yes / No |

**What was the most important condition the model surfaced that you hadn't thought of?**

```
_______________________________________________
```

---

## Prompt 2: The Hiring Decision

**Original prompt:**
> "Should I hire a generalist or a specialist as my first marketing hire?"

---

### Step 1: Identify the Hidden Uncertainty

```
Information missing from this prompt:
- _______________________________________________
- _______________________________________________
- _______________________________________________
- _______________________________________________
```

### Step 2: List the Conditions That Create Branches

Think about this domain. Marketing hire decisions depend heavily on company stage, product type, and existing capabilities.

```
Branch conditions:
1. _______________________________________________
   True → ____________  |  False → ________________

2. _______________________________________________
   True → ____________  |  False → ________________

3. _______________________________________________
   True → ____________  |  False → ________________
```

### Step 3: Rewrite as a Conditional Tree Prompt

For this prompt, use Technique 3 (Condition-First Specification). Start by stating 4–5 conditions about your situation, then ask the narrower question that remains.

```
Your rewritten prompt (Condition-First):

My situation:
- [Company stage]: _______________________________________________
- [Current marketing]: _______________________________________________
- [Product type]: _______________________________________________
- [Budget range]: _______________________________________________
- [Timeline to results]: _______________________________________________

Given these conditions: [narrow question]
_______________________________________________
```

### Step 4: Compare Output Quality

```
Original output — first sentence describes: ___________________________

Rewritten output — first sentence describes: _________________________

Did the model give you a specific role title or archetype to hire?
Original: Yes / No     Rewritten: Yes / No

Which prompt's output could you use to write a job description today?
Original / Rewritten / Neither
```

---

## Prompt 3: The Technical Architecture Choice

**Original prompt:**
> "Should I build my data pipeline in Python or use a dedicated ETL tool like Airflow or dbt?"

---

### Step 1: Identify the Hidden Uncertainty

This is a technical domain with strong conditional structure. Data pipeline tooling decisions depend on team composition, data volume, transformation complexity, and operational requirements.

```
Information missing from this prompt:
- _______________________________________________
- _______________________________________________
- _______________________________________________
- _______________________________________________
- _______________________________________________
```

### Step 2: List the Conditions That Create Branches

```
Branch conditions for this technical decision:
1. _______________________________________________
   True → ____________  |  False → ________________

2. _______________________________________________
   True → ____________  |  False → ________________

3. _______________________________________________
   True → ____________  |  False → ________________

4. _______________________________________________
   True → ____________  |  False → ________________
```

### Step 3: Rewrite Using the Meta-Prompt Technique

For this prompt, use the Meta-Prompt (Technique 2): ask the model to surface conditions and assumptions before answering.

```
Your rewritten prompt:

"Before answering this question, identify:
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________

Then answer for each major branch.

Question: _______________________________________________"
```

### Step 4: Compare and Reflect

```
Original output length (words, approximate): _______
Rewritten output length (words, approximate): _______

How many distinct tool/approach recommendations did the original give? _______
How many distinct branches did the rewritten give? _______

Did the rewritten prompt reveal any conditions that would apply to you
that the original ignored?

_______________________________________________
_______________________________________________
```

---

## Prompt 4: The Compliance Question

**Original prompt:**
> "Do I need a privacy policy on my website?"

---

### Step 1: Identify the Hidden Uncertainty

Compliance questions always have hidden conditional structure based on jurisdiction, data types, and user base.

```
Information missing from this prompt:
- _______________________________________________
- _______________________________________________
- _______________________________________________
- _______________________________________________
```

### Step 2: Identify the Danger of a Flat Answer

For compliance questions, a flat "yes" or "no" can be actively harmful. Before rewriting the prompt, identify:

```
If the model says "yes" when the real answer is "no for your situation":
Risk: _______________________________________________

If the model says "no" when the real answer is "yes for your situation":
Risk: _______________________________________________

Why is "it depends on your jurisdiction" still not good enough?
_______________________________________________
```

### Step 3: Rewrite with Explicit Uncertainty Permission

For compliance questions, add explicit permission for the model to withhold a verdict and ask for conditions instead:

```
Your rewritten prompt:

_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________

[Include: "If you don't have enough information to give me a jurisdiction-specific
answer, list exactly what information you need before I act."]
```

### Step 4: Compare Output Quality

```
Original output — would you rely on this to make a compliance decision?
Yes / No — because: _______________________________________________

Rewritten output — does it list the specific questions you'd need to answer?
Yes / No

What's the most important condition the model identified that the original ignored?
_______________________________________________
```

---

## Prompt 5: Your Own Prompt

Choose a question from your own work, domain, or life that you've previously asked an AI and received a flat or generic answer. This must be a question with genuine conditional structure — one where different conditions would lead to different correct answers.

---

### Your Original Prompt

```
Write your original prompt exactly as you asked (or would ask) it:

_______________________________________________
_______________________________________________
```

**Domain:** _______________________________________________

**What the flat answer typically says:**
```
_______________________________________________
_______________________________________________
_______________________________________________
```

**Why that answer is incomplete or wrong for specific situations:**
```
_______________________________________________
_______________________________________________
```

---

### Step 1: Identify the Hidden Uncertainty

```
Information missing from your prompt:
- _______________________________________________
- _______________________________________________
- _______________________________________________
- _______________________________________________
```

### Step 2: List the Conditions That Create Branches

```
Branch conditions for your question:
1. _______________________________________________
   True → ____________  |  False → ________________

2. _______________________________________________
   True → ____________  |  False → ________________

3. _______________________________________________
   True → ____________  |  False → ________________
```

### Step 3: Choose a Technique and Rewrite

Which technique from Guide 1 are you using?
- [ ] Explicit Branch Request
- [ ] Meta-Prompt
- [ ] Condition-First Specification

```
Your rewritten prompt:

_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________
```

### Step 4: Full Comparison

Run both prompts. Fill in the comparison table:

| Criterion | Original | Rewritten |
|-----------|----------|-----------|
| Number of distinct recommendations | | |
| Number of vague hedges ("it depends," "generally") | | |
| Conditions I can check about my situation | | |
| Could I identify my specific branch? | Yes / No | Yes / No |
| Would I act on this answer? | Yes / No | Yes / No |

**Most valuable insight from the conditional tree response:**
```
_______________________________________________
_______________________________________________
```

**Which branch are you on (based on the tree)?**
```
_______________________________________________
```

---

## Synthesis Questions

After completing all five prompts, answer these questions:

### 1. Pattern Recognition

Looking across all five prompts, what are the most common types of conditions that create branches? List the top three:

```
1. _______________________________________________
2. _______________________________________________
3. _______________________________________________
```

### 2. Technique Selection

Which technique (Explicit Branch Request, Meta-Prompt, or Condition-First) was most useful for each domain?

```
Business decisions → _______________________________________________
Technical architecture → _______________________________________________
Compliance/legal → _______________________________________________
Hiring decisions → _______________________________________________
```

### 3. Failure Cases

Did any of your rewritten prompts fail to produce a better response than the original? If yes, what went wrong?

```
Prompt that underperformed: _______________________________________________
Why the rewrite didn't help: _______________________________________________
What you would change: _______________________________________________
```

### 4. Prompt Template

Based on what you learned, write a reusable conditional tree prompt template you could apply to any new question in your domain:

```
Your domain: _______________________________________________

Your reusable template:

"_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________
_______________________________________________"
```

---

## Connection to the Bayesian Frame

This exercise is a direct application of Module 1's core insight:

$$P(\text{answer} \mid \text{vague question}) \approx P(\text{training prior})$$

$$P(\text{answer} \mid \text{your conditions specified}) \approx P(\text{answer for your branch})$$

When you converted each prompt to a conditional tree:
- You forced the model to make the conditional structure visible
- You identified which branch you're on
- You shifted from getting the training prior to getting your posterior

**The prompts you rewrote are not better because they're longer. They're better because they supply the evidence the Bayesian calculation needs.**

---

## Next Steps

- Complete `notebooks/01_decision_tree_prompts.ipynb` if you haven't already
- Review `guides/02_uncertainty_prompting_guide.md` for techniques on getting the model to ask you questions before answering
- Module 5 applies these skills to agent workflows where the decision tree is executed dynamically
