# Evidence vs. Information: Why "More Detail" Often Fails

> **Reading time:** ~11 min | **Module:** 1 — Bayesian Frame | **Prerequisites:** Module 0 Foundations


## In Brief

Information and evidence are not the same thing. Information is any fact you include in a prompt. Evidence is a fact that changes the posterior — that shifts the model's reasoning away from its training prior toward your specific situation. Most "detailed" prompts are full of information and empty of evidence.


<div class="callout-key">
<strong>Key Concept Summary:</strong> Information and evidence are not the same thing.
</div>

---

## Key Insight

> Adding words to a prompt that match what the model would already assume does not change the posterior. You are not providing new information in the information-theoretic sense. You are confirming the prior. The model ignores confirmation of what it already believes and reasons from the same default world.

---

## The Information-Evidence Distinction

### Information
<div class="callout-insight">
<strong>Insight:</strong> A statement is **information** if it is true and relevant to the topic.
</div>


A statement is **information** if it is true and relevant to the topic.

- "I am working on a tax question related to my business"
- "This is an important decision and I need accurate guidance"
- "I've researched this but haven't found a clear answer"
- "Please consider all relevant factors"

These are all true. They are all relevant. None of them shift the posterior.

Why? Because they are all consistent with the prior world. The model already assumes:
- Tax questions come from people with business or personal tax situations
- Questions are asked because the answer matters
- Questioners have done some prior research
- Complete and accurate guidance is the goal

### Evidence

A statement is **evidence** if it is a fact that **narrows the solution space** — that makes some possible answers more probable and others less probable.

- "This is for a Delaware C-corp, not a sole proprietorship"
- "The transaction closed in Q4 2025, before the regulation change"
- "The constraint is that we cannot change the database schema"
- "The objective is minimizing regulatory risk, not minimizing tax liability"

Each of these eliminates categories of possible answers and directs the model toward a specific subset of answer space.

---

## Why Numbers Are Not Automatically Evidence

A common mistake: adding numbers to a prompt as if they are inherently more specific.
<div class="callout-warning">
<strong>Warning:</strong> A common mistake: adding numbers to a prompt as if they are inherently more specific.
</div>


**Example with numbers (not evidence):**

> "I have about 5-7 years of experience, work with approximately 200-300 clients per year, and generate roughly $800K-$1.2M in annual revenue. What tax strategies should I consider?"

These numbers feel specific. But they don't shift the posterior because:

1. The model has no baseline to compare them against — it doesn't know what "normal" looks like for this profession
2. The range (5-7 years, 200-300 clients) is still consistent with many different strategies
3. Revenue in this range is consistent with many different tax situations
4. None of the numbers constrain the solution space in a meaningful way

**The same information, reframed as evidence:**

> "I am a consultant structured as an S-corp and I have not yet set up a solo 401(k) or SEP-IRA. What tax-advantaged retirement strategies am I missing?"

The phrase "structured as an S-corp" immediately narrows to S-corp-specific strategies (not sole prop, not C-corp). "Not yet set up" shifts from "tell me about all strategies" to "tell me what I'm missing." These are evidence — they constrain the space.

---

## The Four Categories of Discriminating Evidence

Research on what actually changes model responses points to four categories of conditions that consistently shift posteriors:
<div class="callout-key">
<strong>Key Point:</strong> Research on what actually changes model responses points to four categories of conditions that consistently shift posteriors:
</div>


### 1. Constraints

A constraint is a fact that **rules out solution categories**.

- "We cannot use a third-party API" — rules out integration solutions
- "The contract was already signed" — rules out advice about contract terms
- "Budget is fixed at $50K" — rules out strategies above that threshold
- "This is a regulated industry (healthcare)" — rules out approaches that violate HIPAA

Every constraint is evidence because it eliminates answers. When you tell the model a constraint, you are not adding information — you are subtracting answer worlds.

### 2. Timing and Jurisdiction

Timing and jurisdiction are among the most powerful evidence categories because they change what law, precedent, and practice applies.

- **Timing:** "before the 2023 SECURE 2.0 changes" vs. "under current 2025 rules" → different answer
- **Jurisdiction:** "in California" vs. "in Texas" → different answer for almost any legal or regulatory question
- **Phase:** "we're in discovery" vs. "we have a judgment" → different procedural answer
- **Version:** "using Python 3.10" vs. "on Python 3.12" → different answer for certain API questions

The model's prior assumes the most common case (current rules, US federal, the most common jurisdiction for the topic). Any deviation from those defaults is evidence.

### 3. Jurisdiction Over the Objective Function

The objective function is what you're optimizing for. Changing the objective function changes what counts as a good answer.

Compare these two prompts about the same situation:

> "We have a regulatory violation that was identified internally. What should we do?"

vs.

> "We have a regulatory violation that was identified internally. Our objective is to preserve the relationship with the regulator and maintain our operating license, not to minimize the current penalty."

The second prompt specifies the objective function explicitly. This matters because:
- Different objectives favor different actions (disclosure vs. remediation vs. negotiation)
- The model's prior objective for regulatory questions is typically "minimize penalty" or "resolve quickly"
- Stating a different objective shifts the posterior toward recommendations that serve that objective

### 4. The Failure Mode You Observed

When you tell the model what went wrong, you provide evidence about what is *not* the answer.

> "My Python script runs successfully but produces incorrect numerical output on inputs larger than 10,000."

This single sentence contains evidence in multiple directions:
- Syntax errors are ruled out (it runs)
- Import errors are ruled out (it runs)
- The problem is numerical, not structural (incorrect output, not crash)
- There is a threshold effect (>10,000) suggesting integer overflow, floating point precision, or an algorithm with O(n) degradation

Contrast with: "My Python script isn't working correctly." — This is information with no discriminating power.

---

## Side-by-Side Comparison: Same Question, Different Evidence Quality

### Domain: Medical

**Information-heavy (still generic):**
> "I am a 45-year-old adult who has been experiencing fatigue for the past few months. I am generally healthy and try to maintain an active lifestyle. I eat a balanced diet and get 7-8 hours of sleep. What might be causing my fatigue?"

Why this fails: "Generally healthy," "active lifestyle," "balanced diet," "7-8 hours" — all confirm the prior world (healthy adult with unexplained fatigue). The model covers the same differential diagnosis it would for any adult with fatigue.

**Evidence-strong (specific):**
> "I am a 45-year-old adult with fatigue that is notably worse in the morning and improves through the day, specifically accompanied by weight gain despite unchanged diet, cold intolerance, and dry skin over 3 months. No recent stress changes."

Why this works: The symptom cluster (morning-worse fatigue, weight gain, cold intolerance, dry skin, 3-month timeline) provides evidence that discriminates strongly toward hypothyroidism. The model now reasons in that specific diagnostic world.

### Domain: Code Generation

**Information-heavy (still generic):**
> "I need to write a Python function that processes a large CSV file efficiently. The file contains sales data with multiple columns. I want it to be fast and memory-efficient."

Why this fails: "Large file," "sales data," "fast and memory-efficient" — the model produces pandas with chunked reading, the standard pattern for this class of problem.

**Evidence-strong (specific):**
> "I need to process a 50GB CSV file on a machine with 4GB of RAM. The processing is: filter rows where `status == 'completed'`, sum the `amount` column by `customer_id`. The schema is fixed. I cannot use Spark or Dask — only the standard library and pandas."

Why this works: 50GB on 4GB RAM rules out in-memory solutions. "Standard library and pandas only" rules out the Spark/Dask answer. "Filter then aggregate" specifies the exact operations. The model now knows the precise constraints and the exact transformation — the posterior collapses onto chunked streaming aggregation.

### Domain: Business Strategy

**Information-heavy (still generic):**
> "Our SaaS company is growing and we need to think about our pricing strategy. We have different types of customers and we want to maximize revenue. What pricing approaches should we consider?"

Why this fails: "Growing SaaS" and "maximize revenue" with "different customer types" describes essentially every SaaS company. The model covers the standard playbook: tiered pricing, usage-based, freemium, enterprise.

**Evidence-strong (specific):**
> "Our SaaS company has a free tier converting at 8% to paid (industry average is 2-4%), and our paid tier churns at 3% monthly (industry average is 5-7%). We have strong retention but weak initial conversion. We cannot change the product for 6 months."

Why this works: The specific metrics (8% conversion = 2-4x above average, 3% churn = below average) provide evidence about which part of the funnel needs work. "Cannot change the product" is a constraint that rules out product-led growth answers. The model now reasons in the specific world: strong product-market fit, weak top-of-funnel, product-locked.

---

## The Marginal Evidence Test

Before adding any sentence to a prompt, apply this test:
<div class="callout-warning">
<strong>Warning:</strong> Before adding any sentence to a prompt, apply this test:
</div>


> If the model didn't have this sentence, would it produce a meaningfully different response?

If the answer is no — if the model would produce essentially the same response without this sentence — the sentence is information, not evidence. Cut it or replace it with something that passes the test.

This test is rigorous. Most sentences in most prompts fail it.

### Applying the Test

"I'm asking for professional purposes." — Would the model produce a different response if you didn't say this? No. Cut it.

"The data includes timestamps in ISO 8601 format." — Would the model handle timestamps differently without knowing the format? Yes — it might produce strptime code for various formats. This passes. Keep it.

"I need a clear and comprehensive answer." — Would the model produce a less clear or comprehensive answer without this? No. Cut it.

"The database is PostgreSQL 14, not MySQL." — Would the model use different SQL syntax without this? Yes, possibly. This passes. Keep it.

---

## Common Pitfalls

**Pitfall 1: The credential fallacy.**
"As a certified professional with 20 years of experience..." does not change the answer to your question. Your credentials don't shift the posterior — the specific facts of your situation do.

**Pitfall 2: The effort signal.**
"I've spent weeks on this problem and tried everything..." signals effort, not evidence. The model does not reward effort signals with better answers. It responds to conditions.

**Pitfall 3: The hedge response.**
"Please consider all possibilities and be comprehensive" explicitly asks the model to remain in the prior world — to cover all possibilities rather than narrow to yours. If you want specificity, provide evidence. Don't ask for generality and expect specificity.

**Pitfall 4: Providing evidence the model can't use.**
"I need this by tomorrow" — the model cannot prioritize based on your deadline. This is information with no effect on the posterior.

---

## Connections

- **Builds on:** Guide 01 — Prompts as Evidence (the P(A|C) frame)
- **Leads to:** Notebook 02 — Evidence Strength Comparison; Module 3 — Structured Condition Elicitation
- **Related to:** Information theory (information content of a message), sufficient statistics, feature selection in machine learning

---

## Practice

1. Take a prompt you have used in the last week. Apply the marginal evidence test to every sentence. How many sentences pass? How many fail?

2. For each of the four categories (constraints, timing/jurisdiction, objective function, failure mode) — write one example condition in your domain that would shift a posterior significantly.

3. The medical example above contains a symptom cluster. What is the diagnostic equivalent in your field? What cluster of specific facts rules in one answer and rules out everything else?

---


---

## Practice Questions

<div class="callout-info">
<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Key Insight" and why it matters in practice.

2. Given a real-world scenario involving evidence vs. information: why "more detail" often fails, what would be your first three steps to apply the techniques from this guide?
</div>

## Further Reading

- Shannon, C. E. (1948). "A Mathematical Theory of Communication." — The formal definition of information content that underlies the evidence distinction
- Grice, H. P. (1975). "Logic and Conversation." — The maxim of quantity: say exactly as much as needed, no more; a precursor to the evidence framing

---

## Cross-References

<a class="link-card" href="../notebooks/01_posterior_shift_simulator.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
