---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Information Gain
## Why "More Detail" Is Not "Better Conditions"

### Module 2 — Bayesian Prompt Engineering

<!-- Speaker notes: This slide deck builds on the switch variables concept by introducing the mechanism that explains why some conditions matter far more than others. Information gain from information theory gives us a rigorous way to rank conditions. The key insight to deliver early: volume of context does not equal quality of context. A 500-word system prompt with low-gain details can be outperformed by a 50-word prompt with three high-gain switch variables. -->

---

## The Same Question, Two Prompts

<div class="columns">

<div>

**Prompt A — More detail:**

> "We're a US company with 47 employees, founded in 2019, operating in 6 cities, targeting the mid-market HR software space, with $2.3M ARR. How do I structure equity comp?"

</div>

<div>

**Prompt B — Better conditions:**

> "C-Corp. US-based W-2 employees. Pre-Series A. Goal: employee retention, not near-term exit. How do I structure equity comp?"

</div>

</div>

Prompt A is 37 words. Prompt B is 19 words. Prompt B gets a significantly better answer. Why?

<!-- Speaker notes: Read both prompts aloud before revealing the explanation. Ask students which one they think will get a better answer — most will correctly guess Prompt B but won't be able to articulate why. The reason is information gain: Prompt B specifies three high-gain switch variables (C-Corp entity type, employee classification, objective function) while Prompt A provides 37 words of decorative context that doesn't route to any specific answer branch. -->

---

## What Is Information Gain?

For a question $Q$ with answer space $A$:

$$\text{IG}(C) = H(A) - H(A \mid C)$$

Where $H$ is entropy — the spread of probability mass across possible answers.

| $\text{IG}$ value | Meaning |
|-------------------|---------|
| Near $H(A)$ | Condition almost fully resolves the answer |
| Medium | Condition narrows to a few branches |
| Near 0 | Condition barely affects the answer distribution |
| Exactly 0 | Condition is pure noise |

**The goal:** Add conditions in descending order of $\text{IG}$.

<!-- Speaker notes: Don't get bogged down in the math — the intuition is what matters. Entropy measures how spread out the model's uncertainty is across possible answers. Information gain measures how much a specific condition reduces that spread. High-gain conditions dramatically narrow the answer space. Low-gain conditions barely move it. The practical implication is: rank your conditions by gain before adding them. -->

---

## Entropy: Visual Intuition

```
High entropy — before conditions:
Answer distribution for a legal question (no context)

Corporate law  ████████  (16%)
Family law     ████████  (14%)
Criminal def   ███████   (12%)
Contract       ████████  (15%)
IP law         ████████  (13%)
Employment     ████████  (15%)
Other          ████████  (15%)

H ≈ 2.8 bits — model is very uncertain which world applies


Low entropy — after adding jurisdiction + entity type:
Same question + "US federal | corporate | IP dispute"

IP law         █████████████████████████████████  (92%)
Other          ████  (8%)

H ≈ 0.35 bits — model is nearly certain which world applies
```

Two conditions reduced entropy by ~87%. Those were switch variables.

<!-- Speaker notes: Walk through the bar charts carefully. The before state shows near-uniform distribution — the model holds significant probability mass across many legal domains. After adding just two switch variables (jurisdiction and entity type), almost all the mass collapses to the correct domain. This is what high information gain looks like. The two conditions were worth more than a 500-word description of the company would have been. -->

---

## Why Numbers Are Usually Low Gain

**Common mistake:** Adding precise numbers to signal rigor.

> "47 employees, $2.3M ARR, operating since 2019 in 6 locations"

**The problem:** These numbers do not change the answer branch.

The legal advice for 47 employees ≈ legal advice for 53 employees.

Numbers are **high gain only** when they cross a **categorical threshold:**

| Number | Threshold | What changes |
|--------|-----------|-------------|
| Employees | ≥ 50 | ACA employer mandate activates |
| Revenue | ≥ $10M | IRS large-business rules |
| Assets | ≥ $250k | Certain SEC registration thresholds |
| Requests/sec | ≥ 10k | Architecture class changes |

If the number doesn't cross a threshold — it's decoration.

<!-- Speaker notes: This is counterintuitive for analytical professionals. They're trained that more precise data is better data. In the context of information gain, precision without category change is noise. A number that doesn't route to a different answer branch adds nothing. Once students understand the threshold principle, they can evaluate any number they're tempted to add: does this number cross a threshold that changes which rules apply? If yes, it's a switch variable. If no, leave it out. -->

---

## The Diminishing Returns Curve

```
Cumulative information gain as conditions are added:

100% ─────────────────────────────────── ●─────────────
                                       ●
 75% ──────────────────────────────● ──────────────────
                                ●
 50% ─────────────────────────────────────────────────
                          ●
 25% ──────────────────────────────────────────────────
                  ●
  0% ─────────────────────────────────────────────────
      0    1    2    3    4    5    6    7    8    9
              Number of conditions added

  Condition 1 (jurisdiction):  +38% of reducible entropy
  Condition 2 (entity type):   +28%
  Condition 3 (objective):     +17%
  Condition 4 (timing):        +10%
  Condition 5 (constraint):    + 5%
  Conditions 6-9:              + 2% each
```

The first 3-5 conditions do most of the work. After that: diminishing returns.

<!-- Speaker notes: The shape of this curve is what makes the "add more detail" instinct so costly. Adding conditions 6 through 9 provides only 8% combined benefit, but they cost attention, prompt length, and can introduce noise. Students should aim to identify the inflection point — the moment where additional conditions provide minimal new information gain. For most domains, that inflection is around condition 4 or 5. -->

---

## High-Gain vs. Low-Gain Side by Side

<div class="columns">

<div>

**High information gain:**

- Jurisdiction / governing law
- Entity type or status
- Objective function
- Case posture / timing
- Binding constraints
- Regulatory regime

These **route to different solution branches**

</div>

<div>

**Low information gain:**

- Company name
- Founding year
- Headcount (when not at threshold)
- Industry (when not determinative)
- Preferred formatting
- Narrative background
- "We've been working on this for..."

These **don't move the answer distribution**

</div>

</div>

<!-- Speaker notes: Have students categorize examples from their own work. The goal is to build the intuition quickly so it becomes automatic. When writing a prompt, the mental habit should be: "Is this a routing condition or a descriptive detail?" If it's descriptive, it goes at the end or gets dropped entirely. If it routes to a different answer branch, it goes at the top of the prompt. -->

---

## Measuring Gain Empirically

You don't need to calculate entropy formally. Use this practical test:

**Test one condition at a time:**

```
Base prompt:  "What database should I use?"
              → Response: "PostgreSQL is commonly used..."  (generic)

+ Condition 1: "...for real-time analytics at 50k events/sec"
              → Response: "ClickHouse or TimescaleDB — here's why..."
              → CATEGORICAL CHANGE — high IG

+ Condition 2: "...at a company called Acme Corp"
              → Response: "PostgreSQL is commonly used..."
              → NO CHANGE — near-zero IG
```

Categorical change in answer = high information gain.
Rephrasing of same recommendation = low information gain.

<!-- Speaker notes: This empirical approach is what the notebook exercises make concrete. Students will run exactly this test with the Claude API: add one condition at a time and measure how much the response changes. The qualitative signal — categorical change vs. superficial rephrasing — is often sufficient to rank conditions without any formal information theory calculation. The notebook quantifies this using response similarity metrics. -->

---

## Ranking Conditions Before Writing the Prompt

**Before:** Write prompt, then wonder why the answer is generic.

**After:** Rank conditions by gain, then write a tight prompt.

**The process:**
1. Name the question
2. List all conditions you might add
3. Assign each a category (jurisdiction, status, objective, timing, constraint)
4. Ask: "Does this change the answer category, or just its phrasing?"
5. Sort by expected impact
6. Add top 3-5 to the prompt

This takes 2-3 minutes. It produces better results than spending 20 minutes writing a longer prompt.

<!-- Speaker notes: Emphasize the time investment. This ranking process takes 2-3 minutes. The average professional spends 10-15 minutes writing a detailed prompt that adds low-gain details and misses the high-gain switch variables. The ranking process is a better investment of time. Students should practice this before every significant AI interaction until it becomes automatic — ideally within 2 weeks of regular practice. -->

---

## Condition Budget: Spend It Wisely

<div class="columns">

<div>

**Budget of 5 conditions**

Rank by estimated gain:

| Rank | Condition | Estimated gain |
|------|-----------|---------------|
| 1 | Jurisdiction | Very high |
| 2 | Entity type | Very high |
| 3 | Objective | High |
| 4 | Timing | Medium |
| 5 | Constraint | Medium |

</div>

<div>

**Spend vs. skip:**

- Ranks 1-5: **SPEND** → add to prompt
- Rank 6+: **SKIP** → not worth the budget

Each condition you add past rank 5 gives you diminishing returns while adding cognitive overhead for the model.

A tight 5-condition prompt often outperforms a bloated 15-condition prompt.

</div>

</div>

<!-- Speaker notes: The budget metaphor is useful because it makes the tradeoff concrete. You have limited attention — both yours and the model's. Spending that budget on low-gain conditions is a real cost. A 15-condition prompt that includes 10 low-gain conditions can actually perform worse than a 5-condition prompt, because the relevant high-gain conditions get diluted by noise. Quality of conditions beats quantity of conditions every time. -->

---

## Common Mistake: The Narrative Setup

**Prompt with narrative setup (common):**

> "I've been working with a client for three years. They're a healthcare company based in Chicago. We've had some challenges lately with their software infrastructure. They're growing fast and the team is stressed. Here's my question: what database should I use?"

**Same prompt, information gain first:**

> "Healthcare. HIPAA-required. 500k patient records. Relational query patterns. AWS deployment. What database?"

The narrative version added 40 words of near-zero gain context. The lean version specifies all five high-gain switch variables.

<!-- Speaker notes: This is the most common failure mode in practice. Professionals write narrative setups the same way they'd write an email or a business case — with contextual framing. But the model doesn't need narrative framing. It needs switch variables. The healthcare example is instructive: once you specify HIPAA, record count, query pattern, and deployment target, the database recommendation is near-deterministic. None of that information was in the narrative setup. -->

---

## Summary

<div class="columns">

<div>

**Information gain (IG) measures how much a condition reduces answer entropy.**

High IG conditions:
- Route to different solution branches
- Change the answer category
- Are switch variables by definition

Low IG conditions:
- Describe context without routing
- Change phrasing, not substance
- Are decorative (not harmful, just low ROI)

</div>

<div>

**Practical rules:**

1. Rank before adding
2. Numbers only matter at thresholds
3. Top 3-5 conditions do 80%+ of the work
4. Narrative setup is usually low gain
5. Measure gain empirically: categorical answer change = high gain

**Budget = 5 conditions. Spend on high-gain only.**

</div>

</div>

<!-- Speaker notes: Close by reinforcing the mental model shift. Moving from "add more context" to "rank conditions by information gain" is the fundamental shift this module teaches. Students who make this shift will write fewer, tighter, more effective prompts. They'll also develop a diagnostic skill: when a prompt returns a bad answer, they can ask "which high-gain condition am I missing?" rather than "how do I write this more clearly?" The missing condition is almost always the answer. -->

---

<!-- _class: lead -->

## Up Next: Notebook 1

### Build a Claude-Powered Switch Variable Identifier

See information gain in action by measuring how much each condition shifts Claude's response — automatically.

<!-- Speaker notes: Transition to the notebook. Students will implement the empirical information gain measurement they just learned about conceptually. The notebook uses the Claude API to run the same question with and without each candidate condition, measures the semantic distance between responses, and ranks conditions by their impact. This makes the abstract information gain concept tangible and immediately applicable. -->
