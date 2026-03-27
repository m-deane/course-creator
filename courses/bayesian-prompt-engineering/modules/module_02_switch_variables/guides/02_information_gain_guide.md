# Information Gain: Why "More Detail" Is Not "Better Conditions"

## In Brief

Adding more words to a prompt does not automatically improve the answer. What matters is the *information content* of what you add — specifically, how much each new condition reduces the uncertainty in the answer distribution.

> **Core Insight:** Information gain measures how much a condition collapses the answer space. A single high-gain condition (jurisdiction, timing, objective) can reduce a 40-way distribution to a 2-way distribution. A dozen low-gain conditions (company background, formatting preferences, narrative context) can leave the answer distribution almost unchanged.

The skill of prompt engineering, viewed through information theory, is **maximizing entropy reduction per condition added**.

---

## Visual Explanation

### Before Any Conditions

```
Answer distribution for "How should I handle this situation?" (no context):

 Possible answers (representative):
 |
 | [corporate law] [family law] [crim defense] [contract] [IP] [employment]
 | ████████        ████████     ████████        ████████   ████████  ████████
 |
 High entropy — model holds mass across many answer worlds
 H ≈ 3.2 bits (roughly 9 equally probable answer branches)
```

### After Adding Jurisdiction + Entity Type (Two Switch Variables)

```
Answer distribution after "US federal | C-corporation | IP dispute":

 |
 | [corporate law] [crim defense] [family law] [contract] [IP] [employment]
 |                                                              ████████████
 |
 Low entropy — almost all mass on one answer branch
 H ≈ 0.4 bits (one branch dominates)
```

### After Adding Company Size + Founding Year (Two Descriptive Details)

```
Answer distribution after "47 employees | founded 2019":

 |
 | [corporate law] [family law] [crim defense] [contract] [IP] [employment]
 | ████████        ███████      ████████        ████████   ███████   ███████
 |
 Entropy barely changed — these details don't route to one branch
 H ≈ 3.0 bits (almost identical to baseline)
```

Two conditions. One collapses the answer space to near certainty. One changes almost nothing. This is the difference between information gain and noise.

---

## Formal Definition

For a question $Q$ with answer space $A$, define:

**Prior entropy** (before conditions):
$$H(A) = -\sum_{a \in A} P(a \mid Q) \log_2 P(a \mid Q)$$

**Conditional entropy** (after adding condition $C$):
$$H(A \mid C) = -\sum_{c \in C} P(c) \sum_{a \in A} P(a \mid Q, c) \log_2 P(a \mid Q, c)$$

**Information gain of condition $C$:**
$$\text{IG}(C) = H(A) - H(A \mid C)$$

A condition with $\text{IG}(C) = 0$ is pure noise — adding it to the prompt tells the model nothing about which answer world you're in. A condition with $\text{IG}(C) = H(A)$ is a complete resolver — it uniquely determines the answer branch.

**Practical implication:** Rank your conditions by $\text{IG}(C)$ and add them in descending order. The first few will do most of the work.

---

## The Entropy Intuition

Entropy measures how spread out probability mass is across possible answers. Think of it as the model's uncertainty about which world you're in:

| Entropy | Meaning | Prompt state |
|---------|---------|-------------|
| High (3+ bits) | Mass spread across many branches | No conditions — vague prompt |
| Medium (1-2 bits) | Mass concentrated in a few branches | Some conditions added |
| Low (< 0.5 bits) | Mass almost entirely on one branch | Switch variables specified |
| Zero | Complete certainty | Fully conditioned prompt |

The goal is not zero entropy — that would require completely specifying every condition, which is impossible. The goal is to reduce entropy to the point where the model's remaining uncertainty doesn't affect the quality of its answer for your use case.

---

## Why Numbers Are Usually Low Information Gain

Numbers are seductive because they feel precise. "47 employees," "$2.3M ARR," "3 years in operation." They feel like data.

But for most questions, these numbers do not route to different answer branches. The legal advice for a company with 47 employees is nearly identical to the advice for a company with 53 employees. The number adds no information about which legal world applies.

Compare this to the switch from "individual" to "corporation." This is a single binary variable, but it routes to an entirely different set of rules, liabilities, and strategic options. The information gain is enormous.

**Heuristic:** Numbers are high information gain only when they cross a threshold that changes category:
- Revenue > $10M (IRS large business threshold) — changes tax rules
- Employees > 50 (ACA employer mandate) — changes legal obligations
- Assets > $250k (certain SEC thresholds) — changes regulatory status
- Latency < 50ms (real-time interactive) vs. > 1 second (background) — changes architecture class

When a number crosses a threshold that changes the applicable rule set, it becomes a switch variable. When it doesn't cross such a threshold, it is decorative.

---

## Condition Ranking in Practice

### Example: "How do I structure my equity compensation plan?"

| Condition | Type | Information Gain |
|-----------|------|-----------------|
| US vs. international employees | Jurisdiction | Very high — different legal framework |
| C-Corp vs. LLC | Status | Very high — stock options only available to C-Corps |
| Pre- vs. post-Series A | Timing | High — 409A valuation requirements change |
| Goal: retention vs. acquisition prep | Objective | High — vesting cliffs and acceleration differ |
| Employee count > 100 | Constraint/threshold | Medium — affects exemption availability |
| Employees are W-2 vs. contractors | Status | High — equity for contractors has tax differences |
| Company revenue ($2M) | Descriptive number | Low — doesn't change equity plan structure |
| Company founded year | Descriptive | Near zero — irrelevant to plan structure |

**Add in this order:** C-Corp or LLC? US employees? What's the primary goal? Pre or post-Series A? Employee vs. contractor? Then stop — you've captured the high-gain conditions.

---

## The Diminishing Returns Curve

Information gain from conditions follows a concave curve:

```
Information Gain vs. Number of Conditions Added

Cumulative
information
gain
  |
  |         ●
  |       ●
  |     ●
  |   ●
  | ●
  |●____________________________________
  0   1    2    3    4    5    6    7    8
      Number of conditions added

● Condition 1 (jurisdiction):    captures ~40% of total reducible entropy
● Condition 2 (status):          captures ~30% more
● Condition 3 (objective):       captures ~15% more
● Condition 4 (timing):          captures ~10% more
● Condition 5+ (remaining):      each adds <5%
```

This is why the first 3-5 switch variables do the majority of the work. Adding a 20-condition prompt does not produce 4x better results than a 5-condition prompt. It may produce marginally better results at the cost of significant overhead.

**Rule of thumb:** Identify your top 5 switch variables. Add them all. Stop there unless you have a specific reason to add more.

---

## High-Gain vs. Low-Gain Conditions by Domain

### Law

| High information gain | Low information gain |
|-----------------------|---------------------|
| Governing jurisdiction | Company name |
| Case posture (pre/post filing) | Industry sector |
| Entity type of parties | Number of employees |
| Goal of representation | Years in business |
| Existence of arbitration clause | Company's product or service |

### Medicine

| High information gain | Low information gain |
|-----------------------|---------------------|
| Age cohort (pediatric/adult/geriatric) | Patient occupation |
| Immune status | Whether patient exercises |
| Acuity (acute/chronic) | Insurance type (usually) |
| Comorbidities with drug interactions | Patient preferences for communication |
| Care setting (ICU/inpatient/outpatient) | Exact symptom duration (when ordinal matters more) |

### Software Engineering

| High information gain | Low information gain |
|-----------------------|---------------------|
| Deployment target | Team size |
| Language and version | Company industry |
| Scale tier (requests/second) | Codebase age |
| Compliance requirements | Preferred coding style |
| Consistency requirements | Project management methodology |

### Finance

| High information gain | Low information gain |
|-----------------------|---------------------|
| Asset class | Fund manager's name |
| Objective (alpha/hedge/income) | Exact portfolio size (unless at threshold) |
| Market regime | Number of positions |
| Entity type (retail/institutional) | Specific securities held (unless for specific analysis) |
| Time horizon | Exact AUM |

---

## Measuring Information Gain Empirically

You can measure the information gain of a condition empirically by running the same question with and without the condition and comparing the responses.

**Procedure:**

1. Write the base question with no conditions.
2. Add a single condition and re-run.
3. Ask: "Did the answer change categorically (different recommendation, different legal regime, different approach) or superficially (different phrasing, more caveats)?"
4. Categorical change = high information gain condition. Superficial change = low gain.

This is exactly what Notebook 1 does with the Claude API: it measures how much each candidate condition shifts the model's response, and ranks conditions by that shift magnitude.

---

## The "Condition Budget" Mental Model

Think of your prompt as having a **condition budget** — a limited number of conditions before the model's context becomes cluttered or the response becomes over-conditioned.

Spend that budget on high-gain conditions first:

```
Budget = 5 conditions

Rank 1: Jurisdiction (IG: 0.8 bits) — SPEND
Rank 2: Entity status (IG: 0.7 bits) — SPEND
Rank 3: Objective function (IG: 0.6 bits) — SPEND
Rank 4: Timing/posture (IG: 0.5 bits) — SPEND
Rank 5: Primary constraint (IG: 0.4 bits) — SPEND
Rank 6: Secondary constraint (IG: 0.2 bits) — skip
Rank 7: Background detail (IG: 0.05 bits) — skip
```

The skip list is not "wrong" information — it's just not worth the budget. You have captured ~95% of the reducible entropy with 5 high-gain conditions.

---

## Summary

Information gain measures how much a condition collapses the answer space. Switch variables have high information gain — they route reasoning to categorically different solution branches. Descriptive details have low information gain — they refine presentation without changing the underlying answer.

**Key principles:**

1. Conditions have different information gain — rank them before adding them
2. Numbers are usually low gain unless they cross a categorical threshold
3. The first 3-5 high-gain conditions capture most of the reducible entropy
4. Adding more conditions after the top 5 produces diminishing returns
5. Measure information gain empirically: categorical answer change = high gain

**The practice skill:** Before any prompt, ask which conditions are high-gain (change the answer category) and which are low-gain (change the phrasing). Add the high-gain ones first, in order.

**Next:** Notebook 1 — Build a Claude-powered tool that identifies and ranks switch variables automatically.
