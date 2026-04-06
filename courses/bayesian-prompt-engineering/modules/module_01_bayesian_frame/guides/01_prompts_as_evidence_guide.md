# Prompts as Evidence: The P(A|C) Frame

> **Reading time:** ~8 min | **Module:** 1 — Bayesian Frame | **Prerequisites:** Module 0 Foundations


## In Brief

A language model does not retrieve answers. It samples from a conditional distribution over possible answers, where your prompt defines the conditioning event. Understand this and you understand why most prompts fail — and exactly how to fix them.


<div class="callout-key">

<strong>Key Concept Summary:</strong> A language model does not retrieve answers.

</div>

---

## Key Insight

> When you write a prompt, you are not issuing a command. You are providing evidence. The model uses that evidence to narrow the space of plausible answers. If your evidence is weak, the model falls back on the statistical center of its training distribution — the "average world." Your prompt collapses possible worlds. The question is whether it collapses to the *right* one.

---

## Formal Frame

### The Conditional Distribution

Language models assign probability to every possible continuation of a sequence. When you provide a prompt $C$ (conditions), the model generates answers from:
<div class="callout-insight">

<strong>Insight:</strong> Language models assign probability to every possible continuation of a sequence. When you provide a prompt $C$ (conditions), the model generates answers from:

</div>


$$P(A \mid C)$$

This is not a lookup. It is a posterior distribution — the model's belief about the right answer *given* everything you provided.

### Bayes' Theorem Applied

The posterior relates to the model's prior through Bayes' theorem:

$$P(A \mid C) \propto P(C \mid A) \cdot P(A)$$

Each component has a direct interpretation:

| Symbol | Meaning in Prompting | Example |
|--------|---------------------|---------|
| **P(A)** | Prior — the model's "default world" from training | "Tax questions usually mean: standard filing, US, current year, timely submission" |
| **P(C\|A)** | Likelihood — how consistent are your conditions with each possible answer | A late-filing penalty question is very consistent with A = "late filing rules," inconsistent with A = "general tax theory" |
| **P(A\|C)** | Posterior — the answer the model actually produces | With the right conditions, collapses onto the specific rules that apply to your situation |

### The Proportionality is the Lesson

The "∝" sign (proportional to) means the normalizing constant — everything else the model could possibly say — divides out. What determines your answer is the **ratio** between your evidence favoring the right answer versus all other answers.

This means:

- Adding irrelevant detail does not help (it doesn't change the ratio)
- Adding evidence that discriminates between the right answer and the default answer shifts the posterior dramatically
- The same prompt in different contexts produces different answers because context changes the prior

---

## The Prior: What the Model Assumes When You Say Nothing
<div class="callout-warning">

<strong>Warning:</strong> The model's training prior is the aggregate distribution of all text it processed. For any question type, there is a statistical center — the most common framing, the most common jurisdiction, the most common timeframe, the most common objective.

</div>


The model's training prior is the aggregate distribution of all text it processed. For any question type, there is a statistical center — the most common framing, the most common jurisdiction, the most common timeframe, the most common objective.

When you ask a tax question, the model's P(A) encodes:

- United States federal tax law (highest frequency in training)
- Current tax year (most common in training)
- Timely filing (the modal case — most people file on time)
- Standard deductions (the typical choice)
- Ordinary income (the default income type)

Every deviation from this "modal world" requires you to provide evidence that shifts the posterior away from it.

### The Prior is Invisible Until It Dominates

This is why weak prompts produce generic answers that *feel* correct. The prior is not wrong — it describes a real world. It is just not *your* world. And if your prompt doesn't specify otherwise, you get the statistical average, which is correct in the aggregate and wrong for your specific case.

---

## The Accountant Example: Watching a Posterior Shift
<div class="callout-key">

<strong>Key Point:</strong> Consider this question: *"What happens if I miss the filing deadline?"*

</div>


Consider this question: *"What happens if I miss the filing deadline?"*

### Before Adding Evidence: The Prior World

The model reasons in the "typical filing" world:

- **Who asks this?** Someone considering whether to file late, probably still before the deadline
- **Jurisdiction assumed:** US federal
- **Filing type assumed:** Individual 1040
- **Situation assumed:** Hasn't filed yet, asking prospectively

The response covers: failure-to-file penalty (5% per month), failure-to-pay penalty (0.5% per month), interest accrual, the extension option, how to request one.

This is correct — for the prior world.

### After Adding One Sentence: A Posterior Shift

New prompt: *"What happens if I miss the filing deadline? I already filed late in 2026 and didn't request an extension."*

The phrase "filed late in 2026" and "didn't request an extension" collapses several worlds:

| Before the Sentence | After the Sentence |
|--------------------|--------------------|
| Prospective question | Retroactive situation |
| "Should I file late?" | "I did file late — what now?" |
| Extension is relevant | Extension is no longer available |
| Penalty rates (general) | Penalty calculation (specific period) |
| How to avoid penalties | First-time penalty abatement eligibility |
| General IRS guidance | IRS CP2000 / collection timeline |

The model now reasons in a different world. Not because you added more words — because you added *discriminating evidence* that shifted the posterior away from the prior.

### The Diagram

```
                    Prior P(A)
                    (training distribution)
                         │
                         ▼
         ┌───────────────────────────────┐
         │   Possible Worlds             │
         │                               │
         │   [Late filing retroactive]  ← ── "filed late in 2026"
         │   [On-time filing]           ←── collapses this out
         │   [Extension requested]      ←── "didn't request extension"
         │   [Business tax question]    ←── collapses this out
         │   [Non-US jurisdiction]      ←── collapses this out
         └───────────────────────────────┘
                         │
                         ▼
                  Posterior P(A|C)
                  (narrow, specific)
```

One sentence did not "add detail." It provided evidence that eliminated most of the prior probability mass and concentrated it on the relevant world.

---

## Why Most Prompts Fail: The Evidence Gap

The most common failure mode is not that prompts are short. It is that they are long with information that does not discriminate between worlds.
<div class="callout-insight">

<strong>Insight:</strong> The most common failure mode is not that prompts are short. It is that they are long with information that does not discriminate between worlds.

</div>


### Non-Discriminating Information

- "I am a professional working on an important project" — true of nearly everyone asking work questions; does not shift the posterior
- "Please be thorough and detailed" — instruction about output format, not evidence about the situation
- "I have tried other approaches but they haven't worked" — consistent with every possible answer
- Background that the model would assume anyway

### Discriminating Information

- Jurisdiction (shifts away from default US federal assumptions)
- Timing (shifts away from current-year or prospective assumptions)
- Constraints (rules out entire classes of solutions)
- Objective function (changes what "good answer" means)
- Failure mode (narrows the space to problems with a specific symptom)

The difference is not volume. It is **specificity of exclusion** — what does your evidence rule out?

---

## The Stabilization Effect

As you add more discriminating conditions, the posterior narrows. At some point, adding more evidence produces diminishing returns — the posterior is already concentrated on a small region of answer space.

This produces a testable prediction: responses should become more consistent, more specific, and shorter (in the sense of fewer caveats and alternatives) as evidence accumulates. You will verify this directly in notebook `01_posterior_shift_simulator.ipynb`.

---

## Common Pitfalls

**Pitfall 1: Confusing elaboration with evidence.**
Writing three paragraphs about your situation is elaboration. Stating the one fact that rules out five alternative interpretations is evidence.

**Pitfall 2: Providing evidence that matches the prior.**
Saying "I'm asking about US taxes" when the model already assumes US taxes does not shift the posterior. You need evidence that departs from the default.

**Pitfall 3: Providing contradictory evidence.**
Saying "this is an urgent business matter" and "I'm just exploring this theoretically" simultaneously creates a mixed posterior. The model must hedge.

**Pitfall 4: Asking for the posterior before specifying the conditions.**
Asking "what should I do?" before establishing the constraints, timeline, and objective produces the prior-dominated answer. Conditions first, question second.

---

## Connections

- **Builds on:** Basic familiarity with language models, sending prompts
- **Leads to:** Guide 02 (evidence vs. information), Module 2 (prior strength by domain), Module 3 (structured condition elicitation)
- **Related to:** Bayesian updating in statistics, conditional probability, decision theory

---

## Practice

1. Take any prompt you have sent in the past two weeks that produced a generic answer. Identify: what world was the model defaulting to? What one or two facts would have shifted the posterior to your actual world?

2. Write two versions of a prompt for a question in your domain of expertise: one that will activate the prior, one with discriminating evidence. Predict what will differ in the responses before sending them.

3. In the accountant example, identify three more facts (beyond "filed late in 2026") that would further narrow the posterior. What world does each one rule out?

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Key Insight" and why it matters in practice.

2. Given a real-world scenario involving prompts as evidence: the p(a|c) frame, what would be your first three steps to apply the techniques from this guide?

</div>

## Further Reading

- Griffiths, T. L. & Tenenbaum, J. B. (2006). "Optimal predictions in everyday cognition." *Psychological Science* — Bayesian inference as a model of human reasoning
- Brown et al. (2020). "Language Models are Few-Shot Learners." *NeurIPS* — The in-context learning framing that underlies the evidence interpretation

---

## Cross-References

<a class="link-card" href="../notebooks/01_posterior_shift_simulator.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
