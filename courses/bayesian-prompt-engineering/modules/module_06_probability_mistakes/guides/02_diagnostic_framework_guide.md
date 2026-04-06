# Diagnostic Framework: When AI Gives Bad Answers

> **Reading time:** ~10 min | **Module:** 6 — Probability Mistakes | **Prerequisites:** Module 1 Bayesian Frame


## In Brief

When AI gives a bad answer, the cause is almost always structural — a missing condition, a misspecified objective, a temporal context the model couldn't infer. This guide provides a systematic diagnostic flowchart for identifying which of the six mistake types caused the problem, and applying the precise fix.

## Key Insight

Treat every bad AI answer as a diagnostic signal, not a failure. The answer tells you exactly what evidence the model thought it had. Work backwards from the answer to find the missing condition.


<div class="callout-key">

<strong>Key Concept Summary:</strong> When AI gives a bad answer, the cause is almost always structural — a missing condition, a misspecified objective, a temporal context the model couldn't infer.

</div>

---

## The Diagnostic Process

Before running the flowchart, read the bad answer once and ask: "Who would get this answer? What kind of person, in what kind of situation, with what kind of goals?" That person is not you — and the gap between that person and you is where your missing evidence lives.

---

## The Diagnostic Flowchart

```
START: You received a bad answer
         │
         ▼
┌─────────────────────────────┐
│ Is the answer generic?      │
│ (Could apply to anyone)     │
└─────────────────────────────┘
         │
    ┌────┴────┐
   YES        NO
    │          │
    ▼          ▼
┌──────────┐  ┌──────────────────────────────┐
│ Did you  │  │ Is the answer wrong for YOUR │
│ add lots │  │ situation specifically?       │
│ of text  │  └──────────────────────────────┘
│ and still│           │
│ get this?│      ┌────┴────┐
└──────────┘     YES        NO
    │              │          │
   YES             ▼          ▼
    │     ┌──────────────┐  ┌────────────────────┐
    ▼     │ MISTAKE 1    │  │ Is the answer       │
┌───────┐ │ Detail ≠     │  │ inconsistent with   │
│MISTAKE│ │ Conditions   │  │ other answers the   │
│  1    │ │              │  │ model has given you?│
│Detail │ │ Fix: Replace │  └────────────────────┘
│≠      │ │ narrative    │           │
│Conditi│ │ with specific│      ┌────┴────┐
│ons    │ │ discriminatin│     YES        NO
└───────┘ │ g conditions │      │          │
          └──────────────┘      ▼          ▼
                        ┌──────────────┐  ┌──────────────┐
                        │ MISTAKE 2    │  │ Is the answer│
                        │ One answer   │  │ vague or     │
                        │ instead of   │  │ hedge-heavy? │
                        │ tree         │  └──────────────┘
                        │              │           │
                        │ Fix: Ask for │      ┌────┴────┐
                        │ the tree     │     YES        NO
                        │ first        │      │          │
                        └──────────────┘      ▼          ▼
                                     ┌──────────────┐  ┌──────────────┐
                                     │ MISTAKE 4    │  │ Is the answer│
                                     │ No objective │  │ right in     │
                                     │ function     │  │ general but  │
                                     │              │  │ wrong for    │
                                     │ Fix: State   │  │ your moment? │
                                     │ what you're  │  └──────────────┘
                                     │ optimizing   │           │
                                     │ for          │      ┌────┴────┐
                                     └──────────────┘     YES        NO
                                                           │          │
                                                           ▼          ▼
                                                  ┌──────────────┐  ┌──────────────┐
                                                  │ MISTAKE 5    │  │ MISTAKE 6    │
                                                  │ Temporal     │  │ Misaligned   │
                                                  │ conditions   │  │ priors       │
                                                  │ missing      │  │              │
                                                  │              │  │ Fix: Make    │
                                                  │ Fix: Specify │  │ your priors  │
                                                  │ phase/stage/ │  │ explicit     │
                                                  │ timing       │  └──────────────┘
                                                  └──────────────┘
```

---

## Question-by-Question Diagnosis

The flowchart above is the fast path. Below is the deeper diagnostic question set for each branch.
<div class="callout-key">

<strong>Key Point:</strong> The flowchart above is the fast path. Below is the deeper diagnostic question set for each branch.

</div>


### Branch 1: Is the Answer Generic?

**Test:** Replace your name with someone else's name in the prompt. Would they get the same answer? If yes, the answer is generic.

**Why it happens:** Either the evidence was not discriminating (Mistake 1) or the prompt was structured like a search query (Mistake 3).

**Distinguish Mistake 1 from Mistake 3:**
- Mistake 1: You wrote a paragraph of context but got a generic answer → your text was information, not evidence
- Mistake 3: You wrote a short keyword phrase and got a document → you used a retrieval format, not an inference format

**Fixes:**
- Mistake 1: Convert narrative to condition lists. Each condition must be capable of changing the answer.
- Mistake 3: Add: current state → target state → constraints → ask for reasoning under those conditions.

---

### Branch 2: Is the Answer Wrong for Your Situation Specifically?

**Test:** Is there a version of you — a different company size, different industry, different experience level — for whom this answer would be correct? If yes, the model answered for that version of you.

**Why it happens:** The model collapsed a conditional tree to a single verdict (Mistake 2) and picked the branch that represents the most common case in training data.

**Fix:** Ask: "Before answering, what are the 3-5 conditions that would change this answer? Map them as a decision tree. Then tell me which branch I'm on, given [your specific conditions]."

---

### Branch 3: Is the Answer Inconsistent?

**Test:** Ask the same question twice with slight prompt variations. Do you get different answers that contradict each other? Or ask a related question and the answer conflicts with a previous one?

**Why it happens:** The question has hidden conditional structure. Each prompt variation landed you on a different branch of the implicit tree. The answers are both correct — for different branches — but you haven't specified which branch you're on.

**Fix:** The answer inconsistency is actually useful information. It reveals the conditions that are switching you between branches. Make those conditions explicit. Then ask for the full conditional tree with explicit branch labels.

---

### Branch 4: Is the Answer Vague or Hedge-Heavy?

**Test:** Count the number of "it depends," "you might want to consider," "there are tradeoffs," and "it really varies" phrases. More than 2-3 of these in a response means the model doesn't have enough information to commit to a recommendation.

**Why it happens:** The model is doing its job — it genuinely cannot give a precise answer without knowing your objective function. The hedges are the model's way of surfacing the missing variable (Mistake 4).

**Fix:** State the objective function explicitly. What are you optimizing for? What are you willing to sacrifice? What are your hard constraints? Once the objective is clear, the hedges should collapse into a recommendation.

**Useful prompt addition:** "I need a recommendation, not a list of tradeoffs. My primary objective is [X]. Make that assumption and give me a specific answer."

---

### Branch 5: Is the Answer Right in General but Wrong for Your Moment?

**Test:** Read the advice and ask: "Is this correct, but for someone who is 6 months ahead of me? Or 6 months behind me?" Does it describe what you should do eventually, or what you should do right now?

**Why it happens:** The model answered the timeless version of the question, not the temporally-conditioned version (Mistake 5). Most training data describes best practices in steady state, not transition states.

**Fix:** Add explicit temporal conditions: current phase, current constraints, time horizon for the decision, and the trigger that would change the answer. Specifically ask: "Given that I'm currently at [phase/milestone/stage], what's the right action right now — not in steady state?"

---

### Branch 6: Is the Answer Correct for Average but Wrong for You?

**Test:** Is the advice technically sound and clearly explained — but it doesn't account for a constraint or belief you have that you didn't mention? Does it feel like advice for a generic version of your situation?

**Why it happens:** The model answered from training-data priors, which represent the average case. Your priors differ from the average, but you didn't state them (Mistake 6).

**Fix:** Before re-prompting, list your non-standard priors:
- What constraints do you have that the average person in this situation wouldn't?
- What do you already believe that would be news to the model?
- Where would you push back if someone gave you textbook advice?

State these explicitly, then add: "Where your recommendation would normally differ, note how your answer changes given my specific priors."

---

## The Meta-Diagnostic: Read the Answer Backwards

When the above flowchart doesn't immediately place you, use the meta-diagnostic: **read the answer as a description of the conditions the model thought it had**, and ask whether those match the conditions you actually have.

For example:

**Model says:** "The best approach for most teams is to start with a monolith and extract services as you find natural seams."

**Meta-diagnostic reading:** The model thinks you are a new team, starting from scratch, without strong microservices expertise, without specific scalability requirements already identified.

**Your actual conditions:** You're not a new team. You have existing services. You have K8s expertise. You have specific performance requirements from day one.

**The gap:** Three conditions the model assumed that aren't true. Those are the conditions to add in the re-prompt.

---

## Diagnostic Checklist

Use this checklist when a prompt fails:

```
GENERIC ANSWER?
□ Could this answer apply to anyone in my field?
□ Did I add lots of text but still get a generic answer?
  → If both yes: Mistake 1 (detail ≠ conditions) or Mistake 3 (keyword prompt)

WRONG FOR MY CASE?
□ Would this advice work for a different type of person/company/situation?
□ Is there a version of me for whom this is correct?
  → If yes: Mistake 2 (need a conditional tree, not a verdict)

INCONSISTENT?
□ Have I gotten contradictory answers to similar prompts?
□ Did a small prompt change flip the answer?
  → If yes: Mistake 2 (hidden conditional structure — map the tree)

VAGUE/HEDGY?
□ Does the response have more than 2-3 "it depends" phrases?
□ Does the model give me tradeoffs when I want a recommendation?
  → If yes: Mistake 4 (missing objective function)

RIGHT PRINCIPLE, WRONG TIMING?
□ Is this advice correct but for a different phase or stage?
□ Would this be good advice 6 months from now, but not today?
  → If yes: Mistake 5 (temporal conditions missing)

CORRECT BUT GENERIC TO MY SITUATION?
□ Is the advice technically correct but doesn't account for my constraints?
□ Does it feel like advice for the average person, not me specifically?
  → If yes: Mistake 6 (misaligned priors)
```

---

## Example Diagnosis

**Situation:** You ask Claude "How should I approach hiring my first 5 engineers?" You get a thorough response covering culture fit, technical assessment, diversity goals, and employer brand.
<div class="callout-key">

<strong>Key Point:</strong> **Situation:** You ask Claude "How should I approach hiring my first 5 engineers?" You get a thorough response covering culture fit, technical assessment, diversity goals, and employer brand.

</div>


**Diagnosis walkthrough:**

1. Is it generic? Yes — it could apply to any startup. → Branch 1.
2. Did you write a lot of text and still get this? Let's say no, the prompt was short. → Mistake 3 (keyword prompt), not Mistake 1.
3. But wait — is it also wrong for your moment? Yes — you need to hire in 4 weeks for a specific technical problem, not build a long-term hiring system. → Also Mistake 5 (temporal conditions missing).

**Diagnosis:** Two mistakes operating simultaneously. The prompt is a keyword query (Mistake 3) and lacks temporal conditions (Mistake 5).

**Fix:**
```
Current state: I have a working product, 3 months of runway, and need
to hire 2 backend engineers in 4 weeks. I cannot afford a long hiring
process. Technical requirement: Python/FastAPI, prior production API work.

My objective: fill the roles in 4 weeks. I accept higher risk of cultural
mismatch in exchange for speed. I'll revisit process for later hires.

Given this, what's the fastest path to 2 qualified hires in 4 weeks —
not the best long-term hiring process?
```

---

## Connections

- **Builds on:** Guide 1 (the six mistakes), Module 2 (switch variables), Module 3 (condition stack)
- **Applied in:** `notebooks/01_bad_prompt_clinic.ipynb` — live diagnosis of 6 real broken prompts
- **Leads to:** Module 7 (production patterns — automating condition injection so these mistakes can't happen in pipelines)

---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Key Insight" and why it matters in practice.

2. Given a real-world scenario involving diagnostic framework: when ai gives bad answers, what would be your first three steps to apply the techniques from this guide?

</div>
