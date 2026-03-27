# Exercise: Build a Probability Mistake Taxonomy for Your Domain

## Overview

Every bad AI answer is a predictable output of a specific probability mistake. This exercise makes that taxonomy concrete: you will work through two fully-diagnosed examples, then apply the same diagnostic pattern to four prompts from your own domain.

The goal is not to memorize the six mistakes. It is to build the reflex of diagnosing *why* a bad prompt is bad before trying to fix it — because the fix depends on the root cause.

**Time estimate:** 40–50 minutes

**Prerequisites:** `guides/01_six_mistakes_guide.md`

---

## The Six Mistakes: Quick Reference

| # | Mistake | Diagnostic signal | One-line fix |
|---|---------|------------------|--------------|
| 1 | More detail ≠ more conditions | Added text, same wrong answer | Separate information from discriminating evidence |
| 2 | One answer instead of conditional tree | Advice wrong for your specific branch | Ask for the tree, then identify your branch |
| 3 | Treating AI like a search engine | Keyword-style prompt, document-like response | Replace keywords with a state → target → constraints inference spec |
| 4 | Missing objective function | Advice optimizes for the wrong goal | State what you are maximizing, what you are sacrificing, what is off-limits |
| 5 | Missing temporal conditions | Advice correct for a different phase | Specify current phase, relevant deadline, what would trigger a different phase |
| 6 | Assuming shared priors | Standard answer for a non-standard situation | Make your priors explicit before asking for advice |

---

## Taxonomy Template

For each mistake, fill in this template. There are two worked examples below, then four blank templates for your own domain.

```
MISTAKE TYPE: [number and name]

HOW IT MANIFESTS IN [YOUR DOMAIN]:
[What does this mistake look like in practice? How would you recognize it in a prompt someone from your field wrote?]

EXAMPLE BAD PROMPT:
[A realistic prompt someone in this domain would actually write — not a cartoon bad example]

BAYESIAN DIAGNOSIS:
[What prior does the model default to? What posterior is the prompter actually trying to reach? What is the gap?]

CORRECTED PROMPT:
[The fixed version — only the conditions that change the answer, not padding]

WHAT CHANGES:
[One sentence: how the output distribution shifts with the correction]
```

---

## Worked Example 1: Mistake 1 — "More Detail ≠ More Conditions"

Domain: Investment management

### How It Manifests

A portfolio manager asks a detailed question — multiple paragraphs of background, extensive financial history, context about the client relationship — and gets back a generic answer about diversification. They added words but not conditions. The detail describes the situation; it does not constrain which recommendation is correct.

### Example Bad Prompt

> "My client is a 52-year-old executive at a publicly traded company. She has been with the firm for 18 years and has accumulated substantial equity compensation. She received her MBA from a top business school, has been through two market cycles, and understands risk. Her husband is a physician and they have three children in college. She recently mentioned concerns about her portfolio concentration. She has a brokerage account with us and a separate 401(k). What should I do about her portfolio concentration risk?"

### Bayesian Diagnosis

The prior the model defaults to: a moderately affluent, moderately risk-tolerant investor with a diversification problem — recommends diversifying into index funds, selling some concentrated stock, using a systematic divestiture plan. This is the "average case" portfolio concentration answer.

The posterior the prompter actually needs: advice calibrated to (a) whether the concentrated position is restricted stock with trading windows and Rule 10b5-1 constraints, (b) the cost basis of the position and the tax implications of selling, (c) the objective — is she trying to minimize concentrated risk, defer capital gains, satisfy a beneficiary, or hedge without selling?

What the wall of text adds: biographical color and relationship context. What it does not add: the conditions that determine which of the three or four very different strategies is correct. The model has no basis for distinguishing a $2M unrestricted position from a $2M position with a 90-day blackout and a $200k cost basis.

### Corrected Prompt

> "Evaluating portfolio concentration strategy for a client with concentrated equity compensation.
>
> Conditions:
> - Position: 22,000 shares of employer stock, current price $87/share = ~$1.9M
> - Cost basis: $4.20/share (acquired through ESPP over 12 years)
> - Trading status: Insider — subject to Rule 10b5-1 requirements, currently in blackout, next open window opens in 6 weeks
> - Tax situation: MFJ, household income $620k, capital gains rate 23.8% federal (NIIT applies)
> - Objective: reduce concentration below 40% of portfolio over 18 months without triggering AMT event in any single year; she is risk-averse about tax surprises
> - Constraints: cannot use a forward variable prepaid (company policy); 401(k) is already fully diversified, not part of this analysis
>
> What strategy should she use to reduce concentration given these constraints?"

### What Changes

The output shifts from a generic "diversify your concentrated stock" recommendation to a specific analysis of 10b5-1 plan design (automated sales over the window), the gain harvesting schedule constrained by the AMT calculation, and whether charitable remainder trusts or exchange funds are worth modeling given the 23.8% rate and 18-month timeline.

---

## Worked Example 2: Mistake 3 — "Treating AI Like a Search Engine"

Domain: Infrastructure engineering

### How It Manifests

An engineer types a noun phrase — a technology keyword plus a problem keyword — and gets back a structured article summarizing what the technology is and common use cases. They wanted reasoning about their specific situation; they got documentation retrieval. The prompt looked like a search query, and the model completed it like a search result.

### Example Bad Prompt

> "Kubernetes pod eviction memory pressure troubleshooting"

### Bayesian Diagnosis

The prior the model defaults to: "this prompt is the beginning of a technical article or tutorial about Kubernetes pod eviction due to memory pressure." The most probable continuation is a structured explanation: what pod eviction is, what memory pressure means, how to check node conditions with `kubectl describe node`, how to set resource requests and limits.

The posterior the prompter actually needs: a diagnosis of why their specific pods are being evicted and what to change. But the prompt contains no current state, no target state, no constraints. There is nothing to condition on except the topic keywords.

### Corrected Prompt

> "Diagnosing pod eviction on a Kubernetes cluster — need to identify root cause and fix.
>
> Current state:
> - 3 pods in the `payments` namespace have been evicted 4 times in the last 6 hours with reason `Evicted: The node was low on resource: memory`
> - Node memory: 16Gi total, currently reporting 14.2Gi used across all pods
> - The evicted pods have no `resources.limits` or `resources.requests` set
> - Other pods on the same node (with resource limits set) are not being evicted
> - Cluster: EKS 1.28, node type m5.xlarge
>
> Target: stop the eviction cycle without rescheduling the payments pods to dedicated nodes (cost constraint)
>
> What is causing these pods to be evicted before other pods, and what specific changes should I make to stop it? Show me the kubectl commands or YAML changes needed."

### What Changes

The corrected prompt produces a specific diagnosis: pods without resource requests are treated as BestEffort QoS class, which is the first evicted under memory pressure; the fix is to add resource requests and limits to move them to Burstable or Guaranteed QoS. The model provides the exact YAML diff and explains why the un-limited pods were targeted first. The keyword prompt produces a tutorial about memory pressure in general.

---

## Your Turn: Four Templates to Fill

Apply the taxonomy to prompts from your own domain. For each template:
- Pick a mistake type you have seen (or made) in your actual work
- Use a realistic bad prompt — not a cartoonish example
- Write the Bayesian diagnosis precisely: what prior is the model defaulting to, and what posterior are you actually trying to reach?

---

### Your Template 1

**Domain:** _______________________________________________

```
MISTAKE TYPE: [Pick one: 1 / 2 / 3 / 4 / 5 / 6 — name it too]

HOW IT MANIFESTS IN THIS DOMAIN:


EXAMPLE BAD PROMPT:


BAYESIAN DIAGNOSIS:
  Default prior (what the model assumes):

  Target posterior (what you actually need):

  The gap (what conditions are missing or wrong):


CORRECTED PROMPT:


WHAT CHANGES:
```

---

### Your Template 2

**Domain:** _______________________________________________

```
MISTAKE TYPE:

HOW IT MANIFESTS IN THIS DOMAIN:


EXAMPLE BAD PROMPT:


BAYESIAN DIAGNOSIS:
  Default prior:

  Target posterior:

  The gap:


CORRECTED PROMPT:


WHAT CHANGES:
```

---

### Your Template 3

**Domain:** _______________________________________________

```
MISTAKE TYPE:

HOW IT MANIFESTS IN THIS DOMAIN:


EXAMPLE BAD PROMPT:


BAYESIAN DIAGNOSIS:
  Default prior:

  Target posterior:

  The gap:


CORRECTED PROMPT:


WHAT CHANGES:
```

---

### Your Template 4

**Domain:** _______________________________________________

```
MISTAKE TYPE:

HOW IT MANIFESTS IN THIS DOMAIN:


EXAMPLE BAD PROMPT:


BAYESIAN DIAGNOSIS:
  Default prior:

  Target posterior:

  The gap:


CORRECTED PROMPT:


WHAT CHANGES:
```

---

## Synthesis Questions

After completing all four templates, answer these:

### 1. Which mistake is hardest to diagnose?

Which of the six mistakes is easiest to confuse with a different mistake? Give an example of a prompt that looks like Mistake 4 (missing objective function) but is actually Mistake 6 (misaligned priors).

```
Your answer:
```

### 2. Domain-specific frequency

In your domain, which mistakes appear most often? Rank the top three for your field and explain why each is common there.

```
1. Most common in my domain:
   Why:

2. Second most common:
   Why:

3. Third most common:
   Why:
```

### 3. The relationship between mistakes

Mistakes 1, 4, and 6 are related. Explain the connection: how can a single bad prompt simultaneously exhibit all three mistakes?

```
Your answer:
```

### 4. The fastest diagnostic

If you had 10 seconds to diagnose a bad prompt before fixing it, which single question would you ask?

Write it here:

```
Your diagnostic question:
```

---

## Using the Taxonomy Going Forward

The taxonomy is a diagnostic tool, not a checklist. You do not need to identify every mistake in a bad prompt before fixing it. You need to identify the one mistake that, if corrected, would produce the biggest improvement in the output distribution.

The ranking of mistakes by leverage in most domains:

1. **Missing objective function (Mistake 4)** — changes what the model is trying to optimize; highest-leverage fix in most professional domains
2. **Missing temporal conditions (Mistake 5)** — often invisible; produces advice that is correct for the wrong phase
3. **One answer instead of tree (Mistake 2)** — forces the model to surface conditional structure it is hiding
4. **Misaligned priors (Mistake 6)** — hardest to detect because you have to know what the model's prior is before you can see the misalignment

More detail (Mistake 1) and keyword prompting (Mistake 3) are the most visible mistakes. They are also the easiest to fix and often have lower leverage than the others.

**Fix what matters most first.**

---

## Next Steps

- `notebooks/01_bad_prompt_clinic.ipynb` — run all six mistakes through the Claude API and observe the distribution shift live
- `guides/02_diagnostic_framework_guide.md` — a systematic diagnostic process for any bad prompt you encounter
- Module 7 (`guides/01_production_patterns_guide.md`) — how to build systems that prevent these mistakes at the infrastructure level, not just the prompt level
