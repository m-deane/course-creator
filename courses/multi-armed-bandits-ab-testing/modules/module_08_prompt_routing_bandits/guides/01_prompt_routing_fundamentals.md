# Prompt Routing Fundamentals

> **Reading time:** ~20 min | **Module:** 08 — Prompt Routing Bandits | **Prerequisites:** Module 7


## In Brief


<div class="callout-key">

**Key Concept Summary:** Prompt routing bandits treat each prompt template as an "arm" in a multi-armed bandit problem. Instead of manually testing prompts, the system learns which prompt works best for each request type b...

</div>

Prompt routing bandits treat each prompt template as an "arm" in a multi-armed bandit problem. Instead of manually testing prompts, the system learns which prompt works best for each request type by balancing exploration (trying different prompts) and exploitation (using the best-known prompt). This eliminates the "bad prompt tax" — the cost of using suboptimal prompts while you manually iterate.

## Key Insight

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>


**Prompts are a perfect bandit problem** because:
1. You choose one prompt at a time (sequential decision-making)
2. You get fast feedback (LLM response quality is measurable within seconds)
3. The environment changes (different tasks need different prompts; user expectations evolve)
4. Exploration is expensive (every bad prompt costs tokens, time, and user trust)

The traditional approach — manually A/B test prompts for weeks — is exactly the inefficiency that bandits were designed to eliminate.

## The "Bad Prompt Tax"

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


Every production LLM system pays a hidden tax:

> ⚠️ **Warning: The Bad Prompt Tax**

**Time tax:** Weeks spent manually iterating on prompts instead of shipping features
**Cost tax:** Thousands of dollars in wasted tokens on prompts that produce verbose or low-quality outputs
**Trust tax:** User frustration from inconsistent quality, hallucinations, or off-target responses
**Opportunity tax:** Lost insights because you stuck with "good enough" instead of discovering "excellent"

A commodity research assistant using a single "best prompt for everything" approach might:
- Hallucinate inventory numbers when data is sparse (because the prompt doesn't say "say so if uncertain")
- Generate verbose analysis when you need concise data extraction (wrong prompt for the task)
- Miss trading signals because the prompt optimizes for completeness, not actionability

**Prompt routing bandits eliminate this tax** by learning which prompt to use for each request — while the system is running.

## Visual Explanation

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROMPT ROUTING ARCHITECTURE                  │
└─────────────────────────────────────────────────────────────────┘

User Request: "What are latest EIA crude oil inventories?"
      │
      ▼
┌─────────────────┐
│  Feature         │  → Task: extraction
│  Extraction      │  → Commodity: energy
└────────┬─────────┘  → Data: high availability
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BANDIT ALGORITHM                            │
│  (Thompson Sampling or LinUCB)                                   │
│                                                                  │
│  Prompt Arms:                                    Selection Prob  │
│  1. Structured Extraction ................... 15%               │
│  2. Evidence-Only (RAG-safe) ............... 65% ← SELECTED     │
│  3. Clarify-First ....................... 5%                    │
│  4. Quantitative Analysis ................... 10%               │
│  5. Trading Signal .......................... 5%                │
└────────┬─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  SELECTED PROMPT (Evidence-Only):                                │
│                                                                  │
│  "Extract the latest crude oil inventory data from the          │
│   retrieved sources. Use ONLY the data present in the           │
│   provided context. If inventory data is not in the sources,    │
│   respond: 'No inventory data found in retrieved documents.'    │
│   Present data in this format: [Date] [Value] [Source]"         │
└────────┬─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│   LLM Call      │  → GPT-4, Claude, etc.
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  RESPONSE:                                                       │
│  "EIA Report 2024-02-07: 439.5 million barrels                  │
│   Source: EIA Weekly Petroleum Status Report"                   │
└────────┬─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  REWARD CALCULATION:                                             │
│  • Task completion: 1.0 (data extracted)                         │
│  • Factual accuracy: 1.0 (correct number, verifiable)           │
│  • Hallucination flag: 0 (no fabricated data)                   │
│  • Format compliance: 1.0 (matched template)                     │
│  Composite Reward: 0.7 * 1.0 + 0.3 * 1.0 - 0.5 * 0 = 1.0        │
└────────┬─────────────────────────────────────────────────────────┘
         │
         ▼
   UPDATE BANDIT
   (Evidence-Only arm gets positive update)
```

## Formal Definition

A **prompt routing bandit** is a multi-armed bandit problem where:

- **Arms K:** Set of prompt templates {p₁, p₂, ..., pₖ}
- **Context xₜ:** Features of request at time t (task type, commodity, data availability)
- **Action aₜ:** Selected prompt pⱼ ∈ K
- **Reward rₜ:** Quality score of LLM response using prompt pⱼ for request xₜ
- **Policy π:** Mapping from context to prompt selection probabilities

**Objective:** Maximize cumulative reward over T requests:

```
max Σᵀₜ₌₁ E[rₜ | xₜ, aₜ = π(xₜ)]
```

This is equivalent to **minimizing regret** — the difference between the reward you got and the reward you would have gotten with the optimal prompt for each request.

## What Counts as a "Prompt Arm"?

Each arm is a complete prompt template designed for a specific purpose. For commodity trading LLM systems, typical prompt arms include:

### 1. Structured Extraction
**Purpose:** Pull specific data points into a predefined schema
**When it works:** Data-heavy queries with clear structure (EIA reports, USDA tables)
**Template:**
```
Extract the following data from the provided text into JSON format:
{
  "commodity": "",
  "date": "",
  "value": "",
  "unit": "",
  "source": ""
}
Only include data explicitly stated in the text. Leave fields empty if not found.
```

### 2. Evidence-Only (RAG-Safe)
**Purpose:** Prevent hallucinations by requiring citations
**When it works:** When retrieval context is available and accuracy is critical
**Template:**
```
Use ONLY information from the retrieved documents below to answer the question.
If the answer is not in the provided sources, respond: "Not found in sources."
Always cite the source document for each claim.

Retrieved Documents:
{context}

Question: {query}
```

### 3. Clarify-First
**Purpose:** Handle ambiguous requests by asking for specifics
**When it works:** Vague queries where assumptions would be dangerous
**Template:**
```
The user asked: {query}

This query is ambiguous. Ask ONE clarifying question about:
- Which commodity (if multiple are plausible)
- What time period
- What specific metric they want

Do not attempt to answer until they clarify.
```

### 4. Quantitative Analysis
**Purpose:** Generate numerical insights with statistical rigor
**When it works:** Fundamental analysis, trend identification, comparative studies
**Template:**
```
Analyze the provided commodity data and provide:
1. Numerical summary statistics (mean, median, range, volatility)
2. Trend analysis with percentage changes
3. Confidence intervals where applicable
4. Comparison to historical averages

Present all numbers with appropriate precision and units.
Do not make qualitative claims without quantitative support.
```

### 5. Trading Signal
**Purpose:** Produce actionable buy/sell/hold recommendations
**When it works:** When user needs decision support, not just information
**Template:**
```
Based on the commodity data and analysis, generate a trading signal:

Signal: [BUY | SELL | HOLD]
Conviction: [High | Medium | Low]
Rationale: [2-3 bullet points with specific data support]
Risk Factors: [Key downside scenarios]

Do not hedge. Provide a clear directional signal with reasoning.
```

### 6. Scenario Analysis
**Purpose:** Explore bull/base/bear cases with probabilities
**When it works:** Risk assessment, portfolio planning, uncertainty quantification
**Template:**
```
Present three scenarios for the commodity outlook:

BULL CASE (probability: __%)
- Key drivers: [list]
- Price target: [specific level]

BASE CASE (probability: __%)
- Key drivers: [list]
- Price target: [specific level]

BEAR CASE (probability: __%)
- Key drivers: [list]
- Price target: [specific level]

Probabilities must sum to 100%. Justify each scenario with data.
```

## Code Implementation

Here's a minimal prompt routing bandit in ~15 lines:


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np

class PromptRouter:
    def __init__(self, prompts):
        """Initialize with list of prompt templates."""
        self.prompts = prompts
        self.successes = np.ones(len(prompts))  # Beta prior α
        self.failures = np.ones(len(prompts))   # Beta prior β

    def select_prompt(self):
        """Thompson Sampling: sample from Beta posterior for each arm."""
        samples = [np.random.beta(s, f)
                   for s, f in zip(self.successes, self.failures)]
        return np.argmax(samples)

    def update(self, prompt_idx, reward):
        """Update Beta posterior based on reward (1=success, 0=failure)."""
        if reward > 0.7:  # Define success threshold
            self.successes[prompt_idx] += 1
        else:
            self.failures[prompt_idx] += 1
```

</div>
</div>

**Usage:**

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Define your prompt arms
prompts = [
    "Structured extraction prompt...",
    "Evidence-only prompt...",
    "Quantitative analysis prompt...",
]

router = PromptRouter(prompts)

# For each incoming request
for request in requests:
    # Select prompt using Thompson Sampling
    idx = router.select_prompt()
    prompt = prompts[idx]

    # Generate response
    response = llm_call(prompt, request)

    # Calculate reward (quality score)
    reward = evaluate_response(response, request)

    # Update bandit
    router.update(idx, reward)
```

</div>
</div>

## Intuitive Explanation

Think of prompt routing like a commodity trader learning which analysts to trust for different markets.

You have 5 analysts:
- **Analyst A** is great at reading inventory reports (structured data extraction)
- **Analyst B** only reports what's in the official data, never speculates (evidence-only)
- **Analyst C** asks clarifying questions before analyzing (clarify-first)
- **Analyst D** provides deep quantitative analysis (statistical modeling)
- **Analyst E** gives you buy/sell signals (trading recommendations)

When a new question comes in, you need to decide which analyst to ask. At first, you don't know who's best for what. So you:
1. **Explore:** Try different analysts for similar questions
2. **Learn:** Track which analyst gives the best answers for each type of question
3. **Exploit:** Gradually route questions to the best-performing analyst for that type

A prompt router does the same thing — but with prompt templates instead of human analysts.

## Common Pitfalls

### Pitfall 1: "I'll Just Pick the Best Prompt Manually"
**Why it fails:** What works today might not work tomorrow. User expectations change. LLM behavior drifts with model updates. Manual testing takes weeks while your system uses suboptimal prompts.

**The bandit solution:** Continuous learning adapts to changes automatically.

### Pitfall 2: "I Need 20 Different Prompts for Every Edge Case"
**Why it fails:** Too many arms means slow convergence. You won't get enough data per arm to learn reliably.

**The bandit solution:** Start with 5-8 well-designed prompts. Use contextual features (next guide) to handle variation within each prompt.

### Pitfall 3: "User Thumbs-Up = Good Reward"
**Why it fails:** Users often like confident, detailed answers — even when they're wrong. Optimizing for satisfaction alone trains hallucinations.

**The bandit solution:** Composite rewards with guardrails (see next guide). Primary metric + hallucination penalty + format compliance.

### Pitfall 4: "One Prompt to Rule Them All"
**Why it fails:** A single prompt that tries to handle extraction AND analysis AND signal generation will be mediocre at all three.

**The bandit solution:** Specialized prompts for specialized tasks. Let the router learn when to use which.

### Pitfall 5: "I'll Reward Based on Output Length"
**Why it fails:** You'll train verbose, rambling responses that look thorough but waste tokens and user attention.

**The bandit solution:** Reward task completion and accuracy, not word count.

## Connections

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


**Builds on:**
- **Module 0:** Explore-exploit tradeoff applies to prompt selection just like commodity allocation
- **Module 2:** Thompson Sampling is the engine for prompt routing
- **Module 3:** Contextual bandits let you route based on request features (task type, commodity)

**Leads to:**
- **Guide 02:** Reward function design — how to measure prompt quality without training hallucinations
- **Guide 03:** Contextual routing — adapting prompt selection based on request characteristics
- **Guide 04:** Real-world case studies of commodity research assistants

**Connects to other courses:**
- **GenAI for Commodities:** Prompt routing is part of production RAG system architecture
- **Bayesian Commodity Forecasting:** Same Bayesian updating logic, different application

## Practice Problems

### Problem 1: Prompt Arm Design
You're building a commodity report analyzer that processes:
- EIA weekly petroleum status reports (structured tables)
- USDA narrative crop condition reports (unstructured text)
- Analyst research notes (opinions and predictions)

Design 4 prompt arms for this system. For each arm, specify:
- What type of request it handles best
- The core instruction that differentiates it from other arms
- One failure mode it's designed to avoid

### Problem 2: Regret Calculation
You have 3 prompts with true quality scores:
- Prompt A: 0.7 quality (extraction)
- Prompt B: 0.9 quality (analysis)
- Prompt C: 0.5 quality (generic)

You serve 100 requests, all analysis tasks (so optimal is Prompt B every time).
Your router selected: 20 × A, 60 × B, 20 × C

Calculate the cumulative regret.

### Problem 3: When NOT to Use Prompt Routing
List 3 scenarios where prompt routing bandits are the wrong tool. For each, explain what approach you'd use instead.

### Problem 4: Implementation
Implement a prompt router that:
- Accepts 5 prompts as input
- Uses Thompson Sampling for selection
- Accepts rewards in [0, 1]
- Returns selection probabilities for each arm after every update

Test it with simulated rewards where Prompt 3 is best.


---

## Cross-References

<a class="link-card" href="./02_reward_design_llm.md">
  <div class="link-card-title">02 Reward Design Llm</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_reward_design_llm.md">
  <div class="link-card-title">02 Reward Design Llm — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_contextual_prompt_routing.md">
  <div class="link-card-title">03 Contextual Prompt Routing</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_contextual_prompt_routing.md">
  <div class="link-card-title">03 Contextual Prompt Routing — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./04_commodity_research_assistant.md">
  <div class="link-card-title">04 Commodity Research Assistant</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./04_commodity_research_assistant.md">
  <div class="link-card-title">04 Commodity Research Assistant — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

