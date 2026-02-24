# Reward Function Design for LLM Systems

## In Brief

The reward function is the MOST IMPORTANT design decision in prompt routing bandits. A poorly designed reward trains your system to produce confident hallucinations, verbose garbage, or technically-correct-but-useless responses. The key is to use composite rewards: one primary metric (does it solve the task?) plus multiple guardrails (is it factual? is it efficient? does it follow format?).

> 💡 **Key Insight:** Bad rewards produce bad behavior — but in subtle ways.

If you reward only user satisfaction, you train a system that tells users what they want to hear (even if it's wrong).
If you reward only brevity, you train a system that gives shallow, incomplete answers.
If you reward only "no follow-up questions," you train a system that makes confident guesses instead of asking for clarification.

The solution: **Primary metric + guardrails.** Optimize for task completion, but penalize hallucinations, format violations, and excessive cost.

## Bad Rewards and What They Produce

### Bad Reward 1: Thumbs-Up Only
**Design:**
```python
reward = 1 if user_clicked_thumbs_up else 0
```

**What it trains:**
- Confident-sounding answers (users like confidence)
- Detailed explanations (users think more = better)
- Agreeing with user assumptions (users like validation)
- **Hallucinations** (confident lies get thumbs-up if they sound plausible)

**Real example in commodity trading:**
User: "Will crude oil hit $100 by March?"
Bad prompt: "Yes, crude will likely reach $100 by March based on current supply constraints and geopolitical tensions."
(User loves this answer. Gives thumbs-up. Prompt gets reinforced. System learns to make bold predictions without evidence.)

### Bad Reward 2: Brevity Only
**Design:**
```python
reward = 1 / len(response.split())  # Shorter = better
```

**What it trains:**
- One-word answers
- Missing context
- Incomplete analysis
- **Useless output** (technically correct but not helpful)

**Real example:**
User: "Analyze the latest EIA inventory report."
Bad prompt response: "Bearish."
(Short! High reward! But useless for actual trading decisions.)

### Bad Reward 3: No Follow-Up Questions
**Design:**
```python
reward = 1 if no_followup_question_needed else 0
```

**What it trains:**
- Guessing when information is ambiguous
- Assuming context that wasn't provided
- **Confident nonsense** (never admit uncertainty)

**Real example:**
User: "What's the corn outlook?"
Bad prompt response: "December corn futures will rise 8% by harvest."
(System guessed user wanted Dec futures, guessed harvest timeline, made up 8% number. No follow-up needed! High reward!)

### Bad Reward 4: Output Length
**Design:**
```python
reward = min(len(response.split()) / 200, 1.0)  # Reward up to 200 words
```

**What it trains:**
- Verbose rambling
- Repetition to hit word count
- **Token waste** (your LLM bill goes up)

**Real example:**
User: "Latest oil inventories?"
Bad prompt response: "The latest oil inventories, which are released weekly by the Energy Information Administration, commonly known as the EIA, show that crude oil inventories, which represent the total amount of crude oil stored in the United States, have changed. Understanding these inventory levels is crucial for traders because inventories reflect the balance between supply and demand in the market..."
(200 words. Still hasn't answered the question.)

## The Operator Trick: Primary Metric + Guardrails

The solution is **composite reward design:**

```python
reward = primary_metric + Σ(guardrail_penalties)
```

- **Primary metric:** Does it accomplish the task? (0 to 1)
- **Guardrails:** Penalties for hallucinations, format violations, cost overruns

This ensures you optimize for the right thing while preventing catastrophic failures.

## Primary Metrics for Commodity Trading LLM Systems

### Metric 1: Task Completion Rate
**Question:** Did the response answer what was asked?

**How to measure:**
- Automated: Check if required fields are populated (for structured tasks)
- LLM-as-judge: Ask another LLM "Does this response answer the question?" (for unstructured)
- Human labels: Sample 5% of responses for human evaluation

**Example:**
```python
def task_completion_score(query, response):
    """Check if response addresses the query."""
    # Extract task type from query
    if "inventory" in query.lower() or "stockpile" in query.lower():
        # For inventory queries, check if numbers are present
        has_number = bool(re.search(r'\d+\.?\d*\s*(million|billion|MMbbl|MT)', response))
        has_date = bool(re.search(r'\d{4}-\d{2}-\d{2}|\w+ \d{1,2}', response))
        return 1.0 if (has_number and has_date) else 0.3

    # Use LLM-as-judge for complex queries
    judge_prompt = f"""Does this response answer the question?
    Question: {query}
    Response: {response}
    Answer with just "Yes" or "No"."""
    judgment = llm_call(judge_prompt, model="gpt-4o-mini")  # Use cheap model
    return 1.0 if "yes" in judgment.lower() else 0.0
```

### Metric 2: Extraction Accuracy
**Question:** Did it correctly parse the data from the source document?

**How to measure:**
- Compare extracted values to ground truth (for reports with known values)
- Schema compliance check (did it fill the right fields with the right types?)
- Spot-check samples manually

**Example:**
```python
def extraction_accuracy(response, ground_truth):
    """Compare extracted data to known values."""
    try:
        extracted = json.loads(response)
        score = 0.0
        fields = ['commodity', 'date', 'value', 'unit']

        for field in fields:
            if field in extracted and field in ground_truth:
                if extracted[field] == ground_truth[field]:
                    score += 0.25  # Each field worth 25%

        return score
    except json.JSONDecodeError:
        return 0.0  # Not even valid JSON
```

### Metric 3: Signal Quality
**Question:** Was the trading signal actionable and directionally correct?

**How to measure:**
- Backtesting: Track if signals would have been profitable (delayed, but ground truth)
- Consistency: Does the signal match the evidence provided?
- Conviction calibration: Are "high conviction" signals actually more accurate?

**Example:**
```python
def signal_quality(response, actual_price_change):
    """Evaluate trading signal quality after observing outcome."""
    # Extract signal (BUY/SELL/HOLD) and conviction
    signal = extract_signal(response)  # Returns: ("BUY", "High")

    direction, conviction = signal
    correct_direction = (
        (direction == "BUY" and actual_price_change > 0) or
        (direction == "SELL" and actual_price_change < 0) or
        (direction == "HOLD" and abs(actual_price_change) < 0.02)
    )

    # Base reward for correct direction
    reward = 1.0 if correct_direction else 0.0

    # Bonus for high conviction when correct
    if correct_direction and conviction == "High":
        reward += 0.3

    # Penalty for high conviction when wrong
    if not correct_direction and conviction == "High":
        reward -= 0.5

    return max(reward, 0.0)
```

### Metric 4: Research Completeness
**Question:** Did it cover all relevant dimensions of the analysis?

**How to measure:**
- Checklist: For commodity fundamentals, did it cover supply, demand, inventory, price action?
- Source diversity: Did it use multiple data sources?
- Uncertainty acknowledgment: Did it flag data gaps?

**Example:**
```python
def research_completeness(response, commodity):
    """Check if fundamental analysis covers key dimensions."""
    dimensions = {
        'supply': ['production', 'output', 'drilling', 'planting'],
        'demand': ['consumption', 'imports', 'usage', 'offtake'],
        'inventory': ['stocks', 'storage', 'inventories', 'reserves'],
        'price': ['futures', 'spot', 'spread', 'premium'],
    }

    score = 0.0
    response_lower = response.lower()

    for dimension, keywords in dimensions.items():
        if any(kw in response_lower for kw in keywords):
            score += 0.25

    return score
```

## Guardrails

### Guardrail 1: Hallucination Detection
**Most critical for commodity data** where wrong numbers lead to bad trades.

**How to detect:**
- Retrieval verification: Is each claim supported by a retrieved document?
- Fact-checking: Cross-reference numbers against known sources (EIA API, USDA database)
- Uncertainty markers: Penalize responses that state unverified numbers as fact

**Example:**
```python
def hallucination_penalty(response, retrieved_docs):
    """Penalize claims not supported by sources."""
    # Extract factual claims (numbers, dates, specific events)
    claims = extract_factual_claims(response)

    unsupported_claims = 0
    for claim in claims:
        # Check if claim appears in any retrieved document
        if not any(claim_in_doc(claim, doc) for doc in retrieved_docs):
            unsupported_claims += 1

    # Penalty scales with number of unsupported claims
    penalty = -0.3 * unsupported_claims
    return penalty
```

### Guardrail 2: Format Compliance
**Ensures structured outputs can be parsed programmatically.**

**Example:**
```python
def format_compliance_score(response, expected_format):
    """Check if response follows required format."""
    if expected_format == "json":
        try:
            data = json.loads(response)
            return 0.0  # No penalty
        except:
            return -0.5  # Major penalty for invalid JSON

    elif expected_format == "signal":
        # Check for required fields: Signal, Conviction, Rationale
        has_signal = any(s in response for s in ["BUY", "SELL", "HOLD"])
        has_conviction = any(c in response for c in ["High", "Medium", "Low"])
        has_rationale = len(response.split()) > 20

        if has_signal and has_conviction and has_rationale:
            return 0.0
        else:
            return -0.3

    return 0.0
```

### Guardrail 3: Cost and Latency Budgets
**Prevents the system from learning to use expensive, slow prompts.**

**Example:**
```python
def cost_latency_penalty(tokens_used, latency_seconds, budget_tokens=1000, budget_seconds=5):
    """Penalize excessive cost or latency."""
    cost_penalty = 0.0
    if tokens_used > budget_tokens:
        cost_penalty = -0.2 * (tokens_used / budget_tokens - 1)

    latency_penalty = 0.0
    if latency_seconds > budget_seconds:
        latency_penalty = -0.1 * (latency_seconds / budget_seconds - 1)

    return cost_penalty + latency_penalty
```

### Guardrail 4: Citation/Source Verification Rate
**For RAG systems, ensures responses are grounded in retrieved documents.**

**Example:**
```python
def citation_score(response, retrieved_docs):
    """Reward responses that cite sources."""
    # Check if response includes source citations
    has_citations = bool(re.search(r'(Source:|According to|EIA|USDA)', response))

    if not has_citations:
        return -0.2  # Penalty for unsourced claims

    # Bonus: verify citations are actually in retrieved docs
    citations = extract_citations(response)
    valid_citations = sum(1 for c in citations if verify_citation(c, retrieved_docs))

    if len(citations) > 0:
        citation_rate = valid_citations / len(citations)
        return 0.1 * citation_rate
    return 0.0
```

## Composite Reward: Putting It Together

Here's a production-grade reward function for a commodity research assistant:

```python
def compute_reward(query, response, retrieved_docs, ground_truth=None, actual_outcome=None):
    """
    Composite reward for commodity analysis LLM.

    Primary metrics (0 to 1):
    - Task completion: Did it answer the question?
    - Extraction accuracy: Are the numbers right? (if ground truth available)
    - Signal quality: Was the signal correct? (if outcome available)

    Guardrails (penalties):
    - Hallucination: Unsupported claims
    - Format: Non-compliance with expected structure
    - Cost: Excessive token usage
    - Citation: Missing or invalid sources
    """

    # Primary metric: task completion (required)
    primary = task_completion_score(query, response)

    # Secondary metrics (if applicable)
    if ground_truth:
        primary += 0.3 * extraction_accuracy(response, ground_truth)

    if actual_outcome:
        primary += 0.3 * signal_quality(response, actual_outcome)

    # Normalize primary to [0, 1]
    primary = min(primary, 1.0)

    # Guardrails (penalties)
    hallucination_pen = hallucination_penalty(response, retrieved_docs)
    format_pen = format_compliance_score(response, expected_format="auto")
    citation_pen = citation_score(response, retrieved_docs)

    # Total reward
    reward = primary + hallucination_pen + format_pen + citation_pen

    return max(reward, 0.0)  # Floor at 0
```

## Commodity Example: Crude Oil Analysis Bot

**Scenario:** Building an LLM system that analyzes EIA weekly petroleum reports and generates trading insights.

**Prompts being tested:**
1. Generic analysis prompt
2. Evidence-only prompt (RAG-safe)
3. Quantitative-first prompt (stats heavy)
4. Trading signal prompt (actionable recommendations)

**Reward design:**

```python
def crude_oil_bot_reward(query, response, eia_report, price_outcome=None):
    """Reward function for crude oil analysis bot."""

    # Primary: Did it extract the key inventory number?
    inventory_extracted = 0.0
    if re.search(r'\d+\.?\d*\s*million barrels?', response):
        inventory_extracted = 0.5

    # Check if number is correct (if we have ground truth)
    if eia_report and 'crude_stocks' in eia_report:
        true_value = eia_report['crude_stocks']
        extracted_value = extract_number(response)
        if extracted_value and abs(extracted_value - true_value) / true_value < 0.01:
            inventory_extracted = 1.0  # Correct within 1%

    # Guardrail: Hallucination check
    hallucination_pen = 0.0
    if not any(source in response for source in ['EIA', 'Energy Information Administration']):
        hallucination_pen = -0.3  # No source cited

    # Guardrail: Signal quality (if we observe outcome)
    signal_reward = 0.0
    if price_outcome:
        signal_reward = signal_quality(response, price_outcome)

    # Composite
    reward = 0.5 * inventory_extracted + 0.3 * signal_reward + hallucination_pen

    return max(reward, 0.0)
```

**What this rewards:**
- Correct data extraction (50% weight)
- Accurate trading signals (30% weight)
- Proper source attribution (penalty if missing)

**What this penalizes:**
- Wrong inventory numbers
- Unsourced claims
- Signals that would have lost money

## Visual Explanation

```
┌──────────────────────────────────────────────────────────────────┐
│               REWARD FUNCTION ARCHITECTURE                       │
└──────────────────────────────────────────────────────────────────┘

                           COMPOSITE REWARD
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
         PRIMARY METRIC                          GUARDRAILS
         (maximize)                              (penalties)
              │                                       │
    ┌─────────┴─────────┐                 ┌──────────┼──────────┐
    │                   │                 │          │          │
Task            Research          Hallucination  Format   Cost
Completion      Completeness      Detection      Check    Budget
    │                   │                 │          │          │
    ▼                   ▼                 ▼          ▼          ▼
Did it answer?    Cover supply/    Unsourced   Invalid   Token
                  demand/price?     claims?     JSON?     limit?
    │                   │                 │          │          │
    ▼                   ▼                 ▼          ▼          ▼
  [0, 1]            [0, 1]            [-0.5, 0]  [-0.3, 0]  [-0.2, 0]


FINAL REWARD = Primary + Σ(Guardrails)
             = max(0, task_completion + research_score
                      - hallucination_penalty
                      - format_penalty
                      - cost_penalty)
```

## Intuitive Explanation

Think of reward design like hiring a junior analyst.

You don't just evaluate them on "Did the client smile?" (user satisfaction). You evaluate:
- **Primary goal:** Did they answer the research question correctly?
- **Guardrail 1:** Did they make up any numbers? (hallucination penalty)
- **Guardrail 2:** Did they format the report properly? (format compliance)
- **Guardrail 3:** Did they spend 10 hours on a 1-hour task? (cost penalty)

Same logic applies to prompt routing. Reward what you actually care about, and penalize behaviors that lead to long-term problems.

## Common Pitfalls

### Pitfall 1: "I'll Reward Only User Satisfaction"
If users give thumbs-up to confident hallucinations, your system will learn to hallucinate confidently.

**Fix:** Add hallucination detection as a guardrail with heavy penalty.

### Pitfall 2: "I'll Reward Perfect Accuracy Only"
Binary rewards (1 for perfect, 0 otherwise) slow learning. The bandit can't distinguish between "close" and "terrible."

**Fix:** Use continuous rewards. 0.7 for mostly correct, 0.4 for partially correct, 0.0 for wrong.

### Pitfall 3: "I'll Just Use LLM-as-Judge for Everything"
LLM judges are expensive, slow, and can have biases (they prefer responses from the same model family).

**Fix:** Use automated checks where possible (format, extraction accuracy, citation presence). Use LLM-as-judge for nuanced quality assessment only.

### Pitfall 4: "One Reward Function for All Tasks"
A reward designed for data extraction will fail for trading signal generation.

**Fix:** Task-specific rewards, or contextual rewards that adapt based on query type.

## Connections

**Builds on:**
- **Module 2 (Thompson Sampling):** Reward = success/failure for Beta updates
- **Module 5 (Commodity Trading):** Same reward design principles (primary metric + risk guardrails)

**Leads to:**
- **Guide 03:** Contextual routing uses reward as the optimization target
- **Guide 04:** Case studies show reward design in real commodity systems

## Practice Problems

### Problem 1: Design a Reward Function
You're building a commodity report summarizer. Design a composite reward function with:
- One primary metric
- Three guardrails
Specify how each component is calculated.

### Problem 2: Identify the Failure Mode
A prompt routing system uses this reward:
```python
reward = 1 if len(response.split()) > 100 else 0
```
What failure mode will this train? What would you change?

### Problem 3: Hallucination Detection
Implement a function that detects if a response contains numerical claims not present in the source documents. Test it on:
- Response: "EIA reports 450 million barrels in storage."
- Source: "Storage levels increased by 2 million barrels."

Should this be flagged as hallucination? Why or why not?

### Problem 4: Reward Tuning
You have these reward weights:
```python
reward = 0.8 * task_completion + 0.2 * format_compliance - 0.1 * hallucination_penalty
```
The system is producing well-formatted hallucinations. Which weight should you adjust and why?
