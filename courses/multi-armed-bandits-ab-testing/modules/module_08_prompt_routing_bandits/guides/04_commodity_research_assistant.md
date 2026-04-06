# Commodity Research Assistant: Case Studies in Prompt Routing

> **Reading time:** ~20 min | **Module:** 08 — Prompt Routing Bandits | **Prerequisites:** Module 7


## Overview


<div class="callout-key">

**Key Concept Summary:** This guide presents three real-world case studies of prompt routing bandits applied to commodity trading systems. Each case study shows the problem, the bandit setup, the reward function, and the r...

</div>

This guide presents three real-world case studies of prompt routing bandits applied to commodity trading systems. Each case study shows the problem, the bandit setup, the reward function, and the results — demonstrating how adaptive prompt selection improves quality, reduces costs, and eliminates manual prompt engineering.

---

## Case Study 1: A Commodity Research Bot That Stopped Hallucinating

### The Problem

A commodity trading desk built an LLM-powered research assistant to answer analyst questions about energy, metals, and agriculture markets. They used a single "best prompt" optimized through manual testing:

```
You are an expert commodity analyst. Answer the user's question with detailed,
professional analysis. Be confident and thorough.
```

This prompt worked well for open-ended research questions ("Analyze the crude oil market outlook"). But it failed catastrophically on data extraction tasks:

**Query:** "What were last week's EIA crude oil inventories?"

**Response (with generic prompt):**
"Based on recent trends, EIA crude oil inventories likely increased by approximately 3-5 million barrels last week, driven by seasonal refinery maintenance and steady imports. This suggests a bearish near-term outlook for WTI crude futures."

**The problem:** All of this was hallucinated. The system didn't have last week's EIA report. The "confident and thorough" prompt trained the model to sound authoritative even when guessing.

**Cost of hallucinations:**
- Traders made decisions on false data
- Lost trust in the system ("I have to double-check everything anyway")
- Slower adoption (team went back to manual research)

### The Bandit Setup

**Prompt arms (5 total):**

1. **Generic Analysis** (the original prompt)
2. **Evidence-Only (RAG-safe)**
   ```
   Use ONLY information from the retrieved documents below.
   If the answer is not in the sources, respond: "Data not found in available sources."
   Always cite the source for each claim.
   ```

3. **Structured Extraction**
   ```
   Extract the requested data into this format:
   Date: [date]
   Value: [number with units]
   Source: [document name]

   Leave blank if not found.
   ```

4. **Quantitative Analysis**
   ```
   Provide numerical analysis with:
   1. Current level and change from prior period
   2. Historical comparison (percentile, z-score)
   3. Statistical significance of the change
   Avoid qualitative speculation. Focus on data.
   ```

5. **Clarify-First**
   ```
   If the question is ambiguous, ask ONE clarifying question.
   Options to clarify: commodity (if multiple), time period, specific metric.
   Do not attempt to answer until clarified.
   ```

**Context features:**
- Task type (extraction, analysis, signal, general)
- Commodity sector (energy, metals, agriculture)
- Data availability (high, medium, low) — from RAG retrieval score

**Reward function:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def research_bot_reward(query, response, retrieved_docs, ground_truth=None):
    # Primary: task completion
    completion = task_completion_score(query, response)

```

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>

```python
    # Guardrail 1: Hallucination detection (critical!)
    hallucination_penalty = 0.0
    claims = extract_factual_claims(response)
    for claim in claims:
        if not verify_claim_in_docs(claim, retrieved_docs):
            hallucination_penalty -= 0.5  # Heavy penalty

    # Guardrail 2: Source citation
    citation_bonus = 0.1 if has_source_citation(response) else 0.0

    # Composite
    reward = completion + hallucination_penalty + citation_bonus
    return max(reward, 0.0)
```

</div>
</div>

**Key insight:** Hallucination penalty is -0.5 per unsupported claim. This makes hallucinating very expensive for the bandit.

### The Result

After 2 weeks of production use (500 queries):

**For extraction tasks + high data availability:**
- Evidence-Only prompt selected 78% of the time
- Hallucination rate: 2% (down from 35% with generic prompt)
- User satisfaction: 4.2/5 (up from 2.8/5)

**For analysis tasks + low data availability:**
- Quantitative Analysis prompt selected 65% of the time
- Generic Analysis still used 20% (for truly open-ended questions)
- Users appreciated "data-driven" vs "speculative" analysis

**For ambiguous queries:**
- Clarify-First prompt selected 45% of the time
- Reduced wasted tokens on wrong answers by 30%

**Cost savings:**
- 25% reduction in LLM costs (shorter, more targeted responses)
- 40% reduction in "throwaway" responses (answers that were ignored)

### Implementation Code


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# Initialize router
prompts = [generic, evidence_only, structured, quantitative, clarify]
router = CommodityPromptRouter(prompts)

# Production loop
for query in incoming_queries:
    # Retrieve context
    docs = rag.retrieve(query, top_k=5)

    # Build context features
    context = build_context_vector(query, docs, user_profile, market_state)

    # Route to best prompt
    idx, prompt, _ = router.route(query, docs, user_profile, market_state)

    # Generate response
    response = llm_call(prompt, query, docs)

    # Calculate reward
    reward = research_bot_reward(query, response, docs)

    # Update bandit
    router.update(idx, context, reward)

    # Log for monitoring
    log_event({
        'query': query,
        'prompt_idx': idx,
        'reward': reward,
        'hallucination_detected': reward < 0.5,
    })
```

</div>
</div>

### Key Lessons

1. **Hallucination penalty must be severe** — otherwise the bandit learns to hallucinate confidently (users like confidence)
2. **RAG retrieval quality is a critical feature** — evidence-only prompts only work when retrieval succeeds
3. **Task-specific prompts outperform one-size-fits-all** — 25% cost reduction just from using shorter, targeted prompts
4. **Users prefer "I don't know" to confident guessing** — trust recovered once hallucination rate dropped

---

## Case Study 2: EIA Report Processor That Improved While Running

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


### The Problem

A hedge fund built a system to extract data from EIA's weekly petroleum status reports (released every Wednesday). The reports contain:
- Crude oil inventories
- Gasoline stocks
- Refinery utilization
- Production rates
- Imports/exports

They used a manually-tuned extraction prompt. But different sections of the report needed different approaches:
- **Tables:** Structured extraction (pull numbers into schema)
- **Narrative sections:** Summarization (e.g., "Key drivers this week")
- **Historical comparisons:** Trend analysis (e.g., "5-year average")
- **Anomalies:** Change detection (e.g., "Largest weekly build since 2020")

A single prompt was optimized for tables but terrible at narratives.

**Manual prompt engineering:**
- Spent 4 hours every week tweaking prompts after each EIA release
- Different analyst preferences (some wanted raw numbers, some wanted commentary)
- Report format changes occasionally (broke extraction rules)

### The Bandit Setup

**Prompt arms (4 total):**

1. **Structured Table Extraction**
   ```
   Extract data from the table into JSON:
   {
     "crude_oil_stocks": {"value": X, "unit": "million barrels", "change": Y},
     "gasoline_stocks": {...},
     ...
   }
   ```

2. **Narrative Summarization**
   ```
   Summarize the key takeaways from this section in 2-3 bullet points.
   Focus on what changed and why.
   ```

3. **Trend Analysis**
   ```
   Compare current values to:
   - Prior week (change and % change)
   - Year-ago (change and % change)
   - 5-year average (percentile)
   ```

4. **Change Detection**
   ```
   Identify any unusual changes (>2 standard deviations from recent average).
   For each anomaly, state: metric, magnitude, last time this happened.
   ```

**Context features:**
- Report section (inventory, production, refinery, trade)
- Data type (table, narrative, time series)
- User role (trader, analyst, portfolio manager)

**Reward function:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def eia_extraction_reward(section_type, response, ground_truth):
    # For tables: exact match on numbers
    if section_type == 'table':
        accuracy = extraction_accuracy(response, ground_truth)
        format_ok = is_valid_json(response)
        return 0.7 * accuracy + 0.3 * format_ok

    # For narratives: relevance and conciseness
    if section_type == 'narrative':
        relevance = llm_judge_relevance(response, ground_truth)
        brevity = 1.0 if len(response.split()) < 100 else 0.5
        return 0.8 * relevance + 0.2 * brevity

    # For trends: coverage of required comparisons
    if section_type == 'trend':
        has_wow = 'prior week' in response.lower()
        has_yoy = 'year-ago' in response.lower()
        has_avg = 'average' in response.lower()
        coverage = (has_wow + has_yoy + has_avg) / 3.0
        return coverage

    return 0.5  # Default
```

</div>
</div>

### The Result

After 12 weeks (12 EIA reports × 4 sections/report = 48 iterations per section type):

**For table sections:**
- Structured Extraction prompt dominated (95% selection rate)
- Extraction accuracy: 98% (vs 92% with manual prompt)
- Zero manual prompt tuning required

**For narrative sections:**
- Narrative Summarization prompt selected 85%
- Average summary length: 65 words (down from 120 words)
- Traders preferred concise summaries

**For historical comparisons:**
- Trend Analysis prompt selected 90%
- Consistently included all three comparisons (WoW, YoY, 5yr avg)

**For detecting anomalies:**
- Change Detection prompt selected 80%
- Caught 3 data anomalies that manual process missed

**Time savings:**
- Zero hours spent on weekly prompt tuning (down from 4 hours)
- System adapted automatically when EIA changed table format in week 8

### Implementation Code

```python
class EIAReportProcessor:
    def __init__(self):
        self.prompts = [structured, narrative, trend, change]
        self.router = ContextualPromptRouter(self.prompts)

    def process_report(self, report):
        """Process entire EIA report section by section."""
        results = {}

        for section in report.sections:
            # Classify section type
            section_type = classify_section(section.title, section.content)

            # Build context
            context = build_eia_context(section_type, section.has_table, section.length)

            # Route to prompt
            idx, prompt, _ = self.router.select_with_context(context)

            # Extract data
            response = llm_call(prompt, section.content)

            # Store result
            results[section.title] = {
                'response': response,
                'prompt_used': idx,
            }

            # Update bandit (with delayed reward after validation)
            # Reward comes from analyst validation or ground truth comparison
            self.pending_updates.append({
                'section': section.title,
                'context': context,
                'prompt_idx': idx,
            })

        return results

    def validate_and_update(self, section_title, ground_truth, is_correct):
        """After analyst validates, update the bandit."""
        pending = self.pending_updates[section_title]

        reward = 1.0 if is_correct else 0.0
        self.router.update(pending['prompt_idx'], pending['context'], reward)
```

### Key Lessons

1. **Delayed rewards work fine** — update bandit after analyst validates (minutes to hours later)
2. **Section-type is the most predictive feature** — different prompts for different content types
3. **Automated adaptation beats manual tuning** — system handled format changes without human intervention
4. **Start with domain knowledge** — initial prompts designed by experts, then bandit optimizes mix

---

## Case Study 3: Multi-Commodity Trading Signal System

### The Problem

A prop trading firm built an LLM system to generate trading signals across 12 commodities:
- **Energy:** WTI, Brent, Natural Gas, Gasoline
- **Agriculture:** Corn, Wheat, Soybeans
- **Metals:** Gold, Copper, Silver

They used the same signal generation prompt for all commodities:

```
Based on the provided data, generate a trading signal (BUY/SELL/HOLD)
with conviction (High/Medium/Low) and rationale.
```

**The problem:** Signal quality varied dramatically by commodity.

- **WTI crude:** 62% directional accuracy (good)
- **Natural gas:** 48% accuracy (worse than random, due to storage seasonality not captured)
- **Corn:** 55% accuracy (mediocre, needed seasonal adjustment)
- **Gold:** 58% accuracy (decent, but missed macro regime changes)

Manual prompt engineering for each commodity would require:
- 12 commodity-specific prompts
- Re-tuning every time market conditions changed
- Expertise in each market (not scalable)

### The Bandit Setup

**Prompt arms (6 total):**

1. **Momentum Signal**
   ```
   Based on recent price trends, generate a directional signal.
   Focus on: short-term momentum, breakouts, support/resistance.
   ```

2. **Fundamental Signal**
   ```
   Based on supply/demand fundamentals, generate a directional signal.
   Focus on: inventory levels, production, consumption, stocks-to-use.
   ```

3. **Seasonal Signal**
   ```
   Based on seasonal patterns, generate a directional signal.
   Focus on: time of year, historical patterns, weather factors.
   ```

4. **Macro-Driven Signal**
   ```
   Based on macroeconomic indicators, generate a directional signal.
   Focus on: dollar strength, rates, risk sentiment, inflation.
   ```

5. **Spread Signal**
   ```
   Based on term structure and spreads, generate a directional signal.
   Focus on: backwardation/contango, calendar spreads, crack spreads.
   ```

6. **Composite Signal**
   ```
   Synthesize multiple factors (fundamentals, technicals, seasonality).
   Weight each factor by current market regime.
   ```

**Context features:**
- Commodity (12 one-hot features)
- Market regime (trend, mean-revert, high-vol, low-vol)
- Seasonality strength (high, medium, low) — based on historical analysis
- Data freshness (recent release, stale data)

**Reward function:**
```python
def signal_reward(signal, conviction, actual_price_change, horizon_days=5):
    """
    Reward based on:
    1. Directional accuracy
    2. Conviction calibration
    3. Magnitude of move
    """
    # Extract direction
    direction = signal  # BUY, SELL, HOLD

    # Directional accuracy
    price_return = actual_price_change  # e.g., 0.03 = 3% gain

    if direction == "BUY" and price_return > 0:
        directional_reward = 1.0
    elif direction == "SELL" and price_return < 0:
        directional_reward = 1.0
    elif direction == "HOLD" and abs(price_return) < 0.01:
        directional_reward = 1.0
    else:
        directional_reward = 0.0

    # Conviction calibration
    if conviction == "High":
        if directional_reward == 1.0:
            conviction_bonus = 0.5  # Correct high-conviction call
        else:
            conviction_bonus = -0.5  # Incorrect high-conviction call (punish)
    else:
        conviction_bonus = 0.0

    # Magnitude bonus (bigger moves = better signals)
    magnitude_bonus = min(abs(price_return) / 0.05, 0.3)  # Cap at 5% move

    return directional_reward + conviction_bonus + magnitude_bonus
```

**Key:** Reward is delayed by 5 days (time to observe price change). Bandit learns from realized outcomes.

### The Result

After 6 months (500+ signals across 12 commodities):

**System learned commodity-specific preferences:**

| Commodity | Best Prompt | Selection % | Directional Accuracy |
|-----------|------------|-------------|---------------------|
| WTI Crude | Fundamental Signal | 65% | 68% (up from 62%) |
| Natural Gas | Seasonal Signal | 72% | 61% (up from 48%) |
| Corn | Seasonal Signal | 68% | 64% (up from 55%) |
| Gold | Macro-Driven Signal | 58% | 65% (up from 58%) |
| Copper | Macro-Driven Signal | 60% | 63% (new) |

**Key learnings:**
- **Natural gas is seasonal** — system discovered this automatically (no manual rule)
- **Agriculture benefits from seasonality** — corn, wheat, soybeans all preferred seasonal prompts
- **Metals are macro-driven** — gold, copper, silver all routed to macro-driven prompts
- **Energy is mixed** — WTI/Brent use fundamentals; natural gas uses seasonality

**Performance improvement:**
- Overall directional accuracy: 63% (up from 56%)
- High-conviction signals: 72% accuracy (well-calibrated)
- Sharpe ratio of signals-based portfolio: 1.4 (up from 0.9)

**No manual intervention:**
- System adapted when natural gas shifted from mean-reversion to trend regime
- Automatically increased seasonal prompt usage for corn during planting season
- Reduced macro-driven signals during low-volatility periods

### Implementation Code

```python
class MultiCommoditySignalRouter:
    def __init__(self):
        self.prompts = [momentum, fundamental, seasonal, macro, spread, composite]
        self.router = ContextualPromptRouter(self.prompts)
        self.pending_signals = {}  # Store signals awaiting outcomes

    def generate_signal(self, commodity, market_data, regime):
        """Generate trading signal for a commodity."""
        # Build context
        context = build_signal_context(commodity, market_data, regime)

        # Route to best prompt
        idx, prompt, _ = self.router.select_with_context(context)

        # Generate signal
        signal_text = llm_call(prompt, commodity, market_data)
        signal, conviction = parse_signal(signal_text)

        # Store for delayed reward
        signal_id = f"{commodity}_{datetime.now().isoformat()}"
        self.pending_signals[signal_id] = {
            'commodity': commodity,
            'context': context,
            'prompt_idx': idx,
            'signal': signal,
            'conviction': conviction,
            'entry_price': market_data['current_price'],
            'timestamp': datetime.now(),
        }

        return signal, conviction, signal_id

    def update_after_outcome(self, signal_id, current_price):
        """
        Called 5 days later to observe outcome and update bandit.
        """
        if signal_id not in self.pending_signals:
            return

        pending = self.pending_signals[signal_id]

        # Calculate price change
        price_return = (current_price - pending['entry_price']) / pending['entry_price']

        # Calculate reward
        reward = signal_reward(
            pending['signal'],
            pending['conviction'],
            price_return,
            horizon_days=5
        )

        # Update bandit
        self.router.update(pending['prompt_idx'], pending['context'], reward)

        # Clean up
        del self.pending_signals[signal_id]
```

### Key Lessons

1. **Delayed rewards are fine** — 5-day lag for price outcomes didn't prevent learning
2. **Commodity-specific patterns emerge** — system discovered seasonality without being told
3. **Regime adaptation** — routing changed based on market conditions (volatility, trend)
4. **Conviction calibration is critical** — reward/penalize based on conviction accuracy

---

## Summary: When to Use Prompt Routing Bandits

**Use prompt routing bandits when:**

1. **You have distinct prompt strategies** (extraction, analysis, signal generation)
2. **Optimal prompt depends on context** (task type, commodity, data availability)
3. **You get feedback on quality** (user ratings, ground truth, business outcomes)
4. **Manual prompt tuning is expensive** (time-consuming, requires expertise)
5. **The environment changes** (market regimes, user preferences, LLM updates)

**Don't use prompt routing bandits when:**

1. **One prompt clearly dominates** (no context-dependent variation)
2. **Feedback is too noisy** (can't reliably measure quality)
3. **Cold-start is critical** (need optimal behavior from the first request)
4. **Prompt space is infinite** (use meta-prompting or prompt optimization instead)

## Connections to Other Modules

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


- **Module 2 (Bayesian Bandits):** Same Thompson Sampling algorithm
- **Module 3 (Contextual Bandits):** Same LinUCB framework
- **Module 5 (Commodity Trading):** Same domain (commodities), different application (prompts vs allocation)
- **Module 7 (Production Systems):** These case studies show production deployment patterns

## Next Steps

After understanding these case studies:

1. **Try the notebooks** — implement simplified versions of these systems
2. **Design your own prompts** — for your specific commodity trading use case
3. **Define your rewards** — what matters for your application?
4. **Start simple** — 3-5 prompts, basic context features, Thompson Sampling
5. **Monitor and iterate** — track hallucination rates, costs, user satisfaction


---

## Conceptual Practice Questions

**Practice Question 1:** How does non-stationarity in commodity markets affect bandit algorithm assumptions?

**Practice Question 2:** What risk management constraints should be layered on top of a bandit-based allocation system?


---

## Cross-References

<a class="link-card" href="./01_prompt_routing_fundamentals.md">
  <div class="link-card-title">01 Prompt Routing Fundamentals</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_prompt_routing_fundamentals.md">
  <div class="link-card-title">01 Prompt Routing Fundamentals — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

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

