# Contextual Prompt Routing

> **Reading time:** ~20 min | **Module:** 08 — Prompt Routing Bandits | **Prerequisites:** Module 7


## In Brief


<div class="callout-key">

**Key Concept Summary:** One prompt cannot serve all requests. A prompt optimized for extracting inventory numbers from EIA reports will fail at generating fundamental analysis for corn markets. Contextual bandits solve th...

</div>

One prompt cannot serve all requests. A prompt optimized for extracting inventory numbers from EIA reports will fail at generating fundamental analysis for corn markets. Contextual bandits solve this by routing based on request features — task type, commodity sector, data availability, urgency — to select the best prompt for each specific context. This is the same contextual bandit framework from Module 3, applied to prompt selection.

> 💡 **Key Insight:** The optimal prompt depends on context.

- **Task type:** Data extraction needs structured prompts; fundamental analysis needs quantitative prompts
- **Commodity sector:** Energy markets have different data characteristics than agriculture
- **Data availability:** When retrieval returns rich context, use evidence-only prompts; when sparse, use analytical prompts
- **User preference:** Some users want concise signals; others want detailed research
- **Urgency:** Real-time trading decisions need fast, direct prompts; research mode can use thorough analysis

A non-contextual bandit learns "Prompt B is best on average." A contextual bandit learns "Prompt B is best for energy extraction tasks when data is available; Prompt D is best for agriculture analysis when data is sparse."

## Why One Prompt Can't Serve All Requests

Consider a commodity research assistant that handles these queries:

<div class="callout-insight">

**Insight:** Contextual bandits bridge the gap between simple A/B tests and full reinforcement learning. They personalize decisions based on observable features without needing to model state transitions.

</div>


1. "What were last week's crude oil inventories?" (extraction task, energy sector, data-heavy)
2. "Analyze the corn supply-demand balance for 2024/25." (analysis task, agriculture sector, requires synthesis)
3. "Should I buy natural gas futures?" (signal generation task, energy sector, needs conviction)
4. "Give me three scenarios for copper prices." (scenario analysis, metals sector, needs probabilistic thinking)

If you use the same prompt for all four:
- **Generic prompt:** Mediocre at everything, excellent at nothing
- **Extraction-optimized prompt:** Great for query 1, terrible for queries 2-4 (produces tables when you need analysis)
- **Analysis-optimized prompt:** Great for query 2, terrible for query 1 (verbose when you need numbers)

**The solution:** Route to different prompts based on query characteristics.

## Context Features for Commodity LLM Routing

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


### Feature 1: Task Type

**Extraction categories:**
- **Data extraction:** Pull specific numbers from reports (inventories, production, prices)
- **Fundamental analysis:** Synthesize supply/demand drivers into narrative
- **Signal generation:** Produce actionable buy/sell/hold recommendations
- **Scenario modeling:** Generate bull/base/bear cases with probabilities
- **Comparative analysis:** Compare commodities or time periods

**How to extract:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def classify_task_type(query):
    """Classify query into task type."""
    query_lower = query.lower()

    # Keyword-based classification
    if any(kw in query_lower for kw in ['inventory', 'stockpile', 'storage', 'production']):
        if '?' not in query or 'what' in query_lower or 'how much' in query_lower:
            return 'extraction'

    if any(kw in query_lower for kw in ['analyze', 'analysis', 'outlook', 'balance']):
        return 'analysis'

    if any(kw in query_lower for kw in ['buy', 'sell', 'trade', 'should i', 'recommend']):
        return 'signal'

    if any(kw in query_lower for kw in ['scenario', 'cases', 'if', 'risk']):
        return 'scenario'

    return 'general'  # Default
```

</div>
</div>

### Feature 2: Commodity Sector

**Why it matters:**
- **Energy (oil, gas, coal):** Weekly data releases (EIA), liquid markets, geopolitical sensitivity
- **Agriculture (corn, wheat, soybeans):** Seasonal patterns, weather-dependent, USDA reports
- **Metals (copper, gold, aluminum):** Industrial demand signals, slower data releases, macro-driven

Different sectors have different:
- **Data availability:** EIA releases weekly; USDA monthly; LME data varies
- **Volatility patterns:** Energy spikes on geopolitics; agriculture on weather; metals on macro
- **Key drivers:** Energy = inventory; agriculture = yield forecasts; metals = industrial production

**How to extract:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def classify_commodity_sector(query):
    """Identify commodity sector from query."""
    query_lower = query.lower()

    energy_keywords = ['oil', 'crude', 'wti', 'brent', 'gas', 'natural gas', 'gasoline', 'diesel']
    ag_keywords = ['corn', 'wheat', 'soy', 'cotton', 'sugar', 'coffee', 'grain']
    metal_keywords = ['copper', 'gold', 'silver', 'aluminum', 'iron', 'steel', 'platinum']

    if any(kw in query_lower for kw in energy_keywords):
        return 'energy'
    elif any(kw in query_lower for kw in ag_keywords):
        return 'agriculture'
    elif any(kw in query_lower for kw in metal_keywords):
        return 'metals'

    return 'unknown'
```

</div>
</div>

### Feature 3: Data Availability

**Why it matters:**
When your RAG system retrieves rich context (5 relevant documents), use evidence-only prompts.
When retrieval is sparse (0-1 documents), evidence-only prompts will just say "not found" — use analytical prompts instead.

**How to measure:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def assess_data_availability(retrieved_docs, query):
    """Score data availability for this query."""
    # Number of retrieved documents
    doc_count = len(retrieved_docs)

    # Relevance of retrieved docs (semantic similarity)
    avg_relevance = np.mean([doc['relevance_score'] for doc in retrieved_docs])

    # Combined score
    if doc_count >= 3 and avg_relevance > 0.7:
        return 'high'
    elif doc_count >= 1 and avg_relevance > 0.5:
        return 'medium'
    else:
        return 'low'
```


### Feature 4: User Preference

Some users want concise signals; others want detailed research reports.

**How to capture:**
```python
def get_user_preference(user_id, history):
    """Infer user preferences from interaction history."""
    if not history:
        return 'balanced'  # Default

    # Analyze past feedback
    detailed_satisfaction = np.mean([h['reward'] for h in history if h['response_length'] > 200])
    concise_satisfaction = np.mean([h['reward'] for h in history if h['response_length'] <= 200])

    if detailed_satisfaction > concise_satisfaction + 0.2:
        return 'detailed'
    elif concise_satisfaction > detailed_satisfaction + 0.2:
        return 'concise'
    else:
        return 'balanced'
```

### Feature 5: Urgency

**Real-time trading mode** (market hours, breaking news) needs fast, direct prompts.
**Research mode** (planning, backtesting) can use thorough, comprehensive prompts.

**How to detect:**
```python
def assess_urgency(query, market_hours):
    """Determine urgency level."""
    # Check for urgency keywords
    urgent_keywords = ['now', 'current', 'latest', 'real-time', 'live', 'immediate']
    if any(kw in query.lower() for kw in urgent_keywords):
        return 'high'

    # Check if during market hours
    if market_hours:
        return 'medium'

    return 'low'
```

## Implementing Contextual Prompt Routing with LinUCB

LinUCB (Linear Upper Confidence Bound) is a contextual bandit algorithm that learns a linear model:

```
Expected reward of prompt p for context x = θ_p^T · x
```

where:
- θ_p is the learned weight vector for prompt p
- x is the context feature vector

**Implementation:**

```python
import numpy as np

class ContextualPromptRouter:
    def __init__(self, num_prompts, context_dim, alpha=1.0):
        """
        Initialize LinUCB for prompt routing.

        Args:
            num_prompts: Number of prompt templates
            context_dim: Dimension of context feature vector
            alpha: Exploration parameter (higher = more exploration)
        """
        self.num_prompts = num_prompts
        self.context_dim = context_dim
        self.alpha = alpha

        # Initialize for each prompt
        self.A = [np.identity(context_dim) for _ in range(num_prompts)]  # Design matrix
        self.b = [np.zeros(context_dim) for _ in range(num_prompts)]     # Response vector

    def select_prompt(self, context):
        """
        Select prompt using LinUCB.

        Args:
            context: Feature vector (numpy array, length = context_dim)

        Returns:
            Selected prompt index
        """
        ucb_scores = []

        for i in range(self.num_prompts):
            # Estimate parameters: θ = A^(-1) · b
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]

            # Expected reward
            expected_reward = theta @ context

            # Uncertainty bonus
            uncertainty = np.sqrt(context @ A_inv @ context)

            # UCB score
            ucb = expected_reward + self.alpha * uncertainty
            ucb_scores.append(ucb)

        return np.argmax(ucb_scores)

    def update(self, prompt_idx, context, reward):
        """
        Update model after observing reward.

        Args:
            prompt_idx: Index of prompt used
            context: Context vector
            reward: Observed reward
        """
        self.A[prompt_idx] += np.outer(context, context)
        self.b[prompt_idx] += reward * context
```

## Building the Context Vector

Encode all relevant features into a numerical vector:

```python
def build_context_vector(query, retrieved_docs, user_profile, market_state):
    """
    Build context feature vector for prompt routing.

    Returns:
        numpy array of shape (context_dim,)
    """
    features = []

    # Task type (one-hot encoding)
    task = classify_task_type(query)
    task_types = ['extraction', 'analysis', 'signal', 'scenario', 'general']
    features.extend([1 if task == t else 0 for t in task_types])  # 5 features

    # Commodity sector (one-hot encoding)
    sector = classify_commodity_sector(query)
    sectors = ['energy', 'agriculture', 'metals', 'unknown']
    features.extend([1 if sector == s else 0 for s in sectors])  # 4 features

    # Data availability (continuous)
    data_avail = assess_data_availability(retrieved_docs, query)
    avail_map = {'high': 1.0, 'medium': 0.5, 'low': 0.0}
    features.append(avail_map[data_avail])  # 1 feature

    # User preference (one-hot)
    pref = get_user_preference(user_profile.get('id'), user_profile.get('history', []))
    prefs = ['concise', 'balanced', 'detailed']
    features.extend([1 if pref == p else 0 for p in prefs])  # 3 features

    # Urgency (continuous)
    urgency = assess_urgency(query, market_state['is_market_hours'])
    urgency_map = {'high': 1.0, 'medium': 0.5, 'low': 0.0}
    features.append(urgency_map[urgency])  # 1 feature

    # Intercept term
    features.append(1.0)  # 1 feature

    return np.array(features)  # Total: 15 features
```

## Complete Contextual Routing System

```python
class CommodityPromptRouter:
    def __init__(self, prompts):
        """
        Production-ready contextual prompt router.

        Args:
            prompts: List of prompt templates
        """
        self.prompts = prompts
        self.context_dim = 15  # From build_context_vector
        self.router = ContextualPromptRouter(len(prompts), self.context_dim)

    def route(self, query, retrieved_docs, user_profile, market_state):
        """
        Select best prompt for this request.

        Returns:
            (prompt_idx, prompt_text, context_vector)
        """
        # Build context
        context = build_context_vector(query, retrieved_docs, user_profile, market_state)

        # Select prompt
        idx = self.router.select_prompt(context)

        return idx, self.prompts[idx], context

    def update(self, prompt_idx, context, reward):
        """Update router based on observed reward."""
        self.router.update(prompt_idx, context, reward)

    def get_stats(self):
        """Get learned parameters for each prompt."""
        stats = []
        for i in range(len(self.prompts)):
            A_inv = np.linalg.inv(self.router.A[i])
            theta = A_inv @ self.router.b[i]
            stats.append({
                'prompt_idx': i,
                'learned_weights': theta,
                'selections': np.trace(self.router.A[i]) - self.context_dim,
            })
        return stats
```

**Usage:**

```python

# Initialize with prompt templates
prompts = [
    "Structured extraction prompt...",
    "Evidence-only prompt...",
    "Quantitative analysis prompt...",
    "Trading signal prompt...",
    "Scenario analysis prompt...",
]

router = CommodityPromptRouter(prompts)

# For each request
for request in incoming_requests:
    # Extract context
    retrieved_docs = rag_system.retrieve(request['query'])
    user_profile = get_user_profile(request['user_id'])
    market_state = get_market_state()

    # Route to best prompt
    idx, prompt, context = router.route(
        request['query'],
        retrieved_docs,
        user_profile,
        market_state
    )

    # Generate response
    response = llm_call(prompt, request['query'], retrieved_docs)

    # Calculate reward
    reward = compute_reward(request['query'], response, retrieved_docs)

    # Update router
    router.update(idx, context, reward)
```

## Visual Explanation

```
┌────────────────────────────────────────────────────────────────────┐
│               CONTEXTUAL PROMPT ROUTING DECISION TREE              │
└────────────────────────────────────────────────────────────────────┘

User Query: "What are the latest crude oil inventories?"
                                │
                                ▼
                    EXTRACT CONTEXT FEATURES
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
   Task Type:           Commodity Sector:         Data Availability:
   EXTRACTION              ENERGY                     HIGH
   (1,0,0,0,0)           (1,0,0,0)                   (1.0)
                                │
                                ▼
                    CONTEXT VECTOR (x)
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1.0, 0, 1, 0, 0.5, 1]
                                │
                                ▼
                    COMPUTE UCB SCORES FOR EACH PROMPT
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
   Prompt 1:             Prompt 2:               Prompt 3:
   Structured            Evidence-Only           Quantitative
   UCB = 0.73            UCB = 0.91 ← MAX        UCB = 0.65
                                │
                                ▼
                    SELECT PROMPT 2 (Evidence-Only)
                                │
                                ▼
                "Use ONLY data from retrieved documents.
                 If inventory data not found, say so.
                 Format: [Date] [Value] [Source]"
                                │
                                ▼
                        LLM GENERATES RESPONSE
                                │
                                ▼
                    OBSERVE REWARD (r = 0.95)
                                │
                                ▼
                UPDATE: A₂ += x·xᵀ, b₂ += r·x
                (Prompt 2 gets reinforced for this context)
```

## Real-World Example: Learning Prompt Preferences by Context

After 1000 requests, the router learns:

**For energy + extraction + high data:**
- Prompt 2 (Evidence-Only) wins 70% of the time
- Average reward: 0.85

**For agriculture + analysis + low data:**
- Prompt 3 (Quantitative Analysis) wins 65% of the time
- Average reward: 0.78
- Why? Agricultural analysis requires synthesis beyond simple data retrieval

**For any sector + signal + medium data:**
- Prompt 4 (Trading Signal) wins 80% of the time
- Average reward: 0.82

**For metals + scenario + high urgency:**
- Prompt 5 (Scenario Analysis) wins 55% of the time
- Average reward: 0.74

## Intuitive Explanation

Think of contextual routing like a newsroom editor assigning stories to reporters.

You have 5 reporters:
1. **Data reporter** (great at extracting facts from press releases)
2. **Investigative reporter** (only reports verified information)
3. **Analysis reporter** (synthesizes trends and patterns)
4. **Opinion columnist** (makes strong recommendations)
5. **Scenario planner** (explores what-if cases)

When a story comes in, the editor considers:
- **Story type:** Breaking news? Feature? Opinion piece?
- **Topic:** Politics? Sports? Business?
- **Source availability:** Lots of sources? Sparse?
- **Audience:** General public? Expert readers?
- **Deadline:** Immediate? Next week?

A good editor learns which reporter to assign based on these factors. A contextual prompt router does the same thing.

## Common Pitfalls

### Pitfall 1: Too Many Features
**Problem:** 50 context features means slow learning (need more data per feature).

**Fix:** Start with 10-15 most predictive features. Add more only if needed.

### Pitfall 2: Sparse Features
**Problem:** If 95% of requests are energy extraction, you won't learn good prompts for agriculture analysis.

**Fix:** Force exploration of undersampled contexts, or use warm-start with domain knowledge.

### Pitfall 3: Irrelevant Features
**Problem:** Including "day of week" as a feature when it doesn't affect optimal prompt choice.

**Fix:** Feature selection — test if removing a feature hurts performance.

### Pitfall 4: Static Context
**Problem:** Extracting context features once at the start, not updating as conversation evolves.

**Fix:** Rebuild context vector for each turn in a multi-turn conversation.

## Connection to Module 3 (Contextual Bandits)

This is the exact same LinUCB algorithm from Module 3, applied to prompts instead of commodity allocation:

| Module 3 (Commodity Trading) | Module 8 (Prompt Routing) |
|------------------------------|---------------------------|
| Arms = commodities (WTI, corn, gold) | Arms = prompts (extraction, analysis, signal) |
| Context = market features (VIX, term structure) | Context = request features (task type, sector) |
| Reward = portfolio return | Reward = response quality |
| Goal = maximize Sharpe ratio | Goal = maximize task completion + accuracy |

The math is identical. The application is different.

## Connections

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.



**Builds on:**
- **Module 3 (Contextual Bandits):** LinUCB algorithm is the routing engine
- **Guide 01:** Prompt arms are the actions
- **Guide 02:** Reward function is the optimization target

**Leads to:**
- **Guide 04:** Real-world case studies of contextual routing in commodity systems

## Practice Problems

### Problem 1: Feature Engineering
You're building a prompt router for a commodities research assistant. Design a context vector with 10 features. For each feature, explain:
- What it measures
- How you would extract it from a request
- Why it matters for prompt selection

### Problem 2: Context-Specific Prompts
You observe that for energy + extraction tasks, Prompt A (structured) wins 80% of the time. But for agriculture + extraction tasks, Prompt B (evidence-only) wins 75% of the time.

Explain why the same task type (extraction) might need different prompts for different commodity sectors.

### Problem 3: Implementation
Implement a simple contextual router with 3 prompts and 5 context features. Use LinUCB. Simulate 500 requests where:
- Feature 1 (task type) determines the best prompt
- Prompts: extraction, analysis, signal

Show that the router learns to route correctly based on task type.

### Problem 4: Cold Start Problem
A new user arrives with no history. Your contextual router includes "user preference" as a feature. How do you handle this cold start? Provide two approaches.


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

<a class="link-card" href="./04_commodity_research_assistant.md">
  <div class="link-card-title">04 Commodity Research Assistant</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./04_commodity_research_assistant.md">
  <div class="link-card-title">04 Commodity Research Assistant — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

