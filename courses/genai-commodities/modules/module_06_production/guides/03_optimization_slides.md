---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# LLM Cost Optimization, Latency, and Caching

**Module 6: Production**

The triple constraint: cost, latency, and quality

<!-- Speaker notes: Section transition. Briefly preview what this section covers before diving into details. -->

---

## The 80/15/5 Rule

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
    A[All LLM Queries] --> B{Query Type?}

    B -->|80%| C[Repetitive<br/>CACHEABLE<br/>Cost: $0]
    B -->|15%| D[Simple<br/>Use Haiku<br/>Cost: $0.01]
    B -->|5%| E[Complex<br/>Use Sonnet/Opus<br/>Cost: $0.03-0.10]

    C --> F[Effective Cost:<br/>$0.005/signal avg]
    D --> F
    E --> F
```

<div class="callout-key">

Key implementation detail -- study this pattern carefully.

</div>

> 80% of queries are repetitive (cacheable), 15% can use smaller models, and only 5% require the full frontier model.

<!-- Speaker notes: Walk through the diagram step by step. Highlight the key decision points and data flow. -->

---

## Cost Model

**Total Cost per Signal:**
$$C_{\text{total}} = n_{\text{input}} \times p_{\text{input}} + n_{\text{output}} \times p_{\text{output}}$$

**Example (1000 signals/day):**

| Component | Tokens | Price/1M | Daily Cost |
|-----------|--------|----------|------------|
| Input | 2,000/signal | $3.00 | $6.00 |
| Output | 500/signal | $15.00 | $7.50 |
| **Total** | | | **$13.50/day** |

**Monthly: $405** (before optimization)

**With caching (60% hit rate): $162/month** (60% reduction)

<!-- Speaker notes: Review the table contents. Ask learners which rows are most relevant to their use case. -->

---

## Latency Breakdown

$$L_{\text{total}} = L_{\text{network}} + L_{\text{queue}} + L_{\text{inference}} + L_{\text{processing}}$$

| Component | Typical | Optimized |
|-----------|---------|-----------|
| Network | 100-200ms | 100ms (edge) |
| Queue | 0-500ms | 0ms (priority) |
| Inference | 1000-3000ms | 500ms (Haiku) |
| Processing | 100-500ms | 50ms (compiled) |
| **Total** | **1.2-4.2s** | **650ms** |

**Target SLA:** p50 < 500ms, p95 < 2000ms, p99 < 5000ms

<!-- Speaker notes: Review the table contents. Ask learners which rows are most relevant to their use case. -->

---

## Naive vs Optimized Implementation

<div class="columns">
<div>

### Naive ($3000/month)
```python
for news in news_feed:
    context = get_full_context(news)  # 5000 tokens
    signal = claude_opus(context)     # $0.10/call
    # 1000 calls/day → $100/day
```

<div class="callout-insight">

This pattern recurs throughout the course. Understanding it deeply pays dividends later.

</div>

</div>
<div>

### Optimized ($300/month)
```python
for news in news_feed:
    if news in cache:           # 60% hit
        signal = cache[news]    # $0.00
    else:
        context = compress(news)  # 1000 tokens
        if is_simple(news):
            signal = haiku(context)   # $0.01
        else:
            signal = sonnet(context)  # $0.03
        cache[news] = signal
    # Effective: $10/day
```

**90% cost reduction**

</div>
</div>

<!-- Speaker notes: Walk through the code, emphasizing the key patterns. Highlight which parts learners should customize for their own use cases. -->

---

## Parallel Processing for Latency

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
    subgraph Sequential["Sequential: 4.1 seconds"]
        A1[News<br/>100ms] --> A2[Entities<br/>500ms]
        A2 --> A3[Prices<br/>500ms]
        A3 --> A4[Sentiment<br/>1000ms]
        A4 --> A5[Signal<br/>2000ms]
    end

    subgraph Parallel["Parallel: 3.1 seconds"]
        B1[News<br/>100ms] --> B2[Entities<br/>500ms]
        B1 --> B3[Prices<br/>500ms]
        B1 --> B4[Sentiment<br/>1000ms]
        B2 --> B5[Signal<br/>2000ms]
        B3 --> B5
        B4 --> B5
    end

    subgraph Cached["With Cache: 110ms"]
        C1[News<br/>100ms] --> C2[CACHE HIT<br/>10ms]
    end
```

<div class="callout-warning">

Watch for edge cases with this implementation in production use.

</div>

<!-- Speaker notes: Walk through the diagram step by step. Highlight the key decision points and data flow. -->

---

<!-- _class: lead -->

# Semantic Cache Implementation

Intelligent caching with TTL and statistics

<!-- Speaker notes: Section transition. Briefly preview what this section covers before diving into details. -->

---

<!-- Speaker notes: Cover the key points about SemanticCache. Emphasize practical implications and connect to previous material. -->

## SemanticCache

```python
class SemanticCache:
    def __init__(self, redis_client, similarity_threshold=0.95, ttl=3600):
        self.redis = redis_client
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        self.saves_usd = 0

```

<div class="callout-info">

This approach follows established best practices in the field.

</div>

---

```python
    def get(self, prompt, cost_saved=0.05):
        key = f"semantic_cache:{hashlib.sha256(prompt.encode()).hexdigest()}"
        cached = self.redis.get(key)
        if cached:
            self.hits += 1
            self.saves_usd += cost_saved
            return json.loads(cached)
        self.misses += 1
        return None

    def set(self, prompt, response):
        key = f"semantic_cache:{hashlib.sha256(prompt.encode()).hexdigest()}"
        self.redis.setex(key, self.ttl, json.dumps(response))

```

<!-- Speaker notes: Walk through the code, emphasizing the key patterns. Highlight which parts learners should customize for their own use cases. -->

---

## Cache Hit Rate Impact

$$C_{\text{effective}} = C_{\text{total}} \times (1 - r_{\text{cache}})$$

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart LR
    A[1000 signals/day<br/>$0.05 each] --> B{Cache Hit?}
    B -->|60% HIT| C[$0.00<br/>600 signals free]
    B -->|40% MISS| D[$0.05<br/>400 signals billed]

    C --> E[Daily Cost:<br/>$20 instead of $50]
    D --> E

    E --> F[Monthly Savings:<br/>$900]
```

<!-- Speaker notes: Walk through the diagram step by step. Highlight the key decision points and data flow. -->

---

<!-- _class: lead -->

# Model Routing

Automatic model selection based on query complexity

<!-- Speaker notes: Section transition. Briefly preview what this section covers before diving into details. -->

---

<!-- Speaker notes: Cover the key points about ModelRouter. Emphasize practical implications and connect to previous material. -->

## ModelRouter

```python
class ModelRouter:
    def estimate_complexity(self, prompt, context) -> float:
        score = 0.0
        # Context length
        score += min(len(prompt.split()) / 1000, 0.3)
        # Conflicting signals
        if context.get('conflicting_signals'):
            score += 0.3
        # High uncertainty
        if context.get('uncertainty_high'):
            score += 0.2
```

---

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">select_model.py</span>
</div>

```python
        # Multi-asset analysis
        if context.get('multi_asset'):
            score += 0.2
        return min(score, 1.0)

    def select_model(self, complexity) -> str:
        if complexity < 0.3: return 'haiku'    # Fast, cheap
        elif complexity < 0.7: return 'sonnet'  # Balanced
        else: return 'opus'                      # Best quality

```

</div>

<!-- Speaker notes: Walk through the code, emphasizing the key patterns. Highlight which parts learners should customize for their own use cases. -->

---

## Model Routing Decision

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
    A[Incoming Query] --> B[Estimate<br/>Complexity Score]

    B --> C{Score < 0.3?}
    C -->|Yes| D[Haiku<br/>$0.01/call<br/>500ms]
    C -->|No| E{Score < 0.7?}
    E -->|Yes| F[Sonnet<br/>$0.03/call<br/>2000ms]
    E -->|No| G[Opus<br/>$0.10/call<br/>5000ms]

    D --> H[Return Signal]
    F --> H
    G --> H
```

**Quality-Aware Routing:**
$$\text{Model} = \begin{cases}
\text{Haiku} & \text{if } q < 0.3 \\
\text{Sonnet} & \text{if } 0.3 \leq q < 0.7 \\
\text{Opus} & \text{if } q \geq 0.7
\end{cases}$$

<!-- Speaker notes: Walk through the diagram step by step. Highlight the key decision points and data flow. -->

---

<!-- Speaker notes: Cover the key points about Prompt Compression. Emphasize practical implications and connect to previous material. -->

## Prompt Compression

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">promptcompressor.py</span>
</div>

```python
class PromptCompressor:
    def compress_context(self, context, max_tokens=1000):
        lines = context.split('\n')

        # Priority scoring (recency + keyword importance)
        scored_lines = []
        for i, line in enumerate(lines):
            score = 1.0 - (i / len(lines)) * 0.5
            # Boost important keywords
            if any(kw in line.lower()
                   for kw in ['price', 'supply', 'demand', 'opec']):
                score += 0.3
            scored_lines.append((score, line))
```

</div>

---

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

        # Take highest-scoring lines within budget
        scored_lines.sort(reverse=True, key=lambda x: x[0])
        compressed = []
        token_count = 0
        for score, line in scored_lines:
            if token_count + len(line)/4 <= max_tokens:
                compressed.append(line)
                token_count += len(line)/4
        return '\n'.join(compressed)

```

</div>

<!-- Speaker notes: Walk through the code, emphasizing the key patterns. Highlight which parts learners should customize for their own use cases. -->

---

## Full Optimization Pipeline

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
    A[Raw Context<br/>5000 tokens] --> B[Compress<br/>to 1000 tokens]
    B --> C{In Cache?}

    C -->|HIT| D[Return Cached<br/>Latency: 10ms<br/>Cost: $0]

    C -->|MISS| E[Estimate<br/>Complexity]
    E --> F{Route to<br/>Model}

    F -->|Simple| G[Haiku: $0.01<br/>500ms]
    F -->|Medium| H[Sonnet: $0.03<br/>2000ms]
    F -->|Complex| I[Opus: $0.10<br/>5000ms]

    G --> J[Cache Result]
    H --> J
    I --> J

    J --> K[Return Signal<br/>Track Metrics]
```

<!-- Speaker notes: Walk through the diagram step by step. Highlight the key decision points and data flow. -->

---

## Common Pitfalls

<div class="columns">
<div>

### Over-Aggressive Caching
Caching market data for too long

**Solution:** Short TTL (10-30 min) for time-sensitive data; longer for static analysis

### Premature Model Downgrade
Routing too many queries to cheap models

**Solution:** A/B test routing thresholds; monitor quality metrics by model tier

### Compression Losing Critical Info
Aggressive compression removes key context

**Solution:** Preserve high-importance sentences; test compression ratios against signal quality

</div>
<div>

### Cache Key Collisions
Different prompts hash to same key

**Solution:** Include timestamp, asset ID, and model version in cache key

### Ignoring Latency Long Tail
Optimizing average but ignoring p99

**Solution:** Set timeouts; implement fallback responses; monitor all percentiles

</div>
</div>

<!-- Speaker notes: Walk through each pitfall with a real-world example. Ask learners if they have encountered any of these in their own work. -->

---

## Optimization Results Summary

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Cost/signal | $0.05 | $0.005 | 90% reduction |
| Latency p50 | 2000ms | 300ms | 85% faster |
| Latency p99 | 8000ms | 3000ms | 63% faster |
| Cache hit rate | 0% | 65% | From zero |
| Monthly cost | $1500 | $150 | $1350 saved |

> The goal: maintain signal quality while reducing cost from $0.05 to $0.005 and latency from 3s to 300ms.

<!-- Speaker notes: Review the table contents. Ask learners which rows are most relevant to their use case. -->

---

## Key Takeaways

1. **Cache aggressively** -- 80% of queries are repetitive, cache hit rates of 60-80% are achievable

2. **Route intelligently** -- use Haiku for simple queries, Sonnet for medium, Opus only when needed

3. **Compress ruthlessly** -- reduce input tokens from 5000 to 1000 with priority-based compression

4. **Parallelize everything** -- independent data fetches should run concurrently

5. **Monitor the tradeoffs** -- cost optimization must not degrade signal quality below acceptable thresholds

<!-- Speaker notes: Recap the main points. Ask learners which takeaway they found most surprising or useful. -->

---

## Connections

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
graph LR
    A[Production<br/>Deployment] --> B[Optimization<br/>This Guide]
    C[Monitoring &<br/>Drift Detection] --> B
    D[Module 5<br/>Signal Pipeline] --> B

    B --> E[ROI Analysis]
    B --> F[Scaling<br/>Strategy]
    B --> G[Multi-Region<br/>Deployment]

    E --> H[Continuous<br/>Cost-Quality<br/>Improvement]
    F --> H
```

<!-- Speaker notes: Show how this content connects to other modules. Point learners to the next recommended deck. -->
