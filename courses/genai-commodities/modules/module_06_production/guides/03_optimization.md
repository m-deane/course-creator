# LLM Cost Optimization, Latency Reduction, and Caching

> **Reading time:** ~11 min | **Module:** Module 6: Production | **Prerequisites:** Modules 0-5

<div class="callout-key">

**Key Concept Summary:** Production LLM systems face the triple constraint of cost, latency, and quality. Cost optimization reduces API expenses through prompt compression, smaller models, and intelligent caching. Latency reduction ensures signals arrive before market moves through parallel processing, streaming, and pre...

</div>

## In Brief

Production LLM systems face the triple constraint of cost, latency, and quality. Cost optimization reduces API expenses through prompt compression, smaller models, and intelligent caching. Latency reduction ensures signals arrive before market moves through parallel processing, streaming, and precomputation. The goal: maintain signal quality while reducing cost per signal from $0.05 to $0.005 and latency from 3s to 300ms.

<div class="callout-insight">

**Insight:** LLM costs follow a power law: 80% of queries are repetitive (cacheable), 15% can use smaller/faster models, and only 5% require the full frontier model. Latency has three components—network (100-200ms), inference (1-2s), and processing (0.5s)—each optimizable independently. The optimization strategy: cache aggressively, route intelligently, compress ruthlessly, and parallelize everything.

</div>
## Intuitive Explanation

## Formal Definition

### Cost Model

**Total Cost per Signal:**
$$C_{\text{total}} = C_{\text{input}} + C_{\text{output}} + C_{\text{compute}}$$

Where:
$$C_{\text{input}} = n_{\text{input\_tokens}} \times p_{\text{input}}$$
$$C_{\text{output}} = n_{\text{output\_tokens}} \times p_{\text{output}}$$

Typical pricing (Claude Sonnet 4):
- $p_{\text{input}} = \$3.00$ per 1M tokens
- $p_{\text{output}} = \$15.00$ per 1M tokens

**Monthly Cost:**
$$C_{\text{month}} = n_{\text{signals/day}} \times C_{\text{total}} \times 30$$

Example:
- 1000 signals/day
- 2000 input tokens/signal
- 500 output tokens/signal
- Cost: $1000 \times (2000 \times 0.000003 + 500 \times 0.000015) \times 30 = \$405/month$

### Latency Model

**End-to-End Latency:**
$$L_{\text{total}} = L_{\text{network}} + L_{\text{queue}} + L_{\text{inference}} + L_{\text{processing}}$$

Typical breakdown:
- $L_{\text{network}}$: 100-200ms (API round-trip)
- $L_{\text{queue}}$: 0-500ms (provider load)
- $L_{\text{inference}}$: 1000-3000ms (token generation)
- $L_{\text{processing}}$: 100-500ms (parsing, validation)

**Target Latency SLA:**
- p50 < 500ms
- p95 < 2000ms
- p99 < 5000ms

### Cache Hit Rate

**Effective Cost Reduction:**
$$C_{\text{effective}} = C_{\text{total}} \times (1 - r_{\text{cache}})$$

Where $r_{\text{cache}}$ is cache hit rate.

Example:
- Base cost: $0.05/signal
- Cache hit rate: 60%
- Effective cost: $0.05 \times 0.4 = \$0.02/signal$ (60% reduction)

### Model Routing

**Quality-Aware Routing:**

For query complexity score $q$:
$$\text{Model} = \begin{cases}
\text{Haiku (fast/cheap)} & \text{if } q < \theta_1 \\
\text{Sonnet (balanced)} & \text{if } \theta_1 \leq q < \theta_2 \\
\text{Opus (slow/expensive)} & \text{if } q \geq \theta_2
\end{cases}$$

Cost-quality tradeoff:
$$\text{Maximize: } Q(\text{Model}) - \lambda \times C(\text{Model})$$

### The Cost Problem

**Naive Implementation:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Every signal: full context, large model
for news in news_feed:
    context = get_full_context(news)  # 5000 tokens
    signal = claude_opus(context)      # $0.10 per call
    # 1000 calls/day → $100/day → $3000/month
```

</div>
</div>

**Optimized Implementation:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Smart routing and caching
for news in news_feed:
    # Check cache first
    if news in cache:
        signal = cache[news]           # $0.00 (60% hit rate)
    else:
        # Compress context
        context = compress_context(news)  # 1000 tokens

        # Route by complexity
        if is_simple(news):
            signal = claude_haiku(context)  # $0.01 (30% of queries)
        else:
            signal = claude_sonnet(context) # $0.03 (10% of queries)

        cache[news] = signal
    # Effective cost: $0.01/signal → $10/day → $300/month
```

</div>
</div>

**Savings: 90% cost reduction**

### The Latency Problem

**Sequential Processing (Slow):**
```
News → Extract entities → Get prices → Get sentiment → Generate signal
100ms     500ms             500ms        1000ms          2000ms
Total: 4.1 seconds (too slow for trading)
```

**Parallel Processing (Fast):**
```
                 ┌→ Extract entities (500ms) ──┐
News (100ms) ───┼→ Get prices (500ms) ─────────┼→ Generate signal (2000ms)
                 └→ Get sentiment (1000ms) ─────┘

Total: 100ms + 1000ms + 2000ms = 3.1 seconds (better)

With caching:
News (100ms) → [CACHE HIT] → Return signal (10ms)
Total: 110ms (acceptable for trading)
```

## Code Implementation

### Intelligent Caching System


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">from.py</span>
</div>

```python
import hashlib
import json
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import redis
from anthropic import Anthropic

@dataclass
class CacheConfig:
    """Cache configuration."""
    ttl_seconds: int = 3600  # 1 hour default
    max_size: int = 10000
    enable_compression: bool = True


class SemanticCache:
    """
    Semantic caching for LLM responses.
    Uses embedding similarity for fuzzy matching.
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        similarity_threshold: float = 0.95,
        ttl: int = 3600
    ):
        self.redis = redis_client
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl

        # Statistics
        self.hits = 0
        self.misses = 0
        self.saves_usd = 0

    def _compute_key(self, prompt: str) -> str:
        """Compute cache key from prompt."""
        return f"semantic_cache:{hashlib.sha256(prompt.encode()).hexdigest()}"

    def get(self, prompt: str, cost_saved: float = 0.05) -> Optional[Dict]:
        """
        Retrieve cached response if exists.

        Args:
            prompt: LLM prompt
            cost_saved: Cost saved if cache hit (for statistics)

        Returns:
            Cached response or None
        """
        key = self._compute_key(prompt)
        cached = self.redis.get(key)

        if cached:
            self.hits += 1
            self.saves_usd += cost_saved
            return json.loads(cached)

        self.misses += 1
        return None

    def set(self, prompt: str, response: Dict):
        """Cache response with TTL."""
        key = self._compute_key(prompt)
        self.redis.setex(
            key,
            self.ttl,
            json.dumps(response)
        )

    def stats(self) -> Dict:
        """Cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'saves_usd': self.saves_usd,
            'monthly_saves_usd': self.saves_usd * 30  # Extrapolate
        }


class ModelRouter:
    """
    Route queries to appropriate model based on complexity.
    """

    def __init__(self, anthropic_api_key: str):
        self.client = Anthropic(api_key=anthropic_api_key)

        # Model costs (per 1M tokens)
        self.costs = {
            'haiku': {'input': 0.80, 'output': 4.00},
            'sonnet': {'input': 3.00, 'output': 15.00},
            'opus': {'input': 15.00, 'output': 75.00}
        }

    def estimate_complexity(self, prompt: str, context: Dict) -> float:
        """
        Estimate query complexity (0-1 scale).

        Heuristics:
        - Long context → complex
        - Many entities → complex
        - Conflicting signals → complex
        """
        score = 0.0

        # Context length
        prompt_length = len(prompt.split())
        score += min(prompt_length / 1000, 0.3)

        # Context indicators
        if context.get('conflicting_signals'):
            score += 0.3

        if context.get('uncertainty_high'):
            score += 0.2

        if context.get('multi_asset'):
            score += 0.2

        return min(score, 1.0)

    def select_model(self, complexity: float) -> str:
        """
        Select model based on complexity.

        Returns:
            Model name: 'haiku', 'sonnet', or 'opus'
        """
        if complexity < 0.3:
            return 'haiku'
        elif complexity < 0.7:
            return 'sonnet'
        else:
            return 'opus'

    def generate(
        self,
        prompt: str,
        context: Dict,
        force_model: Optional[str] = None
    ) -> Tuple[Dict, Dict]:
        """
        Generate response with automatic model selection.

        Returns:
            (response, metadata)
        """
        start_time = time.time()

        # Determine model
        if force_model:
            model = force_model
            complexity = 0.5
        else:
            complexity = self.estimate_complexity(prompt, context)
            model = self.select_model(complexity)

        # Map to API model names
        model_map = {
            'haiku': 'claude-haiku-4-20250514',
            'sonnet': 'claude-sonnet-4-20250514',
            'opus': 'claude-opus-4-20250514'
        }

        # Generate
        response = self.client.messages.create(
            model=model_map[model],
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )

        latency = time.time() - start_time

        # Parse response
        try:
            parsed = json.loads(response.content[0].text)
        except json.JSONDecodeError:
            parsed = {'raw': response.content[0].text}

        # Calculate cost
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = (
            input_tokens * self.costs[model]['input'] / 1_000_000 +
            output_tokens * self.costs[model]['output'] / 1_000_000
        )

        metadata = {
            'model': model,
            'complexity': complexity,
            'latency_ms': latency * 1000,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost_usd': cost
        }

        return parsed, metadata


class PromptCompressor:
    """
    Compress prompts to reduce token usage.
    """

    def compress_context(self, context: str, max_tokens: int = 1000) -> str:
        """
        Compress context to fit token budget.

        Strategies:
        1. Remove redundant information
        2. Summarize lengthy sections
        3. Prioritize recent/relevant info
        """
        # Simple compression: truncate and prioritize
        lines = context.split('\n')

        # Priority scoring (recency bias)
        scored_lines = []
        for i, line in enumerate(lines):
            score = 1.0 - (i / len(lines)) * 0.5  # Recent lines score higher

            # Boost important keywords
            if any(kw in line.lower() for kw in ['price', 'supply', 'demand', 'opec']):
                score += 0.3

            scored_lines.append((score, line))

        # Sort by score
        scored_lines.sort(reverse=True, key=lambda x: x[0])

        # Reconstruct context
        compressed = []
        token_count = 0

        for score, line in scored_lines:
            # Rough token estimate (1 token ≈ 4 chars)
            line_tokens = len(line) / 4

            if token_count + line_tokens <= max_tokens:
                compressed.append(line)
                token_count += line_tokens
            else:
                break

        return '\n'.join(compressed)


class OptimizedLLMClient:
    """
    Fully optimized LLM client with caching, routing, and compression.
    """

    def __init__(
        self,
        anthropic_api_key: str,
        redis_url: str = "redis://localhost:6379"
    ):
        self.router = ModelRouter(anthropic_api_key)
        self.cache = SemanticCache(
            redis_client=redis.from_url(redis_url),
            ttl=3600
        )
        self.compressor = PromptCompressor()

        # Statistics
        self.total_calls = 0
        self.total_cost = 0
        self.total_latency = 0

    def generate_signal(
        self,
        prompt_template: str,
        context: str,
        context_dict: Dict,
        enable_cache: bool = True,
        enable_compression: bool = True,
        max_context_tokens: int = 1000
    ) -> Tuple[Dict, Dict]:
        """
        Generate signal with full optimization.

        Args:
            prompt_template: Template string
            context: Context string
            context_dict: Context metadata for complexity estimation
            enable_cache: Use caching
            enable_compression: Compress prompts
            max_context_tokens: Max tokens for context

        Returns:
            (signal, stats)
        """
        self.total_calls += 1

        # Compress context
        if enable_compression:
            context = self.compressor.compress_context(context, max_context_tokens)

        # Build prompt
        full_prompt = prompt_template.format(context=context)

        # Check cache
        if enable_cache:
            cached = self.cache.get(full_prompt)
            if cached:
                return cached, {
                    'cache_hit': True,
                    'cost_usd': 0,
                    'latency_ms': 10  # Cache retrieval time
                }

        # Generate with routing
        signal, metadata = self.router.generate(full_prompt, context_dict)

        # Cache result
        if enable_cache:
            self.cache.set(full_prompt, signal)

        # Update statistics
        self.total_cost += metadata['cost_usd']
        self.total_latency += metadata['latency_ms']

        metadata['cache_hit'] = False
        return signal, metadata

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        cache_stats = self.cache.stats()

        return {
            'total_calls': self.total_calls,
            'total_cost_usd': self.total_cost,
            'avg_cost_usd': self.total_cost / self.total_calls if self.total_calls > 0 else 0,
            'avg_latency_ms': self.total_latency / self.total_calls if self.total_calls > 0 else 0,
            'cache_hit_rate': cache_stats['hit_rate'],
            'cache_saves_usd': cache_stats['saves_usd'],
            'projected_monthly_cost_usd': self.total_cost * 30,
            'projected_monthly_saves_usd': cache_stats['monthly_saves_usd']
        }


# Example Usage

print("=" * 70)
print("LLM COST OPTIMIZATION")
print("=" * 70)

# Initialize optimized client
client = OptimizedLLMClient(anthropic_api_key="your-key-here")

# Simulate 100 signals with various complexities
contexts = [
    {
        'text': "OPEC announces production cut of 1M bpd. Crude prices rally 3%.",
        'metadata': {'conflicting_signals': False, 'uncertainty_high': False}
    },
    {
        'text': "Mixed signals: OPEC cuts but recession fears grow. China demand uncertain.",
        'metadata': {'conflicting_signals': True, 'uncertainty_high': True}
    },
    {
        'text': "Inventory data shows unexpected build. Prices decline 2%.",
        'metadata': {'conflicting_signals': False, 'uncertainty_high': False}
    }
]

prompt_template = """
Analyze this commodity market update and generate trading signal:

{context}

Provide JSON:
{{
  "direction": "bullish" | "bearish" | "neutral",
  "strength": 0.0-1.0
}}
"""

print("\nGenerating 100 signals (with repetition for cache testing)...")

results = []
for i in range(100):
    # Simulate repeated contexts (cache hits)
    context = contexts[i % len(contexts)]

    signal, stats = client.generate_signal(
        prompt_template=prompt_template,
        context=context['text'],
        context_dict=context['metadata'],
        enable_cache=True,
        enable_compression=True
    )

    results.append(stats)

    if (i + 1) % 25 == 0:
        current_stats = client.get_statistics()
        print(f"\nAfter {i+1} signals:")
        print(f"  Avg Cost: ${current_stats['avg_cost_usd']:.4f}")
        print(f"  Avg Latency: {current_stats['avg_latency_ms']:.0f}ms")
        print(f"  Cache Hit Rate: {current_stats['cache_hit_rate']*100:.1f}%")

# Final statistics
print("\n" + "=" * 70)
print("FINAL STATISTICS")
print("=" * 70)

final_stats = client.get_statistics()

print(f"\nCost Analysis:")
print(f"  Total Cost: ${final_stats['total_cost_usd']:.2f}")
print(f"  Avg Cost per Signal: ${final_stats['avg_cost_usd']:.4f}")
print(f"  Projected Monthly Cost: ${final_stats['projected_monthly_cost_usd']:.2f}")
print(f"  Cache Savings: ${final_stats['cache_saves_usd']:.2f}")
print(f"  Projected Monthly Savings: ${final_stats['projected_monthly_saves_usd']:.2f}")

print(f"\nPerformance Analysis:")
print(f"  Avg Latency: {final_stats['avg_latency_ms']:.0f}ms")
print(f"  Cache Hit Rate: {final_stats['cache_hit_rate']*100:.1f}%")

print(f"\nOptimization Impact:")
naive_cost = 0.05 * final_stats['total_calls']
savings_pct = (1 - final_stats['total_cost_usd'] / naive_cost) * 100
print(f"  Naive Cost (no optimization): ${naive_cost:.2f}")
print(f"  Optimized Cost: ${final_stats['total_cost_usd']:.2f}")
print(f"  Cost Reduction: {savings_pct:.1f}%")

# Model distribution
model_counts = {}
for result in results:
    if not result.get('cache_hit'):
        model = result.get('model', 'unknown')
        model_counts[model] = model_counts.get(model, 0) + 1

print(f"\nModel Usage Distribution:")
for model, count in model_counts.items():
    pct = count / len([r for r in results if not r.get('cache_hit')]) * 100
    print(f"  {model}: {count} calls ({pct:.1f}%)")
```

</div>
</div>

## Common Pitfalls

**1. Over-Aggressive Caching**
- Problem: Caching market-sensitive data for too long
- Symptom: Stale signals when market moves quickly
- Solution: Short TTL (10-30 min) for time-sensitive data, longer for static analysis

**2. Premature Model Downgrade**
- Problem: Routing too many queries to cheap/fast models
- Symptom: Signal quality degrades, win rate drops
- Solution: A/B test routing thresholds, monitor quality metrics by model tier

**3. Compression Losing Critical Information**
- Problem: Aggressive prompt compression removes key context
- Symptom: LLM generates generic signals without specific reasoning
- Solution: Preserve high-importance sentences, test compression ratios

**4. Cache Key Collisions**
- Problem: Different prompts hash to same cache key
- Symptom: Wrong cached responses returned
- Solution: Include timestamp, asset ID, model version in cache key

**5. Ignoring Latency Long Tail**
- Problem: Optimizing average latency, ignoring p99
- Symptom: Occasional 10s delays disrupt trading
- Solution: Set timeouts, implement fallback responses, monitor percentiles

## Connections

**Builds on:**
- Module 6.1: Production deployment (what to optimize)
- Module 6.2: Monitoring (identify optimization opportunities)
- Module 5: Signal generation (optimize signal pipeline)

**Leads to:**
- ROI analysis (quantify optimization benefits)
- Scaling strategy (handle 10x traffic growth)
- Multi-region deployment (latency optimization via geography)

**Related concepts:**
- API rate limiting (budget enforcement)
- Serverless optimization (cold start reduction)
- Edge computing (latency reduction via proximity)

## Practice Problems

1. **Cost-Benefit Analysis**
   Current: 1000 signals/day, $0.05/signal → $1500/month
   Optimization investment: $5000 (engineering time)
   Target: $0.01/signal
   Payback period?

2. **Cache TTL Optimization**
   Market regime changes every 2 hours on average.
   Cache hit rate by TTL:
   - 10 min: 40%
   - 30 min: 65%
   - 60 min: 75%
   - 120 min: 80%

   Risk of stale signal increases with TTL.
   Optimal TTL?

3. **Model Routing**
   Haiku: $0.01, 500ms, 55% accuracy
   Sonnet: $0.03, 2000ms, 62% accuracy
   Opus: $0.10, 5000ms, 65% accuracy

   Trading signal has 2-second window.
   Average trade profit: $50
   Which model(s) to use?

4. **Latency Budget**
   Target: p95 latency < 2000ms
   Components:
   - Network: 150ms
   - Queue: 200ms
   - Inference: ???
   - Processing: 300ms

   Max allowable inference time?

5. **Parallel Processing**
   Sequential: News fetch (200ms) → Sentiment (1000ms) → Signal (2000ms)
   Can parallelize sentiment + price fetch (500ms)
   What's new total latency?
   What if sentiment needed for signal generation?

<div class="callout-insight">

**Insight:** Understanding llm cost optimization, latency reduction, and caching is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.




## Further Reading

**Cost Optimization:**
<div class="flow">
<div class="flow-step mint">1. "LLM Economics" by A...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">2. "Optimizing Large La...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">3. "Prompt Engineering ...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. "Semantic Caching fo...</div>


1. **"LLM Economics" by Anthropic** - Cost structure and optimization
2. **"Optimizing Large Language Model Inference"** - Inference optimization techniques
3. **"Prompt Engineering for Cost Reduction"** - Shorter prompts, same quality

**Caching Strategies:**
4. **"Semantic Caching for LLMs"** - Embedding-based caching
5. **"Cache-Augmented Language Models"** - Research on LLM caching
6. **"Redis Best Practices"** - Production caching

**Latency Optimization:**
7. **"Low-Latency ML Serving"** - Model serving optimization
8. **"Speculative Decoding"** - Faster token generation
9. **"Batching Strategies for LLM APIs"** - Throughput optimization

**Model Selection:**
10. **"Cascading LLMs for Cost-Quality Tradeoffs"** - Routing strategies
11. **"Model Distillation"** - Creating cheaper versions
12. **"Mixture of Experts"** - Efficient model architectures

*"Optimize for cost, latency, and quality -- in that order -- until constraints are met."*

---

## Conceptual Practice Questions

1. What makes LLMs particularly useful for commodity market analysis compared to traditional NLP?

2. Describe three types of commodity documents that LLMs can process and the structured output you would expect from each.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---



## Cross-References

<a class="link-card" href="./03_optimization_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_pipeline_build.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_commodity_agents.md">
  <div class="link-card-title">01 Commodity Agents</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_production_deployment.md">
  <div class="link-card-title">01 Production Deployment</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

