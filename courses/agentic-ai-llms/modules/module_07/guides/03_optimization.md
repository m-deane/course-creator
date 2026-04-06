# Cost and Latency Optimization for AI Agents

> **Reading time:** ~10 min | **Module:** 7 — Production Deployment | **Prerequisites:** Module 7 — Observability

Optimization for AI agents means reducing cost and latency while maintaining output quality. Cost optimization minimizes token usage and API calls; latency optimization reduces response time through caching, parallelization, and model selection. Both require measurement-driven trade-offs between speed, cost, and quality.

<div class="callout-insight">

**Insight:** Agent systems are fundamentally expensive and slow—LLM calls cost money, tools take time, and complex reasoning requires multiple steps. The goal isn't to make them free or instant, but to eliminate waste: unnecessary tokens, redundant calls, sequential operations that could be parallel, and expensive models for simple tasks.

</div>

**The Optimization Hierarchy:**
1. **Eliminate:** Remove unnecessary work entirely (caching, deduplication)
2. **Reduce:** Use fewer resources for the same work (smaller models, shorter prompts)
3. **Parallelize:** Do multiple things simultaneously (concurrent tool calls)
4. **Accelerate:** Make individual operations faster (streaming, edge deployment)

## Formal Definition

**Cost Optimization** minimizes `total_cost = Σ(llm_calls × tokens × price_per_token + tool_calls × tool_cost)`

**Latency Optimization** minimizes `total_latency = Σ(llm_latency + tool_latency + network_latency + processing_overhead)`

**Subject to constraints:**
- Output quality ≥ quality_threshold
- Success rate ≥ success_threshold
- User experience remains acceptable

**Key Metrics:**
- **Cost per session:** Total cost to handle one user request
- **Tokens per task:** Average tokens needed to complete task type
- **Latency percentiles:** p50, p95, p99 response times
- **Cache hit rate:** Percentage of requests served from cache
- **Quality-cost tradeoff:** Output quality vs. cost spent

## Intuitive Explanation

Think of optimization like route planning:

**Unoptimized Route:**
- Use expensive toll roads for every trip (always use GPT-4)
- Stop at every red light even if road is clear (sequential execution)
- Take the same route even if you've been there before (no caching)
- Bring entire library when you only need one book (full context always)

**Optimized Route:**
- Use toll roads only when speed matters, else use free routes (model routing)
- Anticipate traffic patterns to hit green lights (parallel execution)
- Remember routes you've taken before (caching)
- Pack only what you need for the trip (context pruning)

Both get you there, but optimization saves time and money.

## Code Implementation

### Cost Optimization: Model Routing


<span class="filename">agent.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
from anthropic import Anthropic
from typing import Literal

class ModelRouter:
    """Route requests to appropriate model based on complexity."""

    def __init__(self, client: Anthropic):
        self.client = client

        # Model costs (per 1M tokens)
        self.costs = {
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
            "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
            "claude-opus": {"input": 15.00, "output": 75.00}
        }

    def estimate_complexity(self, task: str) -> Literal["simple", "medium", "complex"]:
        """Estimate task complexity."""

        # Simple heuristics (could use ML classifier)
        task_lower = task.lower()

        # Simple: short, single operation
        if len(task) < 100 and any(keyword in task_lower for keyword in
            ["what is", "define", "translate", "summarize this"]):
            return "simple"

        # Complex: long, multi-step, requires reasoning
        if len(task) > 500 or any(keyword in task_lower for keyword in
            ["analyze", "compare", "evaluate", "design", "create a plan"]):
            return "complex"

        return "medium"

    def select_model(self, task: str) -> str:
        """Select appropriate model for task."""
        complexity = self.estimate_complexity(task)

        model_map = {
            "simple": "claude-3-haiku",
            "medium": "claude-3-5-sonnet",
            "complex": "claude-opus"
        }

        return model_map[complexity]

    def execute(self, task: str, max_tokens: int = 1024) -> tuple[str, float]:
        """Execute with optimal model selection."""

        model = self.select_model(task)

        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": task}]
        )

        # Calculate cost
        input_cost = response.usage.input_tokens * self.costs[model]["input"] / 1_000_000
        output_cost = response.usage.output_tokens * self.costs[model]["output"] / 1_000_000
        total_cost = input_cost + output_cost

        return response.content[0].text, total_cost


# Example: Save 90% on simple queries
router = ModelRouter(client)

# Simple task → Haiku ($0.25/1M vs $3.00/1M) = 12x cheaper
result, cost = router.execute("What is the capital of France?")
print(f"Cost: ${cost:.4f}")  # ~$0.0001

# Complex task → Sonnet/Opus when needed
result, cost = router.execute(
    "Analyze the economic implications of climate change..."
)
print(f"Cost: ${cost:.4f}")  # Higher but appropriate
```

</div>
</div>

### Cost Optimization: Prompt Compression

```python
class PromptOptimizer:
    """Reduce token usage through prompt optimization."""

    def __init__(self, client: Anthropic):
        self.client = client

    def compress_context(self, long_context: str, max_tokens: int = 1000) -> str:
        """Compress long context to essential information."""

        compression_prompt = f"""Compress this text to {max_tokens} tokens while preserving key information:

{long_context}

Return only the compressed version."""

        response = self.client.messages.create(
            model="claude-3-haiku",  # Use cheap model for compression
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": compression_prompt}]
        )

        return response.content[0].text

    def optimize_prompt(self, prompt: str) -> str:
        """Remove unnecessary tokens from prompt."""

        # Remove extra whitespace
        optimized = " ".join(prompt.split())

        # Remove common filler words if over budget
        if len(optimized) > 2000:  # Arbitrary threshold
            fillers = ["basically", "actually", "literally", "really", "very"]
            for filler in fillers:
                optimized = optimized.replace(f" {filler} ", " ")

        return optimized

    def extract_relevant_context(
        self,
        query: str,
        full_context: str,
        max_context_tokens: int = 500
    ) -> str:
        """Extract only relevant portions of context."""

        extraction_prompt = f"""Given this query: "{query}"

Extract the most relevant information from this context (max {max_context_tokens} tokens):

{full_context}

Return only the relevant excerpts."""

        response = self.client.messages.create(
            model="claude-3-haiku",
            max_tokens=max_context_tokens,
            messages=[{"role": "user", "content": extraction_prompt}]
        )

        return response.content[0].text


# Example: Reduce token usage by 70%
optimizer = PromptOptimizer(client)

long_doc = "..." # 10,000 tokens
query = "What does the document say about revenue?"

relevant_context = optimizer.extract_relevant_context(query, long_doc, max_context_tokens=500)
# Now only ~500 tokens instead of 10,000 = 95% reduction
```

### Latency Optimization: Caching

```python
import hashlib
import json
from typing import Optional
import redis

class AgentCache:
    """Cache agent responses for repeated queries."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.ttl_seconds = 3600  # 1 hour default

    def _make_key(self, input: str, context: dict) -> str:
        """Create cache key from input and context."""
        cache_input = {
            "input": input,
            "context": context
        }
        serialized = json.dumps(cache_input, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def get(self, input: str, context: dict = None) -> Optional[str]:
        """Get cached response if available."""
        key = self._make_key(input, context or {})
        cached = self.redis.get(key)

        if cached:
            return json.loads(cached)["response"]
        return None

    def set(self, input: str, response: str, context: dict = None):
        """Cache response."""
        key = self._make_key(input, context or {})
        value = json.dumps({
            "response": response,
            "timestamp": time.time()
        })
        self.redis.setex(key, self.ttl_seconds, value)

    def execute_with_cache(
        self,
        input: str,
        execute_fn: Callable[[str], str],
        context: dict = None
    ) -> tuple[str, bool]:
        """Execute with caching."""

        # Try cache first
        cached = self.get(input, context)
        if cached:
            return cached, True  # Cache hit

        # Execute if not cached
        response = execute_fn(input)

        # Cache for next time
        self.set(input, response, context)

        return response, False  # Cache miss


# Example: 50x speedup on repeated queries
cache = AgentCache(redis.Redis())

def expensive_agent_call(input: str) -> str:
    # Takes 2 seconds, costs $0.10
    return agent.execute(input)

# First call: slow (2s)
result, hit = cache.execute_with_cache("What is AI?", expensive_agent_call)
print(f"Cache hit: {hit}, Result: {result}")  # False

# Second call: instant (~0.04s), free
result, hit = cache.execute_with_cache("What is AI?", expensive_agent_call)
print(f"Cache hit: {hit}, Result: {result}")  # True
```

### Latency Optimization: Parallel Execution

```python
import asyncio
from typing import List

class ParallelAgent:
    """Execute multiple operations in parallel."""

    def __init__(self, client: Anthropic):
        self.client = client

    async def parallel_llm_calls(
        self,
        prompts: List[str]
    ) -> List[str]:
        """Execute multiple LLM calls in parallel."""

        async def single_call(prompt: str) -> str:
            # Use async client
            response = await self.client.messages.create_async(
                model="claude-3-5-sonnet",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        # Execute all in parallel
        results = await asyncio.gather(*[single_call(p) for p in prompts])
        return results

    async def parallel_tool_calls(
        self,
        tool_calls: List[tuple[str, dict]]
    ) -> List[any]:
        """Execute multiple tool calls in parallel."""

        async def execute_tool(tool_name: str, params: dict):
            # Tool execution logic
            return await tools[tool_name](**params)

        results = await asyncio.gather(
            *[execute_tool(name, params) for name, params in tool_calls]
        )
        return results


# Example: 3x speedup for independent operations
agent = ParallelAgent(client)

# Sequential: 6 seconds total (3 calls × 2s each)
start = time.time()
result1 = agent.execute("Summarize article 1")
result2 = agent.execute("Summarize article 2")
result3 = agent.execute("Summarize article 3")
print(f"Sequential: {time.time() - start:.1f}s")  # ~6s

# Parallel: 2 seconds total (all at once)
start = time.time()
results = asyncio.run(agent.parallel_llm_calls([
    "Summarize article 1",
    "Summarize article 2",
    "Summarize article 3"
]))
print(f"Parallel: {time.time() - start:.1f}s")  # ~2s
```

### Latency Optimization: Streaming

```python
class StreamingAgent:
    """Stream responses for faster perceived latency."""

    def __init__(self, client: Anthropic):
        self.client = client

    def execute_streaming(self, prompt: str):
        """Execute with streaming for faster time-to-first-token."""

        with self.client.messages.stream(
            model="claude-3-5-sonnet",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                # In production: yield text to frontend

        print()  # Newline after stream


# Example: User sees results immediately
agent = StreamingAgent(client)

# Non-streaming: wait 2s, then see full response
response = agent.execute("Write a long essay")  # Wait... wait... [full essay]

# Streaming: see words as they're generated
agent.execute_streaming("Write a long essay")  # The... essay... begins... with...
# Perceived latency: 100ms (time to first token)
# Actual latency: still 2s, but feels faster
```

### Integrated Optimization Pipeline

```python
class OptimizedAgent:
    """Agent with full cost and latency optimizations."""

    def __init__(self, client: Anthropic, redis_client: redis.Redis):
        self.model_router = ModelRouter(client)
        self.prompt_optimizer = PromptOptimizer(client)
        self.cache = AgentCache(redis_client)

    def execute(
        self,
        prompt: str,
        context: str = None,
        use_cache: bool = True
    ) -> tuple[str, dict]:
        """Execute with all optimizations."""

        metrics = {
            "cache_hit": False,
            "model_used": None,
            "tokens_saved": 0,
            "latency_ms": 0,
            "cost": 0.0
        }

        start_time = time.time()

        # Optimization 1: Check cache
        if use_cache:
            cached = self.cache.get(prompt, {"context": context})
            if cached:
                metrics["cache_hit"] = True
                metrics["latency_ms"] = (time.time() - start_time) * 1000
                return cached, metrics

        # Optimization 2: Compress context if provided
        if context and len(context) > 1000:
            original_tokens = len(context.split())
            context = self.prompt_optimizer.compress_context(context, max_tokens=500)
            compressed_tokens = len(context.split())
            metrics["tokens_saved"] = original_tokens - compressed_tokens

        # Optimization 3: Optimize prompt
        optimized_prompt = self.prompt_optimizer.optimize_prompt(prompt)

        # Optimization 4: Route to appropriate model
        full_prompt = f"{context}\n\n{optimized_prompt}" if context else optimized_prompt
        response, cost = self.model_router.execute(full_prompt)

        metrics["model_used"] = self.model_router.select_model(full_prompt)
        metrics["cost"] = cost
        metrics["latency_ms"] = (time.time() - start_time) * 1000

        # Cache for future
        if use_cache:
            self.cache.set(prompt, response, {"context": context})

        return response, metrics


# Example: Combined optimizations
agent = OptimizedAgent(client, redis_client)

# First call
response, metrics = agent.execute(
    "What is the main point of this document?",
    context=ten_thousand_word_document
)
print(f"Cost: ${metrics['cost']:.4f}, Latency: {metrics['latency_ms']:.0f}ms")
# Cost: $0.02 (context compressed, routed to Sonnet)
# Latency: 800ms

# Repeated call
response, metrics = agent.execute(
    "What is the main point of this document?",
    context=ten_thousand_word_document
)
print(f"Cost: ${metrics['cost']:.4f}, Latency: {metrics['latency_ms']:.0f}ms")
# Cost: $0.00 (cache hit)
# Latency: 5ms
```

## Common Pitfalls

### 1. Premature Optimization
**Problem:** Optimizing before measuring impact.


<span class="filename">agent.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# DON'T: Optimize everything immediately
optimize_every_single_call()

# DO: Measure first, optimize bottlenecks
measure_performance()
identify_bottlenecks()  # 90% of cost is in one place
optimize_that_bottleneck()
```

</div>
</div>

### 2. Over-Caching
**Problem:** Serving stale data when freshness matters.

```python
# DON'T: Cache everything forever
cache.set(key, value, ttl=FOREVER)

# DO: Appropriate cache TTLs
cache.set("static_content", value, ttl=86400)  # 1 day
cache.set("stock_price", value, ttl=60)  # 1 minute
cache.set("user_specific", value, ttl=300)  # 5 minutes
```

### 3. Quality Degradation
**Problem:** Optimizing cost at expense of quality.

```python
# DON'T: Always use cheapest model
model = "claude-haiku"  # Even for complex tasks

# DO: Balance quality and cost
if task_is_critical:
    model = "claude-opus"
elif task_is_complex:
    model = "claude-sonnet"
else:
    model = "claude-haiku"
```

### 4. False Parallelization
**Problem:** Parallelizing dependent operations.

```python
# DON'T: Parallelize dependent steps
results = await asyncio.gather(
    get_data(),       # Step 2 needs this
    analyze(data)     # Can't run until step 1 completes!
)

# DO: Only parallelize independent operations
data = await get_data()  # Must be first
results = await asyncio.gather(
    analyze(data),    # Independent
    summarize(data),  # Independent
    visualize(data)   # Independent
)
```

## Connections

**Builds on:**
- Observability (measuring what to optimize)
- Production Architecture (designing optimizable systems)
- LLM Fundamentals (understanding token economics)

**Leads to:**
- Cost Management (budgeting and forecasting)
- Scalability (handling higher loads efficiently)
- User Experience (faster, more responsive agents)

**Related to:**
- Database Optimization (caching, indexing)
- API Design (rate limiting, batching)
- Distributed Systems (parallelization, replication)

## Practice Problems

### 1. Optimization Strategy
You have an agent system with these characteristics:
- 10,000 requests/day
- Average: 5,000 input tokens, 500 output tokens per request
- Current model: Claude Opus ($15/$75 per 1M tokens)
- Average latency: 3 seconds
- Cache hit potential: 30% (repeated queries)

Calculate:
- Current daily cost
- Cost after implementing caching
- Cost after model routing (50% simple → Haiku, 30% medium → Sonnet, 20% complex → Opus)
- Combined savings

### 2. Latency Budget
Your agent has a 2-second latency budget. Currently:
- LLM call: 1.5s
- Tool calls (3): 0.3s each (0.9s total)
- Processing: 0.1s
- Total: 2.5s (over budget by 0.5s)

Design optimizations to meet the budget.

### 3. Cache Strategy
Design a caching strategy for:
- Product search results (10M queries/day, high repetition)
- Personalized recommendations (1M users, user-specific)
- Real-time stock analysis (100K queries/day, time-sensitive)

What cache keys, TTLs, and invalidation strategies would you use?

### 4. Cost-Quality Tradeoff
Your agent routes tasks to models:
- Haiku: 95% quality, $0.001/request
- Sonnet: 98% quality, $0.01/request
- Opus: 99.5% quality, $0.05/request

For a customer service agent handling 1M requests/month:
- Calculate cost for 100% Haiku, Sonnet, Opus
- Design routing strategy to maximize quality under $10K/month budget
- What's the expected average quality score?

<div class="callout-key">

**Key Concept Summary:** This guide covered the core concepts. Review the companion slides for visual summaries and the hands-on notebook for practice implementations.

</div>

## Further Reading

**Cost Optimization:**
- "The AI Token Economy" - Understanding LLM pricing
- Anthropic: "Prompt Engineering for Cost Efficiency"
- OpenAI: "Token Optimization Strategies"
- "Optimizing LLM Costs in Production" (Huyen, 2024)

**Latency Optimization:**
- "High Performance Browser Networking" - Latency fundamentals
- "Designing Data-Intensive Applications" - Caching patterns
- "Systems Performance" (Gregg) - Performance analysis
- Anthropic: "Streaming and Latency Optimization"

**Tools & Services:**
- Redis (caching)
- OpenTelemetry (metrics)
- Helicone (LLM cost tracking)
- LangSmith (prompt optimization)
- PromptLayer (caching and analytics)

**Advanced Topics:**
- Semantic caching (similar queries, not just exact matches)
- Speculative execution (predict next calls)
- Model distillation (compress expensive models)
- Prompt compression techniques (LLMLingua, AutoCompressor)
- Dynamic batching for throughput

---

**Next Steps:**

<a class="link-card" href="./03_optimization_slides.md">
  <div class="link-card-title">Performance Optimization for Agents — Companion Slides</div>
  <div class="link-card-description">Visual slide deck with diagrams, speaker notes, and key takeaways.</div>
</a>

<a class="link-card" href="../notebooks/01_deployment_patterns.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with working code and guided exercises.</div>
</a>
