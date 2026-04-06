# Custom Model Wrappers in Dataiku

> **Reading time:** ~12 min | **Module:** 3 — Custom | **Prerequisites:** Module 2 — RAG, Python programming

## In Brief

Custom model wrappers extend Dataiku's LLM Mesh to add preprocessing, postprocessing, routing logic, or integration with custom endpoints. They enable advanced patterns like retry logic, response parsing, multi-model ensembles, and specialized prompt templates while maintaining compatibility with Dataiku's standard LLM interfaces.

<div class="callout-insight">
<strong>Key Insight:</strong> Production LLM applications require more than just API calls—they need error handling, response validation, cost optimization, and domain-specific logic. Custom wrappers provide a clean abstraction layer that encapsulates this complexity, making advanced LLM patterns reusable across projects without duplicating code.
</div>

<div class="callout-key">
<strong>Key Concept:</strong> Custom model wrappers extend Dataiku's LLM Mesh to add preprocessing, postprocessing, routing logic, or integration with custom endpoints. They enable advanced patterns like retry logic, response parsing, multi-model ensembles, and specialized prompt templates while maintaining compatibility with...
</div>

## Formal Definition

**Custom Model Wrapper** is a Python class that:
- **Implements** the standard LLM interface (complete, chat methods)
- **Encapsulates** preprocessing (prompt formatting, validation) and postprocessing (parsing, validation)
- **Extends** base LLM behavior with error handling, retries, logging, and routing
- **Maintains** compatibility with Dataiku recipes and Prompt Studios
- **Provides** domain-specific abstractions for common use cases

## Intuitive Explanation

Think of custom model wrappers like middleware in web development. Just as middleware sits between your application and the web server (handling authentication, logging, error handling), a custom LLM wrapper sits between your application and the LLM API (handling retries, parsing, validation, cost tracking). You write the wrapper once and use it everywhere, ensuring consistent behavior across all LLM interactions.

## Visual Representation

```
┌─────────────────────────────────────────────────────────────┐
│                 Your Application Code                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Custom Model Wrapper                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Preprocessing │  │ Error        │  │Postprocessing│      │
│  │• Validation  │  │ Handling     │  │• Parse JSON  │      │
│  │• Formatting  │  │ • Retries    │  │• Validate    │      │
│  │• Enrichment  │  │ • Fallback   │  │• Transform   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────▲───────┘      │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘              │
└────────────────────────────┼─────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM Mesh                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ Claude   │    │  GPT-4   │    │  Custom  │              │
│  └──────────┘    └──────────┘    └──────────┘              │
└─────────────────────────────────────────────────────────────┘
```

## Code Implementation

### Basic Wrapper Pattern

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from dataiku.llm import LLM
from typing import Optional, Dict, Any
import json
import time

class BaseLLMWrapper:
    """
    Base class for custom LLM wrappers.
    """

    def __init__(self, connection_name: str, **kwargs):
        """
        Initialize wrapper with LLM connection.

        Args:
            connection_name: LLM Mesh connection name
            **kwargs: Additional configuration
        """
        self.llm = LLM(connection_name)
        self.config = kwargs

    def preprocess(self, prompt: str, **kwargs) -> tuple[str, Dict]:
        """
        Preprocess prompt before sending to LLM.

        Args:
            prompt: Raw prompt text
            **kwargs: Additional parameters

        Returns:
            Tuple of (processed_prompt, processed_kwargs)
        """
        # Default: no preprocessing
        return prompt, kwargs

    def postprocess(self, response: Any) -> Any:
        """
        Postprocess LLM response.

        Args:
            response: Raw LLM response

        Returns:
            Processed response
        """
        # Default: return raw response
        return response

    def complete(self, prompt: str, **kwargs) -> Any:
        """
        Generate completion with pre/post processing.

        Args:
            prompt: Input prompt
            **kwargs: LLM parameters

        Returns:
            Processed response
        """
        # Preprocess
        processed_prompt, processed_kwargs = self.preprocess(prompt, **kwargs)

        # Call LLM
        response = self.llm.complete(processed_prompt, **processed_kwargs)

        # Postprocess
        result = self.postprocess(response)

        return result
```

</div>

### JSON Extraction Wrapper

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
class JSONExtractorLLM(BaseLLMWrapper):
    """
    Wrapper that ensures JSON output and parses it automatically.
    """

    def __init__(
        self,
        connection_name: str,
        retry_on_invalid_json: bool = True,
        max_retries: int = 3
    ):
        super().__init__(connection_name)
        self.retry_on_invalid_json = retry_on_invalid_json
        self.max_retries = max_retries

    def preprocess(self, prompt: str, **kwargs) -> tuple[str, Dict]:
        """Add JSON instruction to prompt."""
        json_instruction = "\n\nReturn valid JSON only. No explanatory text."

        if json_instruction not in prompt:
            prompt = prompt + json_instruction

        # Force low temperature for structured output
        kwargs['temperature'] = kwargs.get('temperature', 0)

        return prompt, kwargs

    def postprocess(self, response: Any) -> Dict:
        """Parse and validate JSON response."""
        text = response.text.strip()

        # Try to extract JSON if embedded in markdown
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()

        # Parse JSON
        try:
            data = json.loads(text)
            return {
                'success': True,
                'data': data,
                'raw_response': response
            }
        except json.JSONDecodeError as e:
            return {
                'success': False,
                'error': str(e),
                'raw_text': text,
                'raw_response': response
            }

    def complete(self, prompt: str, **kwargs) -> Dict:
        """Generate completion with automatic retry on invalid JSON."""
        for attempt in range(self.max_retries):
            result = super().complete(prompt, **kwargs)

            if result['success']:
                return result['data']

            if not self.retry_on_invalid_json or attempt == self.max_retries - 1:
                raise ValueError(
                    f"Failed to get valid JSON after {self.max_retries} attempts. "
                    f"Last error: {result['error']}"
                )

            # Retry with more explicit instructions
            prompt = prompt + f"\n\nPrevious attempt returned invalid JSON. Please return ONLY valid JSON."

        raise ValueError("Max retries exceeded")

# Usage
json_llm = JSONExtractorLLM('anthropic-claude')

result = json_llm.complete("""
Extract from this report:
- inventory_change (number)
- sentiment (bullish/bearish/neutral)
- key_factors (list of strings)

Report: U.S. crude inventories fell 5.2 million barrels...
""")

print(result)
# {'inventory_change': -5.2, 'sentiment': 'bullish', 'key_factors': [...]}
```

</div>

### Retry and Fallback Wrapper

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import logging
from typing import List

logger = logging.getLogger(__name__)

class RetryFallbackLLM(BaseLLMWrapper):
    """
    Wrapper with automatic retry and fallback to alternative models.
    """

    def __init__(
        self,
        primary_connection: str,
        fallback_connections: List[str],
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        super().__init__(primary_connection)
        self.fallback_connections = fallback_connections
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def complete(self, prompt: str, **kwargs) -> Any:
        """
        Try primary connection, then fallbacks on error.
        """
        connections = [self.llm] + [
            LLM(conn) for conn in self.fallback_connections
        ]

        last_error = None

        for conn_idx, connection in enumerate(connections):
            conn_name = (
                "primary" if conn_idx == 0
                else f"fallback-{conn_idx}"
            )

            for attempt in range(self.max_retries):
                try:
                    logger.info(
                        f"Attempting {conn_name}, try {attempt + 1}/{self.max_retries}"
                    )

                    processed_prompt, processed_kwargs = self.preprocess(
                        prompt, **kwargs
                    )

                    response = connection.complete(
                        processed_prompt,
                        **processed_kwargs
                    )

                    result = self.postprocess(response)

                    logger.info(f"Success with {conn_name}")
                    return result

                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"{conn_name} attempt {attempt + 1} failed: {e}"
                    )

                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff

            logger.warning(f"{conn_name} exhausted all retries")

        raise RuntimeError(
            f"All connections failed after {self.max_retries} retries. "
            f"Last error: {last_error}"
        )

# Usage
resilient_llm = RetryFallbackLLM(
    primary_connection='anthropic-claude',
    fallback_connections=['openai-gpt4', 'azure-openai'],
    max_retries=3,
    retry_delay=1.0
)

result = resilient_llm.complete("Summarize this report...")
```

</div>

### Cost-Optimized Router Wrapper

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
class CostOptimizedLLM(BaseLLMWrapper):
    """
    Route requests to models based on complexity and cost.
    """

    MODEL_COSTS = {
        'claude-opus-4': {'input': 15.0, 'output': 75.0},  # per 1M tokens
        'claude-sonnet-4': {'input': 3.0, 'output': 15.0},
        'claude-haiku-3': {'input': 0.25, 'output': 1.25}
    }

    def __init__(
        self,
        connection_map: Dict[str, str],
        complexity_threshold_high: int = 5000,
        complexity_threshold_low: int = 1000
    ):
        """
        Args:
            connection_map: Dict mapping model tier to connection name
                          {'high': 'claude-opus', 'medium': 'claude-sonnet', 'low': 'claude-haiku'}
            complexity_threshold_high: Token count for high-complexity tasks
            complexity_threshold_low: Token count for low-complexity tasks
        """
        self.connections = {
            tier: LLM(conn)
            for tier, conn in connection_map.items()
        }
        self.threshold_high = complexity_threshold_high
        self.threshold_low = complexity_threshold_low

    def estimate_complexity(self, prompt: str) -> str:
        """
        Estimate task complexity to select appropriate model.

        Args:
            prompt: Input prompt

        Returns:
            Complexity tier: 'high', 'medium', or 'low'
        """
        # Rough token estimate
        estimated_tokens = len(prompt) // 4

        # Complexity indicators
        has_code = 'def ' in prompt or 'class ' in prompt or '```' in prompt
        has_analysis = any(
            word in prompt.lower()
            for word in ['analyze', 'compare', 'evaluate', 'assess']
        )
        has_reasoning = any(
            word in prompt.lower()
            for word in ['explain', 'justify', 'argue', 'prove']
        )

        # Scoring
        complexity_score = estimated_tokens

        if has_code:
            complexity_score += 1000
        if has_analysis:
            complexity_score += 500
        if has_reasoning:
            complexity_score += 500

        # Classify
        if complexity_score > self.threshold_high:
            return 'high'
        elif complexity_score > self.threshold_low:
            return 'medium'
        else:
            return 'low'

    def complete(self, prompt: str, force_tier: str = None, **kwargs) -> Any:
        """
        Route to appropriate model based on complexity.

        Args:
            prompt: Input prompt
            force_tier: Optional override for model selection
            **kwargs: LLM parameters

        Returns:
            LLM response
        """
        # Determine model tier
        tier = force_tier or self.estimate_complexity(prompt)

        logger.info(f"Routing to {tier} tier model")

        # Get appropriate connection
        connection = self.connections.get(tier, self.connections['medium'])

        # Execute
        processed_prompt, processed_kwargs = self.preprocess(prompt, **kwargs)
        response = connection.complete(processed_prompt, **processed_kwargs)
        result = self.postprocess(response)

        # Log cost
        logger.info(
            f"Used {tier} tier: {response.usage.total_tokens} tokens, "
            f"${response.cost:.4f}"
        )

        return result

# Usage
cost_optimized = CostOptimizedLLM(
    connection_map={
        'high': 'anthropic-claude-opus',
        'medium': 'anthropic-claude-sonnet',
        'low': 'anthropic-claude-haiku'
    }
)

# Simple extraction - routed to Haiku
result1 = cost_optimized.complete("Extract inventory change from: ...")

# Complex analysis - routed to Opus
result2 = cost_optimized.complete("""
Analyze the following 10,000 word report and provide:
1. Comprehensive supply-demand analysis
2. Multi-factor price modeling
3. Risk assessment with probabilities
...
""")

# Force specific tier
result3 = cost_optimized.complete("...", force_tier='high')
```

</div>

### Domain-Specific Wrapper

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
class CommodityAnalysisLLM(BaseLLMWrapper):
    """
    Domain-specific wrapper for commodity market analysis.
    """

    SYSTEM_PROMPT = """You are an expert commodity market analyst specializing in energy markets.

Your analysis should:
- Cite specific data points from source material
- Compare to historical averages when relevant
- Provide bullish/bearish/neutral sentiment with confidence
- Identify key supply and demand drivers
- Be objective and data-driven

Always return structured JSON output."""

    def __init__(self, connection_name: str):
        super().__init__(connection_name)

    def preprocess(self, prompt: str, **kwargs) -> tuple[str, Dict]:
        """Add domain-specific context."""
        # Build full prompt with system message
        full_prompt = f"{self.SYSTEM_PROMPT}\n\n{prompt}"

        # Set domain-appropriate parameters
        kwargs.setdefault('temperature', 0.3)  # Lower for factual analysis
        kwargs.setdefault('max_tokens', 1500)

        return full_prompt, kwargs

    def analyze_report(
        self,
        commodity: str,
        report_text: str,
        metrics: List[str] = None
    ) -> Dict:
        """
        Analyze a commodity market report.

        Args:
            commodity: Commodity type (crude_oil, natural_gas, etc.)
            report_text: Full report text
            metrics: Optional list of specific metrics to extract

        Returns:
            Structured analysis
        """
        metrics_str = ', '.join(metrics) if metrics else 'all relevant metrics'

        prompt = f"""Analyze this {commodity.replace('_', ' ')} market report:

{report_text}

Extract and analyze {metrics_str}.

Return JSON with:
- metrics: dict of extracted values with units
- sentiment: "bullish" | "bearish" | "neutral"
- confidence: 0-1 score
- key_factors: list of driving factors
- comparison_to_average: above/below/inline with historical
- outlook: short-term market outlook"""

        result = self.complete(prompt)

        # Parse JSON from response
        if isinstance(result, str):
            result = json.loads(result)

        return result

    def compare_reports(
        self,
        commodity: str,
        reports: List[Dict[str, str]]
    ) -> Dict:
        """
        Compare multiple reports for consensus and conflicts.

        Args:
            commodity: Commodity type
            reports: List of dicts with 'source' and 'content' keys

        Returns:
            Comparative analysis
        """
        reports_text = "\n\n".join([
            f"## {report['source']}\n{report['content']}"
            for report in reports
        ])

        prompt = f"""Compare these {commodity.replace('_', ' ')} market reports:

{reports_text}

Identify:
1. Consensus views (what all sources agree on)
2. Disagreements (conflicting data or interpretations)
3. Unique insights (mentioned by only one source)
4. Synthesized outlook (your unified view)

Return JSON."""

        result = self.complete(prompt)

        if isinstance(result, str):
            result = json.loads(result)

        return result

# Usage
commodity_llm = CommodityAnalysisLLM('anthropic-claude')

# Single report analysis
analysis = commodity_llm.analyze_report(
    commodity='crude_oil',
    report_text="U.S. commercial crude oil inventories decreased...",
    metrics=['inventory_change', 'production', 'imports']
)

print(f"Sentiment: {analysis['sentiment']}")
print(f"Key factors: {analysis['key_factors']}")

# Multi-report comparison
comparison = commodity_llm.compare_reports(
    commodity='natural_gas',
    reports=[
        {'source': 'EIA', 'content': 'Storage increased...'},
        {'source': 'IEA', 'content': 'Global supply...'},
        {'source': 'Industry', 'content': 'Winter demand...'}
    ]
)

print(f"Consensus: {comparison['consensus']}")
```

</div>

### Wrapper with Caching

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import hashlib
from typing import Optional

class CachedLLM(BaseLLMWrapper):
    """
    Wrapper with response caching to reduce costs.
    """

    def __init__(
        self,
        connection_name: str,
        cache_backend: Optional[Any] = None
    ):
        super().__init__(connection_name)
        self.cache = cache_backend or {}  # Simple dict cache

    def _cache_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key from prompt and parameters."""
        # Include relevant parameters in cache key
        cache_str = f"{prompt}||{kwargs.get('temperature', 0.7)}||{kwargs.get('max_tokens', 1000)}"
        return hashlib.sha256(cache_str.encode()).hexdigest()

    def complete(self, prompt: str, use_cache: bool = True, **kwargs) -> Any:
        """
        Generate completion with caching.

        Args:
            prompt: Input prompt
            use_cache: Whether to use cache
            **kwargs: LLM parameters

        Returns:
            LLM response (possibly from cache)
        """
        if use_cache:
            cache_key = self._cache_key(prompt, **kwargs)

            # Check cache
            if cache_key in self.cache:
                logger.info("Cache hit - returning cached response")
                return self.cache[cache_key]

        # Cache miss - call LLM
        result = super().complete(prompt, **kwargs)

        # Store in cache
        if use_cache:
            self.cache[cache_key] = result

        return result

# Usage with Redis cache (example)
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

cached_llm = CachedLLM(
    connection_name='anthropic-claude',
    cache_backend=redis_client
)

# First call - hits LLM
result1 = cached_llm.complete("Analyze this report...")

# Second call with identical prompt - hits cache
result2 = cached_llm.complete("Analyze this report...")  # Free!
```

</div>

## Common Pitfalls

**Pitfall 1: Not Handling Errors Gracefully**
- LLM calls can fail for many reasons (rate limits, timeouts, invalid responses)
- Always implement retry logic with exponential backoff
- Provide fallback behavior or meaningful error messages

**Pitfall 2: Tight Coupling to Specific Models**
- Hardcoding model-specific behavior makes wrappers inflexible
- Use configuration parameters for model-specific settings
- Abstract away provider differences in the wrapper

**Pitfall 3: Ignoring Token Limits**
- Not validating input size before sending to LLM causes errors
- Check prompt length and truncate or chunk if necessary
- Account for both input and output token limits

**Pitfall 4: No Logging or Monitoring**
- Wrappers without logging make debugging impossible
- Log all requests, responses, errors, and performance metrics
- Include correlation IDs to trace requests across systems

**Pitfall 5: Cache Invalidation Problems**
- Caching without proper invalidation serves stale data
- Include version identifiers in cache keys
- Implement TTL (time-to-live) for cached responses

## Connections

<div class="callout-info">
<strong>How this connects to the rest of the course:</strong>
</div>

**Builds on:**
- LLM Mesh setup and Python integration (Module 0)
- Prompt design principles (Module 1)

**Leads to:**
- Pipeline integration patterns (Module 3.3)
- Production deployment and monitoring (Module 4)

**Related to:**
- Design patterns (Strategy, Decorator)
- Middleware architectures
- Error handling and resilience patterns

## Practice Problems

1. **Validation Wrapper**
   - Build a wrapper that validates all outputs match a Pydantic schema
   - Retry up to 3 times with increasingly explicit instructions
   - Return structured Python objects instead of raw text

2. **Multi-Model Ensemble**
   - Create a wrapper that calls 3 different models in parallel
   - Aggregate their responses using majority voting or confidence weighting
   - Compare ensemble performance to individual models

3. **Budget-Aware Wrapper**
   - Implement a wrapper with daily budget tracking
   - Reject requests when budget exhausted
   - Provide cost estimates before execution
   - Generate end-of-day spending reports

4. **Streaming Response Wrapper**
   - Extend the base wrapper to support streaming responses
   - Process tokens as they arrive rather than waiting for completion
   - Implement early stopping if certain conditions are met

5. **Domain-Specific Wrapper**
   - For a domain of your choice (legal, medical, financial), create a specialized wrapper
   - Include domain-specific preprocessing, prompt templates, and validation
   - Add methods for common domain tasks

## Further Reading

- **Dataiku Documentation**: [Custom LLM Integrations](https://doc.dataiku.com/dss/latest/generative-ai/custom-llm.html) - Official guidance on extending LLM Mesh

- **Design Patterns**: "Gang of Four" Decorator and Strategy Patterns - Classic patterns applicable to LLM wrappers

- **Martin Fowler**: [API Gateway Pattern](https://martinfowler.com/articles/gateway-pattern.html) - Similar architectural pattern for API management

- **Blog Post**: "Production LLM Engineering Patterns" - Real-world wrapper patterns from industry (representative of current practices)

- **Research**: "LLM Middleware: Abstractions for Production Language Model Applications" - Emerging research on LLM infrastructure layers (representative of ongoing work)


## Resources

<a class="link-card" href="../notebooks/01_python_llm_calls.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
