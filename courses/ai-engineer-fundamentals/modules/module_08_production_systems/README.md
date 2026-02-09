# Module 08: Production Systems - The Full Loop

> **"The future rewards the person who can build a system that keeps getting better after it ships."**

## Learning Objectives

By the end of this module, you will:
- Deploy LLM systems to production (Modal, Railway, cloud)
- Implement observability (logging, tracing, monitoring)
- Build feedback flywheels for continuous improvement
- Optimize costs without sacrificing quality
- Implement safety guardrails for production
- Know when NOT to use LLMs

## The Core Insight

Deploying an LLM is just the beginning. Production systems need:

```
┌─────────────────────────────────────────────────────────────────┐
│               THE PRODUCTION LLM STACK                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    FEEDBACK FLYWHEEL                     │   │
│  │  User feedback → Training data → Better model → Repeat  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ▲                                    │
│  ┌─────────────────────────┴───────────────────────────────┐   │
│  │                    OBSERVABILITY                         │   │
│  │  Logging │ Tracing │ Metrics │ Alerts │ Dashboards      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ▲                                    │
│  ┌─────────────────────────┴───────────────────────────────┐   │
│  │                    SAFETY LAYER                          │   │
│  │  Input filtering │ Output guardrails │ Rate limiting    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ▲                                    │
│  ┌─────────────────────────┴───────────────────────────────┐   │
│  │                    INFERENCE                             │   │
│  │  Load balancing │ Caching │ Batching │ Model routing    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ▲                                    │
│  ┌─────────────────────────┴───────────────────────────────┐   │
│  │                    THE MODEL                             │   │
│  │  (finally, the actual LLM)                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Module Contents

### Guides
| Guide | Description | Time |
|-------|-------------|------|
| [01_deployment_patterns.md](guides/01_deployment_patterns.md) | Modal, Railway, cloud options | 15 min |
| [02_observability.md](guides/02_observability.md) | What to log, trace, alert on | 15 min |
| [03_feedback_flywheel.md](guides/03_feedback_flywheel.md) | Data flywheels for improvement | 15 min |
| [04_cost_optimization.md](guides/04_cost_optimization.md) | Caching, model routing, batching | 15 min |
| [05_safety_production.md](guides/05_safety_production.md) | Guardrails, content filtering | 15 min |
| [06_when_not_to_use_llms.md](guides/06_when_not_to_use_llms.md) | Decision framework | 10 min |
| [cheatsheet.md](guides/cheatsheet.md) | Production checklist | 5 min |

### Notebooks
| Notebook | Description | Time |
|----------|-------------|------|
| [01_deploy_to_modal.ipynb](notebooks/01_deploy_to_modal.ipynb) | Ship an endpoint in 10 min | 15 min |
| [02_observability_setup.ipynb](notebooks/02_observability_setup.ipynb) | Logging and monitoring | 15 min |
| [03_feedback_collection.ipynb](notebooks/03_feedback_collection.ipynb) | Build feedback loops | 15 min |

## Key Concepts

### Deployment Patterns

| Pattern | Pros | Cons | Best For |
|---------|------|------|----------|
| **API Proxy** | Simple, no infra | Vendor lock-in, cost | Prototypes, low volume |
| **Serverless (Modal)** | Auto-scaling, pay-per-use | Cold starts | Variable traffic |
| **Container (Railway)** | Full control, always warm | Fixed costs | Predictable traffic |
| **Self-hosted** | Maximum control, lowest marginal cost | Ops burden | High volume, sensitive data |

### Cost Optimization Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                  COST OPTIMIZATION LAYERS                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. CACHING                                               │   │
│  │    • Semantic cache (similar queries → cached response)  │   │
│  │    • Exact match cache (identical queries)               │   │
│  │    • Embedding cache (don't re-embed same text)          │   │
│  │    Savings: 30-70% of requests                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 2. MODEL ROUTING                                         │   │
│  │    • Simple queries → smaller/cheaper model              │   │
│  │    • Complex queries → larger model                      │   │
│  │    • Classification model decides routing                │   │
│  │    Savings: 40-60% on model costs                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 3. PROMPT OPTIMIZATION                                   │   │
│  │    • Compress system prompts                             │   │
│  │    • Remove unnecessary context                          │   │
│  │    • Use structured outputs to reduce tokens             │   │
│  │    Savings: 20-40% on token costs                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 4. BATCHING                                              │   │
│  │    • Batch similar requests                              │   │
│  │    • Trade latency for throughput                        │   │
│  │    Savings: 20-50% on compute costs                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Observability Essentials

```python
import logging
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMRequest:
    request_id: str
    user_id: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    latency_ms: float
    cost: float
    success: bool
    error: Optional[str] = None

class LLMObserver:
    """Production observability for LLM calls."""

    def __init__(self):
        self.logger = logging.getLogger("llm")

    def log_request(self, req: LLMRequest):
        """Log every request with structured data."""
        self.logger.info(
            "llm_request",
            extra={
                "request_id": req.request_id,
                "user_id": req.user_id,
                "model": req.model,
                "prompt_tokens": req.prompt_tokens,
                "completion_tokens": req.completion_tokens,
                "total_tokens": req.prompt_tokens + req.completion_tokens,
                "latency_ms": req.latency_ms,
                "cost_usd": req.cost,
                "success": req.success,
                "error": req.error
            }
        )

    def track_metrics(self, req: LLMRequest):
        """Send metrics to monitoring system."""
        # Prometheus, DataDog, etc.
        metrics.histogram("llm_latency_ms", req.latency_ms, tags={"model": req.model})
        metrics.counter("llm_tokens_total", req.prompt_tokens + req.completion_tokens)
        metrics.counter("llm_cost_usd", req.cost)
        if not req.success:
            metrics.counter("llm_errors", 1, tags={"error": req.error})
```

### When NOT to Use LLMs

```
┌─────────────────────────────────────────────────────────────────┐
│              WHEN NOT TO USE LLMs                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ❌ DON'T USE LLMs FOR:                                         │
│                                                                 │
│  • Deterministic operations (math, lookups, transforms)         │
│    → Use code                                                   │
│                                                                 │
│  • Real-time, sub-10ms requirements                            │
│    → Use traditional ML or rules                               │
│                                                                 │
│  • Perfect accuracy requirements                                │
│    → LLMs are probabilistic; add verification layer            │
│                                                                 │
│  • Highly repetitive, templated content                        │
│    → Use templates with variable substitution                  │
│                                                                 │
│  • Simple classification with labeled data                     │
│    → Traditional ML is cheaper and faster                      │
│                                                                 │
│  • Tasks with clear rules                                      │
│    → Rule engines are more reliable                            │
│                                                                 │
│  ✅ USE LLMs FOR:                                               │
│                                                                 │
│  • Ambiguous, open-ended tasks                                 │
│  • Natural language understanding                              │
│  • Creative generation                                         │
│  • Complex reasoning over text                                 │
│  • When rules are hard to specify                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Production Checklist

Before shipping:
- [ ] **Observability**: Logging, metrics, tracing enabled
- [ ] **Error handling**: Graceful degradation, retries configured
- [ ] **Rate limiting**: Per-user and global limits set
- [ ] **Cost controls**: Spending alerts, budget caps
- [ ] **Safety**: Input filtering, output guardrails
- [ ] **Monitoring**: Dashboards, alerts configured
- [ ] **Feedback**: User feedback collection mechanism
- [ ] **Testing**: Regression tests, red-team tests passing
- [ ] **Documentation**: Runbooks, on-call procedures

## Templates

```
templates/
├── production_api_template.py     # Full production LLM API
├── monitoring_template.py         # Key metrics setup
├── feedback_collector_template.py # User feedback → data
└── guardrails_template.py         # Safety layer
```

## Prerequisites

- Modules 00-07 (recommended to complete full track)
- Cloud platform account (Modal, Railway, or AWS/GCP)
- Understanding of web APIs

## Portfolio Project Connection

This module directly supports:
- **Project 3: Production LLM System** - Apply everything in this module

## Time Estimate

- Quick path: 45 minutes (notebooks only)
- Full path: 2 hours (guides + notebooks)

---

*"The product moat is: closed-loop learning + trustworthy memory + tool execution + evaluation discipline. Not a single clever prompt."*
