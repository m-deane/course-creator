# Portfolio Project 3: Production LLM System

> **Build and deploy a complete production-grade LLM system with the full loop.**

## What You'll Build

A production-ready system that combines everything from the course:
- Custom alignment (SFT on domain data)
- RAG with hybrid retrieval
- Multi-agent orchestration via MCP
- Evaluation harness with regression tests
- Observability and feedback collection
- Cost optimization

**This is your capstone project.** It demonstrates mastery of the full AI Engineer stack.

## Learning Goals

By completing this project, you will demonstrate:
- End-to-end system design
- Integration of alignment, memory, tools, and evaluation
- Production deployment and monitoring
- Continuous improvement via feedback loops
- Cost-quality tradeoff optimization

## Requirements

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                PRODUCTION SYSTEM COMPONENTS                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. ALIGNMENT LAYER                                        │   │
│  │    • Fine-tune base model on domain data (LoRA/SFT)      │   │
│  │    • Or: Implement domain-specific system prompts        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 2. MEMORY LAYER                                           │   │
│  │    • RAG with vector DB                                   │   │
│  │    • Hybrid retrieval (vector + keyword)                 │   │
│  │    • Long-term user/session memory                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 3. AGENT LAYER                                            │   │
│  │    • Multi-tool agent with MCP integration               │   │
│  │    • Error handling and recovery                         │   │
│  │    • Task decomposition for complex queries              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 4. EVALUATION LAYER                                       │   │
│  │    • Automated regression tests                          │   │
│  │    • Quality metrics (accuracy, latency, cost)           │   │
│  │    • Red-team test suite                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 5. PRODUCTION LAYER                                       │   │
│  │    • API with rate limiting and auth                     │   │
│  │    • Observability (logs, metrics, traces)               │   │
│  │    • Feedback collection                                 │   │
│  │    • Cost optimization (caching, model routing)          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Functional Requirements

- [ ] Domain-specific knowledge (RAG or fine-tuning)
- [ ] At least 3 integrated tools via MCP
- [ ] Conversation memory across sessions
- [ ] User feedback collection
- [ ] Admin dashboard for monitoring

### Technical Requirements

- [ ] Deployed API with authentication
- [ ] Structured logging with request tracing
- [ ] Metrics dashboard (latency, cost, errors)
- [ ] Automated regression test suite
- [ ] CI/CD pipeline for updates

### Quality Requirements

- [ ] Response latency p95 < 5 seconds
- [ ] Error rate < 1%
- [ ] Cost per query < $0.05 average
- [ ] Regression tests pass before deploy
- [ ] 99% uptime

## Suggested Domain Options

Choose a domain that interests you:

| Domain | Knowledge Source | Tools Needed |
|--------|------------------|--------------|
| **Legal Assistant** | Legal documents, case law | Document search, citation lookup, draft generator |
| **Developer Helper** | Code repos, documentation | Code search, linter, test runner |
| **Research Assistant** | Papers, datasets | ArXiv search, data analysis, citation manager |
| **Customer Support** | Product docs, FAQ | Ticket search, knowledge base, escalation |
| **Financial Analyst** | Market data, reports | Stock API, calculator, report generator |

## Project Structure

```
project_3_production_system/
├── README.md
├── requirements.txt
├── docker-compose.yml
│
├── src/
│   ├── __init__.py
│   │
│   ├── alignment/
│   │   ├── __init__.py
│   │   ├── fine_tune.py      # LoRA training script
│   │   └── prompts.py        # Domain-specific prompts
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── rag.py            # RAG pipeline
│   │   ├── hybrid_search.py  # Vector + keyword
│   │   └── session_memory.py # Per-user memory
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── orchestrator.py   # Main agent coordinator
│   │   └── specialists/      # Domain-specific sub-agents
│   │
│   ├── mcp_servers/
│   │   ├── __init__.py
│   │   ├── knowledge_server.py
│   │   └── tools_server.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI app
│   │   ├── routes/
│   │   ├── middleware/       # Auth, rate limiting, logging
│   │   └── models.py         # Pydantic models
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── regression_suite.py
│   │   ├── red_team_suite.py
│   │   └── metrics.py
│   │
│   └── observability/
│       ├── __init__.py
│       ├── logging.py
│       ├── metrics.py
│       └── tracing.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── deploy/
│   ├── Dockerfile
│   ├── modal/
│   ├── kubernetes/
│   └── terraform/
│
├── dashboards/
│   ├── grafana/
│   └── admin_ui/
│
└── docs/
    ├── architecture.md
    ├── api.md
    ├── deployment.md
    └── runbook.md
```

## Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Set up project structure
- [ ] Implement basic RAG pipeline
- [ ] Create API skeleton with auth
- [ ] Set up development environment

### Phase 2: Core Features (Week 2)
- [ ] Build MCP servers for tools
- [ ] Implement agent orchestrator
- [ ] Add session memory
- [ ] Create basic evaluation suite

### Phase 3: Production Hardening (Week 3)
- [ ] Add observability (logs, metrics)
- [ ] Implement cost optimization (caching)
- [ ] Build admin dashboard
- [ ] Set up CI/CD

### Phase 4: Polish & Deploy (Week 4)
- [ ] Run full evaluation suite
- [ ] Red-team testing
- [ ] Deploy to production
- [ ] Write documentation

## Key Implementation Details

### Cost Optimization Layer

```python
class CostOptimizer:
    """Optimize costs while maintaining quality."""

    def __init__(self):
        self.cache = SemanticCache()
        self.router = ModelRouter()

    async def process_query(self, query: str, context: dict) -> dict:
        # 1. Check cache
        cached = await self.cache.get(query)
        if cached:
            return {"response": cached, "source": "cache", "cost": 0}

        # 2. Route to appropriate model
        complexity = self.router.estimate_complexity(query)
        model = self.router.select_model(complexity)

        # 3. Generate response
        response = await self.generate(query, context, model)

        # 4. Cache result
        await self.cache.set(query, response)

        return {
            "response": response,
            "source": model,
            "cost": self.estimate_cost(response, model)
        }
```

### Feedback Flywheel

```python
class FeedbackCollector:
    """Collect and process user feedback."""

    async def record_feedback(self, request_id: str, feedback: dict):
        """Record user feedback for a response."""
        await self.db.insert({
            "request_id": request_id,
            "rating": feedback.get("rating"),
            "comment": feedback.get("comment"),
            "timestamp": datetime.now()
        })

        # Flag for review if negative
        if feedback.get("rating", 5) < 3:
            await self.queue_for_review(request_id)

    async def generate_training_data(self) -> list:
        """Generate training data from positive feedback."""
        positive = await self.db.query({"rating": {"$gte": 4}})
        return [
            {
                "prompt": p["original_query"],
                "completion": p["response"],
                "source": "user_feedback"
            }
            for p in positive
        ]
```

## Evaluation Criteria

| Criterion | Weight | Passing |
|-----------|--------|---------|
| System completeness | 20% | All 5 layers implemented |
| Code quality | 15% | Clean, tested, documented |
| Production readiness | 20% | Deployed, monitored, reliable |
| Evaluation suite | 15% | Comprehensive regression + red-team |
| Performance | 15% | Meets latency, cost, error targets |
| Documentation | 15% | Architecture, API, runbook complete |

## Stretch Goals

- [ ] A/B testing framework for model comparisons
- [ ] Automatic fine-tuning from feedback data
- [ ] Multi-tenant support
- [ ] WebSocket streaming responses
- [ ] Mobile SDK
- [ ] Slack/Discord integration

## Resources

- All course modules (this project integrates everything)
- [Paper Summaries](../../resources/paper_summaries.md)
- [Production Systems Module](../../modules/module_08_production_systems/)

## Submission

When complete:
1. Deploy to production environment
2. Share API documentation
3. Provide dashboard access
4. Write architecture decision record
5. Record a 5-minute demo video

---

*"This project is your proof of mastery. A working production system demonstrates that you can build the full loop - not just understand the theory."*
