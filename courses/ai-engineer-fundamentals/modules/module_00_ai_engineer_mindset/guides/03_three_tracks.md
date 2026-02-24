# Three Tracks of AI Engineering

## In Brief

AI Engineering divides into three complementary tracks: Model & Training Core (understanding and modifying the engine), Alignment & Safety (shaping behavior), and Agent Systems (connecting to reality). Mastering all three makes you a complete AI Engineer.

> 💡 **Key Insight:** You don't need to be a researcher to be an AI Engineer, but you need to understand enough of each track to make good architectural decisions.

## Visual Explanation

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     THE THREE TRACKS OF AI ENGINEERING                   │
└──────────────────────────────────────────────────────────────────────────┘

┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
│   TRACK A          │  │   TRACK B          │  │   TRACK C          │
│   MODEL & TRAINING │  │   ALIGNMENT &      │  │   AGENT SYSTEMS    │
│   CORE             │  │   SAFETY           │  │                    │
├────────────────────┤  ├────────────────────┤  ├────────────────────┤
│                    │  │                    │  │                    │
│  "Understand and   │  │  "Make it behave   │  │  "Connect it to    │
│   modify the       │  │   the way you      │  │   the real world"  │
│   engine"          │  │   want"            │  │                    │
│                    │  │                    │  │                    │
├────────────────────┤  ├────────────────────┤  ├────────────────────┤
│ • Transformers     │  │ • SFT              │  │ • RAG              │
│ • Attention        │  │ • RLHF             │  │ • Memory           │
│ • Training         │  │ • DPO              │  │ • Tool use         │
│ • Scaling laws     │  │ • Constitutional   │  │ • Protocols (MCP)  │
│ • Data quality     │  │ • Red-teaming      │  │ • Orchestration    │
│ • Efficiency       │  │ • Evaluation       │  │ • Observability    │
│                    │  │                    │  │                    │
├────────────────────┤  ├────────────────────┤  ├────────────────────┤
│ Modules: 01, 06    │  │ Modules: 02, 07    │  │ Modules: 03-05, 08 │
└────────────────────┘  └────────────────────┘  └────────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   COMPLETE AI ENGINEER │
                    │                        │
                    │   Can build end-to-end │
                    │   production systems   │
                    └────────────────────────┘
```

## Track A: Model & Training Core

### What It Covers

| Topic | Description | Depth Needed |
|-------|-------------|--------------|
| **Transformers** | Attention, FFN, normalization, architecture | Deep understanding |
| **Training** | Optimization, loss functions, stability | Working knowledge |
| **Scaling Laws** | Compute vs params vs data tradeoffs | Conceptual grasp |
| **Data Quality** | Filtering, dedup, balance, contamination | Practical skills |
| **Efficiency** | LoRA, quantization, FlashAttention, MoE | Implementation level |
| **Distributed** | ZeRO, FSDP, tensor parallelism | Awareness + some hands-on |

### Key Skills

```python
# Track A engineers can:

# 1. Load and inspect models
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# 2. Understand attention patterns
attention_weights = model.get_attention_weights(input_ids)
visualize_attention(attention_weights)

# 3. Fine-tune efficiently
from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)

# 4. Quantize for deployment
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# 5. Reason about compute budgets
tokens_needed = 20 * num_parameters  # Chinchilla optimal
```

### When You Need Track A

- Fine-tuning models for specific domains
- Optimizing inference cost and latency
- Debugging unexpected model behavior
- Choosing between model sizes
- Understanding capability limits

### Career Focus

- ML Engineers at AI labs
- Model optimization specialists
- Training infrastructure engineers
- Research engineers

---

## Track B: Alignment & Safety

### What It Covers

| Topic | Description | Depth Needed |
|-------|-------------|--------------|
| **SFT** | Instruction tuning, format learning | Implementation level |
| **RLHF** | Reward models, PPO, preference learning | Conceptual + some hands-on |
| **DPO** | Direct preference optimization | Implementation level |
| **Constitutional AI** | Principle-based self-alignment | Awareness |
| **Safety Policies** | Refusal behavior, harm prevention | Deep practical knowledge |
| **Red-teaming** | Adversarial testing, jailbreaks | Practical skills |
| **Evaluation** | Metrics, benchmarks, regression testing | Deep practical knowledge |

### Key Skills

```python
# Track B engineers can:

# 1. Create instruction-tuning datasets
training_data = [
    {"instruction": "Summarize this article",
     "input": article_text,
     "output": good_summary},
    # ...
]

# 2. Implement preference optimization
from trl import DPOTrainer
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=preference_pairs,  # (prompt, chosen, rejected)
    beta=0.1,  # KL penalty weight
)

# 3. Design safety evaluations
def test_harmful_request_handling(model):
    harmful_prompts = load_red_team_prompts()
    results = []
    for prompt in harmful_prompts:
        response = model.generate(prompt)
        results.append({
            "prompt": prompt,
            "response": response,
            "refused": detect_refusal(response),
            "harmful_content": detect_harm(response)
        })
    return calculate_safety_metrics(results)

# 4. Build regression test suites
def regression_test(model_v1, model_v2, test_cases):
    for case in test_cases:
        v1_result = model_v1.generate(case.input)
        v2_result = model_v2.generate(case.input)
        assert evaluate(v2_result) >= evaluate(v1_result), \
            f"Regression detected on: {case.name}"
```

### When You Need Track B

- Building customer-facing AI products
- Ensuring brand-safe responses
- Meeting compliance requirements
- Reducing support escalations
- Preventing harmful outputs

### Career Focus

- AI Safety engineers
- Product AI engineers
- Trust & Safety teams
- AI policy and governance

---

## Track C: Agent Systems

### What It Covers

| Topic | Description | Depth Needed |
|-------|-------------|--------------|
| **RAG** | Retrieval, embeddings, reranking | Deep implementation |
| **Memory** | Short/long-term, lifecycle management | Deep implementation |
| **Tool Use** | Function calling, API integration | Deep implementation |
| **Agent Loops** | ReAct, planning, execution | Deep implementation |
| **Protocols** | MCP, standardized interfaces | Implementation level |
| **Orchestration** | Multi-agent coordination | Working knowledge |
| **Observability** | Logging, tracing, debugging | Practical skills |

### Key Skills

```python
# Track C engineers can:

# 1. Build production RAG systems
class RAGSystem:
    def __init__(self):
        self.embedder = EmbeddingModel("text-embedding-3-small")
        self.vector_db = ChromaDB(collection="documents")
        self.reranker = CrossEncoderReranker()

    def query(self, question: str, k: int = 5) -> list[Document]:
        # Embed query
        query_embedding = self.embedder.embed(question)

        # Retrieve candidates (retrieve more, rerank to k)
        candidates = self.vector_db.search(query_embedding, k=k*3)

        # Rerank for relevance
        reranked = self.reranker.rerank(question, candidates, k=k)

        return reranked

# 2. Implement agent loops
class ReActAgent:
    def run(self, goal: str) -> str:
        history = []

        for step in range(self.max_steps):
            # Generate thought + action
            thought, action = self.think_and_act(goal, history)

            # Execute action
            if action.type == "finish":
                return action.value

            observation = self.execute(action)

            # Update history
            history.append((thought, action, observation))

        return "Max steps reached"

# 3. Design tool interfaces
@tool(description="Search the web for current information")
def web_search(query: str, num_results: int = 5) -> list[SearchResult]:
    """
    Args:
        query: The search query
        num_results: Number of results to return (default 5)

    Returns:
        List of search results with title, url, and snippet
    """
    return search_api.search(query, num_results)

# 4. Build with protocols (MCP)
from mcp import Server, Tool

server = Server("my-tools")

@server.tool()
def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    return weather_api.get(city)

# 5. Add observability
@traced("agent.run")
def run_agent(goal: str):
    with span("context_building"):
        context = build_context(goal)

    with span("generation"):
        response = generate(goal, context)

    log_metrics({
        "goal": goal,
        "context_tokens": count_tokens(context),
        "response_tokens": count_tokens(response),
        "latency_ms": elapsed_ms()
    })

    return response
```

### When You Need Track C

- Building AI-powered applications
- Integrating LLMs with existing systems
- Creating autonomous agents
- Connecting to enterprise data
- Scaling to production

### Career Focus

- AI Application engineers
- Full-stack AI developers
- Platform engineers
- Startup founders

---

## How the Tracks Combine

### Example: Building a Customer Support Bot

```
Track A Knowledge Used:
- Choose model size based on latency/quality tradeoff
- Quantize for inference efficiency
- Understand why certain queries confuse the model

Track B Knowledge Used:
- Fine-tune on support conversations (SFT)
- Ensure brand-appropriate tone
- Build refusal behavior for out-of-scope requests
- Create regression tests for common issues

Track C Knowledge Used:
- RAG for product documentation
- Tool integration for order lookup, refunds
- Memory for conversation continuity
- Logging for debugging and improvement
```

### Overlap Diagram

```
                    ┌─────────────────────┐
                    │      Track A        │
                    │   Model & Training  │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
    ┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
    │ Fine-tuning for │ │  Efficient  │ │   Embedding     │
    │   alignment     │ │  inference  │ │   models for    │
    │   (A ∩ B)       │ │  (A ∩ C)    │ │   RAG (A ∩ C)   │
    └─────────────────┘ └─────────────┘ └─────────────────┘
              │                │                │
              ▼                │                ▼
    ┌─────────────────┐        │        ┌─────────────────┐
    │     Track B     │        │        │     Track C     │
    │   Alignment &   │◄───────┴───────►│  Agent Systems  │
    │     Safety      │                 │                 │
    └─────────────────┘                 └─────────────────┘
              │                                 │
              └─────────────┬───────────────────┘
                            ▼
                  ┌─────────────────────┐
                  │  Evaluation of      │
                  │  agent behavior     │
                  │  (B ∩ C)            │
                  └─────────────────────┘
```

## Recommended Learning Paths

### Path 1: Application Developer (Track C → B → A)

```
Week 1-2: Module 03 (Memory) + Module 04 (Tools)
Week 3:   Module 05 (MCP) + Module 08 (Production)
Week 4:   Module 02 (Alignment) + Module 07 (Evaluation)
Week 5-6: Module 01 (Transformer) + Module 06 (Efficiency)
```

**Rationale:** Get building quickly, then understand why things work.

### Path 2: ML Engineer (Track A → B → C)

```
Week 1-2: Module 01 (Transformer) + Module 06 (Efficiency)
Week 3:   Module 02 (Alignment) + Module 07 (Evaluation)
Week 4-5: Module 03 (Memory) + Module 04 (Tools)
Week 6:   Module 05 (MCP) + Module 08 (Production)
```

**Rationale:** Understand the engine deeply, then apply to systems.

### Path 3: Product/Startup (All tracks parallel)

```
Week 1: Module 00 (Mindset) + Quick-starts from each track
Week 2: Module 03 (Memory) - enough to build RAG
Week 3: Module 04 (Tools) - enough to build agents
Week 4: Module 02 (Alignment) - enough to tune behavior
Week 5: Module 08 (Production) - deployment focus
Week 6: Deep dive into whatever bottleneck you hit
```

**Rationale:** Get to production fast, fill gaps as needed.

## Self-Assessment

### Track A Skills Check
- [ ] Can explain attention mechanism intuitively and mathematically
- [ ] Can fine-tune a model using LoRA
- [ ] Can quantize a model for deployment
- [ ] Understand compute-optimal training (Chinchilla)
- [ ] Can debug training instabilities

### Track B Skills Check
- [ ] Can create and curate instruction-tuning datasets
- [ ] Can implement DPO training
- [ ] Can design safety evaluations
- [ ] Can build regression test suites
- [ ] Understand RLHF conceptually

### Track C Skills Check
- [ ] Can build production RAG with retrieval evaluation
- [ ] Can implement ReAct-style agent loops
- [ ] Can design reliable tool interfaces
- [ ] Can add observability to agent systems
- [ ] Can deploy and monitor in production

## Connections

- **Builds on:** The closed-loop mental model
- **Leads to:** Detailed exploration of each track in subsequent modules

## Practice Problems

1. **Self-assessment:** Rate yourself 1-5 on each skill in the three checklists above. Where are your biggest gaps?

2. **System design:** Design a system for "AI-powered code review." Which track's knowledge is most critical? Which can you start with minimal depth?

3. **Career planning:** Based on your current role and goals, which learning path makes most sense for you?
