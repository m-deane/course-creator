# AI Engineer Cheatsheet

## The Core Mental Model

```
Goal → Context → Plan/Generate → Act → Observe → Update Memory → Evaluate → Iterate
```

**Whoever runs this loop faster and cleaner wins.**

---

## The Full Stack

| Layer | Purpose | Key Technologies |
|-------|---------|------------------|
| **Evaluation** | Measure progress, prevent regression | Benchmarks, custom metrics, red-teaming |
| **Observability** | Monitor production behavior | Logging, tracing, dashboards |
| **Agent Loop** | Goal → action → observation cycle | ReAct, function calling |
| **Tools** | Actions beyond text | APIs, databases, code execution |
| **Memory** | Updated, relevant context | RAG, vector DBs, long-term stores |
| **Protocols** | Standardized tool integration | MCP |
| **Alignment** | Shape behavior | SFT, RLHF, DPO |
| **Transformer** | The engine | Attention, FFN, embeddings |

---

## Three Tracks at a Glance

| Track | Focus | Key Skills | Modules |
|-------|-------|------------|---------|
| **A: Model Core** | The engine | Attention, training, efficiency | 01, 06 |
| **B: Alignment** | Behavior shaping | SFT, DPO, evaluation | 02, 07 |
| **C: Agents** | World connection | RAG, tools, MCP | 03-05, 08 |

---

## Key Formulas

### Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

### Chinchilla Optimal Training
```
Tokens ≈ 20 × Parameters
```

### DPO Loss
```
L = -E[log σ(β(log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
```

---

## Decision Trees

### When to Use RAG vs Fine-tuning

```
Need to add knowledge?
├── Knowledge changes frequently → RAG
├── Knowledge is domain-specific but stable → Fine-tune
├── Need source attribution → RAG
└── Need to change behavior/style → Fine-tune (SFT/DPO)
```

### Choosing Alignment Method

```
What do you have?
├── Demonstration data (good examples) → SFT
├── Preference pairs (A > B) → DPO
├── Reward signal + compute budget → RLHF
└── Explicit principles → Constitutional AI
```

### Memory Architecture

```
What kind of information?
├── Current conversation → Context window
├── Documents/knowledge base → RAG (vector DB)
├── User preferences → Long-term memory store
├── Task state → Working memory
└── Needs updating in weights → Model editing (rare)
```

---

## Common Patterns

### ReAct Agent Loop
```
Thought: [reasoning about the task]
Action: [tool_name(parameters)]
Observation: [result from tool]
... repeat ...
Thought: [I can now answer]
Action: finish(answer)
```

### RAG Pipeline
```
Query → Embed → Retrieve → Rerank → Generate with context
```

### Production Deployment
```
Request → Cache check → Route to model → Generate → Guardrails → Log → Respond
```

---

## Efficiency Quick Reference

| Technique | Purpose | Memory Reduction | Speed Improvement |
|-----------|---------|------------------|-------------------|
| **LoRA** | Fine-tuning | Train <1% params | Similar to full |
| **QLoRA** | Fine-tune large models | 4-bit base model | Similar to full |
| **Quantization (INT4)** | Inference | 4x smaller | 2-4x faster |
| **FlashAttention** | Training/inference | 5-20x | 2-4x faster |
| **MoE** | Scaling | N/A (more total) | Same compute, more capacity |
| **KV Caching** | Inference | Trades memory for speed | Significant for long context |

---

## Evaluation Metrics

| Task Type | Metrics |
|-----------|---------|
| **General capability** | MMLU, HumanEval, MT-Bench |
| **RAG quality** | Retrieval precision/recall, answer faithfulness |
| **Agent tasks** | Success rate, cost per success, steps to complete |
| **Safety** | Refusal rate on harmful, compliance on benign |
| **Production** | Latency p50/p95/p99, cost per query, error rate |

---

## Canonical Papers (One-Line Summary)

| Paper | One-Line Summary |
|-------|------------------|
| **Attention Is All You Need** | Self-attention replaces recurrence, enables parallel training |
| **Chinchilla** | Train on 20 tokens per parameter for optimal efficiency |
| **InstructGPT** | RLHF turns base models into assistants |
| **DPO** | Simpler alternative to RLHF using direct preference optimization |
| **RAG** | Retrieve documents at inference time, don't memorize everything |
| **MemGPT** | Treat memory like an OS with paging and explicit management |
| **ReAct** | Interleave reasoning and action for interpretable agents |
| **LoRA** | Fine-tune with low-rank adapters, not full weights |
| **FlashAttention** | IO-aware attention reduces memory, increases speed |
| **MCP** | Standard protocol for tool integration |

---

## Anti-Patterns

| Don't | Do Instead |
|-------|------------|
| "Just scale the model" | Design the full system |
| "Fine-tune for everything" | Use RAG for knowledge, fine-tune for behavior |
| "RLHF is always better" | DPO is simpler and often sufficient |
| "Long context solves memory" | Hierarchical memory with intelligent retrieval |
| "Benchmark scores = production quality" | Task-specific evaluation + monitoring |
| "One agent does everything" | Specialized agents with clear responsibilities |

---

## Quick Commands

```bash
# Fine-tune with LoRA
python -m trl.scripts.sft --model_name meta-llama/Llama-2-7b-hf \
  --dataset_name your_dataset --peft_lora_r 8

# Quantize model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True)
)

# Start MCP server
mcp run server.py

# Run evaluation
lm_eval --model hf --model_args pretrained=your_model --tasks mmlu
```

---

## The AI Engineer's Checklist

Before shipping:
- [ ] Evaluation harness with regression tests
- [ ] Safety testing (red-teaming)
- [ ] Observability (logging, tracing, metrics)
- [ ] Cost estimation and optimization
- [ ] Fallback behavior for failures
- [ ] Feedback collection mechanism
- [ ] Documentation for maintenance

---

*"The future rewards the person who can build a system that keeps getting better after it ships."*
