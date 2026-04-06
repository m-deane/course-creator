# AI Engineer Cheatsheet

> **Reading time:** ~5 min | **Module:** 0 — AI Engineer Mindset | **Prerequisites:** None

<span class="badge mint">All Levels</span> <span class="badge amber">~5 min</span> <span class="badge blue">Module 0</span>

## The Core Mental Model

<div class="flow">
  <div class="flow-step mint">Goal</div>
  <div class="flow-arrow">&#8594;</div>
  <div class="flow-step amber">Context</div>
  <div class="flow-arrow">&#8594;</div>
  <div class="flow-step blue">Plan/Generate</div>
  <div class="flow-arrow">&#8594;</div>
  <div class="flow-step lavender">Act</div>
  <div class="flow-arrow">&#8594;</div>
  <div class="flow-step rose">Observe</div>
  <div class="flow-arrow">&#8594;</div>
  <div class="flow-step mint">Update Memory</div>
  <div class="flow-arrow">&#8594;</div>
  <div class="flow-step amber">Evaluate</div>
  <div class="flow-arrow">&#8594;</div>
  <div class="flow-step blue">Iterate</div>
</div>

<div class="callout-key">
<strong>Key Point:</strong> Whoever runs this loop faster and cleaner wins.
</div>

---

## The Full Stack

| Layer | Purpose | Key Technologies |
|-------|---------|------------------|
| **Evaluation** | Measure progress, prevent regression | Benchmarks, custom metrics, red-teaming |
| **Observability** | Monitor production behavior | Logging, tracing, dashboards |
| **Agent Loop** | Goal to action to observation cycle | ReAct, function calling |
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

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### Chinchilla Optimal Training

$$\text{Tokens} \approx 20 \times \text{Parameters}$$

### DPO Loss

$$\mathcal{L} = -\mathbb{E}\left[\log \sigma\left(\beta\left(\log \frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right)\right]$$

---

## Decision Trees

### When to Use RAG vs Fine-tuning

```
Need to add knowledge?
+-- Knowledge changes frequently --> RAG
+-- Knowledge is domain-specific but stable --> Fine-tune
+-- Need source attribution --> RAG
+-- Need to change behavior/style --> Fine-tune (SFT/DPO)
```

### Choosing Alignment Method

```
What do you have?
+-- Demonstration data (good examples) --> SFT
+-- Preference pairs (A > B) --> DPO
+-- Reward signal + compute budget --> RLHF
+-- Explicit principles --> Constitutional AI
```

### Memory Architecture

```
What kind of information?
+-- Current conversation --> Context window
+-- Documents/knowledge base --> RAG (vector DB)
+-- User preferences --> Long-term memory store
+-- Task state --> Working memory
+-- Needs updating in weights --> Model editing (rare)
```

---

## Common Patterns

### ReAct Agent Loop

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">react_pattern.txt</span>
</div>
<div class="code-body">

```
Thought: [reasoning about the task]
Action: [tool_name(parameters)]
Observation: [result from tool]
... repeat ...
Thought: [I can now answer]
Action: finish(answer)
```

</div>
</div>

### RAG Pipeline

<div class="flow">
  <div class="flow-step mint">Query</div>
  <div class="flow-arrow">&#8594;</div>
  <div class="flow-step amber">Embed</div>
  <div class="flow-arrow">&#8594;</div>
  <div class="flow-step blue">Retrieve</div>
  <div class="flow-arrow">&#8594;</div>
  <div class="flow-step lavender">Rerank</div>
  <div class="flow-arrow">&#8594;</div>
  <div class="flow-step rose">Generate with context</div>
</div>

### Production Deployment

<div class="flow">
  <div class="flow-step mint">Request</div>
  <div class="flow-arrow">&#8594;</div>
  <div class="flow-step amber">Cache check</div>
  <div class="flow-arrow">&#8594;</div>
  <div class="flow-step blue">Route to model</div>
  <div class="flow-arrow">&#8594;</div>
  <div class="flow-step lavender">Generate</div>
  <div class="flow-arrow">&#8594;</div>
  <div class="flow-step rose">Guardrails</div>
  <div class="flow-arrow">&#8594;</div>
  <div class="flow-step mint">Log & Respond</div>
</div>

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

<div class="compare">
  <div class="compare-card">
    <div class="header before">Don't</div>
    <div class="body">
      "Just scale the model." Fine-tune for everything. RLHF is always better. Long context solves memory. Benchmark scores = production quality. One agent does everything.
    </div>
  </div>
  <div class="compare-card">
    <div class="header after">Do Instead</div>
    <div class="body">
      Design the full system. Use RAG for knowledge, fine-tune for behavior. DPO is simpler and often sufficient. Hierarchical memory with intelligent retrieval. Task-specific evaluation + monitoring. Specialized agents with clear responsibilities.
    </div>
  </div>
</div>

---

## Quick Commands

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">quick_commands.sh</span>
</div>
<div class="code-body">

```bash
# Fine-tune with LoRA
python -m trl.scripts.sft --model_name meta-llama/Llama-2-7b-hf \
  --dataset_name your_dataset --peft_lora_r 8

# Start MCP server
mcp run server.py

# Run evaluation
lm_eval --model hf --model_args pretrained=your_model --tasks mmlu
```

</div>
</div>

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
