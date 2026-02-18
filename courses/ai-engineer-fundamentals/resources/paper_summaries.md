# Canonical Paper Summaries: The AI Engineer's Reading List

> Every paper that shaped modern LLM systems, summarized for practitioners.

## How to Use This Guide

Each summary includes:
- **TL;DR** - One sentence summary
- **The Problem** - What issue this paper addressed
- **The Solution** - Key innovation
- **Key Insight** - The conceptual breakthrough
- **Practical Impact** - How this affects your work as an AI Engineer
- **Read This If** - When to dive deeper

---

## 1. Transformer Architecture

### Attention Is All You Need (Vaswani et al., 2017)
**ArXiv:** [1706.03762](https://arxiv.org/abs/1706.03762)

**TL;DR:** Replaces recurrence with self-attention, enabling parallel training and better long-range dependencies.

**The Problem:** RNNs and LSTMs process sequences step-by-step, making them slow to train and prone to forgetting long-range information.

**The Solution:** Self-attention mechanism that computes relationships between all positions simultaneously:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Key Insight:** Attention is a learnable routing mechanism. Instead of fixed sequential processing, the model learns which positions to attend to based on content. Multi-head attention lets the model attend to different aspects simultaneously.

**Architecture Innovation:**
- **Encoder-Decoder** structure with stacked layers
- **Multi-head attention** - multiple attention patterns in parallel
- **Positional encoding** - sinusoidal functions to encode position
- **Layer normalization** and **residual connections** for training stability

**Practical Impact:**
- Foundation of ALL modern LLMs (GPT, Claude, Llama, etc.)
- Enables massive parallelization during training
- Quadratic complexity with sequence length (O(n²)) - drives need for efficiency innovations
- The "backbone" you're building systems around

**Read This If:** You want to truly understand how the engine works, not just use it.

---

### Transformer-XL (Dai et al., 2019)
**ArXiv:** [1901.02860](https://arxiv.org/abs/1901.02860)

**TL;DR:** Extends context beyond fixed windows using segment-level recurrence and relative positional encoding.

**The Problem:** Standard Transformers have a fixed context window. Information cannot flow across segment boundaries during training.

**The Solution:**
- **Segment-level recurrence**: Cache hidden states from previous segments
- **Relative positional encoding**: Position is relative to current token, not absolute

**Key Insight:** You can have the parallelism benefits of Transformers while maintaining information flow across arbitrary lengths.

**Practical Impact:**
- Influenced long-context architectures
- Ideas incorporated into modern models for handling longer sequences
- Foundation for understanding context extension techniques

**Read This If:** You're working on long-context applications or need to understand context window limitations.

---

## 2. Scaling Laws

### Scaling Laws for Neural Language Models (Kaplan et al., 2020)
**ArXiv:** [2001.08361](https://arxiv.org/abs/2001.08361)

**TL;DR:** Model performance is predictable from scale - larger models trained on more data perform better in a smooth, power-law relationship.

**The Problem:** How should we allocate compute budget? More parameters? More data? Longer training?

**The Solution:** Empirical discovery of scaling laws:
```
L(N) ∝ N^(-0.076)  # Loss scales with parameters
L(D) ∝ D^(-0.095)  # Loss scales with data
L(C) ∝ C^(-0.050)  # Loss scales with compute
```

**Key Insight:** Performance is remarkably predictable. You can forecast how well a model will perform before training it, based on scale alone. This enables rational compute allocation.

**Critical Finding:** At the time, this suggested prioritizing model size over training duration - train larger models for fewer steps.

**Practical Impact:**
- Justifies investment in larger models
- Enables planning training runs based on target performance
- Explains why "just scale it" worked for years
- Later refined by Chinchilla (see below)

**Read This If:** You're planning training runs or need to justify compute budgets.

---

### Training Compute-Optimal Large Language Models (Hoffmann et al., 2022) - "Chinchilla"
**ArXiv:** [2203.15556](https://arxiv.org/abs/2203.15556)

**TL;DR:** Most large models are undertrained. For optimal performance per compute, scale data and parameters equally.

**The Problem:** Previous scaling laws suggested prioritizing parameters. But is this compute-optimal?

**The Solution:** For a given compute budget C, optimal allocation is:
```
N_opt ∝ C^0.5  # Parameters
D_opt ∝ C^0.5  # Tokens
```
Train on ~20 tokens per parameter for optimal efficiency.

**Key Insight:** GPT-3 (175B parameters, 300B tokens) was undertrained by 4x. A 70B model trained on 1.4T tokens (Chinchilla) outperformed it.

**The Chinchilla Rule:** Tokens ≈ 20 × Parameters

**Practical Impact:**
- Shifted industry toward smaller, better-trained models
- Llama models followed this insight
- Changes how you think about model selection: a well-trained smaller model can beat an undertrained larger one
- Inference cost implications: smaller models are cheaper to serve

**Read This If:** You're deciding between model sizes or planning training data collection.

---

## 3. Alignment & Preference Learning

### Training Language Models to Follow Instructions with Human Feedback (Ouyang et al., 2022) - "InstructGPT"
**ArXiv:** [2203.02155](https://arxiv.org/abs/2203.02155)

**TL;DR:** RLHF transforms a base model into a helpful assistant by training on human preferences.

**The Problem:** Base models predict likely continuations, not helpful responses. They can be toxic, unhelpful, or confidently wrong.

**The Solution:** Three-stage process:
1. **Supervised Fine-Tuning (SFT)**: Train on demonstration data (instruction → good response)
2. **Reward Model Training**: Train a model to predict human preferences between responses
3. **RL Optimization**: Use PPO to optimize the policy against the reward model with KL constraint

```
objective = E[r(x,y)] - β * KL(π || π_ref)
```

**Key Insight:** Preferences are easier to collect than demonstrations. Humans can say "A is better than B" even when they can't write the perfect response.

**The RLHF Loop:**
```
Prompt → Generate multiple responses → Human ranks them → Train reward model → Optimize policy
```

**Practical Impact:**
- The technique behind ChatGPT, Claude, and all modern assistants
- Enables shaping behavior: helpfulness, harmlessness, honesty
- Creates the "assistant persona" we expect from LLMs
- Foundation for all subsequent alignment work

**Read This If:** You need to understand how models become assistants, or you're doing any alignment work.

---

### Deep Reinforcement Learning from Human Preferences (Christiano et al., 2017)
**ArXiv:** [1706.03741](https://arxiv.org/abs/1706.03741)

**TL;DR:** You can train RL agents from human preference comparisons instead of hand-coded reward functions.

**The Problem:** Designing reward functions is hard. Humans know what they want but can't always specify it mathematically.

**The Solution:** Learn a reward function from human comparisons:
1. Agent takes actions, creates trajectory clips
2. Humans compare clips: "Which is better?"
3. Train reward model on preferences
4. Optimize agent against learned reward

**Key Insight:** Preference learning separates "what we want" from "how to specify it". This is the conceptual foundation of RLHF.

**Practical Impact:**
- Theoretical foundation for InstructGPT/RLHF
- Shows preferences are a viable learning signal
- Influenced all subsequent preference-based training

**Read This If:** You want to understand the theoretical roots of RLHF.

---

### Direct Preference Optimization (Rafailov et al., 2023)
**ArXiv:** [2305.18290](https://arxiv.org/abs/2305.18290)

**TL;DR:** Skip the reward model - optimize preferences directly in a simpler, more stable training objective.

**The Problem:** RLHF is complex: train reward model, then run PPO with KL constraints. Lots of hyperparameters, unstable training.

**The Solution:** Reparameterize the RLHF objective to eliminate the explicit reward model:
```
L_DPO = -E[log σ(β log(π(y_w|x)/π_ref(y_w|x)) - β log(π(y_l|x)/π_ref(y_l|x)))]
```

Where y_w = preferred response, y_l = dispreferred response.

**Key Insight:** The optimal policy under RLHF has a closed-form solution. You can directly optimize toward it without intermediate reward modeling.

**DPO vs RLHF:**
| Aspect | RLHF | DPO |
|--------|------|-----|
| Complexity | High (reward model + RL) | Low (single objective) |
| Stability | Requires careful tuning | More stable |
| Compute | More expensive | Cheaper |
| Performance | Strong | Comparable |

**Practical Impact:**
- Simpler alignment pipeline
- Easier to implement and debug
- Widely adopted for fine-tuning
- Often the default choice for practical alignment

**Read This If:** You're implementing alignment and want the simpler path.

---

### Constitutional AI: Harmlessness from AI Feedback (Bai et al., 2022)
**ArXiv:** [2212.08073](https://arxiv.org/abs/2212.08073)

**TL;DR:** Use explicit principles and AI-generated feedback to align models, reducing dependence on human labels.

**The Problem:** Human feedback is expensive and doesn't scale. How do you align models with less human involvement?

**The Solution:** Two phases:
1. **Supervised Learning from Principles**: Model critiques its own outputs against a "constitution" (explicit principles), then revises
2. **RL from AI Feedback (RLAIF)**: Use the model's own judgments (guided by principles) as the reward signal

**The Constitution:** A set of explicit principles like:
- "Choose the response that is most helpful"
- "Choose the response that is least harmful"
- "Choose the response that is most honest"

**Key Insight:** You can bootstrap alignment by having the model apply principles to itself. This creates a "self-improvement" loop.

**Practical Impact:**
- Reduces human labeling costs
- Makes alignment more transparent (explicit principles)
- Enables scaling alignment to new domains
- Influenced Claude's development

**Read This If:** You're interested in scalable alignment or explicit value specification.

---

## 4. Memory & Retrieval

### Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)
**ArXiv:** [2005.11401](https://arxiv.org/abs/2005.11401)

**TL;DR:** Combine a retriever with a generator - retrieve relevant documents at inference time, then generate based on them.

**The Problem:**
- Knowledge in model weights is static (expensive to update)
- Models hallucinate facts they don't know
- Long-tail knowledge is poorly represented

**The Solution:** RAG architecture:
```
Query → Retriever → Top-k Documents → Generator → Response
```

Components:
- **Retriever**: Dense Passage Retrieval (DPR) using BERT embeddings
- **Generator**: BART seq2seq model
- **End-to-end training**: Both components trained jointly

**Key Insight:** Separate "what to know" (retriever) from "how to express it" (generator). This makes knowledge updatable without retraining.

**RAG Advantages:**
- **Freshness**: Update documents, not weights
- **Traceability**: Can cite sources
- **Domain adaptation**: Add domain docs without fine-tuning
- **Reduced hallucination**: Ground generation in retrieved facts

**Practical Impact:**
- Foundation of most production LLM systems
- Enables enterprise deployment (connect to internal docs)
- The "memory system" for modern agents
- Start here for any knowledge-intensive application

**Read This If:** You're building any system that needs to access external knowledge (which is most systems).

---

### MemGPT: Towards LLMs as Operating Systems (Packer et al., 2023)
**ArXiv:** [2310.08560](https://arxiv.org/abs/2310.08560)

**TL;DR:** Treat LLM memory like an operating system - with hierarchical storage, paging, and managed memory lifecycles.

**The Problem:** Context windows are finite. How do agents maintain coherent long-term memory across sessions?

**The Solution:** OS-inspired memory hierarchy:
- **Main Context**: Limited "RAM" (the context window)
- **Archival Memory**: Unlimited "disk" storage (vector DB)
- **Recall Memory**: Retrieved conversation history
- **Memory Management Functions**: The LLM explicitly calls functions to read/write memory

**Key Insight:** Give the model explicit control over memory operations. It decides what to store, retrieve, and forget - like a program managing its own memory.

**Memory Operations:**
```python
core_memory_append(content)    # Add to working memory
core_memory_replace(old, new)  # Update working memory
archival_memory_insert(content) # Save to long-term
archival_memory_search(query)  # Retrieve from long-term
```

**Practical Impact:**
- Framework for building agents with persistent memory
- Solves the "goldfish memory" problem in chatbots
- Model for how to structure agent memory systems
- Influenced modern agent frameworks

**Read This If:** You're building agents that need to remember across sessions or handle very long contexts.

---

### Locating and Editing Factual Associations in GPT (Meng et al., 2022) - "ROME"
**ArXiv:** [2202.05262](https://arxiv.org/abs/2202.05262)

**TL;DR:** You can surgically edit specific facts in a model's weights without retraining.

**The Problem:** RAG doesn't always work - sometimes you need to change what the model "believes". But retraining is expensive.

**The Solution:** Locate where facts are stored (in MLP layers), then directly edit those weights:
1. **Causal tracing**: Find which layers store the fact
2. **Rank-one model editing**: Modify specific weights to change the association

**Key Insight:** Factual associations are surprisingly localized in specific MLP layers. You can edit them like entries in a database.

**Example:**
- Before: "The Eiffel Tower is in Paris"
- Edit: Change "Paris" → "Rome"
- After: Model consistently says "The Eiffel Tower is in Rome"

**Practical Impact:**
- Enables fact correction without retraining
- Understanding of where knowledge lives in models
- Useful for specific corrections (not mass editing)
- Foundation for interpretability research

**Read This If:** You need to understand knowledge storage in models or make targeted corrections.

---

### Mass-Editing Memory in a Transformer (Meng et al., 2022) - "MEMIT"
**ArXiv:** [2210.07229](https://arxiv.org/abs/2210.07229)

**TL;DR:** Edit thousands of facts simultaneously while preserving model behavior on unrelated inputs.

**The Problem:** ROME edits one fact at a time. Doesn't scale to mass corrections.

**The Solution:**
- Spread edits across multiple layers (not just one)
- Optimize for minimal interference with unrelated facts
- Batch processing of thousands of edits

**Key Insight:** Distributing edits across layers prevents catastrophic interference and maintains model coherence.

**Practical Impact:**
- Scalable knowledge updating
- Useful for mass corrections (e.g., updating outdated facts)
- Still experimental for production use
- RAG is usually more practical, but MEMIT shows what's possible

**Read This If:** You need mass fact correction or are researching knowledge editing.

---

## 5. Agents & Tool Use

### ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)
**ArXiv:** [2210.03629](https://arxiv.org/abs/2210.03629)

**TL;DR:** Interleave reasoning traces with actions - the model thinks about what to do, does it, observes results, and repeats.

**The Problem:**
- Pure reasoning (Chain-of-Thought) can't access external information
- Pure acting (tool use) lacks transparent reasoning
- Neither alone handles complex multi-step tasks

**The Solution:** ReAct loop:
```
Thought: I need to find X to answer this question
Action: Search[X]
Observation: [Search results]
Thought: Based on this, I now need to find Y
Action: Search[Y]
Observation: [Search results]
Thought: Now I can answer the question
Action: Finish[answer]
```

**Key Insight:** Explicit reasoning traces make the agent interpretable and allow it to recover from errors. The model can observe failures and reason about corrections.

**ReAct vs Alternatives:**
| Approach | Reasoning | Acting | Interpretable |
|----------|-----------|--------|---------------|
| Standard prompting | ❌ | ❌ | ❌ |
| Chain-of-Thought | ✅ | ❌ | ✅ |
| Act-only | ❌ | ✅ | ❌ |
| ReAct | ✅ | ✅ | ✅ |

**Practical Impact:**
- Standard pattern for LLM agents
- Implemented in LangChain, LlamaIndex, etc.
- Foundation for all modern agent frameworks
- Your go-to pattern for building agents

**Read This If:** You're building any kind of agent system.

---

### Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023)
**ArXiv:** [2302.04761](https://arxiv.org/abs/2302.04761)

**TL;DR:** Train the model to use tools by having it generate and filter its own tool-use training data.

**The Problem:** How do you teach a model to use tools without massive human annotation?

**The Solution:** Self-supervised tool learning:
1. Sample positions where tools might help
2. Generate API calls at those positions
3. Execute calls and get results
4. Keep only calls that improve perplexity (actually help the model)
5. Fine-tune on this filtered data

**Tools Learned:**
- Calculator
- Q&A system
- Search engine
- Translation system
- Calendar

**Key Insight:** The model can learn when and how to use tools by discovering which tool calls actually improve its predictions.

**Practical Impact:**
- Shows models can learn tool use from self-supervision
- Influenced thinking about tool integration
- In practice, most systems use prompted tool use (function calling) instead
- But demonstrates the potential for learned tool selection

**Read This If:** You're interested in how models can learn tool use, or researching agent training.

---

## 6. Efficiency & Optimization

### Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Fedus et al., 2021)
**ArXiv:** [2101.03961](https://arxiv.org/abs/2101.03961)

**TL;DR:** Mixture-of-Experts (MoE) enables massive parameter counts while only activating a fraction per token.

**The Problem:** Bigger models perform better, but compute cost scales with parameters. Can we get the benefits without the cost?

**The Solution:** Sparse Mixture-of-Experts:
- Replace FFN layers with multiple "expert" FFNs
- Router network selects which expert(s) to use per token
- Each token only activates a small subset of total parameters

**Key Insight:** You can scale capacity (total parameters) separately from compute (active parameters). A 1T parameter MoE might only use 100B parameters per forward pass.

**Architecture:**
```
Input → Router → Select Top-k Experts → Expert FFNs → Combine outputs
```

**MoE Trade-offs:**
| Aspect | Dense Model | MoE |
|--------|-------------|-----|
| Capacity | All params used | Huge total capacity |
| Compute/token | All params | Subset only |
| Memory | Params in memory | ALL params in memory |
| Training | Simple | Load balancing challenges |

**Practical Impact:**
- Used in GPT-4 (rumored), Mixtral, and other frontier models
- Key efficiency technique for scaling
- Trade memory for compute efficiency
- Understanding MoE is essential for frontier systems

**Read This If:** You're working on efficient model architectures or need to understand frontier model designs.

---

### LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
**ArXiv:** [2106.09685](https://arxiv.org/abs/2106.09685)

**TL;DR:** Fine-tune large models by training small low-rank decomposition matrices, keeping original weights frozen.

**The Problem:** Fine-tuning a 175B parameter model requires storing and updating all 175B parameters. Expensive and impractical.

**The Solution:** Low-rank decomposition:
```
W' = W + BA
```
- W: Original frozen weights (d × k)
- B: Low-rank matrix (d × r), r << d
- A: Low-rank matrix (r × k)
- Only train A and B (orders of magnitude fewer parameters)

**Key Insight:** The updates during fine-tuning have low "intrinsic rank" - you can capture most of the adaptation with small matrices.

**LoRA Benefits:**
- Train <1% of parameters
- No inference latency (merge weights after training)
- Multiple LoRAs for different tasks
- Democratizes fine-tuning

**Practical Impact:**
- Standard technique for efficient fine-tuning
- Enables fine-tuning on consumer GPUs
- Used everywhere: HuggingFace PEFT, llama.cpp, etc.
- You WILL use this

**Read This If:** You're doing any fine-tuning (you probably are).

---

### QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)
**ArXiv:** [2305.14314](https://arxiv.org/abs/2305.14314)

**TL;DR:** Combine 4-bit quantization with LoRA to fine-tune 65B models on a single 48GB GPU.

**The Problem:** Even LoRA requires holding the base model in memory. Large models don't fit on consumer GPUs.

**The Solution:** Stack optimizations:
- **4-bit NormalFloat**: New quantization format with better precision
- **Double quantization**: Quantize the quantization constants
- **Paged optimizers**: Offload optimizer states to CPU when needed
- **LoRA adapters**: Train small matrices on the quantized model

**Key Insight:** Aggressive quantization of base weights + LoRA training achieves near-full-precision fine-tuning quality.

**Practical Impact:**
- Fine-tune 70B models on single GPU
- Makes LLM fine-tuning accessible to individuals
- Standard for local/affordable fine-tuning
- Often combined with LoRA by default now

**Read This If:** You want to fine-tune large models without expensive hardware.

---

### FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)
**ArXiv:** [2205.14135](https://arxiv.org/abs/2205.14135)

**TL;DR:** Reorder attention computation to minimize memory reads/writes, achieving 2-4x speedup.

**The Problem:** Standard attention is memory-bound, not compute-bound. GPU spends most time moving data, not computing.

**The Solution:** IO-aware algorithm:
- **Tiling**: Process attention in blocks that fit in fast SRAM
- **Recomputation**: Recompute attention in backward pass instead of storing
- **Kernel fusion**: Combine operations to reduce memory round-trips

**Key Insight:** Modern GPUs have fast compute but slow memory bandwidth. Optimizing for memory access patterns matters more than reducing FLOPs.

**Performance:**
- 2-4x faster training
- 5-20x memory reduction
- Enables longer context windows
- Exact attention (no approximation)

**Practical Impact:**
- Standard in all modern training frameworks
- Enables longer contexts affordably
- Often invisible to users (built into libraries)
- Understanding it helps you reason about efficiency

**Read This If:** You're training models or need to understand why context limits exist.

---

### ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2020)
**ArXiv:** [1910.02054](https://arxiv.org/abs/1910.02054)

**TL;DR:** Partition model states across GPUs to train models that don't fit on a single device.

**The Problem:** Training a large model requires storing: parameters + gradients + optimizer states. This exceeds single-GPU memory.

**The Solution:** Zero Redundancy Optimizer (ZeRO) stages:
- **ZeRO-1**: Partition optimizer states
- **ZeRO-2**: Partition gradients
- **ZeRO-3**: Partition parameters
- Each GPU only stores 1/N of each component

**Key Insight:** Data parallelism typically replicates the full model on each GPU. This is wasteful - you can partition and communicate.

**Memory Reduction:**
| Stage | Memory per GPU | Communication |
|-------|----------------|---------------|
| Baseline | Full model | None |
| ZeRO-1 | ~4x reduction | Low |
| ZeRO-2 | ~8x reduction | Medium |
| ZeRO-3 | Linear scaling | Higher |

**Practical Impact:**
- Enables training models larger than single-GPU memory
- Implemented in DeepSpeed, FSDP
- Standard for distributed training
- You'll use this for any serious training

**Read This If:** You're training large models or want to understand distributed training.

---

## 7. Data & Training

### The Pile: An 800GB Dataset of Diverse Text for Language Modeling (Gao et al., 2020)
**ArXiv:** [2101.00027](https://arxiv.org/abs/2101.00027)

**TL;DR:** A curated, diverse, high-quality dataset of 825GB for training language models.

**The Problem:** Web crawls (Common Crawl) are noisy and biased. Good training data is scarce and expensive to create.

**The Solution:** Curate 22 diverse sources:
- Books, academic papers, code (GitHub, StackExchange)
- Wikipedia, news, legal documents
- Math, science, literature
- Careful deduplication and quality filtering

**Key Insight:** Data diversity and quality matter more than raw scale. A curated 800GB beats a noisy 8TB.

**Practical Impact:**
- Influenced all subsequent data curation efforts
- Demonstrated importance of data engineering
- Many open models trained on Pile or derivatives
- Template for building training datasets

**Read This If:** You're curating training data or want to understand data quality.

---

### Self-Instruct: Aligning Language Models with Self-Generated Instructions (Wang et al., 2022)
**ArXiv:** [2212.10560](https://arxiv.org/abs/2212.10560)

**TL;DR:** Use a language model to generate its own instruction-following training data.

**The Problem:** Human-written instruction data is expensive and limited. How do you scale instruction tuning?

**The Solution:** Bootstrap from a small seed:
1. Start with 175 seed instructions (human-written)
2. Use the model to generate new instructions
3. Filter for quality and diversity
4. Generate responses to instructions
5. Fine-tune on this synthetic data

**Key Insight:** Models can generate useful training data for themselves, creating a self-improvement loop.

**Practical Impact:**
- Foundation for synthetic data generation
- Inspired Alpaca, Vicuna, and many open models
- Shows you can bootstrap capability from small seeds
- Standard technique for instruction tuning

**Read This If:** You're creating instruction-tuning datasets or working with synthetic data.

---

## 8. Evaluation

### Measuring Massive Multitask Language Understanding (Hendrycks et al., 2020) - "MMLU"
**ArXiv:** [2009.03300](https://arxiv.org/abs/2009.03300)

**TL;DR:** A benchmark of 57 subjects from elementary to professional level, measuring broad knowledge.

**The Problem:** How do you measure whether a model has broad, useful knowledge? Existing benchmarks were narrow.

**The Solution:** 15,908 questions across 57 subjects:
- STEM: Math, physics, chemistry, CS
- Humanities: History, philosophy, law
- Social Sciences: Economics, psychology
- Other: Professional medicine, accounting

**Key Insight:** Broad evaluation reveals gaps that narrow benchmarks miss. A model can ace coding while failing at basic science.

**MMLU Limitations:**
- Multiple choice format (easy to game)
- Static (can be memorized)
- Doesn't test reasoning depth
- Cultural/geographic biases

**Practical Impact:**
- Standard benchmark for model comparison
- Useful but not sufficient
- Production systems need task-specific evaluation
- Don't over-optimize for MMLU

**Read This If:** You need to understand standard evaluation, but remember benchmarks aren't everything.

---

## 9. Multimodality

### Flamingo: A Visual Language Model for Few-Shot Learning (Alayrac et al., 2022)
**ArXiv:** [2204.14198](https://arxiv.org/abs/2204.14198)

**TL;DR:** A model that can see images and engage in visual dialogue with few-shot learning.

**The Problem:** How do you combine vision and language while leveraging pretrained LLMs?

**The Solution:**
- Freeze a pretrained LLM
- Add visual encoders with Perceiver Resampler
- Cross-attention layers to inject visual information
- Train on interleaved image-text data

**Key Insight:** You can add vision to a frozen LLM with efficient cross-attention, preserving language capabilities.

**Practical Impact:**
- Foundation for modern vision-language models
- Influenced GPT-4V, Claude vision, Gemini
- Architecture pattern for multimodal extension
- Understanding this helps with multimodal systems

**Read This If:** You're working with vision-language models or extending LLMs to new modalities.

---

### GPT-4 Technical Report (OpenAI, 2023)
**ArXiv:** [2303.08774](https://arxiv.org/abs/2303.08774)

**TL;DR:** A large multimodal model with strong reasoning, accepted as images and text.

**The Problem:** How do you build a production-grade multimodal system with strong reasoning?

**The Solution:** [Details intentionally omitted by OpenAI, but report covers]:
- Evaluation methodology
- Safety testing approach
- Capabilities and limitations
- Predictable scaling

**Key Insight:** The report is valuable not for architecture (not disclosed) but for:
- How to evaluate complex systems
- How to think about safety
- What predictable scaling means in practice

**Practical Impact:**
- Benchmark for capabilities
- Safety evaluation template
- Systems thinking over architecture details
- Shows what's possible at frontier

**Read This If:** You want to understand frontier capabilities and safety evaluation methodology.

---

## 10. Interpretability

### A Mathematical Framework for Transformer Circuits (Elhage et al., 2021)
**Anthropic, not on ArXiv** - [transformer-circuits.pub](https://transformer-circuits.pub/2021/framework/index.html)

**TL;DR:** Reverse-engineer what Transformers compute by analyzing them as circuits.

**The Problem:** Neural networks are black boxes. Can we understand what they're actually computing?

**The Solution:** Mechanistic interpretability:
- Analyze attention patterns as information routing
- Identify "circuits" that implement specific behaviors
- Map mathematical structure to semantic meaning
- Find interpretable features in activation space

**Key Insight:** Transformers implement identifiable algorithms. "Induction heads" implement in-context learning. Specific circuits handle specific tasks.

**Found Circuits:**
- **Induction heads**: Copy patterns from context
- **Indirect object identification**: Track which entities did what
- **Negation**: Implement logical negation

**Practical Impact:**
- Foundation for AI safety via understanding
- Debugging model behavior at mechanistic level
- Early-stage but rapidly advancing field
- Important for building trustworthy systems

**Read This If:** You're interested in AI safety, debugging models, or understanding what models actually compute.

---

## 11. Protocols

### Model Context Protocol (MCP) - Anthropic, 2024
**Documentation:** [modelcontextprotocol.io](https://modelcontextprotocol.io)

**TL;DR:** A standard protocol for connecting AI systems to external tools, data, and services.

**The Problem:** Every tool integration is custom. N models × M tools = N×M integrations. Doesn't scale.

**The Solution:** Standardized protocol with:
- **Clients**: AI applications that want to use tools
- **Servers**: Services that expose tools/data
- **Transports**: How they communicate (stdio, HTTP)
- **Capabilities**: Tools, resources, prompts, sampling

**Key Insight:** Protocols enable ecosystems. Just as HTTP enabled the web, MCP enables interoperable AI tools.

**MCP Components:**
```
┌─────────────┐         ┌─────────────┐
│   Client    │◄───────►│   Server    │
│ (Claude,    │   MCP   │ (Your tool, │
│  Agent)     │ Protocol│  Database)  │
└─────────────┘         └─────────────┘
```

**Practical Impact:**
- Reduces integration cost dramatically
- Enables tool marketplaces/ecosystems
- Standard for Claude Desktop, agent frameworks
- Learn this if building agent systems

**Read This If:** You're building agents that use tools (you probably are).

---

## Reading Order Recommendations

### If you have 1 week:
1. Attention Is All You Need (skim math, understand concept)
2. InstructGPT (understand RLHF)
3. RAG paper (understand retrieval augmentation)
4. ReAct (understand agent pattern)
5. MCP docs (understand tool integration)

### If you have 1 month:
Add: DPO, LoRA, FlashAttention, Chinchilla, Self-Instruct

### If you want depth in specific areas:

**Alignment track:** InstructGPT → DPO → Constitutional AI → Christiano 2017

**Memory track:** RAG → MemGPT → ROME → MEMIT

**Efficiency track:** LoRA → QLoRA → FlashAttention → ZeRO → Switch Transformer

**Agents track:** ReAct → Toolformer → MCP → MemGPT

---

## Summary: The Papers That Matter Most

If you remember nothing else:

| Paper | Why It Matters |
|-------|----------------|
| Attention Is All You Need | The architecture everything builds on |
| InstructGPT | How models become assistants |
| DPO | Simpler alignment that works |
| RAG | How models access external knowledge |
| ReAct | The standard agent pattern |
| LoRA | How you actually fine-tune |
| MCP | How you connect to tools at scale |

**The meta-lesson:** Individual papers solve individual problems. The AI Engineer's job is to combine them into working systems.
