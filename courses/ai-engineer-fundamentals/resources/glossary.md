# AI Engineer Glossary

> Key terms and concepts for modern LLM systems.

---

## A

**Alignment**
The process of making AI systems behave according to human values and intentions. Includes techniques like SFT, RLHF, and DPO.

**Attention**
The mechanism in Transformers that allows the model to weigh the relevance of different parts of the input when generating each output token. The core formula: `Attention(Q, K, V) = softmax(QK^T / √d_k) V`

**Agent**
An LLM-based system that can take actions in the world, observe results, and iterate until a goal is achieved. Combines reasoning with tool use.

**Agent Loop**
The iterative cycle of: Interpret goal → Decide action → Execute tool → Observe result → Update plan → Repeat until done.

---

## B

**Base Model**
A language model trained only on next-token prediction, before any alignment or fine-tuning. Good at completion but not at following instructions.

**Batching**
Processing multiple requests together to improve throughput, trading latency for efficiency.

---

## C

**Chain-of-Thought (CoT)**
A prompting technique where the model is encouraged to show its reasoning steps before giving a final answer.

**Chinchilla Scaling**
The finding that models should be trained on approximately 20 tokens per parameter for compute-optimal performance.

**Chunking**
Breaking documents into smaller pieces for embedding and retrieval. Strategies include fixed-size, semantic, and recursive chunking.

**Constitutional AI**
An alignment approach using explicit principles and AI-generated feedback to train models, reducing dependence on human labels.

**Context Window**
The maximum number of tokens a model can process in a single forward pass. Ranges from 4K to 128K+ in modern models.

**Cross-Encoder**
A model that takes a query-document pair as input and outputs a relevance score. More accurate than bi-encoders but slower.

---

## D

**DPO (Direct Preference Optimization)**
A simpler alternative to RLHF that directly optimizes on preference pairs without training a separate reward model.

**Dense Retrieval**
Retrieval using learned embeddings and vector similarity, as opposed to keyword-based sparse retrieval.

---

## E

**Embedding**
A dense vector representation of text that captures semantic meaning. Used for similarity search in RAG systems.

**Evaluation Harness**
A system for running standardized tests on models to measure capabilities and detect regressions.

---

## F

**Few-Shot Learning**
Including examples in the prompt to teach the model a task without fine-tuning.

**Fine-Tuning**
Additional training on a pretrained model to adapt it for specific tasks or domains.

**FlashAttention**
An IO-aware implementation of attention that reduces memory usage and increases speed by optimizing data movement.

**Function Calling**
The ability for LLMs to output structured tool/function invocations rather than just text.

---

## G

**Guardrails**
Safety mechanisms that filter or modify LLM inputs and outputs to prevent harmful content.

---

## H

**Hallucination**
When an LLM generates confident but factually incorrect information. A key challenge for production systems.

**HNSW (Hierarchical Navigable Small World)**
A graph-based algorithm for approximate nearest neighbor search, commonly used in vector databases.

---

## I

**In-Context Learning**
The ability of LLMs to adapt to new tasks based on examples provided in the prompt, without weight updates.

**Instruction Tuning**
Fine-tuning a model on instruction-response pairs to improve its ability to follow directions.

---

## K

**KL Divergence**
A measure of how one probability distribution differs from another. Used in RLHF to keep the policy close to the reference model.

**KV Cache**
Caching of key and value matrices during generation to avoid recomputation, trading memory for speed.

---

## L

**LoRA (Low-Rank Adaptation)**
A parameter-efficient fine-tuning method that trains small adapter matrices instead of the full model weights.

**LLM (Large Language Model)**
A neural network trained on large amounts of text data to predict and generate language.

---

## M

**MCP (Model Context Protocol)**
A standard protocol for connecting AI systems to external tools, data sources, and services.

**MEMIT**
A technique for editing thousands of facts in model weights simultaneously.

**Memory (Agent)**
The ability for agents to store and retrieve information across interactions. Includes context window, RAG, and persistent stores.

**MoE (Mixture of Experts)**
An architecture where only a subset of model parameters are activated for each input, enabling larger total capacity with fixed compute.

**MMLU**
A benchmark measuring knowledge across 57 subjects. Common but limited evaluation metric.

---

## P

**Perplexity**
A measure of how well a probability model predicts a sample. Lower is better for language models.

**PPO (Proximal Policy Optimization)**
A reinforcement learning algorithm used in RLHF to optimize the policy while constraining update size.

**Prompt Engineering**
The practice of crafting effective prompts to elicit desired behavior from LLMs.

---

## Q

**QLoRA**
Combining 4-bit quantization with LoRA to enable fine-tuning of large models on consumer hardware.

**Quantization**
Reducing the precision of model weights (e.g., from 32-bit to 8-bit or 4-bit) to decrease memory and increase speed.

---

## R

**RAG (Retrieval-Augmented Generation)**
Combining a retriever and generator so the model can access external knowledge at inference time.

**ReAct**
A pattern interleaving reasoning traces and action execution, making agents more interpretable and robust.

**Reranking**
A second-stage retrieval step using a more accurate (but slower) model to reorder initial results.

**Reward Model**
A model trained to predict human preferences, used in RLHF to guide policy optimization.

**RLHF (Reinforcement Learning from Human Feedback)**
Training LLMs to align with human preferences by using a learned reward model and RL optimization.

**ROME**
A technique for locating and editing specific factual associations in model weights.

---

## S

**Scaling Laws**
Empirical relationships showing how model performance improves predictably with more data, parameters, and compute.

**Self-Attention**
Attention mechanism where queries, keys, and values all come from the same sequence, allowing tokens to attend to each other.

**SFT (Supervised Fine-Tuning)**
Fine-tuning on instruction-response pairs to teach models to follow instructions.

**Sparse Retrieval**
Traditional keyword-based retrieval methods like BM25, as opposed to dense embedding-based retrieval.

---

## T

**Temperature**
A parameter controlling randomness in generation. Higher = more random, lower = more deterministic.

**Token**
The basic unit of text that LLMs process. Words are typically split into subword tokens.

**Tool Use**
The ability for LLMs to call external functions or APIs to accomplish tasks beyond text generation.

**Transformer**
The neural network architecture underlying modern LLMs, based on self-attention mechanisms.

---

## V

**Vector Database**
A database optimized for storing and searching high-dimensional embedding vectors. Examples: Chroma, Pinecone, Qdrant.

---

## Z

**Zero-Shot**
Using a model on a task without any task-specific examples in the prompt.

**ZeRO (Zero Redundancy Optimizer)**
A distributed training technique that partitions optimizer states, gradients, and parameters across GPUs.

---

## Quick Reference Formulas

**Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

**Chinchilla Optimal:**
```
Tokens ≈ 20 × Parameters
```

**DPO Loss:**
```
L = -E[log σ(β(log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
```

**Cosine Similarity:**
```
cos(a, b) = (a · b) / (||a|| × ||b||)
```

---

*For paper-specific terms, see [Paper Summaries](paper_summaries.md).*
