# Transformer Architecture for Agent Builders

> **Reading time:** ~12 min | **Module:** 0 — Foundations | **Prerequisites:** None

Transformers are the neural network architecture behind all modern LLMs. Understanding how they process and generate text helps you write better prompts, debug agent behavior, and make informed design decisions.

<div class="callout-insight">

**Insight:** Transformers process all tokens simultaneously through attention, but generate output one token at a time. This means the model "sees" your entire prompt at once but must commit to each output token sequentially—once generated, tokens can't be revised.

</div>

---

## The Transformer Pipeline

### Input Processing

```
"Hello world"
    ↓
[15496, 995]              # Tokenization
    ↓
[[0.1, -0.3, ...],        # Token embeddings (d=4096 for large models)
 [0.2,  0.1, ...]]
    ↓
+ Position embeddings     # Add positional information
    ↓
Ready for attention layers
```

### The Attention Mechanism

Self-attention allows each token to "look at" every other token:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Intuition:**
- **Q (Query):** "What am I looking for?"
- **K (Key):** "What do I contain?"
- **V (Value):** "What information do I provide?"

Each token computes attention scores with all other tokens, creating a weighted combination of their values.

### Multi-Head Attention

Multiple attention "heads" run in parallel, each learning different relationships:

```
Head 1: Learns syntax (subject-verb agreement)
Head 2: Learns coreference (pronouns → nouns)
Head 3: Learns semantic similarity
...
Head N: Learned patterns
```

### Layer Stack

Modern LLMs stack many transformer layers:

| Model | Layers | Parameters |
|-------|--------|------------|
| GPT-4 | ~120 | ~1.7T |
| Claude 3 | ~80-100 | Undisclosed |
| Llama 3 70B | 80 | 70B |

Each layer refines the representation, building from tokens → syntax → semantics → reasoning.

---

## Generation: Autoregressive Decoding

### The Generation Loop


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
def generate(prompt, max_tokens):
    tokens = tokenize(prompt)

    for _ in range(max_tokens):
        # Forward pass through all layers
        logits = transformer(tokens)

        # Sample next token from probability distribution
        next_token = sample(logits[-1])

        # Append and continue
        tokens.append(next_token)

        if next_token == EOS:
            break

    return detokenize(tokens)
```

</div>
</div>

### Sampling Strategies

**Temperature:** Controls randomness
- T=0: Always pick highest probability (deterministic)
- T=1: Sample from true distribution
- T>1: More random/creative

**Top-p (Nucleus):** Only sample from tokens comprising top p% probability mass

**Top-k:** Only consider top k most likely tokens

### For Agents: Use Low Temperature

Agent actions should be reliable and predictable:
```python
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    temperature=0,  # Deterministic for tool calls
    ...
)
```

---

## Context Windows and Attention

### Quadratic Complexity

Attention is O(n²) in sequence length—each token attends to every other:

| Context | Attention Computations |
|---------|----------------------|
| 1K tokens | 1M |
| 10K tokens | 100M |
| 100K tokens | 10B |

### Implications for Agents

1. **Longer context ≠ better:** Relevant information should be near the query
2. **"Lost in the middle" problem:** Information in the middle of context is harder to retrieve
3. **RAG helps:** Retrieve relevant chunks rather than stuffing full documents

### Context Window Management


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
def manage_context(messages, max_tokens=100000):
    """Keep conversation within context limits."""
    total_tokens = sum(count_tokens(m) for m in messages)

    while total_tokens > max_tokens:
        # Remove oldest messages (keep system prompt)
        messages.pop(1)  # Index 0 is system prompt
        total_tokens = sum(count_tokens(m) for m in messages)

    return messages
```

</div>
</div>

---

## Practical Implications for Agents

### 1. Prompt Placement Matters

Put critical instructions at the **beginning** and **end** of prompts—these positions get more attention.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
system_prompt = """
[CRITICAL INSTRUCTIONS AT START]
You are an agent that...

[DETAILED CONTEXT IN MIDDLE]
...

[CRITICAL REMINDERS AT END]
Remember to always...
"""
```

</div>
</div>

### 2. Structured Output is Easier

The model predicts token-by-token. Structured formats (JSON, XML) constrain each token's options:

```
{"name": "   # After this, model knows a string is coming
```

### 3. Chain-of-Thought Works Because of Autoregression

When the model writes out reasoning steps, those tokens become part of the context for subsequent tokens. The model literally "thinks out loud."

```
Let me solve this step by step:
1. First, I need to... [model now has this context]
2. Given step 1, I should... [builds on step 1]
3. Therefore... [conclusion informed by steps 1-2]
```

### 4. Tool Calls Are Just Structured Generation

Function calling is the model generating structured JSON that matches a schema:

```json
{
  "tool": "search",
  "arguments": {
    "query": "current weather in NYC"
  }
}
```

---

## Common Misconceptions

### "The model understands my intent"

The model predicts likely continuations based on patterns. It doesn't "understand" in a human sense. Clear, explicit prompts work better than implicit expectations.

### "More context is always better"

Long contexts dilute attention. Focused, relevant context outperforms exhaustive dumps.

### "The model remembers previous conversations"

Each API call is independent. "Memory" requires explicit context management (conversation history, RAG, etc.).

### "Temperature 0 is always best"

For tool calls and structured output: yes. For creative tasks, brainstorming, or diverse sampling: higher temperature helps.

---

## Code: Inspecting Tokenization


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>

```python
import anthropic

client = anthropic.Anthropic()

def count_tokens(text: str) -> int:
    """Count tokens in text using Claude's tokenizer."""
    response = client.messages.count_tokens(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": text}]
    )
    return response.input_tokens

# Examples
examples = [
    "Hello, world!",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "The quick brown fox jumps over the lazy dog.",
]

for text in examples:
    tokens = count_tokens(text)
    ratio = len(text.split()) / tokens
    print(f"'{text[:50]}...' -> {tokens} tokens ({ratio:.2f} words/token)")
```


---

<div class="callout-key">

**Key Concept Summary:** This guide covered the core concepts. Review the companion slides for visual summaries and the hands-on notebook for practice implementations.




## Further Reading

- **"Attention Is All You Need"** (Vaswani et al., 2017) - Original transformer paper
- **"Language Models are Few-Shot Learners"** (Brown et al., 2020) - GPT-3 paper
- **Anthropic's Claude documentation** - Practical guidance for Claude models

---

*Understanding transformers isn't about the math—it's about intuition for how tokens flow through the model and how that affects your agent designs.*


## Practice Questions

1. Explain in your own words how the concepts in this guide relate to building production agents.
2. What are the key tradeoffs you need to consider when applying these techniques?
3. Describe a scenario where the approach from this guide would be the wrong choice, and what you would use instead.

---

**Next Steps:**

<a class="link-card" href="./01_transformer_architecture_slides.md">
  <div class="link-card-title">Transformer Architecture for Agent Builders — Companion Slides</div>
  <div class="link-card-description">Visual slide deck with diagrams, speaker notes, and key takeaways.</div>
</a>

<a class="link-card" href="../notebooks/01_api_setup.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with working code and guided exercises.</div>
</a>
