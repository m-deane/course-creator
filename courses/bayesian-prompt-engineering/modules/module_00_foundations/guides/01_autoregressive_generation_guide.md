# Autoregressive Generation: LLMs as Conditional Probability Machines

## In Brief

Language models generate text one token at a time, where each token is sampled from a conditional probability distribution over the entire vocabulary, conditioned on every token that came before it. Understanding this mechanism is the foundation of principled prompt engineering.

---

## Key Insight

A language model is not a database returning stored answers. It is a function that, given a sequence of tokens, produces a probability distribution over what token should come next. Generation is the repeated application of that function:

$$P(\text{text}) = \prod_{n=1}^{N} P(t_n \mid t_1, t_2, \ldots, t_{n-1})$$

Your prompt is the initial condition $t_1, \ldots, t_k$. Everything the model generates is determined by that starting point.

---

## 1. The Token

Before understanding generation, understand the unit of generation: the **token**.

Tokens are not words. They are chunks of text determined by a compression algorithm (Byte Pair Encoding, or BPE) trained on a large corpus. Some tokens are full words; some are word fragments; some are punctuation or whitespace.

| Text | Approximate Tokenisation |
|------|--------------------------|
| `"hello"` | `[hello]` |
| `"unbelievable"` | `[un][believ][able]` |
| `"2024"` | `[2024]` |
| `"anthropic"` | `[anthrop][ic]` |

**Why this matters for prompting:** When you write a prompt, you are providing an exact token sequence. Subtle differences in wording change the token sequence, which changes the conditional distribution the model computes.

---

## 2. Conditional Probability Over Vocabulary

At every step $n$, the model computes a score (logit) for every token in its vocabulary — typically 50,000–130,000 tokens. Those scores are converted to probabilities via softmax:

$$P(t_n = v \mid t_1, \ldots, t_{n-1}) = \frac{e^{z_v}}{\sum_{v'} e^{z_{v'}}}$$

where $z_v$ is the logit assigned to vocabulary item $v$.

The model then samples from this distribution (or takes the argmax, depending on temperature). The sampled token is appended to the context, and the process repeats.

**Visualising one step:**

```
Context so far:  "The capital of France is"

Vocabulary probabilities (top 5):
  "Paris"       → 0.9421
  " Paris"      → 0.0341
  " the"        → 0.0089
  "Lyon"        → 0.0031
  " a"          → 0.0019
  ... (50,000+ more with tiny probabilities)

Sample: "Paris"

New context: "The capital of France is Paris"
```

The model does not "know" Paris is the capital — it has learned that, in its training corpus, the token sequence `"The capital of France is"` was followed by `"Paris"` with very high frequency.

---

## 3. Temperature and Sampling

**Temperature** $T$ controls how peaked or flat the probability distribution is:

$$P(t_n = v) = \frac{e^{z_v / T}}{\sum_{v'} e^{z_{v'} / T}}$$

| Temperature | Effect | Typical Use |
|-------------|--------|-------------|
| $T \to 0$ | Near-deterministic; always picks highest probability token | Factual Q&A, code |
| $T = 1.0$ | Unmodified distribution | Balanced generation |
| $T > 1$ | Flatter distribution; more random | Creative tasks |

**Implication:** Even at $T = 0$, output is fully determined by the probability distribution the model computes. Temperature does not add external randomness — it reshapes the existing distribution. Prompt engineering shapes that distribution before temperature ever enters.

---

## 4. The Full Generation Process

```
                    ┌─────────────────────┐
    Prompt tokens   │                     │
    t₁ t₂ ... tₖ  →│   Transformer       │→  P(tₖ₊₁ | t₁...tₖ)  → sample tₖ₊₁
                    │   (attention layers)│
                    └─────────────────────┘
                              ↑
    t₁ t₂ ... tₖ tₖ₊₁  ──────┘   (append, repeat)
```

Each pass through the transformer conditions on the entire prior context via the attention mechanism. Attention allows distant tokens to influence the current distribution — this is why a condition stated at the start of your prompt can shift the probability of tokens generated hundreds of positions later.

---

## 5. Why Your Prompt Is a Prior Condition

The chain rule form of the generation probability makes the role of your prompt explicit:

$$P(t_{k+1}, \ldots, t_N \mid \underbrace{t_1, \ldots, t_k}_{\text{your prompt}})$$

Your prompt is not a query — it is a set of conditioning variables that shifts the joint probability distribution over all possible completions. A longer, more specific prompt does not constrain the model arbitrarily; it provides more evidence that concentrates probability mass on the specific subset of the output space you actually want.

### Concrete Example

Consider the prompt fragment: `"The recommended dosage is"`

Without context, this fragment could precede medical information, veterinary instructions, cooking measurements, or instructions for a fictional drug in a novel. The conditional distribution is wide and diffuse.

Now add context: `"You are a pharmacist. A patient is asking about ibuprofen for adult pain relief. The recommended dosage is"`

The conditional distribution has shifted dramatically. Medical, adult, ibuprofen, pain relief — these tokens collectively narrow probability mass to a far smaller region of vocabulary space.

This is condition engineering. The extra words are not politeness or padding; they are likelihood evidence that updates the prior.

---

## 6. Formal Statement

**Autoregressive generation** is the process of factoring a joint distribution over a sequence into a product of conditionals:

$$P(t_1, t_2, \ldots, t_N) = P(t_1) \cdot P(t_2 \mid t_1) \cdot P(t_3 \mid t_1, t_2) \cdots P(t_N \mid t_1, \ldots, t_{N-1})$$

A neural language model parameterises each conditional factor $P(t_n \mid t_1, \ldots, t_{n-1})$ with a transformer. Training adjusts the parameters to maximise the log-likelihood of observed text from the training corpus.

**Critical implication:** The training corpus defines the prior. When your prompt does not specify conditions, the model defaults to the most probable completion given that corpus — not the most probable completion given your actual situation.

---

## 7. Common Pitfalls

**Pitfall 1: Treating the model as a search engine**
The model does not retrieve facts; it generates the most probable continuation of your token sequence. This is why confident-sounding hallucinations exist: the generated text can be syntactically and stylistically indistinguishable from correct text.

**Pitfall 2: Assuming longer prompts are always better**
Length only helps if the additional tokens shift the probability distribution toward your desired output. Padding and repetition can actually dilute the signal of the conditions you care about.

**Pitfall 3: Ignoring token boundaries**
The exact tokenisation of unusual terms, numbers, and proper nouns affects generation. "1,000,000" and "1000000" may tokenise differently, shifting subsequent probabilities slightly.

**Pitfall 4: Confusing temperature with reliability**
Setting temperature to 0 does not make the model correct — it makes it deterministic. If the peak of the distribution is wrong (because the prompt was underspecified), $T = 0$ deterministically produces the wrong answer.

---

## Connections

- **Builds on:** Basic probability (conditional probability, Bayes' theorem)
- **Leads to:** Guide 02 — Prior Dominance and how to counteract it
- **Related to:** Attention mechanisms (how distant tokens condition current generation), tokenisation (what the actual conditioning units are)

---

## Practice Problems

1. **Conceptual:** Given the generation formula $P(t_n \mid t_1, \ldots, t_{n-1})$, explain in plain language why the same question phrased differently can produce completely different answers, even at temperature 0.

2. **Implementation:** Run the notebook `notebooks/01_token_probability.ipynb` and observe how adding a single sentence to a prompt shifts the distribution of model outputs across three different domains.

3. **Extension:** The transformer attention mechanism is what allows long-range dependencies in the conditioning. Sketch an explanation of why early tokens in a long prompt can have weakened influence on generation near the end (the "lost in the middle" phenomenon) from a probabilistic perspective.

---

## Further Reading

- Radford et al. (2019) "Language Models are Unsupervised Multitask Learners" — original GPT-2 paper that demonstrated the power of autoregressive pretraining
- Holtzman et al. (2020) "The Curious Case of Neural Text Degeneration" — explains why argmax decoding produces degenerate text and motivates nucleus sampling
