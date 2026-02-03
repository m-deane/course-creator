# Quiz: Module 0 - Foundations of LLMs and Agent Design

**Estimated Time:** 20 minutes
**Total Points:** 100
**Passing Score:** 70%

---

## Instructions

This quiz assesses your understanding of transformer architecture, LLM providers, and prompt engineering basics. Answer all questions to the best of your ability. For multiple choice questions, select the best answer. For short answer questions, provide concise but complete responses (2-3 sentences).

---

## Part A: Transformer Architecture (30 points)

### Question 1 (5 points)
**True or False:** Transformers process input tokens sequentially, one at a time, just like they generate output tokens.

**Your Answer:**

---

### Question 2 (8 points)
What is the computational complexity of self-attention in transformers, and why does this matter for agent systems with long context windows?

**Your Answer:**

---

### Question 3 (7 points)
In the attention mechanism formula `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V`, what role does each component play?

**Select the correct matching:**

A) Q (Query): What information do I provide?
B) K (Key): What am I looking for?
C) V (Value): What do I contain?

**OR**

D) Q (Query): What am I looking for?
E) K (Key): What do I contain?
F) V (Value): What information do I provide?

**Your Answer:**

---

### Question 4 (5 points)
Why should agent systems typically use low temperature (e.g., temperature=0) when making tool calls?

A) Low temperature reduces API costs
B) Low temperature makes responses more creative and diverse
C) Low temperature produces deterministic, reliable outputs for structured tasks
D) Low temperature increases the context window size

**Your Answer:**

---

### Question 5 (5 points)
**True or False:** The "lost in the middle" problem refers to the fact that LLMs have difficulty retrieving information placed in the middle of long context windows.

**Your Answer:**

---

## Part B: LLM Providers (25 points)

### Question 6 (8 points)
Match each LLM provider with its primary strength:

1. Claude (Anthropic)
2. GPT-4 (OpenAI)
3. Open-source models (Llama, Mistral)

A) Largest ecosystem, extensive fine-tuning options, strong general knowledge
B) Superior instruction following, excellent structured output, large context window (200K)
C) Full data privacy, no API costs if self-hosted, customizable

**Your Answer:**

---

### Question 7 (7 points)
For a high-volume document processing agent that needs to extract structured information from thousands of invoices daily, which model selection strategy would be most cost-effective?

A) Use Claude 3 Opus for all requests to maximize accuracy
B) Use Claude 3 Haiku or GPT-3.5 Turbo for cost efficiency at scale
C) Use a self-hosted Llama model regardless of infrastructure costs
D) Randomly distribute requests across all available models

**Your Answer:**

---

### Question 8 (10 points)
What is the purpose of creating a provider-agnostic abstraction layer when building agent systems? Describe two specific benefits.

**Your Answer:**

---

## Part C: Prompt Engineering Basics (45 points)

### Question 9 (8 points)
In the CLEAR framework for prompt construction, what do the letters stand for?

**Your Answer:**

---

### Question 10 (7 points)
Which of the following demonstrates proper use of delimiters to prevent prompt injection vulnerabilities?

A)
```python
prompt = f"Translate this to French: {user_input}"
```

B)
```python
prompt = f"""Translate the text between <text> tags to French.
Only output the translation, nothing else.

<text>
{user_input}
</text>

French translation:"""
```

C)
```python
prompt = "Translate to French: " + user_input
```

D)
```python
prompt = f"User said: {user_input}. Now translate it to French."
```

**Your Answer:**

---

### Question 11 (8 points)
What is "output priming" in prompt engineering, and when is it useful? Provide an example.

**Your Answer:**

---

### Question 12 (7 points)
**True or False:** When designing prompts, implicit expectations work just as well as explicit instructions because modern LLMs can infer what you mean.

**Your Answer:**

---

### Question 13 (8 points)
You need to create a prompt that extracts contact information from text. Some fields might be missing. Which approach best handles missing information?

A) Assume all fields are present and let the model figure it out
B) Specify which fields are required vs. optional, and define what to return when data is missing
C) Ask the model to guess missing information
D) Return an error if any field is missing

**Your Answer:**

---

### Question 14 (7 points)
Explain why chain-of-thought prompting works from a technical perspective, referencing how transformers generate text.

**Your Answer:**

---

## Bonus Question (5 points)

### Question 15
Why is it important to place critical instructions at both the beginning and end of long prompts, rather than only at the beginning or only in the middle?

**Your Answer:**

---

# Answer Key

## Part A: Transformer Architecture

**Question 1:** False (5 points)
*Explanation:* Transformers process ALL input tokens simultaneously through attention, but generate output tokens one at a time (autoregressive). This is a crucial distinction.

**Question 2:** (8 points)
*Sample Answer:* Self-attention is O(n²) in sequence length, meaning attention computations grow quadratically. For agent systems, this means longer context windows dramatically increase computational cost and latency. For example, 100K tokens requires 10 billion attention computations compared to 100M for 10K tokens. This is why relevant information retrieval (RAG) is often better than stuffing entire documents into context.

**Question 3:** Answer: D, E, F (7 points)
*Explanation:* Q (Query) = "What am I looking for?", K (Key) = "What do I contain?", V (Value) = "What information do I provide?"

**Question 4:** Answer: C (5 points)
*Explanation:* Low temperature produces deterministic, reliable outputs critical for structured tasks like tool calls where consistency is essential.

**Question 5:** True (5 points)
*Explanation:* Research shows LLMs have more difficulty retrieving information from the middle of long contexts compared to the beginning or end.

## Part B: LLM Providers

**Question 6:** 1-B, 2-A, 3-C (8 points)
- Claude: Superior instruction following, excellent structured output, large context window
- GPT-4: Largest ecosystem, extensive fine-tuning options, strong general knowledge
- Open-source: Full data privacy, no API costs if self-hosted, customizable

**Question 7:** Answer: B (7 points)
*Explanation:* High-volume processing requires cost efficiency. Haiku or GPT-3.5 Turbo provide good performance at much lower cost for simple extraction tasks.

**Question 8:** (10 points)
*Sample Answer:* Provider-agnostic abstraction allows: (1) Easy switching between providers without rewriting code, enabling flexibility as models improve or pricing changes, and (2) Using multiple providers strategically (e.g., Claude for reasoning, GPT-4 for broad knowledge) within the same system without duplicating integration logic.

## Part C: Prompt Engineering Basics

**Question 9:** (8 points)
*Answer:*
- C = Context (set the scene)
- L = Limits (define constraints)
- E = Examples (show, don't just tell)
- A = Action (state the task clearly)
- R = Result (specify desired output format)

**Question 10:** Answer: B (7 points)
*Explanation:* Option B uses XML delimiters to clearly separate user input from instructions and explicitly states to only translate, reducing injection risk.

**Question 11:** (8 points)
*Sample Answer:* Output priming is starting the model's response to guide the format. It's useful when you need precise formatting. Example: Ending a prompt with "SQL:\n```sql\nSELECT" primes the model to continue generating SQL code in the correct format.

**Question 12:** False (7 points)
*Explanation:* Explicit instructions consistently outperform implicit expectations. The model predicts likely continuations based on patterns, not true understanding of intent.

**Question 13:** Answer: B (8 points)
*Explanation:* Explicitly specifying required vs. optional fields and defining behavior for missing data (e.g., return "N/A" or "INCOMPLETE") provides clear, predictable results.

**Question 14:** (7 points)
*Sample Answer:* Chain-of-thought works because of autoregressive generation—each token becomes context for subsequent tokens. When the model writes reasoning steps, those tokens are fed back as input, allowing the model to literally build on its own reasoning. Each step informs the next token prediction.

## Bonus Question

**Question 15:** (5 points)
*Sample Answer:* Due to attention patterns and the "lost in the middle" problem, information at the beginning and end of prompts receives stronger attention weights. Placing critical instructions in both positions ensures they aren't missed, especially in long prompts.

---

## Scoring Guide

- **90-100 points:** Excellent - Strong grasp of foundational concepts
- **80-89 points:** Good - Solid understanding with minor gaps
- **70-79 points:** Passing - Adequate comprehension, review some topics
- **Below 70:** Review module materials before proceeding

**Key Topics to Review if Struggling:**
- Transformer architecture and attention mechanisms
- Trade-offs between different LLM providers
- CLEAR framework and structured prompting
- Security considerations in prompt design
- Chain-of-thought reasoning mechanics
