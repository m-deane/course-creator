# Module 3 Quiz: Custom LLM Applications

**Course:** Gen AI & Dataiku: LLM Mesh Use Cases
**Module:** 3 - Custom LLM Applications
**Time Limit:** 20 minutes
**Total Points:** 100
**Passing Score:** 70/100

## Instructions

- This quiz has 13 questions covering Python integration, custom models, and pipeline integration
- Select the best answer for each question
- Point values are indicated for each question
- You have 2 attempts per question
- Refer to module guides and notebooks if needed

---

## Section 1: Python LLM Mesh API (32 points)

### Question 1 (8 points)

Which Python code correctly creates an LLM handle in Dataiku?

A) `llm = dataiku.llm("my-connection")`
B) `llm = LLMHandle.connect("my-connection")`
C) `llm = dataiku.LLMHandle("my-connection")`
D) `llm = dataiku.api.get_llm("my-connection")`

**Answer:** C

**Explanation:**
- **C (Correct):** `dataiku.LLMHandle("connection-name")` is the correct syntax to create an LLM handle.
- **A:** There is no `llm()` method directly on the dataiku module.
- **B:** `LLMHandle` must be imported from dataiku; `connect()` is not the method name.
- **D:** The API uses `LLMHandle()`, not a `get_llm()` method.

---

### Question 2 (8 points)

What is the purpose of the `messages` parameter in `llm.generate()`?

A) To specify error messages for exception handling
B) To provide the conversation history including system, user, and assistant messages
C) To configure logging output for debugging
D) To set notification recipients for long-running tasks

**Answer:** B

**Explanation:**
- **B (Correct):** The `messages` parameter contains the conversation context with roles (system/user/assistant) and content.
- **A:** Error handling uses different mechanisms, not the messages parameter.
- **C:** Logging is configured separately from the generation call.
- **D:** Notifications are unrelated to the messages parameter.

---

### Question 3 (8 points)

Your Python recipe calls an LLM. How do you access the token usage information?

A) `llm.get_usage()`
B) `response.usage.total_tokens`
C) `dataiku.monitor.tokens()`
D) Token usage is not accessible from Python

**Answer:** B

**Explanation:**
- **B (Correct):** The response object has a `usage` attribute containing `prompt_tokens`, `completion_tokens`, and `total_tokens`.
- **A:** Usage is accessed via the response object, not a method on the LLM handle.
- **C:** There's no monitor.tokens() method; usage is in the response.
- **D:** Token usage is accessible and important for cost monitoring.

---

### Question 4 (8 points)

Which parameters can you configure when calling `llm.generate()`?

A) Only the prompt text
B) Temperature, max_tokens, top_p, and other model parameters
C) Only the connection name and prompt
D) Parameters are fixed and cannot be adjusted per call

**Answer:** B

**Explanation:**
- **B (Correct):** You can specify model parameters like temperature, max_tokens, top_p, stop sequences, etc. per request.
- **A:** Many parameters beyond the prompt are configurable.
- **C:** Connection is set at handle creation; generation calls accept model parameters.
- **D:** Parameters are highly configurable for each generation call.

---

## Section 2: Custom Model Patterns (35 points)

### Question 5 (9 points)

What is a "wrapper" pattern in custom LLM applications?

A) Encrypting LLM requests for security
B) Adding pre-processing or post-processing logic around LLM calls
C) Compressing prompts to reduce token usage
D) Converting between different LLM provider APIs

**Answer:** B

**Explanation:**
- **B (Correct):** Wrappers add functionality before/after LLM calls, such as input validation, output formatting, or logging.
- **A:** While encryption could be part of a wrapper, it's not the defining characteristic.
- **C:** Compression might be a specific wrapper feature, but not the general pattern.
- **D:** Provider abstraction is handled by LLM Mesh; wrappers add business logic.

---

### Question 6 (9 points)

You need to call multiple LLMs in sequence, where each uses the previous output. What pattern is this?

A) Parallel ensemble
B) Sequential chain
C) Router pattern
D) Batch processing

**Answer:** B

**Explanation:**
- **B (Correct):** A chain pattern calls LLMs sequentially, passing outputs as inputs to subsequent steps.
- **A:** Parallel ensemble calls multiple models simultaneously, not sequentially.
- **C:** Routers select one model based on criteria, not call multiple in sequence.
- **D:** Batch processing handles multiple inputs, not sequential LLM calls.

---

### Question 7 (8 points)

What is the purpose of a "router" pattern in LLM applications?

A) To automatically retry failed API calls
B) To dynamically select which LLM to use based on request characteristics
C) To distribute load across multiple API keys
D) To route requests to the cheapest available provider

**Answer:** B

**Explanation:**
- **B (Correct):** Routers implement conditional logic to select the appropriate model based on input type, complexity, or other factors.
- **A:** Retry logic is error handling, not routing.
- **C:** Load balancing is infrastructure-level, not application routing.
- **D:** While cost might be a routing criterion, routing is about selection logic, not just cost.

---

### Question 8 (9 points)

You want multiple LLMs to answer the same question and then aggregate their responses. What pattern is this?

A) Sequential chain
B) Router pattern
C) Ensemble pattern
D) Wrapper pattern

**Answer:** C

**Explanation:**
- **C (Correct):** Ensemble patterns call multiple models in parallel and combine their outputs (voting, averaging, etc.).
- **A:** Chains are sequential, not parallel.
- **B:** Routers select one model, not use multiple.
- **D:** Wrappers add logic around single calls, not aggregate multiple models.

---

## Section 3: Pipeline Integration (33 points)

### Question 9 (9 points)

Which Dataiku recipe type is most appropriate for calling LLMs on dataset rows?

A) Sync recipe
B) Python recipe
C) Visual recipe (Prepare)
D) SQL recipe

**Answer:** B

**Explanation:**
- **B (Correct):** Python recipes provide full control over LLM API calls and can iterate over dataset rows.
- **A:** Sync recipes copy data; they don't support custom LLM logic.
- **C:** Prepare recipes have limited scripting; LLM calls require Python.
- **D:** SQL recipes can't make LLM API calls directly.

---

### Question 10 (8 points)

Your Python recipe processes 10,000 rows, calling an LLM for each. What should you implement?

A) Make all 10,000 calls simultaneously for speed
B) Implement error handling, rate limiting, and progress tracking
C) Load all results into memory before writing
D) Skip error handling since LLM APIs are always reliable

**Answer:** B

**Explanation:**
- **B (Correct):** Production pipelines need robust error handling, rate limiting to avoid API throttling, and progress monitoring.
- **A:** Simultaneous calls would hit rate limits and overwhelm the API.
- **C:** Streaming results (write as you go) is better for large datasets.
- **D:** APIs can fail; error handling is essential.

---

### Question 11 (8 points)

How should you handle API rate limits when processing large datasets?

A) Ignore them and retry indefinitely
B) Implement exponential backoff and respect rate limit headers
C) Switch to a different API key after each error
D) Process everything in a single batch request

**Answer:** B

**Explanation:**
- **B (Correct):** Exponential backoff with rate limit awareness is the standard approach for resilient API usage.
- **A:** Infinite retries can cause account suspension and waste resources.
- **C:** Key rotation doesn't solve rate limiting and may violate terms of service.
- **D:** Most LLM APIs don't support batch requests; even if they do, rate limits still apply.

---

### Question 12 (8 points)

What is the benefit of saving LLM responses to a dataset rather than just displaying them?

A) Datasets are required for LLM Mesh to track usage
B) Enables downstream analysis, caching, and audit trails
C) LLM responses are automatically improved when saved
D) Saving is the only way to access response metadata

**Answer:** B

**Explanation:**
- **B (Correct):** Persisting responses enables reuse, analysis, versioning, and auditability without re-calling expensive APIs.
- **A:** Usage tracking works regardless of where responses are saved.
- **C:** Saving doesn't change response quality; it preserves results.
- **D:** Metadata is accessible in memory; saving enables persistence.

---

### Question 13 (Bonus 10 points)

You're building a data quality pipeline that uses LLMs to validate and correct data. What architecture is most appropriate?

A) Real-time API where each validation request calls the LLM synchronously
B) Batch pipeline that processes data in scheduled jobs with LLM enrichment
C) Streaming pipeline that processes each record as it arrives
D) Manual review where humans check LLM suggestions

**Answer:** B

**Explanation:**
- **B (Correct):** Data quality is typically batch-oriented; scheduled jobs allow cost control, monitoring, and efficient processing.
- **A:** Real-time LLM calls for every validation would be expensive and slow.
- **C:** Streaming is possible but adds complexity; batch is more practical for data quality.
- **D:** LLMs should augment, not replace, automated checks; human review can be selective.

---

## Answer Key Summary

| Question | Answer | Points | Topic |
|----------|--------|--------|-------|
| 1 | C | 8 | LLMHandle creation |
| 2 | B | 8 | Messages parameter |
| 3 | B | 8 | Token usage |
| 4 | B | 8 | Generation parameters |
| 5 | B | 9 | Wrapper pattern |
| 6 | B | 9 | Chain pattern |
| 7 | B | 8 | Router pattern |
| 8 | C | 9 | Ensemble pattern |
| 9 | B | 9 | Recipe selection |
| 10 | B | 8 | Production considerations |
| 11 | B | 8 | Rate limiting |
| 12 | B | 8 | Response persistence |
| 13 | B | 10 | Architecture design |

**Total:** 110 points (100 base + 10 bonus)

---

## Performance Indicators

**95-110:** Expert-level understanding of custom LLM integration
**85-94:** Strong grasp of Python API and design patterns
**70-84:** Adequate understanding, practice more with pipeline integration
**Below 70:** Review module materials, especially Python API usage

## Common Mistakes to Avoid

1. **Incorrect API syntax**: Use `dataiku.LLMHandle()`, not variations
2. **Missing error handling**: Always implement robust error handling for API calls
3. **Ignoring rate limits**: Respect API rate limits to avoid throttling
4. **Not tracking usage**: Monitor token usage for cost control
5. **Synchronous processing at scale**: Use batch patterns for large datasets

## Key Concepts to Master

### Python API Essentials
```python
# Create handle
llm = dataiku.LLMHandle("connection-name")

# Generate response
response = llm.generate(
    messages=[
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User prompt"}
    ],
    temperature=0.7,
    max_tokens=1000
)

# Access results
text = response.text
tokens = response.usage.total_tokens
```

### Design Patterns
- **Wrapper**: Add logic around LLM calls
- **Chain**: Sequential multi-step processing
- **Router**: Dynamic model selection
- **Ensemble**: Multiple models with aggregation

### Production Best Practices
- Error handling and retries
- Rate limiting and backoff
- Progress tracking
- Response caching
- Cost monitoring

## Next Steps

After completing this quiz:
1. Review any questions you missed
2. Complete the Python integration hands-on exercises
3. Build a custom LLM application using design patterns
4. Implement production-grade error handling
5. Proceed to Module 4: Deployment and Governance when ready
