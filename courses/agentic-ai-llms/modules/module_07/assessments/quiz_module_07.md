# Quiz: Module 7 - Production Deployment

**Estimated Time:** 20 minutes
**Total Points:** 100
**Passing Score:** 70%

---

## Instructions

This quiz evaluates your understanding of production architecture, observability, and optimization for agent systems at scale. Focus on real-world deployment challenges and solutions.

---

## Part A: Production Architecture (30 points)

### Question 1 (10 points)
List five key production concerns for agent systems and one solution/strategy for each.

**Your Answer:**

---

### Question 2 (8 points)
What is a circuit breaker pattern, and why is it important for production agent systems?

**Your Answer:**

---

### Question 3 (7 points)
Which architectural pattern is best for handling temporary API failures in production?

A) Crash immediately and alert developers
B) Return cached responses indefinitely
C) Retry with exponential backoff, then fallback to degraded functionality
D) Ignore errors and continue

**Your Answer:**

---

### Question 4 (5 points)
**True or False:** In production, you should always use the most capable (and expensive) model to ensure best quality.

**Your Answer:**

---

## Part B: Observability (35 points)

### Question 5 (12 points)
Describe the three pillars of observability and provide specific examples for each in the context of agent systems:

1. **Logs:**
2. **Metrics:**
3. **Traces:**

**Your Answer:**

---

### Question 6 (10 points)
Design a structured log entry for an agent's tool execution. Include at least 6 fields that would be valuable for debugging and monitoring.

**Your Answer:**

---

### Question 7 (8 points)
You receive an alert that your agent's P95 latency has increased from 2s to 8s. What are the first three things you would investigate?

**Your Answer:**

---

### Question 8 (5 points)
What is distributed tracing, and why is it essential for multi-step agent workflows?

**Your Answer:**

---

## Part C: Optimization (35 points)

### Question 9 (10 points)
Describe five cost optimization strategies for production agent systems and estimate the potential savings for each (low/medium/high).

**Your Answer:**

---

### Question 10 (8 points)
You have an agent that processes 100,000 queries daily. Each query uses a 2,000-token system prompt. How could prompt caching reduce costs, and what would be the requirements?

**Your Answer:**

---

### Question 11 (9 points)
Compare these three latency optimization strategies:

1. **Streaming responses**
2. **Parallel tool execution**
3. **Model selection (faster model)**

For each, describe: how it works, when to use it, and one trade-off.

**Your Answer:**

---

### Question 12 (8 points)
What is the purpose of batching requests in production systems? Identify two scenarios where batching is beneficial and one where it's not appropriate.

**Your Answer:**

---

## Part D: Error Handling & Recovery (Bonus - 10 points)

### Question 13 (5 points)
Design a fallback strategy for when the primary LLM API is unavailable. Include at least 3 fallback levels.

**Your Answer:**

---

### Question 14 (5 points)
What is the difference between a "hard failure" and a "soft failure" in production systems? Provide one example of each for an agent system.

**Your Answer:**

---

# Answer Key

## Part A: Production Architecture

**Question 1:** (10 points)
*Sample Answer:*

1. **Reliability** → Implement retry logic with exponential backoff
2. **Observability** → Structured logging, distributed tracing, metrics collection
3. **Performance** → Caching, model routing, parallel execution
4. **Cost Control** → Token budgeting, prompt caching, model selection
5. **Security** → Input validation, output filtering, access control

*Grading:* 2 points per concern + solution

**Question 2:** (8 points)
*Sample Answer:* A circuit breaker prevents cascading failures by stopping requests to failing services after a threshold of failures. When "open," it immediately fails requests without attempting the call. After a timeout, it enters "half-open" to test if service recovered. This is important for agents because: (1) prevents wasting time/resources on calls that will fail, (2) gives failing services time to recover, (3) enables graceful degradation, and (4) prevents cascading failures across systems.

**Question 3:** Answer: C (7 points)
*Explanation:* Retry with exponential backoff handles transient failures, while fallback to degraded functionality (cached data, simpler model, error message) ensures the system remains operational rather than completely failing.

**Question 4:** False (5 points)
*Explanation:* Production systems should use model routing—use cheaper/faster models for simple tasks, expensive models only when needed. This balances quality with cost and latency. Always using the most expensive model wastes resources and increases latency unnecessarily.

## Part B: Observability

**Question 5:** (12 points)
*Sample Answer:*

**1. Logs** - Detailed event records
- Example: "User query received", "Tool call executed", "Error in retrieval", each with structured fields (timestamp, user_id, query, etc.)

**2. Metrics** - Numerical measurements over time
- Example: Request rate (req/sec), token usage (tokens/hour), error rate (%), P95 latency (ms)

**3. Traces** - Request flow through distributed system
- Example: Trace showing: User request → Query embedding → Vector search → Context retrieval → LLM call → Response, with timing for each span

*Grading:* 4 points per pillar (description + agent-specific example)

**Question 6:** (10 points)
*Sample Answer:*
```json
{
  "timestamp": "2024-02-02T10:30:45.123Z",
  "trace_id": "abc123-def456",
  "agent_id": "customer-support-agent-1",
  "user_id": "user_12345",
  "tool_name": "search_knowledge_base",
  "tool_input": {"query": "return policy", "max_results": 5},
  "tool_output": {"results": [...], "count": 3},
  "execution_time_ms": 245,
  "status": "success",
  "error": null,
  "tokens_used": {"input": 150, "output": 200}
}
```

*Grading:* 1.5 points per useful field (need 6+)
Must include: timestamp, identifiers, tool info, execution details, status/error

**Question 7:** (8 points)
*Sample Answer:*
1. **Token usage spike** - Check if queries are longer or responses more verbose (could indicate prompt injection or hallucination)
2. **External API latency** - Check if tools (search, database, etc.) are slow
3. **Model selection** - Verify correct model is being used (accidental switch to slower model)

*Additional valid investigations:*
- Concurrent request load
- Context window size increases
- Network issues
- Resource constraints (CPU/memory)

*Grading:* 2.5 points each for three valid, distinct investigations

**Question 8:** (5 points)
*Sample Answer:* Distributed tracing tracks requests across multiple services/steps, showing the complete path and timing. For multi-step agents, it's essential because workflows span many components (LLM calls, tool executions, retrievals), making it impossible to debug latency or failures without seeing the full request flow and identifying which step is slow or failing.

## Part C: Optimization

**Question 9:** (10 points)
*Sample Answer:*

1. **Prompt Caching** - Cache static system prompts (High savings: 50-80% token reduction)
2. **Model Routing** - Use cheaper models for simple queries (Medium-High: 30-60% cost reduction)
3. **Response Caching** - Cache responses for repeated queries (Medium: 20-40% reduction depending on query diversity)
4. **Prompt Optimization** - Reduce token count while maintaining quality (Low-Medium: 10-30%)
5. **Token Budgeting** - Set max_tokens limits to prevent runaway costs (Low: 5-15%)

*Grading:* 2 points per strategy (name + estimated savings)

**Question 10:** (8 points)
*Sample Answer:* Prompt caching (e.g., Claude's prompt caching) stores the 2,000-token system prompt after first use. Subsequent requests only pay for cache read (90% discount) instead of full input tokens.

**Calculation:**
- Without caching: 100K × 2,000 = 200M tokens daily
- With caching: ~2K (initial) + 100K × 200 (cache read cost) ≈ 20M token equivalent
- Savings: ~90% on system prompt tokens

**Requirements:**
- System prompt must be consistent across requests
- Must exceed minimum cache size (typically 1024+ tokens)
- Cache TTL consideration for updates

*Grading:* Explanation (4 pts) + Requirements (4 pts)

**Question 11:** (9 points)
*Sample Answer:*

**1. Streaming Responses**
- How: Return tokens as generated, not waiting for completion
- When: Long responses where user wants immediate feedback
- Trade-off: Can't validate full response before returning, complex client-side handling

**2. Parallel Tool Execution**
- How: Execute independent tool calls concurrently
- When: Multiple tools needed that don't depend on each other
- Trade-off: Increased complexity, need to manage concurrent state

**3. Model Selection**
- How: Use faster, smaller model (e.g., Haiku instead of Opus)
- When: Simple queries that don't need maximum capability
- Trade-off: Lower quality/accuracy for complex tasks

*Grading:* 3 points per strategy (how + when + trade-off)

**Question 12:** (8 points)
*Sample Answer:*
**Purpose:** Batching groups multiple requests into a single API call to reduce overhead, latency, and costs.

**Beneficial:**
1. **Background processing** - Analyzing 1000 support tickets overnight
2. **Bulk embeddings** - Embedding a large document corpus

**Not appropriate:**
- **Real-time user queries** - User expects immediate response, batching adds unacceptable latency

*Grading:* Purpose (3 pts), beneficial scenarios (2.5 pts), inappropriate scenario (2.5 pts)

## Part D: Error Handling & Recovery

**Question 13:** (5 points)
*Sample Answer:*

**Level 1:** Retry primary API with exponential backoff (3 attempts)

**Level 2:** Switch to backup LLM provider (e.g., Claude → GPT-4)

**Level 3:** Return cached response if query seen before

**Level 4:** Return graceful error message with alternative actions (e.g., "Service temporarily unavailable, try again in 5 minutes or contact support@...")

**Level 5:** (Optional) Queue request for async processing when service recovers

*Grading:* Must include 3+ distinct fallback levels with clear progression

**Question 14:** (5 points)
*Sample Answer:*
**Hard Failure:** Complete system failure requiring manual intervention.
- Example: Database corrupted, cannot start agent service

**Soft Failure:** Partial degradation with reduced functionality but system still operational.
- Example: Vector search unavailable → agent continues with keyword search or cached results

Hard failures stop operations; soft failures enable graceful degradation.

---

## Scoring Guide

- **90-100 points:** Excellent - Ready to deploy production agent systems
- **80-89 points:** Good - Strong understanding, practice with real deployments
- **70-79 points:** Passing - Core concepts understood, gain more operational experience
- **Below 70:** Review module and study production case studies

**Key Topics to Review if Struggling:**
- Circuit breaker and retry patterns
- Three pillars of observability (logs, metrics, traces)
- Cost optimization strategies (caching, routing, batching)
- Latency optimization techniques
- Structured logging design
- Fallback strategies and graceful degradation
- Hard vs soft failures
- When to use streaming vs parallel execution
