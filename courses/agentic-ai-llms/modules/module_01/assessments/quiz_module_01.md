# Quiz: Module 1 - LLM Fundamentals for Agents

**Estimated Time:** 18 minutes
**Total Points:** 100
**Passing Score:** 70%

---

## Instructions

This quiz assesses your understanding of system prompts, chain-of-thought reasoning, and few-shot learning. Answer all questions thoughtfully. Multiple choice questions have one best answer unless otherwise specified.

---

## Part A: System Prompts (35 points)

### Question 1 (8 points)
List the five essential components of an effective system prompt for an agent.

**Your Answer:**

---

### Question 2 (7 points)
Which system prompt structure is most effective for an agent that needs to follow strict procedures?

A)
```
You are a helpful assistant. Please help the user with their request.
```

B)
```
You are a customer support agent.

Capabilities:
- Search knowledge base
- Create support tickets
- Escalate to human agents

Constraints:
- Never make promises about refunds without manager approval
- Always verify customer identity before sharing account details
- Do not share internal system information

Response Format:
1. Acknowledge the issue
2. Propose solution or next steps
3. Ask if anything else is needed
```

C)
```
You are an expert at customer support. Be friendly and helpful.
```

D)
```
Answer customer questions accurately and professionally.
```

**Your Answer:**

---

### Question 3 (10 points)
You're building a code review agent. Write a system prompt that includes appropriate identity, capabilities, and constraints. (4-6 sentences)

**Your Answer:**

---

### Question 4 (5 points)
**True or False:** System prompts should include specific tool definitions and JSON schemas directly in the system prompt text to ensure the agent knows how to use them.

**Your Answer:**

---

### Question 5 (5 points)
What is the primary risk of having an overly long system prompt (e.g., >2000 tokens)?

A) It increases per-request costs and latency
B) It confuses the model with too many instructions
C) It violates API rate limits
D) Both A and B

**Your Answer:**

---

## Part B: Chain-of-Thought Reasoning (40 points)

### Question 6 (8 points)
Explain the difference between zero-shot chain-of-thought and few-shot chain-of-thought prompting. When would you use each?

**Your Answer:**

---

### Question 7 (7 points)
Which prompt modification implements zero-shot chain-of-thought?

A) "Calculate the result: 15% of 240"
B) "Calculate the result step by step: 15% of 240"
C) "Let's think step by step. Calculate: 15% of 240"
D) "Example: 10% of 100 = 10. Now calculate: 15% of 240"

**Your Answer:**

---

### Question 8 (10 points)
You have an agent that needs to determine if a customer is eligible for a premium feature based on multiple criteria (account age, subscription tier, usage metrics). Design a chain-of-thought prompt structure for this task. Show the key steps the agent should follow.

**Your Answer:**

---

### Question 9 (8 points)
What is self-consistency in chain-of-thought reasoning, and what problem does it solve?

**Your Answer:**

---

### Question 10 (7 points)
**True or False:** Tree-of-thought (ToT) reasoning explores multiple reasoning paths simultaneously and is most useful for problems with many possible solution approaches.

**Your Answer:**

---

## Part C: Few-Shot Learning (25 points)

### Question 11 (8 points)
What are the three key elements that should be included in each few-shot example for agent training?

A) Input, output, timestamp
B) Input, reasoning trace, output
C) Input, model used, output
D) Input, temperature setting, output

**Your Answer:**

---

### Question 12 (10 points)
Create 2 few-shot examples for an agent that extracts action items from meeting notes. Each example should demonstrate proper formatting and edge case handling.

**Your Answer:**

---

### Question 13 (7 points)
What is the main trade-off when increasing the number of few-shot examples in a prompt?

A) More examples always improve performance with no downsides
B) More examples improve accuracy but increase token costs and may hit context limits
C) More examples confuse the model
D) More examples only help for simple tasks

**Your Answer:**

---

## Part D: Prompt Optimization (Bonus Section - 10 points)

### Question 14 (5 points)
Which optimization technique can reduce prompt tokens while maintaining quality?

A) Remove all examples and rely on zero-shot prompting
B) Use abbreviations and shortened words throughout
C) Template reusable prompt components and swap in task-specific sections
D) Remove all formatting and whitespace

**Your Answer:**

---

### Question 15 (5 points)
You have a prompt that's 3000 tokens and needs to be called 10,000 times daily. Identify one specific way to reduce costs while maintaining output quality.

**Your Answer:**

---

# Answer Key

## Part A: System Prompts

**Question 1:** (8 points)
*Answer:*
1. Identity - Who is the agent?
2. Capabilities - What can it do?
3. Constraints - What should it avoid?
4. Format - How should it respond?
5. Tools - What tools are available?

**Question 2:** Answer: B (7 points)
*Explanation:* Option B provides clear structure with defined capabilities, explicit constraints, and response format—essential for procedural compliance.

**Question 3:** (10 points)
*Sample Answer:* You are a senior code review agent specializing in security and best practices. Your capabilities include analyzing code for vulnerabilities, identifying code smells, and suggesting improvements. You must focus only on significant issues (security, performance, correctness), not style preferences. Always provide specific line numbers and concrete fix suggestions. Never approve code with critical security vulnerabilities. Format responses as: Issue → Severity → Location → Recommended Fix.

*Grading:*
- Identity (2 pts)
- Capabilities (2 pts)
- Constraints (3 pts)
- Format (3 pts)

**Question 4:** False (5 points)
*Explanation:* Tool definitions should be provided through the API's tool/function parameter, not embedded in the system prompt text. This keeps prompts focused and allows the API to properly handle tool calling.

**Question 5:** Answer: D (5 points)
*Explanation:* Long system prompts increase costs (charged every request) and latency, AND can dilute attention on critical instructions, leading to missed guidance.

## Part B: Chain-of-Thought Reasoning

**Question 6:** (8 points)
*Sample Answer:* Zero-shot CoT uses prompts like "Let's think step by step" without examples, relying on the model's training. Few-shot CoT provides examples showing reasoning traces. Use zero-shot for general reasoning when examples are hard to create. Use few-shot when you need specific reasoning patterns or domain-specific logic.

**Question 7:** Answer: C (7 points)
*Explanation:* "Let's think step by step" is the canonical zero-shot CoT trigger phrase that prompts step-by-step reasoning without examples.

**Question 8:** (10 points)
*Sample Answer:*
```
Check customer eligibility step by step:

Step 1: Verify account age
- Check if account > 90 days old
- Result: PASS/FAIL

Step 2: Verify subscription tier
- Check if tier is "Pro" or "Enterprise"
- Result: PASS/FAIL

Step 3: Check usage metrics
- Verify monthly active usage > threshold
- Result: PASS/FAIL

Step 4: Final determination
- If all steps PASS → ELIGIBLE
- If any step FAIL → NOT ELIGIBLE, specify reason

Input: {customer_data}
```

*Grading:*
- Clear step sequence (4 pts)
- Specific criteria (3 pts)
- Pass/fail logic (3 pts)

**Question 9:** (8 points)
*Sample Answer:* Self-consistency generates multiple independent reasoning paths for the same problem, then selects the most common answer. It solves the problem of reasoning inconsistency—the model might get different answers on different attempts. By sampling multiple paths and taking a majority vote, it improves reliability for complex reasoning.

**Question 10:** True (7 points)
*Explanation:* Tree-of-thought explicitly explores multiple reasoning branches, evaluating and pruning paths, making it ideal for problems with many solution approaches.

## Part C: Few-Shot Learning

**Question 11:** Answer: B (8 points)
*Explanation:* Effective few-shot examples include input (the task), reasoning trace (how to think about it), and output (the expected result). This teaches both what to output and how to reason.

**Question 12:** (10 points)
*Sample Answer:*

Example 1:
```
Input: "Team agreed to finalize Q4 budget by Friday. Sarah will send updated projections. Need to schedule follow-up meeting next week."
Output:
- [ ] Finalize Q4 budget (Due: Friday, Owner: Team)
- [ ] Send updated projections (Owner: Sarah)
- [ ] Schedule follow-up meeting (Due: Next week, Owner: Unassigned)
```

Example 2 (Edge case - no action items):
```
Input: "Discussed the new product launch. Everyone is excited about the potential. Sales numbers look promising."
Output:
No action items identified - this was an informational discussion.
```

*Grading:*
- Two complete examples (4 pts)
- Proper formatting (3 pts)
- Edge case handling (3 pts)

**Question 13:** Answer: B (7 points)
*Explanation:* More examples generally improve accuracy but consume more tokens (increasing costs) and can approach context window limits, requiring careful balance.

## Part D: Prompt Optimization

**Question 14:** Answer: C (5 points)
*Explanation:* Templating reusable components allows you to maintain quality while reducing repetition. Other options sacrifice clarity or quality.

**Question 15:** (5 points)
*Sample Answer:* Extract the static system prompt portion (possibly 2500 tokens) and use prompt caching (supported by Claude). This caches the unchanging prefix, reducing tokens charged on repeated calls. With 10K daily calls, this could save 25M tokens daily.

*Acceptable alternatives:*
- Compress prompt by removing redundant examples while keeping representative ones
- Use a smaller model for simpler sub-tasks
- Implement response caching for repeated queries

---

## Scoring Guide

- **90-100 points:** Excellent - Strong command of prompting fundamentals
- **80-89 points:** Good - Solid understanding with room for refinement
- **70-79 points:** Passing - Core concepts understood, practice more
- **Below 70:** Review module materials before advancing

**Key Topics to Review if Struggling:**
- System prompt structure and components
- Zero-shot vs. few-shot chain-of-thought
- Few-shot example design principles
- Prompt optimization strategies
- Tree-of-thought for complex reasoning
