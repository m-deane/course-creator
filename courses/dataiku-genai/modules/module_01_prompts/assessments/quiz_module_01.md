# Module 1 Quiz: Prompt Design with Prompt Studios

**Course:** Gen AI & Dataiku: LLM Mesh Use Cases
**Module:** 1 - Prompt Design with Prompt Studios
**Time Limit:** 20 minutes
**Total Points:** 100
**Passing Score:** 70/100

## Instructions

- This quiz has 13 questions covering Prompt Studios, template variables, and prompt optimization
- Select the best answer for each question
- Point values are indicated for each question
- You have 2 attempts per question
- Refer to module guides and notebooks if needed

---

## Section 1: Prompt Studios Basics (30 points)

### Question 1 (8 points)

What is the PRIMARY advantage of using Dataiku's Prompt Studios over hardcoding prompts in Python?

A) Prompt Studios automatically improves prompt quality using AI
B) Visual interface enables testing, versioning, and iteration without code changes
C) Prompts created in Prompt Studios run faster than hardcoded prompts
D) Prompt Studios provides free access to all LLM providers

**Answer:** B

**Explanation:**
- **B (Correct):** Prompt Studios provides a visual environment for rapid iteration, testing with different inputs, version control, and team collaboration without modifying code.
- **A:** Prompt Studios doesn't automatically improve prompts; it provides tools for systematic improvement.
- **C:** Execution speed is the same; the benefit is in development workflow.
- **D:** LLM provider access still requires proper connections and API keys.

---

### Question 2 (8 points)

In Prompt Studios, what is the purpose of the "System Prompt" section?

A) To specify technical configuration like temperature and max tokens
B) To define the LLM's role, behavior, and constraints
C) To list the template variables that will be used
D) To store example test inputs for validation

**Answer:** B

**Explanation:**
- **B (Correct):** The system prompt establishes the LLM's persona, expertise, behavior guidelines, and overall context.
- **A:** Technical parameters are configured separately from the prompt content.
- **C:** Variables are defined in the user template, not the system prompt.
- **D:** Test inputs are stored in a separate testing interface.

---

### Question 3 (7 points)

You've created a prompt in Prompt Studios. How do you use it in a Python recipe?

A) Copy and paste the prompt text into your Python code
B) Reference the Prompt Studio ID using the Dataiku API
C) Export the prompt as a JSON file and import it
D) Prompt Studios automatically injects prompts into all recipes

**Answer:** B

**Explanation:**
- **B (Correct):** Use the Dataiku API to reference prompts by their ID: `prompt = dataiku.get_prompt("prompt-id")`.
- **A:** Copying defeats the purpose of centralized prompt management.
- **C:** While export is possible, direct API reference is the standard approach.
- **D:** Prompts must be explicitly referenced; there's no automatic injection.

---

### Question 4 (7 points)

Which statement about Prompt Studios version control is TRUE?

A) Only the most recent version of a prompt is saved
B) Prompt Studios maintains a history of all prompt versions
C) Version control requires Git integration
D) Each project must maintain its own prompt versions

**Answer:** B

**Explanation:**
- **B (Correct):** Prompt Studios automatically tracks all versions, allowing comparison and rollback.
- **A:** All versions are retained for audit and comparison purposes.
- **C:** Version control is built into Prompt Studios; Git integration is optional.
- **D:** Prompts are shared across projects; versioning is centralized.

---

## Section 2: Template Variables (35 points)

### Question 5 (9 points)

You're building a prompt that analyzes customer reviews. Which template variable syntax is correct?

A) `Analyze this review: ${customer_review}`
B) `Analyze this review: %customer_review%`
C) `Analyze this review: {{customer_review}}`
D) `Analyze this review: @customer_review`

**Answer:** C

**Explanation:**
- **C (Correct):** Dataiku Prompt Studios uses Mustache-style `{{variable}}` syntax for template variables.
- **A:** `${}` is used in shell scripts and some templating systems, but not Prompt Studios.
- **B:** `%%` is used in SQL and some other contexts.
- **D:** `@` is typically used for parameters in SQL or decorators in Python.

---

### Question 6 (9 points)

You have a list of product features and want to include all of them in a prompt. Which template syntax allows iteration?

A) `{{product_features}}`
B) `{{#product_features}}{{.}}{{/product_features}}`
C) `{{product_features[*]}}`
D) `{{foreach product_features}}`

**Answer:** B

**Explanation:**
- **B (Correct):** The Mustache section syntax `{{#array}}{{.}}{{/array}}` iterates over array elements, with `{{.}}` representing each item.
- **A:** This would insert the entire array as a string, not iterate.
- **C:** This is pseudo-code; actual iteration requires section syntax.
- **D:** `foreach` is not valid Mustache syntax.

---

### Question 7 (8 points)

Your prompt should include optional content based on whether a premium feature is enabled. Which syntax supports this?

A) `{{if premium}}Premium features: ...{{endif}}`
B) `{{#premium}}Premium features: ...{{/premium}}`
C) `{{premium ? "Premium features: ..." : ""}}`
D) `{{conditional premium "Premium features: ..."}}`

**Answer:** B

**Explanation:**
- **B (Correct):** Mustache sections handle conditionals; `{{#variable}}` renders content only if variable is truthy.
- **A:** Mustache doesn't use `if/endif` keywords.
- **C:** Ternary operator syntax isn't valid in Mustache templates.
- **D:** There's no `conditional` function in Mustache.

---

### Question 8 (9 points)

When passing variables to a Prompt Studio template from Python, what data type is supported?

A) Only strings
B) Strings, numbers, and booleans
C) Any JSON-serializable data structure
D) Only primitive types (no nested structures)

**Answer:** C

**Explanation:**
- **C (Correct):** Template variables can be complex JSON structures including nested objects and arrays.
- **A:** Multiple data types are supported, not just strings.
- **B:** Complex structures like lists and dictionaries are also supported.
- **D:** Nested structures are fully supported and commonly used.

---

## Section 3: Testing and Optimization (35 points)

### Question 9 (9 points)

What is the BEST approach to systematically improve a prompt's performance?

A) Increase the temperature parameter until results improve
B) Create multiple test cases and compare prompt variations side-by-side
C) Use the longest possible system prompt for maximum context
D) Always use few-shot examples regardless of the task

**Answer:** B

**Explanation:**
- **B (Correct):** Systematic testing with representative cases allows objective comparison of prompt variations.
- **A:** Temperature affects randomness, not prompt quality; optimization requires structured testing.
- **C:** Longer prompts aren't always better; clarity and relevance matter more.
- **D:** Few-shot examples are powerful but not always necessary; use them when they improve results.

---

### Question 10 (9 points)

You're testing a sentiment analysis prompt in Prompt Studios. What should your test dataset include?

A) Only positive sentiment examples to verify the prompt works
B) Representative examples covering all sentiment categories and edge cases
C) The largest possible dataset to ensure statistical significance
D) Randomly generated text to test robustness

**Answer:** B

**Explanation:**
- **B (Correct):** Test data should cover the full range of expected inputs including edge cases for comprehensive evaluation.
- **A:** Testing only one category doesn't validate the prompt's ability to distinguish sentiments.
- **C:** Quality and representativeness matter more than quantity for prompt testing.
- **D:** Random text doesn't represent real-world use cases; use authentic examples.

---

### Question 11 (8 points)

When should you use few-shot examples in a prompt?

A) Always, as more examples always improve results
B) When the task is ambiguous or requires specific output formatting
C) Only when the LLM doesn't understand natural language instructions
D) Never, as examples make prompts too long and expensive

**Answer:** B

**Explanation:**
- **B (Correct):** Few-shot examples are most valuable for demonstrating complex patterns, specific formats, or disambiguating instructions.
- **A:** Examples add cost and latency; use them when they provide clear value.
- **C:** Modern LLMs understand instructions well; examples enhance specific behaviors.
- **D:** Examples are valuable when appropriately used, despite added length.

---

### Question 12 (9 points)

You've deployed a prompt to production and notice inconsistent results. What is the FIRST debugging step?

A) Immediately increase the max_tokens parameter
B) Review the actual inputs being passed to the prompt template
C) Switch to a more expensive LLM model
D) Add more few-shot examples to the system prompt

**Answer:** B

**Explanation:**
- **B (Correct):** Verify that template variables are being populated correctly; input issues are the most common cause of inconsistency.
- **A:** Token limits cause truncation, not general inconsistency.
- **C:** Model changes should be evidence-based, not the first troubleshooting step.
- **D:** Adding examples without understanding the root cause is premature.

---

### Question 13 (Bonus 10 points)

A prompt performs well in Prompt Studios testing but poorly in production. What is the MOST likely cause?

A) The production LLM connection uses a different model version
B) The test data doesn't represent actual production input distribution
C) Prompt Studios uses optimized infrastructure not available in production
D) Template variables work differently in production vs. testing

**Answer:** B

**Explanation:**
- **B (Correct):** Test data mismatch is the most common cause of test-production performance gaps.
- **A:** While model versions can differ, this is less common if using the same connection.
- **C:** The underlying infrastructure is the same; only the execution context differs.
- **D:** Template variable behavior is consistent; the API is the same.

---

## Answer Key Summary

| Question | Answer | Points | Topic |
|----------|--------|--------|-------|
| 1 | B | 8 | Prompt Studios benefits |
| 2 | B | 8 | System prompts |
| 3 | B | 7 | Using prompts in Python |
| 4 | B | 7 | Version control |
| 5 | C | 9 | Variable syntax |
| 6 | B | 9 | Iteration syntax |
| 7 | B | 8 | Conditional syntax |
| 8 | C | 9 | Variable data types |
| 9 | B | 9 | Prompt optimization |
| 10 | B | 9 | Test dataset design |
| 11 | B | 8 | Few-shot examples |
| 12 | B | 9 | Debugging prompts |
| 13 | B | 10 | Production issues |

**Total:** 110 points (100 base + 10 bonus)

---

## Performance Indicators

**95-110:** Expert-level understanding of prompt engineering
**85-94:** Strong grasp of Prompt Studios and best practices
**70-84:** Adequate understanding, practice more with template variables
**Below 70:** Review module materials, focus on hands-on exercises

## Common Mistakes to Avoid

1. **Incorrect variable syntax**: Remember to use `{{variable}}`, not other formats
2. **Insufficient testing**: Always test with representative production-like data
3. **Over-reliance on examples**: Use few-shot examples strategically, not always
4. **Ignoring version control**: Track prompt changes for auditability and rollback

## Next Steps

After completing this quiz:
1. Review any questions you missed
2. Complete the Prompt Studios hands-on exercises
3. Practice creating prompts with complex template variables
4. Proceed to Module 2: RAG with Knowledge Banks when ready
