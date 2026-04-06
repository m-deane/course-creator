# Chain-of-Thought Reasoning

> **Reading time:** ~10 min | **Module:** 1 — Advanced Prompting | **Prerequisites:** Module 0 — Transformer Architecture

Chain-of-thought (CoT) prompting makes LLMs show their reasoning process before giving answers. This simple technique dramatically improves performance on complex tasks—reasoning, math, logic, and multi-step problems.

<div class="callout-insight">

**Insight:** When models think out loud, they think better. Autoregressive generation means each token builds on previous tokens. By generating reasoning steps, the model creates useful context for its final answer. The reasoning isn't just explanation—it's computation.

</div>

---

## Why Chain-of-Thought Works

### The Autoregressive Advantage

Remember: LLMs generate tokens sequentially. Each new token is influenced by all previous tokens.

```
Without CoT:
Q: What is 23 × 47?
A: 1081  ← Must compute in one step (often wrong)

With CoT:
Q: What is 23 × 47?
A: Let me calculate step by step:
   23 × 47 = 23 × (40 + 7)
   = 23 × 40 + 23 × 7
   = 920 + 161
   = 1081  ← Each step provides context for the next
```

### When to Use CoT

| Task Type | CoT Benefit | Example |
|-----------|-------------|---------|
| Math/Logic | High | Multi-step calculations |
| Reasoning | High | Causal analysis |
| Code | Medium | Debugging complex issues |
| Classification | Low | Sentiment analysis |
| Translation | Low | Direct language transfer |

---

## Chain-of-Thought Variants

### 1. Zero-Shot CoT

The simplest form—just add "Let's think step by step":


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
prompt = """A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball.
How much does the ball cost?

Let's think step by step."""
```

</div>
</div>

**Output:**
```
Let's think step by step.

Let me call the ball's price "x"
Then the bat's price is "x + $1.00"
Together: x + (x + $1.00) = $1.10
So: 2x + $1.00 = $1.10
Therefore: 2x = $0.10
And: x = $0.05

The ball costs $0.05 (5 cents).
```

### 2. Few-Shot CoT

Provide examples with reasoning:

```python
prompt = """Solve these problems by thinking step by step.

Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?
A: Let's think step by step.
Roger started with 5 balls.
He bought 2 cans × 3 balls per can = 6 balls.
Total: 5 + 6 = 11 tennis balls.
The answer is 11.

Q: A juggler has 16 balls. Half are red and half are blue.
He loses 3 red balls. How many balls does he have now?
A: Let's think step by step.
Started with 16 balls total.
Red balls: 16 / 2 = 8
Blue balls: 16 / 2 = 8
After losing 3 red: 8 - 3 = 5 red balls
Total now: 5 + 8 = 13 balls.
The answer is 13.

Q: {new_problem}
A: Let's think step by step."""
```

### 3. Self-Consistency

Generate multiple reasoning paths, take the majority answer:

```python
import anthropic
from collections import Counter

client = anthropic.Anthropic()


def solve_with_self_consistency(problem: str, n_samples: int = 5) -> str:
    """Solve a problem using self-consistency (multiple reasoning paths)."""

    answers = []

    for _ in range(n_samples):
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.7,  # Higher temperature for diversity
            messages=[{
                "role": "user",
                "content": f"""Solve this problem step by step, then give your final answer.

Problem: {problem}

Think through this carefully, then state your final answer on the last line as:
ANSWER: [your answer]"""
            }]
        )

        # Extract the answer
        text = response.content[0].text
        if "ANSWER:" in text:
            answer = text.split("ANSWER:")[-1].strip()
            answers.append(answer)

    # Return majority answer
    if answers:
        return Counter(answers).most_common(1)[0][0]
    return "Unable to determine answer"


# Usage
problem = "If a train travels 120 miles in 2 hours, then 90 miles in 1.5 hours, what is its average speed for the whole journey?"
answer = solve_with_self_consistency(problem)
print(f"Answer: {answer}")  # ~60 mph
```

### 4. Tree-of-Thought (ToT)

Explore multiple reasoning branches:

```python
def tree_of_thought(problem: str, breadth: int = 3, depth: int = 3) -> str:
    """
    Tree-of-Thought: Generate multiple reasoning paths,
    evaluate each, and explore the most promising.
    """

    def generate_thoughts(context: str, n: int) -> list[str]:
        """Generate n possible next thoughts."""
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.8,
            messages=[{
                "role": "user",
                "content": f"""Given this problem and reasoning so far:

{context}

Generate {n} different possible next steps in the reasoning.
Format as:
1. [thought 1]
2. [thought 2]
...
"""
            }]
        )
        # Parse thoughts from response
        text = response.content[0].text
        thoughts = []
        for line in text.split('\n'):
            if line.strip() and line[0].isdigit():
                thought = line.split('.', 1)[-1].strip()
                thoughts.append(thought)
        return thoughts[:n]

    def evaluate_thought(context: str, thought: str) -> float:
        """Evaluate how promising a thought is (0-1)."""
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            temperature=0,
            messages=[{
                "role": "user",
                "content": f"""Rate this reasoning step from 0.0 to 1.0.

Problem context:
{context}

Proposed step:
{thought}

Consider: Is this step logical? Does it make progress? Is it on the right track?

Reply with just a number between 0.0 and 1.0."""
            }]
        )
        try:
            return float(response.content[0].text.strip())
        except ValueError:
            return 0.5

    # Initialize with problem
    paths = [(problem, 0.5)]  # (context, score)

    for _ in range(depth):
        new_paths = []
        for context, _ in paths[:breadth]:
            thoughts = generate_thoughts(context, breadth)
            for thought in thoughts:
                new_context = f"{context}\n\nStep: {thought}"
                score = evaluate_thought(context, thought)
                new_paths.append((new_context, score))

        # Keep top paths
        paths = sorted(new_paths, key=lambda x: x[1], reverse=True)[:breadth]

    # Get final answer from best path
    best_path = paths[0][0]
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""Based on this reasoning:

{best_path}

What is the final answer?"""
        }]
    )
    return response.content[0].text
```

---

## Structured Reasoning Patterns

### The ReAct Pattern (Preview)

Reasoning + Acting in an interleaved loop:

```
Thought: I need to find the population of France.
Action: search("France population 2024")
Observation: France has a population of approximately 68 million.
Thought: Now I can answer the question.
Answer: France has approximately 68 million people.
```

### Step-Back Prompting

Ask for abstract principles before specific solutions:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
prompt = """Before solving this problem, let's step back.

Problem: {specific_problem}

Step 1: What general principles or concepts apply to this type of problem?
Step 2: How do these principles guide the solution?
Step 3: Now solve the specific problem using these principles.

Begin:"""
```

</div>
</div>

### Decomposition Prompting

Break complex questions into sub-questions:

```python
prompt = """To answer this question, let me break it into smaller parts.

Question: {complex_question}

Sub-questions I need to answer:
1. [identify sub-question 1]
2. [identify sub-question 2]
...

Now I'll answer each:

Sub-question 1: ...
"""
```

---

## Implementation Patterns

### Basic CoT Wrapper


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
def with_cot(prompt: str, cot_trigger: str = "Let's think step by step.") -> str:
    """Add chain-of-thought to any prompt."""
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"{prompt}\n\n{cot_trigger}"
        }]
    )
    return response.content[0].text


# Usage
answer = with_cot("What would happen if the moon suddenly disappeared?")
```

</div>
</div>

### CoT with Answer Extraction

```python
def solve_with_cot(problem: str) -> dict:
    """Solve a problem with CoT and extract structured output."""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"""Solve this problem step by step.

Problem: {problem}

After your reasoning, provide your answer in this exact format:
<answer>your final answer here</answer>"""
        }]
    )

    text = response.content[0].text

    # Extract reasoning and answer
    if "<answer>" in text and "</answer>" in text:
        reasoning = text.split("<answer>")[0].strip()
        answer = text.split("<answer>")[1].split("</answer>")[0].strip()
    else:
        reasoning = text
        answer = text.split('\n')[-1]  # Fallback to last line

    return {
        "reasoning": reasoning,
        "answer": answer,
        "full_response": text
    }
```

### Verifier Pattern

Have the model check its own work:

```python
def solve_and_verify(problem: str) -> dict:
    """Solve a problem, then verify the solution."""

    # Step 1: Solve
    solution = solve_with_cot(problem)

    # Step 2: Verify
    verification = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"""Check if this solution is correct.

Problem: {problem}

Proposed solution:
{solution['reasoning']}

Final answer: {solution['answer']}

Verify step by step. Are there any errors? Is the final answer correct?
End with: VERDICT: CORRECT or VERDICT: INCORRECT"""
        }]
    )

    verdict_text = verification.content[0].text
    is_correct = "VERDICT: CORRECT" in verdict_text

    return {
        **solution,
        "verification": verdict_text,
        "verified_correct": is_correct
    }
```

---

## When CoT Hurts

Chain-of-thought isn't always beneficial:

### Cases Where CoT is Unnecessary

1. **Simple factual recall**: "What is the capital of France?"
2. **Basic classification**: "Is this email spam?"
3. **Direct translation**: "Translate 'hello' to Spanish"

### Cases Where CoT Can Hurt

1. **Time-sensitive applications**: CoT adds latency
2. **Token-limited contexts**: Reasoning consumes output tokens
3. **Overconfident reasoning**: Model may rationalize incorrect answers


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>

```python
def adaptive_cot(problem: str, complexity_threshold: float = 0.5) -> str:
    """Use CoT only for complex problems."""

    # Quick complexity assessment
    complexity = client.messages.create(
        model="claude-3-haiku-20240307",  # Fast, cheap
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": f"Rate problem complexity 0-1: {problem}"
        }]
    )

    try:
        score = float(complexity.content[0].text.strip())
    except ValueError:
        score = 0.5

    # Use CoT only if complex enough
    if score > complexity_threshold:
        return with_cot(problem)
    else:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": problem}]
        )
        return response.content[0].text
```


---

## Best Practices

1. **Match CoT to Task Complexity**: Don't over-engineer simple problems
2. **Use Temperature Wisely**: Low temperature (0) for deterministic reasoning, higher (0.7) for self-consistency
3. **Extract Final Answers**: Use structured output markers to parse results
4. **Verify Critical Results**: For high-stakes tasks, add verification steps
5. **Combine Techniques**: ToT + Self-Consistency for the hardest problems

<div class="callout-key">

**Key Concept Summary:** This guide covered the core concepts. Review the companion slides for visual summaries and the hands-on notebook for practice implementations.


---

*Chain-of-thought transforms LLMs from pattern matchers to reasoners. Master these techniques—they're the foundation for every advanced agent capability.*




## Practice Questions

1. Explain in your own words how the concepts in this guide relate to building production agents.
2. What are the key tradeoffs you need to consider when applying these techniques?
3. Describe a scenario where the approach from this guide would be the wrong choice, and what you would use instead.

---

**Next Steps:**

<a class="link-card" href="./02_chain_of_thought_slides.md">
  <div class="link-card-title">Chain-of-Thought Reasoning — Companion Slides</div>
  <div class="link-card-description">Visual slide deck with diagrams, speaker notes, and key takeaways.</div>
</a>

<a class="link-card" href="../notebooks/01_system_prompt_design.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with working code and guided exercises.</div>
</a>
