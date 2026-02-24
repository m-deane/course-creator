# Few-Shot Learning: Teaching by Example

## In Brief

Few-shot learning provides LLMs with examples of desired input-output pairs before the actual task. This technique dramatically improves consistency, accuracy, and format adherence—especially for tasks where verbal instructions are insufficient.

> 💡 **Key Insight:** **Show, don't just tell.** A single well-chosen example often conveys more than paragraphs of instructions. Examples demonstrate format, tone, edge case handling, and implicit rules that are hard to describe but easy to recognize.

---

## The Spectrum of Shot Learning

### Zero-Shot
No examples, just instructions:
```
Classify the sentiment: "This product exceeded my expectations!"
```

### One-Shot
One example before the task:
```
Example:
Input: "I love this!" → Positive

Classify: "This product exceeded my expectations!"
```

### Few-Shot
Multiple examples (typically 3-8):
```
Examples:
"I love this!" → Positive
"Terrible experience." → Negative
"It was okay." → Neutral

Classify: "This product exceeded my expectations!"
```

### Many-Shot
Large number of examples (10-100+):
- Higher accuracy but more tokens
- Useful for complex or nuanced tasks
- Consider fine-tuning at this scale

---

## Effective Example Selection

### Principle 1: Coverage

Include examples that cover the range of expected inputs:

```python
# Bad: All similar examples
examples = [
    ("I love it!", "Positive"),
    ("This is great!", "Positive"),
    ("Amazing product!", "Positive"),
]

# Good: Diverse coverage
examples = [
    ("I love it!", "Positive"),
    ("Terrible, waste of money.", "Negative"),
    ("It works, nothing special.", "Neutral"),
    ("Great features but poor battery.", "Mixed"),
]
```

### Principle 2: Difficulty Gradient

Include easy and hard examples:

```python
examples = [
    # Easy (clear sentiment)
    ("Best purchase ever!", "Positive"),
    ("Complete garbage.", "Negative"),

    # Hard (nuanced)
    ("Not bad, I suppose.", "Neutral"),  # Faint praise
    ("Could be worse.", "Neutral"),       # Backhanded
    ("Love the idea, hate the execution.", "Mixed"),
]
```

### Principle 3: Edge Cases

Show how to handle unusual situations:

```python
examples = [
    # Standard cases
    ("Great product!", "Positive"),

    # Edge cases
    ("", "Invalid - empty input"),
    ("???", "Invalid - no text content"),
    ("10/10 would not recommend", "Negative"),  # Sarcasm
    ("It's literally the worst best thing ever", "Mixed"),  # Contradiction
]
```

---

## Few-Shot Prompt Structures

### Basic Structure

```python
def create_few_shot_prompt(examples: list[tuple], task: str) -> str:
    """Create a few-shot prompt from examples."""

    prompt = "Examples:\n\n"

    for input_text, output_text in examples:
        prompt += f"Input: {input_text}\nOutput: {output_text}\n\n"

    prompt += f"Input: {task}\nOutput:"

    return prompt
```

### Structured with Instructions

```python
def create_guided_few_shot(
    task_description: str,
    examples: list[dict],
    input_text: str
) -> str:
    """Few-shot with explicit task description."""

    prompt = f"""{task_description}

Follow these examples exactly:

"""

    for ex in examples:
        prompt += f"---\nInput: {ex['input']}\nOutput: {ex['output']}\n"

    prompt += f"""---
Input: {input_text}
Output:"""

    return prompt


# Usage
prompt = create_guided_few_shot(
    task_description="Extract the main entities (people, places, organizations) from text.",
    examples=[
        {
            "input": "Apple CEO Tim Cook announced the event in Cupertino.",
            "output": "People: Tim Cook\nPlaces: Cupertino\nOrganizations: Apple"
        },
        {
            "input": "The UN met in Geneva to discuss climate policy.",
            "output": "People: None\nPlaces: Geneva\nOrganizations: UN"
        }
    ],
    input_text="Elon Musk's SpaceX launched from Cape Canaveral."
)
```

### With Reasoning (Few-Shot CoT)

```python
examples_with_reasoning = [
    {
        "input": "A farmer has 17 sheep. All but 9 run away. How many are left?",
        "reasoning": "This is a trick question. 'All but 9' means 9 remain.",
        "output": "9 sheep"
    },
    {
        "input": "If you have a bowl with 6 apples and take 2, how many do you have?",
        "reasoning": "The question asks how many YOU have, not how many are in the bowl. You took 2.",
        "output": "2 apples"
    }
]


def few_shot_cot(examples: list[dict], question: str) -> str:
    prompt = "Solve these problems, showing your reasoning:\n\n"

    for ex in examples:
        prompt += f"""Question: {ex['input']}
Reasoning: {ex['reasoning']}
Answer: {ex['output']}

"""

    prompt += f"""Question: {question}
Reasoning:"""

    return prompt
```

---

## Dynamic Example Selection

### Embedding-Based Selection

Choose examples most similar to the input:

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')


class DynamicFewShot:
    def __init__(self, examples: list[dict]):
        self.examples = examples
        # Pre-compute embeddings
        self.embeddings = model.encode([ex['input'] for ex in examples])

    def select_examples(self, query: str, k: int = 3) -> list[dict]:
        """Select k most similar examples to the query."""
        query_embedding = model.encode([query])[0]

        # Cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top k indices
        top_indices = np.argsort(similarities)[-k:][::-1]

        return [self.examples[i] for i in top_indices]

    def create_prompt(self, query: str, k: int = 3) -> str:
        """Create prompt with dynamically selected examples."""
        selected = self.select_examples(query, k)

        prompt = "Examples:\n\n"
        for ex in selected:
            prompt += f"Input: {ex['input']}\nOutput: {ex['output']}\n\n"

        prompt += f"Input: {query}\nOutput:"
        return prompt


# Usage
example_bank = [
    {"input": "Revenue increased 20%", "output": "Positive, Financial"},
    {"input": "CEO resigned unexpectedly", "output": "Negative, Leadership"},
    {"input": "New product launch planned", "output": "Neutral, Product"},
    # ... many more examples
]

few_shot = DynamicFewShot(example_bank)
prompt = few_shot.create_prompt("Quarterly profits exceeded estimates")
```

### Category-Based Selection

Ensure examples cover different categories:

```python
def select_diverse_examples(
    examples: list[dict],
    query: str,
    k: int = 4
) -> list[dict]:
    """Select examples ensuring category diversity."""
    # Group by category
    by_category = {}
    for ex in examples:
        cat = ex.get('category', 'default')
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(ex)

    # Take from each category round-robin
    selected = []
    categories = list(by_category.keys())
    i = 0
    while len(selected) < k and any(by_category.values()):
        cat = categories[i % len(categories)]
        if by_category[cat]:
            selected.append(by_category[cat].pop(0))
        i += 1

    return selected
```

---

## Format Examples

### JSON Output

```python
json_examples = """Examples of entity extraction:

Input: "Microsoft acquired GitHub for $7.5 billion"
Output: {
    "entities": [
        {"name": "Microsoft", "type": "company"},
        {"name": "GitHub", "type": "company"},
        {"name": "$7.5 billion", "type": "money"}
    ],
    "relation": "acquisition"
}

Input: "Satya Nadella became CEO in 2014"
Output: {
    "entities": [
        {"name": "Satya Nadella", "type": "person"},
        {"name": "CEO", "type": "title"},
        {"name": "2014", "type": "date"}
    ],
    "relation": "appointment"
}

Input: "{user_input}"
Output:"""
```

### Structured Text

```python
structured_examples = """Convert meeting notes to action items:

Notes: "John will send the report by Friday. Sarah needs to review the budget."
Action Items:
- [ ] John: Send the report (Due: Friday)
- [ ] Sarah: Review the budget (Due: Not specified)

Notes: "Team agreed to postpone the launch. Mike volunteered to notify clients."
Action Items:
- [ ] Team: Postpone the launch (Due: Not specified)
- [ ] Mike: Notify clients (Due: Not specified)

Notes: "{meeting_notes}"
Action Items:"""
```

### Code Generation

```python
code_examples = """Generate Python functions from descriptions:

Description: Check if a number is prime
```python
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
```

Description: Reverse a string without using slicing
```python
def reverse_string(s: str) -> str:
    result = ""
    for char in s:
        result = char + result
    return result
```

Description: {function_description}
```python"""
```

---

## Anti-Patterns

### 1. Inconsistent Formatting

```python
# Bad: Format varies between examples
examples = [
    ("good movie", "POSITIVE"),
    ("bad movie", "negative"),
    ("okay movie", "Neutral sentiment"),
]

# Good: Consistent format
examples = [
    ("good movie", "Positive"),
    ("bad movie", "Negative"),
    ("okay movie", "Neutral"),
]
```

### 2. Examples Too Easy

```python
# Bad: Only obvious cases
examples = [
    ("I LOVE IT!!!", "Positive"),
    ("HATE THIS!!!", "Negative"),
]

# Good: Include subtle cases
examples = [
    ("I LOVE IT!!!", "Positive"),
    ("It's fine, I guess", "Neutral"),
    ("Could have been better", "Negative"),
]
```

### 3. Missing Critical Patterns

```python
# Bad: No example of handling invalid input
examples = [
    ("Calculate 2+2", "4"),
    ("Calculate 10/5", "2"),
]

# Good: Show how to handle edge cases
examples = [
    ("Calculate 2+2", "4"),
    ("Calculate 10/5", "2"),
    ("Calculate 10/0", "Error: Division by zero"),
    ("Calculate xyz", "Error: Invalid expression"),
]
```

---

## Optimal Example Counts

Research and practice suggest:

| Task Complexity | Recommended Examples |
|----------------|---------------------|
| Simple classification | 2-3 |
| Moderate extraction | 4-6 |
| Complex reasoning | 5-8 |
| Highly nuanced | 8-15 |

**Guidelines:**
- Start with 3 examples, add more if quality is inconsistent
- Watch for diminishing returns (usually after 8-10)
- Balance example count against token costs
- Consider dynamic selection for large example banks

---

## Combining with Other Techniques

### Few-Shot + System Prompt

```python
system = """You are a customer support classifier.
Always respond with exactly one category.
When uncertain, choose the closest match."""

few_shot = """
Message: "Can't log in to my account"
Category: Account Access

Message: "When will my order arrive?"
Category: Order Status

Message: "How do I get a refund?"
Category: Billing

Message: "{customer_message}"
Category:"""
```

### Few-Shot + Chain-of-Thought

```python
few_shot_cot = """
Question: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?
Thought: Start with 23 apples. Use 20, leaving 23-20=3. Buy 6 more: 3+6=9.
Answer: 9

Question: {word_problem}
Thought:"""
```

---

## Testing Few-Shot Prompts

```python
def evaluate_few_shot(
    prompt_template: str,
    test_cases: list[dict],
    examples: list[tuple]
) -> dict:
    """Evaluate a few-shot prompt against test cases."""

    results = []

    for case in test_cases:
        prompt = prompt_template.format(
            examples=format_examples(examples),
            input=case["input"]
        )

        response = call_llm(prompt)

        # Check exact match or contains expected
        exact_match = response.strip() == case["expected"]
        contains = case["expected"].lower() in response.lower()

        results.append({
            "input": case["input"],
            "expected": case["expected"],
            "actual": response,
            "exact_match": exact_match,
            "contains": contains
        })

    return {
        "exact_match_rate": sum(r["exact_match"] for r in results) / len(results),
        "contains_rate": sum(r["contains"] for r in results) / len(results),
        "details": results
    }
```

---

*Few-shot learning is about communication through demonstration. Choose your examples as carefully as you would write production code—they directly shape your agent's behavior.*
