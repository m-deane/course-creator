# Self-Reflection: Agents That Learn from Mistakes

> **Reading time:** ~10 min | **Module:** 4 — Agentic Patterns | **Prerequisites:** Module 4 — ReAct Pattern

Self-reflection enables agents to critique their own outputs, identify errors, and improve. Instead of accepting first attempts, reflective agents evaluate their work and iterate toward better solutions.

<div class="callout-insight">

**Insight:** The first answer is rarely the best. Reflection adds a meta-cognitive layer—the agent asks "Is this good? What could be better?" This catches errors, improves reasoning, and produces more reliable outputs.

</div>

---

## Reflection Patterns

### Basic Critique-Revise Loop


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
def reflect_and_revise(task: str, max_iterations: int = 3) -> str:
    """Generate, critique, and revise until satisfactory."""

    # Initial generation
    response = generate(task)

    for i in range(max_iterations):
        # Critique the response
        critique = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"""Critique this response. Identify:
1. Errors or inaccuracies
2. Missing information
3. Areas that could be clearer
4. Overall quality (1-10)

Task: {task}
Response: {response}

If quality >= 8 and no major issues, say "APPROVED".
Otherwise, list specific improvements needed."""
            }]
        )

        critique_text = critique.content[0].text

        if "APPROVED" in critique_text:
            return response

        # Revise based on critique
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"""Revise this response based on the critique.

Original task: {task}
Original response: {response}
Critique: {critique_text}

Provide an improved response:"""
            }]
        ).content[0].text

    return response
```

</div>
</div>

### Reflexion Pattern

Learn from trajectory of attempts:

```python
class ReflexionAgent:
    """Agent that learns from failed attempts."""

    def __init__(self):
        self.memory = []  # Past attempts and reflections

    def run(self, task: str, max_attempts: int = 3) -> str:
        for attempt in range(max_attempts):
            # Include past reflections in context
            context = self._build_context()

            # Generate response
            response = self._generate(task, context)

            # Evaluate success
            evaluation = self._evaluate(task, response)

            if evaluation["success"]:
                return response

            # Reflect on failure
            reflection = self._reflect(task, response, evaluation)
            self.memory.append({
                "attempt": attempt,
                "response": response,
                "evaluation": evaluation,
                "reflection": reflection
            })

        return response  # Return best attempt

    def _reflect(self, task: str, response: str, evaluation: dict) -> str:
        """Generate reflection on why the attempt failed."""

        prompt = f"""The following attempt failed. Reflect on why and how to improve.

Task: {task}
Attempt: {response}
Evaluation: {evaluation}

Reflection (be specific about what went wrong and how to fix it):"""

        return client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        ).content[0].text

    def _build_context(self) -> str:
        """Build context from past attempts."""
        if not self.memory:
            return ""

        context = "Previous attempts and learnings:\n"
        for m in self.memory[-3:]:  # Last 3 attempts
            context += f"\nAttempt {m['attempt']}:\n"
            context += f"Response: {m['response'][:200]}...\n"
            context += f"Reflection: {m['reflection']}\n"

        return context
```

---

## Evaluation Strategies

### Self-Consistency Check


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
def check_consistency(task: str, response: str, n_checks: int = 3) -> dict:
    """Verify response consistency across multiple generations."""

    alternatives = []
    for _ in range(n_checks):
        alt = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            temperature=0.7,
            messages=[{"role": "user", "content": task}]
        ).content[0].text
        alternatives.append(alt)

    # Check if responses agree
    agreement_prompt = f"""Compare these responses and determine if they agree.

Response 1: {response}
Response 2: {alternatives[0]}
Response 3: {alternatives[1]}

Do they reach the same conclusion? (YES/NO)
If NO, what are the disagreements?"""

    check = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=200,
        messages=[{"role": "user", "content": agreement_prompt}]
    ).content[0].text

    return {
        "consistent": "YES" in check.upper(),
        "analysis": check,
        "alternatives": alternatives
    }
```

</div>
</div>

### Factual Verification

```python
def verify_facts(response: str, tools: list) -> dict:
    """Extract and verify factual claims in response."""

    # Extract claims
    extraction = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""Extract all factual claims from this response.
Return as a JSON list of claims.

Response: {response}

Claims:"""
        }]
    ).content[0].text

    claims = json.loads(extraction)

    # Verify each claim
    verified = []
    for claim in claims:
        result = search_tool(claim)  # Use search to verify

        verification = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": f"Does this evidence support the claim?\nClaim: {claim}\nEvidence: {result}\nAnswer YES or NO with brief reason."
            }]
        ).content[0].text

        verified.append({
            "claim": claim,
            "evidence": result,
            "verified": "YES" in verification.upper()
        })

    return {
        "claims": verified,
        "all_verified": all(v["verified"] for v in verified)
    }
```

---

## Reflection Prompts

### Quality Critique


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>
</div>

```python
QUALITY_CRITIQUE = """Evaluate this response on these dimensions:

1. ACCURACY (1-5): Are all facts correct?
2. COMPLETENESS (1-5): Does it fully address the question?
3. CLARITY (1-5): Is it easy to understand?
4. RELEVANCE (1-5): Does it stay on topic?
5. REASONING (1-5): Is the logic sound?

Response to evaluate:
{response}

Provide scores and specific issues for any dimension below 4."""
```

</div>
</div>

### Error Detection

```python
ERROR_DETECTION = """Analyze this response for errors:

Response: {response}

Check for:
1. Logical contradictions
2. Factual inaccuracies (if detectable)
3. Unsupported claims
4. Missing important considerations
5. Ambiguous or unclear statements

List each error found with its location and severity (HIGH/MEDIUM/LOW)."""
```

### Improvement Suggestions

```python
IMPROVEMENT = """How could this response be improved?

Original task: {task}
Current response: {response}

Suggest specific improvements:
1. What should be added?
2. What should be removed?
3. What should be rephrased?
4. What structure changes would help?

Be concrete and actionable."""
```

---

## Multi-Agent Reflection

### Critic Agent


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>

```python
class CriticAgent:
    """Specialized agent for critiquing responses."""

    SYSTEM = """You are a critical reviewer. Your job is to find flaws,
    errors, and areas for improvement. Be thorough but fair.
    Do not suggest improvements unless there are genuine issues."""

    def critique(self, task: str, response: str) -> dict:
        result = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            system=self.SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Task: {task}\n\nResponse to critique:\n{response}"
            }]
        )

        return self._parse_critique(result.content[0].text)

    def _parse_critique(self, text: str) -> dict:
        # Parse into structured feedback
        return {
            "raw": text,
            "has_issues": "no issues" not in text.lower(),
            "severity": self._assess_severity(text)
        }
```


### Debate-Based Reflection

```python
def reflect_through_debate(task: str, response: str) -> str:
    """Use adversarial debate for reflection."""

    # Advocate argues response is correct
    advocate = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"Argue why this response is correct and complete.\nTask: {task}\nResponse: {response}"
        }]
    ).content[0].text

    # Critic argues response has problems
    critic = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"Argue why this response is flawed or incomplete.\nTask: {task}\nResponse: {response}"
        }]
    ).content[0].text

    # Judge synthesizes
    judgment = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"""Given this debate, provide the best answer.

Task: {task}
Original: {response}
For: {advocate}
Against: {critic}

Synthesize the best response considering both perspectives:"""
        }]
    ).content[0].text

    return judgment
```

---

## Best Practices

1. **Set clear criteria**: Define what "good" means before reflecting
2. **Limit iterations**: 2-3 reflection rounds usually suffice
3. **Use different perspectives**: Vary the critique angle each round
4. **Stop when satisfied**: Don't over-refine good responses
5. **Log reflections**: Track what errors are common for improvement

<div class="callout-key">

**Key Concept Summary:** This guide covered the core concepts. Review the companion slides for visual summaries and the hands-on notebook for practice implementations.


---

*Self-reflection transforms agents from single-shot responders to iterative improvers. The best agents question their own outputs.*




## Practice Questions

1. Explain in your own words how the concepts in this guide relate to building production agents.
2. What are the key tradeoffs you need to consider when applying these techniques?
3. Describe a scenario where the approach from this guide would be the wrong choice, and what you would use instead.

---

**Next Steps:**

<a class="link-card" href="./03_self_reflection_slides.md">
  <div class="link-card-title">Self-Reflection and Self-Correction — Companion Slides</div>
  <div class="link-card-description">Visual slide deck with diagrams, speaker notes, and key takeaways.</div>
</a>

<a class="link-card" href="../notebooks/01_react_agents.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with working code and guided exercises.</div>
</a>
