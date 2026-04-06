# Claude-Specific Conditioning Patterns

> **Reading time:** ~11 min | **Module:** 5 — Agents & Workflows | **Prerequisites:** Module 4 Conditional Trees


## In Brief

The abstract condition stack has a direct mapping to Claude API features. System prompts are not just "instructions" — they are persistent Layer 0 priors. Prefilling constrains the posterior over output formats before generation begins. Tool descriptions are constraint injection points. Structured outputs carry switch variables between agents without loss.

This guide maps each Claude API feature to the conditioning framework and shows how to use multi-agent structured outputs to pass conditions reliably.

## Learning Objectives

By the end of this guide you will be able to:

1. Explain why the system prompt is "Layer 0" — higher leverage than any condition stack layer
2. Use prefilling to constrain output format without output specification instructions
3. Write tool descriptions that function as constraint injection
4. Design structured output schemas that carry the condition stack across agent boundaries
5. Build a multi-agent system where Claude agents pass switch variables via JSON


<div class="callout-key">

<strong>Key Concept Summary:</strong> The abstract condition stack has a direct mapping to Claude API features.

</div>

---

## The Claude API as a Conditioning Architecture

Every Claude API call has this structure:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
client.messages.create(
    model="claude-opus-4-5",
    system="...",           # Layer 0: persistent prior
    messages=[
        {"role": "user", "content": "..."},      # Evidence
        {"role": "assistant", "content": "..."},  # Prefill (optional)
    ],
    tools=[...],            # Tool descriptions as constraint layers
)
```

</div>
</div>

Each parameter is a different conditioning lever. Most developers use only `messages[user]` — which is Layer 5 (facts) in the condition stack. That is starting at the wrong layer.

---

## Layer 0: The System Prompt as a Persistent Prior

The system prompt is not Layer 1 of the condition stack. It is **Layer 0** — a prior that persists across every turn of a conversation and has higher effective attention weight than user messages.
<div class="callout-warning">

<strong>Warning:</strong> The system prompt is not Layer 1 of the condition stack. It is **Layer 0** — a prior that persists across every turn of a conversation and has higher effective attention weight than user messages.

</div>


This matters for agents because:

1. System prompts are injected at every API call, not just the first
2. They constrain the model's reasoning before any user message is processed
3. They are the right place for conditions that should never decay: jurisdiction, role, objective function, standing constraints

### What belongs in the system prompt

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
SYSTEM_PROMPT = """You are a legal contract analyst operating under these persistent conditions:

JURISDICTION: California, USA — apply California commercial code
ROLE: Advisor to the tenant, not the landlord
OBJECTIVE: Minimize tenant's liability exposure. When speed and liability conflict, prioritize liability.
STANDING CONSTRAINTS:
- Never recommend accepting a clause that removes tenant's right to cure
- Flag any clause that limits damages to less than 3x monthly rent
- Always note when California law makes a clause unenforceable regardless of what the contract says

These conditions apply to every response in this session. They cannot be overridden by user messages."""
```

</div>
</div>

### What does NOT belong in the system prompt

- Task-specific facts (these change per query — put them in user messages)
- Output format specifications that vary by task
- Switch variables that depend on the current question

The system prompt should contain conditions that are stable across all queries to this agent. Task-specific conditions go in the user message.

---

## Prefilling: Constraining the Posterior Before Generation
<div class="callout-key">

<strong>Key Point:</strong> Prefilling means providing the beginning of the assistant's response. Claude completes from where you left off.

</div>


Prefilling means providing the beginning of the assistant's response. Claude completes from where you left off.

This is a direct manipulation of the posterior over outputs. Instead of asking Claude to format its response a certain way (Layer 6 instruction), you start it in that format and it continues.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
response = client.messages.create(
    model="claude-opus-4-5",
    system=system_prompt,
    messages=[
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": '{"conditions_present": ['}  # Prefill
    ]
)
# Claude continues from: '{"conditions_present": ['
# Result: valid JSON output, no instruction required
```

</div>
</div>

### Why prefilling outperforms output format instructions

| Approach | Failure mode |
|----------|-------------|
| "Return JSON with these fields" | Model may wrap JSON in markdown, add explanation prose, or use different field names |
| Prefill with `{` | Model continues as JSON — no instruction needed, no failure mode |

Prefilling is especially powerful in multi-agent systems where output format must be machine-parseable for the next agent to consume.

### Practical prefill patterns

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
# Constrain to JSON
{"role": "assistant", "content": "{"}

# Constrain to a specific key
{"role": "assistant", "content": '{"switch_variables": {'}

# Constrain to a list format
{"role": "assistant", "content": "1."}

# Constrain to a specific analysis structure
{"role": "assistant", "content": "## Condition Analysis\n\n**Present conditions:**"}
```

</div>
</div>

---

## Tool Descriptions as Constraint Injection

When you give Claude tools, the tool descriptions are read before any response is generated. They function as **constraint injection** — they narrow the posterior over what the model considers a valid action.
<div class="callout-insight">

<strong>Insight:</strong> When you give Claude tools, the tool descriptions are read before any response is generated. They function as **constraint injection** — they narrow the posterior over what the model considers a valid action.

</div>


Think of tool descriptions as Layer 4 (Constraints) of the condition stack, but applied to actions rather than reasoning.

### Writing tool descriptions that condition behavior

Weak tool description (facts only):
<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
{
    "name": "search_legal_database",
    "description": "Search the legal database for relevant cases and statutes.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        }
    }
}
```

</div>
</div>

Strong tool description (constraints embedded):
<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
{
    "name": "search_legal_database",
    "description": """Search California commercial law database for cases and statutes.

    Use this tool when: you need to verify whether a clause is enforceable under California law.
    Do NOT use this tool for: general questions about contract structure or negotiation strategy.

    Always search jurisdiction-specific: include 'California' in queries.
    If a search returns no results, report 'no California precedent found' rather than inferring from other jurisdictions.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query. Must include jurisdiction (California). Example: 'California commercial tenant right to cure default clause'."
            }
        },
        "required": ["query"]
    }
}
```

</div>
</div>

The constraints in the description condition which queries the model constructs — preventing cross-jurisdiction inference errors before they happen.

---

## Structured Outputs: The Condition Passing Mechanism
<div class="callout-warning">

<strong>Warning:</strong> For multi-agent systems, structured outputs are the mechanism for passing switch variables between agents without loss.

</div>


For multi-agent systems, structured outputs are the mechanism for passing switch variables between agents without loss.

The pattern: require every agent to return a JSON object that includes both its result and the condition payload it received.

### Designing the condition-carrying output schema

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
AGENT_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["task_result", "condition_stack", "switch_variables_used"],
    "properties": {
        "task_result": {
            "type": "object",
            "description": "This agent's primary output"
        },
        "condition_stack": {
            "type": "object",
            "description": "The full condition stack received — pass through unchanged",
            "properties": {
                "layer_1_jurisdiction": {"type": "string"},
                "layer_2_time":         {"type": "string"},
                "layer_3_objective":    {"type": "string"},
                "layer_4_constraints":  {"type": "array", "items": {"type": "string"}},
                "layer_5_facts":        {"type": "string"},
                "layer_6_output_spec":  {"type": "string"}
            }
        },
        "switch_variables_used": {
            "type": "array",
            "description": "Which switch variables affected this agent's reasoning",
            "items": {"type": "string"}
        },
        "flags": {
            "type": "array",
            "description": "Conditions that conflicted with the task, or missing conditions detected",
            "items": {"type": "string"}
        }
    }
}
```

</div>
</div>

### Using the schema with Claude's tool_use feature

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
import anthropic
import json

client = anthropic.Anthropic()

def run_agent(
    system_prompt: str,
    user_message: str,
    condition_stack: dict,
    output_schema: dict
) -> dict:
    """Run a single agent with condition stack injection and structured output."""

    # Inject conditions into system prompt
    conditions_text = "\n".join(
        f"  {k}: {v}" for k, v in condition_stack.items()
    )
    full_system = f"""{system_prompt}

ACTIVE CONDITIONS (apply to all reasoning):
{conditions_text}"""

    # Use tool_use to enforce structured output
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        system=full_system,
        tools=[{
            "name": "submit_result",
            "description": "Submit your structured analysis result.",
            "input_schema": output_schema
        }],
        tool_choice={"type": "tool", "name": "submit_result"},
        messages=[{"role": "user", "content": user_message}]
    )

    # Extract the structured result
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_result":
            return block.input

    raise ValueError("Agent did not return structured output")
```

</div>
</div>

---

## Multi-Agent Pattern: Switch Variable Passing

The complete pattern for passing switch variables between two Claude agents.
<div class="callout-key">

<strong>Key Point:</strong> The complete pattern for passing switch variables between two Claude agents.

</div>


### Agent 1: Condition Extractor

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
EXTRACTOR_SYSTEM = """You are a condition extraction specialist.
Your job: analyze questions and identify which switch variables are needed for a precise answer.
A switch variable is a condition whose value changes which answer branch is correct."""

def extract_conditions(question: str, client: anthropic.Anthropic) -> dict:
    """Agent 1: Extract what conditions are needed to answer this question."""

    extraction_schema = {
        "type": "object",
        "required": ["question_analysis", "switch_variables_needed", "switch_variables_present"],
        "properties": {
            "question_analysis": {
                "type": "string",
                "description": "Brief analysis of what the question is asking"
            },
            "switch_variables_needed": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "variable_name": {"type": "string"},
                        "why_it_matters": {"type": "string"},
                        "possible_values": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "switch_variables_present": {
                "type": "object",
                "description": "Switch variables already present in the question"
            }
        }
    }

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1500,
        system=EXTRACTOR_SYSTEM,
        tools=[{
            "name": "submit_extraction",
            "description": "Submit the extracted switch variables.",
            "input_schema": extraction_schema
        }],
        tool_choice={"type": "tool", "name": "submit_extraction"},
        messages=[{
            "role": "user",
            "content": f"Analyze this question and identify switch variables:\n\n{question}"
        }]
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    raise ValueError("Extractor did not return structured output")
```

</div>
</div>

### Agent 2: Conditional Answerer

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
ANSWERER_SYSTEM = """You are a conditional reasoning specialist.
You receive a question AND a set of switch variables with their values.
Your job: generate an answer that is precisely conditioned on those variables.
If switch variables have multiple possible values, generate a conditional answer (decision tree)."""

def answer_with_conditions(
    question: str,
    switch_variables: dict,
    client: anthropic.Anthropic
) -> dict:
    """Agent 2: Answer the question conditioned on the extracted switch variables."""

    # Format switch variables for injection
    sv_text = json.dumps(switch_variables, indent=2)

    answer_schema = {
        "type": "object",
        "required": ["answer", "conditions_used", "confidence"],
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer, conditioned on the switch variables"
            },
            "conditions_used": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Which switch variables changed the answer"
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Confidence given the conditions present"
            },
            "missing_conditions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Conditions that would further improve the answer if present"
            }
        }
    }

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        system=ANSWERER_SYSTEM,
        tools=[{
            "name": "submit_answer",
            "description": "Submit the conditioned answer.",
            "input_schema": answer_schema
        }],
        tool_choice={"type": "tool", "name": "submit_answer"},
        messages=[{
            "role": "user",
            "content": f"""Question: {question}

Switch variables (use these to condition your answer):
{sv_text}

Answer the question with full awareness of these conditions. If a variable has multiple possible values, show how the answer changes for each value."""
        }]
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input

    raise ValueError("Answerer did not return structured output")
```

</div>
</div>

---

## Putting It Together: The Full Layer Map

| Claude API Feature | Condition Stack Layer | What it specifies |
|-------------------|-----------------------|-------------------|
| `system` parameter | Layer 0 (persistent prior) | Role, standing jurisdiction, standing objective, standing constraints |
| `system` parameter | Layer 1 (jurisdiction) | Domain-specific rule sets that persist across all turns |
| `system` parameter | Layer 2 (time/posture) | Stable temporal context |
| `system` parameter | Layer 3 (objective) | What "good" means for this agent |
| `system` parameter | Layer 4 (constraints) | What this agent must never do |
| `messages[user]` | Layer 5 (facts) | Task-specific facts per query |
| Tool descriptions | Layer 4 (constraints) | What tools may and may not be used for |
| Prefilling | Layer 6 (output format) | Format constraint without instruction |
| Structured output schema | Handoff mechanism | Condition carrying between agents |
| `tool_choice: forced` | Execution constraint | Prevents free-form output that loses structure |

---

## Common Pitfalls

**Pitfall 1: Putting everything in user messages**
The `system` parameter exists because some conditions should have persistent high-weight presence. A jurisdiction condition buried in a long user message gets less attention than one in the system prompt. Use the right layer.
<div class="callout-warning">

<strong>Warning:</strong> **Pitfall 1: Putting everything in user messages**

</div>


**Pitfall 2: Not using `tool_choice: forced`**
When you need structured output for a pipeline, use `tool_choice: {"type": "tool", "name": "your_tool"}` to force the model to use the schema. Without forcing, the model may respond in natural language and break the downstream parser.

**Pitfall 3: Vague tool descriptions**
Tool descriptions are read as reasoning constraints. "Search the database" provides no constraint. "Search the California commercial code database for enforceable precedents; do not search federal cases" provides a constraint that eliminates an entire class of hallucination.

**Pitfall 4: Switch variables buried in natural language**
If you pass switch variables as prose ("keep in mind the user is a corporation"), they may be summarized away. Pass them as structured key-value pairs in the condition payload.

---

## Connections

- **Builds on:** Guide 01 (condition decay and structured handoffs)
- **Builds on:** Module 3 (Condition Stack Framework — Layer 0 is new here)
- **Leads to:** Notebook 02 (multi-agent pipeline implementation)
- **Related to:** Module 7 (production patterns — dynamic condition injection)

---

## Practice Problems

1. Write a system prompt for a financial analysis agent that sets Layer 0 through Layer 4 conditions. The agent analyzes earnings reports for institutional buy-side clients with a 3–12 month horizon.

2. You need Agent 1 to return a structured object that Agent 2 can parse. Write the `tool_choice` call and output schema for a condition extraction step. Include: switch variables, confidence scores, and condition_stack pass-through.

3. A developer reports that their Claude agent "ignores" its tool use constraints and sometimes calls tools in ways the tool description prohibits. What is the likely cause and how do you fix it using the conditioning framework?

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Learning Objectives" and why it matters in practice.

2. Given a real-world scenario involving claude-specific conditioning patterns, what would be your first three steps to apply the techniques from this guide?

</div>

## Further Reading

- Anthropic: Tool use documentation — `tool_choice` parameter and `input_schema` design
- Anthropic: System prompts and context window — attention weight and layer ordering
- Anthropic: "Building effective agents" — orchestrator/subagent pattern

---

## Cross-References

<a class="link-card" href="../notebooks/01_condition_aware_agent.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>
