# The ReAct Pattern: Reasoning + Acting

> **Reading time:** ~12 min | **Module:** 4 — Agentic Patterns | **Prerequisites:** Module 2 — Tool Use, Module 1 — Chain-of-Thought

ReAct (Reasoning and Acting) interleaves thinking with action-taking. Instead of generating a complete plan upfront, ReAct agents alternate between reasoning about what to do next and executing actions, adapting based on observations.

<div class="callout-insight">

**Insight:** Think before you act, observe after you act. ReAct makes the agent's decision process explicit through "Thought" traces, enabling debugging, transparency, and better reasoning through intermediate steps.

</div>

---

## The ReAct Loop

### Pattern Structure

```
User Input
    ↓
Thought: [What do I need to do? What information do I need?]
    ↓
Action: [Tool call with specific parameters]
    ↓
Observation: [Result from the tool]
    ↓
Thought: [What did I learn? What should I do next?]
    ↓
... (repeat until complete)
    ↓
Thought: [I have enough information to answer]
    ↓
Final Answer: [Response to user]
```

### Basic Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>

```python
import anthropic
import json
import re

client = anthropic.Anthropic()

REACT_SYSTEM_PROMPT = """You are a helpful assistant that thinks step by step.

When given a task, you should:
1. Think about what you need to do
2. Take an action using available tools
3. Observe the result
4. Repeat until you can provide a final answer

Format your response as:
Thought: <your reasoning about what to do next>
Action: <tool_name>(<parameters as JSON>)

OR, if you have the final answer:
Thought: <your reasoning>
Final Answer: <your answer to the user>

Available tools:
- search(query): Search the web for information
- calculate(expression): Evaluate a mathematical expression
- get_weather(city): Get current weather for a city

Always think before acting. Always explain your reasoning."""


def parse_react_response(response: str) -> dict:
    """Parse a ReAct-formatted response."""

    # Check for final answer
    if "Final Answer:" in response:
        thought = response.split("Final Answer:")[0].replace("Thought:", "").strip()
        answer = response.split("Final Answer:")[1].strip()
        return {"type": "final", "thought": thought, "answer": answer}

    # Parse thought and action
    thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", response, re.DOTALL)
    action_match = re.search(r"Action:\s*(\w+)\((.+?)\)", response)

    if thought_match and action_match:
        return {
            "type": "action",
            "thought": thought_match.group(1).strip(),
            "tool": action_match.group(1),
            "params": json.loads(action_match.group(2))
        }

    return {"type": "error", "raw": response}


def execute_tool(tool_name: str, params: dict) -> str:
    """Execute a tool and return the observation."""

    if tool_name == "search":
        # Simulated search
        return f"Search results for '{params.get('query', '')}': [Relevant information would appear here]"

    elif tool_name == "calculate":
        try:
            result = eval(params.get("expression", "0"))
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {e}"

    elif tool_name == "get_weather":
        return f"Weather in {params.get('city', 'unknown')}: Sunny, 72°F"

    return f"Unknown tool: {tool_name}"


def run_react_agent(task: str, max_steps: int = 10) -> str:
    """Run a ReAct agent on a task."""

    messages = []
    history = f"Task: {task}\n\n"

    for step in range(max_steps):
        # Get next action from LLM
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            system=REACT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": history}]
        )

        response_text = response.content[0].text
        history += response_text + "\n"

        # Parse the response
        parsed = parse_react_response(response_text)

        if parsed["type"] == "final":
            return parsed["answer"]

        elif parsed["type"] == "action":
            # Execute the action
            observation = execute_tool(parsed["tool"], parsed["params"])
            history += f"Observation: {observation}\n\n"

        else:
            return f"Agent error: Could not parse response"

    return "Max steps reached without final answer"


# Usage
answer = run_react_agent("What is 15% of the temperature in New York right now?")
print(answer)
```

</div>
</div>

---

## ReAct with Claude's Tool Use

Leveraging native tool calling for cleaner implementation:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>

```python
import anthropic

client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "search",
        "description": "Search for information on the web",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }
    }
]


REACT_SYSTEM = """You are a helpful assistant that solves problems step by step.

Before taking any action, always explain your reasoning in your response.
Think through what information you need and why.

After receiving tool results, analyze what you learned before deciding next steps.
If you have enough information to answer, provide the final answer directly."""


class ReActAgent:
    """ReAct agent using Claude's native tool use."""

    def __init__(self, tools: list, max_steps: int = 10):
        self.client = anthropic.Anthropic()
        self.tools = tools
        self.max_steps = max_steps

    def execute_tool(self, name: str, params: dict) -> str:
        """Execute a tool by name."""
        if name == "search":
            return f"Search results for '{params['query']}': ..."
        elif name == "calculate":
            try:
                return str(eval(params["expression"]))
            except Exception as e:
                return f"Error: {e}"
        return f"Unknown tool: {name}"

    def run(self, task: str) -> str:
        """Run the agent on a task."""

        messages = [{"role": "user", "content": task}]

        for step in range(self.max_steps):
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                system=REACT_SYSTEM,
                tools=self.tools,
                messages=messages
            )

            # Check if we're done
            if response.stop_reason == "end_turn":
                return self._extract_text(response)

            # Process tool calls
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self.execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "user", "content": tool_results})

        return "Max steps reached"

    def _extract_text(self, response) -> str:
        return "".join(
            block.text for block in response.content
            if hasattr(block, "text")
        )


# Usage
agent = ReActAgent(TOOLS)
result = agent.run("Calculate 15% of 847.50 and tell me if it's more than 100")
```

</div>
</div>

---

## ReAct Trace Visualization

### Structured Logging


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>

```python
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class ReActStep:
    step_number: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[dict] = None
    observation: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ReActTrace:
    task: str
    steps: list[ReActStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    success: bool = False

    def add_step(self, step: ReActStep):
        self.steps.append(step)

    def to_markdown(self) -> str:
        """Convert trace to readable markdown."""
        md = f"# ReAct Trace\n\n**Task:** {self.task}\n\n"

        for step in self.steps:
            md += f"## Step {step.step_number}\n\n"
            md += f"**Thought:** {step.thought}\n\n"
            if step.action:
                md += f"**Action:** `{step.action}({step.action_input})`\n\n"
            if step.observation:
                md += f"**Observation:** {step.observation}\n\n"
            md += "---\n\n"

        if self.final_answer:
            md += f"## Final Answer\n\n{self.final_answer}\n"

        return md


class TracingReActAgent(ReActAgent):
    """ReAct agent with full execution tracing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_trace: Optional[ReActTrace] = None

    def run(self, task: str) -> tuple[str, ReActTrace]:
        """Run agent and return result with trace."""

        self.current_trace = ReActTrace(task=task)
        messages = [{"role": "user", "content": task}]
        step_num = 0

        for _ in range(self.max_steps):
            step_num += 1
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                system=REACT_SYSTEM,
                tools=self.tools,
                messages=messages
            )

            # Extract thought from text content
            thought = self._extract_text(response)

            if response.stop_reason == "end_turn":
                self.current_trace.final_answer = thought
                self.current_trace.success = True
                self.current_trace.add_step(ReActStep(
                    step_number=step_num,
                    thought=thought
                ))
                return thought, self.current_trace

            # Process tool calls
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []

            for block in response.content:
                if block.type == "tool_use":
                    result = self.execute_tool(block.name, block.input)

                    self.current_trace.add_step(ReActStep(
                        step_number=step_num,
                        thought=thought,
                        action=block.name,
                        action_input=block.input,
                        observation=result
                    ))

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "user", "content": tool_results})

        return "Max steps reached", self.current_trace
```

</div>
</div>

---

## Advanced ReAct Patterns

### ReAct with Memory


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>

```python
class ReActWithMemory(ReActAgent):
    """ReAct agent with working memory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = []

    def run(self, task: str) -> str:
        # Include relevant memories in context
        memory_context = ""
        if self.memory:
            memory_context = "Relevant past information:\n"
            memory_context += "\n".join(f"- {m}" for m in self.memory[-5:])
            memory_context += "\n\n"

        augmented_task = f"{memory_context}Current task: {task}"

        result = super().run(augmented_task)

        # Store task and result in memory
        self.memory.append(f"Task: {task} -> Result: {result[:100]}...")

        return result
```

</div>
</div>

### ReAct with Verification

```python
class VerifiedReActAgent(ReActAgent):
    """ReAct agent that verifies its answers."""

    def run(self, task: str) -> str:
        # First pass: solve the task
        initial_answer = super().run(task)

        # Verification pass
        verification_prompt = f"""I solved this task:
Task: {task}

My answer was: {initial_answer}

Please verify this answer:
1. Is the reasoning sound?
2. Did I use the tools correctly?
3. Is the answer complete?

If there are issues, provide the corrected answer.
If the answer is correct, confirm it."""

        verification = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": verification_prompt}]
        )

        verified_text = verification.content[0].text

        if "correct" in verified_text.lower() or "confirm" in verified_text.lower():
            return initial_answer
        else:
            # Extract corrected answer if provided
            return verified_text
```

---

## ReAct Prompting Strategies

### Explicit Reasoning Instructions


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>

```python
DETAILED_REACT_SYSTEM = """You are an AI assistant that solves problems systematically.

For each step:
1. THINK: State what you know, what you need to find out, and your strategy
2. ACT: Choose and use the most appropriate tool
3. OBSERVE: Analyze the result carefully
4. REFLECT: Did this help? What should you do next?

Rules:
- Never skip the thinking step
- If a tool fails, try a different approach
- If you're uncertain, acknowledge it
- Break complex problems into smaller steps
- Verify your final answer before stating it"""
```

</div>
</div>

### Few-Shot ReAct Examples

```python
REACT_EXAMPLES = """
Example 1:
Task: What is the population of the capital of Japan?

Thought: I need to find two things: 1) the capital of Japan, and 2) its population. I already know Tokyo is the capital, so I just need to search for Tokyo's population.
Action: search({"query": "Tokyo population 2024"})
Observation: Tokyo has a population of approximately 13.96 million in the city proper, and 37.4 million in the greater metropolitan area.

Thought: I found the population. The task asks about the capital's population, so I should provide the city proper figure.
Final Answer: The population of Tokyo, the capital of Japan, is approximately 13.96 million people (city proper).

Example 2:
Task: If I invest $1000 at 5% annual interest compounded monthly for 10 years, how much will I have?

Thought: This is a compound interest calculation. The formula is A = P(1 + r/n)^(nt), where P=1000, r=0.05, n=12, t=10. Let me calculate this.
Action: calculate({"expression": "1000 * (1 + 0.05/12) ** (12 * 10)"})
Observation: 1647.0094976902782

Thought: The calculation gives approximately $1647. I should round appropriately for currency.
Final Answer: After 10 years, your investment would grow to approximately $1,647.01.
"""
```

---

## Error Handling in ReAct


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>

```python
class RobustReActAgent(ReActAgent):
    """ReAct agent with error handling and recovery."""

    def run(self, task: str) -> str:
        messages = [{"role": "user", "content": task}]
        consecutive_errors = 0
        max_errors = 3

        for step in range(self.max_steps):
            try:
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2000,
                    system=REACT_SYSTEM,
                    tools=self.tools,
                    messages=messages
                )

                consecutive_errors = 0  # Reset on success

                if response.stop_reason == "end_turn":
                    return self._extract_text(response)

                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        try:
                            result = self.execute_tool(block.name, block.input)
                        except Exception as e:
                            result = f"Tool error: {e}. Try a different approach."

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })

                messages.append({"role": "user", "content": tool_results})

            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= max_errors:
                    return f"Failed after {max_errors} consecutive errors: {e}"
                continue

        return "Max steps reached"
```

</div>
</div>

<div class="callout-key">

**Key Concept Summary:** This guide covered the core concepts. Review the companion slides for visual summaries and the hands-on notebook for practice implementations.

</div>

---

*ReAct is the fundamental reasoning pattern for agentic AI. Master it, and you'll have the foundation for building agents that can tackle any complex task.*


## Practice Questions

1. Explain in your own words how the concepts in this guide relate to building production agents.
2. What are the key tradeoffs you need to consider when applying these techniques?
3. Describe a scenario where the approach from this guide would be the wrong choice, and what you would use instead.

---

**Next Steps:**

<a class="link-card" href="./01_react_pattern_slides.md">
  <div class="link-card-title">The ReAct Pattern: Reasoning + Acting — Companion Slides</div>
  <div class="link-card-description">Visual slide deck with diagrams, speaker notes, and key takeaways.</div>
</a>

<a class="link-card" href="../notebooks/01_react_agents.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with working code and guided exercises.</div>
</a>
