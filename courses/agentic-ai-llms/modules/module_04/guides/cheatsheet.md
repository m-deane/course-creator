# Planning & Reasoning Cheatsheet

> **Reading time:** ~5 min | **Module:** 4 — Agentic Patterns | **Prerequisites:** Module 4 guides

## Key Concepts

| Concept | Definition |
|---------|-----------|
| **ReAct** | Pattern that interleaves reasoning (thinking) with acting (tool use) in a loop |
| **Thought** | Internal reasoning step where the agent decides what to do next |
| **Action** | Execution of a tool or function to gather information or affect the environment |
| **Observation** | Result returned from an action that informs the next thought |
| **Goal Decomposition** | Breaking a complex objective into smaller, manageable subtasks |
| **Self-Reflection** | Process where an agent critiques its own outputs and revises its approach |
| **Planning Horizon** | How far ahead an agent plans before taking action |
| **Replanning** | Adjusting the plan based on new observations or failed attempts |

## Common Patterns

### Basic ReAct Loop


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>

```python
def react_loop(query, tools, max_steps=10):
    history = []

    for step in range(max_steps):
        # THOUGHT: Decide what to do
        thought = llm.generate(
            f"Question: {query}\nHistory: {history}\n"
            f"Thought: What should I do next?"
        )

        # Check if done
        if "Final Answer:" in thought:
            return extract_answer(thought)

        # ACTION: Execute tool
        action, action_input = parse_action(thought)
        observation = tools[action](action_input)

        # OBSERVATION: Record result
        history.append({
            "thought": thought,
            "action": action,
            "input": action_input,
            "observation": observation
        })

    return "Max steps reached"
```

</div>
</div>

### Goal Decomposition

```python
def decompose_goal(goal):
    prompt = f"""
    Break down this goal into 3-5 concrete subtasks:
    Goal: {goal}

    Return as JSON list: [{{"subtask": "...", "dependencies": []}}]
    """

    subtasks = llm.generate(prompt)
    return json.loads(subtasks)

def execute_plan(subtasks, tools):
    completed = {}

    for task in subtasks:
        # Wait for dependencies
        if not all(dep in completed for dep in task["dependencies"]):
            continue

        result = react_loop(task["subtask"], tools)
        completed[task["subtask"]] = result

    return completed
```

### Self-Reflection

```python
def reflect_and_retry(task, initial_result, max_retries=3):
    result = initial_result

    for attempt in range(max_retries):
        # Critique the result
        critique = llm.generate(
            f"Task: {task}\nResult: {result}\n"
            f"Is this result correct and complete? "
            f"If not, what's wrong and how to fix it?"
        )

        if "correct" in critique.lower():
            return result

        # Revise based on critique
        result = llm.generate(
            f"Task: {task}\nPrevious result: {result}\n"
            f"Issues: {critique}\nRevised result:"
        )

    return result
```

### Adaptive Planning

```python
def adaptive_plan_execute(goal, tools):
    plan = create_initial_plan(goal)

    while not is_goal_achieved(goal):
        # Execute next step
        next_step = plan.pop(0)
        result = execute_step(next_step, tools)

        # Check if replanning needed
        if is_unexpected_result(result):
            plan = replan(goal, result, remaining_steps=plan)

        # Update state
        update_world_state(result)

    return get_final_result()
```

## Gotchas

### Problem: Infinite loops in ReAct
**Symptom:** Agent repeats the same action without progress
**Solution:**
- Add step limits and forced termination
- Track observation history to detect duplicates
- Implement early stopping when no new information gained


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">agent.py</span>

```python

# Bad: No loop detection
while not done:
    action = decide_action()
    observation = execute(action)

# Good: Track and prevent loops
seen_observations = set()
while not done and steps < MAX_STEPS:
    action = decide_action()
    observation = execute(action)

    if observation in seen_observations:
        break  # Stuck in loop
    seen_observations.add(observation)
```

</div>
</div>

### Problem: Overly complex plans fail
**Symptom:** Agent creates detailed 10+ step plans that break on first unexpected result
**Solution:**
- Use short planning horizons (2-3 steps)
- Replan frequently based on observations
- Prefer adaptive over complete upfront planning

### Problem: Self-reflection adds latency without value
**Symptom:** Agent critiques every output, even trivial ones
**Solution:**
- Only reflect on complex tasks or after failures
- Use cheaper models for reflection
- Cache reflection patterns for common errors

### Problem: Goal decomposition too granular or too coarse
**Symptom:** Subtasks are either trivial one-liners or still complex multi-step tasks
**Solution:**
- Aim for 3-7 subtasks per goal
- Each subtask should be achievable in 1-3 ReAct iterations
- Test with example: can you describe the subtask in one sentence?

### Problem: Prompt injection through observations
**Symptom:** Tool outputs contain instructions that hijack agent reasoning
**Solution:**
- Sanitize all observations before adding to prompt
- Use XML tags to clearly separate observations from instructions
- Validate observations match expected format

```python

# Bad: Raw observation in prompt
prompt = f"Observation: {tool_output}\nThought:"

# Good: Structured and sanitized
sanitized = remove_instructions(tool_output)
prompt = f"<observation>{sanitized}</observation>\n<thought>"
```

## Quick Decision Guide

**When to use ReAct?**
- Multi-step tasks requiring information gathering
- Unknown number of steps needed upfront
- Need to adapt based on intermediate results

**When to use Plan-and-Execute?**
- Well-defined tasks with predictable structure
- Need to validate plan before execution
- Parallel subtask execution possible

**When to use Self-Reflection?**
- Task requires high accuracy
- Cost of errors is high
- Agent makes systematic mistakes

**When to NOT use complex planning?**
- Simple single-step tasks
- Real-time applications (latency sensitive)
- Tasks where humans will review outputs anyway
