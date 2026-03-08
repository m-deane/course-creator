---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Trajectories
## Recording Agent Episodes for GRPO Training

Module 02 — Reinforcement Learning for AI Agents

<!-- Speaker notes: This guide is the pivot point of the module. Architecture (Guide 01) and installation (Guide 02) were conceptual and procedural. Here we write the actual data structures that make training happen. Trajectories are the interface between your agent code and the GRPO trainer. Understanding their structure precisely—especially the difference between dict messages and Choice objects—prevents the most common ART debugging mistakes. -->

---

## What Is a Trajectory?

A Trajectory is the **complete record of one agent episode**:

```
Trajectory
├── messages_and_choices  ← full conversation history
│   ├── dict: system message
│   ├── dict: user message
│   ├── Choice: assistant response (with log probs)
│   ├── dict: tool result
│   └── Choice: assistant final response
└── reward: float  ← how well this episode went
```

Reward is assigned **once**, after the episode ends.

<!-- Speaker notes: The diagram on this slide is the single most important thing students need to internalize. A Trajectory is not just a list of messages—it's a list of two different types: plain dicts (system, user, tool results) and OpenAI Choice objects (assistant responses). The distinction matters because Choice objects carry log probabilities, which are what GRPO actually trains on. If students mix up the types, they'll get a confusing error deep in the training pipeline. -->

---

## Two Types of Items in messages_and_choices

<div class="columns">

**Plain Dicts**
Context — not trained on

```python
# System
{"role": "system",
 "content": "You are a SQL expert."}

# User
{"role": "user",
 "content": "How many orders?"}

# Tool result
{"role": "tool",
 "content": "42",
 "tool_call_id": "call_abc"}
```

**Choice Objects**
Policy outputs — GRPO trains these

```python
# From completion.choices[0]
completion = await client.chat.completions.create(
    model=model.get_inference_name(),
    messages=trajectory.messages(),
)
choice = completion.choices[0]
# choice.message.content = "SELECT COUNT(*)"
# choice.logprobs = <log probabilities>

trajectory.messages_and_choices.append(choice)
```

</div>

<!-- Speaker notes: Spend time on this slide. The key error students make is appending choice.message instead of choice. The message object (ChatCompletionMessage) has the text content and tool calls, but it doesn't have the log probabilities. The Choice object wraps the message AND includes the logprobs that GRPO needs. Ask students: "Why does GRPO need log probabilities?" Answer: to compute the probability ratio between the new policy and the reference policy—that's the core of the policy gradient update from Module 01. -->

---

## Single-Turn Trajectory

```python
async def rollout_single_turn(
    model: art.Model,
    question: str,
    correct_answer: str,
) -> art.Trajectory:

    client = model.openai_client()

    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": "Answer concisely."},
            {"role": "user",   "content": question},
        ]
    )

    completion = await client.chat.completions.create(
        model=model.get_inference_name(),
        messages=trajectory.messages(),
        temperature=0.8,
    )

    # Append the Choice object (not completion.choices[0].message)
    trajectory.messages_and_choices.append(completion.choices[0])

    answer = completion.choices[0].message.content.strip()
    trajectory.reward = 1.0 if correct_answer.lower() in answer.lower() else 0.0
    return trajectory
```

<!-- Speaker notes: Walk through this code line by line. Point out trajectory.messages()—this is the helper method that converts the mixed list of dicts and Choice objects into the pure dict format that the OpenAI API expects. Students don't need to implement this conversion themselves; ART handles it. The reward assignment at the end is simple here (binary match), but in real tasks you'd run the answer through a test suite, execute SQL against a database, or call a judge LLM. -->

---

## Multi-Turn Trajectory

Agent loops until it succeeds (or runs out of turns):

```python
trajectory = art.Trajectory(
    messages_and_choices=[
        {"role": "system", "content": "Fix your SQL until it works."},
        {"role": "user",   "content": task},
    ]
)

for turn in range(max_turns):
    completion = await client.chat.completions.create(
        model=model.get_inference_name(),
        messages=trajectory.messages(),
        temperature=0.8,
    )
    choice = completion.choices[0]
    trajectory.messages_and_choices.append(choice)   # Choice

    result, error = run_sql(choice.message.content)
    if error is None:
        trajectory.reward = 1.0 - (turn / max_turns) * 0.3  # efficiency bonus
        break

    trajectory.messages_and_choices.append({          # dict
        "role": "user",
        "content": f"Error: {error}. Try again.",
    })
else:
    trajectory.reward = 0.0
```

<!-- Speaker notes: The for/else pattern is clean Python here: the else block runs only if the loop completed without a break (i.e., all turns were used without success). Point out that the trajectory grows each turn—by the end of a 5-turn episode, messages_and_choices has 12 items: system + user + 5 Choice objects + 4 error feedback dicts. The reward shaping (efficiency bonus) rewards fewer turns, which guides the agent to learn concise strategies rather than brute-force retry loops. -->

---

## Multi-Turn Trajectory: What It Looks Like

After a 2-attempt success:

```
messages_and_choices = [
    {role: system}         ← "Fix your SQL until it works."
    {role: user}           ← "Task: count orders by region"
    Choice(assistant)      ← "SELECT region, COUNT(*) FROM orders"  ← WRONG COLUMN
    {role: user}           ← "Error: column 'region' not found"
    Choice(assistant)      ← "SELECT r.name, COUNT(*) FROM orders o JOIN regions r ON ..."
]

reward = 0.85   (succeeded, 1 correction needed)
```

GRPO sees this as one training example.
All assistant decisions get reinforced or discouraged together.

<!-- Speaker notes: The key insight on this slide: GRPO doesn't see two separate decisions (first SQL attempt, second SQL attempt). It sees one episode where a sequence of decisions led to reward 0.85. Through the policy gradient, the probability of the entire sequence is adjusted. This is fundamentally different from training on individual response quality—it's training on episode-level success, which is what agents need. -->

---

## Tool-Calling Trajectory

```python
for _ in range(5):
    completion = await client.chat.completions.create(
        model=model.get_inference_name(),
        messages=trajectory.messages(),
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.8,
    )
    choice = completion.choices[0]
    trajectory.messages_and_choices.append(choice)   # Choice with tool_calls

    if choice.finish_reason == "stop":
        trajectory.reward = 1.0
        break

    for tool_call in choice.message.tool_calls:
        args = json.loads(tool_call.function.arguments)
        result = run_tool(tool_call.function.name, args)
        trajectory.messages_and_choices.append({     # dict: tool result
            "role": "tool",
            "content": str(result),
            "tool_call_id": tool_call.id,
        })
else:
    trajectory.reward = 0.0
```

<!-- Speaker notes: Tool calling adds one wrinkle: the tool_call_id must match between the assistant's tool_calls list and the subsequent tool result message. The OpenAI API enforces this—if the IDs don't match, the API rejects the message sequence. ART doesn't change this behavior; it just stores the messages in whatever format you provide. The for/else pattern appears again: if the loop runs to completion without a break (no stop reason in 5 calls), the agent failed to finish and gets reward 0. -->

---

## Tool-Calling Trajectory: Structure

```
messages_and_choices = [
    {role: system}    "Use run_sql to answer questions"
    {role: user}      "How many orders last week?"
    Choice(assistant) tool_calls=[{name:"run_sql", args:{"query":"SELECT COUNT(*) FROM orders WHERE ..."}}]
    {role: tool}      content="42"  tool_call_id="call_abc"
    Choice(assistant) content="There were 42 orders last week."  finish_reason="stop"
]

reward = 1.0
```

The **tool selection** (which function to call) and **argument formation** (the SQL query) are both part of the policy being trained.

<!-- Speaker notes: This structure makes explicit what GRPO is actually training. The model's decision to call run_sql instead of something else, the specific SQL query it wrote, and the final natural language answer are all assistant-turn Choice objects. They all receive gradient signal based on whether this episode's reward was above or below the group average. An agent that consistently writes correct SQL on the first try will have those patterns reinforced. An agent that writes incorrect SQL will have those patterns suppressed. -->

---

## trajectory.messages() — The Conversion Helper

You never need to manually convert the mixed list. Call `trajectory.messages()`:

```python
# This list contains both dicts AND Choice objects
trajectory.messages_and_choices = [
    {"role": "system", "content": "..."},   # dict
    {"role": "user", "content": "..."},     # dict
    completion.choices[0],                   # Choice object
    {"role": "tool", "content": "..."},     # dict
]

# This call converts everything to pure dicts for the API
messages_for_api = trajectory.messages()

# Then pass to the next completion
next_completion = await client.chat.completions.create(
    model=model.get_inference_name(),
    messages=messages_for_api,   # pure dicts, API-compatible
)
```

`trajectory.messages()` extracts the message content from Choice objects and formats tool calls correctly.

<!-- Speaker notes: This is a quality-of-life feature but an important one. Without it, students would need to write their own serialization logic to convert Choice objects back to message dicts for each subsequent API call. The fact that ART handles this is what makes the append-Choice-then-call-messages() pattern clean. Ask students: what would happen if you called trajectory.messages() before appending a Choice? The list up to that point is returned—it's safe to call at any time to get the current state of the conversation. -->

---

## From Trajectories to GRPO: The Group

```python
# Same scenario, 8 different outcomes due to temperature > 0
group = art.TrajectoryGroup(
    rollout(model, scenario) for _ in range(8)
)

# Example rewards from 8 rollouts of the same question:
# [1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5, 0.0]
# Mean = 0.5

# GRPO advantages (reward - mean):
# [+0.5, -0.5, +0.5, 0.0, -0.5, +0.5, 0.0, -0.5]
```

Trajectories above mean → **reinforced**
Trajectories below mean → **suppressed**
Trajectories at mean → no update

<!-- Speaker notes: Connect back to Module 01 GRPO theory. The advantage is simply reward minus the group mean. This is what makes GRPO "relative"—it doesn't need an absolute scale for rewards, only a relative ordering within the group. A reward of 0.5 can be either above or below the mean depending on how the other 7 trajectories performed. This is why you need temperature > 0 during rollouts: identical trajectories give identical rewards, producing zero advantage for everyone, and the training step does nothing. Variance in outcomes is required for learning. -->

---

## Temperature During Training

During rollouts, always use **temperature > 0**:

```python
completion = await client.chat.completions.create(
    model=model.get_inference_name(),
    messages=trajectory.messages(),
    temperature=0.8,    # not 0.0
)
```

**Why?** Temperature 0 = same output every time = same reward every time = zero advantage = no learning.

```
temperature=0.0:  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] → advantages all 0
temperature=0.8:  [1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5, 0.0] → meaningful advantages
```

GRPO needs **variance** to compute signal.

<!-- Speaker notes: This is a common mistake worth calling out explicitly. Students familiar with inference pipelines often set temperature=0 for deterministic outputs. In a training context, deterministic outputs are useless. The GRPO signal requires diversity in outcomes to know which behaviors to reinforce and which to suppress. The temperature during evaluation (after training, when you're testing the model) can be lower—but during training rollouts it should always be above 0. Typical range: 0.7 to 1.0. -->

---

## Trajectory Validation Checklist

Before sending to training, verify:

```python
def validate_trajectory(t: art.Trajectory) -> list[str]:
    errors = []
    if not t.messages_and_choices:
        errors.append("No messages in trajectory")
    if t.reward is None:
        errors.append("reward is None — must be set")
    if not isinstance(t.reward, (int, float)):
        errors.append(f"reward must be numeric, got {type(t.reward)}")
    # Confirm at least one assistant Choice exists
    has_choice = any(hasattr(item, "message") for item in t.messages_and_choices)
    if not has_choice:
        errors.append("No assistant Choice objects found — check that completions were appended")
    return errors
```

Run this on a test trajectory before starting a long training run.

<!-- Speaker notes: The validate_trajectory function is a practical tool for debugging. The most dangerous error is reward=None—ART will raise a cryptic error during training rather than a clear message about the reward. The second check (no Choice objects) catches the message vs. choice confusion. Encourage students to add this validation to their rollout function during development, and remove it once they're confident the trajectory structure is correct. -->

---

## Common Pitfalls Summary

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Append `choice.message` instead of `choice` | Training fails with log prob error | Append `completion.choices[0]` |
| temperature=0 during rollouts | Training step runs but model doesn't improve | Use temperature=0.7–1.0 |
| All rewards identical in group | Zero advantage, no learning | Vary task difficulty or reward function |
| reward=None | Error during model.train() | Always assign reward before returning |
| Wrong tool_call_id | API error on next message | Use `tool_call.id` from the Choice |

<!-- Speaker notes: This table summarizes the five most common mistakes. Post it somewhere visible during the exercise. The tool_call_id mistake is subtle because the error happens in the next API call, not when you append the tool result—students often blame the wrong line of code. The key to debugging trajectory issues: add the validate_trajectory check, print the trajectory structure, and verify each item's type before submitting to training. -->

---

## Module 02 Summary

**What we covered:**

1. ART's client-backend architecture and LoRA hot-swapping
2. Installation: vLLM → Unsloth → TRL → ART (order matters)
3. Trajectory structure: dict messages + Choice objects + reward

**The Trajectory is the core abstraction:**
- Single-turn: 2 dicts + 1 Choice + reward
- Multi-turn: grows with each agent step
- Tool-calling: Choice with tool_calls + dict tool results

**Next module:** RULER — automatic reward functions using LLM-as-a-judge

<!-- Speaker notes: Summarize the three guides as a cohesive whole. Students now understand what ART does architecturally, how to install it, and how to construct the training data (trajectories) it needs. The exercise (01_art_setup_exercise.py) reinforces trajectory construction without requiring a live backend. Module 03 builds directly on trajectories by showing how to score them automatically using RULER, removing the need to write reward functions by hand. -->

---

<!-- _class: lead -->

# Module 02 Complete

**Exercise:** `exercises/01_art_setup_exercise.py`

**Next Module:** Module 03 — RULER Rewards
Automatic trajectory scoring with LLM-as-a-judge

<!-- Speaker notes: The exercise is self-contained and doesn't require a GPU or live backend. Students construct TrainableModel configs and Trajectory objects, validate their structure, and verify they understand the dict vs. Choice distinction. Completing it takes about 25 minutes. Students who finish early can look at the ART GitHub examples directory to see real training scripts for the tic-tac-toe and email retrieval tasks mentioned earlier. -->
