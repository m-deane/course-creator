# Trajectories: Recording Agent Episodes for GRPO Training

## In Brief

A Trajectory is ART's representation of one complete agent run — the full history of messages, tool calls, tool results, and model responses from start to finish. Trajectories are the training data for GRPO: groups of trajectories from the same scenario are compared by their rewards to compute relative advantages, which drive the policy update.

## Key Insight

The Trajectory is the unit of learning. A reward attached to a Trajectory tells GRPO "this sequence of decisions led to this outcome." GRPO then adjusts the model's parameters to make good-outcome sequences more likely and bad-outcome sequences less likely — for the entire episode, not just the last response.

---

## What a Trajectory Contains

A completed Trajectory has two components:

1. **`messages_and_choices`** — the ordered list of turns in the conversation, including system prompts, user messages, assistant completions, tool calls, and tool results
2. **`reward`** — a float assigned after the episode ends, scoring how well the agent performed

```python
import art

trajectory = art.Trajectory(
    messages_and_choices=[
        {"role": "system", "content": "You are a SQL expert."},
        {"role": "user",   "content": "How many users signed up this week?"},
    ]
)
# ... agent runs ...
trajectory.reward = 1.0  # set after the episode completes
```

The `messages_and_choices` list grows as the agent runs. Each turn appends to it. The reward is assigned exactly once, at the end.

---

## Message Types in a Trajectory

### Plain Messages (dicts)

System, user, and tool result messages are stored as plain Python dicts following the OpenAI message format:

```python
# System message
{"role": "system", "content": "You are a helpful assistant."}

# User message
{"role": "user", "content": "What is the capital of France?"}

# Tool result message
{
    "role": "tool",
    "content": "Paris",          # the result returned by the tool function
    "tool_call_id": "call_abc",  # must match the id in the assistant's tool_calls
}
```

### Completion Choices (OpenAI Choice objects)

Assistant turns — whether they contain text responses or tool calls — are stored as OpenAI `Choice` objects, not dicts. These come directly from the `chat.completions.create` response:

```python
completion = await client.chat.completions.create(
    model=model.get_inference_name(),
    messages=trajectory.messages(),
)

# completion.choices[0] is a Choice object, not a dict
choice = completion.choices[0]
trajectory.messages_and_choices.append(choice)
```

ART stores Choice objects rather than converting them to dicts because they carry additional data — token log probabilities, finish reasons, usage statistics — that GRPO requires for computing the policy gradient.

---

## A Single-Turn Trajectory

The simplest case: one user question, one assistant response.

```python
import art
import asyncio

async def build_single_turn_trajectory(
    model: art.Model,
    question: str,
    correct_answer: str,
) -> art.Trajectory:

    client = model.openai_client()

    # Initialize with system + user turns
    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": "Answer concisely and accurately."},
            {"role": "user",   "content": question},
        ]
    )

    # Get the assistant response
    completion = await client.chat.completions.create(
        model=model.get_inference_name(),
        messages=trajectory.messages(),   # converts the list to the right format
        temperature=0.8,
    )

    # Append the assistant's Choice object
    trajectory.messages_and_choices.append(completion.choices[0])

    # Score the response
    agent_answer = completion.choices[0].message.content.strip().lower()
    trajectory.reward = 1.0 if correct_answer.lower() in agent_answer else 0.0

    return trajectory
```

After this function returns, the trajectory contains:
```
[dict: system] [dict: user] [Choice: assistant_response]
reward = 1.0 or 0.0
```

---

## A Multi-Turn Trajectory

For agents that loop — generating a response, receiving environment feedback, generating another response — the trajectory grows with each turn:

```python
async def build_multi_turn_trajectory(
    model: art.Model,
    scenario: dict,
    db_connection,
    max_turns: int = 5,
) -> art.Trajectory:

    client = model.openai_client()

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": (
                    "You are a SQL debugging assistant. "
                    "You will receive error messages after each query attempt. "
                    "Fix your query until it succeeds."
                ),
            },
            {"role": "user", "content": f"Task: {scenario['task']}\nSchema: {scenario['schema']}"},
        ]
    )

    final_result = None
    for turn in range(max_turns):
        # Get the next agent response
        completion = await client.chat.completions.create(
            model=model.get_inference_name(),
            messages=trajectory.messages(),
            temperature=0.8,
        )
        choice = completion.choices[0]
        trajectory.messages_and_choices.append(choice)

        # Extract the SQL from the response and try running it
        sql_query = extract_sql(choice.message.content)
        result, error = run_sql(db_connection, sql_query)

        if error is None:
            # Success — no more turns needed
            final_result = result
            break

        # Failure — add the error as a user message and continue
        trajectory.messages_and_choices.append({
            "role": "user",
            "content": f"Error: {error}\nTry again.",
        })

    # Score based on whether the agent eventually succeeded
    # Penalize for the number of turns used (reward shaping)
    if final_result is not None:
        efficiency_bonus = 1.0 - (turn / max_turns) * 0.3  # bonus for fewer turns
        trajectory.reward = efficiency_bonus
    else:
        trajectory.reward = 0.0

    return trajectory
```

After a 3-turn run where the agent succeeds on the second attempt:
```
[dict: system]
[dict: user: initial task]
[Choice: agent_sql_attempt_1]  ← assistant, contains SQL
[dict: user: error message]    ← environment feedback
[Choice: agent_sql_attempt_2]  ← assistant, corrected SQL
reward = 0.87  (succeeded with turns to spare)
```

---

## A Tool-Calling Trajectory

When the model uses tool calls, the trajectory structure captures both the tool call request and the tool result:

```python
import json

# Tool definition passed to the model
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Execute a SQL query and return results",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The SQL query to execute"}
                },
                "required": ["query"],
            },
        },
    }
]

async def build_tool_calling_trajectory(
    model: art.Model,
    question: str,
    db_connection,
) -> art.Trajectory:

    client = model.openai_client()

    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": "Use run_sql to answer database questions."},
            {"role": "user",   "content": question},
        ]
    )

    success = False
    for _ in range(5):  # allow up to 5 tool calls
        completion = await client.chat.completions.create(
            model=model.get_inference_name(),
            messages=trajectory.messages(),
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.8,
        )
        choice = completion.choices[0]
        trajectory.messages_and_choices.append(choice)

        # Check if the model finished (no more tool calls)
        if choice.finish_reason == "stop":
            success = True
            break

        # Process tool calls the model requested
        if choice.message.tool_calls:
            for tool_call in choice.message.tool_calls:
                # Execute the tool
                args = json.loads(tool_call.function.arguments)
                result, error = run_sql(db_connection, args["query"])

                tool_result_content = result if error is None else f"Error: {error}"

                # Add the tool result to the trajectory as a dict
                trajectory.messages_and_choices.append({
                    "role": "tool",
                    "content": str(tool_result_content),
                    "tool_call_id": tool_call.id,
                })

    trajectory.reward = 1.0 if success else 0.0
    return trajectory
```

The resulting trajectory for a 2-call successful run:
```
[dict:   system]
[dict:   user: question]
[Choice: assistant, tool_calls=[{name:"run_sql", args:{query:"SELECT ..."}}]]
[dict:   tool, content="Error: column not found", tool_call_id="call_abc"]
[Choice: assistant, tool_calls=[{name:"run_sql", args:{query:"SELECT ..."}}]]
[dict:   tool, content="42", tool_call_id="call_def"]
[Choice: assistant, content="The answer is 42.", finish_reason="stop"]
reward = 1.0
```

Every decision — which SQL to write, how to fix the error, when to stop — is part of the training example.

---

## How ART Records Trajectories Automatically

The `trajectory.messages()` method converts the `messages_and_choices` list into the OpenAI message format expected by `chat.completions.create`. It handles the conversion of Choice objects back to dict format for the API call.

You do not need to manually convert anything between formats. The workflow is:

```
1. Start with dicts (system, user)
2. Append Choice objects from completions
3. Append dicts for tool results and user feedback
4. Repeat until episode ends
5. Call trajectory.messages() to get the current conversation for the next API call
```

ART also records which messages are "trainable" — the assistant's Choice objects, specifically their log probabilities. Plain dict messages (system, user, tool results) are context, not policy outputs, and are excluded from the gradient computation.

---

## How Trajectories Feed into GRPO

GRPO requires groups of trajectories from the same scenario to compute relative advantages. A `TrajectoryGroup` bundles these together:

```python
# Run the same scenario 8 times (the GRPO group)
group = art.TrajectoryGroup(
    rollout(model, scenario) for _ in range(8)
)
```

When GRPO processes this group, it:

1. Collects the 8 reward scores: e.g., `[1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5, 0.0]`
2. Computes the mean reward: `0.5`
3. Computes relative advantages: reward minus mean for each trajectory
   - Trajectories with reward 1.0 → advantage = +0.5 (reinforce these decisions)
   - Trajectories with reward 0.0 → advantage = -0.5 (discourage these decisions)
4. Uses advantages to weight the policy gradient update

This is why you need multiple trajectories per scenario: a single trajectory provides no signal about whether the outcome was good or bad relative to other possible trajectories. The group provides the reference distribution.

```python
# Multiple groups across different scenarios
train_groups = await art.gather_trajectory_groups(
    (
        art.TrajectoryGroup(rollout(model, scenario) for _ in range(8))
        for scenario in training_scenarios
    ),
    pbar_desc="collecting rollouts",
)

# Send all groups to the backend for one GRPO update
await model.train(train_groups, config=art.TrainConfig(learning_rate=1e-5))
```

---

## Trajectory Validation

Before sending trajectories to training, verify they are well-formed:

```python
def validate_trajectory(trajectory: art.Trajectory) -> list[str]:
    """
    Check a trajectory for common structural errors.
    Returns a list of error messages (empty list means valid).
    """
    errors = []

    if not trajectory.messages_and_choices:
        errors.append("Trajectory has no messages")
        return errors

    if trajectory.reward is None:
        errors.append("trajectory.reward is None — must be set before training")

    if not isinstance(trajectory.reward, (int, float)):
        errors.append(f"trajectory.reward must be numeric, got {type(trajectory.reward)}")

    # Check for alternating roles (basic sanity check)
    import openai
    roles = []
    for item in trajectory.messages_and_choices:
        if isinstance(item, dict):
            roles.append(item.get("role", "unknown"))
        elif hasattr(item, "message"):
            roles.append("assistant")  # Choice object

    # Trajectory must start with system or user message
    if roles and roles[0] not in ("system", "user"):
        errors.append(f"First message must be system or user, got '{roles[0]}'")

    # Tool result must follow an assistant message with tool_calls
    for i, role in enumerate(roles):
        if role == "tool" and (i == 0 or roles[i - 1] != "assistant"):
            errors.append(f"Tool result at position {i} must follow an assistant message")

    return errors


# Usage
trajectory = await rollout(model, scenario)
errors = validate_trajectory(trajectory)
if errors:
    for error in errors:
        print(f"Trajectory error: {error}")
else:
    print(f"Valid trajectory with reward={trajectory.reward}")
```

---

## Common Pitfalls

- **Not appending the Choice object:** Appending `completion.choices[0].message` (a ChatCompletionMessage) instead of `completion.choices[0]` (a Choice). The Choice carries log probabilities; the message does not. GRPO will fail without log probabilities.

- **Setting reward before the episode ends:** If reward is assigned partway through a multi-turn loop and then overwritten, there is no bug — `trajectory.reward` is just a float attribute. But it reads poorly. Assign reward once, at the very end of the rollout function.

- **Same reward for every trajectory in a group:** If all 8 trajectories in a group have reward 1.0 (or all have 0.0), GRPO computes zero advantage for all of them and makes no update. The training step completes but the model doesn't change. This is usually a sign that the task is too easy (all trajectories succeed) or too hard (none do). Adjust scenario difficulty or the reward function.

- **Using temperature=0 during training:** Greedy decoding (temperature=0) produces the same trajectory every time a scenario is run. GRPO needs diverse trajectories to compute meaningful relative advantages. Always use temperature > 0 during training rollouts (0.7–1.0 is typical).

---

## Connections

- **Builds on:** Module 01 — GRPO Algorithm (advantage computation from trajectory groups)
- **Builds on:** Guide 01 — ART Architecture (how trajectories flow from client to backend)
- **Leads to:** Module 03 — RULER Rewards (automatic trajectory scoring)
- **Leads to:** Module 05 — Training Loop (full orchestration using these patterns)
- **Leads to:** Module 06 — Text-to-SQL Agent (end-to-end trajectory construction for a real task)

## Further Reading

- [ART Client Documentation](https://art.openpipe.ai/fundamentals/art-client) — Trajectory API reference
- [ART GitHub README](https://github.com/OpenPipe/ART/blob/main/README.md) — full training examples with trajectory construction
- [W&B ART Tutorial](https://wandb.ai/onlineinference/genai-research/reports/Tutorial-The-OpenPipe-ART-project--VmlldzoxMzcxMDQyMg) — walkthrough of trajectory-based training
