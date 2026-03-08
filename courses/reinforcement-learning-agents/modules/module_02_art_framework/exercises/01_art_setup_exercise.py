"""
Exercise 01: ART Setup — Config, Trajectories, and Validation

This exercise validates your understanding of the ART framework's core data
structures before you run any live training. No GPU or backend connection is
required — you will construct and validate ART objects directly.

Learning Objectives
-------------------
1. Define a valid art.TrainableModel configuration
2. Construct art.Trajectory objects with the correct message types
3. Distinguish between dict messages and Choice objects in a trajectory
4. Validate trajectory format using the provided checker
5. Build a TrajectoryGroup from multiple trajectories

Estimated Time: 25 minutes

Run this file directly:
    python 01_art_setup_exercise.py

All checks print PASS or FAIL with an explanation.
"""

import json
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Lightweight stubs for art objects
# These replicate the ART API surface used in this exercise.
# In real training, you import these from the openpipe-art package.
# ---------------------------------------------------------------------------

@dataclass
class FunctionCall:
    name: str
    arguments: str  # JSON string


@dataclass
class ToolCall:
    id: str
    type: str
    function: FunctionCall


@dataclass
class Message:
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


@dataclass
class Choice:
    """
    Represents one completion choice returned by the model.
    Equivalent to openai.types.chat.ChatCompletion.choices[0].
    Stores the message AND logprobs (required for GRPO).
    """
    message: Message
    finish_reason: str
    logprobs: dict | None = None  # GRPO requires log probabilities

    def __repr__(self) -> str:
        tool_info = ""
        if self.message.tool_calls:
            names = [tc.function.name for tc in self.message.tool_calls]
            tool_info = f", tool_calls={names}"
        return f"Choice(role=assistant, finish_reason={self.finish_reason!r}{tool_info})"


@dataclass
class Trajectory:
    """
    Represents one complete agent episode.
    messages_and_choices contains a mix of dict messages and Choice objects.
    """
    messages_and_choices: list[Any] = field(default_factory=list)
    reward: float | None = None

    def messages(self) -> list[dict]:
        """
        Convert the mixed list to pure dicts for use in the next API call.
        Choice objects are converted to their message dict representation.
        """
        result = []
        for item in self.messages_and_choices:
            if isinstance(item, dict):
                result.append(item)
            elif isinstance(item, Choice):
                msg = {"role": "assistant"}
                if item.message.content is not None:
                    msg["content"] = item.message.content
                if item.message.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in item.message.tool_calls
                    ]
                result.append(msg)
        return result


@dataclass
class TrainableModel:
    """
    Configuration for the model to be trained.
    Equivalent to art.TrainableModel.
    """
    name: str
    project: str
    base_model: str


@dataclass
class TrainConfig:
    """
    Hyperparameters for a GRPO training step.
    Equivalent to art.TrainConfig.
    """
    learning_rate: float = 1e-5
    num_epochs: int = 1
    max_grad_norm: float = 0.1
    beta: float = 0.04


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_trajectory(trajectory: Trajectory) -> tuple[bool, list[str]]:
    """
    Check a trajectory for structural errors.

    Returns
    -------
    (is_valid, errors)
        is_valid: True if no errors were found
        errors: list of error strings describing each problem
    """
    errors = []

    if not trajectory.messages_and_choices:
        errors.append("messages_and_choices is empty")
        return False, errors

    if trajectory.reward is None:
        errors.append("reward is None — must be set to a float before training")

    if trajectory.reward is not None and not isinstance(trajectory.reward, (int, float)):
        errors.append(f"reward must be numeric (int or float), got {type(trajectory.reward).__name__}")

    # Check that at least one Choice object is present
    has_choice = any(isinstance(item, Choice) for item in trajectory.messages_and_choices)
    if not has_choice:
        errors.append(
            "No Choice objects found in messages_and_choices. "
            "Assistant completions must be appended as Choice objects (not dicts) "
            "so that log probabilities are preserved for GRPO."
        )

    # Check that the first message is system or user
    first = trajectory.messages_and_choices[0]
    if isinstance(first, dict):
        if first.get("role") not in ("system", "user"):
            errors.append(f"First message role must be 'system' or 'user', got '{first.get('role')}'")
    elif isinstance(first, Choice):
        errors.append("First item cannot be a Choice — trajectory must begin with a system or user message")

    # Check that tool results follow assistant messages with tool_calls
    for i, item in enumerate(trajectory.messages_and_choices):
        if isinstance(item, dict) and item.get("role") == "tool":
            if i == 0:
                errors.append("Tool result at position 0 must follow an assistant message with tool_calls")
                continue
            prev = trajectory.messages_and_choices[i - 1]
            if not (isinstance(prev, Choice) and prev.message.tool_calls):
                errors.append(
                    f"Tool result at position {i} must follow an assistant Choice that has tool_calls, "
                    f"but the previous item is: {prev!r}"
                )
            # Check that tool_call_id is present
            if "tool_call_id" not in item:
                errors.append(f"Tool result at position {i} is missing 'tool_call_id'")

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_model_config(model: TrainableModel) -> tuple[bool, list[str]]:
    """
    Check a TrainableModel for required fields.
    """
    errors = []

    if not model.name or not model.name.strip():
        errors.append("model.name must be a non-empty string")

    if not model.project or not model.project.strip():
        errors.append("model.project must be a non-empty string")

    SUPPORTED_MODEL_FAMILIES = (
        "Qwen/Qwen2.5", "Qwen/Qwen3", "meta-llama/Llama", "mistralai/Mistral"
    )
    if not any(model.base_model.startswith(prefix) for prefix in SUPPORTED_MODEL_FAMILIES):
        errors.append(
            f"base_model '{model.base_model}' may not be supported. "
            f"Recommended: Qwen/Qwen2.5-7B-Instruct or meta-llama/Llama-3.1-8B-Instruct. "
            f"Note: Gemma 3 is NOT supported by Unsloth."
        )

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_train_config(config: TrainConfig) -> tuple[bool, list[str]]:
    """
    Check a TrainConfig for common mistakes.
    """
    errors = []

    if config.learning_rate <= 0:
        errors.append(f"learning_rate must be positive, got {config.learning_rate}")

    if config.learning_rate > 5e-4:
        errors.append(
            f"learning_rate {config.learning_rate} is very high for RL training. "
            f"Values above 5e-4 typically cause divergence. "
            f"Recommended: 1e-5 to 1e-4."
        )

    if config.num_epochs < 1:
        errors.append(f"num_epochs must be at least 1, got {config.num_epochs}")

    if config.num_epochs > 3:
        errors.append(
            f"num_epochs={config.num_epochs} is high for RL training. "
            f"Values above 3 risk overfitting to the current batch. "
            f"Recommended: 1."
        )

    if config.max_grad_norm <= 0:
        errors.append(f"max_grad_norm must be positive, got {config.max_grad_norm}")

    if config.beta < 0:
        errors.append(f"beta must be non-negative, got {config.beta}")

    is_valid = len(errors) == 0
    return is_valid, errors


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_check(name: str, passed: bool, errors: list[str]) -> bool:
    if passed:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}")
        for err in errors:
            print(f"          -> {err}")
    return passed


# ===========================================================================
# EXERCISE PART 1: Define a valid TrainableModel configuration
# ===========================================================================

print("\n" + "=" * 60)
print("PART 1: TrainableModel Configuration")
print("=" * 60)

print("""
Task: Create a TrainableModel with:
  - name: a descriptive identifier for this training run (e.g., "sql-agent-v1")
  - project: a project name grouping related runs (e.g., "text-to-sql")
  - base_model: a supported HuggingFace model ID

Replace the placeholder values below with valid configuration.
""")

# ---- YOUR CODE START ----

my_model = TrainableModel(
    name="YOUR_RUN_NAME",        # Replace with a descriptive name
    project="YOUR_PROJECT_NAME", # Replace with a project name
    base_model="YOUR_BASE_MODEL",# Replace with a supported model ID
)

# ---- YOUR CODE END ----

is_valid, errors = validate_model_config(my_model)
run_check("TrainableModel has valid name", bool(my_model.name and my_model.name != "YOUR_RUN_NAME"), [
    "name is still the placeholder 'YOUR_RUN_NAME' — replace it with a real run name"
])
run_check("TrainableModel has valid project", bool(my_model.project and my_model.project != "YOUR_PROJECT_NAME"), [
    "project is still the placeholder 'YOUR_PROJECT_NAME' — replace it"
])
run_check("TrainableModel has supported base_model", is_valid or my_model.base_model != "YOUR_BASE_MODEL",
          ["base_model is still the placeholder — use e.g. 'Qwen/Qwen2.5-7B-Instruct'"])
if my_model.base_model != "YOUR_BASE_MODEL":
    run_check("base_model is from a supported family", is_valid, errors)


# ===========================================================================
# EXERCISE PART 2: Define a valid TrainConfig
# ===========================================================================

print("\n" + "=" * 60)
print("PART 2: TrainConfig Hyperparameters")
print("=" * 60)

print("""
Task: Create a TrainConfig suitable for a first training run.
  - learning_rate: start conservative (1e-5 is the recommended default)
  - num_epochs: keep at 1 to avoid overfitting to the current batch
  - max_grad_norm: 0.1 prevents catastrophic forgetting
  - beta: 0.04 controls KL divergence from the base model
""")

# ---- YOUR CODE START ----

my_config = TrainConfig(
    learning_rate=0.01,  # Fix this — it's too high for RL training
    num_epochs=5,        # Fix this — too many epochs per batch
    max_grad_norm=0.1,
    beta=0.04,
)

# ---- YOUR CODE END ----

config_valid, config_errors = validate_train_config(my_config)
run_check("TrainConfig has safe learning_rate", my_config.learning_rate <= 5e-4, [
    f"learning_rate={my_config.learning_rate} is too high. Use 1e-5 to 1e-4 for RL training."
])
run_check("TrainConfig has safe num_epochs", my_config.num_epochs <= 3, [
    f"num_epochs={my_config.num_epochs} is too high. Keep at 1 for stable RL."
])
run_check("TrainConfig passes full validation", config_valid, config_errors)


# ===========================================================================
# EXERCISE PART 3: Build a single-turn trajectory
# ===========================================================================

print("\n" + "=" * 60)
print("PART 3: Single-Turn Trajectory")
print("=" * 60)

print("""
Task: Build a valid single-turn trajectory for this scenario:
  - System: "You are a SQL expert."
  - User: "How many users signed up in January?"
  - Assistant responds with: "SELECT COUNT(*) FROM users WHERE signup_month = 1"
  - The SQL is correct, so reward = 1.0

Key requirement: the assistant response MUST be a Choice object, not a dict.
""")

# ---- YOUR CODE START ----

# Simulate what the model returns (in real training, this comes from the API)
simulated_completion_choice = Choice(
    message=Message(
        role="assistant",
        content="SELECT COUNT(*) FROM users WHERE signup_month = 1",
    ),
    finish_reason="stop",
    logprobs={"tokens": [-0.1, -0.3, -0.2]},  # simplified log probs
)

single_turn_trajectory = Trajectory(
    messages_and_choices=[
        # Add the system message here as a dict
        # Add the user message here as a dict
        # Add the assistant Choice object here
    ]
)
single_turn_trajectory.reward = None  # Set the correct reward here

# ---- YOUR CODE END ----

print("\nValidating single-turn trajectory...")
valid, errors = validate_trajectory(single_turn_trajectory)
run_check("Single-turn trajectory has messages", len(single_turn_trajectory.messages_and_choices) > 0, [
    "messages_and_choices is empty — add system, user, and assistant messages"
])
run_check("Single-turn trajectory has system message", any(
    isinstance(m, dict) and m.get("role") == "system"
    for m in single_turn_trajectory.messages_and_choices
), ["No system message found — add {'role': 'system', 'content': '...'}"])
run_check("Single-turn trajectory has user message", any(
    isinstance(m, dict) and m.get("role") == "user"
    for m in single_turn_trajectory.messages_and_choices
), ["No user message found — add {'role': 'user', 'content': '...'}"])
run_check("Single-turn trajectory has Choice object", any(
    isinstance(m, Choice) for m in single_turn_trajectory.messages_and_choices
), ["No Choice object found — append simulated_completion_choice to messages_and_choices"])
run_check("Single-turn trajectory has correct reward", single_turn_trajectory.reward == 1.0, [
    f"Expected reward=1.0 for correct SQL, got reward={single_turn_trajectory.reward}"
])
run_check("Single-turn trajectory is fully valid", valid, errors)


# ===========================================================================
# EXERCISE PART 4: Build a multi-turn trajectory
# ===========================================================================

print("\n" + "=" * 60)
print("PART 4: Multi-Turn Trajectory")
print("=" * 60)

print("""
Task: Build a trajectory for a 2-turn episode where the agent:
  1. First attempt: writes incorrect SQL (gets an error)
  2. Second attempt: writes correct SQL (succeeds)

Messages in order:
  1. system: "Fix your SQL until it runs without errors."
  2. user: "Count orders placed after 2024-01-01"
  3. assistant Choice (first attempt, wrong): "SELECT * FROM order WHERE date > 2024-01-01"
  4. user (error feedback): "Error: table 'order' does not exist. Use 'orders'."
  5. assistant Choice (second attempt, correct): "SELECT COUNT(*) FROM orders WHERE date > '2024-01-01'"

The agent succeeded but needed 2 turns, so reward = 0.7 (efficiency penalty applied).
""")

# Simulated completion choices
first_attempt = Choice(
    message=Message(role="assistant", content="SELECT * FROM order WHERE date > 2024-01-01"),
    finish_reason="stop",
    logprobs={"tokens": [-0.2, -0.4, -0.1]},
)

second_attempt = Choice(
    message=Message(role="assistant", content="SELECT COUNT(*) FROM orders WHERE date > '2024-01-01'"),
    finish_reason="stop",
    logprobs={"tokens": [-0.05, -0.1, -0.08]},
)

# ---- YOUR CODE START ----

multi_turn_trajectory = Trajectory(
    messages_and_choices=[
        # Add all 5 messages/choices in order:
        # 1. system dict
        # 2. user dict
        # 3. first_attempt Choice
        # 4. error feedback dict (role="user")
        # 5. second_attempt Choice
    ]
)
multi_turn_trajectory.reward = None  # Set the correct reward

# ---- YOUR CODE END ----

print("\nValidating multi-turn trajectory...")
valid, errors = validate_trajectory(multi_turn_trajectory)
run_check("Multi-turn trajectory has 5 items", len(multi_turn_trajectory.messages_and_choices) == 5, [
    f"Expected 5 items in messages_and_choices, got {len(multi_turn_trajectory.messages_and_choices)}"
])
run_check("Multi-turn trajectory has 2 Choice objects", sum(
    1 for m in multi_turn_trajectory.messages_and_choices if isinstance(m, Choice)
) == 2, ["Expected 2 Choice objects (one per assistant turn)"])
run_check("Multi-turn trajectory has correct reward", multi_turn_trajectory.reward == 0.7, [
    f"Expected reward=0.7, got reward={multi_turn_trajectory.reward}"
])
run_check("Multi-turn trajectory is fully valid", valid, errors)

# Verify messages() conversion works
if multi_turn_trajectory.messages_and_choices:
    converted = multi_turn_trajectory.messages()
    run_check("messages() returns all-dict list", all(isinstance(m, dict) for m in converted), [
        "messages() should return a list of plain dicts — check the Trajectory.messages() implementation"
    ])
    run_check("messages() has correct length", len(converted) == len(multi_turn_trajectory.messages_and_choices), [
        f"messages() returned {len(converted)} items, expected {len(multi_turn_trajectory.messages_and_choices)}"
    ])


# ===========================================================================
# EXERCISE PART 5: Build a tool-calling trajectory
# ===========================================================================

print("\n" + "=" * 60)
print("PART 5: Tool-Calling Trajectory")
print("=" * 60)

print("""
Task: Build a trajectory for a tool-calling episode:
  1. system: "Use run_sql(query) to answer database questions."
  2. user: "What is the total revenue from completed orders?"
  3. assistant Choice with tool_call: run_sql("SELECT SUM(amount) FROM orders WHERE status='completed'")
  4. tool result dict: content="15234.50", tool_call_id must match the tool call id
  5. assistant Choice: content="The total revenue is $15,234.50." finish_reason="stop"

Reward = 1.0 (agent succeeded)

Note: tool_call_id in the tool result dict must match the id in the assistant's tool_calls list.
""")

# Simulated tool-calling completion
TOOL_CALL_ID = "call_revenue_001"

tool_calling_choice = Choice(
    message=Message(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(
                id=TOOL_CALL_ID,
                type="function",
                function=FunctionCall(
                    name="run_sql",
                    arguments=json.dumps({"query": "SELECT SUM(amount) FROM orders WHERE status='completed'"}),
                ),
            )
        ],
    ),
    finish_reason="tool_calls",
    logprobs={"tokens": [-0.1, -0.2]},
)

final_response_choice = Choice(
    message=Message(
        role="assistant",
        content="The total revenue is $15,234.50.",
    ),
    finish_reason="stop",
    logprobs={"tokens": [-0.05, -0.1, -0.08, -0.03]},
)

# ---- YOUR CODE START ----

tool_trajectory = Trajectory(
    messages_and_choices=[
        # 1. system dict
        # 2. user dict
        # 3. tool_calling_choice (Choice with tool_calls)
        # 4. tool result dict — IMPORTANT: tool_call_id must be TOOL_CALL_ID
        # 5. final_response_choice
    ]
)
tool_trajectory.reward = None  # Set the correct reward

# ---- YOUR CODE END ----

print("\nValidating tool-calling trajectory...")
valid, errors = validate_trajectory(tool_trajectory)
run_check("Tool trajectory has 5 items", len(tool_trajectory.messages_and_choices) == 5, [
    f"Expected 5 items, got {len(tool_trajectory.messages_and_choices)}"
])
run_check("Tool trajectory has tool result dict", any(
    isinstance(m, dict) and m.get("role") == "tool"
    for m in tool_trajectory.messages_and_choices
), ["No tool result message found — add {'role': 'tool', 'content': '...', 'tool_call_id': TOOL_CALL_ID}"])
run_check("Tool result has matching tool_call_id", any(
    isinstance(m, dict) and m.get("role") == "tool" and m.get("tool_call_id") == TOOL_CALL_ID
    for m in tool_trajectory.messages_and_choices
), [f"tool_call_id must be '{TOOL_CALL_ID}' — it must match the id in tool_calling_choice.message.tool_calls[0].id"])
run_check("Tool trajectory has correct reward", tool_trajectory.reward == 1.0, [
    f"Expected reward=1.0, got reward={tool_trajectory.reward}"
])
run_check("Tool trajectory is fully valid", valid, errors)


# ===========================================================================
# EXERCISE PART 6: Understand TrajectoryGroup and GRPO advantages
# ===========================================================================

print("\n" + "=" * 60)
print("PART 6: GRPO Advantage Computation")
print("=" * 60)

print("""
Task: Given a group of 8 trajectories from the same scenario with these rewards:
  [1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5, 0.0]

Compute:
  1. The mean reward (this is the GRPO baseline)
  2. The advantage for each trajectory (reward - mean)
  3. Identify which trajectories will be reinforced (positive advantage)
     and which will be suppressed (negative advantage)
""")

group_rewards = [1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5, 0.0]

# ---- YOUR CODE START ----

mean_reward = None           # Compute the mean of group_rewards
advantages = []              # Compute reward - mean for each trajectory

# For each trajectory, classify as "reinforce", "suppress", or "no_update"
classifications = []
for advantage in advantages:
    if advantage > 0:
        classifications.append("reinforce")
    elif advantage < 0:
        classifications.append("suppress")
    else:
        classifications.append("no_update")

# ---- YOUR CODE END ----

run_check("Mean reward is correct", mean_reward == 0.5, [
    f"Expected mean=0.5, got mean={mean_reward}. "
    f"Hint: sum(group_rewards) / len(group_rewards)"
])
run_check("Advantages list has correct length", len(advantages) == len(group_rewards), [
    f"Expected {len(group_rewards)} advantages, got {len(advantages)}"
])

expected_advantages = [r - 0.5 for r in group_rewards]
advantages_correct = all(
    abs(a - e) < 1e-9 for a, e in zip(advantages, expected_advantages)
) if len(advantages) == len(group_rewards) else False
run_check("Advantages are correctly computed", advantages_correct, [
    f"Expected advantages: {expected_advantages}",
    f"Got advantages:      {advantages}",
    "Hint: advantage = reward - mean_reward for each trajectory"
])

expected_classifications = ["reinforce", "suppress", "reinforce", "no_update",
                             "suppress", "reinforce", "no_update", "suppress"]
classifications_correct = classifications == expected_classifications
run_check("Classifications are correct", classifications_correct, [
    f"Expected: {expected_classifications}",
    f"Got:      {classifications}",
])


# ===========================================================================
# Final Summary
# ===========================================================================

print("\n" + "=" * 60)
print("EXERCISE COMPLETE")
print("=" * 60)
print("""
Key takeaways from this exercise:

1. TrainableModel requires: name, project, and a supported base_model
   Use Qwen/Qwen2.5-7B-Instruct for this course.

2. TrainConfig: start with learning_rate=1e-5, num_epochs=1
   RL training is sensitive to high learning rates.

3. Trajectories mix dict messages (system, user, tool results) and
   Choice objects (assistant completions). Choice objects carry log
   probabilities required by GRPO.

4. Single-turn: [dict, dict, Choice] + reward
   Multi-turn: [dict, dict, Choice, dict, Choice, ...] + reward
   Tool-calling: [dict, dict, Choice(tool_calls), dict(tool), Choice] + reward

5. GRPO advantage = reward - group_mean
   Positive advantage → reinforce those decisions
   Negative advantage → suppress those decisions
   Zero advantage → no update (happens when all rewards are identical)

Next: Module 03 — RULER Rewards
Automatic trajectory scoring using an LLM as the judge.
""")
