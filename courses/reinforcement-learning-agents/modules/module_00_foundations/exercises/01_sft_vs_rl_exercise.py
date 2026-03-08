"""
Exercise 01: SFT vs RL — Scenario Classification, Reward Functions, and Advantages

Module 00 Foundations | Reinforcement Learning for AI Agents

Covers:
  - Part A: Classify scenarios as SFT-appropriate or RL-appropriate (with reasoning)
  - Part B: Implement a binary reward function for a tool-calling agent
  - Part C: Implement scalar reward with multiple components
  - Part D: Calculate relative advantages from a group of rollout scores

Run this file directly to check your implementations:
    python 01_sft_vs_rl_exercise.py

All test functions print PASS or a descriptive error message.
"""

import math
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data structures used throughout the exercise
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """A task description for Part A classification."""
    name: str
    description: str
    n_steps: int           # how many sequential steps the task requires
    has_tool_calls: bool   # does the task require calling external tools?
    recoverable: bool      # does the agent need to recover from intermediate errors?


@dataclass
class Rollout:
    """One complete agent episode."""
    prompt: str
    actions: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    final_answer: str = ""
    reward: float = 0.0


# ---------------------------------------------------------------------------
# PART A: Classify scenarios as SFT or RL
# ---------------------------------------------------------------------------

SCENARIOS = [
    Scenario(
        name="email_subject_generation",
        description=(
            "Given the body of an email, generate an appropriate subject line. "
            "Single-turn, no tools, correct answer is subjective."
        ),
        n_steps=1,
        has_tool_calls=False,
        recoverable=False,
    ),
    Scenario(
        name="research_assistant",
        description=(
            "Given a research question, search the web 3-5 times using different queries, "
            "synthesize findings, check for contradictions, and produce a structured report. "
            "Must recover if a search returns no results."
        ),
        n_steps=8,
        has_tool_calls=True,
        recoverable=True,
    ),
    Scenario(
        name="code_translation",
        description=(
            "Translate a Python function to JavaScript. "
            "Single step. Correct translation is well-defined."
        ),
        n_steps=1,
        has_tool_calls=False,
        recoverable=False,
    ),
    Scenario(
        name="database_query_agent",
        description=(
            "Given a natural language question about a database, "
            "explore the schema, write SQL, execute it, handle errors, "
            "and return the formatted result. Average of 5 tool calls required."
        ),
        n_steps=5,
        has_tool_calls=True,
        recoverable=True,
    ),
    Scenario(
        name="single_turn_qa",
        description=(
            "Answer factual questions from a fixed knowledge base. "
            "Single-turn, answer is right or wrong."
        ),
        n_steps=1,
        has_tool_calls=False,
        recoverable=False,
    ),
    Scenario(
        name="trading_decision_agent",
        description=(
            "Fetch market data, compute technical indicators, check portfolio state, "
            "evaluate risk constraints, and decide whether to buy/sell/hold. "
            "6 sequential tool calls with error handling."
        ),
        n_steps=6,
        has_tool_calls=True,
        recoverable=True,
    ),
]


def classify_scenario(scenario: Scenario) -> str:
    """
    Classify a scenario as 'sft', 'rl', or 'both'.

    Decision rule:
      - 'sft'  if n_steps == 1 AND no tool calls AND no recovery needed
      - 'rl'   if n_steps >= 3 AND has tool calls AND recovery is needed
      - 'both' if 1 < n_steps < 3 OR mixed signals

    Args:
        scenario: a Scenario object with task properties

    Returns:
        'sft', 'rl', or 'both'

    Implement this function.
    """
    # TODO: implement classification logic based on scenario properties
    # Use scenario.n_steps, scenario.has_tool_calls, scenario.recoverable
    raise NotImplementedError("Implement classify_scenario")


def test_part_a():
    """Tests for Part A: scenario classification."""
    print("\n--- Part A: Scenario Classification ---")

    expected = {
        "email_subject_generation": "sft",
        "research_assistant": "rl",
        "code_translation": "sft",
        "database_query_agent": "rl",
        "single_turn_qa": "sft",
        "trading_decision_agent": "rl",
    }

    all_passed = True
    for scenario in SCENARIOS:
        try:
            result = classify_scenario(scenario)
        except NotImplementedError:
            print(f"  SKIP  {scenario.name}: classify_scenario not implemented yet")
            all_passed = False
            continue

        exp = expected[scenario.name]
        if result == exp:
            print(f"  PASS  {scenario.name} → {result}")
        else:
            print(
                f"  FAIL  {scenario.name}: expected '{exp}', got '{result}'\n"
                f"        Hint: n_steps={scenario.n_steps}, "
                f"has_tool_calls={scenario.has_tool_calls}, "
                f"recoverable={scenario.recoverable}"
            )
            all_passed = False

    return all_passed


# ---------------------------------------------------------------------------
# PART B: Implement a binary reward function
# ---------------------------------------------------------------------------

def binary_reward(rollout: Rollout, ground_truth: str) -> float:
    """
    Binary reward function for a tool-calling agent.

    Returns 1.0 if the rollout's final_answer contains the ground_truth
    (case-insensitive, after stripping whitespace), 0.0 otherwise.

    Rules:
      - Comparison must be case-insensitive
      - Strip leading/trailing whitespace before comparing
      - ground_truth must be a substring of final_answer (not exact match)
        to handle "the answer is 42" matching ground_truth "42"

    Args:
        rollout: a Rollout object with a final_answer field
        ground_truth: the expected correct value

    Returns:
        1.0 if correct, 0.0 otherwise

    Implement this function.
    """
    # TODO: implement binary reward
    raise NotImplementedError("Implement binary_reward")


def test_part_b():
    """Tests for Part B: binary reward function."""
    print("\n--- Part B: Binary Reward Function ---")

    test_cases = [
        # (rollout_answer, ground_truth, expected_reward, description)
        ("The answer is 127.05", "127.05", 1.0, "answer contains ground truth"),
        ("$127.05 is the result", "127.05", 1.0, "ground truth in different position"),
        ("127.1", "127.05", 0.0, "wrong answer"),
        ("  127.05  ", "127.05", 1.0, "whitespace in answer"),
        ("The ANSWER is 127.05", "127.05", 1.0, "case insensitive answer"),
        ("", "127.05", 0.0, "empty answer"),
        ("42", "42", 1.0, "exact match"),
        ("43", "42", 0.0, "off-by-one"),
    ]

    all_passed = True
    for answer, truth, expected, description in test_cases:
        rollout = Rollout(prompt="test", final_answer=answer)
        try:
            result = binary_reward(rollout, truth)
        except NotImplementedError:
            print("  SKIP  binary_reward not implemented yet")
            return False

        if abs(result - expected) < 1e-9:
            print(f"  PASS  {description}")
        else:
            print(
                f"  FAIL  {description}\n"
                f"        answer='{answer}', ground_truth='{truth}'\n"
                f"        expected {expected:.1f}, got {result:.1f}"
            )
            all_passed = False

    return all_passed


# ---------------------------------------------------------------------------
# PART C: Implement a scalar reward function with multiple components
# ---------------------------------------------------------------------------

def scalar_reward(rollout: Rollout, ground_truth: str) -> float:
    """
    Composite scalar reward for a tool-calling agent.

    Score components (must sum to 1.0 maximum):

    1. Correctness (0.6 points):
       - 0.6 if ground_truth is a case-insensitive substring of final_answer
       - 0.0 otherwise

    2. Efficiency (0.2 points):
       - 0.2 if len(rollout.actions) <= 2
       - 0.1 if len(rollout.actions) is 3 or 4
       - 0.0 if len(rollout.actions) > 4 or == 0

    3. Format compliance (0.2 points):
       - 0.2 if ALL items in rollout.tool_calls start with "<tool_call>"
         AND end with "</tool_call>"
       - 0.2 if rollout.tool_calls is empty (no tool calls required is also valid)
       - 0.0 if any tool call has invalid format

    Total: correctness + efficiency + format (max 1.0)

    Args:
        rollout: a Rollout with final_answer, actions, and tool_calls fields
        ground_truth: the expected correct value

    Returns:
        float in [0.0, 1.0]

    Implement this function.
    """
    # TODO: implement scalar reward
    raise NotImplementedError("Implement scalar_reward")


def test_part_c():
    """Tests for Part C: scalar reward function."""
    print("\n--- Part C: Scalar Reward Function ---")

    test_cases = [
        # (rollout_kwargs, ground_truth, expected_reward, tolerance, description)
        (
            {
                "final_answer": "The answer is 42",
                "actions": ["<tool_call>calc(x=42)</tool_call>"],
                "tool_calls": ["<tool_call>calc(x=42)</tool_call>"],
            },
            "42",
            1.0,
            0.001,
            "perfect rollout: correct, efficient, valid format",
        ),
        (
            {
                "final_answer": "The answer is 42",
                "actions": [
                    "<tool_call>search(q='42')</tool_call>",
                    "<tool_call>search(q='42 meaning')</tool_call>",
                    "<tool_call>search(q='answer')</tool_call>",
                    "<tool_call>calc(x=42)</tool_call>",
                    "<tool_call>verify(x=42)</tool_call>",
                ],
                "tool_calls": [
                    "<tool_call>search(q='42')</tool_call>",
                    "<tool_call>search(q='42 meaning')</tool_call>",
                    "<tool_call>search(q='answer')</tool_call>",
                    "<tool_call>calc(x=42)</tool_call>",
                    "<tool_call>verify(x=42)</tool_call>",
                ],
            },
            "42",
            0.8,
            0.001,
            "correct but inefficient (5 calls)",
        ),
        (
            {
                "final_answer": "I don't know",
                "actions": ["<tool_call>calc(x=99)</tool_call>"],
                "tool_calls": ["<tool_call>calc(x=99)</tool_call>"],
            },
            "42",
            0.4,
            0.001,
            "wrong answer, efficient, valid format",
        ),
        (
            {
                "final_answer": "The answer is 42",
                "actions": ["calc(x=42)"],   # missing tags
                "tool_calls": ["calc(x=42)"],
            },
            "42",
            0.8,
            0.001,
            "correct, efficient, but invalid format",
        ),
        (
            {
                "final_answer": "wrong",
                "actions": [
                    "bad_call_1()",
                    "bad_call_2()",
                    "bad_call_3()",
                    "bad_call_4()",
                    "bad_call_5()",
                ],
                "tool_calls": [
                    "bad_call_1()",
                    "bad_call_2()",
                    "bad_call_3()",
                    "bad_call_4()",
                    "bad_call_5()",
                ],
            },
            "42",
            0.0,
            0.001,
            "all components fail: wrong answer, 5 actions (inefficient), invalid format",
        ),
        (
            {
                "final_answer": "42 is the answer",
                "actions": [],
                "tool_calls": [],
            },
            "42",
            1.0,
            0.001,
            "correct answer, 0 actions (<=2 → efficient), empty tool_calls (→ valid format)",
        ),
    ]

    all_passed = True
    for kwargs, truth, expected, tol, description in test_cases:
        rollout = Rollout(prompt="test", **kwargs)
        try:
            result = scalar_reward(rollout, truth)
        except NotImplementedError:
            print("  SKIP  scalar_reward not implemented yet")
            return False

        if abs(result - expected) <= tol:
            print(f"  PASS  {description} → {result:.2f}")
        else:
            print(
                f"  FAIL  {description}\n"
                f"        expected {expected:.2f}, got {result:.4f}\n"
                f"        final_answer='{kwargs['final_answer']}', "
                f"ground_truth='{truth}'\n"
                f"        n_actions={len(kwargs['actions'])}"
            )
            all_passed = False

    return all_passed


# ---------------------------------------------------------------------------
# PART D: Calculate relative advantages from a group of rollout scores
# ---------------------------------------------------------------------------

def compute_advantages(rewards: list[float]) -> list[float]:
    """
    Compute normalized group-relative advantages from a list of rewards.

    Formula:
        mean  = sum(rewards) / len(rewards)
        std   = sqrt(sum((r - mean)^2 for r in rewards) / len(rewards))
        A_i   = (rewards[i] - mean) / std   for each i

    Edge cases:
        - If std < 1e-8 (all rewards are identical), return [0.0] * len(rewards)
        - If rewards is empty, return []

    Args:
        rewards: list of scalar reward values for G rollouts of the same prompt

    Returns:
        list of normalized advantage values (same length as rewards)
        Properties:
          - mean of advantages is approximately 0.0
          - std of advantages is approximately 1.0 (unless all rewards are equal)
          - Positive advantage: better than group average → reinforce
          - Negative advantage: worse than group average → suppress

    Implement this function using only Python built-ins and math module.
    Do not use numpy.
    """
    # TODO: implement compute_advantages
    raise NotImplementedError("Implement compute_advantages")


def test_part_d():
    """Tests for Part D: advantage computation."""
    print("\n--- Part D: Relative Advantage Computation ---")

    all_passed = True

    # Test 1: Basic computation
    rewards = [0.2, 0.8, 1.0, 0.6, 0.4, 1.0, 0.0, 0.6]
    try:
        adv = compute_advantages(rewards)
    except NotImplementedError:
        print("  SKIP  compute_advantages not implemented yet")
        return False

    if len(adv) != len(rewards):
        print(f"  FAIL  basic: expected {len(rewards)} values, got {len(adv)}")
        all_passed = False
    else:
        print(f"  PASS  basic: returned {len(adv)} values")

    # Test 2: Mean of advantages should be ~0
    mean_adv = sum(adv) / len(adv)
    if abs(mean_adv) < 1e-6:
        print(f"  PASS  mean of advantages ≈ 0.0 (got {mean_adv:.8f})")
    else:
        print(f"  FAIL  mean should be ~0.0, got {mean_adv:.6f}")
        all_passed = False

    # Test 3: Std of advantages should be ~1
    variance = sum((a - mean_adv) ** 2 for a in adv) / len(adv)
    std_adv = math.sqrt(variance)
    if abs(std_adv - 1.0) < 1e-6:
        print(f"  PASS  std of advantages ≈ 1.0 (got {std_adv:.8f})")
    else:
        print(f"  FAIL  std should be ~1.0, got {std_adv:.6f}")
        all_passed = False

    # Test 4: Ordering preserved — highest reward → highest advantage
    reward_rank = sorted(range(len(rewards)), key=lambda i: rewards[i], reverse=True)
    adv_rank = sorted(range(len(adv)), key=lambda i: adv[i], reverse=True)
    if reward_rank == adv_rank:
        print("  PASS  ordering preserved: highest reward → highest advantage")
    else:
        print(
            f"  FAIL  ordering not preserved\n"
            f"        reward ranking: {reward_rank}\n"
            f"        advantage ranking: {adv_rank}"
        )
        all_passed = False

    # Test 5: All identical rewards → all zero advantages
    identical_rewards = [0.5, 0.5, 0.5, 0.5]
    adv_identical = compute_advantages(identical_rewards)
    if all(abs(a) < 1e-8 for a in adv_identical):
        print("  PASS  identical rewards → all-zero advantages")
    else:
        print(
            f"  FAIL  identical rewards should give all zeros, got {adv_identical}"
        )
        all_passed = False

    # Test 6: Empty list
    adv_empty = compute_advantages([])
    if adv_empty == []:
        print("  PASS  empty rewards → empty advantages")
    else:
        print(f"  FAIL  empty input should return [], got {adv_empty}")
        all_passed = False

    # Test 7: Sign correctness
    simple_rewards = [1.0, 0.0]
    adv_simple = compute_advantages(simple_rewards)
    if adv_simple[0] > 0 and adv_simple[1] < 0:
        print(
            f"  PASS  sign correct: high reward → positive ({adv_simple[0]:.3f}), "
            f"low reward → negative ({adv_simple[1]:.3f})"
        )
    else:
        print(
            f"  FAIL  sign error: rewards [1.0, 0.0] → advantages {adv_simple}\n"
            f"        expected [positive, negative]"
        )
        all_passed = False

    # Show the full output for the main test case
    print("\n  Full advantage table for rewards = [0.2, 0.8, 1.0, 0.6, 0.4, 1.0, 0.0, 0.6]:")
    print(f"  {'Rollout':>8} {'Reward':>8} {'Advantage':>10} {'Signal':>12}")
    print(f"  {'-'*44}")
    for i, (r, a) in enumerate(zip(rewards, adv)):
        signal = "REINFORCE" if a > 0 else ("SUPPRESS" if a < 0 else "NEUTRAL")
        print(f"  {i+1:>8} {r:>8.2f} {a:>10.3f} {signal:>12}")

    return all_passed


# ---------------------------------------------------------------------------
# BONUS: Tie it all together
# ---------------------------------------------------------------------------

def simulate_grpo_step(
    prompts: list[str],
    ground_truths: list[str],
    group_size: int = 4,
) -> dict[str, Any]:
    """
    Simulate one step of GRPO training using the functions you implemented.

    This function demonstrates how Parts B, C, and D connect:
    1. For each prompt, generate group_size simulated rollouts
    2. Score each rollout with scalar_reward
    3. Compute group-relative advantages with compute_advantages
    4. Return a summary of what the policy update would look like

    This uses synthetic rollouts (varying quality) to simulate the process.
    In real training, rollouts come from the actual language model.

    Args:
        prompts: list of prompt strings
        ground_truths: list of correct answers (same length as prompts)
        group_size: number of rollouts per prompt (G in GRPO)

    Returns:
        dict with 'mean_reward', 'mean_advantage_std', 'update_summary'
    """
    import random
    random.seed(42)

    all_rewards = []
    all_advantages = []
    update_summary = []

    for prompt, truth in zip(prompts, ground_truths):
        # Simulate rollouts with varying quality (in real training: LLM generates these)
        group_rollouts = []
        for i in range(group_size):
            quality = random.random()
            is_correct = quality > 0.4
            n_actions = random.choice([1, 2, 3, 4, 5])
            use_valid_format = quality > 0.3

            rollout = Rollout(
                prompt=prompt,
                final_answer=f"The answer is {truth}" if is_correct else "I don't know",
                actions=[f"action_{j}" for j in range(n_actions)],
                tool_calls=(
                    [f"<tool_call>tool_{j}()</tool_call>" for j in range(n_actions)]
                    if use_valid_format
                    else [f"bad_call_{j}()" for j in range(n_actions)]
                ),
            )
            group_rollouts.append(rollout)

        # Score all rollouts in the group
        try:
            rewards = [scalar_reward(r, truth) for r in group_rollouts]
        except NotImplementedError:
            return {"error": "scalar_reward not implemented"}

        # Compute group-relative advantages
        try:
            advantages = compute_advantages(rewards)
        except NotImplementedError:
            return {"error": "compute_advantages not implemented"}

        all_rewards.extend(rewards)
        all_advantages.extend(advantages)

        best_idx = max(range(len(rewards)), key=lambda i: rewards[i])
        worst_idx = min(range(len(rewards)), key=lambda i: rewards[i])

        update_summary.append({
            "prompt": prompt[:40],
            "mean_reward": sum(rewards) / len(rewards),
            "best_reward": rewards[best_idx],
            "worst_reward": rewards[worst_idx],
            "best_advantage": advantages[best_idx],
            "worst_advantage": advantages[worst_idx],
        })

    mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
    mean_adv_std = (
        (sum((a ** 2 for a in all_advantages)) / len(all_advantages)) ** 0.5
        if all_advantages else 0.0
    )

    return {
        "mean_reward": mean_reward,
        "mean_advantage_std": mean_adv_std,
        "update_summary": update_summary,
    }


def run_bonus():
    """Run the bonus simulation and print results."""
    print("\n--- Bonus: GRPO Step Simulation ---")

    prompts = [
        "What is 15% of 847?",
        "What is the square root of 144?",
        "What is 2 to the power of 10?",
    ]
    ground_truths = ["127.05", "12", "1024"]

    result = simulate_grpo_step(prompts, ground_truths, group_size=6)

    if "error" in result:
        print(f"  Cannot run bonus: {result['error']}")
        return

    print(f"  Mean group reward: {result['mean_reward']:.3f}")
    print(f"  Mean advantage std: {result['mean_advantage_std']:.3f}")
    print()
    print(f"  {'Prompt':>42} {'Mean R':>7} {'Best A':>7} {'Worst A':>8}")
    print(f"  {'-'*66}")
    for s in result["update_summary"]:
        print(
            f"  {s['prompt']:>42} {s['mean_reward']:>7.3f} "
            f"{s['best_advantage']:>7.3f} {s['worst_advantage']:>8.3f}"
        )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Module 00 Exercise: SFT vs RL")
    print("=" * 60)

    results = {
        "Part A": test_part_a(),
        "Part B": test_part_b(),
        "Part C": test_part_c(),
        "Part D": test_part_d(),
    }

    run_bonus()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for part, passed in results.items():
        status = "PASS" if passed else "FAIL or INCOMPLETE"
        print(f"  {part}: {status}")

    total_passed = sum(1 for v in results.values() if v)
    print(f"\n  {total_passed}/{len(results)} parts complete")

    if total_passed == len(results):
        print(
            "\n  All parts complete. You are ready for Module 01: GRPO Algorithm."
        )
    else:
        print(
            "\n  Complete the remaining parts, then re-run this file."
        )
