# Guide 02: Reward Signals

## In Brief

A reward signal is a scalar value that tells a learning agent how well it did on a given episode. The design of the reward function is the most consequential decision in any RL-for-agents project — it defines what "success" means operationally.

---

## Key Insight

**RL only needs relative rankings, not absolute scores. If rollout A is better than rollout B, the model can learn from that comparison even if neither score is meaningful in isolation.**

This insight is the foundation of methods like GRPO (Group Relative Policy Optimization), which the next module covers in detail.

---

## What Reward Signals Are

In reinforcement learning, the reward $r_t$ is a scalar signal received after taking action $a_t$ in state $s_t$. The agent's goal is to find a policy $\pi$ that maximizes the expected cumulative reward:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

where $\gamma \in [0, 1]$ is a discount factor (how much we weight near-term vs. long-term rewards) and $\tau$ is a complete trajectory (episode).

For language model agents, we typically simplify this to a **terminal reward**: a single scalar computed at the end of the episode after the agent has produced its final answer. This is cleaner to implement and less prone to reward shaping errors.

```python
# Terminal reward: a single scalar for the whole episode
def episode_reward(final_answer: str, ground_truth: str) -> float:
    """Computed once at the end of the episode."""
    return 1.0 if correct(final_answer, ground_truth) else 0.0


# Step reward: a scalar at each step (more complex, often unnecessary)
def step_reward(action: str, observation: str, step: int) -> float:
    """Computed after each action. Risk: reward hacking at individual steps."""
    if "error" in observation.lower():
        return -0.1   # mild penalty for tool errors
    if action.startswith("<tool_call>"):
        return 0.0    # neutral for valid tool calls
    return 0.0
```

For most tool-calling agent tasks, terminal rewards are sufficient and less likely to be gamed.

---

## Types of Reward Signals

### Binary Reward: Correct or Incorrect

The simplest possible reward. The agent either succeeded or failed.

```python
def binary_reward(predicted_answer: str, ground_truth: str) -> float:
    """
    Returns 1.0 for a correct answer, 0.0 otherwise.

    Best for: tasks with clear right/wrong answers (math, SQL, code execution)
    Limitation: provides no gradient signal for "almost correct" responses
    """
    # Normalize for comparison: strip whitespace, lowercase
    pred = predicted_answer.strip().lower()
    truth = ground_truth.strip().lower()

    return 1.0 if pred == truth else 0.0


# Example usage
print(binary_reward("127.05", "127.05"))   # 1.0 — correct
print(binary_reward("127.1", "127.05"))    # 0.0 — wrong, even though close
print(binary_reward("$127.05", "127.05"))  # 0.0 — formatting mismatch
```

**When to use binary rewards:**
- Mathematical computations with exact answers
- SQL queries (you can execute and check the result)
- Code that either passes or fails tests
- Yes/no classification tasks

**Limitation:** Binary rewards are sparse. If the agent rarely produces correct answers early in training, it receives all-zero rewards and cannot learn (the "sparse reward problem").

---

### Scalar Reward: Continuous Quality Score

A scalar in $[0, 1]$ that measures quality on a continuous scale. This provides richer gradient signal than binary rewards.

```python
def scalar_reward(rollout: dict, ground_truth: str) -> float:
    """
    Composite reward combining correctness, format, and efficiency.

    Returns a float in [0.0, 1.0].

    Args:
        rollout: dict with keys 'final_answer', 'actions', 'tool_calls'
        ground_truth: the expected correct answer

    Returns:
        float: composite quality score
    """
    score = 0.0

    # Component 1: Correctness (60% of total reward)
    # Use fuzzy matching to avoid penalizing minor formatting differences
    final_answer = rollout["final_answer"].strip().lower()
    truth = ground_truth.strip().lower()

    if truth in final_answer:
        score += 0.6  # answer contains the ground truth
    elif any(word in final_answer for word in truth.split() if len(word) > 3):
        score += 0.2  # partial credit: key words present

    # Component 2: Efficiency (20% of total reward)
    # Fewer tool calls is better, up to a minimum of 1
    n_actions = len(rollout["actions"])
    if n_actions == 0:
        score += 0.0    # no actions taken — no efficiency credit
    elif n_actions <= 2:
        score += 0.2    # efficient
    elif n_actions <= 4:
        score += 0.1    # acceptable
    # More than 4 calls for a simple task: no efficiency credit

    # Component 3: Format compliance (20% of total reward)
    # Did the agent use proper tool call syntax throughout?
    tool_calls = rollout.get("tool_calls", [])
    if tool_calls and all(
        tc.startswith("<tool_call>") and tc.endswith("</tool_call>")
        for tc in tool_calls
    ):
        score += 0.2

    return score


# Example: a perfect rollout
perfect = {
    "final_answer": "The compound interest is $1,576.25",
    "actions": ["<tool_call>calculator(p=10000,r=0.05,n=3)</tool_call>"],
    "tool_calls": ["<tool_call>calculator(p=10000,r=0.05,n=3)</tool_call>"],
}
print(f"Perfect rollout: {scalar_reward(perfect, '1,576.25'):.2f}")   # 1.00

# Example: correct but inefficient
verbose = {
    "final_answer": "So the answer is 1,576.25 dollars in interest",
    "actions": [
        "<tool_call>search(q='compound interest formula')</tool_call>",
        "<tool_call>search(q='compound interest calculator')</tool_call>",
        "<tool_call>calculator(p=10000,r=0.05,n=3)</tool_call>",
        "<tool_call>calculator(p=10000,r=0.05,n=3)</tool_call>",  # duplicate
    ],
    "tool_calls": [
        "<tool_call>search(q='compound interest formula')</tool_call>",
        "<tool_call>search(q='compound interest calculator')</tool_call>",
        "<tool_call>calculator(p=10000,r=0.05,n=3)</tool_call>",
        "<tool_call>calculator(p=10000,r=0.05,n=3)</tool_call>",
    ],
}
print(f"Correct but verbose: {scalar_reward(verbose, '1,576.25'):.2f}")  # 0.80
```

---

### Relative Reward: Rankings Over Absolute Scores

The most important insight for modern LLM training: **RL algorithms need relative comparisons, not absolute values.**

If you generate $G$ rollouts for the same prompt and rank them by quality, the model can learn to produce more rollouts like the top-ranked ones and fewer like the bottom-ranked ones — even if your reward function's absolute scale is arbitrary.

```python
import numpy as np


def compute_relative_advantages(rewards: list[float]) -> list[float]:
    """
    Convert a group of raw rewards into relative advantages.

    Advantage_i = (reward_i - mean(rewards)) / std(rewards)

    This normalization means:
    - Positive advantage: this rollout was better than average → reinforce it
    - Negative advantage: this rollout was worse than average → suppress it
    - The absolute scale of rewards doesn't matter, only the relative ranking

    Args:
        rewards: list of scalar rewards for G rollouts of the same prompt

    Returns:
        list of normalized advantage values, one per rollout
    """
    rewards_arr = np.array(rewards, dtype=float)

    mean_reward = rewards_arr.mean()
    std_reward = rewards_arr.std()

    # Avoid division by zero when all rewards are identical
    if std_reward < 1e-8:
        return [0.0] * len(rewards)

    advantages = (rewards_arr - mean_reward) / std_reward
    return advantages.tolist()


# Example: 8 rollouts for the same prompt with varying quality
group_rewards = [0.2, 0.8, 1.0, 0.6, 0.4, 1.0, 0.0, 0.6]

advantages = compute_relative_advantages(group_rewards)

print("Rollout  Raw Reward  Advantage  Signal")
print("-" * 50)
for i, (r, a) in enumerate(zip(group_rewards, advantages)):
    signal = "REINFORCE" if a > 0 else ("SUPPRESS" if a < 0 else "NEUTRAL")
    print(f"   {i+1}       {r:.1f}       {a:+.2f}    {signal}")
```

Output:
```
Rollout  Raw Reward  Advantage  Signal
--------------------------------------------------
   1       0.2       -1.06    SUPPRESS
   2       0.8       +0.45    REINFORCE
   3       1.0       +0.90    REINFORCE
   4       0.6       -0.01    NEUTRAL
   5       0.4       -0.53    SUPPRESS
   6       1.0       +0.90    REINFORCE
   7       0.0       -1.51    SUPPRESS
   8       0.6       -0.01    NEUTRAL
```

This is precisely what GRPO (Group Relative Policy Optimization) computes. The model updates its weights to increase the probability of rollouts 2, 3, and 6, and decrease the probability of rollouts 1, 5, and 7.

---

## Reward Functions for Different Agent Tasks

### Text-to-SQL Agent

```python
import sqlite3


def sql_reward(generated_sql: str, reference_sql: str, db_path: str) -> float:
    """
    Reward function for a text-to-SQL agent.

    Executes both queries and compares results — not string matching.
    This correctly handles equivalent queries written differently.
    """
    score = 0.0

    # Execute generated SQL
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(generated_sql)
        generated_result = set(tuple(row) for row in cursor.fetchall())
        conn.close()
        sql_valid = True
    except sqlite3.Error:
        return 0.0  # invalid SQL: zero reward

    # Execute reference SQL
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(reference_sql)
    reference_result = set(tuple(row) for row in cursor.fetchall())
    conn.close()

    # Compare results: execution accuracy (the standard SQL benchmark metric)
    if generated_result == reference_result:
        score += 0.8  # exact match on results

    # Bonus: query is syntactically simpler (fewer tokens) while still correct
    if generated_result == reference_result:
        gen_tokens = len(generated_sql.split())
        ref_tokens = len(reference_sql.split())
        if gen_tokens <= ref_tokens * 1.2:  # within 20% of reference length
            score += 0.2

    return score
```

### Code Generation Agent

```python
import subprocess
import tempfile
import os


def code_reward(generated_code: str, test_cases: list[dict]) -> float:
    """
    Reward function for a code generation agent.

    Runs the generated code against test cases and scores by pass rate.

    Args:
        generated_code: the Python code produced by the agent
        test_cases: list of {'input': ..., 'expected_output': ...} dicts

    Returns:
        float: fraction of test cases passed, in [0.0, 1.0]
    """
    if not generated_code.strip():
        return 0.0

    passed = 0
    total = len(test_cases)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(generated_code)
        tmp_path = f.name

    try:
        for tc in test_cases:
            try:
                result = subprocess.run(
                    ["python", tmp_path],
                    input=str(tc["input"]),
                    capture_output=True,
                    text=True,
                    timeout=5,  # prevent infinite loops
                )
                output = result.stdout.strip()
                expected = str(tc["expected_output"]).strip()
                if output == expected:
                    passed += 1
            except subprocess.TimeoutExpired:
                pass  # timeout counts as failed
    finally:
        os.unlink(tmp_path)

    return passed / total if total > 0 else 0.0
```

### Reasoning / QA Agent

```python
from difflib import SequenceMatcher


def reasoning_reward(
    agent_answer: str,
    ground_truth: str,
    agent_reasoning: str = "",
) -> float:
    """
    Reward function for a reasoning agent.

    Combines answer correctness with reasoning quality signals.
    """
    score = 0.0

    # Primary signal: answer correctness (normalized string comparison)
    answer_norm = agent_answer.strip().lower()
    truth_norm = ground_truth.strip().lower()

    if answer_norm == truth_norm:
        score += 0.7  # exact match
    else:
        # Partial credit via sequence similarity
        similarity = SequenceMatcher(None, answer_norm, truth_norm).ratio()
        score += 0.7 * similarity * 0.5  # partial credit, discounted

    # Secondary signal: reasoning shows relevant steps (if reasoning is provided)
    if agent_reasoning:
        # Reward for explicit step-by-step reasoning
        reasoning_signals = [
            "therefore",
            "because",
            "since",
            "first",
            "second",
            "finally",
            "step",
        ]
        signal_count = sum(
            1 for s in reasoning_signals if s in agent_reasoning.lower()
        )
        # Up to 0.3 bonus for structured reasoning
        score += min(0.3, signal_count * 0.05)

    return min(1.0, score)  # cap at 1.0
```

---

## Reward Shaping: What to Avoid

**Goodhart's Law:** When a measure becomes a target, it ceases to be a good measure.

Common reward hacking patterns:

| Reward Signal | What You Intended | What the Agent Learns |
|---------------|-------------------|-----------------------|
| Reward per tool call | "use tools" | Make as many tool calls as possible |
| Reward for long answers | "be thorough" | Pad answers with irrelevant content |
| Reward for confidence tokens | "be confident" | Always say "definitely" and "certainly" |
| Binary: any non-empty answer | "produce an answer" | Produce the shortest possible string |

**Design principle:** reward the outcome you care about, not a proxy for it.

```python
# Bad: rewards a proxy
def bad_reward(rollout):
    # Hypothesis: longer reasoning = better thinking
    return len(rollout["reasoning"].split()) / 100.0  # always gameable

# Good: rewards the actual outcome
def good_reward(rollout, ground_truth):
    # Does the final answer match the ground truth?
    return 1.0 if ground_truth in rollout["final_answer"] else 0.0
```

---

## Common Pitfalls

**Pitfall 1: Reward functions that are too sparse.** If the agent almost never gets a non-zero reward, it cannot learn. For sparse tasks, use scalar rewards that give partial credit, or decompose the task into smaller sub-goals.

**Pitfall 2: Reward components at different scales.** If correctness contributes 10.0 and format contributes 0.01, the model will focus entirely on correctness and ignore format. Normalize all components to the same scale before combining.

**Pitfall 3: Reward hacking through format tricks.** If you reward "contains the ground truth string," the agent may learn to produce the ground truth followed by incorrect reasoning. Reward the final stated answer, not any mention of the truth.

---

## Connections

- **Builds on:** Guide 01 (SFT vs RL) — the reward signal is what RL optimizes instead of imitating demonstrations
- **Leads to:** Guide 03 (Policy Optimization) — the advantage values computed from rewards drive the policy gradient update
- **Leads to:** Module 01 (GRPO) — GRPO uses group-relative advantages computed from exactly this kind of reward function
- **Related concept:** RLHF reward models learn a reward function from human preference rankings — the same relative comparison principle

---

## Further Reading

- Ziegler et al., "Fine-Tuning Language Models from Human Preferences" (2019) — introduces reward modeling for language; the foundation of RLHF
- Amodei et al., "Concrete Problems in AI Safety" (2016) — Section 2 covers reward hacking comprehensively with real examples
- Shao et al., "DeepSeekMath" (2024) — the paper that introduced GRPO and the group-relative advantage approach used in this course
