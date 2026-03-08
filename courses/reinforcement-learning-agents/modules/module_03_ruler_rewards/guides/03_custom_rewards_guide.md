# Custom and Hybrid Reward Functions

## In Brief

RULER alone is not always enough. For tasks with hard correctness requirements (SQL validity, API schema compliance, unit test passage), you need programmatic checks that catch failures RULER might overlook. Hybrid reward functions combine the binary precision of programmatic checks with the quality nuance of RULER scores. This guide covers when to use each approach and how to build robust hybrid reward functions.

## When to Use RULER Alone vs. Hybrid Rewards

The decision depends on whether your task has a checkable ground truth:

```
Task has hard correctness criterion?
        │
        ├── YES → Use hybrid reward
        │         (programmatic correctness + RULER quality)
        │
        └── NO  → Use RULER alone
                  (no single right answer, quality is the goal)
```

**RULER alone is appropriate when:**
- There is no objectively correct answer (writing, analysis, explanation)
- The task is open-ended and quality is inherently subjective
- You want the agent to discover creative approaches

**Hybrid rewards are appropriate when:**
- There is a checkable ground truth (SQL returns correct rows, code passes tests)
- Hard failures should receive zero reward regardless of quality
- You want to prevent the agent from learning "confident wrong" answers

## The Structure of a Hybrid Reward

A hybrid reward has two components that multiply rather than add:

$$r_{hybrid} = r_{programmatic} \times r_{ruler}$$

Why multiplication instead of addition?

With addition: `reward = 0 * correctness + 1.0 * quality = 1.0`
A completely wrong but beautifully written answer gets full reward.

With multiplication: `reward = 0 * 1.0 = 0`
A completely wrong answer gets zero reward, regardless of how the judge scores it.

The programmatic check acts as a gate: if the agent fails the hard requirement, RULER quality is irrelevant.

```python
def hybrid_reward(
    programmatic_score: float,  # 0.0 or 1.0 (binary pass/fail)
    ruler_score: float,         # 0.0 to 1.0 (quality judgment)
    ruler_weight: float = 0.3,  # How much quality modulates the final score
) -> float:
    """Combine programmatic correctness with RULER quality score.

    Programmatic score acts as a gate:
    - If 0.0 (hard failure): reward is 0.0 regardless of quality
    - If 1.0 (passes check): reward is modulated by RULER quality

    ruler_weight controls how much quality matters beyond basic correctness:
    - ruler_weight=0.0: binary pass/fail only, RULER ignored
    - ruler_weight=0.3: correctness matters 70%, quality matters 30%
    - ruler_weight=1.0: RULER score applied fully (correctness still gates)

    Examples:
    - SQL syntax error + high quality judgment = 0.0 * 0.9 = 0.0
    - Correct SQL + mediocre style = 1.0 * (0.7 + 0.3 * 0.4) = 0.82
    - Correct SQL + excellent style = 1.0 * (0.7 + 0.3 * 0.9) = 0.97
    """
    if programmatic_score == 0.0:
        return 0.0

    # Programmatic check passed — modulate by RULER quality
    quality_component = (1.0 - ruler_weight) + ruler_weight * ruler_score
    return programmatic_score * quality_component
```

## Code Example: Hybrid Reward for a SQL Agent

This is a complete, working hybrid reward function for a SQL agent. The programmatic checks catch hard failures; RULER scores quality among correct responses.

```python
import sqlite3
import asyncio
import re
from typing import Any
from openai import AsyncOpenAI

client = AsyncOpenAI()


def extract_sql_from_trajectory(trajectory: list[dict[str, Any]]) -> str | None:
    """Extract the final SQL query from an agent trajectory.

    Looks for SQL in the last assistant message, handling:
    - Raw SQL strings
    - Markdown code blocks: ```sql ... ```
    - Tool call arguments with a 'query' parameter

    Returns None if no SQL found.
    """
    for msg in reversed(trajectory):
        if msg.get("role") != "assistant":
            continue

        content = msg.get("content", "") or ""

        # Check for SQL in markdown code blocks
        sql_match = re.search(r"```(?:sql)?\n(.*?)```", content, re.DOTALL | re.IGNORECASE)
        if sql_match:
            return sql_match.group(1).strip()

        # Check for bare SQL (SELECT/INSERT/UPDATE/DELETE)
        sql_keywords = r"^\s*(SELECT|INSERT|UPDATE|DELETE|WITH|CREATE)\b"
        if re.match(sql_keywords, content.strip(), re.IGNORECASE):
            return content.strip()

        # Check tool call arguments
        for tc in msg.get("tool_calls", []):
            args = tc.get("function", {}).get("arguments", "")
            try:
                import json
                parsed = json.loads(args)
                if "query" in parsed:
                    return parsed["query"]
            except (json.JSONDecodeError, AttributeError):
                pass

    return None


def check_sql_validity(
    sql: str,
    db_path: str,
    expected_row_count: int | None = None,
    expected_columns: list[str] | None = None,
) -> tuple[float, str]:
    """Run programmatic checks on a SQL query.

    Args:
        sql: The SQL query to check.
        db_path: Path to the SQLite database file.
        expected_row_count: If provided, check that query returns this many rows.
        expected_columns: If provided, check that result columns match these names.

    Returns:
        Tuple of (score, reason) where score is 0.0 (failure) or 1.0 (pass).
    """
    if not sql or not sql.strip():
        return 0.0, "No SQL query found in trajectory"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check 1: Does the query parse and execute without error?
        cursor.execute(sql)
        results = cursor.fetchall()
        column_names = [description[0] for description in cursor.description or []]
        conn.close()

        # Check 2: Does it return results (not empty when we expect data)?
        if expected_row_count is not None and len(results) != expected_row_count:
            return 0.0, f"Expected {expected_row_count} rows, got {len(results)}"

        # Check 3: Do column names match expectations?
        if expected_columns is not None:
            actual_cols = set(col.lower() for col in column_names)
            expected_cols = set(col.lower() for col in expected_columns)
            if not expected_cols.issubset(actual_cols):
                missing = expected_cols - actual_cols
                return 0.0, f"Missing expected columns: {missing}"

        return 1.0, "All checks passed"

    except sqlite3.Error as e:
        return 0.0, f"SQL execution error: {e}"
    except Exception as e:
        return 0.0, f"Unexpected error: {e}"


async def sql_hybrid_reward(
    trajectory: list[dict[str, Any]],
    task_description: str,
    db_path: str,
    group_trajectories: list[list[dict[str, Any]]],
    group_ruler_scores: list[float] | None = None,
    expected_row_count: int | None = None,
    expected_columns: list[str] | None = None,
    ruler_weight: float = 0.3,
    judge_model: str = "o4-mini",
) -> tuple[float, dict[str, Any]]:
    """Compute hybrid reward for one trajectory in a SQL agent training step.

    Args:
        trajectory: The trajectory to score.
        task_description: Plain-language description of the SQL task.
        db_path: SQLite database file path.
        group_trajectories: All trajectories in the GRPO group (for RULER scoring).
        group_ruler_scores: Pre-computed RULER scores for the group (avoids re-scoring).
        expected_row_count: Optional: expected number of rows in correct result.
        expected_columns: Optional: expected column names in correct result.
        ruler_weight: Weight of RULER score in hybrid reward (0.0-1.0).
        judge_model: Judge model for RULER.

    Returns:
        Tuple of (final_reward, debug_info) where debug_info contains
        the component scores for logging.
    """
    debug = {}

    # Step 1: Extract SQL from trajectory
    sql = extract_sql_from_trajectory(trajectory)
    debug["sql"] = sql

    # Step 2: Run programmatic correctness checks
    programmatic_score, reason = check_sql_validity(
        sql or "",
        db_path,
        expected_row_count=expected_row_count,
        expected_columns=expected_columns,
    )
    debug["programmatic_score"] = programmatic_score
    debug["programmatic_reason"] = reason

    # Step 3: Get RULER quality score (only if not pre-computed)
    if group_ruler_scores is None:
        # Import here to avoid circular dependency in practice
        from guides.ruler_mechanism import ruler_score_group  # adjust import path
        group_ruler_scores = await ruler_score_group(
            group_trajectories, task_description, judge_model
        )

    # Find this trajectory's RULER score within the group
    trajectory_idx = group_trajectories.index(trajectory)
    ruler_score = group_ruler_scores[trajectory_idx]
    debug["ruler_score"] = ruler_score

    # Step 4: Compute hybrid reward
    final_reward = hybrid_reward(programmatic_score, ruler_score, ruler_weight)
    debug["final_reward"] = final_reward

    return final_reward, debug
```

## Reward Shaping for Multi-Step Tasks

For multi-step agentic tasks, a single terminal reward is often too sparse. Intermediate rewards help the agent learn which sub-decisions were good.

**The danger of intermediate rewards:** They can conflict with the terminal reward if not designed carefully. An agent that receives positive reward for searching the web might learn to search excessively, even when it already has the information needed.

**The safe pattern:** Make intermediate rewards smaller than the terminal reward, and ensure they are aligned with the terminal goal.

```python
def multi_step_reward(trajectory: list[dict], terminal_ruler_score: float) -> list[float]:
    """Assign per-step rewards across a multi-step trajectory.

    Strategy:
    - Terminal step: full RULER-scored reward
    - Intermediate steps: small process rewards for sensible tool use
    - Bad intermediate steps: small negative reward

    The terminal reward dominates (0.7 weight) to keep the agent
    goal-focused rather than process-focused.

    Args:
        trajectory: List of message dicts representing the full trajectory.
        terminal_ruler_score: RULER score for the final output quality.

    Returns:
        List of per-step rewards aligned with trajectory length.
    """
    step_rewards = []

    for i, msg in enumerate(trajectory):
        is_final = i == len(trajectory) - 1

        if is_final:
            # Terminal reward: dominant signal
            step_rewards.append(0.7 * terminal_ruler_score)

        elif msg.get("role") == "assistant" and msg.get("tool_calls"):
            # Intermediate tool call: small process reward
            tool_calls = msg["tool_calls"]
            process_score = evaluate_tool_call_quality(tool_calls)
            step_rewards.append(0.3 * process_score)

        else:
            # Non-action steps (user messages, tool results): no reward
            step_rewards.append(0.0)

    return step_rewards


def evaluate_tool_call_quality(tool_calls: list[dict]) -> float:
    """Score the quality of an intermediate tool call.

    Returns a score in [0, 1] based on:
    - Whether the tool exists in the allowed set
    - Whether arguments are syntactically valid
    - Whether this is a reasonable action (not a no-op, not excessive)

    This is intentionally simple — it catches obvious failures without
    trying to evaluate semantic quality (that's what RULER is for).
    """
    ALLOWED_TOOLS = {"search_web", "query_database", "calculate", "read_document"}

    if not tool_calls:
        return 0.0

    scores = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        tool_name = fn.get("name", "")

        # Check: tool exists in allowed set
        if tool_name not in ALLOWED_TOOLS:
            scores.append(0.0)
            continue

        # Check: arguments are valid JSON
        try:
            import json
            args = json.loads(fn.get("arguments", "{}"))
            if isinstance(args, dict) and args:
                scores.append(1.0)
            else:
                scores.append(0.5)  # Valid JSON but empty or not a dict
        except json.JSONDecodeError:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0
```

## Testing and Validating Reward Functions

A reward function that produces wrong signals is worse than no reward function — it actively teaches bad behavior. Test reward functions before any training run.

```python
import pytest
from dataclasses import dataclass


@dataclass
class RewardTestCase:
    """A single test case for reward function validation."""
    name: str
    trajectory: list[dict]
    expected_score_range: tuple[float, float]  # (min, max) acceptable score
    description: str


# Build test cases that cover the full range of agent behaviors
SQL_REWARD_TEST_CASES = [
    RewardTestCase(
        name="perfect_query",
        trajectory=[
            {"role": "user", "content": "Find the top 5 customers by revenue"},
            {"role": "assistant", "content": "```sql\nSELECT customer_id, SUM(revenue) as total_revenue\nFROM orders\nGROUP BY customer_id\nORDER BY total_revenue DESC\nLIMIT 5\n```"},
        ],
        expected_score_range=(0.75, 1.0),
        description="Correct, efficient, readable SQL should score high",
    ),
    RewardTestCase(
        name="syntax_error",
        trajectory=[
            {"role": "user", "content": "Find the top 5 customers by revenue"},
            {"role": "assistant", "content": "```sql\nSELECT customer_id FORM orders LIMIT 5\n```"},
        ],
        expected_score_range=(0.0, 0.0),
        description="Syntax error should score exactly 0 (programmatic gate)",
    ),
    RewardTestCase(
        name="correct_but_inefficient",
        trajectory=[
            {"role": "user", "content": "Find the top 5 customers by revenue"},
            {"role": "assistant", "content": "```sql\nSELECT * FROM orders\n```"},
        ],
        expected_score_range=(0.0, 0.5),
        description="Returns correct rows but wildly inefficient — low score",
    ),
    RewardTestCase(
        name="empty_response",
        trajectory=[
            {"role": "user", "content": "Find the top 5 customers by revenue"},
            {"role": "assistant", "content": "I cannot help with that."},
        ],
        expected_score_range=(0.0, 0.0),
        description="Refusal with no SQL should score 0",
    ),
]


def test_reward_function_on_cases(
    db_path: str,
    test_cases: list[RewardTestCase] | None = None,
) -> dict[str, bool]:
    """Run reward function test cases and report results.

    Returns dict mapping test case name to pass/fail.
    """
    if test_cases is None:
        test_cases = SQL_REWARD_TEST_CASES

    results = {}

    for case in test_cases:
        sql = extract_sql_from_trajectory(case.trajectory)
        score, reason = check_sql_validity(sql or "", db_path)

        # For testing without a real RULER call, use programmatic score only
        actual_score = score
        min_expected, max_expected = case.expected_score_range

        passed = min_expected <= actual_score <= max_expected
        results[case.name] = passed

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {case.name}: score={actual_score:.2f} (expected {min_expected:.2f}-{max_expected:.2f})")
        if not passed:
            print(f"       Reason: {reason}")
            print(f"       Description: {case.description}")

    return results


def run_reward_validation(db_path: str) -> bool:
    """Run full reward validation suite. Returns True if all tests pass."""
    results = test_reward_function_on_cases(db_path)
    all_passed = all(results.values())

    passed = sum(results.values())
    total = len(results)
    print(f"\nReward validation: {passed}/{total} tests passed")

    if not all_passed:
        failed = [name for name, ok in results.items() if not ok]
        print(f"Failed tests: {failed}")
        print("Fix reward function before running training.")

    return all_passed
```

## Decision Guide: Choosing Your Reward Strategy

| Situation | Recommended Approach |
|-----------|---------------------|
| Open-ended task (writing, analysis) | RULER alone |
| Hard correctness criterion exists | Hybrid (programmatic + RULER) |
| Multi-step task with identifiable sub-goals | Multi-step shaping + terminal RULER |
| Very fast training loop needed | Programmatic only (no LLM call) |
| Binary pass/fail sufficient | Programmatic only |
| Quality matters among correct responses | Hybrid with high ruler_weight |

Start simple. Programmatic-only rewards are fast to implement and debug. Add RULER when you observe that all-correct responses vary in quality in ways that matter for your use case.

## Common Pitfalls in Hybrid Design

**Pitfall 1: Adding rewards instead of gating**
`reward = 0.5 * programmatic + 0.5 * ruler` allows a wrong answer to get 0.5 reward just from RULER quality. Use multiplication to gate.

**Pitfall 2: Too many intermediate rewards**
More than 2-3 intermediate reward signals creates conflicting gradients. The agent tries to optimize all simultaneously and often converges to none. Design intermediate rewards conservatively.

**Pitfall 3: Intermediate rewards that dominate the terminal reward**
If intermediate rewards sum to more than the terminal reward, the agent focuses on the process and ignores the final output quality.

**Pitfall 4: Not testing reward functions before training**
A reward function that returns wrong values will teach wrong behaviors silently. The model will appear to train (loss goes down) while learning to do the wrong thing. Always run test cases first.

**Pitfall 5: Reward functions that are too slow**
Hybrid rewards call both a database (programmatic) and an LLM (RULER). If either is slow, the training loop bottlenecks. Target under 5 seconds per training step for the reward computation.

## Connections

- **Builds on:** Guide 01 (reward hacking), Guide 02 (RULER scoring mechanism)
- **Leads to:** Module 05 (Training Loop — integrating rewards into the full pipeline)
- **Related to:** Process Reward Models (PRMs), dense reward shaping in robotics RL

## Practice Questions

1. A customer service agent is being trained to resolve tickets. Design a hybrid reward function: what programmatic checks would you run, and what would you ask the judge LLM to evaluate?

2. Why does `reward = programmatic * ruler_score` behave correctly when the SQL has a syntax error, but `reward = 0.5 * programmatic + 0.5 * ruler_score` does not?

3. You add an intermediate reward for each web search call. During training, the agent starts making 15+ search calls per task. What went wrong and how do you fix it?

## Further Reading

- Ng, Russell, "Policy Invariance Under Reward Transformations" (1999) — foundational theory of reward shaping
- "Let's Reward Step by Step" (Lightman et al., 2023) — process reward models for multi-step reasoning
- "RLHF is Not RLHF" (Lambert et al., 2023) — practical differences between reward model training and RULER-style approaches
- OpenPipe ART GitHub — reference implementations of hybrid reward functions for text-to-SQL tasks
