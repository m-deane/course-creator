"""
Module 03 Exercise: RULER Automatic Rewards
============================================

This exercise has three parts:

  Part 1: Implement a simple LLM-as-a-judge scoring function
  Part 2: Compare absolute vs relative scoring on sample data
  Part 3: Build a hybrid reward combining programmatic + RULER scores

Prerequisites:
  pip install openai scipy

Set your API key:
  export OPENAI_API_KEY=sk-...

Estimated time: 30-45 minutes

Run the exercise:
  python 01_ruler_exercise.py

All tests are self-checking. Green output means you got it right.
"""

import asyncio
import json
import re
import sqlite3
import tempfile
import os
from typing import Any

from openai import AsyncOpenAI

client = AsyncOpenAI()


# ---------------------------------------------------------------------------
# Sample data — 4 agent trajectories for the same SQL task
# These represent what a real training step would produce.
# ---------------------------------------------------------------------------

TASK_DESCRIPTION = "Find the total revenue per product category, sorted from highest to lowest."

# Four agent responses of varying quality — we will score these throughout the exercise.
SAMPLE_TRAJECTORIES: list[list[dict[str, Any]]] = [
    # Trajectory 0: Correct, efficient, readable
    [
        {"role": "user", "content": TASK_DESCRIPTION},
        {
            "role": "assistant",
            "content": (
                "I'll query the orders and products tables to aggregate revenue by category.\n\n"
                "```sql\n"
                "SELECT p.category, SUM(o.quantity * o.unit_price) AS total_revenue\n"
                "FROM orders o\n"
                "JOIN products p ON o.product_id = p.id\n"
                "GROUP BY p.category\n"
                "ORDER BY total_revenue DESC\n"
                "```"
            ),
        },
    ],
    # Trajectory 1: Correct but inefficient (SELECT * instead of needed columns)
    [
        {"role": "user", "content": TASK_DESCRIPTION},
        {
            "role": "assistant",
            "content": (
                "Let me look at the data.\n\n"
                "```sql\n"
                "SELECT * FROM orders o, products p\n"
                "WHERE o.product_id = p.id\n"
                "```"
            ),
        },
    ],
    # Trajectory 2: Syntax error — cannot execute
    [
        {"role": "user", "content": TASK_DESCRIPTION},
        {
            "role": "assistant",
            "content": (
                "```sql\n"
                "SELECT category, SUM(revenue) FORM orders\n"
                "GROUP category\n"
                "ORDER revenue DESC\n"
                "```"
            ),
        },
    ],
    # Trajectory 3: Partial — correct approach but missing the JOIN
    [
        {"role": "user", "content": TASK_DESCRIPTION},
        {
            "role": "assistant",
            "content": (
                "```sql\n"
                "SELECT product_id, SUM(quantity * unit_price) AS total_revenue\n"
                "FROM orders\n"
                "GROUP BY product_id\n"
                "ORDER BY total_revenue DESC\n"
                "```"
            ),
        },
    ],
]

# The human-labeled ranking (best to worst): 0 > 3 > 1 > 2
# Trajectory 0 is ideal; trajectory 2 has a syntax error (worst).
HUMAN_RANKING_BEST_TO_WORST = [0, 3, 1, 2]


# ---------------------------------------------------------------------------
# Helpers — these are provided for you
# ---------------------------------------------------------------------------

def extract_sql(trajectory: list[dict[str, Any]]) -> str | None:
    """Extract SQL from the last assistant message in a trajectory."""
    for msg in reversed(trajectory):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "") or ""
        match = re.search(r"```(?:sql)?\n(.*?)```", content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        if re.match(r"^\s*(SELECT|INSERT|UPDATE|DELETE|WITH)\b", content.strip(), re.IGNORECASE):
            return content.strip()
    return None


def make_test_database() -> str:
    """Create a temporary SQLite database with sample data. Returns the file path."""
    db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = db_file.name
    db_file.close()

    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL
        );
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            product_id INTEGER REFERENCES products(id),
            quantity INTEGER,
            unit_price REAL
        );
        INSERT INTO products VALUES (1, 'Widget A', 'Electronics');
        INSERT INTO products VALUES (2, 'Widget B', 'Electronics');
        INSERT INTO products VALUES (3, 'Gadget X', 'Tools');
        INSERT INTO products VALUES (4, 'Part Y', 'Tools');
        INSERT INTO products VALUES (5, 'Book Z', 'Education');
        INSERT INTO orders VALUES (1, 1, 10, 29.99);
        INSERT INTO orders VALUES (2, 2, 5, 49.99);
        INSERT INTO orders VALUES (3, 3, 20, 9.99);
        INSERT INTO orders VALUES (4, 4, 8, 14.99);
        INSERT INTO orders VALUES (5, 5, 3, 39.99);
        INSERT INTO orders VALUES (6, 1, 7, 29.99);
        INSERT INTO orders VALUES (7, 2, 12, 49.99);
    """)
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# PART 1: Implement LLM-as-a-Judge scoring
# ---------------------------------------------------------------------------
#
# Task: Complete the `absolute_score_single` and `relative_score_group` functions.
#
# `absolute_score_single` asks the judge to rate ONE trajectory from 0.0 to 1.0.
# `relative_score_group` asks the judge to rate ALL trajectories simultaneously.
#
# Both functions should:
# 1. Format the trajectory/trajectories for the judge
# 2. Call the OpenAI API with temperature=0.0
# 3. Parse the JSON response and return scores as floats
# 4. Handle errors gracefully (return 0.5 for absolute, [0.5]*n for relative)
#
# The judge system prompts are provided — do not change them.
# ---------------------------------------------------------------------------

ABSOLUTE_JUDGE_PROMPT = """You are an expert SQL evaluator.

Rate the quality of this agent's SQL query response from 0.0 to 1.0:

- 1.0 = Perfect: correct result, efficient, readable, handles edge cases
- 0.75 = Good: correct or near-correct with minor issues
- 0.5 = Mediocre: partially correct or significant issues
- 0.25 = Poor: mostly incorrect but shows understanding
- 0.0 = Failed: syntax error, crashes, or refusal

Respond with ONLY this JSON: {"score": <float>}
"""

RELATIVE_JUDGE_PROMPT = """You are an expert SQL evaluator.

Rate the quality of each agent trajectory from 0.0 to 1.0:

- 1.0 = Perfect: correct result, efficient, readable
- 0.75 = Good: correct or near-correct with minor issues
- 0.5 = Mediocre: partially correct or significant issues
- 0.25 = Poor: mostly incorrect
- 0.0 = Failed: syntax error, crashes, or refusal

IMPORTANT: Assign scores RELATIVE to each other. The best trajectory must
receive the highest score. Do NOT assign all trajectories the same score.

Respond with ONLY this JSON: {"trajectory_0": <score>, "trajectory_1": <score>, ...}
"""


async def absolute_score_single(
    trajectory: list[dict[str, Any]],
    task_description: str,
    judge_model: str = "gpt-4o-mini",
) -> float:
    """Score a single trajectory using absolute (non-comparative) scoring.

    The judge sees ONLY this trajectory, not the others.

    Args:
        trajectory: Single agent trajectory to score.
        task_description: What the agent was trying to do.
        judge_model: OpenAI model to use as judge.

    Returns:
        Float score in [0.0, 1.0]. Returns 0.5 on any error.
    """
    # TODO: Implement this function.
    #
    # Steps:
    # 1. Build a user message that shows the task and the trajectory contents.
    #    Format: "TASK: {task_description}\n\nAGENT RESPONSE:\n{message contents}"
    # 2. Call client.chat.completions.create with:
    #    - model=judge_model
    #    - messages=[system prompt, user message]
    #    - response_format={"type": "json_object"}
    #    - temperature=0.0
    # 3. Parse the JSON response and extract the "score" field.
    # 4. Clamp the score to [0.0, 1.0] with max(0.0, min(1.0, score)).
    # 5. Return 0.5 on json.JSONDecodeError, KeyError, or ValueError.
    raise NotImplementedError("Implement absolute_score_single")


async def relative_score_group(
    trajectories: list[list[dict[str, Any]]],
    task_description: str,
    judge_model: str = "gpt-4o-mini",
) -> list[float]:
    """Score a group of trajectories using relative (comparative) scoring.

    The judge sees ALL trajectories simultaneously and scores them relative
    to each other.

    Args:
        trajectories: List of N trajectories to score together.
        task_description: What the agent was trying to do.
        judge_model: OpenAI model to use as judge.

    Returns:
        List of N floats in [0.0, 1.0]. Returns [0.5]*N on any error.
    """
    # TODO: Implement this function.
    #
    # Steps:
    # 1. Build a user message that shows the task followed by ALL trajectories,
    #    each labeled "--- TRAJECTORY {i} ---" with their message contents below.
    # 2. Call client.chat.completions.create with:
    #    - model=judge_model
    #    - messages=[system prompt, user message]
    #    - response_format={"type": "json_object"}
    #    - temperature=0.0
    # 3. Parse the JSON response: {"trajectory_0": 0.8, "trajectory_1": 0.3, ...}
    # 4. Build a list of scores in order, defaulting to 0.5 for any missing key.
    # 5. Clamp each score to [0.0, 1.0].
    # 6. Return [0.5] * len(trajectories) on any error.
    raise NotImplementedError("Implement relative_score_group")


# ---------------------------------------------------------------------------
# PART 2: Compare absolute vs relative scoring
# ---------------------------------------------------------------------------
#
# Task: Complete `compare_scoring_methods` to call both scoring functions
# and compute how well each agrees with the human ranking.
#
# We use Spearman rank correlation to measure agreement.
# 1.0 = perfect agreement with human ranking.
# 0.0 = no correlation.
# -1.0 = perfectly reversed.
# ---------------------------------------------------------------------------

def spearman_rank_correlation(scores: list[float], human_ranking: list[int]) -> float:
    """Compute Spearman rank correlation between scores and human ranking.

    Args:
        scores: Model scores, one per trajectory (higher = better).
        human_ranking: Human-labeled list of trajectory indices from best to worst.
                       e.g., [0, 3, 1, 2] means trajectory 0 is best.

    Returns:
        Spearman rho in [-1, 1]. 1.0 = perfect agreement.
    """
    n = len(scores)
    assert n == len(human_ranking), "scores and human_ranking must have same length"

    # Convert human ranking to per-trajectory rank (lower rank = better)
    human_rank = [0] * n
    for rank, idx in enumerate(human_ranking):
        human_rank[idx] = rank

    # Convert model scores to ranks (higher score = lower rank = better)
    sorted_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)
    model_rank = [0] * n
    for rank, idx in enumerate(sorted_indices):
        model_rank[idx] = rank

    # Compute Spearman rho = 1 - (6 * sum(d^2)) / (n * (n^2 - 1))
    d_squared_sum = sum((human_rank[i] - model_rank[i]) ** 2 for i in range(n))
    rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    return rho


async def compare_scoring_methods(
    trajectories: list[list[dict[str, Any]]],
    task_description: str,
    human_ranking: list[int],
    judge_model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Compare absolute vs relative scoring on the same set of trajectories.

    Args:
        trajectories: List of trajectories to score.
        task_description: What the agent was trying to do.
        human_ranking: Ground-truth human ranking (best to worst indices).
        judge_model: Judge model to use.

    Returns:
        Dict with keys:
        - 'absolute_scores': list of scores from absolute scoring
        - 'relative_scores': list of scores from relative scoring
        - 'absolute_rank_correlation': Spearman rho for absolute scoring
        - 'relative_rank_correlation': Spearman rho for relative scoring
        - 'absolute_score_range': max - min of absolute scores (spread)
        - 'relative_score_range': max - min of relative scores (spread)
    """
    # TODO: Implement this function.
    #
    # Steps:
    # 1. Score each trajectory individually using absolute_score_single.
    #    Build a list of scores in order.
    # 2. Score all trajectories together using relative_score_group.
    # 3. Compute rank correlation for each method using spearman_rank_correlation.
    # 4. Compute score range (max - min) for each method.
    # 5. Return all results in the dict described above.
    #
    # Hint: use asyncio.gather to run absolute scores in parallel.
    raise NotImplementedError("Implement compare_scoring_methods")


# ---------------------------------------------------------------------------
# PART 3: Build a hybrid reward function
# ---------------------------------------------------------------------------
#
# Task: Complete `sql_hybrid_reward_for_trajectory` which combines:
# 1. A programmatic check (does the SQL execute against the test DB?)
# 2. A RULER quality score (from relative_score_group)
#
# The final reward should use the gating pattern:
#   if programmatic fails → reward = 0.0
#   if programmatic passes → reward = (1 - ruler_weight) + ruler_weight * ruler_score
# ---------------------------------------------------------------------------

def check_sql_executes(sql: str, db_path: str) -> tuple[float, str]:
    """Check if a SQL query executes without error.

    Args:
        sql: SQL query string.
        db_path: Path to SQLite database.

    Returns:
        Tuple of (score, reason). Score is 1.0 if executes, 0.0 if fails.
    """
    if not sql or not sql.strip():
        return 0.0, "No SQL query found"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        cursor.fetchall()
        conn.close()
        return 1.0, "SQL executed successfully"
    except sqlite3.Error as e:
        return 0.0, f"SQL error: {e}"
    except Exception as e:
        return 0.0, f"Unexpected error: {e}"


def hybrid_reward(
    programmatic_score: float,
    ruler_score: float,
    ruler_weight: float = 0.3,
) -> float:
    """Combine programmatic and RULER scores using the gating pattern.

    Args:
        programmatic_score: 0.0 (fail) or 1.0 (pass).
        ruler_score: Quality score in [0.0, 1.0] from RULER.
        ruler_weight: Weight for RULER quality (0.0-1.0).
                      0.0 = ignore RULER, 1.0 = full RULER weight.

    Returns:
        Final reward in [0.0, 1.0].
    """
    # TODO: Implement the hybrid gating pattern.
    #
    # If programmatic_score == 0.0, return 0.0 immediately.
    # Otherwise: return programmatic_score * ((1 - ruler_weight) + ruler_weight * ruler_score)
    raise NotImplementedError("Implement hybrid_reward")


async def compute_group_hybrid_rewards(
    trajectories: list[list[dict[str, Any]]],
    task_description: str,
    db_path: str,
    ruler_weight: float = 0.3,
    judge_model: str = "gpt-4o-mini",
) -> list[tuple[float, dict[str, Any]]]:
    """Compute hybrid rewards for all trajectories in a GRPO group.

    Runs RULER once for the whole group (efficient), then combines with
    per-trajectory programmatic checks.

    Args:
        trajectories: List of N trajectories.
        task_description: What the agent was trying to do.
        db_path: SQLite database file path.
        ruler_weight: Weight for RULER quality in hybrid reward.
        judge_model: Judge model for RULER scoring.

    Returns:
        List of N tuples (final_reward, debug_info).
        debug_info contains: programmatic_score, ruler_score, final_reward, sql.
    """
    # TODO: Implement this function.
    #
    # Steps:
    # 1. Run relative_score_group ONCE for all trajectories (get ruler_scores list).
    # 2. For each trajectory:
    #    a. Extract SQL using extract_sql(trajectory)
    #    b. Run check_sql_executes(sql, db_path) to get programmatic score
    #    c. Get this trajectory's ruler_score from the list (by index)
    #    d. Compute final_reward using hybrid_reward(programmatic, ruler, ruler_weight)
    #    e. Build debug_info dict with: sql, programmatic_score, ruler_score, final_reward
    # 3. Return list of (final_reward, debug_info) tuples.
    raise NotImplementedError("Implement compute_group_hybrid_rewards")


# ---------------------------------------------------------------------------
# Self-checking tests — run automatically when you execute this file
# ---------------------------------------------------------------------------

def _print_separator(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _check(condition: bool, passed_msg: str, failed_msg: str) -> bool:
    if condition:
        print(f"  [PASS] {passed_msg}")
        return True
    else:
        print(f"  [FAIL] {failed_msg}")
        return False


async def test_part1_absolute_scoring() -> bool:
    """Test that absolute_score_single returns valid scores."""
    _print_separator("Part 1a: Absolute Scoring")

    results = await asyncio.gather(*[
        absolute_score_single(traj, TASK_DESCRIPTION)
        for traj in SAMPLE_TRAJECTORIES
    ])

    all_passed = True

    # Each score must be in [0, 1]
    for i, score in enumerate(results):
        ok = _check(
            0.0 <= score <= 1.0,
            f"Trajectory {i}: score={score:.3f} in valid range [0, 1]",
            f"Trajectory {i}: score={score:.3f} is OUTSIDE [0, 1]",
        )
        all_passed = all_passed and ok

    # Trajectory 2 has a syntax error — it should score 0 or very low
    syntax_error_score = results[2]
    ok = _check(
        syntax_error_score <= 0.4,
        f"Syntax error trajectory scored low: {syntax_error_score:.3f} <= 0.4",
        f"Syntax error trajectory scored too high: {syntax_error_score:.3f} > 0.4 (should be low)",
    )
    all_passed = all_passed and ok

    print(f"\n  Absolute scores: {[f'{s:.3f}' for s in results]}")
    return all_passed


async def test_part1_relative_scoring() -> tuple[bool, list[float]]:
    """Test that relative_score_group returns valid, well-spread scores."""
    _print_separator("Part 1b: Relative Scoring")

    scores = await relative_score_group(SAMPLE_TRAJECTORIES, TASK_DESCRIPTION)

    all_passed = True

    # Must return 4 scores
    ok = _check(
        len(scores) == 4,
        f"Returned {len(scores)} scores (expected 4)",
        f"Returned {len(scores)} scores (expected 4)",
    )
    all_passed = all_passed and ok

    # Each score in [0, 1]
    for i, score in enumerate(scores):
        ok = _check(
            0.0 <= score <= 1.0,
            f"Trajectory {i}: score={score:.3f} in [0, 1]",
            f"Trajectory {i}: score={score:.3f} OUTSIDE [0, 1]",
        )
        all_passed = all_passed and ok

    # Scores must not all be identical (anti-clustering)
    score_range = max(scores) - min(scores)
    ok = _check(
        score_range >= 0.15,
        f"Score range {score_range:.3f} >= 0.15 (scores are differentiated)",
        f"Score range {score_range:.3f} < 0.15 (all scores too similar — check relative scoring instruction)",
    )
    all_passed = all_passed and ok

    # Trajectory 0 (best) should score higher than trajectory 2 (syntax error)
    ok = _check(
        scores[0] > scores[2],
        f"Best trajectory (0) scores higher than syntax-error trajectory (2): {scores[0]:.3f} > {scores[2]:.3f}",
        f"Best trajectory (0) did NOT score higher than syntax-error trajectory (2): {scores[0]:.3f} vs {scores[2]:.3f}",
    )
    all_passed = all_passed and ok

    print(f"\n  Relative scores: {[f'{s:.3f}' for s in scores]}")
    return all_passed, scores


async def test_part2_comparison(relative_scores: list[float]) -> bool:
    """Test that relative scoring agrees better with human ranking than absolute."""
    _print_separator("Part 2: Absolute vs Relative Scoring Comparison")

    results = await compare_scoring_methods(
        SAMPLE_TRAJECTORIES,
        TASK_DESCRIPTION,
        HUMAN_RANKING_BEST_TO_WORST,
    )

    all_passed = True

    required_keys = [
        "absolute_scores", "relative_scores",
        "absolute_rank_correlation", "relative_rank_correlation",
        "absolute_score_range", "relative_score_range",
    ]
    for key in required_keys:
        ok = _check(
            key in results,
            f"Result contains '{key}'",
            f"Result missing '{key}'",
        )
        all_passed = all_passed and ok

    if not all_passed:
        return False

    abs_corr = results["absolute_rank_correlation"]
    rel_corr = results["relative_rank_correlation"]
    abs_range = results["absolute_score_range"]
    rel_range = results["relative_score_range"]

    print(f"\n  Absolute scoring — rank correlation: {abs_corr:.3f}, score range: {abs_range:.3f}")
    print(f"  Relative scoring — rank correlation: {rel_corr:.3f}, score range: {rel_range:.3f}")

    # Relative scoring should have better rank correlation with human judgment
    ok = _check(
        rel_corr >= abs_corr - 0.1,  # Allow small tolerance for LLM variability
        f"Relative scoring rank correlation ({rel_corr:.3f}) >= absolute ({abs_corr:.3f})",
        f"Absolute scoring rank correlation ({abs_corr:.3f}) significantly beats relative ({rel_corr:.3f}) — check your relative scoring prompt",
    )
    all_passed = all_passed and ok

    # Relative scoring should produce more spread (wider range)
    ok = _check(
        rel_range >= abs_range - 0.05,
        f"Relative scoring spread ({rel_range:.3f}) >= absolute spread ({abs_range:.3f})",
        f"Relative scoring spread ({rel_range:.3f}) much smaller than absolute ({abs_range:.3f}) — relative scoring should differentiate more",
    )
    all_passed = all_passed and ok

    return all_passed


async def test_part3_hybrid_reward() -> bool:
    """Test hybrid reward function correctness."""
    _print_separator("Part 3: Hybrid Reward Function")

    all_passed = True

    # Test 1: Programmatic failure → reward = 0 regardless of RULER score
    reward = hybrid_reward(programmatic_score=0.0, ruler_score=0.9, ruler_weight=0.3)
    ok = _check(
        reward == 0.0,
        f"Programmatic fail + high RULER → reward=0.0 (got {reward:.3f})",
        f"Programmatic fail + high RULER → reward should be 0.0, got {reward:.3f}",
    )
    all_passed = all_passed and ok

    # Test 2: Programmatic pass + high quality → reward close to 1.0
    reward = hybrid_reward(programmatic_score=1.0, ruler_score=0.9, ruler_weight=0.3)
    ok = _check(
        0.95 <= reward <= 1.0,
        f"Programmatic pass + high RULER → reward={reward:.3f} near 1.0",
        f"Programmatic pass + high RULER → reward={reward:.3f} should be near 1.0",
    )
    all_passed = all_passed and ok

    # Test 3: Programmatic pass + low quality → reward in middle range
    reward = hybrid_reward(programmatic_score=1.0, ruler_score=0.1, ruler_weight=0.3)
    ok = _check(
        0.70 <= reward <= 0.80,
        f"Programmatic pass + low RULER → reward={reward:.3f} in [0.70, 0.80]",
        f"Programmatic pass + low RULER → reward={reward:.3f} should be in [0.70, 0.80]",
    )
    all_passed = all_passed and ok

    # Test 4: ruler_weight=0 → programmatic only (RULER ignored)
    reward_high = hybrid_reward(programmatic_score=1.0, ruler_score=1.0, ruler_weight=0.0)
    reward_low = hybrid_reward(programmatic_score=1.0, ruler_score=0.0, ruler_weight=0.0)
    ok = _check(
        reward_high == reward_low == 1.0,
        f"ruler_weight=0 → RULER ignored: both rewards = 1.0",
        f"ruler_weight=0 → RULER should be ignored, got high={reward_high:.3f}, low={reward_low:.3f}",
    )
    all_passed = all_passed and ok

    # Test 5: Full hybrid on the sample group
    db_path = make_test_database()
    try:
        group_results = await compute_group_hybrid_rewards(
            SAMPLE_TRAJECTORIES,
            TASK_DESCRIPTION,
            db_path,
        )

        ok = _check(
            len(group_results) == 4,
            f"compute_group_hybrid_rewards returned {len(group_results)} results",
            f"compute_group_hybrid_rewards should return 4 results, got {len(group_results)}",
        )
        all_passed = all_passed and ok

        rewards = [r for r, _ in group_results]
        debug_infos = [d for _, d in group_results]

        # Trajectory 2 has syntax error → reward must be 0
        ok = _check(
            rewards[2] == 0.0,
            f"Syntax-error trajectory reward=0.0 (programmatic gate works)",
            f"Syntax-error trajectory reward={rewards[2]:.3f} should be 0.0 (programmatic gate failed)",
        )
        all_passed = all_passed and ok

        # Trajectory 0 (best) should have higher reward than trajectory 2 (error)
        ok = _check(
            rewards[0] > rewards[2],
            f"Best trajectory reward ({rewards[0]:.3f}) > syntax-error reward ({rewards[2]:.3f})",
            f"Best trajectory reward ({rewards[0]:.3f}) should exceed syntax-error ({rewards[2]:.3f})",
        )
        all_passed = all_passed and ok

        # Debug info must contain required fields
        required_debug = {"sql", "programmatic_score", "ruler_score", "final_reward"}
        for i, debug in enumerate(debug_infos):
            ok = _check(
                required_debug.issubset(debug.keys()),
                f"Trajectory {i} debug_info contains all required fields",
                f"Trajectory {i} debug_info missing fields: {required_debug - set(debug.keys())}",
            )
            all_passed = all_passed and ok

        print("\n  Hybrid rewards per trajectory:")
        for i, (reward, debug) in enumerate(group_results):
            print(f"    [{i}] prog={debug['programmatic_score']:.1f} | "
                  f"ruler={debug['ruler_score']:.3f} | "
                  f"hybrid={reward:.3f}")

    finally:
        os.unlink(db_path)

    return all_passed


async def run_all_tests() -> None:
    """Run all exercise tests and print a summary."""
    print("\nModule 03 Exercise: RULER Automatic Rewards")
    print("=" * 60)
    print("Running tests... (this makes several LLM API calls)")

    results = {}

    try:
        results["part1_absolute"] = await test_part1_absolute_scoring()
    except NotImplementedError as e:
        print(f"\n  [SKIP] Part 1a not implemented yet: {e}")
        results["part1_absolute"] = False

    relative_scores = None
    try:
        passed, relative_scores = await test_part1_relative_scoring()
        results["part1_relative"] = passed
    except NotImplementedError as e:
        print(f"\n  [SKIP] Part 1b not implemented yet: {e}")
        results["part1_relative"] = False

    try:
        results["part2_comparison"] = await test_part2_comparison(relative_scores or [])
    except NotImplementedError as e:
        print(f"\n  [SKIP] Part 2 not implemented yet: {e}")
        results["part2_comparison"] = False

    try:
        results["part3_hybrid"] = await test_part3_hybrid_reward()
    except NotImplementedError as e:
        print(f"\n  [SKIP] Part 3 not implemented yet: {e}")
        results["part3_hybrid"] = False

    # Summary
    _print_separator("Summary")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        all_passed = all_passed and passed

    if all_passed:
        print("\n  All tests passed. You are ready for Module 04.")
    else:
        failed = [n for n, p in results.items() if not p]
        print(f"\n  Tests still failing: {failed}")
        print("  Review the TODO comments above each function and try again.")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
