"""
Module 07 — Production Considerations
Exercise 01: Cost Calculator, Benchmark Harness, and Checkpoint Comparison

This exercise has three parts:

  Part 1: Implement a training cost calculator
  Part 2: Build a benchmark evaluation harness
  Part 3: Compare model performance across checkpoints

Run this file to execute all self-checks:
    python exercises/01_production_exercise.py

All three parts have automated checks that print PASS or FAIL with explanations.
"""

import json
import time
import statistics
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable


# ===========================================================================
# PART 1: Training Cost Calculator
# ===========================================================================
# Implement the functions below. Each function has a docstring describing
# what it should do and what it should return.
#
# The checks at the bottom of Part 1 will verify your implementation.
# ===========================================================================


def estimate_lora_training_hours(
    model_size_b: float,
    training_steps: int,
    gpu_type: str = "a100",
) -> float:
    """
    Estimate the wall-clock hours for a LoRA training run.

    Use these throughput estimates (steps per hour, LoRA):
        "a100":   model_size_b * 2 steps per hour
        "h100":   model_size_b * 3 steps per hour
        "a10g":   model_size_b * 0.8 steps per hour
        "rtx4090": model_size_b * 1.2 steps per hour

    So for a 14B model on an A100:
        steps_per_hour = 14 * 2 = 28
        hours = training_steps / 28

    Args:
        model_size_b: Model size in billions of parameters (e.g. 14 for 14B)
        training_steps: Number of gradient update steps
        gpu_type: One of "a100", "h100", "a10g", "rtx4090"

    Returns:
        Estimated training hours as a float, rounded to 2 decimal places.

    Raises:
        ValueError: If gpu_type is not one of the supported types.
    """
    # YOUR CODE HERE
    pass


def estimate_training_cost_usd(
    training_hours: float,
    gpu_hourly_rate: float,
    num_checkpoints: int,
    use_lora: bool = True,
) -> dict:
    """
    Estimate total training cost, broken down by component.

    Cost components:
        1. GPU rental:   training_hours * gpu_hourly_rate
        2. Storage:      num_checkpoints * checkpoint_size_gb * storage_rate_per_gb
                         LoRA checkpoint: 0.5 GB
                         Full fine-tune checkpoint: model_size_gb (use 28 GB as default)
        3. Validation:   num_checkpoints * 500 * cost_per_validation_query
                         cost_per_validation_query = 0.000085 USD

    Use storage_rate_per_gb = 0.023 (AWS S3 monthly rate).
    For simplicity, treat storage cost as one month regardless of actual duration.

    Args:
        training_hours: Output of estimate_lora_training_hours()
        gpu_hourly_rate: USD per hour for the GPU (e.g. 1.29 for Lambda A100)
        num_checkpoints: Number of checkpoints saved during training
        use_lora: True for LoRA (0.5GB checkpoints), False for full (28GB)

    Returns:
        Dictionary with keys:
            "gpu_cost_usd":        float
            "storage_cost_usd":    float
            "validation_cost_usd": float
            "total_cost_usd":      float

        All values rounded to 2 decimal places.
    """
    # YOUR CODE HERE
    pass


def compute_break_even_queries(
    training_cost_usd: float,
    custom_cost_per_query: float,
    frontier_cost_per_query: float,
) -> int:
    """
    Compute how many queries until the custom model is cheaper than the frontier API.

    Break-even formula:
        break_even = training_cost / (frontier_cost_per_query - custom_cost_per_query)

    If frontier_cost_per_query <= custom_cost_per_query, the custom model never
    breaks even. Return -1 in this case.

    Args:
        training_cost_usd: One-time training cost in USD
        custom_cost_per_query: Cost per query for your deployed model (USD)
        frontier_cost_per_query: Cost per query for the frontier API (USD)

    Returns:
        Number of queries to break even (integer, rounded up), or -1 if break-even
        is not achievable.
    """
    # YOUR CODE HERE
    pass


# ===========================================================================
# Part 1 Self-Checks
# ===========================================================================

def check_part1():
    print("=" * 60)
    print("PART 1: Training Cost Calculator")
    print("=" * 60)
    all_passed = True

    # Check 1: estimate_lora_training_hours — A100, 14B model, 500 steps
    # Expected: 500 / (14 * 2) = 500 / 28 = 17.86 hours
    result = estimate_lora_training_hours(14.0, 500, "a100")
    expected = round(500 / (14 * 2), 2)
    if result is None:
        print("FAIL [1.1] estimate_lora_training_hours returned None — implement the function")
        all_passed = False
    elif not isinstance(result, float):
        print(f"FAIL [1.1] Expected float, got {type(result).__name__}")
        all_passed = False
    elif abs(result - expected) > 0.1:
        print(f"FAIL [1.1] For 14B on A100, 500 steps: expected ~{expected}, got {result}")
        all_passed = False
    else:
        print(f"PASS [1.1] estimate_lora_training_hours: {result} hours (expected ~{expected})")

    # Check 2: estimate_lora_training_hours — H100 is faster than A100
    a100_hours = estimate_lora_training_hours(7.0, 200, "a100")
    h100_hours = estimate_lora_training_hours(7.0, 200, "h100")
    if a100_hours is None or h100_hours is None:
        print("FAIL [1.2] estimate_lora_training_hours returned None")
        all_passed = False
    elif h100_hours >= a100_hours:
        print(f"FAIL [1.2] H100 should be faster than A100: h100={h100_hours}, a100={a100_hours}")
        all_passed = False
    else:
        print(f"PASS [1.2] H100 faster than A100: {h100_hours}h vs {a100_hours}h")

    # Check 3: estimate_lora_training_hours — invalid GPU raises ValueError
    try:
        estimate_lora_training_hours(14.0, 500, "tpu_v5")
        print("FAIL [1.3] Should raise ValueError for unsupported gpu_type")
        all_passed = False
    except ValueError:
        print("PASS [1.3] Raises ValueError for unsupported gpu_type")
    except Exception as e:
        print(f"FAIL [1.3] Expected ValueError, got {type(e).__name__}: {e}")
        all_passed = False

    # Check 4: estimate_training_cost_usd — basic structure and values
    cost = estimate_training_cost_usd(
        training_hours=8.0,
        gpu_hourly_rate=1.29,
        num_checkpoints=10,
        use_lora=True,
    )
    if cost is None:
        print("FAIL [1.4] estimate_training_cost_usd returned None — implement the function")
        all_passed = False
    else:
        required_keys = {"gpu_cost_usd", "storage_cost_usd", "validation_cost_usd", "total_cost_usd"}
        missing = required_keys - set(cost.keys())
        if missing:
            print(f"FAIL [1.4] Missing keys: {missing}")
            all_passed = False
        else:
            expected_gpu = round(8.0 * 1.29, 2)
            if abs(cost["gpu_cost_usd"] - expected_gpu) > 0.01:
                print(f"FAIL [1.4] gpu_cost_usd: expected {expected_gpu}, got {cost['gpu_cost_usd']}")
                all_passed = False
            else:
                # Verify total = sum of components
                component_sum = round(
                    cost["gpu_cost_usd"] + cost["storage_cost_usd"] + cost["validation_cost_usd"], 2
                )
                if abs(cost["total_cost_usd"] - component_sum) > 0.02:
                    print(
                        f"FAIL [1.4] total_cost_usd ({cost['total_cost_usd']}) != "
                        f"sum of components ({component_sum})"
                    )
                    all_passed = False
                else:
                    print(f"PASS [1.4] estimate_training_cost_usd: total=${cost['total_cost_usd']}")

    # Check 5: estimate_training_cost_usd — LoRA cheaper than full fine-tune on storage
    lora_cost = estimate_training_cost_usd(5.0, 1.29, 20, use_lora=True)
    full_cost = estimate_training_cost_usd(5.0, 1.29, 20, use_lora=False)
    if lora_cost is None or full_cost is None:
        print("FAIL [1.5] estimate_training_cost_usd returned None")
        all_passed = False
    elif lora_cost["storage_cost_usd"] >= full_cost["storage_cost_usd"]:
        print(
            f"FAIL [1.5] LoRA storage should be cheaper than full: "
            f"LoRA={lora_cost['storage_cost_usd']}, full={full_cost['storage_cost_usd']}"
        )
        all_passed = False
    else:
        print(
            f"PASS [1.5] LoRA storage cheaper: "
            f"${lora_cost['storage_cost_usd']} vs ${full_cost['storage_cost_usd']}"
        )

    # Check 6: compute_break_even_queries — o3 comparison
    # training=$15, custom=$0.000089/query, o3=$0.0552/query
    # break_even = 15 / (0.0552 - 0.000089) = 15 / 0.055111 ≈ 272
    be = compute_break_even_queries(
        training_cost_usd=15.0,
        custom_cost_per_query=0.000089,
        frontier_cost_per_query=0.0552,
    )
    if be is None:
        print("FAIL [1.6] compute_break_even_queries returned None — implement the function")
        all_passed = False
    elif be == -1:
        print("FAIL [1.6] Should return positive break-even (frontier API is more expensive)")
        all_passed = False
    elif not (200 <= be <= 350):
        print(f"FAIL [1.6] Expected break-even around 272 queries, got {be}")
        all_passed = False
    else:
        print(f"PASS [1.6] Break-even vs o3: {be} queries")

    # Check 7: compute_break_even_queries — custom model more expensive → -1
    be_impossible = compute_break_even_queries(
        training_cost_usd=100.0,
        custom_cost_per_query=0.01,
        frontier_cost_per_query=0.005,  # frontier is cheaper
    )
    if be_impossible is None:
        print("FAIL [1.7] compute_break_even_queries returned None")
        all_passed = False
    elif be_impossible != -1:
        print(f"FAIL [1.7] Expected -1 when frontier is cheaper, got {be_impossible}")
        all_passed = False
    else:
        print("PASS [1.7] Returns -1 when break-even is not achievable")

    return all_passed


# ===========================================================================
# PART 2: Benchmark Evaluation Harness
# ===========================================================================
# Implement BenchmarkHarness below.
# The harness must:
#   - Accept a model_fn, eval_fn, and cost_fn
#   - Call model_fn for each test case and measure latency
#   - Record whether eval_fn returns True or False
#   - Aggregate results into a BenchmarkReport
# ===========================================================================


@dataclass
class QueryResult:
    """Result for a single test query."""
    query_id: str
    correct: bool
    latency_seconds: float
    tokens_used: int
    cost_usd: float
    error: str | None = None


@dataclass
class BenchmarkReport:
    """Aggregated results for a full benchmark run."""
    model_name: str
    total_queries: int
    correct: int
    accuracy: float
    mean_latency_seconds: float
    p95_latency_seconds: float
    cost_per_1000_usd: float
    results: list[dict]

    def summary(self) -> str:
        return (
            f"Model: {self.model_name}\n"
            f"Accuracy: {self.accuracy:.1%} ({self.correct}/{self.total_queries})\n"
            f"Mean Latency: {self.mean_latency_seconds:.3f}s\n"
            f"P95 Latency: {self.p95_latency_seconds:.3f}s\n"
            f"Cost/1K: ${self.cost_per_1000_usd:.2f}"
        )


class BenchmarkHarness:
    """
    Runs a test set against a model function and produces a BenchmarkReport.

    Implement the `run` method. It must:
      1. Call self.model_fn(case["query"]) for each test case.
         model_fn returns (response_text: str, tokens_used: int)
      2. Measure wall-clock latency for each call using time.perf_counter()
      3. Call self.eval_fn(case["expected"], response_text) to get bool correctness
      4. Call self.cost_fn(tokens_used) to get cost in USD
      5. Handle exceptions: if model_fn raises, record error=str(exc), correct=False
      6. Return a BenchmarkReport with correct aggregate metrics

    P95 latency: sort latencies, take the value at index int(len * 0.95)
    cost_per_1000_usd: (total_cost / total_queries) * 1000
    """

    def __init__(
        self,
        model_fn: Callable[[str], tuple[str, int]],
        eval_fn: Callable[[str, str], bool],
        cost_fn: Callable[[int], float],
    ):
        self.model_fn = model_fn
        self.eval_fn = eval_fn
        self.cost_fn = cost_fn

    def run(
        self,
        test_cases: list[dict],
        model_name: str,
    ) -> BenchmarkReport:
        """
        Run the benchmark against all test_cases.

        Args:
            test_cases: List of dicts, each with keys:
                "id":       str — unique identifier
                "query":    str — input to the model
                "expected": str — expected correct output

            model_name: Label for the report (e.g. "qwen2.5-14b-rl")

        Returns:
            BenchmarkReport with all metrics computed.
        """
        # YOUR CODE HERE
        pass


# ===========================================================================
# Part 2 Self-Checks
# ===========================================================================

def check_part2():
    print()
    print("=" * 60)
    print("PART 2: Benchmark Evaluation Harness")
    print("=" * 60)
    all_passed = True

    # Build deterministic fake model and evaluator for testing
    def fake_model(query: str) -> tuple[str, int]:
        """Returns 'CORRECT' for queries containing 'good', 'WRONG' otherwise."""
        time.sleep(0.01)  # Simulate latency
        response = "CORRECT" if "good" in query else "WRONG"
        return response, 100

    def fake_eval(expected: str, actual: str) -> bool:
        return actual == expected

    def fake_cost(tokens: int) -> float:
        return tokens * 0.000001  # $0.001 per 1M tokens

    test_cases = [
        {"id": "q1", "query": "this is a good query", "expected": "CORRECT"},
        {"id": "q2", "query": "this is a good query", "expected": "CORRECT"},
        {"id": "q3", "query": "this is a bad query", "expected": "CORRECT"},
        {"id": "q4", "query": "this is a bad query", "expected": "CORRECT"},
        {"id": "q5", "query": "this is a good query", "expected": "CORRECT"},
    ]

    harness = BenchmarkHarness(
        model_fn=fake_model,
        eval_fn=fake_eval,
        cost_fn=fake_cost,
    )

    report = harness.run(test_cases, "fake-model")

    # Check 1: report is not None
    if report is None:
        print("FAIL [2.1] BenchmarkHarness.run() returned None — implement the method")
        return False

    # Check 2: total_queries
    if report.total_queries != 5:
        print(f"FAIL [2.2] total_queries: expected 5, got {report.total_queries}")
        all_passed = False
    else:
        print(f"PASS [2.2] total_queries = {report.total_queries}")

    # Check 3: accuracy — 3 "good" queries out of 5
    if abs(report.accuracy - 0.6) > 0.01:
        print(f"FAIL [2.3] accuracy: expected 0.60, got {report.accuracy:.2f}")
        all_passed = False
    else:
        print(f"PASS [2.3] accuracy = {report.accuracy:.1%}")

    # Check 4: correct count
    if report.correct != 3:
        print(f"FAIL [2.4] correct count: expected 3, got {report.correct}")
        all_passed = False
    else:
        print(f"PASS [2.4] correct count = {report.correct}")

    # Check 5: latency measured and positive
    if report.mean_latency_seconds is None or report.mean_latency_seconds <= 0:
        print(f"FAIL [2.5] mean_latency_seconds should be positive, got {report.mean_latency_seconds}")
        all_passed = False
    else:
        print(f"PASS [2.5] mean_latency_seconds = {report.mean_latency_seconds:.3f}s")

    # Check 6: p95 latency >= mean latency
    if report.p95_latency_seconds < report.mean_latency_seconds:
        print(
            f"FAIL [2.6] P95 latency ({report.p95_latency_seconds}) "
            f"should be >= mean ({report.mean_latency_seconds})"
        )
        all_passed = False
    else:
        print(f"PASS [2.6] p95_latency_seconds = {report.p95_latency_seconds:.3f}s")

    # Check 7: cost_per_1000_usd is positive
    if report.cost_per_1000_usd is None or report.cost_per_1000_usd <= 0:
        print(f"FAIL [2.7] cost_per_1000_usd should be positive, got {report.cost_per_1000_usd}")
        all_passed = False
    else:
        print(f"PASS [2.7] cost_per_1000_usd = ${report.cost_per_1000_usd:.4f}")

    # Check 8: results list length matches total_queries
    if len(report.results) != 5:
        print(f"FAIL [2.8] results list length: expected 5, got {len(report.results)}")
        all_passed = False
    else:
        print(f"PASS [2.8] results list has {len(report.results)} entries")

    # Check 9: error handling — model_fn that raises
    def crashing_model(query: str) -> tuple[str, int]:
        raise RuntimeError("Simulated model failure")

    crash_harness = BenchmarkHarness(
        model_fn=crashing_model,
        eval_fn=fake_eval,
        cost_fn=fake_cost,
    )
    crash_report = crash_harness.run(
        [{"id": "q1", "query": "test", "expected": "X"}],
        "crashing-model",
    )
    if crash_report is None:
        print("FAIL [2.9] run() returned None even with crashing model_fn")
        all_passed = False
    elif crash_report.correct != 0:
        print(f"FAIL [2.9] A crashing model_fn should produce 0 correct, got {crash_report.correct}")
        all_passed = False
    elif not crash_report.results[0].get("error"):
        print("FAIL [2.9] Error should be recorded in results when model_fn raises")
        all_passed = False
    else:
        print(f"PASS [2.9] Errors handled gracefully: {crash_report.results[0]['error'][:40]}")

    return all_passed


# ===========================================================================
# PART 3: Checkpoint Comparison
# ===========================================================================
# Implement compare_checkpoints() below.
#
# It takes a list of BenchmarkReports (one per checkpoint) and returns
# a summary dict identifying the best checkpoint by accuracy.
# ===========================================================================


def compare_checkpoints(reports: list[BenchmarkReport]) -> dict:
    """
    Compare multiple benchmark reports (one per training checkpoint) and
    identify the best checkpoint.

    Args:
        reports: List of BenchmarkReport instances. Each report's model_name
                 field contains the checkpoint label (e.g. "step_100").

    Returns:
        Dictionary with keys:
            "best_checkpoint":    str  — model_name of the highest-accuracy report
            "best_accuracy":      float — accuracy of the best checkpoint
            "worst_checkpoint":   str  — model_name of the lowest-accuracy report
            "worst_accuracy":     float — accuracy of the worst checkpoint
            "accuracy_gain":      float — best_accuracy - worst_accuracy (rounded to 4 dp)
            "ranking":            list[str] — model_names sorted by accuracy descending

        If reports is empty, return {"error": "no reports provided"}.
        If reports has exactly one entry, best and worst are the same checkpoint.
    """
    # YOUR CODE HERE
    pass


# ===========================================================================
# Part 3 Self-Checks
# ===========================================================================

def _make_report(model_name: str, correct: int, total: int, latency: float) -> BenchmarkReport:
    """Helper: construct a minimal BenchmarkReport for testing."""
    return BenchmarkReport(
        model_name=model_name,
        total_queries=total,
        correct=correct,
        accuracy=correct / total,
        mean_latency_seconds=latency,
        p95_latency_seconds=latency * 1.3,
        cost_per_1000_usd=0.09,
        results=[],
    )


def check_part3():
    print()
    print("=" * 60)
    print("PART 3: Checkpoint Comparison")
    print("=" * 60)
    all_passed = True

    # Three checkpoints simulating training progression
    reports = [
        _make_report("step_100", correct=52, total=100, latency=1.3),
        _make_report("step_250", correct=74, total=100, latency=1.2),
        _make_report("step_500", correct=96, total=100, latency=1.1),
    ]

    result = compare_checkpoints(reports)

    # Check 1: returns something
    if result is None:
        print("FAIL [3.1] compare_checkpoints returned None — implement the function")
        return False

    # Check 2: best checkpoint
    if result.get("best_checkpoint") != "step_500":
        print(f"FAIL [3.2] best_checkpoint: expected 'step_500', got '{result.get('best_checkpoint')}'")
        all_passed = False
    else:
        print(f"PASS [3.2] best_checkpoint = {result['best_checkpoint']}")

    # Check 3: best accuracy
    if abs(result.get("best_accuracy", 0) - 0.96) > 0.001:
        print(f"FAIL [3.3] best_accuracy: expected 0.96, got {result.get('best_accuracy')}")
        all_passed = False
    else:
        print(f"PASS [3.3] best_accuracy = {result['best_accuracy']:.2f}")

    # Check 4: worst checkpoint
    if result.get("worst_checkpoint") != "step_100":
        print(f"FAIL [3.4] worst_checkpoint: expected 'step_100', got '{result.get('worst_checkpoint')}'")
        all_passed = False
    else:
        print(f"PASS [3.4] worst_checkpoint = {result['worst_checkpoint']}")

    # Check 5: accuracy_gain = best - worst
    expected_gain = round(0.96 - 0.52, 4)
    if abs(result.get("accuracy_gain", 0) - expected_gain) > 0.001:
        print(f"FAIL [3.5] accuracy_gain: expected {expected_gain}, got {result.get('accuracy_gain')}")
        all_passed = False
    else:
        print(f"PASS [3.5] accuracy_gain = {result['accuracy_gain']:.4f}")

    # Check 6: ranking is descending by accuracy
    ranking = result.get("ranking", [])
    if ranking != ["step_500", "step_250", "step_100"]:
        print(f"FAIL [3.6] ranking: expected ['step_500', 'step_250', 'step_100'], got {ranking}")
        all_passed = False
    else:
        print(f"PASS [3.6] ranking = {ranking}")

    # Check 7: empty reports
    empty_result = compare_checkpoints([])
    if empty_result is None or "error" not in empty_result:
        print("FAIL [3.7] Should return {'error': ...} for empty reports list")
        all_passed = False
    else:
        print(f"PASS [3.7] Empty reports handled: {empty_result['error']}")

    # Check 8: single report — best and worst are the same
    single_result = compare_checkpoints([_make_report("only_checkpoint", correct=80, total=100, latency=1.5)])
    if single_result is None:
        print("FAIL [3.8] compare_checkpoints returned None for single report")
        all_passed = False
    elif single_result.get("best_checkpoint") != "only_checkpoint":
        print(f"FAIL [3.8] best_checkpoint with single report: expected 'only_checkpoint', got {single_result.get('best_checkpoint')}")
        all_passed = False
    elif single_result.get("worst_checkpoint") != "only_checkpoint":
        print(f"FAIL [3.8] worst_checkpoint with single report: expected 'only_checkpoint', got {single_result.get('worst_checkpoint')}")
        all_passed = False
    else:
        print("PASS [3.8] Single-report case: best and worst are the same checkpoint")

    return all_passed


# ===========================================================================
# Main: Run All Checks
# ===========================================================================

if __name__ == "__main__":
    p1 = check_part1()
    p2 = check_part2()
    p3 = check_part3()

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Part 1 (Cost Calculator):     {'PASS' if p1 else 'FAIL'}")
    print(f"Part 2 (Benchmark Harness):   {'PASS' if p2 else 'FAIL'}")
    print(f"Part 3 (Checkpoint Compare):  {'PASS' if p3 else 'FAIL'}")

    if p1 and p2 and p3:
        print()
        print("All parts complete. You now have the core production tooling:")
        print("  - Cost estimator: know your training budget before committing")
        print("  - Benchmark harness: measure what your RL training actually achieved")
        print("  - Checkpoint comparison: identify the best model to promote")
    else:
        print()
        print("Some checks failed. Review the FAIL messages above and fix your implementations.")
        print("Each function has a detailed docstring explaining the expected behavior.")
