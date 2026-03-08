"""
Reward Function Templates
=========================

Production-ready reward functions for training agents with ART/GRPO.
Includes programmatic rewards, LLM-as-a-judge (RULER), and hybrid approaches.

Usage:
    from reward_functions import SQLReward, HybridReward, ruler_score
"""

import re
import sqlite3
from dataclasses import dataclass


# ============================================================
# Programmatic Reward Functions
# ============================================================

class SQLReward:
    """
    Reward function for text-to-SQL agents.

    Scores responses based on SQL validity, correctness, and efficiency.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _execute_sql(self, sql: str) -> tuple[bool, str]:
        """Execute SQL and return (success, result_or_error)."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            return True, {"columns": columns, "rows": rows}
        except sqlite3.Error as e:
            return False, str(e)
        finally:
            conn.close()

    def score(self, prompt: str, response: str, expected_sql: str = None) -> float:
        """
        Score a SQL agent's response.

        Parameters
        ----------
        prompt : str
            The natural language question
        response : str
            The agent's full response (may contain SQL in code blocks)
        expected_sql : str, optional
            Ground truth SQL for result comparison

        Returns
        -------
        float between 0.0 and 1.0
        """
        # Extract SQL from response
        sql = self._extract_sql(response)
        if not sql:
            return 0.0

        # Check if SQL is valid (executes without error)
        success, result = self._execute_sql(sql)
        if not success:
            return 0.1  # Partial credit for attempting SQL

        # Check if results are non-empty
        if not result["rows"]:
            return 0.3  # Valid SQL but no results

        # If we have expected SQL, compare results
        if expected_sql:
            expected_success, expected_result = self._execute_sql(expected_sql)
            if expected_success and result["rows"] == expected_result["rows"]:
                return 1.0  # Perfect match
            elif expected_success:
                # Partial credit for overlapping results
                expected_set = set(str(r) for r in expected_result["rows"])
                actual_set = set(str(r) for r in result["rows"])
                if expected_set and actual_set:
                    overlap = len(expected_set & actual_set) / len(expected_set)
                    return 0.5 + 0.5 * overlap

        # Valid SQL with results but no ground truth
        return 0.6

    @staticmethod
    def _extract_sql(response: str) -> str | None:
        """Extract SQL query from agent response."""
        # Try code block first
        match = re.search(r"```(?:sql)?\s*\n?(.*?)\n?```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try to find SELECT statement
        match = re.search(r"(SELECT\s+.*?;)", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return None


class ToolUseReward:
    """
    Reward function for multi-tool agents.

    Rewards agents for using tools effectively and in the right order.
    """

    def __init__(self, required_tools: list[str] = None, max_turns: int = 10):
        self.required_tools = required_tools or []
        self.max_turns = max_turns

    def score(self, trajectory: list[dict]) -> float:
        """
        Score an agent trajectory based on tool usage patterns.

        Parameters
        ----------
        trajectory : list[dict]
            List of messages in the trajectory, each with "role" and "content" keys.
            Tool calls have role="assistant" with "tool_calls" field.
            Tool results have role="tool" with "content" field.

        Returns
        -------
        float between 0.0 and 1.0
        """
        tools_used = []
        errors = 0
        total_turns = 0

        for msg in trajectory:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for call in msg["tool_calls"]:
                    tools_used.append(call.get("function", {}).get("name", ""))
                    total_turns += 1

            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if "error" in content.lower() or "Error" in content:
                    errors += 1

        score = 0.0

        # Reward using required tools
        if self.required_tools:
            used_required = sum(1 for t in self.required_tools if t in tools_used)
            score += 0.4 * (used_required / len(self.required_tools))
        else:
            score += 0.4 if tools_used else 0.0

        # Penalize errors
        if tools_used:
            error_rate = errors / len(tools_used)
            score += 0.3 * (1 - error_rate)
        else:
            score += 0.0

        # Reward efficiency (fewer turns is better)
        if total_turns > 0:
            efficiency = max(0, 1 - (total_turns - 1) / self.max_turns)
            score += 0.3 * efficiency

        return min(1.0, score)


# ============================================================
# RULER: LLM-as-a-Judge Scoring
# ============================================================

RULER_SYSTEM_PROMPT = """You are an expert judge evaluating AI agent performance.

You will be given a task description and multiple agent attempts (trajectories).
Score each trajectory from 0.0 to 1.0 based on:
- Correctness: Did the agent achieve the goal?
- Efficiency: Did it use minimal steps?
- Robustness: Did it handle errors gracefully?

Output ONLY a JSON array of scores, one per trajectory.
Example: [0.8, 0.3, 0.95, 0.1]"""


def ruler_score(
    task: str,
    trajectories: list[str],
    judge_model: str = "openai/o4-mini",
) -> list[float]:
    """
    Score multiple trajectories using LLM-as-a-judge (RULER).

    Parameters
    ----------
    task : str
        Description of the task the agent was trying to accomplish
    trajectories : list[str]
        String representations of each trajectory
    judge_model : str
        Model to use as judge

    Returns
    -------
    list[float] - scores between 0.0 and 1.0 for each trajectory
    """
    import json

    # Build the judge prompt
    prompt_parts = [f"Task: {task}\n"]
    for i, traj in enumerate(trajectories):
        prompt_parts.append(f"--- Trajectory {i+1} ---\n{traj}\n")
    prompt_parts.append(
        f"\nScore each trajectory from 0.0 to 1.0. "
        f"Return ONLY a JSON array of {len(trajectories)} scores."
    )
    judge_prompt = "\n".join(prompt_parts)

    # Call the judge model
    # This uses the OpenAI-compatible API format that ART/vLLM provides
    try:
        from openai import OpenAI

        # Parse provider from model string
        if "/" in judge_model:
            provider, model_name = judge_model.split("/", 1)
        else:
            provider, model_name = "openai", judge_model

        client = OpenAI()  # Uses OPENAI_API_KEY env var
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": RULER_SYSTEM_PROMPT},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )

        scores_text = response.choices[0].message.content.strip()
        scores = json.loads(scores_text)

        # Validate and clamp scores
        scores = [max(0.0, min(1.0, float(s))) for s in scores]

        if len(scores) != len(trajectories):
            raise ValueError(
                f"Expected {len(trajectories)} scores, got {len(scores)}"
            )

        return scores

    except Exception as e:
        print(f"RULER scoring failed: {e}. Returning uniform scores.")
        return [0.5] * len(trajectories)


# ============================================================
# Hybrid Reward: Programmatic + RULER
# ============================================================

@dataclass
class HybridReward:
    """
    Combines programmatic and LLM-judge rewards.

    Useful when you have some checkable criteria (SQL validity, test passing)
    plus subjective quality criteria (explanation clarity, reasoning).
    """
    programmatic_weight: float = 0.6
    ruler_weight: float = 0.4
    judge_model: str = "openai/o4-mini"

    def score(
        self,
        task: str,
        trajectories: list[str],
        programmatic_scores: list[float],
    ) -> list[float]:
        """
        Compute hybrid scores combining programmatic and RULER.

        Parameters
        ----------
        task : str
            Task description for the LLM judge
        trajectories : list[str]
            String representations of trajectories
        programmatic_scores : list[float]
            Pre-computed programmatic scores (0-1)

        Returns
        -------
        list[float] - hybrid scores between 0.0 and 1.0
        """
        # Get RULER scores
        ruler_scores = ruler_score(task, trajectories, self.judge_model)

        # Combine
        hybrid = [
            self.programmatic_weight * p + self.ruler_weight * r
            for p, r in zip(programmatic_scores, ruler_scores)
        ]

        return hybrid


# ============================================================
# Advantage Computation (used by GRPO)
# ============================================================

def compute_advantages(rewards: list[float]) -> list[float]:
    """
    Compute group-relative advantages for GRPO.

    Parameters
    ----------
    rewards : list[float]
        Raw reward scores for a group of completions

    Returns
    -------
    list[float] - normalized advantages (mean=0, std=1)
    """
    import numpy as np
    rewards_arr = np.array(rewards, dtype=np.float64)
    mean = rewards_arr.mean()
    std = rewards_arr.std()

    if std < 1e-8:
        return [0.0] * len(rewards)

    advantages = ((rewards_arr - mean) / std).tolist()
    return advantages


if __name__ == "__main__":
    # Demo: compute advantages from sample rewards
    sample_rewards = [0.3, 0.5, 0.7, 0.9]
    advantages = compute_advantages(sample_rewards)

    print("GRPO Advantage Computation Demo")
    print("=" * 40)
    for r, a in zip(sample_rewards, advantages):
        direction = "reinforce" if a > 0 else "suppress"
        print(f"  Reward: {r:.1f} -> Advantage: {a:+.2f} ({direction})")
