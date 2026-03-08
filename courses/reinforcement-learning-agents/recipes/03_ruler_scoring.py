"""
Recipe: RULER LLM-as-a-Judge Scoring
======================================

Use an LLM to rank agent trajectories by quality.
RULER works because relative scoring is more reliable than absolute.

Usage:
    scores = ruler_rank(task, trajectories)
"""

import json
from openai import OpenAI

JUDGE_PROMPT = """You are judging AI agent performance on a task.
Compare the trajectories below and score each from 0.0 to 1.0.
Focus on: correctness, efficiency, and error handling.
Return ONLY a JSON array of scores."""


def ruler_rank(
    task: str,
    trajectories: list[str],
    model: str = "o4-mini",
) -> list[float]:
    """
    Score trajectories using LLM-as-a-judge.

    Parameters
    ----------
    task : str
        What the agent was trying to do
    trajectories : list[str]
        String representations of each agent run
    model : str
        Judge model (default: o4-mini)

    Returns
    -------
    list[float] - scores between 0.0 and 1.0
    """
    # Build comparison prompt
    parts = [f"Task: {task}\n"]
    for i, t in enumerate(trajectories):
        parts.append(f"--- Attempt {i+1} ---\n{t}\n")
    parts.append(f"Score each attempt 0.0-1.0. Return JSON array of {len(trajectories)} scores.")

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": "\n".join(parts)},
        ],
        temperature=0.0,
    )

    scores = json.loads(response.choices[0].message.content.strip())
    return [max(0.0, min(1.0, float(s))) for s in scores]


if __name__ == "__main__":
    # Demo with sample trajectories
    task = "Find all employees earning more than $150,000"
    trajectories = [
        "Called run_query('SELECT * FROM employees WHERE salary > 150000') -> 4 results",
        "Called list_tables() -> departments, employees, projects. "
        "Called describe_table('employees') -> columns listed. "
        "Called run_query('SELECT name, salary FROM employees WHERE salary > 150000 ORDER BY salary DESC') -> 4 results",
        "Called run_query('SELECT * FROM emp') -> SQL Error: no such table",
    ]

    print("RULER Scoring Demo")
    print("=" * 50)
    print(f"Task: {task}\n")

    # In production, this calls the judge LLM
    # For demo, we show expected relative ordering
    expected_scores = [0.6, 0.95, 0.1]
    for i, (traj, score) in enumerate(zip(trajectories, expected_scores)):
        print(f"Attempt {i+1}: score={score:.2f}")
        print(f"  {traj[:80]}...")
        print()
