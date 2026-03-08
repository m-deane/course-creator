# The RULER Mechanism

## In Brief

RULER (Reward Using LLM Evaluation Rubrics) automatically generates reward signals for GRPO training by using an LLM to judge which trajectories in a group are better than others. It exploits two key insights: relative scoring is more reliable than absolute scoring, and GRPO only needs relative scores to work.

## The Two Key Insights

### Insight 1: Relative Scoring Is More Reliable Than Absolute Scoring

Ask an LLM to rate a response from 0 to 10, and you get unreliable results:

- The same response gets different scores across runs
- The scale is uncalibrated (what does "7" mean?)
- LLMs are biased toward middle values and toward leniency
- Small prompt wording changes shift scores by 2-3 points

Ask an LLM "which of these four responses is best?" and you get much more reliable results:

- Comparison forces the model to make distinctions
- Relative quality is easier to perceive than absolute quality
- Results are more consistent across runs
- The same information humans use for preference learning

This is not a new discovery — it mirrors how humans assess quality. A wine expert who struggles to assign 85/100 versus 87/100 can reliably rank five wines from worst to best.

### Insight 2: GRPO Only Needs Relative Scores

Recall from Module 01 how GRPO computes advantages:

$$A_i = r_i - \bar{r}$$

Where $r_i$ is the reward for trajectory $i$ and $\bar{r}$ is the mean reward across the group.

GRPO does not need absolute reward values. It only needs to know which trajectories were better or worse than the group average. This means:

- A score of 0.8 in a group where the mean is 0.5 → positive advantage → reinforce
- A score of 0.3 in a group where the mean is 0.5 → negative advantage → discourage

The absolute scale does not matter. Relative ordering is all GRPO needs to learn.

RULER exploits this: it assigns scores in [0, 1] based on relative quality within the group, and these scores plug directly into GRPO without any normalization step.

## How RULER Works: The Full Process

```
┌─────────────────────────────────────────────────────────────┐
│                     RULER Training Step                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Sample one prompt from the training set                  │
│                                                              │
│  2. Generate N trajectories (typically N=4 or N=8)          │
│     using the current policy                                 │
│                                                              │
│  3. Send all N trajectories to the judge LLM:               │
│     "Here are N agent responses to the same task.           │
│      Assign each a score from 0.0 to 1.0 where              │
│      1.0 = best possible response."                          │
│                                                              │
│  4. Parse judge output → [0.73, 0.41, 0.89, 0.31]           │
│                                                              │
│  5. Use scores directly as GRPO rewards                      │
│     Compute advantages: A_i = r_i - mean(r)                 │
│                                                              │
│  6. Update policy to reinforce high-advantage trajectories   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

The judge sees all trajectories simultaneously. This is what enables reliable relative scoring — the model can compare them directly rather than evaluating each in isolation.

## Code: RULER Scoring Function

```python
import json
import asyncio
from openai import AsyncOpenAI
from typing import Any

client = AsyncOpenAI()

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for AI agent responses.

You will receive multiple agent trajectories for the same task. Your job is to
evaluate their quality relative to each other and assign each a score from 0.0
to 1.0 where:

- 1.0 = Ideal response: correct, efficient, well-reasoned, appropriate tool use
- 0.75 = Good response: correct or nearly correct with minor issues
- 0.5 = Mediocre response: partially correct or correct but poorly executed
- 0.25 = Poor response: mostly incorrect or uses inappropriate approach
- 0.0 = Failed response: completely wrong, crashes, or refuses to attempt

IMPORTANT: Assign scores relative to each other. If trajectory A is clearly
better than trajectory B, its score should be meaningfully higher. Avoid
assigning all trajectories the same score.

Respond with a JSON object mapping trajectory IDs to scores:
{"trajectory_0": 0.85, "trajectory_1": 0.42, ...}
"""


def format_trajectories_for_judge(
    trajectories: list[list[dict[str, Any]]],
    task_description: str,
) -> str:
    """Format a group of trajectories for the judge LLM.

    Args:
        trajectories: List of trajectories, each a list of message dicts.
        task_description: Plain-language description of what the agent was asked to do.

    Returns:
        Formatted string ready to send as user message to the judge.
    """
    sections = [f"TASK: {task_description}\n"]

    for i, trajectory in enumerate(trajectories):
        sections.append(f"\n--- TRAJECTORY {i} ---")
        for msg in trajectory:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")

            # Handle tool calls in the trajectory
            if msg.get("tool_calls"):
                tool_summary = []
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    tool_summary.append(
                        f"TOOL CALL: {fn.get('name', 'unknown')}({fn.get('arguments', '')})"
                    )
                content = "\n".join(tool_summary)

            sections.append(f"[{role}]: {content}")

    sections.append(
        "\nEvaluate these trajectories and return JSON scores for each trajectory ID."
    )
    return "\n".join(sections)


async def ruler_score_group(
    trajectories: list[list[dict[str, Any]]],
    task_description: str,
    judge_model: str = "o4-mini",
) -> list[float]:
    """Score a group of trajectories using RULER's LLM-as-a-judge approach.

    Args:
        trajectories: List of N trajectories to score, where each trajectory
                      is a list of message dicts (role, content, tool_calls).
        task_description: Plain-language description of what the agent was trying to do.
        judge_model: Model to use as judge. o4-mini and claude-3-5-haiku are good choices.

    Returns:
        List of float scores in [0.0, 1.0], one per trajectory.
        Preserves order: scores[i] corresponds to trajectories[i].

    Example:
        >>> trajectories = [traj_a, traj_b, traj_c, traj_d]
        >>> scores = await ruler_score_group(trajectories, "Write a SQL query to find top customers")
        >>> scores
        [0.85, 0.42, 0.91, 0.23]
    """
    if not trajectories:
        return []

    n = len(trajectories)
    user_message = format_trajectories_for_judge(trajectories, task_description)

    try:
        response = await client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,  # Deterministic scoring
        )

        raw = response.choices[0].message.content
        scores_dict = json.loads(raw)

        # Extract scores in order, defaulting to 0.5 if a trajectory is missing
        scores = []
        for i in range(n):
            key = f"trajectory_{i}"
            score = scores_dict.get(key, 0.5)
            # Clamp to valid range
            score = max(0.0, min(1.0, float(score)))
            scores.append(score)

        return scores

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # If the judge fails to produce valid output, return neutral scores
        # This prevents a judge failure from crashing the training run
        print(f"Warning: Judge scoring failed ({e}), returning neutral scores")
        return [0.5] * n
```

## Configuring the Judge LLM

The judge model choice matters. Criteria for a good judge:

**Speed:** The judge runs once per training step, N trajectories at a time. A slow judge bottlenecks training. `o4-mini` and `claude-3-5-haiku` are fast enough for production use.

**Instruction following:** The judge needs to return structured JSON reliably. Models with strong instruction following (GPT-4o, Claude 3.5) work better than smaller models.

**Task knowledge:** The judge needs to know what "good" looks like for your task. For SQL, it needs SQL knowledge. For research tasks, it needs to assess reasoning quality. Claude and GPT-4o have broad enough knowledge for most tasks.

**Do not use the same model as the trainee:** The judge is a fixed reference point. If you train GPT-4o and judge with GPT-4o, you are teaching the student to satisfy the teacher rather than to do the task.

```python
# Recommended judge configurations by use case

JUDGE_CONFIGS = {
    "general_agent": {
        "model": "o4-mini",
        "temperature": 0.0,
        "description": "Fast, reliable for most agent tasks",
    },
    "complex_reasoning": {
        "model": "gpt-4o",
        "temperature": 0.0,
        "description": "Higher quality for tasks requiring deep reasoning evaluation",
    },
    "cost_sensitive": {
        "model": "claude-3-5-haiku-20241022",
        "temperature": 0.0,
        "description": "Cheapest option that still follows instructions reliably",
    },
}
```

## Writing Effective Judge Prompts

The judge prompt is the most important variable in RULER. A poorly written prompt produces noisy, inconsistent scores that slow training or teach the wrong behaviors.

### Structure of an Effective Judge Prompt

```
1. Role definition (1 sentence)
   "You are an expert evaluator for [task domain] agent responses."

2. Scoring scale with anchors (4-5 bullet points)
   Define what 0.0, 0.25, 0.5, 0.75, and 1.0 mean concretely.

3. Explicit relative scoring instruction
   "Assign scores relative to each other. If A is better than B,
    its score should be meaningfully higher."

4. Anti-clustering instruction
   "Avoid assigning all trajectories the same score."

5. Output format specification
   Exact JSON format with example.
```

### Example: SQL Agent Judge Prompt

```python
SQL_JUDGE_PROMPT = """You are an expert SQL evaluator for database query agents.

You will receive multiple agent trajectories for the same SQL task. Evaluate their
quality relative to each other using this scale:

- 1.0 = Perfect query: correct results, efficient execution, readable SQL, handles edge cases
- 0.75 = Good query: correct or near-correct, minor inefficiencies acceptable
- 0.5 = Partial: returns approximately correct results but with significant issues
       (wrong aggregation, missing JOINs, inefficient table scans)
- 0.25 = Poor: mostly wrong but shows understanding of the task
- 0.0 = Failed: syntax error, wrong table, or refuses to attempt

Key evaluation criteria (in order of importance):
1. Does the query return correct results for the given question?
2. Is the approach efficient (avoids full table scans when indexes exist)?
3. Does it use appropriate SQL constructs (not over-engineered)?
4. Are edge cases handled (NULL values, empty results)?

Assign scores relative to each other. The best trajectory in the group should
receive the highest score; the worst should receive the lowest. Do not assign
the same score to all trajectories unless they are genuinely identical in quality.

Respond with only this JSON format:
{"trajectory_0": <score>, "trajectory_1": <score>, ...}
"""
```

### Anti-Patterns in Judge Prompts

**Too vague:** "Rate how good this response is." — The model has no calibration.

**No relative instruction:** Omitting "assign scores relative to each other" causes the judge to evaluate each trajectory in isolation, losing the comparative advantage.

**Overly complex rubric:** More than 5-6 criteria and the judge cannot weigh them all. Prioritize your criteria explicitly.

**No output format anchor:** Without seeing an example JSON structure, the judge sometimes returns prose descriptions.

## Validating Your Judge Before Training

Before using a judge in a training run, validate it on a small held-out set where you know the correct relative ranking:

```python
async def validate_judge(
    judge_prompt: str,
    validation_examples: list[dict],
    judge_model: str = "o4-mini",
) -> dict[str, float]:
    """Validate judge consistency and accuracy before training.

    Args:
        judge_prompt: The system prompt to validate.
        validation_examples: List of dicts with keys:
            - 'trajectories': list of trajectories
            - 'task': task description
            - 'expected_ranking': list of trajectory indices from best to worst
        judge_model: Model to use as judge.

    Returns:
        Dict with validation metrics:
            - 'rank_correlation': Spearman correlation with expected rankings
            - 'consistency': Score variance across repeated runs (lower is better)
    """
    from scipy.stats import spearmanr

    all_correlations = []
    all_variances = []

    for example in validation_examples:
        trajectories = example["trajectories"]
        task = example["task"]
        expected_best_to_worst = example["expected_ranking"]

        # Score twice to check consistency
        scores_run1 = await ruler_score_group(trajectories, task, judge_model)
        scores_run2 = await ruler_score_group(trajectories, task, judge_model)

        # Convert expected ranking to expected scores (best gets highest score)
        n = len(trajectories)
        expected_scores = [0.0] * n
        for rank, idx in enumerate(expected_best_to_worst):
            expected_scores[idx] = 1.0 - (rank / (n - 1))

        # Compute rank correlation
        corr, _ = spearmanr(scores_run1, expected_scores)
        all_correlations.append(corr)

        # Compute inter-run consistency
        variance = sum((s1 - s2) ** 2 for s1, s2 in zip(scores_run1, scores_run2)) / n
        all_variances.append(variance)

    return {
        "rank_correlation": sum(all_correlations) / len(all_correlations),
        "consistency": sum(all_variances) / len(all_variances),
    }


# Minimum acceptable thresholds
JUDGE_VALIDATION_THRESHOLDS = {
    "rank_correlation": 0.7,   # Spearman rho — judge agrees with human ranking 70%+
    "consistency": 0.05,        # Variance across runs — judge is stable
}
```

A judge that fails these thresholds will produce noisy training signal. Revise the prompt before running a full training job.

## How RULER Scores Feed Into GRPO

```python
async def training_step_with_ruler(
    prompt: dict,
    policy_model,
    judge_model: str = "o4-mini",
    n_trajectories: int = 4,
) -> dict:
    """One GRPO training step using RULER rewards.

    Args:
        prompt: Training prompt with keys 'messages' and 'task_description'.
        policy_model: The model being trained (ART client).
        judge_model: Judge model identifier.
        n_trajectories: Number of trajectories to sample per step (GRPO group size).

    Returns:
        Dict with 'trajectories', 'rewards', and 'advantages' for the optimizer.
    """
    # Step 1: Generate group of trajectories from current policy
    trajectories = await asyncio.gather(*[
        policy_model.sample(prompt["messages"])
        for _ in range(n_trajectories)
    ])

    # Step 2: Score all trajectories with RULER
    rewards = await ruler_score_group(
        trajectories=list(trajectories),
        task_description=prompt["task_description"],
        judge_model=judge_model,
    )

    # Step 3: Compute GRPO advantages (relative to group mean)
    mean_reward = sum(rewards) / len(rewards)
    advantages = [r - mean_reward for r in rewards]

    return {
        "trajectories": trajectories,
        "rewards": rewards,
        "advantages": advantages,
        # advantages feed directly into GRPO policy gradient update
    }
```

## Common Pitfalls

**Pitfall 1: Sending too many trajectories to the judge at once**
With N=8 or more, the judge's context window fills up and it loses track of earlier trajectories. Keep N=4 for most judge models; N=8 only with 128K+ context models.

**Pitfall 2: Letting the trainee model become the judge**
The judge must be a fixed, non-trainable reference. If the agent can "game" the judge by learning its biases, training diverges.

**Pitfall 3: Not handling judge API failures gracefully**
LLM APIs fail. Return neutral scores (0.5 for all) on failure rather than crashing the training run. Log failures and monitor the failure rate.

**Pitfall 4: Using the same examples for judge validation and agent training**
The judge might inadvertently "know" the right answers for training examples. Use a separate held-out set for judge validation.

## Connections

- **Builds on:** Guide 01 (why manual rewards fail), Module 01 (GRPO advantage computation)
- **Leads to:** Guide 03 (hybrid rewards combining RULER with programmatic checks)
- **Related to:** RLHF reward models (RULER is lighter-weight and task-flexible)

## Practice Questions

1. Why is temperature=0.0 important for the judge LLM, and what might happen if you use temperature=1.0?

2. A group of 4 trajectories receives scores [0.5, 0.5, 0.5, 0.5] from the judge. What are the GRPO advantages? Why is this a problem?

3. Your judge has rank_correlation=0.4 on the validation set. What does this mean, and what are two ways you might fix it?

## Further Reading

- OpenPipe ART documentation — RULER implementation and configuration reference
- "A Survey of Reinforcement Learning from Human Feedback" (Casper et al., 2023) — context for LLM-as-judge relative to RLHF
- "Judging the Judges" (Zheng et al., 2023) — empirical analysis of LLM judge reliability
- "Constitutional AI" (Bai et al., 2022) — related approach using LLM self-critique for reward generation
