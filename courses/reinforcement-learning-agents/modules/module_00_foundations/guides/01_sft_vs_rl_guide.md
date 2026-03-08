# Guide 01: Supervised Fine-Tuning vs Reinforcement Learning

## In Brief

Supervised fine-tuning teaches a model to reproduce examples. Reinforcement learning teaches a model to achieve outcomes. For single-turn tasks these differ little in practice; for multi-step tool-using agents the distinction is decisive.

---

## Key Insight

**SFT trains a model to say the right things. RL trains a model to do the right things.**

A model that has learned to imitate expert demonstrations will produce plausible-looking sequences. But plausibility is not success. When an agent must execute a ten-step reasoning chain, call three different APIs in the right order, and recover from intermediate errors, imitating what an expert would say at each step is not the same as learning what actually works.

---

## What SFT Does

Supervised fine-tuning is a form of imitation learning. You collect pairs of (input, desired output) and train the model to maximize the probability of the desired output given the input.

### The data format

```python
# SFT training example: each sample is an (input, output) pair
sft_example = {
    "input": (
        "You are a helpful assistant with access to a calculator tool.\n"
        "User: What is the compound interest on $10,000 at 5% for 3 years?\n"
    ),
    "output": (
        "I'll calculate this using the compound interest formula.\n"
        "<tool_call>calculator(principal=10000, rate=0.05, years=3, compound='annual')</tool_call>\n"
        "<tool_result>11576.25</tool_result>\n"
        "The compound interest is $1,576.25, giving a final balance of $11,576.25."
    ),
}

# The training objective: maximize log P(output | input)
# Equivalently: minimize cross-entropy loss on each output token
def sft_loss(logits, target_tokens):
    """
    logits: model predictions, shape (seq_len, vocab_size)
    target_tokens: ground truth token ids, shape (seq_len,)

    The model learns: "given this input, these are the tokens I should produce."
    It does NOT learn: "producing these tokens leads to a successful outcome."
    """
    import torch
    import torch.nn.functional as F
    return F.cross_entropy(logits, target_tokens)
```

### What the model actually learns

The model learns a conditional distribution: given the prompt so far, what token comes next? It approximates the distribution of the expert demonstrations. After training, it can produce outputs that look like expert outputs.

This is powerful. SFT is how instruction-following models are created. For many tasks — summarization, translation, single-turn Q&A — it works well because a single high-quality response is a complete solution.

---

## Where SFT Breaks Down for Agents

### The compounding error problem

In a multi-step agent task, each action changes the state of the world. SFT assumes that if the model produces action $a_1$ correctly, then action $a_2$ correctly, and so on, the task will succeed.

But this ignores that errors compound. If the model deviates from the expert trajectory at step 3 (perhaps by calling the wrong API or misinterpreting a tool result), it finds itself in a state that never appeared in training. It has learned nothing about how to recover, because the training data contained only successful trajectories.

```python
# Illustration: compounding error in a multi-step agent
# In SFT training, all trajectories are "correct." The model never sees recovery.

expert_trajectory = [
    "step_1: search(query='Q3 revenue Apple')",     # correct
    "step_2: parse_result(result_id=0)",              # correct
    "step_3: calculate(op='sum', values=[...])",      # correct
    "step_4: format_answer(value=394.33, unit='B')", # correct — task succeeds
]

# What happens at inference when step 2 fails:
agent_trajectory = [
    "step_1: search(query='Q3 revenue Apple')",      # correct — same as training
    "step_2: parse_result(result_id=99)",             # wrong index — off distribution
    "step_3: ???",                                    # model has never seen this state
    # The model hallucinates plausible-looking next steps with no grounding in reality
]
```

### The exploration problem

SFT can only teach what appears in the training data. For novel problems — new tools, new APIs, new reasoning patterns — the model cannot generalize beyond its demonstrations. It cannot discover strategies that were not shown to it.

RL, by contrast, lets the model explore. It tries actions, observes outcomes, and updates toward actions that worked. This is how agents discover novel multi-step strategies.

### The credit assignment problem

Even with perfect expert data, SFT cannot distinguish which steps in a trajectory were critical versus incidental. If an expert always searches before calculating, the model learns to search. But it does not learn *why* searching before calculating leads to success.

RL with reward shaping can assign credit explicitly: this action at this step contributed to the final success.

---

## The Chess Analogy

Think of the difference between:

**SFT approach:** Reading a rulebook and a collection of annotated grandmaster games. You learn what moves look like, the notation, the typical patterns. You can produce moves that look plausible.

**RL approach:** Playing thousands of games and receiving a signal (win/lose/draw) at the end of each one. You discover what works through experience. Over time you develop genuine strategic understanding.

A player trained only by reading annotated games collapses when facing a novel position not covered in their examples. A player who has played thousands of games has internalized what leads to winning, and can generalize.

For agents: the "game" is the task execution trace, and the "win signal" is task completion.

---

## Code Example: SFT Data vs RL Reward Signal

```python
import random
from dataclasses import dataclass
from typing import Callable


# --- SFT paradigm ---

@dataclass
class SFTExample:
    """One training example for supervised fine-tuning."""
    prompt: str
    completion: str
    # Note: no notion of success or failure. Just imitation.


sft_dataset = [
    SFTExample(
        prompt="Search for the current price of AAPL stock.",
        completion="<tool_call>search(query='AAPL stock price today')</tool_call>",
    ),
    SFTExample(
        prompt="Calculate 15% of 847.",
        completion="<tool_call>calculator(expression='847 * 0.15')</tool_call>",
    ),
]

# Training signal: did the model produce the exact tokens in `completion`?
# The loss is the same whether the model almost got it right or was completely wrong,
# as long as the token probabilities are identical.


# --- RL paradigm ---

@dataclass
class AgentRollout:
    """One episode of an agent interacting with an environment."""
    prompt: str
    actions: list[str]       # sequence of actions the agent took
    observations: list[str]  # tool results / environment responses
    final_answer: str
    reward: float            # scalar signal: how well did this work?


def tool_calling_reward(rollout: AgentRollout, ground_truth: str) -> float:
    """
    A reward function for a tool-calling agent.

    Returns a scalar in [0.0, 1.0] measuring how well the agent did.
    The model learns to maximize this — not to copy a fixed output.
    """
    reward = 0.0

    # Correctness is the primary signal
    if ground_truth.lower().strip() in rollout.final_answer.lower().strip():
        reward += 0.6

    # Efficiency: fewer unnecessary tool calls is better
    max_expected_calls = 3
    call_count = len(rollout.actions)
    if call_count <= max_expected_calls:
        reward += 0.2
    elif call_count <= max_expected_calls * 2:
        reward += 0.1  # partial credit for eventually getting there

    # Format compliance: did the agent use the required tool call syntax?
    tool_calls_formatted = all(
        action.startswith("<tool_call>") and action.endswith("</tool_call>")
        for action in rollout.actions
        if action.strip()  # skip empty actions
    )
    if tool_calls_formatted:
        reward += 0.2

    return reward


# Demonstrate the difference in training signal quality

example_rollout = AgentRollout(
    prompt="What is 15% of 847?",
    actions=[
        "<tool_call>calculator(expression='847 * 0.15')</tool_call>",
    ],
    observations=["127.05"],
    final_answer="15% of 847 is 127.05",
    reward=0.0,  # will be computed
)

example_rollout.reward = tool_calling_reward(example_rollout, ground_truth="127.05")
print(f"Rollout reward: {example_rollout.reward:.2f}")  # 1.0 — perfect

bad_rollout = AgentRollout(
    prompt="What is 15% of 847?",
    actions=[
        "<tool_call>search(query='15 percent of 847')</tool_call>",  # wrong tool
        "<tool_call>search(query='calculator online')</tool_call>",   # still wrong
        "<tool_call>calculator(expression='847 / 15')</tool_call>",  # wrong formula
    ],
    observations=["some search results", "more search results", "56.47"],
    final_answer="The answer is approximately 56.47",
    reward=0.0,
)

bad_rollout.reward = tool_calling_reward(bad_rollout, ground_truth="127.05")
print(f"Bad rollout reward: {bad_rollout.reward:.2f}")  # 0.1 — mostly wrong
```

---

## When to Use SFT vs RL

This is a practical decision framework, not a binary choice. In practice the two are complementary.

| Criterion | Lean toward SFT | Lean toward RL |
|-----------|-----------------|----------------|
| Task type | Single-turn, well-specified | Multi-step, open-ended |
| Data availability | Many high-quality demonstrations | Few demonstrations, but can generate rollouts |
| Success definition | Easy to express as a target output | Easy to express as a reward function |
| Agent steps | 1–2 steps | 3+ steps with tool calls |
| Error recovery | Not needed | Critical |
| Novel strategies | Not needed | Desired |

### Practical recipe

```
Phase 1 — SFT:
    Collect ~1,000 expert demonstrations
    Fine-tune to get the model into the right "neighborhood"
    The model now produces plausible formats and uses tools correctly

Phase 2 — RL:
    Generate rollouts from the SFT model
    Score them with a reward function
    Update toward higher-reward behaviors
    The model now learns what actually works, not just what looks right
```

This is the approach used by systems like DeepSeek-R1, Qwen-Agent, and Anthropic's internal agent training pipelines. SFT provides the base; RL provides the optimization pressure toward actual success.

---

## Common Pitfalls

**Pitfall 1: Assuming SFT data quality fixes the problem.** More expert demonstrations do not solve the compounding error problem. At step 8 of a 10-step task, a model trained purely on demonstrations is in distribution only if every prior step matched training exactly.

**Pitfall 2: Designing reward functions that are easy to game.** If you reward "number of tool calls made," the agent learns to make many calls. Reward signals must capture what you actually care about — task success — not proxies for it.

**Pitfall 3: Skipping SFT entirely.** Starting RL from a base model is extremely sample-inefficient. The model spends most of its exploration producing invalid outputs. SFT first, RL second is the standard approach.

---

## Connections

- **Builds on:** Basic familiarity with language model fine-tuning
- **Leads to:** Guide 02 (Reward Signals) — how to design the reward function that drives RL
- **Leads to:** Module 01 (GRPO Algorithm) — the specific RL method used throughout this course
- **Related concept:** RLHF (Reinforcement Learning from Human Feedback) uses the same SFT→RL pipeline, applied to helpfulness rather than task success

---

## Further Reading

- Ouyang et al., "Training language models to follow instructions with human feedback" (2022) — the original InstructGPT paper establishing the SFT→RL pipeline
- DeepSeek-AI, "DeepSeek-R1" (2025) — shows RL enabling emergent reasoning strategies that SFT alone cannot produce
- Sutton & Barto, "Reinforcement Learning: An Introduction" Ch. 1 — the classic motivation for why trial-and-error learning differs from supervised learning
