---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Custom and Hybrid Reward Functions

## When RULER Needs a Partner

Module 03 — RULER Automatic Rewards

<!-- Speaker notes: This final guide in Module 03 covers the practical question of when RULER alone is sufficient and when it needs to be combined with programmatic checks. The core concept is the gating pattern: hard failures get zero reward regardless of quality. This is where the theory becomes engineering. -->

---

## When RULER Alone Is Not Enough

Some tasks have hard correctness requirements that RULER might miss:

- SQL query has a syntax error → execution fails
- API call uses wrong schema → request rejected
- Code doesn't compile → untestable

**The problem with RULER here:**
An agent that writes beautifully reasoned but syntactically broken SQL might receive a high quality score from the judge — the reasoning looks good, even though the output is useless.

**Solution:** Use RULER as a quality signal, but programmatic checks as a gate.

<!-- Speaker notes: This is the key design decision point. RULER judges quality as it would appear to a reader — the judge sees the text and evaluates it. But the judge cannot always know if the SQL actually executes, if the API call would be accepted, if the code compiles. Those require execution. The hybrid approach separates these concerns cleanly. -->

---

## The Decision Tree

```
Task has hard correctness criterion?
        │
        ├── YES → Hybrid reward
        │         programmatic gate + RULER quality
        │
        └── NO  → RULER alone
                  open-ended quality, no single right answer
```

**RULER alone:** Writing, analysis, explanation, creative tasks

**Hybrid:** SQL, code generation, API calls, structured data extraction

<!-- Speaker notes: Walk through a few examples with the class. "Grade a student essay" → RULER alone. "Write SQL to answer this question" → hybrid. "Summarize this document" → RULER alone. "Extract JSON from this text" → hybrid (can validate JSON schema). The presence of an executable checker is the distinguishing factor. -->

---

## Why Multiplication, Not Addition

<div class="columns">

**Addition (wrong)**
$$r = 0.5 \times r_{prog} + 0.5 \times r_{ruler}$$

SQL syntax error ($r_{prog}=0$) + confident wrong answer ($r_{ruler}=0.8$):
$$r = 0.5(0) + 0.5(0.8) = 0.4$$

Agent gets 0.4 reward for broken SQL.

**Multiplication (correct)**
$$r = r_{prog} \times r_{ruler}$$

Same scenario:
$$r = 0 \times 0.8 = 0.0$$

Programmatic gate: fail = zero reward.

</div>

<!-- Speaker notes: This is a concrete, memorable illustration of why the choice of aggregation function matters. With addition, the RULER score partially rescues a completely broken output. With multiplication, a hard failure produces zero reward regardless of how good the rest of the trajectory looks. The agent cannot compensate for execution failure with good reasoning. -->

---

## Hybrid Reward with Modulated Quality

```python
def hybrid_reward(
    programmatic_score: float,  # 0.0 (fail) or 1.0 (pass)
    ruler_score: float,         # 0.0 to 1.0 quality
    ruler_weight: float = 0.3,  # How much quality modulates reward
) -> float:
    if programmatic_score == 0.0:
        return 0.0  # Hard gate: failure = zero

    # Correct response: reward modulated by quality
    quality_component = (1.0 - ruler_weight) + ruler_weight * ruler_score
    return programmatic_score * quality_component
```

Examples with `ruler_weight=0.3`:

| Programmatic | RULER | Hybrid Reward |
|-------------|-------|---------------|
| 0.0 (fail) | 0.9 | **0.0** — fail is fail |
| 1.0 (pass) | 0.4 | **0.82** — correct, mediocre quality |
| 1.0 (pass) | 0.9 | **0.97** — correct, excellent quality |

<!-- Speaker notes: The ruler_weight parameter is a tunable hyperparameter. At ruler_weight=0.0, the reward is binary — only programmatic pass/fail matters. At ruler_weight=1.0, RULER quality fully modulates the score (but the gate still applies). The 0.3 default is a reasonable starting point: correctness matters most, but quality among correct responses is rewarded. -->

---

## Reward Shaping for Multi-Step Agents

Single terminal reward is often too sparse for long trajectories.

```
Step 1: Search web          ← Did this call make sense?
Step 2: Evaluate results    ← Was this judgment sound?
Step 3: Query database      ← Was this query appropriate?
Step 4: Synthesize          ← Was reasoning correct?
Step 5: Return answer       ← TERMINAL: RULER scores this
```

**Safe pattern:**
- Terminal reward: 70% weight (dominates)
- Intermediate process rewards: 30% weight (guidance only)

If intermediate rewards exceed terminal reward weight, the agent optimizes the process and ignores the output quality.

<!-- Speaker notes: The 70/30 split is a guideline, not a law. The principle is that the terminal reward should dominate. Intermediate rewards are training wheels — they provide signal during early training when the agent rarely reaches a good final state, but they should not compete with the terminal objective. A common failure mode is intermediate rewards that are too generous, causing the agent to make unnecessary tool calls to collect process rewards. -->

---

## Testing Reward Functions Before Training

```python
# Define test cases before writing any training code
test_cases = [
    {
        "name": "perfect_sql",
        "trajectory": [{"role": "assistant",
                        "content": "```sql\nSELECT customer_id, SUM(revenue)\n"
                                   "FROM orders GROUP BY customer_id\n"
                                   "ORDER BY 2 DESC LIMIT 5\n```"}],
        "expected_range": (0.75, 1.0),
    },
    {
        "name": "syntax_error",
        "trajectory": [{"role": "assistant",
                        "content": "```sql\nSELECT customer_id FORM orders\n```"}],
        "expected_range": (0.0, 0.0),  # Must be exactly 0
    },
]

# Run before any training
results = test_reward_function_on_cases(db_path="training.db", test_cases=test_cases)
assert all(results.values()), "Fix reward function before training"
```

**Never skip this step.** A broken reward function teaches wrong behavior silently — the loss goes down while the agent learns to do the wrong thing.

<!-- Speaker notes: This is the most important operational point in the module. Reward function bugs are uniquely dangerous: they don't raise exceptions, they don't produce obvious errors, the training metrics look fine. The agent just learns something different from what you intended. Test cases are the only way to catch these bugs before wasting training compute. -->

---

## Common Hybrid Reward Mistakes

<div class="columns">

**Mistake 1: Addition instead of multiplication**
Allows failed outputs to get partial reward from RULER quality.

**Mistake 2: Too many intermediate rewards**
Creates conflicting gradients. Agent optimizes all simultaneously and converges to none.

**Mistake 3: Intermediate rewards dominate terminal**
Agent optimizes process, ignores output quality.

**Mistake 4: Not testing before training**
Broken reward function trains silently wrong behavior.

**Mistake 5: Slow reward computation**
LLM + DB calls take >10 seconds per step → training bottleneck.

</div>

<!-- Speaker notes: Walk through each mistake and ask learners to think about how they would catch it. Mistake 1 is caught by the syntax error test case. Mistake 2 is visible in training: excessive tool calls that don't improve final quality. Mistake 3 shows up as good process metrics with degrading output quality. Mistake 4 requires proactive testing. Mistake 5 shows up in training throughput monitoring. -->

---

## The Complete Reward Decision Guide

| Situation | Approach |
|-----------|---------|
| Open-ended (writing, analysis) | RULER alone |
| Hard correctness criterion | Hybrid: programmatic gate + RULER |
| Multi-step with sub-goals | Step shaping + terminal RULER |
| Need fast training loop | Programmatic only (no LLM call) |
| Binary pass/fail sufficient | Programmatic only |
| Quality matters among correct | Hybrid with higher `ruler_weight` |

**Start simple.** Programmatic-only rewards are fast to build and debug. Add RULER when correct responses vary in quality that matters.

<!-- Speaker notes: This table is meant to be a practical reference. Emphasize the "start simple" guidance — it's easy to overthink reward design. If binary pass/fail gives sufficient training signal (the agent learns to avoid hard failures), the additional complexity of RULER may not be worth it. Add RULER when you observe that all-correct responses differ meaningfully in quality and you want the agent to improve beyond mere correctness. -->

---

<!-- _class: lead -->

## Module 03 Complete

**What you now know:**

1. Why manual reward engineering fails at scale (Guide 01)
2. How RULER uses relative LLM scoring to replace reward functions (Guide 02)
3. How hybrid rewards combine programmatic gates with RULER quality (Guide 03)

**Next:** Module 04 — MCP Integration
Building the tool environment your trained agent will operate in.

**Do the exercise before moving on.**
`exercises/01_ruler_exercise.py`

<!-- Speaker notes: This is the capstone for Module 03. Make sure learners understand all three guides before moving to Module 04 — the concepts here are prerequisite for understanding the full training loop in Module 05. The exercise synthesizes all three guides: implementing a judge, comparing absolute vs relative scoring, and building a hybrid reward. -->

---

## Practice Questions

1. A customer service agent resolves support tickets. Design a hybrid reward: what programmatic checks would you run, and what would you ask the judge to evaluate?

2. Why is `r = programmatic × ruler` correct when SQL has a syntax error, but `r = 0.5 × programmatic + 0.5 × ruler` is not?

3. You add intermediate reward for each web search. The agent starts making 15+ searches per task. What went wrong and how do you fix it?

**Then complete the exercise.**

<!-- Speaker notes: These questions map directly to the three main concepts: hybrid reward design (Q1), the multiplication vs addition principle (Q2), and the danger of excessive intermediate rewards (Q3). Answers: Q1 - check if ticket is marked resolved, customer sentiment positive; judge evaluates tone, accuracy, completeness. Q2 - with syntax error prog=0, so 0×0.8=0 but 0.5(0)+0.5(0.8)=0.4. Q3 - intermediate reward for searches outweighs terminal reward; reduce intermediate weight or cap number of searches. -->
