# ART Architecture: Client, Backend, and the Training Loop

## In Brief

ART (Agent Reinforcement Trainer) by OpenPipe splits the GRPO training loop into two independent halves: a **client** that runs your agent code and a **backend** that handles GPU inference and weight updates. This separation lets agent code run anywhere — your laptop, a cloud VM, a LangGraph workflow — while expensive GPU operations stay on dedicated hardware.

## Key Insight

Every existing RL framework for LLMs was designed for single-turn interactions: give the model a prompt, get one completion, score it. Agents don't work that way. They call tools, receive results, reason across multiple turns, and accumulate context over many steps. ART was built specifically for this structure.

The name captures the philosophy: give your agents **on-the-job training**, not classroom instruction.

---

## The Full Training Loop

Before examining each component, see the complete loop:

```
┌─────────────────────────────────────────────────────────┐
│                      CLIENT                              │
│                                                         │
│  Scenario ──► rollout() function                        │
│                   │                                     │
│                   ▼                                     │
│  art.Trajectory (messages accumulate)                   │
│       │  ◄── inference requests ──► BACKEND vLLM        │
│       │                                                 │
│  reward = score(trajectory)                             │
│       │                                                 │
│  art.TrajectoryGroup (N trajectories, same scenario)    │
│       │                                                 │
│  model.train(groups) ──────────────────────────────►   │
└─────────────────────────────────────────────────────────┘
                                          │
┌─────────────────────────────────────────▼───────────────┐
│                      BACKEND                             │
│                                                         │
│  Receive TrajectoryGroups                               │
│       │                                                 │
│  Block inference (pause rollouts)                       │
│       │                                                 │
│  GRPO trainer (Unsloth + TRL)                           │
│       │  uses relative advantages from group rewards    │
│       │                                                 │
│  Save new LoRA checkpoint to disk                       │
│       │                                                 │
│  Hot-swap checkpoint into vLLM                          │
│       │                                                 │
│  Unblock inference                                      │
│       │                                                 │
│  Return to client ──► next iteration begins             │
└─────────────────────────────────────────────────────────┘
```

This loop runs for a configured number of steps. Each iteration uses a slightly better model than the one before.

---

## The Client

The client is where your agent code lives. It is responsible for:

1. **Defining rollout functions** — Python async functions that run your agent against a scenario
2. **Building Trajectories** — accumulating messages, tool calls, and tool results into `art.Trajectory` objects
3. **Assigning rewards** — computing a score for each completed trajectory
4. **Grouping trajectories** — using `art.TrajectoryGroup` to bundle multiple runs of the same scenario
5. **Triggering training** — calling `model.train(groups)` to send data to the backend

The client communicates with the backend through an OpenAI-compatible HTTP endpoint. This means any code that works with the OpenAI Python SDK works with ART with minimal changes:

```python
import art

model = art.TrainableModel(
    name="my-agent",
    project="sql-query-agent",
    base_model="Qwen/Qwen2.5-7B-Instruct",
)

async def rollout(model: art.Model, scenario: dict) -> art.Trajectory:
    # Get an OpenAI-compatible client pointing at the ART backend
    client = model.openai_client()

    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": "You are a SQL expert."},
            {"role": "user", "content": scenario["question"]},
        ]
    )

    # This call goes to the vLLM server on your backend, not openai.com
    completion = await client.chat.completions.create(
        model=model.get_inference_name(),
        messages=trajectory.messages(),
    )

    trajectory.messages_and_choices.append(completion.choices[0])

    # Score the result against the expected SQL query
    trajectory.reward = 1.0 if check_sql(completion, scenario["expected"]) else 0.0

    return trajectory
```

The client runs wherever Python runs. It does not require a GPU.

---

## The Backend

The backend is a long-running service that manages two subsystems:

### vLLM Inference Engine

vLLM serves the current model weights through an OpenAI-compatible API. During rollouts, the client sends chat completion requests here. vLLM is optimized for high-throughput batched inference and returns log probabilities, which GRPO requires for the policy gradient computation.

The backend is initialized with a base model. On the first training step, inference uses the base model directly. After the first weight update, a LoRA adapter is loaded. Every subsequent step uses the latest adapter on top of the frozen base weights.

### Unsloth + TRL GRPO Trainer

After each batch of trajectory groups arrives from the client, the backend:

1. Pauses inference (blocks new rollout requests)
2. Runs the GRPO update using Unsloth's memory-efficient training
3. Saves the new LoRA checkpoint to a local directory
4. Hot-swaps the checkpoint into the running vLLM server
5. Resumes inference

This hot-swap is the key technical achievement. The base model weights stay loaded in VRAM throughout training. Only the small LoRA adapter (typically < 2% of total parameters) gets replaced between steps. Swap time is seconds, not minutes.

**Memory optimization:** ART offloads vLLM's KV cache to CPU during the training phase. This recovers up to 4 GB of VRAM on a 7B model with 8K context, making it possible to train 7B models on a single GPU with 24 GB VRAM.

---

## LoRA Checkpoint Hot-Swapping

LoRA (Low-Rank Adaptation) works by keeping the original model weights frozen and adding small trainable rank-decomposition matrices at each attention layer:

```
Output = W₀x + BAx
         ────    ──
         frozen  LoRA adapter (trained)
```

Where `W₀` is the original weight matrix, `B` and `A` are the low-rank matrices added by LoRA, and `x` is the input.

Because the adapter is small (a few hundred MB for a 7B model), ART can:

- Keep `W₀` loaded in VRAM permanently — no model reload between steps
- Train only `B` and `A` — dramatically faster than full fine-tuning
- Swap the adapter after each step — the next batch of rollouts immediately uses improved weights

This is what enables the training loop to be tight. Each step: rollouts → group → train → swap → rollouts.

---

## Native Tool Call Support

ART's Trajectory structure handles tool calls natively. A multi-turn trajectory with tools looks like:

```
messages_and_choices = [
    {"role": "system",    "content": "You have access to a SQL database."},
    {"role": "user",      "content": "How many orders came in last week?"},
    <assistant choice with tool_calls=[{name: "run_sql", arguments: {...}}]>,
    {"role": "tool",      "content": "42", "tool_call_id": "call_abc"},
    <assistant choice with content: "There were 42 orders last week.">,
]
```

The assistant turns are stored as `Choice` objects (from the OpenAI SDK), not plain dicts. ART serializes these for the GRPO trainer, which sees the full sequence including tool invocations as part of the policy's output.

This is what distinguishes ART from single-turn trainers: the entire tool-calling episode is treated as one training example, not just the final text response.

---

## Framework Integrations

ART's OpenAI-compatible client means agent frameworks that already use OpenAI-format APIs require minimal changes:

### LangGraph

LangGraph agents that use a `ChatOpenAI` node can be retargeted at ART's backend by changing the base URL. ART provides helper utilities for extracting messages from LangGraph state objects into Trajectory format.

### CrewAI

CrewAI agents can be wrapped in a rollout function. The agent's internal LLM calls are routed through ART's endpoint by configuring the CrewAI LLM provider to use a custom base URL.

### Google ADK (Agent Development Kit)

ADK agents similarly expose a model configuration that can be pointed at an OpenAI-compatible endpoint, making ART integration straightforward.

The underlying principle: any framework that calls an OpenAI-compatible `/v1/chat/completions` endpoint works with ART. The client intercepts those calls and routes them through the backend for both inference and logging.

---

## What Makes ART Different

Most RL frameworks for LLMs were designed for a specific interaction pattern:

```
Prompt → Single LLM output → Score → Done
```

This works for chatbots and math problem solvers where one completion is the complete answer. It breaks down for agents that:

- Make tool calls and receive results before knowing whether the approach worked
- Accumulate context across 5, 10, or 20 turns before reaching a conclusion
- Succeed or fail based on the cumulative effect of many intermediate decisions

ART's trajectory structure captures the full episode. The reward is assigned to the complete run, and GRPO's credit assignment propagates learning signals back through every decision in the sequence.

Existing frameworks that handle multi-turn conversations do exist, but they typically require the agent environment to be serialized and managed by the training framework itself. ART inverts this: the agent code runs in a normal Python process, and ART observes it through its OpenAI-compatible wrapper.

---

## Common Pitfalls

- **Running client and backend on the same machine without enough VRAM:** The backend needs dedicated GPU memory. On a machine with a single 24 GB GPU, the backend occupies most of it during training. Rollouts that fire during training will be queued, not dropped.
- **Not assigning a reward before the trajectory is sent:** If `trajectory.reward` is `None` when the group is submitted for training, ART will raise an error. Always assign the reward before returning from the rollout function.
- **Using a base model not supported by Unsloth:** ART's trainer layer uses Unsloth for memory efficiency. Gemma 3 and some other architectures are not currently supported. Stick to Qwen 2.5, Qwen 3, or Llama 3.x for reliable support.

---

## Connections

- **Builds on:** Module 01 — GRPO Algorithm (the backend runs GRPO; group structure maps directly to TrajectoryGroup)
- **Leads to:** Module 03 — RULER Rewards (automatic reward functions replace manual `trajectory.reward` assignment)
- **Leads to:** Module 05 — Training Loop (full orchestration with scenarios, checkpoints, and evaluation)

## Further Reading

- [ART GitHub Repository](https://github.com/OpenPipe/ART) — source code and examples
- [ART Documentation](https://art.openpipe.ai) — official docs including FAQ and architecture notes
- [OpenPipe Blog: ART Trainer](https://openpipe.ai/blog/art-trainer-a-new-rl-trainer-for-agents) — original announcement with design rationale
- [ART·E Case Study](https://www.zenml.io/llmops-database/building-art-e-reinforcement-learning-for-email-search-agent-development) — training a 14B model to beat o3 at email retrieval for $80
