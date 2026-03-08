# ART Installation and Configuration

## In Brief

ART installs as a single Python package (`openpipe-art`). The backend requires a CUDA-capable GPU (compute capability 7.0+) running Linux. Configuration is handled through `art.TrainableModel` and `art.TrainConfig` objects rather than config files.

## Key Insight

ART's defaults are well-tuned. For a first training run, the only required configuration decisions are: which base model to use, and how many rollouts per scenario to run per step. Everything else has been set to values that work across a wide range of agentic tasks.

---

## Prerequisites

Before installing ART:

- **Python:** 3.9 or higher
- **OS:** Linux (for the backend with GPU training; client can run on macOS or Windows)
- **CUDA:** 12.1 (required by vLLM's prebuilt wheels)
- **GPU:** NVIDIA with compute capability 7.0+ (V100, T4, RTX 20xx, A100, L4, H100)
- **VRAM:** 16 GB minimum for 3B models; 24 GB for 7B models; 80 GB for 14B models at full precision

For the client only (no training on this machine), no GPU is required.

---

## Installation

### Client (runs anywhere)

```bash
pip install openpipe-art
```

This installs the ART client library, which includes `art.Trajectory`, `art.TrajectoryGroup`, `art.TrainableModel`, and the OpenAI-compatible client wrapper.

### Backend (runs on GPU machine)

The backend requires vLLM and Unsloth in addition to the ART package. Install in this order to avoid CUDA version conflicts:

```bash
# Step 1: Install vLLM (requires CUDA 12.1)
pip install vllm==0.6.6

# Step 2: Install Unsloth with the matching CUDA version
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"

# Step 3: Install TRL (transformer reinforcement learning)
pip install trl==0.12.0

# Step 4: Install ART
pip install openpipe-art
```

Verify the installation:

```python
import art
import vllm
import unsloth

print(f"ART version: {art.__version__}")
print(f"vLLM version: {vllm.__version__}")
# If this runs without error, the installation is correct
```

### Docker Installation (Recommended for Backend)

Using the vLLM Docker image avoids CUDA driver conflicts:

```bash
# Pull the vLLM base image with CUDA 12.1
docker pull vllm/vllm-openai:v0.6.6

# Run with GPU access and shared memory for vLLM
docker run --gpus all \
  --shm-size 8g \
  -v $(pwd):/workspace \
  -p 8000:8000 \
  vllm/vllm-openai:v0.6.6 \
  /bin/bash

# Inside the container, install ART
pip install openpipe-art "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

---

## Model Selection

ART supports any model compatible with both vLLM and Unsloth. The recommended models are:

| Model | Parameters | Min VRAM | Best For |
|-------|-----------|----------|----------|
| `Qwen/Qwen2.5-3B-Instruct` | 3B | 8 GB | Experimentation, Colab free tier |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | 16 GB | Standard training runs |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | 32 GB | High-quality agents |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | 20 GB | Llama ecosystem |
| `meta-llama/Llama-3.3-70B-Instruct` | 70B | 80 GB+ | Production quality |

**Recommendation for this course:** Start with `Qwen/Qwen2.5-7B-Instruct`. The Qwen 2.5 family was trained with strong instruction following and tool-calling capabilities, which accelerates RL training because the base model already has relevant behaviors to reinforce.

Note: Gemma 3 is not currently supported by Unsloth and should not be used with ART.

---

## Complete Configuration Setup

Here is a complete training script showing all configuration options:

```python
import asyncio
import art

# ---------------------------------------------------------
# 1. Define the trainable model
# ---------------------------------------------------------
model = art.TrainableModel(
    name="sql-agent-v1",            # identifier for this run
    project="text-to-sql",          # groups related runs together
    base_model="Qwen/Qwen2.5-7B-Instruct",  # HuggingFace model ID
)

# ---------------------------------------------------------
# 2. Configure and register a backend
# ---------------------------------------------------------
# Option A: Local backend (backend runs on the same machine)
from art.local.backend import LocalBackend

backend = LocalBackend(
    # Directory where LoRA checkpoints are saved between steps
    checkpoint_dir="./checkpoints/sql-agent-v1",
    # Number of GPU IDs to use (0 = first GPU)
    gpu_ids=[0],
)

# Option B: Serverless backend (W&B manages GPU infrastructure)
# from art.serverless.backend import ServerlessBackend
# backend = ServerlessBackend(api_key="your_wandb_api_key")

model.register(backend)

# ---------------------------------------------------------
# 3. Define training scenarios
# ---------------------------------------------------------
# Scenarios are the training examples your agent will practice on.
# Provide at least 10 diverse scenarios for meaningful learning.
SCENARIOS = [
    {
        "question": "How many customers made a purchase last month?",
        "schema": "customers(id, name), orders(id, customer_id, date, amount)",
        "expected_sql": "SELECT COUNT(DISTINCT customer_id) FROM orders WHERE date >= date('now', '-1 month')",
    },
    {
        "question": "What is the total revenue for each product category?",
        "schema": "products(id, name, category), order_items(order_id, product_id, quantity, price)",
        "expected_sql": "SELECT p.category, SUM(oi.quantity * oi.price) FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.category",
    },
    # Add more scenarios...
]

# ---------------------------------------------------------
# 4. Define the rollout function
# ---------------------------------------------------------
async def rollout(model: art.Model, scenario: dict) -> art.Trajectory:
    """Run the agent on one scenario and return a scored trajectory."""
    client = model.openai_client()

    system_prompt = f"""You are a SQL expert. Given a database schema and a question,
write a single SQL query that answers the question.
Respond with only the SQL query, no explanation.

Schema: {scenario['schema']}"""

    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": scenario["question"]},
        ]
    )

    completion = await client.chat.completions.create(
        model=model.get_inference_name(),
        messages=trajectory.messages(),
        max_tokens=256,
        temperature=0.8,   # Use non-zero temperature during training for exploration
    )

    choice = completion.choices[0]
    trajectory.messages_and_choices.append(choice)

    # Score: exact match gives 1.0, no match gives 0.0
    # In practice, use SQL execution comparison for better signal
    generated_sql = choice.message.content.strip()
    expected_sql = scenario["expected_sql"].strip()
    trajectory.reward = 1.0 if generated_sql.lower() == expected_sql.lower() else 0.0

    return trajectory

# ---------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------
async def train():
    num_steps = 50                  # total training iterations
    rollouts_per_scenario = 8       # trajectories per scenario per step (GRPO group size)

    for step in range(await model.get_step(), num_steps):
        print(f"\n=== Step {step + 1}/{num_steps} ===")

        # Gather trajectory groups: each scenario runs rollouts_per_scenario times
        # ART runs these in parallel for efficiency
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, scenario) for _ in range(rollouts_per_scenario)
                )
                for scenario in SCENARIOS
            ),
            pbar_desc=f"step {step + 1} rollouts",
        )

        # Send groups to backend for GRPO training
        await model.train(
            train_groups,
            config=art.TrainConfig(
                learning_rate=1e-5,          # lower LR = more stable but slower
                num_epochs=1,                # passes over the trajectory batch
                max_grad_norm=0.1,           # gradient clipping
            ),
        )
        print(f"Step {step + 1} complete. Model checkpoint updated.")

asyncio.run(train())
```

---

## Training Hyperparameters

The `art.TrainConfig` object controls the GRPO training step:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `1e-5` | Step size for gradient updates. Lower values (1e-6) give more stable training; higher values (5e-5) can diverge. Start with the default. |
| `num_epochs` | `1` | How many passes over the trajectory batch per training step. Values above 1 can cause overfitting to the current batch. |
| `max_grad_norm` | `0.1` | Gradient clipping threshold. Prevents catastrophic forgetting. Keep this low for stable RL training. |
| `beta` | `0.04` | KL divergence penalty weight. Controls how far the policy is allowed to drift from the base model per step. Higher values keep the model closer to the base. |

Recommended starting config for most agentic tasks:

```python
config = art.TrainConfig(
    learning_rate=1e-5,
    num_epochs=1,
    max_grad_norm=0.1,
    beta=0.04,
)
```

---

## Setting Up the Judge LLM for RULER

RULER (Relative Universal LLM Evaluator Rewards) uses a judge LLM to score trajectories automatically, replacing hand-coded reward functions. This is covered in detail in Module 03, but here is the configuration setup:

```python
import os
from art.rewards import ruler_score_group

# RULER calls an LLM to judge trajectory quality.
# Configure the judge model via environment variable.
# OpenAI models work best as judges due to instruction following quality.
os.environ["OPENAI_API_KEY"] = "your-api-key"

# During gather_trajectory_groups, pass ruler_score_group as the after_each callback.
# This scores each group as it completes, before sending to training.
train_groups = await art.gather_trajectory_groups(
    (
        art.TrajectoryGroup(
            rollout(model, scenario) for _ in range(8)
        )
        for scenario in SCENARIOS
    ),
    after_each=lambda group: ruler_score_group(
        group,
        judge_model="openai/gpt-4o",     # model used to score trajectories
        swallow_exceptions=True,          # don't crash if judge call fails
    ),
    pbar_desc="rollouts with RULER scoring",
)
```

RULER replaces manually assigned `trajectory.reward` values with LLM-judged scores. When using RULER, do not assign `trajectory.reward` in the rollout function — the judge handles it.

---

## GPU Requirements Summary

| Scenario | GPU | VRAM | Cost Estimate |
|----------|-----|------|--------------|
| 3B model experimentation | T4 (Colab free) | 15 GB | Free |
| 7B model standard training | A10G, RTX 3090 | 24 GB | ~$1/hr cloud |
| 14B model production | A100 80GB, H100 | 40+ GB | ~$3–5/hr cloud |
| Multi-GPU 70B model | 4x A100 or 8x H100 | 320+ GB | ~$20+/hr cloud |

Cloud providers for GPU rental: RunPod, Lambda Labs, SkyPilot (multi-cloud), Google Colab Pro+.

ART's serverless backend (via W&B) manages GPU provisioning automatically — use it during development to avoid infrastructure setup time.

---

## Common Installation Errors

**CUDA version mismatch:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
Solution: Ensure your CUDA driver matches CUDA 12.1. Run `nvidia-smi` and check the CUDA version displayed. If different, install the matching vLLM wheel.

**Unsloth import error after vLLM install:**
```
ImportError: cannot import name 'FastLanguageModel' from 'unsloth'
```
Solution: Unsloth must be installed from source with the exact CUDA variant specified. Use the `git+https://` install command shown above, not `pip install unsloth`.

**Out of memory during training:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
Solution: Reduce `max_tokens` in rollout completions (shorter trajectories use less KV cache), or use a smaller model. ART's KV cache offloading activates automatically — confirm the backend was initialized with GPU access.

---

## Connections

- **Builds on:** Guide 01 — ART Architecture (understanding what each component does before configuring it)
- **Leads to:** Guide 03 — Trajectories (using `art.Trajectory` in rollout functions)
- **Leads to:** Module 03 — RULER Rewards (RULER judge configuration)

## Further Reading

- [ART Documentation](https://art.openpipe.ai) — official installation guide and backend configuration options
- [vLLM Installation Guide](https://docs.vllm.ai/en/latest/getting_started/installation/) — GPU requirements and CUDA compatibility matrix
- [Unsloth Documentation](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/training-ai-agents-with-rl) — RL training with Unsloth
- [openpipe-art on PyPI](https://pypi.org/project/openpipe-art/) — latest version and release notes
