---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# ART Installation and Configuration

Module 02 — Reinforcement Learning for AI Agents

<!-- Speaker notes: This slide deck is practical and hands-on. Students should have a terminal open alongside this. The goal is a working ART backend by the end of the session. Emphasize that ART's defaults are well-tuned—for a first training run, very little configuration is required. -->

---

## What We're Setting Up

```
Your Machine (Client)           Cloud GPU (Backend)
┌─────────────────┐             ┌──────────────────────┐
│  Python agent   │  HTTP API   │  vLLM inference      │
│  art.Trajectory │◄───────────►│  Unsloth GRPO trainer│
│  rollout()      │             │  LoRA checkpoints    │
└─────────────────┘             └──────────────────────┘
      No GPU needed                  GPU required
      pip install openpipe-art       Full stack install
```

Two installation paths: **client only** or **full backend**.

<!-- Speaker notes: Start by orienting students to what they're actually installing. The client-only install is a standard Python package and takes 30 seconds. The full backend install involves vLLM and Unsloth and requires attention to CUDA versions. Students running in Colab or on a cloud GPU should follow the full backend path. Students who will connect to a remote backend someone else is running only need the client. -->

---

## Prerequisites Checklist

**For client only:**
- Python 3.9+
- Any OS (Linux, macOS, Windows)

**For backend (training):**
- Python 3.9+
- Linux
- CUDA 12.1
- NVIDIA GPU: compute capability 7.0+

| GPU | Compute Cap | Supported |
|-----|------------|-----------|
| V100 | 7.0 | Yes |
| T4 | 7.5 | Yes |
| A100 | 8.0 | Yes |
| H100 | 9.0 | Yes |

<!-- Speaker notes: Compute capability 7.0 is the Volta architecture from 2017. Almost any modern NVIDIA GPU meets this requirement. The CUDA 12.1 requirement is the more common friction point—check with nvidia-smi before starting the install. Students on older systems with CUDA 11.x will need to update their CUDA toolkit or use Docker. -->

---

## Installation: Client

```bash
# All you need for the client
pip install openpipe-art
```

Verify:
```python
import art
print(art.__version__)  # should print without error
```

That's it. The client installs in under a minute.

<!-- Speaker notes: Keep this slide brief. The client install is genuinely simple. The complexity comes in the backend install. Don't dwell here—move to the backend steps where students need guidance. -->

---

## Installation: Backend — Order Matters

```bash
# Step 1: vLLM first (sets CUDA constraints)
pip install vllm==0.6.6

# Step 2: Unsloth with matching CUDA variant
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"

# Step 3: TRL
pip install trl==0.12.0

# Step 4: ART
pip install openpipe-art
```

**Why this order?** vLLM pins specific PyTorch and CUDA versions.
Unsloth must be built against those same versions.

<!-- Speaker notes: The order matters because vLLM installs PyTorch with specific CUDA bindings, and Unsloth must be compiled against those same versions. Installing Unsloth first and then vLLM frequently causes binary compatibility errors. If students encounter import errors after install, the fix is almost always: uninstall everything in this stack, then reinstall in this exact order. Pin the vLLM version to 0.6.6 for this course—later versions may have changed the ART integration. -->

---

## Docker Installation (Recommended)

Avoids CUDA driver conflicts entirely:

```bash
# Pull vLLM base image
docker pull vllm/vllm-openai:v0.6.6

# Run with GPU access
docker run --gpus all \
  --shm-size 8g \
  -v $(pwd):/workspace \
  -p 8000:8000 \
  -it vllm/vllm-openai:v0.6.6 /bin/bash

# Inside container
pip install openpipe-art \
  "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

`--shm-size 8g` is required — vLLM uses shared memory for tensor parallelism.

<!-- Speaker notes: Recommend Docker for anyone who has had CUDA environment problems before. The vLLM Docker image is tested and maintained by the vLLM team—it has the right CUDA toolkit, PyTorch version, and dependencies pre-installed. The only additions needed are ART and Unsloth. The --shm-size flag is critical and easy to forget; without it, vLLM will crash with a shared memory error on multi-GPU setups. -->

---

## Model Selection

| Model | VRAM | Best For |
|-------|------|----------|
| `Qwen/Qwen2.5-3B-Instruct` | 8 GB | Colab free tier, quick experiments |
| `Qwen/Qwen2.5-7B-Instruct` | 16 GB | Standard course exercises |
| `Qwen/Qwen2.5-14B-Instruct` | 32 GB | High-quality production agents |
| `meta-llama/Llama-3.1-8B-Instruct` | 20 GB | Llama ecosystem compatibility |

**This course uses:** `Qwen/Qwen2.5-7B-Instruct`

Qwen 2.5 has strong tool-calling out of the box — less RL needed to reach competence.

**Avoid:** Gemma 3 (not yet supported by Unsloth)

<!-- Speaker notes: The model choice affects both what GPU you need and how fast training converges. Qwen 2.5 is the default recommendation because it was specifically trained with function calling and tool use, which means the base model already has relevant behaviors. RL training reinforces existing capabilities—it can't create them from scratch. Starting with a model that already roughly knows how to call tools means training converges faster and with fewer steps. For students on free Colab, use the 3B model and expect lower final performance but the same learning experience. -->

---

## Complete Training Configuration

```python
import art

# Model definition
model = art.TrainableModel(
    name="sql-agent-v1",       # name this run
    project="text-to-sql",     # group related runs
    base_model="Qwen/Qwen2.5-7B-Instruct",
)

# Backend registration
from art.local.backend import LocalBackend

backend = LocalBackend(
    checkpoint_dir="./checkpoints/sql-agent-v1",
    gpu_ids=[0],
)
model.register(backend)
```

Three decisions: **name**, **base model**, **backend type**.

<!-- Speaker notes: Walk through each parameter. The name and project are for your own tracking—they appear in logs and checkpoint directories. The base_model is a HuggingFace model ID, downloaded automatically on first use. The backend type is either LocalBackend (GPU on this machine) or ServerlessBackend (W&B-managed GPU). For the course exercises, LocalBackend is standard; students without a local GPU should use the serverless option covered in the next slide. -->

---

## Serverless Backend Option

No local GPU? Use W&B's managed infrastructure:

```python
from art.serverless.backend import ServerlessBackend
import os

os.environ["WANDB_API_KEY"] = "your-wandb-api-key"

backend = ServerlessBackend(
    api_key=os.environ["WANDB_API_KEY"]
)
model.register(backend)

# Everything else is identical —
# rollout functions, TrajectoryGroups, model.train() calls
# all work the same way
```

W&B provisions the GPU, manages the vLLM server, and handles checkpoints.
You pay only for training time consumed.

<!-- Speaker notes: The serverless backend is genuinely useful for course work. Students don't need to manage GPU instances, handle CUDA setup, or worry about idle GPU costs. The tradeoff is latency—the serverless backend has higher overhead per step than a local backend—but for learning purposes this is acceptable. The API is identical, which is the point: the client-backend split means switching between local and serverless is just a one-line change in the backend registration. -->

---

## Training Hyperparameters

```python
config = art.TrainConfig(
    learning_rate=1e-5,   # step size — don't go above 5e-5
    num_epochs=1,         # passes per batch — keep at 1
    max_grad_norm=0.1,    # gradient clip — keeps training stable
    beta=0.04,            # KL penalty — how far to drift from base model
)

await model.train(train_groups, config=config)
```

**Start with the defaults.** Only tune if training diverges.

<!-- Speaker notes: RL training is more sensitive to hyperparameters than supervised learning. The default values in ART were tuned by the OpenPipe team across multiple real training runs. The most common student mistake is trying to accelerate training by increasing the learning rate—this frequently causes the policy to collapse or diverge. Keep learning_rate at 1e-5. The beta parameter controls the KL divergence penalty, which prevents the model from straying too far from the base model in a single step. Higher beta = more conservative updates = more stable but slower learning. -->

---

## The RULER Judge Setup

RULER uses an LLM to score trajectories automatically. Configure it with your judge model:

```python
import os
from art.rewards import ruler_score_group

# Judge model credentials
os.environ["OPENAI_API_KEY"] = "sk-..."

# Use as after_each callback in gather_trajectory_groups
train_groups = await art.gather_trajectory_groups(
    (
        art.TrajectoryGroup(rollout(model, s) for _ in range(8))
        for s in scenarios
    ),
    after_each=lambda group: ruler_score_group(
        group,
        judge_model="openai/gpt-4o",
        swallow_exceptions=True,
    ),
)
```

When using RULER: **do not assign trajectory.reward manually.**

<!-- Speaker notes: RULER is covered in depth in Module 03—this slide just shows the setup so students know where it fits in the configuration picture. The key point: when RULER is active, it replaces manual reward assignment. The after_each callback runs on each completed TrajectoryGroup before it's sent to training, scoring all trajectories in the group relative to each other. The swallow_exceptions=True flag prevents a single judge API failure from crashing the entire training run. -->

---

## GPU Requirements at a Glance

| Task | GPU | VRAM | Approx Cost |
|------|-----|------|-------------|
| 3B model, course exercises | T4 (Colab free) | 15 GB | Free |
| 7B model, standard | A10G / RTX 3090 | 24 GB | ~$1/hr |
| 14B model, production | A100 80GB | 40 GB | ~$3–5/hr |
| 70B model | 4x A100 | 320 GB | ~$20+/hr |

**Recommended cloud providers:** RunPod, Lambda Labs, SkyPilot

<!-- Speaker notes: Anchor cost expectations clearly. The $80 cost to train a 14B model that beats o3 wasn't on a cheap instance—it was an H100 for roughly one day. For course exercises using a 7B model, students can complete a meaningful training run in a few hours for $5–10 on a cloud A10G. Colab free tier is viable for the 3B model with reduced training steps. SkyPilot is worth mentioning for students who want to switch between cloud providers based on spot instance availability. -->

---

## Common Errors and Fixes

**CUDA mismatch:**
```
RuntimeError: CUDA error: no kernel image is available
```
Fix: Check `nvidia-smi` for CUDA version. Reinstall vLLM with matching CUDA variant.

**Unsloth import fails:**
```
ImportError: cannot import name 'FastLanguageModel'
```
Fix: Reinstall Unsloth from source with explicit CUDA variant (`cu121-torch240`).

**Out of memory:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
Fix: Reduce `max_tokens` in rollout completions, or switch to a smaller model.

<!-- Speaker notes: These are the three errors students encounter most frequently. Have students check nvidia-smi first before any install troubleshooting—the CUDA version determines everything else. For Unsloth import failures, the fix is almost always the install command: people install with pip install unsloth (without the CUDA specifier) and then get a generic CPU-only build that lacks FastLanguageModel. The OOM error is often fixable by limiting trajectory length before buying a bigger GPU. -->

---

## Module Summary

**Installation:**
- Client: `pip install openpipe-art` (any machine)
- Backend: vLLM → Unsloth → TRL → ART (Linux, CUDA 12.1, GPU)

**Configuration:**
- `art.TrainableModel`: name, project, base model
- `art.LocalBackend` or `art.ServerlessBackend`: GPU location
- `art.TrainConfig`: learning rate, gradient clipping, KL penalty

**Model choice:** Qwen 2.5 7B for this course (good tool-calling base)

**Next:** Trajectories — the data structure that makes this all work.

<!-- Speaker notes: Summarize the three pillars: installation order, model configuration, and backend registration. The next guide on Trajectories is where students start writing agent code that actually runs with ART. Installation was a prerequisite; Trajectories are where the course content begins in earnest. If students had trouble with installation, point them to the exercise file which validates the setup before they attempt trajectory construction. -->

---

<!-- _class: lead -->

# Guide 02 Complete

**Next:** `03_trajectories_guide.md`
What a Trajectory is, how to build one, how GRPO uses it

<!-- Speaker notes: Before moving on, confirm that students have either installed ART or have a plan for accessing a backend. The exercise for this module (exercises/01_art_setup_exercise.py) validates both the config structure and trajectory format—it's a good first check before attempting any live training. -->
