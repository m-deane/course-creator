# Additional Readings — Module 02: ART Framework

## Primary References

### ART Repository and Documentation

**[ART GitHub Repository](https://github.com/OpenPipe/ART)**
The primary source. Contains the full source code, example training scripts (tic-tac-toe, email retrieval, open deep research), and issue tracker. The `examples/` directory is the most useful part for course participants — each example is a complete end-to-end training run.

**[ART Official Documentation](https://art.openpipe.ai)**
Covers the client API, backend configuration, RULER integration, and FAQ. The FAQ page is particularly useful — it addresses common questions about GPU requirements, supported models, and the difference between local and serverless backends.

**[ART FAQ](https://art.openpipe.ai/getting-started/faq)**
Direct link to the FAQ. Answers common questions including: "Can I run ART on a single GPU?", "What models are supported?", "What is the difference between the client and the backend?", and "How does RULER work?"

**[ART Client API Reference](https://art.openpipe.ai/fundamentals/art-client)**
Documents `art.Trajectory`, `art.TrajectoryGroup`, `art.TrainableModel`, `art.TrainConfig`, and `art.gather_trajectory_groups`. The reference for every API call made in this module.

---

## OpenPipe Blog Posts

**[ART Trainer: A New RL Trainer for Agents](https://openpipe.ai/blog/art-trainer-a-new-rl-trainer-for-agents)**
The original announcement post by the OpenPipe team. Explains the design rationale — why existing RL frameworks fail for multi-step agents and what architectural decisions ART makes to address those failures. Read this after the guides to see how the design decisions were motivated.

**[Building ART·E: Reinforcement Learning for Email Search](https://www.zenml.io/llmops-database/building-art-e-reinforcement-learning-for-email-search-agent-development)**
Case study of training Qwen 2.5 14B to beat OpenAI's o3 at email retrieval. Covers scenario design, reward function construction, and the training infrastructure used (single H100 via RunPod, ~$80 total). Demonstrates all Module 02 concepts in a real production context.

---

## Backend Components

### vLLM

**[vLLM Documentation](https://docs.vllm.ai/en/latest/)**
Official documentation for the inference engine ART uses. Relevant sections:
- [Installation Guide](https://docs.vllm.ai/en/latest/getting_started/installation/) — GPU requirements, CUDA versions, Docker setup
- [GPU Sizing Guide](https://www.digitalocean.com/community/conceptual-articles/vllm-gpu-sizing-configuration-guide) — how to select GPU VRAM based on model size and context length

**[vLLM GitHub](https://github.com/vllm-project/vllm)**
Source code and releases. Useful for checking which version of vLLM is compatible with a given CUDA version.

### Unsloth

**[Unsloth Documentation — RL Training Guide](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/training-ai-agents-with-rl)**
Unsloth's own guide to training agents with RL, including integration with ART. Covers memory optimization techniques that make training 7B models possible on 24 GB VRAM.

**[Unsloth GitHub](https://github.com/unslothai/unsloth)**
Source code and installation instructions. The install command with explicit CUDA variant (`cu121-torch240`) is on the README.

**[openpipe-art on PyPI](https://pypi.org/project/openpipe-art/)**
Release history and version notes. Check here when upgrading ART to see what has changed.

---

## Integrations

**[LangGraph ART Integration](https://github.com/OpenPipe/ART)**
The ART repository includes LangGraph integration examples. Search for `langgraph` in the repository to find the relevant helper utilities for extracting message history from LangGraph state into Trajectory format.

**[NVIDIA NeMo Agent Toolkit — GRPO with OpenPipe ART](https://docs.nvidia.com/nemo/agent-toolkit/latest/improve-workflows/finetuning/rl_with_openpipe.html)**
NVIDIA's integration guide for using ART within the NeMo ecosystem. Relevant for practitioners in enterprise environments using NVIDIA infrastructure.

**[W&B ART Tutorial](https://wandb.ai/onlineinference/genai-research/reports/Tutorial-The-OpenPipe-ART-project--VmlldzoxMzcxMDQyMg)**
Weights and Biases walkthrough of the ART framework including the serverless backend option. Good reference for the observability integrations (W&B experiment tracking, Langfuse tracing).

---

## Background Reading

**[GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)**
The paper introducing GRPO (DeepSeek Math, 2024). ART implements GRPO as its training algorithm. Module 01 covers GRPO theory; this paper is the academic reference for students who want the formal treatment.

**[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)**
The original LoRA paper. Understanding LoRA's structure (rank decomposition matrices added to frozen weights) explains why ART's checkpoint hot-swapping is fast.

**[Reinforcement Learning for Long-Horizon Multi-Turn Search Agents](https://arxiv.org/html/2510.24126v1)**
Research paper demonstrating RL training for agents that require many tool calls over extended episodes — directly relevant to the type of agents ART is designed to train.

---

## Cloud Providers for GPU Rental

When running ART's backend on cloud infrastructure:

- **[RunPod](https://runpod.io)** — Commonly used by the ART team for training runs. H100 and A100 instances available.
- **[Lambda Labs](https://lambdalabs.com)** — Competitive pricing on A100 instances. Good for longer training runs.
- **[SkyPilot](https://skypilot.readthedocs.io)** — Multi-cloud orchestration. Automatically finds cheapest available GPU across AWS, GCP, Azure, Lambda, and RunPod. Useful for spot instances.
- **[Google Colab Pro+](https://colab.research.google.com)** — A100 access. Suitable for 7B model training with ART's memory optimizations.

For the course exercises using a 7B model, a single A10G or RTX 3090 (24 GB VRAM) is sufficient. Expected cost for a 50-step training run: $5–15 depending on provider and instance type.
