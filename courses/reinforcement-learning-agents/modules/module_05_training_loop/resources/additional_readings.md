# Additional Readings — Module 05: Training Loop Deep-Dive

## Foundational Papers

### GRPO and Group Relative Policy Optimization

**DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**
Sheng et al., 2024
[arXiv:2402.03300](https://arxiv.org/abs/2402.03300)

The paper that introduced GRPO. Section 3.2 describes the group relative policy optimization algorithm, the advantage normalization formula, and the KL divergence penalty structure. Essential reading for understanding why the group-relative approach replaces the value network.

**DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**
DeepSeek-AI, 2025
[arXiv:2501.12948](https://arxiv.org/abs/2501.12948)

Demonstrates GRPO training at scale producing a model that rivals OpenAI o1. Appendix A shows the exact reward function design for mathematical reasoning tasks. Sections 2 and 3 cover the training pipeline, including rollout collection and checkpoint management practices.

---

### LoRA and Parameter-Efficient Fine-Tuning

**LoRA: Low-Rank Adaptation of Large Language Models**
Hu et al., 2021
[arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

The original LoRA paper. Sections 2 and 3 explain the low-rank decomposition ($W + \alpha BA$), why it works, and how rank affects the capacity-to-efficiency tradeoff. Figure 2 shows the memory savings compared to full fine-tuning.

**QLoRA: Efficient Finetuning of Quantized LLMs**
Dettmers et al., 2023
[arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

Extends LoRA with 4-bit quantization of the base model. The combination — quantized base + LoRA adapter — is what Unsloth uses internally. Section 3 explains why quantization of the base model does not harm adapter quality.

---

### Reinforcement Learning for Language Models

**Proximal Policy Optimization Algorithms**
Schulman et al., 2017
[arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

The PPO paper. Compare Algorithm 1 (PPO-Clip) to GRPO: PPO requires a value network to compute advantages, GRPO replaces it with group normalization. Understanding why PPO needs a critic makes GRPO's elimination of it more meaningful.

**Training Language Models to Follow Instructions with Human Feedback**
Ouyang et al., 2022 (InstructGPT)
[arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

The paper that established RLHF as a practical training paradigm. Section 3 describes the PPO training loop for language models, including reward model training, rollout collection, and KL penalty structure — all concepts directly relevant to the Module 05 training loop.

---

## Implementation References

### ART Framework

**ART: Agent Reinforcement Training**
OpenPipe, 2025
[GitHub: OpenPipe/art](https://github.com/OpenPipe/art)

The ART framework used throughout this course. The `src/art/` directory contains the implementation of `model.train()`, `model.save_checkpoint()`, and `model.reload_vllm()`. Reading the source code of `training.py` makes the training step concrete.

**ART Blog Post: Training Agents That Actually Learn**
OpenPipe Engineering Blog, 2025
[openpipe.ai/blog/art-training-agents](https://openpipe.ai/blog)

The blog post that introduced ART. Covers the text-to-SQL agent training experiment referenced throughout Module 05 and Module 06, including training curves, checkpoint evaluation results, and cost breakdown.

---

### vLLM and LoRA Serving

**vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention**
Kwon et al., 2023
[arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

The vLLM paper. Section 4 on PagedAttention explains why vLLM can serve multiple LoRA adapters simultaneously without loading the base model multiple times — the mechanism that makes hot-swapping efficient.

**vLLM LoRA Documentation**
[docs.vllm.ai/en/latest/models/lora.html](https://docs.vllm.ai/en/latest/models/lora.html)

Official documentation for vLLM's LoRA serving API, including `load_lora_adapter` and `unload_lora_adapter`. The `/v1/load_lora_adapter` endpoint used in Guide 03 is documented here with all parameters.

---

### Unsloth for Efficient GRPO Training

**Unsloth Documentation**
[docs.unsloth.ai](https://docs.unsloth.ai)

Unsloth is the library ART uses for memory-efficient GRPO updates. The "GRPO" section of the documentation covers the specific optimizations Unsloth applies: gradient checkpointing, Flash Attention 2, and 4-bit quantization during the forward pass.

---

## Practical References

### Rollout Collection and Training Efficiency

**RLHF Workflow: From Reward Functions to RL Fine-Tuning**
Lambert et al., 2024
[arXiv:2405.07148](https://arxiv.org/abs/2405.07148)

A practitioner's guide to RLHF pipelines. Section 4 covers rollout collection strategies, including parallelism, batch sizing, and the tradeoff between group size N and wall-clock time. Directly applicable to the `collect_trajectories` function in Guide 01.

**Scaling LLM Test-Time Compute Optimally**
Snell et al., 2024
[arXiv:2408.03314](https://arxiv.org/abs/2408.03314)

Analyzes how sampling multiple completions (the same operation as rollout collection) improves answer quality. Section 3 provides experimental data on the optimal group size N for different task types — relevant when choosing `n_rollouts` in the training loop.

---

### Reward Hacking and Training Stability

**Reward Hacking: Definition, Examples, and Mitigations**
Skalse et al., 2022
[arXiv:2209.13085](https://arxiv.org/abs/2209.13085)

A systematic study of reward hacking in RL systems. Section 3 covers the categories of hacking most relevant to LLM training: length exploitation, judge manipulation, and specification gaming. Essential reading before deploying RULER-based rewards in production.

**Calibrating AI to Avoid Over-Optimization**
Gao et al., 2023
[arXiv:2210.10760](https://arxiv.org/abs/2210.10760)

Studies the relationship between KL divergence from the reference model and the onset of reward hacking. The paper's main finding — that hacking begins predictably as KL grows — directly informs the KL monitoring guidance in Guide 02's diagnostic section.

---

## Tools Used in This Module

| Tool | Purpose | Link |
|------|---------|------|
| ART (OpenPipe) | Agent reinforcement training framework | [github.com/OpenPipe/art](https://github.com/OpenPipe/art) |
| Unsloth | Memory-efficient GRPO implementation | [github.com/unslothai/unsloth](https://github.com/unslothai/unsloth) |
| vLLM | Fast inference with LoRA hot-swap support | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) |
| RULER | LLM-as-a-judge reward function (Module 03) | See Module 03 resources |
| FastMCP | MCP tool server (Module 04) | See Module 04 resources |
| safetensors | Efficient tensor serialization for checkpoints | [github.com/huggingface/safetensors](https://github.com/huggingface/safetensors) |

---

## What to Read First

If you have 30 minutes: read the ART blog post. It covers the full training pipeline with real results.

If you have 2 hours: read DeepSeekMath (GRPO algorithm) and the LoRA paper. Together they explain the two core mechanisms — the update algorithm and the parameter-efficient adapter.

If you want to go deep: read DeepSeek-R1 (full GRPO training at scale) and the reward hacking paper (Gao et al., 2023) to understand the stability risks in long training runs.
