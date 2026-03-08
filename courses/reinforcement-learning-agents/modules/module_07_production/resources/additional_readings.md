# Additional Readings — Module 07: Production Considerations

## Deployment & Serving

**vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention**
Kwon et al., 2023. SOSP.
The inference engine ART uses for serving. PagedAttention enables efficient memory management during generation, making it possible to serve LoRA adapters with minimal overhead.
https://arxiv.org/abs/2309.06180

**LoRAX: Multi-LoRA Inference Server**
Predibase, 2024. Serves multiple LoRA adapters from a single base model. Relevant for A/B testing different training checkpoints in production.
https://github.com/predibase/lorax

## Cost Optimization

**Efficient Fine-Tuning with LoRA**
Hu et al., 2021. ICLR.
The foundational paper for Low-Rank Adaptation. Explains why training only rank-16 matrices on top of a frozen base model achieves comparable quality to full fine-tuning at a fraction of the cost.
https://arxiv.org/abs/2106.09685

**Unsloth: 2x Faster LLM Fine-Tuning**
Unsloth AI, 2024. The training backend ART uses. Achieves 2x speedup through kernel optimization and memory-efficient backpropagation.
https://github.com/unslothai/unsloth

## Benchmarking

**ART-E: Agent Reinforcement Training for Email Search**
OpenPipe, 2025. The benchmark showing a 14B RL-trained model outperforming o3 at 96% accuracy, 5x faster, 64x cheaper. Includes training curves and cost breakdowns.
https://openpipe.ai/blog/art-e

**Holistic Evaluation of Language Models (HELM)**
Liang et al., 2022. Stanford CRFM.
A comprehensive evaluation framework for LLMs. Useful for designing your own benchmark suites beyond simple accuracy metrics.
https://arxiv.org/abs/2211.09110

## Model Selection

**Qwen2.5 Technical Report**
Qwen Team, 2024. The primary model family for ART training. Documents architecture, training data, and capabilities across 0.5B to 72B sizes.
https://arxiv.org/abs/2412.15115

**Llama 3: Open Foundation Models**
Meta AI, 2024. The alternative model family supported by ART. Covers architecture choices and training methodology.
https://arxiv.org/abs/2407.21783
