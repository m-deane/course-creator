# Additional Readings — Module 07: Production Deployment

Resources organized by topic. Start with the vLLM serving documentation and the OpenPipe cost case study before reading the broader deployment and monitoring references.

---

## vLLM: Model Serving

**vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention**
Kwon et al., 2023. SOSP.
The research paper behind vLLM. Section 4 explains PagedAttention — the memory management mechanism that eliminates KV-cache fragmentation and allows near-100% GPU memory utilization during inference. Understanding PagedAttention explains the throughput numbers in benchmarks and informs tuning `--gpu-memory-utilization` for your workload.
[https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)

**vLLM Documentation — OpenAI-Compatible Server**
The authoritative reference for running vLLM as an HTTP serving endpoint. Covers the `vllm serve` CLI, all server arguments, authentication, tensor parallelism, and the `/v1/chat/completions` endpoint. The "Engine Arguments" section documents every configuration option relevant to production: batch sizes, quantization flags, and request scheduling policies.
[https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

**vLLM LoRA Serving Documentation**
Documents how vLLM serves multiple LoRA adapters from a single loaded base model. The `--enable-lora`, `--max-loras`, and `--max-lora-rank` flags are explained with their memory implications. Serving multiple checkpoint versions from one base model is the key technique for A/B testing adapter generations without duplicating GPU memory for the base weights.
[https://docs.vllm.ai/en/latest/models/lora.html](https://docs.vllm.ai/en/latest/models/lora.html)

**vLLM Benchmarking Guide**
How to measure throughput (tokens/second) and latency (time-to-first-token) for your specific deployment configuration. Run these benchmarks before and after applying quantization to quantify the latency-vs-quality tradeoff for your workload.
[https://docs.vllm.ai/en/latest/performance/benchmarks.html](https://docs.vllm.ai/en/latest/performance/benchmarks.html)

**vLLM Prometheus Metrics**
vLLM exports Prometheus metrics for request queue depth, token throughput, GPU utilization, and KV-cache hit rate. The Grafana dashboard template linked on this page provides a production-ready starting point for monitoring your deployment.
[https://docs.vllm.ai/en/latest/serving/metrics.html](https://docs.vllm.ai/en/latest/serving/metrics.html)

**LoRAX: Multi-LoRA Inference Server**
Predibase, 2024. An alternative to vLLM's built-in LoRA support, optimized specifically for serving many adapters simultaneously. Relevant for production systems where multiple SQL agent versions must be served concurrently for A/B testing or multi-tenant deployments.
[https://github.com/predibase/lorax](https://github.com/predibase/lorax)

---

## Cost Optimization

**Reinforcement Learning for Text-to-SQL at Scale**
OpenPipe Engineering Blog, 2025.
The case study this course is built around. Section 5 contains a full cost breakdown: training a 14B SQL agent on a single H100 for 12 hours cost approximately $50, and the resulting model runs at 5× the speed of o3 at 64× lower cost per inference call. The cost model here directly informs the ROI calculation when deciding whether to train a custom model vs. use a frontier API.
[https://openpipe.ai/blog/sql-agent-rl](https://openpipe.ai/blog/sql-agent-rl)

**ART-E: Agent Reinforcement Training for Email Search**
OpenPipe, 2025. A second case study showing a 14B RL-trained model outperforming o3 at 96% accuracy, 5× faster, 64× cheaper. Includes complete training curves and cost breakdowns across training and inference phases.
[https://openpipe.ai/blog/art-e](https://openpipe.ai/blog/art-e)

**Scaling Laws for Neural Language Models**
Kaplan et al., 2020.
The foundational paper on the compute-performance tradeoff. Section 4 establishes that a fine-tuned smaller model at lower inference cost typically matches a larger prompted model. This is the theoretical basis for why a 7B RL-trained agent can compete with a 70B prompted model on narrow tasks.
[https://arxiv.org/abs/2001.08361](https://arxiv.org/abs/2001.08361)

**Efficient Fine-Tuning with LoRA**
Hu et al., 2021. ICLR.
The foundational paper for Low-Rank Adaptation. Explains why training only rank-16 matrices on top of a frozen base model achieves comparable quality to full fine-tuning at 3× lower memory cost. The rank-quality tradeoff in Section 5 informs choosing `lora_rank` for your training config.
[https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

---

## Quantization for Inference

**AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration**
Lin et al., 2023.
AWQ is the recommended quantization method for production deployments. Section 3 explains how channel-wise scaling protects the 1% of weights that account for most model performance. vLLM supports AWQ natively via `--quantization awq`. Use AWQ when deploying newly trained checkpoints quickly.
[https://arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)

**GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**
Frantar et al., 2022.
The other widely-supported quantization format. GPTQ takes longer to quantize but achieves slightly better quality than AWQ at 4-bit. Use GPTQ when quality matters more than quantization turnaround time.
[https://arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)

**Unsloth: 2× Faster LLM Fine-Tuning**
Unsloth AI, 2024. The training backend ART uses internally. Achieves 2× speedup through kernel fusion, Flash Attention 2, and memory-efficient backpropagation. The same optimizations that reduce training memory also apply when running the model for evaluation before deployment.
[https://github.com/unslothai/unsloth](https://github.com/unslothai/unsloth)

---

## Agent Evaluation and Benchmarking

**AgentBench: Evaluating LLMs as Agents**
Liu et al., 2023.
The first systematic benchmark for evaluating LLM agents across diverse environments. Section 3 describes the evaluation protocol: multiple independent runs per task, success rate as the primary metric, cost as a secondary metric. Adopt this protocol for evaluating your deployed SQL agent on held-out queries before and after each retraining cycle.
[https://arxiv.org/abs/2308.03688](https://arxiv.org/abs/2308.03688)

**BIRD: A Big Bench for Large-Scale Database Grounded Text-to-SQL Evaluation**
Li et al., 2023.
The production-realistic text-to-SQL benchmark, harder than Spider. Uses schemas with hundreds of tables, noisy data, and questions requiring external knowledge. Evaluating your deployed agent on BIRD gives a realistic estimate of out-of-distribution performance beyond the training distribution.
[https://arxiv.org/abs/2305.03111](https://arxiv.org/abs/2305.03111)

**Holistic Evaluation of Language Models (HELM)**
Liang et al., 2022. Stanford CRFM.
A comprehensive evaluation framework for LLMs. The methodology for constructing evaluation scenarios — multiple perturbations, calibration checks, and disaggregated reporting — is directly applicable to designing a production eval suite for your SQL agent.
[https://arxiv.org/abs/2211.09110](https://arxiv.org/abs/2211.09110)

**Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**
Zheng et al., 2023.
Validates LLM-as-judge evaluation and documents its failure modes: position bias, verbosity bias, and self-enhancement bias. Section 4 provides mitigation strategies for each. Understanding these biases is essential before deploying RULER-style scoring as your production evaluation metric.
[https://arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)

---

## Observability and Monitoring

**Langfuse: Open-Source LLM Observability**
[https://langfuse.com/docs](https://langfuse.com/docs)
An open-source observability platform designed for LLM applications. Captures the full message history of each agent run, reward score, and per-step latency. The ART integration is documented in the ART repository. Use Langfuse to identify which query patterns your agent handles poorly in production.

**OpenTelemetry Documentation**
[https://opentelemetry.io/docs/](https://opentelemetry.io/docs/)
The standard for distributed tracing, metrics, and logging. vLLM exports OpenTelemetry traces natively. Instrumenting your agent with OpenTelemetry gives per-request latency breakdowns — time in vLLM vs. time in tool calls vs. time in the agent loop — essential for diagnosing production performance bottlenecks.

---

## Continuous Training

**Online Learning for Language Models: A Survey**
[https://arxiv.org/abs/2503.01954](https://arxiv.org/abs/2503.01954)
Surveys approaches for continuously updating deployed language models with new data. Section 4 on continual RL fine-tuning covers how to use production trajectories (real user queries + observed rewards) to periodically retrain your agent without catastrophic forgetting of earlier capabilities.

**Don't Stop Pretraining: Adapt Language Models to Domains and Tasks**
Gururangan et al., 2020.
Establishes the empirical basis for continued domain-adaptive training. The key finding — that continued training on domain data consistently improves task performance even after initial fine-tuning — supports the practice of periodically retraining your SQL agent on new production queries.
[https://arxiv.org/abs/2004.10964](https://arxiv.org/abs/2004.10964)

---

## Model Selection

**Qwen2.5 Technical Report**
Qwen Team, 2024. The primary model family for ART training. Documents architecture, training data, and capabilities across 0.5B to 72B model sizes. Table 2 shows the Qwen2.5-7B-Instruct capability profile — the baseline you are training from.
[https://arxiv.org/abs/2412.15115](https://arxiv.org/abs/2412.15115)

**Llama 3: Open Foundation Models**
Meta AI, 2024. The alternative model family supported by ART. Covers architecture choices and training methodology. Table 4 benchmarks Llama-3.1-8B-Instruct on coding and reasoning tasks — useful for comparing against Qwen2.5-7B as a base model choice.
[https://arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783)

---

## Cloud Infrastructure

**RunPod GPU Cloud**
[https://runpod.io](https://runpod.io)
Commonly used by the ART team. H100 and A100 instances available. Spot instances reduce cost by 60–70% for non-interactive training jobs. Network volumes persist checkpoints across pod restarts.

**SkyPilot: Intercloud Broker**
[https://skypilot.readthedocs.io](https://skypilot.readthedocs.io)
Automatically finds the cheapest available GPU across AWS, GCP, Azure, Lambda, and RunPod. Supports spot instances with automatic recovery. Useful when cost matters more than fixed latency in your training infrastructure.

---

## What to Read First

For getting a model into production quickly (1–2 hours):
1. vLLM OpenAI-compatible server docs — run `vllm serve` with your checkpoint
2. vLLM LoRA docs — confirm the adapter loads correctly alongside the base model
3. OpenPipe blog post cost section — set realistic cost expectations before scaling

For production-grade monitoring and evaluation (half day):
1. vLLM Prometheus metrics docs — set up dashboards before the first production query
2. Langfuse docs — add full trace capture to your agent loop
3. AgentBench evaluation protocol — design your hold-out test harness

For continuous improvement (research background):
1. BIRD benchmark — calibrate out-of-distribution performance
2. Online learning survey Section 4 — plan the retraining cadence
3. LLM-as-judge paper Section 4 — understand evaluation bias in your monitoring
