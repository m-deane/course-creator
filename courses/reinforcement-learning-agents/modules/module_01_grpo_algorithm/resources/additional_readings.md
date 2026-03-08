# Additional Readings — Module 01: GRPO Algorithm

## Primary Sources

### DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
**Authors:** Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxian He, Yichong Wu, Mingchuan Zhang, Keming Lu, Jiayi Pan, Wenbo Ma, Yanping Huang, Zizheng Pan, Bingchen Zhao, Zhi-Hong Deng, Chong Ruan, Jianshu Chen, Damai Dai, Fuli Luo, Yichong Chen, Zhongyu Wei, Wenhao Zhuang, Linfeng Song, Wanjun Zhong, Xiaohan Chen, Fei Nie, Xiaodong Liu, Yu Wu, Chengqi Zha, Fuli Luo
**Date:** April 2024
**URL:** https://arxiv.org/abs/2402.03300

This is the paper that introduced GRPO. The algorithm was developed to train DeepSeekMath-7B to achieve competitive mathematical reasoning performance. Section 3.2 defines the GRPO objective function used throughout this module. Read this before the DeepSeek-R1 paper — it provides the algorithmic foundation that R1 builds on.

Key sections:
- Section 3.2: GRPO algorithm definition and motivation
- Section 3.3: Ablation study comparing GRPO to PPO, SFT, and rejection sampling fine-tuning (RFT)
- Appendix: Training details including group size G and hyperparameter values

---

### DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
**Authors:** DeepSeek-AI
**Date:** January 2025
**URL:** https://arxiv.org/abs/2501.12948

The paper that brought GRPO to widespread attention. DeepSeek-R1 achieves GPT-o1-level reasoning on math and coding benchmarks using GRPO on open-source models. The paper describes a multi-stage training pipeline: cold-start SFT, RL with GRPO, rejection sampling, and final alignment SFT.

Key sections:
- Section 2: Training methodology including the GRPO objective
- Section 3: Results comparing R1 to GPT-o1 on reasoning benchmarks
- Section 4: DeepSeek-R1-Zero (pure RL without SFT cold-start) — the ablation that demonstrates RL alone can induce chain-of-thought reasoning

---

## Background: PPO and Policy Gradients

### Proximal Policy Optimization Algorithms
**Authors:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
**Date:** July 2017
**URL:** https://arxiv.org/abs/1707.06347

The original PPO paper. Read this to understand what GRPO is replacing and why. Section 3 defines the clipped surrogate objective that GRPO adapts. The core clipping mechanism in GRPO ($\min(\rho A, \text{clip}(\rho, 1-\epsilon, 1+\epsilon)A)$) is taken directly from PPO.

Key sections:
- Section 3: The clipped surrogate objective
- Section 4: PPO-Clip algorithm

---

### Training Language Models to Follow Instructions with Human Feedback (InstructGPT)
**Authors:** Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe
**Date:** March 2022
**URL:** https://arxiv.org/abs/2203.02155

The paper that established PPO as the standard for LLM alignment (RLHF). Understanding InstructGPT's PPO setup clarifies exactly what GRPO simplifies: InstructGPT trains a separate reward model, a value network, and applies multiple stabilization techniques. GRPO removes most of this complexity.

---

## Background: DPO

### Direct Preference Optimization: Your Language Model is Secretly a Reward Model
**Authors:** Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn
**Date:** May 2023
**URL:** https://arxiv.org/abs/2305.18290

The DPO paper. Read this alongside Guide 03 (GRPO vs Alternatives) to understand the offline alternative to GRPO. The key insight: the RLHF objective can be reformulated to optimize directly from preference pairs without training a reward model.

---

## Background: REINFORCE

### Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning
**Authors:** Ronald J. Williams
**Date:** 1992
**URL:** https://link.springer.com/article/10.1007/BF00992696

The original REINFORCE paper. A foundational read for understanding where policy gradient methods come from. GRPO can be seen as REINFORCE with variance reduction (baseline subtraction via group mean) plus PPO's clipping mechanism.

---

## Practical Guides and Implementations

### Unsloth GRPO Training Documentation
**URL:** https://docs.unsloth.ai/basics/reinforcement-learning

Unsloth is the efficient fine-tuning library used in Module 02 (ART Framework). This documentation covers how to configure GRPO training with Unsloth, including group size settings, memory optimization, and integration with reward functions. Practical reading before Module 02.

---

### ART (Agent Reinforcement Training) — OpenPipe
**URL:** https://github.com/OpenPipe/ART

The open-source framework this course uses for training. The repository includes example scripts for text-to-SQL and tool-use agents trained with GRPO. Reading the examples in `examples/` before Module 02 is recommended.

---

### The N+1 Problem in GRPO: Why Group Size Matters
**URL:** https://huggingface.co/blog/grpo-explainer

A HuggingFace blog post walking through GRPO hyperparameter sensitivity, particularly around group size G and its tradeoff between baseline quality and inference cost. Practical context for the hyperparameter choices in Module 02.

---

## Further Reading: Variance Reduction in Policy Gradients

### Generalized Advantage Estimation
**Authors:** John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel
**Date:** 2015
**URL:** https://arxiv.org/abs/1506.02438

GAE is the standard technique for computing advantages in PPO. Understanding GAE clarifies why GRPO's group-based approach is a simpler alternative — GAE requires a value function and complex discount weighting; GRPO requires only a group of same-prompt samples.

---

## Recommended Reading Order

For this module:
1. DeepSeekMath paper (Section 3.2 first, then the full paper)
2. PPO paper (Section 3, for context on the clipped surrogate)
3. DeepSeek-R1 paper (Section 2-3, once you understand GRPO's mechanics)

For broader context:
4. InstructGPT (to understand what GRPO replaced)
5. DPO (to understand the offline alternative)
6. REINFORCE (to understand the historical foundation)
