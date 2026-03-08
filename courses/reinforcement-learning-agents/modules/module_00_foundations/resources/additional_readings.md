# Additional Readings — Module 00: Foundations

Organized by topic. Start with the papers marked **Essential** before moving to Module 01.

---

## SFT and the Case for RL

### Essential

**Ouyang et al., "Training language models to follow instructions with human feedback" (2022)**
- The InstructGPT paper. Establishes the canonical SFT → RL pipeline for language models.
- Section 3 describes the SFT phase and why it is insufficient alone.
- https://arxiv.org/abs/2203.02155

**DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025)**
- Demonstrates that RL enables reasoning strategies that SFT alone cannot produce.
- Section 2 shows the SFT cold-start approach; Section 3 shows what RL adds.
- https://arxiv.org/abs/2501.12948

### Background

**Ross, Bagnell & Gordon, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (DAgger, 2011)**
- The formal treatment of why imitation learning fails in sequential settings (compounding errors).
- Section 2 proves the error bound: imitation learning error grows quadratically with horizon length.
- https://arxiv.org/abs/1011.0686

**Ho & Ermon, "Generative Adversarial Imitation Learning" (GAIL, 2016)**
- One alternative to pure SFT: learn a reward function from demonstrations and then apply RL.
- Useful background for understanding the spectrum between pure imitation and pure RL.
- https://arxiv.org/abs/1606.03476

---

## Reward Signal Design

### Essential

**Ziegler et al., "Fine-Tuning Language Models from Human Preferences" (2019)**
- Introduces reward modeling: train a model to predict human preference rankings.
- The foundation of RLHF reward models.
- https://arxiv.org/abs/1909.08593

**Amodei et al., "Concrete Problems in AI Safety" (2016)**
- Section 2 (Reward Hacking) and Section 3 (Safe Exploration) are directly relevant.
- The most comprehensive treatment of reward function failure modes.
- https://arxiv.org/abs/1606.06565

### Background

**Leike et al., "Scalable agent alignment via reward modeling" (2018)**
- Frames reward modeling as the core bottleneck for capable agents.
- Motivates why hand-crafted reward functions are insufficient at scale.
- https://arxiv.org/abs/1811.07871

**Krakovna et al., "Avoiding Side Effects in Complex Environments" (2020)**
- Practical examples of reward hacking and side effects in real agent tasks.
- Useful for understanding what goes wrong when reward functions are underspecified.
- https://arxiv.org/abs/2006.06547

---

## Policy Optimization

### Essential

**Schulman et al., "Proximal Policy Optimization Algorithms" (PPO, 2017)**
- The standard policy gradient method that GRPO simplifies.
- Section 3 explains the clipped surrogate objective that GRPO inherits.
- https://arxiv.org/abs/1707.06347

**Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (GRPO, 2024)**
- Introduces GRPO. Section 3 derives the algorithm step by step from PPO.
- Required reading before Module 01.
- https://arxiv.org/abs/2402.03300

### Background

**Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (REINFORCE, 1992)**
- The original policy gradient paper. Short and readable.
- Theorem 1 is the policy gradient theorem; Section 6 introduces the baseline.
- http://www.cs.toronto.edu/~tingwuwang/REINFORCE.pdf

**Sutton et al., "Policy Gradient Methods for Reinforcement Learning with Function Approximation" (2000)**
- The formal policy gradient theorem for function approximators (including neural networks).
- Section 2 proves that the baseline does not bias the gradient.
- https://papers.nips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf

**Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (GAE, 2016)**
- Introduces GAE — the standard advantage estimation technique that GRPO adapts.
- Section 4 explains the bias-variance tradeoff in advantage estimation.
- https://arxiv.org/abs/1506.02438

---

## RL for Language Models (Broader Context)

**Stiennon et al., "Learning to summarize from human feedback" (2020)**
- The first large-scale demonstration of RL improving LLM outputs beyond SFT.
- Clearly shows where SFT plateaus and RL continues to improve.
- https://arxiv.org/abs/2009.01325

**Bai et al., "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback" (Anthropic RLHF, 2022)**
- Detailed description of the SFT → reward model → RL pipeline at scale.
- Section 3 is particularly relevant: discusses reward model quality and its effect on RL.
- https://arxiv.org/abs/2204.05862

**OpenAI, "Scaling Laws for Reward Model Overoptimization" (2022)**
- Shows what happens when RL optimizes too hard against an imperfect reward model.
- Empirical evidence for the reward hacking failure modes described in Guide 02.
- https://arxiv.org/abs/2210.10760

---

## Textbook References

**Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed., 2018)**
- The canonical RL textbook. Available free online.
- Chapters 1–4: MDP formalism and motivation
- Chapter 13: Policy gradient methods (REINFORCE and baselines)
- http://incompleteideas.net/book/the-book-2nd.html

**Goodfellow, Bengio & Courville, "Deep Learning" (2016)**
- Chapter 6: deep feedforward networks (architecture background)
- Chapter 8: optimization methods (gradient descent mechanics)
- http://www.deeplearningbook.org/

---

## Recommended Reading Order

Before Module 01 (GRPO Algorithm):
1. DeepSeek-R1 (skim Section 1 and 2, focus on Section 3 motivation)
2. PPO paper (Sections 1-3)
3. GRPO paper (Section 3 algorithm derivation)

For deeper understanding of reward design:
1. Concrete Problems in AI Safety (Section 2)
2. Learning to summarize from human feedback (Section 3)
3. Scaling Laws for Reward Model Overoptimization

For formal foundations (optional, recommended for researchers):
1. REINFORCE (Williams 1992)
2. Policy Gradient theorem (Sutton et al. 2000)
3. GAE (Schulman et al. 2016)
