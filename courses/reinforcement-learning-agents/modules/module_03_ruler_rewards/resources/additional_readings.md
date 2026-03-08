# Additional Readings — Module 03: RULER Automatic Rewards

## Foundational Papers

### Reward Specification and Reward Hacking

**"Specification Gaming: The Flip Side of AI Ingenuity"**
DeepMind blog post, Krakovna et al., 2020
A comprehensive, readable catalog of reward hacking examples across simulation, robotics, games, and language tasks. Useful for building intuition about how reward misspecification manifests in practice. Start here if you want concrete examples beyond the ones in Guide 01.

**"Concrete Problems in AI Safety"**
Amodei, Olah, Steinhardt, Christiano, Schulman, Mane (2016)
arXiv:1606.06565
The canonical reference for reward misspecification as a safety problem. Section 3.1 on reward hacking and Section 3.2 on safe exploration are directly relevant to this module. More technical than the DeepMind blog post but worth reading for the formal treatment.

**"Reward is Enough"**
Silver, Singh, Precup, Sutton (2021)
arXiv:2106.04799
The philosophical argument that reward maximization is sufficient for general intelligence — which simultaneously makes the correctness of the reward function critical. Useful context for why reward design matters so much.

### LLM-as-a-Judge

**"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"**
Zheng, Chiang, Sheng, Zhuang, Wu, Zhuang, Lin, Li, Li, Xing, Zhang, Gonzalez, Stoica (2023)
arXiv:2306.05685
The foundational paper on using LLMs as evaluators. Covers position bias, verbosity bias, and self-enhancement bias in LLM judges. Essential reading for understanding the failure modes of the judge you are using in RULER. The MT-Bench benchmark is still widely used.

**"Large Language Models Are Not Robust Multiple Choice Selectors"**
Pezeshkpour and Hruschka (2023)
arXiv:2309.03882
Demonstrates that LLM judges are sensitive to the order in which options are presented. Relevant when formatting trajectories for RULER — randomize the order of trajectories within a group to avoid position bias.

**"Calibrating LLM-Based Evaluators"**
Liu, Iter, Xu, Wang, Xu, Zhu (2023)
arXiv:2309.13714
Methods for improving judge calibration through calibration examples in the prompt. If your judge produces clustered scores (all near 0.5), this paper has practical techniques for improving spread.

## RLHF and Reward Learning

**"Learning to summarize from human feedback"**
Stiennon, Ouyang, Wu, Ziegler, Lowe, Voss, Radford, Amodei, Christiano (2020)
arXiv:2009.01325
The OpenAI paper that brought RLHF to widespread attention. Relevant context for understanding the landscape RULER operates in: RLHF trains a separate reward model from human preferences, while RULER uses an LLM judge directly. Comparing the two approaches clarifies RULER's design choices.

**"Constitutional AI: Harmlessness from AI Feedback"**
Bai, Jones, Ndousse, Askell, Chen, DasSarma, Drain, Fort, Ganguli, Henighan, Joseph, Kadavath, Kernion, Conerly, El-Showk, Elhage, Hatfield-Dodds, Hernandez, Hume, Johnston, Kravec, Lovitt, Nanda, Olsson, Amodei, Brown, Clark, McCandlish, Olah, Mann, Kaplan (2022)
arXiv:2212.08073
Anthropic's approach to using an LLM to generate and evaluate its own critiques as reward signals. Related to RULER in that both use LLM judgment rather than human preference labels. The "self-critique and revision" loop is a useful pattern for tasks where you can generate evaluation criteria from the task description.

**"A Survey of Reinforcement Learning from Human Feedback"**
Casper, Davies, Shi, Gilbert, Scheurer, Rando, Freedman, Korbak, Lindner, Freire, Wang, Marks, Segerie, Carroll, Peng, Christoffersen, Damani, Slocum, Pfau, Astrid Schettler, Gal, Shavit, Sridhar, Krueger, Hadfield-Menell (2023)
arXiv:2307.15217
Comprehensive survey covering reward model training, RLHF pipelines, and practical failure modes. Sections 4-5 on reward modeling are most relevant to Module 03.

## Reward Shaping

**"Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping"**
Ng, Harada, Russell (1999)
ICML 1999
The theoretical foundation for safe reward shaping. Proves that potential-based reward shaping is the only transformation that preserves the optimal policy. Required reading if you design intermediate rewards — the theory tells you exactly which transformations are safe.

**"Let's Reward Step by Step: Step-Level Reward Model as the Navigators for Reasoning"**
Lightman, Kosaraju, Burda, Edwards, Baker, Lee, Leike, Schulman, Sutskever, Cobbe (2023)
arXiv:2305.20050
OpenAI's process reward model (PRM) paper. Demonstrates that rewarding correct reasoning steps (not just correct final answers) significantly improves math reasoning performance. The multi-step reward shaping in Guide 03 is inspired by this approach. The key insight: final-answer-only rewards leave the agent without signal about which reasoning paths are productive.

## GRPO and Training Algorithms

**"DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"**
Shao, Wang, Zhu, Chen, Zeng, Liu, Zhang, Zhang, Li, Zhang, Zhu (2024)
arXiv:2402.03300
The paper that introduced GRPO (Group Relative Policy Optimization). Section 3.2 explains the algorithm used throughout this course. Read this to understand exactly why GRPO only needs relative rewards (Section 3.2, Equation 2), which is the mathematical foundation for RULER's design.

## Practical Implementation

**ART (Agent Reinforcement Training) Documentation**
OpenPipe, 2024-2025
github.com/openPipe/ART
The framework used throughout this course. The examples directory contains working RULER implementations for text-to-SQL and other agent tasks. The `art.reward.ruler_score` module is the production implementation of the scoring function built in this module's exercise.

**"RULER: Reward Using LLM Evaluation Rubrics"**
Internal documentation, OpenPipe, 2024
Describes the RULER design decisions: why relative scoring, why groups of N=4, how to write judge prompts that avoid clustering. Available in the ART documentation.

## Beyond RULER

**"Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models"**
Chen, Deng, Yuan, Ji, Gu (2024)
arXiv:2401.01335
SPIN: an alternative to RLHF that uses self-play instead of a reward model. The agent plays against older versions of itself — recent outputs are preferred over older outputs as a reward signal. No judge needed. Worth understanding as a complementary approach to RULER for tasks where "newer is better" is a reasonable proxy.

**"OpenAI o1 System Card"**
OpenAI, 2024
openai.com/o1/system-card
While not focused on RULER specifically, the o1 system card describes using process reward models at scale for mathematical reasoning — the production version of the ideas in Guide 03's multi-step reward shaping section. Useful for understanding where the field is heading.

## Suggested Reading Order

For learners new to reward learning:
1. DeepMind specification gaming blog post
2. "Judging LLM-as-a-Judge" (Zheng et al.)
3. DeepSeekMath paper (Section 3.2 only for GRPO)
4. ART documentation and examples

For learners wanting theoretical depth:
1. "Concrete Problems in AI Safety" (Amodei et al.)
2. Ng & Russell reward shaping paper
3. "A Survey of RLHF" (Casper et al.)
4. Lightman et al. process reward models

For practitioners building reward functions now:
1. ART documentation (RULER examples)
2. "Calibrating LLM-Based Evaluators" (Liu et al.)
3. "Large Language Models Are Not Robust Multiple Choice Selectors" (position bias)
4. OpenPipe ART GitHub issues and discussions
