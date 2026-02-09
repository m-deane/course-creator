# Additional Readings - Module 00: AI Engineer Mindset

## Essential Papers

### The Foundation
- **Attention Is All You Need** (Vaswani et al., 2017)
  - [ArXiv](https://arxiv.org/abs/1706.03762)
  - The paper that started modern LLMs. Read for understanding the engine.

### Scaling & Training
- **Scaling Laws for Neural Language Models** (Kaplan et al., 2020)
  - [ArXiv](https://arxiv.org/abs/2001.08361)
  - Why bigger models trained on more data work better, predictably.

- **Training Compute-Optimal Large Language Models** (Hoffmann et al., 2022) - "Chinchilla"
  - [ArXiv](https://arxiv.org/abs/2203.15556)
  - The 20-tokens-per-parameter rule. Changed how we think about training.

### Alignment
- **Training Language Models to Follow Instructions** (Ouyang et al., 2022) - "InstructGPT"
  - [ArXiv](https://arxiv.org/abs/2203.02155)
  - How base models become assistants. The RLHF paper.

### Memory & Retrieval
- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (Lewis et al., 2020)
  - [ArXiv](https://arxiv.org/abs/2005.11401)
  - The RAG paper. Why you don't need to put everything in weights.

### Agents
- **ReAct: Synergizing Reasoning and Acting in Language Models** (Yao et al., 2022)
  - [ArXiv](https://arxiv.org/abs/2210.03629)
  - The standard pattern for LLM agents.

## Recommended Blog Posts & Articles

### Mental Models
- [What We Learned from a Year of Building with LLMs](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/) - O'Reilly
- [Building LLM Applications for Production](https://huyenchip.com/2023/04/11/llm-engineering.html) - Chip Huyen
- [Patterns for Building LLM-based Systems & Products](https://eugeneyan.com/writing/llm-patterns/) - Eugene Yan

### System Design
- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) - Lilian Weng
- [The Shift from Models to Compound AI Systems](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/) - Berkeley AI Research

### Practical Engineering
- [Emerging Architectures for LLM Applications](https://a16z.com/emerging-architectures-for-llm-applications/) - a16z
- [LLMOps: MLOps for LLMs](https://wandb.ai/site/articles/llmops-mlops-for-llms) - Weights & Biases

## Video Resources

- [Andrej Karpathy: State of GPT](https://www.youtube.com/watch?v=bZQun8Y4L2A) - 1 hour overview
- [fast.ai Practical Deep Learning](https://course.fast.ai/) - Foundation course
- [Stanford CS324: Large Language Models](https://stanford-cs324.github.io/winter2022/) - Academic depth

## Books

- **Build a Large Language Model (From Scratch)** - Sebastian Raschka
  - Hands-on understanding of the architecture

- **Designing Machine Learning Systems** - Chip Huyen
  - Production ML thinking (applies to LLMs)

## Tools & Documentation

- [Anthropic Claude Documentation](https://docs.anthropic.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)

## Course Paper Summaries

See [/resources/paper_summaries.md](../../../resources/paper_summaries.md) for detailed summaries of all canonical papers.

## Reading Order Recommendation

1. **Day 1:** Attention Is All You Need (skim math, understand architecture)
2. **Day 2:** InstructGPT (understand how assistants are made)
3. **Day 3:** RAG paper (understand retrieval augmentation)
4. **Day 4:** ReAct paper (understand agent patterns)
5. **Day 5:** Blog posts on production LLM systems

This gives you the conceptual foundation to understand the rest of the course.
