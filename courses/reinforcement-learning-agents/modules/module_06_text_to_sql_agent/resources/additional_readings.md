# Additional Readings — Module 06: Text-to-SQL Agent

Resources are organized by topic. Start with the foundational papers before moving to implementation guides.

---

## Text-to-SQL Research

### Foundational

**Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task**
Yu et al., 2018. EMNLP.
The benchmark that established how text-to-SQL systems are evaluated. Introduces cross-database evaluation — testing on schemas the model was never trained on. The hardest Spider examples require the same correlated subquery and multi-table JOIN reasoning your agent learns in this module.
[https://arxiv.org/abs/1809.08887](https://arxiv.org/abs/1809.08887)

**BIRD: A Big Bench for Large-Scale Database Grounded Text-to-SQL Evaluation**
Li et al., 2023. NeurIPS.
A harder benchmark than Spider, with larger databases, noisy data, and questions that require external knowledge. The BIRD leaderboard shows the performance ceiling of prompted GPT-4 compared to fine-tuned models — the gap is where RL training provides its biggest gains.
[https://arxiv.org/abs/2305.03111](https://arxiv.org/abs/2305.03111)

### RL for Text-to-SQL

**Reinforcement Learning for Text-to-SQL at Scale**
OpenPipe Engineering Blog, 2025. The case study behind this module. A 14B model trained with RL on their internal database outperformed o3 on a realistic agentic SQL task, at 5x the speed and 64x lower cost. Includes training curves, cost breakdowns, and ablations on reward function design.
[https://openpipe.ai/blog/sql-agent-rl](https://openpipe.ai/blog/sql-agent-rl)

**CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning**
Le et al., 2022. NeurIPS.
SQL generation is structurally similar to code generation — both require syntactic correctness and semantic accuracy. CodeRL shows how RL with execution-based rewards (does the code run? does it produce the right output?) improves code generation beyond what supervised fine-tuning achieves. The same principles apply to SQL.
[https://arxiv.org/abs/2207.01780](https://arxiv.org/abs/2207.01780)

---

## MCP and Tool Use

**Model Context Protocol Specification**
Anthropic, 2024. The formal specification for MCP — the protocol your FastMCP server implements. Covers the tool schema format, the request/response protocol, and the resource and prompt subsystems (not used in this module but relevant for more complex agents).
[https://modelcontextprotocol.io/specification](https://modelcontextprotocol.io/specification)

**Toolformer: Language Models Can Teach Themselves to Use Tools**
Schick et al., 2023. NeurIPS.
The paper that showed LLMs can learn when to call external tools by generating tool call annotations themselves. The text-to-SQL agent you built does something similar: it learns when to call `list_tables`, when to call `describe_table`, and when it has enough information to answer. Toolformer's approach is supervised; yours is RL-based.
[https://arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761)

**ReAct: Synergizing Reasoning and Acting in Language Models**
Yao et al., 2023. ICLR.
The foundational paper for the thought-action-observation loop that underlies most modern tool-using agents. Your rollout function implements this pattern: the agent reasons (thinks about what it needs), acts (calls a tool), observes (reads the tool result), and repeats. Understanding ReAct will help you extend your agent beyond SQL to more complex multi-step tasks.
[https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629)

---

## SQLite Reference

**SQLite PRAGMA Statements**
SQLite documentation. The `PRAGMA table_info` and `PRAGMA foreign_key_list` commands used in `describe_table()` are documented here. Also covers foreign key enforcement (`PRAGMA foreign_keys`), which must be enabled on every new connection.
[https://www.sqlite.org/pragma.html](https://www.sqlite.org/pragma.html)

**SQLite Query Optimization**
SQLite documentation. When your trained agent starts running in production against large databases, query performance becomes relevant. This page explains SQLite's query planner, how to read `EXPLAIN QUERY PLAN` output, and when indexes matter.
[https://www.sqlite.org/queryplanner.html](https://www.sqlite.org/queryplanner.html)

---

## GRPO and Training

**DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**
Shao et al., 2024. The paper that introduced GRPO, the algorithm behind your training loop. Section 3 covers the objective function, the group relative advantage calculation ($A_i = (r_i - \mu) / \sigma$), and the KL divergence penalty. Reading this alongside Module 01's guides gives you the complete mathematical foundation.
[https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)

**ART: Automatic Reinforcement Learning for Agentic Training**
OpenPipe, 2025. The framework used in this module's training loop. The ART documentation covers the Trainer API, checkpoint management, multi-worker rollout collection, and integration with vLLM and Unsloth.
[https://docs.openpipe.ai/art](https://docs.openpipe.ai/art)

---

## LLM-as-Judge

**Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**
Zheng et al., 2023. NeurIPS.
The paper that validated the LLM-as-judge approach — using a strong LLM to score outputs from a weaker model. RULER builds on this. Understanding the biases documented here (position bias, verbosity bias, self-enhancement bias) helps you write better RULER prompts that produce more reliable training signal.
[https://arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)

---

## Going Further

**DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction**
Pourreza and Rafiei, 2023. A prompting-only approach to text-to-SQL that shows how far you can get without fine-tuning. Comparing DIN-SQL's performance on Spider to your RL-trained agent on the same schema illustrates the gap that RL closes.
[https://arxiv.org/abs/2304.11015](https://arxiv.org/abs/2304.11015)

**Execution-Guided Neural Program Synthesis**
Chen et al., 2018. The idea of using execution results to guide generation — which is exactly what `run_query`'s error messages do for your agent. Understanding the history of execution-guided synthesis gives context for why structured error returns improve training.
[https://arxiv.org/abs/1807.03168](https://arxiv.org/abs/1807.03168)

**Self-Debugging: Teaching Large Language Models to Debug Their Predicted Programs**
Chen et al., 2023. A supervised approach to teaching models to fix their own code errors by reading error messages. Your RL-trained agent learns the same behavior without labeled debugging examples — it learns from trial and error. Reading this paper helps you understand what your agent is doing when it recovers from SQL syntax errors.
[https://arxiv.org/abs/2304.05128](https://arxiv.org/abs/2304.05128)
