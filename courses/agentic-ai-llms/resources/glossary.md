# Glossary: Agentic AI & Large Language Models

## A

**Action**
: A specific operation that an agent can perform using a tool. Actions are selected based on reasoning about the current goal and context.

**Agent**
: An autonomous system that uses an LLM as a reasoning engine to perceive, reason, plan, and act to accomplish goals. Goes beyond simple prompt-response by maintaining state and taking multi-step actions.

**Agent Loop**
: The iterative cycle of observe → reason → act → observe that agents follow until completing their goal or reaching a termination condition.

**Anthropic Claude**
: A family of large language models developed by Anthropic, known for long context windows, strong instruction following, and safety features.

**Attention Mechanism**
: The core component of transformers that allows models to focus on relevant parts of the input when processing each token. Multi-head attention enables parallel processing of different relationships.

**Autogen**
: A Microsoft framework for building multi-agent systems where agents can have conversations, delegate tasks, and collaborate.

---

## B

**Backpressure**
: Managing rate limits and token throughput when agents make multiple API calls. Important for production deployments.

**Beam Search**
: A decoding strategy that maintains multiple candidate sequences, exploring the top-k most likely continuations at each step.

**BM25**
: A ranking function used in information retrieval for scoring document relevance. Often used as a baseline or hybrid with vector search in RAG systems.

---

## C

**Chain-of-Thought (CoT)**
: A prompting technique that elicits step-by-step reasoning by asking the model to show its work. Improves performance on complex reasoning tasks.

**Chroma (ChromaDB)**
: An open-source vector database designed for AI applications, particularly RAG systems. Stores embeddings and enables similarity search.

**Chunking**
: Breaking documents into smaller segments for embedding and retrieval. Strategies include fixed-size, sentence-based, and semantic chunking.

**Context Window**
: The maximum number of tokens an LLM can process in a single request (input + output). Claude 3 supports up to 200k tokens.

**Conversation Memory**
: Techniques for maintaining dialogue history across multiple turns, including full history, summarization, and sliding windows.

**CrewAI**
: A framework for orchestrating role-playing autonomous AI agents that work together as a crew to accomplish complex tasks.

---

## D

**Decoding**
: The process of generating output tokens from an LLM's probability distribution. Strategies include greedy, sampling, top-k, nucleus (top-p).

**Deterministic Routing**
: Using rule-based logic to route agent decisions rather than relying solely on LLM judgment. Improves reliability and reduces cost.

**Document Store**
: A database optimized for storing and retrieving full documents, often paired with vector stores in RAG architectures.

---

## E

**Embedding**
: A dense vector representation of text that captures semantic meaning. Enables similarity search by measuring distances in vector space.

**Embedding Model**
: A neural network that converts text into embeddings. Examples: OpenAI ada-002, Anthropic voyage, sentence-transformers.

**Error Handling**
: Strategies for managing tool failures, API errors, and invalid agent actions. Critical for robust production agents.

**Evaluation Metric**
: Quantitative measures for agent performance: task success rate, tool use accuracy, latency, cost per task.

---

## F

**Few-Shot Learning**
: Providing the model with several examples of the desired behavior in the prompt. Effective for teaching specific patterns.

**Fine-Tuning**
: Training an existing model on task-specific data to improve performance. Generally not needed for agents when prompting suffices.

**Function Calling (Tool Calling)**
: LLM capability to output structured requests for external function execution. Enables agents to use tools deterministically.

---

## G

**Goal Decomposition**
: Breaking complex objectives into smaller subgoals that can be tackled sequentially or in parallel.

**GPT-4**
: OpenAI's fourth-generation large language model with strong reasoning, coding, and instruction-following capabilities.

**Greedy Decoding**
: Selecting the highest probability token at each step. Deterministic but can miss better solutions that require exploring lower-probability paths.

**Grounding**
: Connecting LLM outputs to verifiable external information sources to reduce hallucination and improve factual accuracy.

**Guardrails**
: Safety mechanisms that constrain agent behavior: input validation, output filtering, action whitelisting, toxicity detection.

---

## H

**Hallucination**
: When an LLM generates plausible-sounding but factually incorrect or nonsensical information. RAG and tool use help mitigate this.

**Human-in-the-Loop**
: Requiring human approval or intervention at critical decision points in agent workflows.

**Hybrid Search**
: Combining multiple retrieval methods (e.g., BM25 + vector similarity) to improve recall and precision in RAG systems.

**Hyperparameter**
: Configuration settings for LLMs (temperature, top_p, max_tokens) or agents (max_iterations, timeout) that control behavior.

---

## I

**Instruction Following**
: An LLM's ability to correctly interpret and execute directives in prompts. Critical for agent reliability.

**Iterative Refinement**
: An agent pattern where the agent reviews its own output, identifies issues, and generates improved versions.

---

## J

**JSON Mode**
: LLM output mode that guarantees valid JSON responses. Essential for structured tool calls and data extraction.

---

## K

**k-NN (k-Nearest Neighbors)**
: Algorithm for finding the k most similar vectors to a query vector. Used in vector store retrieval.

**Knowledge Base**
: A structured repository of information that agents can query. Can be vector stores, databases, APIs, or knowledge graphs.

---

## L

**LangChain**
: A popular framework for building LLM applications, providing abstractions for agents, chains, memory, and tools.

**LangGraph**
: An extension of LangChain for building stateful, multi-actor applications with explicit control flow graphs.

**LangSmith**
: Anthropic's observability platform for debugging, testing, and monitoring LLM applications.

**Llama**
: Meta's open-source family of large language models. Llama 3 offers strong performance with local deployment options.

**Long-Term Memory**
: Persistent storage of information across sessions, enabling agents to learn from past interactions.

**LLM (Large Language Model)**
: A neural network trained on vast text corpora to predict and generate text. Foundation of modern AI agents.

---

## M

**Max Tokens**
: The maximum number of tokens the model can generate in a single response. Controls output length and cost.

**Memory Buffer**
: A data structure that stores conversation history or relevant context for agent decision-making.

**Metadata Filtering**
: Narrowing vector search by pre-filtering on structured metadata (date, category, source) before semantic similarity.

**Mistral**
: A French AI company producing efficient open-source LLMs. Mixtral uses mixture-of-experts architecture.

**Multi-Agent System**
: Multiple autonomous agents that communicate, coordinate, and collaborate to solve problems beyond individual agent capabilities.

**Multi-Head Attention**
: Parallel attention mechanisms in transformers that allow the model to focus on different aspects simultaneously.

---

## N

**Nucleus Sampling (top-p)**
: A decoding strategy that samples from the smallest set of tokens whose cumulative probability exceeds p. More dynamic than top-k.

---

## O

**Observability**
: Monitoring and instrumenting agent systems to understand behavior: logging tool calls, tracking token usage, measuring latency.

**Orchestration**
: Coordinating multiple agents or tools to accomplish complex workflows. Strategies include sequential, parallel, and hierarchical.

**Output Parser**
: A component that extracts structured data from LLM text output, handling inconsistencies and format variations.

---

## P

**Parameter**
: The learned weights in a neural network. Modern LLMs have billions to hundreds of billions of parameters.

**Phoenix**
: An open-source observability platform for LLM applications, providing tracing, evaluation, and debugging tools.

**Pinecone**
: A managed vector database service optimized for production RAG applications with features like namespaces and metadata filtering.

**Plan-and-Execute**
: An agent pattern that separates planning (generating a task list) from execution (performing each task), improving reliability.

**Prompt**
: The input text provided to an LLM, including instructions, context, examples, and the user query.

**Prompt Engineering**
: The practice of crafting effective prompts to elicit desired LLM behavior through instruction clarity, examples, and structure.

**Prompt Template**
: A reusable prompt structure with variables that are filled in at runtime, enabling consistent agent behavior.

---

## Q

**Query Transformation**
: Reformulating user questions to improve retrieval quality: rephrasing, decomposition, hypothetical document embedding (HyDE).

---

## R

**RAG (Retrieval-Augmented Generation)**
: Augmenting LLM responses with information retrieved from external knowledge sources. Reduces hallucination and provides citations.

**Rate Limiting**
: Controlling the frequency of API calls to comply with provider limits and manage costs.

**ReAct (Reasoning + Acting)**
: An agent framework that interleaves reasoning traces with action execution, improving interpretability and performance.

**Red Teaming**
: Adversarial testing of agents to discover vulnerabilities, jailbreaks, and failure modes before production deployment.

**Reflection**
: An agent evaluating its own performance and adjusting its approach. Can be self-critique or using a separate critic model.

**Retrieval**
: The process of finding relevant information from a knowledge base, typically using semantic similarity in vector space.

**Reranking**
: Applying a second-stage model to reorder retrieved documents by relevance, improving precision over initial retrieval.

---

## S

**Sampling**
: A probabilistic decoding strategy that selects tokens according to the model's probability distribution, controlled by temperature.

**Semantic Search**
: Finding documents by meaning rather than keyword matching, using embedding similarity.

**Self-Consistency**
: Generating multiple reasoning paths and selecting the most frequent answer, improving accuracy on reasoning tasks.

**Sentence Transformers**
: A library for generating sentence and document embeddings using pre-trained models like all-MiniLM-L6-v2.

**Short-Term Memory**
: Temporary storage of recent context within a conversation or task, typically held in the agent's prompt.

**State**
: The agent's current context including goals, observations, tool outputs, and memory. State management is crucial for multi-step reasoning.

**System Prompt**
: Instructions at the beginning of the conversation that define the agent's role, capabilities, and behavioral guidelines.

---

## T

**Temperature**
: A decoding parameter that controls randomness. Low (0-0.3) = deterministic, high (0.7-1.0) = creative.

**Tokenization**
: Breaking text into subword units (tokens) that the model processes. Different models use different tokenizers.

**Tool**
: An external capability that an agent can invoke: APIs, databases, calculators, search engines, code execution.

**Tool Schema**
: A structured description of a tool's name, purpose, parameters, and return format that the LLM uses to generate correct tool calls.

**Top-k Sampling**
: Restricting sampling to the k most likely tokens at each step, balancing diversity and coherence.

**Transformer**
: The neural network architecture underlying modern LLMs, based on self-attention mechanisms and parallel processing.

**Tree-of-Thought (ToT)**
: A reasoning pattern that explores multiple reasoning branches in parallel, evaluating and selecting the most promising paths.

---

## V

**Vector Database**
: A specialized database for storing and querying high-dimensional embeddings, optimized for similarity search.

**Vector Store**
: See Vector Database.

**Viterbi Algorithm**
: A dynamic programming algorithm for finding the most likely sequence of states. Not typically used in modern LLMs but relevant for understanding sequence modeling.

---

## W

**Working Memory**
: The portion of an agent's context actively used for current reasoning and decision-making.

**Wrapper**
: An abstraction layer that provides a consistent interface to different LLM providers or tools.

---

## Z

**Zero-Shot**
: Prompting an LLM to perform a task without providing examples, relying on its pre-training knowledge and instruction-following abilities.

---

## Common Acronyms

| Acronym | Full Name |
|---------|-----------|
| LLM | Large Language Model |
| RAG | Retrieval-Augmented Generation |
| CoT | Chain-of-Thought |
| ToT | Tree-of-Thought |
| ReAct | Reasoning + Acting |
| API | Application Programming Interface |
| JSON | JavaScript Object Notation |
| k-NN | k-Nearest Neighbors |
| BM25 | Best Match 25 (ranking function) |
| HyDE | Hypothetical Document Embedding |
| MCP | Model Context Protocol |

---

## Framework Comparison

| Framework | Best For |
|-----------|----------|
| LangChain | General-purpose RAG and simple agents |
| LangGraph | Complex stateful workflows with control flow |
| CrewAI | Role-based multi-agent collaboration |
| Autogen | Conversational multi-agent systems |
| Custom | Maximum control and optimization |

---

## Typical Agent Hyperparameters

| Parameter | Typical Range | Purpose |
|-----------|---------------|---------|
| Temperature | 0-1 | Controls randomness (0=deterministic) |
| Max Tokens | 500-4000 | Limits response length |
| Max Iterations | 5-20 | Prevents infinite loops |
| Timeout (seconds) | 30-300 | Task time limit |
| Context Window | 4k-200k | How much history to include |

---

*This glossary covers core concepts for the Agentic AI & LLMs course. For deeper explanations, see module guides.*
