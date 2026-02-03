# Glossary: Gen AI & Dataiku LLM Mesh

## A

**API Endpoint**
: A URL that exposes LLM capabilities as a service. Dataiku can publish LLM applications as API endpoints for integration.

**API Key Management**
: Centralized storage and access control for LLM provider API keys in Dataiku connections.

**Azure OpenAI**
: Microsoft's enterprise deployment of OpenAI models with additional compliance and security features. Supported LLM Mesh provider.

---

## B

**Batch Prediction**
: Running LLM inference on multiple inputs in a dataset using Dataiku recipes. More efficient than individual calls.

**Bundle**
: A Dataiku package containing code, models, and configurations that can be deployed across environments.

---

## C

**Chain**
: A sequence of LLM calls where output from one becomes input to another. Supported in Dataiku's Python recipes.

**Chat Completion**
: LLM API format for multi-turn conversations with message history. Standard interface in LLM Mesh.

**Claude (Anthropic)**
: Large language model family known for long context and safety. Key LLM Mesh provider.

**Code Environment**
: Isolated Python environment in Dataiku with specific package versions. Required for LLM libraries.

**Completion**
: Single-turn text generation from an LLM. Legacy API format, largely replaced by chat completion.

**Connection**
: Dataiku object storing credentials and configuration for external services like LLM providers.

**Context Length**
: Maximum number of tokens an LLM can process in one request. Varies by model (4k to 200k+).

**Cost Tracking**
: Monitoring token usage and API costs across LLM calls. Built into Dataiku LLM Mesh.

---

## D

**Dataset**
: Fundamental Dataiku object storing tabular data. Can contain LLM inputs, outputs, and embeddings.

**Dataiku DSS (Data Science Studio)**
: Dataiku's unified platform for data preparation, ML, and AI application development.

**Deployment**
: Publishing LLM applications to production environments (API Deployer, automation node).

**Det**
: Deprecated term; see Project.

---

## E

**Embedding**
: Dense vector representation of text. Dataiku supports embedding generation through LLM Mesh and integration with vector stores.

**Embedding Model**
: Specialized model that converts text to vectors. Examples: OpenAI ada-002, Cohere embed-english.

**Enrichment Recipe**
: Dataiku recipe that adds LLM-generated columns to existing datasets.

---

## F

**Few-Shot Prompting**
: Including examples in prompts to guide LLM behavior. Can be configured in Prompt Studios.

**Flow**
: Dataiku's visual representation of data pipelines and transformations.

**Function Calling**
: LLM capability to output structured tool/API calls. Supported by GPT-4 and Claude in LLM Mesh.

---

## G

**Gemini**
: Google's multimodal LLM family. Can be integrated with Dataiku via Vertex AI.

**Governance**
: Policies and controls around LLM usage, costs, and access. Centralized in Dataiku.

**GPT-3.5 / GPT-4**
: OpenAI's large language model families. Primary LLM Mesh providers.

**Guardrails**
: Safety mechanisms that filter inputs/outputs or constrain LLM behavior. Implemented via Python recipes or plugins.

---

## H

**Hugging Face**
: Platform for open-source ML models. Can integrate custom models with Dataiku.

**Hyperparameter**
: LLM configuration like temperature, max_tokens, top_p. Controlled in Dataiku LLM Mesh connections.

---

## I

**Inference**
: Running a model on inputs to generate outputs. For LLMs, this is text generation.

**Input Token**
: Token in the prompt or input. Typically cheaper than output tokens.

**Instance (Dataiku)**
: A Dataiku DSS installation. LLM Mesh configured at instance level.

---

## J

**JSON Mode**
: LLM output mode that guarantees valid JSON. Useful for structured data extraction in Dataiku recipes.

---

## K

**Knowledge Bank**
: Dataiku's RAG implementation. Stores documents, generates embeddings, and enables semantic retrieval.

**Knowledge Source**
: Documents added to a Knowledge Bank: datasets, files, folders, or external URLs.

---

## L

**LangChain**
: Python framework for LLM applications. Can be used in Dataiku Python recipes for advanced workflows.

**LLM (Large Language Model)**
: Neural network trained on vast text data for generation and understanding. Core of Gen AI.

**LLM Connection**
: Dataiku connection object storing LLM provider credentials and default parameters.

**LLM Mesh**
: Dataiku's abstraction layer providing unified interface to multiple LLM providers with governance.

**LLM Recipe**
: Dataiku visual recipe for applying LLM operations to datasets without code.

---

## M

**Managed Folder**
: Dataiku object for storing unstructured files. Used for document storage in Knowledge Banks.

**Max Tokens**
: Maximum output length. Configured in LLM connections or per-call.

**Metadata Filtering**
: Narrowing retrieval in Knowledge Banks by filtering on document attributes before semantic search.

**Model Context Protocol (MCP)**
: Emerging standard for LLM-tool integration. May be supported in future Dataiku versions.

**Multimodal**
: LLMs that process multiple input types (text, images, audio). Example: GPT-4 Vision.

---

## N

**Notebook**
: Interactive coding environment in Dataiku (Jupyter). Useful for LLM experimentation.

---

## O

**OpenAI**
: Leading LLM provider (GPT-3.5, GPT-4). Primary LLM Mesh integration.

**Output Parser**
: Code that extracts structured data from LLM text responses. Implemented in Python recipes.

**Output Token**
: Token generated by the LLM. Typically 2-3x more expensive than input tokens.

---

## P

**Plugin**
: Extension adding functionality to Dataiku. LLM-related plugins available from store.

**Preparator**
: Dataiku recipe for data cleaning and transformation. Often used to format inputs for LLMs.

**Project**
: Organizational unit in Dataiku containing datasets, recipes, models, and workflows.

**Prompt**
: Input text provided to an LLM, including instructions, context, and query.

**Prompt Engineering**
: Crafting effective prompts to elicit desired LLM behavior. Supported by Prompt Studios.

**Prompt Studio**
: Dataiku visual interface for designing, testing, and versioning prompts with template variables.

**Prompt Template**
: Reusable prompt structure with variables filled from dataset columns or user input.

**Provider (LLM)**
: Company offering LLM API access: OpenAI, Anthropic, Google, Azure, etc.

**Python Recipe**
: Dataiku recipe executing custom Python code. Used for advanced LLM workflows.

---

## Q

**Query Transformation**
: Reformulating user questions before retrieval in RAG. Can improve Knowledge Bank results.

---

## R

**RAG (Retrieval-Augmented Generation)**
: Augmenting LLM responses with retrieved information. Implemented via Knowledge Banks in Dataiku.

**Rate Limiting**
: Controlling API call frequency to comply with provider limits. Handled automatically by LLM Mesh.

**Recipe**
: Dataiku transformation step in a flow. Various types: visual, code, ML, LLM.

**Reranking**
: Re-ordering retrieved results by relevance. Can be implemented in Python recipes for Knowledge Banks.

**Retrieval**
: Finding relevant documents from Knowledge Bank using semantic similarity.

**Role (Message)**
: In chat completion format: system, user, or assistant. Defines message source.

---

## S

**Scenario**
: Dataiku automation workflow. Can trigger batch LLM processing or Knowledge Bank updates.

**Semantic Search**
: Finding information by meaning rather than keywords. Core of Knowledge Bank retrieval.

**Shared Code**
: Reusable Python functions available across project recipes. Useful for LLM utilities.

**System Prompt**
: Initial instructions defining LLM behavior and role. Configured in Prompt Studios or code.

---

## T

**Temperature**
: LLM parameter controlling randomness (0=deterministic, 1=creative). Set in connections or per-call.

**Token**
: Subword unit processed by LLMs. English text averages ~1.3 tokens per word.

**Tokenization**
: Breaking text into tokens. Different models use different tokenizers.

**Top-p (Nucleus Sampling)**
: Decoding parameter controlling output diversity. Alternative to temperature.

**Trigger**
: Event that starts a Dataiku scenario, enabling automated LLM workflows.

---

## U

**User Prompt**
: The actual query or task description, distinct from system instructions.

---

## V

**Variable (Prompt)**
: Placeholder in prompt template filled with dataset values or user input. Syntax: `${variable_name}`.

**Vector Database**
: Specialized database for storing and querying embeddings. Dataiku Knowledge Banks use internal vector storage.

**Vector Store**
: See Vector Database.

**Vertex AI**
: Google Cloud's ML platform, providing access to Gemini and other models. Can integrate with Dataiku.

**Version Control**
: Tracking changes to prompts, code, and configurations. Git integration available in Dataiku.

---

## W

**Webapp**
: Interactive application built in Dataiku, can expose LLM capabilities to end users.

**Webapp Framework**
: Dataiku's system for building custom web interfaces using Python/R backends and HTML/JS frontends.

**Workflow**
: See Flow.

---

## Z

**Zero-Shot**
: Prompting without examples, relying on model's pre-training. Simpler than few-shot.

---

## LLM Mesh Providers

| Provider | Models | Key Features |
|----------|--------|--------------|
| **OpenAI** | GPT-3.5, GPT-4, GPT-4 Turbo | Industry standard, function calling |
| **Anthropic** | Claude 2, Claude 3 (Opus/Sonnet/Haiku) | Long context (200k), safety focus |
| **Azure OpenAI** | GPT-3.5, GPT-4 | Enterprise compliance, regional deployment |
| **Google Vertex AI** | Gemini Pro, PaLM 2 | Multimodal, Google Cloud integration |
| **Cohere** | Command, Embed | Specialized for enterprise RAG |
| **Hugging Face** | Various open-source | Self-hosted, customizable |

---

## Knowledge Bank Components

| Component | Purpose |
|-----------|---------|
| **Sources** | Documents/datasets to index |
| **Chunking** | Splitting documents for embedding |
| **Embeddings** | Vector representations of chunks |
| **Vector Store** | Efficient similarity search |
| **Retrieval Config** | Number of results, metadata filters |
| **LLM Integration** | Generating answers from retrieved context |

---

## Dataiku Recipe Types for LLM Work

| Recipe Type | Use Case |
|-------------|----------|
| **LLM Recipe** | No-code LLM operations on datasets |
| **Python Recipe** | Custom LLM logic, complex workflows |
| **Prepare Recipe** | Format data for LLM input |
| **Split Recipe** | Create train/test for LLM evaluation |
| **Join Recipe** | Combine LLM outputs with original data |

---

## Typical LLM Hyperparameters

| Parameter | Range | Purpose | Dataiku Location |
|-----------|-------|---------|------------------|
| Temperature | 0-2 | Randomness control | Connection or recipe |
| Max Tokens | 1-4096+ | Output length limit | Connection or recipe |
| Top-p | 0-1 | Nucleus sampling | Connection or recipe |
| Frequency Penalty | -2 to 2 | Reduce repetition | Connection or recipe |
| Presence Penalty | -2 to 2 | Encourage topic diversity | Connection or recipe |

---

## Cost Optimization Strategies

| Strategy | Implementation in Dataiku |
|----------|---------------------------|
| **Caching** | Store LLM outputs in datasets |
| **Batching** | Process datasets in recipes vs. individual calls |
| **Model Selection** | Use GPT-3.5 where GPT-4 unnecessary |
| **Prompt Optimization** | Use Prompt Studios to minimize tokens |
| **Monitoring** | Track costs via LLM Mesh dashboards |
| **Rate Limiting** | Configure in scenario execution |

---

## Common Abbreviations

| Abbreviation | Full Name |
|--------------|-----------|
| DSS | Data Science Studio |
| LLM | Large Language Model |
| RAG | Retrieval-Augmented Generation |
| API | Application Programming Interface |
| JSON | JavaScript Object Notation |
| SQL | Structured Query Language |
| NLP | Natural Language Processing |
| MLOps | Machine Learning Operations |
| GPT | Generative Pre-trained Transformer |

---

## Dataiku LLM Workflow Patterns

### Pattern 1: Enrichment
```
Dataset → LLM Recipe → Enriched Dataset
```
Add LLM-generated columns (summaries, classifications, extractions).

### Pattern 2: RAG
```
Documents → Knowledge Bank → Query → LLM → Response
```
Answer questions using enterprise documents.

### Pattern 3: Classification
```
Text Dataset → LLM Recipe (classify) → Labeled Dataset → Model
```
Use LLMs for data labeling, then train specialized classifier.

### Pattern 4: Analysis
```
Dataset → Python Recipe (LLM) → Insights Dataset → Visualization
```
Generate analytical insights from structured data.

---

*This glossary covers Dataiku-specific Gen AI terminology. For general LLM concepts, see foundational modules.*
