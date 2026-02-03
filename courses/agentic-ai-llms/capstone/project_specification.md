# Capstone Project: Production Agentic AI System

## Overview

Design, implement, and deploy a complete production-grade agentic AI system that autonomously accomplishes complex tasks using LLMs, tools, memory, and reasoning. This project integrates all course concepts into a real-world application.

**Weight:** 35% of final grade
**Duration:** Weeks 10-13
**Team Size:** Individual (groups of 2 optional with approval)

---

## Learning Objectives Demonstrated

By completing this project, you will demonstrate mastery of:

1. **Agent Architecture:** Designing systems with appropriate reasoning patterns, memory, and tool sets
2. **Tool Integration:** Building robust tool interfaces with error handling and validation
3. **RAG Implementation:** Creating knowledge bases with effective retrieval strategies
4. **Multi-Agent Coordination:** Orchestrating specialized agents or complex workflows
5. **Production Engineering:** Deploying with observability, cost management, and safety guardrails
6. **Evaluation:** Systematically measuring agent performance and reliability

---

## Project Options

Choose ONE project type:

### Option A: Research Assistant Agent

Build an agent that autonomously conducts research by:
- Querying multiple information sources (web search, ArXiv, databases)
- Synthesizing information across sources
- Fact-checking and source attribution
- Generating structured reports with citations

**Example Use Cases:**
- Scientific literature review
- Market research compilation
- Competitive intelligence gathering
- Technical documentation synthesis

### Option B: Customer Support Agent

Build a multi-capability support agent that:
- Retrieves information from documentation (RAG)
- Accesses customer data and order history
- Performs actions (refunds, ticket creation, password resets)
- Escalates complex issues appropriately

**Example Use Cases:**
- E-commerce support
- SaaS product support
- Technical troubleshooting
- Account management

### Option C: Data Analysis Agent

Build an agent that autonomously analyzes data by:
- Interpreting natural language data questions
- Writing and executing SQL or pandas code
- Generating visualizations
- Producing narrative insights and recommendations

**Example Use Cases:**
- Business intelligence automation
- Exploratory data analysis
- Report generation
- Anomaly investigation

### Option D: Content Creation Agent

Build an agent that produces high-quality content by:
- Researching topics using RAG and web search
- Planning content structure
- Drafting with appropriate tone and style
- Self-reviewing and refining output

**Example Use Cases:**
- Technical blog posts
- Marketing copy
- Documentation generation
- Report writing

### Option E: Custom Proposal (Requires Approval)

Propose your own agentic system. Must demonstrate:
- Multi-step reasoning and planning
- Tool use (minimum 3 tools)
- Memory or state management
- Clear evaluation criteria

---

## Core Requirements (Must Complete All)

### 1. Agent Architecture (20 points)

- [ ] Clear system design with component diagram
- [ ] Appropriate reasoning pattern (ReAct, Plan-Execute, or custom)
- [ ] Well-defined agent roles if multi-agent
- [ ] State management strategy documented
- [ ] Failure recovery mechanisms

**Evaluation Criteria:**
- Architecture clarity and justification
- Appropriate pattern selection
- Scalability considerations
- Error handling design

### 2. Tool Implementation (20 points)

Implement at least THREE tools with:
- [ ] Clear tool schemas (name, description, parameters)
- [ ] Input validation and type checking
- [ ] Comprehensive error handling
- [ ] Appropriate timeouts and rate limiting
- [ ] Unit tests for each tool

**Evaluation Criteria:**
- Tool interface quality
- Error handling robustness
- Security considerations (input sanitization)
- Test coverage

### 3. Memory & RAG (20 points)

If applicable to your project (required for options A, B):
- [ ] Knowledge base construction with appropriate chunking
- [ ] Embedding model selection and justification
- [ ] Retrieval strategy (hybrid search, reranking, etc.)
- [ ] Source attribution in responses
- [ ] Conversation memory management

**Evaluation Criteria:**
- Retrieval quality (precision/recall)
- Chunking strategy appropriateness
- Memory efficiency
- Context management

### 4. Reasoning & Planning (15 points)

- [ ] Clear demonstration of multi-step reasoning
- [ ] Goal decomposition for complex tasks
- [ ] Self-reflection or self-correction capability
- [ ] Reasoning traces logged for interpretability

**Evaluation Criteria:**
- Reasoning depth and coherence
- Task completion success rate
- Transparency of decision-making
- Handling of ambiguous inputs

### 5. Evaluation & Testing (15 points)

- [ ] Define success metrics for your agent
- [ ] Create test suite with at least 10 test cases
- [ ] Measure task success rate, accuracy, latency
- [ ] Compare against baseline (e.g., single-shot LLM)
- [ ] Error analysis and failure mode documentation

**Evaluation Criteria:**
- Metrics appropriateness
- Test coverage breadth
- Rigorous comparison methodology
- Insightful failure analysis

### 6. Production Engineering (10 points)

- [ ] Observability: logging, tracing, token counting
- [ ] Cost tracking and optimization strategies
- [ ] Safety guardrails (input validation, output filtering)
- [ ] API rate limiting and timeout handling
- [ ] Deployment instructions (Docker optional but recommended)

**Evaluation Criteria:**
- Observability completeness
- Cost-consciousness
- Safety implementation
- Deployment clarity

---

## Extension Options (Choose 1-2 for Bonus)

Each extension worth up to 5 bonus points:

1. **Multi-Agent System:** Implement multiple specialized agents with coordination
2. **Human-in-the-Loop:** Add approval workflows for critical actions
3. **Web UI:** Build interactive frontend (Streamlit, Gradio, or custom)
4. **Advanced RAG:** Implement query transformation, hybrid search, or reranking
5. **Continuous Learning:** Agent learns from feedback to improve over time
6. **Comprehensive Benchmarking:** Evaluate on standard benchmark (e.g., WebArena, SWE-bench subset)

---

## Milestones & Checkpoints

### Milestone 1: Proposal (Week 10) — 5%
**Deliverable:** 2-page proposal

- Project option selection and justification
- System architecture diagram
- Tool requirements identified
- Success metrics defined
- Timeline with weekly goals

**Grading:**
- Clear, feasible scope: 2 points
- Appropriate architecture: 2 points
- Realistic timeline: 1 point

### Milestone 2: Prototype (Week 11) — 10%
**Deliverable:** Working prototype + demo video (3 min)

- Basic agent loop functional
- At least 1 tool working
- Simple test cases passing
- Demo showing core capability

**Grading:**
- Functional prototype: 4 points
- Tool quality: 3 points
- Demo clarity: 3 points

### Milestone 3: Evaluation Results (Week 12) — 10%
**Deliverable:** Evaluation report + test suite

- Complete test suite
- Performance metrics calculated
- Comparison with baseline
- Error analysis documented

**Grading:**
- Test comprehensiveness: 4 points
- Metrics rigor: 3 points
- Analysis depth: 3 points

### Milestone 4: Final Submission (Week 13) — 75%
**Deliverables:** Complete system + report + presentation

See detailed rubric below.

---

## Technical Report Template

### Structure (4-6 pages, excluding figures/code)

1. **Executive Summary** (0.5 pages)
   - Problem statement
   - Solution approach
   - Key results and insights

2. **System Architecture** (1-1.5 pages)
   - Component diagram
   - Reasoning pattern and justification
   - Tool descriptions
   - Memory/RAG strategy (if applicable)

3. **Implementation Details** (1-1.5 pages)
   - Key technical decisions
   - Challenges encountered and solutions
   - Code organization
   - Technology stack

4. **Evaluation** (1-1.5 pages)
   - Methodology
   - Quantitative results (tables/graphs)
   - Qualitative analysis
   - Comparison with baselines

5. **Discussion** (0.5-1 page)
   - Limitations and failure modes
   - Production considerations
   - Future improvements
   - Lessons learned

6. **Appendix**
   - Tool schemas
   - Example agent traces
   - Test case details

### Formatting
- 11pt font, 1.5 spacing
- Include code snippets where illustrative
- Clear figures and diagrams

---

## Presentation Rubric

### Structure (12 minutes total)
- Problem and motivation: 2 min
- System architecture: 3 min
- Live demo: 4 min
- Evaluation results: 2 min
- Q&A: 1 min

### Evaluation Criteria

| Criterion | Excellent (5) | Good (4) | Adequate (3) | Needs Work (1-2) |
|-----------|---------------|----------|--------------|------------------|
| **Clarity** | Crystal clear, engaging | Clear, well-paced | Understandable | Confusing |
| **Technical Depth** | Demonstrates mastery | Strong understanding | Basic understanding | Superficial |
| **Demo Quality** | Impressive, smooth | Works well | Functions | Buggy or limited |
| **Results Analysis** | Insightful, rigorous | Good analysis | Basic reporting | Weak or missing |
| **Q&A Handling** | Excellent responses | Answers well | Adequate | Struggles |

---

## Final Grading Rubric

### Agent Architecture (20 points)
| Points | Criteria |
|--------|----------|
| 18-20 | Excellent design; well-justified; robust error handling; clear documentation |
| 15-17 | Good architecture; solid justification; minor issues |
| 12-14 | Adequate design; some justification; notable gaps |
| 0-11 | Poor architecture; weak justification; major issues |

### Tool Implementation (20 points)
| Points | Criteria |
|--------|----------|
| 18-20 | Multiple robust tools; excellent error handling; well-tested; secure |
| 15-17 | Good tools; solid error handling; adequately tested |
| 12-14 | Basic tools; some error handling; limited testing |
| 0-11 | Poor tool quality; inadequate error handling; minimal testing |

### Memory & RAG (20 points)
| Points | Criteria |
|--------|----------|
| 18-20 | Excellent retrieval quality; optimal chunking; effective memory |
| 15-17 | Good retrieval; reasonable chunking; working memory |
| 12-14 | Adequate retrieval; basic chunking; simple memory |
| 0-11 | Poor retrieval; weak chunking; minimal memory |

*Note: If RAG not applicable, these points redistribute to Architecture and Tools*

### Reasoning & Planning (15 points)
| Points | Criteria |
|--------|----------|
| 14-15 | Sophisticated multi-step reasoning; effective planning; strong self-correction |
| 11-13 | Good reasoning; solid planning; some self-correction |
| 8-10 | Basic reasoning; simple planning; limited self-correction |
| 0-7 | Weak reasoning; poor planning; no self-correction |

### Evaluation & Testing (15 points)
| Points | Criteria |
|--------|----------|
| 14-15 | Comprehensive testing; rigorous metrics; insightful analysis |
| 11-13 | Good testing; solid metrics; useful analysis |
| 8-10 | Basic testing; adequate metrics; limited analysis |
| 0-7 | Minimal testing; poor metrics; weak analysis |

### Production Engineering (10 points)
| Points | Criteria |
|--------|----------|
| 9-10 | Excellent observability; strong safety; clear deployment |
| 7-8 | Good observability; solid safety; working deployment |
| 5-6 | Basic observability; some safety; unclear deployment |
| 0-4 | Poor observability; weak safety; no deployment path |

---

## Academic Integrity

- This is individual work (or approved pairs)
- Cite any external code, libraries, or resources
- You may use LLM assistants for debugging and code generation, but:
  - You must understand all code you submit
  - Core architecture and design must be your own
  - Document AI assistance used
- All code must be explainable in Q&A

---

## Resources

### Frameworks & Libraries
- **LangChain:** [python.langchain.com](https://python.langchain.com)
- **LangGraph:** [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph)
- **CrewAI:** [crewai.com](https://crewai.com)
- **Autogen:** [microsoft.github.io/autogen](https://microsoft.github.io/autogen)

### Vector Databases
- **Chroma:** [trychroma.com](https://trychroma.com)
- **Pinecone:** [pinecone.io](https://pinecone.io)
- **Weaviate:** [weaviate.io](https://weaviate.io)

### Observability
- **LangSmith:** [smith.langchain.com](https://smith.langchain.com)
- **Phoenix:** [phoenix.arize.com](https://phoenix.arize.com)

### Code Templates
- See `capstone/templates/` for starter code
- Reference implementations in course notebooks

### Office Hours
- Extended hours during capstone period
- Architecture review sessions available

---

## Submission Instructions

1. **Create GitHub repository** with clear structure:
   ```
   project-name/
   ├── README.md              # Setup and usage instructions
   ├── src/
   │   ├── agent.py          # Core agent implementation
   │   ├── tools/            # Tool implementations
   │   └── utils/            # Helper functions
   ├── tests/                # Test suite
   ├── data/                 # Sample data or knowledge base
   ├── docs/
   │   ├── report.pdf        # Technical report
   │   └── architecture.png  # System diagram
   └── requirements.txt      # Dependencies
   ```

2. **Submit via course platform:**
   - Link to GitHub repository
   - Technical report (PDF)
   - Presentation slides (PDF)
   - Demo video (5 min max, optional)

3. **Ensure reproducibility:**
   - Clear setup instructions in README
   - Requirements.txt or environment.yml
   - Sample data or instructions to obtain data
   - .env.example showing required API keys (not actual keys!)

---

## Evaluation Timeline

- **Proposals due:** End of Week 10
- **Prototypes due:** End of Week 11
- **Evaluation reports due:** End of Week 12
- **Final submissions due:** End of Week 13
- **Presentations:** Week 13 (scheduled)

---

*"The measure of an agent is not just what it can do, but how reliably it does it. Build systems you'd trust in production."*
