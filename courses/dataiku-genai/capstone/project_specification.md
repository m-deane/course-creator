# Capstone Project: Enterprise Gen AI Application in Dataiku

## Overview

Design and build a production-ready generative AI application using Dataiku's LLM Mesh. Apply RAG, prompt engineering, and MLOps best practices to solve a real business problem with LLMs.

**Weight:** 30% of final grade
**Duration:** Weeks 7-9

---

## Learning Objectives Demonstrated

By completing this project, you will demonstrate mastery of:

1. **LLM Mesh Configuration:** Setting up and managing multiple LLM providers
2. **RAG Implementation:** Building effective Knowledge Banks with retrieval optimization
3. **Prompt Engineering:** Designing robust prompts using Prompt Studios
4. **Application Development:** Creating end-user interfaces (Webapps or APIs)
5. **MLOps:** Implementing monitoring, cost tracking, and deployment workflows
6. **Business Impact:** Delivering measurable value with Gen AI

---

## Project Options

Choose ONE application type:

### Option A: Intelligent Document Assistant

Build a RAG system that answers questions about enterprise documents.

**Components:**
- Knowledge Bank with company documents (PDFs, manuals, policies)
- Query understanding and reformulation
- Source attribution and citation
- Confidence scoring
- User-friendly webapp interface

**Example Use Cases:**
- HR policy assistant
- Technical documentation search
- Compliance question answering
- Onboarding assistant

### Option B: Data Analysis Copilot

Build an LLM that generates insights from structured data.

**Components:**
- SQL/Python code generation from natural language
- Automated data quality checking
- Narrative insight generation
- Visualization recommendations
- Result explanation in business terms

**Example Use Cases:**
- Business intelligence assistant
- Sales analytics copilot
- Financial reporting automation
- Marketing metrics interpreter

### Option C: Content Generation System

Build an automated content creation pipeline.

**Components:**
- Research gathering (web search, internal docs)
- Content planning and outlining
- Draft generation with brand voice
- Quality review and refinement
- Multi-format output (reports, emails, summaries)

**Example Use Cases:**
- Customer communication drafting
- Report generation from data
- Marketing content creation
- Technical documentation writing

### Option D: Classification & Labeling Pipeline

Build an LLM-powered data labeling and classification system.

**Components:**
- Few-shot classification prompts
- Batch processing of unlabeled data
- Confidence-based active learning
- Human-in-the-loop review
- Training data generation for specialized models

**Example Use Cases:**
- Customer feedback categorization
- Ticket routing and prioritization
- Content moderation
- Sentiment analysis

### Option E: Custom Proposal (Requires Approval)

Propose your own Gen AI application. Must demonstrate:
- LLM Mesh integration with 2+ providers
- Knowledge Bank or advanced prompt engineering
- End-user interface (webapp or API)
- Business value measurement

---

## Core Requirements (Must Complete All)

### 1. LLM Mesh Configuration (10 points)

- [ ] Configure at least 2 LLM providers (e.g., OpenAI + Anthropic)
- [ ] Set up connections with appropriate defaults
- [ ] Implement cost tracking
- [ ] Document provider selection rationale
- [ ] Test failover between providers

**Grading:**
- Configuration completeness: 4 points
- Provider selection justification: 3 points
- Documentation: 3 points

### 2. Prompt Engineering (20 points)

- [ ] Design prompts using Prompt Studios
- [ ] Create templates with variables
- [ ] Implement few-shot examples where appropriate
- [ ] Test with diverse inputs
- [ ] Version control prompts
- [ ] Document prompt evolution and iteration

**Grading:**
- Prompt quality and robustness: 10 points
- Template design: 5 points
- Testing thoroughness: 3 points
- Documentation: 2 points

### 3. Knowledge Bank / RAG (20 points)
*(Skip if Option B or D; redistribute points to other sections)*

- [ ] Create Knowledge Bank with relevant documents
- [ ] Choose appropriate chunking strategy
- [ ] Select and justify embedding model
- [ ] Configure retrieval parameters (top-k, metadata filters)
- [ ] Implement source attribution
- [ ] Evaluate retrieval quality

**Grading:**
- Knowledge Bank design: 8 points
- Retrieval quality: 7 points
- Source attribution: 3 points
- Evaluation: 2 points

### 4. Data Processing & Integration (15 points)

- [ ] Build data flow with appropriate recipes
- [ ] Handle input validation and cleaning
- [ ] Implement error handling for LLM failures
- [ ] Create output datasets with results
- [ ] Optimize for batch processing

**Grading:**
- Flow design: 6 points
- Error handling: 5 points
- Optimization: 4 points

### 5. User Interface (15 points)

Choose ONE:
- [ ] **Webapp:** Interactive UI for end users
- [ ] **API Endpoint:** RESTful API for integration
- [ ] **Dashboard:** Visualization of results and metrics

**Grading:**
- Functionality: 8 points
- User experience: 4 points
- Documentation: 3 points

### 6. Evaluation & Quality Assurance (10 points)

- [ ] Define success metrics for your application
- [ ] Create test dataset with ground truth (if applicable)
- [ ] Measure accuracy, relevance, or quality
- [ ] Implement quality checks (output validation, hallucination detection)
- [ ] Document failure modes and limitations

**Grading:**
- Metrics appropriateness: 4 points
- Testing rigor: 4 points
- Failure analysis: 2 points

### 7. MLOps & Production Readiness (10 points)

- [ ] Implement logging and monitoring
- [ ] Track token usage and costs
- [ ] Set up automated scenarios (if applicable)
- [ ] Document deployment instructions
- [ ] Consider security (data privacy, access control)

**Grading:**
- Observability: 4 points
- Cost management: 3 points
- Deployment readiness: 3 points

---

## Extension Options (Choose 1-2 for Bonus)

Each extension worth up to 5 bonus points:

1. **Multi-Provider Intelligence**
   - Route queries to optimal provider based on task
   - Compare responses across providers
   - Implement ensemble or voting

2. **Human-in-the-Loop Workflow**
   - Review queue for uncertain predictions
   - Feedback collection mechanism
   - Continuous improvement from corrections

3. **Advanced RAG Techniques**
   - Hybrid search (keyword + semantic)
   - Query transformation or expansion
   - Reranking with cross-encoder
   - Hypothetical document embedding (HyDE)

4. **Multi-Lingual Support**
   - Handle queries in multiple languages
   - Translate documents automatically
   - Language detection and routing

5. **Comprehensive Benchmarking**
   - Compare against baseline methods
   - A/B testing framework
   - Systematic prompt optimization

---

## Milestones & Checkpoints

### Milestone 1: Design & Setup (Week 7) — 10%
**Deliverable:** Project proposal + initial Dataiku project

- Application type selection and justification
- LLM Mesh connections configured
- Data sources identified
- Initial flow design

**Grading:**
- Feasibility and scope: 4 points
- Configuration quality: 4 points
- Planning: 2 points

### Milestone 2: Core Implementation (Week 8) — 15%
**Deliverable:** Working prototype

- Basic LLM functionality working
- Knowledge Bank built (if applicable)
- Initial prompts tested
- Sample outputs generated

**Grading:**
- Functionality: 8 points
- Prompt quality: 5 points
- Documentation: 2 points

### Milestone 3: Final Submission (Week 9) — 75%
**Deliverables:** Complete application + report + demo

See detailed rubric below.

---

## Technical Report Template

### Structure (4-5 pages, excluding appendices)

1. **Executive Summary** (0.5 pages)
   - Business problem addressed
   - Solution approach
   - Key results and impact
   - Recommendations

2. **Business Context** (0.5-1 page)
   - Problem description
   - Current manual process (if applicable)
   - Success criteria
   - Stakeholder requirements

3. **Technical Architecture** (1-1.5 pages)
   - Dataiku flow diagram
   - LLM providers and selection rationale
   - Knowledge Bank design (if applicable)
   - Prompt strategy
   - Key technical decisions

4. **Implementation Details** (1-1.5 pages)
   - Prompt examples
   - Retrieval configuration
   - Error handling approach
   - Cost optimization strategies

5. **Evaluation & Results** (0.5-1 page)
   - Quantitative metrics
   - Qualitative assessment
   - Example outputs
   - Limitations and failure modes

6. **Deployment & Operations** (0.5 page)
   - Deployment approach
   - Monitoring and logging
   - Cost projections
   - Maintenance plan

7. **Appendix**
   - Full prompt templates
   - API documentation (if applicable)
   - Screenshot walkthroughs

---

## Presentation Rubric

### Structure (10 minutes total)
- Problem and business case: 2 min
- Architecture and approach: 2 min
- Live demo: 4 min
- Results and impact: 1 min
- Q&A: 1 min

### Evaluation Criteria

| Criterion | Excellent (5) | Good (4) | Adequate (3) | Needs Work (1-2) |
|-----------|---------------|----------|--------------|------------------|
| **Business Value** | Clear ROI, strong impact | Good value case | Some value | Weak justification |
| **Technical Quality** | Robust, production-ready | Solid implementation | Basic functionality | Buggy or incomplete |
| **Demo** | Impressive, smooth | Works well | Functions | Struggles or limited |
| **Understanding** | Deep Dataiku/LLM mastery | Strong grasp | Adequate | Superficial |
| **Q&A** | Insightful responses | Answers well | Basic answers | Cannot defend |

---

## Final Grading Rubric

### LLM Mesh Configuration (10 points)
| Points | Criteria |
|--------|----------|
| 9-10 | Multiple providers; optimal settings; excellent documentation |
| 7-8 | Good configuration; reasonable settings; solid documentation |
| 5-6 | Basic setup; default settings; minimal documentation |
| 0-4 | Incomplete or poorly configured |

### Prompt Engineering (20 points)
| Points | Criteria |
|--------|----------|
| 18-20 | Sophisticated prompts; thorough testing; excellent iteration |
| 15-17 | Good prompts; solid testing; reasonable iteration |
| 12-14 | Basic prompts; some testing; limited iteration |
| 0-11 | Weak prompts; minimal testing; no iteration |

### Knowledge Bank / RAG (20 points)
| Points | Criteria |
|--------|----------|
| 18-20 | Excellent retrieval; optimal chunking; strong evaluation |
| 15-17 | Good retrieval; solid chunking; adequate evaluation |
| 12-14 | Basic retrieval; simple chunking; weak evaluation |
| 0-11 | Poor retrieval; inappropriate chunking; no evaluation |

### Data Processing (15 points)
| Points | Criteria |
|--------|----------|
| 14-15 | Robust flow; excellent error handling; well-optimized |
| 11-13 | Good flow; solid error handling; adequately optimized |
| 8-10 | Basic flow; some error handling; minimal optimization |
| 0-7 | Weak flow; poor error handling; not optimized |

### User Interface (15 points)
| Points | Criteria |
|--------|----------|
| 14-15 | Professional UI; excellent UX; comprehensive documentation |
| 11-13 | Good UI; solid UX; good documentation |
| 8-10 | Basic UI; adequate UX; minimal documentation |
| 0-7 | Poor UI; weak UX; no documentation |

### Evaluation & QA (10 points)
| Points | Criteria |
|--------|----------|
| 9-10 | Rigorous evaluation; appropriate metrics; thorough testing |
| 7-8 | Good evaluation; solid metrics; adequate testing |
| 5-6 | Basic evaluation; simple metrics; limited testing |
| 0-4 | Minimal evaluation; poor metrics; no testing |

### MLOps & Production (10 points)
| Points | Criteria |
|--------|----------|
| 9-10 | Comprehensive observability; cost tracking; deployment-ready |
| 7-8 | Good monitoring; cost awareness; mostly deployment-ready |
| 5-6 | Basic monitoring; some cost tracking; unclear deployment |
| 0-4 | No monitoring; no cost tracking; not deployment-ready |

---

## Academic Integrity

- This is individual work
- You may use Dataiku documentation and community resources
- Cite any external code or prompt templates
- Document any AI assistance (e.g., using ChatGPT for debugging)
- All work must be explainable in presentation

---

## Resources

### Dataiku Documentation
- **LLM Mesh:** [doc.dataiku.com/dss/latest/machine-learning/llm](https://doc.dataiku.com/dss/latest/machine-learning/llm)
- **Knowledge Banks:** [doc.dataiku.com/dss/latest/knowledge-banks](https://doc.dataiku.com/dss/latest/knowledge-banks)
- **Webapps:** [doc.dataiku.com/dss/latest/webapps](https://doc.dataiku.com/dss/latest/webapps)

### Sample Datasets
- Company documentation (PDFs, manuals)
- Customer support tickets
- Sales/financial data
- Product catalogs
- Policy documents

### Code Libraries
- LangChain (for advanced workflows)
- OpenAI/Anthropic Python SDKs
- Sentence-transformers (embeddings)

### Office Hours
- Extended support during capstone period
- Architecture review sessions available

---

## Submission Instructions

1. **Export Dataiku Project:**
   - Project → Export → Include all datasets/scenarios
   - Save bundle for submission

2. **Create Documentation Package:**
   ```
   dataiku-genai-capstone/
   ├── README.md                # Setup and usage instructions
   ├── project_bundle.zip       # Exported Dataiku project
   ├── docs/
   │   ├── report.pdf          # Technical report
   │   ├── architecture.png    # Flow diagram
   │   └── screenshots/        # UI screenshots
   ├── prompts/
   │   └── prompt_templates.md # Prompt documentation
   └── evaluation/
       ├── test_cases.csv      # Test data
       └── results.csv         # Evaluation results
   ```

3. **Submit via course platform:**
   - Dataiku project bundle
   - PDF technical report
   - Presentation slides
   - Demo video (5 min max, optional)

4. **Dataiku Project Requirements:**
   - Clear project name and description
   - Organized flow with meaningful recipe names
   - Comments in Python recipes
   - README dataset with setup instructions
   - Sample data included (or instructions to obtain)

---

## Evaluation Timeline

- **Proposals due:** End of Week 7
- **Prototypes due:** End of Week 8
- **Final submissions due:** End of Week 9
- **Presentations:** Week 9 (scheduled)

---

*"Enterprise Gen AI is not just about the LLM—it's about the entire system: data integration, governance, user experience, and business value. Build complete solutions, not just prompts."*
