# Multi-Course Development Prompt

## Overview

Create seven advanced university-level courses following the framework in `.claude_prompts/course_creator.md`. Each course should balance theoretical rigor with practical implementation, targeting graduate students and industry professionals.

---

## Course 1: Agentic AI & Large Language Models

### Course Vision
Build production-grade AI agent systems using modern LLM architectures. Students learn to design, implement, and deploy autonomous agents that can reason, plan, use tools, and collaborate.

### Target Audience
- ML engineers transitioning to LLM-based systems
- Software engineers building AI-powered applications
- Technical leads designing agent architectures

### Module Structure (8 modules)

```
Module 0: Foundations
- Transformer architecture refresher
- Attention mechanisms and context windows
- API-based vs local LLM deployment

Module 1: LLM Fundamentals for Agents
- Prompt engineering principles
- Chain-of-thought and reasoning patterns
- System prompts and persona design
- Token economics and cost optimization

Module 2: Tool Use and Function Calling
- Function calling APIs (OpenAI, Anthropic, open-source)
- Tool design patterns
- Error handling and retry strategies
- Sandboxing and security considerations

Module 3: Memory and Context Management
- Short-term vs long-term memory
- RAG (Retrieval Augmented Generation)
- Vector databases and embeddings
- Conversation summarization strategies

Module 4: Planning and Reasoning
- ReAct pattern (Reasoning + Acting)
- Tree-of-thought and graph-based planning
- Goal decomposition
- Self-reflection and correction

Module 5: Multi-Agent Systems
- Agent communication protocols
- Orchestration patterns (sequential, parallel, hierarchical)
- Consensus and conflict resolution
- Agent specialization and delegation

Module 6: Evaluation and Safety
- Agent benchmarks and evaluation metrics
- Hallucination detection and mitigation
- Guardrails and content filtering
- Red teaming and adversarial testing

Module 7: Production Deployment
- Observability and tracing (LangSmith, Phoenix)
- Caching and latency optimization
- Cost management at scale
- A/B testing agent behaviors

Capstone: Build a multi-agent system for a real-world task
```

### Key Technologies
- Claude API, OpenAI API, open-source models (Llama, Mistral)
- LangChain, LangGraph, CrewAI
- Vector stores (Pinecone, Chroma, Weaviate)
- Evaluation frameworks (RAGAS, DeepEval)

### Practical Focus
- Every module includes working code implementations
- Real API integrations (not mocks)
- Cost-conscious design patterns
- Production deployment considerations

---

## Course 2: Bayesian Time Series Forecasting

### Course Vision
*Already implemented - see `courses/bayesian-commodity-forecasting/`*

Build probabilistic forecasting systems that quantify uncertainty, incorporate domain knowledge through priors, and provide decision-ready predictions.

### Key Differentiators from Existing Course
- Generalize beyond commodities to any time series domain
- Add modules on:
  - Bayesian Neural Networks for time series
  - Conformal prediction for uncertainty calibration
  - Online/streaming Bayesian updates
  - Bayesian optimization for hyperparameter tuning

---

## Course 3: Generative AI for Commodities Trading

### Course Vision
Apply generative AI and LLMs to commodity market analysis, from automated fundamental research to AI-assisted trading strategy development. Bridge the gap between quantitative finance and modern AI.

### Target Audience
- Commodity traders and analysts
- Quantitative researchers in energy/agriculture/metals
- Data scientists in trading firms

### Module Structure (10 modules)

```
Module 0: Foundations
- Commodity market structure review
- LLM capabilities and limitations
- API setup (Claude, OpenAI, open-source)

Module 1: Automated Fundamental Research
- Parsing EIA, USDA, OPEC reports with LLMs
- Extracting structured data from unstructured text
- Building knowledge graphs of market relationships
- Fact verification and hallucination prevention

Module 2: News and Sentiment Analysis
- Real-time news processing pipelines
- Commodity-specific sentiment models
- Event extraction (supply disruptions, policy changes)
- Sentiment-return relationship modeling

Module 3: Report Generation and Summarization
- Automated market commentary generation
- Multi-source synthesis (combining EIA + USDA + news)
- Template-based vs free-form generation
- Human-in-the-loop review workflows

Module 4: RAG for Market Intelligence
- Building commodity knowledge bases
- Embedding strategies for financial text
- Retrieval optimization for market queries
- Combining RAG with real-time data

Module 5: LLM-Assisted Feature Engineering
- Using LLMs to propose features from domain knowledge
- Automated feature documentation
- Feature importance explanation generation
- Code generation for feature pipelines

Module 6: Natural Language Interfaces for Trading
- Conversational market analysis
- Query-to-SQL for fundamental databases
- Voice interfaces for traders
- Audit trails and compliance

Module 7: Generative Models for Scenario Analysis
- Generating plausible market scenarios
- Stress testing with LLM-generated narratives
- Monte Carlo with narrative constraints
- Communicating scenarios to stakeholders

Module 8: AI-Assisted Strategy Development
- LLMs for strategy ideation
- Backtesting code generation
- Automated strategy documentation
- Risk narrative generation

Module 9: Production Systems
- Latency requirements for trading applications
- Model versioning and governance
- Compliance and explainability
- Monitoring and alerting

Capstone: Build an AI-powered commodity research system
```

### Key Technologies
- Claude/GPT-4 for reasoning tasks
- Embedding models for retrieval
- Structured output parsing
- Commodity data APIs (EIA, Bloomberg, Reuters)

### Unique Elements
- Focus on accuracy and verifiability (trading requires correctness)
- Compliance and audit trail considerations
- Integration with existing trading infrastructure
- Real commodity data throughout

---

## Course 4: Panel Regression Models with Fixed and Random Effects

### Course Vision
Master panel data econometrics from theory to implementation. Understand when and why to use fixed vs random effects, handle clustered errors, and build robust models for longitudinal data.

### Target Audience
- Applied economists and policy researchers
- Data scientists working with longitudinal data
- Graduate students in economics, finance, sociology

### Module Structure (8 modules)

```
Module 0: Foundations
- Panel data structure and notation
- Pooled OLS limitations
- Sources of panel data (surveys, administrative, financial)

Module 1: Fixed Effects Models
- The within transformation
- Entity and time fixed effects
- Two-way fixed effects
- Interpreting coefficients with FE
- Computational approaches for large panels

Module 2: Random Effects Models
- The random effects assumption
- GLS estimation
- Comparing RE to pooled OLS
- When RE is appropriate

Module 3: Fixed vs Random Effects
- The Hausman test
- Economic interpretation of the choice
- Correlated random effects (Mundlak approach)
- Practical decision frameworks

Module 4: Standard Errors and Inference
- Heteroskedasticity in panels
- Serial correlation
- Clustered standard errors (one-way, two-way)
- Wild cluster bootstrap
- When clusters are few

Module 5: Dynamic Panel Models
- Lagged dependent variables
- Nickell bias
- GMM estimation (Arellano-Bond, Blundell-Bond)
- Weak instruments in dynamic panels

Module 6: Advanced Topics
- Instrumental variables with panel data
- Difference-in-differences
- Synthetic control methods
- Interactive fixed effects
- High-dimensional fixed effects

Module 7: Implementation
- Python (linearmodels, pyfixest)
- R (plm, fixest, lfe)
- Stata comparison
- Computational efficiency for large panels
- Visualization of panel results

Capstone: Empirical project with real panel dataset
```

### Key Technologies
- Python: linearmodels, pyfixest, statsmodels
- R: fixest, plm, lfe
- Julia: FixedEffectModels.jl

### Datasets for Practice
- Penn World Tables (cross-country economics)
- CRSP/Compustat (firm-level finance)
- NLSY (individual labor economics)
- OECD panels (country-year)

---

## Course 5: Gen AI & Dataiku

### Course Vision
Master Dataiku's Generative AI capabilities to build enterprise AI applications. From LLM Mesh configuration to production deployment, learn to leverage Dataiku's visual and code-based tools for AI solutions.

### Target Audience
- Dataiku users expanding to Gen AI
- Enterprise data scientists
- Analytics teams adopting LLMs

### Module Structure (8 modules)

```
Module 0: Dataiku Platform Foundations
- Dataiku architecture overview
- Projects, datasets, and flows
- Code environments and plugins
- Collaboration and governance features

Module 1: LLM Mesh Configuration
- Connecting LLM providers (OpenAI, Azure, Anthropic, Bedrock)
- Model selection and cost management
- API key management and security
- Rate limiting and quotas

Module 2: Prompt Engineering in Dataiku
- Prompt Studio basics
- Template variables and dynamic prompts
- Few-shot examples management
- Prompt versioning and testing

Module 3: RAG with Dataiku
- Knowledge banks and document processing
- Embedding configuration
- Retrieval settings optimization
- Hybrid search strategies
- Evaluation of RAG quality

Module 4: LLM Recipes and Flows
- Text generation recipes
- Classification and extraction
- Batch processing at scale
- Error handling and retries
- Combining LLM steps with traditional ML

Module 5: Custom LLM Applications
- Dataiku Applications with LLM backends
- Webapp development with Gen AI
- API endpoints for LLM services
- Streamlit integration

Module 6: Fine-Tuning and Custom Models
- When to fine-tune vs prompt engineering
- Dataiku fine-tuning workflows
- Evaluation and comparison
- Model registry integration

Module 7: Production and Governance
- Model deployment bundles
- Monitoring LLM applications
- Cost tracking and optimization
- Audit logs and compliance
- Model governance for Gen AI

Capstone: End-to-end Gen AI application in Dataiku
```

### Use Case Focus
Each module includes real use cases:
- Document Q&A systems
- Automated report generation
- Customer feedback analysis
- Code documentation generation
- Data quality assessment with LLMs

### Prerequisites
- Dataiku DSS experience
- Basic Python knowledge
- Understanding of ML concepts

---

## Course 6: Genetic Algorithms for Feature Selection

### Course Vision
Apply evolutionary optimization to the feature selection problem in time series forecasting. Learn when genetic algorithms outperform traditional methods and how to implement robust feature selection pipelines.

### Target Audience
- Data scientists working with high-dimensional time series
- ML engineers optimizing forecasting pipelines
- Researchers in automated machine learning

### Module Structure (8 modules)

```
Module 0: Foundations
- The curse of dimensionality
- Feature selection vs feature extraction
- Filter, wrapper, and embedded methods
- Why time series is special (temporal leakage, autocorrelation)

Module 1: Genetic Algorithm Fundamentals
- Evolutionary computation principles
- Chromosomes, genes, and fitness
- Selection operators (tournament, roulette, rank)
- Crossover operators (single-point, uniform, specialized)
- Mutation operators and rates

Module 2: Encoding Features for GAs
- Binary encoding (feature in/out)
- Integer encoding (feature subsets)
- Real-valued encoding (feature weights)
- Handling feature interactions
- Encoding time series lags

Module 3: Fitness Functions for Forecasting
- Prediction accuracy metrics (RMSE, MAE, MAPE)
- Information criteria (AIC, BIC)
- Cross-validation strategies for time series
- Multi-objective fitness (accuracy + parsimony)
- Computational cost considerations

Module 4: GA Implementation
- From-scratch implementation in Python
- DEAP library deep dive
- Parallelization strategies
- Handling constraints (min/max features)
- Early stopping and convergence detection

Module 5: Advanced Operators
- Adaptive mutation rates
- Niching and speciation
- Memetic algorithms (local search + GA)
- Island models for diversity
- Co-evolution strategies

Module 6: GA vs Alternative Methods
- Comparison with LASSO/Elastic Net
- Comparison with tree-based importance
- Comparison with recursive feature elimination
- Comparison with Bayesian optimization
- When GA wins, when it loses

Module 7: Production Pipelines
- Integrating GA selection with forecasting models
- Feature stability across time windows
- Ensemble feature selection
- MLOps for feature selection
- Monitoring feature drift

Capstone: GA-based feature selection for commodity forecasting
```

### Key Technologies
- DEAP (Distributed Evolutionary Algorithms in Python)
- scikit-learn for baseline comparisons
- Optuna/TPOT for AutoML comparison
- Time series libraries (statsmodels, sktime)

### Practical Focus
- Real-world time series datasets
- Computational efficiency at scale
- Reproducibility and stability
- Integration with existing ML pipelines

---

## Course 7: Hidden Markov Models

### Course Vision
Master Hidden Markov Models from mathematical foundations to modern applications. Build intuition for when HMMs are appropriate and implement them for regime detection, sequence labeling, and time series analysis.

### Target Audience
- Data scientists working with sequential data
- Quantitative researchers in finance
- Engineers building speech/NLP systems
- Bioinformatics researchers

### Module Structure (8 modules)

```
Module 0: Foundations
- Markov chains review
- Stationarity and ergodicity
- Transition matrices and limiting distributions
- From visible to hidden states

Module 1: HMM Formulation
- The three problems (evaluation, decoding, learning)
- Generative model perspective
- Graphical model representation
- Emission distributions (discrete, Gaussian, mixture)

Module 2: The Forward-Backward Algorithm
- Forward algorithm derivation
- Backward algorithm derivation
- Computing state probabilities
- Numerical stability (log-space computation)
- Scaling factors

Module 3: The Viterbi Algorithm
- Most likely state sequence
- Dynamic programming solution
- Backtracking
- Comparison with posterior decoding
- Applications in sequence labeling

Module 4: Parameter Estimation
- Baum-Welch (EM) algorithm
- E-step: expected sufficient statistics
- M-step: parameter updates
- Convergence properties
- Initialization strategies

Module 5: Bayesian HMMs
- Prior distributions on parameters
- Gibbs sampling for HMMs
- Variational inference for HMMs
- PyMC and NumPyro implementations
- Advantages over MLE

Module 6: Extensions and Variants
- Input-output HMMs
- Hierarchical HMMs
- Infinite HMMs (HDP-HMM)
- Switching state space models
- Continuous-time HMMs

Module 7: Applications
- Financial regime detection
- Speech recognition fundamentals
- Biological sequence analysis
- Anomaly detection
- Predictive maintenance

Capstone: HMM-based regime detection system
```

### Key Technologies
- hmmlearn (Python)
- pomegranate
- PyMC for Bayesian HMMs
- Custom implementations for learning

### Mathematical Rigor
- Full derivations of all algorithms
- Proofs of convergence properties
- Computational complexity analysis
- Connection to broader probabilistic graphical models

---

## Implementation Instructions

### For Each Course:

1. **Create Directory Structure**
```bash
courses/[course-name]/
├── README.md
├── syllabus/
│   ├── course_syllabus.md
│   └── learning_objectives.md
├── modules/
│   ├── module_00_foundations/
│   │   ├── README.md
│   │   ├── guides/
│   │   ├── notebooks/
│   │   └── assessments/
│   └── module_XX_.../
├── capstone/
│   └── project_specification.md
└── resources/
    ├── glossary.md
    └── environment_setup.md
```

2. **Follow Guide Template**
```markdown
# [Concept Name]

## In Brief
[1-2 sentence summary]

## Key Insight
[Core idea in plain language]

## Formal Definition
[Mathematical formulation]

## Intuitive Explanation
[Analogy or visual]

## Code Implementation
[Working example]

## Common Pitfalls
[What to avoid]

## Practice Problems
[Exercises]

## Further Reading
[Resources]
```

3. **Follow Notebook Template**
- Learning objectives at start
- Markdown before every code cell
- Auto-graded exercises with tests
- Summary with key takeaways

4. **Quality Standards**
- No mocks or placeholders
- Real data and APIs
- Complete working implementations
- Multiple explanation approaches

---

## Execution Order

Recommended order based on dependencies and reusability:

1. **Panel Regression** (foundational econometrics)
2. **Hidden Markov Models** (foundational for regime models)
3. **Genetic Algorithms for Feature Selection** (self-contained)
4. **Bayesian Time Series** (extend existing course)
5. **Agentic AI & LLMs** (high demand, evolving field)
6. **Gen AI for Commodities** (builds on #4 and #5)
7. **Dataiku Gen AI** (platform-specific, can be parallel)

---

## Cross-Course Connections

Create explicit links between courses:

- **HMMs → Bayesian Time Series:** Regime-switching modules
- **Panel Regression → Commodities:** Cross-sectional commodity analysis
- **Agentic AI → Commodities GenAI:** Agent architectures for trading
- **Feature Selection → Bayesian TS:** Feature selection for forecasting
- **HMMs → Commodities GenAI:** Regime-aware AI systems

---

## Success Metrics

For each course, target:
- **Completion rate:** >50% (vs 13% industry average)
- **Assessment scores:** Normal distribution centered at 75-80%
- **Student satisfaction:** >4.2/5.0
- **Time accuracy:** Actual time within 20% of estimates

---

*Use `/create-course [topic]` command after this prompt is processed to begin building individual courses.*
