# Slide Deck Quality Review: Dataiku GenAI Foundations

**Reviewer:** Automated Quality Audit
**Date:** 2026-02-20
**Scope:** All 17 `_slides.md` files across 5 modules
**Methodology:** Each deck evaluated against its companion source guide and scored across 6 weighted dimensions (1-5 scale)

---

## Executive Summary

The Dataiku GenAI Foundations slide collection contains **17 Marp-based decks** totaling approximately **6,500 lines** of content. The decks demonstrate strong technical consistency and effective use of Mermaid diagrams, but suffer from three systemic problems that drag overall quality below what a polished course demands:

1. **Content duplication across paired decks.** Module 00 has two pairs of near-duplicate decks (architecture/setup, model_connections/provider_setup) and Module 01 has one pair (prompt_studio_basics/prompt_studios). This represents ~30% redundant slide content in the first two modules -- the ones learners encounter first.

2. **Uniform structure breeds monotony.** Nearly every deck follows the identical skeleton: Key Insight blockquote, component diagrams, code implementation, pitfalls table, takeaways list. By the third module, the rhythm becomes predictable and disengaging.

3. **Questionable SDK accuracy.** Multiple decks use Dataiku SDK patterns (`dataiku.llm.LLM`, `KnowledgeBank.search()`, `ChatSession`, `PromptStudio`) that appear to be illustrative rather than verified against actual API documentation. This is a serious credibility risk for a platform-specific course.

**Overall Weighted Score: 3.4 / 5.0**

The decks are functional and internally consistent, but they are not yet at the standard of a polished professional course. The duplication problem alone would confuse learners navigating the module structure.

---

## Dimension Scores (Aggregate Across All 17 Decks)

| Dimension | Weight | Avg Score | Weighted |
|-----------|--------|-----------|----------|
| Design & Visual Quality | 20% | 3.6 | 0.72 |
| Narrative & Story Flow | 20% | 3.2 | 0.64 |
| Comprehensiveness | 20% | 3.5 | 0.70 |
| Technical Accuracy | 20% | 3.1 | 0.62 |
| Added Visual Value | 10% | 3.8 | 0.38 |
| Production Readiness | 10% | 3.3 | 0.33 |
| **TOTAL** | **100%** | | **3.39** |

### Dimension Commentary

**Design & Visual Quality (3.6/5):** Consistent Marp frontmatter, clean CSS grid columns, well-sized fonts. Mermaid diagrams render correctly. However, every deck uses the same `default` theme with identical styling -- no visual differentiation between modules. No speaker notes. No imagery beyond diagrams.

**Narrative & Story Flow (3.2/5):** Each deck has a clear opening analogy and consistent structure. The problem is that this structure is too rigid -- every deck feels interchangeable. Transitions between sections use bare `<!-- _class: lead -->` headers without bridging text. The "Five Common Pitfalls" tables, while useful, appear mechanically in every deck regardless of whether five pitfalls genuinely exist.

**Comprehensiveness (3.5/5):** Decks generally cover the key topics from their companion guides. Source guide content like practice problems, further reading, and formal definitions are appropriately excluded from slides. However, some decks omit important nuances present in guides (e.g., retrieval strategies deck skips evaluation metrics detail, knowledge banks deck glosses over embedding model selection).

**Technical Accuracy (3.1/5):** This is the most concerning dimension. The Dataiku SDK usage throughout (`from dataiku.llm import LLM`, `LLM().complete()`, `KnowledgeBank()`, `ChatSession`, `PromptStudio`) appears to be based on assumed/illustrative API patterns rather than verified documentation. If these APIs do not exist exactly as shown, every code slide in the course is misleading. The general Python patterns (ThreadPoolExecutor, Flask, SSE) are accurate. Mermaid syntax is correct throughout.

**Added Visual Value (3.8/5):** This is the strongest dimension. The Mermaid diagrams genuinely add explanatory value -- architecture diagrams, sequence flows, decision trees, and comparison charts make abstract concepts concrete. The retrieval strategies deck's use of LaTeX for distance formulas is appropriate. The Gantt chart in python_recipes for parallel processing comparison is a creative touch.

**Production Readiness (3.3/5):** Decks render correctly in Marp. Pagination is enabled. Code blocks have syntax highlighting. However: no speaker notes exist, no handout variants, no print-friendly layouts, slide counts vary wildly (15-25 per deck with no apparent targeting), and the duplicate decks would need to be resolved before any delivery.

---

## Per-Deck Reviews

### Module 00: LLM Mesh Foundations

#### 01_llm_mesh_architecture_slides.md
**Score: 3.6** | Lines: ~504 | Slides: ~25

| Dimension | Score | Notes |
|-----------|-------|-------|
| Design & Visual Quality | 4 | Clean architecture diagrams. Good use of subgraph groupings in Mermaid. Well-structured code blocks. |
| Narrative & Story Flow | 3 | "Air traffic control" analogy is strong. But 25 slides is long for one topic -- could be split or tightened. Middle section drags with sequential component explanations. |
| Comprehensiveness | 4 | Covers architecture, components (router, governance, telemetry), request flow, usage patterns, and connections to other modules. Good breadth. |
| Technical Accuracy | 3 | SDK patterns (`LLM("anthropic-claude").complete()`) are plausible but unverified. The `response.usage.total_tokens` pattern is standard but may not match Dataiku's actual response object. |
| Added Visual Value | 4 | Architecture diagram with color-coded subgraphs is the standout. Request flow sequence diagram adds genuine clarity. |
| Production Readiness | 3 | No speaker notes. 25 slides is too many for a single presentation section without clear break points. |

**Key Issues:**
- Significant overlap with `01_llm_mesh_setup_slides.md` -- both cover LLM Mesh concept, basic usage, and connection management
- Some code examples are too long for slides (8-15 lines that would be hard to read projected)

---

#### 01_llm_mesh_setup_slides.md
**Score: 3.3** | Lines: ~338 | Slides: ~17

| Dimension | Score | Notes |
|-----------|-------|-------|
| Design & Visual Quality | 4 | Consistent styling. Good Mermaid usage. |
| Narrative & Story Flow | 3 | Opens with same LLM Mesh concept as architecture deck -- immediately feels redundant. The "dial tones" analogy is weaker than "air traffic control." |
| Comprehensiveness | 3 | Covers basic completion, chat, structured output, failover. But this is a subset of what the architecture deck already covers. |
| Technical Accuracy | 3 | Same SDK concerns. The `llm.complete()` with `temperature` and `max_tokens` kwargs is standard but unverified for Dataiku. |
| Added Visual Value | 4 | Connection diagram and cost routing flowchart are useful. |
| Production Readiness | 3 | Would need to be merged with or replaced by architecture deck. |

**Key Issues:**
- **DUPLICATE CANDIDATE.** This deck covers a strict subset of the architecture deck. Recommend merging or eliminating.
- The structured output section (JSON extraction) reappears in Module 03's custom models deck.

---

#### 02_model_connections_slides.md
**Score: 3.5** | Lines: ~314 | Slides: ~16

| Dimension | Score | Notes |
|-----------|-------|-------|
| Design & Visual Quality | 4 | Clean provider config code blocks. Good comparison tables. |
| Narrative & Story Flow | 3 | Logical progression from config to testing to monitoring. But opens cold without a motivating analogy. |
| Comprehensiveness | 4 | Covers OpenAI, Azure, Bedrock configs. Health checks, benchmarking, load balancing, failover, cost tracking, security -- good breadth for 16 slides. |
| Technical Accuracy | 3 | Provider configuration patterns are reasonable for an abstraction layer but unverified. The `dataiku.api_client()` call pattern needs verification. |
| Added Visual Value | 4 | Load balancing flowchart and failover sequence diagram are strong. |
| Production Readiness | 3 | Overlaps with provider_setup deck. |

**Key Issues:**
- **DUPLICATE CANDIDATE** with `02_provider_setup_slides.md`. Both cover provider configuration, multi-connection management, and failover.
- Benchmarking code block is too dense for a slide.

---

#### 02_provider_setup_slides.md
**Score: 3.5** | Lines: ~402 | Slides: ~20

| Dimension | Score | Notes |
|-----------|-------|-------|
| Design & Visual Quality | 4 | Good use of columns layout for side-by-side configs. Table comparisons are clear. |
| Narrative & Story Flow | 4 | "Phone system" analogy (old world: direct dial each carrier; new world: dial one number) is effective and memorable. Best analogy in Module 00. |
| Comprehensiveness | 4 | Anthropic, OpenAI, Azure configurations. Multi-connection fallback. Pitfalls specific to each provider. |
| Technical Accuracy | 3 | Azure deployment name gotcha is a real-world concern -- good inclusion. But SDK patterns still unverified. |
| Added Visual Value | 3 | Diagrams are adequate but less creative than other Module 00 decks. The "old vs new" comparison is text-heavy where a diagram would help. |
| Production Readiness | 3 | Overlaps with model_connections. Pitfalls table is the strongest differentiator. |

**Key Issues:**
- **DUPLICATE CANDIDATE** with `02_model_connections_slides.md`. The phone analogy and pitfalls table are the unique value -- could be folded into the other deck.

---

#### 03_governance_slides.md (Module 00)
**Score: 3.7** | Lines: ~328 | Slides: ~17

| Dimension | Score | Notes |
|-----------|-------|-------|
| Design & Visual Quality | 4 | Clean governance hierarchy diagram. Good use of color coding (green for allowed, red for denied). |
| Narrative & Story Flow | 4 | "Expense policy" analogy grounds an abstract topic. Logical flow from access control to quotas to auditing. |
| Comprehensiveness | 4 | Access control, project quotas, cost tracking, audit logging, rate limiting. Covers the full governance surface. |
| Technical Accuracy | 3 | `GovernanceConfig` dataclass is illustrative. Rate limiting implementation uses a `deque` pattern that is correct algorithmically but the integration with Dataiku is speculative. |
| Added Visual Value | 4 | Governance hierarchy and audit flow diagrams are strong additions beyond what the source guide text describes. |
| Production Readiness | 3 | No speaker notes. Some code blocks need context about where they run (admin setup vs project code). |

**Key Issues:**
- Best standalone deck in Module 00 -- no duplication problems
- Could benefit from a real-world scenario walkthrough (e.g., "What happens when a team exceeds their quota?")

---

### Module 01: Prompt Engineering

#### 01_prompt_studio_basics_slides.md
**Score: 3.4** | Lines: ~333 | Slides: ~17

| Dimension | Score | Notes |
|-----------|-------|-------|
| Design & Visual Quality | 4 | Clean interface layout diagram. Good numbered step sequences. |
| Narrative & Story Flow | 3 | "IDE for prompts" analogy is appropriate but not vivid. Progression from creation to testing to deployment is logical. |
| Comprehensiveness | 3 | Covers core Prompt Studio features but is shallow on each. Test cases and version control get only 1-2 slides each. |
| Technical Accuracy | 3 | `dataiku.PromptStudio()` API patterns are plausible for the platform but need verification. |
| Added Visual Value | 4 | Interface layout diagram helps learners orient before opening the actual tool. |
| Production Readiness | 3 | Overlaps with prompt_studios deck. |

**Key Issues:**
- **DUPLICATE CANDIDATE** with `01_prompt_studios_slides.md`. Both cover what Prompt Studios are, how to create prompts, test them, and deploy them. The "basics" version is slightly more focused on UI walkthrough; the "studios" version adds template patterns.

---

#### 01_prompt_studios_slides.md
**Score: 3.5** | Lines: ~378 | Slides: ~18

| Dimension | Score | Notes |
|-----------|-------|-------|
| Design & Visual Quality | 4 | Good columns layout for prompt pattern comparisons. |
| Narrative & Story Flow | 3 | Covers similar ground to basics deck with more emphasis on prompt patterns (extraction, analysis, comparison). Better technical depth but weaker opening. |
| Comprehensiveness | 4 | Adds prompt patterns, test-driven development approach, and deployment options beyond what basics covers. |
| Technical Accuracy | 3 | Same SDK concerns. The prompt patterns themselves (extraction, analysis, comparison) are sound prompt engineering. |
| Added Visual Value | 4 | Workflow diagram for test-driven prompt development is a strong addition. |
| Production Readiness | 3 | Needs to be merged with or replace the basics deck. |

**Key Issues:**
- **DUPLICATE CANDIDATE** with basics deck. Recommend keeping this one (more complete) and folding in the UI walkthrough content from basics.

---

#### 02_template_variables_slides.md
**Score: 3.8** | Lines: ~413 | Slides: ~20

| Dimension | Score | Notes |
|-----------|-------|-------|
| Design & Visual Quality | 4 | Strong use of columns for variable definition syntax examples. Clean execution flow diagram. |
| Narrative & Story Flow | 4 | "Mail merge" analogy is immediately intuitive. Progression from simple variables to conditional logic to looping to batch processing is well-paced. |
| Comprehensiveness | 4 | Covers variable types, definition syntax, execution flow, validation, conditionals (`{{#if}}`), loops (`{{#each}}`), batch processing. Complete treatment. |
| Technical Accuracy | 3 | Handlebars-like template syntax (`{{variable}}`, `{{#if}}`, `{{#each}}`) is plausible for Dataiku but needs verification. The batch processing with ThreadPoolExecutor is solid Python. |
| Added Visual Value | 4 | Execution flow diagram and validation flowchart add clarity beyond the source guide's text explanations. |
| Production Readiness | 4 | Standalone topic, no duplication. Good slide count for the content. |

**Key Issues:**
- Best deck in Module 01
- Batch processing section feels like it belongs in Module 03 (pipeline integration) rather than template variables

---

#### 03_testing_iteration_slides.md
**Score: 3.7** | Lines: ~429 | Slides: ~22

| Dimension | Score | Notes |
|-----------|-------|-------|
| Design & Visual Quality | 4 | Good code structure for PromptEvaluator and PromptTestSuite classes. Clean comparison tables. |
| Narrative & Story Flow | 3 | "Test harness for prompts" is accurate but not a vivid analogy. 22 slides is on the long side. The jump from evaluation to iteration to regression testing is logical but could use more bridging. |
| Comprehensiveness | 4 | Evaluation criteria, test suites, version comparison, automated iteration, regression testing. Thorough coverage. |
| Technical Accuracy | 3 | The PromptEvaluator pattern with required vs weighted criteria is well-designed as a concept. The `PromptStudio.load_version()` API is unverified. |
| Added Visual Value | 4 | Version comparison bar chart (Mermaid) and iteration improvement flowchart are effective. |
| Production Readiness | 3 | Could trim 3-4 slides without losing substance. |

**Key Issues:**
- Solid standalone deck
- The PromptIterator class with automated improvement feels aspirational -- learners might expect this to work out of the box

---

### Module 02: RAG (Retrieval Augmented Generation)

#### 01_knowledge_banks_slides.md
**Score: 3.7** | Lines: ~431 | Slides: ~20

| Dimension | Score | Notes |
|-----------|-------|-------|
| Design & Visual Quality | 4 | RAG pipeline diagram is clean and well-colored. CommodityRAG class code is well-structured. |
| Narrative & Story Flow | 4 | "Library with a research assistant" analogy is accessible. Flow from creation to querying to evaluation to maintenance is natural. |
| Comprehensiveness | 4 | Creating KBs (UI + code), chunking strategies (3 types), querying (basic/filtered/hybrid), CommodityRAG class, evaluation, maintenance. Good breadth. |
| Technical Accuracy | 3 | `KnowledgeBank()` API with `.search()`, `.add_documents()` is plausible but unverified. Chunking strategy parameters need platform verification. |
| Added Visual Value | 4 | RAG pipeline sequence diagram and chunking comparison diagrams add genuine value. The evaluation metrics visualization is a nice touch. |
| Production Readiness | 4 | No duplication. Good standalone deck. Appropriate length. |

**Key Issues:**
- Embedding model selection is glossed over -- the source guide has more detail that could be included
- The CommodityRAG class is a good teaching vehicle but combines too many concepts in one code block

---

#### 02_retrieval_strategies_slides.md
**Score: 3.9** | Lines: ~495 | Slides: ~24

| Dimension | Score | Notes |
|-----------|-------|-------|
| Design & Visual Quality | 4 | LaTeX formulas for distance metrics are well-formatted. Decision tree diagram at the end is excellent. |
| Narrative & Story Flow | 4 | Logical progression from basic similarity to hybrid search to advanced techniques (reranking, caching). Good pacing despite 24 slides. |
| Comprehensiveness | 5 | Vector search, distance metrics (3 types with formulas), index types (Flat/IVF/HNSW with trade-offs), hybrid retrieval, query expansion, contextual compression, multi-index, reranking (cross-encoder + LLM-based), caching, decision tree. Most thorough deck in the course. |
| Technical Accuracy | 3 | Distance metric formulas are correct. Index type descriptions are accurate for vector databases generally. The `HybridRetriever` class pattern is sound but Dataiku-specific integration is unverified. |
| Added Visual Value | 4 | Decision tree for strategy selection is the single best diagram in the entire course. LaTeX formulas are appropriate here. |
| Production Readiness | 4 | 24 slides is long but justified by the topic's complexity. Could be split into "Fundamentals" and "Advanced" if needed. |

**Key Issues:**
- **Highest-scoring deck in the course**
- The only deck that uses LaTeX math, which is appropriate
- Could benefit from a "When to Use What" summary slide earlier (currently only at the end as a decision tree)

---

### Module 03: Custom Code Integration

#### 01_python_recipes_slides.md
**Score: 3.5** | Lines: ~361 | Slides: ~18

| Dimension | Score | Notes |
|-----------|-------|-------|
| Design & Visual Quality | 4 | Gantt chart for parallel processing comparison is creative. Clean code blocks. |
| Narrative & Story Flow | 3 | Jumps between several patterns (recipe, parallel, chunked, API endpoint, webapp, pipeline class) without a clear through-line. Feels like a grab-bag. |
| Comprehensiveness | 3 | Covers many patterns but each thinly. The multi-stage pipeline class gets only one slide. |
| Technical Accuracy | 3 | ThreadPoolExecutor usage is correct. Flask patterns are standard. Dataiku recipe pattern (`dataiku.Dataset().get_dataframe()`) is plausible. |
| Added Visual Value | 4 | Gantt chart showing sequential vs parallel processing is effective. Pipeline flow diagram is good. |
| Production Readiness | 3 | Topic is too broad for one deck. Should be split or focused. |

**Key Issues:**
- Tries to cover too many patterns in one deck -- recipes, parallel processing, chunking, API endpoints, webapps, and pipeline classes
- Overlap with Module 04 decks (API endpoints, webapps)
- The Gantt chart is a standout visual that other decks could learn from

---

#### 02_custom_models_slides.md
**Score: 3.7** | Lines: ~415 | Slides: ~20

| Dimension | Score | Notes |
|-----------|-------|-------|
| Design & Visual Quality | 4 | Middleware diagram is clean. Wrapper extension points diagram clearly shows the pre/post pattern. Good columns layout for side-by-side code. |
| Narrative & Story Flow | 4 | "Middleware" analogy is immediately intuitive for developers. Logical progression from base pattern to specialized wrappers. The composition note at the end ties everything together. |
| Comprehensiveness | 4 | BaseLLMWrapper, JSONExtractorLLM, RetryFallbackLLM, CostOptimizedLLM, CommodityAnalysisLLM, CachedLLM, wrapper selection guide. Good coverage. |
| Technical Accuracy | 3 | The wrapper patterns themselves are sound software engineering. The complexity estimation in CostOptimizedLLM is simplistic (keyword matching) but acknowledged as such. The `hashlib.sha256` cache key pattern is correct. |
| Added Visual Value | 4 | Retry/fallback flowchart makes the strategy immediately clear. Cost routing decision tree is useful. |
| Production Readiness | 4 | Good standalone deck. Each wrapper is self-contained and well-explained. |

**Key Issues:**
- Second-best deck in Module 03
- The CostOptimizedLLM complexity estimation deserves a caveat slide about its limitations
- Wrapper composition (`CachedLLM(RetryFallbackLLM(JSONExtractorLLM(...)))`) is mentioned but not diagrammed -- missed opportunity

---

#### 03_pipeline_integration_slides.md
**Score: 3.6** | Lines: ~411 | Slides: ~20

| Dimension | Score | Notes |
|-----------|-------|-------|
| Design & Visual Quality | 4 | Assembly line analogy table is well-formatted. Pipeline flow diagram clearly shows data movement. |
| Narrative & Story Flow | 4 | "Assembly line" analogy grounds the concept well. Progression from basic recipe to batch processing to chunking to monitoring to incremental processing is logical and well-paced. |
| Comprehensiveness | 4 | Basic integration, parallel batch processing (BatchLLMProcessor), scaling strategy, chunked processing, robust pipeline with monitoring, incremental processing with cost comparison. Solid coverage. |
| Technical Accuracy | 3 | ThreadPoolExecutor pattern is correct. Chunking implementation with sentence boundary detection is reasonable. Incremental processing pattern is sound. The `response.cost` attribute is unverified. |
| Added Visual Value | 4 | Scaling strategy decision tree (< 100 rows / 100-10K / > 10K) is practical and clear. Incremental vs full processing comparison diagram is effective. |
| Production Readiness | 4 | Good standalone deck. No duplication with other Module 03 decks. |

**Key Issues:**
- Overlap with `01_python_recipes_slides.md` in the parallel processing and chunking sections
- The cost comparison table (Full: $50/day vs Incremental: $0.25/day) is compelling and should appear earlier
- Scaling strategy table has a minor inconsistency: "1s/row" assumption may not hold for different model tiers

---

### Module 04: Deployment & Production

#### 01_deployment_monitoring_slides.md
**Score: 3.6** | Lines: ~352 | Slides: ~17

| Dimension | Score | Notes |
|-----------|-------|-------|
| Design & Visual Quality | 4 | API Node architecture diagram with load balancer is clean. Monitoring dashboard layout is clear. |
| Narrative & Story Flow | 3 | Goes from deployment architecture to monitoring to alerting to cost management. Logical but feels like four mini-topics rather than one narrative. |
| Comprehensiveness | 4 | API Node deployment, built-in metrics, MonitoredLLM wrapper, dashboard, alert configuration (YAML + programmatic), CostTracker, BudgetEnforcedLLM, production checklist. Thorough. |
| Technical Accuracy | 3 | YAML config for API service is plausible. The MonitoredLLM wrapper using `MetricsClient()` is illustrative. CostTracker pricing for `claude-sonnet-4` and `gpt-4o` will go stale quickly. |
| Added Visual Value | 4 | Budget enforcement flowchart is simple and effective. Alert flow diagram clearly shows severity routing. |
| Production Readiness | 3 | Model pricing in CostTracker will be outdated rapidly -- needs a "check current pricing" note. |

**Key Issues:**
- Hardcoded model pricing is a maintenance liability
- Overlap with Module 04's governance deck in monitoring and alerting sections
- Production checklist is a strong practical addition but could be its own handout

---

#### 02_webapp_integration_slides.md
**Score: 3.5** | Lines: ~433 | Slides: ~20

| Dimension | Score | Notes |
|-----------|-------|-------|
| Design & Visual Quality | 4 | Full stack architecture diagram is well-structured. Good use of columns for backend/frontend side-by-side code. |
| Narrative & Story Flow | 3 | Progression from basic webapp to chatbot to streaming is logical. But the "Key Insight" about hiding complexity is generic -- doesn't tell learners what specifically they will build. |
| Comprehensiveness | 4 | Basic Flask endpoint, frontend request pattern, chatbot with session management, SSE streaming (backend + frontend). Covers the core webapp patterns. |
| Technical Accuracy | 3 | Flask patterns are correct. The `chat_sessions = {}` in-memory dict is acknowledged as a pitfall (data lost on restart). SSE implementation with `stream_with_context(generate())` is correct Flask. The frontend ReadableStream reader pattern is standard. |
| Added Visual Value | 3 | Architecture diagram is good but the chatbot sequence diagram adds limited value beyond what the code already shows. The SSE flow diagram is more useful. |
| Production Readiness | 3 | In-memory session storage is a known issue that should have a concrete alternative shown, not just mentioned in pitfalls. |

**Key Issues:**
- Overlap with Module 03's `01_python_recipes_slides.md` which also covers Flask webapp backends
- The frontend JavaScript could benefit from a framework mention (React, Vue) even if not used
- The streaming frontend code has a bug: `data.content` is used but `data` is never parsed from the SSE chunk text

---

#### 03_governance_slides.md (Module 04)
**Score: 3.6** | Lines: ~362 | Slides: ~18

| Dimension | Score | Notes |
|-----------|-------|-------|
| Design & Visual Quality | 4 | DeploymentConfig dataclass is clean. Environment comparison diagram with color-coded subgraphs is effective. |
| Narrative & Story Flow | 3 | "Air traffic control" tagline reuses the same metaphor from Module 00's architecture deck -- creates a confusing echo. Progression from config to pipeline to monitoring is logical. |
| Comprehensiveness | 4 | DeploymentConfig, environment comparison table, pre-deployment checks flowchart, DeploymentPipeline class, rollback procedure, ProductionMonitor, alert configuration, monitoring cycle. Complete treatment. |
| Technical Accuracy | 3 | DeploymentConfig dataclass is well-designed. Pre-deployment check flow is sound. Rollback procedure using deployment log is reasonable. The `dataiku.PromptStudio("market-analyzer").load_version()` is unverified. |
| Added Visual Value | 4 | Pre-deployment checks flowchart is one of the best decision-flow diagrams in the course. Rollback sequence diagram clearly shows the multi-step process. Environment comparison diagram with budget numbers is immediately useful. |
| Production Readiness | 3 | Overlap with `01_deployment_monitoring_slides.md` in the monitoring and alerting sections. |

**Key Issues:**
- Monitoring overlap with deployment_monitoring deck -- the ProductionMonitor class here and the MonitoredLLM there cover similar ground from different angles
- The "air traffic control" tagline collision with Module 00 is a minor but real brand confusion issue
- Rollback to previous version using `iloc[1]` is fragile -- should explain the assumption

---

## Dimension Score Matrix (All 17 Decks)

| Deck | Design | Narrative | Comprehensive | Accuracy | Visual | Production | **Weighted** |
|------|--------|-----------|---------------|----------|--------|------------|-------------|
| M00: architecture | 4 | 3 | 4 | 3 | 4 | 3 | **3.5** |
| M00: setup | 4 | 3 | 3 | 3 | 4 | 3 | **3.3** |
| M00: model_connections | 4 | 3 | 4 | 3 | 4 | 3 | **3.5** |
| M00: provider_setup | 4 | 4 | 4 | 3 | 3 | 3 | **3.5** |
| M00: governance | 4 | 4 | 4 | 3 | 4 | 3 | **3.7** |
| M01: basics | 4 | 3 | 3 | 3 | 4 | 3 | **3.3** |
| M01: studios | 4 | 3 | 4 | 3 | 4 | 3 | **3.5** |
| M01: template_vars | 4 | 4 | 4 | 3 | 4 | 4 | **3.8** |
| M01: testing | 4 | 3 | 4 | 3 | 4 | 3 | **3.5** |
| M02: knowledge_banks | 4 | 4 | 4 | 3 | 4 | 4 | **3.8** |
| M02: retrieval | 4 | 4 | 5 | 3 | 4 | 4 | **3.9** |
| M03: python_recipes | 4 | 3 | 3 | 3 | 4 | 3 | **3.3** |
| M03: custom_models | 4 | 4 | 4 | 3 | 4 | 4 | **3.8** |
| M03: pipeline | 4 | 4 | 4 | 3 | 4 | 4 | **3.8** |
| M04: deployment | 4 | 3 | 4 | 3 | 4 | 3 | **3.5** |
| M04: webapp | 4 | 3 | 4 | 3 | 3 | 3 | **3.4** |
| M04: governance | 4 | 3 | 4 | 3 | 4 | 3 | **3.5** |

---

## Comparison to Best-in-Class

### What the Best Decks Do Right

**02_retrieval_strategies_slides.md (3.9)** -- Top scorer:
- Comprehensive coverage without feeling rushed (24 slides justified by topic complexity)
- LaTeX formulas used appropriately for distance metrics
- Decision tree at the end synthesizes all strategies into actionable guidance
- Trade-off tables for index types give learners real decision-making tools

**02_template_variables_slides.md (3.8):**
- "Mail merge" analogy is instantly understood by any audience
- Clean progression from simple to complex (variables to conditionals to loops to batch)
- Standalone topic with no duplication issues

**02_custom_models_slides.md (3.8):**
- "Middleware" analogy connects to developers' existing mental models
- Each wrapper pattern is self-contained with clear use case
- Wrapper selection guide and composition hint tie everything together

### What Best-in-Class Slide Decks (External) Do That These Don't

1. **Speaker notes.** Zero decks include Marp speaker notes (`<!-- presenter notes -->`). A presenter delivering these decks has no guidance on timing, emphasis, or talking points.

2. **Progressive disclosure.** External best-in-class decks build diagrams incrementally across slides. These decks show complete diagrams on single slides, losing the pedagogical benefit of step-by-step construction.

3. **Real screenshots.** Platform-specific courses (Databricks, Snowflake, Google Cloud) include actual UI screenshots. These decks rely entirely on Mermaid diagrams to represent UI layouts, which is less convincing for a platform course.

4. **Recap/bridge slides.** Between major sections, best-in-class decks include a "What we covered / What's next" bridge. These decks use bare `<!-- _class: lead -->` section headers without context.

5. **Consistent slide counts.** Best-in-class courses target a consistent slide count per topic (e.g., 12-15). These range from 15-25 with no apparent target.

6. **Interactivity prompts.** Best-in-class decks include "Think about..." or "Discussion:" slides. These are purely lecture-format.

---

## Priority Fix List

### P0: Must Fix Before Delivery

1. **Eliminate duplicate decks.** Merge or remove:
   - `01_llm_mesh_architecture_slides.md` + `01_llm_mesh_setup_slides.md` -> single deck
   - `02_model_connections_slides.md` + `02_provider_setup_slides.md` -> single deck
   - `01_prompt_studio_basics_slides.md` + `01_prompt_studios_slides.md` -> single deck

   **Impact:** Removes ~5 redundant decks, reducing learner confusion and maintenance burden.

2. **Verify Dataiku SDK patterns.** Every code example using `dataiku.llm.LLM`, `KnowledgeBank`, `ChatSession`, `PromptStudio` must be verified against actual Dataiku documentation or a working instance. If the APIs differ, every code slide needs correction.

   **Impact:** Addresses the single largest credibility risk in the entire course.

3. **Fix the streaming frontend bug in `02_webapp_integration_slides.md`.** The frontend code references `data.content` but never parses the SSE `chunk` text into a `data` object. This is a broken code example in a production-readiness-focused module.

### P1: Should Fix Before Delivery

4. **Add speaker notes to all decks.** At minimum: timing guidance, key talking points, and transition cues. Marp supports `<!-- notes: ... -->` syntax.

5. **Resolve cross-module overlap.** Module 03's `01_python_recipes_slides.md` covers API endpoints and webapps that Module 04 covers in more depth. Either remove from Module 03 or explicitly frame as "preview."

6. **Update hardcoded model pricing in `01_deployment_monitoring_slides.md`.** CostTracker class has prices for `claude-sonnet-4`, `gpt-4o`, `claude-3-haiku` that will be outdated. Add a "verify current pricing" note or make it configurable.

7. **Differentiate the "air traffic control" metaphor.** Module 00 architecture deck and Module 04 governance deck both use "air traffic control." Either change one or make the connection explicit ("Remember the air traffic control analogy from Module 0? Now we are adding...").

### P2: Nice to Have

8. **Add progressive diagram builds.** For complex diagrams (RAG pipeline, deployment architecture, governance framework), split into 2-3 slides that build incrementally.

9. **Add recap/bridge slides between sections.** A one-line "What we just covered | What's next" slide between each `<!-- _class: lead -->` section.

10. **Target consistent slide counts.** Aim for 15-18 slides per deck. Trim the 22-25 slide decks and expand thin coverage in shorter decks.

11. **Add at least one real Dataiku UI screenshot per module** to ground the platform-specific content in reality rather than abstract diagrams.

12. **Add "Discussion" or "Think About" slides** to break lecture monotony -- at least one per deck.

13. **Vary the deck structure.** Not every deck needs to end with "Five Common Pitfalls" followed by "Key Takeaways." Some decks could end with a challenge question, a comparison table, or a decision framework.

14. **Add a visual theme variation per module.** Even small changes (accent color, header style) would help learners orient which module they are in.

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total decks reviewed | 17 |
| Total lines of slide content | ~6,500 |
| Duplicate/overlapping decks | 5 (3 pairs) |
| Unique standalone decks | 12 |
| Decks scoring >= 3.8 | 4 |
| Decks scoring < 3.5 | 4 |
| Decks with speaker notes | 0 |
| Decks using LaTeX | 1 |
| Decks using Mermaid | 17 (100%) |
| Decks using column layouts | 10 |
| P0 issues | 3 |
| P1 issues | 4 |
| P2 issues | 7 |
