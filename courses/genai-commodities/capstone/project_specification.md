# Capstone Project: LLM-Powered Commodity Analysis System

## Overview

Build an end-to-end generative AI system that automates commodity fundamental analysis—from data collection and report processing to signal generation and backtesting. Apply LLMs to extract, synthesize, and reason over unstructured commodity market data.

**Weight:** 35% of final grade
**Duration:** Weeks 9-13
**Team Size:** Individual (groups of 2 optional with approval)

---

## Learning Objectives Demonstrated

By completing this project, you will demonstrate mastery of:

1. **Data Engineering:** Automating collection of commodity reports and market data
2. **Information Extraction:** Using LLMs to extract structured data from unstructured reports
3. **RAG Systems:** Building knowledge bases for fundamental research
4. **Signal Generation:** Converting fundamental analysis to actionable trading insights
5. **Validation:** Backtesting LLM-generated signals against historical data
6. **Production Systems:** Deploying automated analysis pipelines

---

## Project Requirements

Choose ONE commodity market to focus on:

### Energy Options
- **Crude Oil (WTI)** — EIA weekly petroleum, production, inventories
- **Natural Gas** — EIA storage, weather, LNG exports
- **Gasoline (RBOB)** — Refinery data, crack spreads, demand

### Agriculture Options
- **Corn** — USDA WASDE, planting/harvest, weather
- **Soybeans** — Crush spread, export data, South American weather
- **Wheat** — Multiple varieties, global production

### Metals Options
- **Copper** — Mine production, LME stocks, China demand
- **Gold** — Central bank buying, dollar correlation, safe haven flows

---

## Core Requirements (Must Complete All)

### 1. Data Pipeline (15 points)

Build automated system to collect:
- [ ] Price data (minimum 5 years daily)
- [ ] Fundamental reports (EIA, USDA, IEA, or relevant agency)
- [ ] News feeds or commentary (optional but recommended)
- [ ] Weather data (if applicable for your commodity)

**Requirements:**
- Automated data retrieval (APIs, web scraping)
- Handling missing values and data quality
- Organized storage (databases, files)
- Documentation of all sources

**Grading:**
- Automation completeness: 6 points
- Data quality: 5 points
- Documentation: 4 points

### 2. LLM-Based Report Processing (25 points)

Extract structured data from unstructured reports:
- [ ] Identify key entities (production, stocks, demand, etc.)
- [ ] Extract numerical data with units and timestamps
- [ ] Parse tables from PDFs or HTML
- [ ] Handle report format variations
- [ ] Validate extractions against ground truth sample

**Technical Requirements:**
- Use GPT-4, Claude, or equivalent
- Implement robust prompts with few-shot examples
- JSON output format for structured data
- Error handling for extraction failures
- Create test set with manually verified extractions

**Grading:**
- Extraction accuracy (vs. ground truth): 12 points
- Prompt quality and robustness: 8 points
- Error handling: 5 points

### 3. RAG for Fundamental Research (20 points)

Build knowledge base containing:
- [ ] Historical commodity reports
- [ ] Research notes or analyst commentary
- [ ] Market event summaries
- [ ] Seasonal pattern documentation

**Requirements:**
- Semantic search over historical context
- Query reformulation for better retrieval
- Source attribution in responses
- Metadata filtering (date, commodity, report type)

**Grading:**
- Retrieval quality (precision/recall): 10 points
- Knowledge base design: 6 points
- Query handling: 4 points

### 4. Fundamental Analysis & Signal Generation (20 points)

Use LLMs to:
- [ ] Analyze supply/demand balance
- [ ] Compare current data to historical norms
- [ ] Identify bullish/bearish factors
- [ ] Generate trading signals with rationale

**Requirements:**
- Chain-of-thought reasoning in prompts
- Structured signal output (direction, conviction, rationale)
- Incorporation of multiple data sources
- Economic interpretation of fundamentals

**Grading:**
- Analysis quality: 10 points
- Signal structure: 6 points
- Economic reasoning: 4 points

### 5. Backtesting & Validation (15 points)

Evaluate signal performance:
- [ ] Backtest signals on historical prices
- [ ] Compute performance metrics (returns, Sharpe, drawdown)
- [ ] Compare to benchmarks (buy-and-hold, moving averages)
- [ ] Analyze winning vs. losing trades
- [ ] Assess look-ahead bias and data snooping

**Grading:**
- Backtest rigor: 7 points
- Performance analysis: 5 points
- Benchmark comparison: 3 points

### 6. Production Pipeline (5 points)

- [ ] End-to-end automated workflow
- [ ] Logging and error monitoring
- [ ] Cost tracking (API usage)
- [ ] Schedule for report processing (if applicable)
- [ ] Deployment instructions

**Grading:**
- Automation quality: 2 points
- Monitoring: 2 points
- Documentation: 1 point

---

## Extension Options (Choose 1-2 for Bonus)

Each extension worth up to 5 bonus points:

1. **Multi-Commodity Analysis**
   - Analyze related commodities (e.g., crude + gasoline)
   - Identify spread opportunities
   - Cross-commodity dependencies

2. **Sentiment Integration**
   - Extract sentiment from news or Twitter
   - Combine with fundamental signals
   - Sentiment-adjusted positioning

3. **Ensemble Approach**
   - Multiple LLM providers
   - Voting or weighted consensus
   - Provider comparison and selection

4. **Real-Time Dashboard**
   - Streamlit/Gradio interface
   - Live data updates
   - Signal history and performance tracking

5. **Alternative Data**
   - Satellite imagery (crop monitoring)
   - Shipping data (imports/exports)
   - Social media (demand signals)

6. **Agent-Based System**
   - Autonomous agent that decides what data to fetch
   - Multi-step reasoning with tool use
   - Self-correction and refinement

---

## Milestones & Checkpoints

### Milestone 1: Proposal & Data Pipeline (Week 9) — 10%
**Deliverable:** Proposal + initial data pipeline

- Commodity selection and justification
- Data sources identified and tested
- Sample report extraction
- Initial prompt designs

**Grading:**
- Feasibility: 4 points
- Data pipeline: 4 points
- Prompt quality: 2 points

### Milestone 2: Extraction & RAG (Week 10-11) — 15%
**Deliverable:** Working extraction + knowledge base

- Extraction accuracy on test set
- RAG system functional
- Query examples
- Preliminary analysis outputs

**Grading:**
- Extraction performance: 8 points
- RAG quality: 5 points
- Documentation: 2 points

### Milestone 3: Signals & Backtesting (Week 12) — 10%
**Deliverable:** Signal generation + backtest results

- Signal generation working
- Backtest complete
- Performance metrics calculated

**Grading:**
- Signal quality: 5 points
- Backtest rigor: 4 points
- Analysis: 1 point

### Milestone 4: Final Submission (Week 13) — 65%
**Deliverables:** Complete system + report + presentation

See detailed rubric below.

---

## Technical Report Template

### Structure (5-7 pages, excluding appendices)

1. **Executive Summary** (0.5-1 page)
   - Commodity and approach
   - Key findings
   - Signal performance summary
   - Recommendations for deployment

2. **Data & Sources** (1-1.5 pages)
   - Data sources enumerated
   - Collection methodology
   - Data quality and completeness
   - Feature engineering

3. **LLM Extraction Pipeline** (1.5-2 pages)
   - Prompt engineering approach
   - Few-shot example selection
   - Extraction validation methodology
   - Accuracy results vs. ground truth

4. **RAG & Fundamental Analysis** (1-1.5 pages)
   - Knowledge base design
   - Retrieval configuration
   - Analysis prompt structure
   - Example analyses

5. **Signal Generation & Backtesting** (1.5-2 pages)
   - Signal methodology
   - Backtest period and assumptions
   - Performance metrics and charts
   - Comparison to benchmarks
   - Trade examples (winners and losers)

6. **Discussion** (0.5-1 page)
   - Limitations and failure modes
   - LLM hallucination risks and mitigations
   - Production deployment considerations
   - Future improvements

7. **Appendix**
   - Full prompt templates
   - Extraction examples
   - Signal history table
   - Code snippets (key algorithms)

---

## Presentation Rubric

### Structure (12 minutes total)
- Problem and commodity overview: 2 min
- Data pipeline and extraction: 3 min
- Analysis and signal generation: 3 min
- Backtest results: 3 min
- Q&A: 1 min

### Evaluation Criteria

| Criterion | Excellent (5) | Good (4) | Adequate (3) | Needs Work (1-2) |
|-----------|---------------|----------|--------------|------------------|
| **Commodity Knowledge** | Deep market understanding | Good fundamentals grasp | Basic knowledge | Superficial |
| **Technical Quality** | Robust LLM pipeline | Solid implementation | Basic functionality | Weak or buggy |
| **Results** | Strong performance, insightful | Good results, useful | Adequate results | Poor or missing |
| **Insights** | Novel, actionable findings | Useful observations | Basic conclusions | No real insights |
| **Q&A** | Handles tough questions | Answers most questions | Struggles with some | Cannot defend |

---

## Final Grading Rubric

### Data Pipeline (15 points)
| Points | Criteria |
|--------|----------|
| 13-15 | Fully automated; excellent quality; comprehensive sources |
| 10-12 | Good automation; solid quality; adequate sources |
| 7-9 | Partial automation; acceptable quality; limited sources |
| 0-6 | Manual; poor quality; insufficient sources |

### Report Processing (25 points)
| Points | Criteria |
|--------|----------|
| 22-25 | >90% extraction accuracy; robust prompts; excellent error handling |
| 17-21 | 80-90% accuracy; good prompts; solid error handling |
| 12-16 | 70-80% accuracy; basic prompts; some error handling |
| 0-11 | <70% accuracy; weak prompts; poor error handling |

### RAG System (20 points)
| Points | Criteria |
|--------|----------|
| 18-20 | Excellent retrieval; optimal design; strong query handling |
| 14-17 | Good retrieval; solid design; adequate query handling |
| 10-13 | Basic retrieval; simple design; limited query handling |
| 0-9 | Poor retrieval; weak design; minimal query handling |

### Analysis & Signals (20 points)
| Points | Criteria |
|--------|----------|
| 18-20 | Sophisticated analysis; well-reasoned signals; strong economics |
| 14-17 | Good analysis; solid signals; reasonable economics |
| 10-13 | Basic analysis; simple signals; limited economics |
| 0-9 | Weak analysis; poor signals; no economic reasoning |

### Backtesting (15 points)
| Points | Criteria |
|--------|----------|
| 13-15 | Rigorous methodology; comprehensive metrics; insightful analysis |
| 10-12 | Good methodology; solid metrics; useful analysis |
| 7-9 | Basic methodology; adequate metrics; limited analysis |
| 0-6 | Weak methodology; poor metrics; minimal analysis |

### Production Pipeline (5 points)
| Points | Criteria |
|--------|----------|
| 5 | Fully automated; monitored; production-ready |
| 4 | Mostly automated; some monitoring; nearly ready |
| 3 | Partially automated; minimal monitoring |
| 0-2 | Manual; no monitoring; not production-ready |

---

## Technical Specifications

### Minimum Requirements
- **Python 3.8+**
- **LLM Provider:** GPT-4, Claude 3, or equivalent
- **Data:** 5+ years historical, weekly or higher frequency
- **Extraction Test Set:** 20+ manually verified examples
- **Backtest Period:** 2+ years out-of-sample

### Recommended Stack

```python
# Core LLM
import anthropic  # or openai
from langchain import ...

# Data
import pandas as pd
import requests
from bs4 import BeautifulSoup

# RAG
import chromadb
from sentence_transformers import SentenceTransformer

# Backtesting
import numpy as np
import matplotlib.pyplot as plt
```

### Sample Workflow

```python
# 1. Data Collection
def fetch_eia_report(week):
    """Retrieve weekly petroleum report"""
    ...

# 2. Extraction
def extract_fundamentals(report_text, llm_client):
    """Extract structured data from report"""
    prompt = f"""
    Extract the following from this EIA report:
    - Crude production: MMbbl/d
    - Crude stocks: MMbbl
    - Gasoline stocks: MMbbl

    Report:
    {report_text}

    Output valid JSON.
    """
    response = llm_client.complete(prompt)
    return json.loads(response)

# 3. Analysis
def generate_signal(fundamentals, historical_context, llm_client):
    """Generate trading signal with rationale"""
    ...

# 4. Backtest
def backtest_signals(signals_df, prices_df):
    """Calculate returns and metrics"""
    ...
```

---

## Academic Integrity

- This is individual work (or approved pairs)
- Cite all data sources and external code
- You may use LLM assistants for coding, but:
  - Core logic must be your own
  - Understand all code submitted
  - Document AI assistance
- No sharing of prompt templates with classmates

---

## Resources

### Data Sources
- **EIA API:** [eia.gov/opendata](https://www.eia.gov/opendata/)
- **USDA APIs:** [usda.gov/developers](https://www.usda.gov/developers)
- **FRED:** [fred.stlouisfed.org](https://fred.stlouisfed.org)
- **Yahoo Finance:** yfinance library
- **Quandl:** Financial and commodity data

### LLM Providers
- **Anthropic Claude:** [anthropic.com](https://www.anthropic.com)
- **OpenAI:** [openai.com](https://www.openai.com)
- **Documentation:** API docs for extraction patterns

### Code Templates
- See `capstone/templates/` for starter code
- Reference notebooks for extraction and RAG

---

## Submission Instructions

1. **Create GitHub repository:**
   ```
   llm-commodity-analysis/
   ├── README.md
   ├── data/
   │   ├── raw/               # Downloaded reports
   │   ├── processed/         # Extracted data
   │   └── prices/            # Historical prices
   ├── src/
   │   ├── data_collection.py
   │   ├── extraction.py
   │   ├── rag.py
   │   ├── signals.py
   │   └── backtest.py
   ├── prompts/
   │   ├── extraction_prompts.md
   │   └── analysis_prompts.md
   ├── tests/
   │   └── extraction_test_set.csv
   ├── results/
   │   ├── extraction_accuracy.csv
   │   ├── backtest_results.csv
   │   └── figures/
   ├── docs/
   │   └── report.pdf
   └── requirements.txt
   ```

2. **Submit via course platform:**
   - GitHub repository link
   - Technical report (PDF)
   - Presentation slides (PDF)
   - Demo video (optional, 5 min max)

3. **Reproducibility requirements:**
   - requirements.txt with package versions
   - .env.example for API keys (not actual keys!)
   - Data download instructions or sample data
   - README with complete setup steps

---

*"In commodities, information advantage is alpha. LLMs that extract insights from reports faster and more comprehensively than human analysts create edge."*
