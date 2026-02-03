# Glossary: Generative AI for Commodities Trading & Fundamentals Analysis

## Commodity Market Terms

**Backwardation**
: Market condition where near-dated futures prices exceed deferred contracts. Often signals supply tightness.

**Barrel (bbl)**
: Unit for crude oil (42 U.S. gallons). Production and consumption typically measured in millions of barrels per day (MMbbl/d).

**Basis**
: Price differential between cash (spot) and futures, or between different delivery locations.

**BCF (Billion Cubic Feet)**
: Natural gas volume measurement. U.S. storage reported weekly by EIA in BCF.

**Bushel**
: Agricultural volume unit. Corn, soybeans, wheat measured in bushels. 1 bushel corn ≈ 56 lbs.

**Carry**
: Cost of holding physical commodity (storage, insurance, financing) minus benefits (convenience yield).

**Contango**
: Market condition where deferred futures exceed near-dated contracts. Normal when carry costs exceed convenience yield.

**Crack Spread**
: Refining margin. Difference between crude oil cost and refined products (gasoline, diesel) revenue.

**Crush Spread**
: Soybean processing margin. Revenue from soybean meal and oil minus soybean cost.

**Fundamentals**
: Supply and demand data: production, consumption, inventories, imports/exports, weather.

**Futures Curve (Term Structure)**
: Prices across different contract expiration months. Shape indicates market expectations.

**Henry Hub**
: Natural gas pricing point in Louisiana. Benchmark for U.S. natural gas prices.

**Inventory (Stocks)**
: Physical commodity held in storage. Key fundamental indicator.

**LNG (Liquefied Natural Gas)**
: Natural gas cooled to liquid form for transport. Major U.S. export.

**Roll Yield**
: Return from rolling expiring futures to next month. Positive in backwardation, negative in contango.

**Seasonal Pattern**
: Regular price patterns tied to production cycles, weather, or demand patterns.

**Sentiment**
: Market psychology extracted from news, positioning data (COT), or analyst reports.

**Spot Price**
: Current cash market price for immediate delivery.

**WTI (West Texas Intermediate)**
: U.S. crude oil benchmark, priced at Cushing, Oklahoma.

---

## Gen AI / LLM Terms

**Agent**
: Autonomous LLM system that uses tools, maintains state, and takes multi-step actions.

**Chain-of-Thought (CoT)**
: Prompting technique eliciting step-by-step reasoning. Useful for complex fundamental analysis.

**Completion**
: LLM-generated text response to a prompt.

**Context Window**
: Maximum tokens an LLM can process. Critical for long reports (e.g., full EIA petroleum reports).

**Embedding**
: Vector representation of text enabling semantic similarity search.

**Few-Shot Learning**
: Providing examples in prompts to guide LLM behavior. Example: showing how to extract data from previous reports.

**Function Calling (Tool Use)**
: LLM capability to invoke external functions (APIs, databases). Essential for data retrieval agents.

**Hallucination**
: LLM generating plausible but factually incorrect information. Major risk for trading applications.

**Instruction Following**
: LLM's ability to execute specific directives. Critical for structured data extraction.

**JSON Mode**
: LLM output format guaranteeing valid JSON. Ideal for extracting structured fundamental data.

**Prompt Engineering**
: Crafting effective prompts to elicit desired LLM behavior.

**RAG (Retrieval-Augmented Generation)**
: Augmenting LLM with retrieved information to ground responses in facts.

**Temperature**
: Sampling parameter controlling randomness. Low (0-0.3) for factual extraction, higher for creative analysis.

**Token**
: Subword unit processed by LLMs. Cost and context limits measured in tokens.

**Zero-Shot**
: Prompting without examples, relying on pre-training knowledge.

---

## Data Sources

**EIA (Energy Information Administration)**
: U.S. government agency publishing petroleum, natural gas, electricity data. Weekly reports critical for energy trading.

**USDA (United States Department of Agriculture)**
: Publishes crop reports, WASDE (supply/demand estimates), planting/harvest data.

**WASDE (World Agricultural Supply and Demand Estimates)**
: Monthly USDA report with global crop forecasts. Market-moving.

**COT (Commitments of Traders)**
: CFTC report showing futures positioning by trader type. Sentiment indicator.

**IEA (International Energy Agency)**
: Global energy data and forecasts. Monthly Oil Market Report influential.

**NOAA (National Oceanic and Atmospheric Administration)**
: Weather data and forecasts. Critical for energy demand, crop yields.

**Baker Hughes Rig Count**
: Weekly report of active oil/gas drilling rigs. Production leading indicator.

---

## Information Extraction Terms

**Entity Recognition (NER)**
: Identifying named entities (commodities, locations, dates, quantities) in text.

**Relation Extraction**
: Identifying relationships between entities (e.g., "U.S. production rose to X MMbbl/d").

**Sentiment Analysis**
: Classifying text tone (bullish, bearish, neutral). Applied to news, analyst reports.

**Structured Extraction**
: Converting unstructured text to structured data (tables, JSON).

**Table Extraction**
: Parsing tables from PDFs or HTML. Common in EIA, USDA reports.

**Time Series Extraction**
: Identifying and extracting numerical data with timestamps.

---

## Trading & Analysis Terms

**Alpha**
: Excess return above benchmark. Goal of fundamental trading strategies.

**Backtest**
: Evaluating strategy performance on historical data.

**Long**
: Betting on price increase (buy position).

**Short**
: Betting on price decrease (sell position).

**Signal**
: Trading indicator suggesting buy, sell, or hold action.

**Sharpe Ratio**
: Risk-adjusted return metric. (Return - Risk-free rate) / Volatility.

**Drawdown**
: Peak-to-trough decline in portfolio value.

**Position Sizing**
: Determining trade size based on conviction and risk.

---

## LLM Frameworks & Tools

**LangChain**
: Python framework for building LLM applications with chains, agents, and memory.

**LangGraph**
: Extension of LangChain for stateful, multi-actor workflows.

**ChromaDB**
: Open-source vector database for RAG applications.

**Pinecone**
: Managed vector database service for production RAG.

**BeautifulSoup**
: Python library for parsing HTML and XML. Useful for scraping commodity reports.

**Requests**
: Python HTTP library for API calls (EIA API, weather APIs).

**Pandas**
: Data manipulation library. Essential for time series and fundamental data.

---

## Prompt Patterns for Commodities

**Extraction Prompt**
: "Extract the following fields from this EIA report: production, imports, stocks..."

**Analysis Prompt**
: "Analyze the supply/demand balance for natural gas based on this data..."

**Signal Generation Prompt**
: "Based on this fundamental data, suggest a trading position with rationale..."

**Summarization Prompt**
: "Summarize the key takeaways from this week's petroleum status report..."

**Comparison Prompt**
: "Compare current inventory levels to 5-year average and explain implications..."

---

## RAG for Commodities

**Knowledge Base**
: Collection of historical reports, research notes, and market commentary.

**Retrieval**
: Finding relevant historical context for current market situation.

**Chunking**
: Splitting long reports (e.g., monthly OPEC reports) into semantically coherent segments.

**Metadata Filtering**
: Narrowing search by date, commodity, report type before semantic search.

**Hybrid Search**
: Combining keyword (BM25) and semantic (vector) search for better recall.

---

## Report Types

**Weekly Petroleum Status Report**
: EIA's flagship report (Wednesdays 10:30 AM ET). Crude stocks, production, refinery runs.

**Natural Gas Storage Report**
: EIA weekly (Thursdays 10:30 AM ET). Natural gas inventory changes.

**Monthly STEO (Short-Term Energy Outlook)**
: EIA monthly forecast for next 2 years.

**OPEC Monthly Oil Market Report**
: OPEC supply/demand estimates and production data.

**Crop Progress Report**
: USDA weekly report on planting/harvest progress, crop conditions.

**Quarterly Earnings Calls**
: Oil company earnings with production guidance and capital spending plans.

---

## Evaluation Metrics

**Extraction Accuracy**
: Percentage of correctly extracted data points vs. ground truth.

**Precision**
: True positives / (True positives + False positives) for classification tasks.

**Recall**
: True positives / (True positives + False negatives).

**F1 Score**
: Harmonic mean of precision and recall.

**BLEU Score**
: Metric for text generation quality (summaries, reports).

**Faithfulness**
: Measure of whether LLM output is grounded in retrieved sources.

---

## Acronyms

| Acronym | Full Name |
|---------|-----------|
| API | Application Programming Interface |
| BBL | Barrel |
| BCF | Billion Cubic Feet |
| BPD | Barrels Per Day |
| COT | Commitments of Traders |
| EIA | Energy Information Administration |
| GPT | Generative Pre-trained Transformer |
| IEA | International Energy Agency |
| JSON | JavaScript Object Notation |
| LLM | Large Language Model |
| LNG | Liquefied Natural Gas |
| MMBbl/d | Million Barrels Per Day |
| NER | Named Entity Recognition |
| NOAA | National Oceanic and Atmospheric Administration |
| OPEC | Organization of the Petroleum Exporting Countries |
| RAG | Retrieval-Augmented Generation |
| STEO | Short-Term Energy Outlook |
| USDA | United States Department of Agriculture |
| WASDE | World Agricultural Supply and Demand Estimates |
| WTI | West Texas Intermediate |

---

## Python Libraries for Commodities + Gen AI

| Library | Purpose |
|---------|---------|
| `langchain` | LLM application framework |
| `anthropic` | Claude API client |
| `openai` | GPT API client |
| `chromadb` | Vector database |
| `pandas` | Data manipulation |
| `requests` | API calls (EIA, USDA) |
| `beautifulsoup4` | Web scraping |
| `feedparser` | RSS feed parsing (news) |
| `pdfplumber` | PDF text extraction |
| `yfinance` | Financial data |
| `matplotlib` | Visualization |

---

## EIA API Endpoints

| Endpoint | Data Type |
|----------|-----------|
| `/petroleum/sum/sndw` | Weekly petroleum summary |
| `/natural-gas/stor/wkly` | Natural gas storage |
| `/petroleum/crd/crpdn` | Crude production |
| `/petroleum/pri/spt` | Spot prices |

---

## Common Prompt Templates

### Extraction Template
```
Extract the following from the EIA report:
- Crude oil production: [number] MMbbl/d
- Crude imports: [number] MMbbl/d
- Commercial crude stocks: [number] MMbbl
- Gasoline stocks: [number] MMbbl
- Refinery utilization: [number]%

Report text:
{report_text}
```

### Analysis Template
```
You are a commodity analyst. Analyze the following data:

Current Week Data:
{current_data}

5-Year Average:
{historical_avg}

Provide:
1. Supply/demand balance assessment
2. Inventory position vs. normal
3. Price implications
4. Risk factors
```

### Signal Template
```
Based on this fundamental data, generate a trading signal.

Data:
{fundamental_data}

Output JSON:
{
  "position": "long|short|neutral",
  "conviction": 1-10,
  "rationale": "...",
  "risks": ["...", "..."]
}
```

---

## Commodity-Specific Features

### Crude Oil
- Production (shale, conventional, offshore)
- Refinery runs and utilization
- Crack spreads (3-2-1, 5-3-2)
- Strategic Petroleum Reserve (SPR)
- OPEC+ production quotas

### Natural Gas
- Storage levels vs. 5-year range
- Heating degree days (HDD) / Cooling degree days (CDD)
- LNG export volumes
- Power generation demand
- Pipeline constraints

### Agriculture
- Planting progress vs. 5-year average
- Crop condition ratings (good/excellent %)
- Yield estimates
- Ending stocks-to-use ratio
- Weather (drought, floods, temperatures)

---

## Risk Factors in LLM Trading Applications

| Risk | Mitigation |
|------|------------|
| **Hallucination** | RAG, structured output, validation |
| **Stale Data** | Real-time data refresh, timestamps |
| **Misinterpretation** | Few-shot examples, chain-of-thought |
| **Bias** | Multiple sources, human review |
| **API Failures** | Retry logic, fallback providers |
| **Cost Overruns** | Token budgets, caching, monitoring |

---

*This glossary combines commodity market knowledge with Gen AI terminology for trading applications.*
