# Quiz: Module 2 - RAG for Commodity Research

**Course:** GenAI for Commodity Markets
**Module:** 2 - RAG for Commodity Research
**Total Points:** 100
**Time Limit:** 30 minutes
**Attempts Allowed:** 2

## Instructions

This quiz assesses your understanding of retrieval-augmented generation systems for commodity research, including knowledge base design, chunking strategies, and grounded analysis generation.

---

## Section 1: Knowledge Base Design (25 points)

### Question 1 (10 points)

You're designing a vector knowledge base for commodity research that will contain:
- 5 years of EIA weekly petroleum reports (~260 documents)
- 3 years of USDA monthly WASDE reports (~36 documents)
- Daily news articles (~5,000 documents)

Which knowledge base architecture is MOST appropriate?

**Option A: Single Collection**
```python
# All documents in one vector store
kb = VectorStore(collection="commodity_data")
kb.add(eia_reports + usda_reports + news_articles)
```

**Option B: Separate Collections by Source**
```python
# Three separate collections
eia_kb = VectorStore(collection="eia_reports")
usda_kb = VectorStore(collection="usda_reports")
news_kb = VectorStore(collection="news_articles")

# Query all and merge results
results = merge_results(
    eia_kb.query(q),
    usda_kb.query(q),
    news_kb.query(q)
)
```

**Option C: Hierarchical Collections with Metadata Filtering**
```python
# Single collection with rich metadata
kb = VectorStore(collection="commodity_research")

for doc in all_documents:
    kb.add(
        text=doc.content,
        metadata={
            "source_type": "eia|usda|news",
            "commodity": ["crude_oil", "natural_gas", ...],
            "date": doc.date,
            "document_type": "report|news|forecast",
            "temporal_scope": "weekly|monthly|daily"
        }
    )

# Query with metadata filters
results = kb.query(
    query="crude oil inventory trends",
    filter={
        "source_type": ["eia", "news"],
        "commodity": "crude_oil",
        "date": {"$gte": "2024-01-01"}
    }
)
```

**Option D: Time-Partitioned Collections**
```python
# Separate collections by year
kb_2024 = VectorStore(collection="commodity_2024")
kb_2023 = VectorStore(collection="commodity_2023")
# ...
```

A) Option A - Simplest implementation
B) Option B - Best separation of concerns
C) Option C - Flexible metadata-driven architecture
D) Option D - Optimized for temporal queries

**Answer: C**

**Explanation:**

**A - Single Collection (No Metadata):**
- Problem: Can't filter by source, date, or commodity
- Results mix irrelevant documents
- No way to prefer recent data over old

**B - Separate Collections:**
- Problem: Must query 3 collections and merge results
- Harder to rank cross-source results
- Can't easily query "all energy reports" (EIA + relevant news)
- Complexity in result aggregation

**C (Correct) - Metadata-Driven:**
```python
# Enables flexible queries:

# Recent EIA crude oil data only
eia_recent = kb.query(
    "latest inventory levels",
    filter={"source_type": "eia", "commodity": "crude_oil",
            "date": {"$gte": "2024-11-01"}}
)

# Historical context: same period last year
eia_history = kb.query(
    "inventory levels",
    filter={"source_type": "eia", "commodity": "crude_oil",
            "date": {"$gte": "2023-11-01", "$lte": "2023-12-01"}}
)

# Multi-source analysis
multi = kb.query(
    "natural gas supply outlook",
    filter={"commodity": "natural_gas",
            "source_type": ["eia", "news"],
            "date": {"$gte": "2024-01-01"}}
)

# Cross-commodity queries
energy = kb.query(
    "energy market trends",
    filter={"commodity": ["crude_oil", "natural_gas"]}
)
```

**Benefits:**
- Single system, flexible querying
- Easy to add new metadata fields
- Supports complex temporal logic
- Efficient ranking across all documents
- Standard vector DB capability

**D - Time Partitions:**
- Operational complexity (managing multiple collections)
- Hard to query across time periods
- No benefit over metadata filtering by date

---

### Question 2 (8 points)

For commodity reports, which metadata fields are MOST critical for effective retrieval? Rank from most to least important:

I. Document source (EIA, USDA, IEA)
II. Commodity mentioned (crude oil, natural gas, corn, etc.)
III. Publication date
IV. Author name
V. Document format (PDF, HTML, API)
VI. Temporal coverage (the time period the data describes)

A) I, II, III, VI, IV, V
B) II, III, VI, I, IV, V
C) III, II, I, VI, V, IV
D) II, VI, III, I, IV, V

**Answer: B**

**Explanation:**

**B (Correct): II, III, VI, I, IV, V**

**1. Commodity (II) - MOST CRITICAL**
```python
# Users always search by commodity
"What are crude oil inventories?" → filter: commodity="crude_oil"
"Corn production forecast?" → filter: commodity="corn"
```
Without this, you get irrelevant cross-commodity results.

**2. Publication Date (III) - CRITICAL**
```python
# Commodity markets are time-sensitive
"Latest crude oil data" → filter: date = most_recent
"Historical comparison" → filter: date range
```
Stale data is worse than no data in trading.

**3. Temporal Coverage (VI) - VERY IMPORTANT**
```python
# Distinguishes forecast from historical
{
    "publication_date": "2024-11-15",  # When published
    "data_period": "2024-12-01"        # What period it covers
}

# A December forecast published in November
# vs November actuals published in December
```
Critical for understanding forecast vs. reality.

**4. Document Source (I) - IMPORTANT**
```python
# Credibility and update frequency differ
"eia" → weekly, authoritative for US
"news" → daily, sentiment signal
"usda" → monthly, authoritative for agriculture
```

**5. Author Name (IV) - MINOR**
- Occasionally useful for analyst research
- Not critical for government reports

**6. Document Format (V) - LEAST IMPORTANT**
- Implementation detail
- Doesn't affect content relevance

---

### Question 3 (7 points)

You're storing EIA reports with weekly crude oil inventory data. Each report contains multiple metrics (production, imports, inventory, refinery utilization). What is the BEST granularity for chunking?

A) Store entire report as single chunk
B) One chunk per metric (separate chunks for production, imports, inventory)
C) One chunk per paragraph
D) One chunk per sentence with overlapping context

**Answer: B**

**Explanation:**

**A - Entire Report:**
```python
# Problems:
# - Report may exceed optimal chunk size (1000-1500 tokens)
# - User asks "What are crude oil inventories?"
# - System retrieves entire report including irrelevant gasoline, distillate data
# - Wastes context window on irrelevant content
```

**B (Correct) - One Chunk Per Metric:**
```python
@dataclass
class EIAChunk:
    text: str          # "U.S. crude oil inventories decreased by 5.2 million barrels..."
    metric: str        # "crude_oil_inventory"
    value: float       # 430.0
    change: float      # -5.2
    units: str         # "million_barrels"
    date: date         # Report date
    vs_5yr_avg: float  # -3.0 (percent)

# Benefits:
# 1. Precise retrieval
user_query = "crude oil inventory"
retrieved = kb.query(query, filter={"metric": "crude_oil_inventory"})
# Returns ONLY inventory chunks, not production or refinery data

# 2. Structured context
# Can include both text AND structured data
chunk_text = f"""Crude Oil Inventory (as of {date}):
Current Level: {value} million barrels
Weekly Change: {change:+.1f} million barrels
vs 5-Year Average: {vs_5yr_avg:+.1f}%

Full context: {original_text}
"""

# 3. Better ranking
# Similarity search finds specific metric discussions
# rather than general report matches
```

**C - Paragraph-based:**
- May split metric context across chunks
- No semantic structure

**D - Sentence-based:**
- Too granular
- Loses context
- Inefficient (too many chunks)

**Implementation:**
```python
def chunk_eia_report(report):
    """Chunk by commodity and metric."""
    chunks = []

    # Extract each metric section
    metrics = [
        "crude_oil_inventory",
        "crude_oil_production",
        "crude_oil_imports",
        "gasoline_inventory",
        "distillate_inventory",
        # ...
    ]

    for metric in metrics:
        section = extract_section(report, metric)
        structured_data = extract_structured_data(section)

        chunk = {
            "text": section,
            "metadata": {
                "metric": metric,
                "date": report.date,
                "source": "eia_wpsr",
                **structured_data  # Add extracted values
            }
        }
        chunks.append(chunk)

    return chunks
```

---

## Section 2: Retrieval Strategies (30 points)

### Question 4 (12 points)

A trader asks: "How do current crude oil inventories compare to last year and the five-year average?"

This question requires data from:
- Current week's EIA report (most recent)
- Same week last year's report
- Five-year average calculation

Which retrieval strategy is MOST effective?

**Strategy A: Single Semantic Search**
```python
results = kb.query("crude oil inventories comparison current vs last year vs average")
answer = llm.generate(context=results, query=user_query)
```

**Strategy B: Multi-Query Retrieval**
```python
current = kb.query(
    "current crude oil inventory",
    filter={"date": most_recent_date}
)
last_year = kb.query(
    "crude oil inventory",
    filter={"date": same_week_last_year}
)
five_year = kb.query(
    "crude oil inventory five year average",
    filter={"date": {"$gte": five_years_ago}}
)

answer = llm.generate(
    context={"current": current, "last_year": last_year, "five_year": five_year},
    query=user_query
)
```

**Strategy C: Hybrid: Structured + Semantic**
```python
# First: Get current report's structured data
current_report = get_latest_eia_report()
current_value = current_report.crude_inventory
current_vs_5yr = current_report.vs_5yr_avg_pct  # Already in report!
current_vs_ly = current_report.vs_last_year_pct  # Already in report!

# Second: Retrieve narrative context for "why"
context_chunks = kb.query(
    "factors affecting crude oil inventory changes",
    filter={"date": {"$gte": thirty_days_ago}}
)

# Third: Generate answer with structured + narrative
answer = llm.generate(f"""Using this data:
Current crude oil inventory: {current_value} million barrels
vs Five-Year Average: {current_vs_5yr:+.1f}%
vs Last Year: {current_vs_ly:+.1f}%

And this context about recent market factors:
{context_chunks}

Answer: {user_query}
""")
```

**Strategy D: Time-Series Database Query**
```python
# Directly query time-series DB, skip RAG entirely
data = timeseries_db.query("""
    SELECT date, crude_inventory, five_year_avg
    FROM eia_data
    WHERE date IN (CURRENT_WEEK, SAME_WEEK_LAST_YEAR)
""")
```

A) Strategy A
B) Strategy B
C) Strategy C
D) Strategy D

**Answer: C**

**Explanation:**

**Strategy A - Single Semantic Search:**
- Problem: May not retrieve all necessary time periods
- Semantic search finds "similar" content, not specific dates
- May miss last year's data or mix up time periods

**Strategy B - Multi-Query:**
- Better than A: explicitly fetches each time period
- Problem: Still relies on semantic search to find right data
- May retrieve prose descriptions instead of actual numbers
- Inefficient: 3 separate retrievals

**Strategy C (Correct) - Hybrid:**
```python
# Advantages:
# 1. EIA reports ALREADY CONTAIN the comparisons!
# "...inventories are 3% below the five-year average"
# "...6% higher than last year"

# 2. Structured data extraction (from Module 1) gives exact numbers
current_report = {
    "crude_inventory": 430.0,
    "vs_5yr_avg_pct": -3.0,
    "vs_last_year_pct": +6.0
}

# 3. Use RAG for explanatory context, not raw data
# "Why are inventories low?"
# → Retrieve: "Strong refinery demand, reduced imports from Canada..."

# 4. Best accuracy + best context
# Structured data: 100% accurate (no hallucination)
# Semantic context: Explains the "why" behind numbers
```

**Real implementation:**
```python
def answer_comparative_query(user_query):
    # Step 1: Parse intent
    intent = analyze_query(user_query)
    # → {"type": "comparison", "metric": "inventory",
    #    "timeframes": ["current", "last_year", "5yr_avg"]}

    # Step 2: Fetch structured data
    latest = get_latest_structured_eia_data()
    current_value = latest['crude_inventory']
    comparisons = {
        'vs_5yr_avg': latest['vs_5yr_avg_pct'],
        'vs_last_year': latest['vs_last_year_pct']
    }

    # Step 3: Retrieve explanatory context
    if comparisons['vs_5yr_avg'] < -5:  # Significantly below average
        context = kb.query(
            "factors causing low crude oil inventories",
            filter={"date": {"$gte": thirty_days_ago}}
        )

    # Step 4: Generate answer
    return llm.generate(f"""
Current crude oil inventories: {current_value} million barrels
- {comparisons['vs_5yr_avg']:+.1f}% vs five-year average
- {comparisons['vs_last_year']:+.1f}% vs last year

Context: {context}

Question: {user_query}
""")
```

**Strategy D:**
- Correct approach for pure data queries
- But user's question implies they want explanation, not just numbers
- Hybrid approach provides both

**Key Principle:** Use structured data for precision, RAG for explanation.

---

### Question 5 (10 points)

You've built a RAG system for commodity research. A user asks: "What's driving natural gas prices higher?"

Your system retrieves these 5 chunks (ranked by similarity):

1. **Similarity: 0.92** - News article from 3 months ago: "Natural gas prices surged on cold weather forecast"
2. **Similarity: 0.89** - EIA report from last week: "Natural gas storage fell 85 Bcf, larger than expected"
3. **Similarity: 0.87** - Blog post from 6 months ago: "Why natural gas prices could rise in 2025"
4. **Similarity: 0.85** - EIA report from last week: "LNG exports reached record levels"
5. **Similarity: 0.83** - USDA report: "Corn prices rise on strong demand"

Which re-ranking strategy would be MOST effective before sending to the LLM?

A) Use the top 3 by similarity score (chunks 1, 2, 3)
B) Filter by recency, then take top 3 (chunks 2, 4, and next most recent)
C) Implement hybrid scoring: `score = 0.4 * similarity + 0.4 * recency + 0.2 * source_credibility`
D) Use all 5 chunks and let the LLM determine relevance

**Answer: C**

**Explanation:**

**A - Top 3 by Similarity:**
- Includes chunk #1: 3 months old, no longer relevant
- Includes chunk #3: Speculative blog, not current driver
- Misses chunk #4: Recent, relevant (LNG exports drive current prices)

**B - Recency Filter:**
- Better than A: focuses on current drivers
- Problem: Discards similarity scores entirely
- Might include recent but irrelevant content

**C (Correct) - Hybrid Scoring:**
```python
def rerank_chunks(chunks, query_date):
    """Multi-factor ranking for commodity queries."""

    for chunk in chunks:
        # Factor 1: Semantic similarity (base score)
        similarity_score = chunk.similarity  # 0-1

        # Factor 2: Recency (commodity markets are time-sensitive)
        days_old = (query_date - chunk.date).days
        recency_score = np.exp(-days_old / 30)  # Exponential decay
        # Recent (1 week): ~0.8
        # 1 month old: ~0.37
        # 3 months old: ~0.05

        # Factor 3: Source credibility
        source_scores = {
            'eia': 1.0,      # Authoritative government data
            'usda': 1.0,     # Authoritative government data
            'iea': 0.9,      # International authority
            'news': 0.7,     # News from known sources
            'blog': 0.4,     # Lower credibility
        }
        source_score = source_scores.get(chunk.source, 0.5)

        # Factor 4: Commodity relevance
        commodity_match = 1.0 if query_commodity in chunk.commodities else 0.5

        # Weighted combination
        chunk.final_score = (
            0.3 * similarity_score +
            0.4 * recency_score +      # Most important for "what's driving" questions
            0.2 * source_score +
            0.1 * commodity_match
        )

    # Re-rank by final score
    ranked = sorted(chunks, key=lambda c: c.final_score, reverse=True)
    return ranked[:3]  # Top 3

# Applying to our example:
# Chunk 1: 0.92 * 0.3 + 0.05 * 0.4 + 0.7 * 0.2 + 1.0 * 0.1 = 0.536
# Chunk 2: 0.89 * 0.3 + 0.80 * 0.4 + 1.0 * 0.2 + 1.0 * 0.1 = 0.887  ← Top!
# Chunk 3: 0.87 * 0.3 + 0.01 * 0.4 + 0.4 * 0.2 + 1.0 * 0.1 = 0.445
# Chunk 4: 0.85 * 0.3 + 0.80 * 0.4 + 1.0 * 0.2 + 1.0 * 0.1 = 0.875  ← 2nd
# Chunk 5: 0.83 * 0.3 + 0.80 * 0.4 + 1.0 * 0.2 + 0.5 * 0.1 = 0.819  ← 3rd (but wrong commodity!)

# After commodity filtering (must match natural_gas):
# Final Top 3: Chunks 2, 4, and next relevant natural gas chunk
```

**Benefits:**
- Balances semantic relevance with temporal freshness
- Prioritizes authoritative sources
- Filters cross-commodity contamination
- Adaptable weights based on query type

**D - All 5 Chunks:**
- Wastes context window on irrelevant chunks
- Dilutes signal with noise
- May confuse LLM with outdated information
- Chunk #5 (corn prices) completely irrelevant

**Query-Type Specific Weights:**
```python
def get_ranking_weights(query_type):
    """Adjust weights based on query type."""
    if "latest" in query_type or "current" in query_type:
        return {"similarity": 0.2, "recency": 0.6, "source": 0.15, "commodity": 0.05}
    elif "historical" in query_type or "trend" in query_type:
        return {"similarity": 0.4, "recency": 0.1, "source": 0.3, "commodity": 0.2}
    else:  # General query
        return {"similarity": 0.3, "recency": 0.4, "source": 0.2, "commodity": 0.1}
```

---

### Question 6 (8 points)

Your RAG system needs to answer: "Compare crude oil production forecasts from the last three EIA STEO reports."

The STEO (Short-Term Energy Outlook) is published monthly. Which retrieval approach is MOST effective?

A) Semantic search for "crude oil production forecast" and hope the top results include all three reports
B) Query with metadata filter: `source="eia_steo", metric="crude_production_forecast"`, sort by date DESC, take top 3
C) Query for "EIA STEO crude oil forecast" and manually parse the LLM response for the three most recent
D) Query all EIA documents and let the LLM filter for STEO reports

**Answer: B**

**Explanation:**

**A - Semantic Search Only:**
- Unreliable: May not return all three reports
- May return news articles discussing the forecasts instead of actual reports
- No guarantee of chronological coverage

**B (Correct) - Metadata-Driven Query:**
```python
def get_last_n_steo_forecasts(commodity, metric, n=3):
    """Retrieve last N STEO forecast values for comparison."""

    results = kb.query(
        query=f"{commodity} {metric}",  # Semantic component
        filter={
            "source": "eia_steo",       # Exact source
            "metric": metric,            # Exact metric
            "document_type": "forecast"  # Not historical data
        },
        sort_by="publication_date",
        order="DESC",
        limit=n
    )

    # Extract structured data
    forecasts = []
    for doc in results:
        forecasts.append({
            "publication_date": doc.metadata["publication_date"],
            "forecast_period": doc.metadata["forecast_period"],
            "value": doc.metadata["forecast_value"],
            "narrative": doc.text
        })

    return forecasts

# Usage:
forecasts = get_last_n_steo_forecasts("crude_oil", "production_forecast", n=3)

# Generate comparison
answer = llm.generate(f"""Compare these three EIA STEO forecasts for crude oil production:

1. {forecasts[0]["publication_date"]} STEO:
   Forecast: {forecasts[0]["value"]} million bpd
   Context: {forecasts[0]["narrative"]}

2. {forecasts[1]["publication_date"]} STEO:
   Forecast: {forecasts[1]["value"]} million bpd
   Context: {forecasts[1]["narrative"]}

3. {forecasts[2]["publication_date"]} STEO:
   Forecast: {forecasts[2]["value"]} million bpd
   Context: {forecasts[2]["narrative"]}

Analysis: {query}
""")
```

**Advantages:**
- Guaranteed to retrieve exactly the right reports
- Chronologically ordered
- Includes structured values for comparison
- No false positives (news, summaries, etc.)

**C - Semantic + Manual Parsing:**
- Adds unnecessary complexity
- LLM may miss reports or misidentify them
- Manual parsing is error-prone

**D - Query All + LLM Filtering:**
- Extremely inefficient
- May exceed context window
- Expensive (process many irrelevant documents)
- LLM may miss reports

**Key Insight:** When you need specific documents from known sources, use metadata filters. Save semantic search for exploratory queries.

---

## Section 3: Grounded Analysis Generation (25 points)

### Question 7 (10 points)

You're generating commodity research reports using RAG. A user requests: "Analyze the natural gas market outlook for winter 2024-25."

Your RAG system retrieves relevant EIA reports and news. Which generation prompt is MOST likely to produce accurate, grounded analysis?

```python
# Option A
prompt = f"""Analyze the natural gas market outlook for winter 2024-25.

Context:
{retrieved_chunks}
"""

# Option B
prompt = f"""You are an expert commodity analyst. Based on the following data,
provide a comprehensive analysis of the natural gas market outlook for winter 2024-25.

Context:
{retrieved_chunks}

Provide your expert opinion on price direction.
"""

# Option C
prompt = f"""Analyze the natural gas market outlook for winter 2024-25 based
ONLY on the provided data. Do not use external knowledge.

Context:
{retrieved_chunks}

Structure your response:
1. Current supply/demand balance (cite specific data)
2. Storage levels vs historical averages (cite specific data)
3. Key factors for winter (cite specific data)
4. Synthesis and outlook

For each point, cite the specific data source and value.
If data is unavailable, state "Data not available" rather than speculating.
"""

# Option D
prompt = f"""Using this context: {retrieved_chunks}

Answer: What is the natural gas outlook for winter 2024-25?

Response:"""
```

A) Option A
B) Option B
C) Option C
D) Option D

**Answer: C**

**Explanation:**

**A - Minimal Prompt:**
- No guidance on how to use retrieved information
- May mix retrieved data with pre-training knowledge
- No citation requirements
- Unclear structure

**B - "Expert" Role-Play:**
- "Expert opinion" encourages speculation beyond retrieved data
- Anthropomorphizing increases hallucination risk
- No grounding requirements
- "Comprehensive analysis" may go beyond available data

**C (Correct) - Grounded with Structure:**
```python
# Example output from this prompt:
"""
Natural Gas Market Outlook: Winter 2024-25

1. CURRENT SUPPLY/DEMAND BALANCE
Storage injections totaled 85 Bcf for the week ending Nov 8, bringing total
working gas in storage to 3,842 Bcf (Source: EIA NGSR, Nov 15 2024).

This is 152 Bcf (4.1%) higher than last year and 181 Bcf (4.9%) above the
five-year average (Source: EIA NGSR, Nov 15 2024).

Production averaged 103.2 Bcf/d in October (Source: EIA NGMS, Nov 2024).

2. STORAGE LEVELS VS HISTORICAL
Current storage of 3,842 Bcf represents:
- 4.1% above last year's 3,690 Bcf
- 4.9% above 5-year average of 3,661 Bcf
(Source: EIA NGSR, Nov 15 2024)

At this injection pace, winter entry storage could reach 3,900-4,000 Bcf,
well above historical norms.

3. KEY FACTORS FOR WINTER
- LNG exports averaging 12.1 Bcf/d, up 15% year-over-year (Source: EIA, Oct 2024)
- Weather forecasts: Data not available in provided sources
- Production outlook: Data not available in provided sources

4. SYNTHESIS
Higher-than-average storage entering winter provides a supply cushion.
However, elevated LNG exports continue to tighten the market. The outlook
depends on winter weather severity, which is not yet determinable from
available data.
"""
```

**Key Features:**
- Every claim cites specific source
- Acknowledges data gaps explicitly
- Structure makes verification easy
- No speculation beyond data

**D - Minimal Question:**
- Too casual
- No grounding requirements
- No structure
- Likely to produce unverified claims

**Production Best Practice:**
```python
def generate_grounded_analysis(query, chunks):
    """Generate analysis with citation requirements."""

    # Prepare context with source labels
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[Source {i+1}: {chunk.source}, {chunk.date}]\n"
        context += chunk.text + "\n"

    prompt = f"""Analyze: {query}

Available data:
{context}

Requirements:
1. Cite every factual claim as [Source N]
2. Distinguish facts from analysis
3. State "Data not available" for missing information
4. Do not speculate beyond provided data

Structure:
- Facts from data (with citations)
- Analysis based on facts
- Data gaps and limitations
"""

    response = llm.generate(prompt)

    # Post-process: Verify all citations link to actual chunks
    verify_citations(response, chunks)

    return response
```

---

### Question 8 (8 points)

Your RAG system generates this response:

> "Crude oil inventories declined by 5.2 million barrels last week to 430.0 million barrels, which is 3% below the five-year average [Source: EIA WPSR, Nov 15]. This drawdown was driven by strong refinery demand, with runs averaging 16.2 million bpd [Source: EIA WPSR, Nov 15]. Gasoline demand typically peaks in summer months [General Knowledge]."

Which statement is TRUE about this response?

A) Completely grounded - all statements cite sources
B) Partially grounded - mix of retrieved and general knowledge, properly labeled
C) Poor practice - should not mix retrieved and general knowledge
D) Hallucination risk - the refinery runs number is likely incorrect

**Answer: B**

**Explanation:**

**A - Completely Grounded:**
- False: The third statement explicitly uses general knowledge

**B (Correct) - Partially Grounded with Transparency:**
```python
# This is GOOD PRACTICE when general knowledge adds value:

Response breakdown:
1. "5.2 million barrel decline to 430.0 million" - [Source: EIA WPSR]
   → Grounded in retrieved data ✓

2. "3% below five-year average" - [Source: EIA WPSR]
   → Grounded in retrieved data ✓

3. "Driven by strong refinery demand, 16.2 million bpd" - [Source: EIA WPSR]
   → Grounded in retrieved data ✓

4. "Gasoline demand typically peaks in summer" - [General Knowledge]
   → Contextual knowledge, clearly labeled ✓

# The key: TRANSPARENT LABELING
# User knows which claims come from which source
# Can verify cited claims
# Understands contextual claims are general knowledge

# This is better than:
# - Refusing to use any general knowledge (too restrictive)
# - Mixing sources without labels (no transparency)
```

**Good practice pattern:**
```python
system_prompt = """When analyzing commodity data:
1. Primary: Use retrieved data and cite sources [Source: X]
2. Secondary: Use general commodity knowledge when helpful, label as [General Knowledge]
3. Never: Speculate about specific data points or forecasts without data

Example:
"Crude oil inventories fell 5.2 million barrels [Source: EIA WPSR, Nov 15].
This is unusual for this time of year, as inventories typically build in
shoulder season months [General Knowledge]."
"""
```

**C - Should Not Mix:**
- Too restrictive
- General domain knowledge (seasonal patterns, market structure) adds valuable context
- Key is transparency through labeling

**D - Hallucination Risk:**
- False: The number is cited from the EIA report
- Validation would confirm this against the original source

**Anti-Pattern (Bad Practice):**
```python
# DO NOT DO THIS:
"Crude oil inventories fell 5.2 million barrels to 430.0 million barrels,
which is 3% below average. This suggests strong refinery demand and indicates
prices will likely rise 10-15% over the next month based on historical patterns."

# Problems:
# - No citations
# - Mixes data with speculation
# - Makes specific price forecast without data support
# - User can't distinguish facts from analysis
```

---

### Question 9 (7 points)

You're implementing citation verification for your RAG system. After generating an analysis, you want to verify that cited data actually appears in the retrieved chunks. Which verification approach is MOST robust?

A) Check if citation number exists (e.g., [Source 1] exists if you retrieved 3+ chunks)
B) Extract cited values and verify they appear exactly in the source chunk
C) Use LLM to verify: "Does this source support this claim?"
D) Combine: Check exact value matches + verify values are within reasonable range + use LLM for contextual claims

**Answer: D**

**Explanation:**

**A - Citation Number Exists:**
```python
def weak_verification(response, chunks):
    """Weak: Only checks citation format."""
    citations = re.findall(r'\[Source (\d+)\]', response)
    return all(int(c) <= len(chunks) for c in citations)

# Problem: [Source 1] might cite wrong value
# "Inventories are 500 million barrels [Source 1]"
# But Source 1 actually says "430 million barrels"
```

**B - Exact Value Match:**
```python
def exact_match_verification(response, chunks):
    """Check if specific values appear in sources."""

    # Extract numerical claims
    claims = extract_claims(response)
    # [("430.0 million barrels", "Source 1"),
    #  ("-5.2 million barrels", "Source 1")]

    for claim_value, cited_source in claims:
        source_text = chunks[int(cited_source)-1].text

        if claim_value not in source_text:
            raise CitationError(f"Value '{claim_value}' not found in {cited_source}")

    return True

# Better but still limited:
# - Only works for exact numerical matches
# - Doesn't verify contextual or analytical claims
# - May fail on equivalent phrasings ("5.2" vs "5.20")
```

**C - LLM Verification:**
```python
def llm_verification(claim, source_chunk):
    """Use LLM to verify claim support."""

    verification_prompt = f"""Does this source support this claim?

Claim: {claim}
Source: {source_chunk}

Answer: Yes/No and explain."""

    result = llm.generate(verification_prompt)
    return result

# Problems:
# - Expensive (additional LLM call per claim)
# - LLM may be lenient or make errors
# - Not suitable for numerical precision
```

**D (Correct) - Hybrid Verification:**
```python
def robust_citation_verification(response, chunks):
    """Multi-layered verification of citations."""

    claims = extract_claims_with_citations(response)

    for claim in claims:
        source_idx = claim.source_number - 1
        source_chunk = chunks[source_idx]

        # Layer 1: Exact match for numerical values
        if claim.type == "numerical":
            value_str = str(claim.value)
            # Handle equivalent formats
            equivalent_formats = [
                value_str,
                f"{claim.value:.1f}",
                f"{claim.value:.2f}",
                value_str.replace(".", ",")  # International formats
            ]

            if not any(fmt in source_chunk.text for fmt in equivalent_formats):
                raise CitationError(
                    f"Numerical value {claim.value} not found in Source {source_idx+1}"
                )

            # Verify value is reasonable for metric
            if not validate_range(claim.metric, claim.value):
                raise ValidationError(
                    f"{claim.value} outside reasonable range for {claim.metric}"
                )

        # Layer 2: LLM verification for contextual claims
        elif claim.type == "contextual":
            # E.g., "strong refinery demand" citing "runs at 95% utilization"

            verification_prompt = f"""Strictly verify if this source supports this claim:

Claim: {claim.text}
Source: {source_chunk.text}

Does the source directly support this claim?
- YES: Explicit support (e.g., "strong demand" in source)
- PARTIAL: Implicit support (e.g., high utilization implies strong demand)
- NO: Not supported

Answer: YES/PARTIAL/NO
Reasoning: [explanation]"""

            result = llm.generate(verification_prompt)

            if result.startswith("NO"):
                raise CitationError(
                    f"Claim '{claim.text}' not supported by Source {source_idx+1}"
                )

            if result.startswith("PARTIAL"):
                # Flag for review or add qualifier
                claim.add_qualifier("based on indirect indicators")

        # Layer 3: Cross-reference with structured data (if available)
        if claim.metric in source_chunk.metadata:
            metadata_value = source_chunk.metadata[claim.metric]
            if abs(claim.value - metadata_value) > 0.1:  # Tolerance
                raise CitationError(
                    f"Claim value {claim.value} doesn't match structured data {metadata_value}"
                )

    return True
```

**Benefits of Hybrid Approach:**
- Exact verification for numbers (no hallucination risk)
- LLM verification for nuanced contextual claims
- Cross-reference with structured data when available
- Range validation catches obvious errors
- Flags partial support for transparency

---

## Section 4: Production Considerations (20 points)

### Question 10 (10 points)

Your commodity RAG system has been in production for 2 months. You notice:
- Query latency increased from 2 seconds to 8 seconds
- Knowledge base grew from 1,000 to 10,000 documents
- Users complain about occasionally outdated information
- Cost per query doubled

Rank these optimization strategies by implementation priority:

I. Implement semantic caching for common queries
II. Add date-based filtering to prioritize recent documents
III. Increase vector database index parameters for faster search
IV. Reduce chunk size to decrease context window usage
V. Implement tiered storage (recent docs in fast DB, archived docs in cold storage)

A) II, III, I, V, IV
B) I, II, III, IV, V
C) II, I, III, V, IV
D) III, II, I, V, IV

**Answer: C**

**Explanation:**

**C (Correct): II, I, III, V, IV**

**Priority 1 - Date Filtering (II):**
```python
# Solves two problems simultaneously:
# 1. Reduces search space → faster queries
# 2. Prioritizes recent data → fixes staleness complaints

def query_with_recency_bias(query, lookback_days=90):
    """Prioritize recent documents for commodity queries."""

    # First: Try recent data only
    recent_results = kb.query(
        query,
        filter={"date": {"$gte": days_ago(lookback_days)}},
        limit=5
    )

    # If insufficient recent results, expand search
    if len(recent_results) < 3:
        all_results = kb.query(query, limit=5)
        return rerank_by_recency(all_results)

    return recent_results

# Impact:
# - Latency: Search 1k recent docs instead of 10k total → 80% faster
# - Relevance: Recent docs more relevant for commodity markets → better quality
# - Cost: Fewer tokens in context → lower cost
```

**Priority 2 - Semantic Caching (I):**
```python
# Common queries in commodity markets:
# - "Latest crude oil inventory"
# - "Natural gas storage report"
# - "Corn production forecast"

def query_with_cache(query, ttl=3600):
    """Cache query results for common questions."""

    cache_key = generate_cache_key(query)
    cached_result = redis.get(cache_key)

    if cached_result and not is_stale(cached_result):
        return cached_result  # Instant response

    # Cache miss: perform query
    result = kb.query(query)
    redis.setex(cache_key, ttl, result)  # Cache for 1 hour

    return result

# Impact:
# - Latency: 0.1s for cache hits vs 8s for full query → 98% improvement
# - Cost: No LLM/DB costs for cached queries → 60-70% cost reduction
#   (assuming 60-70% query overlap)
```

**Priority 3 - Index Optimization (III):**
```python
# Tune vector database for speed/accuracy tradeoff
# (Example for FAISS)

index_params = {
    "index_type": "IVF_FLAT",      # Faster than brute force
    "nlist": 100,                   # Number of clusters
    "nprobe": 10,                   # Clusters to search
}

# Impact:
# - Latency: 2-3x faster search on large indexes
# - Accuracy: Minimal degradation with proper tuning
# - Complexity: Requires experimentation to tune parameters
```

**Priority 4 - Tiered Storage (V):**
```python
# Move old documents to cheaper, slower storage
def tiered_knowledge_base():
    """Separate hot and cold data."""

    # Tier 1: Recent (3 months) - Fast SSD-backed DB
    hot_kb = ChromaDB(path="/fast/storage")
    hot_kb.add(docs_last_90_days)

    # Tier 2: Archive (3+ months) - Slower/cheaper storage
    cold_kb = ChromaDB(path="/slow/storage")
    cold_kb.add(historical_docs)

    def query(q, recent_only=True):
        results = hot_kb.query(q)

        # If user explicitly needs historical
        if not recent_only:
            historical = cold_kb.query(q)
            results = merge_results(results, historical)

        return results

# Impact:
# - Cost: Storage costs reduced 50-70%
# - Performance: Hot queries remain fast
# - Complexity: Adds infrastructure complexity
```

**Priority 5 - Reduce Chunk Size (IV):**
```python
# LAST PRIORITY - May reduce quality

# Current: 1500 token chunks → 5 chunks = 7500 tokens in context
# Proposed: 750 token chunks → 5 chunks = 3750 tokens in context

# Impact:
# - Cost: 50% reduction in context tokens → 30-40% cost savings
# - Risk: Smaller chunks may lose important context
# - Quality: May need more chunks to maintain coverage

# Only do this if:
# 1. Other optimizations insufficient
# 2. Testing shows quality doesn't degrade
# 3. Cost is truly prohibitive
```

**Key Principle:** Optimize for speed and relevance before optimizing for cost. Users value fast, accurate responses over cheap, slow ones.

---

### Question 11 (10 points)

You're evaluating your RAG system's answer quality. A user asks: "Why did crude oil prices rise last week?"

Your system returns:

> "Crude oil prices rose last week due to inventory drawdowns and geopolitical tensions. Inventories fell by 5.2 million barrels [Source: EIA WPSR, Nov 15], while OPEC+ announced production cuts [Source: Reuters, Nov 12]."

The actual sources say:
- EIA WPSR: "Inventories decreased 5.2 million barrels"
- Reuters article: "OPEC+ members discussed potential production adjustments but made no formal announcements"

What is the PRIMARY quality issue?

A) Missing citation for price movement
B) Hallucination - OPEC+ announced cuts is incorrect
C) Too brief - needs more detail
D) No issue - response is acceptable

**Answer: B**

**Explanation:**

**A - Missing Citation:**
- Secondary issue: The price movement itself should be cited
- But not the PRIMARY problem

**B (Correct) - Hallucination:**
```python
# Critical error breakdown:

System claimed: "OPEC+ announced production cuts"
Source actually said: "discussed potential production adjustments but made no formal announcements"

# This is a HALLUCINATION:
# - Transformed "discussed" → "announced"
# - Added specificity "cuts" not in source
# - Changed uncertainty (potential) to certainty (announced)

# In commodity markets, this is CRITICAL:
# - "Discussed" vs "Announced" has massively different market implications
# - Trading on this misinformation would be costly

# Root cause analysis:
# 1. LLM inferred from discussion to announcement (logical but wrong)
# 2. No verification step caught the discrepancy
# 3. Citation alone doesn't prevent hallucination

# Prevention:
def generate_with_verification(query, chunks):
    """Generate answer and verify against sources."""

    response = llm.generate(query, context=chunks)

    # Extract factual claims
    claims = extract_factual_claims(response)

    # Verify each claim
    for claim in claims:
        source_idx = claim.source - 1
        source_text = chunks[source_idx].text

        # Verify using LLM (better for nuanced claims)
        verification = llm.generate(f"""
Source text: {source_text}

Claim: {claim.text}

Is the claim EXACTLY supported by the source, or is it an inference/exaggeration?
- EXACT: The source explicitly states this
- INFERENCE: The claim goes beyond what the source states
- EXAGGERATION: The claim is stronger than the source

Answer: EXACT/INFERENCE/EXAGGERATION
""")

        if "INFERENCE" in verification or "EXAGGERATION" in verification:
            # Rewrite claim with qualification
            claim_rewrite = llm.generate(f"""
Original claim: {claim.text}
Source: {source_text}
Issue: {verification}

Rewrite the claim to accurately reflect ONLY what the source states, without inference.
""")

            response = response.replace(claim.text, claim_rewrite)

    return response
```

**Corrected response:**
> "Crude oil prices rose last week. Contributing factors may include inventory drawdowns and OPEC+ discussions. Inventories fell by 5.2 million barrels [Source: EIA WPSR, Nov 15], and OPEC+ members discussed potential production adjustments but made no formal announcements [Source: Reuters, Nov 12]."

**C - Too Brief:**
- Reasonable critique but not a quality *issue*
- Briefness is often desirable for trader dashboards

**D - No Issue:**
- False: The OPEC+ hallucination is a critical error

**Key Takeaway:** Citations don't prevent hallucinations. You must verify that cited claims accurately reflect source content, not just that sources exist.

---

## Answer Key Summary

| Question | Answer | Points | Topic |
|----------|--------|--------|-------|
| 1 | C | 10 | Knowledge base architecture |
| 2 | B | 8 | Critical metadata fields |
| 3 | B | 7 | Chunking granularity |
| 4 | C | 12 | Hybrid retrieval strategy |
| 5 | C | 10 | Multi-factor re-ranking |
| 6 | B | 8 | Metadata-driven queries |
| 7 | C | 10 | Grounded generation prompts |
| 8 | B | 8 | Transparent sourcing |
| 9 | D | 7 | Citation verification |
| 10 | C | 10 | Production optimization priorities |
| 11 | B | 10 | Hallucination detection |

**Total:** 100 points

---

## Grading Scale

- **90-100:** Excellent - RAG system expertise
- **80-89:** Good - Solid understanding, minor gaps
- **70-79:** Adequate - Review retrieval and verification strategies
- **Below 70:** Needs improvement - Revisit core RAG concepts

---

## Key Takeaways

1. **Metadata is Critical:** Rich metadata enables precise filtering and ranking
2. **Hybrid Approaches Win:** Combine structured data, semantic search, and metadata filtering
3. **Recency Matters:** Commodity markets are time-sensitive; prioritize recent data
4. **Verify Everything:** Citations alone don't prevent hallucinations
5. **Optimize Strategically:** Speed and relevance before cost

---

## Next Steps

**Score 90-100:** Proceed to Module 3 (Sentiment Analysis)
**Score 80-89:** Review re-ranking and verification techniques
**Score 70-79:** Practice implementing RAG pipelines with proper grounding
**Score <70:** Revisit Module 2 materials, focus on retrieval strategies
