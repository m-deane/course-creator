# Retrieval Strategies for Commodity Analysis

> **Reading time:** ~13 min | **Module:** Module 2: Rag Research | **Prerequisites:** Modules 0-1

<div class="callout-key">

**Key Concept Summary:** Retrieval strategies for commodity analysis combine semantic search with temporal, geographical, and commodity-specific filters to ensure LLMs receive contextually appropriate information—preventing the mixing of outdated data, wrong geographies, or unrelated commodities that would lead to halluc...

</div>

## In Brief

Retrieval strategies for commodity analysis combine semantic search with temporal, geographical, and commodity-specific filters to ensure LLMs receive contextually appropriate information—preventing the mixing of outdated data, wrong geographies, or unrelated commodities that would lead to hallucinated analysis.

<div class="callout-insight">

**Insight:** Standard RAG retrieval fails for commodities because semantic similarity alone can't distinguish between "crude oil storage was high in January 2024" and "crude oil storage was high in January 2020"—both are semantically identical but lead to opposite trading conclusions. Effective commodity retrieval requires **multi-dimensional filtering** that respects time, space, and domain boundaries.

</div>
<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Intuitive Explanation

Think of retrieval like asking a specialized librarian for commodity research:

**Bad librarian (semantic search only):**
- You: "What's the crude oil inventory situation?"
- Returns: Mix of 2019 and 2024 data, some about U.S., some about OPEC, plus unrelated gasoline info
- Result: Confusion and wrong conclusions

**Good librarian (multi-dimensional retrieval):**
- You: "What's the crude oil inventory situation?"
- Librarian asks: "Which geography? What time period?"
- You: "U.S., most recent data"
- Returns: Only recent U.S. crude inventory reports, properly dated and contextualized
- Result: Accurate, actionable insights

The challenge is that commodity questions have implicit context ("current" means this week, "storage" implies specific geography, "crude" excludes products) that must be made explicit in retrieval.

## Formal Definition

A commodity retrieval strategy is a function **R: (Q, KB, F) → C** where:

**Inputs:**
- **Q** = query (user question or analysis task)
- **KB** = knowledge base of commodity documents
- **F** = filter constraints {time_range, commodity_type, geography, section_type}

**Output:**
- **C** = ordered list of chunks {c₁, c₂, ..., cₖ} ranked by relevance score

**Relevance score:** S(q, c) = α·sim(q, c) + β·temporal_score(c) + γ·specificity_score(c)
- **sim(q, c)**: semantic similarity (cosine distance in embedding space)
- **temporal_score(c)**: recency weighting (more recent = higher score)
- **specificity_score(c)**: exact commodity/geography match bonus

**Constraints:**
- Temporal coherence: max(date(c₁)) - min(date(cₖ)) ≤ acceptable_time_window
- Commodity consistency: all chunks reference same commodity unless explicitly multi-commodity query
- Geography consistency: maintain regional specificity (don't mix U.S. and China data)

## Code Implementation

### Multi-Dimensional Retrieval System


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">from.py</span>
</div>

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from enum import Enum
import numpy as np
from anthropic import Anthropic
import chromadb

class QueryType(Enum):
    CURRENT_STATE = "current_state"  # "What is inventory now?"
    HISTORICAL_COMPARISON = "historical_comparison"  # "Compare to year ago"
    TREND_ANALYSIS = "trend_analysis"  # "What's the trend?"
    FORECAST = "forecast"  # "What's expected?"
    CROSS_COMMODITY = "cross_commodity"  # "How does X affect Y?"

@dataclass
class QueryIntent:
    """Parsed query with extracted intent and filters."""
    original_query: str
    query_type: QueryType
    commodity: str
    geography: Optional[str]
    time_reference: str  # "current", "last_month", "year_ago", "2024-Q1"
    specific_metrics: List[str]  # ["inventory", "production", "demand"]
    comparison_needed: bool

@dataclass
class RetrievalContext:
    """Context for retrieval strategy."""
    query_intent: QueryIntent
    current_date: datetime
    acceptable_time_window: timedelta
    min_chunks: int
    max_chunks: int

class CommodityRetrievalStrategy:
    """
    Advanced retrieval strategies for commodity knowledge bases.
    """

    def __init__(self, knowledge_base, anthropic_api_key: str):
        self.kb = knowledge_base
        self.anthropic_client = Anthropic(api_key=anthropic_api_key)

    def retrieve(
        self,
        query: str,
        strategy: str = "auto",
        top_k: int = 5,
        **kwargs
    ) -> List[Dict]:
        """
        Retrieve chunks using specified strategy.

        Args:
            query: User question
            strategy: "auto" | "current_state" | "historical" | "trend" | "multi_commodity"
            top_k: Maximum chunks to return
            **kwargs: Additional filters (commodity, date_range, etc.)

        Returns:
            List of retrieved chunks with metadata and scores
        """
        # Step 1: Parse query intent
        intent = self._parse_query_intent(query)

        # Step 2: Select strategy
        if strategy == "auto":
            strategy = intent.query_type.value

        # Step 3: Build retrieval context
        context = self._build_retrieval_context(intent, top_k, **kwargs)

        # Step 4: Execute strategy-specific retrieval
        if strategy == "current_state":
            return self._retrieve_current_state(context)
        elif strategy == "historical_comparison":
            return self._retrieve_historical_comparison(context)
        elif strategy == "trend_analysis":
            return self._retrieve_trend_analysis(context)
        elif strategy == "forecast":
            return self._retrieve_forecast(context)
        elif strategy == "cross_commodity":
            return self._retrieve_cross_commodity(context)
        else:
            # Fallback to semantic search with filters
            return self._retrieve_semantic(context)

    def _parse_query_intent(self, query: str) -> QueryIntent:
        """
        Use LLM to parse query and extract intent.
        """
        prompt = f"""Parse this commodity market query and extract structured intent.

Query: "{query}"

Return JSON with this structure:
{{
  "query_type": "current_state | historical_comparison | trend_analysis | forecast | cross_commodity",
  "commodity": "crude_oil | natural_gas | gasoline | corn | wheat | etc.",
  "geography": "US | PADD1 | Europe | Global | null if not specified",
  "time_reference": "current | last_week | last_month | year_ago | specific_date",
  "specific_metrics": ["inventory", "production", "demand", "price"],
  "comparison_needed": true/false
}}

Examples:
- "What's the current crude inventory?" → current_state, crude_oil, US, current
- "How does this week's storage compare to last year?" → historical_comparison, needs year_ago data
- "What's the trend in natural gas production?" → trend_analysis, natural_gas, multi-week data"""

        response = self.anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        intent_data = json.loads(response.content[0].text)

        return QueryIntent(
            original_query=query,
            query_type=QueryType(intent_data["query_type"]),
            commodity=intent_data["commodity"],
            geography=intent_data.get("geography"),
            time_reference=intent_data["time_reference"],
            specific_metrics=intent_data["specific_metrics"],
            comparison_needed=intent_data["comparison_needed"]
        )

    def _build_retrieval_context(
        self,
        intent: QueryIntent,
        top_k: int,
        **kwargs
    ) -> RetrievalContext:
        """
        Build retrieval context from intent and parameters.
        """
        # Determine acceptable time window based on query type
        if intent.query_type == QueryType.CURRENT_STATE:
            time_window = timedelta(days=7)  # Last week only
        elif intent.query_type == QueryType.TREND_ANALYSIS:
            time_window = timedelta(days=180)  # 6 months
        elif intent.query_type == QueryType.HISTORICAL_COMPARISON:
            time_window = timedelta(days=365)  # Up to 1 year
        else:
            time_window = timedelta(days=30)

        return RetrievalContext(
            query_intent=intent,
            current_date=kwargs.get("current_date", datetime.now()),
            acceptable_time_window=time_window,
            min_chunks=kwargs.get("min_chunks", 3),
            max_chunks=top_k
        )

    def _retrieve_current_state(self, context: RetrievalContext) -> List[Dict]:
        """
        Retrieve most recent data for current state queries.

        Strategy:
        1. Strict recency filter (last 7 days)
        2. Exact commodity match
        3. Prefer "inventory" or "production" sections
        4. Boost chunks with data tables
        """
        intent = context.query_intent

        # Build strict filters
        end_date = context.current_date
        start_date = end_date - timedelta(days=7)

        where_filter = {
            "commodity": intent.commodity,
            "report_date": {
                "$gte": start_date.isoformat(),
                "$lte": end_date.isoformat()
            }
        }

        if intent.geography:
            where_filter["geography"] = intent.geography

        # Semantic search with filters
        results = self.kb.collection.query(
            query_texts=[intent.original_query],
            n_results=context.max_chunks,
            where=where_filter
        )

        # Re-rank results
        chunks = self._rerank_for_current_state(results, intent)

        return chunks[:context.max_chunks]

    def _retrieve_historical_comparison(self, context: RetrievalContext) -> List[Dict]:
        """
        Retrieve data for historical comparison.

        Strategy:
        1. Get current period data (e.g., this week)
        2. Get comparison period data (e.g., year ago same week)
        3. Ensure seasonal alignment (winter to winter)
        4. Return balanced set (50% current, 50% historical)
        """
        intent = context.query_intent

        # Determine comparison period from query
        if "year ago" in intent.original_query.lower():
            comparison_offset = timedelta(days=365)
        elif "month ago" in intent.original_query.lower():
            comparison_offset = timedelta(days=30)
        else:
            comparison_offset = timedelta(days=365)  # Default to YoY

        # Retrieve current period
        current_end = context.current_date
        current_start = current_end - timedelta(days=7)

        current_chunks = self._retrieve_time_period(
            intent,
            current_start,
            current_end,
            max_chunks=context.max_chunks // 2
        )

        # Retrieve comparison period (same time window, offset back)
        comparison_end = current_end - comparison_offset
        comparison_start = current_start - comparison_offset

        comparison_chunks = self._retrieve_time_period(
            intent,
            comparison_start,
            comparison_end,
            max_chunks=context.max_chunks // 2
        )

        # Combine and label
        results = []
        for chunk in current_chunks:
            chunk["period"] = "current"
            results.append(chunk)

        for chunk in comparison_chunks:
            chunk["period"] = "comparison"
            results.append(chunk)

        return results

    def _retrieve_trend_analysis(self, context: RetrievalContext) -> List[Dict]:
        """
        Retrieve data showing trends over time.

        Strategy:
        1. Get evenly spaced samples over time window (weekly or monthly)
        2. Prioritize chunks with numerical data
        3. Ensure temporal ordering
        4. Include enough points to show trend (minimum 8-10 weeks)
        """
        intent = context.query_intent

        # Determine sampling frequency
        if "weekly" in intent.original_query.lower() or intent.commodity in ["crude_oil", "gasoline"]:
            sample_interval = timedelta(days=7)
            num_samples = 12  # 3 months weekly
        else:
            sample_interval = timedelta(days=30)
            num_samples = 6  # 6 months monthly

        # Retrieve samples going back in time
        chunks = []
        current_date = context.current_date

        for i in range(num_samples):
            sample_end = current_date - (i * sample_interval)
            sample_start = sample_end - timedelta(days=3)  # Small window around sample point

            sample_chunks = self._retrieve_time_period(
                intent,
                sample_start,
                sample_end,
                max_chunks=1  # One chunk per sample point
            )

            if sample_chunks:
                chunks.extend(sample_chunks)

        # Sort by date (oldest to newest for trend display)
        chunks.sort(key=lambda x: x["metadata"]["report_date"])

        return chunks

    def _retrieve_forecast(self, context: RetrievalContext) -> List[Dict]:
        """
        Retrieve forecast and outlook data.

        Strategy:
        1. Look for report types that contain forecasts (STEO, WASDE)
        2. Get most recent forecasts
        3. Include historical actuals for context
        4. Prioritize "forecast" or "outlook" sections
        """
        intent = context.query_intent

        where_filter = {
            "commodity": intent.commodity,
            "section_type": {"$in": ["forecast", "outlook", "projection"]}
        }

        if intent.geography:
            where_filter["geography"] = intent.geography

        # Get forecast chunks
        results = self.kb.collection.query(
            query_texts=[intent.original_query],
            n_results=context.max_chunks,
            where=where_filter
        )

        chunks = self._format_results(results)

        # Add recent actuals for context (20% of chunks)
        num_actuals = max(1, context.max_chunks // 5)
        actuals = self._retrieve_current_state(context)[:num_actuals]

        for actual in actuals:
            actual["chunk_type"] = "actual"

        for chunk in chunks:
            chunk["chunk_type"] = "forecast"

        return chunks + actuals

    def _retrieve_cross_commodity(self, context: RetrievalContext) -> List[Dict]:
        """
        Retrieve data for cross-commodity analysis.

        Strategy:
        1. Identify both commodities from query
        2. Retrieve relevant data for each
        3. Focus on related metrics (e.g., crude production + gasoline demand)
        4. Ensure temporal alignment
        """
        intent = context.query_intent

        # Parse multiple commodities from query
        commodities = self._extract_commodities(intent.original_query)

        if len(commodities) < 2:
            # Fallback to single commodity
            return self._retrieve_current_state(context)

        # Retrieve for each commodity
        chunks_per_commodity = context.max_chunks // len(commodities)

        all_chunks = []
        for commodity in commodities:
            # Create intent for this commodity
            commodity_intent = QueryIntent(
                original_query=intent.original_query,
                query_type=QueryType.CURRENT_STATE,
                commodity=commodity,
                geography=intent.geography,
                time_reference="current",
                specific_metrics=intent.specific_metrics,
                comparison_needed=False
            )

            commodity_context = RetrievalContext(
                query_intent=commodity_intent,
                current_date=context.current_date,
                acceptable_time_window=timedelta(days=30),
                min_chunks=1,
                max_chunks=chunks_per_commodity
            )

            chunks = self._retrieve_current_state(commodity_context)
            for chunk in chunks:
                chunk["related_commodity"] = commodity
            all_chunks.extend(chunks)

        return all_chunks

    def _retrieve_time_period(
        self,
        intent: QueryIntent,
        start_date: datetime,
        end_date: datetime,
        max_chunks: int
    ) -> List[Dict]:
        """
        Retrieve chunks for specific time period.
        """
        where_filter = {
            "commodity": intent.commodity,
            "report_date": {
                "$gte": start_date.isoformat(),
                "$lte": end_date.isoformat()
            }
        }

        if intent.geography:
            where_filter["geography"] = intent.geography

        results = self.kb.collection.query(
            query_texts=[intent.original_query],
            n_results=max_chunks,
            where=where_filter
        )

        return self._format_results(results)

    def _retrieve_semantic(self, context: RetrievalContext) -> List[Dict]:
        """
        Fallback to basic semantic search with commodity filter.
        """
        intent = context.query_intent

        where_filter = {"commodity": intent.commodity}
        if intent.geography:
            where_filter["geography"] = intent.geography

        results = self.kb.collection.query(
            query_texts=[intent.original_query],
            n_results=context.max_chunks,
            where=where_filter
        )

        return self._format_results(results)

    def _rerank_for_current_state(self, results: Dict, intent: QueryIntent) -> List[Dict]:
        """
        Re-rank results for current state queries.

        Boosting factors:
        - Contains data tables: +0.2
        - Exact metric match: +0.3
        - More recent: +0.1 per day
        """
        chunks = self._format_results(results)

        for chunk in chunks:
            boost = 0.0

            # Data table boost
            if chunk["metadata"].get("contains_table"):
                boost += 0.2

            # Metric match boost
            for metric in intent.specific_metrics:
                if metric in chunk["text"].lower():
                    boost += 0.3
                    break

            # Recency boost (newer is better)
            report_date = datetime.fromisoformat(chunk["metadata"]["report_date"])
            days_old = (datetime.now() - report_date).days
            recency_boost = max(0, 0.5 - (days_old * 0.01))
            boost += recency_boost

            chunk["boosted_score"] = chunk["similarity_score"] + boost

        # Sort by boosted score
        chunks.sort(key=lambda x: x["boosted_score"], reverse=True)

        return chunks

    def _format_results(self, results: Dict) -> List[Dict]:
        """
        Format ChromaDB results into standard chunk format.
        """
        chunks = []
        for i, doc in enumerate(results['documents'][0]):
            chunk = {
                "text": doc,
                "metadata": results['metadatas'][0][i],
                "chunk_id": results['ids'][0][i],
                "similarity_score": 1.0 - results['distances'][0][i]  # Convert distance to similarity
            }
            chunks.append(chunk)

        return chunks

    def _extract_commodities(self, query: str) -> List[str]:
        """
        Extract multiple commodities from query.
        """
        # Simple keyword matching (in production, use LLM)
        commodities = []
        commodity_keywords = {
            "crude_oil": ["crude", "oil", "wti", "brent"],
            "natural_gas": ["natural gas", "nat gas", "gas"],
            "gasoline": ["gasoline", "gas", "rbob"],
            "distillates": ["distillate", "diesel", "heating oil"],
            "corn": ["corn"],
            "wheat": ["wheat"],
            "soybeans": ["soybean", "soy"]
        }

        query_lower = query.lower()
        for commodity, keywords in commodity_keywords.items():
            if any(kw in query_lower for kw in keywords):
                commodities.append(commodity)

        return commodities
```

</div>
</div>

### Hybrid Retrieval with Re-ranking


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">hybridretrieval.py</span>
</div>

```python
class HybridRetrieval:
    """
    Combine multiple retrieval methods and re-rank.
    """

    def __init__(self, kb, anthropic_client):
        self.kb = kb
        self.anthropic_client = anthropic_client

    def retrieve_hybrid(
        self,
        query: str,
        commodity: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Hybrid retrieval combining:
        1. Semantic search (embeddings)
        2. Keyword/BM25 search
        3. Metadata filtering
        4. LLM-based re-ranking
        """
        # Method 1: Semantic search
        semantic_results = self._semantic_search(query, commodity, top_k * 2)

        # Method 2: Keyword search (simulate BM25)
        keyword_results = self._keyword_search(query, commodity, top_k * 2)

        # Merge results (deduplicate by chunk_id)
        merged = self._merge_results(semantic_results, keyword_results)

        # Re-rank using LLM
        reranked = self._llm_rerank(query, merged, top_k)

        return reranked

    def _llm_rerank(self, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
        """
        Use LLM to re-rank retrieved chunks for relevance.
        """
        # Prepare chunks for LLM
        chunk_texts = []
        for i, chunk in enumerate(chunks[:20]):  # Limit to 20 for LLM context
            chunk_texts.append(f"[{i}] {chunk['text'][:300]}...")

        prompt = f"""Rank these commodity report chunks by relevance to the query.

Query: "{query}"

Chunks:
{chr(10).join(chunk_texts)}

Return JSON array of chunk indices in order of relevance (most relevant first):
{{"ranked_indices": [3, 7, 1, ...]}}"""

        response = self.anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        ranking = json.loads(response.content[0].text)

        # Reorder chunks
        reranked = [chunks[i] for i in ranking["ranked_indices"][:top_k]]

        return reranked
```

</div>
</div>

## Common Pitfalls

**1. Ignoring Temporal Context in Retrieval**
- **Problem**: Retrieving mix of old and new data leads to contradictory analysis
- **Why it happens**: Semantic similarity doesn't respect time boundaries
- **Solution**: Always include date range filters; use recency boosting

**2. Over-Reliance on Semantic Similarity**
- **Problem**: "High crude oil storage" matches "Low crude oil storage" semantically
- **Why it happens**: Embeddings capture topic similarity, not directional differences
- **Solution**: Combine semantic with keyword matching; use LLM re-ranking

**3. Insufficient Chunks for Trend Analysis**
- **Problem**: Returning only 3-5 data points for "show me the trend"
- **Why it happens**: Default top_k is too small for time-series queries
- **Solution**: Dynamically adjust top_k based on query type (trends need 10-15 points)

**4. Geography Mixing**
- **Problem**: Combining U.S. PADD regions or mixing countries
- **Why it happens**: No geography hierarchy in metadata
- **Solution**: Implement geography taxonomy (US > PADD1 > Northeast); enforce consistency

**5. Missing Cross-References**
- **Problem**: Crude oil query misses related refinery utilization data
- **Why it happens**: No relationship modeling between related metrics
- **Solution**: Build commodity relationship graph; expand queries to related metrics

## Connections

**Builds on:**
- Knowledge base design (proper chunking and metadata)
- Vector databases (similarity search fundamentals)
- LLM query understanding (intent parsing)

**Leads to:**
- Multi-document synthesis (combining retrieved chunks)
- Answer generation (using retrieved context in prompts)
- Confidence scoring (measuring retrieval quality)

**Related to:**
- Information retrieval theory (BM25, TF-IDF, embeddings)
- Temporal databases (time-aware queries)
- Graph databases (relationship modeling)

## Practice Problems

1. **Intent-Aware Retrieval**
   - Implement query intent classification for these queries:
     - "What's the crude oil inventory?"
     - "How has natural gas storage trended over the past quarter?"
     - "Compare current corn production to last year's harvest"
   - For each, design the appropriate retrieval strategy

2. **Seasonal Alignment Challenge**
   - User asks: "How does this winter's natural gas storage compare to last winter?"
   - Current date: January 15, 2025
   - Design retrieval that:
     - Gets winter 2024-25 data (Dec-Jan)
     - Gets winter 2023-24 data (same months)
     - Excludes summer data
   - What date ranges would you use?

3. **Multi-Commodity Retrieval**
   - Query: "How do crude oil fundamentals affect gasoline prices?"
   - This requires data on:
     - Crude oil production and inventory
     - Refinery utilization (links the two)
     - Gasoline demand and inventory
   - Design a retrieval strategy that gets all necessary context

4. **Forecast vs Actual Retrieval**
   - User wants to evaluate forecast accuracy
   - Query: "How accurate was last month's production forecast?"
   - You need to retrieve:
     - Forecast from 1 month ago
     - Actual data from most recent report
   - How would you structure this retrieval?

5. **Hybrid Re-Ranking System**
   - Implement a re-ranking function that combines:
     - Semantic similarity score (0-1)
     - Recency score (0-1, exponential decay)
     - Data table presence (boolean)
     - Geography exactness (exact match > region match > no match)
   - Design the weighting formula

<div class="callout-insight">

**Insight:** Understanding retrieval strategies for commodity analysis is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.




## Further Reading

**Information Retrieval:**
- "Introduction to Information Retrieval" (Manning et al.) - Chapter 6: Scoring, term weighting, and the vector space model
- "Dense Passage Retrieval for Open-Domain Question Answering" - Semantic search foundations

**Hybrid Search:**
- Pinecone: "Hybrid Search Explained" - Combining dense and sparse retrieval
- Elasticsearch: "Text Similarity with Vector Search" - Production hybrid systems

**LLM-Augmented Retrieval:**
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" - Original RAG paper
- "Lost in the Middle: How Language Models Use Long Contexts" - Position bias in retrieval

**Re-Ranking:**
- "RankGPT: Listwise Passage Re-ranking with LLMs" - Using LLMs for re-ranking
- "Cross-Encoder vs Bi-Encoder for Semantic Search" - Re-ranking architectures

**Temporal Retrieval:**
- "Time-Aware Recommender Systems" - Handling temporal dynamics
- "Temporal Information Retrieval" - Academic survey

**Production RAG:**
- LangChain: "Advanced Retrieval Strategies" - Practical implementation patterns
- "Building RAG Systems at Scale" (LlamaIndex) - Performance optimization

---

## Conceptual Practice Questions

1. Compare dense retrieval vs. hybrid retrieval for commodity research queries.

2. How do you evaluate whether your retrieval strategy is returning relevant documents?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---



## Cross-References

<a class="link-card" href="./03_retrieval_strategies_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_eia_knowledge_base.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_knowledge_base_design.md">
  <div class="link-card-title">01 Knowledge Base Design</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_document_processing.md">
  <div class="link-card-title">02 Document Processing</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

