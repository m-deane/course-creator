# Knowledge Base Design for Commodity Research

> **Reading time:** ~9 min | **Module:** Module 2: Rag Research | **Prerequisites:** Modules 0-1

<div class="callout-key">

**Key Concept Summary:** Knowledge base design for commodity markets involves structuring unstructured commodity data (EIA reports, weather forecasts, corporate filings) into retrievable chunks optimized for LLM context windows while preserving temporal, geographical, and commodity-specific relationships.

</div>

## In Brief

Knowledge base design for commodity markets involves structuring unstructured commodity data (EIA reports, weather forecasts, corporate filings) into retrievable chunks optimized for LLM context windows while preserving temporal, geographical, and commodity-specific relationships.

<div class="callout-insight">

**Insight:** Commodity data requires time-aware chunking strategies because the same concept (e.g., "crude oil inventory") has drastically different interpretations based on date, location, and market context. A knowledge base that loses temporal ordering produces hallucinated or outdated analysis.

</div>
<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Intuitive Explanation

Think of a commodity knowledge base like a specialized library where:
- Books are organized not just by topic, but by publication date (critical for commodities)
- Each book has tags for geography (U.S. vs. OPEC), commodity type (crude vs. products), and season
- When you ask "What's the storage situation?", the librarian retrieves only recent books for your specific commodity and region
- Old books are kept but clearly marked to prevent confusion

The challenge is that commodity reports mix narrative analysis with data tables, have strong temporal dependencies (winter vs. summer natural gas), and contain nested concepts (crude oil affects gasoline affects consumer spending).

## Formal Definition

A commodity knowledge base is a tuple **KB = (D, C, E, M)** where:
- **D** = set of source documents (reports, transcripts, news articles)
- **C** = chunking strategy that maps D → text segments with metadata
- **E** = embedding function mapping chunks → vector space
- **M** = metadata schema capturing {commodity, date, location, report_type, seasonality}

**Retrieval function**: R(query, k) → top-k chunks by semantic similarity + metadata filters

**Quality metric**: Retrieved chunks must be temporally coherent (no mixing Q1 2023 and Q3 2024 data in same context) and commodity-specific (crude oil chunks shouldn't retrieve natural gas unless explicitly querying correlations).

## Code Implementation

### Basic Knowledge Base Structure


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">from.py</span>
</div>

```python
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
from enum import Enum
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from anthropic import Anthropic

class CommodityType(Enum):
    CRUDE_OIL = "crude_oil"
    NATURAL_GAS = "natural_gas"
    GASOLINE = "gasoline"
    DISTILLATES = "distillates"
    CORN = "corn"
    WHEAT = "wheat"
    SOYBEANS = "soybeans"
    GOLD = "gold"
    COPPER = "copper"

class ReportType(Enum):
    EIA_WPSR = "eia_wpsr"
    EIA_NGSR = "eia_natural_gas_storage"
    USDA_WASDE = "usda_wasde"
    USDA_CROP_PROGRESS = "usda_crop_progress"
    EARNINGS_CALL = "earnings_call"
    NEWS_ARTICLE = "news_article"

@dataclass
class CommodityDocument:
    """Source document with commodity-specific metadata."""
    doc_id: str
    content: str
    commodity: CommodityType
    report_type: ReportType
    report_date: datetime
    geography: str  # "US", "PADD1", "GLOBAL"
    source_url: Optional[str]

@dataclass
class CommodityChunk:
    """Text chunk with rich metadata for retrieval."""
    chunk_id: str
    text: str
    commodity: CommodityType
    report_type: ReportType
    report_date: datetime
    geography: str
    section_type: str  # "inventory", "production", "demand", "forecast"
    contains_data_table: bool
    seasonal_period: str  # "winter", "summer", "planting", "harvest"

class CommodityKnowledgeBase:
    """
    Knowledge base optimized for commodity research.

    Design principles:
    1. Temporal coherence - preserve time relationships
    2. Commodity separation - avoid cross-contamination
    3. Geography awareness - regional variations matter
    4. Data vs narrative separation - different chunking for tables
    """

    def __init__(self, collection_name: str = "commodity_kb"):
        # Initialize vector store
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"description": "Commodity market intelligence"}
        )

        # Initialize LLM for embeddings and analysis
        self.anthropic_client = Anthropic()

        # Chunking strategy for different content types
        self.narrative_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " "]
        )

        self.table_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Tables need more context
            chunk_overlap=100,
            separators=["\n\n", "\n"]
        )

    def add_document(self, doc: CommodityDocument):
        """
        Add document to knowledge base with intelligent chunking.
        """
        # Detect if document contains data tables
        has_tables = self._detect_tables(doc.content)

        # Separate narrative from tables
        if has_tables:
            narrative, tables = self._separate_tables(doc.content)
            chunks = self._chunk_narrative(narrative, doc)
            chunks.extend(self._chunk_tables(tables, doc))
        else:
            chunks = self._chunk_narrative(doc.content, doc)

        # Add chunks to vector store
        for chunk in chunks:
            self._add_chunk(chunk)

    def _detect_tables(self, text: str) -> bool:
        """
        Detect if text contains data tables using LLM.
        """
        prompt = f"""Does this text contain data tables with numerical values?
Reply with only "yes" or "no".

Text:
{text[:500]}..."""

        response = self.anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}]
        )

        return "yes" in response.content[0].text.lower()

    def _separate_tables(self, text: str) -> tuple[str, List[str]]:
        """
        Separate narrative text from data tables using LLM.
        """
        prompt = f"""Separate this commodity report into:
1. Narrative analysis (prose explanations)
2. Data tables (structured numerical data)

Return as JSON:
{{
  "narrative": "...",
  "tables": ["table1", "table2", ...]
}}

Report:
{text}"""

        response = self.anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        result = json.loads(response.content[0].text)
        return result["narrative"], result["tables"]

    def _chunk_narrative(
        self,
        text: str,
        doc: CommodityDocument
    ) -> List[CommodityChunk]:
        """
        Chunk narrative text with context preservation.
        """
        splits = self.narrative_splitter.split_text(text)

        chunks = []
        for i, split_text in enumerate(splits):
            # Classify section type using LLM
            section_type = self._classify_section(split_text)

            chunk = CommodityChunk(
                chunk_id=f"{doc.doc_id}_narrative_{i}",
                text=split_text,
                commodity=doc.commodity,
                report_type=doc.report_type,
                report_date=doc.report_date,
                geography=doc.geography,
                section_type=section_type,
                contains_data_table=False,
                seasonal_period=self._infer_season(doc.report_date)
            )
            chunks.append(chunk)

        return chunks

    def _chunk_tables(
        self,
        tables: List[str],
        doc: CommodityDocument
    ) -> List[CommodityChunk]:
        """
        Chunk data tables with special handling.
        """
        chunks = []
        for i, table in enumerate(tables):
            # Keep tables mostly intact - they need context
            chunk = CommodityChunk(
                chunk_id=f"{doc.doc_id}_table_{i}",
                text=table,
                commodity=doc.commodity,
                report_type=doc.report_type,
                report_date=doc.report_date,
                geography=doc.geography,
                section_type="data_table",
                contains_data_table=True,
                seasonal_period=self._infer_season(doc.report_date)
            )
            chunks.append(chunk)

        return chunks

    def _classify_section(self, text: str) -> str:
        """
        Classify section type for targeted retrieval.
        """
        prompt = f"""What is the primary focus of this commodity text?
Choose ONE: inventory, production, demand, forecast, price_analysis, geopolitical

Text: {text[:300]}..."""

        response = self.anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip()

    def _infer_season(self, report_date: datetime) -> str:
        """
        Infer seasonal period from date (critical for commodities).
        """
        month = report_date.month

        # Northern hemisphere seasons (adjust for global commodities)
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"

    def _add_chunk(self, chunk: CommodityChunk):
        """
        Add chunk to vector store with metadata.
        """
        # Create metadata for filtering
        metadata = {
            "commodity": chunk.commodity.value,
            "report_type": chunk.report_type.value,
            "report_date": chunk.report_date.isoformat(),
            "geography": chunk.geography,
            "section_type": chunk.section_type,
            "contains_table": chunk.contains_data_table,
            "season": chunk.seasonal_period,
            "year": chunk.report_date.year,
            "month": chunk.report_date.month
        }

        self.collection.add(
            documents=[chunk.text],
            metadatas=[metadata],
            ids=[chunk.chunk_id]
        )

    def retrieve(
        self,
        query: str,
        commodity: Optional[CommodityType] = None,
        date_range: Optional[tuple[datetime, datetime]] = None,
        geography: Optional[str] = None,
        top_k: int = 5
    ) -> List[CommodityChunk]:
        """
        Retrieve relevant chunks with metadata filtering.
        """
        # Build metadata filters
        where = {}
        if commodity:
            where["commodity"] = commodity.value
        if geography:
            where["geography"] = geography
        if date_range:
            # ChromaDB date filtering
            where["report_date"] = {
                "$gte": date_range[0].isoformat(),
                "$lte": date_range[1].isoformat()
            }

        # Query vector store
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where if where else None
        )

        # Reconstruct chunks from results
        chunks = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            chunk = CommodityChunk(
                chunk_id=results['ids'][0][i],
                text=doc,
                commodity=CommodityType(metadata['commodity']),
                report_type=ReportType(metadata['report_type']),
                report_date=datetime.fromisoformat(metadata['report_date']),
                geography=metadata['geography'],
                section_type=metadata['section_type'],
                contains_data_table=metadata['contains_table'],
                seasonal_period=metadata['season']
            )
            chunks.append(chunk)

        return chunks
```

</div>
</div>

### Example Usage


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from datetime import datetime, timedelta

# Initialize knowledge base
kb = CommodityKnowledgeBase()

# Add EIA weekly petroleum report
wpsr_doc = CommodityDocument(
    doc_id="wpsr_2024_11_13",
    content="""
    U.S. commercial crude oil inventories decreased by 5.2 million barrels
    from the previous week. At 430.0 million barrels, U.S. crude oil
    inventories are about 3% below the five year average...

    [TABLE: Weekly Petroleum Stocks]
    Product                Current    Prior Week    Year Ago
    Crude Oil (MMB)        430.0      435.2         445.3
    Gasoline (MMB)         215.4      213.3         220.1
    """,
    commodity=CommodityType.CRUDE_OIL,
    report_type=ReportType.EIA_WPSR,
    report_date=datetime(2024, 11, 13),
    geography="US",
    source_url="https://www.eia.gov/petroleum/supply/weekly/"
)

kb.add_document(wpsr_doc)

# Retrieve with temporal and commodity filtering
results = kb.retrieve(
    query="What is the current crude oil inventory situation?",
    commodity=CommodityType.CRUDE_OIL,
    date_range=(datetime(2024, 11, 1), datetime(2024, 11, 30)),
    geography="US",
    top_k=3
)

for chunk in results:
    print(f"Date: {chunk.report_date}")
    print(f"Section: {chunk.section_type}")
    print(f"Text: {chunk.text[:200]}...")
    print("---")
```

</div>
</div>

## Common Pitfalls

**1. Ignoring Temporal Context**
- **Problem**: Mixing data from different time periods in retrieval
- **Why it happens**: Standard RAG systems don't enforce temporal coherence
- **Solution**: Always include date_range filters in queries; use recency weighting

**2. Chunk Size Mismatch for Tables**
- **Problem**: Splitting data tables across chunks loses context
- **Why it happens**: Using same chunk size for narrative and structured data
- **Solution**: Detect tables and use larger chunk sizes (2000+ tokens)

**3. Cross-Commodity Contamination**
- **Problem**: Crude oil query returns natural gas results
- **Why it happens**: Semantic similarity without commodity filtering
- **Solution**: Mandatory commodity metadata in all chunks; filter on retrieval

**4. Losing Seasonal Context**
- **Problem**: Comparing summer natural gas demand to winter demand
- **Why it happens**: No seasonal metadata in chunks
- **Solution**: Tag all chunks with seasonal period; account for seasonality in analysis

**5. Geography Ambiguity**
- **Problem**: Mixing U.S. PADD regions or global vs. regional data
- **Why it happens**: Commodity reports reference multiple geographies
- **Solution**: Extract geography during chunking; use hierarchical geography tags (US > PADD1 > NY)

## Connections

**Builds on:**
- LLM fundamentals (prompt engineering for classification)
- Vector databases (ChromaDB, Pinecone basics)
- Commodity market structure (understanding report types)

**Leads to:**
- Retrieval strategies (how to query the knowledge base effectively)
- Multi-document synthesis (combining chunks into coherent analysis)
- Real-time knowledge base updates (handling new reports)

**Related to:**
- Document processing pipelines (ETL for commodity data)
- Metadata extraction (automated tagging of documents)
- Temporal reasoning (handling time-series relationships)

## Practice Problems

1. **Chunking Strategy Design**
   - Given a USDA WASDE report (20 pages, narrative + 15 data tables), design a chunking strategy that:
     - Preserves table integrity
     - Maintains crop-specific sections
     - Captures forecast vs. historical data distinctions
   - What chunk sizes would you use? What metadata would you extract?

2. **Temporal Query Challenge**
   - You have 5 years of EIA weekly petroleum reports in a knowledge base
   - A user asks: "How does current crude inventory compare to pre-COVID levels?"
   - Design a retrieval strategy that:
     - Identifies "current" as most recent report
     - Defines "pre-COVID" as Q4 2019 - Q1 2020
     - Retrieves comparable seasonal periods only
   - What filters would you apply? How would you handle seasonality?

3. **Multi-Commodity Analysis**
   - Design a knowledge base schema that supports queries like:
     - "How do crude oil fundamentals affect gasoline prices?"
     - "What's the correlation between corn production and ethanol demand?"
   - Should you create separate collections per commodity or one unified collection?
   - How would you handle cross-commodity relationships?

4. **Data Quality Detection**
   - Implement a function that validates chunk quality:
     - Detects incomplete table splits
     - Identifies orphaned table headers
     - Flags contradictory data across chunks
   - What heuristics would you use? When should you reject a chunk?

5. **Real-Time Update Strategy**
   - Every Wednesday at 10:30 AM ET, a new EIA WPSR is released
   - Design an update strategy that:
     - Adds new report to knowledge base within 5 minutes
     - Deprecates (but doesn't delete) superseded forecasts
     - Maintains retrieval performance during updates
   - How would you handle versioning? What about rollback if data is revised?

<div class="callout-insight">

**Insight:** Understanding knowledge base design for commodity research is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.


## Further Reading

**Vector Databases for Time-Series Data:**
- Pinecone: "Metadata Filtering and Hybrid Search" - Best practices for temporal filtering
- ChromaDB Documentation: "Managing Time-Aware Collections"

**Commodity-Specific Knowledge Representation:**
- "Temporal Knowledge Graphs for Financial Markets" (2023) - Represents time-dependent relationships
- "Domain-Specific RAG Systems" (LangChain blog) - Customizing retrieval for specialized domains

**Chunking Strategies:**
- "Precision RAG: Prompt Tuning For Building Enterprise Grade RAG Systems" - Advanced chunking techniques
- LlamaIndex Guide: "Optimizing Chunk Size for Different Document Types"

**Production RAG Systems:**
- "Building Production-Ready RAG Applications" (Anthropic) - Best practices for reliability
- "RAG at Scale: Handling 10M+ Documents" - Performance optimization strategies

**Commodity Market Data APIs:**
- EIA API v2 Documentation - Understanding available data series
- USDA Quick Stats API - Agricultural data access patterns

---

## Conceptual Practice Questions

1. What is the difference between a keyword search and a vector similarity search for commodity research?

2. How should you chunk commodity reports for optimal retrieval performance?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---

## Cross-References

<a class="link-card" href="./01_knowledge_base_design_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_eia_knowledge_base.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./02_document_processing.md">
  <div class="link-card-title">02 Document Processing</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_retrieval_strategies.md">
  <div class="link-card-title">03 Retrieval Strategies</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

