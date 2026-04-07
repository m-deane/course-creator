# Processing Commodity Documents for RAG Systems

> **Reading time:** ~11 min | **Module:** Module 2: Rag Research | **Prerequisites:** Modules 0-1

<div class="callout-key">

**Key Concept Summary:** Commodity document processing transforms heterogeneous reports (PDFs, HTML tables, time-series data) into LLM-ready formats while preserving critical metadata (timestamps, units, geographies) and handling commodity-specific challenges like seasonal adjustments and data revisions.

</div>

## In Brief

Commodity document processing transforms heterogeneous reports (PDFs, HTML tables, time-series data) into LLM-ready formats while preserving critical metadata (timestamps, units, geographies) and handling commodity-specific challenges like seasonal adjustments and data revisions.

<div class="callout-insight">

**Insight:** The value of commodity documents isn't just in the text—it's in the structured data they contain. A crude oil inventory number without its date, unit, and geography is meaningless. Processing pipelines must extract, validate, and enrich data while maintaining traceability to source documents.

</div>
## Intuitive Explanation

Think of processing commodity documents like preparing ingredients for cooking:
- Raw ingredients (PDFs, HTML) arrive in different packaging
- You need to unpack them (parsing), wash them (cleaning), and prep them (structuring)
- Some ingredients are liquids (narrative text), others are solids (data tables)—handle differently
- Check for spoilage (validation)—bad data can ruin the whole dish
- Label everything (metadata)—you need to know what it is and when you got it

The challenge is that commodity reports are like IKEA instructions written in multiple languages with some pages missing and tables that span multiple pages. You need robust error handling and validation at every step.

## Formal Definition

A commodity document processing pipeline is a function **P: D_raw → D_structured** where:

**Input:** D_raw = raw document in format {PDF, HTML, CSV, API_response}

**Output:** D_structured = tuple (text, data_tables, metadata, validation_status)
- **text**: cleaned narrative content
- **data_tables**: structured DataFrames with validated units
- **metadata**: {commodity, date, source, geography, revision_number}
- **validation_status**: quality checks passed/failed

**Processing stages:**
<div class="flow">
<div class="flow-step mint">1. Acquisition</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">2. Parsing</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">3. Cleaning</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Structuring</div>
</div>


1. **Acquisition**: Fetch from source (API, web scraping, file upload)
2. **Parsing**: Extract text and tables from format-specific structure
3. **Cleaning**: Remove artifacts, normalize whitespace, fix encoding
4. **Structuring**: Separate narrative from data, identify relationships
5. **Validation**: Unit checking, range validation, historical consistency
6. **Enrichment**: Add derived fields, seasonality tags, cross-references

## Code Implementation

### Document Acquisition Layer


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">from.py</span>
</div>

```python
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import feedparser
from bs4 import BeautifulSoup
import pdfplumber
import pandas as pd
from dataclasses import dataclass
from enum import Enum

class DocumentSource(Enum):
    EIA_API = "eia_api"
    EIA_PDF = "eia_pdf"
    USDA_REPORT = "usda_report"
    EARNINGS_TRANSCRIPT = "earnings_transcript"
    NEWS_RSS = "news_rss"

@dataclass
class RawDocument:
    """Raw document before processing."""
    source: DocumentSource
    content: bytes | str
    url: str
    fetch_date: datetime
    metadata: Dict

class DocumentAcquisition:
    """
    Acquire commodity documents from various sources.
    """

    def __init__(self, eia_api_key: str):
        self.eia_api_key = eia_api_key

    def fetch_eia_report(
        self,
        report_type: str,
        report_date: Optional[datetime] = None
    ) -> RawDocument:
        """
        Fetch EIA report from their website.

        Args:
            report_type: "wpsr" (weekly petroleum) or "steo" (outlook)
            report_date: Specific report date, defaults to latest
        """
        if report_type == "wpsr":
            # Weekly Petroleum Status Report is published Wednesdays
            if report_date is None:
                # Get most recent Wednesday
                report_date = self._get_last_wednesday()

            # Construct PDF URL
            date_str = report_date.strftime("%Y%m%d")
            url = f"https://www.eia.gov/petroleum/supply/weekly/pdf/wpsrall.pdf"

            response = requests.get(url)
            response.raise_for_status()

            return RawDocument(
                source=DocumentSource.EIA_PDF,
                content=response.content,
                url=url,
                fetch_date=datetime.now(),
                metadata={
                    "report_type": report_type,
                    "report_date": report_date,
                    "commodity": "petroleum"
                }
            )

        elif report_type == "steo":
            # Monthly Short-Term Energy Outlook
            url = "https://www.eia.gov/outlooks/steo/pdf/steo_full.pdf"
            response = requests.get(url)
            response.raise_for_status()

            return RawDocument(
                source=DocumentSource.EIA_PDF,
                content=response.content,
                url=url,
                fetch_date=datetime.now(),
                metadata={
                    "report_type": "steo",
                    "report_date": datetime.now().replace(day=1),
                    "commodity": "energy"
                }
            )

    def fetch_eia_api_data(
        self,
        series_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> RawDocument:
        """
        Fetch time-series data from EIA API.

        Common series IDs:
        - PET.WCESTUS1.W: Weekly crude oil stocks
        - NG.NW2_EPG0_SWO_R48_BCF.W: Natural gas storage
        """
        base_url = "https://api.eia.gov/v2"

        params = {
            "api_key": self.eia_api_key,
            "frequency": "weekly",
            "data[0]": "value",
            "start": start_date.strftime("%Y-%m-%d"),
            "end": end_date.strftime("%Y-%m-%d"),
            "sort[0][column]": "period",
            "sort[0][direction]": "desc"
        }

        # Determine endpoint from series ID
        if series_id.startswith("PET"):
            endpoint = f"{base_url}/petroleum/sum/sndw/data"
        elif series_id.startswith("NG"):
            endpoint = f"{base_url}/natural-gas/sum/sndw/data"

        params["facets[series][]"] = series_id

        response = requests.get(endpoint, params=params)
        response.raise_for_status()

        return RawDocument(
            source=DocumentSource.EIA_API,
            content=response.text,
            url=endpoint,
            fetch_date=datetime.now(),
            metadata={
                "series_id": series_id,
                "start_date": start_date,
                "end_date": end_date,
                "data_type": "time_series"
            }
        )

    def fetch_usda_report(self, report_type: str) -> RawDocument:
        """
        Fetch USDA commodity reports.

        Args:
            report_type: "wasde" (World Ag Supply/Demand) or "crop_progress"
        """
        if report_type == "wasde":
            # WASDE is published monthly
            url = "https://www.usda.gov/oce/commodity/wasde/latest.pdf"
            response = requests.get(url)
            response.raise_for_status()

            return RawDocument(
                source=DocumentSource.USDA_REPORT,
                content=response.content,
                url=url,
                fetch_date=datetime.now(),
                metadata={
                    "report_type": "wasde",
                    "commodity": "agriculture",
                    "report_date": datetime.now().replace(day=1)
                }
            )

    def fetch_commodity_news(
        self,
        commodity: str,
        hours_back: int = 24
    ) -> List[RawDocument]:
        """
        Fetch recent commodity news from RSS feeds.
        """
        # Example feeds (in production, use comprehensive feed aggregator)
        feeds = {
            "energy": [
                "https://www.rigzone.com/news/rss.aspx",
                "https://www.naturalgasintel.com/feed/"
            ],
            "agriculture": [
                "https://www.agriculture.com/rss"
            ]
        }

        documents = []
        for feed_url in feeds.get(commodity, []):
            feed = feedparser.parse(feed_url)

            for entry in feed.entries:
                pub_date = datetime(*entry.published_parsed[:6])

                # Filter by recency
                if datetime.now() - pub_date > timedelta(hours=hours_back):
                    continue

                documents.append(RawDocument(
                    source=DocumentSource.NEWS_RSS,
                    content=entry.summary,
                    url=entry.link,
                    fetch_date=datetime.now(),
                    metadata={
                        "title": entry.title,
                        "pub_date": pub_date,
                        "commodity": commodity,
                        "source": feed_url
                    }
                ))

        return documents

    def _get_last_wednesday(self) -> datetime:
        """Get the most recent Wednesday (EIA release day)."""
        today = datetime.now()
        days_since_wednesday = (today.weekday() - 2) % 7
        last_wednesday = today - timedelta(days=days_since_wednesday)
        return last_wednesday.replace(hour=10, minute=30, second=0)
```

</div>
</div>

### PDF Processing with Table Extraction


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">pdfprocessor.py</span>
</div>

```python
from anthropic import Anthropic

class PDFProcessor:
    """
    Process commodity PDF reports with table extraction.
    """

    def __init__(self):
        self.anthropic_client = Anthropic()

    def process_pdf(self, pdf_bytes: bytes) -> Dict:
        """
        Extract text and tables from commodity PDF.
        """
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            # Extract all text
            full_text = ""
            tables = []

            for page in pdf.pages:
                # Get page text
                page_text = page.extract_text()
                full_text += page_text + "\n\n"

                # Extract tables
                page_tables = page.extract_tables()
                for table in page_tables:
                    # Convert to DataFrame for easier handling
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        tables.append(df)

        return {
            "text": full_text,
            "tables": tables,
            "page_count": len(pdf.pages)
        }

    def clean_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean extracted table data.
        """
        # Remove empty rows/columns
        df = df.dropna(how='all').dropna(axis=1, how='all')

        # Strip whitespace from all string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()

        # Attempt to convert numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                pass

        return df

    def parse_eia_wpsr_table(self, table_df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse EIA Weekly Petroleum Status Report tables.

        These tables have specific formats with:
        - Product names in first column
        - Multiple time periods across columns
        - Units in headers or footnotes
        """
        # Identify table type using LLM
        table_text = table_df.to_string()

        prompt = f"""Identify this EIA table type and extract data.

Table:
{table_text}

Return JSON:
{{
  "table_type": "inventory|supply|demand|imports|exports",
  "unit": "thousand_barrels|million_barrels|barrels_per_day",
  "time_columns": ["current_week", "prior_week", "year_ago"],
  "products": [
    {{"product": "crude_oil", "current_week": 430000, "prior_week": 435200}}
  ]
}}"""

        response = self.anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        parsed = json.loads(response.content[0].text)

        # Convert to structured DataFrame
        structured_df = pd.DataFrame(parsed["products"])
        structured_df["unit"] = parsed["unit"]
        structured_df["table_type"] = parsed["table_type"]

        return structured_df
```

</div>
</div>

### Data Validation Layer


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">datavalidator.py</span>
</div>

```python
from typing import List, Tuple

class DataValidator:
    """
    Validate commodity data for quality and consistency.
    """

    def __init__(self):
        # Historical ranges for sanity checks
        self.valid_ranges = {
            "crude_oil_inventory_mmb": (200, 600),  # Million barrels
            "gasoline_inventory_mmb": (180, 260),
            "natural_gas_storage_bcf": (1000, 4500),  # Billion cubic feet
            "corn_production_million_bu": (8000, 16000),  # Million bushels
            "refinery_utilization_pct": (70, 100)
        }

        # Maximum reasonable weekly changes
        self.max_weekly_changes = {
            "crude_oil_inventory_mmb": 20,
            "gasoline_inventory_mmb": 10,
            "natural_gas_storage_bcf": 150
        }

    def validate_value(
        self,
        metric: str,
        value: float,
        commodity: str
    ) -> Tuple[bool, str]:
        """
        Check if value is within reasonable range.

        Returns: (is_valid, error_message)
        """
        # Check if metric is known
        if metric not in self.valid_ranges:
            return True, ""  # Can't validate unknown metrics

        min_val, max_val = self.valid_ranges[metric]

        if not (min_val <= value <= max_val):
            return False, f"{metric} value {value} outside valid range [{min_val}, {max_val}]"

        return True, ""

    def validate_weekly_change(
        self,
        metric: str,
        current: float,
        prior: float
    ) -> Tuple[bool, str]:
        """
        Check if weekly change is reasonable.
        """
        if metric not in self.max_weekly_changes:
            return True, ""

        change = abs(current - prior)
        max_change = self.max_weekly_changes[metric]

        if change > max_change:
            return False, f"Weekly change of {change} exceeds maximum {max_change}"

        return True, ""

    def validate_unit_consistency(
        self,
        value: float,
        stated_unit: str,
        expected_unit: str
    ) -> Tuple[bool, str]:
        """
        Check if units match expectations.
        """
        # Unit conversion factors
        conversions = {
            ("thousand_barrels", "million_barrels"): 1000,
            ("barrels", "thousand_barrels"): 1000,
            ("cubic_feet", "billion_cubic_feet"): 1e9
        }

        if stated_unit == expected_unit:
            return True, ""

        # Check if convertible
        conversion_key = (stated_unit, expected_unit)
        if conversion_key in conversions:
            return True, f"Unit mismatch: {stated_unit} vs {expected_unit} (convertible)"

        return False, f"Incompatible units: {stated_unit} vs {expected_unit}"

    def cross_reference_api_vs_pdf(
        self,
        api_value: float,
        pdf_value: float,
        tolerance_pct: float = 1.0
    ) -> Tuple[bool, str]:
        """
        Validate that API data matches PDF report.
        """
        diff_pct = abs(api_value - pdf_value) / api_value * 100

        if diff_pct > tolerance_pct:
            return False, f"API/PDF mismatch: {api_value} vs {pdf_value} ({diff_pct:.1f}% diff)"

        return True, ""
```

</div>
</div>

### Complete Processing Pipeline


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">class.py</span>
</div>

```python
import io

@dataclass
class ProcessedDocument:
    """Fully processed commodity document."""
    original_source: DocumentSource
    narrative_text: str
    data_tables: List[pd.DataFrame]
    metadata: Dict
    validation_results: Dict
    processing_date: datetime

class CommodityDocumentProcessor:
    """
    End-to-end commodity document processing pipeline.
    """

    def __init__(self, eia_api_key: str):
        self.acquisition = DocumentAcquisition(eia_api_key)
        self.pdf_processor = PDFProcessor()
        self.validator = DataValidator()

    def process_eia_wpsr(self, report_date: Optional[datetime] = None) -> ProcessedDocument:
        """
        Process complete EIA Weekly Petroleum Status Report.
        """
        # Step 1: Acquire document
        raw_pdf = self.acquisition.fetch_eia_report("wpsr", report_date)

        # Step 2: Extract text and tables
        extracted = self.pdf_processor.process_pdf(raw_pdf.content)

        # Step 3: Process tables
        processed_tables = []
        for table_df in extracted["tables"]:
            cleaned = self.pdf_processor.clean_table(table_df)
            if not cleaned.empty:
                structured = self.pdf_processor.parse_eia_wpsr_table(cleaned)
                processed_tables.append(structured)

        # Step 4: Validate data (cross-check with API)
        validation_results = {}
        try:
            # Fetch same data from API for validation
            api_crude = self.acquisition.fetch_eia_api_data(
                series_id="PET.WCESTUS1.W",
                start_date=report_date - timedelta(days=14),
                end_date=report_date
            )

            import json
            api_data = json.loads(api_crude.content)
            api_value = float(api_data["response"]["data"][0]["value"])

            # Find crude oil value in PDF tables
            for table in processed_tables:
                if "crude_oil" in str(table.get("product", "")).lower():
                    pdf_value = float(table.iloc[0]["current_week"])

                    is_valid, msg = self.validator.cross_reference_api_vs_pdf(
                        api_value,
                        pdf_value / 1000  # Convert thousands to millions
                    )

                    validation_results["crude_oil_cross_check"] = {
                        "valid": is_valid,
                        "message": msg,
                        "api_value": api_value,
                        "pdf_value": pdf_value
                    }
        except Exception as e:
            validation_results["cross_check_error"] = str(e)

        # Step 5: Return processed document
        return ProcessedDocument(
            original_source=raw_pdf.source,
            narrative_text=extracted["text"],
            data_tables=processed_tables,
            metadata={
                **raw_pdf.metadata,
                "page_count": extracted["page_count"],
                "table_count": len(processed_tables)
            },
            validation_results=validation_results,
            processing_date=datetime.now()
        )

    def process_batch(
        self,
        source_type: DocumentSource,
        date_range: Tuple[datetime, datetime]
    ) -> List[ProcessedDocument]:
        """
        Process multiple documents in batch.
        """
        documents = []

        if source_type == DocumentSource.EIA_PDF:
            # Process weekly reports in date range
            current_date = date_range[0]
            while current_date <= date_range[1]:
                # Find next Wednesday
                days_until_wednesday = (2 - current_date.weekday()) % 7
                wednesday = current_date + timedelta(days=days_until_wednesday)

                if wednesday <= date_range[1]:
                    try:
                        doc = self.process_eia_wpsr(wednesday)
                        documents.append(doc)
                    except Exception as e:
                        print(f"Failed to process {wednesday}: {e}")

                current_date = wednesday + timedelta(days=1)

        return documents
```


## Common Pitfalls

**1. Ignoring PDF Extraction Errors**
- **Problem**: Tables span multiple pages or have merged cells
- **Why it happens**: PDFs are designed for human reading, not parsing
- **Solution**: Use LLM to reconstruct broken tables; validate against API data

**2. Unit Confusion**
- **Problem**: Mixing thousand barrels with million barrels
- **Why it happens**: Different reports use different units
- **Solution**: Mandatory unit extraction and validation; convert to standard units

**3. Assuming Clean Data**
- **Problem**: OCR errors, typos, missing values in source documents
- **Why it happens**: Reports are manually created and updated
- **Solution**: Range validation, cross-referencing, outlier detection

**4. Missing Revisions**
- **Problem**: Historical data gets revised but you use old values
- **Why it happens**: Agencies update past reports when new information arrives
- **Solution**: Track revision numbers; update knowledge base when revisions published

**5. Timezone Issues**
- **Problem**: Mixing UTC with local times for report releases
- **Why it happens**: EIA releases at 10:30 AM ET, USDA times vary
- **Solution**: Store all timestamps in UTC with timezone metadata

## Connections

**Builds on:**
- PDF parsing libraries (pdfplumber, PyPDF2)
- Web scraping (BeautifulSoup, requests)
- Data validation techniques (pandas, unit testing)

**Leads to:**
- Knowledge base ingestion (adding processed docs to vector stores)
- Real-time monitoring (detecting new report releases)
- Data quality dashboards (tracking validation failures)

**Related to:**
- ETL pipelines (Extract-Transform-Load patterns)
- Data lineage tracking (knowing where data came from)
- Error handling and logging (production-grade systems)

## Practice Problems

1. **Multi-Format Table Extraction**
   - Download the latest EIA WPSR PDF
   - Extract all inventory tables
   - Convert to a single normalized DataFrame with columns: [commodity, date, value, unit, geography]
   - Validate that weekly changes are within historical norms

2. **API vs PDF Reconciliation**
   - For crude oil stocks, fetch data from both EIA API and PDF report
   - Identify any discrepancies
   - Implement automatic resolution strategy (which source to trust?)
   - Log discrepancies for manual review

3. **USDA Report Parser**
   - Parse a USDA WASDE report
   - Extract crop production forecasts for corn, soybeans, wheat
   - Identify which values are forecasts vs. actuals
   - Calculate month-over-month forecast changes

4. **News Article Processing**
   - Fetch 24 hours of commodity news from RSS feeds
   - Extract mentioned commodities (use LLM)
   - Classify sentiment (bullish/bearish/neutral)
   - Deduplicate articles covering same story

5. **Revision Tracking System**
   - Design a system to track when EIA revises historical data
   - Detect revisions by comparing new report to stored version
   - Quantify impact of revisions on your analysis
   - Trigger re-analysis if revisions are significant

<div class="callout-insight">

**Insight:** Understanding processing commodity documents for rag systems is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.




## Further Reading

**PDF Processing:**
- pdfplumber documentation - "Advanced table extraction techniques"
- "Parsing Complex PDFs with LLMs" (Anthropic blog) - Using Claude for table understanding

**Data Validation:**
- "Great Expectations for Data Quality" - Automated data testing framework
- "Data Validation Patterns for Financial Data" - Industry best practices

**Commodity Data Sources:**
- EIA API v2 Documentation - Complete reference
- USDA Data APIs - Quick Stats and reports
- CME Group Data Services - Futures market data

**Production ETL:**
- "Building Robust Data Pipelines" - Error handling and retry logic
- Apache Airflow Documentation - Workflow orchestration
- "Data Pipeline Design Patterns" - Scalable architecture approaches

**LLM-Augmented Processing:**
- "Using LLMs for Data Extraction" (LangChain) - Structured output patterns
- "Claude for Document Intelligence" (Anthropic) - Best practices for table/text extraction

---

## Conceptual Practice Questions

1. Explain the core idea of processing commodity documents for rag systems in your own words to a colleague who has not studied it.

2. What is the most common mistake practitioners make when applying processing commodity documents for rag systems, and how would you avoid it?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---



## Cross-References

<a class="link-card" href="./02_document_processing_slides.md">
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

<a class="link-card" href="./03_retrieval_strategies.md">
  <div class="link-card-title">03 Retrieval Strategies</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

