# Quiz: Module 1 - Report Processing

**Course:** GenAI for Commodity Markets
**Module:** 1 - Report Processing
**Total Points:** 100
**Time Limit:** 30 minutes
**Attempts Allowed:** 2

## Instructions

This quiz assesses your ability to design and implement LLM-powered report processing systems for commodity markets. Focus on extraction pipelines, validation strategies, and production-ready implementations.

---

## Section 1: EIA Report Processing (30 points)

### Question 1 (10 points)

You're extracting data from this EIA Weekly Petroleum Status Report excerpt:

> "U.S. crude oil imports averaged 6.5 million barrels per day last week, up by 0.3 million barrels per day from the previous week. Over the past four weeks, crude oil imports averaged 6.1 million barrels per day, 5.2% more than the same four-week period last year."

Which extraction approach produces the MOST reliable structured output?

```python
# Option A
prompt = "Extract all the numbers from this text."

# Option B
prompt = """Extract crude oil import data as JSON:
{
    "imports_current_week": <number>,
    "imports_change_wow": <number>,
    "imports_4wk_avg": <number>,
    "imports_yoy_change_pct": <number>
}
"""

# Option C
prompt = """Extract crude oil import data. Return JSON with these exact fields:
{
    "commodity": "crude_oil",
    "metric": "imports",
    "current_value": <float, million bpd>,
    "current_period": "week",
    "change_vs_previous": <float, million bpd>,
    "change_period": "week_over_week",
    "rolling_average": <float, million bpd>,
    "rolling_period": "4_week",
    "yoy_change": <float, percent>,
    "units": "million_barrels_per_day"
}

If any value is not present, use null. All numbers should be floats."""

# Option D
prompt = "What were crude oil imports last week?"
```

A) Option A - Simplest approach
B) Option B - Adequate structure
C) Option C - Comprehensive schema with validation
D) Option D - Direct question

**Answer: C**

**Explanation:**
- **A:** Extracts numbers without context or structure. Can't distinguish between imports, changes, or percentages.
- **B:** Better structure but lacks units, doesn't specify data types, no null handling for missing values.
- **C (Correct):** Production-ready extraction:
  - Specifies exact field names and data types
  - Includes units and periods for context
  - Handles missing data with null values
  - Normalizes temporal references (week_over_week, 4_week)
  - Can be validated programmatically with Pydantic

  This enables direct database insertion and time-series analysis.
- **D:** Conversational but loses precision and structure.

**Validation Example:**
```python
from pydantic import BaseModel, Field
from typing import Optional

class CrudeImportData(BaseModel):
    commodity: str
    metric: str
    current_value: Optional[float] = Field(ge=0, le=15)  # Validate range
    current_period: str
    units: str
```

---

### Question 2 (10 points)

The EIA report contains this table:

```
                           Current Week    Previous Week    Change
Crude Oil (mb)                430.0           435.2         -5.2
Gasoline (mb)                 234.1           232.8         +1.3
Distillate (mb)               125.3           127.1         -1.8
```

Which Python approach would be MOST reliable for extracting this table?

A) Use the LLM to describe the table in natural language
B) Request JSON array with explicit structure for each commodity and field
C) Ask the LLM to recreate the table in markdown format
D) Extract only the "Change" column as that's most important

**Answer: B**

**Explanation:**
- **A:** Loses structure and makes downstream analysis harder; requires parsing natural language descriptions.
- **B (Correct):** Request structured format:
```python
prompt = """Extract inventory data from the table as JSON array:
[
  {
    "commodity": "<name>",
    "current_week": <float>,
    "previous_week": <float>,
    "change": <float>,
    "units": "million_barrels"
  },
  ...
]

Preserve positive/negative signs for changes."""

# Produces parseable output
data = json.loads(response)
df = pd.DataFrame(data)
```
  This enables immediate use in pandas, databases, or APIs.
- **C:** Markdown tables still require parsing and may introduce formatting inconsistencies.
- **D:** Discards valuable absolute inventory levels needed for historical analysis.

**Post-Processing Validation:**
```python
for item in data:
    # Verify math
    calculated_change = item['current_week'] - item['previous_week']
    assert abs(calculated_change - item['change']) < 0.1, "Math error detected"
```

---

### Question 3 (10 points)

You're processing 52 weeks of historical EIA reports. Week 23's report has a formatting error and the LLM extracts incorrect inventory values (returns 43,000 instead of 430.0 million barrels). What is the BEST error detection strategy?

A) Manually review all 52 extractions
B) Compare each extraction against the previous week and flag large jumps
C) Implement multi-layered validation:
   - Range checks (inventory must be 300-600 million barrels)
   - Sequential validation (change from previous week < 50 million)
   - Statistical outlier detection (>3 standard deviations)
   - Cross-reference with EIA API when available
D) Re-run the extraction with a different model

**Answer: C**

**Explanation:**
- **A:** Not scalable; manual review defeats automation purpose.
- **B:** Helpful but misses errors that occur gradually or when both adjacent weeks are wrong.
- **C (Correct):** Defense-in-depth validation:

```python
def validate_extraction(current_data, previous_data, historical_data):
    """Multi-layer validation for commodity data."""

    # Layer 1: Range validation
    if not (300 <= current_data['inventory'] <= 600):
        raise ValidationError(f"Inventory {current_data['inventory']} outside normal range")

    # Layer 2: Sequential validation
    if previous_data:
        change = abs(current_data['inventory'] - previous_data['inventory'])
        if change > 50:
            flag_for_review(f"Unusual change: {change} million barrels")

    # Layer 3: Statistical validation
    if historical_data:
        mean = np.mean([d['inventory'] for d in historical_data])
        std = np.std([d['inventory'] for d in historical_data])
        z_score = abs(current_data['inventory'] - mean) / std
        if z_score > 3:
            flag_for_review(f"Statistical outlier: z={z_score:.2f}")

    # Layer 4: API cross-reference
    if eia_api_available:
        api_value = fetch_eia_data(current_data['date'])
        if abs(current_data['inventory'] - api_value) > 1.0:
            log_discrepancy(current_data['inventory'], api_value)
            return api_value  # Use authoritative source

    return current_data['inventory']
```

  This catches the 43,000 error immediately (fails range check).
- **D:** Different models may make different errors; doesn't solve validation problem.

---

## Section 2: USDA Report Processing (25 points)

### Question 4 (10 points)

The USDA WASDE report contains this supply/demand table for corn:

```
                        2023/24         2024/25
                        (Million Bushels)
Beginning Stocks          1,377           2,162
Production               15,342          14,980
Imports                      25              25
Total Supply            16,744          17,167

Feed & Residual          5,425           5,400
Food, Seed & Industrial  6,505           6,555
Exports                  2,150           2,000
Total Use              14,080          13,955

Ending Stocks            2,162           3,212
```

What is the MOST important validation check after LLM extraction?

A) Verify that Total Supply = Beginning Stocks + Production + Imports
B) Check that all numbers are positive
C) Confirm that Ending Stocks = Total Supply - Total Use
D) Both A and C must be validated

**Answer: D**

**Explanation:**
- **A:** Supply identity must hold: Total Supply = Beginning Stocks + Production + Imports
- **B:** While typically positive, some fields (e.g., statistical adjustments) can be negative.
- **C:** Balance identity must hold: Ending Stocks = Total Supply - Total Use
- **D (Correct):** USDA balance sheets have accounting identities that MUST balance:

```python
def validate_wasde_balance(data):
    """Validate USDA supply/demand balance sheet."""

    # Supply identity
    calculated_supply = (
        data['beginning_stocks'] +
        data['production'] +
        data['imports']
    )
    assert abs(calculated_supply - data['total_supply']) < 1, \
        f"Supply doesn't balance: {calculated_supply} != {data['total_supply']}"

    # Demand identity
    calculated_demand = (
        data['feed_residual'] +
        data['fsi'] +
        data['exports']
    )
    assert abs(calculated_demand - data['total_use']) < 1, \
        f"Demand doesn't balance: {calculated_demand} != {data['total_use']}"

    # Balance identity
    calculated_ending = data['total_supply'] - data['total_use']
    assert abs(calculated_ending - data['ending_stocks']) < 1, \
        f"Ending stocks don't balance: {calculated_ending} != {data['ending_stocks']}"

    # Next year continuity
    if 'next_year' in data:
        assert abs(data['ending_stocks'] - data['next_year']['beginning_stocks']) < 1, \
            "Ending stocks don't match next year's beginning stocks"

    return True
```

If these validations fail, the extraction contains errors and should be re-attempted or flagged for manual review.

---

### Question 5 (8 points)

You need to track how USDA corn production forecasts change across multiple WASDE reports throughout the growing season. Which data structure is MOST appropriate?

A) Store only the latest forecast for each marketing year
B) Store each forecast with report date, marketing year, and forecast value
C) Store forecast changes (deltas) between reports
D) Store only final end-of-season production values

**Answer: B**

**Explanation:**
- **A:** Loses valuable revision history that traders use to gauge uncertainty and USDA confidence.
- **B (Correct):** Time-series of forecasts enables crucial analysis:

```python
@dataclass
class USDAForecast:
    report_date: date              # When report was published
    marketing_year: str            # e.g., "2024/25"
    commodity: str                 # "corn", "soybeans", etc.
    metric: str                    # "production", "ending_stocks"
    forecast_value: float
    units: str

# Example queries enabled by this structure:
# 1. Forecast revision analysis
revisions = df.groupby('marketing_year').apply(
    lambda x: x.sort_values('report_date')['forecast_value'].diff()
)

# 2. Forecast accuracy
final = df[df.report_date == df.marketing_year_end]['forecast_value']
early = df[df.report_date == df.marketing_year_start]['forecast_value']
accuracy = abs(final - early) / final

# 3. Seasonal patterns in revisions
monthly_revisions = df.groupby([df.report_date.dt.month, 'marketing_year'])['forecast_value'].mean()
```

- **C:** Deltas are useful but can be derived from absolute values; storing only deltas loses baseline context.
- **D:** Loses the entire forecasting process which contains trading signals.

**Key Insight:** USDA forecast revisions are themselves trading signals. Large unexpected revisions can move markets significantly.

---

### Question 6 (7 points)

When extracting data from USDA Crop Progress reports that use prose descriptions like "As of June 15, corn planting was 95% complete, compared to 92% last week and 89% for the five-year average," which extraction challenge is MOST critical to address?

A) Converting percentages to decimal format
B) Disambiguating between multiple temporal comparisons (last week vs. five-year average vs. current)
C) Handling missing data when states don't report
D) Converting state-level to national aggregates

**Answer: B**

**Explanation:**
- **A:** Straightforward conversion (95% → 0.95); low complexity.
- **B (Correct):** Temporal disambiguation is critical:

```python
# Clear schema for temporal comparisons
@dataclass
class CropProgress:
    commodity: str
    state: str
    report_date: date
    metric: str                    # "planting_complete", "emerged", etc.
    current_value: float           # 0.95
    comparison_values: dict = field(default_factory=dict)
    # {
    #     "last_week": 0.92,
    #     "last_year": 0.88,
    #     "five_year_avg": 0.89
    # }

# Prompt must explicitly request structure:
prompt = """Extract crop progress with clear temporal labels:
- current_value: This week's percentage
- last_week: Previous week (labeled as "last week", "previous week")
- last_year: Same week last year
- five_year_avg: Five-year average for this week
"""
```

Without this structure, it's easy to confuse which comparison is which, especially in complex sentences.

- **C:** Important but typically handled by noting null values in the schema.
- **D:** USDA reports already provide national aggregates; state data is separate.

---

## Section 3: Earnings Transcript Analysis (20 points)

### Question 7 (10 points)

You're analyzing earnings call transcripts from major oil companies to detect commodity price sensitivity. An executive says:

> "While we don't provide specific price guidance, we're seeing healthy demand across all refined products. Our integrated model provides natural hedges across the value chain, and we're well-positioned regardless of where crude prices settle in the near term."

What is the MOST accurate characterization of this statement's commodity exposure?

A) Bullish on crude oil prices
B) Bearish on crude oil prices
C) Neutral/hedged position with limited price sensitivity
D) Insufficient information to determine

**Answer: C**

**Explanation:**
- **A/B:** The statement explicitly avoids price direction implications.
- **C (Correct):** Key phrases indicate hedged position:
  - "integrated model" - upstream production offsets downstream refining
  - "natural hedges across the value chain" - profit from both ends
  - "well-positioned regardless of where crude prices settle" - explicit statement of neutrality

```python
# LLM extraction prompt for earnings calls:
prompt = """Analyze this earnings call excerpt for commodity exposure:

Quote: {quote}

Extract:
{
    "commodity_mentioned": ["crude_oil", "refined_products"],
    "exposure_type": "hedged|long|short|neutral",
    "price_sensitivity": "high|medium|low",
    "key_phrases": ["integrated model", "natural hedges"],
    "confidence": 0.85,
    "rationale": "Company explicitly states hedged position across value chain"
}
"""
```

- **D:** While cautious, the statement provides clear hedging signals.

**Trading Implication:** This company's earnings are less sensitive to crude price moves than pure-play producers or refiners.

---

### Question 8 (10 points)

You're building a system to extract commodity mentions from 100+ earnings transcripts. Which text processing approach is MOST efficient and accurate?

**Architecture A: Full Transcript Processing**
- Send entire transcript (20,000 tokens) to LLM
- Ask for all commodity mentions
- Extract in single pass

**Architecture B: Chunk-then-Extract**
- Split transcript into Q&A chunks
- Process each chunk separately
- Aggregate results

**Architecture C: Pre-filter-then-Extract**
1. Use regex/keyword search to find commodity-related sections
2. Send only relevant sections to LLM for detailed extraction
3. Use prompt caching for extraction instructions

**Architecture D: Fine-tune BERT**
- Fine-tune BERT for commodity NER
- Run inference on all transcripts
- Post-process results

A) Architecture A - Most comprehensive
B) Architecture B - Best accuracy
C) Architecture C - Optimal cost/accuracy balance
D) Architecture D - Highest quality

**Answer: C**

**Explanation:**

**Cost/Performance Analysis:**

**Architecture A:**
- Cost: 20k tokens × 100 transcripts = 2M tokens (~$16 for Sonnet)
- Accuracy: Attention dilution on long contexts
- Latency: Very long processing times
- Risk: May miss details in lengthy transcripts

**Architecture B:**
- Cost: Similar to A but with overhead for chunking
- Accuracy: Better focused attention per chunk
- Complexity: Aggregation logic needed to avoid double-counting
- Issue: Chunk boundaries may split relevant context

**Architecture C (Correct):**
```python
def efficient_transcript_processing(transcript, cached_prompt):
    # Step 1: Pre-filter (cheap)
    commodity_keywords = ['oil', 'gas', 'crude', 'natural gas', 'LNG',
                          'barrel', 'mcf', 'btu', 'upstream', 'downstream']

    relevant_chunks = []
    for paragraph in transcript.split('\n\n'):
        if any(kw in paragraph.lower() for kw in commodity_keywords):
            relevant_chunks.append(paragraph)

    if not relevant_chunks:
        return None  # No commodity discussion

    # Step 2: Extract from relevant sections only
    # Use cached prompt for extraction instructions
    filtered_text = '\n\n'.join(relevant_chunks)
    # Reduced tokens: maybe 3k tokens vs 20k

    result = llm_extract(cached_prompt, filtered_text)
    return result

# Savings: ~85% token reduction per transcript
# Cost: ~$2.40 for 100 transcripts (vs $16)
# Accuracy: Focused on relevant content
```

**Architecture D:**
- Cost: High upfront (labeling data, training, infrastructure)
- Complexity: Requires ML engineering expertise
- Maintenance: Need retraining for new commodity types
- Overkill: LLMs already excel at NER tasks

---

## Section 4: Production Pipeline Design (25 points)

### Question 9 (12 points)

You're deploying an EIA report processing pipeline that must:
- Process reports within 5 minutes of release (10:30 AM ET Wednesdays)
- Handle occasional LLM API failures
- Store results in a database
- Alert traders if significant inventory changes detected

Which implementation is MOST production-appropriate?

```python
# Option A: Simple Script
def process_report():
    report = fetch_eia_report()
    data = llm_extract(report)
    db.insert(data)
    if abs(data['change']) > 5:
        send_alert(data)

schedule.every().wednesday.at("10:35").do(process_report)

# Option B: Robust Pipeline
def process_eia_report():
    try:
        # Fetch with timeout
        report = fetch_eia_report(timeout=30)
        if not report:
            raise ValueError("Empty report")

        # Extract with retry
        data = llm_extract_with_retry(
            report,
            max_retries=3,
            backoff_factor=2
        )

        # Validate
        validate_eia_data(data)

        # Store with transaction
        with db.transaction():
            db.insert(data)
            db.log_processing_metadata({
                'timestamp': datetime.now(),
                'tokens_used': data.get('token_count'),
                'processing_time': elapsed
            })

        # Alert on significant changes
        if abs(data['crude_change']) > 5:
            send_alert(
                message=f"Large crude draw: {data['crude_change']} mb",
                priority='high',
                data=data
            )

        logger.info(f"Successfully processed report for {data['date']}")
        return data

    except LLMAPIError as e:
        logger.error(f"LLM API failed: {e}")
        # Fall back to cached previous week's structure
        return process_with_fallback(report)

    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        # Flag for manual review
        send_alert(
            message=f"EIA processing validation failed: {e}",
            priority='critical'
        )
        raise

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # Dead letter queue
        dlq.put({'report': report, 'error': str(e)})
        raise

# Option C: Microservices
# 3 separate services: Fetcher, Processor, Alerter
# Message queue between each
# Docker + Kubernetes

# Option D: Serverless
# AWS Lambda triggered by EventBridge schedule
# S3 for report storage
# DynamoDB for results
```

A) Option A - Simplest implementation
B) Option B - Robust error handling and monitoring
C) Option C - Most scalable architecture
D) Option D - Most modern technology

**Answer: B**

**Explanation:**
- **A:** Production-inappropriate:
  - No error handling (API failures crash pipeline)
  - No validation (bad extractions go to database)
  - No logging (can't debug failures)
  - No retry logic (single API timeout fails entire process)

- **B (Correct):** Production-ready features:
  - **Error Handling:** Try-catch blocks for different failure modes
  - **Retry Logic:** Exponential backoff for transient failures
  - **Validation:** Catches data quality issues before storage
  - **Fallback:** Graceful degradation if LLM unavailable
  - **Observability:** Logging, metadata tracking
  - **Alerting:** Both data alerts and operational alerts
  - **Transaction Safety:** Database transactions for consistency

  This handles real-world failures without requiring complex infrastructure.

- **C:** Over-engineered for a single weekly report:
  - Operational complexity far exceeds need
  - 3 services for a 5-minute/week workload
  - K8s overhead not justified
  - Harder to debug

- **D:** Reasonable but:
  - Cold start latency (5-10 seconds) cuts into 5-minute SLA
  - Multiple AWS services increase complexity
  - Lambda limitations (15-minute timeout should be fine, but adds constraints)
  - Option B achieves same reliability with simpler infrastructure

**Key Principle:** Choose simplest architecture that meets requirements. A well-written monolith with proper error handling beats a poorly-managed microservices setup.

---

### Question 10 (13 points)

Your report processing system has been running for 3 months. You notice:
- Week 1-8: 100% success rate, avg 2.5 min processing time
- Week 9-12: 85% success rate, failures all on Week 10 (EIA report had unusual formatting)

You're now implementing improvements. Rank these enhancements by PRIORITY:

I. Add comprehensive logging to capture raw API responses for debugging
II. Implement A/B testing between two different extraction prompts
III. Add validation checks to detect and flag unusual report formatting
IV. Build a dashboard showing extraction success rates and processing times
V. Implement automated retry with alternative prompts when primary extraction fails

A) III, V, I, IV, II
B) II, IV, III, V, I
C) V, III, I, IV, II
D) I, III, IV, V, II

**Answer: A**

**Explanation:**

**A (Correct): III, V, I, IV, II**

**Priority 1 - Validation (III):**
The Week 10 failure shows formatting changes broke extraction. Validation would have:
- Detected the issue immediately
- Prevented bad data from reaching database
- Triggered fallback logic

```python
def validate_report_format(report_text):
    """Detect unusual formatting before processing."""
    expected_sections = ['Summary', 'Crude Oil', 'Gasoline', 'Distillate']
    found_sections = [s for s in expected_sections if s in report_text]

    if len(found_sections) < 3:
        logger.warning(f"Unusual format: only {len(found_sections)} sections found")
        return False
    return True
```

**Priority 2 - Automated Fallback (V):**
When primary extraction fails, try alternative approach:
```python
def extract_with_fallback(report):
    try:
        return primary_extraction_prompt(report)
    except ValidationError:
        logger.info("Primary extraction failed, trying alternative")
        return alternative_extraction_prompt(report)
```

**Priority 3 - Logging (I):**
Captures raw responses for post-mortem analysis of failures.

**Priority 4 - Dashboard (IV):**
Visibility helps but doesn't prevent failures.

**Priority 5 - A/B Testing (II):**
Optimization is premature when reliability isn't yet established. 85% success rate needs fixing before optimizing the 85%.

**Key Insight:** In production systems, prioritize:
1. **Preventing failures** (validation)
2. **Recovering from failures** (fallback)
3. **Debugging failures** (logging)
4. **Monitoring failures** (dashboard)
5. **Optimizing success cases** (A/B testing)

---

## Answer Key Summary

| Question | Answer | Points | Topic |
|----------|--------|--------|-------|
| 1 | C | 10 | Structured extraction with validation |
| 2 | B | 10 | Table extraction to JSON |
| 3 | C | 10 | Multi-layer validation strategy |
| 4 | D | 10 | USDA balance sheet validation |
| 5 | B | 8 | Time-series data structure |
| 6 | B | 7 | Temporal disambiguation |
| 7 | C | 10 | Earnings call sentiment |
| 8 | C | 10 | Efficient transcript processing |
| 9 | B | 12 | Production pipeline design |
| 10 | A | 13 | Production enhancement prioritization |

**Total:** 100 points

---

## Grading Scale

- **90-100:** Excellent - Ready for advanced modules
- **80-89:** Good - Strong understanding with minor gaps
- **70-79:** Adequate - Review validation and production patterns
- **Below 70:** Needs improvement - Revisit extraction and error handling

---

## Common Mistakes to Avoid

1. **Under-specifying extraction schemas** - Always include units, data types, and null handling
2. **Skipping validation** - Never trust LLM output without validation
3. **Ignoring accounting identities** - USDA tables must balance mathematically
4. **Over-engineering** - Simple robust solutions beat complex fragile ones
5. **No error handling** - Production systems must handle API failures gracefully

---

## Next Steps

**Score 90-100:** Proceed to Module 2 (RAG for Commodity Research)
**Score 80-89:** Review validation strategies, then proceed
**Score 70-79:** Practice implementing extraction pipelines with error handling
**Score <70:** Revisit Module 1 guides and complete all notebooks
