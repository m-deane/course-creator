# Quiz: Module 0 - Foundations

**Course:** GenAI for Commodity Markets
**Module:** 0 - Foundations
**Total Points:** 100
**Time Limit:** 25 minutes
**Attempts Allowed:** 2

## Instructions

This quiz assesses your understanding of commodity market fundamentals, LLM capabilities, and development environment setup. Select the best answer for each question. Partial credit may be awarded for multi-part questions.

---

## Section 1: Commodity Market Fundamentals (30 points)

### Question 1 (8 points)

Which of the following best describes the primary difference between physical commodity markets and financial commodity derivatives?

A) Physical markets trade actual commodities for delivery, while derivatives are contracts based on future price movements
B) Physical markets are regulated while derivatives are not
C) Physical markets only trade agricultural products while derivatives trade energy
D) Physical markets are settled daily while derivatives are settled monthly

**Answer: A**

**Explanation:**
- **A (Correct):** Physical markets involve the actual exchange of tangible commodities (oil, grain, metals) for delivery, while derivatives (futures, options, swaps) are financial contracts whose value derives from underlying commodity prices without necessarily involving physical delivery.
- **B:** Both physical and derivatives markets are regulated, with derivatives often more heavily regulated (CFTC in the US).
- **C:** Both market types trade across all commodity sectors (energy, agriculture, metals).
- **D:** Settlement frequency varies by contract type and is not the defining difference.

---

### Question 2 (7 points)

The EIA Weekly Petroleum Status Report (WPSR) provides inventory data for U.S. crude oil stocks. If the report shows crude inventories decreased by 3.2 million barrels and are 5% below the five-year average, what is the most likely immediate market reaction?

A) Bearish - lower inventories indicate oversupply
B) Bullish - lower inventories indicate tight supply conditions
C) Neutral - inventory changes don't affect prices
D) Unpredictable - depends entirely on OPEC decisions

**Answer: B**

**Explanation:**
- **A:** Incorrect interpretation; lower inventories suggest tighter, not looser, supply.
- **B (Correct):** Declining inventories, especially below seasonal averages, typically signal supply tightness and are bullish for prices. Traders interpret inventory drawdowns as indication of strong demand or constrained supply.
- **C:** Inventory data is a critical fundamental indicator that directly impacts price expectations.
- **D:** While OPEC influences prices, inventory data has immediate, measurable market impact independent of OPEC.

---

### Question 3 (8 points)

You're building an LLM system to analyze USDA crop reports. Which data frequency and timing would be MOST critical for a grain trading application?

A) Daily updates on global weather patterns
B) Monthly WASDE reports released mid-month
C) Annual harvest statistics released in December
D) Weekly export sales data

**Answer: B**

**Explanation:**
- **A:** While important, weather data frequency alone doesn't capture the comprehensive supply/demand balance.
- **B (Correct):** The USDA World Agricultural Supply and Demand Estimates (WASDE) report, released monthly around the 10th, is the single most important data release for grain fundamentals. It provides comprehensive supply/demand balance sheets, production forecasts, and ending stocks estimates that drive market expectations.
- **C:** Annual data is too infrequent for active trading strategies.
- **D:** Weekly export sales are useful but secondary to the comprehensive WASDE monthly updates.

---

### Question 4 (7 points)

Which commodity market structure is characterized by higher near-term prices relative to deferred contracts, typically indicating tight current supply?

A) Contango
B) Backwardation
C) Convergence
D) Arbitrage

**Answer: B**

**Explanation:**
- **A:** Contango is the opposite structure (deferred > spot), indicating adequate supply and storage costs.
- **B (Correct):** Backwardation (spot > deferred) reflects tight current supply conditions where immediate delivery commands a premium. This incentivizes producers to sell now and storage holders to release inventory.
- **C:** Convergence refers to spot and futures prices coming together at expiration, not a market structure.
- **D:** Arbitrage is a trading strategy exploiting price differences, not a market structure.

---

## Section 2: LLM Capabilities and Limitations (35 points)

### Question 5 (10 points)

When using an LLM to extract structured data from an EIA report, which of the following outputs would be MOST problematic and require additional validation?

A) Extracting the exact text: "U.S. crude oil production averaged 13.2 million barrels per day"
B) Calculating the percentage change between two explicitly stated inventory levels
C) Summarizing the report's overall tone as bullish or bearish
D) Converting units from "thousand barrels per day" to "million barrels per day"

**Answer: B**

**Explanation:**
- **A:** Direct extraction of stated facts is a strength of LLMs with high reliability.
- **B (Correct):** Mathematical calculations, especially multi-step arithmetic, are a known weakness of LLMs. They may produce incorrect percentages, round improperly, or make arithmetic errors. Critical numerical calculations should be performed programmatically and only use the LLM for extraction of the base numbers.
- **C:** Sentiment analysis is within LLM capabilities, though subject to verification.
- **D:** Unit conversion is typically reliable for standard conversions, though validation is good practice.

**Key Insight:** Use LLMs for pattern recognition and extraction, but implement programmatic logic for calculations and numerical operations.

---

### Question 6 (8 points)

You need to analyze 200 historical USDA reports to extract corn production forecasts over the past 15 years. Which approach is MOST cost-effective and accurate?

A) Send all 200 reports in a single prompt to Claude with a 200k context window
B) Process each report individually with a well-crafted extraction prompt
C) Use prompt caching to store report template structures and process reports individually
D) Fine-tune a small LLM specifically for USDA report extraction

**Answer: C**

**Explanation:**
- **A:** While technically possible with large context windows, sending 200 reports together would be extremely expensive per request and reduce extraction accuracy due to attention dilution.
- **B:** This works but doesn't optimize for the repeated structure of USDA reports, incurring full processing costs for each report.
- **C (Correct):** Prompt caching allows you to cache the instruction template and report format structure, paying full price only for the variable content (each new report). This dramatically reduces costs for repetitive tasks while maintaining high accuracy through individual processing.
- **D:** Fine-tuning is expensive, time-consuming, requires ML expertise, and is overkill when prompt engineering with caching achieves excellent results.

**Cost comparison:** With caching, you might pay 90% less for the cached instruction portion across 200 reports.

---

### Question 7 (9 points)

Which prompt engineering technique would be MOST effective for extracting commodity price forecasts from an analyst report that contains both explicit predictions and implicit sentiment?

```python
# Option A
"Extract any price forecasts from this report."

# Option B
"You are an expert commodity analyst. Read this report and tell me
what the author thinks about future prices."

# Option C
"""Extract price forecasts from this report using the following structure:
{
    "commodity": "<name>",
    "forecast_price": <number or null>,
    "forecast_timeframe": "<timeframe or null>",
    "forecast_direction": "bullish|bearish|neutral",
    "confidence_level": "high|medium|low",
    "key_reasoning": "<brief explanation>"
}

If multiple commodities or timeframes are discussed, return an array.
Include explicit forecasts if stated, or implicit directional views if not."""

# Option D
"Is this report bullish or bearish?"
```

A) Option A - Simple and direct
B) Option B - Establishes expertise
C) Option C - Structured output with clear schema
D) Option D - Most specific question

**Answer: C**

**Explanation:**
- **A:** Too vague; doesn't specify format or handle edge cases (implicit vs explicit forecasts).
- **B:** Role-playing helps but lacks output structure and doesn't distinguish explicit vs implicit forecasts.
- **C (Correct):** Provides clear output schema, handles both explicit and implicit forecasts, specifies data types, and requests confidence levels. The structured format enables programmatic processing and consistent handling of edge cases.
- **D:** Oversimplified; loses numerical forecasts and timeframe information.

**Best Practice:** For data extraction tasks, always specify:
1. Output format (JSON schema)
2. Data types and nullable fields
3. How to handle edge cases
4. Examples if the task is complex

---

### Question 8 (8 points)

An LLM-powered system is extracting natural gas storage data from EIA reports. The model occasionally "hallucinates" plausible-sounding but incorrect storage figures. What is the MOST effective mitigation strategy?

A) Use a larger model like GPT-4 instead of GPT-3.5
B) Implement validation checks against known data ranges and cross-reference with the source API
C) Add "Be accurate and don't make up numbers" to the prompt
D) Run the extraction twice and average the results

**Answer: B**

**Explanation:**
- **A:** Larger models reduce but don't eliminate hallucinations, especially for numerical data.
- **B (Correct):** Implement programmatic validation:
  - Check if extracted values fall within reasonable ranges (e.g., U.S. gas storage: 1,000-4,000 Bcf)
  - Cross-reference with the EIA API when available
  - Flag anomalies for human review
  - Use structured extraction with type validation

  This creates a safety layer independent of model behavior.
- **C:** Prompt instructions have minimal effect on hallucinations; models don't consciously "choose" to hallucinate.
- **D:** Averaging incorrect extractions doesn't improve accuracy; both runs might hallucinate similarly or differently.

**Production Pattern:**
```python
extracted_value = llm_extract(report)
if not validate_range(extracted_value, min_val, max_val):
    flag_for_review(extracted_value)
if api_available:
    api_value = fetch_from_api(report_date)
    if abs(extracted_value - api_value) > threshold:
        log_discrepancy(extracted_value, api_value)
```

---

## Section 3: Development Environment and APIs (35 points)

### Question 9 (10 points)

You're setting up API access for a production commodity analysis system. Rank these API considerations from MOST to LEAST critical:

I. Rate limiting and request throttling
II. API key security and rotation
III. Error handling and retry logic
IV. Cost monitoring and budget alerts

A) II, I, III, IV
B) IV, II, I, III
C) II, III, I, IV
D) I, II, III, IV

**Answer: C**

**Explanation:**

**C (Correct):** Security first, reliability second, performance third, cost fourth.

1. **API Key Security (II):** Compromised keys can lead to unauthorized access, data breaches, massive unexpected bills, and service disruption. Use environment variables, never commit keys, implement rotation policies.

2. **Error Handling and Retry Logic (III):** In production systems, APIs fail. Without proper error handling, a single API timeout could crash your entire pipeline. Implement exponential backoff, circuit breakers, and graceful degradation.

3. **Rate Limiting (I):** Important to avoid service disruption, but typically handled with queuing and backoff strategies. Less critical than security and error handling.

4. **Cost Monitoring (IV):** Important for operations but shouldn't compromise security or reliability. Can be addressed with budget alerts and cost caps.

**Production Checklist:**
- Store API keys in environment variables or secure vaults (AWS Secrets Manager, Azure Key Vault)
- Implement retry logic with exponential backoff
- Add circuit breakers for failing APIs
- Monitor costs but don't let cost concerns compromise security

---

### Question 10 (8 points)

Which Python libraries would form the MINIMUM viable tech stack for building an LLM-powered commodity report processing system?

```python
# Option A
import anthropic  # or openai
import requests
import pandas as pd
import json

# Option B
import anthropic
import langchain
import chromadb
import pytorch
import transformers

# Option C
import anthropic
import requests
import beautifulsoup4
import pandas as pd
import pydantic

# Option D
import openai
import numpy as np
import matplotlib
```

A) Option A
B) Option B
C) Option C
D) Option D

**Answer: C**

**Explanation:**
- **A:** Missing HTML parsing (beautifulsoup4) and structured validation (pydantic). JSON alone isn't sufficient for robust parsing.
- **B:** Overcomplicated for a minimum viable system. Langchain adds abstraction overhead, chromadb and pytorch are unnecessary for basic report processing, transformers is needed only if running local models.
- **C (Correct):** Provides everything needed:
  - `anthropic`: LLM API access
  - `requests`: Fetch reports from APIs/web
  - `beautifulsoup4`: Parse HTML/XML reports
  - `pandas`: Structure and analyze extracted data
  - `pydantic`: Validate extracted data against schemas

  This stack handles fetching, parsing, extracting, validating, and analyzing data.
- **D:** Missing core components (web scraping, data validation) and includes visualization (matplotlib) which isn't essential for processing.

**Minimal Installation:**
```bash
pip install anthropic requests beautifulsoup4 pandas pydantic python-dotenv
```

---

### Question 11 (7 points)

You're designing a system to process EIA reports released every Wednesday at 10:30 AM ET. The LLM extraction takes 2-3 minutes. Which architecture pattern is MOST appropriate?

A) Real-time webhook triggered immediately upon report release
B) Scheduled batch job running every Wednesday at 10:35 AM ET
C) Continuous polling every 5 minutes checking for new reports
D) Manual trigger via a Jupyter notebook

**Answer: B**

**Explanation:**
- **A:** Overengineered; EIA doesn't offer webhooks, and the 5-minute processing window doesn't require real-time architecture.
- **B (Correct):** For predictable, weekly releases, scheduled batch processing is optimal:
  - Reliable and simple
  - Runs at 10:35 AM (5 minutes after release)
  - Allows time for report to be posted
  - Can be implemented with cron, Airflow, or cloud schedulers
  - Efficient resource usage (only runs when needed)
- **C:** Wasteful; polling 10,080 times per week when only 1 release occurs. Increases costs and complexity.
- **D:** Not production-appropriate; requires manual intervention and isn't reliable.

**Implementation Example (Airflow):**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator

dag = DAG(
    'eia_wpsr_processing',
    schedule_interval='35 10 * * 3',  # Every Wed 10:35 AM
    catchup=False
)
```

---

### Question 12 (10 points)

When using Claude with a 200k token context window to analyze multiple commodity reports simultaneously, which scenario provides the BEST cost-to-performance ratio?

**Scenario A:**
Send 5 reports (total 180k tokens) in one prompt asking for comparative analysis

**Scenario B:**
Process each report individually (5 prompts × 40k tokens each) then synthesize results with a 6th prompt

**Scenario C:**
Use prompt caching: Cache common instructions (10k tokens), send 5 individual requests with report content (30k tokens each), then synthesize

**Scenario D:**
Use Claude Opus for comprehensive analysis of all reports together

A) Scenario A - Most efficient single prompt
B) Scenario B - Better accuracy through focused processing
C) Scenario C - Optimal cost and accuracy balance
D) Scenario D - Highest quality model

**Answer: C**

**Explanation:**

**Cost Analysis:**

**Scenario A:**
- Tokens: 180k input
- Cost: ~$1.44 (at $8/MTok for Sonnet)
- Risk: Attention dilution across large context may reduce accuracy
- Latency: Single long processing time

**Scenario B:**
- Tokens: 5 × 40k = 200k input + synthesis prompt
- Cost: ~$1.60 + synthesis
- Benefit: Focused attention on each report
- Latency: 6 sequential API calls

**Scenario C (Correct):**
- Cached tokens: 10k instructions × 1 = 10k (full price) + 10k × 5 = 50k (cached at 90% discount)
- Uncached tokens: 30k × 5 = 150k (full price)
- Total cost: ~$1.28
- Benefits:
  - Focused attention on each report
  - Lower cost than A or B
  - Faster than A (parallel processing possible)
  - Consistent extraction instructions

**Scenario D:**
- Opus costs 5x more than Sonnet
- Cost: ~$7.20
- Only justified if Sonnet fails quality requirements

**Best Practice:** For repetitive structured tasks, prompt caching + individual processing provides optimal cost, accuracy, and latency.

---

## Answer Key Summary

| Question | Answer | Points | Topic |
|----------|--------|--------|-------|
| 1 | A | 8 | Physical vs Derivatives Markets |
| 2 | B | 7 | Inventory Data Interpretation |
| 3 | B | 8 | USDA Report Timing |
| 4 | B | 7 | Market Structure (Backwardation) |
| 5 | B | 10 | LLM Limitations (Math) |
| 6 | C | 8 | Prompt Caching Strategy |
| 7 | C | 9 | Structured Extraction Prompts |
| 8 | B | 8 | Hallucination Mitigation |
| 9 | C | 10 | API Security Priorities |
| 10 | C | 8 | Tech Stack Selection |
| 11 | B | 7 | Batch Processing Architecture |
| 12 | C | 10 | Cost Optimization with Caching |

**Total:** 100 points

---

## Grading Scale

- **90-100:** Excellent - Strong grasp of foundations
- **80-89:** Good - Ready to proceed with minor review
- **70-79:** Adequate - Review weak areas before continuing
- **Below 70:** Needs improvement - Revisit module materials

---

## Learning Recommendations by Score Range

**Below 70:**
- Review Module 0 guides on commodity markets and LLM capabilities
- Complete the environment setup notebook
- Understand prompt engineering basics before proceeding

**70-79:**
- Focus on questions missed in Section 2 (LLM Capabilities)
- Practice structured prompt design
- Review API best practices

**80-89:**
- Solid foundation; minor gaps to address
- Ensure understanding of cost optimization strategies
- Review validation and error handling patterns

**90-100:**
- Excellent preparation for Module 1
- Consider exploring advanced prompt engineering techniques
- Ready for report processing challenges
