# Environment Setup for GenAI Commodities Trading

> **Reading time:** ~6 min | **Module:** Module 0: Foundations | **Prerequisites:** Basic linear algebra, Python

<div class="callout-key">

**Key Concept Summary:** Setting up the development environment for applying generative AI to commodity markets requires API access to both LLM providers and commodity data sources, along with Python libraries for data processing and machine learning.

</div>

## In Brief

Setting up the development environment for applying generative AI to commodity markets requires API access to both LLM providers and commodity data sources, along with Python libraries for data processing and machine learning.

<div class="callout-insight">

**Insight:** Successful Gen AI commodities work requires three layers: (1) data access to market information, (2) LLM infrastructure for processing unstructured content, and (3) analytical tools for signal generation. Most failures stem from inadequate API setup or missing dependencies.

</div>
<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Required Components

### 1. LLM API Access

**Primary: Anthropic Claude**
```bash

# Sign up at: https://console.anthropic.com/

# Set environment variable
export ANTHROPIC_API_KEY="sk-ant-api..."
```

**Alternative: OpenAI GPT**
```bash

# Sign up at: https://platform.openai.com/
export OPENAI_API_KEY="sk-..."
```

**Cost Considerations:**
- Anthropic Claude 3.5 Sonnet: $3/1M input tokens, $15/1M output tokens
- OpenAI GPT-4o: $2.50/1M input tokens, $10/1M output tokens
- Budget for development: $50-100/month for moderate usage

### 2. Commodity Data APIs

**Energy Information Administration (EIA)**
```bash

# Free registration: https://www.eia.gov/opendata/register.php
export EIA_API_KEY="your_eia_key"

# Key datasets:

# - Weekly Petroleum Status Report (WPSR)

# - Natural Gas Weekly Update

# - Short-Term Energy Outlook (STEO)
```

**USDA Agricultural Data**
```bash

# No API key required for most reports

# Key sources:

# - WASDE (World Agricultural Supply and Demand Estimates)

# - Crop Progress Reports

# - Export Sales Reports

# Direct download URLs available
```

**NOAA Weather Data (Optional)**
```bash

# For weather-dependent commodities (natural gas, agriculture)

# Register at: https://www.ncdc.noaa.gov/cdo-web/token
export NOAA_TOKEN="your_token"
```

### 3. Market Data (Optional)

**For Price Data:**
- **Free:** Yahoo Finance API (via yfinance)
- **Paid:** Bloomberg API (subscription required)
- **Paid:** Quandl/Nasdaq Data Link

```bash

# Free options don't require API keys
pip install yfinance

# Quandl (free tier available)
export QUANDL_API_KEY="your_key"
```

## Installation

### Python Environment

**Requirements:**
- Python 3.9+
- Virtual environment manager (venv, conda, or poetry)

**Create Virtual Environment:**
```bash

# Using venv
python -m venv genai-commodities
source genai-commodities/bin/activate  # On Windows: genai-commodities\Scripts\activate

# Using conda
conda create -n genai-commodities python=3.11
conda activate genai-commodities
```

### Core Dependencies

**Create requirements.txt:**
```txt

# LLM Frameworks
anthropic>=0.25.0
openai>=1.30.0
langchain>=0.2.0
langchain-anthropic>=0.1.0

# Vector Databases
chromadb>=0.4.0
pinecone-client>=3.0.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
requests>=2.31.0
beautifulsoup4>=4.12.0

# Visualization
matplotlib>=3.7.0
plotly>=5.14.0
seaborn>=0.12.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
feedparser>=6.0.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0

# Optional: Advanced features
langsmith>=0.1.0  # For LLM observability
instructor>=0.2.0  # For structured outputs
```

**Install:**
```bash
pip install -r requirements.txt
```

### Configuration

**Create .env file:**
```bash

# .env (never commit this file!)
ANTHROPIC_API_KEY=sk-ant-api...
OPENAI_API_KEY=sk-...
EIA_API_KEY=your_eia_key
NOAA_TOKEN=your_noaa_token
QUANDL_API_KEY=your_quandl_key

# Vector database (if using Pinecone)
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=us-west1-gcp

# Optional: LangSmith for monitoring
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=genai-commodities
```

**Load in Python:**
```python
from dotenv import load_dotenv
import os

load_dotenv()

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
EIA_API_KEY = os.getenv('EIA_API_KEY')
```

## Verification

### Test LLM Access

```python
from anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=100,
    messages=[{
        "role": "user",
        "content": "What are the main factors affecting crude oil prices?"
    }]
)

print(response.content[0].text)

# Should return a coherent response about oil price drivers
```

### Test EIA API Access

```python
import requests
import os

EIA_API_KEY = os.getenv('EIA_API_KEY')

response = requests.get(
    "https://api.eia.gov/v2/petroleum/sum/sndw/data",
    params={
        'api_key': EIA_API_KEY,
        'frequency': 'weekly',
        'data[0]': 'value',
        'facets[series][]': 'WCESTUS1',  # Crude oil stocks
        'length': 1
    }
)

if response.status_code == 200:
    data = response.json()
    latest_value = data['response']['data'][0]['value']
    print(f"Latest US crude oil stocks: {latest_value} thousand barrels")
else:
    print(f"Error: {response.status_code}")
```

### Test Vector Database

**ChromaDB (local, no API key needed):**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("test")
collection.add(
    documents=["Crude oil inventories declined"],
    ids=["test1"]
)

results = collection.query(
    query_texts=["oil stocks"],
    n_results=1
)
print(results)

# Should return the document
```

</div>
</div>

## Directory Structure

**Recommended project layout:**
```
genai-commodities/
├── .env                    # API keys (gitignored)
├── .gitignore
├── requirements.txt
├── README.md
│
├── data/                   # Raw and processed data
│   ├── raw/
│   ├── processed/
│   └── cache/
│
├── notebooks/              # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_llm_experiments.ipynb
│   └── ...
│
├── src/                    # Source code
│   ├── __init__.py
│   ├── data/              # Data fetching
│   ├── llm/               # LLM utilities
│   ├── analysis/          # Analysis functions
│   └── signals/           # Signal generation
│
├── tests/                  # Unit tests
│   ├── test_data.py
│   ├── test_llm.py
│   └── ...
│
└── outputs/                # Results and reports
    ├── reports/
    ├── signals/
    └── figures/
```

## Common Pitfalls

**1. API Rate Limits**
- **Issue:** Exceeding free tier limits or hitting rate limits
- **Solution:** Implement caching, use exponential backoff, monitor usage

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">retry_with_backoff.py</span>
</div>

```python
import time
from functools import wraps

def retry_with_backoff(max_retries=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    wait = 2 ** attempt
                    print(f"Retry {attempt + 1}/{max_retries} after {wait}s")
                    time.sleep(wait)
        return wrapper
    return decorator
```

</div>
</div>

**2. Environment Variables Not Loading**
- **Issue:** API keys not found even after setting in .env
- **Solution:** Ensure python-dotenv is installed and load_dotenv() is called before accessing variables
```python
from dotenv import load_dotenv
load_dotenv()  # Must be called before os.getenv()
```

**3. ChromaDB Persistence Issues**
- **Issue:** Vector database data lost between sessions
- **Solution:** Use persistent client

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import chromadb

# Persistent storage
client = chromadb.PersistentClient(path="./chroma_db")
```

</div>
</div>

**4. Large Response Truncation**
- **Issue:** LLM responses cut off mid-sentence
- **Solution:** Increase max_tokens parameter

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,  # Increase from default 1024
    messages=[...]
)
```

</div>
</div>

**5. JSON Parsing Failures**
- **Issue:** LLM returns malformed JSON
- **Solution:** Use libraries like instructor or manual validation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">safe_json_parse.py</span>

```python
import json
import re as re_mod

def safe_json_parse(text):
    """Extract and parse JSON from LLM response."""
    try:
        # Try direct parse first
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        fence = "`" * 3
        pattern = fence + r"json\n(.*?)\n" + fence
        json_match = re_mod.search(pattern, text, re_mod.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        raise ValueError("No valid JSON found in response")
```


## Connections

**Builds on:**
- Python programming fundamentals
- REST API concepts
- Environment variable management

**Leads to:**
- Module 1: Report Processing (using these APIs)
- Module 2: RAG Research (vector databases configured here)
- All subsequent modules rely on this foundation

**Related to:**
- DevOps practices (environment management)
- Security (API key protection)
- Cost optimization (API usage monitoring)

## Practice Problems

1. **Basic Setup:**
   - Set up your environment with all required API keys
   - Verify access to both Anthropic and EIA APIs
   - Create a simple script that fetches crude oil inventory data and summarizes it using an LLM

2. **Error Handling:**
   - Implement a robust data fetching function with retry logic
   - Add logging to track API calls and failures
   - Handle cases where API keys are missing or invalid

3. **Cost Tracking:**
   - Create a token usage tracker for LLM calls
   - Estimate monthly costs based on expected usage patterns
   - Design a caching strategy to minimize redundant API calls

<div class="callout-insight">

**Insight:** Understanding environment setup for genai commodities trading is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.


## Further Reading

- **Anthropic API Documentation:** https://docs.anthropic.com/
  - Comprehensive guide to Claude API, includes best practices and examples

- **EIA API Guide:** https://www.eia.gov/opendata/documentation.php
  - Detailed API v2 documentation with all available datasets

- **LangChain Documentation:** https://python.langchain.com/docs/get_started/introduction
  - Framework for building LLM applications, useful for complex workflows

- **ChromaDB Docs:** https://docs.trychroma.com/
  - Vector database documentation, essential for RAG systems

- **Python dotenv Guide:** https://github.com/theskumar/python-dotenv
  - Best practices for environment variable management

- **API Security Best Practices:**
  - https://owasp.org/www-project-api-security/
  - Critical reading for protecting API keys and handling sensitive data

---

## Conceptual Practice Questions

1. What API configuration decisions affect cost vs. quality tradeoffs for commodity LLM applications?

2. Why is structured output (JSON mode) important for downstream pipeline integration?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---

## Cross-References

<a class="link-card" href="./03_environment_setup_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_market_data_access.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_llm_fundamentals.md">
  <div class="link-card-title">01 Llm Fundamentals</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_prompt_engineering_basics.md">
  <div class="link-card-title">02 Prompt Engineering Basics</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

