# Custom LLM Applications in Dataiku

> **Reading time:** ~5 min | **Module:** 3 — Custom | **Prerequisites:** Module 2 — RAG, Python programming

## Python Recipes with LLMs

Dataiku Python recipes provide full flexibility for custom LLM applications.

### Basic LLM Recipe


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Python Recipe: process_reports
import dataiku
import pandas as pd
from dataiku.llm import LLM

# Input/Output
input_dataset = dataiku.Dataset("raw_reports")
output_dataset = dataiku.Dataset("processed_reports")

# Read input
df = input_dataset.get_dataframe()

# Setup LLM
llm = LLM("anthropic-claude")

def extract_data(report_text: str) -> dict:
    """Extract structured data from report."""
    prompt = f"""Extract the following from this commodity report:
    - Commodity name
    - Key metric
    - Value
    - Change vs prior period
    - Sentiment (bullish/bearish/neutral)

    Report: {report_text}

    Return JSON only."""

    response = llm.complete(prompt, max_tokens=200, temperature=0)

    try:
        import json
        return json.loads(response.text)
    except json.JSONDecodeError:
        return {'error': 'Failed to parse', 'raw': response.text}

# Process each report
results = []
for idx, row in df.iterrows():
    extracted = extract_data(row['report_text'])
    results.append({
        'report_id': row['id'],
        'report_date': row['date'],
        **extracted
    })

# Write output
output_df = pd.DataFrame(results)
output_dataset.write_with_schema(output_df)
```

</div>
</div>

## Batch Processing Patterns

### Parallel Processing


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import dataiku
import pandas as pd
from dataiku.llm import LLM
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_with_llm(row_data: dict, llm_connection: str) -> dict:
    """Process single row with LLM."""
    llm = LLM(llm_connection)

    prompt = f"Analyze: {row_data['text']}"
    response = llm.complete(prompt, max_tokens=200)

    return {
        'id': row_data['id'],
        'analysis': response.text,
        'tokens_used': response.usage.total_tokens
    }

# Read data
input_dataset = dataiku.Dataset("reports")
df = input_dataset.get_dataframe()
rows = df.to_dict('records')

# Process in parallel
results = []
max_workers = 5  # Respect rate limits

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(process_with_llm, row, "anthropic-claude"): row['id']
        for row in rows
    }

    for future in tqdm(as_completed(futures), total=len(futures)):
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            row_id = futures[future]
            results.append({'id': row_id, 'error': str(e)})

# Write results
output_df = pd.DataFrame(results)
dataiku.Dataset("analyzed_reports").write_with_schema(output_df)
```

</div>
</div>

### Chunked Processing for Large Datasets


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import dataiku
import pandas as pd
from dataiku.llm import LLM

def process_in_chunks(input_name: str, output_name: str, chunk_size: int = 100):
    """Process large dataset in chunks."""

    input_ds = dataiku.Dataset(input_name)
    output_ds = dataiku.Dataset(output_name)

    llm = LLM("anthropic-claude")
    first_chunk = True

    # Process chunks
    for chunk_df in input_ds.iter_dataframes(chunksize=chunk_size):
        results = []

        for _, row in chunk_df.iterrows():
            try:
                response = llm.complete(
                    f"Summarize: {row['text']}",
                    max_tokens=100
                )
                results.append({
                    'id': row['id'],
                    'summary': response.text
                })
            except Exception as e:
                results.append({
                    'id': row['id'],
                    'summary': None,
                    'error': str(e)
                })

        # Write chunk
        result_df = pd.DataFrame(results)
        if first_chunk:
            output_ds.write_with_schema(result_df)
            first_chunk = False
        else:
            output_ds.write_dataframe(result_df, infer_schema=False)

process_in_chunks("large_reports", "summarized_reports")
```

</div>
</div>

## Building Custom Endpoints

### API Node Endpoint


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">api_endpoint.py</span>
</div>

```python

# api_endpoint.py - Deploy as API endpoint
import json
from dataiku.llm import LLM
# Pseudocode — KnowledgeBank is not a real Dataiku Python import.
# Verify the Knowledge Bank API against your Dataiku version's docs.
# from dataiku.knowledge_bank import KnowledgeBank  # not a real import

def api_handler(request):
    """Handle API request for commodity Q&A."""

    # Parse request
    body = json.loads(request.get('body', '{}'))
    question = body.get('question', '')
    commodity = body.get('commodity', None)

    if not question:
        return {
            'status_code': 400,
            'body': json.dumps({'error': 'Question required'})
        }

    # Setup RAG
    kb = KnowledgeBank("commodity_reports_kb")
    llm = LLM("anthropic-claude")

    # Retrieve context
    filters = {'commodity': commodity} if commodity else None
    results = kb.search(query=question, top_k=5, filters=filters)

    context = "\n\n".join([r.text for r in results])

    # Generate answer
    prompt = f"""Based on this context, answer the question.
    Context: {context}
    Question: {question}"""

    response = llm.complete(prompt, max_tokens=300)

    return {
        'status_code': 200,
        'body': json.dumps({
            'answer': response.text,
            'sources': [r.metadata.get('source') for r in results]
        })
    }
```

</div>
</div>

### Webapp Backend


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">webapp_backend.py</span>
</div>

```python

# webapp_backend.py - For Dataiku Webapp
# Pseudocode — ChatSession is not a real Dataiku import.
# Multi-turn conversations are managed by maintaining message history
# and passing it to the LLM completion API.
import dataiku
from flask import request, jsonify

# Store conversation histories per session
chat_histories = {}

def get_or_create_history(session_id: str) -> list:
    """Get existing or create new conversation history."""
    if session_id not in chat_histories:
        chat_histories[session_id] = [
            {"role": "system", "content":
             "You are a commodity market analyst. "
             "Provide accurate, data-driven analysis."}
        ]
    return chat_histories[session_id]

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat message."""
    data = request.json
    session_id = data.get('session_id', 'default')
    message = data.get('message', '')

    history = get_or_create_history(session_id)
    history.append({"role": "user", "content": message})

    client = dataiku.api_client()
    project = client.get_default_project()
    llm = project.get_llm("anthropic-claude")
    completion = llm.new_completion()
    for msg in history:
        completion.with_message(msg["content"], role=msg["role"])
    response = completion.execute()

    history.append({"role": "assistant", "content": response.text})

    return jsonify({
        'response': response.text,
        'session_id': session_id
    })

@app.route('/chat/history', methods=['GET'])
def get_history():
    """Get chat history."""
    session_id = request.args.get('session_id', 'default')

    if session_id not in chat_histories:
        return jsonify({'messages': []})

    messages = chat_histories[session_id]

    return jsonify({'messages': messages})
```

</div>
</div>

## Complex Pipelines

### Multi-Stage Processing


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import dataiku
from dataiku.llm import LLM
import pandas as pd

class CommodityAnalysisPipeline:
    """Multi-stage commodity analysis pipeline."""

    def __init__(self, llm_connection: str):
        self.llm = LLM(llm_connection)

    def stage1_extract(self, report_text: str) -> dict:
        """Stage 1: Extract key data points."""
        prompt = f"""Extract from this report:
        - Commodity
        - Key metrics (as list)
        - Timeframe

        Report: {report_text}

        Return JSON."""

        response = self.llm.complete(prompt, max_tokens=200, temperature=0)
        return json.loads(response.text)

    def stage2_analyze(self, extracted: dict) -> dict:
        """Stage 2: Analyze extracted data."""
        prompt = f"""Analyze this commodity data:
        {json.dumps(extracted)}

        Provide:
        - Trend assessment
        - Key drivers
        - Outlook

        Return JSON."""

        response = self.llm.complete(prompt, max_tokens=300)
        return json.loads(response.text)

    def stage3_signal(self, analysis: dict) -> dict:
        """Stage 3: Generate trading signal."""
        prompt = f"""Based on this analysis:
        {json.dumps(analysis)}

        Generate trading signal:
        - Direction: long/short/neutral
        - Confidence: 0-1
        - Timeframe: immediate/short/medium term
        - Key risk

        Return JSON."""

        response = self.llm.complete(prompt, max_tokens=150)
        return json.loads(response.text)

    def process(self, report_text: str) -> dict:
        """Run full pipeline."""
        extracted = self.stage1_extract(report_text)
        analysis = self.stage2_analyze(extracted)
        signal = self.stage3_signal(analysis)

        return {
            'extracted': extracted,
            'analysis': analysis,
            'signal': signal
        }

# Usage in recipe
pipeline = CommodityAnalysisPipeline("anthropic-claude")

input_ds = dataiku.Dataset("reports")
df = input_ds.get_dataframe()

results = []
for _, row in df.iterrows():
    result = pipeline.process(row['report_text'])
    results.append({
        'report_id': row['id'],
        **result['signal']
    })

output_df = pd.DataFrame(results)
dataiku.Dataset("trading_signals").write_with_schema(output_df)
```


## Error Handling

### Robust Processing


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import dataiku
from dataiku.llm import LLM
import json
import time

class RobustLLMProcessor:
    """LLM processor with retry and error handling."""

    def __init__(self, connection: str, max_retries: int = 3):
        self.llm = LLM(connection)
        self.max_retries = max_retries

    def process(self, prompt: str, **kwargs) -> dict:
        """Process with retries and error handling."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.llm.complete(prompt, **kwargs)

                # Try to parse as JSON if expected
                if kwargs.get('expect_json', False):
                    return {
                        'success': True,
                        'data': json.loads(response.text),
                        'tokens': response.usage.total_tokens
                    }
                else:
                    return {
                        'success': True,
                        'text': response.text,
                        'tokens': response.usage.total_tokens
                    }

            except json.JSONDecodeError as e:
                # Retry with clearer instruction
                prompt = prompt + "\n\nIMPORTANT: Return valid JSON only."
                last_error = e

            except Exception as e:
                last_error = e
                time.sleep(2 ** attempt)  # Exponential backoff

        return {
            'success': False,
            'error': str(last_error),
            'attempts': self.max_retries
        }
```


## Key Takeaways

1. **Python recipes** provide full flexibility for custom LLM logic

2. **Parallel processing** speeds up batch operations while respecting rate limits

3. **Chunked processing** handles large datasets efficiently

4. **API endpoints** expose LLM capabilities as services

5. **Error handling** is essential for production reliability

<div class="callout-key">

<strong>Key Concept:</strong> 5. **Error handling** is essential for production reliability





## Resources

<a class="link-card" href="../notebooks/01_python_llm_calls.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
