# Integrating LLMs into Dataiku Pipelines

> **Reading time:** ~11 min | **Module:** 3 — Custom | **Prerequisites:** Module 2 — RAG, Python programming

## In Brief

LLM integration into Dataiku data pipelines transforms unstructured text into structured, actionable data at scale. By embedding LLM processing steps alongside traditional ETL operations, you create end-to-end workflows that combine data engineering and generative AI capabilities in a unified, governed environment.

<div class="callout-insight">

<strong>Key Insight:</strong> The real power of LLMs in data pipelines comes from treating them as transformation steps—not standalone applications. When LLMs process data as it flows through pipelines, you gain automatic dependency tracking, version control, scheduling, monitoring, and governance that would require extensive custom engineering otherwise.

</div>

<div class="callout-key">

<strong>Key Concept:</strong> LLM integration into Dataiku data pipelines transforms unstructured text into structured, actionable data at scale. By embedding LLM processing steps alongside traditional ETL operations, you create end-to-end workflows that combine data engineering and generative AI capabilities in a unified, go...

</div>

## Formal Definition

**Pipeline Integration** is the practice of embedding LLM operations within data processing workflows:
- **Dataiku Flow**: Visual representation of data dependencies from sources to LLM processing to outputs
- **Recipe-Based Processing**: Python, LLM, or Visual recipes that apply LLMs to datasets
- **Batch Processing**: Scheduled execution on datasets of any size with parallelization
- **Dependency Management**: Automatic tracking of upstream changes triggering downstream LLM processing
- **Monitoring**: Built-in metrics for token usage, costs, errors, and processing time

## Intuitive Explanation

Think of a manufacturing assembly line: raw materials enter, pass through various processing stations (cutting, welding, painting), and emerge as finished products. A data pipeline with LLM integration works the same way: raw text data enters, passes through preprocessing (cleaning, chunking), LLM processing (extraction, analysis), postprocessing (validation, formatting), and emerges as structured insights. Each station knows its inputs and outputs, enabling automation and quality control at scale.

## Visual Representation

```

┌─────────────────────────────────────────────────────────────┐
│              Dataiku Pipeline with LLM Integration          │
└─────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │ Raw Reports  │  (Dataset: unstructured text)
  │   Dataset    │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Preprocess   │  (Python Recipe: clean, chunk)
  │   Recipe     │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │   LLM        │  (LLM Recipe: extract structured data)
  │  Processing  │   • Uses Prompt Studio
  │   Recipe     │   • Parallel execution
  └──────┬───────┘   • Error handling
         │
         ▼
  ┌──────────────┐
  │ Validate &   │  (Python Recipe: validate, enrich)
  │  Transform   │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Structured   │  (Dataset: JSON/tabular output)
  │    Output    │
  └──────────────┘
```

## Code Implementation

### Basic LLM Recipe Integration


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Python Recipe: process_reports_with_llm
import dataiku
from dataiku.llm import LLM
import pandas as pd
from typing import Dict, Any

# Input dataset
input_dataset = dataiku.Dataset("raw_market_reports")
df_input = input_dataset.get_dataframe()

# LLM setup
llm = LLM("anthropic-claude")

# Prompt template
ANALYSIS_PROMPT = """Analyze this {commodity} market report:

{report_text}

Extract:
- inventory_change: number (million barrels/Bcf)
- production: number
- sentiment: bullish/bearish/neutral
- key_factors: list

Return JSON only."""

def process_row(row: pd.Series) -> Dict[str, Any]:
    """Process single row with LLM."""
    try:
        prompt = ANALYSIS_PROMPT.format(
            commodity=row['commodity'],
            report_text=row['report_text']
        )

        response = llm.complete(
            prompt=prompt,
            temperature=0,
            max_tokens=500
        )

        # Parse JSON response
        import json
        data = json.loads(response.text)

        return {
            'report_id': row['report_id'],
            'status': 'success',
            'tokens_used': response.usage.total_tokens,
            'cost_usd': response.cost,
            **data  # Unpack extracted fields
        }

    except Exception as e:
        return {
            'report_id': row['report_id'],
            'status': 'error',
            'error_message': str(e),
            'tokens_used': 0,
            'cost_usd': 0
        }

# Process all rows
results = df_input.apply(process_row, axis=1, result_type='expand')

# Output dataset
output_dataset = dataiku.Dataset("analyzed_reports")
output_dataset.write_with_schema(results)

# Log summary
print(f"Processed {len(results)} reports")
print(f"Success: {(results['status'] == 'success').sum()}")
print(f"Total tokens: {results['tokens_used'].sum()}")
print(f"Total cost: ${results['cost_usd'].sum():.2f}")
```

</div>
</div>

### Parallel Batch Processing


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Python Recipe: parallel_llm_processing
import dataiku
from dataiku.llm import LLM
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class BatchLLMProcessor:
    """
    Parallel batch processing with LLMs.
    """

    def __init__(
        self,
        connection_name: str,
        max_workers: int = 10,
        batch_size: int = 100
    ):
        self.connection_name = connection_name
        self.max_workers = max_workers
        self.batch_size = batch_size

    def process_batch(
        self,
        df: pd.DataFrame,
        prompt_template: str,
        variable_columns: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Process DataFrame in parallel batches.

        Args:
            df: Input DataFrame
            prompt_template: Template with {variable} placeholders
            variable_columns: Map template vars to DataFrame columns

        Returns:
            DataFrame with LLM outputs
        """
        # Initialize LLM per worker thread
        def get_llm():
            return LLM(self.connection_name)

        def process_row(row_tuple):
            """Process single row (called in parallel)."""
            idx, row = row_tuple
            llm = get_llm()

            try:
                # Build prompt from template and row data
                prompt_vars = {
                    var: row[col]
                    for var, col in variable_columns.items()
                }
                prompt = prompt_template.format(**prompt_vars)

                # Call LLM
                response = llm.complete(prompt, temperature=0, max_tokens=500)

                return {
                    'index': idx,
                    'status': 'success',
                    'output': response.text,
                    'tokens': response.usage.total_tokens,
                    'cost': response.cost
                }

            except Exception as e:
                logger.error(f"Row {idx} failed: {e}")
                return {
                    'index': idx,
                    'status': 'error',
                    'error': str(e),
                    'output': None,
                    'tokens': 0,
                    'cost': 0
                }

        # Process in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all rows
            futures = {
                executor.submit(process_row, row_tuple): row_tuple[0]
                for row_tuple in df.iterrows()
            }

            # Collect results as they complete
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                # Progress logging
                if len(results) % 100 == 0:
                    logger.info(f"Processed {len(results)}/{len(df)} rows")

        # Convert to DataFrame
        result_df = pd.DataFrame(results).set_index('index')

        # Merge with original DataFrame
        output_df = df.join(result_df)

        # Summary
        success_count = (output_df['status'] == 'success').sum()
        total_tokens = output_df['tokens'].sum()
        total_cost = output_df['cost'].sum()

        logger.info(f"""
Batch Processing Complete:
  Success: {success_count}/{len(df)} ({success_count/len(df)*100:.1f}%)
  Total tokens: {total_tokens:,}
  Total cost: ${total_cost:.2f}
  Avg tokens/row: {total_tokens/len(df):.0f}
        """)

        return output_df

# Usage in recipe
processor = BatchLLMProcessor(
    connection_name='anthropic-claude',
    max_workers=10  # Process 10 rows concurrently
)

input_df = dataiku.Dataset("raw_reports").get_dataframe()

output_df = processor.process_batch(
    df=input_df,
    prompt_template="Analyze this {commodity} report:\n\n{report_text}",
    variable_columns={
        'commodity': 'commodity_type',
        'report_text': 'full_text'
    }
)

dataiku.Dataset("analyzed_reports").write_with_schema(output_df)
```

</div>
</div>

### Chunked Processing for Large Texts


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Python Recipe: chunk_and_process
from typing import List

def chunk_text(
    text: str,
    max_chunk_tokens: int = 3000,
    overlap_tokens: int = 200
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Full text to chunk
        max_chunk_tokens: Max tokens per chunk
        overlap_tokens: Overlap between chunks

    Returns:
        List of text chunks
    """
    # Rough estimate: 4 chars per token
    max_chars = max_chunk_tokens * 4
    overlap_chars = overlap_tokens * 4

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars

        # Try to break on sentence boundary
        if end < len(text):
            # Look for sentence end in last 20% of chunk
            search_start = end - (max_chars // 5)
            sentence_end = text.rfind('. ', search_start, end)

            if sentence_end > start:
                end = sentence_end + 1

        chunks.append(text[start:end])

        # Move start with overlap
        start = end - overlap_chars if end < len(text) else end

    return chunks

def process_with_chunking(
    df: pd.DataFrame,
    text_column: str,
    llm: LLM,
    chunk_size: int = 3000
) -> pd.DataFrame:
    """
    Process long texts by chunking.

    Args:
        df: Input DataFrame
        text_column: Column containing long text
        llm: LLM instance
        chunk_size: Max tokens per chunk

    Returns:
        DataFrame with aggregated results
    """
    results = []

    for idx, row in df.iterrows():
        text = row[text_column]
        chunks = chunk_text(text, max_chunk_tokens=chunk_size)

        logger.info(f"Row {idx}: {len(text)} chars -> {len(chunks)} chunks")

        # Process each chunk
        chunk_results = []
        total_tokens = 0
        total_cost = 0

        for i, chunk in enumerate(chunks):
            prompt = f"""Analyze this section (part {i+1}/{len(chunks)}) of a market report:

{chunk}

Extract key points and metrics."""

            response = llm.complete(prompt, max_tokens=500)

            chunk_results.append(response.text)
            total_tokens += response.usage.total_tokens
            total_cost += response.cost

        # Synthesize chunk results
        synthesis_prompt = f"""Synthesize these {len(chunks)} analyses into a unified view:

""" + "\n\n---\n\n".join(chunk_results) + """

Provide consolidated analysis with no duplication."""

        final_response = llm.complete(synthesis_prompt, max_tokens=800)

        results.append({
            'id': row['id'],
            'num_chunks': len(chunks),
            'chunk_summaries': chunk_results,
            'final_analysis': final_response.text,
            'total_tokens': total_tokens + final_response.usage.total_tokens,
            'total_cost': total_cost + final_response.cost
        })

    return pd.DataFrame(results)

# Usage
input_df = dataiku.Dataset("long_reports").get_dataframe()
llm = LLM("anthropic-claude")

output_df = process_with_chunking(
    df=input_df,
    text_column='full_report_text',
    llm=llm,
    chunk_size=3000
)

dataiku.Dataset("processed_long_reports").write_with_schema(output_df)
```

</div>
</div>

### Pipeline with Error Handling and Monitoring


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Python Recipe: robust_pipeline_integration
import dataiku
from dataiku.llm import LLM
import pandas as pd
from typing import Optional
import time
from datetime import datetime

class RobustLLMPipeline:
    """
    Production-grade LLM pipeline integration.
    """

    def __init__(
        self,
        connection_name: str,
        metrics_dataset: Optional[str] = None
    ):
        self.llm = LLM(connection_name)
        self.metrics_dataset = metrics_dataset
        self.metrics = []

    def process_with_retry(
        self,
        prompt: str,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> dict:
        """Process with retry logic."""
        last_error = None

        for attempt in range(max_retries):
            try:
                start_time = time.time()

                response = self.llm.complete(
                    prompt=prompt,
                    temperature=0,
                    max_tokens=500
                )

                latency = time.time() - start_time

                # Record metrics
                self.metrics.append({
                    'timestamp': datetime.now(),
                    'status': 'success',
                    'attempt': attempt + 1,
                    'tokens': response.usage.total_tokens,
                    'cost': response.cost,
                    'latency_sec': latency
                })

                return {
                    'status': 'success',
                    'output': response.text,
                    'tokens': response.usage.total_tokens,
                    'cost': response.cost,
                    'attempts': attempt + 1
                }

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

                # Record failure
                self.metrics.append({
                    'timestamp': datetime.now(),
                    'status': 'error',
                    'attempt': attempt + 1,
                    'error': str(e),
                    'tokens': 0,
                    'cost': 0
                })

                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))

        # All retries failed
        return {
            'status': 'error',
            'error': str(last_error),
            'output': None,
            'tokens': 0,
            'cost': 0,
            'attempts': max_retries
        }

    def process_dataset(
        self,
        input_dataset_name: str,
        output_dataset_name: str,
        prompt_template: str,
        variable_mapping: dict
    ) -> dict:
        """
        Process entire dataset with monitoring.

        Returns:
            Summary statistics
        """
        logger.info(f"Starting pipeline: {input_dataset_name} -> {output_dataset_name}")

        # Load input
        input_df = dataiku.Dataset(input_dataset_name).get_dataframe()
        logger.info(f"Loaded {len(input_df)} rows")

        # Process rows
        results = []
        for idx, row in input_df.iterrows():
            # Build prompt
            prompt_vars = {
                var: row[col]
                for var, col in variable_mapping.items()
            }
            prompt = prompt_template.format(**prompt_vars)

            # Process with retry
            result = self.process_with_retry(prompt)

            results.append({
                'row_id': idx,
                **result
            })

            # Progress
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(input_df)} rows")

        # Create output DataFrame
        output_df = pd.DataFrame(results)

        # Write output
        dataiku.Dataset(output_dataset_name).write_with_schema(output_df)

        # Write metrics if configured
        if self.metrics_dataset:
            metrics_df = pd.DataFrame(self.metrics)
            dataiku.Dataset(self.metrics_dataset).write_with_schema(metrics_df)

        # Summary statistics
        summary = {
            'total_rows': len(output_df),
            'successful': (output_df['status'] == 'success').sum(),
            'failed': (output_df['status'] == 'error').sum(),
            'total_tokens': output_df['tokens'].sum(),
            'total_cost': output_df['cost'].sum(),
            'avg_attempts': output_df['attempts'].mean(),
            'success_rate': (output_df['status'] == 'success').mean()
        }

        logger.info(f"""
Pipeline Complete:
  Total rows: {summary['total_rows']}
  Successful: {summary['successful']} ({summary['success_rate']*100:.1f}%)
  Failed: {summary['failed']}
  Total tokens: {summary['total_tokens']:,}
  Total cost: ${summary['total_cost']:.2f}
  Avg retries: {summary['avg_attempts']:.1f}
        """)

        return summary

# Usage
pipeline = RobustLLMPipeline(
    connection_name='anthropic-claude',
    metrics_dataset='llm_processing_metrics'
)

summary = pipeline.process_dataset(
    input_dataset_name='raw_reports',
    output_dataset_name='analyzed_reports',
    prompt_template="Analyze: {report_text}",
    variable_mapping={'report_text': 'content'}
)
```

</div>

### Incremental Processing Pattern


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# Python Recipe: incremental_llm_processing
import dataiku
from datetime import datetime, timedelta

def incremental_llm_processing(
    input_dataset_name: str,
    output_dataset_name: str,
    timestamp_column: str,
    lookback_days: int = 1
):
    """
    Process only new/updated records since last run.

    Args:
        input_dataset_name: Source dataset
        output_dataset_name: Output dataset
        timestamp_column: Column tracking record updates
        lookback_days: How far back to process
    """
    # Load existing output to identify processed records
    try:
        output_df = dataiku.Dataset(output_dataset_name).get_dataframe()
        last_processed = output_df[timestamp_column].max()
        logger.info(f"Last processed timestamp: {last_processed}")
    except:
        # First run - no existing output
        last_processed = datetime.now() - timedelta(days=lookback_days)
        output_df = pd.DataFrame()
        logger.info("First run - processing all records")

    # Load input - only new records
    input_df = dataiku.Dataset(input_dataset_name).get_dataframe()
    input_df[timestamp_column] = pd.to_datetime(input_df[timestamp_column])

    new_records = input_df[input_df[timestamp_column] > last_processed]

    logger.info(f"Found {len(new_records)} new records to process")

    if len(new_records) == 0:
        logger.info("No new records - skipping processing")
        return

    # Process new records with LLM
    llm = LLM("anthropic-claude")
    results = []

    for idx, row in new_records.iterrows():
        response = llm.complete(
            prompt=f"Analyze: {row['text']}",
            max_tokens=500
        )

        results.append({
            'id': row['id'],
            'processed_at': datetime.now(),
            'output': response.text,
            'tokens': response.usage.total_tokens,
            'cost': response.cost
        })

    # Append to existing output
    new_output_df = pd.DataFrame(results)
    combined_df = pd.concat([output_df, new_output_df], ignore_index=True)

    # Write back
    dataiku.Dataset(output_dataset_name).write_with_schema(combined_df)

    logger.info(f"Processed {len(results)} new records")

# Run incremental processing
incremental_llm_processing(
    input_dataset_name='market_reports',
    output_dataset_name='analyzed_reports',
    timestamp_column='report_date',
    lookback_days=1
)
```


## Common Pitfalls

**Pitfall 1: Not Handling Dataset Size**
- Processing 100,000 rows sequentially takes days and costs thousands
- Always use parallel processing with ThreadPoolExecutor or Dask
- Consider incremental processing for regularly updated datasets

**Pitfall 2: No Error Isolation**
- One failed LLM call shouldn't crash the entire pipeline
- Wrap each row's processing in try/except
- Store errors with partial results for debugging

**Pitfall 3: Missing Cost Controls**
- Running unmonitored pipelines can exhaust budgets quickly
- Implement cost estimates before processing
- Add budget checks that stop processing if threshold exceeded

**Pitfall 4: Ignoring Token Limits**
- Long documents exceed model context windows causing errors
- Implement automatic chunking for texts over limit
- Synthesize chunk results into final output

**Pitfall 5: No Observability**
- Pipelines without metrics are black boxes
- Log token usage, costs, errors, and latency for every run
- Create monitoring dashboards for production pipelines

## Connections

<div class="callout-info">

<strong>How this connects to the rest of the course:</strong>


**Builds on:**
- LLM Mesh Python integration (Module 0, Module 3.1)
- Custom model wrappers (Module 3.2)

**Leads to:**
- Production deployment patterns (Module 4)
- Monitoring and cost optimization (Module 4)

**Related to:**
- ETL and data pipeline design
- Batch processing architectures
- Error handling and resilience patterns



## Practice Problems

1. **Basic Pipeline**
   - Create a 3-step pipeline: preprocess text → LLM analysis → validation
   - Process a dataset of 100 documents
   - Calculate total cost and average processing time

2. **Parallel Processing**
   - Implement parallel processing for 1,000 documents
   - Compare execution time with 1, 5, 10, and 20 workers
   - Identify optimal parallelism level for your infrastructure

3. **Incremental Updates**
   - Design an incremental processing pattern for daily report analysis
   - Only process new reports since last run
   - Maintain historical analysis in output dataset

4. **Chunking Strategy**
   - Process documents averaging 20,000 tokens (exceeding context limit)
   - Implement chunking with overlap
   - Synthesize chunk results into coherent final analysis

5. **Pipeline Monitoring**
   - Add comprehensive monitoring to an LLM pipeline
   - Track: success rate, avg tokens, cost, latency, error types
   - Create a metrics dataset and build a monitoring dashboard

## Further Reading

- **Dataiku Documentation**: [LLM Recipes](https://doc.dataiku.com/dss/latest/generative-ai/llm-recipes.html) - Official guide to LLM integration in recipes

- **Dataiku Documentation**: [Python Recipes](https://doc.dataiku.com/dss/latest/code_recipes/python.html) - Python recipe fundamentals

- **Martin Kleppmann**: "Designing Data-Intensive Applications" - Principles applicable to LLM data pipelines

- **Blog Post**: "Building Reliable LLM Pipelines at Scale" - Production patterns from real deployments (representative of industry practices)

- **Research**: "Batch Processing Optimization for LLM Workloads" - Techniques for efficient large-scale processing (representative of ongoing optimization work)


## Resources

<a class="link-card" href="../notebooks/01_python_llm_calls.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>
