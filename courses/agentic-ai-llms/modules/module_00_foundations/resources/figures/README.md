# Figures Directory

This directory contains visual diagrams and figures for Module 0: Foundations.

## Recommended Diagrams

The following diagrams should be created to support this module's content:

1. **transformer_architecture.png**
   - Complete transformer block showing encoder-decoder structure
   - Attention mechanism visualization
   - Multi-head attention detail

2. **attention_mechanism.png**
   - Query-Key-Value computation flow
   - Attention score calculation and softmax
   - Weighted value aggregation

3. **tokenization_process.png**
   - Text to token conversion example
   - Subword tokenization (BPE) visualization
   - Token ID mapping

4. **context_window_comparison.png**
   - Bar chart comparing context windows across models
   - Visual representation of token limits
   - Cost per token comparison

5. **llm_inference_flow.png**
   - Input text processing pipeline
   - Embedding generation
   - Autoregressive token generation
   - Output text reconstruction

6. **token_economics.png**
   - Cost breakdown by model and task type
   - Input vs output token pricing
   - Context window utilization strategies

## Usage

Reference these figures in notebooks and guides using:
```markdown
![Description](../resources/figures/filename.png)
```

For interactive notebooks:
```python
from IPython.display import Image, display
display(Image('../resources/figures/filename.png'))
```
