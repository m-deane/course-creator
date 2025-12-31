# Module 3: Custom LLM Applications

## Overview

Build custom Gen AI applications using Python recipes, custom models, and the Dataiku API. Extend beyond visual tools for advanced use cases.

**Time Estimate:** 8-10 hours

## Learning Objectives

By completing this module, you will:
1. Call LLM Mesh from Python recipes
2. Build custom model wrappers
3. Create reusable LLM components
4. Integrate with Dataiku pipelines

## Contents

### Guides
- `01_python_integration.md` - LLM Mesh in Python
- `02_custom_models.md` - Building custom wrappers
- `03_pipeline_integration.md` - LLMs in data pipelines

### Notebooks
- `01_python_llm_calls.ipynb` - Python integration
- `02_custom_application.ipynb` - Building custom apps

## Key Concepts

### Python LLM Mesh API

```python
import dataiku
from dataiku import LLMHandle

# Get LLM connection
llm = LLMHandle("my-claude-connection")

# Generate completion
response = llm.generate(
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Summarize this text..."}
    ],
    temperature=0.7,
    max_tokens=1000
)

print(response.text)
print(response.usage)  # Token counts
```

### Custom Model Patterns

| Pattern | Use Case |
|---------|----------|
| Wrapper | Add pre/post processing |
| Chain | Multi-step reasoning |
| Router | Dynamic model selection |
| Ensemble | Multiple model consensus |

### Pipeline Integration

```
Dataset → Python Recipe → LLM Processing → Output Dataset
              ↓
         LLM Mesh
              ↓
         Prompt Studio
```

## Prerequisites

- Module 0-2 completed
- Python proficiency
