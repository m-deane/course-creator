# Figures Directory

This directory contains visual diagrams and figures for Module 1: LLM Fundamentals for Agents.

## Recommended Diagrams

The following diagrams should be created to support this module's content:

1. **prompt_hierarchy.png**
   - Visual flow showing System Prompt → Few-Shot Examples → User Message → Assistant Response
   - Context injection points
   - Message ordering and persistence

2. **chain_of_thought_comparison.png**
   - Side-by-side comparison of direct answer vs CoT reasoning
   - Example problem with reasoning trace
   - Accuracy improvements visualization

3. **system_prompt_components.png**
   - Breakdown of effective system prompt structure
   - Identity, Capabilities, Constraints, Format, Tools sections
   - Example annotations

4. **cot_variants_flowchart.png**
   - Decision tree for choosing CoT variant
   - Zero-shot vs Few-shot vs Self-consistency vs Tree-of-Thought
   - Use case mapping

5. **few_shot_pattern.png**
   - Example structure with input-output pairs
   - Pattern recognition visualization
   - Optimal number of examples guidance

6. **token_optimization_strategies.png**
   - Before/after examples of prompt compression
   - Token count reduction techniques
   - Quality vs cost trade-offs

7. **reasoning_quality_spectrum.png**
   - Spectrum from simple Q&A to complex multi-step reasoning
   - Appropriate technique for each complexity level
   - Performance vs cost visualization

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
