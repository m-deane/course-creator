# Figures Directory

This directory contains visual diagrams and figures for Module 2: Tool Use & Function Calling.

## Recommended Diagrams

The following diagrams should be created to support this module's content:

1. **tool_calling_flow.png**
   - Complete flow: User Query → LLM Decision → Tool Call → Execution → Result → LLM Response
   - JSON structure for tool call and result
   - Multi-turn tool interaction sequence

2. **tool_definition_anatomy.png**
   - Breakdown of tool schema components
   - Name, description, parameters, required fields
   - JSON Schema examples with annotations

3. **multi_tool_agent_architecture.png**
   - Agent with multiple tool connections
   - Decision flow for tool selection
   - Error handling and fallback paths

4. **tool_security_layers.png**
   - Input validation layer
   - Sandboxing/isolation layer
   - Output sanitization layer
   - Audit logging layer

5. **error_handling_flowchart.png**
   - Tool execution decision tree
   - Retry logic with exponential backoff
   - Graceful degradation paths
   - User communication for failures

6. **tool_design_principles.png**
   - Single responsibility examples
   - Good vs bad tool naming
   - Parameter design patterns
   - Output format standards

7. **react_loop_diagram.png**
   - Reasoning → Action → Observation cycle
   - Multi-step problem solving visualization
   - Tool chaining example

8. **tool_types_taxonomy.png**
   - Read-only tools (search, retrieve, calculate)
   - Write tools (create, update, delete)
   - External API tools (weather, stock data)
   - Code execution tools (sandboxed computation)

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
