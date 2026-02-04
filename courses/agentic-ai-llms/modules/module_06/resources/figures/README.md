# Module 6 Figures

This directory should contain visual diagrams for Module 6: Evaluation & Safety.

## Suggested Diagrams

### 1. Safety Layers Architecture
- Visual stack showing defense in depth: Input Validation → Agent Processing → Output Filtering → Action Verification
- Show what each layer checks for and blocks
- Include example attacks blocked at each layer

### 2. Evaluation Framework Pyramid
- Base: Unit tests for individual components
- Middle: Integration tests for agent workflows
- Top: End-to-end benchmarks and human evaluation
- Show metrics appropriate for each level

### 3. Red Teaming Attack Tree
- Root: Agent system
- Branches: Different attack vectors (prompt injection, jailbreaking, data extraction, resource abuse, hallucinations)
- Leaves: Specific attack techniques
- Color-code by severity and likelihood

### 4. Guardrail Decision Flow
- Input enters system → Guardrail checks → Pass/Block/Modify decision
- Show branching logic for different rule types
- Include fallback behaviors when guardrails trigger

### 5. Evaluation Metrics Dashboard
- Multi-panel dashboard layout showing key metrics:
  - Accuracy (correctness)
  - Reliability (consistency)
  - Safety (harmful outputs)
  - Helpfulness (user satisfaction)
  - Efficiency (cost/latency)
- Include example values and thresholds

### 6. Prompt Injection Attack Flow
- Sequence diagram showing how malicious input bypasses intended behavior
- Show user input → injected instructions → hijacked output
- Include defense mechanisms that break the attack

### 7. Benchmark Test Coverage Matrix
- Rows: Different capabilities (reasoning, knowledge, coding, etc.)
- Columns: Different test types (unit, integration, adversarial)
- Cells: Checkmarks for covered areas, red X for gaps

### 8. Safety vs Capability Tradeoff
- Graph showing relationship between safety restrictions and model capability
- Show Pareto frontier of optimal tradeoffs
- Mark different deployment scenarios (public, enterprise, research)

## Format Recommendations

- Use flowcharts for decision processes
- Create attack trees with Mermaid or GraphViz
- Design dashboard mockups for monitoring UIs
- Use sequence diagrams for attack scenarios
- Include real examples (sanitized) not just abstract concepts
- Color-code by severity: green (safe), yellow (warning), red (critical)
