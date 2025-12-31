# Module 4: Fundamentals Modeling with LLMs

## Overview

Combine traditional fundamental analysis with LLM capabilities. Build supply/demand models enhanced by unstructured data extraction and reasoning.

**Time Estimate:** 10-12 hours

## Learning Objectives

By completing this module, you will:
1. Extract fundamental data from reports
2. Build supply/demand balance models
3. Integrate qualitative and quantitative factors
4. Generate fundamental-based forecasts

## Contents

### Guides
- `01_supply_demand.md` - S/D balance construction
- `02_storage_analysis.md` - Inventory and storage theory
- `03_term_structure.md` - Curve analysis with fundamentals

### Notebooks
- `01_crude_fundamentals.ipynb` - Oil S/D model
- `02_nat_gas_balance.ipynb` - Gas storage model
- `03_integrated_model.ipynb` - Full fundamental system

## Key Concepts

### Supply/Demand Framework

```
Supply                    Demand
├── Production            ├── Refinery runs
├── Imports               ├── Exports
├── Stock draws           ├── Stock builds
└── Other supply          └── Other demand

Balance = Supply - Demand
```

### LLM-Enhanced Fundamentals

| Traditional | LLM Enhancement |
|-------------|-----------------|
| EIA data tables | Extract from prose |
| Fixed categories | Flexible parsing |
| Historical only | Forward commentary |
| Quantitative | Qualitative context |

### Fundamental Extraction

```python
FUNDAMENTAL_PROMPT = """Extract supply/demand data from this report.

Report: {report_text}

Extract for each commodity mentioned:
- Production (current, change, units)
- Consumption (current, change, units)
- Inventories (current, vs average, units)
- Key factors affecting balance
- Forward outlook indicators

Return structured JSON."""
```

## Prerequisites

- Module 0-3 completed
- Basic commodity fundamentals
