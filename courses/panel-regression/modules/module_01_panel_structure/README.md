# Module 1: Panel Data Structure

## Overview

Understand the unique structure of panel data and how to prepare it for analysis. Learn to handle long vs wide formats, balanced vs unbalanced panels, and common data quality issues.

**Time Estimate:** 4-6 hours

## Learning Objectives

By completing this module, you will:
1. Distinguish panel data from cross-sectional and time series data
2. Convert between long and wide data formats
3. Handle unbalanced panels and missing data
4. Identify and address common panel data issues

## Contents

### Guides
- `01_panel_fundamentals.md` - What makes data "panel"
- `02_data_formats.md` - Long vs wide, reshaping
- `03_data_quality.md` - Missing data, attrition, balance

### Notebooks
- `01_data_preparation.ipynb` - Preparing panel datasets
- `02_exploration.ipynb` - Exploratory analysis for panels

## Key Concepts

### Panel Data Structure

| Entity (i) | Time (t) | X₁ | X₂ | Y |
|------------|----------|-----|-----|-----|
| Firm A | 2020 | 100 | 5.2 | 45 |
| Firm A | 2021 | 110 | 5.5 | 48 |
| Firm A | 2022 | 105 | 5.3 | 46 |
| Firm B | 2020 | 200 | 6.1 | 82 |
| Firm B | 2021 | 210 | 6.3 | 85 |
| Firm B | 2022 | 220 | 6.5 | 88 |

### Notation

- $N$ = number of entities
- $T$ = number of time periods
- $n = N \times T$ = total observations (if balanced)

### Panel Types

| Type | Description | Example |
|------|-------------|---------|
| Balanced | Same T for all entities | Survey with fixed waves |
| Unbalanced | Different T across entities | Firms entering/exiting |
| Short Panel | Small T, large N | Many firms, few years |
| Long Panel | Large T, small N | Few countries, many years |

## Prerequisites

- Module 0 completed
- pandas proficiency
