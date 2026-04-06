# All Courses Display Audit — 2026-04-06

## Summary

All 19 courses were audited across 7 display-correctness checks. **2 issues** were found and fixed in prior commits. All courses now pass all checks.

### Fixes Applied (in prior commits)

1. **genetic-algorithms-feature-selection** — orphaned `</div>` in `module_00_foundations/guides/02_optimization_basics.md` (batch C fix)
2. **power-automate** — malformed `</div<div>` tag in `module_05_sharepoint_excel/guides/02_excel_integration_guide.md` (batch C/D fix)
3. **34 guide files** across power-automate, reinforcement-learning, and time-series-forecasting-neuralforecast — mermaid blocks missing `%%{init}` theme headers (batch C/D fix)

## Results by Course

| Course | Slides Match | SVG Refs | HTML Balance | Notebooks | No Leaks | No Old CSS | Mermaid Theme | Status |
|--------|-------------|----------|-------------|-----------|----------|-----------|--------------|--------|
| agentic-ai-llms | 32/32 | PASS | PASS | 20/20 | PASS | PASS | PASS | PASS |
| agentic-ai-practical | 0/0 | PASS | PASS | 3/3 | PASS | PASS | PASS | PASS |
| ai-engineer-fundamentals | 11/11 | PASS | PASS | 1/1 | PASS | PASS | PASS | PASS |
| bayesian-commodity-forecasting | 27/27 | PASS | PASS | 24/24 | PASS | PASS | PASS | PASS |
| bayesian-prompt-engineering | 16/16 | PASS | PASS | 6/6 | PASS | PASS | PASS | PASS |
| captum-neural-network-interpretability | 22/22 | PASS | PASS | 28/28 | PASS | PASS | PASS | PASS |
| causalpy-interrupted-time-series | 21/21 | PASS | PASS | 25/25 | PASS | PASS | PASS | PASS |
| dataiku-genai | 14/14 | PASS | PASS | 10/10 | PASS | PASS | PASS | PASS |
| double-machine-learning | 10/10 | PASS | PASS | 11/11 | PASS | PASS | PASS | PASS |
| dynamic-factor-models | 28/28 | PASS | PASS | 15/15 | PASS | PASS | PASS | PASS |
| genai-commodities | 21/21 | PASS | PASS | 17/17 | PASS | PASS | PASS | PASS |
| genetic-algorithms-feature-selection | 20/20 | PASS | FIXED | 12/12 | PASS | PASS | PASS | PASS |
| hidden-markov-models | 16/16 | PASS | PASS | 14/14 | PASS | PASS | PASS | PASS |
| midas-mixed-frequency-nowcasting | 23/23 | PASS | PASS | 28/28 | PASS | PASS | PASS | PASS |
| multi-armed-bandits-ab-testing | 38/38 | PASS | PASS | 30/30 | PASS | PASS | PASS | PASS |
| panel-regression | 22/22 | PASS | PASS | 13/13 | PASS | PASS | PASS | PASS |
| power-automate | 20/20 | PASS | FIXED | 13/13 | PASS | PASS | FIXED | PASS |
| reinforcement-learning | 32/32 | PASS | PASS | 26/26 | PASS | PASS | FIXED | PASS |
| time-series-forecasting-neuralforecast | 15/15 | PASS | PASS | 16/16 | PASS | PASS | FIXED | PASS |

## Check Descriptions

| Check | What It Verifies |
|-------|-----------------|
| Slides Match | Every `_slides.md` has a corresponding `_slides.html` |
| SVG Refs | All `![...](*.svg)` references in guides point to existing files |
| HTML Balance | Opening `<div>` tags match closing `</div>` tags in guides |
| Notebooks | All `.ipynb` files parse with `nbformat.read()` |
| No Leaks | No `_slides` files appear in the Streamlit guide listing |
| No Old CSS | No `callout-success` or bare `callout` classes remain |
| Mermaid Theme | All mermaid blocks include `%%{init: {...}}%%` theme headers |

## Batch Details

- **Batch A** (agentic-ai-llms, agentic-ai-practical, ai-engineer-fundamentals, bayesian-commodity-forecasting, bayesian-prompt-engineering) — all clean
- **Batch B** (captum, causalpy, dataiku-genai, double-ml, dynamic-factor-models) — all clean
- **Batch C** (genai-commodities, genetic-algorithms, hidden-markov, midas, multi-armed-bandits) — 1 fix (orphaned div)
- **Batch D** (panel-regression, power-automate, reinforcement-learning, neuralforecast) — 35 fixes (1 malformed div, 34 mermaid init headers)

**Total**: 19 courses audited, 36 issues fixed, 0 remaining.
