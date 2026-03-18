# Project Status

**Last updated:** 2026-03-18

## Content Inventory Summary

| Metric | Count |
|--------|-------|
| Courses | 17 |
| Slide decks (Marp) | 357 |
| Rendered HTML slides | 259 (98 pending render) |
| Jupyter notebooks | 293 |
| Concept guides | 132 |
| Python templates | 25 |
| Recipes | 37 |
| Portfolio projects | 47 |
| Quick-starts | 27 |
| Exercises | 56 |

## Course Completion Status

### Fully-Featured Courses (guides + notebooks + templates + recipes + projects + quick-starts)

| Course | Decks | Notebooks | Guides | Templates | Recipes | Projects | Quick-starts | Status |
|--------|-------|-----------|--------|-----------|---------|----------|-------------|--------|
| `captum-neural-network-interpretability` | 22 | 28 | 22 | 5 | 7 | 4 | 3 | ✅ Complete |
| `causalpy-interrupted-time-series` | 21 | 25 | 21 | 4 | 6 | 4 | 3 | ✅ Complete |
| `midas-mixed-frequency-nowcasting` | 23 | 28 | 23 | 4 | 6 | 4 | 3 | ✅ Complete |
| `reinforcement-learning` | 32 | 26 | 32 | 5 | 3 | 7 | 4 | ✅ Complete |
| `power-automate` | 20 | 13 | 20 | 1 | 4 | 2 | 3 | ✅ Complete |
| `multi-armed-bandits-ab-testing` | 38 | 33 | — | 7 | 4 | 14 | 6 | ✅ Complete |

### Slide + Notebook Courses (missing guides/templates/recipes)

| Course | Decks | Notebooks | Guides | Status |
|--------|-------|-----------|--------|--------|
| `agentic-ai-llms` | 32 | 20 | — | ⚠️ Needs guides, templates, recipes |
| `bayesian-commodity-forecasting` | 27 | 24 | — | ⚠️ Needs guides, templates, recipes |
| `dataiku-genai` | 14 | 10 | — | ⚠️ Needs guides, templates, recipes |
| `dynamic-factor-models` | 28 | 15 | — | ⚠️ Needs guides, templates, recipes |
| `genai-commodities` | 21 | 17 | — | ⚠️ Needs guides, templates, recipes |
| `genetic-algorithms-feature-selection` | 20 | 12 | — | ⚠️ Needs guides, templates, recipes |
| `hidden-markov-models` | 16 | 14 | — | ⚠️ Needs guides, templates, recipes |
| `panel-regression` | 22 | 13 | — | ⚠️ Needs guides, templates, recipes |

### Minimal Courses (early stage)

| Course | Decks | Notebooks | Status |
|--------|-------|-----------|--------|
| `double-machine-learning` | 10 | 11 | ⚠️ Has guides, needs templates/recipes/projects |
| `ai-engineer-fundamentals` | 11 | 1 | ⚠️ Has projects, needs notebooks/guides |
| `agentic-ai-practical` | — | 3 | ⚠️ Has templates/projects, needs decks/guides |

## Slide Rendering Gap

- **357** slide decks exist
- **259** rendered to HTML
- **98** decks need rendering

To render all pending:
```bash
npx @marp-team/marp-cli --html --theme-set resources/themes/course-theme.css -- "courses/**/*_slides.md"
```

## Priority Actions

1. **Render 98 pending slide decks** to close the HTML gap
2. **Add guides** to the 8 courses that have decks+notebooks but no guides
3. **Add templates/recipes/projects/quick-starts** to courses missing supplementary materials
4. **Complete `agentic-ai-practical`** — only course with no slide decks
5. **Expand `ai-engineer-fundamentals`** — only 1 notebook across 11 modules
