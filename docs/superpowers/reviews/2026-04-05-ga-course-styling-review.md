# GA Course Styling Review

**Date:** 2026-04-05
**Course:** genetic-algorithms-feature-selection
**Scope:** All 21 guide markdowns, 20 slide decks, 3 notebooks (sampled), STYLING_GUIDE.md, course-theme.css, notebook_style.py
**Reviewer:** Automated styling audit

---

## Findings Table

| # | Issue | Location | Severity | Recommendation |
|---|-------|----------|----------|----------------|
| 1 | **Two distinct guide structures coexist** | 12 guides use "full" structure (In Brief/Formal Definition/Intuitive Explanation/Code Implementation/Common Pitfalls/Connections/Practice Problems/Further Reading). 9 guides use "compact" structure (topic-driven headings, Key Takeaways closing). No rule in STYLING_GUIDE.md distinguishes when to use which. | High | STYLING_GUIDE.md must define both patterns explicitly with criteria for when each is appropriate. |
| 2 | **Zero guides have a "Learning Objectives" section** | All 21 guide markdowns (`grep -c "## Learning Objectives"` returns 0 for every file) | High | STYLING_GUIDE.md Section 8.1 prescribes `## Learning Objectives` as part of the guide template, but no guide in the GA course includes it. Either enforce it or update the guide to mark it optional. |
| 3 | **Zero guides have a "Summary" or "Next Steps" section** | None of the 21 guides contain `## Summary` or `## Next Steps` as prescribed in STYLING_GUIDE.md Section 8.1 | High | The guide template says to close with a Summary table and Next Steps links. GA guides close with either `## Key Takeaways` (callout-key) or `## Further Reading` + a one-line Next link. Reconcile the template with reality. |
| 4 | **Only 1 code-window per guide (first code block only)** | All 21 guides have exactly 1 `code-window` HTML wrapper. Remaining code blocks (2-13 per guide) are bare triple-backtick fenced blocks. | High | The styling guide says code blocks should use "macOS terminal windows with traffic-light dots, filenames, and annotation callouts." In practice, only the first code example per guide gets this treatment. Either relax the rule or enforce it. |
| 5 | **No flow components or comparison cards in any guide** | 0/21 guides use `<div class="flow">` or `<div class="compare">`. These are only used in slide decks. | Medium | STYLING_GUIDE.md documents these components for guides too (Section 8.4 lists "Guides: SVG diagrams, callout boxes, comparison cards, tables"). Add guidance on when guides should use flow/compare vs. callouts. |
| 6 | **Callout type distribution is narrow in guides** | Most guides use exactly 1 insight + 1 info callout. Warning/key/danger callouts appear in only a subset. Typical: insight(1), info(1), occasionally warning(1) or key(1). Never danger. | Medium | Add minimum callout diversity guidance: e.g., "Each guide should use at least 3 distinct callout types appropriate to the content." |
| 7 | **SVG images referenced but may not exist** | Every guide references 1-3 SVGs (e.g., `feature_selection_pipeline.svg`, `ga_lifecycle.svg`). Many guides reference the SAME SVG (`feature_selection_pipeline.svg` appears in 5+ guides). No verification that referenced SVGs actually exist. | Medium | Add cross-reference validation checklist. Guides should reference module-specific SVGs, not reuse the same generic diagram everywhere. |
| 8 | **Callout `<strong>` labels missing in guides** | STYLING_GUIDE.md examples show `<strong>Insight:</strong>` inside callout-insight. GA course guides typically omit the `<strong>` label, using just the text directly inside the callout div. | Low | Enforce callout structure: always include `<strong>Type:</strong>` prefix inside callout divs. |
| 9 | **Emoji inconsistency in callouts** | Some callouts include emoji (e.g., `info` callouts in Module 00 use `info icon + "How this connects"`). Others are plain text. No consistent pattern. | Low | STYLING_GUIDE.md should specify: callouts in guides use emoji prefix, or not. Current state is mixed. |
| 10 | **Bare code blocks lack filenames and context** | After the first code-window, subsequent code blocks are bare ```` ```python ```` without any filename, purpose comment, or annotation. In 02_selection_approaches.md, 9 bare code blocks appear consecutively. | Medium | Add rule: every code block over 5 lines should either use a code-window wrapper OR include a `# Purpose:` comment as the first line. |
| 11 | **Slides: compare cards used as "Builds On / Leads To" containers** | Module 00 feature_selection_challenge_slides.md uses compare-card with `before`/`after` headers for "Builds On" / "Leads To" content. The semantic mismatch (red="Builds On", green="Leads To") is confusing. | Medium | Add guidance for using compare-cards for non-before/after content, or define additional header color classes (e.g., `.header.blue`, `.header.purple`). |
| 12 | **Slide decks: callout count drops sharply after Module 00-01** | Module 00 slides: 8-9 callouts per deck. Module 03-05 slides: 1-2 callouts per deck. This creates inconsistent visual density. | Medium | Set minimum callout count per slide deck (e.g., at least 3-4 per deck). |
| 13 | **Slide decks: comparison cards disappear after Module 01** | Module 00-01 decks use 1-2 comparison cards each. Modules 02-05 decks use 0 comparison cards. | Low | Comparison cards are effective for presenting tradeoffs, which are abundant in later modules (e.g., walk-forward vs. expanding window, memetic vs. pure GA). |
| 14 | **Guide metadata banner format varies** | "Full" guides: `> **Reading time:** ~X min \| **Module:** N -- Topic \| **Prerequisites:** ...`. "Compact" guides: `> **Reading time:** ~X min \| **Module:** N -- Topic \| **Prerequisites:** ...`. Same format, BUT some compact guides omit the metadata banner entirely (e.g., 01_ga_components.md has no metadata banner). | Medium | Make the metadata banner mandatory for all guides. |
| 15 | **No `## Learning Objectives` slide in all decks** | All slide decks correctly include a Learning Objectives slide after the lead slide, matching STYLING_GUIDE.md Section 8.2. However, this is absent from the guide markdowns. | Low | This discrepancy between guides and slides should be reconciled. |
| 16 | **Notebook structure: callout() before setup imports** | In `01_deap_ga.ipynb`, Cell 4 calls `callout()` BEFORE Cell 5 imports `apply_course_theme`. This will fail if the notebook is run top-to-bottom since `callout` was imported in Cell 5. | High | STYLING_GUIDE.md Section 8.3 says Cell 2 should be the setup cell. Enforce: imports and theme application MUST be Cell 2, before any styled output. |
| 17 | **Notebooks: `learning_objectives()` called as code cell, not markdown** | All three sampled notebooks call `learning_objectives([...])` as a Python code cell (Cell 1) rather than a markdown cell with HTML. This means objectives display as code output, not native markdown. | Low | This is actually the correct pattern per `notebook_style.py` design. STYLING_GUIDE.md should document this pattern explicitly. |
| 18 | **Notebooks: sys.path depth varies** | Module 00 notebooks use `sys.path.insert(0, '../../../../..')` (5 levels up). Module 04 uses the same. But path depth depends on nesting, and there is no validation. | Low | Add a note to STYLING_GUIDE.md about sys.path calculation: count directory levels from notebook to project root. |
| 19 | **Math/LaTeX: `\argmin` used without `\DeclareMathOperator`** | Multiple guides use `\argmin` directly. MathJax supports this, but it renders as italic "argmin" instead of upright. No `\operatorname{argmin}` or `\DeclareMathOperator` convention. | Low | Add LaTeX convention: use `\operatorname{argmin}` or include a MathJax config preamble. |
| 20 | **No algorithm pseudocode styling convention** | Several guides present algorithms as plain text (e.g., "Algorithm: RFE" in `02_selection_approaches.md`) or ASCII art (e.g., GA framework box in `01_ga_components.md`). No consistent pseudocode formatting. | Medium | Add a pseudocode presentation convention to STYLING_GUIDE.md (e.g., use a callout-info with monospace formatting, or a dedicated `algorithm` HTML component). |
| 21 | **No parameter table convention** | GA parameters (population size, mutation rate, crossover probability, tournament size) appear as inline text, bulleted lists, and table rows inconsistently across guides. | Medium | Define a standard "GA Parameter Table" format in the styling guide. |
| 22 | **Practice problems use bare code blocks** | All practice problem sections use bare ```` ```python ```` blocks, never code-windows. This is consistent but inconsistent with the styling guide's preference for code-windows. | Low | Define explicitly: practice problem code may use bare code blocks (to distinguish from teaching code), or require code-windows everywhere. |
| 23 | **"Connections" section uses callout-info as header decoration** | In guides with the full structure, the Connections section starts with `<div class="callout-info">` containing only a header line like `"How this connects to the rest of the course:"`. This is decorative rather than informative -- the callout adds no content. | Low | Either make the Connections callout contain substantive content, or use an H3 heading instead. |
| 24 | **Duplicate guide naming in Module 00** | Module 00 has `01_feature_selection_problem.md` AND `01_optimization_basics.md` (both numbered 01). Also `02_evolutionary_operators.md` AND `02_selection_approaches.md` (both numbered 02). This creates ambiguity. | Medium | STYLING_GUIDE.md should prescribe unique numbering within each module. |
| 25 | **Mermaid init directive not always present** | STYLING_GUIDE.md Section 6 requires every Mermaid diagram to include the init directive. All sampled slide decks do include it, which is correct. Guides do not use Mermaid (they use SVGs instead), which is a reasonable separation. | Low | Clarify in STYLING_GUIDE.md that Mermaid is for slides and SVGs are for guides. |

---

## Styling Guide Improvement Recommendations

### 1. Define Two Guide Templates (Full vs. Compact)

The GA course has two clearly distinct guide structures:

**Full Template** (used by 12/21 guides):
```
# Title
> Metadata banner
## In Brief (with callout-insight)
## Formal Definition (LaTeX)
## Intuitive Explanation (analogy)
## [Mathematical Formulation] (optional)
## Code Implementation (code-window first, then bare blocks)
## Common Pitfalls (numbered, with code examples)
## Connections (callout-info, Builds On / Leads To / Related To)
## Practice Problems (numbered, with skeleton code)
## Further Reading (academic papers, books, online)
---
**Next:** [Companion Slides](link) | [Notebook](link)
```

**Compact Template** (used by 9/21 guides):
```
# Title
> Metadata banner (sometimes omitted)
## Topic-driven sections (no fixed pattern)
## Key Takeaways (callout-key with numbered points)
---
**Next:** [Companion Slides](link) | [Notebook](link)
```

**Recommendation:** Add both templates to STYLING_GUIDE.md Section 8.1 with clear criteria:
- **Full template:** For concept introduction guides (first guide in a topic area). Use when introducing a concept for the first time.
- **Compact template:** For reference/implementation guides. Use when the reader already has the conceptual foundation and needs working code.

### 2. Add Learning Objectives Rule for Guides

Current template prescribes `## Learning Objectives` but zero guides implement it. Options:
- **Option A (Recommended):** Remove `## Learning Objectives` from the guide template. Guides already have the "In Brief" section that serves a similar purpose. Learning Objectives are better suited to notebooks and slide decks where they frame a hands-on session.
- **Option B:** Require it and enforce it.

### 3. Standardize Guide Closing Section

Replace the current template's `## Summary` + `## Next Steps` with what the course actually uses:

For Full template: close with `## Further Reading` + one-line Next link.
For Compact template: close with `## Key Takeaways` callout + one-line Next link.

### 4. Add Code Presentation Tiering

Define three tiers of code presentation:

| Tier | When | Format |
|------|------|--------|
| **Primary example** | First/key code block per section | Full code-window with filename and optional annotations |
| **Supporting example** | Follow-up code blocks | Bare fenced block with `# Purpose:` first-line comment |
| **Practice/exercise** | Practice problems | Bare fenced block (distinguishes from teaching content) |

### 5. Add Visual Density Metrics by Content Type

Extend STYLING_GUIDE.md Section 8.4 with specific minimums:

| Content Type | Minimum Visual Elements | Required Types |
|-------------|------------------------|----------------|
| Guide (Full, 10+ min reading) | 5+ visual elements | At least: 1 SVG diagram, 1 code-window, 1 callout-insight, 1 callout-warning or callout-key, 1 table |
| Guide (Compact, 5-7 min) | 3+ visual elements | At least: 1 SVG or code-window, 1 callout, 1 table |
| Slide deck (10+ slides) | 1 visual per slide average | At least: 2 Mermaid diagrams, 2 code-windows, 1 flow component, 2 callouts |
| Notebook (any length) | 1 plot/output per 3 code cells | `apply_course_theme()` and `apply_plot_theme()` in Cell 2 |

### 6. Add Callout Usage Guidelines

Add a "Callout Selection Guide" to STYLING_GUIDE.md Section 4.3:

| Callout Type | Use For | Do NOT Use For |
|-------------|---------|----------------|
| `callout-insight` | Non-obvious implications, "aha" moments, design rationale | Restating what the text already says |
| `callout-warning` | Common mistakes, performance pitfalls, subtle errors | General caution with no specific consequence |
| `callout-key` | Summary takeaways, critical facts to remember | Long paragraphs (keep to 1-3 sentences) |
| `callout-danger` | Data leakage, security issues, irreversible mistakes | Minor coding errors |
| `callout-info` | Context, prerequisites, cross-references | Decorative headers with no content |

**Minimum per guide:** At least 3 callouts using at least 2 distinct types.

### 7. Add Algorithm Pseudocode Convention

GA courses frequently present algorithms. Add a convention:

```html
<div class="callout-info" style="font-family: var(--font-code);">
<strong>Algorithm:</strong> Tournament Selection<br/>
<strong>Input:</strong> Population P, tournament size k<br/>
<strong>Output:</strong> Selected individual<br/><br/>
1. Sample k individuals from P uniformly at random<br/>
2. Return individual with best fitness from sample
</div>
```

### 8. Add Parameter Table Convention

Define a standard "configuration table" format for GA parameters:

```markdown
| Parameter | Symbol | Range | Recommended | Notes |
|-----------|--------|-------|-------------|-------|
| Population size | N | 20-200 | 50 | Larger for more features |
| Mutation rate | p_m | 0.001-0.1 | 1/n | n = chromosome length |
| Crossover rate | p_c | 0.6-0.95 | 0.8 | |
| Tournament size | k | 2-7 | 3 | Higher = more pressure |
```

### 9. Add Cross-Reference Convention

Add explicit rules:
- Every guide MUST end with `**Next:** [Companion Slides](./XX_slides.md) | [Notebook](../notebooks/XX.ipynb)` (already done in GA course).
- SVG references should use module-specific diagrams, not the same generic diagram repeated across modules.
- Guides should NOT reference the same SVG more than twice across the entire course.

### 10. Add LaTeX Convention Section

Add to STYLING_GUIDE.md:
- Use `\operatorname{argmin}` instead of `\argmin` for proper upright rendering.
- Display math ($$...$$) for standalone equations. Inline math ($...$) for variables and short expressions.
- Multi-line equations use `\begin{cases}` or `\begin{bmatrix}`, never aligned plain text.
- Where/Given/Find blocks: use bold labels followed by a bulleted list.

### 11. Clarify Mermaid vs. SVG Separation

Add to Section 5 or Section 6:
- **Slide decks:** Use Mermaid diagrams (rendered live by Marp).
- **Guide markdowns:** Use pre-generated SVG files (embedded via `![alt](./path.svg)`).
- **Notebooks:** Use inline SVG via `diagram_generator.py` or matplotlib plots.

### 12. Add Notebook Cell Ordering Rule

Strengthen STYLING_GUIDE.md Section 8.3:

| Cell # | Type | Content | Notes |
|--------|------|---------|-------|
| 0 | Markdown | Title, objectives heading | No code |
| 1 | Code | `learning_objectives([...])` | Styled HTML output |
| 2 | Code | `section_divider("Setup")` | Visual break |
| 3 | Markdown | Brief setup context | 1-2 sentences |
| 4 | Code | `import sys; sys.path...` + all imports + `apply_course_theme()` + `apply_plot_theme()` | ALL imports here |
| N-2 | Code | `section_divider("Summary")` | |
| N-1 | Markdown | Summary heading + text | |
| N | Code | `key_takeaways([...])` | Styled HTML output |

**Critical rule:** No styled helper functions (`callout()`, `section_divider()`, etc.) may be called before the import cell.

---

## Quick Wins

These 10 changes would immediately improve the GA course with minimal effort:

1. **Add metadata banners to the 3 compact guides missing them** (`01_ga_components.md`, `01_fitness_functions.md`, `02_cross_validation_fitness.md`, `01_timeseries_considerations.md`, `01_deap_implementation.md`, `01_advanced_techniques.md`). Each needs a `> **Reading time:** ~X min | **Module:** N -- Topic | **Prerequisites:** ...` line after the H1.

2. **Fix `01_deap_ga.ipynb` cell ordering**: Move the `callout()` call in Cell 4 to AFTER the imports/theme application in Cell 5-6. Currently Cell 4 calls `callout("DEAP uses a registration-based system.", kind="insight")` before `callout` is imported.

3. **Add `<strong>Type:</strong>` prefix to all guide callouts**: Most callouts in guides contain just the text without the label prefix shown in STYLING_GUIDE.md examples. A find-and-replace can add `<strong>Insight:</strong>` etc. to each callout div.

4. **Replace `\argmin` with `\operatorname{argmin}`** in all guide and slide LaTeX. This affects `01_feature_selection_problem.md`, `01_optimization_basics.md`, `02_selection_approaches.md`, and `01_feature_selection_challenge_slides.md`.

5. **Add at least 1 comparison card to guides in Module 02-05**: The comparison card component exists in the theme CSS but is never used outside of slide decks. Prime candidates:
   - `01_fitness_functions.md`: Compare weighted-sum vs. Pareto fitness
   - `01_timeseries_considerations.md`: Compare walk-forward vs. expanding window
   - `01_deap_implementation.md`: Compare eaSimple vs. custom loop

6. **Reduce SVG reuse**: `feature_selection_pipeline.svg` is referenced in at least 5 guides and 2 slide decks. Create module-specific SVGs or at least reference the generic one only in Module 00 where it is introduced.

7. **Add `# Purpose:` comments to bare code blocks**: The 140+ bare code blocks across guides have no filename or purpose annotation. Adding a `# Purpose: ...` first-line comment to each would improve scanability at minimal cost.

8. **Increase callout count in Module 03-05 slide decks**: Decks like `01_timeseries_considerations_slides.md` (1 callout across 15 slides) and `01_advanced_techniques_slides.md` (1 callout across 15 slides) are visually sparse compared to Module 00-01 decks (8-9 callouts). Add 2-3 callouts to each.

9. **Rename duplicate-numbered guides in Module 00**: Currently has two `01_*.md` and two `02_*.md` files. Renumber to `01`, `02`, `03`, `04` for unambiguous ordering.

10. **Add flow components to 3-4 guides**: The `<div class="flow">` component is used in every slide deck but zero guides. Key candidates:
    - `01_ga_components.md`: Initialize -> Evaluate -> Select -> Crossover -> Mutate -> Replace
    - `01_fitness_functions.md`: Chromosome -> Select Features -> Train Model -> CV Error -> Fitness
    - `01_timeseries_considerations.md`: Walk-Forward -> Rolling Stability -> Multi-Horizon

---

## Summary Statistics

| Metric | Count | Notes |
|--------|-------|-------|
| Total guides reviewed | 21 | 12 full-template, 9 compact-template |
| Total slide decks reviewed | 20 | All have correct Marp frontmatter |
| Total notebooks sampled | 3 | Modules 00, 01, 04 |
| Guides with Learning Objectives | 0/21 | Template prescribes it, none implement it |
| Guides with Summary section | 0/21 | Template prescribes it, none implement it |
| Guides with code-windows | 21/21 | But only 1 per guide (first block) |
| Guides with flow components | 0/21 | Only in slides |
| Guides with comparison cards | 0/21 | Only in slides |
| Slide decks with speaker notes | 20/20 | Every slide has notes |
| Slide decks with Mermaid init | 20/20 | Consistent |
| Notebooks with apply_course_theme() | 3/3 | Consistent |
| Notebooks with apply_plot_theme() | 3/3 | Consistent |
| Notebooks with learning_objectives() | 3/3 | Consistent |
| Notebooks with key_takeaways() | 3/3 | Consistent |
