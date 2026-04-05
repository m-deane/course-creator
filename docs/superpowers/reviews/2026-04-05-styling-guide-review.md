# Styling Guide Audit Review

**Date:** 2026-04-05
**Scope:** STYLING_GUIDE.md, styling_guide.html, course-theme.css, notebook_style.py, streamlit custom.css, GA course implementation
**References:** Daily Dose of Data Science (DDoDS) design patterns, GA Feature Selection course as implemented

---

## 1. Missing Components

DDoDS patterns not covered or only partially covered in the styling guide.

### 1.1 Embedded Link Preview Cards
**Priority:** High

DDoDS uses rich link preview cards for cross-references between articles -- styled boxes with a title, description snippet, and arrow/icon. The styling guide mentions "Next Steps" links at the end of guides as plain markdown links (`[text](url)`), and the GA course implements these as simple inline links (e.g., `**Next:** [Companion Slides](./01_feature_selection_challenge_slides.md)`). No component spec or CSS class exists for a visually styled link preview card.

**Recommendation:** Add a `.link-card` component with thumbnail area, title, description, and hover state. Define markup, CSS class spec, and add to the HTML preview.

### 1.2 Gradient Section Backgrounds
**Priority:** Medium

DDoDS uses gradient pastel backgrounds between sections for visual grouping. The styling guide documents flat pastel backgrounds (`--bg-section-mint`, `--bg-section-amber`, etc.) but never specifies gradient usage within content sections. The `section.lead` slide class in course-theme.css does use `linear-gradient(135deg, var(--bg-sidebar), #16213e)`, and `key_takeaways()` in notebook_style.py uses `linear-gradient(135deg, #f3e5f5, #e3f2fd)`, but these are isolated instances without a documented pattern.

**Recommendation:** Document a `.section-gradient` pattern with 2-3 preset gradient combinations (e.g., mint-to-blue, lavender-to-blue) that content creators can apply to full-width section backgrounds.

### 1.3 Character/Mascot Illustrations
**Priority:** Low

DDoDS includes custom illustrated characters and mascots in educational contexts. The styling guide has no provisions for illustrated characters. The icon set in `resources/graphics/icons/` contains 12 utility SVG icons (robot, brain, gear, etc.) but these are 24x24 functional icons, not illustrative characters.

**Recommendation:** This is a nice-to-have. If pursued, define a spot illustration specification (size, positioning, color palette constraints) and add 3-4 mascot/character SVGs for common scenarios (thinking, success, error, tip).

### 1.4 Comparison Tables with Icons
**Priority:** Medium

DDoDS uses comparison tables where column headers or rows have colored icons alongside text. The styling guide documents comparison cards (`.compare` / `.compare-card`) with before/after headers, and standard tables with dark headers. Neither supports inline icons within cells or colored icon badges in table headers. The GA course uses plain markdown tables throughout without icon enhancement.

**Recommendation:** Add a `.table-with-icons` variant or document how to embed inline SVG icons within table cells. Define icon size (16-20px), alignment, and color rules for table contexts.

### 1.5 Read-Time Badges
**Priority:** Medium

DDoDS shows read-time badges prominently on each page. The styling guide documents metadata badges (`.badge` class in course-theme.css with `.badge.mint`, `.badge.amber`, etc.) and the guide template includes a reading time line (`> **Reading time:** ~X min`). However, the badge CSS class and the guide reading time pattern are not connected -- guides use a blockquote for metadata, not the `.badge` component. The GA course uses the blockquote pattern consistently (e.g., `> **Reading time:** ~9 min | **Module:** 0 -- Foundations`).

**Recommendation:** Document a styled reading-time badge using the existing `.badge` class, positioned at the top of guides and slide decks. Provide markup examples that match the GA course pattern but use the badge component for visual consistency.

### 1.6 Dark Sidebar Navigation with Collapsible Sections and Progress Bar
**Priority:** Low (already implemented in Streamlit, not in styling guide)

DDoDS has a dark sidebar with collapsible sections and a progress bar. The Streamlit `custom.css` implements this fully (`.progress-container`, `.progress-bar-bg`, `.progress-bar-fill`, sidebar styling). However, STYLING_GUIDE.md only mentions `--bg-sidebar` in the color table. The sidebar component is not documented as a reusable pattern.

**Recommendation:** Add a sidebar section to the styling guide documenting the Streamlit implementation as a reference pattern, even if it is Streamlit-specific. This ensures the design intent is preserved if the browser is rebuilt in another framework.

---

## 2. Incomplete Component Specs

Components documented in the styling guide but lacking sufficient detail for consistent reproduction.

### 2.1 Code Annotation Positioning
**Priority:** High

The styling guide describes `.code-annotation.right`, `.code-annotation.left`, and `.code-annotation.top` positions but does not specify:
- How to calculate the `top` offset for multi-line code blocks (the GA course uses inline `style="top: 60px;"` as a manual override)
- Maximum annotation text length before truncation or wrapping
- What happens when multiple annotations overlap vertically
- Arrow/connector line styling between annotation pill and code (course-theme.css includes `::before` pseudo-element arrows, but the styling guide does not mention these)

The HTML preview demonstrates only a single right-positioned annotation with a hardcoded `top: 175px; right: 20px; transform: none;`, which contradicts the documented CSS that uses `transform: translateX(100%)`.

**Recommendation:** Add a multi-annotation example showing stacked annotations. Document the `::before`/`::after` arrow connector pseudo-elements from course-theme.css. Specify min/max annotation widths and the expected inline style pattern for `top` positioning.

### 2.2 Process Flow Responsive Behavior
**Priority:** Medium

The styling guide documents `.flow` as a flex container with `flex-wrap: wrap` but does not specify:
- When steps should wrap to a new row
- How arrow direction changes on wrap (the HTML preview handles this with `@media (max-width: 768px)` by rotating arrows 90 degrees and switching to column layout, but the STYLING_GUIDE.md never mentions this)
- Maximum recommended number of steps (the GA course uses 5 steps in slides, which works, but 7+ steps will overflow on standard Marp slide dimensions)

**Recommendation:** Document the responsive breakpoint behavior. Specify a maximum of 5-6 steps for horizontal flows and recommend vertical orientation for 7+ steps. Add the responsive media query rules to the component spec.

### 2.3 Comparison Card Variants
**Priority:** Medium

The styling guide only documents the before/after (red/green) variant. The GA course uses comparison cards creatively for "Builds On / Leads To" content (slide `01_feature_selection_challenge_slides.md`, line 367-389), applying `.header.before` and `.header.after` for non-before/after semantics. The styling guide does not document:
- Additional header color variants beyond red/green (e.g., blue/amber, two neutral colors)
- How to use comparison cards for non-before/after content (e.g., "Option A / Option B", "Theory / Practice")
- Whether the cross/check mark pseudo-elements (`::before` with `\2717` / `\2713`) should appear for non-before/after uses

**Recommendation:** Add 2-3 comparison card header color variants (`.header.option-a` / `.header.option-b` with blue/amber, for example). Document that the cross/check marks are specific to the before/after variant.

### 2.4 Callout Box Emoji Icons
**Priority:** Low

The styling guide's semantic color sets table (Section 2.2) lists icons (Lightbulb, Warning triangle, Key, Alert, Info circle) but does not specify whether these are Unicode emoji, SVG icons, or CSS `::before` content. The GA course uses Unicode emoji within the callout text (e.g., `<strong>Key Point:</strong>` without icon, or inline emoji like `🔑`). The notebook_style.py `_CALLOUT_STYLES` dict uses specific Unicode codepoints. There is no consistency mandate.

**Recommendation:** Standardize on Unicode emoji for callout labels. Document the exact emoji for each type: Insight = bulb, Warning = triangle, Key = key, Danger = siren, Info = info. Add these to the callout box spec in Section 4.3.

### 2.5 Table Hover State Specifics
**Priority:** Low

The styling guide says tables have "hover highlighting" with `--bg-section-blue` background. Course-theme.css implements `tbody tr:hover { background: var(--bg-section-blue); }`. The HTML preview implements this. However, the guide does not specify:
- Transition duration for hover
- Whether hover applies on mobile/touch devices
- How hover interacts with striped rows (even row hover remains blue, not a blend)

**Recommendation:** Add `transition: background 0.15s ease` to the table spec. Note that hover is desktop-only and overrides the stripe color.

---

## 3. Visual Density Gaps

### 3.1 Graphic Type Selection Matrix
**Priority:** High

Section 8.4 states "one visual element every 2-3 paragraphs" and lists applicable visual types per medium (guides, slides, notebooks). It does not specify which graphic type is best for different content types. The GA course demonstrates effective choices:
- **Conceptual explanations** -> Mermaid flowcharts or SVG architecture diagrams
- **Code walkthroughs** -> annotated code windows
- **Process/workflow** -> `.flow` components or process flow SVGs
- **Tradeoffs/alternatives** -> comparison cards
- **Summary/reference** -> tables
- **Danger/warning/insight** -> callout boxes

But this mapping is implicit, not documented.

**Recommendation:** Add a "Graphic Selection Guide" matrix to Section 8.4:

| Content Type | Recommended Visual | Fallback |
|---|---|---|
| Conceptual/theory | Mermaid diagram or SVG architecture | Process flow |
| Algorithm steps | Process flow (HTML or SVG) | Mermaid flowchart |
| Code explanation | Annotated code window | Code block + callout |
| Comparison/tradeoff | Comparison cards | Side-by-side table |
| Warning/gotcha | Callout box (warning/danger) | Blockquote |
| Summary | Table | Bulleted callout |
| Timeline/history | Timeline SVG | Numbered list |

### 3.2 Notebook Visual Density Enforcement
**Priority:** Medium

Section 8.3 says "a plot, diagram, or styled output every 2-3 code cells" but provides no tooling or linting to enforce this. The GA course notebooks were not audited for compliance as part of this review, but the guide and slide content does maintain good density. The notebook_style.py provides `callout()`, `learning_objectives()`, `section_divider()`, and `key_takeaways()` helpers, which help achieve density, but there is no automated check.

**Recommendation:** Consider adding a `check_visual_density.py` script that parses `.ipynb` files and flags sequences of 4+ consecutive code cells without a markdown cell containing an image, HTML, or plot.

---

## 4. CSS Coverage Gaps

### 4.1 Classes in course-theme.css Not Documented in STYLING_GUIDE.md
**Priority:** High

The following CSS classes/selectors exist in `course-theme.css` but are absent from STYLING_GUIDE.md:

| CSS Selector | Purpose | Documentation Status |
|---|---|---|
| `section.lead` | Title/lead slides with gradient background | Mentioned in Section 8.2 template as `<!-- _class: lead -->` but no CSS spec |
| `section.lead::before` | Radial overlay for visual depth | Not documented |
| `section.lead h1`, `section.lead h2` | Overridden heading styles on lead slides | Not documented |
| `section.lead strong` | Orange accent for bold text on lead slides | Not documented |
| `section.lead blockquote` | Styled blockquote on lead slides | Not documented |
| `section.comparison` | 2-column grid layout for comparison slides | Not documented at all |
| `.flow-step.rose` | Rose color variant for flow steps | Not documented (only mint, amber, blue, lavender listed in Section 4.2) |
| `.badge` (all variants) | Metadata badge pills (mint, amber, blue, lavender, rose) | Mentioned in typography size table but no component spec |
| `.caption` | Centered italic caption text | Not documented as a component |
| `blockquote` | Styled blockquote with orange left border and amber background | Not documented |
| `pre`, `pre code`, `code` (inline) | Standalone code block styling (outside `.code-window`) | Not documented |
| `section::after` | Pagination footer styling | Not documented |
| `strong`, `em` | Emphasis styling (`--text-heading` for bold, `--text-muted` for italic) | Not documented |
| `hr` | Horizontal rule styling | Not documented |
| `img` | Image max-width and border-radius | Not documented |
| `ul`, `ol`, `li` | List margin/spacing | Not documented |
| `.MathJax` | MathJax font size override | Not documented |
| `.mermaid` | Mermaid diagram centering | Not documented |

**Recommendation:** Add a "Base Element Styles" section documenting blockquote, inline code, lists, images, horizontal rules, and emphasis. Add full component specs for `section.lead`, `section.comparison`, `.badge`, and `.caption`. Add `.flow-step.rose` to the process flow documentation.

### 4.2 Classes in STYLING_GUIDE.md Not Implemented in course-theme.css
**Priority:** Low

All classes documented in the styling guide appear to be implemented in course-theme.css. No gaps found in this direction.

### 4.3 Streamlit CSS Classes Not Cross-Referenced
**Priority:** Low

The Streamlit `custom.css` (888 lines) defines many components not documented anywhere in the styling guide: `.hero-section`, `.course-card`, `.module-card`, `.metric-card`, `.breadcrumb`, `.progress-container`, `.slide-viewer-container`, `.content-container`, `.section-divider`, `.file-type-badge`, `.nav-buttons`, `.viewer-header`, etc. These are Streamlit-specific but share the same design tokens.

**Recommendation:** Add a brief "Streamlit Components" appendix to the styling guide, or note in Section 10 (File Map) that `resources/streamlit/custom.css` contains additional browser-specific components that extend the base design system.

---

## 5. Template Gaps

### 5.1 Slide Template Missing Lead Slide Patterns
**Priority:** High

The starter template at `resources/templates/slide_template.md` includes a basic lead slide (`<!-- _class: lead -->`) but does not demonstrate:
- The module-break pattern with a return to content after the break
- The `section.comparison` class (which is in course-theme.css but undocumented)
- Nested comparison cards with code blocks inside them (used extensively in the GA course, e.g., `01_feature_selection_challenge_slides.md` lines 258-297)
- Multiple code windows on a single slide (the GA course does this, e.g., `01_ga_components_slides.md` lines 63-106)

**Recommendation:** Add a "slide_template_advanced.md" with patterns for comparison-with-code, multi-code-window slides, and the comparison class.

### 5.2 Guide Template Missing Connection Patterns
**Priority:** Medium

The guide template at `resources/templates/guide_template.md` lacks:
- A "Connections" section pattern (the GA course uses a `## Connections` section with "Builds On", "Leads To", and "Related To" subsections)
- A "Practice Problems" section pattern (used in the GA course with Question/Solution format)
- A "Further Reading" section with structured citation format
- Multiple SVG diagram references (the template has one placeholder comment; GA guides embed 2-3 SVGs)

**Recommendation:** Add these sections to the guide template. They are present in every GA course guide and represent established patterns.

### 5.3 No Recipe or Quick-Start Template
**Priority:** Medium

The `resources/templates/` directory contains `slide_template.md`, `guide_template.md`, and `notebook_template.ipynb`. The course architecture also includes `recipes/`, `quick-starts/`, `templates/`, and `projects/` directories per course. There are no starter templates for these content types.

**Recommendation:** Add `recipe_template.md`, `quick_start_template.ipynb`, and `project_template.md` with the expected frontmatter and structure.

---

## 6. Color/Typography Inconsistencies

### 6.1 H2 Border Implementation Variance
**Priority:** Medium

The styling guide specifies H2 has a "3px bottom border in `--accent-orange` spanning 60px (via `::after` pseudo-element, not full width)." This is correctly implemented in course-theme.css and styling_guide.html. However, notebook_style.py implements it differently:

```python
# notebook_style.py line 38-39
border-bottom: 3px solid #ff9800;
display: inline-block;
```

The notebook version uses a full `border-bottom` constrained by `display: inline-block` (so width matches text), while the Marp theme uses a `::after` pseudo-element fixed at 60px. These produce visually different results -- the notebook border stretches to text width, the slide border is always 60px.

**Recommendation:** Align the notebook implementation to use a fixed-width approach or document the intentional divergence (inline-block may be appropriate for notebooks where heading widths vary more).

### 6.2 Code Body Text Color Inconsistency
**Priority:** Low

In course-theme.css, `pre code` uses `color: var(--text-on-dark)` (#f5f5f5), which is a warm off-white. The styling_guide.html code window uses `color: #cdd6f4` (Catppuccin Mocha "text" color), which is a cooler blue-white. Both are readable on `--bg-code` (#1e1e2e), but they are different colors.

**Recommendation:** Pick one. The `#cdd6f4` Catppuccin value is more consistent with the syntax highlighting scheme used in the HTML preview. Update course-theme.css `pre code` color to `#cdd6f4` or document the intentional difference.

### 6.3 Font Stack Order: Georgia vs Playfair Display
**Priority:** Low

The styling guide and all CSS files use `Georgia, 'Playfair Display', serif` -- Georgia first. This means Playfair Display (the Google Font) will never be used unless Georgia is unavailable, which is rare since Georgia ships with every modern OS. The `@import` statement loads Playfair Display but it sits unused behind Georgia.

**Recommendation:** If the intent is to prefer Playfair Display when available, reverse the order to `'Playfair Display', Georgia, serif`. If Georgia is the intended primary, consider removing the Playfair Display import to reduce font loading overhead. Document the decision either way.

### 6.4 Slide Body Font Size vs Guide Body Font Size
**Priority:** Low

Section 3.2 specifies Body (slides) = 26px and Body (guides) = 18px. Course-theme.css sets `section { font-size: 26px; }`, matching the slides spec. The guide font size of 18px is not enforced anywhere in CSS -- guides are rendered as markdown, and the browser default (16px) applies. The styling_guide.html uses `font-size: 16px` on `body`.

**Recommendation:** Either implement the 18px guide body size in a guide-specific stylesheet or update the spec to match reality (16px for HTML guides, 18px as a recommendation for print/PDF rendering).

---

## 7. Accessibility Gaps

### 7.1 Accent Colors Used as Text (Identified but Insufficiently Mitigated)
**Priority:** High

Section 2.3 correctly identifies that `--accent-blue` (#2196f3) and `--accent-green` (#4caf50) fail WCAG AA contrast on white backgrounds (3.26:1 and 3.07:1 respectively, both below the 4.5:1 minimum for normal text). The guide states these are "never used as the sole text color for essential content."

However, in practice:
- Course-theme.css uses `a { color: var(--accent-blue); }` -- all hyperlinks are accent blue on white, which is **essential navigation content** at 3.26:1 contrast
- `section.lead strong { color: var(--accent-orange); }` -- bold text on lead slides uses orange, which has adequate contrast on the dark gradient background (this is fine)
- `.compare-card:first-child .body ul li::before { content: "\2717"; color: var(--accent-red); }` -- the cross mark uses red, but this is decorative alongside text (acceptable)

The link color is the most critical issue. Every link in slides, guides, and the Streamlit browser fails AA for normal text.

**Recommendation:** Change link color to a darker blue that passes AA. `#1565c0` (blue 800) provides 7.16:1 contrast on white. Alternatively, add an underline to all links (the CSS currently removes underlines with `text-decoration: none`), which provides a non-color visual indicator per WCAG 1.4.1.

### 7.2 Flow Step Text Colors on Pastel Backgrounds
**Priority:** Medium

The flow step text colors in the HTML preview (e.g., `.flow-step.mint { color: #1b5e20; }`) are taken from the dark variants in the `COLORS` dict. These pass AA against their pastel backgrounds:
- #1b5e20 on #e8f5e9 (mint): ~7.2:1 -- Pass
- #e65100 on #fff8e1 (amber): ~4.7:1 -- Pass (borderline for large text, passes for large, fails for small at some sizes)
- #0d47a1 on #e3f2fd (blue): ~8.0:1 -- Pass
- #4a148c on #f3e5f5 (lavender): ~9.8:1 -- Pass
- #b71c1c on #fce4ec (rose): ~5.8:1 -- Pass

The amber combination (#e65100 on #fff8e1) at 4.7:1 is marginal. At the flow step font size of 0.9rem (~14.4px), this needs 4.5:1 for AA normal text, so it technically passes but has no margin.

**Recommendation:** Darken the amber text to #bf360c (~6.5:1) for a comfortable margin. Document the contrast ratios for all flow step color combinations.

### 7.3 Muted Text on Pastel Backgrounds
**Priority:** Medium

The styling guide documents `--text-muted` (#757575) at 4.48:1 on white, which passes AA for normal text (minimally). However, the guide does not check muted text on pastel backgrounds, which would have lower contrast:
- #757575 on #e8f5e9 (mint): ~3.9:1 -- **Fail** for normal text
- #757575 on #fff8e1 (amber): ~4.0:1 -- **Fail** for normal text
- #757575 on #e3f2fd (blue): ~3.8:1 -- **Fail** for normal text
- #757575 on #f3e5f5 (lavender): ~3.8:1 -- **Fail** for normal text

If muted text is used inside callout boxes or on pastel section backgrounds, it fails AA.

**Recommendation:** Add a rule: "Do not use `--text-muted` as body text on pastel backgrounds. Use `--text-primary` (#212121) instead." Update the contrast ratio table to include muted-on-pastel combinations.

### 7.4 Focus Indicators
**Priority:** Medium

The styling guide and CSS files do not specify focus indicators for interactive elements. The styling_guide.html's TOC links have `border-bottom: 2px solid transparent` with hover state, but no `:focus` or `:focus-visible` style. The Streamlit custom.css removes focus on sidebar buttons (`box-shadow: none`). This fails WCAG 2.4.7 (Focus Visible).

**Recommendation:** Add `:focus-visible` styles for all interactive elements (links, buttons, form inputs). Use a 2px outline in `--accent-blue` with 2px offset as the standard focus indicator.

### 7.5 Font Size Minimums
**Priority:** Low

The smallest documented font size is 0.65em (pagination footer in course-theme.css `section::after`). At a slide base size of 26px, this computes to ~17px, which is fine. For guides at 16px base, 0.65em = ~10.4px, which is below the recommended 12px minimum for legibility. The `.badge` at 0.75em = 12px (borderline). The swatch `.token` class in the HTML preview uses 0.75rem.

**Recommendation:** Set a minimum rendered font size of 12px. Ensure no element computes below this. The pagination footer on guides should use 0.75em minimum.

---

## Summary of Findings by Priority

### High Priority (should fix before creating new courses)
1. **Missing link preview card component** (Section 1.1)
2. **Code annotation positioning spec incomplete** (Section 2.1)
3. **Graphic type selection matrix missing** (Section 3.1)
4. **18 CSS classes/selectors undocumented** (Section 4.1) -- notably `section.lead`, `section.comparison`, `.badge`, `.caption`, blockquote, inline code, and base element styles
5. **Slide template missing advanced patterns** (Section 5.1)
6. **Link color fails WCAG AA** (Section 7.1)

### Medium Priority (should fix in next styling guide revision)
1. Missing gradient section background pattern (Section 1.2)
2. Missing comparison table with icons component (Section 1.4)
3. Missing read-time badge connection to `.badge` class (Section 1.5)
4. Process flow responsive behavior undocumented (Section 2.2)
5. Comparison card variants limited to before/after (Section 2.3)
6. Notebook visual density not enforced (Section 3.2)
7. Guide template missing Connections/Practice/Reading sections (Section 5.2)
8. No recipe/quick-start/project templates (Section 5.3)
9. H2 border implementation diverges between slides and notebooks (Section 6.1)
10. Amber flow step text contrast is marginal (Section 7.2)
11. Muted text fails AA on pastel backgrounds (Section 7.3)
12. No focus indicators defined (Section 7.4)

### Low Priority (track for future improvement)
1. Character/mascot illustrations (Section 1.3)
2. Sidebar documentation in styling guide (Section 1.6)
3. Callout emoji standardization (Section 2.4)
4. Table hover transition spec (Section 2.5)
5. No orphan classes in styling guide (Section 4.2)
6. Streamlit CSS cross-reference (Section 4.3)
7. Code body text color inconsistency (Section 6.2)
8. Font stack order Georgia vs Playfair Display (Section 6.3)
9. Guide body font size not enforced (Section 6.4)
10. Minimum font size enforcement (Section 7.5)
