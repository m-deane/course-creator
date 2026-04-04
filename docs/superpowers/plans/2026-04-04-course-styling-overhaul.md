# Course Styling Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle all course content types (Marp slides, markdown guides, Jupyter notebooks, Streamlit browser) to match the Daily Dose of DS visual quality, and produce a standalone styling guide for reuse across projects.

**Architecture:** Theme Layer approach — a unified `resources/` styling system that all content types inherit from. Marp CSS theme provides slide components, Python modules handle SVG generation and notebook styling, Streamlit app gets a full rewrite. Applied to `genetic-algorithms-feature-selection` as test course.

**Tech Stack:** Marp CLI (slides), Python stdlib `xml.etree.ElementTree` (SVG generation), matplotlib/seaborn (plot theme), Streamlit (course browser), CSS (all styling)

**Spec:** `docs/superpowers/specs/2026-04-04-course-styling-overhaul-design.md`

---

## Task Dependency Map

```
Task 1 (Marp CSS Theme) ─────────────────────┐
Task 2 (SVG Diagram Generator) ──────────────┤
Task 3 (Plot Theme) ─────────────────────────┤──→ Task 7 (Restyle GA Slides)
Task 4 (Notebook Style Injector) ────────────┤──→ Task 8 (Restyle GA Notebooks)
Task 5 (SVG Icon Library) ───────────────────┤──→ Task 9 (Generate GA SVG Diagrams)
Task 6 (Streamlit App Overhaul) ─────────────┤──→ Task 10 (Restyle GA Guides)
                                              └──→ Task 11 (Render Slides + Verify)
Task 12 (Styling Guide + Templates) ── depends on Tasks 1-6 being complete
```

**Parallelizable groups:**
- **Group A (infrastructure, parallel):** Tasks 1, 2, 3, 4, 5, 6
- **Group B (test course, parallel after Group A):** Tasks 7, 8, 9, 10
- **Group C (finalize, sequential after B):** Tasks 11, 12

---

### Task 1: Marp CSS Theme Rewrite

**Files:**
- Rewrite: `resources/themes/course-theme.css`

- [ ] **Step 1: Write the new CSS theme file**

Complete rewrite of `resources/themes/course-theme.css` per the spec. Include:

```css
/* === Font Loading === */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

/* === Marp Theme Registration === */
/* @theme course */

/* === CSS Custom Properties === */
:root {
  /* Backgrounds */
  --bg-primary: #ffffff;
  --bg-section-mint: #e8f5e9;
  --bg-section-amber: #fff8e1;
  --bg-section-blue: #e3f2fd;
  --bg-section-lavender: #f3e5f5;
  --bg-section-rose: #fce4ec;
  --bg-code: #1e1e2e;
  --bg-sidebar: #1a1a2e;

  /* Accents */
  --accent-green: #4caf50;
  --accent-orange: #ff9800;
  --accent-blue: #2196f3;
  --accent-red: #ef5350;
  --accent-purple: #7c4dff;

  /* Text */
  --text-primary: #212121;
  --text-heading: #1a1a2e;
  --text-muted: #757575;
  --text-on-dark: #f5f5f5;

  /* Borders & Shadows */
  --border-light: #e0e0e0;
  --shadow: rgba(0,0,0,0.08);

  /* Fonts */
  --font-heading: Georgia, 'Playfair Display', serif;
  --font-body: 'Inter', 'Segoe UI', system-ui, sans-serif;
  --font-code: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
}
```

Then implement all sections from spec Section 5:
- Base `section` element: 26px body font, `--font-body`, `--text-primary` color, 40px/60px padding
- H1: `--font-heading`, 2.2em, 700 weight, `--text-heading`, `letter-spacing: -0.01em`
- H2: `--font-heading`, 1.6em, 600 weight, 3px bottom border `--accent-orange` (60px wide via `::after`)
- H3: `--font-heading`, 1.2em, 600 weight
- `section.lead`: gradient background, white text, centered, 2.4em H1 with text-shadow
- `section.module-break`: `--bg-section-mint`, centered H2, 4px top border `--accent-green`
- `section.comparison`: inherits `.compare` grid layout
- `.code-window`, `.code-header`, `.dots`, `.dot-red/.dot-yellow/.dot-green`, `.filename`
- `.code-annotation`, `.code-annotation.right/.left/.top` with arrow pseudo-elements
- `.flow`, `.flow-step`, `.flow-step.mint/.amber/.blue/.lavender`, `.flow-arrow`
- `.callout-insight/.callout-warning/.callout-key/.callout-danger/.callout-info`
- `.compare`, `.compare-card`, `.compare-card .header.before/.after`, `.compare-card .body`
- Table styling: `--bg-sidebar` header, striped rows, hover `--bg-section-blue`
- Code blocks: `--bg-code` background, 12px border-radius, `--font-code`
- Blockquotes: left border `--accent-orange`, `--bg-section-amber` background
- Images: max-width 100%, 8px border-radius
- Links: `--accent-blue`, underline on hover

- [ ] **Step 2: Verify theme renders with Marp CLI**

Create a test slide file and render:
```bash
cat > /tmp/test_theme.md << 'SLIDE'
---
marp: true
theme: course
paginate: true
---

<!-- _class: lead -->

# Test Course Title
## Subtitle here

---

## Regular Slide with Code Window

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">test.py</span>
</div>

```python
def hello():
    return "world"
```

</div>

---

## Process Flow

<div class="flow">
<div class="flow-step mint">Step 1</div>
<div class="flow-arrow">→</div>
<div class="flow-step amber">Step 2</div>
<div class="flow-arrow">→</div>
<div class="flow-step blue">Step 3</div>
</div>

---

## Callouts

<div class="callout-insight">

💡 **Key Insight:** This is an insight callout.

</div>

<div class="callout-warning">

⚠️ **Warning:** This is a warning callout.

</div>

SLIDE

npx @marp-team/marp-cli --html --theme-set resources/themes/course-theme.css -- /tmp/test_theme.md -o /tmp/test_theme.html
```

Open `/tmp/test_theme.html` in browser and verify all components render correctly.

- [ ] **Step 3: Commit**

```bash
git add resources/themes/course-theme.css
git commit -m "feat: rewrite Marp CSS theme with DDoDS-inspired components"
```

---

### Task 2: SVG Diagram Generator

**Files:**
- Create: `resources/graphics/__init__.py`
- Create: `resources/graphics/diagram_generator.py`
- Create: `tests/test_diagram_generator.py`

- [ ] **Step 1: Create `resources/graphics/__init__.py`**

```python
from .diagram_generator import (
    process_flow, architecture_diagram, comparison_graphic,
    concept_map, timeline, annotated_code, SVGDocument, COLORS, FONTS
)
from .plot_theme import apply_plot_theme

__all__ = [
    "process_flow", "architecture_diagram", "comparison_graphic",
    "concept_map", "timeline", "annotated_code", "SVGDocument",
    "COLORS", "FONTS", "apply_plot_theme"
]
```

- [ ] **Step 2: Write tests for SVGDocument and process_flow**

Create `tests/test_diagram_generator.py`:

```python
import os
import tempfile
from resources.graphics.diagram_generator import (
    SVGDocument, process_flow, comparison_graphic, concept_map,
    timeline, annotated_code, architecture_diagram, COLORS, FONTS
)


def test_svg_document_save():
    svg = process_flow(["Step 1", "Step 2", "Step 3"])
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as f:
        svg.save(f.name)
        assert os.path.exists(f.name)
        content = open(f.name).read()
        assert "<svg" in content
        assert "Step 1" in content
        assert "Step 2" in content
        os.unlink(f.name)


def test_svg_document_to_string():
    svg = process_flow(["A", "B"])
    result = svg.to_string()
    assert isinstance(result, str)
    assert "<svg" in result
    assert "A" in result


def test_svg_document_to_data_uri():
    svg = process_flow(["X"])
    uri = svg.to_data_uri()
    assert uri.startswith("data:image/svg+xml;base64,")


def test_svg_document_repr_svg():
    svg = process_flow(["Test"])
    html = svg._repr_svg_()
    assert "<svg" in html


def test_process_flow_horizontal():
    svg = process_flow(["Init", "Evaluate", "Select"], colors=["mint", "blue", "amber"])
    content = svg.to_string()
    assert "Init" in content
    assert "Evaluate" in content
    assert COLORS["mint"]["bg"] in content


def test_process_flow_vertical():
    svg = process_flow(["A", "B", "C"], orientation="vertical")
    content = svg.to_string()
    assert "A" in content


def test_process_flow_with_title():
    svg = process_flow(["A", "B"], title="My Flow")
    assert "My Flow" in svg.to_string()


def test_process_flow_default_colors():
    svg = process_flow(["A", "B", "C", "D"])
    content = svg.to_string()
    assert "<svg" in content


def test_comparison_graphic():
    svg = comparison_graphic(
        left={"title": "Before", "items": ["Slow", "Manual"]},
        right={"title": "After", "items": ["Fast", "Automated"]},
        title="Improvement"
    )
    content = svg.to_string()
    assert "Before" in content
    assert "After" in content
    assert "Improvement" in content


def test_concept_map():
    svg = concept_map(
        nodes=[
            {"id": "a", "label": "Node A", "color": "mint"},
            {"id": "b", "label": "Node B", "color": "blue"},
        ],
        edges=[{"from": "a", "to": "b", "label": "connects"}],
        title="Test Map"
    )
    content = svg.to_string()
    assert "Node A" in content
    assert "Node B" in content


def test_timeline():
    svg = timeline(
        events=[
            {"label": "Start", "detail": "Begin here", "color": "mint"},
            {"label": "Middle", "detail": "Halfway", "color": "amber"},
            {"label": "End", "detail": "Done", "color": "blue"},
        ],
        title="Project Timeline"
    )
    content = svg.to_string()
    assert "Start" in content
    assert "Project Timeline" in content


def test_annotated_code():
    svg = annotated_code(
        code="def foo():\n    return 42",
        language="python",
        annotations=[{"line": 1, "text": "Function definition", "side": "right"}],
        filename="example.py"
    )
    content = svg.to_string()
    assert "def foo" in content
    assert "Function definition" in content
    assert "example.py" in content


def test_architecture_diagram():
    svg = architecture_diagram(
        layers=[
            {"name": "Frontend", "nodes": ["UI", "API Client"], "color": "blue"},
            {"name": "Backend", "nodes": ["Server", "Database"], "color": "mint"},
        ],
        connections=[("UI", "Server"), ("Server", "Database")],
        title="System Architecture"
    )
    content = svg.to_string()
    assert "Frontend" in content
    assert "System Architecture" in content


def test_each_function_returns_independent_instance():
    a = process_flow(["A"])
    b = process_flow(["B"])
    assert "A" in a.to_string()
    assert "B" in b.to_string()
    assert "B" not in a.to_string()
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd "/Users/matthewdeane/Documents/Data Science/python/_projects/course-creator"
python -m pytest tests/test_diagram_generator.py -v
```
Expected: ImportError or ModuleNotFoundError

- [ ] **Step 4: Implement `diagram_generator.py`**

Create `resources/graphics/diagram_generator.py` implementing:
- `COLORS` and `FONTS` constants per spec Section 6
- `SVGDocument` class with `save()`, `to_string()`, `to_data_uri()`, `_repr_svg_()` methods
- `process_flow()` — creates numbered step boxes connected by arrows, horizontal or vertical layout
- `architecture_diagram()` — multi-row layer layout with connection lines
- `comparison_graphic()` — two-column side-by-side with colored headers (red left, green right)
- `concept_map()` — positioned nodes with labeled edge lines
- `timeline()` — horizontal or vertical milestones with detail text
- `annotated_code()` — dark background code block with macOS chrome + positioned annotation labels

All using `xml.etree.ElementTree` — no external dependencies. Each function returns a new independent `SVGDocument` instance.

Key implementation details:
- Rounded rectangles via `<rect rx="12" ry="12">`
- Arrow heads via `<defs><marker>` elements
- Text positioning via `<text>` with `text-anchor`, `dominant-baseline`
- Font embedding via `<style>` element within SVG
- Pastel fills from `COLORS` dict
- Consistent 20px padding, 12px border-radius

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_diagram_generator.py -v
```
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add resources/graphics/__init__.py resources/graphics/diagram_generator.py tests/test_diagram_generator.py
git commit -m "feat: add SVG diagram generator with process flow, architecture, comparison, concept map, timeline, and annotated code"
```

---

### Task 3: Matplotlib/Seaborn Plot Theme

**Files:**
- Create: `resources/graphics/plot_theme.py`
- Create: `tests/test_plot_theme.py`

- [ ] **Step 1: Write test**

```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from resources.graphics.plot_theme import apply_plot_theme


def test_apply_plot_theme_sets_color_cycle():
    apply_plot_theme()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = [c['color'] for c in prop_cycle]
    assert colors[0] == '#4caf50'
    assert colors[1] == '#2196f3'


def test_apply_plot_theme_sets_figure_size():
    apply_plot_theme()
    assert plt.rcParams['figure.figsize'] == [10, 6]


def test_apply_plot_theme_sets_white_background():
    apply_plot_theme()
    assert plt.rcParams['figure.facecolor'] == 'white'
    assert plt.rcParams['axes.facecolor'] == 'white'


def test_apply_plot_theme_spine_visibility():
    apply_plot_theme()
    assert plt.rcParams['axes.spines.top'] == False
    assert plt.rcParams['axes.spines.right'] == False


def test_apply_plot_theme_grid():
    apply_plot_theme()
    assert plt.rcParams['axes.grid'] == True
    assert plt.rcParams['grid.color'] == '#f5f5f5'
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_plot_theme.py -v
```

- [ ] **Step 3: Implement `plot_theme.py`**

```python
"""Course-consistent matplotlib/seaborn plot theme."""
import matplotlib.pyplot as plt
import matplotlib as mpl


COURSE_COLORS = ['#4caf50', '#2196f3', '#ff9800', '#7c4dff', '#ef5350', '#00bcd4']


def apply_plot_theme():
    """Apply course-consistent matplotlib/seaborn styling."""
    plt.rcParams.update({
        # Colors
        'axes.prop_cycle': mpl.cycler(color=COURSE_COLORS),
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',

        # Figure
        'figure.figsize': [10, 6],
        'figure.dpi': 100,

        # Fonts
        'font.family': 'sans-serif',
        'font.sans-serif': ['Inter', 'Segoe UI', 'Helvetica', 'Arial', 'sans-serif'],
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelsize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,

        # Spines
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.edgecolor': '#e0e0e0',

        # Grid
        'axes.grid': True,
        'grid.color': '#f5f5f5',
        'grid.linewidth': 1,
        'grid.alpha': 1.0,

        # Legend
        'legend.frameon': False,
        'legend.fontsize': 11,
    })

    # Try to apply seaborn theme if available
    try:
        import seaborn as sns
        sns.set_palette(COURSE_COLORS)
    except ImportError:
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_plot_theme.py -v
```

- [ ] **Step 5: Commit**

```bash
git add resources/graphics/plot_theme.py tests/test_plot_theme.py
git commit -m "feat: add matplotlib/seaborn course plot theme"
```

---

### Task 4: Notebook Style Injector

**Files:**
- Create: `resources/notebook_style.py`
- Create: `tests/test_notebook_style.py`

- [ ] **Step 1: Write tests**

```python
from resources.notebook_style import (
    apply_course_theme, learning_objectives, section_divider,
    callout, key_takeaways
)


def test_apply_course_theme_returns_html():
    result = apply_course_theme()
    # apply_course_theme returns an IPython HTML object
    html_str = result.data
    assert '<style>' in html_str
    assert 'Georgia' in html_str  # serif headings
    assert '#1a1a2e' in html_str  # heading color


def test_learning_objectives():
    result = learning_objectives(["Learn X", "Understand Y"])
    html_str = result.data
    assert "Learn X" in html_str
    assert "Understand Y" in html_str
    assert "#e8f5e9" in html_str  # mint background


def test_section_divider_with_title():
    result = section_divider("Part 2")
    assert "Part 2" in result.data


def test_section_divider_without_title():
    result = section_divider()
    assert "<hr" in result.data or "border" in result.data


def test_callout_insight():
    result = callout("This is important", kind="insight")
    html_str = result.data
    assert "This is important" in html_str
    assert "#f3e5f5" in html_str  # lavender


def test_callout_warning():
    result = callout("Be careful", kind="warning")
    assert "#fff8e1" in result.data  # amber


def test_callout_key():
    result = callout("Key point", kind="key")
    assert "#e8f5e9" in result.data  # mint


def test_callout_danger():
    result = callout("Danger!", kind="danger")
    assert "#fce4ec" in result.data  # rose


def test_callout_info():
    result = callout("FYI", kind="info")
    assert "#e3f2fd" in result.data  # blue


def test_key_takeaways():
    result = key_takeaways(["Point 1", "Point 2"])
    html_str = result.data
    assert "Point 1" in html_str
    assert "Point 2" in html_str
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_notebook_style.py -v
```

- [ ] **Step 3: Implement `notebook_style.py`**

Implement per spec Section 8:
- `apply_course_theme()` — returns `IPython.display.HTML` with `<style>` block for notebook cells. Styles H1/H2 with serif font, DataFrames with striped rows and colored headers, code output with monospace.
- `learning_objectives(items)` — styled mint card with checklist items
- `section_divider(title)` — centered horizontal rule with optional title overlay
- `callout(text, kind)` — colored box matching the 5 semantic types (insight/warning/key/danger/info)
- `key_takeaways(items)` — styled summary card with numbered items

All functions return `IPython.display.HTML` objects. Use inline CSS (not external stylesheets) so notebooks are portable.

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_notebook_style.py -v
```

- [ ] **Step 5: Commit**

```bash
git add resources/notebook_style.py tests/test_notebook_style.py
git commit -m "feat: add notebook style injector with callouts, objectives, and takeaways"
```

---

### Task 5: SVG Icon Library

**Files:**
- Create: `resources/graphics/icons/robot.svg`
- Create: `resources/graphics/icons/brain.svg`
- Create: `resources/graphics/icons/gear.svg`
- Create: `resources/graphics/icons/chart.svg`
- Create: `resources/graphics/icons/database.svg`
- Create: `resources/graphics/icons/dna.svg`
- Create: `resources/graphics/icons/target.svg`
- Create: `resources/graphics/icons/magnifier.svg`
- Create: `resources/graphics/icons/arrow_right.svg`
- Create: `resources/graphics/icons/arrow_down.svg`
- Create: `resources/graphics/icons/checkmark.svg`
- Create: `resources/graphics/icons/warning.svg`

- [ ] **Step 1: Create icons directory**

```bash
mkdir -p "resources/graphics/icons"
```

- [ ] **Step 2: Create all 12 SVG icon files**

Each icon is 24x24 viewBox, single-color (`currentColor`), minimal path data. These are simple line/fill icons in a consistent style — not photorealistic.

Create each icon as a clean SVG with `viewBox="0 0 24 24"`, `fill="currentColor"`, and simple path data. Icons should be immediately recognizable at 24px and scalable to any size.

- [ ] **Step 3: Verify icons render**

Open a few icons in a browser to verify they display correctly at 24px and 48px sizes.

- [ ] **Step 4: Commit**

```bash
git add resources/graphics/icons/
git commit -m "feat: add SVG icon library (12 icons for diagrams)"
```

---

### Task 6: Streamlit App Overhaul

**Files:**
- Rewrite: `app.py`
- Create: `resources/streamlit/custom.css`
- Modify: `.streamlit/config.toml`

- [ ] **Step 1: Update `.streamlit/config.toml`**

```toml
[theme]
primaryColor = "#2196f3"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f5f5f5"
textColor = "#212121"
font = "sans serif"
```

- [ ] **Step 2: Create `resources/streamlit/custom.css`**

Custom CSS for:
- Dark sidebar: target `.stSidebar` with `--bg-sidebar` background, white text
- Module cards: pastel backgrounds, rounded corners, shadows, lesson count badges
- Content type tabs: styled horizontal tab bar
- Slide viewer: dark surround, 800px height
- Metrics cards: icon + number + label grid
- Progress bar: green fill on dark background
- Breadcrumb navigation styling
- Typography: serif headings, `--font-body` body text
- Callout/flow classes (same as Marp theme) for guide rendering

- [ ] **Step 3: Rewrite `app.py`**

Full rewrite per spec Section 9. Key changes from current app:
- Load custom CSS from `resources/streamlit/custom.css` via `st.markdown()`
- Sidebar: dark-styled with collapsible course/module tree, progress bar at bottom
- Main content: breadcrumb navigation, card-based module grid on course landing pages
- Content viewers: tabbed interface (Slides / Guide / Notebook / Exercises)
- Session state: `visited_pages` set for progress tracking
- Preserve all existing course discovery logic (walking `courses/` directory)
- Light theme base with dark sidebar via CSS injection

- [ ] **Step 4: Run Streamlit app locally and verify**

```bash
streamlit run app.py
```

Verify: sidebar is dark, module cards are pastel, slide viewer renders, navigation works.

- [ ] **Step 5: Commit**

```bash
git add app.py resources/streamlit/custom.css .streamlit/config.toml
git commit -m "feat: overhaul Streamlit course browser with DDoDS-inspired design"
```

---

### Task 7: Restyle GA Course — Slide Decks

**Depends on:** Task 1 (Marp CSS Theme), Task 9 (SVG diagrams)

**Files:**
- Modify: All `_slides.md` files in `courses/genetic-algorithms-feature-selection/modules/`

- [ ] **Step 1: List all slide files to restyle**

```bash
find courses/genetic-algorithms-feature-selection -name "*_slides.md" -type f | sort
```

- [ ] **Step 2: Restyle each slide deck**

For each `_slides.md` file:
1. Keep `theme: course` frontmatter (unchanged)
2. Replace old CSS classes (`.callout` → `.callout-info`, `.callout-success` → `.callout-key`, `.columns` → `.compare`)
3. Add code windows around key code blocks: wrap with `.code-window`, `.code-header` with dots + filename, `.code-annotation` labels
4. Add process flow components where sequential steps are explained
5. Add callout boxes (insight/warning/key) for key points
6. Replace inline Mermaid `style` directives with `%%{init}%%` theme header
7. Ensure visual density: graphic/component every 2-3 slides
8. Every slide should have speaker notes: `<!-- Speaker notes: ... -->`

- [ ] **Step 3: Render all restyled slides**

```bash
npx @marp-team/marp-cli --html --theme-set resources/themes/course-theme.css -- "courses/genetic-algorithms-feature-selection/**/*_slides.md"
```

- [ ] **Step 4: Spot-check rendered HTML in browser**

Open 3-4 HTML files and verify: lead slides gradient, code windows have dots, callouts colored, flows render, tables styled.

- [ ] **Step 5: Commit**

```bash
git add courses/genetic-algorithms-feature-selection/
git commit -m "feat: restyle GA course slide decks with new theme components"
```

---

### Task 8: Restyle GA Course — Notebooks

**Depends on:** Task 4 (Notebook Style Injector), Task 3 (Plot Theme)

**Files:**
- Modify: All `.ipynb` files in `courses/genetic-algorithms-feature-selection/modules/`

- [ ] **Step 1: List all notebooks**

```bash
find courses/genetic-algorithms-feature-selection -name "*.ipynb" -type f | sort
```

- [ ] **Step 2: Update each notebook**

For each notebook:
1. Add `from resources.notebook_style import apply_course_theme, learning_objectives, section_divider, callout, key_takeaways` to the imports cell
2. Add `from resources.graphics.plot_theme import apply_plot_theme` to imports
3. Add `apply_course_theme()` and `apply_plot_theme()` calls right after imports
4. Replace plain markdown learning objectives with `learning_objectives([...])` call
5. Add `section_divider()` between major sections
6. Add `callout()` for key insights and warnings
7. Add `key_takeaways()` at end of notebook
8. Ensure matplotlib plots will use the course palette (via `apply_plot_theme()`)

- [ ] **Step 3: Verify notebooks load without errors**

```bash
python -c "
import nbformat
import glob
for nb_path in glob.glob('courses/genetic-algorithms-feature-selection/**/*.ipynb', recursive=True):
    nb = nbformat.read(nb_path, as_version=4)
    print(f'OK: {nb_path} ({len(nb.cells)} cells)')
"
```

- [ ] **Step 4: Commit**

```bash
git add courses/genetic-algorithms-feature-selection/
git commit -m "feat: restyle GA course notebooks with styled theme and components"
```

---

### Task 9: Generate GA Course SVG Diagrams

**Depends on:** Task 2 (SVG Diagram Generator), Task 5 (Icon Library)

**Files:**
- Create: `courses/genetic-algorithms-feature-selection/modules/module_00_foundations/guides/feature_selection_pipeline.svg`
- Create: `courses/genetic-algorithms-feature-selection/modules/module_01_ga_fundamentals/guides/ga_lifecycle.svg`
- Create: `courses/genetic-algorithms-feature-selection/modules/module_01_ga_fundamentals/guides/selection_methods.svg`
- Create: `courses/genetic-algorithms-feature-selection/modules/module_01_ga_fundamentals/guides/crossover_types.svg`
- Create: `courses/genetic-algorithms-feature-selection/modules/module_01_ga_fundamentals/guides/mutation_types.svg`
- Create: `courses/genetic-algorithms-feature-selection/modules/module_02_fitness/guides/fitness_landscape.svg`
- Create: `courses/genetic-algorithms-feature-selection/modules/module_03_time_series/guides/walk_forward_timeline.svg`
- Create: `scripts/generate_ga_diagrams.py`

- [ ] **Step 1: Write diagram generation script**

Create `scripts/generate_ga_diagrams.py` that uses the diagram generator to produce all 7 SVGs:

```python
"""Generate SVG diagrams for the GA Feature Selection course."""
import sys
sys.path.insert(0, '.')

from resources.graphics import process_flow, comparison_graphic, architecture_diagram, timeline, concept_map

BASE = "courses/genetic-algorithms-feature-selection/modules"

# 1. GA Lifecycle — process flow
ga_lifecycle = process_flow(
    steps=["Initialize\nPopulation", "Evaluate\nFitness", "Selection", "Crossover", "Mutation", "New\nGeneration"],
    colors=["mint", "blue", "amber", "lavender", "rose", "mint"],
    title="Genetic Algorithm Lifecycle"
)
ga_lifecycle.save(f"{BASE}/module_01_ga_fundamentals/guides/ga_lifecycle.svg")

# 2. Selection Methods — comparison
selection = comparison_graphic(
    left={"title": "Tournament Selection", "items": ["Pick k random individuals", "Best one wins", "Adjustable pressure via k"]},
    right={"title": "Roulette Wheel", "items": ["Probability ∝ fitness", "Spin wheel to select", "High fitness dominates"]},
    title="Selection Method Comparison"
)
selection.save(f"{BASE}/module_01_ga_fundamentals/guides/selection_methods.svg")

# 3. Crossover Types — process flow
crossover = process_flow(
    steps=["Single-Point", "Two-Point", "Uniform"],
    colors=["mint", "amber", "blue"],
    title="Crossover Operators"
)
crossover.save(f"{BASE}/module_01_ga_fundamentals/guides/crossover_types.svg")

# 4. Mutation Types — process flow
mutation = process_flow(
    steps=["Bit Flip", "Swap", "Scramble"],
    colors=["rose", "lavender", "amber"],
    title="Mutation Operators"
)
mutation.save(f"{BASE}/module_01_ga_fundamentals/guides/mutation_types.svg")

# 5. Fitness Landscape — concept map
fitness = concept_map(
    nodes=[
        {"id": "landscape", "label": "Fitness\nLandscape", "color": "blue"},
        {"id": "global", "label": "Global\nOptimum", "color": "mint"},
        {"id": "local", "label": "Local\nOptima", "color": "amber"},
        {"id": "plateau", "label": "Plateaus", "color": "gray"},
        {"id": "diversity", "label": "Population\nDiversity", "color": "lavender"},
    ],
    edges=[
        {"from": "landscape", "to": "global", "label": "contains"},
        {"from": "landscape", "to": "local", "label": "contains"},
        {"from": "landscape", "to": "plateau", "label": "contains"},
        {"from": "diversity", "to": "local", "label": "escapes"},
    ],
    title="Fitness Landscape Concepts"
)
fitness.save(f"{BASE}/module_02_fitness/guides/fitness_landscape.svg")

# 6. Feature Selection Pipeline — architecture
pipeline = architecture_diagram(
    layers=[
        {"name": "Data", "nodes": ["Raw Features", "Preprocessing"], "color": "gray"},
        {"name": "GA Engine", "nodes": ["Population", "Fitness Eval", "Selection", "Operators"], "color": "mint"},
        {"name": "Output", "nodes": ["Best Features", "Model Training", "Evaluation"], "color": "blue"},
    ],
    connections=[
        ("Preprocessing", "Population"),
        ("Population", "Fitness Eval"),
        ("Fitness Eval", "Selection"),
        ("Selection", "Operators"),
        ("Operators", "Population"),
        ("Best Features", "Model Training"),
        ("Model Training", "Evaluation"),
    ],
    title="GA Feature Selection Pipeline"
)
pipeline.save(f"{BASE}/module_00_foundations/guides/feature_selection_pipeline.svg")

# 7. Walk-Forward Timeline
wf = timeline(
    events=[
        {"label": "Train 1", "detail": "t₀ → t₁", "color": "mint"},
        {"label": "Test 1", "detail": "t₁ → t₂", "color": "blue"},
        {"label": "Train 2", "detail": "t₀ → t₂", "color": "mint"},
        {"label": "Test 2", "detail": "t₂ → t₃", "color": "blue"},
        {"label": "Train 3", "detail": "t₀ → t₃", "color": "mint"},
        {"label": "Test 3", "detail": "t₃ → t₄", "color": "blue"},
    ],
    title="Walk-Forward Validation"
)
wf.save(f"{BASE}/module_03_time_series/guides/walk_forward_timeline.svg")

print("All 7 SVG diagrams generated successfully.")
```

- [ ] **Step 2: Run the script**

```bash
python scripts/generate_ga_diagrams.py
```

- [ ] **Step 3: Verify SVGs render in browser**

Open 2-3 SVGs in a browser to verify they display correctly with pastel colors, readable text, proper layout.

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_ga_diagrams.py courses/genetic-algorithms-feature-selection/modules/*/guides/*.svg
git commit -m "feat: generate SVG diagrams for GA course (lifecycle, selection, crossover, mutation, fitness, pipeline, walk-forward)"
```

---

### Task 10: Restyle GA Course — Guide Markdowns

**Depends on:** Task 1 (for class names), Task 9 (for SVG references)

**Files:**
- Modify: All non-`_slides.md` guide `.md` files in `courses/genetic-algorithms-feature-selection/modules/`

- [ ] **Step 1: List all guide files**

```bash
find courses/genetic-algorithms-feature-selection -path "*/guides/*.md" -not -name "*_slides*" -type f | sort
```

- [ ] **Step 2: Restyle each guide**

For each guide markdown:
1. Add metadata header: `> **Reading time:** ~X min | **Module:** N — Title | **Prerequisites:** ...`
2. Replace old callout classes (`.callout` → `.callout-info`, etc.)
3. Add callout divs for key insights, warnings, and key points
4. Reference SVG diagrams: `![GA Lifecycle](ga_lifecycle.svg)`
5. Ensure visual density: image or callout every 2-3 paragraphs
6. Add cross-reference links to companion slides and notebooks at the end
7. Use `<div class="code-window">` around key code blocks with filename headers

- [ ] **Step 3: Commit**

```bash
git add courses/genetic-algorithms-feature-selection/
git commit -m "feat: restyle GA course guide markdowns with callouts, SVG refs, and visual density"
```

---

### Task 11: Render All Slides + Final Verification

**Depends on:** Tasks 7, 9, 10

**Files:**
- Regenerate: All `.html` files in `courses/genetic-algorithms-feature-selection/`

- [ ] **Step 1: Delete old HTML renders**

```bash
find courses/genetic-algorithms-feature-selection -name "*_slides.html" -type f -delete
```

- [ ] **Step 2: Render all slides with new theme**

```bash
npx @marp-team/marp-cli --html --theme-set resources/themes/course-theme.css -- "courses/genetic-algorithms-feature-selection/**/*_slides.md"
```

- [ ] **Step 3: Verify all renders succeed**

```bash
find courses/genetic-algorithms-feature-selection -name "*_slides.html" -type f | wc -l
```

Compare count to number of `_slides.md` files — should match.

- [ ] **Step 4: Visual spot-check**

Open at least one HTML from each module in browser. Verify:
- Lead slides have gradient background
- Code windows have macOS chrome (dots + filename)
- Process flows render with colored steps
- Callouts have correct pastel backgrounds
- Tables have dark headers with striped rows
- Mermaid diagrams use course palette

- [ ] **Step 5: Commit**

```bash
git add courses/genetic-algorithms-feature-selection/
git commit -m "feat: regenerate all GA course slide HTML with new theme"
```

---

### Task 12: Styling Guide + Templates

**Depends on:** Tasks 1-6

**Files:**
- Create: `STYLING_GUIDE.md`
- Create: `styling_guide.html`
- Create: `resources/templates/slide_template.md`
- Create: `resources/templates/guide_template.md`
- Create: `resources/templates/notebook_template.ipynb`

- [ ] **Step 1: Write `STYLING_GUIDE.md`**

Comprehensive portable document per spec Section 10. Include:
1. Design Principles
2. Color Reference (full token table + contrast ratios)
3. Typography Spec
4. Component Library (every CSS class with HTML markup)
5. SVG Graphics Guide (API reference + examples)
6. Mermaid Theme Config (copy-paste `%%{init}%%` snippet)
7. Plot Theme config
8. Content Conventions (guide/slide/notebook templates)
9. Quick Start Checklist
10. File Map

- [ ] **Step 2: Write `styling_guide.html`**

Standalone HTML file with embedded CSS showing live component previews:
- Color swatches
- Typography scale
- Code window demo
- Process flow demo
- All 5 callout types
- Comparison cards
- Table styling
- Interactive hover states

- [ ] **Step 3: Create starter templates**

`resources/templates/slide_template.md`:
```markdown
---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Module Title
## Course Name

<!-- Speaker notes: Welcome to this module... -->

---

## Topic Heading

Content here.

<div class="callout-key">

🔑 **Key Point:** Important concept here.

</div>

---

## Code Example

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# Your code here
```

</div>

<!-- Speaker notes: Explain the code... -->
```

`resources/templates/guide_template.md`:
```markdown
# Guide Title

> **Reading time:** ~X min | **Module:** N — Module Name | **Prerequisites:** None

---

## Introduction

Opening paragraph.

<div class="callout-insight">

💡 **Key Insight:** Main takeaway.

</div>

## Section

Content with visual density — image or component every 2-3 paragraphs.

![Diagram](diagram_name.svg)

## Summary

- Key point 1
- Key point 2

**Next:** [Companion Slides](./guide_name_slides.md) | [Notebook](../notebooks/notebook_name.ipynb)
```

`resources/templates/notebook_template.ipynb` — a Jupyter notebook JSON with:
- Cell 1: markdown title + learning objectives placeholder
- Cell 2: code cell with theme imports and `apply_course_theme()` + `apply_plot_theme()`
- Cell 3: markdown section header
- Cell 4: code cell placeholder
- Cell 5: markdown summary with key takeaways placeholder

- [ ] **Step 4: Commit**

```bash
git add STYLING_GUIDE.md styling_guide.html resources/templates/
git commit -m "feat: add standalone styling guide, HTML preview, and starter templates"
```

---

## Verification Checklist

After all tasks complete:

- [ ] `npx @marp-team/marp-cli --html --theme-set resources/themes/course-theme.css -- /tmp/test_theme.md` renders all components
- [ ] `python -m pytest tests/ -v` — all tests pass
- [ ] `python scripts/generate_ga_diagrams.py` — generates 7 SVGs without errors
- [ ] `streamlit run app.py` — app loads with new styling
- [ ] GA course slides render with new theme (open 3+ HTML files)
- [ ] `STYLING_GUIDE.md` is self-contained and references all components
- [ ] `styling_guide.html` opens in browser with live component demos
