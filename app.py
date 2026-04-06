"""
Course Content Browser — Streamlit App

Browse all courses, slides, notebooks, and guides.
Deploy free on Streamlit Community Cloud: https://streamlit.io/cloud

Design: DDoDS-inspired light theme with dark sidebar, pastel module cards,
progress tracking, and tabbed content viewers.
"""

import json
import re
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
COURSES_DIR = PROJECT_ROOT / "courses"
CUSTOM_CSS_PATH = PROJECT_ROOT / "resources" / "streamlit" / "custom.css"

ACRONYMS = {
    "llm": "LLM", "llms": "LLMs", "rag": "RAG", "eia": "EIA",
    "hmm": "HMM", "hmms": "HMMs", "ai": "AI", "genai": "GenAI",
    "ab": "A/B", "mab": "MAB", "api": "API", "ml": "ML",
    "nlp": "NLP", "gpt": "GPT", "lda": "LDA", "dml": "DML",
}

COURSE_ICONS = {
    "agentic-ai-llms": "\U0001f916",
    "agentic-ai-practical": "\U0001f916",
    "ai-engineer-fundamentals": "\u2699\ufe0f",
    "bayesian-commodity-forecasting": "\U0001f4ca",
    "bayesian-prompt-engineering": "\U0001f4ca",
    "captum-neural-network-interpretability": "\U0001f9e0",
    "causalpy-interrupted-time-series": "\U0001f4c8",
    "dataiku-genai": "\U0001f9e9",
    "double-machine-learning": "\U0001f3af",
    "dynamic-factor-models": "\U0001f4d0",
    "genai-commodities": "\U0001f4ac",
    "genetic-algorithms-feature-selection": "\U0001f9ec",
    "hidden-markov-models": "\U0001f517",
    "midas-mixed-frequency-nowcasting": "\U0001f554",
    "multi-armed-bandits-ab-testing": "\U0001f3b0",
    "panel-regression": "\U0001f4c8",
    "power-automate": "\u26a1",
    "reinforcement-learning": "\U0001f3ae",
    "time-series-forecasting-neuralforecast": "\U0001f4c9",
}

CONTENT_SECTIONS = [
    ("quick-starts", "Quick Starts"),
    ("templates", "Templates"),
    ("recipes", "Recipes"),
    ("concepts", "Concepts"),
    ("projects", "Projects"),
    ("capstone", "Capstone"),
]

FILE_ICONS = {
    "markdown": "\U0001f4dd",
    "notebook": "\U0001f4d3",
    "python": "\U0001f40d",
    "html": "\U0001f310",
    "csv": "\U0001f4ca",
    "yaml": "\u2699\ufe0f",
    "json": "{ }",
    "text": "\U0001f4c4",
}

TYPE_BADGES = {
    "html": ("Slide Deck", "#ff9800"),
    "markdown": ("Guide", "#2196f3"),
    "notebook": ("Notebook", "#4caf50"),
    "python": ("Python", "#7c4dff"),
    "exercises": ("Exercise", "#ef5350"),
    "csv": ("Data", "#757575"),
    "yaml": ("Config", "#757575"),
    "json": ("Config", "#757575"),
    "text": ("Text", "#757575"),
}

SECTION_ICONS = {
    "guides": "\U0001f4dd",
    "notebooks": "\U0001f4d3",
    "exercises": "\u270d\ufe0f",
    "resources": "\U0001f4ce",
}

SECTION_LABELS = {
    "guides": "Guides",
    "notebooks": "Notebooks",
    "exercises": "Exercises",
    "resources": "Resources",
}

# Pastel color cycle for module cards
MODULE_COLORS = ["mint", "amber", "blue", "lavender", "rose"]


# ---------------------------------------------------------------------------
# Custom CSS Injection
# ---------------------------------------------------------------------------

def _load_custom_css() -> str:
    """Load custom CSS text from resources/streamlit/custom.css (cached)."""
    if CUSTOM_CSS_PATH.exists():
        return CUSTOM_CSS_PATH.read_text(encoding="utf-8")
    return ""


_CUSTOM_CSS_CACHE: str | None = None


def _get_custom_css() -> str:
    """Return cached CSS string."""
    global _CUSTOM_CSS_CACHE
    if _CUSTOM_CSS_CACHE is None:
        _CUSTOM_CSS_CACHE = _load_custom_css()
    return _CUSTOM_CSS_CACHE


def inject_custom_css():
    """Inject custom CSS into the main Streamlit page via st.markdown.

    This styles Streamlit-native elements (sidebar, buttons, tabs, etc.).
    For st.html() calls (which render in isolated iframes), use styled_html().
    """
    css = _get_custom_css()
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def styled_html(html_content: str):
    """Render HTML inside st.html() with the custom CSS inlined.

    st.html() renders in an iframe with its own DOM, so it does NOT inherit
    CSS injected via st.markdown().  This helper bundles the stylesheet
    inside the iframe so all custom classes work correctly.
    """
    css = _get_custom_css()
    fonts = (
        '<link rel="stylesheet" '
        'href="https://fonts.googleapis.com/css2?'
        'family=Inter:wght@400;500;600;700&'
        'family=Playfair+Display:wght@600;700&display=swap">'
    )
    st.html(f"{fonts}<style>{css}</style>{html_content}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def humanize(name: str) -> str:
    parts = name.replace("-", " ").replace("_", " ").split()
    result = []
    for p in parts:
        low = p.lower()
        if low in ACRONYMS:
            result.append(ACRONYMS[low])
        else:
            result.append(p.capitalize())
    return " ".join(result)


def humanize_filename(filename: str) -> str:
    """Convert a filename into a human-readable title."""
    stem = Path(filename).stem
    for suffix in ("_slides", "_guide", "_exercises", "_exercise", "_notebook"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    stem = re.sub(r"^\d+[a-z]?_", "", stem)
    return humanize(stem)


def extract_readme_info(readme_path: Path) -> tuple:
    if not readme_path.exists():
        return "", ""
    text = readme_path.read_text(encoding="utf-8", errors="replace")
    title, desc = "", ""
    for line in text.split("\n"):
        if line.startswith("# ") and not title:
            title = line[2:].strip()
        elif title and not desc and line.strip() and not line.startswith("#"):
            raw = line.strip()
            if raw.startswith(">") or raw.startswith("![") or raw.startswith("[!"):
                continue
            raw = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", raw)
            raw = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", raw)
            if raw:
                desc = raw
                break
    return title, desc


def get_file_type_badge_html(file_type: str) -> str:
    label, color = TYPE_BADGES.get(file_type, ("File", "#757575"))
    return (
        f'<span class="file-type-badge" '
        f'style="background:{color}15; color:{color}; border: 1px solid {color}33;">'
        f'{label}</span>'
    )


def get_module_number(dir_name: str) -> str:
    """Extract module number from dir name like module_01_topic."""
    parts = dir_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return ""


def get_module_clean_title(mod: dict) -> str:
    """Get a clean module title without leading number/dash or 'Module N:' prefix."""
    title = mod["name"]
    title = re.sub(r"^\d+\s*[—\-]\s*", "", title)
    title = re.sub(r"^Module\s+\d+\s*:\s*", "", title)
    return title


# ---------------------------------------------------------------------------
# Progress Tracking
# ---------------------------------------------------------------------------

def track_page_visit(course_slug: str, module_dir: str, content_type: str, filename: str):
    """Record a page visit for progress tracking."""
    if "visited_pages" not in st.session_state:
        st.session_state.visited_pages = set()
    st.session_state.visited_pages.add((course_slug, module_dir, content_type, filename))


def get_course_progress(courses: dict, course_slug: str) -> tuple:
    """Return (visited_count, total_count) for a course."""
    course = courses.get(course_slug)
    if not course:
        return 0, 0

    total = 0
    visited = 0
    visited_pages = st.session_state.get("visited_pages", set())

    for mod in course["modules"]:
        for sl in mod["slides"]:
            total += 1
            key = (course_slug, mod["dir_name"], "slides", Path(sl["path"]).name)
            if key in visited_pages:
                visited += 1
        for sec_key in ["guides", "notebooks", "exercises", "resources"]:
            for f in mod["content"].get(sec_key, []):
                total += 1
                key = (course_slug, mod["dir_name"], sec_key, f["name"])
                if key in visited_pages:
                    visited += 1

    for folder, sec in course["sections"].items():
        for f in sec["files"]:
            total += 1
            key = (course_slug, folder, "section", f["name"])
            if key in visited_pages:
                visited += 1

    return visited, total


# ---------------------------------------------------------------------------
# Content Scanner (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def scan_all_courses() -> dict:
    courses = {}
    if not COURSES_DIR.exists():
        return courses
    for d in sorted(COURSES_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            courses[d.name] = _scan_course(d)
    return courses


def _scan_course(course_dir: Path) -> dict:
    readme = course_dir / "README.md"
    title, desc = extract_readme_info(readme)
    if not title:
        title = humanize(course_dir.name)

    modules = []
    mod_dir = course_dir / "modules"
    if mod_dir.exists():
        for m in sorted(mod_dir.iterdir()):
            if m.is_dir() and m.name.startswith("module_"):
                modules.append(_scan_module(m))

    sections = {}
    for folder, label in CONTENT_SECTIONS:
        sec_dir = course_dir / folder
        if sec_dir.exists() and sec_dir.is_dir():
            files = _list_content_files(sec_dir)
            if files:
                sections[folder] = {"label": label, "files": files}

    stats = {
        "slides": len(list(course_dir.rglob("*_slides.html"))),
        "notebooks": len(list(course_dir.rglob("*.ipynb"))),
        "guides": len([f for f in course_dir.rglob("*.md")
                       if f.name != "README.md" and "_slides" not in f.name]),
        "modules": 0,
    }
    if mod_dir.exists():
        stats["modules"] = len([d for d in mod_dir.iterdir()
                                if d.is_dir() and d.name.startswith("module_")])

    return {
        "title": title,
        "description": desc,
        "dir": str(course_dir.relative_to(PROJECT_ROOT)),
        "modules": modules,
        "sections": sections,
        "stats": stats,
    }


def _scan_module(mod_dir: Path) -> dict:
    readme = mod_dir / "README.md"
    title, _ = extract_readme_info(readme)
    if not title:
        parts = mod_dir.name.split("_", 2)
        num = parts[1] if len(parts) > 1 else ""
        name = humanize(parts[2]) if len(parts) > 2 else humanize(mod_dir.name)
        title = f"{num} — {name}" if num else name

    content = {}
    for sub in ["guides", "notebooks", "exercises", "resources"]:
        sub_dir = mod_dir / sub
        if sub_dir.exists():
            files = _list_content_files(sub_dir)
            if sub == "guides":
                # Filter out _slides.md and _slides.html — those are handled
                # separately as slide decks, not study guides
                files = [f for f in files if "_slides" not in f["name"]]
            if files:
                content[sub] = files

    slides = []
    guides_dir = mod_dir / "guides"
    if guides_dir.exists():
        for f in sorted(guides_dir.iterdir()):
            if f.suffix == ".html" and "_slides" in f.name:
                slides.append({
                    "name": f.stem.replace("_", " ").title(),
                    "path": str(f.relative_to(PROJECT_ROOT)),
                })

    return {
        "name": title,
        "dir_name": mod_dir.name,
        "dir": str(mod_dir.relative_to(PROJECT_ROOT)),
        "content": content,
        "slides": slides,
    }


def _list_content_files(directory: Path) -> list:
    files = []
    for f in sorted(directory.rglob("*")):
        if f.is_file() and not f.name.startswith("."):
            ext = f.suffix.lower()
            type_map = {
                ".md": "markdown", ".ipynb": "notebook", ".py": "python",
                ".html": "html", ".txt": "text", ".yaml": "yaml",
                ".yml": "yaml", ".json": "json", ".csv": "csv",
            }
            if ext in type_map:
                files.append({
                    "name": f.name,
                    "path": str(f.relative_to(PROJECT_ROOT)),
                    "type": type_map[ext],
                    "size": f.stat().st_size,
                })
    return files


# ---------------------------------------------------------------------------
# Content Renderers
# ---------------------------------------------------------------------------

def _resolve_images(text: str, base_dir: Path) -> str:
    """Replace relative image paths with base64 data URIs so they render in st.markdown."""
    import base64

    def _replace_img(match):
        alt = match.group(1)
        img_path_str = match.group(2)
        # Skip URLs
        if img_path_str.startswith(("http://", "https://", "data:")):
            return match.group(0)
        # Resolve relative to the markdown file's directory
        img_path = base_dir / img_path_str
        if not img_path.exists():
            return match.group(0)  # leave as-is if not found
        suffix = img_path.suffix.lower()
        if suffix == ".svg":
            svg_content = img_path.read_text(encoding="utf-8", errors="replace")
            encoded = base64.b64encode(svg_content.encode("utf-8")).decode("ascii")
            return f'<img src="data:image/svg+xml;base64,{encoded}" alt="{alt}" style="max-width:100%; height:auto; margin: 1rem 0; border-radius: 8px;">'
        elif suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            mime = {
                ".png": "image/png", ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg", ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(suffix, "image/png")
            img_bytes = img_path.read_bytes()
            encoded = base64.b64encode(img_bytes).decode("ascii")
            return f'<img src="data:{mime};base64,{encoded}" alt="{alt}" style="max-width:100%; height:auto; margin: 1rem 0; border-radius: 8px;">'
        return match.group(0)

    # Match markdown image syntax: ![alt](path)
    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', _replace_img, text)


def render_markdown_file(file_path: Path):
    text = file_path.read_text(encoding="utf-8", errors="replace")
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            frontmatter = text[3:end]
            if "marp:" in frontmatter:
                text = text[end + 3:].lstrip("\n")
    # Resolve relative image paths to inline data URIs
    text = _resolve_images(text, file_path.parent)

    # Split content into markdown and HTML blocks.
    # st.markdown strips complex nested divs, so we render HTML blocks
    # (code-window, callout-*, flow, compare) via styled_html() and
    # the rest via st.markdown().
    _render_mixed_content(text)


def _render_mixed_content(text: str):
    """Render mixed markdown + HTML content.

    Strategy: walk through lines, accumulate markdown text, and when we
    encounter an HTML component block (code-window, callout-*, flow, compare)
    flush the accumulated markdown, then render the HTML block with styled_html().
    This avoids regex issues with code blocks inside HTML divs.
    """
    _COMPONENT_CLASSES = ("code-window", "callout-", "flow", "compare")
    lines = text.split("\n")
    md_buffer = []
    html_buffer = []
    html_depth = 0  # track nested div depth inside a component block

    def flush_markdown():
        chunk = "\n".join(md_buffer).strip()
        md_buffer.clear()
        if chunk:
            _render_markdown_chunk(chunk)

    def flush_html():
        chunk = "\n".join(html_buffer).strip()
        html_buffer.clear()
        if chunk:
            _render_html_component(chunk)

    for line in lines:
        stripped = line.strip()

        if html_depth > 0:
            # We're inside an HTML component block — accumulate
            html_buffer.append(line)
            html_depth += stripped.count("<div")
            html_depth -= stripped.count("</div>")
            if html_depth <= 0:
                html_depth = 0
                flush_html()
        elif any(
            f'class="{cls}' in stripped
            for cls in _COMPONENT_CLASSES
        ) and stripped.startswith("<div"):
            # Starting a new HTML component block
            flush_markdown()
            html_buffer.append(line)
            html_depth = stripped.count("<div") - stripped.count("</div>")
            if html_depth <= 0:
                html_depth = 0
                flush_html()
        else:
            md_buffer.append(line)

    # Flush remaining
    flush_markdown()
    if html_buffer:
        flush_html()


def _render_html_component(chunk: str):
    """Render an HTML component block. For code-windows, extract the fenced
    code and render the header via styled_html() + code via st.code().
    For other components (callouts, flows, comparisons), render via styled_html()."""
    if 'class="code-window"' in chunk:
        # Extract code fence from the code-window
        fence_match = re.search(r'```(\w*)\n(.*?)```', chunk, re.DOTALL)
        if fence_match:
            lang = fence_match.group(1) or 'python'
            code = fence_match.group(2).rstrip('\n')

            # Render the header (everything before the code fence)
            header_html = chunk[:fence_match.start()].strip()
            # Remove trailing blank lines from header
            if header_html:
                styled_html(header_html + '\n</div>')

            # Render the code with syntax highlighting
            st.code(code, language=lang)
        else:
            # No code fence found — render as-is
            styled_html(chunk)
    else:
        # Callouts, flows, comparisons — render as styled HTML
        styled_html(chunk)


def _render_markdown_chunk(text: str):
    """Render a chunk of pure markdown, extracting $$...$$ LaTeX blocks."""
    latex_pattern = re.compile(r'(\$\$.*?\$\$)', re.DOTALL)
    segments = latex_pattern.split(text)
    for seg in segments:
        seg_stripped = seg.strip()
        if not seg_stripped:
            continue
        if seg_stripped.startswith('$$') and seg_stripped.endswith('$$'):
            inner = seg_stripped[2:-2].strip()
            st.latex(inner)
        else:
            st.markdown(seg, unsafe_allow_html=True)


def render_notebook_file(file_path: Path):
    data = json.loads(file_path.read_text(encoding="utf-8", errors="replace"))
    cells = data.get("cells", [])

    # Content container (Streamlit handles layout)
    for cell in cells:
        ctype = cell.get("cell_type", "code")
        source = "".join(cell.get("source", []))

        if not source.strip():
            continue

        if ctype == "markdown":
            st.markdown(source, unsafe_allow_html=True)
        elif ctype == "code":
            st.code(source, language="python")
            for out in cell.get("outputs", []):
                out_type = out.get("output_type", "")
                if out_type == "stream":
                    text = "".join(out.get("text", []))
                    st.text(text)
                elif out_type in ("execute_result", "display_data"):
                    od = out.get("data", {})
                    if "image/png" in od:
                        import base64
                        img_bytes = base64.b64decode(od["image/png"])
                        st.image(img_bytes)
                    elif "text/html" in od:
                        html = "".join(od["text/html"])
                        st.html(html)
                    elif "text/plain" in od:
                        st.text("".join(od["text/plain"]))
                elif out_type == "error":
                    tb = "\n".join(out.get("traceback", []))
                    tb = re.sub(r'\x1b\[[0-9;]*m', '', tb)
                    st.error(tb)
    # End content container


def render_python_file(file_path: Path):
    code = file_path.read_text(encoding="utf-8", errors="replace")
    # Content container (Streamlit handles layout)
    st.code(code, language="python")
    # End content container


def render_html_slides(file_path: Path):
    """Render HTML slides in an iframe with dark surround."""
    html = file_path.read_text(encoding="utf-8", errors="replace")
    st.components.v1.html(html, height=800, scrolling=False)


def render_csv_file(file_path: Path):
    import pandas as pd
    # Content container (Streamlit handles layout)
    df = pd.read_csv(file_path)
    st.dataframe(df, use_container_width=True)
    # End content container


def render_text_file(file_path: Path):
    text = file_path.read_text(encoding="utf-8", errors="replace")
    # Content container (Streamlit handles layout)
    st.code(text)
    # End content container


def render_file(rel_path: str):
    """Render any supported file by relative path."""
    full = PROJECT_ROOT / rel_path
    if not full.exists():
        st.error(f"File not found: {rel_path}")
        return

    ext = full.suffix.lower()
    if ext == ".md":
        render_markdown_file(full)
    elif ext == ".ipynb":
        render_notebook_file(full)
    elif ext == ".py":
        render_python_file(full)
    elif ext == ".html":
        render_html_slides(full)
    elif ext == ".csv":
        render_csv_file(full)
    else:
        render_text_file(full)


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------

def nav_to(page: str, **kwargs):
    st.session_state["page"] = page
    for k, v in kwargs.items():
        st.session_state[k] = v


def render_breadcrumb(items: list):
    """Render a breadcrumb trail.

    items: list of (label, page, kwargs) for navigable items, or (label,) for current page.
    """
    parts = []
    for item in items:
        if len(item) == 1:
            parts.append(f'<span class="breadcrumb-current">{item[0]}</span>')
        else:
            parts.append(f'<span class="breadcrumb-item">{item[0]}</span>')

    breadcrumb_html = ' <span class="breadcrumb-separator">\u203a</span> '.join(parts)
    styled_html(f'<div class="breadcrumb">{breadcrumb_html}</div>')

    nav_items = [item for item in items if len(item) >= 3]
    if nav_items:
        cols = st.columns(len(nav_items) + 2)
        for idx, item in enumerate(nav_items):
            label, page, kwargs = item
            with cols[idx]:
                if st.button(f"\u21a9 {label}", key=f"bc_{label}_{page}", type="secondary"):
                    nav_to(page, **kwargs)
                    st.rerun()


def find_adjacent_files(courses: dict, slug: str, current_path: str):
    """Find previous and next files within the same module/section."""
    course = courses.get(slug)
    if not course:
        return None, None

    all_files = []

    for folder, sec in course["sections"].items():
        for f in sec["files"]:
            all_files.append(f["path"])

    for mod in course["modules"]:
        for sl in mod["slides"]:
            all_files.append(sl["path"])
        for sec_key in ["guides", "notebooks", "exercises", "resources"]:
            for f in mod["content"].get(sec_key, []):
                all_files.append(f["path"])

    if current_path not in all_files:
        return None, None

    idx = all_files.index(current_path)
    prev_path = all_files[idx - 1] if idx > 0 else None
    next_path = all_files[idx + 1] if idx < len(all_files) - 1 else None
    return prev_path, next_path


def find_module_for_file(course: dict, file_path: str) -> dict | None:
    """Find which module a file belongs to."""
    for mod in course.get("modules", []):
        for sl in mod["slides"]:
            if sl["path"] == file_path:
                return mod
        for sec_key in ["guides", "notebooks", "exercises", "resources"]:
            for f in mod["content"].get(sec_key, []):
                if f["path"] == file_path:
                    return mod
    return None


def get_file_type_from_path(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    type_map = {
        ".md": "markdown", ".ipynb": "notebook", ".py": "python",
        ".html": "html", ".txt": "text", ".yaml": "yaml",
        ".yml": "yaml", ".json": "json", ".csv": "csv",
    }
    return type_map.get(ext, "text")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(courses: dict):
    """Render sidebar with course/module navigation tree."""
    current_slug = st.session_state.get("course_slug", "")
    current_module = st.session_state.get("module_dir", "")
    search = st.session_state.get("search", "").lower()

    with st.sidebar:
        st.markdown("### \U0001f393 Course Browser")

        # Home button
        if st.button("\U0001f3e0  Home", key="sidebar_home", use_container_width=True):
            nav_to("home")
            st.rerun()

        st.divider()

        # Search
        st.text_input(
            "Search courses",
            key="search",
            placeholder="\U0001f50d Search courses...",
            label_visibility="collapsed",
        )

        # Course navigation tree — each course is an expander
        for slug, course in courses.items():
            if search and search not in slug and search not in course["title"].lower():
                continue

            icon = COURSE_ICONS.get(slug, "\U0001f4d6")
            is_active = slug == current_slug
            stats = course["stats"]

            with st.expander(
                f"{icon} {course['title']}",
                expanded=is_active,
            ):
                # Course stats
                st.caption(
                    f"{stats['modules']} modules \u2022 "
                    f"{stats['slides']} slides \u2022 "
                    f"{stats['notebooks']} notebooks"
                )

                # Course overview button
                btn_type = "primary" if is_active else "secondary"
                if st.button(
                    "\U0001f4cb Course Overview",
                    key=f"nav_{slug}",
                    use_container_width=True,
                    type=btn_type,
                ):
                    nav_to("course", course_slug=slug)
                    st.rerun()

                # Module list
                if course["modules"]:
                    for mod in course["modules"]:
                        mod_num = get_module_number(mod["dir_name"])
                        mod_title = get_module_clean_title(mod)
                        is_current_mod = (is_active and mod["dir_name"] == current_module)

                        n_guides = len(mod["content"].get("guides", []))
                        n_notebooks = len(mod["content"].get("notebooks", []))
                        n_slides = len(mod["slides"])

                        # Module button — highlighted if current
                        prefix = "\u25b6 " if is_current_mod else "\u00a0\u00a0 "
                        label = f"{prefix}{mod_num}. {mod_title}"
                        if st.button(
                            label,
                            key=f"sidebar_mod_{slug}_{mod['dir_name']}",
                            use_container_width=True,
                            type="primary" if is_current_mod else "secondary",
                        ):
                            nav_to("module", course_slug=slug, module_dir=mod["dir_name"])
                            st.rerun()

                        # Content counts for current module
                        if is_current_mod:
                            st.caption(
                                f"\u00a0\u00a0\u00a0\u00a0"
                                f"{n_slides} slides \u2022 "
                                f"{n_guides} guides \u2022 "
                                f"{n_notebooks} notebooks"
                            )

                # Progress for active course
                if is_active:
                    visited, total = get_course_progress(courses, slug)
                    pct = int((visited / total * 100) if total > 0 else 0)
                    st.progress(pct / 100, text=f"Progress: {visited}/{total} pages")



# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_home(courses: dict):
    """Home page with hero section, metrics grid, and course cards."""

    # Aggregate stats
    total = {"courses": len(courses), "modules": 0, "slides": 0, "notebooks": 0, "guides": 0}
    for c in courses.values():
        for k in ("modules", "slides", "notebooks", "guides"):
            total[k] += c["stats"][k]

    # Hero section
    styled_html(f"""
    <div class="hero-section">
        <p class="hero-title">Course Browser</p>
        <p class="hero-subtitle">
            Practical-first courses spanning ML, GenAI, econometrics, and trading systems
        </p>
        <div class="stat-pills">
            <span class="stat-pill"><span class="stat-value">{total["courses"]}</span> Courses</span>
            <span class="stat-pill"><span class="stat-value">{total["modules"]}</span> Modules</span>
            <span class="stat-pill"><span class="stat-value">{total["slides"]}</span> Slide Decks</span>
            <span class="stat-pill"><span class="stat-value">{total["notebooks"]}</span> Notebooks</span>
            <span class="stat-pill"><span class="stat-value">{total["guides"]}</span> Guides</span>
        </div>
    </div>
    """)

    # Metrics cards
    styled_html(f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <span class="metric-icon">\U0001f4da</span>
            <div class="metric-value">{total["courses"]}</div>
            <div class="metric-label">Courses</div>
        </div>
        <div class="metric-card">
            <span class="metric-icon">\U0001f4e6</span>
            <div class="metric-value">{total["modules"]}</div>
            <div class="metric-label">Modules</div>
        </div>
        <div class="metric-card">
            <span class="metric-icon">\U0001f4d3</span>
            <div class="metric-value">{total["notebooks"]}</div>
            <div class="metric-label">Notebooks</div>
        </div>
        <div class="metric-card">
            <span class="metric-icon">\U0001f310</span>
            <div class="metric-value">{total["slides"]}</div>
            <div class="metric-label">Slide Decks</div>
        </div>
    </div>
    """)

    # Search filter
    search = st.session_state.get("search", "").lower()
    filtered = {k: v for k, v in courses.items()
                if not search or search in k.lower() or search in v["title"].lower()
                or search in v.get("description", "").lower()}

    # Course card grid
    items = list(filtered.items())
    for i in range(0, len(items), 3):
        cols = st.columns(3)
        for j, (slug, course) in enumerate(items[i:i + 3]):
            with cols[j]:
                icon = COURSE_ICONS.get(slug, "\U0001f4d6")
                stats = course["stats"]
                desc = course["description"][:120] if course["description"] else "Explore course content"
                styled_html(f"""
                <div class="course-card">
                    <div class="course-card-icon">{icon}</div>
                    <div class="course-card-title">{course['title']}</div>
                    <div class="course-card-desc">{desc}</div>
                    <div class="course-card-stats">
                        <span class="card-badge"><strong>{stats['modules']}</strong> modules</span>
                        <span class="card-badge"><strong>{stats['slides']}</strong> slides</span>
                        <span class="card-badge"><strong>{stats['notebooks']}</strong> notebooks</span>
                    </div>
                </div>
                """)
                if st.button(f"Open \u2192", key=f"open_{slug}", use_container_width=True):
                    nav_to("course", course_slug=slug)
                    st.rerun()


def page_course(courses: dict):
    """Course landing page with module card grid."""
    slug = st.session_state.get("course_slug", "")
    course = courses.get(slug)
    if not course:
        st.error("Course not found")
        return

    icon = COURSE_ICONS.get(slug, "\U0001f4d6")

    # Breadcrumb
    render_breadcrumb([
        ("Home", "home", {}),
        (course["title"],),
    ])

    # Course header
    s = course["stats"]
    desc_html = f"<p>{course['description']}</p>" if course["description"] else ""
    styled_html(f"""
    <div class="course-header">
        <div class="course-header-icon">{icon}</div>
        <div class="course-header-info">
            <h1>{course['title']}</h1>
            {desc_html}
            <div class="stat-pills">
                <span class="stat-pill"><span class="stat-value">{s['modules']}</span> Modules</span>
                <span class="stat-pill"><span class="stat-value">{s['slides']}</span> Slides</span>
                <span class="stat-pill"><span class="stat-value">{s['notebooks']}</span> Notebooks</span>
                <span class="stat-pill"><span class="stat-value">{s['guides']}</span> Guides</span>
            </div>
        </div>
    </div>
    """)

    # Top-level sections (quick-starts, templates, etc.)
    for folder, sec in course["sections"].items():
        styled_html(f"""
        <div class="section-divider">
            <span class="section-divider-icon">\U0001f4c1</span>
            <span class="section-divider-label">{sec['label']}</span>
            <span class="section-divider-count">{len(sec['files'])} files</span>
        </div>
        """)

        for f in sec["files"]:
            fi = FILE_ICONS.get(f["type"], "\U0001f4c4")
            display_name = humanize_filename(f["name"])
            badge_html = get_file_type_badge_html(f["type"])

            col1, col2 = st.columns([6, 1])
            with col1:
                if st.button(
                    f"{fi}  {display_name}",
                    key=f"sec_{folder}_{f['path']}",
                    use_container_width=True,
                ):
                    nav_to("viewer", view_path=f["path"], course_slug=slug)
                    st.rerun()
            with col2:
                styled_html(badge_html)

    # Module card grid
    if course["modules"]:
        styled_html(f"""
        <div class="section-divider" style="margin-top: 2rem;">
            <span class="section-divider-icon">\U0001f4da</span>
            <span class="section-divider-label">Modules</span>
            <span class="section-divider-count">{len(course['modules'])} modules</span>
        </div>
        """)

        # Render module cards as clickable Streamlit containers
        cols_per_row = 3
        mods = course["modules"]
        for i in range(0, len(mods), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, mod in enumerate(mods[i:i + cols_per_row]):
                idx = i + j
                with cols[j]:
                    mod_num = get_module_number(mod["dir_name"])
                    mod_title = get_module_clean_title(mod)
                    color = MODULE_COLORS[idx % len(MODULE_COLORS)]

                    n_slides = len(mod["slides"])
                    n_guides = len(mod["content"].get("guides", []))
                    n_notebooks = len(mod["content"].get("notebooks", []))
                    n_exercises = len(mod["content"].get("exercises", []))

                    badges = ""
                    if n_slides:
                        badges += f'\U0001f310 {n_slides} slides  '
                    if n_guides:
                        badges += f'\U0001f4dd {n_guides} guides  '
                    if n_notebooks:
                        badges += f'\U0001f4d3 {n_notebooks} notebooks  '
                    if n_exercises:
                        badges += f'\u270d\ufe0f {n_exercises} exercises'

                    # Use a styled container with a button inside
                    card_html = f"""
                    <div class="module-card {color}" style="pointer-events: none;">
                        <span class="module-card-number">Module {mod_num}</span>
                        <div class="module-card-title">{mod_title}</div>
                        <div class="module-card-badges">{badges.strip()}</div>
                    </div>
                    """
                    styled_html(card_html)
                    if st.button(
                        f"Open Module {mod_num} \u2192",
                        key=f"open_mod_{slug}_{mod['dir_name']}",
                        use_container_width=True,
                    ):
                        nav_to("module", course_slug=slug, module_dir=mod["dir_name"])
                        st.rerun()


def _pair_guides_and_slides(mod: dict) -> list:
    """Pair each guide markdown with its companion slide deck HTML.

    Returns a list of dicts:
        {"guide": guide_file_dict | None,
         "slides": slide_file_dict | None,
         "label": str}
    Ordered by filename prefix (01_, 02_, ...).
    """
    guides = mod["content"].get("guides", [])
    slides = mod["slides"]  # list of {"name": ..., "path": ...}

    # Build lookup: base name (without _slides suffix/extension) -> slide
    slide_lookup = {}
    for s in slides:
        # path like .../guides/01_ga_components_slides.html
        fname = Path(s["path"]).stem  # 01_ga_components_slides
        base = fname.replace("_slides", "")
        slide_lookup[base] = s

    paired = []
    seen_bases = set()

    # Walk guides, find matching slides
    for g in guides:
        fname = Path(g["path"]).stem  # e.g. 01_ga_components
        # Skip any _slides files (source .md or rendered .html)
        if "_slides" in g["name"]:
            continue
        base = fname
        seen_bases.add(base)
        paired.append({
            "guide": g,
            "slides": slide_lookup.get(base),
            "label": humanize_filename(g["name"]),
        })

    # Any slides without a matching guide
    for base, s in slide_lookup.items():
        if base not in seen_bases:
            paired.append({
                "guide": None,
                "slides": s,
                "label": humanize_filename(Path(s["path"]).name),
            })

    return paired


def page_module(courses: dict):
    """Module detail page — guides and slides shown together inline."""
    slug = st.session_state.get("course_slug", "")
    module_dir_name = st.session_state.get("module_dir", "")
    course = courses.get(slug)
    if not course:
        st.error("Course not found")
        return

    # Find the module
    mod = None
    for m in course["modules"]:
        if m["dir_name"] == module_dir_name:
            mod = m
            break
    if not mod:
        st.error("Module not found")
        return

    mod_num = get_module_number(mod["dir_name"])
    mod_title = get_module_clean_title(mod)

    # Breadcrumb
    render_breadcrumb([
        ("Home", "home", {}),
        (course["title"], "course", {"course_slug": slug}),
        (f"Module {mod_num}: {mod_title}",),
    ])

    # Module header
    styled_html(f"""
    <div class="course-header">
        <div class="course-header-icon">\U0001f4e6</div>
        <div class="course-header-info">
            <h1>Module {mod_num}: {mod_title}</h1>
        </div>
    </div>
    """)

    # ---- Paired guide + slides content ----
    pairs = _pair_guides_and_slides(mod)
    notebooks = mod["content"].get("notebooks", [])
    exercises = mod["content"].get("exercises", [])
    resources = mod["content"].get("resources", [])

    if not pairs and not notebooks and not exercises:
        st.info("No content available in this module.")
        return

    # Render each guide/slides pair as an expander section
    if pairs:
        for idx, pair in enumerate(pairs):
            label = pair["label"]
            icon = "\U0001f4d6"  # open book
            with st.expander(f"{icon}  {label}", expanded=(idx == 0)):
                # Slides first (if available)
                if pair["slides"]:
                    st.markdown("#### \U0001f3ac Slide Deck")
                    slide_path = PROJECT_ROOT / pair["slides"]["path"]
                    if slide_path.exists():
                        html_content = slide_path.read_text(
                            encoding="utf-8", errors="replace"
                        )
                        st.components.v1.html(html_content, height=600, scrolling=False)
                        track_page_visit(
                            slug, mod["dir_name"], "slides",
                            slide_path.name,
                        )

                # Guide below slides
                if pair["guide"]:
                    st.markdown("---")
                    st.markdown("#### \U0001f4dd Study Guide")
                    guide_path = PROJECT_ROOT / pair["guide"]["path"]
                    if guide_path.exists():
                        render_markdown_file(guide_path)
                        track_page_visit(
                            slug, mod["dir_name"], "guides",
                            guide_path.name,
                        )

    # ---- Notebooks section ----
    if notebooks:
        st.markdown("---")
        st.markdown("### \U0001f4d3 Notebooks")
        for nb in notebooks:
            nb_name = humanize_filename(nb["name"])
            with st.expander(f"\U0001f4d3  {nb_name}"):
                nb_path = PROJECT_ROOT / nb["path"]
                if nb_path.exists():
                    render_notebook_file(nb_path)
                    track_page_visit(
                        slug, mod["dir_name"], "notebooks",
                        nb_path.name,
                    )

    # ---- Exercises section ----
    if exercises:
        st.markdown("---")
        st.markdown("### \u270d\ufe0f Exercises")
        for ex in exercises:
            ex_name = humanize_filename(ex["name"])
            with st.expander(f"\u270d\ufe0f  {ex_name}"):
                ex_path = PROJECT_ROOT / ex["path"]
                if ex_path.exists():
                    render_file(ex["path"])

    # ---- Resources section ----
    if resources:
        st.markdown("---")
        st.markdown("### \U0001f4da Resources")
        for res in resources:
            res_name = humanize_filename(res["name"])
            with st.expander(f"\U0001f4da  {res_name}"):
                res_path = PROJECT_ROOT / res["path"]
                if res_path.exists():
                    render_file(res["path"])

    # Previous / Next module navigation
    mod_index = next(
        (i for i, m in enumerate(course["modules"]) if m["dir_name"] == module_dir_name),
        None
    )
    if mod_index is not None:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if mod_index > 0:
                prev_mod = course["modules"][mod_index - 1]
                prev_num = get_module_number(prev_mod["dir_name"])
                if st.button(
                    f"\u2190 Module {prev_num}",
                    key="prev_module",
                    use_container_width=True,
                ):
                    nav_to("module", course_slug=slug, module_dir=prev_mod["dir_name"])
                    st.rerun()
        with col3:
            if mod_index < len(course["modules"]) - 1:
                next_mod = course["modules"][mod_index + 1]
                next_num = get_module_number(next_mod["dir_name"])
                if st.button(
                    f"Module {next_num} \u2192",
                    key="next_module",
                    use_container_width=True,
                ):
                    nav_to("module", course_slug=slug, module_dir=next_mod["dir_name"])
                    st.rerun()


def page_viewer(courses: dict):
    """Content viewer page with breadcrumbs and prev/next navigation."""
    slug = st.session_state.get("course_slug", "")
    view_path = st.session_state.get("view_path", "")
    course = courses.get(slug, {})
    course_title = course.get("title", "Course")

    file_name = Path(view_path).name
    display_name = humanize_filename(file_name)
    file_type = get_file_type_from_path(view_path)

    # Track visit
    mod = find_module_for_file(course, view_path)
    if mod:
        mod_title = get_module_clean_title(mod)
        mod_num = get_module_number(mod["dir_name"])

        # Determine content type for tracking
        content_type = "other"
        for sec_key in ["guides", "notebooks", "exercises", "resources"]:
            for f in mod["content"].get(sec_key, []):
                if f["path"] == view_path:
                    content_type = sec_key
                    break
        for sl in mod["slides"]:
            if sl["path"] == view_path:
                content_type = "slides"
                break

        track_page_visit(slug, mod["dir_name"], content_type, file_name)
    else:
        mod_title = None
        mod_num = None

    # Breadcrumb
    if mod:
        render_breadcrumb([
            ("Home", "home", {}),
            (course_title, "course", {"course_slug": slug}),
            (f"Module {mod_num}: {mod_title}", "module",
             {"course_slug": slug, "module_dir": mod["dir_name"]}),
            (display_name,),
        ])
    else:
        render_breadcrumb([
            ("Home", "home", {}),
            (course_title, "course", {"course_slug": slug}),
            (display_name,),
        ])

    # Viewer header
    badge_html = get_file_type_badge_html(file_type)
    styled_html(f"""
    <div class="viewer-header">
        <span class="viewer-title">{display_name}</span>
        {badge_html}
    </div>
    """)

    # Render content
    render_file(view_path)

    # Prev/Next navigation
    prev_path, next_path = find_adjacent_files(courses, slug, view_path)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if prev_path:
            prev_name = humanize_filename(Path(prev_path).name)
            if st.button(f"\u2190 {prev_name}", key="prev_file", use_container_width=True):
                nav_to("viewer", view_path=prev_path, course_slug=slug)
                st.rerun()
    with col3:
        if next_path:
            next_name = humanize_filename(Path(next_path).name)
            if st.button(f"{next_name} \u2192", key="next_file", use_container_width=True):
                nav_to("viewer", view_path=next_path, course_slug=slug)
                st.rerun()


# ---------------------------------------------------------------------------
# App Layout
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Course Browser",
        page_icon="\U0001f393",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    if "visited_pages" not in st.session_state:
        st.session_state.visited_pages = set()

    inject_custom_css()

    courses = scan_all_courses()

    # Sidebar
    render_sidebar(courses)

    # Router
    page = st.session_state.get("page", "home")
    if page == "course":
        page_course(courses)
    elif page == "module":
        page_module(courses)
    elif page == "viewer":
        page_viewer(courses)
    else:
        page_home(courses)


if __name__ == "__main__":
    main()
