"""
Course Content Browser — Streamlit App

Browse all courses, slides, notebooks, and guides.
Deploy free on Streamlit Community Cloud: https://streamlit.io/cloud
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


def extract_readme_info(readme_path: Path) -> tuple:
    if not readme_path.exists():
        return "", ""
    text = readme_path.read_text(encoding="utf-8", errors="replace")
    title, desc = "", ""
    for line in text.split("\n"):
        if line.startswith("# ") and not title:
            title = line[2:].strip()
        elif title and not desc and line.strip() and not line.startswith("#"):
            desc = line.strip()
            break
    return title, desc


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

def render_markdown_file(file_path: Path):
    text = file_path.read_text(encoding="utf-8", errors="replace")
    # Strip Marp frontmatter if present
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            frontmatter = text[3:end]
            if "marp:" in frontmatter:
                text = text[end + 3:].lstrip("\n")
    st.markdown(text, unsafe_allow_html=True)


def render_notebook_file(file_path: Path):
    data = json.loads(file_path.read_text(encoding="utf-8", errors="replace"))
    cells = data.get("cells", [])

    for i, cell in enumerate(cells):
        ctype = cell.get("cell_type", "code")
        source = "".join(cell.get("source", []))

        if not source.strip():
            continue

        if ctype == "markdown":
            st.markdown(source, unsafe_allow_html=True)
        elif ctype == "code":
            st.code(source, language="python")
            # Show outputs
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


def render_python_file(file_path: Path):
    code = file_path.read_text(encoding="utf-8", errors="replace")
    st.code(code, language="python")


def render_html_slides(file_path: Path):
    html = file_path.read_text(encoding="utf-8", errors="replace")
    st.components.v1.html(html, height=620, scrolling=True)


def render_csv_file(file_path: Path):
    import pandas as pd
    df = pd.read_csv(file_path)
    st.dataframe(df, use_container_width=True)


def render_text_file(file_path: Path):
    text = file_path.read_text(encoding="utf-8", errors="replace")
    st.code(text)


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


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_home(courses: dict):
    st.title("Course Browser")
    st.caption("Practical-first courses spanning ML, GenAI, econometrics, and trading systems")

    # Aggregate stats
    total = {"courses": len(courses), "modules": 0, "slides": 0, "notebooks": 0, "guides": 0}
    for c in courses.values():
        for k in ("modules", "slides", "notebooks", "guides"):
            total[k] += c["stats"][k]

    cols = st.columns(5)
    for col, (label, val) in zip(cols, [
        ("Courses", total["courses"]),
        ("Modules", total["modules"]),
        ("Slide Decks", total["slides"]),
        ("Notebooks", total["notebooks"]),
        ("Guides", total["guides"]),
    ]):
        col.metric(label, val)

    st.divider()

    # Course grid
    search = st.session_state.get("search", "").lower()
    filtered = {k: v for k, v in courses.items()
                if not search or search in k.lower() or search in v["title"].lower()
                or search in v.get("description", "").lower()}

    for i in range(0, len(filtered), 3):
        cols = st.columns(3)
        for col, (slug, course) in zip(cols, list(filtered.items())[i:i+3]):
            with col:
                icon = COURSE_ICONS.get(slug, "\U0001f4d6")
                stats = course["stats"]
                st.markdown(f"### {icon} {course['title']}")
                if course["description"]:
                    st.caption(course["description"][:120])
                st.markdown(
                    f"**{stats['modules']}** modules · "
                    f"**{stats['slides']}** slides · "
                    f"**{stats['notebooks']}** notebooks"
                )
                if st.button("Open", key=f"open_{slug}"):
                    nav_to("course", course_slug=slug)
                    st.rerun()


def page_course(courses: dict):
    slug = st.session_state.get("course_slug", "")
    course = courses.get(slug)
    if not course:
        st.error("Course not found")
        return

    if st.button("\u2190 All Courses"):
        nav_to("home")
        st.rerun()

    icon = COURSE_ICONS.get(slug, "\U0001f4d6")
    st.title(f"{icon} {course['title']}")
    if course["description"]:
        st.markdown(course["description"])

    # Stats row
    s = course["stats"]
    cols = st.columns(4)
    cols[0].metric("Modules", s["modules"])
    cols[1].metric("Slide Decks", s["slides"])
    cols[2].metric("Notebooks", s["notebooks"])
    cols[3].metric("Guides", s["guides"])

    st.divider()

    # Top-level sections (quick-starts, templates, etc.)
    for folder, sec in course["sections"].items():
        with st.expander(f"\U0001f4c1 {sec['label']} ({len(sec['files'])} files)"):
            for f in sec["files"]:
                fi = FILE_ICONS.get(f["type"], "\U0001f4c4")
                if st.button(f"{fi} {f['name']}", key=f"sec_{folder}_{f['path']}"):
                    nav_to("viewer", view_path=f["path"], course_slug=slug)
                    st.rerun()

    # Modules
    if course["modules"]:
        st.subheader("Modules")
        for mod in course["modules"]:
            with st.expander(f"\U0001f4d6 {mod['name']}"):
                # Slides
                if mod["slides"]:
                    st.markdown("**Slide Decks**")
                    for sl in mod["slides"]:
                        if st.button(f"\U0001f3ac {sl['name']}", key=f"sl_{sl['path']}"):
                            nav_to("viewer", view_path=sl["path"], course_slug=slug)
                            st.rerun()

                # Content sections
                section_labels = {
                    "guides": ("\U0001f4dd", "Guides"),
                    "notebooks": ("\U0001f4d3", "Notebooks"),
                    "exercises": ("\u270d\ufe0f", "Exercises"),
                    "resources": ("\U0001f4ce", "Resources"),
                }
                for sec_key, (sec_icon, sec_label) in section_labels.items():
                    files = mod["content"].get(sec_key, [])
                    if files:
                        st.markdown(f"**{sec_label}**")
                        for f in files:
                            fi = FILE_ICONS.get(f["type"], "\U0001f4c4")
                            if st.button(f"{fi} {f['name']}", key=f"mod_{sec_key}_{f['path']}"):
                                nav_to("viewer", view_path=f["path"], course_slug=slug)
                                st.rerun()


def page_viewer(courses: dict):
    slug = st.session_state.get("course_slug", "")
    view_path = st.session_state.get("view_path", "")

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("\u2190 Back"):
            nav_to("course", course_slug=slug)
            st.rerun()
    with col2:
        file_name = Path(view_path).name
        st.markdown(f"**{file_name}**")

    st.divider()
    render_file(view_path)


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

    courses = scan_all_courses()

    # Sidebar
    with st.sidebar:
        st.title("\U0001f393 Courses")
        search = st.text_input("Search courses", key="search", placeholder="Filter...")

        st.divider()
        for slug, course in courses.items():
            if search and search.lower() not in slug and search.lower() not in course["title"].lower():
                continue
            icon = COURSE_ICONS.get(slug, "\U0001f4d6")
            if st.button(f"{icon} {course['title']}", key=f"nav_{slug}", use_container_width=True):
                nav_to("course", course_slug=slug)
                st.rerun()

        st.divider()
        with st.expander("About"):
            st.markdown(
                "**Practical-first courses** producing professional-grade "
                "educational materials across ML, GenAI, econometrics, and "
                "trading systems.\n\n"
                "Content includes Marp slide decks, Jupyter notebooks, "
                "Python templates, and markdown guides."
            )

    # Router
    page = st.session_state.get("page", "home")
    if page == "course":
        page_course(courses)
    elif page == "viewer":
        page_viewer(courses)
    else:
        page_home(courses)


if __name__ == "__main__":
    main()
