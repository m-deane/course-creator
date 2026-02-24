#!/usr/bin/env python3
"""
Course Content Browser — browse all courses, slides, notebooks, and guides at http://127.0.0.1:8000

Usage:
    python scripts/course_browser.py              # default port 8000
    python scripts/course_browser.py --port 9000   # custom port
"""

import argparse
import hashlib
import json
import re
import urllib.parse
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer, get_lexer_by_name, TextLexer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
COURSES_DIR = PROJECT_ROOT / "courses"

ACRONYMS = {
    "llm": "LLM", "llms": "LLMs", "rag": "RAG", "eia": "EIA",
    "hmm": "HMM", "hmms": "HMMs", "ai": "AI", "genai": "GenAI",
    "ab": "A/B", "mab": "MAB", "api": "API", "ml": "ML",
    "nlp": "NLP", "gpt": "GPT", "lda": "LDA",
}

CONTENT_SECTIONS = [
    ("quick-starts", "Quick Starts"),
    ("templates", "Templates"),
    ("recipes", "Recipes"),
    ("concepts", "Concepts"),
    ("projects", "Projects"),
    ("capstone", "Capstone"),
]

# Iteration 5: Course icons and accent colors
COURSE_ICONS = {
    "agentic-ai-llms": "🤖",
    "agentic-ai-practical": "🤖",
    "ai-engineer-fundamentals": "⚙️",
    "bayesian-commodity-forecasting": "📊",
    "dataiku-genai": "🧩",
    "dynamic-factor-models": "📐",
    "genai-commodities": "💬",
    "genetic-algorithms-feature-selection": "🧬",
    "hidden-markov-models": "🔗",
    "multi-armed-bandits-ab-testing": "🎰",
    "panel-regression": "📈",
}

ACCENT_COLORS = [
    "#3182ce",  # blue
    "#dd6b20",  # orange
    "#38a169",  # green
    "#805ad5",  # purple
    "#d53f8c",  # pink
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def humanize(name: str) -> str:
    """Convert directory name to a human-readable title."""
    parts = name.replace("-", " ").replace("_", " ").split()
    result = []
    for p in parts:
        low = p.lower()
        if low in ACRONYMS:
            result.append(ACRONYMS[low])
        else:
            result.append(p.capitalize())
    return " ".join(result)


def extract_readme_info(readme_path: Path) -> tuple[str, str]:
    """Extract title (H1) and first paragraph from README.md."""
    if not readme_path.exists():
        return "", ""
    text = readme_path.read_text(encoding="utf-8", errors="replace")
    title = ""
    desc = ""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("# ") and not title:
            title = line[2:].strip()
        elif title and not desc and line.strip() and not line.startswith("#"):
            desc = line.strip()
            break
    return title, desc


def safe_path(requested: str) -> Path | None:
    """Resolve a path and ensure it stays within PROJECT_ROOT."""
    try:
        resolved = (PROJECT_ROOT / requested).resolve()
        if str(resolved).startswith(str(PROJECT_ROOT)):
            return resolved
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Content Scanner
# ---------------------------------------------------------------------------

class ContentScanner:
    def __init__(self):
        self._cache: dict | None = None

    def scan_all(self, force: bool = False) -> dict:
        if self._cache and not force:
            return self._cache
        courses = {}
        if not COURSES_DIR.exists():
            self._cache = courses
            return courses
        for d in sorted(COURSES_DIR.iterdir()):
            if d.is_dir() and not d.name.startswith("."):
                courses[d.name] = self._scan_course(d)
        self._cache = courses
        return courses

    def _scan_course(self, course_dir: Path) -> dict:
        readme = course_dir / "README.md"
        title, desc = extract_readme_info(readme)
        if not title:
            title = humanize(course_dir.name)

        modules = []
        mod_dir = course_dir / "modules"
        if mod_dir.exists():
            for m in sorted(mod_dir.iterdir()):
                if m.is_dir() and m.name.startswith("module_"):
                    modules.append(self._scan_module(m))

        sections = {}
        for folder, label in CONTENT_SECTIONS:
            sec_dir = course_dir / folder
            if sec_dir.exists() and sec_dir.is_dir():
                files = self._list_content_files(sec_dir)
                if files:
                    sections[folder] = {"label": label, "files": files}

        stats = self._count_stats(course_dir)

        return {
            "title": title,
            "description": desc,
            "dir": str(course_dir.relative_to(PROJECT_ROOT)),
            "has_readme": readme.exists(),
            "modules": modules,
            "sections": sections,
            "stats": stats,
        }

    def _scan_module(self, mod_dir: Path) -> dict:
        readme = mod_dir / "README.md"
        title, _ = extract_readme_info(readme)
        if not title:
            # Parse module_03_sentiment -> "03 - Sentiment"
            parts = mod_dir.name.split("_", 2)
            num = parts[1] if len(parts) > 1 else ""
            name = humanize(parts[2]) if len(parts) > 2 else humanize(mod_dir.name)
            title = f"{num} - {name}" if num else name

        content = {}
        for sub in ["guides", "notebooks", "exercises", "resources"]:
            sub_dir = mod_dir / sub
            if sub_dir.exists():
                files = self._list_content_files(sub_dir)
                if files:
                    content[sub] = files

        # Slides (HTML files in guides/)
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
            "has_readme": readme.exists(),
            "content": content,
            "slides": slides,
        }

    def _list_content_files(self, directory: Path) -> list[dict]:
        files = []
        for f in sorted(directory.rglob("*")):
            if f.is_file() and not f.name.startswith("."):
                ext = f.suffix.lower()
                if ext in (".md", ".ipynb", ".py", ".html", ".txt", ".yaml", ".yml", ".json", ".csv"):
                    ftype = {
                        ".md": "markdown", ".ipynb": "notebook", ".py": "python",
                        ".html": "html", ".txt": "text", ".yaml": "yaml",
                        ".yml": "yaml", ".json": "json", ".csv": "csv",
                    }.get(ext, "other")
                    files.append({
                        "name": f.name,
                        "path": str(f.relative_to(PROJECT_ROOT)),
                        "type": ftype,
                        "size": f.stat().st_size,
                    })
        return files

    def _count_stats(self, course_dir: Path) -> dict:
        slides = len(list(course_dir.rglob("*_slides.html")))
        notebooks = len(list(course_dir.rglob("*.ipynb")))
        guides = len([f for f in course_dir.rglob("*.md")
                       if f.name != "README.md" and "_slides" not in f.name])
        modules = 0
        mod_dir = course_dir / "modules"
        if mod_dir.exists():
            modules = len([d for d in mod_dir.iterdir()
                          if d.is_dir() and d.name.startswith("module_")])
        return {
            "modules": modules,
            "slides": slides,
            "notebooks": notebooks,
            "guides": guides,
        }


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

LANG_LABELS = {
    "python": "Python", "py": "Python", "javascript": "JavaScript",
    "js": "JavaScript", "bash": "Bash", "sh": "Shell", "shell": "Shell",
    "json": "JSON", "yaml": "YAML", "yml": "YAML", "sql": "SQL",
    "html": "HTML", "css": "CSS", "text": "Text", "plaintext": "Text",
    "r": "R", "rust": "Rust", "go": "Go", "java": "Java",
    "cpp": "C++", "c": "C", "typescript": "TypeScript", "ts": "TypeScript",
    "markdown": "Markdown", "md": "Markdown", "xml": "XML",
    "dockerfile": "Dockerfile", "makefile": "Makefile",
    "toml": "TOML", "ini": "INI", "env": "ENV",
}


def _extract_fenced_languages(text: str) -> list[str]:
    """
    Scan markdown source to extract language identifiers from fenced code blocks
    in the order they appear. Returns a list of language strings (may be empty string).
    """
    langs = []
    for m in re.finditer(r'^```(\w*)', text, re.MULTILINE):
        langs.append(m.group(1).lower())
    return langs


def _add_code_headers(html: str, lang_queue: list[str]) -> str:
    """
    Iteration 2: Post-process highlighted HTML to inject a language header bar
    before each <div class="highlight"> block.
    Uses a pre-extracted queue of language identifiers from the markdown source.
    """
    idx = [0]  # mutable counter for closure

    def inject_header(m):
        lang = ""
        if idx[0] < len(lang_queue):
            lang = lang_queue[idx[0]]
        idx[0] += 1

        label = LANG_LABELS.get(lang, lang.upper() if lang else "Code")
        header = f'<div class="code-header"><span class="code-lang">{label}</span></div>'
        return header + m.group(0)

    # Match each highlight div
    html = re.sub(
        r'<div class="highlight">',
        inject_header,
        html
    )
    return html


def _transform_callouts(html: str) -> str:
    """
    Iteration 3: Transform <blockquote> elements into styled callout boxes.
    Detects emoji/keyword prefixes to assign variant classes.
    """
    def replace_blockquote(m):
        inner = m.group(1).strip()

        # Determine callout type from content prefix
        callout_type = "info"
        icon = "ℹ️"

        inner_text = re.sub(r'<[^>]+>', '', inner).strip()

        if re.match(r'^(⚠️|\*\*Warning\*\*)', inner_text) or re.match(r'^<[^>]*>(⚠️|\*\*Warning\*\*)', inner):
            callout_type = "warning"
            icon = "⚠️"
        elif re.match(r'^(💡|\*\*Tip\*\*)', inner_text) or '💡' in inner_text[:30]:
            callout_type = "tip"
            icon = "💡"
        elif re.match(r'^(📝|\*\*Note\*\*)', inner_text) or '📝' in inner_text[:30]:
            callout_type = "note"
            icon = "📝"
        elif re.match(r'^(🚨|\*\*Important\*\*|\*\*Critical\*\*)', inner_text):
            callout_type = "danger"
            icon = "🚨"

        return f'<div class="callout callout-{callout_type}"><div class="callout-icon">{icon}</div><div class="callout-body">{inner}</div></div>'

    html = re.sub(r'<blockquote>\s*(.*?)\s*</blockquote>', replace_blockquote, html, flags=re.DOTALL)
    return html


def _extract_toc(html: str) -> list[dict]:
    """
    Iteration 4: Extract headings from rendered HTML for TOC generation.
    Returns list of {level, text, id} dicts.
    """
    headings = []
    for m in re.finditer(r'<h([23])([^>]*)>(.*?)</h\1>', html, re.DOTALL):
        level = int(m.group(1))
        attrs = m.group(2)
        text = re.sub(r'<[^>]+>', '', m.group(3)).strip()

        # Extract existing id or generate one
        id_match = re.search(r'id="([^"]+)"', attrs)
        if id_match:
            hid = id_match.group(1)
        else:
            # Generate id from text
            hid = re.sub(r'[^a-z0-9-]', '-', text.lower())
            hid = re.sub(r'-+', '-', hid).strip('-')

        headings.append({"level": level, "text": text, "id": hid})
    return headings


def _inject_heading_ids(html: str) -> str:
    """Ensure all h2/h3 headings have id attributes for TOC anchor links."""
    counter = {}

    def add_id(m):
        tag = m.group(1)
        attrs = m.group(2)
        content = m.group(3)

        if 'id=' in attrs:
            return m.group(0)

        text = re.sub(r'<[^>]+>', '', content).strip()
        hid = re.sub(r'[^a-z0-9-]', '-', text.lower())
        hid = re.sub(r'-+', '-', hid).strip('-') or 'heading'

        # Handle duplicates
        if hid in counter:
            counter[hid] += 1
            hid = f"{hid}-{counter[hid]}"
        else:
            counter[hid] = 0

        return f'<{tag}{attrs} id="{hid}">{content}</{tag}>'

    html = re.sub(r'<(h[23])([^>]*)>(.*?)</h[23]>', add_id, html, flags=re.DOTALL)
    return html


def render_markdown(text: str, add_toc: bool = False) -> tuple[str, list[dict]]:
    """Render markdown to HTML with syntax highlighting.
    Returns (html, headings) where headings is a list for TOC generation.
    """
    # Iteration 2: Pre-extract language identifiers from fenced code blocks
    lang_queue = _extract_fenced_languages(text)

    md = markdown.Markdown(extensions=[
        "fenced_code", "tables", "codehilite", "toc", "nl2br",
    ], extension_configs={
        "codehilite": {"css_class": "highlight", "guess_lang": True},
    })
    html = md.convert(text)

    # Iteration 2: Add code header bars with language labels (using pre-extracted queue)
    html = _add_code_headers(html, lang_queue)

    # Iteration 3: Transform blockquotes to callout boxes
    html = _transform_callouts(html)

    # Iteration 4: Inject heading ids for TOC
    html = _inject_heading_ids(html)

    headings = _extract_toc(html) if add_toc else []
    return html, headings


def render_notebook(nb_path: Path) -> str:
    """Render a Jupyter notebook to HTML without nbconvert."""
    data = json.loads(nb_path.read_text(encoding="utf-8", errors="replace"))
    cells = data.get("cells", [])
    kernel = data.get("metadata", {}).get("kernelspec", {}).get("language", "python")

    parts = []
    cell_index = 0
    for cell in cells:
        ctype = cell.get("cell_type", "code")
        source = "".join(cell.get("source", []))

        if ctype == "markdown":
            html, _ = render_markdown(source)
            parts.append(f'<div class="nb-cell nb-md">{html}</div>')

        elif ctype == "code":
            cell_index += 1
            try:
                lexer = get_lexer_by_name(kernel)
            except Exception:
                lexer = PythonLexer()
            formatter = HtmlFormatter(cssclass="highlight", nowrap=False)
            highlighted = highlight(source, lexer, formatter)
            # Iteration 2: Add "In [n]:" label for notebook cells
            label = f'<div class="nb-in-label">In [{cell_index}]:</div>'
            parts.append(f'<div class="nb-cell nb-code">{label}<div class="nb-code-inner">{highlighted}</div></div>')

            # Outputs
            out_index = 0
            for out in cell.get("outputs", []):
                out_type = out.get("output_type", "")
                if out_type == "stream":
                    text = "".join(out.get("text", []))
                    out_label = f'<div class="nb-out-label">Out [{cell_index}]:</div>'
                    parts.append(f'<div class="nb-cell nb-output">{out_label}<pre>{_escape(text)}</pre></div>')
                elif out_type in ("execute_result", "display_data"):
                    od = out.get("data", {})
                    out_label = f'<div class="nb-out-label">Out [{cell_index}]:</div>'
                    if "text/html" in od:
                        html = "".join(od["text/html"])
                        parts.append(f'<div class="nb-cell nb-output">{out_label}{html}</div>')
                    elif "image/png" in od:
                        img = od["image/png"]
                        parts.append(f'<div class="nb-cell nb-output">{out_label}<img src="data:image/png;base64,{img}"></div>')
                    elif "text/plain" in od:
                        text = "".join(od["text/plain"])
                        parts.append(f'<div class="nb-cell nb-output">{out_label}<pre>{_escape(text)}</pre></div>')
                elif out_type == "error":
                    tb = "\n".join(out.get("traceback", []))
                    # Strip ANSI codes
                    tb = re.sub(r'\x1b\[[0-9;]*m', '', tb)
                    parts.append(f'<div class="nb-cell nb-error"><pre>{_escape(tb)}</pre></div>')

    return "\n".join(parts)


def render_python(py_path: Path) -> str:
    """Render Python file with syntax highlighting."""
    code = py_path.read_text(encoding="utf-8", errors="replace")
    formatter = HtmlFormatter(cssclass="highlight", linenos=True, nowrap=False)
    return highlight(code, PythonLexer(), formatter)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# HTML Templates
# ---------------------------------------------------------------------------

PYGMENTS_CSS = HtmlFormatter(style="monokai").get_style_defs(".highlight")

BASE_CSS = """
:root {
    --navy: #1a365d;
    --navy-light: #2a4a7f;
    --orange: #dd6b20;
    --orange-light: #ed8936;
    --bg: #f0f4f8;
    --card-bg: #ffffff;
    --text: #1a202c;
    --text-light: #4a5568;
    --border: #e2e8f0;
    --code-bg: #1e1e2e;
    --accent-blue: #3182ce;
    --accent-green: #38a169;
    --accent-purple: #805ad5;
    --accent-teal: #319795;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.75;
    font-size: 16px;
}
a { color: var(--navy); text-decoration: none; }
a:hover { color: var(--orange); text-decoration: underline; }
.container { max-width: 1280px; margin: 0 auto; padding: 0 32px; }

/* ---- HEADER ---- */
header {
    background: linear-gradient(135deg, var(--navy) 0%, #1e4080 50%, var(--navy-light) 100%);
    color: white; padding: 24px 0; margin-bottom: 36px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.18);
    position: relative; overflow: hidden;
}
header::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, var(--orange), var(--orange-light), var(--orange));
}
header h1 { font-size: 1.65rem; font-weight: 700; letter-spacing: -0.02em; }
header a { color: white; }
header .subtitle { color: rgba(255,255,255,0.75); font-size: 0.92rem; margin-top: 4px; font-weight: 400; }

/* ---- BREADCRUMB ---- */
.breadcrumb {
    padding: 10px 0; font-size: 0.85rem; color: var(--text-light);
    border-bottom: 1px solid var(--border); margin-bottom: 28px;
    display: flex; align-items: center; gap: 4px;
}
.breadcrumb a { color: var(--navy); font-weight: 500; }
.breadcrumb a:hover { color: var(--orange); }
.breadcrumb span { color: #cbd5e0; margin: 0 2px; }

/* ---- SEARCH BAR (Iteration 5) ---- */
.search-bar {
    margin-bottom: 28px;
    position: relative;
}
.search-bar input {
    width: 100%; max-width: 480px;
    padding: 10px 16px 10px 40px;
    border: 2px solid var(--border);
    border-radius: 24px;
    font-size: 0.95rem;
    color: var(--text);
    background: var(--card-bg);
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.search-bar input:focus {
    border-color: var(--navy);
    box-shadow: 0 0 0 3px rgba(26,54,93,0.12);
}
.search-bar .search-icon {
    position: absolute; left: 13px; top: 50%; transform: translateY(-50%);
    color: var(--text-light); font-size: 1rem; pointer-events: none;
}
.no-results {
    display: none; text-align: center; padding: 48px 0;
    color: var(--text-light); font-size: 1.1rem;
}

/* ---- CARDS (Iteration 5 polish) ---- */
.card-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 24px; }
.card {
    background: var(--card-bg); border-radius: 12px; padding: 0;
    border: 1px solid var(--border);
    transition: box-shadow 0.25s, transform 0.25s;
    overflow: hidden; display: flex; flex-direction: column;
    position: relative;
}
.card:hover {
    box-shadow: 0 8px 32px rgba(0,0,0,0.13);
    transform: translateY(-4px);
}
.card-accent { height: 4px; width: 100%; flex-shrink: 0; }
.card-body { padding: 24px; flex: 1; display: flex; flex-direction: column; }
.card-icon { font-size: 2rem; margin-bottom: 10px; line-height: 1; }
.card h3 {
    color: var(--navy); margin-bottom: 8px;
    font-size: 1.15rem; font-weight: 700; line-height: 1.3;
}
.card p { color: var(--text-light); font-size: 0.9rem; margin-bottom: 16px; line-height: 1.6; flex: 1; }
.card .stats { display: flex; gap: 8px; flex-wrap: wrap; margin-top: auto; }

/* Pill-shaped stat badges (Iteration 5) */
.stat {
    padding: 3px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
    display: inline-flex; align-items: center; gap: 4px;
}
.stat-slides { background: #c6f6d5; color: #22543d; }
.stat-notebooks { background: #bee3f8; color: #2a4365; }
.stat-guides { background: #fefcbf; color: #7b6b00; }
.stat-modules { background: #e9d8fd; color: #44337a; }

/* ---- TABLES ---- */
table { width: 100%; border-collapse: collapse; margin: 20px 0; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
th { background: var(--navy); color: white; padding: 12px 16px; text-align: left; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.03em; }
td { padding: 12px 16px; border-bottom: 1px solid var(--border); font-size: 0.9rem; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #edf2f7; }

/* ---- SECTIONS ---- */
.section { margin-bottom: 40px; }
.section h2 {
    color: var(--navy); font-size: 1.3rem; font-weight: 700; margin-bottom: 20px;
    padding-bottom: 10px; border-bottom: 2px solid var(--orange);
    letter-spacing: -0.01em;
}
.section h3 { color: var(--navy); font-size: 1.05rem; font-weight: 600; margin: 20px 0 10px; }

/* ---- MODULE NAV PILLS (Iteration 4) ---- */
.module-nav {
    display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 28px;
    padding: 16px; background: var(--card-bg); border-radius: 10px;
    border: 1px solid var(--border);
}
.module-nav-label { font-size: 0.8rem; font-weight: 700; color: var(--text-light); text-transform: uppercase; letter-spacing: 0.06em; width: 100%; margin-bottom: 4px; }
.module-pill {
    padding: 5px 14px; border-radius: 20px; font-size: 0.82rem; font-weight: 500;
    border: 1px solid var(--border); color: var(--text-light);
    background: var(--bg); transition: all 0.15s;
    text-decoration: none !important;
}
.module-pill:hover { background: var(--navy); color: white; border-color: var(--navy); text-decoration: none !important; }
.module-pill.active { background: var(--navy); color: white; border-color: var(--navy); font-weight: 700; }

/* ---- VIEWER (Iterations 1-4) ---- */
.viewer-wrapper { display: flex; gap: 32px; align-items: flex-start; }
.viewer {
    background: var(--card-bg); border-radius: 12px; padding: 44px 48px;
    border: 1px solid var(--border);
    min-width: 0; flex: 1;
    box-shadow: 0 2px 12px rgba(0,0,0,0.05);
}
.viewer img { max-width: 100%; border-radius: 6px; }

/* Typography — Iteration 1 */
.viewer h1 {
    color: var(--navy);
    font-size: 2.1rem; font-weight: 800; line-height: 1.2;
    margin-top: 0; margin-bottom: 20px;
    letter-spacing: -0.03em;
    padding-bottom: 16px;
    border-bottom: 3px solid var(--orange);
}
.viewer h2 {
    color: var(--navy);
    font-size: 1.55rem; font-weight: 700;
    margin-top: 48px; margin-bottom: 16px;
    padding-left: 14px;
    border-left: 4px solid var(--orange);
    letter-spacing: -0.02em;
    line-height: 1.3;
}
.viewer h3 {
    color: #2d4a7a;
    font-size: 1.2rem; font-weight: 700;
    margin-top: 36px; margin-bottom: 12px;
    letter-spacing: -0.01em;
}
.viewer h4 {
    color: var(--text); font-size: 1.05rem; font-weight: 700;
    margin-top: 24px; margin-bottom: 10px;
}
.viewer p {
    margin-bottom: 16px; line-height: 1.8;
    color: #2d3748;
}
.viewer ul, .viewer ol { margin: 0 0 16px 24px; }
.viewer li { margin-bottom: 6px; line-height: 1.75; }
.viewer table { font-size: 0.875rem; }
.viewer hr { border: none; border-top: 2px solid var(--border); margin: 32px 0; }

/* Inline code — Iteration 1: orange tint */
.viewer p code, .viewer li code, .viewer td code {
    background: #fff3e0;
    color: #b7410e;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.84em;
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', Menlo, monospace;
    border: 1px solid #fbd38d;
    font-weight: 500;
}

/* ---- CODE BLOCKS — Iteration 2 ---- */
.viewer pre, .viewer .highlight {
    border-radius: 0 0 8px 8px !important;
    overflow-x: auto;
}
.viewer .highlight {
    background: var(--code-bg);
    max-height: 420px;
    overflow-y: auto;
    margin: 0 !important;
    border-radius: 0 0 8px 8px !important;
}
.viewer .highlight pre {
    background: transparent;
    color: #cdd6f4;
    padding: 18px 20px;
    border-radius: 0;
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', Menlo, monospace;
    font-size: 0.84rem;
    line-height: 1.6;
    overflow-x: auto;
    overflow-y: visible;
    max-height: none;
}
/* Code header bar — Iteration 2 */
.code-header {
    display: flex; align-items: center; justify-content: flex-start;
    background: #13131f; /* slightly darker than code-bg */
    border-radius: 8px 8px 0 0;
    padding: 8px 16px 7px;
    border-bottom: 2px solid var(--orange);
    margin-top: 20px;
}
.code-lang {
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', Menlo, monospace;
    font-size: 0.72rem; font-weight: 700;
    color: var(--orange-light);
    text-transform: uppercase; letter-spacing: 0.08em;
    background: rgba(221,107,32,0.15);
    padding: 2px 10px; border-radius: 10px;
}
/* Ensure highlight div follows directly after code-header */
.code-header + .highlight,
.code-header + div > .highlight {
    border-radius: 0 0 8px 8px !important;
    margin-top: 0 !important;
}

/* ---- CALLOUT BOXES — Iteration 3 ---- */
.callout {
    display: flex; gap: 14px; align-items: flex-start;
    border-radius: 10px; padding: 16px 20px; margin: 24px 0;
    border-left: 5px solid;
}
.callout-icon { font-size: 1.35rem; flex-shrink: 0; margin-top: 2px; line-height: 1; }
.callout-body { flex: 1; }
.callout-body p:last-child { margin-bottom: 0; }
.callout-body p { margin-bottom: 8px; }

.callout-info {
    background: #ebf8ff; border-left-color: #3182ce; color: #1a365d;
}
.callout-warning {
    background: #fffbeb; border-left-color: #d69e2e; color: #744210;
}
.callout-tip {
    background: #f0fff4; border-left-color: #38a169; color: #1c4532;
}
.callout-note {
    background: #e8f4fd; border-left-color: #3182ce; color: #1a365d;
}
.callout-danger {
    background: #fff5f5; border-left-color: #e53e3e; color: #742a2a;
}

/* ---- TOC SIDEBAR — Iteration 4 ---- */
.toc-sidebar {
    width: 240px; flex-shrink: 0;
    position: sticky; top: 24px;
    max-height: calc(100vh - 80px); overflow-y: auto;
    background: var(--card-bg); border-radius: 10px;
    border: 1px solid var(--border); padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.toc-sidebar h4 {
    font-size: 0.72rem; font-weight: 800; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-light);
    margin-bottom: 12px; padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}
.toc-sidebar ul { list-style: none; margin: 0; padding: 0; }
.toc-sidebar li { margin: 0; }
.toc-sidebar a {
    display: block; padding: 4px 8px; border-radius: 5px;
    font-size: 0.82rem; color: var(--text-light);
    line-height: 1.4; transition: all 0.15s;
    border-left: 2px solid transparent;
}
.toc-sidebar a:hover { color: var(--navy); background: #edf2f7; text-decoration: none; }
.toc-sidebar a.toc-active {
    color: var(--navy); font-weight: 600;
    border-left-color: var(--orange);
    background: #fffaf0;
}
.toc-sidebar li.toc-h3 a { padding-left: 20px; font-size: 0.78rem; }

/* ---- NOTEBOOK CELLS — Iterations 1 & 2 ---- */
.nb-cell { margin-bottom: 20px; border-radius: 10px; overflow: hidden; }

/* Markdown cells */
.nb-md {
    padding: 20px 24px;
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
}
.nb-md h1 { font-size: 1.8rem; font-weight: 800; color: var(--navy); margin-top: 0; margin-bottom: 16px; border-bottom: 3px solid var(--orange); padding-bottom: 12px; }
.nb-md h2 { font-size: 1.35rem; font-weight: 700; color: var(--navy); margin-top: 24px; margin-bottom: 10px; border-left: 4px solid var(--orange); padding-left: 12px; }
.nb-md h3 { font-size: 1.1rem; font-weight: 700; color: #2d4a7a; margin-top: 18px; margin-bottom: 8px; }
.nb-md p { margin-bottom: 12px; line-height: 1.75; }
.nb-md code { background: #fff3e0; color: #b7410e; padding: 1px 5px; border-radius: 3px; font-size: 0.85em; border: 1px solid #fbd38d; }

/* Code cells — Iteration 2 */
.nb-code {
    border: 1px solid #2d2d3f; border-radius: 10px; overflow: hidden;
    display: flex; flex-direction: column;
}
.nb-in-label {
    background: #13131f; color: #6272a4;
    font-family: 'JetBrains Mono', 'Fira Code', Menlo, monospace;
    font-size: 0.72rem; font-weight: 700;
    padding: 6px 16px 5px;
    border-bottom: 1px solid #2d2d3f;
    letter-spacing: 0.03em;
}
.nb-code-inner { position: relative; }
.nb-code .highlight { margin: 0 !important; border-radius: 0 !important; max-height: 420px; overflow-y: auto; }
.nb-code .highlight pre { padding: 14px 18px; background: var(--code-bg); color: #cdd6f4; font-size: 0.83rem; line-height: 1.6; }

/* Output cells */
.nb-output {
    background: #f8fafb; border: 1px solid var(--border);
    border-top: none; border-radius: 0 0 10px 10px;
    overflow: hidden;
}
.nb-out-label {
    background: #edf2f7; color: #718096;
    font-family: 'JetBrains Mono', 'Fira Code', Menlo, monospace;
    font-size: 0.72rem; font-weight: 700;
    padding: 5px 16px;
    border-bottom: 1px solid var(--border);
    letter-spacing: 0.03em;
}
.nb-output > pre { padding: 14px 18px; background: transparent; color: var(--text); font-size: 0.85rem; line-height: 1.6; }
.nb-output img { display: block; max-width: 100%; padding: 16px; }
.nb-error {
    background: #fff5f5; border: 1px solid #fed7d7;
    border-top: none; padding: 14px 18px;
    border-radius: 0 0 10px 10px;
}
.nb-error pre { background: transparent; color: #c53030; font-size: 0.83rem; padding: 0; }

/* ---- FILE LIST ---- */
.file-list { list-style: none; }
.file-list li {
    padding: 9px 12px; border-radius: 7px;
    transition: background 0.12s; display: flex; align-items: center; gap: 8px;
}
.file-list li:hover { background: #edf2f7; }
.file-list .badge {
    display: inline-block; padding: 2px 9px; border-radius: 20px;
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; flex-shrink: 0;
}
.badge-slides { background: #c6f6d5; color: #22543d; }
.badge-notebook { background: #bee3f8; color: #2a4365; }
.badge-markdown { background: #fefcbf; color: #7b6b00; }
.badge-python { background: #e9d8fd; color: #44337a; }
.badge-html { background: #fed7d7; color: #742a2a; }

/* ---- SLIDE LINKS ---- */
.slide-link {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--navy); color: white; padding: 8px 18px;
    border-radius: 8px; font-size: 0.85rem; margin: 4px 6px 4px 0;
    transition: background 0.2s, transform 0.15s;
    font-weight: 500;
}
.slide-link:hover { background: var(--orange); color: white; text-decoration: none; transform: translateY(-1px); }

.refresh-link {
    float: right; font-size: 0.8rem; color: rgba(255,255,255,0.65);
    border: 1px solid rgba(255,255,255,0.25); padding: 3px 10px;
    border-radius: 12px; transition: all 0.15s;
}
.refresh-link:hover { color: white; background: rgba(255,255,255,0.1); text-decoration: none; }

/* ---- FOOTER ---- */
footer {
    text-align: center; padding: 32px;
    color: var(--text-light); font-size: 0.82rem;
    margin-top: 56px;
    border-top: 1px solid var(--border);
    background: var(--card-bg);
}
footer strong { color: var(--navy); }
.footer-stats { display: flex; justify-content: center; gap: 32px; margin-top: 10px; }
.footer-stat { display: flex; flex-direction: column; align-items: center; }
.footer-stat .number { font-size: 1.4rem; font-weight: 800; color: var(--navy); line-height: 1; }
.footer-stat .label { font-size: 0.75rem; color: var(--text-light); margin-top: 2px; }

/* ---- INDEX HEADER ---- */
.index-header {
    background: linear-gradient(135deg, var(--navy) 0%, #1e4080 60%, var(--navy-light) 100%);
    border-radius: 14px; padding: 40px 48px; margin-bottom: 32px;
    color: white; position: relative; overflow: hidden;
}
.index-header::before {
    content: ''; position: absolute; top: -40px; right: -40px;
    width: 200px; height: 200px; border-radius: 50%;
    background: rgba(255,255,255,0.04);
}
.index-header::after {
    content: ''; position: absolute; bottom: -60px; left: 20%;
    width: 300px; height: 300px; border-radius: 50%;
    background: rgba(255,255,255,0.03);
}
.index-header h2 { font-size: 1.8rem; font-weight: 800; color: white; margin-bottom: 8px; position: relative; z-index: 1; }
.index-header p { color: rgba(255,255,255,0.7); font-size: 1rem; position: relative; z-index: 1; }
"""

# ---- JavaScript for Iteration 4 & 5 ----
TOC_JS = """
<script>
(function() {
    // TOC active heading via IntersectionObserver
    var headings = document.querySelectorAll('.viewer h2, .viewer h3');
    var tocLinks = document.querySelectorAll('.toc-sidebar a');
    if (!headings.length || !tocLinks.length) return;

    var observer = new IntersectionObserver(function(entries) {
        entries.forEach(function(entry) {
            if (entry.isIntersecting) {
                var id = entry.target.id;
                tocLinks.forEach(function(link) {
                    link.classList.remove('toc-active');
                    if (link.getAttribute('href') === '#' + id) {
                        link.classList.add('toc-active');
                    }
                });
            }
        });
    }, { rootMargin: '-10% 0px -80% 0px', threshold: 0 });

    headings.forEach(function(h) { if (h.id) observer.observe(h); });
})();
</script>
"""

SEARCH_JS = """
<script>
(function() {
    var input = document.getElementById('course-search');
    var cards = document.querySelectorAll('.course-card-wrapper');
    var noResults = document.getElementById('no-results');
    if (!input) return;

    input.addEventListener('input', function() {
        var q = this.value.toLowerCase().trim();
        var visible = 0;
        cards.forEach(function(card) {
            var text = card.textContent.toLowerCase();
            var match = !q || text.includes(q);
            card.style.display = match ? '' : 'none';
            if (match) visible++;
        });
        if (noResults) noResults.style.display = visible === 0 ? 'block' : 'none';
    });
})();
</script>
"""


def page_html(title: str, body: str, breadcrumbs: list[tuple[str, str]] | None = None, extra_js: str = "") -> str:
    bc = ""
    if breadcrumbs:
        parts = []
        for label, href in breadcrumbs[:-1]:
            parts.append(f'<a href="{href}">{label}</a>')
        parts.append(f"<strong>{breadcrumbs[-1][0]}</strong>")
        bc = f'<div class="breadcrumb container">{" <span>/</span> ".join(parts)}</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} - Course Browser</title>
<style>{BASE_CSS}\n{PYGMENTS_CSS}</style>
</head>
<body>
<header>
<div class="container">
<h1><a href="/">&#128218; Course Browser</a> <a class="refresh-link" href="/?refresh=1">&#8635; Refresh</a></h1>
<div class="subtitle">{title}</div>
</div>
</header>
{bc}
<div class="container">
{body}
</div>
<footer>
<strong>Course Browser</strong> &mdash; {len(list(COURSES_DIR.iterdir()))} courses available
</footer>
{extra_js}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Request Handler
# ---------------------------------------------------------------------------

class CourseBrowserHandler(BaseHTTPRequestHandler):
    scanner = ContentScanner()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = urllib.parse.parse_qs(parsed.query)

        # Refresh cache
        if "refresh" in query:
            self.scanner.scan_all(force=True)

        try:
            if path == "/":
                self._serve_index()
            elif path.startswith("/course/") and "/module/" in path:
                # /course/{name}/module/{mod}
                parts = path.split("/")
                course = parts[2]
                mod = parts[4]
                self._serve_module(course, mod)
            elif path.startswith("/course/"):
                course = path.split("/")[2]
                self._serve_course(course)
            elif path == "/view":
                fpath = query.get("path", [""])[0]
                self._serve_view(fpath)
            elif path.startswith("/slides/"):
                fpath = path[len("/slides/"):]
                self._serve_slides(fpath)
            else:
                self._send_404()
        except Exception as e:
            self._send_error(500, str(e))

    def _serve_index(self):
        courses = self.scanner.scan_all()
        cards = []
        for idx, (key, info) in enumerate(courses.items()):
            stats = info["stats"]
            stat_badges = []
            if stats["modules"]:
                stat_badges.append(f'<span class="stat stat-modules">{stats["modules"]} modules</span>')
            if stats["slides"]:
                stat_badges.append(f'<span class="stat stat-slides">{stats["slides"]} slides</span>')
            if stats["notebooks"]:
                stat_badges.append(f'<span class="stat stat-notebooks">{stats["notebooks"]} notebooks</span>')
            if stats["guides"]:
                stat_badges.append(f'<span class="stat stat-guides">{stats["guides"]} guides</span>')

            desc = info["description"][:130] + "..." if len(info.get("description", "")) > 130 else info.get("description", "No description available.")
            icon = COURSE_ICONS.get(key, "📚")
            accent = ACCENT_COLORS[idx % len(ACCENT_COLORS)]

            cards.append(f"""
            <div class="course-card-wrapper">
            <a href="/course/{key}" style="text-decoration:none;color:inherit;display:block;height:100%;">
            <div class="card">
                <div class="card-accent" style="background: {accent};"></div>
                <div class="card-body">
                    <div class="card-icon">{icon}</div>
                    <h3>{info['title']}</h3>
                    <p>{desc}</p>
                    <div class="stats">{''.join(stat_badges)}</div>
                </div>
            </div>
            </a>
            </div>""")

        total_slides = sum(c["stats"]["slides"] for c in courses.values())
        total_nb = sum(c["stats"]["notebooks"] for c in courses.values())
        total_guides = sum(c["stats"]["guides"] for c in courses.values())
        total_modules = sum(c["stats"]["modules"] for c in courses.values())

        body = f"""
        <div class="index-header">
            <h2>&#127891; Course Library</h2>
            <p>{len(courses)} courses &middot; {total_modules} modules &middot; {total_slides} slide decks &middot; {total_nb} notebooks &middot; {total_guides} guides</p>
        </div>
        <div class="search-bar">
            <span class="search-icon">&#128269;</span>
            <input id="course-search" type="text" placeholder="Filter courses by name..." autocomplete="off">
        </div>
        <div class="card-grid" id="course-grid">{''.join(cards)}</div>
        <div class="no-results" id="no-results">No courses match your search.</div>
        <div class="footer-stats" style="margin-top:40px;margin-bottom:8px;">
            <div class="footer-stat"><span class="number">{len(courses)}</span><span class="label">Courses</span></div>
            <div class="footer-stat"><span class="number">{total_modules}</span><span class="label">Modules</span></div>
            <div class="footer-stat"><span class="number">{total_slides}</span><span class="label">Slide Decks</span></div>
            <div class="footer-stat"><span class="number">{total_nb}</span><span class="label">Notebooks</span></div>
            <div class="footer-stat"><span class="number">{total_guides}</span><span class="label">Guides</span></div>
        </div>"""

        self._send_html(page_html("All Courses", body, extra_js=SEARCH_JS))

    def _serve_course(self, course_key: str):
        courses = self.scanner.scan_all()
        if course_key not in courses:
            self._send_404()
            return

        info = courses[course_key]
        parts = []

        # README
        if info["has_readme"]:
            readme_path = COURSES_DIR / course_key / "README.md"
            readme_html, _ = render_markdown(readme_path.read_text(encoding="utf-8", errors="replace"))
            parts.append(f'<div class="viewer" style="margin-bottom:28px">{readme_html}</div>')

        # Modules table
        if info["modules"]:
            rows = []
            for mod in info["modules"]:
                counts = []
                if mod["slides"]:
                    counts.append(f'<span class="stat stat-slides" style="display:inline-block">{len(mod["slides"])} slides</span>')
                for sub, files in mod["content"].items():
                    css_class = {"notebooks": "stat-notebooks", "guides": "stat-guides"}.get(sub, "stat-modules")
                    counts.append(f'<span class="stat {css_class}" style="display:inline-block">{len(files)} {sub}</span>')
                count_str = " ".join(counts) if counts else "—"
                rows.append(f"""
                <tr>
                    <td><a href="/course/{course_key}/module/{mod['dir_name']}">{mod['name']}</a></td>
                    <td style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;">{count_str}</td>
                </tr>""")

            parts.append(f"""
            <div class="section">
                <h2>Modules ({len(info['modules'])})</h2>
                <table>
                    <tr><th>Module</th><th>Content</th></tr>
                    {''.join(rows)}
                </table>
            </div>""")

        # Extra sections (quick-starts, templates, etc.)
        for folder, sec in info["sections"].items():
            file_links = self._file_list_html(sec["files"])
            parts.append(f"""
            <div class="section">
                <h2>{sec['label']} ({len(sec['files'])} files)</h2>
                {file_links}
            </div>""")

        breadcrumbs = [("Home", "/"), (info["title"], "")]
        self._send_html(page_html(info["title"], "\n".join(parts), breadcrumbs))

    def _serve_module(self, course_key: str, mod_name: str):
        courses = self.scanner.scan_all()
        if course_key not in courses:
            self._send_404()
            return

        course = courses[course_key]
        mod = None
        for m in course["modules"]:
            if m["dir_name"] == mod_name:
                mod = m
                break
        if not mod:
            self._send_404()
            return

        parts = []

        # Iteration 4: Module nav pill strip
        if course["modules"]:
            pills = []
            pills.append('<span class="module-nav-label">All Modules</span>')
            for m in course["modules"]:
                is_active = m["dir_name"] == mod_name
                active_class = " active" if is_active else ""
                pills.append(
                    f'<a class="module-pill{active_class}" href="/course/{course_key}/module/{m["dir_name"]}">'
                    f'{m["name"]}</a>'
                )
            parts.append(f'<div class="module-nav">{"".join(pills)}</div>')

        # Module README
        if mod["has_readme"]:
            readme_path = PROJECT_ROOT / mod["dir"] / "README.md"
            readme_html, _ = render_markdown(readme_path.read_text(encoding="utf-8", errors="replace"))
            parts.append(f'<div class="viewer" style="margin-bottom:28px">{readme_html}</div>')

        # Slides
        if mod["slides"]:
            slide_links = []
            for s in mod["slides"]:
                slide_links.append(f'<a class="slide-link" href="/slides/{s["path"]}" target="_blank">&#9654; {s["name"]}</a>')
            parts.append(f"""
            <div class="section">
                <h2>&#127913; Slide Decks ({len(mod['slides'])})</h2>
                <div>{''.join(slide_links)}</div>
            </div>""")

        # Content sections
        section_labels = {
            "guides": "&#128196; Guides",
            "notebooks": "&#128203; Notebooks",
            "exercises": "&#9997; Exercises",
            "resources": "&#128190; Resources",
        }
        for sub, label in section_labels.items():
            if sub in mod["content"]:
                files = mod["content"][sub]
                file_links = self._file_list_html(files)
                parts.append(f"""
                <div class="section">
                    <h2>{label} ({len(files)} files)</h2>
                    {file_links}
                </div>""")

        breadcrumbs = [
            ("Home", "/"),
            (course["title"], f"/course/{course_key}"),
            (mod["name"], ""),
        ]
        self._send_html(page_html(mod["name"], "\n".join(parts), breadcrumbs))

    def _serve_view(self, rel_path: str):
        if not rel_path:
            self._send_404()
            return

        full = safe_path(rel_path)
        if not full or not full.exists():
            self._send_404()
            return

        ext = full.suffix.lower()
        name = full.name
        extra_js = ""
        toc_html = ""

        try:
            if ext == ".md":
                text = full.read_text(encoding="utf-8", errors="replace")
                rendered, headings = render_markdown(text, add_toc=True)

                # Iteration 4: Build TOC sidebar if enough headings
                if len(headings) >= 3:
                    toc_items = []
                    for h in headings:
                        h_class = "toc-h3" if h["level"] == 3 else "toc-h2"
                        toc_items.append(
                            f'<li class="{h_class}"><a href="#{h["id"]}">{_escape(h["text"])}</a></li>'
                        )
                    toc_html = f'''
                    <aside class="toc-sidebar">
                        <h4>Contents</h4>
                        <ul>{"".join(toc_items)}</ul>
                    </aside>'''
                    extra_js = TOC_JS

                content = f'<div class="viewer-wrapper"><div class="viewer">{rendered}</div>{toc_html}</div>'

            elif ext == ".ipynb":
                rendered = render_notebook(full)
                content = f'<div class="viewer">{rendered}</div>'
            elif ext == ".py":
                rendered = render_python(full)
                content = f'<div class="viewer">{rendered}</div>'
            elif ext == ".html":
                # Redirect to slides endpoint for HTML
                self.send_response(302)
                self.send_header("Location", f"/slides/{rel_path}")
                self.end_headers()
                return
            else:
                text = full.read_text(encoding="utf-8", errors="replace")
                content = f'<div class="viewer"><pre>{_escape(text)}</pre></div>'
        except Exception as e:
            content = f'<div class="viewer"><p style="color:red">Error rendering: {_escape(str(e))}</p></div>'

        # Build breadcrumbs from path
        rel = Path(rel_path)
        crumbs = [("Home", "/")]
        # Try to find course context
        if rel.parts[0] == "courses" and len(rel.parts) > 1:
            course_key = rel.parts[1]
            courses = self.scanner.scan_all()
            if course_key in courses:
                crumbs.append((courses[course_key]["title"], f"/course/{course_key}"))
                if len(rel.parts) > 3 and rel.parts[2] == "modules":
                    mod_name = rel.parts[3]
                    crumbs.append((humanize(mod_name), f"/course/{course_key}/module/{mod_name}"))
        crumbs.append((name, ""))

        self._send_html(page_html(name, content, crumbs, extra_js=extra_js))

    def _serve_slides(self, rel_path: str):
        full = safe_path(rel_path)
        if not full or not full.exists() or full.suffix.lower() != ".html":
            self._send_404()
            return

        html = full.read_text(encoding="utf-8", errors="replace")

        # Marp's bespoke-marp sets body{background:#000}, which shows as
        # letterbox bars when the viewport aspect ratio doesn't match 16:9.
        # Inject CSS to default to the course navy, then JS to dynamically
        # sync the body background to each active slide's background color.
        _SLIDE_INJECT = """<style>
html,body{background:#1a365d!important}
</style>
<script>
(function(){
  var SLIDE_W=1280,SLIDE_H=720;
  function syncBg(){
    var s=document.querySelector('section.bespoke-marp-active');
    if(!s)return;
    var cs=getComputedStyle(s);
    var bi=cs.backgroundImage;
    var bc=cs.backgroundColor;
    var fill=(bi&&bi!=='none')?bi+','+bc:bc;
    if(fill&&fill.indexOf('rgba(0, 0, 0, 0)')<0){
      document.documentElement.style.background=fill;
      document.body.style.background=fill;
    }
  }
  // Sync on load and after bespoke initialises
  [100,300,600,1200].forEach(function(t){setTimeout(syncBg,t)});
  // Sync on every navigation gesture
  document.addEventListener('keydown',function(){setTimeout(syncBg,30)});
  document.addEventListener('click',function(){setTimeout(syncBg,30)});
})();
</script>"""
        html = html.replace('</head>', _SLIDE_INJECT + '</head>', 1)

        content = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _file_list_html(self, files: list[dict]) -> str:
        items = []
        for f in files:
            ftype = f["type"]
            badge_class = {
                "notebook": "badge-notebook", "markdown": "badge-markdown",
                "python": "badge-python", "html": "badge-html",
            }.get(ftype, "")
            badge = f'<span class="badge {badge_class}">{ftype}</span>' if badge_class else ""

            # Choose link behavior
            if ftype == "html" and "_slides" in f["name"]:
                link = f'<a href="/slides/{f["path"]}" target="_blank">{f["name"]}</a>'
            elif ftype in ("markdown", "notebook", "python"):
                link = f'<a href="/view?path={urllib.parse.quote(f["path"])}">{f["name"]}</a>'
            elif ftype == "html":
                link = f'<a href="/slides/{f["path"]}" target="_blank">{f["name"]}</a>'
            else:
                link = f["name"]

            size_kb = f["size"] / 1024
            size_str = f"{size_kb:.0f} KB" if size_kb >= 1 else f'{f["size"]} B'
            items.append(f'<li>{link} {badge} <span style="color:var(--text-light);font-size:0.8rem;margin-left:auto">({size_str})</span></li>')

        return f'<ul class="file-list">{"".join(items)}</ul>'

    def _send_html(self, html: str):
        data = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_404(self):
        body = page_html("Not Found", '<div class="viewer"><h2>404 - Not Found</h2><p><a href="/">Back to Home</a></p></div>')
        data = body.encode("utf-8")
        self.send_response(404)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, code: int, msg: str):
        body = page_html("Error", f'<div class="viewer"><h2>Error {code}</h2><p>{_escape(msg)}</p></div>')
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        # Quieter logging
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Course Content Browser")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()

    server = HTTPServer(("127.0.0.1", args.port), CourseBrowserHandler)
    print(f"Course Browser running at http://127.0.0.1:{args.port}")
    print(f"Scanning {COURSES_DIR}")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
