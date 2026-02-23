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

def render_markdown(text: str) -> str:
    """Render markdown to HTML with syntax highlighting."""
    md = markdown.Markdown(extensions=[
        "fenced_code", "tables", "codehilite", "toc", "nl2br",
    ], extension_configs={
        "codehilite": {"css_class": "highlight", "guess_lang": True},
    })
    return md.convert(text)


def render_notebook(nb_path: Path) -> str:
    """Render a Jupyter notebook to HTML without nbconvert."""
    data = json.loads(nb_path.read_text(encoding="utf-8", errors="replace"))
    cells = data.get("cells", [])
    kernel = data.get("metadata", {}).get("kernelspec", {}).get("language", "python")

    parts = []
    for cell in cells:
        ctype = cell.get("cell_type", "code")
        source = "".join(cell.get("source", []))

        if ctype == "markdown":
            parts.append(f'<div class="nb-cell nb-md">{render_markdown(source)}</div>')

        elif ctype == "code":
            try:
                lexer = get_lexer_by_name(kernel)
            except Exception:
                lexer = PythonLexer()
            formatter = HtmlFormatter(cssclass="highlight", nowrap=False)
            highlighted = highlight(source, lexer, formatter)
            parts.append(f'<div class="nb-cell nb-code">{highlighted}</div>')

            # Outputs
            for out in cell.get("outputs", []):
                out_type = out.get("output_type", "")
                if out_type == "stream":
                    text = "".join(out.get("text", []))
                    parts.append(f'<div class="nb-cell nb-output"><pre>{_escape(text)}</pre></div>')
                elif out_type in ("execute_result", "display_data"):
                    od = out.get("data", {})
                    if "text/html" in od:
                        html = "".join(od["text/html"])
                        parts.append(f'<div class="nb-cell nb-output">{html}</div>')
                    elif "image/png" in od:
                        img = od["image/png"]
                        parts.append(f'<div class="nb-cell nb-output"><img src="data:image/png;base64,{img}"></div>')
                    elif "text/plain" in od:
                        text = "".join(od["text/plain"])
                        parts.append(f'<div class="nb-cell nb-output"><pre>{_escape(text)}</pre></div>')
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
    --bg: #f7fafc;
    --card-bg: #ffffff;
    --text: #2d3748;
    --text-light: #718096;
    --border: #e2e8f0;
    --code-bg: #1e1e2e;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6;
}
a { color: var(--navy); text-decoration: none; }
a:hover { color: var(--orange); text-decoration: underline; }
.container { max-width: 1200px; margin: 0 auto; padding: 0 24px; }
header {
    background: linear-gradient(135deg, var(--navy) 0%, var(--navy-light) 100%);
    color: white; padding: 20px 0; margin-bottom: 32px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
header h1 { font-size: 1.5rem; }
header a { color: white; }
header .subtitle { color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-top: 4px; }
.breadcrumb {
    padding: 12px 0; font-size: 0.85rem; color: var(--text-light);
    border-bottom: 1px solid var(--border); margin-bottom: 24px;
}
.breadcrumb a { color: var(--navy); }
.breadcrumb span { margin: 0 6px; }

/* Cards */
.card-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 20px; }
.card {
    background: var(--card-bg); border-radius: 10px; padding: 24px;
    border: 1px solid var(--border); transition: box-shadow 0.2s, transform 0.2s;
}
.card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.1); transform: translateY(-2px); }
.card h3 { color: var(--navy); margin-bottom: 8px; font-size: 1.1rem; }
.card p { color: var(--text-light); font-size: 0.9rem; margin-bottom: 12px; }
.card .stats { display: flex; gap: 16px; flex-wrap: wrap; }
.stat {
    background: var(--bg); padding: 4px 10px; border-radius: 6px;
    font-size: 0.8rem; color: var(--text-light);
}
.stat strong { color: var(--navy); }

/* Tables */
table { width: 100%; border-collapse: collapse; margin: 16px 0; }
th { background: var(--navy); color: white; padding: 10px 14px; text-align: left; font-size: 0.85rem; }
td { padding: 10px 14px; border-bottom: 1px solid var(--border); font-size: 0.9rem; }
tr:hover td { background: #edf2f7; }

/* Sections */
.section { margin-bottom: 32px; }
.section h2 {
    color: var(--navy); font-size: 1.2rem; margin-bottom: 16px;
    padding-bottom: 8px; border-bottom: 2px solid var(--orange);
}
.section h3 { color: var(--navy); font-size: 1rem; margin: 16px 0 8px; }

/* Content viewer */
.viewer { background: var(--card-bg); border-radius: 10px; padding: 32px; border: 1px solid var(--border); }
.viewer img { max-width: 100%; }
.viewer h1, .viewer h2, .viewer h3 { color: var(--navy); margin-top: 24px; margin-bottom: 12px; }
.viewer table { font-size: 0.85rem; }
.viewer pre { background: var(--code-bg); color: #cdd6f4; padding: 16px; border-radius: 8px; overflow-x: auto; }
.viewer code { font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.85rem; }
.viewer p code {
    background: #edf2f7; color: var(--navy); padding: 2px 6px;
    border-radius: 4px; font-size: 0.85rem;
}
.viewer blockquote {
    border-left: 4px solid var(--orange); padding: 12px 16px; margin: 16px 0;
    background: #fffaf0; color: #744210; border-radius: 0 6px 6px 0;
}

/* Notebook cells */
.nb-cell { margin-bottom: 16px; border-radius: 8px; overflow: hidden; }
.nb-md { padding: 16px; }
.nb-code { border: 1px solid #313244; }
.nb-code .highlight { margin: 0; padding: 12px; border-radius: 0; }
.nb-output { background: #f1f5f9; padding: 12px; border-left: 3px solid var(--border); }
.nb-output pre { background: transparent; color: var(--text); padding: 0; }
.nb-error { background: #fff5f5; padding: 12px; border-left: 3px solid #e53e3e; }
.nb-error pre { background: transparent; color: #c53030; padding: 0; }

/* File list */
.file-list { list-style: none; }
.file-list li { padding: 6px 0; }
.file-list .badge {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.7rem; font-weight: 600; margin-left: 8px; text-transform: uppercase;
}
.badge-slides { background: #c6f6d5; color: #22543d; }
.badge-notebook { background: #bee3f8; color: #2a4365; }
.badge-markdown { background: #fefcbf; color: #744210; }
.badge-python { background: #e9d8fd; color: #44337a; }
.badge-html { background: #fed7d7; color: #742a2a; }

/* Icon links for slides */
.slide-link {
    display: inline-flex; align-items: center; gap: 6px;
    background: var(--navy); color: white; padding: 6px 14px;
    border-radius: 6px; font-size: 0.85rem; margin: 4px 4px 4px 0;
    transition: background 0.2s;
}
.slide-link:hover { background: var(--orange); color: white; text-decoration: none; }

.refresh-link {
    float: right; font-size: 0.8rem; color: rgba(255,255,255,0.7);
}
.refresh-link:hover { color: white; }

footer { text-align: center; padding: 24px; color: var(--text-light); font-size: 0.8rem; margin-top: 40px; }
"""


def page_html(title: str, body: str, breadcrumbs: list[tuple[str, str]] | None = None) -> str:
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
<h1><a href="/">Course Browser</a> <a class="refresh-link" href="/?refresh=1">Refresh</a></h1>
<div class="subtitle">{title}</div>
</div>
</header>
{bc}
<div class="container">
{body}
</div>
<footer>Course Browser &middot; {len(list(COURSES_DIR.iterdir()))} courses</footer>
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
        for key, info in courses.items():
            stats = info["stats"]
            stat_badges = []
            if stats["modules"]:
                stat_badges.append(f'<span class="stat"><strong>{stats["modules"]}</strong> modules</span>')
            if stats["slides"]:
                stat_badges.append(f'<span class="stat"><strong>{stats["slides"]}</strong> slides</span>')
            if stats["notebooks"]:
                stat_badges.append(f'<span class="stat"><strong>{stats["notebooks"]}</strong> notebooks</span>')
            if stats["guides"]:
                stat_badges.append(f'<span class="stat"><strong>{stats["guides"]}</strong> guides</span>')

            desc = info["description"][:120] + "..." if len(info.get("description", "")) > 120 else info.get("description", "")
            cards.append(f"""
            <a href="/course/{key}" style="text-decoration:none;color:inherit;">
            <div class="card">
                <h3>{info['title']}</h3>
                <p>{desc}</p>
                <div class="stats">{''.join(stat_badges)}</div>
            </div>
            </a>""")

        total_slides = sum(c["stats"]["slides"] for c in courses.values())
        total_nb = sum(c["stats"]["notebooks"] for c in courses.values())
        total_guides = sum(c["stats"]["guides"] for c in courses.values())

        body = f"""
        <div class="section">
            <h2>All Courses ({len(courses)} courses &middot; {total_slides} slide decks &middot; {total_nb} notebooks &middot; {total_guides} guides)</h2>
            <div class="card-grid">{''.join(cards)}</div>
        </div>"""

        self._send_html(page_html("All Courses", body))

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
            readme_html = render_markdown(readme_path.read_text(encoding="utf-8", errors="replace"))
            parts.append(f'<div class="viewer" style="margin-bottom:24px">{readme_html}</div>')

        # Modules table
        if info["modules"]:
            rows = []
            for mod in info["modules"]:
                counts = []
                if mod["slides"]:
                    counts.append(f'{len(mod["slides"])} slides')
                for sub, files in mod["content"].items():
                    counts.append(f'{len(files)} {sub}')
                count_str = " &middot; ".join(counts) if counts else "—"
                rows.append(f"""
                <tr>
                    <td><a href="/course/{course_key}/module/{mod['dir_name']}">{mod['name']}</a></td>
                    <td>{count_str}</td>
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

        # Module README
        if mod["has_readme"]:
            readme_path = PROJECT_ROOT / mod["dir"] / "README.md"
            readme_html = render_markdown(readme_path.read_text(encoding="utf-8", errors="replace"))
            parts.append(f'<div class="viewer" style="margin-bottom:24px">{readme_html}</div>')

        # Slides
        if mod["slides"]:
            slide_links = []
            for s in mod["slides"]:
                slide_links.append(f'<a class="slide-link" href="/slides/{s["path"]}" target="_blank">&#9654; {s["name"]}</a>')
            parts.append(f"""
            <div class="section">
                <h2>Slide Decks ({len(mod['slides'])})</h2>
                <div>{''.join(slide_links)}</div>
            </div>""")

        # Content sections
        section_labels = {
            "guides": "Guides", "notebooks": "Notebooks",
            "exercises": "Exercises", "resources": "Resources",
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

        try:
            if ext == ".md":
                text = full.read_text(encoding="utf-8", errors="replace")
                rendered = render_markdown(text)
                content = f'<div class="viewer">{rendered}</div>'
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

        self._send_html(page_html(name, content, crumbs))

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
            items.append(f'<li>{link} {badge} <span style="color:var(--text-light);font-size:0.8rem">({size_str})</span></li>')

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
