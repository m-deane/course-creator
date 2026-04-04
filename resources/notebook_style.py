"""Notebook style injector for course-consistent Jupyter styling."""
from IPython.display import HTML


# Callout type configuration: (background, border, icon, label)
_CALLOUT_STYLES = {
    "insight": {"bg": "#f3e5f5", "border": "#7c4dff", "icon": "\U0001f4a1", "label": "Key Insight"},
    "warning": {"bg": "#fff8e1", "border": "#ff9800", "icon": "\u26a0\ufe0f", "label": "Warning"},
    "key":     {"bg": "#e8f5e9", "border": "#4caf50", "icon": "\U0001f511", "label": "Key Point"},
    "danger":  {"bg": "#fce4ec", "border": "#ef5350", "icon": "\U0001f6a8", "label": "Danger"},
    "info":    {"bg": "#e3f2fd", "border": "#2196f3", "icon": "\u2139\ufe0f", "label": "Info"},
}


def apply_course_theme() -> HTML:
    """Inject CSS into notebook for consistent styling.

    Returns an IPython HTML object with a <style> block that styles
    headings (serif), DataFrames (striped rows, colored header),
    and code output (monospace, subtle background).
    """
    css = """<style>
/* === Course Theme: Headings === */
.rendered_html h1, .jp-RenderedHTMLCommon h1 {
    font-family: Georgia, 'Playfair Display', serif;
    color: #1a1a2e;
    font-size: 2.2em;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin-bottom: 0.5em;
}
.rendered_html h2, .jp-RenderedHTMLCommon h2 {
    font-family: Georgia, 'Playfair Display', serif;
    color: #1a1a2e;
    font-size: 1.6em;
    font-weight: 600;
    letter-spacing: -0.01em;
    border-bottom: 3px solid #ff9800;
    padding-bottom: 0.3em;
    display: inline-block;
}
.rendered_html h3, .jp-RenderedHTMLCommon h3 {
    font-family: Georgia, 'Playfair Display', serif;
    color: #212121;
    font-size: 1.2em;
    font-weight: 600;
    letter-spacing: -0.01em;
}

/* === Course Theme: DataFrame === */
.dataframe, .rendered_html table, .jp-RenderedHTMLCommon table {
    border-collapse: collapse;
    width: 100%;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #e0e0e0;
}
.dataframe thead th, .rendered_html table thead th, .jp-RenderedHTMLCommon table thead th {
    background-color: #1a1a2e;
    color: #f5f5f5;
    font-weight: 600;
    padding: 10px 14px;
    text-align: left;
    font-size: 0.9em;
}
.dataframe tbody tr:nth-child(even), .rendered_html table tbody tr:nth-child(even) {
    background-color: #fafafa;
}
.dataframe tbody tr:nth-child(odd), .rendered_html table tbody tr:nth-child(odd) {
    background-color: #ffffff;
}
.dataframe tbody td, .rendered_html table tbody td, .jp-RenderedHTMLCommon table tbody td {
    padding: 8px 14px;
    border: 1px solid #e0e0e0;
    font-size: 0.9em;
}
.dataframe tbody tr:hover {
    background-color: #e3f2fd;
}

/* === Course Theme: Code Output === */
.output_text pre, .jp-OutputArea-output pre {
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
    background-color: #f5f5f5;
    padding: 12px;
    border-radius: 6px;
    font-size: 0.85em;
    line-height: 1.5;
}
</style>"""
    return HTML(css)


def learning_objectives(items: list) -> HTML:
    """Render a styled learning objectives card.

    Args:
        items: List of objective strings displayed as a checklist.

    Returns:
        IPython HTML object with a mint-colored checklist card.
    """
    checklist = "".join(
        f'<li style="margin-bottom: 8px; list-style: none; padding-left: 0;">'
        f'\u2705 {item}</li>'
        for item in items
    )
    html = f"""<div style="
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        border-radius: 8px;
        padding: 20px 24px;
        margin: 16px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    ">
        <h3 style="
            margin: 0 0 12px 0;
            font-family: Georgia, 'Playfair Display', serif;
            color: #1b5e20;
            font-size: 1.1em;
            font-weight: 600;
        ">\U0001f3af Learning Objectives</h3>
        <ul style="margin: 0; padding: 0;">{checklist}</ul>
    </div>"""
    return HTML(html)


def section_divider(title: str = None) -> HTML:
    """Render a styled section break.

    Args:
        title: Optional title text centered over the divider.

    Returns:
        IPython HTML object with a centered horizontal rule.
    """
    if title:
        html = f"""<div style="
            text-align: center;
            margin: 32px 0;
            position: relative;
        ">
            <hr style="
                border: none;
                border-top: 2px solid #e0e0e0;
                margin: 0;
            ">
            <span style="
                background: white;
                padding: 0 16px;
                position: relative;
                top: -12px;
                font-family: Georgia, 'Playfair Display', serif;
                color: #757575;
                font-size: 0.95em;
                font-weight: 600;
            ">{title}</span>
        </div>"""
    else:
        html = """<div style="
            text-align: center;
            margin: 32px 0;
        ">
            <hr style="
                border: none;
                border-top: 2px solid #e0e0e0;
            ">
        </div>"""
    return HTML(html)


def callout(text: str, kind: str = "insight") -> HTML:
    """Render a styled callout box.

    Args:
        text: The callout body text.
        kind: One of 'insight', 'warning', 'key', 'danger', 'info'.

    Returns:
        IPython HTML object with a colored callout box.
    """
    style = _CALLOUT_STYLES.get(kind, _CALLOUT_STYLES["insight"])
    html = f"""<div style="
        background-color: {style['bg']};
        border-left: 4px solid {style['border']};
        border-radius: 8px;
        padding: 16px 20px;
        margin: 16px 0;
        font-size: 0.95em;
        line-height: 1.6;
    ">
        <strong>{style['icon']} {style['label']}:</strong> {text}
    </div>"""
    return HTML(html)


def key_takeaways(items: list) -> HTML:
    """Render a styled takeaways summary card.

    Args:
        items: List of takeaway strings displayed as numbered items.

    Returns:
        IPython HTML object with a styled summary card.
    """
    numbered = "".join(
        f'<li style="margin-bottom: 8px; padding-left: 4px;">{item}</li>'
        for item in items
    )
    html = f"""<div style="
        background: linear-gradient(135deg, #f3e5f5, #e3f2fd);
        border-radius: 8px;
        padding: 20px 24px;
        margin: 16px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    ">
        <h3 style="
            margin: 0 0 12px 0;
            font-family: Georgia, 'Playfair Display', serif;
            color: #1a1a2e;
            font-size: 1.1em;
            font-weight: 600;
        ">\U0001f4cb Key Takeaways</h3>
        <ol style="margin: 0; padding-left: 20px;">{numbered}</ol>
    </div>"""
    return HTML(html)
