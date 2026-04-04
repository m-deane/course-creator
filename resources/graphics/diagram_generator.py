"""SVG diagram generator for course materials.

Produces professional-grade SVG diagrams using only Python stdlib
(xml.etree.ElementTree). Each public function returns an independent
SVGDocument instance with pastel fills, rounded corners, and clean
typography matching the course styling system.
"""

from __future__ import annotations

import base64
import xml.etree.ElementTree as ET
from typing import Optional

# ---------------------------------------------------------------------------
# Style Constants
# ---------------------------------------------------------------------------

COLORS = {
    "mint": {"bg": "#e8f5e9", "border": "#4caf50", "text": "#1b5e20"},
    "amber": {"bg": "#fff8e1", "border": "#ff9800", "text": "#e65100"},
    "blue": {"bg": "#e3f2fd", "border": "#2196f3", "text": "#0d47a1"},
    "lavender": {"bg": "#f3e5f5", "border": "#7c4dff", "text": "#4a148c"},
    "rose": {"bg": "#fce4ec", "border": "#ef5350", "text": "#b71c1c"},
    "gray": {"bg": "#f5f5f5", "border": "#9e9e9e", "text": "#424242"},
}

FONTS = {
    "heading": "Georgia, serif",
    "body": "Inter, sans-serif",
    "code": "JetBrains Mono, monospace",
}

NODE_RADIUS = 12
PADDING = 20
ARROW_HEAD_SIZE = 8

_DEFAULT_COLOR_CYCLE = ["mint", "amber", "blue", "lavender", "rose", "gray"]

_SVG_NS = "http://www.w3.org/2000/svg"

# ---------------------------------------------------------------------------
# SVGDocument
# ---------------------------------------------------------------------------


class SVGDocument:
    """Immutable wrapper around a completed SVG element tree.

    Returned by every public generator function. Supports saving to disk,
    string serialization, base64 data-URI encoding, and Jupyter rich display.
    """

    def __init__(self, root: ET.Element) -> None:
        self._root = root

    # -- public API ---------------------------------------------------------

    def save(self, path: str) -> None:
        """Write the SVG to *path*."""
        tree = ET.ElementTree(self._root)
        ET.indent(tree, space="  ")
        with open(path, "wb") as fh:
            tree.write(fh, encoding="utf-8", xml_declaration=True)

    def to_string(self) -> str:
        """Return the SVG as a UTF-8 string."""
        ET.indent(self._root, space="  ")
        return ET.tostring(self._root, encoding="unicode")

    def to_data_uri(self) -> str:
        """Return a ``data:image/svg+xml;base64,...`` URI for inline embedding."""
        raw = self.to_string().encode("utf-8")
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/svg+xml;base64,{b64}"

    def _repr_svg_(self) -> str:
        """Jupyter notebook rich display integration."""
        return self.to_string()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_svg(width: int, height: int) -> ET.Element:
    """Create a root ``<svg>`` element with standard attributes."""
    svg = ET.Element("svg")
    svg.set("xmlns", _SVG_NS)
    svg.set("width", str(width))
    svg.set("height", str(height))
    svg.set("viewBox", f"0 0 {width} {height}")
    svg.set("font-family", FONTS["body"])
    return svg


def _add_style(svg: ET.Element) -> None:
    """Inject a ``<style>`` element with embedded font declarations."""
    style = ET.SubElement(svg, "style")
    style.text = (
        f"text {{ font-family: {FONTS['body']}; }}"
        f" .heading {{ font-family: {FONTS['heading']}; font-weight: 700; }}"
        f" .code {{ font-family: {FONTS['code']}; }}"
    )


def _add_arrow_marker(
    defs: ET.Element,
    marker_id: str = "arrowhead",
    color: str = "#757575",
) -> None:
    """Append an arrowhead ``<marker>`` to *defs*."""
    marker = ET.SubElement(defs, "marker")
    marker.set("id", marker_id)
    marker.set("markerWidth", str(ARROW_HEAD_SIZE))
    marker.set("markerHeight", str(ARROW_HEAD_SIZE))
    marker.set("refX", str(ARROW_HEAD_SIZE))
    marker.set("refY", str(ARROW_HEAD_SIZE // 2))
    marker.set("orient", "auto")
    marker.set("markerUnits", "userSpaceOnUse")
    poly = ET.SubElement(marker, "polygon")
    s = ARROW_HEAD_SIZE
    poly.set("points", f"0 0, {s} {s // 2}, 0 {s}")
    poly.set("fill", color)


def _add_drop_shadow(defs: ET.Element, filter_id: str = "shadow") -> None:
    """Append a subtle drop-shadow ``<filter>`` to *defs*."""
    filt = ET.SubElement(defs, "filter")
    filt.set("id", filter_id)
    filt.set("x", "-5%")
    filt.set("y", "-5%")
    filt.set("width", "110%")
    filt.set("height", "120%")
    offset = ET.SubElement(filt, "feOffset")
    offset.set("dx", "0")
    offset.set("dy", "2")
    offset.set("in", "SourceAlpha")
    offset.set("result", "offOut")
    blur = ET.SubElement(filt, "feGaussianBlur")
    blur.set("in", "offOut")
    blur.set("stdDeviation", "3")
    blur.set("result", "blurOut")
    blend = ET.SubElement(filt, "feBlend")
    blend.set("in", "SourceGraphic")
    blend.set("in2", "blurOut")
    blend.set("mode", "normal")


def _color(name: str) -> dict:
    """Resolve a color name to its dict, defaulting to gray."""
    return COLORS.get(name, COLORS["gray"])


def _add_title(svg: ET.Element, title: str, cx: int, y: int) -> None:
    """Add a centered heading title to the SVG."""
    t = ET.SubElement(svg, "text")
    t.set("x", str(cx))
    t.set("y", str(y))
    t.set("text-anchor", "middle")
    t.set("dominant-baseline", "middle")
    t.set("class", "heading")
    t.set("font-size", "22")
    t.set("fill", "#1a1a2e")
    t.text = title


def _rounded_rect(
    parent: ET.Element,
    x: int,
    y: int,
    w: int,
    h: int,
    fill: str,
    stroke: str,
    stroke_width: int = 2,
    rx: int = NODE_RADIUS,
    filter_id: Optional[str] = "shadow",
) -> ET.Element:
    """Append a rounded ``<rect>`` with optional drop-shadow filter."""
    rect = ET.SubElement(parent, "rect")
    rect.set("x", str(x))
    rect.set("y", str(y))
    rect.set("width", str(w))
    rect.set("height", str(h))
    rect.set("rx", str(rx))
    rect.set("ry", str(rx))
    rect.set("fill", fill)
    rect.set("stroke", stroke)
    rect.set("stroke-width", str(stroke_width))
    if filter_id:
        rect.set("filter", f"url(#{filter_id})")
    return rect


def _text(
    parent: ET.Element,
    x: int,
    y: int,
    content: str,
    *,
    font_size: int = 14,
    fill: str = "#212121",
    anchor: str = "middle",
    baseline: str = "middle",
    cls: Optional[str] = None,
    font_weight: Optional[str] = None,
) -> ET.Element:
    """Append a ``<text>`` element to *parent*."""
    t = ET.SubElement(parent, "text")
    t.set("x", str(x))
    t.set("y", str(y))
    t.set("text-anchor", anchor)
    t.set("dominant-baseline", baseline)
    t.set("font-size", str(font_size))
    t.set("fill", fill)
    if cls:
        t.set("class", cls)
    if font_weight:
        t.set("font-weight", font_weight)
    t.text = content
    return t


def _line(
    parent: ET.Element,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    stroke: str = "#757575",
    stroke_width: int = 2,
    marker_end: Optional[str] = "arrowhead",
) -> ET.Element:
    """Append a ``<line>`` element, optionally with an arrowhead marker."""
    ln = ET.SubElement(parent, "line")
    ln.set("x1", str(x1))
    ln.set("y1", str(y1))
    ln.set("x2", str(x2))
    ln.set("y2", str(y2))
    ln.set("stroke", stroke)
    ln.set("stroke-width", str(stroke_width))
    if marker_end:
        ln.set("marker-end", f"url(#{marker_end})")
    return ln


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_flow(
    steps: list[str],
    colors: Optional[list[str]] = None,
    title: Optional[str] = None,
    orientation: str = "horizontal",
) -> SVGDocument:
    """Generate a numbered step-by-step process flow diagram.

    Parameters
    ----------
    steps : list[str]
        Labels for each step box.
    colors : list[str], optional
        Color names (keys of ``COLORS``) per step. Cycles through the default
        palette when not provided.
    title : str, optional
        Optional title rendered above the diagram.
    orientation : str
        ``"horizontal"`` (default) or ``"vertical"``.
    """
    n = len(steps)
    if colors is None:
        colors = [_DEFAULT_COLOR_CYCLE[i % len(_DEFAULT_COLOR_CYCLE)] for i in range(n)]
    while len(colors) < n:
        colors.append(_DEFAULT_COLOR_CYCLE[len(colors) % len(_DEFAULT_COLOR_CYCLE)])

    box_w, box_h = 160, 60
    gap = 50  # space between boxes (includes arrow)
    title_h = 50 if title else 0

    if orientation == "vertical":
        canvas_w = box_w + 2 * PADDING
        canvas_h = n * box_h + (n - 1) * gap + 2 * PADDING + title_h
    else:
        canvas_w = n * box_w + (n - 1) * gap + 2 * PADDING
        canvas_h = box_h + 2 * PADDING + title_h

    svg = _make_svg(canvas_w, canvas_h)
    _add_style(svg)
    defs = ET.SubElement(svg, "defs")
    _add_drop_shadow(defs)
    _add_arrow_marker(defs)

    if title:
        _add_title(svg, title, canvas_w // 2, PADDING + 16)

    for i, step in enumerate(steps):
        c = _color(colors[i])
        if orientation == "vertical":
            bx = PADDING
            by = PADDING + title_h + i * (box_h + gap)
        else:
            bx = PADDING + i * (box_w + gap)
            by = PADDING + title_h

        _rounded_rect(svg, bx, by, box_w, box_h, c["bg"], c["border"])

        # Step number circle
        circle = ET.SubElement(svg, "circle")
        circle.set("cx", str(bx + 20))
        circle.set("cy", str(by + box_h // 2))
        circle.set("r", "12")
        circle.set("fill", c["border"])
        _text(svg, bx + 20, by + box_h // 2, str(i + 1),
              font_size=12, fill="#ffffff", font_weight="700")

        # Step label
        _text(svg, bx + box_w // 2 + 10, by + box_h // 2, step,
              font_size=14, fill=c["text"], font_weight="600")

        # Arrow to next step
        if i < n - 1:
            if orientation == "vertical":
                ax1 = bx + box_w // 2
                ay1 = by + box_h
                ax2 = ax1
                ay2 = by + box_h + gap - ARROW_HEAD_SIZE
                _line(svg, ax1, ay1, ax2, ay2)
            else:
                ax1 = bx + box_w
                ay1 = by + box_h // 2
                ax2 = bx + box_w + gap - ARROW_HEAD_SIZE
                ay2 = ay1
                _line(svg, ax1, ay1, ax2, ay2)

    return SVGDocument(svg)


def architecture_diagram(
    layers: list[dict],
    connections: list[tuple],
    title: Optional[str] = None,
) -> SVGDocument:
    """Generate a multi-layer architecture diagram with labeled connections.

    Parameters
    ----------
    layers : list[dict]
        Each dict has ``name`` (layer label), ``nodes`` (list[str] of box
        labels), and optionally ``color`` (key of ``COLORS``, default
        ``"blue"``).
    connections : list[tuple]
        ``(source_node_label, target_node_label)`` pairs. Connections are
        drawn as lines between matching node labels across layers.
    title : str, optional
        Optional heading above the diagram.
    """
    node_w, node_h = 140, 50
    layer_gap = 80
    node_gap = 30
    label_w = 120
    title_h = 50 if title else 0

    # Pre-compute positions for each node
    max_nodes = max(len(layer["nodes"]) for layer in layers)
    content_w = max_nodes * (node_w + node_gap) - node_gap
    canvas_w = content_w + label_w + 3 * PADDING
    canvas_h = len(layers) * (node_h + layer_gap) - layer_gap + 2 * PADDING + title_h

    svg = _make_svg(canvas_w, canvas_h)
    _add_style(svg)
    defs = ET.SubElement(svg, "defs")
    _add_drop_shadow(defs)
    _add_arrow_marker(defs)

    if title:
        _add_title(svg, title, canvas_w // 2, PADDING + 16)

    node_positions: dict[str, tuple[int, int]] = {}

    for li, layer in enumerate(layers):
        c = _color(layer.get("color", "blue"))
        ly = PADDING + title_h + li * (node_h + layer_gap)
        nodes = layer["nodes"]
        total_w = len(nodes) * (node_w + node_gap) - node_gap
        start_x = label_w + PADDING + (content_w - total_w) // 2

        # Layer label
        _text(svg, PADDING + label_w // 2, ly + node_h // 2, layer["name"],
              font_size=15, fill=c["text"], font_weight="700", cls="heading")

        # Layer background band
        band = ET.SubElement(svg, "rect")
        band.set("x", str(label_w + PADDING - 10))
        band.set("y", str(ly - 8))
        band.set("width", str(content_w + 20))
        band.set("height", str(node_h + 16))
        band.set("rx", "8")
        band.set("ry", "8")
        band.set("fill", c["bg"])
        band.set("stroke", c["border"])
        band.set("stroke-width", "1")
        band.set("stroke-dasharray", "4,4")
        band.set("opacity", "0.5")

        for ni, node_label in enumerate(nodes):
            nx = start_x + ni * (node_w + node_gap)
            ny = ly
            _rounded_rect(svg, nx, ny, node_w, node_h, "#ffffff", c["border"],
                          stroke_width=2, filter_id="shadow")
            _text(svg, nx + node_w // 2, ny + node_h // 2, node_label,
                  font_size=13, fill=c["text"], font_weight="600")
            node_positions[node_label] = (nx + node_w // 2, ny + node_h // 2)

    # Draw connections
    for src, tgt in connections:
        if src in node_positions and tgt in node_positions:
            sx, sy = node_positions[src]
            tx, ty = node_positions[tgt]
            # Offset to connect from bottom of source to top of target
            sy_start = sy + node_h // 2
            ty_end = ty - node_h // 2
            _line(svg, sx, sy_start, tx, ty_end - ARROW_HEAD_SIZE)

    return SVGDocument(svg)


def comparison_graphic(
    left: dict,
    right: dict,
    title: Optional[str] = None,
) -> SVGDocument:
    """Generate a side-by-side comparison graphic.

    Parameters
    ----------
    left : dict
        ``{"title": str, "items": list[str]}``.  Rendered with a red/rose
        header (the *before* / *option A* side).
    right : dict
        Same structure as *left*. Rendered with a green/mint header (the
        *after* / *option B* side).
    title : str, optional
        Optional heading above the comparison.
    """
    col_w = 260
    header_h = 40
    item_h = 30
    gap = 40
    title_h = 50 if title else 0

    max_items = max(len(left.get("items", [])), len(right.get("items", [])))
    body_h = max(max_items * item_h + 20, 80)
    card_h = header_h + body_h

    canvas_w = 2 * col_w + gap + 2 * PADDING
    canvas_h = card_h + 2 * PADDING + title_h

    svg = _make_svg(canvas_w, canvas_h)
    _add_style(svg)
    defs = ET.SubElement(svg, "defs")
    _add_drop_shadow(defs)

    if title:
        _add_title(svg, title, canvas_w // 2, PADDING + 16)

    # Clip paths for rounded header corners
    for side_idx, (side_data, base_x, header_color) in enumerate([
        (left, PADDING, "#ef5350"),
        (right, PADDING + col_w + gap, "#4caf50"),
    ]):
        x = base_x
        y = PADDING + title_h

        # Card background
        _rounded_rect(svg, x, y, col_w, card_h, "#ffffff", "#e0e0e0",
                      stroke_width=1, filter_id="shadow")

        # Header
        header = ET.SubElement(svg, "rect")
        header.set("x", str(x))
        header.set("y", str(y))
        header.set("width", str(col_w))
        header.set("height", str(header_h))
        header.set("rx", str(NODE_RADIUS))
        header.set("ry", str(NODE_RADIUS))
        header.set("fill", header_color)

        # Square off bottom corners of header
        sq = ET.SubElement(svg, "rect")
        sq.set("x", str(x))
        sq.set("y", str(y + header_h - NODE_RADIUS))
        sq.set("width", str(col_w))
        sq.set("height", str(NODE_RADIUS))
        sq.set("fill", header_color)

        _text(svg, x + col_w // 2, y + header_h // 2, side_data.get("title", ""),
              font_size=16, fill="#ffffff", font_weight="700")

        # Items
        for ii, item in enumerate(side_data.get("items", [])):
            iy = y + header_h + 15 + ii * item_h
            # Bullet
            bullet = ET.SubElement(svg, "circle")
            bullet.set("cx", str(x + 25))
            bullet.set("cy", str(iy + 5))
            bullet.set("r", "4")
            bullet.set("fill", header_color)
            _text(svg, x + 40, iy + 5, item,
                  font_size=13, fill="#424242", anchor="start")

    return SVGDocument(svg)


def concept_map(
    nodes: list[dict],
    edges: list[dict],
    title: Optional[str] = None,
) -> SVGDocument:
    """Generate a concept relationship map.

    Parameters
    ----------
    nodes : list[dict]
        Each dict: ``{"id": str, "label": str, "color": str}``.
        An optional ``"x"`` and ``"y"`` can be provided; otherwise nodes are
        auto-positioned in a grid.
    edges : list[dict]
        Each dict: ``{"from": str, "to": str, "label": str (optional)}``.
    title : str, optional
        Optional heading above the map.
    """
    node_w, node_h = 140, 50
    title_h = 50 if title else 0
    n = len(nodes)

    # Auto-position: arrange in rows of up to 3
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    h_gap = 80
    v_gap = 80

    canvas_w = cols * (node_w + h_gap) - h_gap + 2 * PADDING
    canvas_h = rows * (node_h + v_gap) - v_gap + 2 * PADDING + title_h

    svg = _make_svg(max(canvas_w, 300), max(canvas_h, 200))
    _add_style(svg)
    defs = ET.SubElement(svg, "defs")
    _add_drop_shadow(defs)
    _add_arrow_marker(defs)

    if title:
        _add_title(svg, title, max(canvas_w, 300) // 2, PADDING + 16)

    node_centers: dict[str, tuple[int, int]] = {}

    for idx, node in enumerate(nodes):
        c = _color(node.get("color", "gray"))
        col = idx % cols
        row = idx // cols
        if "x" in node and "y" in node:
            nx = int(node["x"])
            ny = int(node["y"])
        else:
            nx = PADDING + col * (node_w + h_gap)
            ny = PADDING + title_h + row * (node_h + v_gap)

        _rounded_rect(svg, nx, ny, node_w, node_h, c["bg"], c["border"],
                      filter_id="shadow")
        _text(svg, nx + node_w // 2, ny + node_h // 2, node["label"],
              font_size=13, fill=c["text"], font_weight="600")
        node_centers[node["id"]] = (nx + node_w // 2, ny + node_h // 2)

    # Draw edges
    for edge in edges:
        src_id = edge["from"]
        tgt_id = edge["to"]
        if src_id in node_centers and tgt_id in node_centers:
            sx, sy = node_centers[src_id]
            tx, ty = node_centers[tgt_id]

            # Shorten the line so arrow doesn't overlap the node
            dx = tx - sx
            dy = ty - sy
            dist = max((dx ** 2 + dy ** 2) ** 0.5, 1)
            shorten = node_w // 2 + ARROW_HEAD_SIZE
            ratio_start = (node_w // 2) / dist
            ratio_end = shorten / dist
            lx1 = int(sx + dx * ratio_start)
            ly1 = int(sy + dy * ratio_start)
            lx2 = int(sx + dx * (1 - ratio_end / dist))
            ly2 = int(sy + dy * (1 - ratio_end / dist))

            # Simpler: just offset target by arrow head
            end_offset = ARROW_HEAD_SIZE / dist if dist > 0 else 0
            lx2 = int(tx - dx * end_offset)
            ly2 = int(ty - dy * end_offset)

            _line(svg, lx1, ly1, lx2, ly2)

            # Edge label at midpoint
            label = edge.get("label", "")
            if label:
                mx = (sx + tx) // 2
                my = (sy + ty) // 2 - 8
                # Label background
                lbl_bg = ET.SubElement(svg, "rect")
                lbl_bg.set("x", str(mx - 35))
                lbl_bg.set("y", str(my - 10))
                lbl_bg.set("width", "70")
                lbl_bg.set("height", "20")
                lbl_bg.set("rx", "10")
                lbl_bg.set("ry", "10")
                lbl_bg.set("fill", "#ffffff")
                lbl_bg.set("stroke", "#e0e0e0")
                lbl_bg.set("stroke-width", "1")
                _text(svg, mx, my, label,
                      font_size=11, fill="#757575")

    return SVGDocument(svg)


def timeline(
    events: list[dict],
    title: Optional[str] = None,
    orientation: str = "horizontal",
) -> SVGDocument:
    """Generate a timeline with labeled milestones.

    Parameters
    ----------
    events : list[dict]
        Each dict: ``{"label": str, "detail": str, "color": str (optional)}``.
    title : str, optional
        Optional heading above the timeline.
    orientation : str
        ``"horizontal"`` (default) or ``"vertical"``.
    """
    n = len(events)
    dot_r = 10
    title_h = 50 if title else 0

    if orientation == "vertical":
        segment = 100
        canvas_w = 400
        canvas_h = n * segment + 2 * PADDING + title_h
    else:
        segment = 180
        canvas_w = n * segment + 2 * PADDING
        canvas_h = 180 + title_h

    svg = _make_svg(canvas_w, canvas_h)
    _add_style(svg)
    defs = ET.SubElement(svg, "defs")
    _add_drop_shadow(defs)

    if title:
        _add_title(svg, title, canvas_w // 2, PADDING + 16)

    if orientation == "vertical":
        line_x = PADDING + 30
        # Main vertical line
        vline = ET.SubElement(svg, "line")
        vline.set("x1", str(line_x))
        vline.set("y1", str(PADDING + title_h))
        vline.set("x2", str(line_x))
        vline.set("y2", str(PADDING + title_h + (n - 1) * segment))
        vline.set("stroke", "#e0e0e0")
        vline.set("stroke-width", "3")

        for i, event in enumerate(events):
            c = _color(event.get("color", "gray"))
            cy = PADDING + title_h + i * segment

            # Dot
            dot = ET.SubElement(svg, "circle")
            dot.set("cx", str(line_x))
            dot.set("cy", str(cy))
            dot.set("r", str(dot_r))
            dot.set("fill", c["border"])
            dot.set("stroke", "#ffffff")
            dot.set("stroke-width", "3")

            # Label
            _text(svg, line_x + 30, cy - 8, event["label"],
                  font_size=14, fill=c["text"], anchor="start", font_weight="700")
            # Detail
            _text(svg, line_x + 30, cy + 12, event.get("detail", ""),
                  font_size=12, fill="#757575", anchor="start")
    else:
        line_y = PADDING + title_h + 60
        # Main horizontal line
        hline = ET.SubElement(svg, "line")
        hline.set("x1", str(PADDING))
        hline.set("y1", str(line_y))
        hline.set("x2", str(PADDING + (n - 1) * segment))
        hline.set("y2", str(line_y))
        hline.set("stroke", "#e0e0e0")
        hline.set("stroke-width", "3")

        for i, event in enumerate(events):
            c = _color(event.get("color", "gray"))
            cx = PADDING + i * segment

            # Dot
            dot = ET.SubElement(svg, "circle")
            dot.set("cx", str(cx))
            dot.set("cy", str(line_y))
            dot.set("r", str(dot_r))
            dot.set("fill", c["border"])
            dot.set("stroke", "#ffffff")
            dot.set("stroke-width", "3")

            # Label above
            _text(svg, cx, line_y - 25, event["label"],
                  font_size=14, fill=c["text"], font_weight="700")
            # Detail below
            detail = event.get("detail", "")
            if detail:
                # Wrap long detail text
                words = detail.split()
                line1 = ""
                line2 = ""
                for w in words:
                    test = line1 + " " + w if line1 else w
                    if len(test) <= 18:
                        line1 = test
                    else:
                        line2 += (" " + w if line2 else w)
                _text(svg, cx, line_y + 25, line1,
                      font_size=11, fill="#757575")
                if line2:
                    _text(svg, cx, line_y + 40, line2,
                          font_size=11, fill="#757575")

    return SVGDocument(svg)


def annotated_code(
    code: str,
    language: str,
    annotations: list[dict],
    filename: Optional[str] = None,
) -> SVGDocument:
    """Generate a code block with macOS window chrome and annotation callouts.

    Parameters
    ----------
    code : str
        The source code to display.
    language : str
        Language name (shown in the header if no *filename* given).
    annotations : list[dict]
        Each dict: ``{"line": int (1-based), "text": str, "side": str}``.
        ``side`` is ``"right"`` (default), ``"left"``, or ``"top"``.
    filename : str, optional
        Displayed in the window title bar. Falls back to *language*.
    """
    lines = code.split("\n")
    line_h = 22
    header_h = 36
    code_padding = 16
    annotation_w = 160
    annotation_margin = 20

    code_w = max(len(line) * 9 + 2 * code_padding for line in lines) if lines else 200
    code_w = max(code_w, 300)
    code_h = len(lines) * line_h + 2 * code_padding

    window_w = code_w + annotation_w + annotation_margin
    window_h = header_h + code_h + 10

    canvas_w = window_w + 2 * PADDING
    canvas_h = window_h + 2 * PADDING

    svg = _make_svg(canvas_w, canvas_h)
    _add_style(svg)
    defs = ET.SubElement(svg, "defs")
    _add_drop_shadow(defs)

    wx = PADDING
    wy = PADDING

    # Window background with shadow
    _rounded_rect(svg, wx, wy, window_w, window_h, "#1e1e2e", "#2d2d44",
                  stroke_width=1, rx=NODE_RADIUS, filter_id="shadow")

    # Header bar
    header_bg = ET.SubElement(svg, "rect")
    header_bg.set("x", str(wx))
    header_bg.set("y", str(wy))
    header_bg.set("width", str(window_w))
    header_bg.set("height", str(header_h))
    header_bg.set("rx", str(NODE_RADIUS))
    header_bg.set("ry", str(NODE_RADIUS))
    header_bg.set("fill", "#2d2d44")

    # Square off bottom corners of header
    sq = ET.SubElement(svg, "rect")
    sq.set("x", str(wx))
    sq.set("y", str(wy + header_h - NODE_RADIUS))
    sq.set("width", str(window_w))
    sq.set("height", str(NODE_RADIUS))
    sq.set("fill", "#2d2d44")

    # Traffic light dots
    dot_colors = ["#ff5f57", "#ffbd2e", "#28ca41"]
    for di, dc in enumerate(dot_colors):
        dot = ET.SubElement(svg, "circle")
        dot.set("cx", str(wx + 20 + di * 20))
        dot.set("cy", str(wy + header_h // 2))
        dot.set("r", "6")
        dot.set("fill", dc)

    # Filename
    fname = filename or language
    _text(svg, wx + window_w // 2, wy + header_h // 2, fname,
          font_size=12, fill="#a0a0b0", cls="code")

    # Code lines
    code_x = wx + code_padding
    code_y_start = wy + header_h + code_padding

    for li, line_text in enumerate(lines):
        ly = code_y_start + li * line_h
        # Line number
        _text(svg, code_x + 20, ly + line_h // 2, str(li + 1),
              font_size=12, fill="#555566", anchor="end", cls="code")
        # Code text
        _text(svg, code_x + 30, ly + line_h // 2, line_text,
              font_size=13, fill="#cdd6f4", anchor="start", cls="code")

    # Annotations
    for ann in annotations:
        line_num = ann.get("line", 1)
        ann_text = ann.get("text", "")
        side = ann.get("side", "right")

        # Annotation Y position: aligned with the target line
        ay = code_y_start + (line_num - 1) * line_h + line_h // 2

        if side == "right":
            ax = wx + code_w + annotation_margin
        elif side == "left":
            ax = wx - annotation_w - annotation_margin + PADDING
        else:  # top
            ax = wx + code_w // 2
            ay = wy - 20

        # Pill background
        pill_w = max(len(ann_text) * 7 + 20, 80)
        pill_h = 26
        pill_x = ax
        pill_y = ay - pill_h // 2

        pill = ET.SubElement(svg, "rect")
        pill.set("x", str(pill_x))
        pill.set("y", str(pill_y))
        pill.set("width", str(pill_w))
        pill.set("height", str(pill_h))
        pill.set("rx", "13")
        pill.set("ry", "13")
        pill.set("fill", "#ffffff")
        pill.set("stroke", "#e0e0e0")
        pill.set("stroke-width", "1")
        pill.set("filter", "url(#shadow)")

        _text(svg, pill_x + pill_w // 2, ay, ann_text,
              font_size=11, fill="#1a1a2e", font_weight="600")

        # Connecting line from annotation to code line
        if side == "right":
            _line(svg, wx + code_w - 10, ay, pill_x, ay,
                  stroke="#e0e0e0", marker_end=None)
        elif side == "left":
            _line(svg, pill_x + pill_w, ay, wx + code_padding + 30, ay,
                  stroke="#e0e0e0", marker_end=None)

    return SVGDocument(svg)
