"""Tests for the SVG diagram generator module."""

import os
import tempfile

from resources.graphics.diagram_generator import (
    COLORS,
    FONTS,
    SVGDocument,
    annotated_code,
    architecture_diagram,
    comparison_graphic,
    concept_map,
    process_flow,
    timeline,
)


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------


def test_colors_has_all_keys():
    expected = {"mint", "amber", "blue", "lavender", "rose", "gray"}
    assert set(COLORS.keys()) == expected


def test_each_color_has_bg_border_text():
    for name, palette in COLORS.items():
        assert "bg" in palette, f"{name} missing 'bg'"
        assert "border" in palette, f"{name} missing 'border'"
        assert "text" in palette, f"{name} missing 'text'"


def test_fonts_has_all_keys():
    expected = {"heading", "body", "code"}
    assert set(FONTS.keys()) == expected


# ---------------------------------------------------------------------------
# SVGDocument
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# process_flow
# ---------------------------------------------------------------------------


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


def test_process_flow_single_step():
    svg = process_flow(["Only Step"])
    content = svg.to_string()
    assert "Only Step" in content
    # Single step should have no arrows (line elements)
    assert content.count("<line") == 0 or "Only Step" in content


def test_process_flow_many_steps():
    steps = [f"Step {i}" for i in range(8)]
    svg = process_flow(steps)
    content = svg.to_string()
    for s in steps:
        assert s in content


# ---------------------------------------------------------------------------
# comparison_graphic
# ---------------------------------------------------------------------------


def test_comparison_graphic():
    svg = comparison_graphic(
        left={"title": "Before", "items": ["Slow", "Manual"]},
        right={"title": "After", "items": ["Fast", "Automated"]},
        title="Improvement",
    )
    content = svg.to_string()
    assert "Before" in content
    assert "After" in content
    assert "Improvement" in content


def test_comparison_graphic_no_title():
    svg = comparison_graphic(
        left={"title": "Old", "items": ["A"]},
        right={"title": "New", "items": ["B"]},
    )
    content = svg.to_string()
    assert "Old" in content
    assert "New" in content


def test_comparison_graphic_empty_items():
    svg = comparison_graphic(
        left={"title": "Left", "items": []},
        right={"title": "Right", "items": []},
    )
    content = svg.to_string()
    assert "Left" in content
    assert "Right" in content


# ---------------------------------------------------------------------------
# concept_map
# ---------------------------------------------------------------------------


def test_concept_map():
    svg = concept_map(
        nodes=[
            {"id": "a", "label": "Node A", "color": "mint"},
            {"id": "b", "label": "Node B", "color": "blue"},
        ],
        edges=[{"from": "a", "to": "b", "label": "connects"}],
        title="Test Map",
    )
    content = svg.to_string()
    assert "Node A" in content
    assert "Node B" in content


def test_concept_map_no_edges():
    svg = concept_map(
        nodes=[
            {"id": "x", "label": "Solo", "color": "lavender"},
        ],
        edges=[],
    )
    content = svg.to_string()
    assert "Solo" in content


def test_concept_map_with_positions():
    svg = concept_map(
        nodes=[
            {"id": "a", "label": "A", "color": "mint", "x": 50, "y": 50},
            {"id": "b", "label": "B", "color": "blue", "x": 250, "y": 50},
        ],
        edges=[{"from": "a", "to": "b"}],
    )
    content = svg.to_string()
    assert "A" in content
    assert "B" in content


def test_concept_map_edge_without_label():
    svg = concept_map(
        nodes=[
            {"id": "a", "label": "Src", "color": "amber"},
            {"id": "b", "label": "Dst", "color": "rose"},
        ],
        edges=[{"from": "a", "to": "b"}],
    )
    content = svg.to_string()
    assert "Src" in content


# ---------------------------------------------------------------------------
# timeline
# ---------------------------------------------------------------------------


def test_timeline():
    svg = timeline(
        events=[
            {"label": "Start", "detail": "Begin here", "color": "mint"},
            {"label": "Middle", "detail": "Halfway", "color": "amber"},
            {"label": "End", "detail": "Done", "color": "blue"},
        ],
        title="Project Timeline",
    )
    content = svg.to_string()
    assert "Start" in content
    assert "Project Timeline" in content


def test_timeline_vertical():
    svg = timeline(
        events=[
            {"label": "Phase 1", "detail": "First", "color": "mint"},
            {"label": "Phase 2", "detail": "Second", "color": "blue"},
        ],
        orientation="vertical",
    )
    content = svg.to_string()
    assert "Phase 1" in content
    assert "Phase 2" in content


def test_timeline_no_detail():
    svg = timeline(
        events=[{"label": "Milestone"}],
    )
    content = svg.to_string()
    assert "Milestone" in content


# ---------------------------------------------------------------------------
# annotated_code
# ---------------------------------------------------------------------------


def test_annotated_code():
    svg = annotated_code(
        code="def foo():\n    return 42",
        language="python",
        annotations=[{"line": 1, "text": "Function definition", "side": "right"}],
        filename="example.py",
    )
    content = svg.to_string()
    assert "def foo" in content
    assert "Function definition" in content
    assert "example.py" in content


def test_annotated_code_no_filename():
    svg = annotated_code(
        code="print('hello')",
        language="python",
        annotations=[],
    )
    content = svg.to_string()
    assert "python" in content


def test_annotated_code_multiple_annotations():
    svg = annotated_code(
        code="import os\npath = os.getcwd()\nprint(path)",
        language="python",
        annotations=[
            {"line": 1, "text": "Import", "side": "right"},
            {"line": 2, "text": "Assignment", "side": "right"},
            {"line": 3, "text": "Output", "side": "right"},
        ],
        filename="demo.py",
    )
    content = svg.to_string()
    assert "Import" in content
    assert "Assignment" in content
    assert "Output" in content


def test_annotated_code_left_annotation():
    svg = annotated_code(
        code="x = 1",
        language="python",
        annotations=[{"line": 1, "text": "Variable", "side": "left"}],
    )
    content = svg.to_string()
    assert "Variable" in content


# ---------------------------------------------------------------------------
# architecture_diagram
# ---------------------------------------------------------------------------


def test_architecture_diagram():
    svg = architecture_diagram(
        layers=[
            {"name": "Frontend", "nodes": ["UI", "API Client"], "color": "blue"},
            {"name": "Backend", "nodes": ["Server", "Database"], "color": "mint"},
        ],
        connections=[("UI", "Server"), ("Server", "Database")],
        title="System Architecture",
    )
    content = svg.to_string()
    assert "Frontend" in content
    assert "System Architecture" in content


def test_architecture_diagram_no_connections():
    svg = architecture_diagram(
        layers=[
            {"name": "Layer", "nodes": ["A", "B"], "color": "gray"},
        ],
        connections=[],
    )
    content = svg.to_string()
    assert "Layer" in content


def test_architecture_diagram_default_color():
    svg = architecture_diagram(
        layers=[
            {"name": "Tier", "nodes": ["Node"]},
        ],
        connections=[],
    )
    content = svg.to_string()
    assert "Tier" in content


# ---------------------------------------------------------------------------
# Independence
# ---------------------------------------------------------------------------


def test_each_function_returns_independent_instance():
    a = process_flow(["AlphaStep"])
    b = process_flow(["BetaStep"])
    assert "AlphaStep" in a.to_string()
    assert "BetaStep" in b.to_string()
    assert "BetaStep" not in a.to_string()


def test_different_generators_independent():
    flow = process_flow(["FlowStep"])
    comp = comparison_graphic(
        left={"title": "L", "items": ["CompItem"]},
        right={"title": "R", "items": []},
    )
    assert "FlowStep" in flow.to_string()
    assert "CompItem" in comp.to_string()
    assert "CompItem" not in flow.to_string()
    assert "FlowStep" not in comp.to_string()
