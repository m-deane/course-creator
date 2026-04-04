"""Course graphics package — SVG diagrams and plot theming."""

from .diagram_generator import (
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
from .plot_theme import apply_plot_theme

__all__ = [
    "process_flow",
    "architecture_diagram",
    "comparison_graphic",
    "concept_map",
    "timeline",
    "annotated_code",
    "SVGDocument",
    "COLORS",
    "FONTS",
    "apply_plot_theme",
]
