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
