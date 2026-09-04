"""Regression tests for the persistent Streamlit visual theme."""

from __future__ import annotations

import re
import tomllib
from html.parser import HTMLParser
from pathlib import Path

from nlp_academic_search.ui.styles import stylesheet


class _SemanticTextExtractor(HTMLParser):
    """Approximate copied/assistive text while respecting aria-hidden content."""

    def __init__(self) -> None:
        super().__init__()
        self._hidden_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._hidden_depth or dict(attrs).get("aria-hidden") == "true":
            self._hidden_depth += 1

    def handle_endtag(self, _tag: str) -> None:
        if self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._hidden_depth:
            self._parts.append(data)

    def text(self) -> str:
        return re.sub(r"\s+", " ", "".join(self._parts)).strip()


def _semantic_text(markup: str) -> str:
    parser = _SemanticTextExtractor()
    parser.feed(markup)
    return parser.text()


def _source_markup(monkeypatch, source: dict, *, cited: bool) -> str:
    from nlp_academic_search.ui import app as ui_app

    rendered: list[str] = []
    monkeypatch.setattr(ui_app.st, "markdown", lambda body, **_kwargs: rendered.append(body))
    citation_validation = {"cited_indices": [source["index"]] if cited else []}
    ui_app.render_sources([source], {"citation_validation": citation_validation})
    return next(item for item in rendered if 'class="source-item ' in item)


def _source_heading_markup(source_markup: str) -> str:
    return source_markup.split('<div class="source-heading">', maxsplit=1)[1].split(
        "</div>", maxsplit=1
    )[0]


def test_stylesheet_has_one_stable_theme_marker():
    css = stylesheet()

    assert css.count('<style id="academic-search-theme">') == 1
    assert "--canvas: #f3efe6" in css
    assert "--oxblood: #762f35" in css


def test_stylesheet_embeds_local_fonts_without_external_font_requests():
    css = stylesheet()

    assert "@font-face {" in css
    assert 'font-family: "IM Fell French Canon"' in css
    assert 'font-family: "IBM Plex Mono"' in css
    assert css.count("data:font/ttf;base64,") == 2
    assert "fonts.googleapis.com" not in css
    assert "@import url(" not in css


def test_stylesheet_avoids_generic_ai_visual_effects():
    css = stylesheet().casefold()

    for prohibited in (
        "linear-gradient",
        "radial-gradient",
        "backdrop-filter",
        "box-shadow",
        "primary-glow",
        "border-glow",
    ):
        assert prohibited not in css


def test_desktop_masthead_stays_on_one_line_and_wraps_responsively():
    css = stylesheet()

    masthead_rules = css.split("/* Masthead */", maxsplit=1)[1].split(
        "/* Editorial mode navigation */", maxsplit=1
    )[0]
    responsive_rules = css.split("/* Responsive composition */", maxsplit=1)[1]

    assert "max-width: none;" in masthead_rules
    assert "white-space: nowrap;" in masthead_rules
    assert ".masthead h1 { white-space: normal; text-wrap: balance; }" in responsive_rules


def test_mode_switch_and_form_controls_use_soft_bounded_surfaces():
    css = stylesheet()

    mode_rules = css.split("/* Editorial mode navigation */", maxsplit=1)[1].split(
        "/* Search and Ask instruments */", maxsplit=1
    )[0]

    assert "--radius-control: 8px;" in css
    assert "--radius-panel: 12px;" in css
    assert '[data-testid="stRadioGroup"]' in mode_rules
    assert "border-bottom" not in mode_rules
    assert "border-radius: 10px;" in mode_rules
    assert "display: none !important;" in mode_rules
    assert '[data-testid="stFormSubmitButton"] button p' in css


def test_buttons_center_their_markdown_content_without_layout_offsets():
    css = stylesheet()
    button_rules = css.split('[data-testid="stTextInput"] input', maxsplit=1)[1].split(
        '[data-testid="stToggle"]', maxsplit=1
    )[0]

    assert "display: inline-flex !important;" in button_rules
    assert "align-items: center !important;" in button_rules
    assert "justify-content: center !important;" in button_rules
    assert "min-height: 44px !important;" in button_rules
    assert "text-align: center !important;" in button_rules
    assert 'button [data-testid="stMarkdownContainer"] p' in button_rules
    assert "margin: 0 !important;" in button_rules
    assert "line-height: 1.25 !important;" in button_rules


def test_mode_options_are_equal_accessible_centered_touch_targets():
    css = stylesheet()
    mode_rules = css.split("/* Editorial mode navigation */", maxsplit=1)[1].split(
        "/* Search and Ask instruments */", maxsplit=1
    )[0]

    assert "display: flex !important;" in mode_rules
    assert "align-items: center !important;" in mode_rules
    assert "justify-content: center !important;" in mode_rules
    assert "flex: 1 1 0;" in mode_rules
    assert "min-height: 44px;" in mode_rules
    assert "margin: 0 !important;" in mode_rules
    assert "border-bottom" not in mode_rules
    assert '[data-testid="stRadioOption"] input {' not in mode_rules


def test_form_columns_use_structural_bottom_alignment_without_blank_spacers():
    app_path = Path(__file__).parents[2] / "src" / "nlp_academic_search" / "ui" / "app.py"
    app_source = app_path.read_text(encoding="utf-8")

    assert 'st.columns([5, 1.15], vertical_alignment="bottom")' in app_source
    assert 'st.columns([4, 1], vertical_alignment="bottom")' in app_source
    assert 'st.columns([1, 2], vertical_alignment="bottom")' in app_source
    assert 'st.write("")' not in app_source


def test_evidence_rows_wrap_titles_and_preserve_text_separators():
    css = stylesheet()
    app_path = Path(__file__).parents[2] / "src" / "nlp_academic_search" / "ui" / "app.py"
    app_source = app_path.read_text(encoding="utf-8")

    source_rules = css.split(".source-heading {", maxsplit=1)[1].split(
        ".citation-badge {", maxsplit=1
    )[0]
    assert "grid-template-columns: 2rem minmax(0, 1fr);" in source_rules
    assert ".source-title-row {" in source_rules
    assert "display: block;" in source_rules
    assert "flex: 1 1 12rem;" not in source_rules
    assert "white-space: nowrap;" in source_rules
    assert '<span class="source-title-row">' in app_source
    assert '<div class="source-title-row">' not in app_source
    assert "source_markup = (" in app_source
    assert "st.markdown(source_markup, unsafe_allow_html=True)" in app_source
    assert 'class="cited-label"' in app_source
    assert 'class="cited-separator"> · </span>' in app_source
    assert 'class="cited-status">Cited</span>' in app_source
    assert 'aria-label="Cited source"' not in app_source
    assert 'aria-hidden="true"> · </span>' not in app_source
    assert "[{safe(source_index)}]&ensp;" in app_source
    assert '<span class="status-dot ready"></span> Cited' not in app_source


def test_ask_failure_copy_distinguishes_repair_failures():
    app_path = Path(__file__).parents[2] / "src" / "nlp_academic_search" / "ui" / "app.py"
    app_source = app_path.read_text(encoding="utf-8")

    assert '"repair_disabled":' in app_source
    assert '"repair_provider_error":' in app_source
    assert '"final_validation_failed":' in app_source


def test_evidence_item_is_emitted_as_flat_html(monkeypatch):
    source_markup = _source_markup(
        monkeypatch,
        {
            "index": 1,
            "title": "A long evidence title",
            "authors": ["Researcher"],
            "categories": ["cs.IR"],
            "year": 2026,
            "source": "fixture",
        },
        cited=False,
    )

    assert "\n" not in source_markup
    assert '<span class="source-title-row">' in source_markup
    assert "</span></div>" in source_markup


def test_cited_evidence_heading_preserves_semantic_separator(monkeypatch):
    source_markup = _source_markup(
        monkeypatch,
        {
            "index": 1,
            "title": "Example paper",
            "authors": ["Researcher"],
            "categories": ["cs.IR"],
            "year": 2026,
            "source": "fixture",
        },
        cited=True,
    )

    heading_markup = _source_heading_markup(source_markup)
    assert _semantic_text(heading_markup) == "[1] Example paper · Cited"
    assert "Example paperCited" not in _semantic_text(heading_markup)
    assert heading_markup.count("Cited") == 1
    assert "aria-hidden" not in heading_markup
    assert "aria-label" not in heading_markup


def test_uncited_evidence_heading_has_no_dangling_separator(monkeypatch):
    source_markup = _source_markup(
        monkeypatch,
        {
            "index": 2,
            "title": "Another paper",
            "authors": ["Researcher"],
            "categories": ["cs.IR"],
            "year": 2026,
            "source": "fixture",
        },
        cited=False,
    )

    heading_text = _semantic_text(_source_heading_markup(source_markup))
    assert heading_text == "[2] Another paper"
    assert not heading_text.endswith("·")


def test_evidence_heading_escapes_untrusted_title_and_supports_long_copy(monkeypatch):
    long_title = (
        'Paper <script>alert("x")</script> & evaluation — '
        + "multilingual retrieval evidence " * 12
    ).strip()
    source_markup = _source_markup(
        monkeypatch,
        {
            "index": 3,
            "title": long_title,
            "authors": ["Researcher"],
            "categories": ["cs.IR"],
            "year": 2026,
            "source": "fixture",
        },
        cited=True,
    )

    heading_markup = _source_heading_markup(source_markup)
    heading_text = _semantic_text(heading_markup)
    assert "<script>" not in heading_markup
    assert "&lt;script&gt;" in heading_markup
    assert "&amp; evaluation" in heading_markup
    assert heading_text == f"[3] {long_title} · Cited"
    assert heading_markup.count("Cited") == 1


def test_streamlit_native_theme_is_a_warm_light_fallback():
    config_path = Path(__file__).parents[2] / ".streamlit" / "config.toml"
    with config_path.open("rb") as config_file:
        theme = tomllib.load(config_file)["theme"]

    assert theme == {
        "base": "light",
        "primaryColor": "#762f35",
        "backgroundColor": "#f3efe6",
        "secondaryBackgroundColor": "#eae4d9",
        "textColor": "#1d1d1a",
        "font": "sans serif",
    }
