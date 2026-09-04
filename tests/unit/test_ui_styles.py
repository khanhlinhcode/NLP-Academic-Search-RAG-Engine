"""Regression tests for the persistent Streamlit visual theme."""

from __future__ import annotations

import tomllib
from pathlib import Path

from nlp_academic_search.ui.styles import stylesheet


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
