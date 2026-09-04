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
