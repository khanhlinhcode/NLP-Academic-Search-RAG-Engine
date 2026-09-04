"""Regression tests for the persistent Streamlit visual theme."""

from __future__ import annotations

import tomllib
from pathlib import Path

from nlp_academic_search.ui.styles import stylesheet


def test_stylesheet_has_one_stable_theme_marker():
    css = stylesheet()

    assert css.count('<style id="academic-search-theme">') == 1
    assert "--bg-main: #0b0f19" in css


def test_streamlit_native_theme_is_a_dark_fallback():
    config_path = Path(__file__).parents[2] / ".streamlit" / "config.toml"
    with config_path.open("rb") as config_file:
        theme = tomllib.load(config_file)["theme"]

    assert theme == {
        "base": "dark",
        "primaryColor": "#6366f1",
        "backgroundColor": "#0b0f19",
        "secondaryBackgroundColor": "#111827",
        "textColor": "#f8fafc",
        "font": "sans serif",
    }
