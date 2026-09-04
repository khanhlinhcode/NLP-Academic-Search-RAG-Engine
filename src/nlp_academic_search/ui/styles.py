"""Premium editorial visual system for the Streamlit research workspace."""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

_FONT_DIRECTORY = Path(__file__).with_name("assets") / "fonts"


@lru_cache(maxsize=2)
def _embedded_font(filename: str) -> str:
    """Return a bundled font as a data URL payload for Streamlit Cloud."""
    return base64.b64encode((_FONT_DIRECTORY / filename).read_bytes()).decode("ascii")


@lru_cache(maxsize=1)
def stylesheet() -> str:
    """Return the self-contained Premium Scholarly Editorial stylesheet."""
    css = """
    <style id="academic-search-theme">
    @font-face {
        font-family: "IM Fell French Canon";
        src: url("data:font/ttf;base64,__DISPLAY_FONT__") format("truetype");
        font-display: swap;
        font-style: normal;
        font-weight: 400;
    }

    @font-face {
        font-family: "IBM Plex Mono";
        src: url("data:font/ttf;base64,__MONO_FONT__") format("truetype");
        font-display: swap;
        font-style: normal;
        font-weight: 400;
    }

    :root {
        color-scheme: light;
        --canvas: #f3efe6;
        --surface: #fbf9f4;
        --surface-control: #fffdf8;
        --surface-muted: #eae4d9;
        --ink: #1d1d1a;
        --ink-secondary: #625e56;
        --ink-muted: #6f685e;
        --rule: #d5cec0;
        --rule-strong: #b9b0a1;
        --oxblood: #762f35;
        --oxblood-hover: #5d2228;
        --oxblood-soft: #efe1df;
        --link: #315c68;
        --success: #38624d;
        --success-soft: #e3ece5;
        --warning: #76571f;
        --warning-soft: #f3ead5;
        --danger: #963e36;
        --danger-soft: #f3dfdc;
        --radius-control: 8px;
        --radius-panel: 12px;
        --font-display: "IM Fell French Canon", Georgia, "Times New Roman", serif;
        --font-body: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        --font-mono: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
    }

    /* Native Streamlit normalization */
    html, body, [class*="css"] {
        color: var(--ink);
        font-family: var(--font-body);
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background: var(--canvas) !important;
    }

    [data-testid="stHeader"] { height: 0; }
    [data-testid="stDecoration"], #MainMenu, footer { display: none; }

    .block-container {
        max-width: 1380px;
        padding: 1.5rem 2.75rem 5rem;
    }

    ::selection { background: var(--oxblood); color: var(--surface); }
    ::-webkit-scrollbar { width: 9px; height: 9px; }
    ::-webkit-scrollbar-track { background: var(--canvas); }
    ::-webkit-scrollbar-thumb { background: var(--rule-strong); border: 2px solid var(--canvas); }
    ::-webkit-scrollbar-thumb:hover { background: var(--ink-secondary); }

    h1, h2, h3, .paper-title, .question {
        color: var(--ink) !important;
        font-family: var(--font-display) !important;
        font-weight: 400 !important;
        letter-spacing: -0.02em;
    }

    p, [data-testid="stMarkdownContainer"] li {
        color: var(--ink-secondary);
        font-family: var(--font-body);
        line-height: 1.65;
    }

    a {
        color: var(--link);
        text-decoration-color: color-mix(in srgb, var(--link) 40%, transparent);
        text-underline-offset: 0.18em;
    }

    a:hover { color: var(--oxblood); text-decoration-thickness: 1px; }

    code {
        padding: 0.1rem 0.25rem;
        background: var(--surface-muted);
        color: var(--ink);
        font-family: var(--font-mono);
        overflow-wrap: anywhere;
    }

    label, [data-testid="stCaptionContainer"] {
        color: var(--ink-secondary) !important;
        font-family: var(--font-body) !important;
    }

    *:focus-visible {
        outline: 3px solid color-mix(in srgb, var(--link) 58%, transparent) !important;
        outline-offset: 2px;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] [role="combobox"],
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        min-height: 2.75rem !important;
        border: 1px solid var(--rule-strong) !important;
        border-radius: var(--radius-control) !important;
        background: var(--surface-control) !important;
        color: var(--ink) !important;
        font-family: var(--font-body) !important;
        font-size: 0.96rem !important;
        transition: border-color 140ms ease, background-color 140ms ease;
    }

    [data-testid="stTextArea"] textarea { min-height: 6.25rem !important; }

    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {
        color: var(--ink-muted) !important;
        opacity: 1;
    }

    [data-testid="stSelectbox"] [role="combobox"] *,
    [data-testid="stSelectbox"] svg {
        color: var(--ink) !important;
        fill: currentColor !important;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: var(--oxblood) !important;
        background: #ffffff !important;
        outline: 3px solid color-mix(in srgb, var(--oxblood) 18%, transparent) !important;
        outline-offset: 0;
    }

    [data-baseweb="popover"], [data-baseweb="menu"] {
        background: var(--surface) !important;
        color: var(--ink) !important;
    }

    .stButton > button,
    [data-testid="stButton"] button,
    [data-testid="stLinkButton"] a,
    [data-testid="stFormSubmitButton"] button {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-height: 44px !important;
        padding: 0.65rem 1rem !important;
        border: 1px solid var(--rule-strong) !important;
        border-radius: var(--radius-control) !important;
        background: transparent !important;
        color: var(--ink) !important;
        font: 600 0.87rem/1 var(--font-body) !important;
        text-align: center !important;
        transition: border-color 140ms ease, color 140ms ease, background-color 140ms ease !important;
    }

    .stButton > button [data-testid="stMarkdownContainer"],
    [data-testid="stButton"] button [data-testid="stMarkdownContainer"],
    [data-testid="stLinkButton"] a [data-testid="stMarkdownContainer"],
    [data-testid="stFormSubmitButton"] button [data-testid="stMarkdownContainer"] {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100%;
        min-width: 0;
        text-align: center !important;
    }

    .stButton > button [data-testid="stMarkdownContainer"] p,
    [data-testid="stButton"] button [data-testid="stMarkdownContainer"] p,
    [data-testid="stLinkButton"] a [data-testid="stMarkdownContainer"] p,
    [data-testid="stFormSubmitButton"] button [data-testid="stMarkdownContainer"] p {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100%;
        margin: 0 !important;
        color: inherit !important;
        line-height: 1.25 !important;
        text-align: center !important;
    }

    .stButton > button:hover,
    [data-testid="stButton"] button:hover,
    [data-testid="stLinkButton"] a:hover {
        border-color: var(--ink) !important;
        background: var(--surface-muted) !important;
        color: var(--ink) !important;
    }

    [data-testid="stFormSubmitButton"] button,
    button[data-testid^="stBaseButton-primary"],
    .stButton > button[kind="primary"] {
        border-color: var(--oxblood) !important;
        background: var(--oxblood) !important;
        color: #ffffff !important;
    }

    [data-testid="stFormSubmitButton"] button p,
    button[data-testid^="stBaseButton-primary"] p,
    .stButton > button[kind="primary"] p {
        color: #ffffff !important;
    }

    [data-testid="stFormSubmitButton"] button:hover,
    button[data-testid^="stBaseButton-primary"]:hover,
    .stButton > button[kind="primary"]:hover {
        border-color: var(--oxblood-hover) !important;
        background: var(--oxblood-hover) !important;
        color: #ffffff !important;
    }

    .stButton > button:disabled,
    [data-testid="stButton"] button:disabled,
    [data-testid="stFormSubmitButton"] button:disabled {
        border-color: var(--rule) !important;
        background: var(--surface-muted) !important;
        color: var(--ink-muted) !important;
        cursor: not-allowed !important;
        opacity: 0.72;
    }

    [data-testid="stFormSubmitButton"] button:disabled p,
    button[data-testid^="stBaseButton-primary"]:disabled p,
    .stButton > button[kind="primary"]:disabled p {
        color: var(--ink-muted) !important;
    }

    [data-testid="stToggle"] [role="switch"][aria-checked="true"] {
        background: var(--oxblood) !important;
    }

    [data-testid="stExpander"] {
        border: 1px solid var(--rule) !important;
        border-radius: var(--radius-control) !important;
        background: #f7f3eb !important;
        overflow: hidden;
    }

    [data-testid="stExpander"] summary {
        min-height: 2.6rem;
        padding-inline: 0.3rem;
        color: var(--ink-secondary) !important;
        font-size: 0.88rem;
        transition: background-color 140ms ease, color 140ms ease;
    }

    [data-testid="stExpander"] summary:hover {
        background: var(--surface-muted) !important;
        color: var(--ink) !important;
    }

    [data-testid="stAlert"] {
        border-radius: var(--radius-control) !important;
        color: var(--ink) !important;
    }

    [data-testid="stSpinner"] { color: var(--oxblood) !important; }
    hr { border-color: var(--rule) !important; }

    /* Masthead */
    .masthead {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(286px, auto);
        gap: 2.25rem;
        align-items: end;
        padding: 1.15rem 0 1.25rem;
        margin-bottom: 1.35rem;
        border-top: 4px solid var(--oxblood);
        border-bottom: 1px solid var(--rule-strong);
    }

    .masthead h1 {
        max-width: none;
        margin: 0;
        font-size: clamp(2.2rem, 2.8vw, 2.9rem);
        line-height: 1;
        white-space: nowrap;
    }

    .strap {
        margin-top: 0.55rem;
        color: var(--oxblood);
        font: 400 0.7rem/1.4 var(--font-mono);
        letter-spacing: 0.08em;
    }

    .engine-meta {
        display: grid;
        grid-template-columns: auto auto;
        min-width: 286px;
        margin: 0;
        padding-left: 1.5rem;
        border-left: 1px solid var(--rule);
        gap: 0.3rem 1.4rem;
        font: 400 0.68rem/1.45 var(--font-mono);
        font-variant-numeric: tabular-nums;
    }

    .engine-meta dt { color: var(--ink-muted); }
    .engine-meta dd {
        margin: 0;
        color: var(--ink);
        text-align: right;
        overflow-wrap: anywhere;
    }
    .engine-meta dd.ready { color: var(--success); }
    .engine-meta dd.offline { color: var(--danger); }

    /* Editorial mode navigation */
    [data-testid="stRadioGroup"] {
        display: flex;
        width: min(100%, 17rem);
        margin: 0 0 1.4rem;
        padding: 0.25rem;
        border: 1px solid var(--rule);
        border-radius: 10px;
        background: var(--surface-muted);
        gap: 0.25rem;
    }

    [data-testid="stRadioGroup"] [data-testid="stRadioOption"] {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        flex: 1 1 0;
        min-width: 0;
        min-height: 44px;
        margin: 0;
        padding: 0.52rem 1rem;
        border: 1px solid transparent !important;
        border-radius: 7px !important;
        box-sizing: border-box;
        background: transparent !important;
        color: var(--ink-secondary) !important;
        font-family: var(--font-body) !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        text-align: center !important;
        white-space: nowrap;
        cursor: pointer;
    }

    [data-testid="stRadioGroup"] [data-testid="stRadioOption"]:hover {
        background: color-mix(in srgb, var(--surface) 58%, transparent) !important;
        color: var(--ink) !important;
    }

    [data-testid="stRadioGroup"] [data-testid="stRadioOption"]:has(input:checked) {
        border-color: var(--rule-strong) !important;
        background: var(--surface-control) !important;
        color: var(--oxblood) !important;
    }

    [data-testid="stRadioGroup"] [data-testid="stRadioOption"] p {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100%;
        margin: 0 !important;
        color: inherit !important;
        line-height: 1.25 !important;
        text-align: center !important;
    }

    [data-testid="stRadioOption"] > div > div:first-child > div:first-child {
        display: none !important;
    }

    [data-testid="stRadioOption"] > div {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100%;
        min-width: 0;
    }

    [data-testid="stRadioOption"] > div > div:first-child {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100%;
        gap: 0 !important;
    }

    [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100%;
        min-width: 0;
        text-align: center !important;
    }

    /* Search and Ask instruments */
    [data-testid="stForm"] {
        margin-bottom: 1.5rem;
        padding: 1.25rem 1.35rem 1.4rem;
        border: 1px solid var(--rule) !important;
        border-radius: var(--radius-panel) !important;
        background: var(--surface);
    }

    .method-explainer {
        max-width: 54ch;
        margin: 0.55rem 0 0;
        color: var(--ink-muted);
        font: 400 0.8rem/1.55 var(--font-body);
    }

    .method-explainer strong { color: var(--oxblood); font-weight: 600; }

    /* Proceedings-style search results */
    .result-summary {
        display: flex;
        align-items: baseline;
        flex-wrap: wrap;
        gap: 0.35rem 1rem;
        margin-bottom: 0.2rem;
        padding: 0 0 0.75rem;
        border-bottom: 1px solid var(--ink);
        color: var(--ink-muted);
        font: 400 0.73rem/1.4 var(--font-mono);
        font-variant-numeric: tabular-nums;
    }

    .result-summary strong { color: var(--ink); font-size: 0.82rem; font-weight: 400; }
    .result-summary span + span { border-left: 1px solid var(--rule); padding-left: 1rem; }

    .paper-entry {
        margin: 0;
        padding: 1.25rem 0 1rem;
        border-bottom: 1px solid var(--rule);
        background: transparent;
    }

    .paper-entry:hover { background: color-mix(in srgb, var(--surface) 70%, transparent); }

    .paper-head {
        display: grid;
        grid-template-columns: 2.35rem minmax(0, 1fr) auto;
        gap: 0.85rem;
        align-items: baseline;
    }

    .paper-rank {
        color: var(--oxblood);
        font: 400 1rem/1 var(--font-mono);
        font-variant-numeric: tabular-nums;
    }

    .paper-rank::after { content: "."; }

    .paper-title {
        max-width: 48ch;
        color: var(--ink);
        font-size: 1.32rem;
        line-height: 1.24;
    }

    .paper-score {
        color: var(--ink-muted);
        font: 400 0.68rem/1.35 var(--font-mono);
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
    }

    .paper-meta-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.2rem 0.55rem;
        margin: 0.55rem 0 0 3.2rem;
    }

    .meta-chip {
        display: inline-flex;
        align-items: center;
        padding: 0;
        border: 0;
        background: transparent;
        color: var(--ink-muted);
        font: 400 0.7rem/1.45 var(--font-mono);
    }

    .meta-chip + .meta-chip::before {
        content: "·";
        margin-right: 0.55rem;
        color: var(--rule-strong);
    }

    .meta-chip.highlight { color: var(--oxblood); }

    .paper-abstract {
        max-width: 75ch;
        margin: 0.9rem 0 0.35rem 3.2rem;
        padding: 0.75rem 1rem;
        border-left: 1px solid var(--rule-strong);
        background: var(--surface);
        color: var(--ink-secondary);
        font-size: 0.94rem;
        line-height: 1.65;
    }

    /* Research answer sheet and validation states */
    .st-key-answer_sheet {
        min-height: 16rem;
        padding: 1.35rem 1.5rem 1.7rem;
        border-top: 3px solid var(--ink);
        border-bottom: 1px solid var(--rule);
        background: var(--surface);
    }

    .st-key-answer_sheet [data-testid="stMarkdownContainer"] > p {
        max-width: 72ch;
        color: var(--ink);
        font-size: 1.02rem;
        line-height: 1.72;
    }

    .content-label, .answer-label {
        margin-bottom: 0.45rem;
        color: var(--oxblood);
        font: 400 0.68rem/1.4 var(--font-mono);
        letter-spacing: 0.05em;
    }

    .question {
        margin: 0 0 1.1rem;
        padding: 0 0 0.9rem;
        border-bottom: 1px solid var(--rule);
        font-size: clamp(1.35rem, 2.3vw, 1.85rem);
        line-height: 1.18;
    }

    .pipeline-state {
        display: grid;
        grid-template-columns: 0.55rem auto minmax(0, 1fr);
        align-items: baseline;
        gap: 0.55rem;
        margin-bottom: 1rem;
        padding: 0.62rem 0;
        border-top: 1px solid var(--rule);
        border-bottom: 1px solid var(--rule);
        color: var(--ink-secondary);
        font: 400 0.79rem/1.45 var(--font-body);
        overflow-wrap: anywhere;
    }

    .pipeline-state strong { color: var(--ink); font-weight: 600; }
    .pipeline-state.is-complete { color: var(--success); }
    .pipeline-state.is-complete strong { color: var(--success); }
    .pipeline-state.is-error { color: var(--danger); background: var(--danger-soft); }
    .pipeline-state.is-warning { color: var(--warning); background: var(--warning-soft); }

    .status-dot {
        display: inline-block;
        width: 0.45rem;
        height: 0.45rem;
        border: 1px solid currentColor;
        border-radius: 50%;
        background: currentColor;
    }

    .status-dot.ready, .ready { color: var(--success); }
    .status-dot.offline, .offline { color: var(--danger); }
    .status-dot.warning { color: var(--warning); }

    /* Evidence ledger */
    .st-key-search_rail,
    .st-key-evidence_rail,
    .st-key-empty_evidence_rail {
        padding: 1rem 1.1rem 1.3rem;
        border-top: 3px solid var(--ink);
        border-bottom: 1px solid var(--rule);
        background: var(--surface);
    }

    .rail-title {
        margin: 0 0 0.85rem !important;
        color: var(--ink) !important;
        font: 400 1.18rem/1.25 var(--font-display) !important;
        letter-spacing: -0.01em;
    }

    .ledger-heading {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 1rem;
    }

    .ledger-heading > span {
        color: var(--ink-muted);
        font: 400 0.68rem/1.3 var(--font-mono);
    }

    .ledger-timing {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 0.3rem 0.65rem;
        padding: 0.45rem 0 0.7rem;
        border-bottom: 1px solid var(--rule);
        color: var(--ink-muted);
        font: 400 0.67rem/1.45 var(--font-mono);
        font-variant-numeric: tabular-nums;
    }

    .ledger-timing strong { color: var(--oxblood); font-weight: 400; }

    .source-item {
        margin: 0;
        padding: 0.9rem 0;
        border-bottom: 1px solid var(--rule);
        background: transparent;
    }

    .source-item.is-cited { background: var(--oxblood-soft); }

    .source-heading {
        display: grid;
        grid-template-columns: 2rem minmax(0, 1fr);
        align-items: baseline;
        gap: 0.45rem;
        min-width: 0;
    }

    .source-index {
        color: var(--oxblood);
        font: 400 0.78rem/1 var(--font-mono);
    }

    .source-title-row {
        display: block;
        min-width: 0;
        line-height: 1.25;
    }

    .source-title {
        min-width: 0;
        color: var(--ink);
        font-family: var(--font-display);
        font-size: 1rem;
        line-height: 1.25;
    }

    .source-meta {
        margin: 0.4rem 0 0 2.45rem;
        color: var(--ink-muted);
        font: 400 0.73rem/1.55 var(--font-body);
    }

    .cited-label {
        display: inline-flex;
        align-items: center;
        white-space: nowrap;
        vertical-align: baseline;
        color: var(--success);
        font: 400 0.62rem/1 var(--font-mono);
    }

    .cited-separator { color: var(--ink-muted); }

    .citation-badge {
        margin: 0 0.12rem;
        padding: 0.05rem 0.2rem;
        border-bottom: 1px solid var(--oxblood);
        color: var(--oxblood);
        font: 400 0.72rem/1 var(--font-mono);
    }

    /* Pipeline disclosure and utility information */
    .pipeline-note {
        margin: 0 0 0.55rem;
        color: var(--ink-muted);
        font-size: 0.76rem;
        line-height: 1.5;
    }

    .pipeline-steps {
        display: grid;
        list-style: none;
        gap: 0;
        margin: 0;
        padding: 0;
    }

    .pipeline-steps li {
        display: grid;
        grid-template-columns: 1.8rem minmax(0, 1fr);
        align-items: start;
        gap: 0.6rem;
        padding: 0.55rem 0;
        border-bottom: 1px solid var(--rule);
        font-family: var(--font-body);
    }

    .pipeline-steps li > span {
        color: var(--oxblood);
        font: 400 0.66rem/1.4 var(--font-mono);
    }

    .pipeline-steps strong { color: var(--ink); font-size: 0.82rem; font-weight: 600; }
    .pipeline-steps small {
        display: block;
        margin-top: 0.1rem;
        color: var(--ink-muted);
        font-size: 0.73rem;
    }

    .readiness-row {
        display: grid;
        grid-template-columns: 0.55rem minmax(0, 1fr) auto;
        align-items: center;
        gap: 0.55rem;
        padding: 0.38rem 0;
        color: var(--ink-secondary);
        font: 400 0.75rem/1.4 var(--font-body);
    }

    .readiness-row strong { color: var(--ink); font-size: 0.7rem; font-weight: 600; }

    .section-rule { margin: 1.15rem 0; border-top: 1px solid var(--rule); }

    /* Empty and waiting states */
    .empty-sheet {
        max-width: 70ch;
        padding: 1.5rem 0;
        border-top: 1px solid var(--rule-strong);
        border-bottom: 1px solid var(--rule);
        text-align: left;
    }

    .empty-sheet h3 {
        margin: 0 0 0.45rem;
        color: var(--ink);
        font-size: 1.45rem;
    }

    .empty-sheet p {
        max-width: 62ch;
        margin: 0;
        color: var(--ink-secondary);
        font-size: 0.94rem;
        line-height: 1.6;
    }

    .st-key-answer_sheet .empty-sheet {
        padding: 0;
        border: 0;
    }

    .evidence-empty {
        display: flex;
        min-height: 5rem;
        flex-direction: column;
        justify-content: center;
        margin: 0.35rem 0;
        padding: 1rem 0;
        border-top: 1px solid var(--rule);
        border-bottom: 1px solid var(--rule);
        text-align: left;
    }

    .evidence-empty strong { color: var(--ink); font-size: 0.88rem; font-weight: 600; }
    .evidence-empty span {
        max-width: 34ch;
        margin-top: 0.25rem;
        color: var(--ink-muted);
        font: 400 0.75rem/1.5 var(--font-body);
    }

    .source-title, .source-meta, .paper-title, .paper-abstract, .question,
    [data-testid="stAlert"], [data-testid="stMarkdownContainer"] {
        overflow-wrap: anywhere;
        word-break: normal;
    }

    /* Responsive composition */
    @media (max-width: 1100px) {
        .masthead { grid-template-columns: 1fr; align-items: start; gap: 0.9rem; }
        .masthead h1 { white-space: normal; text-wrap: balance; }
        .engine-meta {
            width: 100%;
            min-width: 0;
            padding: 0.75rem 0 0;
            border-top: 1px solid var(--rule);
            border-left: 0;
        }
        .engine-meta dd { max-width: 32ch; }
    }

    @media (max-width: 900px) {
        .block-container { padding: 1.25rem 1.5rem 4rem; }
        .st-key-search_rail, .st-key-evidence_rail, .st-key-empty_evidence_rail {
            margin-top: 1.25rem;
        }
    }

    @media (max-width: 640px) {
        .block-container { padding: 0.85rem 0.8rem 3rem; }
        .masthead { padding-top: 0.8rem; margin-bottom: 1rem; }
        .masthead h1 { max-width: 18ch; font-size: clamp(1.8rem, 10vw, 2.35rem); }
        .engine-meta { grid-template-columns: minmax(0, 1fr) minmax(0, 1.5fr); gap: 0.3rem 0.7rem; }
        .engine-meta dd { text-align: left; }
        [data-testid="stRadioGroup"] { width: 100%; }
        [data-testid="stRadioGroup"] [data-testid="stRadioOption"] {
            min-height: 2.75rem;
            padding-inline: 0.75rem;
        }
        [data-testid="stForm"] { padding: 0.95rem 0.85rem 1rem; }
        .paper-head { grid-template-columns: 1.65rem minmax(0, 1fr); }
        .paper-score { grid-column: 2; white-space: normal; }
        .paper-meta-chips, .paper-abstract { margin-left: 2.5rem; }
        .paper-title { font-size: 1.18rem; }
        .result-summary span + span { border-left: 0; padding-left: 0; }
        .source-heading { grid-template-columns: 1.7rem minmax(0, 1fr); }
        .source-meta { margin-left: 2.15rem; font-size: 0.78rem; }
        .pipeline-state { grid-template-columns: 0.55rem minmax(0, 1fr); }
        .pipeline-state > span:last-child { grid-column: 2; }
        .st-key-answer_sheet { padding: 1.1rem 1rem 1.35rem; }
        .st-key-answer_sheet [data-testid="stMarkdownContainer"] > p { font-size: 1rem; }
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] [role="combobox"],
        [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            min-height: 2.75rem !important;
            font-size: 1rem !important;
        }
        .stButton > button,
        [data-testid="stButton"] button,
        [data-testid="stLinkButton"] a,
        [data-testid="stFormSubmitButton"] button { min-height: 44px !important; }
    }

    @media (max-width: 420px) {
        .block-container { padding-inline: 0.65rem; }
        .paper-entry { padding-block: 1rem 0.85rem; }
        .paper-meta-chips, .paper-abstract { margin-left: 0; }
        .source-meta { margin-left: 0; }
        .st-key-search_rail, .st-key-evidence_rail, .st-key-empty_evidence_rail {
            padding-inline: 0.8rem;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            scroll-behavior: auto !important;
            transition: none !important;
            animation: none !important;
        }
    }
    </style>
    """
    return css.replace("__DISPLAY_FONT__", _embedded_font("IMFellFrenchCanon-Regular.ttf")).replace(
        "__MONO_FONT__", _embedded_font("IBMPlexMono-Regular.ttf")
    )
