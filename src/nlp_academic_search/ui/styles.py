"""Visual system for the Streamlit research workspace."""

from __future__ import annotations


def stylesheet() -> str:
    """Return the complete, self-contained CSS for the modern research workspace."""
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;1,400&family=Source+Sans+3:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #0b0f19;
        --bg-surface: #111827;
        --bg-card: rgba(17, 24, 39, 0.75);
        --bg-card-hover: rgba(30, 41, 59, 0.85);
        --bg-pill: #1e293b;
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-glow: rgba(99, 102, 241, 0.4);

        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;

        --primary: #6366f1;
        --primary-glow: rgba(99, 102, 241, 0.25);
        --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%);
        --accent-gradient-hover: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #c026d3 100%);

        --success: #10b981;
        --success-glow: rgba(16, 185, 129, 0.2);
        --warning: #f59e0b;
        --danger: #ef4444;

        --font-sans: 'Source Sans 3', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        --font-mono: 'JetBrains Mono', Consolas, monospace;
    }

    html, body, [class*="css"] {
        color: var(--text-primary);
        font-family: var(--font-sans);
    }

    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: var(--bg-main) !important;
        background-image:
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.14) 0px, transparent 50%),
            radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.12) 0px, transparent 50%);
        background-attachment: fixed;
    }

    [data-testid="stHeader"] { height: 0; }
    [data-testid="stDecoration"], #MainMenu, footer { display: none; }
    .block-container { max-width: 1440px; padding: 2rem 3rem 6rem; }

    ::selection { background: var(--primary); color: #fff; }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-main); }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }

    h1, h2, h3, .paper-title {
        font-family: var(--font-sans) !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Masthead Header */
    .masthead {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 2rem;
        align-items: center;
        padding: 1.5rem 2rem;
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-glow);
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }

    .masthead h1 {
        font-size: clamp(1.8rem, 3.2vw, 2.5rem);
        line-height: 1.1;
        margin: 0;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .strap {
        font: 500 0.72rem/1.4 var(--font-mono);
        letter-spacing: 0.2em;
        text-transform: uppercase;
        margin-top: 0.5rem;
        color: var(--primary);
    }

    .engine-meta {
        display: grid;
        grid-template-columns: auto auto;
        gap: 0.4rem 1.5rem;
        font: 500 0.72rem/1.5 var(--font-mono);
        min-width: 280px;
        margin: 0;
        padding-left: 1.5rem;
        border-left: 1px solid var(--border-subtle);
    }

    .engine-meta dt { color: var(--text-muted); }
    .engine-meta dd { margin: 0; text-align: right; color: var(--text-primary); font-weight: 600; }
    .engine-meta dd.ready { color: var(--success); }
    .engine-meta dd.offline { color: var(--danger); }

    /* Custom Mode Toggle Pills (PROMPT 1) */
    div[role="radiogroup"] {
        display: flex;
        gap: 0.5rem;
        background: rgba(15, 23, 42, 0.8);
        padding: 0.35rem;
        border-radius: 14px;
        border: 1px solid var(--border-subtle);
        margin-bottom: 1.75rem;
    }

    div[role="radiogroup"] label {
        flex: 1;
        justify-content: center;
        border-radius: 10px !important;
        border: none !important;
        padding: 0.65rem 1.25rem;
        font-family: var(--font-sans) !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        color: var(--text-secondary) !important;
        background: transparent !important;
        transition: all 0.25s ease;
        cursor: pointer;
    }

    div[role="radiogroup"] label:hover {
        color: var(--text-primary) !important;
        background: rgba(255, 255, 255, 0.05) !important;
    }

    div[role="radiogroup"] label:has(input:checked) {
        background: var(--accent-gradient) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 16px var(--primary-glow);
    }

    div[role="radiogroup"] label > div:first-child { display: none; }

    /* Paper Cards & Meta Chips (PROMPT 2) */
    .paper-entry {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.35rem 1.6rem;
        margin-bottom: 1.25rem;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .paper-entry:hover {
        background: var(--bg-card-hover);
        border-color: var(--border-glow);
        transform: translateY(-2px);
        box-shadow: 0 12px 28px -8px rgba(99, 102, 241, 0.25);
    }

    .paper-head {
        display: grid;
        grid-template-columns: 2.3rem minmax(0, 1fr) auto;
        gap: 0.85rem;
        align-items: center;
    }

    .paper-rank {
        color: #ffffff;
        font: 700 1.1rem/1 var(--font-mono);
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
        width: 2.3rem;
        height: 2.3rem;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
    }

    .paper-title {
        font-size: 1.15rem;
        line-height: 1.35;
        color: var(--text-primary);
        font-weight: 600;
    }

    .paper-score {
        color: #a7f3d0;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.3rem 0.65rem;
        border-radius: 20px;
        font: 600 0.7rem/1.2 var(--font-mono);
    }

    .paper-meta-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem 0.5rem;
        margin: 0.65rem 0 0 3.15rem;
    }

    .meta-chip {
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: var(--text-secondary);
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font: 500 0.73rem/1.2 var(--font-mono);
    }

    .meta-chip.highlight {
        background: rgba(99, 102, 241, 0.15);
        border-color: rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
    }

    .paper-abstract {
        margin: 0.85rem 0 0.4rem 3.15rem;
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.6;
        background: rgba(15, 23, 42, 0.6);
        padding: 0.9rem 1.1rem;
        border-radius: 10px;
        border: 1px solid rgba(99, 102, 241, 0.24);
    }

    /* Textarea & Inputs Focus Glow (PROMPT 3) */
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-baseweb="select"] > div {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-family: var(--font-sans) !important;
        font-size: 0.95rem !important;
        transition: all 0.25s ease;
    }

    [data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px var(--primary-glow) !important;
    }

    .stButton > button, [data-testid="stFormSubmitButton"] > button {
        border-radius: 12px !important;
        border: 1px solid var(--border-subtle) !important;
        font: 600 0.85rem/1 var(--font-sans) !important;
        min-height: 3rem !important;
        transition: all 0.25s ease !important;
    }

    [data-testid="stFormSubmitButton"] > button[kind="primary"], .stButton > button[kind="primary"] {
        background: var(--accent-gradient) !important;
        border: none !important;
        color: white !important;
        box-shadow: 0 4px 16px var(--primary-glow) !important;
    }

    [data-testid="stFormSubmitButton"] > button[kind="primary"]:hover, .stButton > button[kind="primary"]:hover {
        background: var(--accent-gradient-hover) !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
    }

    /* Sidebar Rails & Pipeline State */
    .st-key-search_rail, .st-key-evidence_rail, .st-key-empty_evidence_rail {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-glow);
        border-radius: 16px;
        padding: 1.5rem;
    }

    .rail-title {
        font: 700 0.75rem/1.2 var(--font-mono) !important;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--primary) !important;
        margin: 0 0 1.1rem !important;
    }

    .pipeline-steps {
        list-style: none;
        padding: 0;
        margin: 0;
        display: grid;
        gap: 0.8rem;
    }

    .pipeline-steps li {
        display: grid;
        grid-template-columns: 1.8rem 1fr;
        gap: 0.75rem;
        align-items: start;
        font-family: var(--font-sans) !important;
    }

    .pipeline-steps li > span {
        color: #ffffff;
        font: 700 0.72rem/1.2 var(--font-mono);
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 0.25rem;
        border-radius: 6px;
        text-align: center;
    }

    .pipeline-steps strong {
        color: var(--text-primary);
        font-size: 0.88rem;
        font-weight: 600;
    }

    .pipeline-steps small {
        color: var(--text-muted);
        font-size: 0.78rem;
        display: block;
        margin-top: 0.15rem;
    }

    .readiness-row {
        display: grid;
        grid-template-columns: 0.6rem minmax(0, 1fr) auto;
        gap: 0.6rem;
        align-items: center;
        padding: 0.45rem 0;
        font: 500 0.8rem/1.4 var(--font-sans);
    }

    .status-dot {
        width: 0.45rem;
        height: 0.45rem;
        border-radius: 50%;
        display: inline-block;
    }

    .status-dot.ready { background: var(--success); box-shadow: 0 0 8px var(--success-glow); }
    .status-dot.offline { background: var(--danger); }

    /* Evidence Ledger & Cited Label (PROMPT 2 & 3) */
    .source-item {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }

    .source-item.is-cited {
        border-color: var(--primary);
        background: rgba(99, 102, 241, 0.14);
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.15);
    }

    .source-index {
        color: var(--primary);
        font: 700 0.95rem/1 var(--font-mono);
    }

    .source-title {
        color: var(--text-primary);
        font-size: 0.92rem;
        font-weight: 600;
    }

    .cited-label {
        color: #a7f3d0;
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 0.2rem 0.55rem;
        border-radius: 20px;
        font: 600 0.65rem/1 var(--font-mono);
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }

    /* In-Text Citation Badges (PROMPT 3) */
    .citation-badge {
        color: #a5b4fc;
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.35);
        padding: 0.15rem 0.45rem;
        border-radius: 6px;
        font: 600 0.75rem/1 var(--font-mono);
        margin: 0 0.15rem;
    }

    /* Progress & Pipeline State (PROMPT 3) */
    .pipeline-state {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.65rem;
        margin-bottom: 1.1rem;
        padding: 0.65rem 1rem;
        background: rgba(30, 41, 59, 0.6);
        border-radius: 10px;
        border: 1px solid var(--border-subtle);
        font: 500 0.84rem/1.4 var(--font-sans);
        overflow-wrap: anywhere;
    }

    .pipeline-state.is-complete {
        border-color: var(--success);
        color: var(--success);
        background: rgba(16, 185, 129, 0.1);
    }

    .pipeline-state.is-error {
        border-color: var(--danger);
        color: var(--danger);
        background: rgba(239, 68, 68, 0.1);
    }

    .pipeline-state.is-warning {
        border-color: var(--warning);
        color: var(--warning);
        background: rgba(245, 158, 11, 0.1);
    }

    .status-dot.warning {
        background: var(--warning);
        box-shadow: 0 0 8px rgba(245, 158, 11, 0.2);
    }

    .question {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        padding: 0.8rem 0 1rem;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 1rem;
    }

    .answer-label {
        color: var(--primary);
        font: 700 0.72rem/1.4 var(--font-mono);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }

    [data-testid="stExpander"] {
        border-radius: 12px !important;
        border: 1px solid var(--border-subtle) !important;
        background: rgba(15, 23, 42, 0.5) !important;
    }

    a { color: #818cf8; text-decoration: none; }
    a:hover { color: #a5b4fc; text-decoration: underline; }

    /* Result Summary Bar */
    .result-summary {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.55rem 1rem;
        padding: 0.75rem 1rem;
        margin-bottom: 1.25rem;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        font: 500 0.8rem/1.4 var(--font-mono);
        color: var(--text-secondary);
    }

    .result-summary strong {
        color: var(--text-primary);
        font-weight: 700;
        font-size: 0.88rem;
    }

    .result-summary span + span {
        border-left: 1px solid var(--border-subtle);
        padding-left: 1rem;
    }

    /* Section Dividers */
    .section-rule {
        border-top: 1px solid var(--border-subtle);
        margin: 1.25rem 0;
    }

    /* Empty State */
    .empty-sheet {
        border: 1px dashed rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        text-align: center;
    }

    .empty-sheet h3 {
        font-size: 1.3rem;
        color: var(--text-primary);
        margin: 0 0 0.65rem;
    }

    .empty-sheet p {
        color: var(--text-muted);
        font-size: 0.95rem;
        line-height: 1.6;
        max-width: 52ch;
        margin: 0 auto;
    }

    /* Method Explainer */
    .method-explainer {
        color: var(--text-muted);
        font: 500 0.78rem/1.55 var(--font-sans);
        margin: 0.5rem 0 0;
        max-width: 54ch;
    }

    .method-explainer strong {
        color: var(--primary);
        font-weight: 600;
    }

    /* Source Metadata & Heading */
    .source-meta {
        color: var(--text-secondary);
        font: 400 0.78rem/1.6 var(--font-sans);
        margin: 0.4rem 0 0 2.45rem;
    }

    .source-heading {
        display: grid;
        grid-template-columns: 2rem minmax(0, 1fr) auto;
        gap: 0.5rem;
        align-items: baseline;
        min-width: 0;
    }

    .source-title, .source-meta, .paper-title, .paper-abstract, .question,
    [data-testid="stAlert"], [data-testid="stMarkdownContainer"] {
        overflow-wrap: anywhere;
        word-break: normal;
    }

    /* Evidence Ledger Header & Timing */
    .ledger-heading {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 1rem;
    }

    .ledger-heading > span {
        color: var(--text-muted);
        font: 500 0.72rem/1.3 var(--font-mono);
    }

    .ledger-timing {
        display: flex;
        flex-wrap: wrap;
        gap: 0.3rem 0.65rem;
        justify-content: space-between;
        color: var(--text-secondary);
        font: 500 0.72rem/1.45 var(--font-mono);
        padding: 0.5rem 0 0.7rem;
        border-bottom: 1px solid var(--border-subtle);
    }

    .ledger-timing strong {
        color: var(--primary);
        font-weight: 600;
    }

    /* Evidence Empty State */
    .evidence-empty {
        min-height: 6rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        border: 1px dashed rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }

    .evidence-empty strong {
        color: var(--text-primary);
        font-size: 0.95rem;
        font-weight: 600;
    }

    .evidence-empty span {
        font: 400 0.75rem/1.5 var(--font-sans);
        color: var(--text-muted);
        margin-top: 0.3rem;
        max-width: 34ch;
    }

    /* Content & Answer Labels */
    .content-label {
        color: var(--primary);
        font: 700 0.72rem/1.4 var(--font-mono);
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* Standalone status classes for dd elements */
    .ready { color: var(--success); }
    .offline { color: var(--danger); }

    /* Streamlit native overrides */
    [data-testid="stAlert"] { border-radius: 10px !important; }
    [data-testid="stSpinner"] { color: var(--primary) !important; }
    hr { border-color: var(--border-subtle) !important; }

    label, [data-testid="stCaptionContainer"] {
        font-family: var(--font-sans) !important;
        color: var(--text-secondary) !important;
    }

    p, [data-testid="stMarkdownContainer"] li {
        font-family: var(--font-sans);
        color: var(--text-secondary);
    }

    code {
        font-family: var(--font-mono);
        overflow-wrap: anywhere;
        color: #a5b4fc;
        background: rgba(99, 102, 241, 0.12);
        padding: 0.15rem 0.35rem;
        border-radius: 4px;
    }

    *:focus-visible {
        outline: 2px solid rgba(99, 102, 241, 0.5) !important;
        outline-offset: 2px;
    }

    /* Disabled buttons */
    .stButton > button:disabled, [data-testid="stFormSubmitButton"] > button:disabled {
        background: rgba(30, 41, 59, 0.6) !important;
        border-color: rgba(255, 255, 255, 0.06) !important;
        color: var(--text-muted) !important;
        cursor: not-allowed !important;
        box-shadow: none !important;
    }

    /* Link buttons */
    .stButton > button:hover {
        border-color: var(--primary) !important;
        color: var(--primary) !important;
    }

    /* Responsive */
    @media (max-width: 800px) {
        .block-container { padding: 1rem 1rem 3rem; }
        .masthead { grid-template-columns: 1fr; gap: 1rem; border-radius: 12px; }
        .masthead h1 { font-size: clamp(1.5rem, 8vw, 2rem); }
        .engine-meta { min-width: 0; padding: 0.8rem 0 0; border-left: 0; border-top: 1px solid var(--border-subtle); }
        .paper-head { grid-template-columns: 1.6rem minmax(0, 1fr); }
        .paper-score { grid-column: 2; }
        .paper-meta-chips { margin-left: 2.35rem; }
        .paper-abstract { margin-left: 2.35rem; }
        .paper-title { font-size: 1.05rem; }
        .result-summary span + span { border-left: 0; padding-left: 0; }
        .source-heading { grid-template-columns: 1.8rem minmax(0, 1fr); }
        .cited-label { grid-column: 2; }
        .source-meta { margin-left: 2.25rem; }
        .st-key-search_rail, .st-key-evidence_rail, .st-key-empty_evidence_rail {
            border-radius: 12px; padding: 1rem; margin-top: 1.5rem;
        }
    }

    @media (max-width: 420px) {
        .block-container { padding: 0.75rem 0.65rem 2.5rem; }
        .masthead { padding: 1rem; margin-bottom: 1.25rem; }
        .engine-meta { grid-template-columns: minmax(0, 1fr) auto; gap: 0.35rem 0.75rem; }
        .paper-entry { padding: 1rem; }
        .paper-meta-chips, .paper-abstract { margin-left: 0; }
        .source-meta { margin-left: 0; font-size: 0.875rem; }
        .source-heading { grid-template-columns: 1.7rem minmax(0, 1fr); }
        .pipeline-state { align-items: flex-start; font-size: 0.875rem; }
        [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
        [data-baseweb="select"] > div { font-size: 1rem !important; }
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
