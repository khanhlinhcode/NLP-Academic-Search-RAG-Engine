"""Pure UI output-safety helpers that can be unit tested without Streamlit."""

from __future__ import annotations

import html
from urllib.parse import quote, urlparse


def escape_html(value: object) -> str:
    return html.escape(str(value if value not in (None, "") else "—"))


def safe_external_url(record: dict[str, object]) -> str | None:
    candidate = record.get("source_url")
    if candidate:
        parsed = urlparse(str(candidate))
        if parsed.scheme == "https" and parsed.netloc:
            return str(candidate)
    doi = record.get("doi")
    if doi:
        return f"https://doi.org/{quote(str(doi).strip(), safe='/():;.-_')}"
    return None
