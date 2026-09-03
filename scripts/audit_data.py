"""CLI entry point for the corpus/index integrity report."""

from __future__ import annotations

import json
from pathlib import Path

from nlp_academic_search.data.audit import build_data_audit


def main() -> None:
    report = build_data_audit()
    output = Path("reports/data_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
