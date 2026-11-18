"""Validate a completed human review CSV and create an unlocked scoring JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from nlp_academic_search.evaluation.cloud_review_sheet import convert_cloud_review_sheet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("review_sheet", type=Path)
    parser.add_argument("--app-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    with args.review_sheet.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    try:
        result = convert_cloud_review_sheet(candidates, rows, app_commit=args.app_commit)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Validated {len(result['cases'])} human-reviewed rows; output remains unlocked.")


if __name__ == "__main__":
    main()
