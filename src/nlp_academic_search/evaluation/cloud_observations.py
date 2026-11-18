"""Collect final synchronous Ask observations without touching reviewer labels."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from nlp_academic_search.api.schemas import AskResponse
from nlp_academic_search.rag.citations import is_refusal, validate_citations

OBSERVATION_FIELDS = (
    "observed_answer",
    "observed_refused",
    "observed_error",
    "total_latency_ms",
    "cited_source_ids",
    "generation_model",
    "verification_model",
    "verification_status",
    "http_status",
    "request_id",
    "error_kind",
    "observed_at_utc",
)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as f:
        temp_path = Path(f.name)
        os.chmod(temp_path, 0o600)
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(temp_path, path)


def _atomic_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as f:
        temp_path = Path(f.name)
        os.chmod(temp_path, 0o600)
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def _load_sheet(path: Path, template_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    source = path if path.exists() else template_path
    with source.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields or not rows:
        raise ValueError("review sheet must contain headers and rows")
    if len(set(fields)) != len(fields):
        raise ValueError("review sheet contains duplicate headers")
    if any(None in row for row in rows):
        raise ValueError("review sheet contains more fields than its header")
    return fields, rows


def _question_digest(rows: list[dict[str, str]]) -> str:
    pairs = [(row.get("id"), row.get("question")) for row in rows]
    return hashlib.sha256(json.dumps(pairs, ensure_ascii=False).encode()).hexdigest()


def _pre_run_label_digest(rows: list[dict[str, str]]) -> str:
    keys = (
        "id",
        "question",
        "human_expected_answerability",
        "reference_source_ids",
        "expected_answer",
        "evidence_quote",
        "review_notes",
        "reviewer_kind",
        "reviewer_name_or_id",
        "reviewed_at_utc",
    )
    values = [[row.get(key, "") for key in keys] for row in rows]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()


def _validate_sheet(candidates: dict[str, Any], rows: list[dict[str, str]]) -> None:
    cases = candidates.get("cases")
    if candidates.get("dataset_role") != "candidate_draft_not_scored" or not isinstance(
        cases, list
    ):
        raise ValueError("expected a candidate question bank")
    candidate_ids = [case["id"] for case in cases]
    sheet_ids = [row.get("id") for row in rows]
    if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique")
    if candidate_ids != sheet_ids:
        raise ValueError("review sheet IDs or order differ from candidate bank")
    pending = any(not (row.get("observed_error") or row.get("observed_refused")) for row in rows)
    if pending and any(
        row.get(key)
        for row in rows
        for key in ("answer_correct", "factual_claims_json", "refusal_appropriate")
    ):
        raise ValueError("post-response labels must remain blank until observations finish")
    for candidate, row in zip(cases, rows, strict=True):
        if row.get("anchor_paper_id", "") != (candidate.get("anchor_paper_id") or ""):
            raise ValueError(f"{row.get('id')}: anchor differs from candidate bank")
        question = row.get("question") or ""
        if not 5 <= len(question) <= 1000:
            raise ValueError(f"{row.get('id')}: question length is outside Ask limits")
        case_id = row["id"]
        answerability = row.get("human_expected_answerability")
        if answerability not in {"answerable", "unanswerable"}:
            raise ValueError(f"{case_id}: human_expected_answerability must be reviewed before Ask")
        try:
            reference_ids = json.loads(row.get("reference_source_ids") or "")
        except json.JSONDecodeError as exc:
            raise ValueError(f"{case_id}: reference_source_ids must be a JSON array") from exc
        if not isinstance(reference_ids, list) or not all(
            isinstance(value, str) and value.startswith("arxiv:") for value in reference_ids
        ):
            raise ValueError(f"{case_id}: reference_source_ids must contain arxiv IDs")
        if len(set(reference_ids)) != len(reference_ids):
            raise ValueError(f"{case_id}: reference_source_ids contains duplicates")
        if answerability == "answerable":
            if not reference_ids or not row.get("expected_answer") or not row.get("evidence_quote"):
                raise ValueError(f"{case_id}: answerable case needs reference, answer and quote")
        elif reference_ids or not row.get("review_notes"):
            raise ValueError(f"{case_id}: unanswerable case needs [] and an explanation")
        if (
            row.get("reviewer_kind") != "human"
            or not row.get("reviewer_name_or_id")
            or not row.get("reviewed_at_utc")
        ):
            raise ValueError(f"{case_id}: independent human pre-review is required before Ask")


def _readiness(client: httpx.Client, corpus_version: str, paper_count: int) -> dict[str, Any]:
    response = client.get("/health/ready")
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or data.get("status") != "ready":
        raise ValueError("cloud API is not ready")
    expected_provenance = f"qdrant-cloud:{corpus_version}"
    if data.get("index_provenance") != expected_provenance:
        raise ValueError("cloud corpus version differs from candidate bank")
    if data.get("total_papers") != paper_count:
        raise ValueError("cloud paper count differs from candidate bank")
    if not data.get("verification_required") or not data.get("verification_available"):
        raise ValueError("cloud verification is not required and available")
    models = data.get("models")
    if not isinstance(models, dict) or not models.get("llm") or not models.get("verifier"):
        raise ValueError("cloud readiness did not report generation and verification models")
    return {
        "index_provenance": expected_provenance,
        "generation_model": models["llm"],
        "verification_model": models["verifier"],
    }


def _observation(client: httpx.Client, question: str, models: dict[str, str]) -> dict[str, str]:
    started = time.perf_counter()
    http_status = ""
    request_id = ""
    answer_for_error = ""
    cited_ids_for_error = "[]"
    try:
        response = client.post(
            "/api/v1/ask", json={"question": question, "top_k": 5, "use_reranker": False}
        )
        http_status = str(response.status_code)
        request_id = response.headers.get("x-request-id", "")
        if not response.is_success:
            raise ValueError(f"http_{response.status_code}")
        try:
            result = AskResponse.model_validate(response.json())
        except (ValidationError, ValueError) as exc:
            raise ValueError("invalid_response") from exc
        indices = validate_citations(result.answer, len(result.sources)).cited_indices
        answer_for_error = result.answer
        cited_ids_for_error = json.dumps([result.sources[index - 1].id for index in indices])
        if any(source.index != index for index, source in enumerate(result.sources, start=1)):
            raise ValueError("source_index_mismatch")
        if result.metadata.model != models["generation_model"]:
            raise ValueError("generation_model_mismatch")
        status = result.metadata.answer_status
        refused_by_text = is_refusal(result.answer)
        refused_by_status = status.startswith("refused_")
        if refused_by_text != refused_by_status:
            raise ValueError("refusal_status_answer_mismatch")
        return {
            "observed_answer": result.answer,
            "observed_refused": str(refused_by_text).lower(),
            "observed_error": "false",
            "total_latency_ms": str(round((time.perf_counter() - started) * 1000, 2)),
            "cited_source_ids": cited_ids_for_error,
            "generation_model": result.metadata.model,
            "verification_model": models["verification_model"],
            "verification_status": status,
            "http_status": http_status,
            "request_id": request_id,
            "error_kind": "",
            "observed_at_utc": datetime.now(UTC).isoformat(),
        }
    except (httpx.HTTPError, ValueError) as exc:
        error_kind = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        return {
            "observed_answer": answer_for_error,
            "observed_refused": "false",
            "observed_error": "true",
            "total_latency_ms": str(round((time.perf_counter() - started) * 1000, 2)),
            "cited_source_ids": cited_ids_for_error,
            "generation_model": models["generation_model"],
            "verification_model": models["verification_model"],
            "verification_status": "request_error",
            "http_status": http_status,
            "request_id": request_id,
            "error_kind": error_kind,
            "observed_at_utc": datetime.now(UTC).isoformat(),
        }


def collect_cloud_observations(
    *,
    candidates: dict[str, Any],
    candidate_sha256: str,
    template_path: Path,
    sheet_path: Path,
    manifest_path: Path,
    api_base_url: str,
    app_commit: str,
    limit: int,
    delay_seconds: float,
    client: httpx.Client,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, int | str | None]:
    """Run at most ``limit`` pending questions, checkpointing after each response."""
    if limit < 1 or limit > 100 or not math.isfinite(delay_seconds) or delay_seconds < 0:
        raise ValueError("limit must be 1..100 and delay_seconds must be finite/nonnegative")
    if len(app_commit) != 40 or any(char not in "0123456789abcdef" for char in app_commit):
        raise ValueError("app_commit must be a full lowercase Git SHA")
    fields, rows = _load_sheet(sheet_path, template_path)
    _validate_sheet(candidates, rows)
    corpus_version = candidates.get("corpus", {}).get("corpus_version")
    paper_count = candidates.get("corpus", {}).get("paper_count")
    if not isinstance(corpus_version, str) or not corpus_version:
        raise ValueError("candidate bank lacks corpus_version")
    if not isinstance(paper_count, int) or paper_count < 1:
        raise ValueError("candidate bank lacks paper_count")
    models = _readiness(client, corpus_version, paper_count)
    manifest = {
        "dataset_role": "pre_reviewed_unlocked_observation_run",
        "measurement_type": "synchronous_ask_not_streamlit_sse",
        "candidate_sha256": candidate_sha256,
        "question_sha256": _question_digest(rows),
        "pre_run_label_sha256": _pre_run_label_digest(rows),
        "api_base_url": api_base_url.rstrip("/"),
        "app_commit_asserted": app_commit,
        **models,
    }
    if manifest_path.exists() and not sheet_path.exists():
        raise ValueError("run manifest exists but review sheet is missing")
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("run manifest differs; use a new output directory")
    elif sheet_path.exists() and any(
        row.get("observed_error") or row.get("observed_refused") for row in rows
    ):
        raise ValueError("existing observations lack a run manifest")
    else:
        _atomic_json(manifest_path, manifest)

    for field in OBSERVATION_FIELDS:
        if field not in fields:
            fields.append(field)
    attempted = 0
    errors = 0
    stopped_on_http_status: str | None = None
    for row in rows:
        if attempted >= limit:
            break
        if row.get("observed_error") or row.get("observed_refused"):
            continue
        if attempted:
            sleep(delay_seconds)
        observed = _observation(client, row["question"], models)
        row.update(observed)
        attempted += 1
        errors += observed["observed_error"] == "true"
        _atomic_csv(sheet_path, fields, rows)
        if observed["http_status"] in {"401", "403", "429", "502", "503", "504"}:
            stopped_on_http_status = observed["http_status"]
            break
    return {
        "attempted": attempted,
        "errors": errors,
        "stopped_on_http_status": stopped_on_http_status,
        "remaining": sum(
            not (row.get("observed_error") or row.get("observed_refused")) for row in rows
        ),
    }
