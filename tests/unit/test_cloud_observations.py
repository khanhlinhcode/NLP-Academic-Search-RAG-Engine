import csv
import json
import math

import httpx
import pytest

from nlp_academic_search.evaluation.cloud_observations import collect_cloud_observations
from nlp_academic_search.rag.citations import validate_citations


def _source(index, paper_id):
    return {
        "index": index,
        "id": paper_id,
        "arxiv_id": paper_id.removeprefix("arxiv:"),
        "doi": None,
        "title": "Test paper",
        "authors": [],
        "categories": [],
        "year": 2001,
        "source_url": "https://arxiv.org/abs/1234",
        "pdf_url": None,
        "source": "arxiv",
    }


def _ask_response(question, answer, sources, status):
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "metadata": {
            "model": "generation-model",
            "retrieval_method": "rrf",
            "source_ids": [source["id"] for source in sources],
            "prompt_version": "test",
            "estimated_context_tokens": 100,
            "context_truncated": False,
            "latencies": {"retrieval_ms": 1, "generation_ms": 2, "total_ms": 3},
            "warnings": [],
            "citation_validation": validate_citations(answer, len(sources)).model_dump(),
            "answer_status": status,
        },
    }


@pytest.fixture
def collection_files(tmp_path):
    template = tmp_path / "template.csv"
    with template.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "question",
                "anchor_paper_id",
                "human_expected_answerability",
                "reference_source_ids",
                "expected_answer",
                "evidence_quote",
                "review_notes",
                "reviewer_kind",
                "reviewer_name_or_id",
                "reviewed_at_utc",
                "observed_answer",
                "observed_refused",
                "observed_error",
                "answer_correct",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "id": "one",
                    "question": "What is the answer?",
                    "anchor_paper_id": "arxiv:1",
                    "human_expected_answerability": "answerable",
                    "reference_source_ids": '["arxiv:1"]',
                    "expected_answer": "A finding",
                    "evidence_quote": "A finding",
                    "reviewer_kind": "human",
                    "reviewer_name_or_id": "reviewer-1",
                    "reviewed_at_utc": "2026-11-18T00:00:00Z",
                },
                {
                    "id": "two",
                    "question": "What is not known?",
                    "anchor_paper_id": "",
                    "human_expected_answerability": "unanswerable",
                    "reference_source_ids": "[]",
                    "review_notes": "No indexed evidence",
                    "reviewer_kind": "human",
                    "reviewer_name_or_id": "reviewer-1",
                    "reviewed_at_utc": "2026-11-18T00:00:00Z",
                },
            ]
        )
    candidates = {
        "dataset_role": "candidate_draft_not_scored",
        "corpus": {"corpus_version": "v1", "paper_count": 1000},
        "cases": [{"id": "one", "anchor_paper_id": "arxiv:1"}, {"id": "two"}],
    }
    return (
        candidates,
        template,
        tmp_path / "private" / "review_sheet.csv",
        tmp_path / "private" / "ask_run_manifest.json",
    )


def _client(handler):
    return httpx.Client(
        base_url="https://api.example.test",
        headers={"Authorization": "Bearer secret-for-test"},
        transport=httpx.MockTransport(handler),
    )


def _ready():
    return {
        "status": "ready",
        "index_provenance": "qdrant-cloud:v1",
        "total_papers": 1000,
        "verification_required": True,
        "verification_available": True,
        "models": {"llm": "generation-model", "verifier": "verification-model"},
    }


def _run(files, client, *, limit=1, delay=0):
    candidates, template, sheet, manifest = files
    return collect_cloud_observations(
        candidates=candidates,
        candidate_sha256="candidate-hash",
        template_path=template,
        sheet_path=sheet,
        manifest_path=manifest,
        api_base_url="https://api.example.test",
        app_commit="a" * 40,
        limit=limit,
        delay_seconds=delay,
        client=client,
        sleep=lambda _: None,
    )


def test_collects_only_final_cited_sources_and_resumes_without_touching_labels(collection_files):
    posted = []

    def handler(request):
        if request.url.path == "/health/ready":
            return httpx.Response(200, json=_ready())
        question = json.loads(request.content)["question"]
        posted.append(question)
        if question == "What is the answer?":
            answer = "The second paper answers it [2]."
            sources = [_source(1, "arxiv:1"), _source(2, "arxiv:2")]
            status = "verified"
        else:
            answer = "Not enough evidence in the retrieved sources."
            sources = []
            status = "refused_insufficient_context"
        return httpx.Response(
            200,
            json=_ask_response(question, answer, sources, status),
            headers={"x-request-id": "request-123"},
        )

    with _client(handler) as client:
        assert _run(collection_files, client) == {
            "attempted": 1,
            "errors": 0,
            "remaining": 1,
            "stopped_on_http_status": None,
        }
        assert _run(collection_files, client) == {
            "attempted": 1,
            "errors": 0,
            "remaining": 0,
            "stopped_on_http_status": None,
        }
        assert _run(collection_files, client) == {
            "attempted": 0,
            "errors": 0,
            "remaining": 0,
            "stopped_on_http_status": None,
        }
    _, _, sheet, manifest = collection_files
    rows = list(csv.DictReader(sheet.open(newline="")))
    assert posted == ["What is the answer?", "What is not known?"]
    assert rows[0]["cited_source_ids"] == '["arxiv:2"]'
    assert rows[0]["answer_correct"] == ""
    assert rows[0]["expected_answer"] == "A finding"
    assert rows[0]["reviewer_name_or_id"] == "reviewer-1"
    assert rows[0]["human_expected_answerability"] == "answerable"
    assert rows[0]["request_id"] == "request-123"
    assert rows[1]["observed_refused"] == "true"
    assert rows[1]["cited_source_ids"] == "[]"
    assert "secret-for-test" not in manifest.read_text()
    assert (
        json.loads(manifest.read_text())["measurement_type"] == "synchronous_ask_not_streamlit_sse"
    )


@pytest.mark.parametrize("status", [401, 403, 429, 503])
def test_auth_quota_or_outage_stops_after_first_error(collection_files, status):
    posts = []

    def handler(request):
        if request.url.path == "/health/ready":
            return httpx.Response(200, json=_ready())
        posts.append(1)
        return httpx.Response(status, headers={"x-request-id": "failed-123"})

    with _client(handler) as client:
        assert _run(collection_files, client, limit=2) == {
            "attempted": 1,
            "errors": 1,
            "remaining": 1,
            "stopped_on_http_status": str(status),
        }
    _, _, sheet, _ = collection_files
    row = next(csv.DictReader(sheet.open(newline="")))
    assert row["observed_error"] == "true"
    assert row["http_status"] == str(status)
    assert row["error_kind"] == f"http_{status}"
    assert row["request_id"] == "failed-123"
    assert posts == [1]


def test_wrong_corpus_stops_before_posting(collection_files):
    posts = []

    def handler(request):
        if request.url.path == "/health/ready":
            bad = _ready() | {"index_provenance": "qdrant-cloud:other"}
            return httpx.Response(200, json=bad)
        posts.append(1)
        return httpx.Response(200)

    with _client(handler) as client, pytest.raises(ValueError, match="corpus version differs"):
        _run(collection_files, client)
    assert not posts


def test_wrong_paper_count_stops_before_posting(collection_files):
    posts = []

    def handler(request):
        if request.url.path == "/health/ready":
            return httpx.Response(200, json=_ready() | {"total_papers": 999})
        posts.append(1)
        return httpx.Response(200)

    with _client(handler) as client, pytest.raises(ValueError, match="paper count differs"):
        _run(collection_files, client)
    assert not posts


def test_refusal_status_answer_mismatch_keeps_leaked_answer_for_review(collection_files):
    def handler(request):
        if request.url.path == "/health/ready":
            return httpx.Response(200, json=_ready())
        question = json.loads(request.content)["question"]
        leaked = "Unsupported answer [1]."
        return httpx.Response(
            200,
            json=_ask_response(question, leaked, [_source(1, "arxiv:1")], "refused_unverified"),
        )

    with _client(handler) as client:
        summary = _run(collection_files, client)
    _, _, sheet, _ = collection_files
    row = next(csv.DictReader(sheet.open(newline="")))
    assert summary["errors"] == 1
    assert row["observed_answer"] == "Unsupported answer [1]."
    assert row["error_kind"] == "refusal_status_answer_mismatch"


def test_changed_question_blocks_resume(collection_files):
    def handler(request):
        if request.url.path == "/health/ready":
            return httpx.Response(200, json=_ready())
        return httpx.Response(503)

    with _client(handler) as client:
        _run(collection_files, client)
    _, _, sheet, _ = collection_files
    text = sheet.read_text().replace("What is not known?", "What else is not known?")
    sheet.write_text(text)
    with _client(handler) as client, pytest.raises(ValueError, match="manifest differs"):
        _run(collection_files, client)


def test_blank_pre_review_blocks_before_any_network_call(collection_files):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=_ready())

    _, template, _, _ = collection_files
    content = template.read_text().replace("answerable", "")
    template.write_text(content)
    with (
        _client(handler) as client,
        pytest.raises(ValueError, match="human_expected_answerability"),
    ):
        _run(collection_files, client)
    assert calls == []


def test_post_response_label_filled_too_early_blocks_network(collection_files):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=_ready())

    _, template, _, _ = collection_files
    with template.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    rows[0]["answer_correct"] = "true"
    with template.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with _client(handler) as client, pytest.raises(ValueError, match="post-response labels"):
        _run(collection_files, client)
    assert calls == []


@pytest.mark.parametrize("delay", [math.nan, math.inf, -1.0])
def test_nonfinite_or_negative_delay_is_rejected_before_network(collection_files, delay):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=_ready())

    with _client(handler) as client, pytest.raises(ValueError, match="delay_seconds"):
        _run(collection_files, client, delay=delay)
    assert calls == []
