from __future__ import annotations

import json

import httpx
import pytest

from nlp_academic_search.config import GroqConfig, VerificationConfig
from nlp_academic_search.data.loader import Paper
from nlp_academic_search.providers.verification.base import (
    SemanticVerificationUnavailable,
)
from nlp_academic_search.providers.verification.groq import GroqSemanticVerificationProvider
from nlp_academic_search.rag.verification import (
    ClaimAssessment,
    EvidenceSpan,
    validate_semantic_assessment,
)


@pytest.fixture
def sample_papers() -> list[Paper]:
    return [
        Paper(
            id="p1",
            title="Dense Passage Retrieval",
            authors=["Alice"],
            abstract="Passage retrieval works via bi-encoders.",
            categories=["cs.CL"],
            source="arxiv",
            arxiv_id="2005.11401",
        )
    ]


def test_validate_semantic_assessment_supported(sample_papers: list[Paper]) -> None:
    answer = "Passage retrieval works via bi-encoders [1]."
    claims = [
        ClaimAssessment(
            claim_text="Passage retrieval works via bi-encoders [1].",
            factual=True,
            cited_indices=[1],
            verdict="supported",
            evidence=[
                EvidenceSpan(
                    source_index=1,
                    quote="Passage retrieval works via bi-encoders.",
                )
            ],
        )
    ]
    validation = validate_semantic_assessment(
        answer, sample_papers, claims, provider="groq", model="qwen-2.5-32b", independent=True
    )
    assert validation.valid is True
    assert validation.supported_claim_count == 1
    assert validation.unsupported_claim_count == 0
    assert len(validation.claims) == 1


def test_validate_semantic_assessment_unsupported_quote_not_found(
    sample_papers: list[Paper],
) -> None:
    answer = "Passage retrieval is instant [1]."
    claims = [
        ClaimAssessment(
            claim_text="Passage retrieval is instant [1].",
            factual=True,
            cited_indices=[1],
            verdict="supported",
            evidence=[
                EvidenceSpan(
                    source_index=1,
                    quote="Non-existent fake quote here",
                )
            ],
        )
    ]
    validation = validate_semantic_assessment(
        answer, sample_papers, claims, provider="groq", model="qwen-2.5-32b", independent=True
    )
    assert validation.valid is False
    assert validation.unsupported_claim_count == 0


def test_groq_verification_provider_success(sample_papers: list[Paper]) -> None:
    groq_cfg = GroqConfig(
        base_url="https://api.groq.com/openai/v1",
        api_key="fake-key",
        model_name="qwen-2.5-32b",
        timeout_seconds=30.0,
        max_output_tokens=1024,
    )
    ver_cfg = VerificationConfig(
        enabled=True,
        provider="groq",
        model_name="qwen-2.5-32b",
        timeout_seconds=30.0,
        fail_closed=False,
        max_repair_attempts=1,
    )

    def handle_request(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "qwen-2.5-32b"}]})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "claims": [
                                        {
                                            "claim_text": "Passage retrieval works via bi-encoders [1].",
                                            "factual": True,
                                            "cited_indices": [1],
                                            "verdict": "supported",
                                            "evidence": [
                                                {
                                                    "source_index": 1,
                                                    "quote": "Passage retrieval works via bi-encoders.",
                                                }
                                            ],
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handle_request)
    client = httpx.Client(base_url="https://api.groq.com/openai/v1", transport=transport)
    provider = GroqSemanticVerificationProvider(groq_cfg, ver_cfg, client=client)

    res = provider.assess("question", "answer [1].", sample_papers)
    assert "claims" in res
    assert len(res["claims"]) == 1
    assert provider.is_available() is True


def test_groq_verification_provider_api_error(sample_papers: list[Paper]) -> None:
    groq_cfg = GroqConfig(
        base_url="https://api.groq.com/openai/v1",
        api_key="fake-key",
        model_name="qwen-2.5-32b",
        timeout_seconds=30.0,
        max_output_tokens=1024,
    )
    ver_cfg = VerificationConfig(
        enabled=True,
        provider="groq",
        model_name="qwen-2.5-32b",
        timeout_seconds=30.0,
        fail_closed=False,
        max_repair_attempts=1,
    )

    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"Service Unavailable")

    transport = httpx.MockTransport(handle_request)
    client = httpx.Client(base_url="https://api.groq.com/openai/v1", transport=transport)
    provider = GroqSemanticVerificationProvider(groq_cfg, ver_cfg, client=client)

    with pytest.raises(SemanticVerificationUnavailable):
        provider.assess("question", "answer [1].", sample_papers)
    assert provider.is_available() is False
