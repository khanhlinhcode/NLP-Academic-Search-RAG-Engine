import pytest
from fastapi.testclient import TestClient

from nlp_academic_search.api.main import create_app
from nlp_academic_search.config import settings
from nlp_academic_search.providers.verification.base import (
    SemanticVerificationInvalidRequest,
    SemanticVerificationTimeout,
)
from nlp_academic_search.rag.generator import ModelUnavailableError
from nlp_academic_search.rag.verification import SemanticValidation


def test_health_search_and_cors(services, monkeypatch):
    monkeypatch.setattr(services, "ollama_available", lambda: True)
    with TestClient(create_app(services)) as client:
        assert client.get("/health/live").json()["status"] == "alive"
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        response = client.get(
            "/api/v1/search",
            params={"q": "attention", "top_k": 2},
            headers={"Origin": "http://localhost:8501"},
        )
        assert response.status_code == 200
        assert response.json()["results"][0]["score_type"] == "rrf_score"
        assert response.headers["access-control-allow-origin"] == "http://localhost:8501"
        assert response.headers["x-request-id"]


def test_untrusted_origin_is_not_allowed(services, monkeypatch):
    monkeypatch.setattr(services, "ollama_available", lambda: True)
    with TestClient(create_app(services)) as client:
        response = client.get("/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in response.headers


def test_ask_response_has_grounding_metadata(services, monkeypatch):
    monkeypatch.setattr(services, "ollama_available", lambda: True)
    with TestClient(create_app(services)) as client:
        response = client.post(
            "/api/v1/ask",
            json={"question": "How does attention replace recurrence?", "top_k": 1},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["metadata"]["prompt_version"] == "academic-grounding-v3"
        assert payload["metadata"]["citation_validation"]["valid"] is True
        assert payload["metadata"]["citation_repair_attempted"] is False
        assert payload["sources"][0]["source_url"].startswith("https://arxiv.org/")


def test_sse_event_order_and_metadata(services, monkeypatch):
    monkeypatch.setattr(services, "ollama_available", lambda: True)
    with TestClient(create_app(services)) as client:
        response = client.post(
            "/ask/stream",
            json={"question": "What evidence supports retrieval?", "top_k": 1},
        )
        assert response.status_code == 200
        body = response.text
        assert body.index("event: sources") < body.index("event: token") < body.index("event: done")
        assert '"name": "citation_validation"' in body
        assert body.rindex('"status": "complete"') < body.index("event: done")


class RepairingGenerator:
    model_name = "repairing-model"

    def __init__(self) -> None:
        self.sync_calls = 0
        self.async_calls = 0

    def generate(self, messages, temperature=0.2):
        self.sync_calls += 1
        if self.sync_calls == 1:
            return "Novelty rewards unseen results. It adds retrieval value [1]."
        assert temperature == 0.0
        assert "citation-only editor" in messages[0]["content"]
        return "Novelty rewards unseen results [1]. It adds retrieval value [1]."

    async def generate_stream_async(self, messages, temperature=0.2):
        self.async_calls += 1
        if self.async_calls == 1:
            yield "Novelty rewards unseen results. "
            yield "It adds retrieval value [1]."
        else:
            assert temperature == 0.0
            assert "citation-only editor" in messages[0]["content"]
            yield "Novelty rewards unseen results [1]. "
            yield "It adds retrieval value [1]."

    def is_available(self):
        return True

    def close(self):
        return None


class CountingValidGenerator(RepairingGenerator):
    def generate(self, messages, temperature=0.2):
        del messages, temperature
        self.sync_calls += 1
        return "Novelty rewards unseen results [1]."

    async def generate_stream_async(self, messages, temperature=0.2):
        del messages, temperature
        self.async_calls += 1
        yield "Novelty rewards unseen results [1]."


class IncompleteRepairGenerator(RepairingGenerator):
    def generate(self, messages, temperature=0.2):
        del messages, temperature
        self.sync_calls += 1
        return "Novelty rewards unseen results. It adds retrieval value [1]."


class SemanticRepairGenerator(RepairingGenerator):
    def generate(self, messages, temperature=0.2):
        self.sync_calls += 1
        return (
            "The Transformer uses attention instead of recurrence [1]."
            if self.sync_calls > 1
            else "The Transformer is always perfect [1]."
        )

    async def generate_stream_async(self, messages, temperature=0.2):
        self.async_calls += 1
        yield (
            "The Transformer uses attention instead of recurrence [1]."
            if self.async_calls > 1
            else "The Transformer is always perfect [1]."
        )


class SequencedVerifier:
    provider_name = "groq"
    model_name = "verifier-model"
    verifier_independent = True

    def __init__(self, outcomes: list[bool]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def verify(self, answer, sources, question):
        del answer, sources, question
        valid = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        return SemanticValidation(
            valid=valid,
            total_factual_claims=1,
            supported_claim_count=int(valid),
            unsupported_claim_count=int(not valid),
            insufficient_claim_count=0,
            semantic_claim_coverage=float(valid),
            evidence_quote_validity=float(valid),
            verifier_provider=self.provider_name,
            verifier_model=self.model_name,
            verifier_independent=True,
        )

    def is_available(self):
        return True

    def close(self):
        return None


class TimeoutVerifier(SequencedVerifier):
    def verify(self, answer, sources, question):
        del answer, sources, question
        self.calls += 1
        raise SemanticVerificationTimeout("verification timeout")

    def is_available(self):
        return False


class InvalidRequestVerifier(SequencedVerifier):
    def verify(self, answer, sources, question):
        del answer, sources, question
        self.calls += 1
        raise SemanticVerificationInvalidRequest(
            "verification request rejected",
            provider_http_status=400,
            provider_request_id="groq-request-123",
        )


def enable_semantic_verification(monkeypatch, *, fail_closed=True):
    monkeypatch.setattr(settings, "semantic_verification_enabled", True)
    monkeypatch.setattr(settings, "verification_provider", "groq")
    monkeypatch.setattr(settings, "verification_fail_closed", fail_closed)
    monkeypatch.setattr(settings, "max_rag_repair_attempts", 1)


def test_sync_answer_repairs_citations_once(services, monkeypatch):
    generator = RepairingGenerator()
    services.rag_generator = generator  # type: ignore[assignment]
    monkeypatch.setattr(services, "ollama_available", lambda: True)

    with TestClient(create_app(services)) as client:
        response = client.post(
            "/api/v1/ask",
            json={
                "question": ("Why is novelty-based evaluation useful in information retrieval?"),
                "top_k": 1,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert generator.sync_calls == 2
    assert payload["answer"].count("[1]") == 2
    assert payload["metadata"]["citation_repair_attempted"] is True
    assert payload["metadata"]["citation_repair_succeeded"] is True
    validation = payload["metadata"]["citation_validation"]
    assert validation["valid"] is True
    assert validation["uncited_claim_count"] == 0
    assert validation["claim_citation_coverage"] == 1.0
    assert validation["citation_precision"] == 1.0
    assert validation["invalid_indices"] == []


def test_valid_sync_answer_does_not_call_repair(services, monkeypatch):
    generator = CountingValidGenerator()
    services.rag_generator = generator  # type: ignore[assignment]
    monkeypatch.setattr(services, "ollama_available", lambda: True)

    with TestClient(create_app(services)) as client:
        response = client.post(
            "/api/v1/ask",
            json={"question": "Why is novelty evaluation useful?", "top_k": 1},
        )

    assert response.status_code == 200
    assert generator.sync_calls == 1
    metadata = response.json()["metadata"]
    assert metadata["citation_repair_attempted"] is False
    assert metadata["citation_repair_succeeded"] is None


def test_incomplete_repair_stops_after_one_pass_and_returns_refusal(services, monkeypatch):
    generator = IncompleteRepairGenerator()
    services.rag_generator = generator  # type: ignore[assignment]
    monkeypatch.setattr(services, "ollama_available", lambda: True)

    with TestClient(create_app(services)) as client:
        response = client.post(
            "/api/v1/ask",
            json={"question": "Why is novelty evaluation useful?", "top_k": 1},
        )

    assert response.status_code == 200
    assert generator.sync_calls == 2
    metadata = response.json()["metadata"]
    assert metadata["citation_repair_attempted"] is True
    assert metadata["citation_repair_succeeded"] is False
    assert metadata["initial_citation_validation"]["valid"] is False
    assert metadata["citation_validation"]["valid"] is True
    assert metadata["answer_status"] == "refused_unverified"
    assert metadata["final_answer_replaced"] is True
    assert response.json()["answer"] == "Not enough verified evidence in the retrieved sources."


def test_stream_repairs_once_and_replaces_draft_before_done(services, monkeypatch):
    generator = RepairingGenerator()
    services.rag_generator = generator  # type: ignore[assignment]
    monkeypatch.setattr(services, "ollama_available", lambda: True)

    with TestClient(create_app(services)) as client:
        response = client.post(
            "/api/v1/ask/stream",
            json={"question": "Why is novelty evaluation useful?", "top_k": 1},
        )

    body = response.text
    assert response.status_code == 200
    assert generator.async_calls == 2
    assert body.index("event: sources") < body.index("event: token")
    assert body.index('"name": "citation_validation"') < body.index('"name": "citation_repair"')
    assert body.index("event: answer_replacement") < body.index("event: done")
    assert '"citation_repair_attempted": true' in body
    assert '"claim_citation_coverage": 1.0' in body


def test_semantic_failure_repairs_once_then_becomes_verified(services, monkeypatch):
    enable_semantic_verification(monkeypatch)
    generator = SemanticRepairGenerator()
    verifier = SequencedVerifier([False, True])
    services.rag_generator = generator  # type: ignore[assignment]
    services.semantic_verifier = verifier  # type: ignore[assignment]

    with TestClient(create_app(services)) as client:
        response = client.post(
            "/api/v1/ask",
            json={"question": "How does the Transformer avoid recurrence?", "top_k": 1},
        )

    assert response.status_code == 200
    payload = response.json()
    assert generator.sync_calls == 2
    assert verifier.calls == 2
    assert payload["metadata"]["answer_status"] == "verified"
    assert payload["metadata"]["semantic_verification_attempted"] is True
    assert payload["metadata"]["semantic_verification_succeeded"] is True
    assert payload["metadata"]["semantic_validation"]["valid"] is True
    assert payload["metadata"]["semantic_validation"]["semantic_claim_coverage"] == 1.0
    assert payload["metadata"]["semantic_validation"]["evidence_quote_validity"] == 1.0
    assert payload["metadata"]["semantic_validation"]["unsupported_claim_count"] == 0
    assert payload["metadata"]["failure_reason"] is None
    assert payload["metadata"]["initial_semantic_validation"]["valid"] is False
    assert payload["metadata"]["final_answer_replaced"] is True
    latencies = payload["metadata"]["latencies"]
    assert latencies["total_ms"] == pytest.approx(
        latencies["retrieval_ms"] + latencies["generation_ms"] + latencies["verification_ms"],
        abs=0.03,
    )


def test_semantic_failure_after_one_repair_is_withheld(services, monkeypatch):
    enable_semantic_verification(monkeypatch)
    generator = SemanticRepairGenerator()
    verifier = SequencedVerifier([False, False])
    services.rag_generator = generator  # type: ignore[assignment]
    services.semantic_verifier = verifier  # type: ignore[assignment]

    with TestClient(create_app(services)) as client:
        response = client.post(
            "/api/v1/ask",
            json={"question": "How does the Transformer avoid recurrence?", "top_k": 1},
        )

    payload = response.json()
    assert generator.sync_calls == 2
    assert verifier.calls == 2
    assert payload["answer"] == "Not enough verified evidence in the retrieved sources."
    assert payload["metadata"]["answer_status"] == "refused_unverified"
    assert payload["metadata"]["citation_repair_succeeded"] is False


def test_verifier_timeout_fails_closed_without_wasting_repair(services, monkeypatch):
    enable_semantic_verification(monkeypatch)
    generator = CountingValidGenerator()
    verifier = TimeoutVerifier([False])
    services.rag_generator = generator  # type: ignore[assignment]
    services.semantic_verifier = verifier  # type: ignore[assignment]

    with TestClient(create_app(services)) as client:
        response = client.post(
            "/api/v1/ask",
            json={"question": "How does attention replace recurrence?", "top_k": 1},
        )

    metadata = response.json()["metadata"]
    assert generator.sync_calls == 1
    assert verifier.calls == 1
    assert metadata["answer_status"] == "refused_unverified"
    assert metadata["semantic_verification_attempted"] is True
    assert metadata["semantic_verification_succeeded"] is False
    assert metadata["failure_reason"] == "SemanticVerificationTimeout"


def test_verifier_invalid_request_is_non_retryable_and_observable(services, monkeypatch, caplog):
    enable_semantic_verification(monkeypatch)
    monkeypatch.setattr(settings, "backend_api_token", "backend-secret")
    generator = CountingValidGenerator()
    verifier = InvalidRequestVerifier([False])
    services.rag_generator = generator  # type: ignore[assignment]
    services.semantic_verifier = verifier  # type: ignore[assignment]

    with caplog.at_level("INFO", logger="academic_search.rag"):
        with TestClient(create_app(services)) as client:
            response = client.post(
                "/api/v1/ask",
                json={"question": "How does attention replace recurrence?", "top_k": 1},
                headers={"Authorization": "Bearer backend-secret"},
            )

    metadata = response.json()["metadata"]
    assert generator.sync_calls == 1
    assert verifier.calls == 1
    assert metadata["answer_status"] == "refused_unverified"
    assert metadata["semantic_verification_succeeded"] is False
    assert metadata["citation_repair_attempted"] is False
    assert metadata["failure_reason"] == "SemanticVerificationInvalidRequest"
    assert metadata["verification_provider_http_status"] == 400
    assert metadata["verification_provider_request_id"] == "groq-request-123"
    logs = caplog.text
    assert '"provider_http_status": 400' in logs
    assert '"provider_error_category": "SemanticVerificationInvalidRequest"' in logs
    assert "backend-secret" not in logs


def test_verifier_invalid_request_streams_final_refusal_metadata_without_repair(
    services, monkeypatch
):
    enable_semantic_verification(monkeypatch)
    generator = CountingValidGenerator()
    verifier = InvalidRequestVerifier([False])
    services.rag_generator = generator  # type: ignore[assignment]
    services.semantic_verifier = verifier  # type: ignore[assignment]

    with TestClient(create_app(services)) as client:
        response = client.post(
            "/api/v1/ask/stream",
            json={"question": "How does attention replace recurrence?", "top_k": 1},
        )

    body = response.text
    assert generator.async_calls == 1
    assert verifier.calls == 1
    assert "event: answer_replacement" in body
    assert "event: citation_repair" not in body
    assert '"failure_reason": "SemanticVerificationInvalidRequest"' in body
    assert '"citation_repair_attempted": false' in body
    assert body.index("event: answer_replacement") < body.index("event: done")


def test_semantic_sse_order_replaces_draft_before_final_done(services, monkeypatch):
    enable_semantic_verification(monkeypatch)
    generator = SemanticRepairGenerator()
    verifier = SequencedVerifier([False, True])
    services.rag_generator = generator  # type: ignore[assignment]
    services.semantic_verifier = verifier  # type: ignore[assignment]

    with TestClient(create_app(services)) as client:
        response = client.post(
            "/api/v1/ask/stream",
            json={"question": "How does the Transformer avoid recurrence?", "top_k": 1},
        )

    body = response.text
    positions = [
        body.index("event: sources"),
        body.index("event: token"),
        body.index('"name": "structural_validation"'),
        body.index('"name": "semantic_validation"'),
        body.index('"name": "answer_repair", "status": "running"'),
        body.index("event: answer_replacement"),
        body.index('"name": "final_validation"'),
        body.index('"name": "generation", "status": "complete"'),
        body.index("event: done"),
    ]
    assert positions == sorted(positions)
    assert '"answer_status": "verified"' in body


def test_validation_error_is_structured(services):
    with TestClient(create_app(services)) as client:
        response = client.post("/ask", json={"question": "x"})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"


class UnavailableGenerator:
    model_name = "missing-model"

    def generate(self, _messages):
        raise ModelUnavailableError("model unavailable")

    def generate_stream(self, _messages):
        raise ModelUnavailableError("model unavailable")
        yield ""  # pragma: no cover

    async def generate_stream_async(self, _messages):
        raise ModelUnavailableError("model unavailable")
        yield ""  # pragma: no cover


def test_generation_errors_have_http_and_sse_contracts(services):
    services.rag_generator = UnavailableGenerator()  # type: ignore[assignment]
    with TestClient(create_app(services)) as client:
        response = client.post(
            "/ask", json={"question": "What evidence discusses attention?", "top_k": 1}
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_unavailable"
        streamed = client.post(
            "/ask/stream", json={"question": "What evidence discusses attention?", "top_k": 1}
        )
        assert streamed.status_code == 200
        assert "event: error" in streamed.text
        assert "model_unavailable" in streamed.text


def test_readiness_fails_when_rag_model_is_unavailable(services, monkeypatch):
    monkeypatch.setattr(services, "ollama_available", lambda: False)
    with TestClient(create_app(services)) as client:
        assert client.get("/health/ready").status_code == 503
        assert client.get("/health").json()["status"] == "degraded"


def test_readiness_reports_required_verifier_unavailable(services, monkeypatch):
    enable_semantic_verification(monkeypatch)
    services.semantic_verifier = TimeoutVerifier([False])  # type: ignore[assignment]
    monkeypatch.setattr(services, "ollama_available", lambda: True)
    with TestClient(create_app(services)) as client:
        response = client.get("/health/ready")
    payload = response.json()
    assert response.status_code == 503
    assert payload["verification_provider"] == "groq"
    assert payload["verification_available"] is False
    assert payload["verification_required"] is True


def test_bearer_auth_protects_api_but_not_liveness(services, monkeypatch):
    monkeypatch.setattr(settings, "backend_api_token", "backend-secret")
    monkeypatch.setattr(services, "ollama_available", lambda: True)
    with TestClient(create_app(services)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/api/v1/search", params={"q": "attention"}).status_code == 401
        authorized = client.get(
            "/api/v1/search",
            params={"q": "attention"},
            headers={"Authorization": "Bearer backend-secret"},
        )
        assert authorized.status_code == 200
