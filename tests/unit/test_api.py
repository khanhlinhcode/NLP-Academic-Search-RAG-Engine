from fastapi.testclient import TestClient

from nlp_academic_search.api.main import create_app
from nlp_academic_search.config import settings
from nlp_academic_search.rag.generator import ModelUnavailableError


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


class IncompleteRepairGenerator(RepairingGenerator):
    def generate(self, messages, temperature=0.2):
        del messages, temperature
        self.sync_calls += 1
        return "Novelty rewards unseen results. It adds retrieval value [1]."


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


def test_incomplete_repair_stops_after_one_pass_and_stays_invalid(services, monkeypatch):
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
    assert metadata["citation_validation"]["valid"] is False
    assert any("review the answer" in warning for warning in metadata["warnings"])


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
