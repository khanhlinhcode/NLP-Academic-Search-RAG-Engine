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
        assert payload["metadata"]["prompt_version"] == "academic-grounding-v2"
        assert payload["metadata"]["citation_validation"]["valid"] is True
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
        assert "event: stage" in body


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
