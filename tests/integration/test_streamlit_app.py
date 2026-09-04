from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import uvicorn
from streamlit.testing.v1 import AppTest

from nlp_academic_search.api.main import create_app
from nlp_academic_search.config import settings
from nlp_academic_search.providers.verification.base import SemanticVerificationInvalidRequest
from nlp_academic_search.rag.verification import SemanticValidation
from nlp_academic_search.ui.api_client import AcademicSearchClient


@pytest.fixture(autouse=True)
def _keep_ui_tests_out_of_streamlit_secrets(monkeypatch):
    monkeypatch.setenv("BACKEND_API_TOKEN", "integration-test-token")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def _running_api(services):
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(services), host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    if not server.started:
        raise RuntimeError("test API did not start")
    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _theme_block_count(app: AppTest) -> int:
    return sum("academic-search-theme" in str(item.value) for item in app.markdown)


@pytest.mark.integration
def test_streamlit_reinjects_theme_once_across_mode_switches(services, monkeypatch):
    monkeypatch.setattr(services, "ollama_available", lambda: True)
    with _running_api(services) as port:
        monkeypatch.setenv("API_BASE_URL", f"http://127.0.0.1:{port}")
        app = AppTest.from_file(
            Path(__file__).parents[2] / "scripts" / "streamlit_app.py", default_timeout=10
        ).run()
        assert _theme_block_count(app) == 1
        assert sum(widget.label == "Query" for widget in app.text_input) == 1
        assert app.radio[0].options == ["Search", "Ask"]
        assert any(button.label == "Search papers" for button in app.button)
        assert sum(expander.label == "How this search works" for expander in app.expander) == 1

        app.radio[0].set_value("Ask").run()
        assert _theme_block_count(app) == 1
        assert len(app.text_area) == 1
        assert any(button.label == "Ask" for button in app.button)
        assert (
            sum(expander.label == "How this answer is produced" for expander in app.expander) == 1
        )

        app.radio[0].set_value("Search").run()
        assert _theme_block_count(app) == 1
        assert sum(widget.label == "Query" for widget in app.text_input) == 1
        assert any(button.label == "Search papers" for button in app.button)

        app.radio[0].set_value("Ask").run()

    rendered = [str(item.value) for item in app.markdown]
    assert not app.exception
    assert _theme_block_count(app) == 1
    assert len(app.text_area) == 1
    assert any(button.label == "Ask" for button in app.button)
    assert sum('class="masthead"' in item for item in rendered) == 1
    assert app.session_state["ask_history"] == []


@pytest.mark.integration
def test_streamlit_search_and_ask_flows(services, monkeypatch):
    monkeypatch.setattr(services, "ollama_available", lambda: True)
    with _running_api(services) as port:
        monkeypatch.setenv("API_BASE_URL", f"http://127.0.0.1:{port}")
        app_path = Path(__file__).parents[2] / "scripts" / "streamlit_app.py"
        app = AppTest.from_file(app_path, default_timeout=10).run()
        assert not app.exception
        app.text_input[0].input("attention transformer")
        app.button[0].click().run()
        assert not app.exception
        assert any("Attention Is All You Need" in item.value for item in app.markdown)

        app.radio[0].set_value("Ask").run()
        app.text_area[0].input("How does attention replace recurrence?")
        app.button[0].click().run(timeout=10)
        assert not app.exception
        assert any("Grounded answer" in item.value for item in app.markdown)


class RepairingStreamGenerator:
    model_name = "repairing-stream-model"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, temperature=0.2):
        del messages, temperature
        return "A supported answer [1]."

    async def generate_stream_async(self, messages, temperature=0.2):
        self.calls += 1
        if self.calls == 1:
            yield "The draft describes retrieval systems. "
            yield "The second factual claim is supported [1]."
        else:
            assert temperature == 0.0
            assert "citation-only editor" in messages[0]["content"]
            yield "The repaired factual claim is supported [1]. "
            yield "The second factual claim is supported [1]."

    def is_available(self):
        return True

    def close(self):
        return None


class IncompleteRepairStreamGenerator(RepairingStreamGenerator):
    async def generate_stream_async(self, messages, temperature=0.2):
        del messages, temperature
        self.calls += 1
        yield "The final answer still has an uncited factual claim. "
        yield "Only this factual sentence has support [1]."


class ValidStreamGenerator(RepairingStreamGenerator):
    async def generate_stream_async(self, messages, temperature=0.2):
        del messages, temperature
        self.calls += 1
        yield "The Transformer uses attention instead of recurrence [1]."


class SemanticRepairStreamGenerator(RepairingStreamGenerator):
    async def generate_stream_async(self, messages, temperature=0.2):
        del messages, temperature
        self.calls += 1
        if self.calls == 1:
            yield "The Transformer is always perfect [1]."
        else:
            yield "The Transformer uses attention instead of recurrence [1]."


class SemanticVerifier:
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
            verifier_provider="groq",
            verifier_model=self.model_name,
            verifier_independent=True,
        )

    def is_available(self):
        return True

    def close(self):
        return None


class InvalidRequestSemanticVerifier(SemanticVerifier):
    def verify(self, answer, sources, question):
        del answer, sources, question
        self.calls += 1
        raise SemanticVerificationInvalidRequest(
            "verification request rejected", provider_http_status=400
        )


def enable_verification(monkeypatch, *, fail_closed=True):
    monkeypatch.setattr(settings, "semantic_verification_enabled", True)
    monkeypatch.setattr(settings, "verification_provider", "groq")
    monkeypatch.setattr(settings, "verification_fail_closed", fail_closed)
    monkeypatch.setattr(settings, "max_rag_repair_attempts", 1)


@pytest.mark.integration
def test_streamlit_replaces_uncited_draft_with_structurally_valid_answer(services, monkeypatch):
    generator = RepairingStreamGenerator()
    services.rag_generator = generator  # type: ignore[assignment]
    monkeypatch.setattr(services, "ollama_available", lambda: True)

    with _running_api(services) as port:
        monkeypatch.setenv("API_BASE_URL", f"http://127.0.0.1:{port}")
        app_path = Path(__file__).parents[2] / "scripts" / "streamlit_app.py"
        app = AppTest.from_file(app_path, default_timeout=10).run()
        app.radio[0].set_value("Ask").run()
        app.text_area[0].input("What evidence supports the factual claim?")
        app.button[0].click().run(timeout=10)

    rendered = [str(item.value) for item in app.markdown]
    assert not app.exception
    assert generator.calls == 2
    assert any("The repaired factual claim is supported [1]." in item for item in rendered)
    assert not any("The draft describes retrieval systems." in item for item in rendered)
    assert any("Citation format valid" in item for item in rendered)
    assert not any("Evidence verified" in item for item in rendered)
    assert not any("Running" in item for item in rendered)
    assert not any("Waiting for evidence" in item for item in rendered)
    assert len(app.session_state["ask_history"]) == 1
    assert app.session_state["ask_history"][0]["answer"].startswith("The repaired factual claim")


@pytest.mark.integration
def test_streamlit_treats_stream_without_done_as_interrupted(services, monkeypatch):
    monkeypatch.setattr(services, "ollama_available", lambda: True)

    def truncated_stream(self, question, *, top_k=5, use_reranker=False):
        del self, question, top_k, use_reranker
        yield {"event": "token", "data": {"token": "Unverified draft."}}

    monkeypatch.setattr(AcademicSearchClient, "stream_answer", truncated_stream)
    with _running_api(services) as port:
        monkeypatch.setenv("API_BASE_URL", f"http://127.0.0.1:{port}")
        app = AppTest.from_file(
            Path(__file__).parents[2] / "scripts" / "streamlit_app.py", default_timeout=10
        ).run()
        app.radio[0].set_value("Ask").run()
        app.text_area[0].input("What evidence supports the factual claim?")
        app.button[0].click().run(timeout=10)

    rendered = [str(item.value) for item in app.markdown]
    assert not app.exception
    assert any("Connection" in item and "Interrupted" in item for item in rendered)
    assert not any("Answer ready" in item for item in rendered)
    assert not any("Running" in item for item in rendered)
    assert app.session_state["ask_history"] == []


@pytest.mark.integration
def test_streamlit_does_not_mark_incomplete_repair_as_ready(services, monkeypatch):
    generator = IncompleteRepairStreamGenerator()
    services.rag_generator = generator  # type: ignore[assignment]
    monkeypatch.setattr(services, "ollama_available", lambda: True)

    with _running_api(services) as port:
        monkeypatch.setenv("API_BASE_URL", f"http://127.0.0.1:{port}")
        app_path = Path(__file__).parents[2] / "scripts" / "streamlit_app.py"
        app = AppTest.from_file(app_path, default_timeout=10).run()
        app.radio[0].set_value("Ask").run()
        app.text_area[0].input("What evidence supports the factual claim?")
        app.button[0].click().run(timeout=10)

    rendered = [str(item.value) for item in app.markdown]
    assert not app.exception
    assert generator.calls == 2
    assert any("Not enough verified evidence" in item for item in rendered)
    assert any("Answer withheld" in item for item in rendered)
    assert not any("The final answer still has" in item for item in rendered)
    assert not any("Answer ready" in item for item in rendered)
    assert not any("Running" in item for item in rendered)
    assert not any("Waiting for evidence" in item for item in rendered)
    assert app.session_state["ask_history"][0]["answer"] == (
        "Not enough verified evidence in the retrieved sources."
    )


@pytest.mark.integration
def test_streamlit_marks_only_semantic_valid_answer_as_evidence_verified(services, monkeypatch):
    enable_verification(monkeypatch)
    generator = ValidStreamGenerator()
    verifier = SemanticVerifier([True])
    services.rag_generator = generator  # type: ignore[assignment]
    services.semantic_verifier = verifier  # type: ignore[assignment]
    monkeypatch.setattr(services, "ollama_available", lambda: True)

    with _running_api(services) as port:
        monkeypatch.setenv("API_BASE_URL", f"http://127.0.0.1:{port}")
        app = AppTest.from_file(
            Path(__file__).parents[2] / "scripts" / "streamlit_app.py", default_timeout=10
        ).run()
        app.radio[0].set_value("Ask").run()
        app.text_area[0].input("How does attention replace recurrence?")
        app.button[0].click().run(timeout=10)

    rendered = [str(item.value) for item in app.markdown]
    assert not app.exception
    assert generator.calls == 1
    assert verifier.calls == 1
    assert any("Answer ready" in item and "Evidence verified" in item for item in rendered)
    assert any('<span class="source-title-row">' in item for item in rendered)
    assert not any('<div class="source-title-row">' in item for item in rendered)
    assert any('class="cited-separator"' in item and " · " in item for item in rendered)
    assert not any('aria-hidden="true"> · </span>' in item for item in rendered)
    assert not any('aria-label="Cited source"' in item for item in rendered)
    assert not any(item.strip() in {"</span>", "</div>"} for item in rendered)
    assert not any("Running" in item for item in rendered)
    assert not any("Waiting for evidence" in item for item in rendered)
    assert len(app.session_state["ask_history"]) == 1


@pytest.mark.integration
def test_streamlit_semantic_repair_keeps_only_authoritative_final_answer(services, monkeypatch):
    enable_verification(monkeypatch)
    generator = SemanticRepairStreamGenerator()
    verifier = SemanticVerifier([False, True])
    services.rag_generator = generator  # type: ignore[assignment]
    services.semantic_verifier = verifier  # type: ignore[assignment]
    monkeypatch.setattr(services, "ollama_available", lambda: True)

    with _running_api(services) as port:
        monkeypatch.setenv("API_BASE_URL", f"http://127.0.0.1:{port}")
        app = AppTest.from_file(
            Path(__file__).parents[2] / "scripts" / "streamlit_app.py", default_timeout=10
        ).run()
        app.radio[0].set_value("Ask").run()
        app.text_area[0].input("How does attention replace recurrence?")
        app.button[0].click().run(timeout=10)

    rendered = [str(item.value) for item in app.markdown]
    assert not app.exception
    assert generator.calls == 2
    assert verifier.calls == 2
    assert not any("always perfect" in item for item in rendered)
    assert sum("uses attention instead of recurrence [1]" in item for item in rendered) == 1
    assert app.session_state["ask_history"][0]["metadata"]["answer_status"] == "verified"
    assert not any("Running" in item for item in rendered)
    assert not any("Waiting for evidence" in item for item in rendered)


@pytest.mark.integration
def test_streamlit_renders_insufficient_context_as_terminal_refusal(services, monkeypatch):
    monkeypatch.setattr(services, "ollama_available", lambda: True)
    monkeypatch.setattr(services, "retrieve_for_rag", lambda *_: ([], [], "rrf"))

    with _running_api(services) as port:
        monkeypatch.setenv("API_BASE_URL", f"http://127.0.0.1:{port}")
        app = AppTest.from_file(
            Path(__file__).parents[2] / "scripts" / "streamlit_app.py", default_timeout=10
        ).run()
        app.radio[0].set_value("Ask").run()
        app.text_area[0].input("What evidence supports this claim?")
        app.button[0].click().run(timeout=10)

    rendered = [str(item.value) for item in app.markdown]
    assert not app.exception
    assert any("Not enough evidence in the retrieved sources." in item for item in rendered)
    assert any("Answer withheld" in item for item in rendered)
    assert not any("Answer ready" in item for item in rendered)
    assert not any("Running" in item for item in rendered)
    assert not any("Waiting for evidence" in item for item in rendered)
    assert len(app.session_state["ask_history"]) == 1


@pytest.mark.integration
def test_streamlit_mode_switch_does_not_duplicate_surface_or_history(services, monkeypatch):
    monkeypatch.setattr(services, "ollama_available", lambda: True)
    with _running_api(services) as port:
        monkeypatch.setenv("API_BASE_URL", f"http://127.0.0.1:{port}")
        app = AppTest.from_file(
            Path(__file__).parents[2] / "scripts" / "streamlit_app.py", default_timeout=10
        ).run()
        app.text_input[0].input("attention transformer")
        app.button[0].click().run()
        app.radio[0].set_value("Ask").run()
        app.text_area[0].input("How does attention replace recurrence?")
        app.button[0].click().run(timeout=10)
        app.radio[0].set_value("Search").run()
        app.radio[0].set_value("Ask").run()

    rendered = [str(item.value) for item in app.markdown]
    assert not app.exception
    assert sum('class="masthead"' in item for item in rendered) == 1
    assert sum('class="ledger-heading"' in item for item in rendered) == 1
    assert sum("Grounded answer [1]." in item for item in rendered) == 1
    assert len(app.session_state["ask_history"]) == 1
    assert app.session_state["selected_paper"]["title"] == "Attention Is All You Need"


@pytest.mark.integration
def test_streamlit_warns_when_semantic_verification_is_unavailable(services, monkeypatch):
    enable_verification(monkeypatch, fail_closed=False)
    services.rag_generator = ValidStreamGenerator()  # type: ignore[assignment]
    services.semantic_verifier = None
    monkeypatch.setattr(services, "ollama_available", lambda: True)

    with _running_api(services) as port:
        monkeypatch.setenv("API_BASE_URL", f"http://127.0.0.1:{port}")
        app = AppTest.from_file(
            Path(__file__).parents[2] / "scripts" / "streamlit_app.py", default_timeout=10
        ).run()
        app.radio[0].set_value("Ask").run()
        app.text_area[0].input("How does attention replace recurrence?")
        app.button[0].click().run(timeout=10)

    rendered = [str(item.value) for item in app.markdown]
    assert not app.exception
    assert any("Semantic verification unavailable" in item for item in rendered)
    assert not any("Answer ready" in item for item in rendered)


@pytest.mark.integration
def test_streamlit_never_marks_rejected_verifier_request_as_evidence_verified(
    services, monkeypatch
):
    enable_verification(monkeypatch, fail_closed=True)
    generator = ValidStreamGenerator()
    verifier = InvalidRequestSemanticVerifier([False])
    services.rag_generator = generator  # type: ignore[assignment]
    services.semantic_verifier = verifier  # type: ignore[assignment]
    monkeypatch.setattr(services, "ollama_available", lambda: True)

    with _running_api(services) as port:
        monkeypatch.setenv("API_BASE_URL", f"http://127.0.0.1:{port}")
        app = AppTest.from_file(
            Path(__file__).parents[2] / "scripts" / "streamlit_app.py", default_timeout=10
        ).run()
        app.radio[0].set_value("Ask").run()
        app.text_area[0].input("How does attention replace recurrence?")
        app.button[0].click().run(timeout=10)

    rendered = [str(item.value) for item in app.markdown]
    assert not app.exception
    assert generator.calls == 1
    assert verifier.calls == 1
    assert any("Answer withheld" in item for item in rendered)
    assert any("verifier request was rejected" in item.casefold() for item in rendered)
    assert not any("Evidence verified" in item for item in rendered)


@pytest.mark.integration
def test_streamlit_disables_ask_when_required_verifier_is_unavailable(services, monkeypatch):
    enable_verification(monkeypatch, fail_closed=True)
    services.rag_generator = ValidStreamGenerator()  # type: ignore[assignment]
    services.semantic_verifier = None
    monkeypatch.setattr(services, "ollama_available", lambda: True)

    with _running_api(services) as port:
        monkeypatch.setenv("API_BASE_URL", f"http://127.0.0.1:{port}")
        app = AppTest.from_file(
            Path(__file__).parents[2] / "scripts" / "streamlit_app.py", default_timeout=10
        ).run()
        app.radio[0].set_value("Ask").run()

    assert not app.exception
    assert app.button[0].disabled is True
    assert any(
        "semantic verification is required" in str(message.value).casefold() for message in app.info
    )


@pytest.mark.integration
def test_streamlit_backend_unavailable_state_is_explicit(monkeypatch):
    unavailable_port = _free_port()
    monkeypatch.setenv("API_BASE_URL", f"http://127.0.0.1:{unavailable_port}")
    app = AppTest.from_file(
        Path(__file__).parents[2] / "scripts" / "streamlit_app.py", default_timeout=10
    ).run()

    assert not app.exception
    assert any("FastAPI is not reachable" in str(message.value) for message in app.warning)
    assert any("OFFLINE" in str(markdown.value) for markdown in app.markdown)
