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
            yield "The draft makes an uncited factual claim. "
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


@pytest.mark.integration
def test_streamlit_replaces_uncited_draft_with_verified_answer(services, monkeypatch):
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
    assert not any("The draft makes an uncited factual claim." in item for item in rendered)
    assert any("Citations verified" in item for item in rendered)


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
    assert any("Needs citation review" in item for item in rendered)
    assert not any("Answer ready" in item for item in rendered)
