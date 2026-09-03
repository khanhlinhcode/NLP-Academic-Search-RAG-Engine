from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from streamlit.testing.v1 import AppTest

from nlp_academic_search.api.main import create_app


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.integration
def test_streamlit_search_and_ask_flows(services, monkeypatch):
    monkeypatch.setattr(services, "ollama_available", lambda: True)
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(services), host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    monkeypatch.setenv("API_BASE_URL", f"http://127.0.0.1:{port}")
    try:
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
    finally:
        server.should_exit = True
        thread.join(timeout=5)
