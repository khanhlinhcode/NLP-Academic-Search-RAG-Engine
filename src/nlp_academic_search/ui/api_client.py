"""Small typed HTTP client used by the Streamlit interface."""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import httpx


class APIError(RuntimeError):
    """A recoverable problem while communicating with the FastAPI service."""


class AcademicSearchClient:
    """Client for search, health, statistics, and streaming RAG endpoints."""

    def __init__(
        self,
        base_url: str,
        *,
        api_token: str | None = None,
        timeout: float = 300.0,
        transport: Any = None,
        client: Any = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_token}"} if api_token else None
        self._client = client or httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=3.0),
            transport=transport,
        )

    def close(self) -> None:
        """Close pooled HTTP connections."""
        self._client.close()

    def _json_request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self._client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError as exc:
            hint = (
                "Start it with `make api`."
                if "localhost" in self.base_url or "127.0.0.1" in self.base_url
                else "The free backend may be waking up; retry shortly and verify API_BASE_URL."
            )
            raise APIError(f"FastAPI is not reachable at {self.base_url}. {hint}") from exc
        except httpx.TimeoutException as exc:
            message = "The API timed out while loading local models or search results."
            raise APIError(message) from exc
        except httpx.HTTPStatusError as exc:
            detail = _response_detail(exc.response)
            if exc.response.status_code in {401, 403}:
                raise APIError("Backend authentication failed. Verify BACKEND_API_TOKEN.") from exc
            raise APIError(f"The API returned {exc.response.status_code}: {detail}") from exc
        except (json.JSONDecodeError, ValueError) as exc:
            raise APIError("The API returned an unreadable response.") from exc

    def health(self) -> dict:
        return self._json_request("GET", "/health")

    def stats(self) -> dict:
        return self._json_request("GET", "/stats")

    def search(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 10,
        *,
        category: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        author: str | None = None,
        source: str | None = None,
        offset: int = 0,
    ) -> dict:
        path = "/search" if method == "hybrid" else f"/search/{method}"
        params = {
            "q": query,
            "top_k": top_k,
            "category": category,
            "year_from": year_from,
            "year_to": year_to,
            "author": author,
            "source": source,
            "offset": offset if offset else None,
        }
        return self._json_request(
            "GET",
            path,
            params={key: value for key, value in params.items() if value not in (None, "")},
        )

    def stream_answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        use_reranker: bool = False,
    ) -> Generator[dict, None, None]:
        """Yield decoded SSE events from the streaming RAG endpoint."""
        payload = {
            "question": question,
            "top_k": top_k,
            "use_reranker": use_reranker,
        }
        try:
            with self._client.stream("POST", "/ask/stream", json=payload) as response:
                response.raise_for_status()
                yield from _iter_sse(response.iter_lines())
        except httpx.ConnectError as exc:
            hint = (
                "Start it with `make api`."
                if "localhost" in self.base_url or "127.0.0.1" in self.base_url
                else "The free backend may be waking up; retry shortly and verify API_BASE_URL."
            )
            raise APIError(f"FastAPI is not reachable at {self.base_url}. {hint}") from exc
        except httpx.TimeoutException as exc:
            raise APIError(
                "RAG generation timed out. Check the generation provider and retry."
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = _response_detail(exc.response)
            if exc.response.status_code in {401, 403}:
                raise APIError("Backend authentication failed. Verify BACKEND_API_TOKEN.") from exc
            raise APIError(f"The API returned {exc.response.status_code}: {detail}") from exc


def _iter_sse(lines) -> Generator[dict, None, None]:
    """Parse an iterable of SSE lines into event dictionaries."""
    event_name = "message"
    data_lines: list[str] = []

    def decode_event() -> dict | None:
        if not data_lines:
            return None
        raw_data = "\n".join(data_lines)
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            data = {"text": raw_data}
        return {"event": event_name, "data": data}

    for raw_line in lines:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if line == "":
            decoded = decode_event()
            if decoded is not None:
                yield decoded
            event_name = "message"
            data_lines = []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    decoded = decode_event()
    if decoded is not None:
        yield decoded


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict) and error.get("message"):
                return str(error["message"])
            return str(payload.get("detail", payload))
    except (json.JSONDecodeError, ValueError):
        pass
    return response.text[:240] or "unknown error"
