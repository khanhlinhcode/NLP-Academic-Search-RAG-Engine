"""Ollama generation with typed failures and correct chat roles."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Generator
from typing import Any

import httpx

from nlp_academic_search.config import settings
from nlp_academic_search.providers.generation.base import (
    GenerationTimeoutError,
    ModelUnavailableError,
    RAGGenerationError,
)


class RAGGenerator:
    def __init__(
        self,
        model_name: str | None = None,
        *,
        client: Any | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        import ollama

        self._ollama = ollama
        self.model_name = model_name or settings.ollama.model_name
        self.base_url = settings.ollama.base_url
        self.timeout_seconds = settings.ollama.timeout_seconds
        self.async_transport = async_transport
        self.client = client or ollama.Client(host=self.base_url, timeout=self.timeout_seconds)

    def _map_error(self, exc: Exception) -> RAGGenerationError:
        if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
            return GenerationTimeoutError("Ollama generation timed out")
        if isinstance(exc, (httpx.ConnectError, ConnectionError)):
            return ModelUnavailableError("Ollama is unavailable")
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {
            404,
            502,
            503,
        }:
            return ModelUnavailableError(f"Ollama model '{self.model_name}' is unavailable")
        if isinstance(exc, self._ollama.ResponseError) and exc.status_code in {404, 502, 503}:
            return ModelUnavailableError(f"Ollama model '{self.model_name}' is unavailable")
        return RAGGenerationError("Ollama generation failed")

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": temperature, "num_predict": 1024},
            )
            answer = response["message"]["content"].strip()
            if not answer:
                raise RAGGenerationError("Ollama returned an empty answer")
            return answer
        except RAGGenerationError:
            raise
        except Exception as exc:
            raise self._map_error(exc) from exc

    def generate_stream(
        self, messages: list[dict[str, str]], temperature: float = 0.2
    ) -> Generator[str, None, None]:
        try:
            stream = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": temperature, "num_predict": 1024},
                stream=True,
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                if token:
                    yield token
        except Exception as exc:
            raise self._map_error(exc) from exc

    async def generate_stream_async(
        self, messages: list[dict[str, str]], temperature: float = 0.2
    ) -> AsyncGenerator[str, None]:
        """Stream Ollama NDJSON with cancellation-safe connection cleanup."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": 1024},
            "stream": True,
        }
        emitted = False
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(
                    transport=self.async_transport, timeout=self.timeout_seconds
                ) as client:
                    async with client.stream(
                        "POST", f"{self.base_url}/api/chat", json=payload
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            chunk = json.loads(line)
                            if chunk.get("error"):
                                raise RAGGenerationError("Ollama generation failed")
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                emitted = True
                                yield token
                return
            except asyncio.CancelledError:
                raise
            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                transient_status = isinstance(
                    exc, httpx.HTTPStatusError
                ) and exc.response.status_code in {502, 503}
                if (
                    attempt == 0
                    and not emitted
                    and (
                        isinstance(exc, (httpx.ConnectError, httpx.TimeoutException))
                        or transient_status
                    )
                ):
                    await asyncio.sleep(0.25)
                    continue
                raise self._map_error(exc) from exc
            except (json.JSONDecodeError, TypeError, RAGGenerationError) as exc:
                if isinstance(exc, RAGGenerationError):
                    raise
                raise RAGGenerationError("Ollama returned an invalid stream") from exc

    def is_available(self) -> bool:
        try:
            models = self.client.list()
            names = [getattr(model, "model", "") for model in models.models]
            return any(
                self.model_name == name or name.startswith(f"{self.model_name}:") for name in names
            )
        except Exception:
            return False

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    provider_name = "ollama"
