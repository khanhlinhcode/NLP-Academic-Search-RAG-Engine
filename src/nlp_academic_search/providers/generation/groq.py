"""Lightweight Groq Chat Completions provider with bounded streaming retries."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from nlp_academic_search.config import GroqConfig
from nlp_academic_search.providers.generation.base import (
    GenerationInvalidResponseError,
    GenerationRateLimitedError,
    GenerationTimeoutError,
    ModelUnavailableError,
    RAGGenerationError,
)


class GroqGenerationProvider:
    provider_name = "groq"

    def __init__(
        self,
        config: GroqConfig,
        *,
        client: httpx.Client | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model_name = config.model_name
        self.base_url = config.base_url
        self.timeout_seconds = config.timeout_seconds
        self.max_output_tokens = config.max_output_tokens
        self.reasoning_effort = config.reasoning_effort
        self._async_transport = async_transport
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.base_url,
            headers=self._headers(config.api_key),
            timeout=httpx.Timeout(self.timeout_seconds, connect=5.0),
        )
        self._api_key = config.api_key

    @staticmethod
    def _headers(api_key: str | None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _payload(
        self, messages: list[dict[str, str]], temperature: float, *, stream: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": self.max_output_tokens,
            "stream": stream,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    @staticmethod
    def _retry_after(response: httpx.Response) -> float | None:
        value = response.headers.get("retry-after")
        try:
            return max(0.0, float(value)) if value is not None else None
        except ValueError:
            return None

    def _map_http_error(self, exc: Exception) -> RAGGenerationError:
        if isinstance(exc, httpx.TimeoutException):
            return GenerationTimeoutError("Generation provider timed out")
        if isinstance(exc, httpx.ConnectError):
            return ModelUnavailableError("Generation provider is unavailable")
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status == 429:
                return GenerationRateLimitedError(
                    "Generation provider rate limit exceeded",
                    retry_after=self._retry_after(exc.response),
                )
            if status in {401, 403}:
                return ModelUnavailableError("Generation provider authentication failed")
            if status in {404, 502, 503, 504}:
                return ModelUnavailableError("Generation provider or model is unavailable")
        return RAGGenerationError("Generation provider request failed")

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        try:
            response = self._client.post(
                "/chat/completions",
                json=self._payload(messages, temperature, stream=False),
                headers=self._headers(self._api_key),
            )
            response.raise_for_status()
            payload = response.json()
            answer = payload["choices"][0]["message"]["content"].strip()
            if not answer:
                raise GenerationInvalidResponseError("Generation provider returned an empty answer")
            return answer
        except RAGGenerationError:
            raise
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as exc:
            raise GenerationInvalidResponseError(
                "Generation provider returned an invalid response"
            ) from exc
        except httpx.HTTPError as exc:
            raise self._map_http_error(exc) from exc

    async def generate_stream_async(
        self, messages: list[dict[str, str]], temperature: float = 0.2
    ) -> AsyncGenerator[str, None]:
        emitted = False
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(
                    base_url=self.base_url,
                    headers=self._headers(self._api_key),
                    timeout=httpx.Timeout(self.timeout_seconds, connect=5.0),
                    transport=self._async_transport,
                ) as client:
                    async with client.stream(
                        "POST",
                        "/chat/completions",
                        json=self._payload(messages, temperature, stream=True),
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line or line.startswith(":"):
                                continue
                            if not line.startswith("data:"):
                                continue
                            raw = line.removeprefix("data:").strip()
                            if raw == "[DONE]":
                                return
                            try:
                                chunk = json.loads(raw)
                                token = chunk["choices"][0]["delta"].get("content") or ""
                            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                                raise GenerationInvalidResponseError(
                                    "Generation provider returned an invalid stream"
                                ) from exc
                            if token:
                                emitted = True
                                yield token
                return
            except asyncio.CancelledError:
                raise
            except GenerationInvalidResponseError:
                raise
            except httpx.HTTPError as exc:
                mapped = self._map_http_error(exc)
                retryable = isinstance(exc, (httpx.ConnectError, httpx.TimeoutException)) or (
                    isinstance(exc, httpx.HTTPStatusError)
                    and exc.response.status_code in {502, 503, 504}
                )
                if attempt == 0 and not emitted and retryable:
                    await asyncio.sleep(0.25)
                    continue
                raise mapped from exc

    def is_available(self) -> bool:
        if not self._api_key:
            return False
        try:
            response = self._client.get(
                "/models", timeout=5.0, headers=self._headers(self._api_key)
            )
            response.raise_for_status()
            models = response.json().get("data", [])
            return any(item.get("id") == self.model_name for item in models)
        except (httpx.HTTPError, ValueError, TypeError):
            return False

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
