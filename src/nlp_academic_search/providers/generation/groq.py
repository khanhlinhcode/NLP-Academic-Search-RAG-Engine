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

    @staticmethod
    def _request_id(response: httpx.Response | None) -> str | None:
        if response is None:
            return None
        return response.headers.get("x-request-id") or response.headers.get("x-groq-request-id")

    def _map_http_error(
        self,
        exc: Exception,
        *,
        tokens_emitted: bool = False,
    ) -> RAGGenerationError:
        if isinstance(exc, httpx.TimeoutException):
            return GenerationTimeoutError(
                "Generation provider timed out",
                tokens_emitted=tokens_emitted,
            )
        if isinstance(exc, httpx.ConnectError):
            return ModelUnavailableError(
                "Generation provider is unavailable",
                tokens_emitted=tokens_emitted,
            )
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            request_id = self._request_id(exc.response)
            retry_after = self._retry_after(exc.response)
            if status == 429:
                return GenerationRateLimitedError(
                    "Generation provider rate limit exceeded",
                    retry_after=retry_after,
                    provider_http_status=status,
                    provider_request_id=request_id,
                    tokens_emitted=tokens_emitted,
                )
            if status in {401, 403}:
                return ModelUnavailableError(
                    "Generation provider authentication failed",
                    provider_http_status=status,
                    provider_request_id=request_id,
                    tokens_emitted=tokens_emitted,
                )
            if status in {404, 502, 503, 504}:
                return ModelUnavailableError(
                    "Generation provider or model is unavailable",
                    provider_http_status=status,
                    provider_request_id=request_id,
                    tokens_emitted=tokens_emitted,
                )
            return RAGGenerationError(
                f"Generation provider request failed with status {status}",
                provider_http_status=status,
                provider_request_id=request_id,
                retry_after=retry_after,
                tokens_emitted=tokens_emitted,
            )
        return RAGGenerationError(
            "Generation provider request failed",
            tokens_emitted=tokens_emitted,
        )

    def generate(self, messages: list[dict[str, str]], temperature: float = 0.2) -> str:
        response: httpx.Response | None = None
        try:
            response = self._client.post(
                "/chat/completions",
                json=self._payload(messages, temperature, stream=False),
                headers=self._headers(self._api_key),
            )
            response.raise_for_status()
            payload = response.json()
            choices = payload.get("choices") or []
            if not choices:
                raise GenerationInvalidResponseError(
                    "Generation provider returned empty choices",
                    provider_http_status=response.status_code,
                    provider_request_id=self._request_id(response),
                    output_token_usage=payload.get("usage"),
                )
            choice = choices[0]
            finish_reason = choice.get("finish_reason")
            usage = payload.get("usage")
            message = choice.get("message") or {}
            content = message.get("content") or ""
            answer = content.strip()
            if finish_reason == "length":
                raise GenerationInvalidResponseError(
                    "Generation provider truncated response (finish_reason=length)",
                    provider_http_status=response.status_code,
                    provider_request_id=self._request_id(response),
                    finish_reason="length",
                    output_token_usage=usage,
                )
            if not answer:
                raise GenerationInvalidResponseError(
                    "Generation provider returned an empty answer",
                    provider_http_status=response.status_code,
                    provider_request_id=self._request_id(response),
                    finish_reason=finish_reason,
                    output_token_usage=usage,
                )
            return answer
        except RAGGenerationError:
            raise
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as exc:
            raise GenerationInvalidResponseError(
                "Generation provider returned an invalid response",
                provider_http_status=response.status_code if response is not None else None,
                provider_request_id=self._request_id(response),
            ) from exc
        except httpx.HTTPError as exc:
            raise self._map_http_error(exc) from exc

    async def generate_stream_async(
        self, messages: list[dict[str, str]], temperature: float = 0.2
    ) -> AsyncGenerator[str, None]:
        emitted = False
        last_finish_reason: str | None = None
        last_usage: dict[str, Any] | None = None
        for attempt in range(2):
            try:
                completed_done = False
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
                        request_id = self._request_id(response)
                        async for line in response.aiter_lines():
                            if not line or line.startswith(":"):
                                continue
                            if not line.startswith("data:"):
                                continue
                            raw = line.removeprefix("data:").strip()
                            if raw == "[DONE]":
                                completed_done = True
                                return
                            try:
                                chunk = json.loads(raw)
                                if chunk.get("usage"):
                                    last_usage = chunk["usage"]
                                choices = chunk.get("choices") or []
                                if not choices:
                                    continue
                                choice = choices[0]
                                finish_reason = choice.get("finish_reason")
                                if finish_reason:
                                    last_finish_reason = finish_reason
                                if finish_reason == "length":
                                    raise GenerationInvalidResponseError(
                                        "Generation provider truncated stream (finish_reason=length)",
                                        provider_http_status=response.status_code,
                                        provider_request_id=request_id,
                                        finish_reason="length",
                                        output_token_usage=last_usage,
                                        tokens_emitted=emitted,
                                    )
                                delta = choice.get("delta") or {}
                                token = delta.get("content") or ""
                            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                                raise GenerationInvalidResponseError(
                                    "Generation provider returned an invalid stream",
                                    provider_http_status=response.status_code,
                                    provider_request_id=request_id,
                                    finish_reason=last_finish_reason,
                                    output_token_usage=last_usage,
                                    tokens_emitted=emitted,
                                ) from exc
                            if token:
                                emitted = True
                                yield token

                        if not completed_done:
                            raise GenerationInvalidResponseError(
                                "Generation stream closed prematurely without [DONE]",
                                provider_http_status=response.status_code,
                                provider_request_id=request_id,
                                finish_reason=last_finish_reason,
                                output_token_usage=last_usage,
                                tokens_emitted=emitted,
                            )
                return
            except asyncio.CancelledError:
                raise
            except GenerationInvalidResponseError:
                raise
            except httpx.HTTPError as exc:
                mapped = self._map_http_error(exc, tokens_emitted=emitted)
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
