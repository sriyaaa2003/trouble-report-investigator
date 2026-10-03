"""LLM provider clients (Anthropic Messages API and OpenAI-compatible chat completions).

Both are thin httpx wrappers returning the text plus provider-reported token usage, so cost is
computed from what the provider actually billed, not from an estimate.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Protocol

import httpx

from investigator.settings import Settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int


class LLMError(Exception):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class ProviderConfig:
    kind: str
    base_url: str
    api_key_env: str = ""
    anthropic_version: str = "2023-06-01"


class LLMClient(Protocol):
    async def complete(
        self, *, model: str, system: str, user: str, max_output_tokens: int, timeout_s: float
    ) -> LLMResult: ...


_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code < 400:
        return
    detail = resp.text[:300]
    raise LLMError(
        f"provider returned HTTP {resp.status_code}: {detail}",
        retryable=resp.status_code in _RETRYABLE_STATUS,
    )


class AnthropicClient:
    def __init__(self, cfg: ProviderConfig, http: httpx.AsyncClient) -> None:
        self._cfg = cfg
        self._http = http
        self._key = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else ""
        if not self._key:
            raise RuntimeError(f"environment variable {cfg.api_key_env!r} is not set")

    async def complete(
        self, *, model: str, system: str, user: str, max_output_tokens: int, timeout_s: float
    ) -> LLMResult:
        try:
            resp = await self._http.post(
                self._cfg.base_url.rstrip("/") + "/v1/messages",
                headers={
                    "x-api-key": self._key,
                    "anthropic-version": self._cfg.anthropic_version,
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": max_output_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
                timeout=timeout_s,
            )
        except httpx.TimeoutException as e:
            raise LLMError(f"timeout after {timeout_s}s", retryable=True) from e
        except httpx.TransportError as e:
            raise LLMError(f"network error: {e}", retryable=True) from e
        _raise_for_status(resp)
        body = resp.json()
        text = "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
        usage = body.get("usage", {})
        return LLMResult(text, int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)))


class OpenAICompatClient:
    """Works with OpenAI, Ollama (`/v1`), vLLM, LM Studio and other compatible servers."""

    def __init__(self, cfg: ProviderConfig, http: httpx.AsyncClient) -> None:
        self._cfg = cfg
        self._http = http
        self._key = os.environ.get(cfg.api_key_env, "") if cfg.api_key_env else ""
        if cfg.api_key_env and not self._key:
            raise RuntimeError(f"environment variable {cfg.api_key_env!r} is not set")

    async def complete(
        self, *, model: str, system: str, user: str, max_output_tokens: int, timeout_s: float
    ) -> LLMResult:
        headers = {"content-type": "application/json"}
        if self._key:
            headers["authorization"] = f"Bearer {self._key}"
        try:
            resp = await self._http.post(
                self._cfg.base_url.rstrip("/") + "/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "max_tokens": max_output_tokens,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                },
                timeout=timeout_s,
            )
        except httpx.TimeoutException as e:
            raise LLMError(f"timeout after {timeout_s}s", retryable=True) from e
        except httpx.TransportError as e:
            raise LLMError(f"network error: {e}", retryable=True) from e
        _raise_for_status(resp)
        body = resp.json()
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError("malformed provider response", retryable=False) from e
        usage = body.get("usage", {})
        return LLMResult(text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))


class Completer(Protocol):
    async def complete(self, *, system: str, user: str) -> LLMResult: ...


class LLMService:
    """One configured model with retry/backoff on retryable provider errors."""

    def __init__(self, settings: Settings, http: httpx.AsyncClient) -> None:
        cfg = ProviderConfig(settings.llm_kind, settings.llm_base_url, settings.llm_api_key_env)
        self._client: LLMClient = (
            AnthropicClient(cfg, http) if cfg.kind == "anthropic" else OpenAICompatClient(cfg, http)
        )
        self._s = settings

    async def complete(self, *, system: str, user: str) -> LLMResult:
        attempt = 0
        while True:
            try:
                return await self._client.complete(
                    model=self._s.llm_model,
                    system=system,
                    user=user,
                    max_output_tokens=self._s.llm_max_output_tokens,
                    timeout_s=self._s.llm_timeout_s,
                )
            except LLMError as e:
                if not e.retryable or attempt >= self._s.llm_max_retries:
                    raise
                delay = self._s.llm_retry_backoff_s * (2**attempt)
                log.warning("LLM call failed (%s); retry %d in %.1fs", e, attempt + 1, delay)
                attempt += 1
                await asyncio.sleep(delay)
