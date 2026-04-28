"""Groq API client with retry, error mapping, and token usage tracking.

Only Groq is implemented here.  When Gemini support is added, extract a
LLMClient protocol/ABC and make GroqClient implement it.  Don't add the
abstraction before it has two implementations — that's the right time.

Streaming usage note: Groq populates usage on the final chunk's .usage field
when stream=True.  The stream() method captures it internally and exposes it
via the returned GenerationStream wrapper so callers can read it after the
stream completes without a separate API call.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import AsyncIterator

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncGroq,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
)

from fastapi_helper.config import Settings

log = logging.getLogger(__name__)

# Delays (seconds) before retry attempt 1, 2, … — length == max_retries.
_RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Base for all LLM errors.  Generator catches this to return a graceful
    error string instead of propagating to the API layer."""


class LLMRateLimitError(LLMError):
    """HTTP 429.  Raised after retries exhausted.  Retried internally."""


class LLMAuthError(LLMError):
    """HTTP 401.  Not retried — the API key is wrong; ops must fix it."""


class LLMUnavailableError(LLMError):
    """5xx, network error, or timeout.  Retried internally."""


class LLMBadRequestError(LLMError):
    """HTTP 400.  Not retried — the request is malformed."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class LLMResponse:
    """Result of a non-streaming LLM call."""
    text: str
    usage: TokenUsage | None = None


class GenerationStream:
    """Async iterable over token strings with usage available after iteration.

    Usage::

        gs = await client.stream(system, user)
        async for token in gs:
            print(token, end="", flush=True)
        print(gs.usage)   # TokenUsage or None
    """

    def __init__(self, raw: AsyncIterator[str], usage_holder: list[TokenUsage | None]) -> None:
        self._raw = raw
        self._usage_holder = usage_holder

    def __aiter__(self) -> AsyncIterator[str]:
        return self._raw

    @property
    def usage(self) -> TokenUsage | None:
        return self._usage_holder[0] if self._usage_holder else None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_usage(raw) -> TokenUsage | None:
    if raw is None:
        return None
    try:
        return TokenUsage(
            prompt_tokens=raw.prompt_tokens,
            completion_tokens=raw.completion_tokens,
            total_tokens=raw.total_tokens,
        )
    except AttributeError:
        return None


def _parse_retry_after(exc: Exception) -> float | None:
    """Extract Retry-After seconds from a Groq API error response header.

    Tries to read exc.response.headers regardless of exception type — if the
    attribute path doesn't exist it returns None silently.
    """
    try:
        header = exc.response.headers.get("retry-after")  # type: ignore[attr-defined]
        return float(header) if header else None
    except (AttributeError, TypeError, ValueError):
        return None


def _map_exception(exc: Exception) -> LLMError:
    """Map a groq SDK exception to the project's LLMError hierarchy."""
    if isinstance(exc, LLMError):
        return exc
    if isinstance(exc, RateLimitError):
        return LLMRateLimitError(str(exc))
    if isinstance(exc, AuthenticationError):
        return LLMAuthError(str(exc))
    if isinstance(exc, BadRequestError):
        return LLMBadRequestError(str(exc))
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return LLMUnavailableError(str(exc))
    if isinstance(exc, APIStatusError):
        if exc.status_code == 429:
            return LLMRateLimitError(str(exc))
        if exc.status_code >= 500:
            return LLMUnavailableError(str(exc))
        return LLMError(str(exc))
    return LLMError(str(exc))


def _is_retryable(exc: LLMError) -> bool:
    return isinstance(exc, (LLMRateLimitError, LLMUnavailableError))


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class GroqClient:
    """Async Groq API client with retry and structured error mapping.

    Instantiate once at application startup and reuse across requests.
    The underlying AsyncGroq client manages its own connection pool.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 1024,
        temperature: float = 0.1,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self._client = AsyncGroq(api_key=api_key, timeout=timeout)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_retries = max_retries

    def _messages(self, system: str, user: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    async def complete(self, system: str, user: str) -> LLMResponse:
        """Call Groq and return the full response text with token usage.

        Retries on rate limits and transient errors.  Raises LLMError
        subclasses on terminal failures.  Never returns None.
        """
        messages = self._messages(system, user)
        last_exc: LLMError | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    stream=False,
                )
                text = response.choices[0].message.content or ""
                usage = _parse_usage(response.usage)
                log.debug(
                    "Groq complete: %d prompt + %d completion tokens",
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0,
                )
                return LLMResponse(text=text, usage=usage)

            except Exception as exc:
                mapped = _map_exception(exc)
                if not _is_retryable(mapped):
                    raise mapped from exc
                last_exc = mapped
                if attempt < self._max_retries:
                    delay = _RETRY_DELAYS[attempt]
                    if isinstance(exc, RateLimitError):
                        delay = min(_parse_retry_after(exc) or delay, 30.0)
                    log.warning(
                        "Groq retryable error on attempt %d/%d, retrying in %.1fs: %s",
                        attempt + 1,
                        self._max_retries + 1,
                        delay,
                        type(exc).__name__,
                    )
                    await asyncio.sleep(delay)

        assert last_exc is not None
        raise last_exc

    async def stream(self, system: str, user: str) -> GenerationStream:
        """Return a GenerationStream that yields tokens and captures final usage.

        No retry — once streaming starts, yielded tokens can't be rolled back.
        Any error during streaming raises the appropriate LLMError subclass.
        Read .usage on the returned object after iteration to get token counts.
        """
        messages = self._messages(system, user)
        usage_holder: list[TokenUsage | None] = []

        async def _token_gen() -> AsyncIterator[str]:
            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    stream=True,
                )
                last_chunk = None
                async for chunk in response:
                    last_chunk = chunk
                    if chunk.choices:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content
                # Capture usage from the final chunk (populated by Groq on last event).
                if last_chunk is not None:
                    raw_usage = getattr(last_chunk, "usage", None) or (
                        getattr(last_chunk.x_groq, "usage", None)
                        if getattr(last_chunk, "x_groq", None)
                        else None
                    )
                    usage_holder.append(_parse_usage(raw_usage))
                else:
                    usage_holder.append(None)
            except LLMError:
                raise
            except Exception as exc:
                raise _map_exception(exc) from exc

        return GenerationStream(_token_gen(), usage_holder)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_llm_client(settings: Settings) -> GroqClient:
    """Construct the configured LLM client from application settings.

    Raises NotImplementedError if settings.llm_provider is not "groq".
    Gemini support will be added when needed — extract a protocol then.
    """
    if settings.llm_provider != "groq":
        raise NotImplementedError(
            f"LLM provider {settings.llm_provider!r} is not yet implemented; "
            f"only 'groq' is supported. Set LLM_PROVIDER=groq."
        )
    return GroqClient(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
    )
