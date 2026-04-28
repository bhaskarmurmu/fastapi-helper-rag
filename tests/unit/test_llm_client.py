"""Unit tests for generation/llm_client.py.  Groq SDK is fully mocked."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fastapi_helper.generation.llm_client import (
    GenerationStream,
    GroqClient,
    LLMAuthError,
    LLMBadRequestError,
    LLMError,
    LLMRateLimitError,
    LLMUnavailableError,
    TokenUsage,
    _map_exception,
    _parse_retry_after,
    _parse_usage,
    get_llm_client,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_client(max_retries: int = 0) -> tuple[GroqClient, MagicMock]:
    """Return (GroqClient, mock_groq_instance) with AsyncGroq patched."""
    with patch("fastapi_helper.generation.llm_client.AsyncGroq") as MockGroq:
        mock_instance = MagicMock()
        MockGroq.return_value = mock_instance
        client = GroqClient(api_key="sk-test", max_retries=max_retries)
    client._client = mock_instance
    return client, mock_instance


def _make_completion_response(text: str, prompt_tokens: int = 10, completion_tokens: int = 20) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.usage.total_tokens = prompt_tokens + completion_tokens
    return resp


async def _make_stream_chunks(tokens: list[str], final_usage: MagicMock | None = None):
    """Async generator that yields mock ChatCompletionChunk objects."""
    for i, token in enumerate(tokens):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = token
        chunk.usage = final_usage if i == len(tokens) - 1 else None
        chunk.x_groq = None
        yield chunk


# ---------------------------------------------------------------------------
# _parse_usage
# ---------------------------------------------------------------------------

def test_parse_usage_returns_token_usage() -> None:
    raw = MagicMock()
    raw.prompt_tokens = 50
    raw.completion_tokens = 100
    raw.total_tokens = 150
    usage = _parse_usage(raw)
    assert usage is not None
    assert usage.prompt_tokens == 50
    assert usage.completion_tokens == 100
    assert usage.total_tokens == 150


def test_parse_usage_none_returns_none() -> None:
    assert _parse_usage(None) is None


def test_parse_usage_missing_attr_returns_none() -> None:
    assert _parse_usage(object()) is None


# ---------------------------------------------------------------------------
# _parse_retry_after
# ---------------------------------------------------------------------------

def test_parse_retry_after_returns_float_from_header() -> None:
    # _parse_retry_after reads exc.response.headers without isinstance check
    exc = MagicMock()
    exc.response.headers.get.return_value = "5"
    assert _parse_retry_after(exc) == pytest.approx(5.0)


def test_parse_retry_after_returns_none_for_non_api_error() -> None:
    assert _parse_retry_after(ValueError("oops")) is None


def test_parse_retry_after_returns_none_when_header_absent() -> None:
    exc = MagicMock()
    exc.response.headers.get.return_value = None
    assert _parse_retry_after(exc) is None


# ---------------------------------------------------------------------------
# _map_exception
# ---------------------------------------------------------------------------

def test_map_exception_rate_limit() -> None:
    from groq import RateLimitError
    exc = MagicMock(spec=RateLimitError)
    assert isinstance(_map_exception(exc), LLMRateLimitError)


def test_map_exception_auth() -> None:
    from groq import AuthenticationError
    exc = MagicMock(spec=AuthenticationError)
    assert isinstance(_map_exception(exc), LLMAuthError)


def test_map_exception_bad_request() -> None:
    from groq import BadRequestError
    exc = MagicMock(spec=BadRequestError)
    assert isinstance(_map_exception(exc), LLMBadRequestError)


def test_map_exception_connection_error() -> None:
    from groq import APIConnectionError
    exc = MagicMock(spec=APIConnectionError)
    assert isinstance(_map_exception(exc), LLMUnavailableError)


def test_map_exception_timeout() -> None:
    from groq import APITimeoutError
    exc = MagicMock(spec=APITimeoutError)
    assert isinstance(_map_exception(exc), LLMUnavailableError)


def test_map_exception_500() -> None:
    from groq import APIStatusError
    exc = MagicMock(spec=APIStatusError)
    exc.status_code = 500
    assert isinstance(_map_exception(exc), LLMUnavailableError)


def test_map_exception_429_via_status_error() -> None:
    from groq import APIStatusError
    exc = MagicMock(spec=APIStatusError)
    exc.status_code = 429
    assert isinstance(_map_exception(exc), LLMRateLimitError)


def test_map_exception_unknown_4xx_returns_base_llm_error() -> None:
    from groq import APIStatusError
    exc = MagicMock(spec=APIStatusError)
    exc.status_code = 422
    assert type(_map_exception(exc)) is LLMError


def test_map_exception_already_llm_error_returned_as_is() -> None:
    exc = LLMRateLimitError("already mapped")
    assert _map_exception(exc) is exc


# ---------------------------------------------------------------------------
# GroqClient.complete — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_returns_llm_response() -> None:
    client, mock = _make_client()
    mock.chat.completions.create = AsyncMock(return_value=_make_completion_response("hello"))
    result = await client.complete("sys", "user")
    assert result.text == "hello"


@pytest.mark.asyncio
async def test_complete_captures_token_usage() -> None:
    client, mock = _make_client()
    mock.chat.completions.create = AsyncMock(
        return_value=_make_completion_response("hi", prompt_tokens=15, completion_tokens=25)
    )
    result = await client.complete("sys", "user")
    assert result.usage is not None
    assert result.usage.prompt_tokens == 15
    assert result.usage.completion_tokens == 25
    assert result.usage.total_tokens == 40


@pytest.mark.asyncio
async def test_complete_sends_system_and_user_messages() -> None:
    client, mock = _make_client()
    mock.chat.completions.create = AsyncMock(return_value=_make_completion_response("ok"))
    await client.complete("my system", "my question")
    call_args = mock.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "my system"}
    assert messages[1] == {"role": "user", "content": "my question"}


@pytest.mark.asyncio
async def test_complete_passes_model_and_params() -> None:
    client, mock = _make_client()
    mock.chat.completions.create = AsyncMock(return_value=_make_completion_response("ok"))
    await client.complete("s", "u")
    kwargs = mock.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "llama-3.3-70b-versatile"
    assert kwargs["stream"] is False
    assert "max_tokens" in kwargs
    assert "temperature" in kwargs


@pytest.mark.asyncio
async def test_complete_none_content_returns_empty_string() -> None:
    client, mock = _make_client()
    resp = _make_completion_response("x")
    resp.choices[0].message.content = None
    mock.chat.completions.create = AsyncMock(return_value=resp)
    result = await client.complete("s", "u")
    assert result.text == ""


# ---------------------------------------------------------------------------
# GroqClient.complete — error handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_complete_auth_error_raises_immediately() -> None:
    # Patch the imported name so isinstance checks in _map_exception hit our fake class.
    class FakeAuthError(Exception):
        pass

    client, mock = _make_client(max_retries=2)
    mock.chat.completions.create = AsyncMock(side_effect=FakeAuthError("bad key"))
    with patch("fastapi_helper.generation.llm_client.AuthenticationError", FakeAuthError):
        with pytest.raises(LLMAuthError):
            await client.complete("s", "u")
    assert mock.chat.completions.create.call_count == 1  # no retries


@pytest.mark.asyncio
async def test_complete_bad_request_raises_immediately() -> None:
    class FakeBadRequestError(Exception):
        pass

    client, mock = _make_client(max_retries=2)
    mock.chat.completions.create = AsyncMock(side_effect=FakeBadRequestError("bad"))
    with patch("fastapi_helper.generation.llm_client.BadRequestError", FakeBadRequestError):
        with pytest.raises(LLMBadRequestError):
            await client.complete("s", "u")
    assert mock.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_complete_rate_limit_retries_and_raises() -> None:
    class FakeRateLimitError(Exception):
        pass

    client, mock = _make_client(max_retries=2)
    mock.chat.completions.create = AsyncMock(side_effect=FakeRateLimitError("429"))
    with patch("fastapi_helper.generation.llm_client.RateLimitError", FakeRateLimitError), \
         patch("fastapi_helper.generation.llm_client.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(LLMRateLimitError):
            await client.complete("s", "u")
    assert mock.chat.completions.create.call_count == 3  # 1 + 2 retries


@pytest.mark.asyncio
async def test_complete_unavailable_retries_and_raises() -> None:
    class FakeConnError(Exception):
        pass

    client, mock = _make_client(max_retries=1)
    mock.chat.completions.create = AsyncMock(side_effect=FakeConnError("conn"))
    with patch("fastapi_helper.generation.llm_client.APIConnectionError", FakeConnError), \
         patch("fastapi_helper.generation.llm_client.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(LLMUnavailableError):
            await client.complete("s", "u")
    assert mock.chat.completions.create.call_count == 2  # 1 + 1 retry


@pytest.mark.asyncio
async def test_complete_retries_then_succeeds() -> None:
    class FakeConnError(Exception):
        pass

    client, mock = _make_client(max_retries=2)
    mock.chat.completions.create = AsyncMock(side_effect=[
        FakeConnError("transient"),
        _make_completion_response("success"),
    ])
    with patch("fastapi_helper.generation.llm_client.APIConnectionError", FakeConnError), \
         patch("fastapi_helper.generation.llm_client.asyncio.sleep", new_callable=AsyncMock):
        result = await client.complete("s", "u")
    assert result.text == "success"
    assert mock.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# GroqClient.stream — happy path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_returns_generation_stream() -> None:
    client, mock = _make_client()
    mock.chat.completions.create = AsyncMock(
        return_value=_make_stream_chunks(["hello", " world"])
    )
    gs = await client.stream("sys", "user")
    assert isinstance(gs, GenerationStream)


@pytest.mark.asyncio
async def test_stream_yields_tokens_in_order() -> None:
    client, mock = _make_client()
    mock.chat.completions.create = AsyncMock(
        return_value=_make_stream_chunks(["To", " add", " middleware"])
    )
    gs = await client.stream("sys", "user")
    tokens = [token async for token in gs]
    assert tokens == ["To", " add", " middleware"]


@pytest.mark.asyncio
async def test_stream_passes_stream_true_to_create() -> None:
    # create() is called lazily inside the async generator — must iterate to trigger it.
    client, mock = _make_client()
    mock.chat.completions.create = AsyncMock(return_value=_make_stream_chunks([]))
    gs = await client.stream("sys", "user")
    async for _ in gs:
        pass
    kwargs = mock.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True


@pytest.mark.asyncio
async def test_stream_captures_usage_from_final_chunk() -> None:
    final_usage = MagicMock()
    final_usage.prompt_tokens = 30
    final_usage.completion_tokens = 60
    final_usage.total_tokens = 90
    client, mock = _make_client()
    mock.chat.completions.create = AsyncMock(
        return_value=_make_stream_chunks(["hello"], final_usage=final_usage)
    )
    gs = await client.stream("sys", "user")
    async for _ in gs:
        pass
    assert gs.usage is not None
    assert gs.usage.prompt_tokens == 30
    assert gs.usage.completion_tokens == 60


@pytest.mark.asyncio
async def test_stream_usage_none_when_no_chunks() -> None:
    client, mock = _make_client()
    mock.chat.completions.create = AsyncMock(return_value=_make_stream_chunks([]))
    gs = await client.stream("sys", "user")
    async for _ in gs:
        pass
    assert gs.usage is None


@pytest.mark.asyncio
async def test_stream_skips_none_content_chunks() -> None:
    async def chunks_with_none():
        for content in [None, "hello", None, " world"]:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = content
            chunk.usage = None
            chunk.x_groq = None
            yield chunk

    client, mock = _make_client()
    mock.chat.completions.create = AsyncMock(return_value=chunks_with_none())
    gs = await client.stream("sys", "user")
    tokens = [t async for t in gs]
    assert tokens == ["hello", " world"]


# ---------------------------------------------------------------------------
# GroqClient.stream — error handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_auth_error_raises_llm_auth_error() -> None:
    # create() runs lazily — error surfaces during iteration, not at await time.
    class FakeAuthError(Exception):
        pass

    client, mock = _make_client()
    mock.chat.completions.create = AsyncMock(side_effect=FakeAuthError("bad key"))
    with patch("fastapi_helper.generation.llm_client.AuthenticationError", FakeAuthError):
        with pytest.raises(LLMAuthError):
            gs = await client.stream("s", "u")
            async for _ in gs:
                pass


@pytest.mark.asyncio
async def test_stream_rate_limit_raises_llm_rate_limit_error() -> None:
    class FakeRateLimitError(Exception):
        pass

    client, mock = _make_client()
    mock.chat.completions.create = AsyncMock(side_effect=FakeRateLimitError("429"))
    with patch("fastapi_helper.generation.llm_client.RateLimitError", FakeRateLimitError):
        with pytest.raises(LLMRateLimitError):
            gs = await client.stream("s", "u")
            async for _ in gs:
                pass


@pytest.mark.asyncio
async def test_stream_timeout_raises_llm_unavailable_error() -> None:
    class FakeTimeoutError(Exception):
        pass

    client, mock = _make_client()
    mock.chat.completions.create = AsyncMock(side_effect=FakeTimeoutError("timeout"))
    with patch("fastapi_helper.generation.llm_client.APITimeoutError", FakeTimeoutError):
        with pytest.raises(LLMUnavailableError):
            gs = await client.stream("s", "u")
            async for _ in gs:
                pass


# ---------------------------------------------------------------------------
# GenerationStream
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generation_stream_is_async_iterable() -> None:
    async def gen():
        yield "a"
        yield "b"

    gs = GenerationStream(gen(), [None])
    tokens = [t async for t in gs]
    assert tokens == ["a", "b"]


def test_generation_stream_usage_from_holder() -> None:
    usage = TokenUsage(10, 20, 30)
    gs = GenerationStream(iter([]), [usage])
    assert gs.usage is usage


def test_generation_stream_usage_none_when_holder_empty() -> None:
    gs = GenerationStream(iter([]), [None])
    assert gs.usage is None


# ---------------------------------------------------------------------------
# get_llm_client factory
# ---------------------------------------------------------------------------

def test_get_llm_client_returns_groq_client() -> None:
    settings = MagicMock()
    settings.llm_provider = "groq"
    settings.groq_api_key = "sk-test"
    settings.groq_model = "llama-3.3-70b-versatile"
    settings.llm_max_tokens = 1024
    settings.llm_temperature = 0.1
    with patch("fastapi_helper.generation.llm_client.AsyncGroq"):
        client = get_llm_client(settings)
    assert isinstance(client, GroqClient)


def test_get_llm_client_raises_for_unsupported_provider() -> None:
    settings = MagicMock()
    settings.llm_provider = "gemini"
    with pytest.raises(NotImplementedError, match="gemini"):
        get_llm_client(settings)
