"""Unit tests for generation/prompt.py.

The prompt is the primary control surface for LLM grounding and refusal.
Tests are more prescriptive than elsewhere: exact substrings matter because
a prompt regression is a quality regression.
"""
from __future__ import annotations

import pytest

from fastapi_helper.generation.prompt import (
    REFUSAL_SIGNAL,
    SYSTEM_PROMPT,
    _MAX_CHUNK_CHARS,
    _truncate,
    build_context_block,
    build_user_prompt,
    format_chunk,
)
from fastapi_helper.retrieval.types import RetrievedChunk


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def doc_chunk(
    url: str = "https://fastapi.tiangolo.com/tutorial/middleware/",
    text: str = "Add middleware using app.add_middleware().",
    idx: int = 0,
    metadata: dict | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source_url=url,
        source_type="docs",
        chunk_index=idx,
        score=0.9,
        metadata=metadata or {},
    )


def issue_chunk(
    url: str = "https://github.com/tiangolo/fastapi/issues/1234",
    text: str = "After adding a dependency the session closed early.",
    title: str = "Background task with DB session",
    idx: int = 0,
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source_url=url,
        source_type="issue",
        chunk_index=idx,
        score=0.7,
        metadata={"title": title},
    )


# ---------------------------------------------------------------------------
# SYSTEM_PROMPT contract
# ---------------------------------------------------------------------------

def test_system_prompt_contains_refusal_signal() -> None:
    assert REFUSAL_SIGNAL in SYSTEM_PROMPT


def test_system_prompt_instructs_sources_only() -> None:
    assert "ONLY from the provided sources" in SYSTEM_PROMPT


def test_system_prompt_instructs_inline_citation() -> None:
    assert "[N]" in SYSTEM_PROMPT


def test_system_prompt_forbids_invented_behaviour() -> None:
    # The "never invent" rule must be explicit
    lower = SYSTEM_PROMPT.lower()
    assert "never invent" in lower or "do not invent" in lower


def test_system_prompt_asks_for_conciseness() -> None:
    assert "concise" in SYSTEM_PROMPT.lower() or "3" in SYSTEM_PROMPT


def test_refusal_signal_is_unique_non_empty_string() -> None:
    assert isinstance(REFUSAL_SIGNAL, str)
    assert len(REFUSAL_SIGNAL) > 20
    # Should not appear in a normal factual sentence
    assert "don't have enough information" in REFUSAL_SIGNAL


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------

def test_truncate_short_text_unchanged() -> None:
    text = "Hello world"
    assert _truncate(text) == text


def test_truncate_exactly_max_chars_unchanged() -> None:
    text = "x" * _MAX_CHUNK_CHARS
    assert _truncate(text) == text


def test_truncate_long_text_ends_with_ellipsis() -> None:
    text = "a" * (_MAX_CHUNK_CHARS + 100)
    result = _truncate(text)
    assert result.endswith("…")


def test_truncate_long_text_length_bounded() -> None:
    text = "b" * (_MAX_CHUNK_CHARS * 3)
    result = _truncate(text)
    # The result can be slightly under max due to rstrip + ellipsis
    assert len(result) <= _MAX_CHUNK_CHARS + 2  # 1 space + 1 ellipsis char headroom


def test_truncate_custom_max() -> None:
    result = _truncate("hello world extra", max_chars=5)
    assert result.startswith("hello")
    assert "…" in result


def test_truncate_empty_string() -> None:
    assert _truncate("") == ""


# ---------------------------------------------------------------------------
# format_chunk — docs
# ---------------------------------------------------------------------------

def test_format_chunk_docs_contains_number() -> None:
    c = doc_chunk()
    result = format_chunk(c, 1)
    assert result.startswith("[1]")


def test_format_chunk_docs_contains_url() -> None:
    c = doc_chunk(url="https://fastapi.tiangolo.com/tutorial/cors/")
    result = format_chunk(c, 2)
    assert "https://fastapi.tiangolo.com/tutorial/cors/" in result


def test_format_chunk_docs_contains_source_type_label() -> None:
    result = format_chunk(doc_chunk(), 1)
    assert "docs:" in result


def test_format_chunk_docs_contains_text() -> None:
    c = doc_chunk(text="Use CORSMiddleware to allow origins.")
    result = format_chunk(c, 1)
    assert "Use CORSMiddleware to allow origins." in result


def test_format_chunk_docs_number_reflects_argument() -> None:
    c = doc_chunk()
    assert format_chunk(c, 3).startswith("[3]")
    assert format_chunk(c, 5).startswith("[5]")


def test_format_chunk_docs_long_text_truncated() -> None:
    long_text = "word " * 500
    c = doc_chunk(text=long_text)
    result = format_chunk(c, 1)
    assert "…" in result


def test_format_chunk_docs_short_text_not_truncated() -> None:
    c = doc_chunk(text="Short text.")
    result = format_chunk(c, 1)
    assert "…" not in result


# ---------------------------------------------------------------------------
# format_chunk — issues
# ---------------------------------------------------------------------------

def test_format_chunk_issue_contains_issue_number() -> None:
    c = issue_chunk(url="https://github.com/tiangolo/fastapi/issues/5678")
    result = format_chunk(c, 1)
    assert "#5678" in result


def test_format_chunk_issue_contains_issue_label() -> None:
    result = format_chunk(issue_chunk(), 1)
    assert "issue" in result.lower()


def test_format_chunk_issue_contains_title_when_present() -> None:
    c = issue_chunk(title="Dependency injection fails with background tasks")
    result = format_chunk(c, 1)
    assert "Dependency injection fails with background tasks" in result


def test_format_chunk_issue_no_title_does_not_crash() -> None:
    c = RetrievedChunk(
        text="some text",
        source_url="https://github.com/tiangolo/fastapi/issues/99",
        source_type="issue",
        chunk_index=0,
        score=0.5,
        metadata={},  # no title key
    )
    result = format_chunk(c, 1)
    assert "[1]" in result
    assert "#99" in result


def test_format_chunk_issue_contains_text() -> None:
    c = issue_chunk(text="The session closes before the background task runs.")
    result = format_chunk(c, 2)
    assert "The session closes before the background task runs." in result


def test_format_chunk_issue_contains_url() -> None:
    url = "https://github.com/tiangolo/fastapi/issues/1234"
    result = format_chunk(issue_chunk(url=url), 1)
    assert url in result


# ---------------------------------------------------------------------------
# build_context_block
# ---------------------------------------------------------------------------

def test_build_context_block_numbers_chunks_sequentially() -> None:
    chunks = [doc_chunk(url=f"https://fastapi.tiangolo.com/{i}/") for i in range(3)]
    block = build_context_block(chunks)
    assert "[1]" in block
    assert "[2]" in block
    assert "[3]" in block


def test_build_context_block_chunks_separated_by_blank_line() -> None:
    chunks = [doc_chunk(), issue_chunk()]
    block = build_context_block(chunks)
    assert "\n\n" in block


def test_build_context_block_empty_list_returns_empty_string() -> None:
    assert build_context_block([]) == ""


def test_build_context_block_single_chunk_no_blank_line() -> None:
    block = build_context_block([doc_chunk()])
    # No trailing double newline
    assert not block.endswith("\n\n")


def test_build_context_block_preserves_order() -> None:
    a = doc_chunk(url="https://fastapi.tiangolo.com/a/", text="text A")
    b = doc_chunk(url="https://fastapi.tiangolo.com/b/", text="text B")
    block = build_context_block([a, b])
    assert block.index("[1]") < block.index("[2]")
    assert block.index("text A") < block.index("text B")


# ---------------------------------------------------------------------------
# build_user_prompt
# ---------------------------------------------------------------------------

def test_build_user_prompt_contains_question() -> None:
    prompt = build_user_prompt("How do I add CORS?", [doc_chunk()])
    assert "How do I add CORS?" in prompt


def test_build_user_prompt_contains_sources_header() -> None:
    prompt = build_user_prompt("q?", [doc_chunk()])
    assert "Sources:" in prompt


def test_build_user_prompt_contains_chunk_number() -> None:
    prompt = build_user_prompt("q?", [doc_chunk()])
    assert "[1]" in prompt


def test_build_user_prompt_contains_answer_cue() -> None:
    prompt = build_user_prompt("q?", [doc_chunk()])
    # The answer cue signals to the LLM where to continue
    assert "Answer" in prompt and "[N]" in prompt


def test_build_user_prompt_numbers_match_chunk_count() -> None:
    chunks = [doc_chunk(url=f"https://fastapi.tiangolo.com/{i}/") for i in range(5)]
    prompt = build_user_prompt("q?", chunks)
    for n in range(1, 6):
        assert f"[{n}]" in prompt
    assert "[6]" not in prompt


def test_build_user_prompt_empty_chunks_has_no_sources_placeholder() -> None:
    prompt = build_user_prompt("q?", [])
    # Should include a note that no sources were retrieved rather than being blank
    assert "no sources" in prompt.lower()


def test_build_user_prompt_empty_chunks_does_not_contain_numbered_bracket() -> None:
    prompt = build_user_prompt("q?", [])
    assert "[1]" not in prompt


def test_build_user_prompt_sources_appear_before_question() -> None:
    prompt = build_user_prompt("How do I add CORS?", [doc_chunk()])
    assert prompt.index("Sources:") < prompt.index("How do I add CORS?")


def test_build_user_prompt_question_appears_before_answer_cue() -> None:
    prompt = build_user_prompt("How do I add CORS?", [doc_chunk()])
    assert prompt.index("How do I add CORS?") < prompt.index("Answer")
