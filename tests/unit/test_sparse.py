"""Unit tests for retrieval/sparse.py. psycopg connection is fully mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from fastapi_helper.retrieval.sparse import SparseRetriever, _row_to_chunk
from fastapi_helper.retrieval.types import RetrievedChunk


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_row(
    source_url: str = "https://fastapi.tiangolo.com/tutorial/middleware/",
    source_type: str = "docs",
    chunk_index: int = 0,
    text: str = "Middleware text.",
    rank: float = 0.075,
    metadata: dict | None = None,
) -> dict:
    return {
        "source_url": source_url,
        "source_type": source_type,
        "chunk_index": chunk_index,
        "text": text,
        "rank": rank,
        "metadata": metadata or {"title": "Middleware"},
    }


def _make_conn(rows: list[dict] | None = None) -> MagicMock:
    """Build a mock psycopg.Connection whose cursor context manager returns rows."""
    conn = MagicMock()
    cursor = MagicMock()
    cursor.__enter__ = lambda s: s
    cursor.__exit__ = MagicMock(return_value=False)
    cursor.fetchall.return_value = rows or []
    conn.cursor.return_value = cursor
    return conn


# ---------------------------------------------------------------------------
# _row_to_chunk
# ---------------------------------------------------------------------------

def test_row_to_chunk_maps_fields() -> None:
    row = _make_row(source_url="https://a/", chunk_index=2, rank=0.085)
    chunk = _row_to_chunk(row)
    assert chunk.source_url == "https://a/"
    assert chunk.chunk_index == 2
    assert chunk.score == pytest.approx(0.085)
    assert chunk.text == "Middleware text."


def test_row_to_chunk_score_is_float() -> None:
    row = _make_row(rank=0.05)
    chunk = _row_to_chunk(row)
    assert isinstance(chunk.score, float)


def test_row_to_chunk_metadata_preserved() -> None:
    row = _make_row(metadata={"title": "Test", "section_path": ["Tutorial"]})
    chunk = _row_to_chunk(row)
    assert chunk.metadata["title"] == "Test"
    assert chunk.metadata["section_path"] == ["Tutorial"]


def test_row_to_chunk_none_metadata_becomes_empty_dict() -> None:
    row = _make_row(metadata=None)
    # Override key so metadata is explicitly None
    row["metadata"] = None
    chunk = _row_to_chunk(row)
    assert chunk.metadata == {}


def test_row_to_chunk_returns_retrieved_chunk() -> None:
    chunk = _row_to_chunk(_make_row())
    assert isinstance(chunk, RetrievedChunk)


# ---------------------------------------------------------------------------
# SparseRetriever.query — connection / cursor interaction
# ---------------------------------------------------------------------------

def test_query_opens_cursor_with_dict_row_factory() -> None:
    import psycopg.rows
    conn = _make_conn()
    retriever = SparseRetriever(conn, top_k=30)
    retriever.query("middleware")
    conn.cursor.assert_called_once_with(row_factory=psycopg.rows.dict_row)


def test_query_executes_parameterized_sql() -> None:
    conn = _make_conn()
    retriever = SparseRetriever(conn, top_k=10)
    retriever.query("middleware")
    cursor = conn.cursor.return_value
    execute_call = cursor.execute.call_args
    sql, params = execute_call.args
    # SQL must use placeholders, not f-string interpolation
    assert "%s" in sql
    assert "middleware" in params
    assert params[-1] == 10  # LIMIT is last param


def test_query_strips_whitespace_before_executing() -> None:
    conn = _make_conn()
    retriever = SparseRetriever(conn, top_k=5)
    retriever.query("  uvicorn workers  ")
    cursor = conn.cursor.return_value
    params = cursor.execute.call_args.args[1]
    assert params[0] == "uvicorn workers"


def test_query_sql_contains_plainto_tsquery() -> None:
    conn = _make_conn()
    SparseRetriever(conn).query("test")
    sql = conn.cursor.return_value.execute.call_args.args[0]
    assert "plainto_tsquery" in sql


def test_query_sql_contains_order_by_rank() -> None:
    conn = _make_conn()
    SparseRetriever(conn).query("test")
    sql = conn.cursor.return_value.execute.call_args.args[0].lower()
    assert "order by" in sql
    assert "rank" in sql


# ---------------------------------------------------------------------------
# SparseRetriever.query — top_k
# ---------------------------------------------------------------------------

def test_query_default_top_k_used_as_limit() -> None:
    conn = _make_conn()
    retriever = SparseRetriever(conn, top_k=20)
    retriever.query("something")
    params = conn.cursor.return_value.execute.call_args.args[1]
    assert params[-1] == 20


def test_query_top_k_override_respected() -> None:
    conn = _make_conn()
    retriever = SparseRetriever(conn, top_k=30)
    retriever.query("something", top_k=7)
    params = conn.cursor.return_value.execute.call_args.args[1]
    assert params[-1] == 7


# ---------------------------------------------------------------------------
# SparseRetriever.query — blank input guard
# ---------------------------------------------------------------------------

def test_query_blank_text_returns_empty_list() -> None:
    conn = _make_conn()
    retriever = SparseRetriever(conn)
    assert retriever.query("") == []
    assert retriever.query("   ") == []


def test_query_blank_text_does_not_open_cursor() -> None:
    conn = _make_conn()
    SparseRetriever(conn).query("   ")
    conn.cursor.assert_not_called()


# ---------------------------------------------------------------------------
# SparseRetriever.query — return values
# ---------------------------------------------------------------------------

def test_query_returns_list_of_retrieved_chunks() -> None:
    rows = [_make_row(rank=0.09), _make_row(rank=0.07, chunk_index=1)]
    conn = _make_conn(rows)
    results = SparseRetriever(conn).query("middleware")
    assert len(results) == 2
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_query_scores_match_row_ranks() -> None:
    rows = [_make_row(rank=0.091), _make_row(rank=0.063, chunk_index=1)]
    conn = _make_conn(rows)
    results = SparseRetriever(conn).query("middleware")
    assert results[0].score == pytest.approx(0.091)
    assert results[1].score == pytest.approx(0.063)


def test_query_no_results_returns_empty_list() -> None:
    conn = _make_conn([])
    assert SparseRetriever(conn).query("xyzzy_no_match") == []


# ---------------------------------------------------------------------------
# SparseRetriever.query — error handling
# ---------------------------------------------------------------------------

def test_query_db_exception_returns_empty_list() -> None:
    conn = MagicMock()
    conn.cursor.side_effect = Exception("connection lost")
    results = SparseRetriever(conn).query("middleware")
    assert results == []


def test_query_fetchall_exception_returns_empty_list() -> None:
    conn = _make_conn()
    conn.cursor.return_value.fetchall.side_effect = Exception("timeout")
    results = SparseRetriever(conn).query("middleware")
    assert results == []
