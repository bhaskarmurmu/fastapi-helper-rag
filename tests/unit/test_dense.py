"""Unit tests for retrieval/dense.py. Qdrant and Embedder are fully mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fastapi_helper.retrieval.dense import DenseRetriever, _point_to_chunk
from fastapi_helper.retrieval.types import RetrievedChunk


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_embedder(dim: int = 384) -> MagicMock:
    emb = MagicMock()
    emb.encode.return_value = np.ones((1, dim), dtype=np.float32)
    return emb


def _make_point(
    score: float = 0.85,
    text: str = "FastAPI middleware example.",
    source_url: str = "https://fastapi.tiangolo.com/tutorial/middleware/",
    source_type: str = "docs",
    chunk_index: int = 0,
    extra_payload: dict | None = None,
) -> MagicMock:
    point = MagicMock()
    point.score = score
    point.payload = {
        "text": text,
        "source_url": source_url,
        "source_type": source_type,
        "chunk_index": chunk_index,
        "title": "Middleware",
        "section_path": ["Tutorial", "Middleware"],
        **(extra_payload or {}),
    }
    return point


def _make_client(points: list | None = None) -> MagicMock:
    client = MagicMock()
    client.query_points.return_value.points = points or []
    return client


def _make_retriever(top_k: int = 30) -> tuple[DenseRetriever, MagicMock, MagicMock]:
    embedder = _make_embedder()
    client = _make_client()
    retriever = DenseRetriever(client, embedder, collection="test_col", top_k=top_k)
    return retriever, client, embedder


# ---------------------------------------------------------------------------
# _point_to_chunk
# ---------------------------------------------------------------------------

def test_point_to_chunk_maps_core_fields() -> None:
    point = _make_point(score=0.91, text="hello", source_url="https://a/", chunk_index=3)
    chunk = _point_to_chunk(point)
    assert chunk.text == "hello"
    assert chunk.source_url == "https://a/"
    assert chunk.source_type == "docs"
    assert chunk.chunk_index == 3
    assert chunk.score == pytest.approx(0.91)


def test_point_to_chunk_core_fields_excluded_from_metadata() -> None:
    point = _make_point()
    chunk = _point_to_chunk(point)
    assert "text" not in chunk.metadata
    assert "source_url" not in chunk.metadata
    assert "source_type" not in chunk.metadata
    assert "chunk_index" not in chunk.metadata


def test_point_to_chunk_extra_payload_in_metadata() -> None:
    point = _make_point(extra_payload={"custom_key": "custom_val"})
    chunk = _point_to_chunk(point)
    assert chunk.metadata["title"] == "Middleware"
    assert chunk.metadata["section_path"] == ["Tutorial", "Middleware"]
    assert chunk.metadata["custom_key"] == "custom_val"


def test_point_to_chunk_empty_payload_uses_defaults() -> None:
    point = MagicMock()
    point.score = 0.5
    point.payload = None
    chunk = _point_to_chunk(point)
    assert chunk.text == ""
    assert chunk.source_url == ""
    assert chunk.source_type == "docs"
    assert chunk.chunk_index == 0


def test_point_to_chunk_returns_retrieved_chunk_instance() -> None:
    chunk = _point_to_chunk(_make_point())
    assert isinstance(chunk, RetrievedChunk)


# ---------------------------------------------------------------------------
# DenseRetriever.query — embedder interaction
# ---------------------------------------------------------------------------

def test_query_calls_encode_with_is_query_true() -> None:
    retriever, _, embedder = _make_retriever()
    retriever.query("how do I add middleware")
    call = embedder.encode.call_args
    assert call.kwargs.get("is_query") is True or (call.args and call.args[1:])
    # Also verify the text was passed as a single-element list
    texts_arg = call.args[0] if call.args else call.kwargs.get("texts", call.kwargs.get("text"))
    assert texts_arg == ["how do I add middleware"]


def test_query_passes_vector_to_qdrant() -> None:
    retriever, client, embedder = _make_retriever()
    expected_vec = np.array([[0.5] * 384], dtype=np.float32)
    embedder.encode.return_value = expected_vec

    retriever.query("test query")

    call = client.query_points.call_args
    passed_query = call.kwargs.get("query") if call.kwargs else call.args[1]
    np.testing.assert_array_equal(passed_query, expected_vec[0])


# ---------------------------------------------------------------------------
# DenseRetriever.query — qdrant interaction
# ---------------------------------------------------------------------------

def test_query_uses_collection_name() -> None:
    retriever, client, _ = _make_retriever()
    retriever.query("something")
    call = client.query_points.call_args
    collection = call.kwargs.get("collection_name") or call.args[0]
    assert collection == "test_col"


def test_query_default_top_k_passed_to_qdrant() -> None:
    retriever, client, _ = _make_retriever(top_k=25)
    retriever.query("something")
    call = client.query_points.call_args
    limit = call.kwargs.get("limit")
    assert limit == 25


def test_query_top_k_override_respected() -> None:
    retriever, client, _ = _make_retriever(top_k=30)
    retriever.query("something", top_k=5)
    call = client.query_points.call_args
    limit = call.kwargs.get("limit")
    assert limit == 5


def test_query_requests_payload() -> None:
    retriever, client, _ = _make_retriever()
    retriever.query("something")
    call = client.query_points.call_args
    assert call.kwargs.get("with_payload") is True


# ---------------------------------------------------------------------------
# DenseRetriever.query — return value
# ---------------------------------------------------------------------------

def test_query_returns_list_of_retrieved_chunks() -> None:
    points = [_make_point(score=0.9), _make_point(score=0.8)]
    client = _make_client(points)
    retriever = DenseRetriever(client, _make_embedder(), top_k=30)
    results = retriever.query("test")
    assert len(results) == 2
    assert all(isinstance(r, RetrievedChunk) for r in results)


def test_query_scores_preserved() -> None:
    points = [_make_point(score=0.91), _make_point(score=0.77)]
    client = _make_client(points)
    retriever = DenseRetriever(client, _make_embedder(), top_k=30)
    results = retriever.query("test")
    assert results[0].score == pytest.approx(0.91)
    assert results[1].score == pytest.approx(0.77)


def test_query_empty_collection_returns_empty_list() -> None:
    retriever, _, _ = _make_retriever()
    results = retriever.query("anything")
    assert results == []


def test_query_blank_text_returns_empty_without_calling_qdrant() -> None:
    retriever, client, embedder = _make_retriever()
    results = retriever.query("   ")
    assert results == []
    embedder.encode.assert_not_called()
    client.query_points.assert_not_called()


def test_query_qdrant_exception_returns_empty_list() -> None:
    client = _make_client()
    client.query_points.side_effect = RuntimeError("connection refused")
    retriever = DenseRetriever(client, _make_embedder(), top_k=30)
    results = retriever.query("test")
    assert results == []


# ---------------------------------------------------------------------------
# RetrievedChunk type contract
# ---------------------------------------------------------------------------

def test_retrieved_chunk_is_pydantic_model() -> None:
    chunk = RetrievedChunk(
        text="t", source_url="u", source_type="docs", chunk_index=0, score=1.0
    )
    assert chunk.model_dump()["text"] == "t"


def test_retrieved_chunk_metadata_defaults_to_empty_dict() -> None:
    chunk = RetrievedChunk(
        text="t", source_url="u", source_type="docs", chunk_index=0, score=1.0
    )
    assert chunk.metadata == {}
