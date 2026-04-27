"""Unit tests for retrieval/reranker.py. CrossEncoder is fully mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fastapi_helper.retrieval.reranker import Reranker
from fastapi_helper.retrieval.types import RetrievedChunk


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def chunk(
    url: str = "https://fastapi.tiangolo.com/a/",
    idx: int = 0,
    score: float = 0.5,
    text: str = "some text",
    metadata: dict | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source_url=url,
        source_type="docs",
        chunk_index=idx,
        score=score,
        metadata=metadata or {},
    )


def _make_reranker(scores: list[float], top_n: int = 5) -> Reranker:
    """Return a Reranker whose CrossEncoder.predict returns `scores`."""
    with patch("fastapi_helper.retrieval.reranker.CrossEncoder") as MockCE:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(scores, dtype=np.float32)
        MockCE.return_value = mock_model
        r = Reranker(top_n=top_n)
    r._model = mock_model  # keep the mock active after the patch exits
    return r


# ---------------------------------------------------------------------------
# __init__ — model loading
# ---------------------------------------------------------------------------

def test_init_loads_cross_encoder_once() -> None:
    with patch("fastapi_helper.retrieval.reranker.CrossEncoder") as MockCE:
        MockCE.return_value = MagicMock()
        Reranker(model_name="BAAI/bge-reranker-base")
    MockCE.assert_called_once_with("BAAI/bge-reranker-base")


def test_init_custom_model_name() -> None:
    with patch("fastapi_helper.retrieval.reranker.CrossEncoder") as MockCE:
        MockCE.return_value = MagicMock()
        Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    MockCE.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ---------------------------------------------------------------------------
# rerank — empty / blank guards
# ---------------------------------------------------------------------------

def test_rerank_empty_chunks_returns_empty() -> None:
    r = _make_reranker([])
    assert r.rerank("query", []) == []


def test_rerank_empty_chunks_does_not_call_predict() -> None:
    r = _make_reranker([])
    r.rerank("query", [])
    r._model.predict.assert_not_called()


def test_rerank_blank_query_returns_rrf_order_truncated() -> None:
    chunks = [chunk("https://a/"), chunk("https://b/"), chunk("https://c/")]
    r = _make_reranker([0.9, 0.1, 0.5], top_n=2)
    result = r.rerank("   ", chunks)
    assert len(result) == 2
    # RRF order (input order) preserved
    assert result[0].source_url == "https://a/"
    assert result[1].source_url == "https://b/"


def test_rerank_blank_query_does_not_call_predict() -> None:
    r = _make_reranker([])
    r.rerank("", [chunk()])
    r._model.predict.assert_not_called()


# ---------------------------------------------------------------------------
# rerank — predict call contract
# ---------------------------------------------------------------------------

def test_rerank_calls_predict_with_query_chunk_pairs() -> None:
    c = chunk(text="middleware text")
    r = _make_reranker([0.8])
    r.rerank("middleware", [c])
    call_args = r._model.predict.call_args
    pairs = call_args.args[0]
    assert pairs == [("middleware", "middleware text")]


def test_rerank_predict_receives_all_chunks_as_pairs() -> None:
    chunks = [chunk(url=f"https://{i}/", text=f"text {i}") for i in range(5)]
    r = _make_reranker([0.5, 0.4, 0.3, 0.2, 0.1])
    r.rerank("query", chunks)
    pairs = r._model.predict.call_args.args[0]
    assert len(pairs) == 5
    assert all(p[0] == "query" for p in pairs)


def test_rerank_passes_batch_size_to_predict() -> None:
    with patch("fastapi_helper.retrieval.reranker.CrossEncoder") as MockCE:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5])
        MockCE.return_value = mock_model
        r = Reranker(batch_size=16)
    r._model = mock_model
    r.rerank("q", [chunk()])
    kwargs = r._model.predict.call_args.kwargs
    assert kwargs.get("batch_size") == 16


def test_rerank_passes_show_progress_bar_false() -> None:
    r = _make_reranker([0.5])
    r.rerank("q", [chunk()])
    kwargs = r._model.predict.call_args.kwargs
    assert kwargs.get("show_progress_bar") is False


# ---------------------------------------------------------------------------
# rerank — score ordering
# ---------------------------------------------------------------------------

def test_rerank_sorts_by_score_descending() -> None:
    c1 = chunk("https://a/", score=0.3)
    c2 = chunk("https://b/", score=0.3)
    c3 = chunk("https://c/", score=0.3)
    # cross-encoder scores: c1=0.1, c2=0.9, c3=0.5
    r = _make_reranker([0.1, 0.9, 0.5])
    result = r.rerank("query", [c1, c2, c3])
    assert result[0].source_url == "https://b/"
    assert result[1].source_url == "https://c/"
    assert result[2].source_url == "https://a/"


def test_rerank_score_field_is_cross_encoder_logit() -> None:
    c = chunk("https://a/", score=0.5)
    r = _make_reranker([2.345])
    result = r.rerank("query", [c])
    assert result[0].score == pytest.approx(2.345, rel=1e-5)


def test_rerank_score_is_float() -> None:
    r = _make_reranker([1.0])
    result = r.rerank("q", [chunk()])
    assert isinstance(result[0].score, float)


# ---------------------------------------------------------------------------
# rerank — top_n truncation
# ---------------------------------------------------------------------------

def test_rerank_default_top_n_truncates_output() -> None:
    chunks = [chunk(url=f"https://{i}/") for i in range(10)]
    r = _make_reranker(list(range(10, 0, -1)), top_n=5)
    result = r.rerank("q", chunks)
    assert len(result) == 5


def test_rerank_top_n_override_respected() -> None:
    chunks = [chunk(url=f"https://{i}/") for i in range(8)]
    r = _make_reranker(list(range(8, 0, -1)), top_n=5)
    result = r.rerank("q", chunks, top_n=3)
    assert len(result) == 3


def test_rerank_top_n_larger_than_chunks_returns_all() -> None:
    chunks = [chunk(url=f"https://{i}/") for i in range(3)]
    r = _make_reranker([0.3, 0.1, 0.2], top_n=10)
    result = r.rerank("q", chunks)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# rerank — metadata preservation
# ---------------------------------------------------------------------------

def test_rerank_preserves_rrf_score_in_metadata() -> None:
    c = chunk("https://a/", score=0.032567)
    r = _make_reranker([1.5])
    result = r.rerank("query", [c])
    assert result[0].metadata["rrf_score"] == pytest.approx(0.032567, rel=1e-5)


def test_rerank_rrf_score_rounded_to_8_decimals() -> None:
    c = chunk("https://a/", score=1 / 61)
    r = _make_reranker([1.0])
    result = r.rerank("q", [c])
    assert result[0].metadata["rrf_score"] == round(1 / 61, 8)


def test_rerank_existing_metadata_preserved() -> None:
    c = chunk("https://a/", metadata={"title": "Tutorial", "section": "intro"})
    r = _make_reranker([0.7])
    result = r.rerank("q", [c])
    assert result[0].metadata["title"] == "Tutorial"
    assert result[0].metadata["section"] == "intro"
    assert "rrf_score" in result[0].metadata


def test_rerank_non_score_fields_unchanged() -> None:
    c = chunk("https://a/", idx=3, text="my text")
    r = _make_reranker([0.9])
    result = r.rerank("q", [c])
    assert result[0].source_url == "https://a/"
    assert result[0].chunk_index == 3
    assert result[0].text == "my text"


def test_rerank_does_not_mutate_input_metadata() -> None:
    meta = {"title": "Original"}
    c = chunk("https://a/", metadata=meta)
    r = _make_reranker([0.5])
    r.rerank("q", [c])
    # Original chunk's metadata must not gain "rrf_score"
    assert "rrf_score" not in c.metadata


# ---------------------------------------------------------------------------
# rerank — fallback on model failure
# ---------------------------------------------------------------------------

def test_rerank_model_exception_returns_rrf_order() -> None:
    with patch("fastapi_helper.retrieval.reranker.CrossEncoder") as MockCE:
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("CUDA OOM")
        MockCE.return_value = mock_model
        r = Reranker(top_n=5)
    r._model = mock_model
    chunks = [chunk(url=f"https://{i}/") for i in range(4)]
    result = r.rerank("query", chunks)
    assert [c.source_url for c in result] == [c.source_url for c in chunks]


def test_rerank_model_exception_truncates_to_top_n() -> None:
    with patch("fastapi_helper.retrieval.reranker.CrossEncoder") as MockCE:
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("fail")
        MockCE.return_value = mock_model
        r = Reranker(top_n=2)
    r._model = mock_model
    chunks = [chunk(url=f"https://{i}/") for i in range(5)]
    result = r.rerank("q", chunks)
    assert len(result) == 2


def test_rerank_model_exception_truncates_with_override_top_n() -> None:
    with patch("fastapi_helper.retrieval.reranker.CrossEncoder") as MockCE:
        mock_model = MagicMock()
        mock_model.predict.side_effect = ValueError("bad input")
        MockCE.return_value = mock_model
        r = Reranker(top_n=10)
    r._model = mock_model
    chunks = [chunk(url=f"https://{i}/") for i in range(8)]
    result = r.rerank("q", chunks, top_n=3)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# rerank — output contract
# ---------------------------------------------------------------------------

def test_rerank_returns_retrieved_chunk_instances() -> None:
    r = _make_reranker([0.5, 0.3])
    result = r.rerank("q", [chunk("https://a/"), chunk("https://b/")])
    assert all(isinstance(c, RetrievedChunk) for c in result)


def test_rerank_single_chunk_returns_single_chunk() -> None:
    r = _make_reranker([1.23])
    result = r.rerank("q", [chunk()])
    assert len(result) == 1
    assert result[0].score == pytest.approx(1.23, rel=1e-5)
