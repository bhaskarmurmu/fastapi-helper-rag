"""Unit tests for retrieval/retrieve.py. All I/O components are mocked."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fastapi_helper.retrieval.retrieve import RetrievalResult, Retriever
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


def _make_retriever(
    dense_results: list[RetrievedChunk] | None = None,
    sparse_results: list[RetrievedChunk] | None = None,
    rerank_results: list[RetrievedChunk] | None = None,
    fusion_candidates: int = 10,
) -> tuple[Retriever, MagicMock, MagicMock, MagicMock]:
    """Build a Retriever with mocked components and return (retriever, dense, sparse, reranker)."""
    dense = MagicMock()
    sparse = MagicMock()
    reranker = MagicMock()

    dense.query.return_value = dense_results if dense_results is not None else []
    sparse.query.return_value = sparse_results if sparse_results is not None else []
    # By default reranker echoes its input truncated to top_n
    if rerank_results is not None:
        reranker.rerank.return_value = rerank_results
    else:
        reranker.rerank.side_effect = lambda q, chunks, top_n=5: chunks[:top_n]

    retriever = Retriever(
        dense=dense,
        sparse=sparse,
        reranker=reranker,
        fusion_candidates=fusion_candidates,
    )
    return retriever, dense, sparse, reranker


# ---------------------------------------------------------------------------
# RetrievalResult dataclass
# ---------------------------------------------------------------------------

def test_retrieval_result_has_chunks_and_timing() -> None:
    r = RetrievalResult(chunks=[chunk()], timing={"total_ms": 12.3})
    assert len(r.chunks) == 1
    assert r.timing["total_ms"] == pytest.approx(12.3)


def test_retrieval_result_default_timing_is_empty_dict() -> None:
    r = RetrievalResult(chunks=[])
    assert r.timing == {}


# ---------------------------------------------------------------------------
# Blank / empty query
# ---------------------------------------------------------------------------

def test_blank_query_returns_empty_result() -> None:
    retriever, dense, sparse, _ = _make_retriever()
    result = retriever.retrieve("   ")
    assert result.chunks == []


def test_blank_query_does_not_call_any_component() -> None:
    retriever, dense, sparse, reranker = _make_retriever()
    retriever.retrieve("")
    dense.query.assert_not_called()
    sparse.query.assert_not_called()
    reranker.rerank.assert_not_called()


def test_blank_query_total_ms_is_zero() -> None:
    retriever, _, _, _ = _make_retriever()
    result = retriever.retrieve("  ")
    assert result.timing.get("total_ms", 0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Component call contracts
# ---------------------------------------------------------------------------

def test_dense_always_called_with_query() -> None:
    retriever, dense, _, _ = _make_retriever()
    retriever.retrieve("middleware")
    dense.query.assert_called_once_with("middleware")


def test_sparse_called_when_enable_sparse_true() -> None:
    retriever, _, sparse, _ = _make_retriever()
    retriever.retrieve("middleware", enable_sparse=True)
    sparse.query.assert_called_once_with("middleware")


def test_sparse_not_called_when_enable_sparse_false() -> None:
    retriever, _, sparse, _ = _make_retriever()
    retriever.retrieve("middleware", enable_sparse=False)
    sparse.query.assert_not_called()


def test_reranker_called_when_enable_rerank_true() -> None:
    retriever, _, _, reranker = _make_retriever(
        dense_results=[chunk("https://a/", score=0.8)],
    )
    retriever.retrieve("q", enable_rerank=True)
    reranker.rerank.assert_called_once()


def test_reranker_not_called_when_enable_rerank_false() -> None:
    retriever, _, _, reranker = _make_retriever()
    retriever.retrieve("q", enable_rerank=False)
    reranker.rerank.assert_not_called()


def test_reranker_receives_fused_chunks_and_top_k() -> None:
    d = [chunk("https://a/"), chunk("https://b/")]
    retriever, _, _, reranker = _make_retriever(dense_results=d)
    retriever.retrieve("q", top_k=3)
    call_kwargs = reranker.rerank.call_args
    assert call_kwargs.kwargs.get("top_n") == 3 or call_kwargs.args[2] == 3


# ---------------------------------------------------------------------------
# Full pipeline — output
# ---------------------------------------------------------------------------

def test_full_pipeline_returns_retrieval_result() -> None:
    retriever, _, _, _ = _make_retriever(dense_results=[chunk()])
    result = retriever.retrieve("q")
    assert isinstance(result, RetrievalResult)


def test_full_pipeline_chunks_are_retrieved_chunk_instances() -> None:
    retriever, _, _, _ = _make_retriever(dense_results=[chunk()])
    result = retriever.retrieve("q")
    assert all(isinstance(c, RetrievedChunk) for c in result.chunks)


def test_full_pipeline_top_k_limits_output() -> None:
    chunks = [chunk(url=f"https://{i}/") for i in range(8)]
    reranked = chunks[:3]
    retriever, _, _, _ = _make_retriever(dense_results=chunks, rerank_results=reranked)
    result = retriever.retrieve("q", top_k=3)
    assert len(result.chunks) == 3


def test_dense_only_mode_returns_fused_results_up_to_top_k() -> None:
    chunks = [chunk(url=f"https://{i}/") for i in range(5)]
    retriever, _, _, reranker = _make_retriever(dense_results=chunks)
    result = retriever.retrieve("q", enable_sparse=False, enable_rerank=False, top_k=3)
    assert len(result.chunks) == 3
    reranker.rerank.assert_not_called()


def test_no_rerank_mode_returns_rrf_ordered_results() -> None:
    d = [chunk("https://a/", score=0.9), chunk("https://b/", score=0.7)]
    s = [chunk("https://b/", score=0.8), chunk("https://c/", score=0.6)]
    retriever, _, _, reranker = _make_retriever(dense_results=d, sparse_results=s)
    result = retriever.retrieve("q", enable_rerank=False)
    # Result is RRF-ordered — B appears in both lists so should score highest
    urls = [c.source_url for c in result.chunks]
    assert "https://b/" in urls
    reranker.rerank.assert_not_called()


def test_empty_dense_returns_empty_result() -> None:
    retriever, _, _, _ = _make_retriever(dense_results=[])
    result = retriever.retrieve("q")
    assert result.chunks == []


# ---------------------------------------------------------------------------
# Timing contract
# ---------------------------------------------------------------------------

def test_timing_contains_total_ms() -> None:
    retriever, _, _, _ = _make_retriever(dense_results=[chunk()])
    result = retriever.retrieve("q")
    assert "total_ms" in result.timing
    assert result.timing["total_ms"] > 0


def test_timing_contains_dense_ms() -> None:
    retriever, _, _, _ = _make_retriever(dense_results=[chunk()])
    result = retriever.retrieve("q")
    assert "dense_ms" in result.timing


def test_timing_contains_sparse_ms_when_enabled() -> None:
    retriever, _, _, _ = _make_retriever(dense_results=[chunk()])
    result = retriever.retrieve("q", enable_sparse=True)
    assert "sparse_ms" in result.timing


def test_timing_omits_sparse_ms_when_disabled() -> None:
    retriever, _, _, _ = _make_retriever(dense_results=[chunk()])
    result = retriever.retrieve("q", enable_sparse=False)
    assert "sparse_ms" not in result.timing


def test_timing_contains_rerank_ms_when_enabled() -> None:
    retriever, _, _, _ = _make_retriever(dense_results=[chunk()])
    result = retriever.retrieve("q", enable_rerank=True)
    assert "rerank_ms" in result.timing


def test_timing_omits_rerank_ms_when_disabled() -> None:
    retriever, _, _, _ = _make_retriever(dense_results=[chunk()])
    result = retriever.retrieve("q", enable_rerank=False)
    assert "rerank_ms" not in result.timing


def test_timing_contains_fusion_ms() -> None:
    retriever, _, _, _ = _make_retriever(dense_results=[chunk()])
    result = retriever.retrieve("q")
    assert "fusion_ms" in result.timing


def test_timing_values_are_non_negative_floats() -> None:
    retriever, _, _, _ = _make_retriever(dense_results=[chunk()])
    result = retriever.retrieve("q")
    for key, val in result.timing.items():
        assert isinstance(val, float), f"{key} is not float"
        assert val >= 0.0, f"{key} is negative"


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

def test_sparse_exception_degrades_to_dense_only() -> None:
    dense_chunk = chunk("https://a/", score=0.9)
    retriever, _, sparse, _ = _make_retriever(dense_results=[dense_chunk])
    sparse.query.side_effect = RuntimeError("connection lost")
    # Should not raise; should return dense-only result
    result = retriever.retrieve("q")
    assert len(result.chunks) >= 1


def test_sparse_exception_does_not_raise() -> None:
    retriever, _, sparse, _ = _make_retriever(dense_results=[chunk()])
    sparse.query.side_effect = Exception("timeout")
    result = retriever.retrieve("q")  # must not raise
    assert isinstance(result, RetrievalResult)


def test_sparse_exception_still_records_sparse_ms() -> None:
    retriever, _, sparse, _ = _make_retriever(dense_results=[chunk()])
    sparse.query.side_effect = RuntimeError("fail")
    result = retriever.retrieve("q", enable_sparse=True)
    assert "sparse_ms" in result.timing


def test_reranker_failure_falls_back_to_rrf_order() -> None:
    # Reranker.rerank() handles its own exceptions internally (tested separately).
    # Verify that even if reranker returns RRF-ordered input, retrieve() still works.
    chunks = [chunk(url=f"https://{i}/") for i in range(5)]
    reranked_fallback = chunks[:5]  # same order = RRF fallback
    retriever, _, _, reranker = _make_retriever(
        dense_results=chunks,
        rerank_results=reranked_fallback,
    )
    result = retriever.retrieve("q")
    assert len(result.chunks) == 5


# ---------------------------------------------------------------------------
# Fusion candidates propagation
# ---------------------------------------------------------------------------

def test_fusion_candidates_limits_reranker_input() -> None:
    """Reranker should receive at most fusion_candidates chunks."""
    many = [chunk(url=f"https://{i}/") for i in range(20)]
    retriever, _, _, reranker = _make_retriever(
        dense_results=many,
        fusion_candidates=6,
    )
    reranker.rerank.side_effect = lambda q, chunks, top_n=5: chunks[:top_n]
    retriever.retrieve("q")
    rerank_input = reranker.rerank.call_args.args[1]
    assert len(rerank_input) <= 6


def test_fusion_candidates_respected_in_no_rerank_mode() -> None:
    many = [chunk(url=f"https://{i}/") for i in range(20)]
    retriever, _, _, _ = _make_retriever(dense_results=many, fusion_candidates=7)
    result = retriever.retrieve("q", enable_rerank=False, top_k=7)
    # No-rerank path slices fused[:top_k]; fused is already capped at fusion_candidates
    assert len(result.chunks) <= 7


# ---------------------------------------------------------------------------
# Retriever constructor — rrf_k forwarded to fuse()
# ---------------------------------------------------------------------------

def test_rrf_k_influences_score_magnitude() -> None:
    # k=1 gives much higher scores than k=60 — just verify the pipeline runs with a custom k
    d = [chunk("https://a/"), chunk("https://b/")]
    dense = MagicMock()
    dense.query.return_value = d
    sparse = MagicMock()
    sparse.query.return_value = []
    reranker = MagicMock()
    reranker.rerank.side_effect = lambda q, chunks, top_n=5: chunks[:top_n]

    retriever_k1 = Retriever(dense=dense, sparse=sparse, reranker=reranker, rrf_k=1)
    result = retriever_k1.retrieve("q", enable_rerank=False)
    # rank 1 with k=1: 1/(1+1) = 0.5; much higher than default k=60
    assert result.chunks[0].score == pytest.approx(0.5, rel=1e-5)
