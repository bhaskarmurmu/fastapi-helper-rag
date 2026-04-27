"""Top-level retrieval pipeline: dense + sparse → RRF fusion → reranker.

Usage::

    retriever = Retriever(dense, sparse, reranker)
    result = retriever.retrieve("how do I add middleware?")
    # result.chunks — list[RetrievedChunk], rerank-scored, top-K
    # result.timing — {"dense_ms": …, "sparse_ms": …, "rerank_ms": …, "total_ms": …}

Ablation via flags on retrieve():
    retriever.retrieve(q, enable_sparse=False)           # dense-only
    retriever.retrieve(q, enable_rerank=False)           # hybrid, no reranker
    retriever.retrieve(q, enable_sparse=False, enable_rerank=False)  # dense + RRF only
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from fastapi_helper.retrieval.dense import DenseRetriever
from fastapi_helper.retrieval.fusion import fuse
from fastapi_helper.retrieval.reranker import Reranker
from fastapi_helper.retrieval.sparse import SparseRetriever
from fastapi_helper.retrieval.types import RetrievedChunk

log = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Structured return value from Retriever.retrieve().

    chunks: final ranked list (rerank-scored if reranker ran, RRF-scored otherwise)
    timing: per-component wall-clock ms; absent keys mean that stage was skipped.
            Keys: dense_ms, sparse_ms, fusion_ms, rerank_ms, total_ms
    """
    chunks: list[RetrievedChunk]
    timing: dict[str, float] = field(default_factory=dict)


class Retriever:
    """Chain DenseRetriever + SparseRetriever → RRF fusion → Reranker.

    All three components are injected so they can be mocked in tests and
    shared (pooled) by the API layer.  A single instance is constructed at
    application startup and reused for every request.

    Args:
        dense:             Dense vector retriever (required).
        sparse:            BM25 Postgres retriever (required for hybrid mode).
        reranker:          Cross-encoder reranker (required for rerank mode).
        fusion_candidates: How many fused chunks to pass to the reranker.
                           Should be ≥ 2×top_k so the reranker has meaningful
                           reordering headroom. Default 10.
        rrf_k:             Smoothing constant passed to fuse(). Default 60.
    """

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: SparseRetriever,
        reranker: Reranker,
        fusion_candidates: int = 10,
        rrf_k: int = 60,
    ) -> None:
        self._dense = dense
        self._sparse = sparse
        self._reranker = reranker
        self._fusion_candidates = fusion_candidates
        self._rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        enable_sparse: bool = True,
        enable_rerank: bool = True,
    ) -> RetrievalResult:
        """Run the full hybrid retrieval pipeline.

        Args:
            query:          User query string.
            top_k:          Number of chunks to return.
            enable_sparse:  Include BM25 sparse results in fusion. Set False
                            for dense-only ablations.
            enable_rerank:  Apply cross-encoder reranking. Set False to return
                            RRF-ranked results directly.

        Returns:
            RetrievalResult with chunks sorted by the best available signal
            (rerank score > RRF score) and per-component timing in ms.

        Never raises. If a component fails, the pipeline degrades:
            sparse failure  → continues with dense-only
            reranker failure→ already handled inside Reranker.rerank()
            dense failure   → returns RetrievalResult(chunks=[])
        """
        t_total = time.perf_counter()
        timing: dict[str, float] = {}

        if not query.strip():
            return RetrievalResult(chunks=[], timing={"total_ms": 0.0})

        # --- dense retrieval (always runs) ---
        t0 = time.perf_counter()
        dense_chunks = self._dense.query(query)
        timing["dense_ms"] = (time.perf_counter() - t0) * 1000

        log.debug(
            "dense: %d results, top score %.4f",
            len(dense_chunks),
            dense_chunks[0].score if dense_chunks else 0.0,
        )

        # --- sparse retrieval (optional) ---
        sparse_chunks: list[RetrievedChunk] = []
        if enable_sparse:
            t0 = time.perf_counter()
            try:
                sparse_chunks = self._sparse.query(query)
            except Exception:
                # SparseRetriever.query() already catches internally;
                # this outer guard is defence-in-depth.
                log.warning("Sparse retriever raised unexpectedly — using dense only")
                sparse_chunks = []
            timing["sparse_ms"] = (time.perf_counter() - t0) * 1000

            log.debug(
                "sparse: %d results, top score %.4f",
                len(sparse_chunks),
                sparse_chunks[0].score if sparse_chunks else 0.0,
            )

        # --- RRF fusion ---
        t0 = time.perf_counter()
        fused = fuse(
            dense_chunks,
            sparse_chunks,
            k=self._rrf_k,
            top_k=self._fusion_candidates,
        )
        timing["fusion_ms"] = (time.perf_counter() - t0) * 1000

        log.debug("fusion: %d candidates after dedup", len(fused))

        # --- reranker ---
        if enable_rerank:
            t0 = time.perf_counter()
            chunks = self._reranker.rerank(query, fused, top_n=top_k)
            timing["rerank_ms"] = (time.perf_counter() - t0) * 1000

            log.debug(
                "rerank: top score %.4f, bottom score %.4f",
                chunks[0].score if chunks else 0.0,
                chunks[-1].score if chunks else 0.0,
            )
        else:
            chunks = fused[:top_k]

        timing["total_ms"] = (time.perf_counter() - t_total) * 1000

        log.info(
            "retrieve: %d chunks in %.0fms "
            "(dense=%.0f sparse=%.0f fusion=%.0f rerank=%.0f) "
            "query_len=%d",
            len(chunks),
            timing["total_ms"],
            timing.get("dense_ms", 0),
            timing.get("sparse_ms", 0),
            timing.get("fusion_ms", 0),
            timing.get("rerank_ms", 0),
            len(query),
        )

        return RetrievalResult(chunks=chunks, timing=timing)
