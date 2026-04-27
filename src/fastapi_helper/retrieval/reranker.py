"""Cross-encoder reranker: score (query, chunk) pairs and return top-N.

Model: BAAI/bge-reranker-v2-m3 (~550 MB, downloaded once to ~/.cache/huggingface).
On exception, degrades gracefully to the incoming RRF-ordered list.
"""
from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import CrossEncoder

from fastapi_helper.retrieval.types import RetrievedChunk

log = logging.getLogger(__name__)


class Reranker:
    """Wrap a BGE cross-encoder for single-pass (query, passage) scoring.

    The CrossEncoder is constructed once in __init__ and reused across all
    calls — never re-loaded per request.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        top_n: int = 5,
        batch_size: int = 32,
    ) -> None:
        log.info("Loading reranker model %r", model_name)
        self._model = CrossEncoder(model_name)
        self._top_n = top_n
        self._batch_size = batch_size

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        """Score each (query, chunk.text) pair and return the top-*top_n* chunks.

        Args:
            query:  The user query string.
            chunks: Candidate chunks (typically from fusion), in any order.
            top_n:  How many to return. Defaults to the instance default.

        Returns:
            Up to *top_n* chunks sorted by cross-encoder score descending.
            The score field carries the raw cross-encoder logit; the original
            RRF score is preserved in metadata["rrf_score"].
            Falls back to the RRF-ordered input on any model failure.
        """
        n = top_n if top_n is not None else self._top_n

        if not chunks:
            return []

        if not query.strip():
            log.warning("Reranker called with blank query — returning RRF order")
            return chunks[:n]

        pairs = [(query, c.text) for c in chunks]

        try:
            raw_scores: np.ndarray = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception:
            log.exception("Reranker model failed — falling back to RRF order")
            return chunks[:n]

        ranked = sorted(
            zip(chunks, raw_scores.tolist()),
            key=lambda pair: pair[1],
            reverse=True,
        )[:n]

        return [
            c.model_copy(update={
                "score": float(s),
                "metadata": {**c.metadata, "rrf_score": round(c.score, 8)},
            })
            for c, s in ranked
        ]
