"""Reciprocal Rank Fusion of N ranked RetrievedChunk lists.

Formula (Cormack et al. 2009):
    score(d) = Σ_r  1 / (k + rank_r(d))

rank is 1-indexed; k=60 is the standard default.
Chunks are deduplicated by (source_url, chunk_index).
The output score field carries the RRF score; original retriever scores are discarded.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from fastapi_helper.retrieval.types import RetrievedChunk

log = logging.getLogger(__name__)


def fuse(
    *lists: list[RetrievedChunk],
    k: int = 60,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse N ranked chunk lists into one using Reciprocal Rank Fusion.

    Args:
        *lists: Any number of ranked lists (dense, sparse, …). Empty lists are
                ignored — they contribute no scores but don't raise.
        k:      Smoothing constant (default 60). Higher k → gentler rank-position
                sensitivity.
        top_k:  If set, truncate the result to the top *top_k* chunks. None
                returns all fused chunks.

    Returns:
        Chunks sorted by descending RRF score. Ties are broken by
        (source_url, chunk_index) for determinism. Each chunk's score field
        holds the accumulated RRF score.
    """
    if not lists:
        return []

    scores: dict[tuple[str, int], float] = defaultdict(float)
    # Keep the first-seen RetrievedChunk for each key (text, metadata, etc. are identical
    # across retrievers for the same chunk).
    canonical: dict[tuple[str, int], RetrievedChunk] = {}

    for ranked_list in lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            key = (chunk.source_url, chunk.chunk_index)
            scores[key] += 1.0 / (k + rank)
            if key not in canonical:
                canonical[key] = chunk

    if not scores:
        return []

    # Primary sort: RRF score descending.
    # Tiebreaker: (source_url, chunk_index) ascending — fully deterministic.
    sorted_keys = sorted(
        scores,
        key=lambda key: (-scores[key], key[0], key[1]),
    )

    if top_k is not None:
        sorted_keys = sorted_keys[:top_k]

    return [
        canonical[key].model_copy(update={"score": round(scores[key], 8)})
        for key in sorted_keys
    ]
