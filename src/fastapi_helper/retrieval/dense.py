"""Dense retriever: embed query → Qdrant vector search → RetrievedChunk list."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint

from fastapi_helper.ingest.embedder import Embedder
from fastapi_helper.retrieval.types import RetrievedChunk

log = logging.getLogger(__name__)

_CORE_PAYLOAD_KEYS = frozenset({"text", "source_url", "source_type", "chunk_index"})


def _point_to_chunk(point: ScoredPoint) -> RetrievedChunk:
    payload: dict[str, Any] = point.payload or {}
    return RetrievedChunk(
        text=payload.get("text", ""),
        source_url=payload.get("source_url", ""),
        source_type=payload.get("source_type", "docs"),
        chunk_index=int(payload.get("chunk_index", 0)),
        score=float(point.score),
        metadata={k: v for k, v in payload.items() if k not in _CORE_PAYLOAD_KEYS},
    )


class DenseRetriever:
    """Wrap a Qdrant collection for single-vector approximate-nearest-neighbour search.

    The Embedder is injected so it can be shared across retrievers and mocked in tests.
    """

    def __init__(
        self,
        client: QdrantClient,
        embedder: Embedder,
        collection: str = "fastapi_helper_chunks",
        top_k: int = 30,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._collection = collection
        self._top_k = top_k

    def query(self, text: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Embed *text* and return up to *top_k* nearest chunks by dot-product score.

        Returns an empty list if the collection is empty or the query produces no hits.
        """
        k = top_k if top_k is not None else self._top_k
        if not text.strip():
            return []

        vector: np.ndarray = self._embedder.encode([text], is_query=True)[0]

        try:
            points = self._client.query_points(
                collection_name=self._collection,
                query=vector,
                limit=k,
                with_payload=True,
            ).points
        except Exception:
            log.exception("Qdrant query failed for collection %r", self._collection)
            return []

        chunks = [_point_to_chunk(p) for p in points]
        log.debug("Dense query returned %d results (top score %.4f)",
                  len(chunks), chunks[0].score if chunks else 0.0)
        return chunks
