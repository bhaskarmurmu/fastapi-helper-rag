"""Sparse retriever: Postgres tsvector BM25 search → RetrievedChunk list."""
from __future__ import annotations

import logging
from typing import Any

import psycopg
import psycopg.rows

from fastapi_helper.retrieval.types import RetrievedChunk

log = logging.getLogger(__name__)

# plainto_tsquery handles natural-language input safely — special chars are ignored.
# CTE avoids computing the tsquery twice (rank + WHERE clause).
_SQL = """\
WITH q AS (SELECT plainto_tsquery('english', %s) AS tsq)
SELECT
    source_url,
    source_type,
    chunk_index,
    text,
    metadata,
    ts_rank(tsv, q.tsq) AS rank
FROM chunks, q
WHERE tsv @@ q.tsq
ORDER BY rank DESC, chunk_index ASC
LIMIT %s
"""


def _row_to_chunk(row: dict[str, Any]) -> RetrievedChunk:
    return RetrievedChunk(
        text=row["text"],
        source_url=row["source_url"],
        source_type=row["source_type"],
        chunk_index=int(row["chunk_index"]),
        score=float(row["rank"]),
        metadata=row.get("metadata") or {},
    )


class SparseRetriever:
    """Postgres full-text search using tsvector + ts_rank.

    A psycopg.Connection is injected so it can be mocked in tests and
    managed (pooled) by the caller in the API layer.
    """

    def __init__(
        self,
        conn: psycopg.Connection,  # type: ignore[type-arg]
        top_k: int = 30,
    ) -> None:
        self._conn = conn
        self._top_k = top_k

    def query(self, text: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Run a BM25 full-text search and return up to *top_k* chunks.

        Returns an empty list on blank input or any database error.
        """
        k = top_k if top_k is not None else self._top_k
        if not text.strip():
            return []

        try:
            with self._conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(_SQL, (text.strip(), k))
                rows = cur.fetchall()
        except Exception:
            log.exception("Postgres BM25 query failed")
            return []

        chunks = [_row_to_chunk(row) for row in rows]
        log.debug(
            "Sparse query returned %d results (top rank %.4f)",
            len(chunks),
            chunks[0].score if chunks else 0.0,
        )
        return chunks
