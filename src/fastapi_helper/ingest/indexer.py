"""Write chunks + embeddings to Qdrant (dense) and Postgres (tsvector/BM25)."""
from __future__ import annotations

import hashlib
import json
import logging
import uuid

import numpy as np
import psycopg
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from fastapi_helper.ingest.chunker import Chunk

log = logging.getLogger(__name__)

COLLECTION = "fastapi_chunks"
_QDRANT_BATCH = 100

_DDL = """\
CREATE TABLE IF NOT EXISTS chunks (
    id          UUID PRIMARY KEY,
    source_url  TEXT NOT NULL,
    source_type TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    metadata    JSONB,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);
"""


def chunk_id(chunk: Chunk) -> str:
    """Stable UUID from (source_url, chunk_index) — enables idempotent upserts."""
    key = f"{chunk.source_url}:{chunk.chunk_index}"
    raw = hashlib.sha256(key.encode()).digest()[:16]
    return str(uuid.UUID(bytes=raw))


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

def _ensure_qdrant_collection(client: QdrantClient, dim: int) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.DOT),
        )
        log.info("Created Qdrant collection %r (dim=%d)", COLLECTION, dim)
    else:
        log.debug("Qdrant collection %r already exists", COLLECTION)


def _upsert_qdrant(
    chunks: list[Chunk], vectors: np.ndarray, client: QdrantClient
) -> None:
    for i in range(0, len(chunks), _QDRANT_BATCH):
        batch_c = chunks[i : i + _QDRANT_BATCH]
        batch_v = vectors[i : i + _QDRANT_BATCH]
        points = [
            PointStruct(
                id=chunk_id(c),
                vector=v.tolist(),
                payload={
                    "source_url": c.source_url,
                    "source_type": c.source_type,
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                    **c.metadata,
                },
            )
            for c, v in zip(batch_c, batch_v)
        ]
        client.upsert(collection_name=COLLECTION, points=points, wait=True)
        log.debug("Qdrant upsert batch %d–%d", i, i + len(batch_c) - 1)


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------

def _ensure_pg_schema(conn: psycopg.Connection) -> None:  # type: ignore[type-arg]
    with conn.cursor() as cur:
        cur.execute(_DDL)
    conn.commit()
    log.debug("Postgres schema ensured")


def _upsert_postgres(
    chunks: list[Chunk], conn: psycopg.Connection  # type: ignore[type-arg]
) -> None:
    ids = [chunk_id(c) for c in chunks]
    urls = [c.source_url for c in chunks]
    types = [c.source_type for c in chunks]
    indices = [c.chunk_index for c in chunks]
    texts = [c.text for c in chunks]
    metas = [json.dumps(c.metadata) for c in chunks]

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunks (id, source_url, source_type, chunk_index, text, metadata)
            SELECT * FROM unnest(
                %s::uuid[], %s::text[], %s::text[], %s::int[], %s::text[], %s::jsonb[]
            )
            ON CONFLICT (id) DO UPDATE SET
                text     = EXCLUDED.text,
                metadata = EXCLUDED.metadata
            """,
            (ids, urls, types, indices, texts, metas),
        )
    conn.commit()
    log.debug("Postgres upserted %d rows", len(chunks))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(
    sample_chunk: Chunk,
    sample_vector: np.ndarray,
    qdrant: QdrantClient,
    conn: psycopg.Connection,  # type: ignore[type-arg]
) -> dict[str, object]:
    cid = chunk_id(sample_chunk)

    # Qdrant: retrieve by ID
    hits = qdrant.retrieve(collection_name=COLLECTION, ids=[cid], with_payload=True)
    if not hits:
        raise AssertionError(f"chunk {cid} not found in Qdrant")
    if hits[0].payload is None or hits[0].payload.get("source_url") != sample_chunk.source_url:
        raise AssertionError(f"Qdrant payload mismatch for chunk {cid}")

    # Qdrant: vector search — the sample chunk's own embedding should score ≈ 1.0
    # (top-1 UUID may differ from cid if identical text was indexed under another URL,
    # but a perfect dot-product score proves the retrieval path is working)
    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=sample_vector.tolist(),
        limit=1,
    ).points
    if not results:
        raise AssertionError("Vector search returned no results")
    top_score = float(results[0].score)
    if results[0].id != cid and top_score < 0.9999:
        raise AssertionError(
            f"Vector search top-1 {results[0].id!r} score={top_score:.4f} — "
            f"expected {cid!r} or score ≥ 0.9999"
        )

    # Postgres: row present and tsvector populated
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, tsv IS NOT NULL FROM chunks WHERE id = %s::uuid", (cid,)
        )
        row = cur.fetchone()
    if row is None:
        raise AssertionError(f"chunk {cid} not found in Postgres")
    if not row[1]:
        raise AssertionError("tsvector column is NULL")

    log.info(
        "Smoke test passed — Qdrant top score %.4f, Postgres row confirmed, tsvector present",
        top_score,
    )
    return {"qdrant_top_score": top_score, "pg_row_found": True}


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def index_chunks(
    chunks: list[Chunk],
    vectors: np.ndarray,
    *,
    qdrant_url: str = "http://localhost:6333",
    postgres_url: str = "postgresql://postgres:postgres@localhost:5432/fastapi_helper",
    run_smoke_test: bool = True,
) -> None:
    """Upsert chunks + pre-computed embeddings into Qdrant and Postgres.

    Idempotent: re-running with the same chunks overwrites in place.
    Qdrant is written first; a failed Postgres write can be repaired by re-running.
    """
    if not chunks:
        log.warning("index_chunks called with empty list — nothing to do")
        return

    dim = int(vectors.shape[1])
    qdrant = QdrantClient(url=qdrant_url)
    _ensure_qdrant_collection(qdrant, dim)

    conn = psycopg.connect(postgres_url)
    try:
        _ensure_pg_schema(conn)
        _upsert_qdrant(chunks, vectors, qdrant)
        _upsert_postgres(chunks, conn)
        log.info("Indexed %d chunks into Qdrant + Postgres", len(chunks))

        if run_smoke_test:
            smoke_test(chunks[0], vectors[0], qdrant, conn)
    finally:
        conn.close()
