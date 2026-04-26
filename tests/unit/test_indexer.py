"""Unit tests for indexer.py. Qdrant and psycopg are fully mocked."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from fastapi_helper.ingest.chunker import Chunk
from fastapi_helper.ingest.indexer import (
    COLLECTION,
    _ensure_pg_schema,
    _ensure_qdrant_collection,
    _upsert_postgres,
    _upsert_qdrant,
    chunk_id,
    index_chunks,
    smoke_test,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_chunk(source_url: str = "https://fastapi.tiangolo.com/", index: int = 0) -> Chunk:
    return Chunk(
        text="FastAPI is a modern, fast web framework for building APIs with Python.",
        source_url=source_url,
        source_type="docs",
        chunk_index=index,
        metadata={"title": "FastAPI", "section_path": []},
    )


def make_vector(dim: int = 384) -> np.ndarray:
    rng = np.random.default_rng(42)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def make_chunks(n: int) -> tuple[list[Chunk], np.ndarray]:
    chunks = [make_chunk(index=i) for i in range(n)]
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((n, 384)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return chunks, vecs


# ---------------------------------------------------------------------------
# chunk_id
# ---------------------------------------------------------------------------

def test_chunk_id_is_valid_uuid() -> None:
    c = make_chunk()
    cid = chunk_id(c)
    uuid.UUID(cid)  # raises if invalid


def test_chunk_id_is_deterministic() -> None:
    c = make_chunk()
    assert chunk_id(c) == chunk_id(c)


def test_chunk_id_differs_by_url() -> None:
    c1 = make_chunk(source_url="https://fastapi.tiangolo.com/a/")
    c2 = make_chunk(source_url="https://fastapi.tiangolo.com/b/")
    assert chunk_id(c1) != chunk_id(c2)


def test_chunk_id_differs_by_index() -> None:
    c1 = make_chunk(index=0)
    c2 = make_chunk(index=1)
    assert chunk_id(c1) != chunk_id(c2)


# ---------------------------------------------------------------------------
# _ensure_qdrant_collection
# ---------------------------------------------------------------------------

def _mock_qdrant_client(has_collection: bool = False) -> MagicMock:
    client = MagicMock()
    col = MagicMock()
    col.name = COLLECTION if has_collection else "other"
    client.get_collections.return_value.collections = [col] if has_collection else []
    return client


def test_creates_collection_when_missing() -> None:
    client = _mock_qdrant_client(has_collection=False)
    _ensure_qdrant_collection(client, dim=384)
    client.create_collection.assert_called_once()
    kwargs = client.create_collection.call_args
    assert kwargs.args[0] == COLLECTION or kwargs.kwargs.get("collection_name") == COLLECTION


def test_skips_create_when_collection_exists() -> None:
    client = _mock_qdrant_client(has_collection=True)
    _ensure_qdrant_collection(client, dim=384)
    client.create_collection.assert_not_called()


def test_collection_created_with_correct_dim() -> None:
    client = _mock_qdrant_client(has_collection=False)
    _ensure_qdrant_collection(client, dim=384)
    call_kwargs = client.create_collection.call_args.kwargs
    assert call_kwargs["vectors_config"].size == 384


# ---------------------------------------------------------------------------
# _upsert_qdrant
# ---------------------------------------------------------------------------

def test_upsert_qdrant_calls_upsert() -> None:
    chunks, vecs = make_chunks(5)
    client = MagicMock()
    _upsert_qdrant(chunks, vecs, client)
    client.upsert.assert_called()


def test_upsert_qdrant_batches_correctly() -> None:
    # 250 chunks with batch size 100 → 3 upsert calls
    chunks, vecs = make_chunks(250)
    client = MagicMock()
    with patch("fastapi_helper.ingest.indexer._QDRANT_BATCH", 100):
        _upsert_qdrant(chunks, vecs, client)
    assert client.upsert.call_count == 3


def test_upsert_qdrant_point_ids_are_chunk_ids() -> None:
    chunks, vecs = make_chunks(3)
    client = MagicMock()
    _upsert_qdrant(chunks, vecs, client)
    points = client.upsert.call_args.kwargs["points"]
    for c, p in zip(chunks, points):
        assert p.id == chunk_id(c)


def test_upsert_qdrant_payload_contains_text() -> None:
    chunks, vecs = make_chunks(2)
    client = MagicMock()
    _upsert_qdrant(chunks, vecs, client)
    points = client.upsert.call_args.kwargs["points"]
    for c, p in zip(chunks, points):
        assert p.payload["text"] == c.text
        assert p.payload["source_url"] == c.source_url


# ---------------------------------------------------------------------------
# _ensure_pg_schema
# ---------------------------------------------------------------------------

def _mock_pg_conn() -> MagicMock:
    conn = MagicMock()
    cursor = MagicMock()
    cursor.__enter__ = lambda s: s
    cursor.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor
    return conn


def test_ensure_pg_schema_executes_ddl() -> None:
    conn = _mock_pg_conn()
    _ensure_pg_schema(conn)
    conn.cursor.return_value.execute.assert_called_once()
    conn.commit.assert_called_once()


def test_ensure_pg_schema_ddl_contains_table() -> None:
    conn = _mock_pg_conn()
    _ensure_pg_schema(conn)
    sql = conn.cursor.return_value.execute.call_args.args[0]
    assert "CREATE TABLE IF NOT EXISTS chunks" in sql
    assert "tsvector" in sql
    assert "GIN" in sql


# ---------------------------------------------------------------------------
# _upsert_postgres
# ---------------------------------------------------------------------------

def test_upsert_postgres_executes_and_commits() -> None:
    chunks, _ = make_chunks(3)
    conn = _mock_pg_conn()
    _upsert_postgres(chunks, conn)
    conn.cursor.return_value.execute.assert_called_once()
    conn.commit.assert_called_once()


def test_upsert_postgres_sql_contains_on_conflict() -> None:
    chunks, _ = make_chunks(2)
    conn = _mock_pg_conn()
    _upsert_postgres(chunks, conn)
    sql = conn.cursor.return_value.execute.call_args.args[0]
    assert "ON CONFLICT" in sql
    assert "DO UPDATE" in sql


def test_upsert_postgres_passes_correct_row_count() -> None:
    n = 7
    chunks, _ = make_chunks(n)
    conn = _mock_pg_conn()
    _upsert_postgres(chunks, conn)
    args = conn.cursor.return_value.execute.call_args.args[1]
    ids_list = args[0]
    assert len(ids_list) == n


# ---------------------------------------------------------------------------
# index_chunks (integration of the pipeline, all IO mocked)
# ---------------------------------------------------------------------------

def test_index_chunks_empty_list_is_noop() -> None:
    with patch("fastapi_helper.ingest.indexer.QdrantClient") as mock_q, \
         patch("fastapi_helper.ingest.indexer.psycopg.connect") as mock_pg:
        index_chunks([], np.empty((0, 384), dtype=np.float32))
    mock_q.assert_not_called()
    mock_pg.assert_not_called()


def test_index_chunks_calls_both_stores() -> None:
    chunks, vecs = make_chunks(5)

    mock_qdrant = MagicMock()
    mock_qdrant.get_collections.return_value.collections = []

    mock_conn = _mock_pg_conn()

    with patch("fastapi_helper.ingest.indexer.QdrantClient", return_value=mock_qdrant), \
         patch("fastapi_helper.ingest.indexer.psycopg.connect", return_value=mock_conn):
        index_chunks(chunks, vecs, run_smoke_test=False)

    mock_qdrant.upsert.assert_called()
    mock_conn.cursor.return_value.execute.assert_called()
    mock_conn.close.assert_called_once()


def test_index_chunks_closes_pg_conn_on_error() -> None:
    chunks, vecs = make_chunks(3)

    mock_qdrant = MagicMock()
    mock_qdrant.get_collections.return_value.collections = []
    mock_qdrant.upsert.side_effect = RuntimeError("qdrant error")

    mock_conn = _mock_pg_conn()

    with patch("fastapi_helper.ingest.indexer.QdrantClient", return_value=mock_qdrant), \
         patch("fastapi_helper.ingest.indexer.psycopg.connect", return_value=mock_conn):
        with pytest.raises(RuntimeError, match="qdrant error"):
            index_chunks(chunks, vecs, run_smoke_test=False)

    mock_conn.close.assert_called_once()


# ---------------------------------------------------------------------------
# smoke_test
# ---------------------------------------------------------------------------

def test_smoke_test_passes_when_stores_consistent() -> None:
    chunk = make_chunk()
    vec = make_vector()
    cid = chunk_id(chunk)

    qdrant = MagicMock()
    # retrieve returns the chunk
    hit = MagicMock()
    hit.payload = {"source_url": chunk.source_url}
    qdrant.retrieve.return_value = [hit]
    # vector search returns same chunk as top-1
    top = MagicMock()
    top.id = cid
    top.score = 0.9999
    qdrant.query_points.return_value.points = [top]

    conn = _mock_pg_conn()
    # Postgres row present, tsvector not null
    conn.cursor.return_value.fetchone.return_value = (cid, True)

    result = smoke_test(chunk, vec, qdrant, conn)
    assert result["pg_row_found"] is True
    assert result["qdrant_top_score"] == pytest.approx(0.9999)


def test_smoke_test_raises_if_vector_search_empty() -> None:
    chunk = make_chunk()
    vec = make_vector()
    cid = chunk_id(chunk)

    qdrant = MagicMock()
    hit = MagicMock()
    hit.payload = {"source_url": chunk.source_url}
    qdrant.retrieve.return_value = [hit]
    qdrant.query_points.return_value.points = []  # no results

    conn = _mock_pg_conn()

    with pytest.raises(AssertionError, match="no results"):
        smoke_test(chunk, vec, qdrant, conn)


def test_smoke_test_raises_if_chunk_not_in_qdrant() -> None:
    chunk = make_chunk()
    vec = make_vector()

    qdrant = MagicMock()
    qdrant.retrieve.return_value = []  # not found in retrieve

    conn = _mock_pg_conn()

    with pytest.raises(AssertionError, match="not found in Qdrant"):
        smoke_test(chunk, vec, qdrant, conn)


def test_smoke_test_raises_if_chunk_not_in_postgres() -> None:
    chunk = make_chunk()
    vec = make_vector()
    cid = chunk_id(chunk)

    qdrant = MagicMock()
    hit = MagicMock()
    hit.payload = {"source_url": chunk.source_url}
    qdrant.retrieve.return_value = [hit]
    top = MagicMock()
    top.id = cid
    top.score = 1.0
    qdrant.query_points.return_value.points = [top]

    conn = _mock_pg_conn()
    conn.cursor.return_value.fetchone.return_value = None  # not found in pg

    with pytest.raises(AssertionError, match="not found in Postgres"):
        smoke_test(chunk, vec, qdrant, conn)
