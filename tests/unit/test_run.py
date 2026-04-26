"""Unit tests for run.py helpers. All IO mocked."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fastapi_helper.ingest.chunker import Chunk
from fastapi_helper.ingest.run import _chunk_source, _embed_and_index, _months_ago


# ---------------------------------------------------------------------------
# _months_ago
# ---------------------------------------------------------------------------

def test_months_ago_returns_iso_string() -> None:
    s = _months_ago(6)
    assert "T" in s and "Z" in s


def test_months_ago_is_in_the_past() -> None:
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(_months_ago(6).replace("Z", "+00:00"))
    assert dt < datetime.now(tz=timezone.utc)


def test_months_ago_six_is_earlier_than_one() -> None:
    from datetime import datetime, timezone
    six = datetime.fromisoformat(_months_ago(6).replace("Z", "+00:00"))
    one = datetime.fromisoformat(_months_ago(1).replace("Z", "+00:00"))
    assert six < one


# ---------------------------------------------------------------------------
# _chunk_source
# ---------------------------------------------------------------------------

def _make_doc(url: str = "https://fastapi.tiangolo.com/") -> MagicMock:
    doc = MagicMock()
    doc.source_url = url
    return doc


def test_chunk_source_calls_chunk_document() -> None:
    doc = _make_doc()
    with patch("fastapi_helper.ingest.run.chunk_document", return_value=[]) as mock_cd:
        _chunk_source("test", [doc], chunk_size=512, overlap=50)
    mock_cd.assert_called_once_with(doc, chunk_size=512, overlap=50)


def test_chunk_source_skips_failing_doc(caplog) -> None:
    import logging
    doc1, doc2 = _make_doc("https://a/"), _make_doc("https://b/")
    call_count = {"n": 0}

    def side_effect(doc, **kwargs):
        call_count["n"] += 1
        if doc.source_url == "https://a/":
            raise ValueError("parse error")
        return [MagicMock(spec=Chunk)]

    with patch("fastapi_helper.ingest.run.chunk_document", side_effect=side_effect):
        with caplog.at_level(logging.WARNING):
            chunks = _chunk_source("test", [doc1, doc2], chunk_size=512, overlap=50)

    assert len(chunks) == 1
    assert "chunking failed" in caplog.text


# ---------------------------------------------------------------------------
# _embed_and_index
# ---------------------------------------------------------------------------

def _make_chunk(i: int = 0) -> Chunk:
    return Chunk(
        text=f"text {i}",
        source_url="https://fastapi.tiangolo.com/",
        source_type="docs",
        chunk_index=i,
        metadata={},
    )


def _make_embedder(dim: int = 384) -> MagicMock:
    emb = MagicMock()
    emb.encode.side_effect = lambda texts, **kw: np.ones(
        (len(texts), dim), dtype=np.float32
    )
    return emb


def test_embed_and_index_calls_index_chunks() -> None:
    chunks = [_make_chunk(i) for i in range(3)]
    embedder = _make_embedder()
    with patch("fastapi_helper.ingest.run.index_chunks") as mock_ix:
        ok, fail = _embed_and_index(
            "test", chunks, embedder, "http://q:6333", "pg://", batch_size=10
        )
    mock_ix.assert_called_once()
    assert ok == 3
    assert fail == 0


def test_embed_and_index_batches_correctly() -> None:
    chunks = [_make_chunk(i) for i in range(7)]
    embedder = _make_embedder()
    with patch("fastapi_helper.ingest.run.index_chunks") as mock_ix:
        ok, fail = _embed_and_index(
            "test", chunks, embedder, "http://q:6333", "pg://", batch_size=3
        )
    assert mock_ix.call_count == 3  # ceil(7/3) = 3 batches
    assert ok == 7
    assert fail == 0


def test_embed_and_index_isolates_batch_failure() -> None:
    chunks = [_make_chunk(i) for i in range(6)]
    embedder = _make_embedder()
    call_n = {"n": 0}

    def side_effect(*args, **kwargs):
        call_n["n"] += 1
        if call_n["n"] == 2:
            raise RuntimeError("Qdrant down")

    with patch("fastapi_helper.ingest.run.index_chunks", side_effect=side_effect):
        ok, fail = _embed_and_index(
            "test", chunks, embedder, "http://q:6333", "pg://", batch_size=2
        )

    # Batch 1 (2 chunks) succeeds, batch 2 (2 chunks) fails, batch 3 (2 chunks) succeeds
    assert ok == 4
    assert fail == 2


def test_embed_and_index_empty_chunks() -> None:
    embedder = _make_embedder()
    with patch("fastapi_helper.ingest.run.index_chunks") as mock_ix:
        ok, fail = _embed_and_index(
            "test", [], embedder, "http://q:6333", "pg://", batch_size=500
        )
    mock_ix.assert_not_called()
    assert ok == 0
    assert fail == 0
