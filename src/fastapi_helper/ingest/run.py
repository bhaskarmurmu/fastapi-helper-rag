"""End-to-end ingestion: FastAPI docs + GitHub issues → Qdrant + Postgres."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import psycopg
from qdrant_client import QdrantClient

from fastapi_helper.ingest.chunker import Chunk, chunk_document
from fastapi_helper.ingest.docs_loader import load_docs
from fastapi_helper.ingest.embedder import Embedder
from fastapi_helper.ingest.indexer import COLLECTION, index_chunks
from fastapi_helper.ingest.issues_loader import load_issues

log = logging.getLogger(__name__)

_FASTAPI_REPO = Path(__file__).parents[4] / "data" / "raw" / "fastapi"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _months_ago(n: int) -> str:
    dt = datetime.now(tz=timezone.utc) - timedelta(days=n * 30)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _chunk_source(label: str, docs: list, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        try:
            chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
        except Exception as exc:
            log.warning("[%s] chunking failed for %s: %s", label, doc.source_url, exc)
    log.info("[%s] %d chunks from %d documents", label, len(chunks), len(docs))
    return chunks


def _embed_and_index(
    label: str,
    chunks: list[Chunk],
    embedder: Embedder,
    qdrant_url: str,
    postgres_url: str,
    batch_size: int,
) -> tuple[int, int]:
    """Embed + index chunks in batches. Returns (succeeded, failed)."""
    succeeded = 0
    failed = 0
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        try:
            vectors = embedder.encode([c.text for c in batch])
            index_chunks(
                batch,
                vectors,
                qdrant_url=qdrant_url,
                postgres_url=postgres_url,
                run_smoke_test=False,
            )
            succeeded += len(batch)
            log.info(
                "[%s] %d / %d chunks indexed (%.0f%%)",
                label,
                succeeded,
                total,
                100 * succeeded / total,
            )
        except Exception as exc:
            failed += len(batch)
            log.error(
                "[%s] batch %d–%d FAILED (%d chunks): %s",
                label,
                i,
                i + len(batch) - 1,
                len(batch),
                exc,
            )

    return succeeded, failed


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def _verify(qdrant_url: str, postgres_url: str, embedder: Embedder) -> dict[str, int]:
    """Count rows/vectors, run sample dense + BM25 queries, assert counts match."""
    qdrant = QdrantClient(url=qdrant_url)
    q_count = qdrant.get_collection(COLLECTION).points_count

    conn = psycopg.connect(postgres_url)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks")
        pg_total: int = cur.fetchone()[0]  # type: ignore[index]

        cur.execute(
            "SELECT source_url, chunk_index, source_type "
            "FROM chunks "
            "WHERE tsv @@ plainto_tsquery('english', %s) "
            "ORDER BY ts_rank(tsv, plainto_tsquery('english', %s)) DESC "
            "LIMIT 3",
            ("background tasks", "background tasks"),
        )
        bm25_hits = cur.fetchall()

        cur.execute(
            "SELECT source_type, COUNT(*) FROM chunks GROUP BY source_type ORDER BY source_type"
        )
        by_source = cur.fetchall()
    conn.close()

    # Dense query
    query_vec = embedder.encode(["how do I add middleware"], is_query=True)
    dense_hits = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vec[0].tolist(),
        limit=3,
    ).points

    log.info("=== Verification ===")
    log.info("Qdrant vectors : %d", q_count)
    log.info("Postgres rows  : %d", pg_total)

    if q_count != pg_total:
        log.error("COUNT MISMATCH: Qdrant=%d Postgres=%d", q_count, pg_total)
    else:
        log.info("Counts match ✓")

    log.info("Dense 'how do I add middleware' — top 3:")
    for hit in dense_hits:
        payload = hit.payload or {}
        log.info(
            "  score=%.4f  %s  chunk=%s",
            hit.score,
            payload.get("source_url", "?"),
            payload.get("chunk_index", "?"),
        )

    log.info("BM25 'background tasks' — top 3:")
    for row in bm25_hits:
        log.info("  [%s] %s  chunk=%s", row[2], row[0], row[1])

    log.info("Chunks by source type:")
    for row in by_source:
        log.info("  %s: %d chunks", row[0], row[1])

    return {"qdrant": q_count, "postgres": pg_total}


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------

def run(
    fastapi_repo: Path,
    *,
    chunk_size: int = 512,
    overlap: int = 50,
    issues_since: str | None = None,
    max_issues: int | None = None,
    embed_batch_size: int = 500,
    qdrant_url: str = "http://localhost:6333",
    postgres_url: str = "postgresql://postgres:postgres@localhost:5432/fastapi_helper",
    github_token: str | None = None,
) -> dict[str, int]:
    """Run the full ingestion pipeline. Returns per-source chunk counts."""

    # --- docs ---
    log.info("=== [1/5] Loading FastAPI docs from %s ===", fastapi_repo)
    docs = load_docs(fastapi_repo)
    log.info("Loaded %d doc files", len(docs))
    doc_chunks = _chunk_source("docs", docs, chunk_size, overlap)

    # --- issues ---
    since_label = issues_since or _months_ago(6)
    log.info("=== [2/5] Loading GitHub issues (since=%s) ===", since_label)
    issues = load_issues(
        since=issues_since or since_label,
        max_issues=max_issues,
        include_comments=True,
        token=github_token,
    )
    log.info("Loaded %d issues", len(issues))
    issue_chunks = _chunk_source("issues", issues, chunk_size, overlap)

    # --- embedder ---
    log.info("=== [3/5] Loading embedder ===")
    embedder = Embedder()

    # --- index docs ---
    log.info("=== [4/5] Indexing docs (%d chunks) ===", len(doc_chunks))
    doc_ok, doc_fail = _embed_and_index(
        "docs", doc_chunks, embedder, qdrant_url, postgres_url, embed_batch_size
    )

    # --- index issues ---
    log.info("=== [5/5] Indexing issues (%d chunks) ===", len(issue_chunks))
    issue_ok, issue_fail = _embed_and_index(
        "issues", issue_chunks, embedder, qdrant_url, postgres_url, embed_batch_size
    )

    results = {"docs": doc_ok, "issues": issue_ok}
    total_ok = doc_ok + issue_ok
    total_fail = doc_fail + issue_fail

    log.info("=== Ingestion complete ===")
    for label, count in results.items():
        log.info("  %s: %d chunks", label, count)
    log.info("  total indexed: %d", total_ok)
    if total_fail:
        log.warning("  total failed:  %d", total_fail)

    # --- verify ---
    counts = _verify(qdrant_url, postgres_url, embedder)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest FastAPI docs + GitHub issues")
    parser.add_argument(
        "fastapi_repo",
        type=Path,
        nargs="?",
        default=_FASTAPI_REPO,
        help=f"Path to cloned fastapi/fastapi repo (default: {_FASTAPI_REPO})",
    )
    parser.add_argument(
        "--issues-since",
        default=None,
        metavar="ISO8601",
        help="Only fetch issues updated after this datetime (default: 6 months ago)",
    )
    parser.add_argument(
        "--max-issues",
        type=int,
        default=None,
        metavar="N",
        help="Cap on issues fetched (useful for smoke-testing the pipeline)",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--embed-batch-size", type=int, default=500)
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
    )
    parser.add_argument(
        "--postgres-url",
        default=os.environ.get(
            "POSTGRES_URL",
            "postgresql://postgres:postgres@localhost:5432/fastapi_helper",
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.fastapi_repo.exists():
        log.error(
            "FastAPI repo not found at %s — clone it with:\n"
            "  git clone https://github.com/fastapi/fastapi %s",
            args.fastapi_repo,
            args.fastapi_repo,
        )
        sys.exit(1)

    run(
        args.fastapi_repo,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        issues_since=args.issues_since,
        max_issues=args.max_issues,
        embed_batch_size=args.embed_batch_size,
        qdrant_url=args.qdrant_url,
        postgres_url=args.postgres_url,
        github_token=os.environ.get("GITHUB_TOKEN"),
    )


if __name__ == "__main__":
    main()
