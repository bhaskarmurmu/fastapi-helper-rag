"""Shared types for the retrieval pipeline."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """Single chunk returned by any retrieval step.

    score carries whatever is most meaningful for the current stage:
    dot-product similarity after dense retrieval, ts_rank after BM25,
    RRF score after fusion, cross-encoder logit after reranking.
    """

    text: str
    source_url: str
    source_type: str          # "docs" | "issue" — str so new types don't crash retrieval
    chunk_index: int
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
