# ADR-0003: Hybrid vs. dense retrieval

- **Status:** pending
- **Date:** *(fill in after Experiment 2 produces real numbers)*

## Context

## Decision

## Alternatives considered

- **Dense only:**
- **Sparse only:**

## Consequences

## Notes

*(Quote the recall@5 lift from Experiment 2 here.)*

## Validation (added during phase 3 implementation)

Verified empirically that BM25 surfaces issue corpus content that dense
retrieval ranks lower or misses entirely. Concrete example: for query
"pydantic validation error", BM25 returns issue chunks at positions 4-5
(rank 0.21-0.25) alongside docs. RRF fusion will lift these into the
final top-K when dense ranks them lower.

This confirms the original decision to invest in hybrid over dense-only.
Cost: one additional postgres query per request (~5-10ms). Benefit:
issue corpus is reachable for support-style queries, which is the
primary use case for this RAG.