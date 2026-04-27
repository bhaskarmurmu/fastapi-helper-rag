# ADR-0004: Reranker yes or no

- **Status:** accepted
- **Date:** 2026-04-27

## Context

After hybrid retrieval (dense + sparse RRF), the fused list is ranked by
accumulated RRF score — a signal that rewards appearance in multiple
retriever results but is blind to fine-grained query semantics. A
cross-encoder reranker reads the query and each candidate passage jointly,
producing a score that captures relevance at the token level rather than
rank position.

The question was whether the precision lift justifies the added latency and
operational complexity.

## Decision

Add a cross-encoder reranker as the final stage, operating on the RRF
fusion top-10 and returning top-5 to the LLM.

**Default model: `BAAI/bge-reranker-base`** (~278M params, ~110 MB on disk).

See the model choice section below for why this is not `bge-reranker-v2-m3`.

## Alternatives considered

- **No reranker — use top-5 from RRF directly:** Real-data verification
  showed RRF rank-1 is not always semantically best. On a "background tasks
  dependency injection database session" query, the highest-RRF chunk
  (`/advanced/advanced-dependencies`) was demoted to rank 4 by the
  cross-encoder, which promoted a GitHub issue thread to rank 1. The issue
  thread contained a concrete working example that better answered the
  query. Without reranking, the LLM would have received the wrong leading
  context.

- **Re-rank all fusion results (not just top-10):** Diminishing returns:
  cross-encoders are slow on CPU. Scoring 10 pairs at ~100 ms/pair is
  already ~1 s; scoring 30 would triple that. Top-10 captures enough
  diversity for the reranker to reorder meaningfully while keeping latency
  bounded.

- **Use a BM25-based scoring heuristic instead:** Doesn't capture
  semantic relevance. The reranker's value is joint query+passage scoring,
  not keyword counting a second time.

## Consequences

- Reranker adds ~1–2 s latency on CPU (bge-reranker-base) or ~15–22 s
  on CPU (bge-reranker-v2-m3). See model choice section.
- Model is loaded once in `__init__` and reused; no per-request reload.
- On any model failure, `rerank()` degrades gracefully to returning the
  RRF-ordered input, so retrieval quality falls back but the API never
  errors out.
- Original RRF score is preserved as `metadata["rrf_score"]`; the `score`
  field carries the cross-encoder logit for downstream consumers that need
  the most meaningful relevance signal.

---

## Model size / latency tradeoff

### The models

| Model | Params | Disk | CPU latency (10 pairs) | Notes |
|---|---|---|---|---|
| `bge-reranker-base` | ~278 M | ~110 MB | ~1–2 s | English-optimised; good for technical docs |
| `bge-reranker-v2-m3` | ~568 M | ~550 MB | ~15–22 s | Multilingual, higher quality |

Observed on WSL2, CPU-only (no GPU), scoring 10 (query, passage) pairs.

### Why base for development

During phase 3 implementation, `bge-reranker-v2-m3` was tried first. The
cross-encoder was visibly doing real semantic work — on the
"background tasks + DI + DB session" query it promoted GitHub issue
#15111 from fusion rank 2 to reranker rank 1, ahead of the canonical
tutorial. But each rerank call took 15–22 s, making interactive
development and test-iteration impractical.

Switching to `bge-reranker-base` drops latency to ~1–2 s with no
observable quality regression on English technical content. The ranking
decisions on the same three test queries were equivalent in all
meaningful respects.

### Production path

If the production host has a GPU (even a small one — a T4 or equivalent),
`bge-reranker-v2-m3` is worth reverting to. GPU inference for 10 pairs
runs in ~30–50 ms, well within interactive response bounds. The model name
is controlled by `Settings.reranker_model` / `RERANKER_MODEL` env var, so
no code change is required — only an env var override on the deploy target.

`fly.toml` defaults to `bge-reranker-base` to match the CPU `shared-cpu-1x`
machine size. Upgrade the machine tier and set
`RERANKER_MODEL=BAAI/bge-reranker-v2-m3` as a Fly secret to switch.

### Quality note

The base model uses the same architecture and training objective as v2-m3,
just smaller. For a single-language (English), single-domain (FastAPI
technical docs + issues) corpus, the quality difference is minor. Ragas
evaluation in phase 5 will quantify the gap if needed. If precision@5
drops measurably on the eval set, revert to v2-m3 and add a GPU machine.
