# Resume bullet

> Fill in with real numbers after eval runs complete.

**fastapi-helper** — production RAG assistant over FastAPI docs + GitHub issues
*Personal project · Python, FastAPI, Next.js, Qdrant, Langfuse · [GitHub] [Live]*

- Built and deployed an end-to-end RAG service answering FastAPI questions over a corpus of {{n}} doc chunks and {{n}} GitHub issues; achieved {{n}}% Ragas faithfulness and recall@5 of {{n}} on a 50-question hand-curated eval set.
- Designed hybrid retrieval (dense BGE embeddings + Postgres BM25 fused via RRF) with BGE-reranker-v2-m3, lifting recall@5 from {{n_baseline}} (dense-only) to {{n_final}} — documented across 3 controlled experiments with cost/latency tradeoffs.
- Shipped CI-gated evaluation, semantic caching, and self-hosted Langfuse observability; full stack runs on free-tier infrastructure (Fly.io + Vercel) with zero monthly cost.
