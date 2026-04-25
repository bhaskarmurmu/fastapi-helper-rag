# fastapi-helper

> An assistant that answers FastAPI questions using the official docs and 5,000+ closed GitHub issues — with hybrid retrieval, citation-grounded answers, and CI-gated evaluation.

[![CI](https://github.com/{{your_username}}/fastapi-helper/actions/workflows/ci.yml/badge.svg)](https://github.com/{{your_username}}/fastapi-helper/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Live demo:** [{{your-app}}.vercel.app]({{your-app}}.vercel.app)
**Demo video (90s):** [{{loom-link}}]({{loom-link}})

---

## Why I built this

I wanted my first end-to-end production project to teach me the parts of RAG that tutorials skip: evaluation, observability, retrieval tradeoffs, and shipping something other people can actually use. Customer-support-style RAG was the most realistic shape — most companies building RAG today are building exactly this.

I picked FastAPI as the corpus for three reasons. First, the docs are excellent and openly licensed, so retrieval quality has a high ceiling. Second, the GitHub issue tracker has 5,000+ closed issues representing real, paraphrased questions with authoritative resolutions — perfect natural eval signal. Third, building a tool *about* FastAPI *with* FastAPI was a forcing function to use the framework properly.

Constraint: zero budget. Every component runs on a free tier or self-hosts.

## Demo

Try one of:

- "How do I add CORS middleware?"
- "What's the difference between path parameters and query parameters?"
- "How do I run background tasks after returning a response?"
- "Can I use FastAPI with Django ORM?" *(answer should cite issues)*

The system refuses out-of-scope questions instead of confabulating — try "What's the capital of France?" and you'll see what that looks like.

## Architecture

```mermaid
{{paste your final mermaid diagram here}}
```

**Three pipelines:** ingestion (offline), retrieval (online), generation (online). Full architecture detail in [ARCHITECTURE.md](ARCHITECTURE.md).

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| LLM | Groq Llama 3.3 70B (Gemini Flash fallback) | Free, fast, strong enough with strict prompting |
| Embedding | BGE-small-en-v1.5 (local) | 384-dim, top of MTEB at its size, no API cost |
| Vector DB | Qdrant (Docker) | Best filtering, rich Python SDK, self-hostable |
| Sparse | Postgres `tsvector` | One less service to run, BM25-equivalent quality |
| Reranker | BGE-reranker-v2-m3 (local) | Best open-source cross-encoder |
| Backend | FastAPI + Uvicorn | The corpus is *about* FastAPI |
| Frontend | Next.js 15 + Tailwind + shadcn/ui | Polish that Streamlit can't match |
| Eval | Ragas with Gemini Flash judge | RAG-specific metrics, separate model from generator |
| Observability | Self-hosted Langfuse | Free, full traces, screenshot in `docs/images/` |
| Deploy | Fly.io (backend) + Vercel (frontend) | Free tiers, native Docker, native Next.js |

Every choice has a one-pager in [`docs/adr/`](docs/adr/).

## Features

- Hybrid retrieval (dense + Postgres BM25) fused with Reciprocal Rank Fusion
- BGE cross-encoder reranking, top-30 → top-5
- Streaming responses via Server-Sent Events
- Inline `[N]` citations parsed and linked back to source URLs
- Refusal mode for out-of-scope queries (no confabulation)
- Semantic cache (DiskCache, cosine similarity > 0.97)
- Full Langfuse tracing per request with span-level breakdown
- CI gate that blocks PRs dropping faithfulness below 0.85 or recall@5 below 0.75
- Feedback (thumbs up/down) writes to Postgres for offline review

## Evaluation results

Hand-curated 50-question eval set, stratified across docs-answerable / issue-answerable / mixed / out-of-scope buckets. Three documented experiments in [EXPERIMENTS.md](EXPERIMENTS.md).

| Configuration | recall@5 | faithfulness | answer_relevance | p50 latency |
|---|---|---|---|---|
| Dense only | {{n}} | {{n}} | {{n}} | {{n}} |
| Hybrid (dense+BM25) | {{n}} | {{n}} | {{n}} | {{n}} |
| Hybrid + rerank *(prod)* | **{{n}}** | **{{n}}** | **{{n}}** | {{n}} |

Refusal rate on 10 out-of-scope questions: **{{n}}%**. False refusal on 40 in-scope: **{{n}}%**.

> Methodology, full per-experiment tables, and discussion in [EXPERIMENTS.md](EXPERIMENTS.md).

## What I learned

> *fill in 4–6 bullets at the end, drawn from your real `LEARNINGS.md`. Specifics, not platitudes.*

- {{...}}

## What I'd do differently

> *fill in honestly at the end.*

- {{...}}

## Limitations

- **Cold starts:** with `min_machines_running=0` on Fly.io, the first request after idle takes ~30s for BGE + reranker to load.
- **Refusal isn't perfect:** {{n}}% of in-scope questions get an incorrect refusal (false-refusal rate).
- **Single-turn only:** no multi-turn conversation memory. Each question is independent.
- **Stale corpus:** ingestion is a manual command, not a webhook.
- **No PII or auth:** single shared API key, public-facing demo.

## Local setup

```bash
# Prereqs: Python 3.11, Docker, Node.js 20, pnpm
git clone https://github.com/{{your_username}}/fastapi-helper.git
cd fastapi-helper

# 1. Python env
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt -r requirements-dev.txt

# 2. Services (Qdrant + Postgres + Langfuse)
cp .envrc.example .envrc  # edit with your keys
docker compose up -d
direnv allow

# 3. Ingest (start small)
MAX_ISSUES=200 make ingest

# 4. Backend
make dev                  # uvicorn fastapi_helper.api.main:app --reload

# 5. Frontend (separate terminal)
cd frontend && pnpm install && pnpm dev

# 6. Visit http://localhost:3000
```

## Deployment

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the full Fly.io + Vercel walkthrough.

## Project structure

```
src/fastapi_helper/   # backend Python package
  ingest/             # docs + issues loaders, chunker, embedder, indexer
  retrieval/          # dense, sparse, fusion, reranker, pipeline
  generation/         # prompts, LLM provider abstraction, generator
  cache/              # semantic cache
  observability/      # Langfuse tracing
  api/                # FastAPI routes + middleware
frontend/             # Next.js 15 app
eval/                 # eval set + Ragas runner + experiment results
tests/                # unit + integration
docs/adr/             # 7 architecture decision records
```

## Roadmap

- [ ] Multi-turn conversation memory
- [ ] Query rewriting for chat history
- [ ] Webhook-driven re-ingestion on FastAPI repo updates
- [ ] Replace BGE reranker with a fine-tuned model on collected feedback
- [ ] A/B harness comparing two retrievers on live traffic

## License

MIT. The corpus is sourced from the FastAPI project (also MIT) — citations preserve attribution to original docs and issue authors.
