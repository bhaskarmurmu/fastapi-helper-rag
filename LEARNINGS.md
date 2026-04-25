# Learnings

> Running notes from building fastapi-helper. First-person, dated, often messy.
> If something here is wrong, it's because I was wrong at the time of writing.

## Format
Each session: date, hours worked, what I did, what I learned, what I got stuck on, links to anything I read.


## 2026-04-25 — day 0: scaffolding

Spent today setting up the project structure. Decided to go with FastAPI docs +
GitHub issues as the corpus. FastAPI is well-known so the demo will land quickly,
docs are clean for chunking, and the issues give me messy real user queries.

Stack ended up being:
- Qdrant for vectors (self-hosted via docker)
- Postgres for metadata + cache
- Groq free tier for LLM
- BGE-small embeddings (local, no API cost)
- Langfuse self-hosted for tracing
- FastAPI backend, Next.js frontend
- Fly.io + Vercel for deploy

Things I'm not sure about yet:
- Whether reranking actually adds enough vs. plain hybrid to be worth the latency
- How long building the eval set will really take. Spec says 8 hours but I bet
  it's more once I'm actually doing it
- Self-hosted Langfuse fitting in Fly.io's free tier — might need to fall back
  to cloud free tier later

Plan for next session: Phase 1. config.py, models.py, exceptions, logging setup.

