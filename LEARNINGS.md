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



## 2026-04-27 — day 1 / phase 1: plumbing

Plumbing day. config.py, models.py, exceptions.py, logging_setup.py. No
interesting RAG logic yet but everything has tests so I won't be debugging
config issues at 2am later.

Choices:
- pydantic-settings for config. reads env vars, validates types, exposes
  a module-level `settings` singleton.
- json structured logging from day 1. probably overkill locally but pays
  off when langfuse + prod logs come in.
- pythonpath = ["src"] in pyproject.toml so I don't need pip install -e .
  every time.

Two windows-specific things hit me today:
1. uv isn't on the default PATH after install. Had to call it as
   C:\Users\bhmurmu\.local\bin\uv.exe. Need to add to PATH or alias it.
2. PowerShell blocks .ps1 scripts by default — venv activation failed
   with "running scripts is disabled." Fixed with
   `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`. Should add this
   to README for any Windows users.

30 unit tests passing. Mostly env var loading and llm_provider validation.
Felt over-engineered for config but I bet I'll be glad later when something
silently breaks and the test catches it.

Next: phase 2, ingestion. docs_loader → chunker → embedder → indexer.
This is where the actual RAG work starts and the BGE model gets downloaded
locally. Need to make sure docker is running and i have a github token
ready.