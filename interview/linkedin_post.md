# LinkedIn post draft

> Fill in {{placeholders}} with real numbers after the project ships.

Just shipped fastapi-helper — my first production-deployed RAG project.

The TL;DR: an assistant that answers FastAPI questions by retrieving over the official docs and 5,000+ closed GitHub issues, with citation-grounded answers and a CI gate that blocks regressions in retrieval quality.

A few things I'm proud of:

→ Hand-built a 50-question eval set across docs / issues / mixed / out-of-scope buckets. Without that, every "improvement" is a vibe.
→ Three documented experiments — chunk size, dense vs. hybrid, with vs. without reranker — with real numbers, not hand-waving.
→ Hybrid retrieval lifted recall@5 from {{n_baseline}} to {{n_final}}; reranking added ~{{n_ms}}ms p50 but bought {{n_pct}}pp on faithfulness.
→ Full stack on free-tier infra: BGE locally, Qdrant in Docker, Groq for generation, self-hosted Langfuse. Zero monthly cost.
→ CI gate that fails PRs dropping recall@5 below 0.75 or faithfulness below 0.85.

Lessons that surprised me:
{{1–2 specifics from your LEARNINGS.md}}

Code, eval results, and writeup → {{github}}
Live demo → {{vercel}}

Build was ~{{n}} hours over {{n}} weeks.
