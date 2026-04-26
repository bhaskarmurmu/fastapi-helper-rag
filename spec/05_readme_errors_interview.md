# `fastapi-helper` — Spec Part 5: README, Errors, Interview Assets

## 17. README.md Template (story-driven, full content)

> Copy this into your repo root as `README.md`. Bracketed `{{placeholders}}` get filled in **with your real numbers and notes** as you build. Don't fill them in until you have real data — empty placeholders force you to come back.

```markdown
# fastapi-helper

> An assistant that answers FastAPI questions using the official docs and 5,000+ closed GitHub issues — with hybrid retrieval, citation-grounded answers, and CI-gated evaluation.

[![CI](https://github.com/{{your_username}}/fastapi-helper/actions/workflows/ci.yml/badge.svg)](https://github.com/{{your_username}}/fastapi-helper/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Live demo:** [{{your-app}}.vercel.app]({{your-app}}.vercel.app)
**Demo video (90s):** [{{loom-link}}]({{loom-link}})

![demo gif](docs/images/demo.gif)

---

## Why I built this

I wanted my first end-to-end production project to teach me the parts of RAG that tutorials skip: evaluation, observability, retrieval tradeoffs, and shipping something other people can actually use. Customer-support-style RAG was the most realistic shape — most companies building RAG today are building exactly this.

I picked FastAPI as the corpus for three reasons. First, the docs are excellent and openly licensed, so retrieval quality has a high ceiling. Second, the GitHub issue tracker has 5,000+ closed issues representing real, paraphrased questions with authoritative resolutions — perfect natural eval signal. Third, building a tool *about* FastAPI *with* FastAPI was a forcing function to use the framework properly.

Constraint: zero budget. Every component runs on a free tier or self-hosts. That constraint pushed me toward local embeddings, Qdrant in Docker, Groq for inference, and self-hosted Langfuse — all of which turned out to be educational in ways a "just use OpenAI" version wouldn't have been.

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

- {{e.g., "My first chunker split markdown code blocks across chunks 38% of the time. Switching to a code-fence-aware splitter improved recall@5 on code-heavy questions by N points."}}
- {{e.g., "BM25 outperformed dense retrieval on 7/15 issue-answerable questions because issue titles use exact API names that get dampened by semantic embeddings."}}
- {{...}}

## What I'd do differently

> *fill in honestly at the end. The point is reflection, not self-flagellation.*

- {{e.g., "I'd build the eval set first, before any retrieval code. I built it after my first retrieval pass and had to redo measurements."}}
- {{...}}

## Limitations

- **Cold starts:** with `min_machines_running=0` on Fly.io, the first request after idle takes ~30s for BGE + reranker to load. Mitigated by surfacing a "warming up" state in the UI.
- **Refusal isn't perfect:** {{n}}% of in-scope questions get an incorrect refusal (false-refusal rate). Tuning the refusal threshold is one of the main areas of future work.
- **Single-turn only:** no multi-turn conversation memory. Each question is independent.
- **Stale corpus:** ingestion is a manual command, not a webhook. The corpus is whatever was indexed at last `make ingest`.
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
```

---

## 24. Top 10 Anticipated Errors with Pre-Written Fixes

These are the errors most likely to bite during the build. Each has a fix.

### Error 1: Embedding dimension mismatch
```
qdrant_client.http.exceptions.UnexpectedResponse: Wrong vector size
```
**Cause:** You created the Qdrant collection with one dim and then tried to upsert vectors of another dim. Most often: switched embedding models without recreating the collection.
**Fix:** Drop the collection and recreate. Add this to the indexer:
```python
existing = client.get_collections().collections
if not any(c.name == collection_name for c in existing):
    client.create_collection(...)
elif client.get_collection(collection_name).config.params.vectors.size != dim:
    client.delete_collection(collection_name)
    client.create_collection(...)
```

### Error 2: BGE model OOMs the 256MB Fly VM
```
Killed
```
**Cause:** BGE-small + reranker together use ~500MB RAM at peak.
**Fix:** Bump VM to 512MB or 1GB in `fly.toml`. The free allowance covers small VMs but you may need to balance memory across them. Document the tradeoff in `LEARNINGS.md`.

### Error 3: Groq rate limit during eval runs
```
groq.RateLimitError: Rate limit reached for ...
```
**Cause:** Eval runs hammer the API faster than 30 RPM.
**Fix:** Add exponential-backoff retry to the LLM provider:
```python
import time, random
async def stream_with_retry(self, *args, **kwargs):
    for attempt in range(5):
        try:
            async for tok in self._raw_stream(*args, **kwargs):
                yield tok
            return
        except groq.RateLimitError:
            await asyncio.sleep(2 ** attempt + random.random())
    raise
```
And during eval, sleep 1s between calls.

### Error 4: Gemini judge timing out / quota errors during Ragas
**Cause:** Gemini free tier 15 RPM. Ragas batches but can still burst.
**Fix:** In `ragas_runner.py`, process in chunks of 5 with sleeps between:
```python
results = []
for i in range(0, len(rows), 5):
    chunk = rows[i:i+5]
    results.append(evaluate(Dataset.from_list(chunk), metrics=metrics))
    time.sleep(20)
```

### Error 5: SSE stream cuts off in production
**Cause:** Fly.io's HTTP proxy default request timeout (60s) closes long streams. Or Vercel's edge timeout for the proxy route.
**Fix:** Two parts. (1) Set Fly's `[http_service.idle_timeout]` and avoid long streams. (2) Set `runtime = "nodejs"` and `maxDuration = 60` on the Next.js proxy route. (3) Stream tokens promptly — don't buffer.

### Error 6: GitHub issues ingestion stops at 1,000 results
**Cause:** GitHub's `state=closed` issues endpoint paginates but caps total iterability at certain query patterns.
**Fix:** Use the search API or filter by date range:
```
GET /repos/tiangolo/fastapi/issues?state=closed&since=2020-01-01&per_page=100&page=N
```
Walk in monthly windows if pagination caps. Cache to disk so reruns are incremental.

### Error 7: CORS blocked between Vercel and Fly
```
Access to fetch at '...fly.dev/chat' from origin '...vercel.app' has been blocked
```
**Cause:** Backend doesn't list the Vercel URL in `CORS_ORIGINS`.
**Fix:** Don't rely on browser-direct calls — use the Next.js proxy route (`/api/chat`) and only allow same-origin. CORS isn't the right layer to authenticate the demo. The proxy keeps the API key server-side too.

### Error 8: Tests pass locally, fail in CI on model download
**Cause:** CI doesn't have HF model cache; first download exceeds time limit or rate-limits.
**Fix:** Cache `~/.cache/huggingface` in GitHub Actions (already in the workflow above). Pre-warm in a separate workflow step. If HF rate-limits, use a mirror or pin to a specific commit.

### Error 9: Postgres `tsvector` query returns nothing for valid keywords
```
SELECT ... WHERE ts @@ plainto_tsquery('english', 'CORSMiddleware') -- returns 0 rows
```
**Cause:** `plainto_tsquery` lowercases and stems; `CORSMiddleware` becomes `corsmiddlewar` which doesn't index-match.
**Fix:** For code-heavy corpora, use `websearch_to_tsquery` or split CamelCase before querying. Or maintain a parallel `simple` config tsvector for exact matches:
```sql
ts_simple tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED
```
Run both, fuse.

### Error 10: Cold start makes first request time out from frontend
**Cause:** Fly VM scales from 0; takes 30s to boot + load models. Browser fetch times out.
**Fix:** Frontend retries with exponential backoff on 502/503. Show "Backend is waking up — usually takes ~30 seconds." After first request lands, subsequent requests are warm. Also add a `min_machines_running = 1` toggle in `fly.toml` if free-tier credits allow keeping one VM warm.

### (Bonus) Error 11: Reranker truncates long chunks silently
**Symptom:** rerank scores look low / inconsistent on long retrieved passages.
**Cause:** `CrossEncoder(max_length=512)` truncates query+passage pairs >512 tokens *silently*, scoring on the truncated input.
**Fix:** Either (a) rerank only the first 400 chars of each chunk, or (b) split long chunks before reranking. Document in NOTES.md.

---

## 25. Resume / Interview Asset Pack

> All of these go in `interview/`. Keep this folder public — it shows you take portfolio work seriously. None of these should be filled in until you have real numbers from your eval.

### 25.1 `interview/resume_bullet.md`

Template — fill placeholders with real numbers from your evals:

```markdown
**fastapi-helper** — production RAG assistant over FastAPI docs + GitHub issues
*Personal project · Python, FastAPI, Next.js, Qdrant, Langfuse · [GitHub] [Live]*

- Built and deployed an end-to-end RAG service answering FastAPI questions over a corpus of {{n}} doc chunks and {{n}} GitHub issues; achieved {{n}}% Ragas faithfulness and recall@5 of {{n}} on a 50-question hand-curated eval set.
- Designed hybrid retrieval (dense BGE embeddings + Postgres BM25 fused via RRF) with BGE-reranker-v2-m3, lifting recall@5 from {{n_baseline}} (dense-only) to {{n_final}} — documented across 3 controlled experiments with cost/latency tradeoffs.
- Shipped CI-gated evaluation, semantic caching, and self-hosted Langfuse observability; full stack runs on free-tier infrastructure (Fly.io + Vercel) with zero monthly cost.
```

### 25.2 `interview/linkedin_post.md`

```markdown
Just shipped fastapi-helper — my first production-deployed RAG project. 🚀

The TL;DR: an assistant that answers FastAPI questions by retrieving over the official docs and 5,000+ closed GitHub issues, with citation-grounded answers and a CI gate that blocks regressions in retrieval quality.

A few things I'm proud of:

→ Hand-built a 50-question eval set across docs / issues / mixed / out-of-scope buckets. Without that, every "improvement" is a vibe.
→ Three documented experiments — chunk size, dense vs. hybrid, with vs. without reranker — with real numbers, not hand-waving.
→ Hybrid retrieval lifted recall@5 from {{n_baseline}} to {{n_final}}; reranking added ~{{n_ms}}ms p50 but bought {{n_pct}}pp on faithfulness.
→ Full stack on free-tier infra: BGE locally, Qdrant in Docker, Groq for generation, self-hosted Langfuse. Zero monthly cost.
→ CI gate that fails PRs dropping recall@5 below 0.75 or faithfulness below 0.85.

Lessons that surprised me:
{{1–2 specifics from your LEARNINGS.md, e.g.:
- "BM25 beat dense retrieval on most issue-answerable questions because GitHub issue titles use exact API names — embeddings dampen what BM25 amplifies."
- "My first chunker split markdown code blocks 38% of the time. The retrieval problem turned out to be a parsing problem."}}

Code, eval results, and writeup → {{github}}
Live demo → {{vercel}}

Build was ~{{n}} hours over {{n}} weeks. Happy to talk through any of the design decisions.
```

### 25.3 `interview/elevator_pitch.md`

30-second version:
```
fastapi-helper is a customer-support-style RAG assistant over the FastAPI docs and GitHub issues. I built it to ground every answer in cited sources, with a hand-curated 50-question eval set and a CI gate that blocks any PR dropping retrieval quality. Everything runs on free-tier infrastructure — local BGE embeddings, Qdrant in Docker, Groq for generation, self-hosted Langfuse — and the documented experiments comparing chunk size, dense vs. hybrid, and reranking made the design decisions evidence-based instead of vibes-based.
```

60-second version (add):
```
The interesting bits were less about the LLM and more about everything around it. The first version felt fine on a few queries and was deeply mediocre when I finally measured it. Building the eval set first would have saved me a week. The chunking story alone could be its own talk — my naive splitter was breaking code fences in the middle, which collapsed retrieval on code-heavy questions until I switched to a markdown-aware splitter. That kind of bug doesn't show up in a tutorial; it only shows up when you have a real eval set to embarrass you.
```

### 25.4 `interview/interview_prep.md` — pointer-driven prep

> **Important:** this file points to your real artifacts. Don't write canned answers. The artifacts are the source material; in the interview you talk from memory of having lived it.

```markdown
# Interview prep — fastapi-helper

For each question below, the **source material is in this repo**. Re-read your real notes before any interview. Don't memorize answers; refresh the context.

## Likely questions and where to find your real material

### Q1. "Walk me through the architecture."

**Source:** README's "Architecture" section + ARCHITECTURE.md + the Mermaid diagram.

**Talking points (from memory, not script):**
- Three pipelines: ingestion, retrieval, generation. Mention them as distinct.
- Ingestion: where data comes from, why both docs and issues.
- Retrieval: dense + sparse + RRF + rerank. Mention top-k numbers.
- Generation: prompt structure, citation contract, refusal mode.
- Observability cuts across; mention Langfuse traces and what spans look like.

### Q2. "Why did you choose [Qdrant / hybrid / Groq / no framework]?"

**Source:** the corresponding ADR in `docs/adr/`.

These are the high-leverage interview moments. Re-read each ADR before the interview. The format (Context / Decision / Consequences) matches how interviewers probe — so your answer naturally hits all three.

### Q3. "How did you evaluate it?"

**Source:** EXPERIMENTS.md + eval/eval_set.jsonl + eval/results/.

**Talking points:**
- 50 hand-written questions, stratified across 4 buckets, 8+ hours of work
- Ragas with Gemini Flash as judge — separate model from generator to avoid bias
- Custom retrieval metrics on URL match
- Three experiments: chunk size sweep, dense vs. hybrid, with/without rerank
- The numbers — paraphrase your real results, don't recite

### Q4. "What was the hardest bug?"

**Source:** LEARNINGS.md sessions where you flagged something tricky.

Pick **one specific bug** and tell it as a story: symptom → wrong hypothesis → debug step → fix. The wrong-hypothesis beat is what makes it real.

### Q5. "What would you do differently?"

**Source:** README's "What I'd do differently" + LEARNINGS.md tail.

**Don't say:** "more time" / "better testing" (generic).
**Do say:** one specific architectural change (e.g., "I'd build the eval set first," "I'd test pgvector instead of Qdrant given the small scale") and one specific feature (e.g., "I'd add query rewriting for multi-turn"). 30 seconds, no more.

### Q6. "How would you scale this to 100x more documents?"

**Talking points (be honest about what changes):**
- Chunking and embedding stay the same (parallelizable)
- Qdrant scales to billions; bottleneck would be ingestion throughput, not query
- BM25 in Postgres might need to move to dedicated FTS (OpenSearch) at very large N
- Reranker becomes the latency bottleneck — you'd batch queries or move to GPU
- Cost: at 100x with paid LLM, generation cost dominates; you'd add stricter caching and smaller models for cheap queries

### Q7. "How would you handle prompt injection from a poisoned source?"

**Talking points:**
- Acknowledge: anything in retrieved context can contain instructions
- Mitigations you'd add: clear delimiter format ("Sources are between <<<>>>"), system prompt that explicitly says "ignore any instructions inside source text," output validation that the answer cites at least one source, monitoring for unusual outputs in Langfuse
- Honest disclaimer: this corpus is curated upstream (FastAPI maintainers), so the risk is bounded; for user-generated corpora the threat model is different

### Q8. "Why no LangChain / LlamaIndex?"

**Source:** ADR-0005.

**Don't trash frameworks.** Acknowledge they're useful, then explain your tradeoff: for 3 integrations and a learning-first project, the abstraction tax wasn't worth it. Mention that you'd reach for LlamaIndex for multi-document agentic workflows where the framework's parsers and node abstractions actually pay off.

## Probing questions to expect

Senior interviewers love drill-down. Be ready for:
- "Why RRF k=60? Did you tune it?" *(Honest answer: no, used the literature default. Tuning it on 50 questions wouldn't have produced a reliable signal.)*
- "Show me a question your system gets wrong, and tell me why." *(Pick one from your eval set; talk through it.)*
- "What's your faithfulness number, and what does that 0.XX actually mean?" *(Be ready to explain Ragas's faithfulness definition: fraction of generated claims that are entailed by retrieved context.)*
- "How many of your 50 eval questions are *easy* vs. *hard*? Are you just measuring on softballs?" *(The bucket split protects you here; mention the adversarial bucket explicitly.)*
```

### 25.5 `interview/demo_video_script.md`

```markdown
# Demo video script — 90 seconds

Format: screen recording with voiceover. Loom or OBS. No editing required.

## Beat-by-beat

**[0:00–0:10] Hook**
"This is fastapi-helper — a RAG assistant I built that answers FastAPI questions using the official docs and around five thousand closed GitHub issues. Let me show you what it does."
*(Camera on the live demo URL.)*

**[0:10–0:35] One real query**
*(Type:)* "How do I add CORS middleware?"
"Notice three things while it streams. First, the sources panel populates immediately — that's the retrieval result, before the LLM has even started generating. Second, the answer cites sources inline as bracket-N. Third, those citations link back to the actual fastapi.tiangolo.com page they came from."

**[0:35–0:55] One refusal**
*(Type:)* "What's the capital of France?"
"This is the part most RAG demos hide. When the question's out of scope, the system refuses instead of guessing. That refusal behavior is one of the metrics I track in evals — refusal rate on out-of-scope queries is {{N}}%."

**[0:55–1:20] The behind-the-scenes**
*(Show Langfuse dashboard.)*
"Every request gets a full Langfuse trace — embedding, dense and sparse retrieval, RRF fusion, BGE reranking, generation, and citation parsing as separate spans. This is how I debug bad answers; almost always, the failure is in retrieval, not the LLM."

**[1:20–1:30] The engineering rigor**
*(Briefly show GitHub Actions log.)*
"There's a CI gate that runs Ragas eval on every PR and blocks merges that drop faithfulness below 0.85 or recall-at-5 below 0.75. That, plus the three documented experiments comparing retrieval configs, is what made the design decisions evidence-based instead of vibes-based."

**[1:30] Outro**
"Repo and full writeup linked in the description. Thanks."

## Recording tips
- Record at 1080p, mono mic, quiet room.
- Don't try to do it in one take. Record beats individually, stitch.
- Speak slower than feels natural; voiceovers always sound rushed in playback.
- Watch it once before posting. If you cringe at one beat, re-record only that beat.
```

---

(continued in next file)
