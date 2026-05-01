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



## 2026-04-26 — day 1 / phase 1: plumbing

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



## 2026-04-27 — major detour: switching environments

Tonight was supposed to be Phase 2 (ingestion). Instead it became 4-5 hours
of infrastructure work. Worth writing this down because it's the kind of
problem that doesn't show up in tutorials.

The chain of dead ends:
1. Tried to install Docker Desktop locally. No admin rights on my laptop.
2. Considered IT ticket, but didn't want to involve them for a personal
   project.
3. Tried GitHub Codespaces — Docker works there out of the box. Hit a
   different wall: my Claude Code is bound to my employer Anthropic
   account, and Codespaces runs under my personal GitHub. Couldn't
   authenticate Claude Code in the Codespace.
4. Also discovered that Codespaces auto-injects GITHUB_TOKEN into the
   shell, which broke a Phase 1 test that asserted on default values —
   and worse, leaked tokens into pytest's error output. Had to rotate
   tokens 3-4 times before I figured out the test was the source.
5. Tried `wsl --install --no-distribution` on a hunch. Surprisingly,
   most of it succeeded despite no admin — only the very last step
   needed elevation, but a reboot finalized everything that had already
   installed.
6. After reboot, WSL was functional. Installed Ubuntu via
   `wsl --install -d Ubuntu`. Installed Docker inside Ubuntu via the
   official convenience script — no admin needed once you're inside WSL.
7. Now I have a full Linux + Docker dev environment on a locked-down
   Windows laptop with no admin and no money spent.

Lessons:
- Cloud dev environments aren't magic — they have their own auth boundaries
  that can bite you.
- Tests that read host environment variables are fragile. Need proper
  isolation with monkeypatch or env-clearing fixtures.
- Pytest's AssertionError message includes the actual value of variables —
  so if you assert on a secret, the secret ends up in test output. Need
  custom messages that don't echo the value.
- Windows + Python + Docker is a known pain stack. WSL is the right
  answer when you can get to it.
- The project moves to /home/bhmurmu/dev/fastapi-helper-rag (Linux side)
  going forward. The Windows-side copy in OneDrive is no longer used.

Tomorrow: actually start Phase 2 (ingestion).





- docs_loader gotcha: FastAPI's MkDocs uses `{ #anchor-id }` syntax in H1 
  headings. Naive title extraction picks up "About { #about }" instead of 
  just "About". Needed a second regex pass to strip the anchor fragment.






- chunker emits oversized chunks for code blocks rather than splitting them.
  5 oversized blocks in FastAPI corpus, max 1084 tokens. Right call — splitting
  code mid-function would break retrieval. Tradeoff: uneven chunk distribution
  (max 3861 chars vs avg 1658).
- noise in corpus: data/raw/fastapi/docs/en/_llm-test.md is a meta-test file
  for FastAPI's translation system. Decided to leave it for now and revisit
  during eval if it surfaces incorrectly.
- 153 docs → 988 chunks, avg 1658 chars. metadata (source_url, title,
  section_path) preserved on every chunk. spot-checked sample chunks: real
  readable docs text, not fragments.



  - embedder uses sentence-transformers with BGE-small-en-v1.5 (384-dim, L2-normalized).
  verified all output norms ≈ 1.0 → can use dot product instead of cosine in Qdrant.
- batch size [whatever it is] from Settings. CPU-only WSL, so [reasoning about why
  that batch size — fits in memory / good throughput].
- 13 unit tests mock SentenceTransformer to avoid model download in tests. Real
  embedding only verified in integration. clean separation.






  - step (e) verified directly via psql, not just via the indexer's smoke test:
  - chunks table: 80 rows, unique chunk_ids
  - tsvector populated (length 682 for sample row)
  - BM25 query "middleware request" → 5 hits, all from /middleware/. correct.
- chose Distance.DOT in qdrant since BGE embeddings are L2-normalized
  (verified earlier: norms = 1.0 exactly). dot product is cheaper than
  cosine and equivalent for unit vectors.
- chunk_id = uuid from sha256(source_url:chunk_index). deterministic
  → upserts are idempotent. ON CONFLICT DO UPDATE in postgres, native
  upsert in qdrant.
- relaxed smoke test from "exact UUID match" to "score >= 0.9999"
  because identical text in different files (license headers etc.)
  produces identical embeddings and any of them is a legitimate hit.
  ADR-worthy. (will write up in Phase 2 wrap.)














  ## 2026-04-27 — day 3: phase 2, ingestion pipeline end-to-end

Long day. Started with Docker not working (admin restrictions), pivoted
through Codespaces (couldn't auth Claude Code there), then discovered WSL
could install on this locked-down laptop and got the original spec stack
running. Most of "today" was actually that environment fight, but once it
worked, Phase 2 went smoothly across ~7-8 hours.

Built (in order):
- docker-compose with qdrant + postgres + langfuse (self-hosted)
- docs_loader: clones FastAPI repo, parses MkDocs files, strips YAML
  frontmatter and {#anchor} fragments from H1s. 153 docs.
- chunker: token-aware splitter that preserves code blocks intact even
  when they exceed chunk_size. 988 chunks from docs. Avg 1658 chars,
  max 3861 (one of those oversized code blocks).
- embedder: sentence-transformers wrapping BGE-small-en-v1.5. 384 dim,
  L2-normalized (verified empirically: norms = 1.0 exactly), float32.
- indexer: dual-write to qdrant (DOT distance) and postgres (text +
  GENERATED tsvector + GIN index). Shared chunk_id from sha256 hash —
  deterministic, idempotent.
- issues_loader: GitHub REST + httpx with redirect following. Filters
  PRs (which REST treats as issues). Includes comments. 80 issues
  across 297 chunks.
- run.py: full orchestrator. Total: 1285 chunks indexed.

Things I learned / had to think about:

1. Code-block preservation in chunking is a real tradeoff. Splitting a
   function in the middle breaks retrieval — fragments can't reason
   about code. Chose oversized chunks over fragmented code. 5 chunks
   exceeded chunk_size (max 1084 tokens); accepted as deliberate choice.

2. BGE embeddings are L2-normalized natively, which means dot product
   ≡ cosine for them. Configured Qdrant with Distance.DOT instead of
   COSINE — mathematically equivalent, slightly cheaper. Verified
   empirically with norms() before committing.

3. Smoke-test relaxation: original assertion was "round-trip retrieval
   returns the exact UUID I indexed." Failed for chunks with identical
   text in different files (license headers, etc.). Relaxed to
   "score >= 0.9999" which is the right behavior for production too.

4. tiangolo/fastapi → fastapi/fastapi rename: GitHub redirects (301)
   transparently via httpx, but stored source_urls still use the old
   org. Decided to leave it — GitHub's redirect is stable, and a
   migration to canonical URLs would add complexity for marginal gain.

5. `since` filter on GitHub API uses updated_at, not created_at. Old
   issues that get reopened or commented on appear in "recent" pulls.
   Turned out useful — those are exactly the relevant issues for
   support RAG.

Verifications:
- 139 unit tests passing
- 1285 chunks confirmed in both qdrant and postgres (exact match)
- BM25 query "background tasks" → top hit /tutorial/background-tasks/ ✓
- Dense query "how do I add middleware" → top hit /tutorial/middleware/ ✓
- Re-run produces same counts (idempotency)

Tomorrow: phase 3, hybrid retrieval. Plan to wire dense + BM25
together via RRF, add the BGE reranker on top of the fused list,
verify end-to-end query latency. Should take 4-5 hours fresh.





- step (c) RRF fusion verified empirically. for query "dependency
  injection with yield", dense missed the relevant doc entirely (not
  in top-5) but sparse caught it (rank 2). fusion lifted it to position
  3 of the fused result. that's the case for hybrid in one observation.
- RRF score for k=60 + dense_rank=1 + sparse_rank=2 = 1/61 + 1/62
  = 0.03252. tiny absolute number; relative ordering is what matters.
  worth flagging for future me if fusion ever "looks broken" in logs.
- test fixture bug caught: chunks with single-letter URLs collide on
  identity key (url, chunk_index) when test data is too minimal. fix:
  use distinct realistic-looking fixtures. lesson: tests with single-
  letter inputs hide identity bugs that real data would expose.



  ## 2026-04-28 — day 4: phase 3, hybrid retrieval pipeline

Five steps today. Built the heart of RAG: query in, ranked relevant
chunks out. Dense + sparse + RRF fusion + cross-encoder reranker, all
chained.

Steps:
- (a) DenseRetriever wraps qdrant query_points. DI for embedder so
  the model is shared across queries (130MB, 3-5s warmup).
  RetrievedChunk pydantic model is the shared currency throughout
  the pipeline.
- (b) SparseRetriever wraps postgres tsvector with ts_rank.
  plainto_tsquery with AND semantics — flagged as a future fix if
  Phase 11 evals show low BM25 recall.
- (c) RRF fusion. Pure function, k=60. The math is simple but easy
  to get wrong — caught a test fixture bug where chunks with
  single-letter URLs collided on (url, idx) identity. Distinct
  realistic fixtures matter.
- (d) Cross-encoder reranker. Original spec called for BGE-reranker-v2-m3
  but that's 15-22 seconds on CPU per query. Unworkable. Swapped to
  bge-reranker-base — 3s warm. Quality differences are at ranks 3-5
  within closely-scored candidates; major semantic judgments unchanged.
  ADR-0004 documents the v2-m3 vs base tradeoff and the GPU upgrade
  path for production.
- (e) Retriever orchestrator with configurable components for ablation.
  enable_sparse and enable_rerank as parameters so Phase 11 evaluation
  can compare configurations.

Key empirical findings:
- Hybrid retrieval real value, measured: for query "dependency injection
  with yield", dense missed the relevant doc page entirely (not in
  top-5) but BM25 caught it via keyword match on "yield". Fusion lifted
  it to rank 3.
- Cross-encoder vs RRF make different mistakes: for "background tasks
  + DI + DB session", RRF picked the canonical tutorial first; reranker
  promoted a release-notes chunk that specifically covered the pattern
  asked about. RRF aggregates votes across retrievers; reranker reads
  query+chunk jointly.
- Score distribution itself is signal: a query with all reranker scores
  in the 0.4-0.5 range tells you the corpus has thin coverage. A query
  with scores in 0.85+ tells you the corpus has direct answers. Future
  ADR — confidence-aware response generation.

Latency breakdown (CPU, warm):
- Dense:    70-148ms
- Sparse:   1-20ms
- Fusion:   <1ms
- Rerank:   3.0-4.7s    ← dominates. 95% of total latency.
- Total:    3.2-7.0s
The reranker is the cost. Production with GPU would drop total to <1s.

Architecture lessons:
- Bi-encoder vs cross-encoder cost asymmetry. Embedder runs once per
  query (amortized). Reranker runs once per (query, candidate) pair.
  N candidates means N forward passes, no parallelism on CPU. This
  is why GPU matters more for rerankers than embedders.
- Empty-list-as-natural-signal threads cleanly through every layer.
  Each retriever returns list[RetrievedChunk]. Empty composes correctly:
  fusion of empty + empty → empty, reranker on empty → empty, retrieve()
  on empty → empty, API layer maps empty to "no information found".
  No exceptions, no sentinels, no special cases.
- DI pays off. Every retriever takes its dependencies via constructor
  rather than instantiating internally. Mocking is trivial; tests run
  fast; the contract is explicit.

Tests: 259 unit tests passing across the full retrieval package.

Tomorrow: Phase 4, generation. LLM call wrapping the retrieved chunks
into a prompt, citation handling, response streaming. With retrieval
solid, generation is mostly prompt engineering.




- step (a) prompt design: spent meaningful time on Rule 1's grounding 
  scope. original draft said "use ONLY the provided context" which 
  contradicted the permitted use of general programming knowledge for 
  interpretation. fix: precisely scope to "FastAPI-specific claims must 
  come from sources" while explicitly permitting general programming 
  concepts (Python, HTTP, async/await) for interpretation. without this 
  the LLM would either refuse questions where async knowledge is needed 
  to explain FastAPI docs, or hallucinate FastAPI specifics.
- Rule 6 (partial answers) is what makes the system useful for compound 
  questions. without it, "deploy FastAPI to AWS Lambda with JWT auth" 
  forces a full refusal because Lambda+JWT specifics may not be in the 
  corpus. with it, deployment portion gets answered (citations) and 
  the gap is explicitly flagged.
- REFUSAL_SIGNAL is verbatim in the prompt so Phase 11 evaluation can 
  string-match it. partial-answer detection uses looser substring match 
  ("cannot answer from the provided sources").
- "every factual claim needs a citation" is followed imperfectly by 
  LLMs (~80-90% compliance is realistic). the citation extractor in 
  step (c) handles this by validating cited indices exist, not by 
  enforcing per-claim citation density.



  - step (b) verified the prompt design from step (a) actually works in
  practice. first real LLM call: model cited [2] for code, [1] for 
  descriptive claim, copied code verbatim from source, no invented 
  imports. exactly the behavior Rules 2/3/5 are designed to elicit. 
  this is the moment the prompt review paid off — building everything 
  else on top of a verified-working prompt.
- TTFT (time to first token) ~278ms streaming. total streaming response 
  ~560ms. non-streaming ~711ms. acceptable for an interactive system.
  user-perceived: full pipeline projects to ~3.5s first-token (dominated 
  by reranker on CPU; reranker drops dramatically on GPU).
- groq quirk: streaming and non-streaming token counts differ by 1 on 
  identical prompts (627 vs 628). expected behavior, not a bug. 
  worth flagging for phase 11 eval so cost-per-query metrics aren't 
  misread as inconsistent.