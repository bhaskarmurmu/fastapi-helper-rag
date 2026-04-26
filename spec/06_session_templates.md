# `fastapi-helper` — Spec Part 6: Session Templates & Build Process

Everything in this file is **a template you fill in as you build**, not pre-written content. The whole point: your real notes accumulate as you go, your real numbers fill the tables, your real decisions get recorded at the moment you make them.

---

## 19. `LEARNINGS.md` — session-end template

Create the file with this header at the start of the project. Then **add a new entry at the end of every work session.** A "session" is any block of work, ~1–4 hours. By project end you'll have ~30–50 entries. That accumulation is the artifact.

### Header (paste into LEARNINGS.md once)

```markdown
# Learnings

> Running notes from building fastapi-helper. First-person, dated, often messy.
> If something here is wrong, it's because I was wrong at the time of writing.

## Format
Each session: date, hours worked, what I did, what I learned, what I got stuck on, links to anything I read.
```

### Per-session template (copy this for each session)

```markdown
## YYYY-MM-DD (Nh)

**What I did:**
- ...

**What I learned / what surprised me:**
- ...

**What got me stuck / time-sink of the day:**
- ...

**Links / things I read:**
- ...

**Next session:**
- ...
```

### Prompts to answer at the end of each session

Don't blank-stare at the template. Use these prompts. **Answer at least 3 of them per session.** Be specific.

1. **What did I get working today?** (Concrete: "issue ingestion paginated through ~5K issues in ~6 minutes" not "worked on ingestion.")
2. **What did I get wrong, then fix?** (The pre-fix and post-fix versions. This is the most valuable thing to log; future-you will thank you in interviews.)
3. **What did I read or skim that helped?** (URLs only. Even 5 seconds of saving the link is worth it.)
4. **What confused me that I still haven't fully resolved?** (It's OK to leave this open. List unresolved confusion; the resolution shows up later.)
5. **What did I almost do that I'm glad I didn't?** (Often the better engineering decision.)
6. **What number did I see today that I want to remember?** (Eval scores, latency numbers, chunk counts, cost-per-query estimates. Numbers stick when written down.)
7. **What's blocking the next session?** (One sentence. Reduces friction tomorrow.)

### Suggested seed entries (the kinds of things to capture, *if you actually encounter them*)

These are examples of the *shape* of good entries — not pre-written content to paste in. Capture the equivalents from your real build.

- "Spent 90 min on what turned out to be a Postgres `tsvector` quirk: `plainto_tsquery` lowercases and stems, so 'CORSMiddleware' became 'corsmiddlewar' and never matched. Switched to `websearch_to_tsquery` and a parallel 'simple' tsvector for exact matches. Added test."
- "Tried to use the LangChain `RecursiveCharacterTextSplitter` with markdown_separators; it still split inside code fences ~30% of time on the FastAPI tutorial pages. Wrote a custom splitter that walks the document treating ```...``` as atomic. Tests in test_chunker.py."
- "Read [the Anthropic Contextual Retrieval blog]. Considered prepending an LLM-generated summary to each chunk before embedding. Decided against — extra ingestion cost (one LLM call per chunk × ~12K chunks) doesn't fit the zero-budget constraint. Worth revisiting in v2 if Groq stays free."
- "Eval results dropped a lot when I 'improved' the prompt. Reverted. Lesson: change one thing at a time, run eval before and after."

### Word count target

By project end, LEARNINGS.md should be ~2,000–3,000 words of real content. This isn't a goal to pad to; it's roughly what 30–50 honest session entries naturally produce.

---

## 20. `EXPERIMENTS.md` — protocol & table templates

This file documents the experiments you'll actually run. **Do not fill in numbers ahead of time; run the experiments and paste the real numbers in.**

### File structure

```markdown
# Experiments

This is the documented record of the controlled experiments I ran while tuning the retrieval pipeline. Hypotheses, configs, real numbers, conclusions. Each experiment changes one variable at a time.

## How I ran these

- All experiments use the same 50-question eval set (`eval/eval_set.jsonl`).
- LLM judge: Gemini 2.0 Flash (separate from the generator, which is Llama 3.3 via Groq), to avoid self-grading bias.
- Each experiment writes its full results to `eval/results/exp{N}_*.json`.
- Numbers are mean across the eval set unless otherwise stated.

## Glossary

- **recall@5**: of the expected source URLs for a question, what fraction appear in the retrieved top-5
- **MRR**: mean reciprocal rank of the first correct source URL
- **faithfulness**: Ragas metric — fraction of generated claims entailed by retrieved context
- **answer_relevancy**: Ragas metric — how well the answer addresses the question
- **p50 latency**: median end-to-end query latency, ms

---

## Experiment 1: chunk size sweep

**Hypothesis:** Larger chunks give the LLM more context per source (helping faithfulness) but degrade retrieval precision (chunks become topically diffuse).

**Setup:** re-ingest the same corpus 3 times into separate Qdrant collections. Hold everything else constant: BGE-small embeddings, hybrid retrieval, BGE reranker, Llama 3.3 generation.

| Config | chunk_size | overlap | total_chunks | ingestion_time | recall@5 | MRR | faithfulness | answer_relevancy | p50_latency |
|---|---|---|---|---|---|---|---|---|---|
| A | 256 | 25 | __ | __ | __ | __ | __ | __ | __ |
| B | 512 | 50 | __ | __ | __ | __ | __ | __ | __ |
| C | 1024 | 100 | __ | __ | __ | __ | __ | __ | __ |

**Conclusion:** _(fill in 2–4 sentences based on real numbers. Mention which config you adopted and why. Be honest about close calls.)_

**Caveats / what I'd do with more time:** _(e.g., test 768 token chunks, semantic chunking comparison, sweep overlap independently.)_

---

## Experiment 2: dense vs. sparse vs. hybrid retrieval

**Hypothesis:** Hybrid (dense + Postgres BM25 fused via RRF) outperforms either alone, especially on issue-answerable questions where exact API names (BackgroundTasks, Depends) matter.

**Setup:** corpus from winning chunk size in Exp 1. All three configs use the same BGE reranker downstream. Top-30 from each retriever, top-5 after rerank.

| Config | recall@5 | recall@5 (issue bucket) | recall@5 (docs bucket) | MRR | faithfulness | p50_latency |
|---|---|---|---|---|---|---|
| Dense only | __ | __ | __ | __ | __ | __ |
| Sparse only | __ | __ | __ | __ | __ | __ |
| Hybrid (RRF) | __ | __ | __ | __ | __ | __ |

**Per-bucket qualitative notes:** _(any patterns across buckets — e.g., "sparse beat dense on 7/15 issue-answerable questions because issue titles are keyword-dense; on docs-answerable questions, dense was at parity or better.")_

**Conclusion:** _(2–4 sentences. If hybrid won, by how much, and where it didn't help.)_

---

## Experiment 3: with vs. without reranker

**Hypothesis:** BGE reranker improves precision@5 measurably; latency cost (~100–500ms on CPU) is acceptable for a portfolio-scale deployment.

**Setup:** corpus from Exp 1, hybrid retrieval from Exp 2.
- Config A: hybrid → top-5 directly (no rerank)
- Config B: hybrid → top-30 → BGE rerank → top-5

| Config | recall@5 | precision@5 | faithfulness | answer_relevancy | p50_latency | p95_latency |
|---|---|---|---|---|---|---|
| No rerank | __ | __ | __ | __ | __ | __ |
| With rerank | __ | __ | __ | __ | __ | __ |

**Conclusion:** _(...)_

**Latency breakdown for config B (median over eval):**
- Embedding: __ ms
- Dense retrieval: __ ms
- Sparse retrieval: __ ms
- RRF: __ ms
- Reranking: __ ms
- LLM generation (full stream): __ ms

---

## (Optional) Experiment 4: BGE-small vs. BGE-base embeddings

If time allows. Compares MTEB scores (~64 vs. ~67 for these two on retrieval) against your own data. Often the gap on a domain-specific eval is smaller than on MTEB.

| Config | embedding_dim | model_size | embed_time | recall@5 | faithfulness | p50_latency |
|---|---|---|---|---|---|---|
| BGE-small-en-v1.5 | 384 | 130MB | __ | __ | __ | __ |
| BGE-base-en-v1.5 | 768 | 440MB | __ | __ | __ | __ |

---

## Combined picture: how the final config got there

A short paragraph after all experiments: chunk size from Exp 1 → adopted into Exp 2 baseline → hybrid winner from Exp 2 carried into Exp 3 → reranker decision in Exp 3 → final production config: ____.
```

---

## 21. ADR template & list of decisions to document

### Template (`docs/adr/template.md`)

```markdown
# ADR-NNNN: <Decision title>

- **Status:** accepted | superseded by ADR-XXXX | deprecated
- **Date:** YYYY-MM-DD

## Context

What's the situation? What forces are at play? What constraints exist?
2–4 sentences. Don't editorialize; describe the problem.

## Decision

What did we decide? Pick one option, name it explicitly.
Then 1–3 sentences of *why this option over the others considered*.

## Alternatives considered

- **Alternative A:** rejected because ___
- **Alternative B:** rejected because ___

## Consequences

What does this commit us to? What becomes harder? What becomes easier?
What can we revisit later?
2–4 sentences.

## Notes

Any links, benchmarks, or related ADRs.
```

### The 7 ADRs to write (write each at the moment you make the decision, not at the end)

For each: when to write it, what to put in Context, what the Decision is, the alternatives.

#### `0001-vector-db-choice.md`
- **When:** first time you instantiate a vector DB client (early in ingestion work).
- **Context:** You need a vector store for ~10K embeddings, with metadata filtering, on free-tier infra.
- **Decision:** Qdrant via Docker, locally and on Fly.io.
- **Alternatives to compare:** pgvector (rejected because separating vector concerns from relational concerns lets you swap each independently; Qdrant filtering DSL is cleaner); Chroma (rejected for production seriousness — it's prototype-grade); Pinecone (rejected: free tier is small and you don't want a hard cloud dependency).

#### `0002-embedding-model-choice.md`
- **When:** when you instantiate the embedder.
- **Context:** Need a free, CPU-friendly embedder with good retrieval performance on technical English.
- **Decision:** `BAAI/bge-small-en-v1.5`.
- **Alternatives:** `bge-base-en-v1.5` (deferred to Experiment 4; gap on portfolio scale didn't justify 3× the dim); `all-MiniLM-L6-v2` (faster but materially worse on MTEB retrieval); OpenAI `text-embedding-3-small` (rejected on the zero-cost constraint).

#### `0003-hybrid-vs-dense-retrieval.md`
- **When:** after Experiment 2 produces real numbers.
- **Context:** Question whether the operational complexity of running both retrievers + RRF is worth the recall lift.
- **Decision:** hybrid (dense + Postgres BM25 + RRF).
- **Numbers in the ADR:** quote the recall@5 lift from Exp 2.

#### `0004-reranker-yes-or-no.md`
- **When:** after Experiment 3.
- **Context:** Reranker adds latency (~hundreds of ms on CPU). Tradeoff against precision@5 gain.
- **Decision:** keep reranker; rerank top-30 to top-5.
- **Numbers:** quote precision@5 lift and latency cost from Exp 3.

#### `0005-no-orchestration-framework.md`
- **When:** when scaffolding the project (early).
- **Context:** Choice of LangChain vs. LlamaIndex vs. DIY.
- **Decision:** DIY, ~400 lines of pipeline code.
- **Alternatives:** LangChain (rejected for this project's scope: leaky abstractions outweigh the integration ecosystem when you have 3 integrations); LlamaIndex (rejected for similar reasons; would consider for v2 with multi-doc agentic flows).
- **Consequences:** more code to maintain, but every line is understood; easier to debug; resume signal of "I can build this without a framework" stronger than "I imported LangChain."

#### `0006-self-hosted-langfuse.md`
- **When:** when you set up observability.
- **Context:** Need traceability for debugging, evals burning trace volume.
- **Decision:** self-hosted via Docker.
- **Alternatives:** Langfuse Cloud free tier (acceptable; trace volume limit was the deciding factor); LangSmith (paid only); Phoenix (acceptable alternative).
- **Consequence:** one more service to operate; but full data control and no rate limits.

#### `0007-llm-provider-and-fallback.md`
- **When:** when you write `llm.py`.
- **Context:** Free-tier LLM serving, with provider abstraction for resilience.
- **Decision:** Groq Llama 3.3 70B as primary; Gemini 2.0 Flash as fallback. Provider toggle via `LLM_PROVIDER` env var.
- **Alternatives:** OpenAI/Anthropic/Mistral cloud APIs (cost); Ollama local (acceptable for dev, too slow for shared free-tier deployment).

#### (Optional, write only if it actually happens) `0008-when-things-changed.md`
If during the build you make a meaningfully different decision than originally planned (e.g., switch chunker strategy after eval, downgrade Langfuse from self-hosted to cloud free tier), write a new ADR and mark the old one "superseded by ADR-0008." This *trail of decisions* is the single most senior-feeling artifact in the repo.

### Where to write them

- Open `docs/adr/template.md` to start
- Copy to `0NNN-decision-name.md`
- Fill in at the time of decision (3–10 minutes; don't perfect them)
- Reference from README's "Tech stack" table and from EXPERIMENTS.md where relevant

---

## 22. `BLOG.md` — outline & writing guide

Write at the end. ~1500 words. The point is reflection, not promotion. The most-shared engineering blog posts are specific and humble. If you write a generic "I built RAG and here are the steps" post, it will sink without trace. If you write "here's what I learned from building this, including 3 things that didn't work," it might land on Hacker News.

### Outline (8 sections)

1. **Hook (~100 words).** A single specific moment. e.g., "The day I built my eval set was the day my project changed from 'feels good' to 'numbers say 0.62.' I want to talk about how the second half of that journey unfolded." Or pick a different opening — but make it a *concrete moment*, not a thesis.

2. **Why I built this (~150 words).** The real reason — first portfolio project, wanted to learn the unsexy parts. Don't pretend it was a startup idea.

3. **The naive first version (~200 words).** Walk through what you built before measurement. Recursive chunker, dense-only, no reranker. Honesty about what looked fine and wasn't.

4. **The eval-set inflection point (~250 words).** Building the 50-question hand-curated eval set. The bucket split (docs / issues / mixed / out-of-scope). What "felt fine" turned into when measured. This is the section most readers will quote — make it specific. Mention the time investment honestly (~8 hours).

5. **Three experiments that mattered (~350 words).** Walk through each. Don't recite numbers like a table; tell the story of what you expected vs. what happened. The interesting moment is when an experiment didn't go as expected. Pull at least one such moment forward.

6. **The unsexy 30% (~200 words).** Streaming, deployment, cold starts, CORS, the SSE proxy through Next.js, semantic caching tradeoff. The infrastructure-was-half-the-work story. Engineers love this.

7. **What I'd do differently (~150 words).** Specifics. "Build the eval set first." "Test pgvector against Qdrant given the small scale." Whatever your real reflections are.

8. **What I'm taking forward (~100 words).** What you understand now that you didn't before. Don't be grand; one or two specific things.

### Writing prompts (use to generate first draft, then rewrite)

- "Describe the bug that took you longest to find. What was the wrong hypothesis you held? What was the actual cause?"
- "Describe a moment when an experiment surprised you."
- "Describe a tradeoff you made where the 'best' option wasn't the one you picked. What was the constraint that forced the choice?"
- "If you could time-travel to the start of the project, what's the *one* thing you'd whisper to past-you?"

### Tone calibration

Avoid:
- Phrases like "I'm thrilled to share", "excited to announce", "leveraged"
- Bullet-list-heavy writing for the blog (it reads as outline-not-essay)
- Overstating impact ("revolutionized", "production-grade" — show, don't claim)

Aim for:
- Concrete numbers, named libraries, specific file names where they tell the story
- Acknowledgement of what didn't work
- One mild self-deprecating moment (not affected, just real)

### Where to publish

- Primary: `BLOG.md` in the repo
- Secondary (optional): mirror to dev.to or your own site after the project ships
- Don't cross-post until you've re-read it 24 hours later. Almost always you'll catch something off.

---

## 23. Demo video script

(Already in §25.5. The recording is a separate session — record it after the project is fully deployed and you've slept at least once on it. Don't record it the same day as deployment; you'll be too close to it.)

---

## 18. Commit cadence guide

Your real history will look like a real history because it *is* one. The point of this section is to give you a few habits that make your commits look like a working engineer's, not an LLM's.

### Cadence

- **Commit every meaningful unit of work**, roughly every 30–90 minutes of coding. Not after every keystroke; not once per session.
- **Write the commit before you stop**, not at end of session. End-of-session commits get sloppy and rolled-up.
- **Don't squash everything into "feat: ingestion"**. Squashing kills the story; the story is the value.

### Commit message style

- Conventional Commits prefix when natural: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`, `wip:`. Don't be religious about it.
- Lowercase first word, concise, imperative ("add" not "added"/"adds").
- Body when you need it (`-m` twice or use editor) for context: *why*, not *what*. The diff shows *what*.
- Use `wip:` on dev branches when you stop mid-thought; squash or rephrase before merging to main.

### Examples of real-engineer commits (the shape, not the literal content)

These illustrate the *texture*. Yours will be specific to your real work.

```
feat: chunker prepends section headers to each chunk
fix: tsvector query was lowercasing CamelCase API names
refactor: extract retrieval pipeline class from chat handler
test: pin system prompt with byte-equality test
docs: ADR-0003 hybrid vs dense, with exp2 numbers
chore: bump bge-small to v1.5, regenerate embeddings
wip: trying parent-document retrieval, not convinced yet
fix: rerank truncates >512 token pairs silently — chunk before rerank
docs(readme): add evaluation results table
ci: cache hf models between runs to keep eval gate under 5 min
```

### What *not* to do

- ❌ One commit named "initial commit" with the entire project in it
- ❌ "fix bug" with no body and no context — useless six months later
- ❌ Commits that mix unrelated changes ("fix chunker + add deploy config" — split into two)
- ❌ Force-pushing to main and erasing history (recruiters do look at the graph)
- ❌ Commits at 3am every day (the timestamps look weird; commit when you actually work)

### What you'll naturally end up with

After ~150 hours over 6–8 weeks, you'll have something like 60–120 commits. Don't aim for a number; aim for honest commits at honest cadence. The graph will look like reality because it is.

### Branching

- `main` is always green (CI passes; demo works).
- Feature branches for anything risky (chunker rewrite, deploy config). Merge with squash *only when the feature is small enough that the internal commits add no value*.
- Tag releases (`v0.1.0` for the first deploy, `v0.2.0` after experiments incorporated, etc.). Tags trigger the deploy workflow.

### One PR with the eval gate visible

Open at least one PR that *demonstrably exercises the CI eval gate* — either it passes after you tighten thresholds, or you make a change that drops a metric and watch the gate fail (then fix and re-run). This is a great screenshot for the README and a great talking point in interviews.

---

## 26. Claude Code handoff instructions

> This section is addressed directly to Claude Code (or any AI coding assistant being asked to scaffold the project). It is also the one section the human user (you) should re-read before starting each major build session.

### Build order (strict)

1. **Repo scaffolding** — directory tree, pyproject.toml, requirements.txt, .gitignore, .envrc.example, Makefile, docker-compose.yml. Make a first commit. CI not yet wired.
2. **Models, config, exceptions** — `src/fastapi_helper/{config,models,exceptions,logging_setup}.py`. Tests for config loading.
3. **Ingestion (smallest path first)** — `docs_loader.py` only, against a 5-file fixture. `chunker.py`. Tests. Commit. Then add `embedder.py`. Then `indexer.py`. Then `issues_loader.py`. Then `run.py`. At each step: it should run end-to-end.
4. **Retrieval (smallest path first)** — `dense.py`. Hand-test with one query in a notebook or script. Then `sparse.py`. Then `fusion.py` with unit tests. Then `reranker.py`. Then `pipeline.py`.
5. **Generation** — `prompts.py`, `llm.py`, `generator.py`. Mockable LLM interface for tests.
6. **API** — `api/main.py`, middleware, deps, three routes. Hit them with curl before writing the frontend.
7. **Cache** — `semantic_cache.py`. Wire into chat route. Test miss → hit transition.
8. **Observability** — Langfuse Docker compose, `tracing.py`, instrument the chat route. Verify traces show up.
9. **Frontend** — Next.js scaffolding, the proxy route, ChatWindow + components, SSE handling. Hit local backend.
10. **Tests** — fill in unit tests as you go, integration test at the end.
11. **CI** — `.github/workflows/ci.yml`. First run lint+test. Add eval gate after the eval set exists.
12. **Eval set construction** — this is when you spend ~8 hours. Don't rush.
13. **Experiments** — Exp 1 → ADR-0003 update + EXPERIMENTS.md entry. Then Exp 2. Then Exp 3.
14. **Deployment** — Fly.io backend, Vercel frontend. Iterate until `/health` is green from public URL.
15. **README, BLOG, demo video, interview assets** — last. Polish phase.

### Acceptance criteria at each milestone

After step 3: `make ingest MAX_ISSUES=20` runs to completion and you can `python -c "from qdrant_client import QdrantClient; print(QdrantClient(...).count('fastapi_helper'))"` and see a non-zero count.

After step 4: a Python REPL session can call `pipeline("How do I add CORS?")` and get back 5 sources with non-zero rerank scores from real fastapi.tiangolo.com URLs.

After step 6: `curl -X POST localhost:8000/chat -H 'X-API-Key: dev-secret-change-me' -H 'Content-Type: application/json' -d '{"question":"..."}'` returns SSE.

After step 9: full local round-trip from browser → Vercel route (locally `pnpm dev`) → backend → answer streams back, citations link out.

After step 11: a PR with a no-op change passes CI green.

After step 12: the eval set has 50 entries across the 4 buckets. Spot-check 5 entries by re-reading their `expected_source_urls`.

After step 13: three experiment JSONs in `eval/results/`, three filled-in tables in EXPERIMENTS.md, three sentences of conclusion each, and one updated ADR.

After step 14: live URL serves a streamed answer end-to-end.

After step 15: README has zero `{{placeholder}}` remaining.

### Acceptance criteria for "done"

- All checkboxes in §3 (Final Deliverables Checklist) ticked
- Live demo URL serves three different chat requests successfully (warm requests, after the first)
- Eval shows real numbers above thresholds
- CI is green on main
- BLOG.md is ≥1200 words and free of generic phrasing
- LEARNINGS.md has ≥30 dated session entries
- ≥6 ADRs filled in
- Demo video is recorded and linked
- Resume bullet has real numbers in place of placeholders

### Final reminder for the AI assistant

The artifacts that make this project credible are not the code, they're the **eval set, experiment results, ADRs written at decision time, session-by-session learnings, and honest README**. Help the human make the *code* fast, so they have time to actually do the *evaluation, decision-making, and writing* themselves. Those parts have to come from them.

If asked to "write LEARNINGS.md", refuse politely and instead help the human reflect on their actual session — ask the prompts in §19, listen, then format their real answers. The same applies to BLOG.md, the per-ADR Context/Decision/Consequences (the human articulates the decision rationale; the AI helps phrase it), and the experiments (the AI runs the eval, but the human writes the conclusion paragraph because only they have the context for what surprised them).

Code is fungible; the artifacts of judgment are not.
