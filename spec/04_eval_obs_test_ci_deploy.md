# `fastapi-helper` — Spec Part 4: Evaluation, Observability, Testing, CI/CD, Deployment

## 12. Evaluation Plan

This section is the heart of the project. **The eval set you build is the single highest-leverage artifact in the entire repo.** Don't outsource it. Don't skip it. Don't rush it.

### 12.1 Building the eval set (your protocol)

**Target:** 50 hand-curated Q&A pairs in `eval/eval_set.jsonl`.

**Composition (deliberate split):**

| Bucket | Count | Source of truth |
|---|---|---|
| Docs-answerable | 20 | Pull from real FastAPI docs sections; write a question whose answer is *unambiguously* in one specific paragraph |
| Issue-answerable | 15 | Pick 15 closed issues with clear resolutions; write a paraphrased question and capture the issue URL as the expected source |
| Mixed (docs + issues) | 5 | Questions that genuinely benefit from both (e.g., "How do I do X?" where docs explain the API and an issue confirms a gotcha) |
| Adversarial / out-of-scope | 10 | Questions the system *should refuse* — Django questions, totally unrelated topics, or FastAPI questions whose answer isn't in the corpus |

> **Why include adversarial:** measures refusal behavior. A system that confidently answers off-corpus questions has a hallucination problem regardless of how well it does on in-corpus questions.

### 12.2 Eval set format

`eval/eval_set.jsonl` — one JSON per line:

```json
{
  "id": "q001",
  "question": "How do I add CORS middleware in FastAPI?",
  "ground_truth": "Use CORSMiddleware from fastapi.middleware.cors, then app.add_middleware(CORSMiddleware, allow_origins=[...], allow_credentials=True, allow_methods=['*'], allow_headers=['*']).",
  "expected_source_urls": ["https://fastapi.tiangolo.com/tutorial/cors/"],
  "bucket": "docs",
  "difficulty": "easy"
}
```

For **adversarial / out-of-scope**, set `expected_source_urls: []` and `ground_truth: "REFUSE"` — the eval scoring treats `REFUSE` as a special target meaning the system should produce the configured refusal phrase.

**How to write good ones (template you can paste into `interview/interview_prep.md` later):**

1. Pick a real concept (CORS, dependency injection, background tasks, OAuth2 flow, etc.)
2. Find the canonical doc page. Read it.
3. Write the question as a real user would phrase it — not as the docs phrase it. The whole point is testing retrieval over paraphrase.
4. Write the ground truth in 1–2 sentences, specific enough to grade against.
5. Save the source URL.

**Time budget:** ~10 minutes per question × 50 questions = ~8 hours. Spread over a few sessions. This is the most valuable 8 hours in the project.

### 12.3 Smaller seed set for CI

`eval/eval_set_seed.jsonl` — 10 questions, a stratified sample of `eval_set.jsonl` (4 docs / 3 issues / 1 mixed / 2 adversarial). The CI gate runs against this seed set on every PR; the full 50 runs nightly or manually.

> **Why a smaller CI set:** CI must be fast and cheap. 50-question Ragas eval against Gemini takes ~5 minutes and burns judge tokens. 10-question takes ~1 minute and is enough to catch regressions in CI.

### 12.4 Metrics

**Retrieval metrics** (computed from `expected_source_urls` ↔ retrieved chunks' source URLs):
- `recall@5` — primary metric; threshold for CI gate: ≥0.75
- `recall@10` — secondary
- `mrr` — sanity check
- `precision@5` — secondary

URL match logic: a retrieved URL counts as a hit if it equals an expected URL OR shares the same path prefix (handles cases where multiple chunks come from the same doc page).

**Generation metrics** (Ragas, with Gemini Flash as judge):
- `faithfulness` — primary; threshold for CI gate: ≥0.85
- `answer_relevancy` — secondary
- `context_precision` — secondary
- `context_recall` — secondary

**Refusal metrics** (custom):
- `refusal_rate_oos` — fraction of out-of-scope queries where the system produced the configured refusal phrase. Target: ≥0.80.
- `false_refusal_rate_in_scope` — fraction of in-scope queries where the system *incorrectly* refused. Target: ≤0.05.

### 12.5 Ragas code skeleton

```python
# eval/ragas_runner.py
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from datasets import Dataset

def run_ragas(rows: list[dict], gemini_api_key: str) -> dict:
    judge = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=gemini_api_key,
        temperature=0.0,
    ))
    ds = Dataset.from_list(rows)  # {question, answer, contexts, ground_truth}
    result = evaluate(
        ds,
        metrics=[Faithfulness(llm=judge), AnswerRelevancy(llm=judge),
                 ContextPrecision(llm=judge), ContextRecall(llm=judge)],
    )
    return result.to_pandas().mean(numeric_only=True).to_dict()
```

> **Note on rate limits:** Gemini Flash free tier is ~15 RPM. 50 questions × 4 metrics = 200 judge calls = ~14 minutes minimum even if Ragas batches. Add a `time.sleep` retry wrapper.

### 12.6 The three required experiments

> Each experiment is a single config sweep with everything else held constant. Run one, write up the result in `EXPERIMENTS.md`, then run the next.

#### Experiment 1: Chunk size sweep

**Hypothesis:** Larger chunks improve faithfulness (more context per source) but hurt retrieval precision (less granular).

**Configurations** (only chunk size + overlap vary; everything else = baseline):
- A: `chunk_size=256, overlap=25`
- B: `chunk_size=512, overlap=50` *(default)*
- C: `chunk_size=1024, overlap=100`

For each: re-ingest from scratch into a separate Qdrant collection (`fastapi_helper_chunk_256`, etc.), run the full 50-question eval, record metrics.

**Record:**
- Total chunks indexed
- Ingestion time
- recall@5, MRR
- Faithfulness, answer_relevancy
- p50 latency

**Pick:** the config with best composite score (you'll define the weighting; document it).

#### Experiment 2: Dense vs. hybrid retrieval

**Hypothesis:** Hybrid retrieval improves recall on questions with FastAPI-specific terms (`Depends`, `BackgroundTasks`, etc.) where BM25 catches the keyword that dense embedding might miss.

**Configurations:**
- A: dense only (Qdrant top-30 → rerank top-5)
- B: sparse only (BM25 top-30 → rerank top-5)
- C: hybrid (dense + sparse + RRF → rerank top-5) *(default)*

For each: same corpus (chunked at the winning size from Exp 1), same reranker, same generator. Run full 50-question eval.

**Record:** the same metrics as Exp 1, plus a manual qualitative breakdown by question bucket — does sparse beat dense more on issue-answerable questions? Often yes (issue titles are keyword-heavy).

#### Experiment 3: With vs. without reranker

**Hypothesis:** Reranker meaningfully improves precision@5; latency cost is acceptable.

**Configurations:**
- A: hybrid retrieval, top-5 directly (no rerank)
- B: hybrid retrieval, top-30 → rerank → top-5 *(default)*

**Record:** recall@5, precision@5 (key metric here), faithfulness, latency p50/p95.

> **Bonus experiment** if time permits: BGE-small vs. BGE-base. The point isn't to "win" but to **document the cost-quality curve**, which is the senior-engineer move.

### 12.7 Where the experiment results go

Three places:

1. `eval/results/exp{N}_*.json` — raw numbers, machine-readable
2. `EXPERIMENTS.md` — human-readable writeup with tables, hypothesis, conclusion
3. `README.md` — top-line numbers in the "Evaluation Results" section, linking to EXPERIMENTS.md

---

## 13. Observability & Logging

### 13.1 Langfuse setup (self-hosted)

Add to `docker-compose.yml`:

```yaml
services:
  langfuse-db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: langfuse
      POSTGRES_PASSWORD: langfuse
      POSTGRES_DB: langfuse
    volumes:
      - langfuse_db:/var/lib/postgresql/data

  langfuse:
    image: langfuse/langfuse:3
    depends_on: [langfuse-db]
    environment:
      DATABASE_URL: postgresql://langfuse:langfuse@langfuse-db:5432/langfuse
      NEXTAUTH_SECRET: change-me-32-chars-or-more-for-real
      SALT: change-me-also
      NEXTAUTH_URL: http://localhost:3000
      TELEMETRY_ENABLED: "false"
    ports:
      - "3001:3000"
volumes:
  langfuse_db:
```

> Note: the Langfuse server uses port 3000 internally; we map to 3001 to avoid clashing with Next.js dev server.

**First-run setup:**
1. `docker compose up -d langfuse`
2. Open `http://localhost:3001`, create an account (local-only)
3. Create a project, copy the public + secret keys into `.envrc`
4. Restart the backend to pick up env vars

### 13.2 What to log per request

Every `/chat` request creates one Langfuse trace with these spans:

```
trace: chat
├── span: embed_query (input: question; output: dim, time)
├── span: cache_check (output: hit/miss, similarity score)
├── span: retrieve (parallel)
│   ├── span: dense (input: question; output: top-30 with scores)
│   └── span: sparse (input: question; output: top-30 with scores)
├── span: rrf_fusion
├── span: rerank (input: top-30; output: top-5 with rerank scores)
├── generation: llm_call (input: prompt; output: answer; metadata: model, tokens, cost-estimate)
└── span: parse_citations
```

Trace metadata (top-level): `request_id`, `cache_hit`, `latency_ms_total`, `n_cited_sources`.

### 13.3 Dashboard expectations

After a few hundred queries you should see in Langfuse:
- Median end-to-end latency
- Latency breakdown by span (which step is slow)
- Cache hit rate over time
- Cost per request (Groq is free but log token counts so the dashboard works)
- Failed requests grouped by error type

Take a screenshot when you have ~100 traces. Put it in `docs/images/langfuse_dashboard.png`. Reference it in README and BLOG.

### 13.4 Application logs

`structlog`-style structured JSON logs to stdout (Fly.io captures stdout into its log shipper). Logger setup in `src/fastapi_helper/logging_setup.py`:

```python
import logging, json, sys, time
class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **(getattr(record, "extra", {}) or {}),
        })

def setup_logging(level: str = "INFO"):
    h = logging.StreamHandler(sys.stdout); h.setFormatter(JsonFormatter())
    root = logging.getLogger(); root.handlers = [h]; root.setLevel(level)
```

---

## 14. Testing Strategy

### 14.1 Unit tests (in `tests/unit/`)

| File | What it tests |
|---|---|
| `test_chunker.py` | (a) no chunk splits a code fence; (b) overlap is honored; (c) headers are prepended; (d) chunk size never exceeds `chunk_size` by more than 1.5× (the soft cap for atomic code blocks) |
| `test_fusion.py` | (a) RRF on two identical lists returns the same order; (b) RRF combines two disjoint lists into a single deduped list; (c) higher-ranked items in either input list rank higher in output |
| `test_generator.py` | (a) `parse_citations` extracts `[N]` markers; (b) prompt builder handles 0 sources; (c) prompt is under context budget |
| `test_prompts.py` | (a) system prompt is unchanged byte-for-byte (locks the prompt down — change requires explicit test update, which is the whole point) |
| `test_semantic_cache.py` | (a) cache miss on empty cache; (b) hit on identical query; (c) miss on dissimilar query; (d) eviction when over `max_keys` |

### 14.2 Integration test (in `tests/integration/`)

`test_chat_endpoint.py` — uses `testcontainers-python` to spin up Qdrant + Postgres in Docker, ingests a tiny fixture corpus (5 docs, defined in `tests/fixtures/`), and runs end-to-end:

```python
@pytest.mark.integration
async def test_chat_endpoint_returns_grounded_answer():
    # 1. Start qdrant + postgres containers
    # 2. Ingest fixture (3 docs about FastAPI features)
    # 3. POST /chat with a question matching one fixture
    # 4. Assert: 200, sources non-empty, answer cites at least one [N], latency < 30s
```

> **Important:** the integration test should NOT call Groq/Gemini in CI. Mock the LLM at the `LLMProvider` interface — return a fixed string with `[1]` citation. The test verifies the *plumbing*, not the LLM.

### 14.3 Running tests

```bash
make test          # pytest -m 'not integration'  (fast, no Docker)
make test-int      # pytest -m integration       (slower, requires Docker)
make test-all      # both
```

`pyproject.toml` configuration:

```toml
[tool.pytest.ini_options]
markers = [
    "integration: marks tests requiring Docker (deselect with '-m \"not integration\"')",
]
asyncio_mode = "auto"
```

---

## 15. CI/CD Pipeline

### 15.1 `.github/workflows/ci.yml` (complete)

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - name: Install uv
        run: pip install uv
      - name: Install deps
        run: uv pip install --system -r requirements-dev.txt
      - name: Ruff lint
        run: ruff check src/ tests/ eval/
      - name: Ruff format check
        run: ruff format --check src/ tests/ eval/
      - name: Mypy
        run: mypy src/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - name: Cache HF models
        uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface
          key: hf-${{ runner.os }}-bge-small-v1
      - name: Install deps
        run: pip install uv && uv pip install --system -r requirements.txt -r requirements-dev.txt
      - name: Unit tests
        run: pytest -m "not integration" --maxfail=1 -q

  eval-gate:
    runs-on: ubuntu-latest
    needs: [lint, test]
    services:
      qdrant:
        image: qdrant/qdrant:v1.11.0
        ports: ["6333:6333"]
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_USER: postgres
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: fastapi_helper
        ports: ["5432:5432"]
        options: >-
          --health-cmd="pg_isready -U postgres" --health-interval=5s
          --health-timeout=5s --health-retries=5
    env:
      GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
      GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
      GITHUB_TOKEN: ${{ secrets.GH_PAT_RO }}
      QDRANT_URL: http://localhost:6333
      POSTGRES_URL: postgresql://postgres:postgres@localhost:5432/fastapi_helper
      LLM_PROVIDER: groq
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - name: Cache HF models
        uses: actions/cache@v4
        with:
          path: ~/.cache/huggingface
          key: hf-${{ runner.os }}-bge-small-and-reranker-v1
      - name: Cache fixture ingestion
        uses: actions/cache@v4
        with:
          path: data/raw/fastapi
          key: fastapi-repo-${{ github.sha }}
          restore-keys: fastapi-repo-
      - name: Install deps
        run: pip install uv && uv pip install --system -r requirements.txt -r requirements-dev.txt
      - name: Ingest small fixture
        run: MAX_ISSUES=50 bash scripts/ingest_local.sh
      - name: Run seed eval
        run: python -m eval.run_eval --eval-set eval/eval_set_seed.jsonl --output eval/results/ci.json
      - name: Eval gate
        run: |
          python -c "
          import json, sys
          r = json.load(open('eval/results/ci.json'))
          fails = []
          if r['retrieval']['recall@5'] < 0.75: fails.append(f'recall@5={r[\"retrieval\"][\"recall@5\"]:.2f} < 0.75')
          if r['generation']['faithfulness'] < 0.85: fails.append(f'faithfulness={r[\"generation\"][\"faithfulness\"]:.2f} < 0.85')
          if fails: print('Eval gate FAILED:', fails); sys.exit(1)
          print('Eval gate PASSED')
          "
      - uses: actions/upload-artifact@v4
        with: { name: eval-results, path: eval/results/ci.json }
```

### 15.2 `.github/workflows/deploy.yml`

```yaml
name: Deploy

on:
  push:
    tags: ["v*"]

jobs:
  deploy-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: superfly/flyctl-actions/setup-flyctl@master
      - run: flyctl deploy --remote-only
        env:
          FLY_API_TOKEN: ${{ secrets.FLY_API_TOKEN }}

  # Vercel auto-deploys on tag/push via its GitHub integration; no workflow needed
  # for the frontend. Document that in deployment docs.
```

### 15.3 Pre-commit

`.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.7.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: [--maxkb=500]
```

---

## 16. Deployment Steps

> **Order:** Langfuse first (so you have observability ready), then Postgres + Qdrant, then backend, then frontend. Do this end-to-end *once* manually before automating, even though it's tempting to script it.

### 16.1 Prerequisites (one-time)

```bash
# 1. Install Fly CLI
curl -L https://fly.io/install.sh | sh
fly auth signup  # or: fly auth login

# 2. Install Vercel CLI
npm i -g vercel
vercel login

# 3. Sign up for Groq (free): https://console.groq.com — generate API key
# 4. Sign up for Google AI Studio (free): https://aistudio.google.com — generate Gemini API key
# 5. Generate a GitHub PAT (read-only public_repo): https://github.com/settings/tokens
```

### 16.2 Deploy Postgres + Qdrant on Fly.io

```bash
# Postgres: use Fly Postgres (free tier exists for small clusters)
fly postgres create --name fastapi-helper-db --region sjc \
  --vm-size shared-cpu-1x --volume-size 1
# Capture the DATABASE_URL printed at end

# Qdrant: deploy as a fly app from the official image
fly launch --no-deploy --name fastapi-helper-qdrant --region sjc \
  --image qdrant/qdrant:v1.11.0
fly volumes create qdrant_data --size 1 --region sjc -a fastapi-helper-qdrant
# Edit fly.toml for qdrant: add [mounts] source="qdrant_data" destination="/qdrant/storage"
fly deploy -a fastapi-helper-qdrant
```

> **Free-tier note:** Fly's free allowance has changed over time. As of build, you get a small monthly credit that covers 1–2 shared-cpu-1x VMs full-time. If your account ends up requiring a card on file, document that honestly in NOTES.md — "free with credit card on file" is still zero-cost but worth mentioning.

### 16.3 Deploy Langfuse on Fly.io (single VM, two containers)

```bash
# Use a custom Dockerfile that runs langfuse + langfuse-db with supervisord,
# OR launch as a separate Fly app with two services. Simpler: use Fly's
# multi-process apps via the [processes] section in fly.toml.
fly launch --no-deploy --name fastapi-helper-langfuse --region sjc
# Configure fly.toml manually with langfuse + postgres processes
fly secrets set NEXTAUTH_SECRET="$(openssl rand -base64 32)" -a fastapi-helper-langfuse
fly secrets set SALT="$(openssl rand -base64 32)" -a fastapi-helper-langfuse
fly deploy -a fastapi-helper-langfuse
# Visit https://fastapi-helper-langfuse.fly.dev, create account, copy keys
```

> If this proves too memory-tight for free tier, fall back to **Langfuse Cloud free tier** for production observability. The point is having traces somewhere reviewable, not necessarily self-hosting in prod. **Document that decision honestly** in ADR-0006.

### 16.4 Deploy backend on Fly.io

`fly.toml`:

```toml
app = "fastapi-helper"
primary_region = "sjc"

[build]
  dockerfile = "Dockerfile"

[env]
  PORT = "8080"
  LOG_LEVEL = "INFO"
  LLM_PROVIDER = "groq"
  EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
  RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = "stop"
  auto_start_machines = true
  min_machines_running = 0
  [[http_service.checks]]
    grace_period = "30s"
    interval = "30s"
    method = "GET"
    timeout = "10s"
    path = "/health"

[[vm]]
  size = "shared-cpu-1x"
  memory = "512mb"  # bump to 1gb if BGE+reranker OOMs

[[mounts]]
  source = "model_cache"
  destination = "/root/.cache/huggingface"
```

```bash
# Set secrets (everything sensitive)
fly secrets set \
  GROQ_API_KEY="..." \
  GEMINI_API_KEY="..." \
  QDRANT_URL="https://fastapi-helper-qdrant.internal:6333" \
  POSTGRES_URL="postgres://..." \
  LANGFUSE_HOST="https://fastapi-helper-langfuse.fly.dev" \
  LANGFUSE_PUBLIC_KEY="pk-lf-..." \
  LANGFUSE_SECRET_KEY="sk-lf-..." \
  API_KEY="$(openssl rand -hex 24)" \
  CORS_ORIGINS="https://fastapi-helper.vercel.app"

fly volumes create model_cache --size 2 --region sjc
fly deploy
```

> **Cold start consideration:** with `min_machines_running=0`, the first request after idle takes ~30s to spin up the VM and load BGE + reranker into memory. Document this in the README's limitations section and surface a "waking up" message in the frontend if the first request times out.

### 16.5 Run ingestion against prod

```bash
# Option A (preferred): SSH into the backend VM and run ingestion
fly ssh console -a fastapi-helper
$ python -m fastapi_helper.ingest.run --max-issues 5500
$ exit

# Option B: run ingestion locally pointed at prod Qdrant + Postgres
# (requires opening ports or using fly proxy)
fly proxy 6333 -a fastapi-helper-qdrant &
fly proxy 5432 -a fastapi-helper-db &
QDRANT_URL=http://localhost:6333 \
POSTGRES_URL=postgres://...localhost:5432/fastapi_helper \
python -m fastapi_helper.ingest.run --max-issues 5500
```

### 16.6 Deploy frontend on Vercel

```bash
cd frontend
vercel link
vercel env add BACKEND_URL production    # https://fastapi-helper.fly.dev
vercel env add BACKEND_API_KEY production # the API_KEY you set above
vercel --prod
```

### 16.7 Verify end-to-end

```bash
# Backend health
curl https://fastapi-helper.fly.dev/health

# Frontend
open https://fastapi-helper.vercel.app

# Sample chat via curl
curl -N -X POST https://fastapi-helper.fly.dev/chat \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I add CORS in FastAPI?"}'
```

If all three return useful output, you're shipped.

### 16.8 Post-deploy verification checklist

- [ ] `/health` returns 200 with `qdrant: true, postgres: true`
- [ ] Chat from the frontend returns a streamed answer with citations
- [ ] Citations link to real fastapi.tiangolo.com or github.com URLs
- [ ] An out-of-scope question ("What is the capital of France?") triggers refusal
- [ ] Langfuse dashboard shows traces with full span breakdown
- [ ] Feedback button writes to Postgres (`SELECT * FROM feedback;`)
- [ ] Cold start works: idle the app 10 minutes, send a request, verify it eventually responds

---

(continued in next file)
