# `fastapi-helper` — Spec Part 2: Repository Structure & File-by-File Plan

## 6. Repository Structure

```
fastapi-helper/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # lint + test + eval gate on every PR
│       └── deploy.yml                # deploy on tag push
├── .dockerignore
├── .envrc.example                    # env var template (direnv)
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version                   # 3.11
├── docker-compose.yml                # local dev: qdrant + postgres + langfuse
├── Dockerfile                        # backend image
├── Makefile                          # `make ingest`, `make eval`, `make dev`, etc.
├── pyproject.toml                    # ruff + mypy config + project metadata
├── requirements.txt                  # pinned deps
├── requirements-dev.txt              # pinned dev deps
├── README.md                         # story-driven (template in §17)
├── LEARNINGS.md                      # your real session notes (template in §19)
├── EXPERIMENTS.md                    # 3+ experiments (template in §20)
├── BLOG.md                           # ~1500-word writeup (template in §22)
├── ARCHITECTURE.md                   # detailed architecture, mermaid lives here
├── CONTRIBUTING.md                   # short, even if no contributors expected
├── LICENSE                           # MIT
├── docs/
│   ├── adr/
│   │   ├── README.md                 # ADR index
│   │   ├── template.md               # ADR template
│   │   ├── 0001-vector-db-choice.md
│   │   ├── 0002-embedding-model-choice.md
│   │   ├── 0003-hybrid-vs-dense-retrieval.md
│   │   ├── 0004-reranker-yes-or-no.md
│   │   ├── 0005-no-orchestration-framework.md
│   │   ├── 0006-self-hosted-langfuse.md
│   │   └── 0007-llm-provider-and-fallback.md
│   └── images/
│       ├── architecture.png          # exported from Mermaid
│       └── eval_results.png          # screenshot of dashboard or eval table
├── src/
│   └── fastapi_helper/
│       ├── __init__.py
│       ├── config.py                 # pydantic-settings, all env vars
│       ├── logging_setup.py          # structured logging config
│       ├── models.py                 # pydantic models for requests/responses
│       ├── exceptions.py             # custom exception hierarchy
│       ├── ingest/
│       │   ├── __init__.py
│       │   ├── run.py                # CLI entrypoint: `python -m ...ingest.run`
│       │   ├── docs_loader.py        # parse FastAPI markdown docs
│       │   ├── issues_loader.py      # fetch issues + comments via GitHub API
│       │   ├── chunker.py            # markdown-aware splitter
│       │   ├── embedder.py           # BGE-small wrapper
│       │   └── indexer.py            # upserts into Qdrant + Postgres
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── dense.py              # Qdrant retriever
│       │   ├── sparse.py             # Postgres BM25 retriever
│       │   ├── fusion.py             # RRF
│       │   ├── reranker.py           # BGE CrossEncoder wrapper
│       │   └── pipeline.py           # orchestrates dense+sparse+fusion+rerank
│       ├── generation/
│       │   ├── __init__.py
│       │   ├── prompts.py            # all prompt templates as constants
│       │   ├── llm.py                # provider abstraction (Groq + Gemini)
│       │   └── generator.py          # build prompt, call LLM, parse citations
│       ├── cache/
│       │   ├── __init__.py
│       │   └── semantic_cache.py     # DiskCache + similarity threshold
│       ├── observability/
│       │   ├── __init__.py
│       │   └── tracing.py            # Langfuse setup + decorators
│       └── api/
│           ├── __init__.py
│           ├── main.py               # FastAPI app factory
│           ├── deps.py               # dependency injection (auth, db, etc.)
│           ├── middleware.py         # CORS, rate-limit, request id
│           └── routes/
│               ├── __init__.py
│               ├── chat.py           # POST /chat (streaming SSE)
│               ├── feedback.py       # POST /feedback (thumbs up/down)
│               └── health.py         # GET /health
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # pytest fixtures
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_fusion.py
│   │   ├── test_generator.py
│   │   ├── test_prompts.py
│   │   └── test_semantic_cache.py
│   └── integration/
│       └── test_chat_endpoint.py     # spins up app + Qdrant via testcontainers
├── eval/
│   ├── __init__.py
│   ├── eval_set.jsonl                # 50 hand-curated Q/A/expected-sources
│   ├── eval_set_seed.jsonl           # smaller 10-question set used in CI gate
│   ├── run_eval.py                   # CLI: runs eval, writes results
│   ├── metrics.py                    # custom retrieval metrics (recall@k, MRR)
│   ├── ragas_runner.py               # Ragas integration with Gemini judge
│   └── results/
│       ├── .gitkeep
│       ├── exp1_chunk_size.json
│       ├── exp2_dense_vs_hybrid.json
│       └── exp3_with_vs_without_rerank.json
├── frontend/
│   ├── .gitignore
│   ├── package.json
│   ├── pnpm-lock.yaml
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   ├── postcss.config.js
│   ├── next.config.mjs
│   ├── components.json               # shadcn config
│   ├── public/
│   │   └── favicon.ico
│   └── src/
│       ├── app/
│       │   ├── layout.tsx
│       │   ├── page.tsx              # main chat UI
│       │   ├── globals.css
│       │   └── api/
│       │       └── chat/
│       │           └── route.ts      # proxy to backend, handles SSE
│       ├── components/
│       │   ├── chat/
│       │   │   ├── ChatWindow.tsx
│       │   │   ├── MessageBubble.tsx
│       │   │   ├── SourceList.tsx
│       │   │   ├── FeedbackButtons.tsx
│       │   │   └── EmptyState.tsx
│       │   └── ui/                   # shadcn components
│       ├── lib/
│       │   ├── api.ts                # fetch wrappers
│       │   └── types.ts              # mirror of backend Pydantic types
│       └── hooks/
│           └── useChat.ts            # SSE streaming hook
├── scripts/
│   ├── bootstrap_dev.sh              # one-shot local setup
│   ├── ingest_local.sh               # ingest into local docker-compose
│   └── seed_eval_set.py              # interactive helper to write eval Q's
├── data/
│   ├── .gitkeep
│   ├── raw/                          # gitignored, downloads land here
│   └── processed/                    # gitignored, jsonl outputs here
├── interview/                        # public; portfolio assets
│   ├── resume_bullet.md
│   ├── linkedin_post.md
│   ├── elevator_pitch.md
│   ├── interview_prep.md
│   └── demo_video_script.md
├── scratch/                          # playground for experiments you didn't ship
│   └── README.md                     # explain what's here
└── fly.toml                          # Fly.io config
```

### 6.1 Notes on the structure

- **`src/` layout** (not flat) — modern Python packaging best practice; prevents accidental imports of test code, makes packaging explicit.
- **`scratch/`** — keep one or two genuine experiments you tried and abandoned. Don't fabricate; there will be real ones (e.g., a `chunk_v0_failed.py` you wrote before settling on the markdown-aware version).
- **No `__pycache__/.DS_Store/.idea` etc.** — `.gitignore` properly.
- **`docs/adr/template.md`** is the source of truth for ADR format. Copy from it for each new ADR.

### 6.2 `.gitignore` (exact contents)

```gitignore
# python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
.pytest_cache/
.ruff_cache/
.mypy_cache/
*.egg-info/
.coverage
htmlcov/

# env
.envrc
.env
.env.local

# data
data/raw/
data/processed/
*.jsonl.gz

# models (BGE downloads)
.cache/
models/

# eval
eval/results/*.json
!eval/results/.gitkeep

# qdrant local persistence
qdrant_storage/

# frontend
frontend/node_modules/
frontend/.next/
frontend/out/
frontend/.vercel

# os
.DS_Store
Thumbs.db

# editor
.vscode/
.idea/

# logs
*.log
```

### 6.3 `.envrc.example`

```bash
# copy to .envrc and fill in. direnv will load it.
export GROQ_API_KEY="gsk_..."
export GEMINI_API_KEY="AIza..."          # used by Ragas judge
export GITHUB_TOKEN="ghp_..."             # for issue ingestion (read-only)
export QDRANT_URL="http://localhost:6333"
export POSTGRES_URL="postgresql://postgres:postgres@localhost:5432/fastapi_helper"
export LANGFUSE_HOST="http://localhost:3000"
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export API_KEY="dev-secret-change-me"     # for X-API-Key header
export LLM_PROVIDER="groq"                # or "gemini"
export EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
export RERANKER_MODEL="BAAI/bge-reranker-v2-m3"
export LOG_LEVEL="INFO"
```

---

## 7. File-by-File Implementation Plan

I'll spec the *interfaces* (signatures, behavior, exposed names) without writing the entire body of every function — that's Claude Code's job during implementation. Where a tricky algorithm is involved (RRF, citation parsing), I'll write it out in full.

### 7.1 `src/fastapi_helper/config.py`

```python
"""Centralized config via pydantic-settings. All env vars in one place."""
from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = Field(default="groq", pattern="^(groq|gemini)$")
    groq_api_key: str = ""
    gemini_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-2.0-flash"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1

    # Embedding & rerank
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    embedding_dim: int = 384

    # Retrieval
    dense_top_k: int = 30
    sparse_top_k: int = 30
    rerank_top_n: int = 5
    rrf_k: int = 60
    cache_similarity_threshold: float = 0.97

    # Ingestion
    chunk_size: int = 512
    chunk_overlap: int = 50
    github_token: str = ""

    # Storage
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "fastapi_helper"
    postgres_url: str = "postgresql://postgres:postgres@localhost:5432/fastapi_helper"

    # Observability
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""

    # API
    api_key: str = "dev-secret-change-me"
    cors_origins: list[str] = ["http://localhost:3000"]
    rate_limit_per_minute: int = 30

    # Misc
    log_level: str = "INFO"
    cache_dir: str = ".cache/semantic"
    data_dir: str = "data"

settings = Settings()
```

### 7.2 `src/fastapi_helper/models.py`

```python
"""Pydantic request/response models. Used by API and consumed by frontend."""
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

class Source(BaseModel):
    id: int
    text: str
    url: str
    type: Literal["docs", "issue"]
    score: float
    metadata: dict = Field(default_factory=dict)

class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    stream: bool = True

class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    cache_hit: bool
    latency_ms: int
    request_id: str

class FeedbackRequest(BaseModel):
    request_id: str
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=500)

class FeedbackResponse(BaseModel):
    ok: bool

class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    qdrant: bool
    postgres: bool
    timestamp: datetime
```

### 7.3 `src/fastapi_helper/exceptions.py`

```python
class FastAPIHelperError(Exception): ...
class IngestError(FastAPIHelperError): ...
class RetrievalError(FastAPIHelperError): ...
class GenerationError(FastAPIHelperError): ...
class RateLimitError(FastAPIHelperError): ...
class ConfigError(FastAPIHelperError): ...
```

### 7.4 `src/fastapi_helper/ingest/docs_loader.py`

```python
"""Parse FastAPI markdown docs into Document objects."""
from pathlib import Path
from pydantic import BaseModel

class Document(BaseModel):
    text: str
    source_url: str            # canonical URL on fastapi.tiangolo.com
    source_type: str           # "docs"
    title: str | None
    section_path: list[str]    # e.g., ["Tutorial", "Path Parameters"]

def load_docs(repo_path: Path) -> list[Document]:
    """Walk docs/en/docs/**/*.md, parse each, return Documents.

    For each .md file:
      - Read front-matter (if any)
      - Extract H1 as title
      - Build canonical URL by stripping `docs/en/docs/` prefix and `.md` suffix,
        prepending `https://fastapi.tiangolo.com/`
      - Section path inferred from directory structure
    Returns one Document per file (chunking happens later).
    """
```

### 7.5 `src/fastapi_helper/ingest/issues_loader.py`

```python
"""Fetch closed issues from tiangolo/fastapi via GitHub REST API."""
import httpx
from .docs_loader import Document

def load_issues(
    github_token: str,
    repo: str = "tiangolo/fastapi",
    state: str = "closed",
    cache_path: Path | None = None,
    max_issues: int | None = None,
) -> list[Document]:
    """Paginate issues, fetch each issue's comments, flatten to one Document per issue.

    Document.text format:
        # {title}
        {body}
        ---
        Comment by @{user}:
        {comment_body}
        ---
        ...
    metadata: {issue_number, labels, closed_at, num_comments}
    source_url: https://github.com/tiangolo/fastapi/issues/{number}
    source_type: "issue"

    Caches paginated results to JSONL on disk so reruns are incremental.
    Skips PRs (issues API returns both; filter by `pull_request` field).
    """
```

### 7.6 `src/fastapi_helper/ingest/chunker.py`

```python
"""Markdown-aware chunker. Never splits code blocks. Respects headers."""
from pydantic import BaseModel

class Chunk(BaseModel):
    text: str
    source_url: str
    source_type: str
    chunk_index: int           # position within document
    metadata: dict

def chunk_document(
    doc: Document,
    chunk_size: int = 512,    # tokens, approximated by tiktoken cl100k
    overlap: int = 50,
) -> list[Chunk]:
    """Algorithm:
    1. Tokenize with tiktoken cl100k_base for size accounting.
    2. First, split on top-level markdown structure: code fences, headers.
    3. Recursively split sections that exceed chunk_size, prefer split points
       in this priority order: '\\n## ', '\\n### ', '\\n\\n', '\\n', '. ', ' '.
    4. Code fences (```...```) are atomic — never split internally.
       If a code fence alone exceeds chunk_size, emit it as one oversized chunk
       and log a warning (this is fine, retrieval will handle it).
    5. Apply overlap by carrying the last `overlap` tokens of chunk N
       into the start of chunk N+1.
    6. Always include the most recent header(s) at the start of each chunk
       so retrieved chunks have context (e.g., '## Path Parameters\\n\\n...').
    """
```

> **Test in `tests/unit/test_chunker.py`:** assert no chunk splits a code block (no chunk contains an unmatched ```). Assert overlap is honored. Assert headers are prepended.

### 7.7 `src/fastapi_helper/ingest/embedder.py`

```python
"""Wrap sentence-transformers BGE-small. Batched, normalized."""
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        is_query: bool = False,
    ) -> np.ndarray:
        """BGE convention: prepend 'query: ' for queries (some BGE variants).
        For bge-small-en-v1.5 specifically, no prefix is needed.
        Returns L2-normalized embeddings (sets normalize_embeddings=True).
        """
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
```

### 7.8 `src/fastapi_helper/ingest/indexer.py`

```python
"""Upsert chunks into Qdrant + Postgres."""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import sqlalchemy as sa

class Indexer:
    def __init__(self, qdrant_url: str, postgres_url: str, collection: str): ...

    def init_qdrant(self, dim: int) -> None:
        """Create collection if absent. distance=Cosine (BGE is normalized)."""

    def init_postgres(self) -> None:
        """Create chunks table:
          id BIGSERIAL PK,
          source_url TEXT NOT NULL,
          source_type TEXT NOT NULL,
          chunk_index INT NOT NULL,
          text TEXT NOT NULL,
          metadata JSONB NOT NULL DEFAULT '{}',
          ts tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
          UNIQUE (source_url, chunk_index)
        Plus GIN index on ts.
        Plus feedback table:
          request_id TEXT, rating TEXT, comment TEXT, created_at TIMESTAMP
        """

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Write to both stores. ID strategy: deterministic hash of
        (source_url, chunk_index) so re-ingestion is idempotent.
        Postgres uses ON CONFLICT (source_url, chunk_index) DO UPDATE.
        Qdrant uses point IDs derived from same hash.
        """
```

### 7.9 `src/fastapi_helper/ingest/run.py`

```python
"""CLI entrypoint for ingestion. `python -m fastapi_helper.ingest.run`."""
import argparse, logging
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-issues", action="store_true")
    parser.add_argument("--skip-docs", action="store_true")
    parser.add_argument("--max-issues", type=int, default=None,
                        help="Limit issues for dev iteration")
    args = parser.parse_args()

    # 1. Ensure data/raw/fastapi exists; clone or pull
    # 2. Load docs (docs_loader)
    # 3. Load issues (issues_loader, with cache)
    # 4. Chunk all documents
    # 5. Embed all chunks (batched)
    # 6. Indexer.upsert
    # 7. Print summary: N docs, M chunks, K vectors indexed, time, peak RAM

if __name__ == "__main__":
    main()
```

### 7.10 `src/fastapi_helper/retrieval/dense.py`

```python
class DenseRetriever:
    def __init__(self, qdrant: QdrantClient, collection: str, embedder: Embedder): ...

    def retrieve(self, query: str, top_k: int = 30) -> list[Source]:
        """Embed query, search Qdrant, return Source objects with score."""
```

### 7.11 `src/fastapi_helper/retrieval/sparse.py`

```python
class SparseRetriever:
    def __init__(self, postgres_url: str): ...

    def retrieve(self, query: str, top_k: int = 30) -> list[Source]:
        """SQL: SELECT id, source_url, source_type, text,
                  ts_rank_cd(ts, plainto_tsquery('english', :q)) AS score
                FROM chunks
                WHERE ts @@ plainto_tsquery('english', :q)
                ORDER BY score DESC LIMIT :k;
        Convert rank scores to a normalized [0,1] for downstream use.
        """
```

### 7.12 `src/fastapi_helper/retrieval/fusion.py`

```python
"""Reciprocal Rank Fusion. Standard k=60."""

def reciprocal_rank_fusion(
    rankings: list[list[Source]],
    k: int = 60,
) -> list[Source]:
    """Combine N ranked lists. Each Source's fused score is sum over lists of
    1/(k + rank_in_list). Dedupe by (source_url, chunk_index).
    Returns sorted desc by fused score.
    """
    scores: dict[tuple[str, int], float] = {}
    sources: dict[tuple[str, int], Source] = {}
    for ranking in rankings:
        for rank, src in enumerate(ranking):
            key = (src.url, src.metadata.get("chunk_index", 0))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in sources:
                sources[key] = src
    fused = []
    for key, score in sorted(scores.items(), key=lambda x: -x[1]):
        s = sources[key].model_copy(update={"score": score})
        fused.append(s)
    return fused
```

### 7.13 `src/fastapi_helper/retrieval/reranker.py`

```python
class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, sources: list[Source], top_n: int = 5) -> list[Source]:
        """Score (query, source.text) pairs, sort desc, return top_n.
        If len(sources) <= top_n, still rerank to attach proper scores.
        """
        if not sources:
            return []
        pairs = [[query, s.text] for s in sources]
        scores = self._model.predict(pairs)  # numpy array
        ranked = sorted(zip(sources, scores), key=lambda x: -x[1])
        return [s.model_copy(update={"score": float(score)}) for s, score in ranked[:top_n]]
```

### 7.14 `src/fastapi_helper/retrieval/pipeline.py`

```python
class RetrievalPipeline:
    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever,
                 reranker: Reranker, rrf_k: int = 60): ...

    def __call__(self, query: str, top_n: int = 5) -> list[Source]:
        """Run dense (top 30) and sparse (top 30) in parallel via threadpool,
        fuse with RRF, rerank to top_n, return."""
```

### 7.15 `src/fastapi_helper/generation/prompts.py`

```python
"""All prompt templates. Constants only — no logic. Easy to diff over time."""

SYSTEM_PROMPT = """You are FastAPIHelper, a focused assistant that answers \
questions about the FastAPI Python framework using ONLY the provided context.

Rules:
1. Answer ONLY from the provided sources. If the answer is not in the sources, \
say "I don't have enough information in the provided sources to answer that \
confidently." and stop.
2. Cite sources inline using [N] notation matching the source numbers below. \
Every factual claim needs a citation.
3. Prefer code examples from the sources verbatim where they help.
4. Be concise. Aim for 3–8 sentences plus code if relevant. No filler.
5. Never invent imports, function signatures, or behavior not present in sources.
"""

USER_PROMPT_TEMPLATE = """Sources:
{sources_block}

Question: {question}

Answer (cite [N] inline):"""

# Source block format produced by generator.py:
# [1] (docs: https://fastapi.tiangolo.com/tutorial/path-params/)
# Path parameters... <chunk text truncated to ~800 chars>
#
# [2] (issue: https://github.com/tiangolo/fastapi/issues/1234)
# Title: ...
# <chunk text>
```

### 7.16 `src/fastapi_helper/generation/llm.py`

```python
"""Provider abstraction. Same interface for Groq and Gemini."""
from typing import AsyncIterator
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    async def stream(self, system: str, user: str,
                     max_tokens: int, temperature: float) -> AsyncIterator[str]: ...

class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model: str): ...
    async def stream(self, system, user, max_tokens, temperature):
        # Use groq SDK with stream=True; yield delta tokens.
        ...

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str): ...
    async def stream(self, system, user, max_tokens, temperature):
        # Use google-generativeai with stream=True; yield text chunks.
        ...

def get_llm(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "groq":
        return GroqProvider(settings.groq_api_key, settings.groq_model)
    return GeminiProvider(settings.gemini_api_key, settings.gemini_model)
```

### 7.17 `src/fastapi_helper/generation/generator.py`

```python
class Generator:
    def __init__(self, llm: LLMProvider): ...

    def build_prompt(self, question: str, sources: list[Source]) -> tuple[str, str]:
        """Returns (system_prompt, user_prompt). Truncates long source text to
        ~800 chars per source to fit context budget."""

    async def generate(self, question: str, sources: list[Source]
                       ) -> AsyncIterator[str]:
        """Yields token chunks. Caller is responsible for assembling final string
        and parsing citations after stream completes."""

    @staticmethod
    def parse_citations(answer: str, sources: list[Source]) -> list[Source]:
        """Find all [N] tokens in answer; return the corresponding sources
        in citation order (deduped, only those actually cited)."""
```

### 7.18 `src/fastapi_helper/cache/semantic_cache.py`

```python
import diskcache
import numpy as np

class SemanticCache:
    def __init__(self, cache_dir: str, threshold: float = 0.97, max_keys: int = 5000):
        self._cache = diskcache.Cache(cache_dir)
        self.threshold = threshold
        self.max_keys = max_keys
        # Maintain a separate "index" of (query_text, embedding) for similarity check.
        # On startup, rebuild index from cache contents.
        self._index: list[tuple[str, np.ndarray]] = []
        self._rebuild_index()

    def _rebuild_index(self) -> None: ...

    def get(self, query_embedding: np.ndarray) -> dict | None:
        """Linear scan of self._index, return cached value if max similarity >= threshold."""

    def set(self, query: str, query_embedding: np.ndarray, value: dict) -> None:
        """Store {query, value, embedding} keyed by query hash. Evict LRU when full."""
```

### 7.19 `src/fastapi_helper/observability/tracing.py`

```python
from langfuse import Langfuse
from contextlib import contextmanager

class Tracer:
    def __init__(self, settings: Settings):
        if not settings.langfuse_public_key:
            self._lf = None  # no-op mode for tests/CI
            return
        self._lf = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )

    @contextmanager
    def trace(self, name: str, **kwargs):
        if self._lf is None:
            yield None; return
        trace = self._lf.trace(name=name, metadata=kwargs)
        try:
            yield trace
        finally:
            trace.update(...)
```

### 7.20 `src/fastapi_helper/api/main.py`

```python
"""FastAPI app factory."""
from fastapi import FastAPI
from .middleware import setup_middleware
from .routes import chat, feedback, health

def create_app() -> FastAPI:
    app = FastAPI(
        title="fastapi-helper",
        version="0.1.0",
        description="Customer-support RAG over FastAPI docs + issues.",
    )
    setup_middleware(app)
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(feedback.router)
    return app

app = create_app()
```

### 7.21 `src/fastapi_helper/api/middleware.py`

```python
"""CORS, rate limit, request id."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
import uuid

limiter = Limiter(key_func=get_remote_address)

def setup_middleware(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    # Request ID middleware: generate uuid4 if absent, attach to request.state and
    # echo in `X-Request-ID` response header.
    app.state.limiter = limiter
```

### 7.22 `src/fastapi_helper/api/deps.py`

```python
"""Dependency injection: settings, retriever, generator, tracer, auth."""
from fastapi import Header, HTTPException, status
from functools import lru_cache

@lru_cache
def get_pipeline() -> RetrievalPipeline: ...

@lru_cache
def get_generator() -> Generator: ...

@lru_cache
def get_cache() -> SemanticCache: ...

@lru_cache
def get_tracer() -> Tracer: ...

def require_api_key(x_api_key: str = Header(...)) -> None:
    if x_api_key != settings.api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")
```

### 7.23 `src/fastapi_helper/api/routes/chat.py`

```python
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
import json, time, uuid

router = APIRouter()

@router.post("/chat")
@limiter.limit("30/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    _: None = Depends(require_api_key),
    pipeline: RetrievalPipeline = Depends(get_pipeline),
    generator: Generator = Depends(get_generator),
    cache: SemanticCache = Depends(get_cache),
    tracer: Tracer = Depends(get_tracer),
):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    t0 = time.time()
    with tracer.trace("chat", request_id=request_id, question=body.question) as t:
        # 1. Embed question
        # 2. Cache check
        # 3. If miss: pipeline(question, top_n=5) → sources
        # 4. Stream generator.generate(question, sources)
        # 5. After stream completes, parse citations, write to cache
        # 6. Return SSE stream

        async def event_stream():
            # First event: 'sources' with the retrieved sources (so frontend can render
            # the source list while answer streams)
            yield f"event: sources\ndata: {json.dumps([s.model_dump() for s in sources])}\n\n"
            full = ""
            async for token in generator.generate(body.question, sources):
                full += token
                yield f"event: token\ndata: {json.dumps({'t': token})}\n\n"
            cited = generator.parse_citations(full, sources)
            yield f"event: done\ndata: {json.dumps({'request_id': request_id, 'latency_ms': int((time.time()-t0)*1000), 'cited': [c.id for c in cited]})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
```

### 7.24 `src/fastapi_helper/api/routes/feedback.py`

```python
@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(body: FeedbackRequest, _: None = Depends(require_api_key)):
    """INSERT INTO feedback (request_id, rating, comment, created_at) VALUES (...)"""
    return FeedbackResponse(ok=True)
```

### 7.25 `src/fastapi_helper/api/routes/health.py`

```python
@router.get("/health", response_model=HealthResponse)
def health():
    """Ping Qdrant + Postgres; return status + timestamps."""
```

### 7.26 `eval/run_eval.py`

```python
"""CLI: python -m eval.run_eval --config <name> --output <path>

Steps:
1. Load eval_set.jsonl
2. For each question, run the pipeline + generator (no cache for eval)
3. Capture: question, ground_truth, answer, retrieved_contexts, expected_sources, retrieved_sources
4. Score retrieval: recall@5, MRR, precision@5
5. Score generation with Ragas (faithfulness, answer_relevance, context_precision, context_recall)
   using Gemini Flash as judge (LangChainLLM wrapper)
6. Write JSON results to eval/results/{config}.json
7. Print summary table to stdout
"""
```

### 7.27 `eval/metrics.py`

```python
def recall_at_k(retrieved_urls: list[str], expected_urls: list[str], k: int) -> float:
    """Of expected URLs, how many appear in retrieved[:k]. Allows partial URL match."""

def mean_reciprocal_rank(retrieved_urls: list[str], expected_urls: list[str]) -> float: ...

def precision_at_k(retrieved_urls: list[str], expected_urls: list[str], k: int) -> float: ...
```

### 7.28 `eval/ragas_runner.py`

```python
"""Ragas integration with Gemini Flash judge."""
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from datasets import Dataset

def run_ragas(rows: list[dict]) -> dict:
    """rows: [{question, answer, contexts, ground_truth}, ...]
    Returns dict of metric → mean score.
    Sleeps between rows to respect Gemini RPM limit.
    """
```

---

(continued in next file)
