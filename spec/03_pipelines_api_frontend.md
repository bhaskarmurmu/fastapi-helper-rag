# `fastapi-helper` — Spec Part 3: Pipelines, API, Frontend

## 8. Data & Ingestion Pipeline

### 8.1 Sources

**Source 1: FastAPI documentation**
- Repo: `https://github.com/tiangolo/fastapi`
- Path within repo: `docs/en/docs/**/*.md`
- License: MIT (commercial use OK; attribution preserved via citation)
- Approximate size: ~150 markdown files, ~600KB

**Source 2: GitHub closed issues**
- Endpoint: `GET https://api.github.com/repos/tiangolo/fastapi/issues?state=closed&per_page=100`
- Auth: `Authorization: Bearer $GITHUB_TOKEN` (a personal access token with `public_repo` scope)
- Approximate volume: ~5,500 closed issues + ~25,000 comments
- Rate limit: 5,000 req/hr authenticated; pagination + comments will use ~6,000 calls. **Will require pagination + caching.**

> **Important:** the GitHub Issues endpoint returns both Issues and Pull Requests. Filter where `pull_request` is absent.

### 8.2 Download / prep commands

These live in `scripts/ingest_local.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw

# 1. Clone or pull FastAPI repo
if [[ ! -d data/raw/fastapi ]]; then
  git clone --depth 1 https://github.com/tiangolo/fastapi.git data/raw/fastapi
else
  git -C data/raw/fastapi pull --rebase
fi

# 2. Run ingestion
python -m fastapi_helper.ingest.run --max-issues "${MAX_ISSUES:-2000}"
```

> **Dev iteration tip:** start with `MAX_ISSUES=200` so the first ingestion takes ~5 minutes, not 45. Bump to full ~5,500 only when the pipeline is stable.

### 8.3 Chunking parameters (settled)

- **Tokenizer:** `tiktoken` `cl100k_base` (the GPT-4 tokenizer; close enough for our purposes — actual token counts in BGE differ slightly but the relative ratio is what matters for chunking).
- **Chunk size:** 512 tokens
- **Overlap:** 50 tokens
- **Rules:** code fences atomic, headers prepended to each chunk

These specific numbers come from Experiment 1 (which you'll run; see §12). The defaults are 512/50; adjust if your experiment data says otherwise.

### 8.4 Embedding generation

- **Model:** `BAAI/bge-small-en-v1.5`
- **Batch size:** 64 (fits comfortably in 1GB RAM CPU)
- **Normalization:** yes (`normalize_embeddings=True`)
- **Throughput on CPU:** ~200 chunks/sec on a modern laptop. For ~10K chunks (docs + issues), ingestion-side embedding takes ~1 minute. The bottleneck is the GitHub issue download, not embedding.

### 8.5 Indexing logic

**Qdrant collection setup:**

```python
client.create_collection(
    collection_name="fastapi_helper",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)
client.create_payload_index("fastapi_helper", "source_type", "keyword")
client.create_payload_index("fastapi_helper", "source_url", "keyword")
```

**Postgres schema:**

```sql
CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    source_url TEXT NOT NULL,
    source_type TEXT NOT NULL CHECK (source_type IN ('docs', 'issue')),
    chunk_index INT NOT NULL,
    text TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    ts tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_url, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_ts_gin ON chunks USING GIN (ts);
CREATE INDEX IF NOT EXISTS chunks_source_type ON chunks (source_type);

CREATE TABLE IF NOT EXISTS feedback (
    id BIGSERIAL PRIMARY KEY,
    request_id TEXT NOT NULL,
    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    comment TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

**Idempotency strategy:** point IDs in Qdrant are `int64(blake2b(source_url||chunk_index)[:8])`. Postgres uses the unique constraint with `ON CONFLICT (source_url, chunk_index) DO UPDATE SET text=EXCLUDED.text, metadata=EXCLUDED.metadata`. Re-running ingestion overwrites in place.

### 8.6 Reproducible entrypoint

```bash
make ingest
# expands to:
# bash scripts/ingest_local.sh
```

After ingestion, you should see (numbers approximate):
```
Loaded 153 docs, 5,412 issues
Generated 11,847 chunks
Embedded in 58.3s
Indexed 11,847 vectors in Qdrant; 11,847 rows in Postgres
Total: 4m 12s
```

> **Save this output.** Paste the real numbers into `EXPERIMENTS.md` when you write it up.

---

## 9. Retrieval & Generation Logic

### 9.1 Retrieval method (settled)

**Hybrid (dense + sparse) → RRF → BGE rerank.** Justification in ADR-0003 and Experiment 2.

### 9.2 Exact parameters

| Parameter | Value | Why |
|---|---|---|
| `dense_top_k` | 30 | Enough recall headroom for the reranker |
| `sparse_top_k` | 30 | Same |
| `rrf_k` | 60 | Standard in the literature; not tuned |
| `rerank_top_n` | 5 | Keeps prompt context tight; 5 chunks × ~800 chars = ~4000 chars context, well within Llama 3.3's 128K window but more isn't better (lost-in-the-middle) |
| Per-source truncation | 800 chars | Aggressive but prevents one long chunk from dominating |

### 9.3 Query preprocessing

- **Trim and lowercase whitespace-normalize** the question. No fancy rewriting in v1; it's a stretch goal. Document the tradeoff in an ADR if you skip it.
- **Reject** queries < 3 chars or > 2000 chars at the API layer (Pydantic constraint).

### 9.4 Full prompt template

System (constant — `SYSTEM_PROMPT` in `prompts.py`):

```
You are FastAPIHelper, a focused assistant that answers questions about the FastAPI Python framework using ONLY the provided context.

Rules:
1. Answer ONLY from the provided sources. If the answer is not in the sources, say "I don't have enough information in the provided sources to answer that confidently." and stop.
2. Cite sources inline using [N] notation matching the source numbers below. Every factual claim needs a citation.
3. Prefer code examples from the sources verbatim where they help.
4. Be concise. Aim for 3–8 sentences plus code if relevant. No filler.
5. Never invent imports, function signatures, or behavior not present in sources.
```

User (templated per request):

```
Sources:
[1] (docs: https://fastapi.tiangolo.com/tutorial/path-params/)
{chunk text, max 800 chars}

[2] (issue: https://github.com/tiangolo/fastapi/issues/1234)
{chunk text, max 800 chars}

[3] (docs: ...)
...

Question: {question}

Answer (cite [N] inline):
```

### 9.5 Citation strategy

- **Format:** inline `[N]` markers in the model's output where N matches the source list.
- **Parsing:** regex `r"\[(\d+)\]"` on the final assembled answer; map each N back to the source by index. Dedupe in citation order.
- **Frontend rendering:** every `[N]` in the answer becomes a clickable superscript that opens the source URL in a new tab. The "Sources" panel below the answer shows only sources that were actually cited (i.e., parsed out of the answer), not all 5 retrieved.
- **Why only cited sources are shown:** retrieval recall ≠ relevance to the answer. Showing 5 sources when 2 were used confuses users and pads the UI. Show retrieved-but-uncited sources behind a "show all retrieved" toggle for transparency.

### 9.6 Streaming approach

**Server-Sent Events (SSE)**, three event types:

1. `event: sources` — fired immediately after retrieval, before generation. Payload: full retrieved-source list (so frontend can render the sources panel while the answer streams).
2. `event: token` — fired per generated token chunk. Payload: `{"t": "..."}`.
3. `event: done` — fired after generation completes. Payload: `{"request_id": "...", "latency_ms": ..., "cited": [1, 3]}` so the frontend knows which sources to keep visible.

> **Why SSE over WebSocket:** SSE is one-directional (server → client), simpler to deploy through HTTP proxies, native support in browsers via `EventSource`. WebSocket would be overkill here.

---

## 10. API Specification

### 10.1 Auth

Header: `X-API-Key: <key>`. Single shared key for the demo. In production this would be per-user JWTs; for a portfolio project, a single rotating key is honest and adequate. **Document this tradeoff in ADR-0008** if you add one.

### 10.2 Rate limiting

`slowapi` middleware, 30 requests/minute per IP. Returns `429 Too Many Requests` with a `Retry-After` header.

### 10.3 CORS

Allowed origins from `CORS_ORIGINS` env var (comma-separated list). Default in dev: `http://localhost:3000`. In prod: your Vercel URL.

### 10.4 Endpoints

#### `POST /chat`

**Request:**
```json
{
  "question": "How do I add CORS middleware in FastAPI?",
  "stream": true
}
```

**Headers:** `X-API-Key: <key>`, `Content-Type: application/json`.

**Response (streaming, `text/event-stream`):**

```
event: sources
data: [{"id":1,"text":"...","url":"https://fastapi.tiangolo.com/tutorial/cors/","type":"docs","score":0.91,"metadata":{}}, ...]

event: token
data: {"t": "To "}

event: token
data: {"t": "add "}

...

event: done
data: {"request_id":"abc-123","latency_ms":1432,"cited":[1,3]}
```

**Errors:**
- `400` invalid body (Pydantic validation)
- `401` missing/wrong API key
- `429` rate limited
- `500` internal — generic message, full trace in Langfuse + logs
- `503` if Qdrant or Postgres health checks fail

#### `POST /feedback`

**Request:**
```json
{
  "request_id": "abc-123",
  "rating": "up",
  "comment": "Great answer, exactly what I needed."
}
```

**Response:**
```json
{ "ok": true }
```

#### `GET /health`

**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "qdrant": true,
  "postgres": true,
  "timestamp": "2026-04-25T10:00:00Z"
}
```

If `qdrant` or `postgres` is `false`, returns `503`.

### 10.5 Error format (consistent)

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Question must be at least 3 characters.",
    "request_id": "abc-123"
  }
}
```

---

## 11. Frontend Specification

Single-page app. Don't over-build.

### 11.1 Pages

- `/` — chat interface. That's it. No login, no settings, no history page (history is in stretch goals).

### 11.2 Components

| Component | File | Purpose |
|---|---|---|
| `ChatWindow` | `components/chat/ChatWindow.tsx` | Top-level layout: input at bottom, scrolling message list above |
| `MessageBubble` | `components/chat/MessageBubble.tsx` | Single Q or A message; assistant messages render markdown + citation pills |
| `SourceList` | `components/chat/SourceList.tsx` | Below each assistant message, the cited sources as clickable cards |
| `FeedbackButtons` | `components/chat/FeedbackButtons.tsx` | Thumbs up/down per assistant message |
| `EmptyState` | `components/chat/EmptyState.tsx` | Initial state: 3 example questions as clickable chips |

shadcn/ui components used: `Button`, `Card`, `Input`, `Textarea`, `Skeleton`, `Toast`, `Tooltip`.

### 11.3 State management

Single React state in the `useChat` hook:

```ts
interface ChatState {
  messages: Message[];        // {role, content, sources?, requestId?}
  isStreaming: boolean;
  error: string | null;
}
```

No Redux, no Zustand. The state is small and lives in one component tree. Don't over-engineer.

### 11.4 SSE handling

Use the `EventSource` API directly (not a library). One important quirk: native `EventSource` doesn't support custom headers, which means we can't send `X-API-Key`. Workaround: route the request through a Next.js API route (`app/api/chat/route.ts`) that proxies to the backend, attaching the API key server-side from `process.env.BACKEND_API_KEY`. This also keeps the API key out of the browser bundle entirely. **This proxy pattern is itself a portfolio signal** — document it in NOTES.md when you build it.

```ts
// app/api/chat/route.ts (Next.js App Router)
import { NextRequest } from "next/server";

export async function POST(req: NextRequest) {
  const body = await req.json();
  const upstream = await fetch(`${process.env.BACKEND_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": process.env.BACKEND_API_KEY!,
    },
    body: JSON.stringify(body),
  });
  return new Response(upstream.body, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    },
  });
}
```

### 11.5 Loading / error / empty states

- **Empty (initial):** centered `EmptyState` with three clickable example questions:
  - "How do I add CORS middleware in FastAPI?"
  - "What's the difference between path parameters and query parameters?"
  - "How do I run background tasks after returning a response?"
- **Loading (during stream):** skeleton placeholder for the assistant bubble that fills in as tokens arrive. A subtle "..." pulse below while sources are being retrieved (before first `token` event).
- **Error:** inline error toast (shadcn `Toast`) with retry button. Specific messages for `429` ("Slow down — try again in a moment") and `503` ("Backend is waking up — try again in 30s").
- **Empty answer (the "I don't have enough information" case):** rendered with a different border color (amber) to make the refusal visually distinct. This is intentional UX — surfacing the refusal as a feature, not hiding it.

### 11.6 Styling

- Tailwind, default config + Inter font from `next/font/google`.
- Color palette: shadcn default neutral. One accent color (Tailwind `emerald-600`) for the FastAPI green vibe, used sparingly (citation pills, send button).
- Mobile responsive: stacked layout on narrow screens; sources collapse into an expandable section.
- Dark mode: yes, via shadcn's theme system. Store preference in localStorage.

### 11.7 Markdown rendering

Use `react-markdown` + `remark-gfm` + `rehype-highlight` for code syntax highlighting. The assistant's answer is markdown — render it properly, code blocks with monospace and a "Copy" button.

---

(continued in next file)
