# Architecture

## System diagram

```mermaid
flowchart TB
    subgraph Ingestion["Ingestion (offline, run via Make target)"]
        D1[FastAPI docs<br/>git clone tiangolo/fastapi]
        D2[GitHub Issues API<br/>closed issues, paginated]
        D1 --> P1[Markdown parser<br/>preserves code blocks]
        D2 --> P2[Issue + comments<br/>thread flattener]
        P1 --> C[Chunker<br/>markdown-aware<br/>512 tokens, 50 overlap]
        P2 --> C
        C --> E[BGE-small-en-v1.5<br/>local CPU encoding]
        E --> Q[(Qdrant<br/>dense vectors<br/>+ payload metadata)]
        C --> B[(Postgres tsvector<br/>BM25 index)]
    end

    subgraph Query["Query (online)"]
        U[User question] --> API[FastAPI<br/>/chat endpoint]
        API --> QE[Query embed<br/>BGE-small]
        QE --> DR[Dense retrieval<br/>Qdrant top-30]
        API --> SR[Sparse retrieval<br/>BM25 top-30]
        DR --> RRF[Reciprocal Rank<br/>Fusion]
        SR --> RRF
        RRF --> RR[BGE reranker v2-m3<br/>top-30 → top-5]
        RR --> CACHE{Semantic<br/>cache hit?}
        CACHE -- yes --> RESP[Cached response]
        CACHE -- no --> LLM[Groq Llama 3.3 70B<br/>cited answer generation]
        LLM --> RESP
        RESP --> User[User]
    end

    subgraph Obs["Observability"]
        API -.trace.-> LF[Langfuse<br/>self-hosted]
        LLM -.trace.-> LF
        RR -.trace.-> LF
    end

    Q -.queried by.-> DR
    B -.queried by.-> SR
```

## Component breakdown

| Component | Responsibility |
|---|---|
| **Ingestion CLI** | One-shot script that clones FastAPI repo, fetches issues, chunks, embeds, indexes |
| **Qdrant** | Stores 384-dim BGE embeddings + payload (chunk text, source URL, type, metadata) |
| **Postgres** | Stores raw chunks for BM25 (`tsvector` column with GIN index) and feedback log |
| **Retrieval service** | Hybrid retriever: dense + sparse → RRF → rerank → top-k |
| **Generation service** | Builds prompt, calls Groq, parses citations, returns structured response |
| **Semantic cache** | DiskCache on disk, keyed by query embedding similarity (>0.97 threshold) |
| **FastAPI backend** | `/chat`, `/health`, `/feedback` endpoints; auth via API key header |
| **Next.js frontend** | Single-page chat UI with streaming responses and source rendering |
| **Langfuse** | Self-hosted observability; traces every request component |
| **GitHub Actions** | Lint (ruff), test (pytest), eval gate (Ragas on cached fixture) |

## Sequence diagram

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Next.js
    participant BE as FastAPI
    participant C as DiskCache
    participant Q as Qdrant
    participant P as Postgres
    participant R as BGE Reranker
    participant L as Groq LLM
    participant LF as Langfuse

    U->>FE: Types question, hits Send
    FE->>BE: POST /chat {question}
    BE->>LF: trace.start
    BE->>BE: Embed query (BGE-small)
    BE->>C: Check semantic cache
    alt Cache hit
        C-->>BE: Cached response
        BE-->>FE: Stream cached answer
    else Cache miss
        par Dense retrieval
            BE->>Q: Vector search top-30
            Q-->>BE: 30 candidates
        and Sparse retrieval
            BE->>P: BM25 search top-30
            P-->>BE: 30 candidates
        end
        BE->>BE: RRF fusion → 30 unique
        BE->>R: Rerank top-30
        R-->>BE: top-5 with scores
        BE->>L: Generate with context (streaming)
        L-->>BE: Streamed tokens
        BE-->>FE: SSE stream + citations
        BE->>C: Store in cache
    end
    BE->>LF: trace.end (latency, cost, scores)
    FE-->>U: Renders answer + sources
```
