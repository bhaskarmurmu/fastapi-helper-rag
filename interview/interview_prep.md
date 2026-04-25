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

These are the high-leverage interview moments. Re-read each ADR before the interview.

### Q3. "How did you evaluate it?"

**Source:** EXPERIMENTS.md + eval/eval_set.jsonl + eval/results/.

**Talking points:**
- 50 hand-written questions, stratified across 4 buckets, ~8 hours of work
- Ragas with Gemini Flash as judge — separate model from generator to avoid bias
- Custom retrieval metrics on URL match
- Three experiments: chunk size sweep, dense vs. hybrid, with/without rerank
- The numbers — paraphrase your real results, don't recite

### Q4. "What was the hardest bug?"

**Source:** LEARNINGS.md sessions where you flagged something tricky.

Pick **one specific bug** and tell it as a story: symptom → wrong hypothesis → debug step → fix.

### Q5. "What would you do differently?"

**Source:** README's "What I'd do differently" + LEARNINGS.md tail.

**Don't say:** "more time" / "better testing" (generic).
**Do say:** one specific architectural change and one specific feature.

### Q6. "How would you scale this to 100x more documents?"

**Talking points:**
- Chunking and embedding stay the same (parallelizable)
- Qdrant scales to billions; bottleneck would be ingestion throughput, not query
- BM25 in Postgres might need to move to dedicated FTS at very large N
- Reranker becomes the latency bottleneck — you'd batch queries or move to GPU
- Cost: at 100x with paid LLM, generation cost dominates; stricter caching and smaller models

### Q7. "How would you handle prompt injection from a poisoned source?"

**Talking points:**
- Acknowledge: anything in retrieved context can contain instructions
- Mitigations: clear delimiter format, system prompt that explicitly rejects in-context instructions, output validation, Langfuse monitoring for unusual outputs
- Honest disclaimer: this corpus is curated upstream (FastAPI maintainers), so the risk is bounded

### Q8. "Why no LangChain / LlamaIndex?"

**Source:** ADR-0005.

Don't trash frameworks. Acknowledge they're useful, then explain your tradeoff: for 3 integrations and a learning-first project, the abstraction tax wasn't worth it.

## Probing questions to expect

- "Why RRF k=60? Did you tune it?" *(Honest answer: no, used the literature default.)*
- "Show me a question your system gets wrong, and tell me why."
- "What's your faithfulness number, and what does that 0.XX actually mean?"
- "How many of your 50 eval questions are *easy* vs. *hard*?"
