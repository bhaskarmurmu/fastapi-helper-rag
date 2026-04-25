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

**Conclusion:** *(fill in 2–4 sentences based on real numbers)*

**Caveats / what I'd do with more time:** *(e.g., test 768 token chunks, semantic chunking comparison)*

---

## Experiment 2: dense vs. sparse vs. hybrid retrieval

**Hypothesis:** Hybrid (dense + Postgres BM25 fused via RRF) outperforms either alone, especially on issue-answerable questions where exact API names matter.

**Setup:** corpus from winning chunk size in Exp 1. All three configs use the same BGE reranker downstream. Top-30 from each retriever, top-5 after rerank.

| Config | recall@5 | recall@5 (issue bucket) | recall@5 (docs bucket) | MRR | faithfulness | p50_latency |
|---|---|---|---|---|---|---|
| Dense only | __ | __ | __ | __ | __ | __ |
| Sparse only | __ | __ | __ | __ | __ | __ |
| Hybrid (RRF) | __ | __ | __ | __ | __ | __ |

**Per-bucket qualitative notes:** *(any patterns across buckets)*

**Conclusion:** *(2–4 sentences)*

---

## Experiment 3: with vs. without reranker

**Hypothesis:** BGE reranker improves precision@5 measurably; latency cost (~100–500ms on CPU) is acceptable.

**Setup:** corpus from Exp 1, hybrid retrieval from Exp 2.
- Config A: hybrid → top-5 directly (no rerank)
- Config B: hybrid → top-30 → BGE rerank → top-5

| Config | recall@5 | precision@5 | faithfulness | answer_relevancy | p50_latency | p95_latency |
|---|---|---|---|---|---|---|
| No rerank | __ | __ | __ | __ | __ | __ |
| With rerank | __ | __ | __ | __ | __ | __ |

**Conclusion:** *(...)​*

**Latency breakdown for config B (median over eval):**
- Embedding: __ ms
- Dense retrieval: __ ms
- Sparse retrieval: __ ms
- RRF: __ ms
- Reranking: __ ms
- LLM generation (full stream): __ ms

---

## Combined picture: how the final config got there

*(A short paragraph after all experiments: chunk size from Exp 1 → adopted into Exp 2 baseline → hybrid winner from Exp 2 carried into Exp 3 → reranker decision in Exp 3 → final production config: ____)*
