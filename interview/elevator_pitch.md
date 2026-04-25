# Elevator pitch

## 30-second version

fastapi-helper is a customer-support-style RAG assistant over the FastAPI docs and GitHub issues. I built it to ground every answer in cited sources, with a hand-curated 50-question eval set and a CI gate that blocks any PR dropping retrieval quality. Everything runs on free-tier infrastructure — local BGE embeddings, Qdrant in Docker, Groq for generation, self-hosted Langfuse — and the documented experiments comparing chunk size, dense vs. hybrid, and reranking made the design decisions evidence-based instead of vibes-based.

## 60-second version (add)

The interesting bits were less about the LLM and more about everything around it. The first version felt fine on a few queries and was deeply mediocre when I finally measured it. Building the eval set first would have saved me a week. The chunking story alone could be its own talk — {{fill in your real chunking story from LEARNINGS.md}}. That kind of bug doesn't show up in a tutorial; it only shows up when you have a real eval set to embarrass you.
