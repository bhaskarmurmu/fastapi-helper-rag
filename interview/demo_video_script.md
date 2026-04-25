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
