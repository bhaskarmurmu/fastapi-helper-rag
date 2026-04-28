"""Prompt templates and chunk-formatting helpers.

All constants and pure functions — no I/O, no side effects.
Easy to diff and tune over time without touching generation logic.
"""
from __future__ import annotations

from fastapi_helper.retrieval.types import RetrievedChunk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Verbatim complete-refusal phrase (Rule 1).  Used when the entire question
# is unanswerable from sources.  Phase 11 evals pattern-match this exact
# string.  Partial gaps (Rule 6 — multi-part questions) use natural phrasing
# ("cannot answer from the provided sources") and are not matched here.
REFUSAL_SIGNAL = (
    "I don't have enough information in the provided sources to answer that confidently."
)

# Characters allowed per chunk in the prompt.  Aggressive but intentional:
# 5 chunks × 800 chars ≈ 4 000 chars ≈ 1 000 tokens, well inside Llama 3's
# smallest context window.  A single unusually long chunk can't dominate.
_MAX_CHUNK_CHARS = 800

SYSTEM_PROMPT = f"""\
You are FastAPIHelper, a focused assistant that answers questions about the \
FastAPI Python framework, grounding all FastAPI-specific claims in the \
provided context.

Rules:
1. All FastAPI-specific claims — function names, parameters, behaviour, \
configuration — must come from the provided sources. You may use general \
programming knowledge (Python, HTTP, async/await) to interpret and explain \
what the sources say, but not to fill in FastAPI-specific gaps. If a \
FastAPI-specific question cannot be answered from the sources, say \
"{REFUSAL_SIGNAL}" and stop.
2. Cite sources inline using [N] notation matching the source numbers in the \
context. Every factual claim needs a citation.
3. Prefer code examples from the sources verbatim where they help.
4. Be concise. Aim for 3–8 sentences plus code if relevant. No filler.
5. Never invent imports, function signatures, or behaviour not present in the \
provided sources.
6. If the question has multiple parts and only some are covered by the sources, \
answer the covered parts with citations and explicitly note which parts you \
cannot answer from the provided sources.\
"""

# The {sources_block} placeholder is filled by build_user_prompt().
_USER_TEMPLATE = """\
Sources:
{sources_block}

Question: {question}

Answer (cite [N] inline):"""


# ---------------------------------------------------------------------------
# Chunk formatting
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " …"


def _chunk_header(chunk: RetrievedChunk) -> str:
    """Return the [N]-prefixed header line for a chunk.

    docs:  [N] (docs: https://fastapi.tiangolo.com/tutorial/middleware/)
    issue: [N] (issue #1234: https://github.com/.../issues/1234)
           Title: Background task with DB session
    """
    # This function does not receive N — callers embed N themselves.
    # Kept separate so tests can validate header logic independently.
    if chunk.source_type == "issue":
        number = chunk.source_url.rstrip("/").rsplit("/", 1)[-1]
        header = f"(issue #{number}: {chunk.source_url})"
        title = chunk.metadata.get("title", "").strip()
        if title:
            return f"{header}\nTitle: {title}"
        return header
    return f"(docs: {chunk.source_url})"


def format_chunk(chunk: RetrievedChunk, number: int) -> str:
    """Format a single retrieved chunk as a numbered source block.

    Args:
        chunk:  The retrieved chunk with text and metadata.
        number: 1-indexed position in the source list (matches [N] citations).

    Returns:
        A string block like:
            [1] (docs: https://fastapi.tiangolo.com/tutorial/middleware/)
            Add middleware using app.add_middleware()...
    """
    header = _chunk_header(chunk)
    body = _truncate(chunk.text)
    return f"[{number}] {header}\n{body}"


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format all chunks into the sources block, separated by blank lines.

    Chunks should already be sorted in the desired order (highest reranker
    score first).  Numbering is 1-indexed from the list order.
    """
    return "\n\n".join(format_chunk(c, i) for i, c in enumerate(chunks, start=1))


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Assemble the complete user message from a question and retrieved chunks.

    Args:
        question: The user's query string (already validated/trimmed upstream).
        chunks:   Retrieved chunks in desired citation order (highest score first).

    Returns:
        The fully assembled user message string ready to send to the LLM.
        If chunks is empty, the sources block will be empty — the LLM will
        produce REFUSAL_SIGNAL per its system prompt, but callers should
        short-circuit before reaching the LLM when chunks is empty.
    """
    sources_block = build_context_block(chunks) if chunks else "(no sources retrieved)"
    return _USER_TEMPLATE.format(sources_block=sources_block, question=question)
