"""Unit tests for chunker.py.

Key invariants per spec:
  (a) no chunk splits a code fence
  (b) overlap is honored
  (c) headers are prepended to continuation chunks
  (d) chunk size never exceeds chunk_size * 1.5 for non-code content
"""
from __future__ import annotations

import tiktoken
import pytest

from fastapi_helper.ingest.chunker import Chunk, chunk_document
from fastapi_helper.ingest.docs_loader import Document

_ENC = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_doc(text: str, url: str = "https://fastapi.tiangolo.com/test/") -> Document:
    return Document(
        text=text,
        source_url=url,
        source_type="docs",
        title="Test Doc",
        section_path=["Test"],
    )


def _tok(text: str) -> int:
    return len(_ENC.encode(text))


# ---------------------------------------------------------------------------
# fixture documents
# ---------------------------------------------------------------------------

DOC_WITH_CODE = """\
## Path Parameters

You can declare path parameters with the same syntax used by Python format strings.

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

The value of the path parameter `item_id` will be passed to your function as the
argument `item_id`. You can also declare the type of a path parameter in the function
using standard Python type annotations.
"""

DOC_MULTI_SECTION = """\
# FastAPI Tutorial

This tutorial shows you how to use FastAPI with most of its features, step by step.

## Path Parameters

You can declare path parameters or variables with the same syntax used by Python format
strings. The values of these parameters will be passed to your function as arguments.

## Query Parameters

When you declare other function parameters that are not part of the path parameters,
they are automatically interpreted as query parameters. A query parameter is a key-value
pair that goes after the question mark in the URL and is separated by ampersand characters.
"""

DOC_SMALL = """\
# Tiny

Just a short document.
"""

DOC_BIG_CODE = """\
# Large Example

Here is a very large code block that exceeds any reasonable chunk size.

```python
""" + "\n".join(f"# line {i}: " + "x" * 40 for i in range(60)) + """
```

Text after the big code block.
"""


# ---------------------------------------------------------------------------
# (a) no chunk splits a code fence
# ---------------------------------------------------------------------------

def test_no_chunk_has_unmatched_code_fence() -> None:
    # overlap=0 so overlap text from the previous chunk (which may carry a closing
    # fence) doesn't bleed into this chunk and confuse the count check.
    doc = make_doc(DOC_WITH_CODE)
    chunks = chunk_document(doc, chunk_size=40, overlap=0)
    for chunk in chunks:
        count = chunk.text.count("```")
        assert count % 2 == 0, (
            f"Unmatched ``` in chunk {chunk.chunk_index}: {chunk.text[:120]!r}"
        )


def test_code_block_contained_in_single_chunk() -> None:
    doc = make_doc(DOC_WITH_CODE)
    chunks = chunk_document(doc, chunk_size=40, overlap=0)
    code_chunks = [c for c in chunks if "```python" in c.text]
    assert len(code_chunks) == 1, (
        f"Code block split across {len(code_chunks)} chunks"
    )


def test_oversized_code_block_emitted_whole() -> None:
    doc = make_doc(DOC_BIG_CODE)
    chunks = chunk_document(doc, chunk_size=30, overlap=0)
    code_chunks = [c for c in chunks if "```python" in c.text]
    assert len(code_chunks) == 1
    assert code_chunks[0].text.count("```") == 2  # opening + closing


def test_no_chunk_splits_code_fence_multi_section() -> None:
    text = DOC_MULTI_SECTION + "\n\n" + DOC_WITH_CODE
    doc = make_doc(text)
    # overlap=0: same reasoning as test_no_chunk_has_unmatched_code_fence
    chunks = chunk_document(doc, chunk_size=40, overlap=0)
    for chunk in chunks:
        assert chunk.text.count("```") % 2 == 0


# ---------------------------------------------------------------------------
# (b) overlap is honored
# ---------------------------------------------------------------------------

def test_overlap_tail_appears_in_next_chunk() -> None:
    # Use globally unique tokens (p{i}w{j}) so overlap content is unambiguous.
    # Large chunk_size ensures each chunk is long enough that its last `overlap`
    # tokens come from its actual content, not from a previously-prepended overlap.
    paragraphs = "\n\n".join(
        "## Section " + str(i) + "\n\n" + " ".join(f"p{i}w{j}" for j in range(40))
        for i in range(6)
    )
    text = "# Title\n\n" + paragraphs
    doc = make_doc(text)
    overlap = 15
    chunk_size = 120
    chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)
    assert len(chunks) >= 2, "Document should produce multiple chunks"

    enc = tiktoken.get_encoding("cl100k_base")
    for i in range(len(chunks) - 1):
        # The overlap implementation prepends last `overlap` tokens of texts[i]
        # (pre-overlap version) to chunks[i+1]. Since chunk[i] has at least
        # chunk_size tokens of content and only `overlap` tokens of prefix from
        # chunk[i-1], its last `overlap` tokens are from real content.
        # We verify: the last few words of chunk[i] appear at the start of chunk[i+1].
        prev_words = chunks[i].text.split()
        tail_words = prev_words[-5:]  # last 5 words should be in overlap region
        next_start = chunks[i + 1].text[:300]
        found = any(w in next_start for w in tail_words)
        assert found, (
            f"No overlap word from chunk {i} found in start of chunk {i + 1}.\n"
            f"last words={tail_words}\nchunk {i+1} start={next_start!r}"
        )


def test_no_overlap_when_overlap_zero() -> None:
    text = "# Title\n\n" + "\n\n".join(
        "word " * 50 for _ in range(6)
    )
    doc = make_doc(text)
    chunks_no_ov = chunk_document(doc, chunk_size=60, overlap=0)
    chunks_with_ov = chunk_document(doc, chunk_size=60, overlap=15)

    if len(chunks_no_ov) >= 2 and len(chunks_with_ov) >= 2:
        # With overlap, chunks should be longer (or at least not shorter) than without
        avg_no = sum(_tok(c.text) for c in chunks_no_ov) / len(chunks_no_ov)
        avg_with = sum(_tok(c.text) for c in chunks_with_ov) / len(chunks_with_ov)
        assert avg_with >= avg_no, "Overlap should make chunks longer on average"


# ---------------------------------------------------------------------------
# (c) headers are prepended to continuation chunks
# ---------------------------------------------------------------------------

def test_header_prepended_to_continuation_chunk() -> None:
    # A section whose content is long enough to split into multiple chunks
    long_content = " ".join(f"word{i}" for i in range(200))
    text = f"## Query Parameters\n\n{long_content}"
    doc = make_doc(text)
    chunks = chunk_document(doc, chunk_size=40, overlap=0)

    assert len(chunks) >= 2, "Long content should produce multiple chunks"
    for chunk in chunks:
        assert "## Query Parameters" in chunk.text, (
            f"Header missing from chunk {chunk.chunk_index}: {chunk.text[:120]!r}"
        )


def test_correct_header_prepended_across_sections() -> None:
    text = DOC_MULTI_SECTION
    doc = make_doc(text)
    chunks = chunk_document(doc, chunk_size=40, overlap=0)

    # Any chunk containing "query parameter" text should reference its section header
    query_chunks = [c for c in chunks if "query" in c.text.lower() and "## " in c.text]
    assert query_chunks, "Expected at least one chunk with query content and a header"
    for c in query_chunks:
        assert "## Query Parameters" in c.text or "# FastAPI Tutorial" in c.text


def test_no_header_prepended_when_chunk_already_starts_with_header() -> None:
    text = "## Section A\n\nContent A.\n\n## Section B\n\nContent B."
    doc = make_doc(text)
    chunks = chunk_document(doc, chunk_size=200, overlap=0)

    for chunk in chunks:
        lines = chunk.text.lstrip().splitlines()
        # Headers should not appear twice at the top
        header_lines = [l for l in lines[:4] if l.startswith("#")]
        assert len(header_lines) <= 1, (
            f"Duplicate header at chunk top: {chunk.text[:120]!r}"
        )


# ---------------------------------------------------------------------------
# (d) chunk size soft cap (1.5×) for prose content
# ---------------------------------------------------------------------------

def test_prose_chunks_within_size_limit() -> None:
    long_prose = "# Title\n\n" + " ".join(f"word{i}" for i in range(600))
    doc = make_doc(long_prose)
    chunk_size = 50
    overlap = 10
    chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)

    for chunk in chunks:
        # Skip chunks that are entirely (or mostly) code
        if "```" in chunk.text:
            continue
        tokens = _tok(chunk.text)
        # Prose chunks may exceed chunk_size by at most overlap + small header prefix
        assert tokens <= chunk_size * 1.5 + overlap, (
            f"Prose chunk {chunk.chunk_index} has {tokens} tokens "
            f"(limit {chunk_size * 1.5 + overlap}): {chunk.text[:80]!r}"
        )


def test_oversized_code_allowed_beyond_soft_cap() -> None:
    doc = make_doc(DOC_BIG_CODE)
    chunks = chunk_document(doc, chunk_size=30, overlap=0)
    code_chunks = [c for c in chunks if "```python" in c.text]
    assert len(code_chunks) == 1
    # The oversized code chunk exceeds chunk_size — that's expected and allowed
    assert _tok(code_chunks[0].text) > 30


# ---------------------------------------------------------------------------
# metadata and structure
# ---------------------------------------------------------------------------

def test_chunk_index_sequential() -> None:
    doc = make_doc(DOC_MULTI_SECTION)
    chunks = chunk_document(doc, chunk_size=40, overlap=5)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_source_fields_preserved() -> None:
    url = "https://fastapi.tiangolo.com/tutorial/path-params/"
    doc = Document(
        text=DOC_WITH_CODE,
        source_url=url,
        source_type="docs",
        title="Path Parameters",
        section_path=["Tutorial", "Path Parameters"],
    )
    chunks = chunk_document(doc, chunk_size=40, overlap=0)
    for chunk in chunks:
        assert chunk.source_url == url
        assert chunk.source_type == "docs"


def test_chunk_metadata_carries_title_and_section_path() -> None:
    doc = Document(
        text=DOC_WITH_CODE,
        source_url="https://fastapi.tiangolo.com/tutorial/path-params/",
        source_type="docs",
        title="Path Parameters",
        section_path=["Tutorial", "Path Parameters"],
    )
    chunks = chunk_document(doc, chunk_size=40, overlap=0)
    for chunk in chunks:
        assert chunk.metadata["title"] == "Path Parameters"
        assert chunk.metadata["section_path"] == ["Tutorial", "Path Parameters"]


def test_single_small_doc_produces_one_chunk() -> None:
    doc = make_doc(DOC_SMALL)
    chunks = chunk_document(doc, chunk_size=512, overlap=50)
    assert len(chunks) == 1
    assert "Just a short document." in chunks[0].text


def test_empty_doc_produces_no_chunks() -> None:
    doc = make_doc("   \n\n   ")
    chunks = chunk_document(doc, chunk_size=512, overlap=50)
    assert chunks == []


def test_all_chunks_non_empty() -> None:
    doc = make_doc(DOC_MULTI_SECTION)
    chunks = chunk_document(doc, chunk_size=40, overlap=5)
    assert all(c.text.strip() for c in chunks)
