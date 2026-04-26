"""Markdown-aware chunker. Code fences are atomic; headers prepend each chunk."""
from __future__ import annotations

import logging
import re
from typing import Any

import tiktoken
from pydantic import BaseModel, Field

from fastapi_helper.ingest.docs_loader import Document

log = logging.getLogger(__name__)

_ENC = tiktoken.get_encoding("cl100k_base")
_CODE_FENCE_RE = re.compile(r"(```[^\n]*\n.*?```)", re.DOTALL)
_HEADER_RE = re.compile(r"^#{1,6} .+", re.MULTILINE)
_STARTS_WITH_HEADER = re.compile(r"^#{1,6} ")
_SEPS = ["\n\n", "\n", ". ", " "]


class Chunk(BaseModel):
    text: str
    source_url: str
    source_type: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


def _tok(text: str) -> int:
    return len(_ENC.encode(text))


def _split_text(text: str, limit: int) -> list[str]:
    """Recursively split text into pieces each ≤ limit tokens."""
    if _tok(text) <= limit:
        return [text] if text.strip() else []
    for sep in _SEPS:
        if sep not in text:
            continue
        pieces = text.split(sep)
        result: list[str] = []
        buf = ""
        for piece in pieces:
            candidate = (buf + sep + piece) if buf else piece
            if _tok(candidate) <= limit:
                buf = candidate
            else:
                if buf:
                    result.append(buf)
                if _tok(piece) > limit:
                    result.extend(_split_text(piece, limit))
                    buf = ""
                else:
                    buf = piece
        if buf:
            result.append(buf)
        return [r for r in result if r.strip()]
    return [text]  # no separator found; emit as-is


def _last_header(text: str) -> str | None:
    headers = _HEADER_RE.findall(text)
    return headers[-1] if headers else None


def chunk_document(
    doc: Document,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[Chunk]:
    """Split a Document into overlapping Chunks. Code fences are never split."""
    # Phase 1: separate code fences from prose
    segments: list[tuple[str, str]] = []
    for i, part in enumerate(_CODE_FENCE_RE.split(doc.text)):
        if part:
            segments.append(("code" if i % 2 == 1 else "text", part))

    # Phase 2: split prose into fine pieces ≤ chunk_size; code stays atomic
    fine: list[tuple[str, str]] = []
    for kind, content in segments:
        if kind == "code":
            fine.append(("code", content))
        else:
            for piece in _split_text(content, chunk_size):
                fine.append(("text", piece))

    # Phase 3: greedy accumulation with header context tracking
    raw: list[tuple[str, str | None]] = []  # (content, header_at_chunk_start)
    current_header: str | None = None
    buf = ""
    buf_tokens = 0
    buf_header: str | None = None

    for kind, content in fine:
        if kind == "text":
            h = _last_header(content)
            if h:
                current_header = h

        content_tokens = _tok(content)

        if kind == "code" and content_tokens > chunk_size:
            if buf.strip():
                raw.append((buf, buf_header))
                buf = ""
                buf_tokens = 0
            log.warning(
                "Oversized code block (%d tokens) emitted as single chunk in %s",
                content_tokens,
                doc.source_url,
            )
            raw.append((content, current_header))
            buf_header = None
            continue

        sep_tokens = 2 if buf else 0  # "\n\n" ≈ 2 tokens
        candidate_tokens = buf_tokens + sep_tokens + content_tokens

        if buf and candidate_tokens > chunk_size:
            raw.append((buf, buf_header))
            buf = ""
            buf_tokens = 0
            buf_header = current_header

        if not buf:
            buf_header = current_header

        sep = "\n\n" if buf else ""
        buf = (buf + sep + content) if buf else content
        buf_tokens = _tok(buf)

    if buf.strip():
        raw.append((buf, buf_header))

    # Phase 4: prepend header context to chunks that don't already open with one
    texts: list[str] = []
    for content, header in raw:
        if header and not _STARTS_WITH_HEADER.match(content.lstrip()):
            texts.append(header + "\n\n" + content)
        else:
            texts.append(content)

    # Phase 5: apply token overlap — last `overlap` tokens of chunk N prepend chunk N+1
    if overlap > 0 and len(texts) > 1:
        with_overlap: list[str] = [texts[0]]
        for i in range(1, len(texts)):
            prev_toks = _ENC.encode(texts[i - 1])
            if len(prev_toks) > overlap:
                tail = _ENC.decode(prev_toks[-overlap:])
                with_overlap.append(tail + "\n\n" + texts[i])
            else:
                with_overlap.append(texts[i])
        texts = with_overlap

    # Phase 6: build Chunk objects
    meta: dict[str, Any] = {
        "title": doc.title,
        "section_path": doc.section_path,
        **doc.metadata,
    }
    valid = [t for t in texts if t.strip()]
    return [
        Chunk(
            text=t,
            source_url=doc.source_url,
            source_type=doc.source_type,
            chunk_index=i,
            metadata=meta,
        )
        for i, t in enumerate(valid)
    ]
