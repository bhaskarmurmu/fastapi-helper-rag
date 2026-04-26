"""Parse FastAPI markdown docs into Document objects."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    text: str
    source_url: str
    source_type: str
    title: str | None
    section_path: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_MKDOCS_ANCHOR_RE = re.compile(r"\s*\{[^}]*\}\s*$")  # strip { #anchor } suffixes
_BASE_URL = "https://fastapi.tiangolo.com/"
_DOCS_ROOT = Path("docs") / "en" / "docs"


def _strip_frontmatter(text: str) -> str:
    return _FRONTMATTER_RE.sub("", text, count=1).lstrip()


def _extract_title(text: str) -> str | None:
    m = _H1_RE.search(text)
    if not m:
        return None
    raw = m.group(1).strip()
    return _MKDOCS_ANCHOR_RE.sub("", raw).strip() or None


def _to_url(rel: Path) -> str:
    parts = list(rel.parts)
    stem = parts[-1][: -len(".md")] if parts[-1].endswith(".md") else parts[-1]
    parts[-1] = stem
    if stem == "index":
        parts.pop()
    if parts:
        return _BASE_URL + "/".join(parts) + "/"
    return _BASE_URL


def _to_section_path(rel: Path) -> list[str]:
    dir_parts = [p.replace("-", " ").title() for p in rel.parts[:-1]]
    stem = rel.stem
    if stem != "index":
        dir_parts.append(stem.replace("-", " ").title())
    return dir_parts


def load_docs(repo_path: Path) -> list[Document]:
    """Walk docs/en/docs/**/*.md, parse each, return one Document per file."""
    docs_dir = repo_path / _DOCS_ROOT
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    documents: list[Document] = []
    for md_file in sorted(docs_dir.rglob("*.md")):
        rel = md_file.relative_to(docs_dir)
        raw = md_file.read_text(encoding="utf-8", errors="replace")
        text = _strip_frontmatter(raw)
        documents.append(
            Document(
                text=text,
                source_url=_to_url(rel),
                source_type="docs",
                title=_extract_title(text),
                section_path=_to_section_path(rel),
            )
        )

    return documents
