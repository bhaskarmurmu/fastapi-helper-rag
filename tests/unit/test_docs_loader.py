"""Unit tests for docs_loader.py."""
from pathlib import Path

import pytest

from fastapi_helper.ingest.docs_loader import Document, load_docs

FIXTURE_REPO = Path(__file__).parent.parent / "fixtures" / "fastapi_docs_sample"


@pytest.fixture(scope="module")
def docs() -> list[Document]:
    return load_docs(FIXTURE_REPO)


@pytest.fixture(scope="module")
def by_url(docs: list[Document]) -> dict[str, Document]:
    return {d.source_url: d for d in docs}


# --- count & types ---

def test_returns_four_documents(docs: list[Document]) -> None:
    assert len(docs) == 4


def test_all_have_source_type_docs(docs: list[Document]) -> None:
    assert all(d.source_type == "docs" for d in docs)


def test_all_documents_are_document_instances(docs: list[Document]) -> None:
    assert all(isinstance(d, Document) for d in docs)


# --- URL construction ---

def test_root_index_url(by_url: dict[str, Document]) -> None:
    assert "https://fastapi.tiangolo.com/" in by_url


def test_tutorial_index_url(by_url: dict[str, Document]) -> None:
    assert "https://fastapi.tiangolo.com/tutorial/" in by_url


def test_tutorial_page_url(by_url: dict[str, Document]) -> None:
    assert "https://fastapi.tiangolo.com/tutorial/first-steps/" in by_url


def test_advanced_page_url(by_url: dict[str, Document]) -> None:
    assert "https://fastapi.tiangolo.com/advanced/middleware/" in by_url


# --- section_path ---

def test_section_path_root(by_url: dict[str, Document]) -> None:
    assert by_url["https://fastapi.tiangolo.com/"].section_path == []


def test_section_path_tutorial_index(by_url: dict[str, Document]) -> None:
    assert by_url["https://fastapi.tiangolo.com/tutorial/"].section_path == ["Tutorial"]


def test_section_path_tutorial_page(by_url: dict[str, Document]) -> None:
    assert by_url["https://fastapi.tiangolo.com/tutorial/first-steps/"].section_path == [
        "Tutorial",
        "First Steps",
    ]


def test_section_path_advanced(by_url: dict[str, Document]) -> None:
    assert by_url["https://fastapi.tiangolo.com/advanced/middleware/"].section_path == [
        "Advanced",
        "Middleware",
    ]


# --- title extraction ---

def test_title_root(by_url: dict[str, Document]) -> None:
    assert by_url["https://fastapi.tiangolo.com/"].title == "FastAPI"


def test_title_first_steps(by_url: dict[str, Document]) -> None:
    assert by_url["https://fastapi.tiangolo.com/tutorial/first-steps/"].title == "First Steps"


def test_title_advanced_middleware(by_url: dict[str, Document]) -> None:
    assert by_url["https://fastapi.tiangolo.com/advanced/middleware/"].title == "Advanced Middleware"


# --- front-matter stripping ---

def test_frontmatter_stripped_from_root(by_url: dict[str, Document]) -> None:
    doc = by_url["https://fastapi.tiangolo.com/"]
    assert not doc.text.startswith("---")


def test_frontmatter_stripped_from_advanced(by_url: dict[str, Document]) -> None:
    doc = by_url["https://fastapi.tiangolo.com/advanced/middleware/"]
    assert not doc.text.startswith("---")


def test_no_document_text_starts_with_frontmatter(docs: list[Document]) -> None:
    for d in docs:
        assert not d.text.startswith("---"), f"Front matter not stripped: {d.source_url}"


# --- text content ---

def test_text_non_empty(docs: list[Document]) -> None:
    assert all(d.text.strip() for d in docs)


def test_text_contains_h1(docs: list[Document]) -> None:
    for d in docs:
        assert "# " in d.text, f"No heading found in {d.source_url}"


def test_code_block_preserved_in_first_steps(by_url: dict[str, Document]) -> None:
    doc = by_url["https://fastapi.tiangolo.com/tutorial/first-steps/"]
    assert "```python" in doc.text
    assert "FastAPI()" in doc.text


# --- error handling ---

def test_raises_on_missing_repo() -> None:
    with pytest.raises(FileNotFoundError):
        load_docs(Path("/nonexistent/fastapi/repo"))
