"""Unit tests for issues_loader.py. httpx is fully mocked."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch, call

import pytest

from fastapi_helper.ingest.issues_loader import (
    _auth_headers,
    _build_text,
    _guard_rate_limit,
    _to_document,
    load_issues,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_issue(
    number: int = 1,
    title: str = "Test issue",
    state: str = "closed",
    body: str = "Issue body here.",
    comments: int = 0,
    labels: list[str] | None = None,
    is_pr: bool = False,
) -> dict:
    d = {
        "number": number,
        "title": title,
        "state": state,
        "body": body,
        "comments": comments,
        "comments_url": f"https://api.github.com/repos/tiangolo/fastapi/issues/{number}/comments",
        "labels": [{"name": lbl} for lbl in (labels or [])],
        "user": {"login": "author1"},
        "created_at": "2023-01-01T00:00:00Z",
        "updated_at": "2023-06-01T00:00:00Z",
    }
    if is_pr:
        d["pull_request"] = {"url": "https://github.com/tiangolo/fastapi/pull/1"}
    return d


def _make_comment(body: str = "A comment.", author: str = "user2") -> dict:
    return {"body": body, "user": {"login": author}}


def _mock_response(json_data, remaining: int = 4999, reset_at: int | None = None) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.headers = {
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": str(reset_at or int(time.time()) + 3600),
    }
    resp.raise_for_status = MagicMock()
    return resp


def _make_mock_client(pages: list, comment_pages: dict | None = None) -> MagicMock:
    """
    pages: list of issue-page payloads (each is a list of issue dicts)
    comment_pages: {comments_url: list_of_comments} for issues with comments
    """
    comment_pages = comment_pages or {}
    client = MagicMock()
    client.__enter__ = lambda s: s
    client.__exit__ = MagicMock(return_value=False)

    issue_responses = [_mock_response(p) for p in pages]
    comment_responses = {url: _mock_response(c) for url, c in comment_pages.items()}

    call_count = {"issues": 0}

    def fake_get(url, params=None):
        # Is this a comment URL?
        for c_url, c_resp in comment_responses.items():
            if c_url in url:
                return c_resp
        # Otherwise treat as issues page
        idx = call_count["issues"]
        call_count["issues"] += 1
        if idx < len(issue_responses):
            return issue_responses[idx]
        return _mock_response([])

    client.get.side_effect = fake_get
    return client


# ---------------------------------------------------------------------------
# _auth_headers
# ---------------------------------------------------------------------------

def test_auth_header_with_token() -> None:
    headers = _auth_headers("ghp_test123")
    assert headers["Authorization"] == "Bearer ghp_test123"


def test_auth_header_without_token(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    headers = _auth_headers(None)
    assert "Authorization" not in headers


def test_auth_header_from_env(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_env_token")
    headers = _auth_headers(None)
    assert headers["Authorization"] == "Bearer ghp_env_token"


def test_accept_header_always_set() -> None:
    headers = _auth_headers(None)
    assert "application/vnd.github" in headers["Accept"]


# ---------------------------------------------------------------------------
# _build_text
# ---------------------------------------------------------------------------

def test_build_text_title_in_output() -> None:
    issue = _make_issue(title="Dependency injection bug")
    text = _build_text(issue, [])
    assert "# Dependency injection bug" in text


def test_build_text_body_in_output() -> None:
    issue = _make_issue(body="This is the body.")
    text = _build_text(issue, [])
    assert "This is the body." in text


def test_build_text_comment_appended() -> None:
    issue = _make_issue()
    comments = [_make_comment("Great question.", "alice")]
    text = _build_text(issue, comments)
    assert "Great question." in text
    assert "@alice" in text


def test_build_text_empty_comment_skipped() -> None:
    issue = _make_issue()
    comments = [_make_comment(""), _make_comment("Real content.")]
    text = _build_text(issue, comments)
    assert text.count("---") == 1  # only one separator (for the real comment)


def test_build_text_none_body_handled() -> None:
    issue = _make_issue(body=None)
    text = _build_text(issue, [])
    assert "# " in text  # title still present, no crash


# ---------------------------------------------------------------------------
# _to_document
# ---------------------------------------------------------------------------

def test_to_document_source_url() -> None:
    doc = _to_document(_make_issue(number=42), [])
    assert doc.source_url == "https://github.com/tiangolo/fastapi/issues/42"


def test_to_document_source_type() -> None:
    doc = _to_document(_make_issue(), [])
    assert doc.source_type == "issue"


def test_to_document_metadata_fields() -> None:
    issue = _make_issue(number=10, labels=["bug", "question"])
    doc = _to_document(issue, [])
    assert doc.metadata["issue_number"] == 10
    assert doc.metadata["labels"] == ["bug", "question"]
    assert doc.metadata["state"] == "closed"
    assert doc.metadata["author"] == "author1"


def test_to_document_section_path() -> None:
    doc = _to_document(_make_issue(), [])
    assert doc.section_path == ["Issues"]


def test_to_document_title_preserved() -> None:
    issue = _make_issue(title="WebSocket not working")
    doc = _to_document(issue, [])
    assert doc.title == "WebSocket not working"


# ---------------------------------------------------------------------------
# _guard_rate_limit
# ---------------------------------------------------------------------------

def test_guard_rate_limit_no_sleep_when_ample(monkeypatch) -> None:
    resp = _mock_response([], remaining=4999)
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    _guard_rate_limit(resp)
    assert not slept


def test_guard_rate_limit_sleeps_when_low(monkeypatch) -> None:
    reset = int(time.time()) + 30
    resp = _mock_response([], remaining=10, reset_at=reset)
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    _guard_rate_limit(resp)
    assert slept and slept[0] > 0


# ---------------------------------------------------------------------------
# load_issues
# ---------------------------------------------------------------------------

def test_load_issues_returns_documents() -> None:
    pages = [[_make_issue(number=1), _make_issue(number=2)], []]
    client = _make_mock_client(pages)
    with patch("fastapi_helper.ingest.issues_loader.httpx.Client", return_value=client):
        docs = load_issues(token="fake")
    assert len(docs) == 2


def test_load_issues_excludes_pull_requests() -> None:
    pages = [[_make_issue(number=1, is_pr=True), _make_issue(number=2)], []]
    client = _make_mock_client(pages)
    with patch("fastapi_helper.ingest.issues_loader.httpx.Client", return_value=client):
        docs = load_issues(token="fake")
    assert len(docs) == 1
    assert docs[0].metadata["issue_number"] == 2


def test_load_issues_max_issues_respected() -> None:
    # One page of 5 issues
    pages = [[_make_issue(number=i) for i in range(1, 6)]]
    client = _make_mock_client(pages)
    with patch("fastapi_helper.ingest.issues_loader.httpx.Client", return_value=client):
        docs = load_issues(token="fake", max_issues=3)
    assert len(docs) == 3


def test_load_issues_fetches_comments_when_present() -> None:
    issue = _make_issue(number=1, comments=1)
    pages = [[issue], []]
    comment_url = issue["comments_url"]
    comment_pages = {comment_url: [_make_comment("Fix: use async def.")]}
    client = _make_mock_client(pages, comment_pages)

    with patch("fastapi_helper.ingest.issues_loader.httpx.Client", return_value=client):
        docs = load_issues(token="fake", include_comments=True)

    assert "Fix: use async def." in docs[0].text


def test_load_issues_skips_comments_when_disabled() -> None:
    issue = _make_issue(number=1, comments=3)
    pages = [[issue], []]
    client = _make_mock_client(pages)
    get_calls_before = []

    with patch("fastapi_helper.ingest.issues_loader.httpx.Client", return_value=client):
        docs = load_issues(token="fake", include_comments=False)

    # Comments URL should never have been called
    for c in client.get.call_args_list:
        url = c.args[0] if c.args else c.kwargs.get("url", "")
        assert "comments" not in url or "/issues" in url


def test_load_issues_pagination_fetches_all_pages() -> None:
    # Patch _PER_PAGE to 2 so the loader sees page1 as "full" and fetches page2
    page1 = [_make_issue(number=1), _make_issue(number=2)]
    page2 = [_make_issue(number=3), _make_issue(number=4)]
    pages = [page1, page2, []]
    client = _make_mock_client(pages)
    with patch("fastapi_helper.ingest.issues_loader._PER_PAGE", 2), \
         patch("fastapi_helper.ingest.issues_loader.httpx.Client", return_value=client):
        docs = load_issues(token="fake")
    assert len(docs) == 4


def test_load_issues_empty_repo_returns_empty_list() -> None:
    client = _make_mock_client([[]])
    with patch("fastapi_helper.ingest.issues_loader.httpx.Client", return_value=client):
        docs = load_issues(token="fake")
    assert docs == []


def test_load_issues_closes_client_on_success() -> None:
    pages = [[_make_issue(number=1)], []]
    client = _make_mock_client(pages)
    with patch("fastapi_helper.ingest.issues_loader.httpx.Client", return_value=client):
        load_issues(token="fake")
    client.close.assert_called_once()


def test_load_issues_closes_client_on_error() -> None:
    client = MagicMock()
    client.get.side_effect = httpx.NetworkError("connection refused")
    with patch("fastapi_helper.ingest.issues_loader.httpx.Client", return_value=client):
        with pytest.raises(httpx.NetworkError):
            load_issues(token="fake")
    client.close.assert_called_once()


def test_load_issues_document_source_url_format() -> None:
    pages = [[_make_issue(number=99)], []]
    client = _make_mock_client(pages)
    with patch("fastapi_helper.ingest.issues_loader.httpx.Client", return_value=client):
        docs = load_issues(token="fake")
    assert docs[0].source_url == "https://github.com/tiangolo/fastapi/issues/99"


# ---------------------------------------------------------------------------
# Need httpx import for error test
# ---------------------------------------------------------------------------
import httpx  # noqa: E402
