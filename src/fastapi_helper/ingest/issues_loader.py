"""Load closed GitHub issues + comments into Document objects."""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from fastapi_helper.ingest.docs_loader import Document

log = logging.getLogger(__name__)

_API_BASE = "https://api.github.com"
_PER_PAGE = 100
_RATE_LIMIT_BUFFER = 50  # sleep when remaining drops below this


def _auth_headers(token: str | None) -> dict[str, str]:
    t = token or os.environ.get("GITHUB_TOKEN", "")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if t:
        headers["Authorization"] = f"Bearer {t}"
    return headers


def _guard_rate_limit(resp: httpx.Response) -> None:
    """Sleep if approaching GitHub's rate limit."""
    remaining = int(resp.headers.get("X-RateLimit-Remaining", 999))
    reset_at = int(resp.headers.get("X-RateLimit-Reset", 0))
    if remaining < _RATE_LIMIT_BUFFER:
        wait = max(0, reset_at - int(time.time())) + 5
        log.warning(
            "Rate limit low (%d remaining). Sleeping %ds until reset.", remaining, wait
        )
        time.sleep(wait)


def _fetch_comments(
    client: httpx.Client, comments_url: str, n_comments: int
) -> list[dict[str, Any]]:
    """Fetch all comment pages for one issue."""
    comments: list[dict[str, Any]] = []
    page = 1
    while True:
        resp = client.get(
            comments_url, params={"per_page": _PER_PAGE, "page": page}
        )
        resp.raise_for_status()
        _guard_rate_limit(resp)
        batch = resp.json()
        if not batch:
            break
        comments.extend(batch)
        if len(batch) < _PER_PAGE:
            break
        page += 1
    return comments


def _build_text(issue: dict[str, Any], comments: list[dict[str, Any]]) -> str:
    parts: list[str] = [f"# {issue['title']}\n\n{issue.get('body') or ''}"]
    for c in comments:
        body = (c.get("body") or "").strip()
        if not body:
            continue
        author = c["user"]["login"] if c.get("user") else "unknown"
        parts.append(f"**@{author}:**\n\n{body}")
    return "\n\n---\n\n".join(parts)


def _to_document(issue: dict[str, Any], comments: list[dict[str, Any]]) -> Document:
    number: int = issue["number"]
    labels: list[str] = [lbl["name"] for lbl in issue.get("labels", [])]
    return Document(
        text=_build_text(issue, comments),
        source_url=f"https://github.com/tiangolo/fastapi/issues/{number}",
        source_type="issue",
        title=issue["title"],
        section_path=["Issues"],
        metadata={
            "issue_number": number,
            "state": issue["state"],
            "labels": labels,
            "author": issue["user"]["login"] if issue.get("user") else None,
            "created_at": issue["created_at"],
            "updated_at": issue["updated_at"],
            "comment_count": issue.get("comments", 0),
        },
    )


def load_issues(
    repo: str = "tiangolo/fastapi",
    *,
    state: str = "closed",
    max_issues: int | None = None,
    since: str | None = None,
    include_comments: bool = True,
    token: str | None = None,
) -> list[Document]:
    """Fetch GitHub issues (excluding PRs) and return one Document per issue.

    Args:
        repo: owner/name, e.g. "tiangolo/fastapi"
        state: "closed", "open", or "all"
        max_issues: stop after this many issues (useful for sampling)
        since: ISO 8601 datetime; only issues updated at or after this time
        include_comments: whether to fetch and concatenate comment bodies
        token: GitHub PAT; falls back to GITHUB_TOKEN env var
    """
    client = httpx.Client(
        headers=_auth_headers(token),
        timeout=30.0,
        follow_redirects=True,
    )
    documents: list[Document] = []
    page = 1

    try:
        while True:
            params: dict[str, Any] = {
                "state": state,
                "per_page": _PER_PAGE,
                "page": page,
                "sort": "created",
                "direction": "asc",
            }
            if since:
                params["since"] = since

            resp = client.get(f"{_API_BASE}/repos/{repo}/issues", params=params)
            resp.raise_for_status()
            _guard_rate_limit(resp)

            batch = resp.json()
            if not batch:
                break

            for issue in batch:
                if "pull_request" in issue:
                    continue

                comments: list[dict[str, Any]] = []
                if include_comments and issue.get("comments", 0) > 0:
                    comments = _fetch_comments(
                        client, issue["comments_url"], issue["comments"]
                    )

                documents.append(_to_document(issue, comments))
                log.debug(
                    "Loaded issue #%d (%d comments)", issue["number"], len(comments)
                )

                if max_issues is not None and len(documents) >= max_issues:
                    log.info("Reached max_issues=%d — stopping early", max_issues)
                    return documents

            log.info("Page %d — %d issues loaded so far", page, len(documents))

            if len(batch) < _PER_PAGE:
                break
            page += 1
    finally:
        client.close()

    log.info("Done — %d issues loaded from %s", len(documents), repo)
    return documents
