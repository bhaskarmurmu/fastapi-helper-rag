"""Unit tests for retrieval/fusion.py.

Correctness of the RRF math is the primary concern — numerical examples are
worked by hand and checked against the implementation.
"""
from __future__ import annotations

import pytest

from fastapi_helper.retrieval.fusion import fuse
from fastapi_helper.retrieval.types import RetrievedChunk


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def chunk(url: str, idx: int = 0, score: float = 0.5, src: str = "docs") -> RetrievedChunk:
    return RetrievedChunk(
        text=f"text for {url} chunk {idx}",
        source_url=url,
        source_type=src,
        chunk_index=idx,
        score=score,
        metadata={"title": url},
    )


# Convenience aliases for shorter test bodies
A = chunk("https://fastapi.tiangolo.com/a/", idx=0)
B = chunk("https://fastapi.tiangolo.com/b/", idx=0)
C = chunk("https://fastapi.tiangolo.com/c/", idx=0)
D = chunk("https://fastapi.tiangolo.com/d/", idx=0)


# ---------------------------------------------------------------------------
# Known-value RRF math (worked by hand)
# ---------------------------------------------------------------------------

def test_rrf_single_item_single_list_k60() -> None:
    # rank 1 in one list: 1 / (60 + 1) = 1/61
    result = fuse([A], k=60)
    assert len(result) == 1
    assert result[0].score == pytest.approx(1 / 61, rel=1e-6)


def test_rrf_single_item_both_lists_rank1_k60() -> None:
    # A at rank 1 in both: 1/61 + 1/61 = 2/61
    result = fuse([A], [A], k=60)
    assert len(result) == 1
    assert result[0].score == pytest.approx(2 / 61, rel=1e-6)


def test_rrf_worked_example_k10() -> None:
    """Full worked example with k=10 for easy mental arithmetic.

    List 1: A(rank1), B(rank2), C(rank3)
    List 2: C(rank1), A(rank2), D(rank3)

    RRF scores (k=10):
      A: 1/11 + 1/12 = 0.09091 + 0.08333 = 0.17424
      C: 1/13 + 1/11 = 0.07692 + 0.09091 = 0.16783
      B: 1/12          = 0.08333
      D: 1/13          = 0.07692

    Expected order: A > C > B > D
    """
    list1 = [A, B, C]
    list2 = [C, A, D]
    result = fuse(list1, list2, k=10)

    assert [r.source_url for r in result] == [A.source_url, C.source_url, B.source_url, D.source_url]

    expected = {
        A.source_url: 1/11 + 1/12,
        C.source_url: 1/13 + 1/11,
        B.source_url: 1/12,
        D.source_url: 1/13,
    }
    for r in result:
        assert r.score == pytest.approx(expected[r.source_url], rel=1e-6)


def test_rrf_chunk_in_one_list_only() -> None:
    # A appears only in dense (rank 1), B only in sparse (rank 1).
    # Both should score 1/(k+1); tiebreaker is lexicographic source_url.
    result = fuse([A], [B], k=60)
    assert len(result) == 2
    assert result[0].score == pytest.approx(1 / 61, rel=1e-6)
    assert result[1].score == pytest.approx(1 / 61, rel=1e-6)


def test_rrf_chunk_in_both_lists_scores_higher_than_single_list() -> None:
    # A in both lists at rank 1 vs B in one list at rank 1.
    result = fuse([A, B], [A], k=60)
    a_result = next(r for r in result if r.source_url == A.source_url)
    b_result = next(r for r in result if r.source_url == B.source_url)
    assert a_result.score > b_result.score


def test_rrf_rank_order_matters_within_list() -> None:
    # A at rank 1, B at rank 3 — A scores higher from one list with k=10.
    # Use a unique filler chunk at rank 2 so no key collision with B.
    filler = chunk("https://fastapi.tiangolo.com/filler/", idx=99)
    result = fuse([A, filler, B], k=10)
    scores = {(r.source_url, r.chunk_index): r.score for r in result}
    a_score = scores[(A.source_url, A.chunk_index)]
    b_score = scores[(B.source_url, B.chunk_index)]
    # rank 1: 1/11 ≈ 0.0909; rank 3: 1/13 ≈ 0.0769
    assert a_score == pytest.approx(1 / 11, rel=1e-6)
    assert b_score == pytest.approx(1 / 13, rel=1e-6)
    assert a_score > b_score


def test_rrf_different_ranks_in_two_lists() -> None:
    """A at rank 3 in list1, rank 1 in list2 — verify accumulation is correct."""
    list1 = [B, C, A]   # A is rank 3
    list2 = [A, B, C]   # A is rank 1
    result = fuse(list1, list2, k=10)
    scores = {r.source_url: r.score for r in result}

    # A: 1/13 + 1/11
    # B: 1/11 + 1/12
    # C: 1/12 + 1/13
    assert scores[A.source_url] == pytest.approx(1/13 + 1/11, rel=1e-6)
    assert scores[B.source_url] == pytest.approx(1/11 + 1/12, rel=1e-6)
    assert scores[C.source_url] == pytest.approx(1/12 + 1/13, rel=1e-6)
    # B scores highest (rank 1 + rank 2 > rank 1 + rank 3 > rank 2 + rank 3)
    assert scores[B.source_url] > scores[A.source_url] > scores[C.source_url]


# ---------------------------------------------------------------------------
# Empty / degenerate inputs
# ---------------------------------------------------------------------------

def test_fuse_no_lists_returns_empty() -> None:
    assert fuse() == []


def test_fuse_single_empty_list_returns_empty() -> None:
    assert fuse([]) == []


def test_fuse_both_empty_returns_empty() -> None:
    assert fuse([], []) == []


def test_fuse_one_empty_one_nonempty_returns_nonempty() -> None:
    result = fuse([], [A, B, C])
    assert len(result) == 3
    # Scores come from only the non-empty list
    assert result[0].score == pytest.approx(1 / 61, rel=1e-6)  # rank 1
    assert result[1].score == pytest.approx(1 / 62, rel=1e-6)  # rank 2
    assert result[2].score == pytest.approx(1 / 63, rel=1e-6)  # rank 3


def test_fuse_single_list_assigns_rank_based_scores() -> None:
    result = fuse([A, B], k=60)
    assert result[0].score == pytest.approx(1 / 61, rel=1e-6)
    assert result[1].score == pytest.approx(1 / 62, rel=1e-6)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_fuse_deduplicates_same_chunk_across_lists() -> None:
    result = fuse([A, B], [A, C])
    urls = [r.source_url for r in result]
    assert urls.count(A.source_url) == 1


def test_fuse_deduplication_key_is_url_and_chunk_index() -> None:
    # Same URL, different chunk_index → NOT duplicates
    c0 = chunk("https://fastapi.tiangolo.com/x/", idx=0)
    c1 = chunk("https://fastapi.tiangolo.com/x/", idx=1)
    result = fuse([c0], [c1])
    assert len(result) == 2


def test_fuse_deduplication_same_url_same_index_is_one_chunk() -> None:
    c0a = chunk("https://fastapi.tiangolo.com/x/", idx=0, score=0.9)
    c0b = chunk("https://fastapi.tiangolo.com/x/", idx=0, score=0.5)
    result = fuse([c0a], [c0b])
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Canonical chunk selection (first-seen text/metadata preserved)
# ---------------------------------------------------------------------------

def test_fuse_preserves_text_from_first_seen_chunk() -> None:
    c_dense = RetrievedChunk(
        text="dense text", source_url="https://a/", source_type="docs",
        chunk_index=0, score=0.9,
    )
    c_sparse = RetrievedChunk(
        text="sparse text", source_url="https://a/", source_type="docs",
        chunk_index=0, score=0.05,
    )
    result = fuse([c_dense], [c_sparse])
    assert result[0].text == "dense text"


# ---------------------------------------------------------------------------
# top_k
# ---------------------------------------------------------------------------

def test_fuse_top_k_truncates_output() -> None:
    result = fuse([A, B, C, D], top_k=2)
    assert len(result) == 2


def test_fuse_top_k_none_returns_all() -> None:
    result = fuse([A, B, C, D], top_k=None)
    assert len(result) == 4


def test_fuse_top_k_larger_than_results_returns_all() -> None:
    result = fuse([A, B], top_k=100)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Tied RRF scores — deterministic ordering
# ---------------------------------------------------------------------------

def test_tied_scores_ordered_deterministically() -> None:
    # A and B each appear in only one list at rank 1 → equal RRF score.
    # Tiebreaker: source_url lexicographic ascending.
    first_url = "https://fastapi.tiangolo.com/a/"
    second_url = "https://fastapi.tiangolo.com/b/"
    ca = chunk(first_url)
    cb = chunk(second_url)
    result = fuse([ca], [cb], k=60)
    assert len(result) == 2
    assert result[0].score == pytest.approx(result[1].score, rel=1e-9)
    assert result[0].source_url == first_url
    assert result[1].source_url == second_url


def test_tied_scores_stable_across_input_order() -> None:
    # Reversing the input order should not change the output (tiebreaker is content, not input order)
    first_url = "https://fastapi.tiangolo.com/a/"
    second_url = "https://fastapi.tiangolo.com/b/"
    ca = chunk(first_url)
    cb = chunk(second_url)
    result1 = fuse([ca], [cb], k=60)
    result2 = fuse([cb], [ca], k=60)
    assert [r.source_url for r in result1] == [r.source_url for r in result2]


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

def test_output_is_sorted_descending_by_score() -> None:
    result = fuse([A, B, C], [C, D], k=10)
    scores = [r.score for r in result]
    assert scores == sorted(scores, reverse=True)


def test_output_score_replaces_input_score() -> None:
    # Input score of A is 0.9; output score should be the RRF value, not 0.9
    result = fuse([A], k=60)
    assert result[0].score != pytest.approx(0.9)
    assert result[0].score == pytest.approx(1 / 61, rel=1e-6)


def test_output_chunks_are_retrieved_chunk_instances() -> None:
    result = fuse([A, B], [B, C])
    assert all(isinstance(r, RetrievedChunk) for r in result)


def test_output_non_score_fields_unchanged() -> None:
    result = fuse([A])
    r = result[0]
    assert r.source_url == A.source_url
    assert r.chunk_index == A.chunk_index
    assert r.text == A.text
    assert r.metadata == A.metadata
