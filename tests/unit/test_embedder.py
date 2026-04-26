"""Unit tests for embedder.py. SentenceTransformer is mocked — no model download."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fastapi_helper.ingest.embedder import Embedder


def _make_embedder(dim: int = 384) -> tuple[Embedder, MagicMock]:
    """Return an Embedder backed by a mock model."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = dim

    def fake_encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False):
        rng = np.random.default_rng(seed=42)
        raw = rng.standard_normal((len(texts), dim)).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            return raw / norms
        return raw

    mock_model.encode.side_effect = fake_encode

    with patch("fastapi_helper.ingest.embedder.SentenceTransformer", return_value=mock_model):
        embedder = Embedder()

    return embedder, mock_model


# ---------------------------------------------------------------------------
# shape and dtype
# ---------------------------------------------------------------------------

def test_output_shape_single() -> None:
    embedder, _ = _make_embedder()
    result = embedder.encode(["hello world"])
    assert result.shape == (1, 384)


def test_output_shape_batch() -> None:
    embedder, _ = _make_embedder()
    texts = ["text one", "text two", "text three"]
    result = embedder.encode(texts)
    assert result.shape == (3, 384)


def test_output_dtype_float32() -> None:
    embedder, _ = _make_embedder()
    result = embedder.encode(["check dtype"])
    assert result.dtype == np.float32


def test_dim_attribute_matches_model() -> None:
    embedder, _ = _make_embedder(dim=384)
    assert embedder.dim == 384


# ---------------------------------------------------------------------------
# empty input
# ---------------------------------------------------------------------------

def test_empty_input_returns_empty_array() -> None:
    embedder, mock_model = _make_embedder()
    result = embedder.encode([])
    assert result.shape == (0, 384)
    mock_model.encode.assert_not_called()


def test_empty_input_dtype() -> None:
    embedder, _ = _make_embedder()
    result = embedder.encode([])
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# L2 normalization
# ---------------------------------------------------------------------------

def test_embeddings_are_l2_normalized() -> None:
    embedder, _ = _make_embedder()
    texts = [f"sentence {i}" for i in range(8)]
    result = embedder.encode(texts)
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# is_query flag (interface symmetry — does not change behavior)
# ---------------------------------------------------------------------------

def test_is_query_true_does_not_raise() -> None:
    embedder, _ = _make_embedder()
    result = embedder.encode(["a query"], is_query=True)
    assert result.shape == (1, 384)


def test_is_query_false_same_call_signature() -> None:
    embedder, _ = _make_embedder()
    r1 = embedder.encode(["same text"], is_query=False)
    r2 = embedder.encode(["same text"], is_query=True)
    assert r1.shape == r2.shape


# ---------------------------------------------------------------------------
# batch_size forwarded to underlying model
# ---------------------------------------------------------------------------

def test_batch_size_forwarded() -> None:
    embedder, mock_model = _make_embedder()
    embedder.encode(["a", "b"], batch_size=16)
    call_kwargs = mock_model.encode.call_args
    assert call_kwargs.kwargs.get("batch_size") == 16 or call_kwargs.args[1] == 16


# ---------------------------------------------------------------------------
# progress bar threshold
# ---------------------------------------------------------------------------

def test_no_progress_bar_for_small_input() -> None:
    embedder, mock_model = _make_embedder()
    embedder.encode(["x"] * 50)
    call_kwargs = mock_model.encode.call_args.kwargs
    assert call_kwargs["show_progress_bar"] is False


def test_progress_bar_for_large_input() -> None:
    embedder, mock_model = _make_embedder()
    embedder.encode(["x"] * 101)
    call_kwargs = mock_model.encode.call_args.kwargs
    assert call_kwargs["show_progress_bar"] is True


# ---------------------------------------------------------------------------
# normalize_embeddings always True
# ---------------------------------------------------------------------------

def test_normalize_embeddings_always_true() -> None:
    embedder, mock_model = _make_embedder()
    embedder.encode(["normalize check"])
    call_kwargs = mock_model.encode.call_args.kwargs
    assert call_kwargs["normalize_embeddings"] is True
