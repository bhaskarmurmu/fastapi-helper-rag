"""BGE-small-en-v1.5 wrapper. Batched, L2-normalized embeddings."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._model = SentenceTransformer(model_name)
        self.dim: int = self._model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        is_query: bool = False,
    ) -> np.ndarray:
        """Return L2-normalized embeddings, shape (len(texts), self.dim).

        is_query is accepted for interface symmetry but bge-small-en-v1.5
        does not require a query prefix — the same model handles both.
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
