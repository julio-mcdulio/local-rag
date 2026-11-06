"""Embedding model providers (local, offline after first warm-up).

Each configured embedding model gets its own provider. Providers are
loaded lazily from Hugging Face model names via sentence-transformers and
cached in memory. Dense models produce fixed-size vectors; late-interaction
models (ColBERT) produce per-token vectors plus a mean-pooled vector used
for coarse ranking.
"""

from __future__ import annotations

import re
import threading
from typing import Optional

import numpy as np


def slugify(name: str) -> str:
    """Safe sqlite table-name fragment for a model name."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return s[:60]


class BaseProvider:
    late_interaction = False

    def __init__(self, model_name: str, category: str):
        self.model_name = model_name
        self.category = category  # "standard" | "llm-retrieval" | "late-interaction"
        self.dim: int = 0
        self._model = None
        self._lock = threading.Lock()

    # -- lazy load -----------------------------------------------------------
    def _ensure(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    model = SentenceTransformer(
                        self.model_name,
                        trust_remote_code=True,
                    )
                    # Warm up / determine dimensionality.
                    self._model = model
        return self._model

    # -- encoding -------------------------------------------------------------
    def encode_docs(self, texts: list[str]) -> np.ndarray:
        """Return (n, dim) float32 vectors for documents."""
        model = self._ensure()
        if not texts:
            return np.zeros((0, self._dim_or_infer(model)), dtype=np.float32)
        out = model.encode(
            texts,
            convert_to_numpy=True,
            convert_to_tensor=True,
            batch_size=32,
            show_progress_bar=False,
        )
        arr = np.asarray(out, dtype=np.float32)
        if arr.ndim == 3:  # token-level output
            arr = arr.mean(axis=1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self.dim = int(arr.shape[-1])
        return arr

    def encode_tokens(self, texts: list[str]) -> list[np.ndarray]:
        """Token-level embeddings; returns list of (seq, dim) arrays.

        Only meaningful for late-interaction models.
        """
        model = self._ensure()
        out = []
        for t in texts:
            v = model.encode(
                [t],
                convert_to_numpy=True,
                convert_to_tensor=True,
                batch_size=1,
                show_progress_bar=False,
            )
            arr = np.asarray(v, dtype=np.float32)
            if arr.ndim == 2:  # already (seq, dim)
                arr = arr[0]
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1)
            self.dim = int(arr.shape[-1])
            out.append(arr)
        return out

    def encode_query(self, text: str) -> np.ndarray:
        """Query vector: (dim) for dense, (seq, dim) for late-interaction."""
        model = self._ensure()
        v = model.encode(
            [text],
            convert_to_numpy=True,
            convert_to_tensor=True,
            batch_size=1,
            show_progress_bar=False,
        )
        arr = np.asarray(v, dtype=np.float32)
        if self.late_interaction:
            if arr.ndim == 3:
                arr = arr[0]
            elif arr.ndim == 1:
                arr = arr.reshape(1, -1)
            self.dim = int(arr.shape[-1])
        else:
            if arr.ndim == 3:
                arr = arr.mean(axis=1)[0]
            elif arr.ndim == 1:
                arr = arr.reshape(-1)
            self.dim = int(arr.shape[-1])
        return arr

    def _dim_or_infer(self, model) -> int:
        try:
            d = int(model.get_sentence_embedding_dimension())
            return d
        except Exception:
            return 0


class LateInteractionProvider(BaseProvider):
    late_interaction = True


def make_provider(model_name: str, category: str) -> BaseProvider:
    if category == "late-interaction":
        return LateInteractionProvider(model_name, category)
    return BaseProvider(model_name, category)


class ProviderRegistry:
    """One provider per configured embedding model (cached)."""

    def __init__(self, embedding_models: dict[str, str]):
        self._provs: dict[str, BaseProvider] = {}
        for cat, name in embedding_models.items():
            self._provs[name] = make_provider(name, cat)

    def get(self, model_name: str) -> BaseProvider:
        return self._provs[model_name]

    def all(self) -> list[BaseProvider]:
        return list(self._provs.values())
