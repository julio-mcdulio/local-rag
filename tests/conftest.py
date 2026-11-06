"""Shared test fixtures.

Uses a deterministic fake embedding provider so tests run offline without
downloading Hugging Face models or loading torch.
"""

from __future__ import annotations

import hashlib
import numpy as np

import rag.embeddings as emb
from rag.config import Config
from rag.embeddings import ProviderRegistry


class FakeProvider(emb.BaseProvider):
    """Deterministic pseudo-embeddings derived from the text bytes."""

    def __init__(self, name: str, category: str = "standard", dim: int = 16):
        super().__init__(name, category)
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        # Map digest bytes to finite floats (raw float32 reinterpretation can
        # produce inf/nan, which would poison L2 normalization).
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        v = np.frombuffer(digest[: self.dim], dtype=np.uint8).astype(np.float32) - 127.5
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def encode_tokens(self, texts: list[str]) -> list[np.ndarray]:
        # one pseudo token vector per whitespace word (min 1)
        out = []
        for t in texts:
            words = t.split() or [t]
            arr = np.stack([self._vec(w) for w in words])
            out.append(arr)
        return out

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)


class FakeLateInteractionProvider(FakeProvider):
    late_interaction = True


def make_registry(models: dict[str, str] | None = None) -> ProviderRegistry:
    """Registry over fake providers, keyed by the (fake) model names."""
    models = models or {
        "standard": "fake-standard",
        "llm-retrieval": "fake-jina",
        "late-interaction": "fake-colbert",
    }
    orig = emb.make_provider
    emb.make_provider = lambda name, cat: (
        FakeLateInteractionProvider(name, cat)
        if cat == "late-interaction"
        else FakeProvider(name, cat)
    )
    try:
        return ProviderRegistry(models)
    finally:
        emb.make_provider = orig


def make_config(data_dir: str) -> Config:
    cfg = Config()
    cfg.embedding_models = {
        "standard": "fake-standard",
        "llm-retrieval": "fake-jina",
        "late-interaction": "fake-colbert",
    }
    cfg.data_dir = data_dir
    cfg.db_path = data_dir + "/index.db"
    return cfg
