"""Search engines: dense (embedding), late-interaction, BM25 and raw grep.

All results carry citation coordinates: file path, start line and start char
offset, plus the score.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Optional

from .chunker import word_tokens
from .embeddings import ProviderRegistry
from .store import Store


@dataclasses.dataclass
class Result:
    path: str
    start_line: int
    start_char: int
    text: str
    score: Optional[float]

    def citation(self) -> str:
        s = f"{self.path}:{self.start_line}:{self.start_char}"
        if self.score is not None:
            s += f"  score={self.score:.4f}"
        return s

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _fetch(store: Store, ids_scores: list[tuple[int, float]], limit: int) -> list[Result]:
    if not ids_scores:
        return []
    results: list[Result] = []
    for cid, score in ids_scores[:limit]:
        row = store.get_chunk(cid)
        if row is None:
            continue
        results.append(
            Result(
                path=row["path"],
                start_line=row["start_line"],
                start_char=row["start_char"],
                text=row["text"],
                score=float(score),
            )
        )
    return results


def search_embedding(store: Store, provider: ProviderRegistry, model_name: str,
                     query: str, k: int) -> list[Result]:
    prov = provider.get(model_name)
    qv = prov.encode_query(query)
    return _fetch(store, store.search_dense(model_name, qv, k), k)


def search_late_interaction(store: Store, provider: ProviderRegistry, model_name: str,
                            query: str, k: int, coarse: int = 200) -> list[Result]:
    """Two-stage ColBERT retrieval: coarse cosine rank on pooled vectors,
    then MaxSim re-rank on token-level vectors."""
    prov = provider.get(model_name)
    qv = prov.encode_query(query)
    coarse_ids = store.search_dense(model_name, qv, coarse)
    if not coarse_ids:
        return []
    tok = store.load_token_vectors(model_name, [cid for cid, _ in coarse_ids], prov.dim)
    qv = qv.reshape(-1, qv.shape[-1])
    scored: list[tuple[int, float]] = []
    for cid, _ in coarse_ids:
        doc = tok.get(cid)
        if doc is None or doc.size == 0:
            continue
        # MaxSim: sum over query tokens of max dot product with doc tokens.
        sims = doc @ qv.T  # (doc_tokens, query_tokens)
        score = float(sims.max(axis=0).sum())
        scored.append((cid, score))
    scored.sort(key=lambda kv: -kv[1])
    return _fetch(store, scored, k)


def search_bm25(store: Store, query: str, k: int) -> list[Result]:
    terms = [t.lower() for t in word_tokens(query)]
    return _fetch(store, store.bm25_search(terms, k), k)


def grep_files(store: Store, pattern: str, k: int,
               dirs: Optional[list[str]] = None) -> list[Result]:
    """Regex grep over the raw original files (full-text, per-file)."""
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        rx = re.compile(re.escape(pattern), re.IGNORECASE)
    if dirs:
        dir_set = {str(Path(d).expanduser().resolve()) for d in dirs}
    else:
        dir_set = None

    results: list[Result] = []
    for f in store.iter_files():
        path = f["path"]
        if dir_set is not None and not any(
            path == d or path.startswith(d.rstrip("/") + "/") for d in dir_set
        ):
            continue
        try:
            raw = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in rx.finditer(raw):
            line = raw[: m.start()].count("\n") + 1
            start_char = m.start()
            line_start = raw.rfind("\n", 0, m.start()) + 1
            line_end = raw.find("\n", m.end())
            if line_end == -1:
                line_end = len(raw)
            results.append(
                Result(
                    path=path,
                    start_line=line,
                    start_char=start_char,
                    text=raw[line_start:line_end].strip(),
                    score=None,
                )
            )
            if len(results) >= k:
                return results
    return results
