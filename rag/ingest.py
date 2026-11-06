"""Per-file ingestion pipeline.

extract -> chunk (structure-aware) -> precompute dense / late-interaction
vectors and BM25 postings -> store. One pass per file; embeddings and the
BM25 index are precomputed at ingestion time, one vector index per embedding
model.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunker import chunk_file, count_tokens, word_tokens
from .config import Config
from .embeddings import BaseProvider, ProviderRegistry
from .store import Store
from .text_extract import OcrClient, extract_text, file_kind

log = logging.getLogger("rag.ingest")


@dataclass
class IngestResult:
    path: str
    chunks: int
    skipped: bool = False
    error: str = ""


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ingest_file(
    path: str,
    store: Store,
    config: Config,
    providers: ProviderRegistry,
    ocr: OcrClient,
    force: bool = False,
) -> IngestResult:
    p = Path(path)
    try:
        mtime = p.stat().st_mtime
        sha = _file_hash(p)
        existing = store.conn.execute(
            "SELECT sha256 FROM files WHERE path = ?", (str(p),)
        ).fetchone()
        if existing and existing["sha256"] == sha and not force:
            return IngestResult(str(p), 0, skipped=True)

        text = extract_text(p, ocr)
        kind = file_kind(p)
        chunks = chunk_file(str(p), text, kind)
        if not chunks:
            # Still remove stale data for an emptied file.
            for row in store.chunks_for_path(str(p)):
                store.remove_chunk(row["id"])
            store.remove_file(str(p))
            store.record_file(str(p), sha, mtime)
            return IngestResult(str(p), 0)

        # Remove old chunks for this file (per-file granularity).
        for row in store.chunks_for_path(str(p)):
            store.remove_chunk(row["id"])

        texts = [c.text for c in chunks]
        # Precompute vectors per model (one index per embedding model).
        model_dense: dict[str, np.ndarray] = {}
        model_tok: dict[str, list] = {}
        for prov in providers.all():
            if prov.late_interaction:
                toks = prov.encode_tokens(texts)
                model_tok[prov.model_name] = toks
                pooled = [t.mean(axis=0) for t in toks]
                model_dense[prov.model_name] = np.stack(pooled) if pooled else np.zeros((0,))
            else:
                model_dense[prov.model_name] = prov.encode_docs(texts)
                model_tok[prov.model_name] = None

        for i, ch in enumerate(chunks):
            toks = [t.lower() for t in word_tokens(ch.text)]
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            chunk_id = store.insert_chunk(
                str(p), ch.start_line, ch.start_char, ch.text, ch.token_count, tf
            )
            for prov in providers.all():
                dim = 0
                if prov.model_name in model_dense:
                    vec = model_dense[prov.model_name][i]
                    dim = int(vec.shape[-1])
                store.ensure_model_tables(prov.model_name, dim)
                if prov.late_interaction:
                    store.store_vectors(
                        prov.model_name, chunk_id, model_dense[prov.model_name][i],
                        model_tok[prov.model_name][i],
                    )
                else:
                    store.store_vectors(prov.model_name, chunk_id, model_dense[prov.model_name][i])

        store.record_file(str(p), sha, mtime)
        log.info("Ingested %s: %d chunks", p, len(chunks))
        return IngestResult(str(p), len(chunks))
    except Exception as e:  # keep the pipeline alive for other files
        log.exception("Failed to ingest %s", p)
        return IngestResult(str(p), 0, error=str(e))


def ingest_dir(
    path: str,
    store: Store,
    config: Config,
    providers: ProviderRegistry,
    ocr: OcrClient,
) -> list[IngestResult]:
    root = Path(path)
    results: list[IngestResult] = []
    if not root.is_dir():
        return results
    for f in sorted(root.rglob("*.md")):
        if f.is_file():
            results.append(ingest_file(str(f), store, config, providers, ocr))
    for f in sorted(root.rglob("*.json")):
        if f.is_file():
            results.append(ingest_file(str(f), store, config, providers, ocr))
    for f in sorted(root.rglob("*.pdf")):
        if f.is_file():
            results.append(ingest_file(str(f), store, config, providers, ocr))
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() not in {".md", ".json", ".pdf"}:
            # Plain / other text files; skip obvious binaries.
            head = f.read_bytes()[:512]
            if b"\x00" in head:
                continue
            results.append(ingest_file(str(f), store, config, providers, ocr))
    return results
