"""Ingest pipeline tests: per-file ingest, hash-skip, re-ingest, removal.

Uses a temp corpus with fake embedding providers (no HF models, no torch).
"""

import json

import pytest

from conftest import make_config, make_registry
from rag.config import Config
from rag.ingest import ingest_dir, ingest_file
from rag.store import Store
from rag.text_extract import OcrClient


@pytest.fixture
def env(tmp_path):
    cfg = make_config(str(tmp_path))
    store = Store(cfg.db_path_obj)
    providers = make_registry()
    ocr = OcrClient("http://localhost:11434", "maternion/LightOnOCR-2")
    root = tmp_path / "corpus"
    root.mkdir()
    yield cfg, store, providers, ocr, root
    store.close()


def test_ingest_markdown_json_and_txt(env):
    cfg, store, providers, ocr, root = env
    (root / "notes.md").write_text("# Title\n" + "word " * 200 + "\n")
    # two records, each >128 tokens so they are NOT merged into one chunk
    records = [{"id": i, "body": " ".join(f"w{j}" for j in range(195))}
               for i in (1, 2)]
    (root / "data.json").write_text(json.dumps(records, indent=1))
    (root / "plain.txt").write_text("alpha beta\n\ngamma delta\n")
    (root / "binary.bin").write_bytes(b"\x00\x01\x02garbage")

    results = ingest_dir(str(root), store, cfg, providers, ocr)
    ingested = {r.path.split("/")[-1]: r for r in results}
    assert "notes.md" in ingested and ingested["notes.md"].chunks >= 1
    assert "data.json" in ingested
    # JSON split by top-level record, one chunk per record
    assert store.chunk_count(str(root / "data.json")) == 2
    assert "plain.txt" in ingested
    # binaries are not ingested
    assert "binary.bin" not in ingested

    stats = store.index_stats()
    # one vector per chunk per model (3 fake models)
    for model in cfg.embedding_models.values():
        from rag.embeddings import slugify

        t = "vec_" + slugify(model)
        assert stats["models"][t] == store.chunk_count()


def test_hash_skip_and_force_reingest(env):
    cfg, store, providers, ocr, root = env
    f = root / "a.md"
    f.write_text("hello world one two\n")
    r1 = ingest_file(str(f), store, cfg, providers, ocr)
    assert r1.chunks == 1 and not r1.skipped
    before = store.chunk_count(str(f))

    # unchanged -> skipped, no extra chunks
    r2 = ingest_file(str(f), store, cfg, providers, ocr)
    assert r2.skipped
    assert store.chunk_count(str(f)) == before

    # changed -> re-ingest replaces chunks (per-file granularity)
    f.write_text("completely different text now\n")
    r3 = ingest_file(str(f), store, cfg, providers, ocr)
    assert not r3.skipped
    assert store.chunk_count(str(f)) == r3.chunks
    rows = store.chunks_for_path(str(f))
    assert len(rows) == 1
    assert store.get_chunk(rows[0]["id"])["text"] == "completely different text now"


def test_remove_file_cleans_all_indices(env):
    cfg, store, providers, ocr, root = env
    f = root / "rm.md"
    f.write_text("to be removed soon\n")
    ingest_file(str(f), store, cfg, providers, ocr)
    assert store.chunk_count(str(f)) == 1

    store.remove_file(str(f))
    assert store.chunk_count() == 0
    assert store.bm25_search(["removed"], 5) == []
    for model in cfg.embedding_models.values():
        from rag.embeddings import slugify

        n = store.conn.execute(
            f"SELECT COUNT(*) FROM vec_{slugify(model)}"
        ).fetchone()[0]
        assert n == 0
        n = store.conn.execute(
            f"SELECT COUNT(*) FROM tok_{slugify(model)}"
        ).fetchone()[0]
        assert n == 0


def test_ingest_empty_file_records_no_chunks(env):
    cfg, store, providers, ocr, root = env
    f = root / "empty.txt"
    f.write_text("   \n")
    r = ingest_file(str(f), store, cfg, providers, ocr)
    assert r.chunks == 0
    assert store.chunk_count(str(f)) == 0


def test_ingest_error_is_reported_not_raised(env):
    cfg, store, providers, ocr, root = env
    f = root / "missing.md"
    f.write_text("x")
    f.unlink()
    r = ingest_file(str(f), store, cfg, providers, ocr)
    assert r.error
    assert r.chunks == 0
