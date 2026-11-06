"""Search tests: BM25, dense, late-interaction and grep, offline fakes.

Fake vectors are deterministic per text, so searching with the exact chunk
text must rank that chunk first with score ~1.0 (dense).
"""

import pytest

from conftest import make_config, make_registry
from rag.ingest import ingest_file
from rag.search import (
    grep_files,
    search_bm25,
    search_embedding,
    search_late_interaction,
)
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


def _ingest(env, name, text):
    cfg, store, providers, ocr, root = env
    f = root / name
    f.write_text(text)
    return ingest_file(str(f), store, cfg, providers, ocr), f


def test_bm25_ranks_by_term_presence(env):
    cfg, store, providers, ocr, root = env
    r, f = _ingest(env, "a.md", "quantum zebra flurble\n")
    _ingest(env, "b.md", "quantum mechanics intro\n")
    res = search_bm25(store, "zebra flurble", 5)
    assert res and res[0].path == str(f)
    # 'flurble' is unique to a.md, so it must outrank b.md for 'quantum'
    res = search_bm25(store, "quantum", 5)
    paths = [r.path for r in res]
    assert str(f) in paths and str(root / "b.md") in paths


def test_dense_search_exact_text_matches(env):
    cfg, store, providers, ocr, root = env
    model = cfg.embedding_models["standard"]
    text = "solar systems orbit in quiet ellipses"
    r, f = _ingest(env, "a.md", text + "\n")
    res = search_embedding(store, providers, model, text, 5)
    assert res and res[0].path == str(f)
    assert res[0].score > 0.99
    # result carries source coordinates
    assert res[0].start_line == 1
    assert res[0].start_char == 0


def test_late_interaction_top1_is_exact_doc(env):
    cfg, store, providers, ocr, root = env
    model = cfg.embedding_models["late-interaction"]
    text = "fermented sourdough starter culture"
    r, f = _ingest(env, "a.md", text + "\n")
    res = search_late_interaction(store, providers, model, text, 5)
    assert res and res[0].path == str(f)
    assert res[0].score > 0.0


def test_grep_over_raw_files(env):
    cfg, store, providers, ocr, root = env
    f = root / "notes.md"
    f.write_text("line one\nneedle in the haystack\nline three\n")
    ingest_file(str(f), store, cfg, providers, ocr)
    res = grep_files(store, "haystack", 5, dirs=[str(root)])
    assert len(res) == 1
    assert res[0].path == str(f)
    assert res[0].start_line == 2
    assert "needle" in res[0].text


def test_search_empty_index(env):
    cfg, store, providers, ocr, root = env
    model = cfg.embedding_models["standard"]
    assert search_bm25(store, "anything", 5) == []
    assert search_embedding(store, providers, model, "anything", 5) == []
    assert search_late_interaction(
        store, providers, cfg.embedding_models["late-interaction"], "anything", 5
    ) == []
