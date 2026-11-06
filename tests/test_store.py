"""Store tests: round-trip, BM25, dense search (blob fallback), removal."""

import numpy as np
import pytest

from rag.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "index.db")
    yield s
    s.close()


def _insert(store: Store, path: str, text: str, tf: dict[str, int]) -> int:
    return store.insert_chunk(path, 1, 0, text, len(tf), tf)


def test_insert_and_get_chunk_roundtrip(store):
    cid = _insert(store, "a.md", "hello world", {"hello": 1, "world": 1})
    row = store.get_chunk(cid)
    assert row["path"] == "a.md"
    assert row["text"] == "hello world"
    assert store.chunk_count() == 1
    assert store.chunk_count("a.md") == 1


def test_file_bookkeeping_and_remove(store):
    _insert(store, "a.md", "alpha beta", {"alpha": 1, "beta": 1})
    store.record_file("a.md", "sha1", 123.0)
    files = {f["path"]: f for f in store.iter_files()}
    assert files["a.md"]["sha256"] == "sha1"

    store.remove_file("a.md")
    assert store.chunk_count() == 0
    assert list(store.iter_files()) == []
    # bm25 postings gone
    assert store.bm25_search(["alpha"], 5) == []


def test_bm25_ranks_by_term_frequency(store):
    # doc 1: "cat" x1; doc 2: "cat" x5; doc 3: "cat dog"
    _insert(store, "a.md", "cat", {"cat": 1})
    _insert(store, "b.md", "cat cat cat cat cat", {"cat": 5})
    _insert(store, "c.md", "cat dog", {"cat": 1, "dog": 1})
    res = store.bm25_search(["cat"], 10)
    # doc 2 has the most occurrences -> first; doc 1 (shorter) beats doc 3
    assert [cid for cid, _ in res] == [2, 1, 3]

    res = store.bm25_search(["dog"], 10)
    assert res[0][0] == 3
    # unknown term -> no results
    assert store.bm25_search(["zzz"], 10) == []
    assert store.bm25_search([], 10) == []


def test_bm25_case_insensitive_lookup(store):
    _insert(store, "a.md", "Hello World", {"hello": 1, "world": 1})
    res = store.bm25_search(["HELLO"], 5)
    assert res and res[0][0] == 1


def test_dense_search_blob_fallback(store):
    # Force the blob backend (no sqlite-vec extension in this env, but be
    # explicit so the test is meaningful either way).
    store.vec_backend = "blob"
    model = "test-model"
    store.ensure_model_tables(model, 8)

    # orthogonal-ish vectors: v0 parallel to query, v1 perpendicular
    v0 = np.eye(8, dtype=np.float32)[0]
    v1 = np.eye(8, dtype=np.float32)[1]
    q = np.eye(8, dtype=np.float32)[0]
    _insert(store, "a.md", "doc zero", {"zero": 1})
    _insert(store, "b.md", "doc one", {"one": 1})
    store.store_vectors(model, 1, v0)
    store.store_vectors(model, 2, v1)

    res = store.search_dense(model, q, 2)
    assert res[0][0] == 1
    assert res[0][1] == pytest.approx(1.0, abs=1e-5)
    assert res[1][0] == 2
    assert res[1][1] == pytest.approx(0.0, abs=1e-5)
    # empty index -> no results
    store2 = Store(store.path)
    store2.vec_backend = "blob"
    store2.ensure_model_tables("other", 8)
    assert store2.search_dense("other", q, 3) == []
    store2.close()


def test_sqlite_vec_backend_detection():
    store = Store(__import__("pathlib").Path("/tmp/rag-test-sv.db"))
    # Either backend is acceptable; API must be self-consistent.
    assert store.vec_backend in ("sqlite-vec", "blob")
    if store.vec_backend == "sqlite-vec":
        store.ensure_model_tables("m", 4)
        v = np.ones(4, dtype=np.float32)
        store.store_vectors("m", 1, v)
        res = store.search_dense("m", np.ones(4, dtype=np.float32), 3)
        assert res and res[0][0] == 1
        assert res[0][1] == pytest.approx(1.0, abs=1e-4)
    store.close()


def test_token_vectors_roundtrip(store):
    store.vec_backend = "blob"
    model = "li-model"
    store.ensure_model_tables(model, 4)
    toks = np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    _insert(store, "a.md", "tok doc", {"tok": 1, "doc": 1})
    store.store_vectors(model, 1, np.ones(4, dtype=np.float32), toks)
    out = store.load_token_vectors(model, [1], dim=4)
    assert out[1].shape == (3, 4)
    np.testing.assert_allclose(out[1], toks, atol=1e-6)
    assert store.load_token_vectors(model, [999], dim=4) == {}


def test_remove_chunk_cleans_vector_indices(store):
    store.vec_backend = "blob"
    model = "m"
    store.ensure_model_tables(model, 4)
    cid = _insert(store, "a.md", "x", {"x": 1})
    store.store_vectors(model, cid, np.ones(4, dtype=np.float32))
    store.remove_chunk(cid)
    assert store.chunk_count() == 0
    assert store.bm25_search(["x"], 3) == []
    # no vector rows left
    rows = store.conn.execute("SELECT COUNT(*) FROM vec_m").fetchone()[0]
    assert rows == 0


def test_index_stats(store):
    _insert(store, "a.md", "a b", {"a": 1, "b": 1})
    store.record_file("a.md", "s", 1.0)
    stats = store.index_stats()
    assert stats["files"] == 1
    assert stats["chunks"] == 1
    assert stats["bm25_terms"] == 2
    assert stats["vec_backend"] in ("sqlite-vec", "blob")
