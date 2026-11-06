"""Watcher tests: poll loop, mtime fast path, add/change/remove, config reload.

All offline: fake embedding providers, no real dirs outside tmp_path.
"""

import pytest

from conftest import make_config, make_registry
from rag.config import DirEntry
from rag.store import Store
from rag.text_extract import OcrClient
from rag.watcher import Watcher


@pytest.fixture
def env(tmp_path):
    cfg = make_config(str(tmp_path))
    root = tmp_path / "corpus"
    root.mkdir()
    cfg.dirs = [DirEntry(str(root), watch=True)]
    store = Store(cfg.db_path_obj)
    providers = make_registry()
    ocr = OcrClient("http://localhost:11434", "maternion/LightOnOCR-2")
    yield cfg, store, providers, ocr, root
    store.close()


def test_initial_sync_and_steady_state(env):
    cfg, store, providers, ocr, root = env
    (root / "a.md").write_text("alpha beta gamma\n")
    (root / "b.json").write_text('[{"k": "v"}]')
    (root / "skip.bin").write_bytes(b"\x00\x01")

    w = Watcher(store, cfg, providers, ocr, interval=0.01)
    n = w.initial_sync()
    assert n >= 2  # a.md + b.json
    assert store.chunk_count(str(root / "a.md")) == 1
    assert store.chunk_count(str(root / "b.json")) == 1
    assert store.chunk_count(str(root / "skip.bin")) == 0

    # steady state: nothing to do
    assert w.step() == {"added": 0, "changed": 0, "removed": 0}


def test_detect_change_remove_add(env):
    cfg, store, providers, ocr, root = env
    f = root / "a.md"
    f.write_text("alpha beta gamma\n")
    w = Watcher(store, cfg, providers, ocr, interval=0.01)
    w.initial_sync()

    # change
    f.write_text("totally new content\n")
    stats = w.step()
    assert stats["changed"] == 1
    rows = store.chunks_for_path(str(f))
    assert len(rows) == 1
    assert "totally new content" in rows[0]["text"]

    # remove
    f.unlink()
    stats = w.step()
    assert stats["removed"] == 1
    assert store.chunk_count(str(f)) == 0

    # add
    g = root / "c.md"
    g.write_text("hello there\n")
    stats = w.step()
    assert stats["added"] == 1
    assert store.chunk_count(str(g)) == 1


def test_mtime_unchanged_skips_hash(env):
    cfg, store, providers, ocr, root = env
    f = root / "a.md"
    f.write_text("stable text\n")
    w = Watcher(store, cfg, providers, ocr, interval=0.01)
    w.initial_sync()

    # snapshot entry should be reused when mtime is unchanged
    snap1 = w._scan()
    snap2 = w._scan()
    assert snap1[str(f)] is snap2[str(f)]  # same cached object, no re-hash


def test_config_reload_resets_snapshot(env, tmp_path):
    cfg, store, providers, ocr, root = env
    f = root / "a.md"
    f.write_text("alpha beta gamma\n")
    w = Watcher(store, cfg, providers, ocr, interval=0.01)
    w.initial_sync()

    # point the watcher at a fresh dir and reload
    other = tmp_path / "corpus2"
    other.mkdir()
    other.joinpath("x.md").write_text("new dir content\n")
    cfg.dirs = [DirEntry(str(other), watch=True)]
    w.on_config_reload()

    stats = w.step()
    # old file left the dirs -> removed; new file -> added
    assert stats["removed"] == 1
    assert stats["added"] == 1
    assert store.chunk_count(str(f)) == 0
    assert store.chunk_count(str(other / "x.md")) == 1
