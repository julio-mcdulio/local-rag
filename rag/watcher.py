"""File watching: poll-based watcher over configured ingestion directories.

Compares (mtime, sha256) snapshots; new/changed files are (re)ingested,
deleted files are removed from the index. Runs as a loop inside the daemon.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .embeddings import ProviderRegistry
from .ingest import ingest_dir
from .store import Store
from .text_extract import OcrClient

log = logging.getLogger("rag.watch")

@dataclass
class _Entry:
    mtime: float
    sha256: str


SUPPORTED_SUFFIXES = {".md", ".markdown", ".json", ".pdf", ".txt"}


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Watcher:
    def __init__(
        self,
        store: Store,
        config: Config,
        providers: ProviderRegistry,
        ocr: OcrClient,
        interval: float = 5.0,
    ):
        self.store = store
        self.config = config
        self.providers = providers
        self.ocr = ocr
        self.interval = interval
        self._snapshot: dict[str, _Entry] = {}

    def _scan(self) -> dict[str, _Entry]:
        """Snapshot (mtime, sha256) of all ingestible files.

        Perf: files whose mtime is unchanged since the last scan reuse the
        stored hash, so SHA-256 is only computed for new/changed files.
        """
        snap: dict[str, _Entry] = {}
        prev = self._snapshot
        for d in self.config.dirs:
            root = Path(d.path)
            if not root.is_dir():
                continue
            for f in root.rglob("*"):
                if not f.is_file():
                    continue
                if f.suffix.lower() not in SUPPORTED_SUFFIXES:
                    # plain/other text; skip obvious binaries
                    try:
                        with f.open("rb") as fh:
                            if b"\x00" in fh.read(512):
                                continue
                    except OSError:
                        continue
                try:
                    st = f.stat()
                except OSError:
                    continue
                key = str(f)
                cached = prev.get(key)
                if cached is not None and cached.mtime == st.st_mtime:
                    snap[key] = cached
                else:
                    snap[key] = _Entry(st.st_mtime, _hash(f))
        return snap

    def step(self) -> dict:
        """One watch iteration. Returns counts."""
        snap = self._scan()
        added = changed = removed = 0
        for path, entry in snap.items():
            prev = self._snapshot.get(path)
            if prev is None:
                res = self._ingest_one(path)
                if res and not res.skipped:
                    added += 1
            elif entry.sha256 != prev.sha256 or entry.mtime != prev.mtime:
                res = self._ingest_one(path, force=True)
                if res and not res.skipped:
                    changed += 1
        for path in self._snapshot:
            if path not in snap:
                self.store.remove_file(path)
                removed += 1
        self._snapshot = snap
        return {"added": added, "changed": changed, "removed": removed}

    def _ingest_one(self, path: str, force: bool = False):
        from .ingest import ingest_file

        return ingest_file(path, self.store, self.config, self.providers, self.ocr,
                           force=force)

    def initial_sync(self):
        """Ingest everything present in configured dirs."""
        from .ingest import ingest_file

        count = 0
        for d in self.config.dirs:
            for res in ingest_dir(d.path, self.store, self.config, self.providers, self.ocr):
                if res.chunks or res.error:
                    log.info("ingest %s: chunks=%d skipped=%s error=%s",
                             res.path, res.chunks, res.skipped, res.error)
                count += 1
        self._snapshot = self._scan()
        return count

    def on_config_reload(self):
        """Apply runtime dir changes (e.g. via SIGHUP) without restart.

        Keeps the previous snapshot so the next ``step()`` rescans the
        current dirs: files in newly added dirs are ingested, unchanged
        files skip via hash, and files that left a dir are removed.
        """

    def run_forever(self):
        self.initial_sync()
        while True:
            time.sleep(self.interval)
            stats = self.step()
            if any(stats.values()):
                log.info("watch: %s", stats)
