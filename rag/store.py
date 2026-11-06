"""SQLite storage for chunks, per-model vector indices and the BM25 index.

Layout:
- files:        path -> sha256/mtime (per-file granularity bookkeeping)
- chunks:       chunk text + source coordinates (path, start_line, start_char)
- bm25_terms:   (term, chunk_id, tf) postings, precomputed at ingestion
- vec_<slug>:   one vector table per embedding model
    * dense models: one pooled vector per chunk
    * late-interaction models: one pooled (coarse-rank) vector per chunk plus
      per-token vectors kept as a BLOB in tok_<slug> for MaxSim re-ranking

Uses sqlite-vec when the native extension is available; otherwise falls back
to BLOB storage with an in-memory numpy scan (still fully local/offline).
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from .embeddings import slugify


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize along the last axis (per pooled vector / per token row)."""
    arr = np.asarray(arr, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    return arr / norm


class Store:
    def __init__(self, db_path: str | Path):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()  # serializes writer-side work (watcher thread)
        self._setup()

    def _setup(self):
        c = self.conn
        try:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=5000")
        except sqlite3.OperationalError:
            pass
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                path   TEXT PRIMARY KEY,
                sha256 TEXT,
                mtime  REAL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id         INTEGER PRIMARY KEY,
                path       TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                start_char INTEGER NOT NULL,
                text       TEXT NOT NULL,
                token_count INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks (path);
            CREATE TABLE IF NOT EXISTS bm25_terms (
                term     TEXT NOT NULL,
                chunk_id INTEGER NOT NULL,
                tf       INTEGER NOT NULL,
                PRIMARY KEY (term, chunk_id)
            );
            """
        )
        c.commit()
        self.vec_backend = self._try_sqlite_vec()

    def _try_sqlite_vec(self) -> str:
        try:
            import sqlite_vec

            self.conn.enable_extension(True)
            self.conn.load_extension(sqlite_vec.loadable_library_path())
            self.conn.enable_extension(False)
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS _vecprobe USING vec0(embedding FLOAT[2])"
            )
            self.conn.execute("DROP TABLE _vecprobe")
            return "sqlite-vec"
        except Exception:
            return "blob"

    def close(self):
        self.conn.close()

    # ------------------------------------------------------------------ files
    def record_file(self, path: str, sha256: str, mtime: float):
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO files (path, sha256, mtime) VALUES (?, ?, ?)",
                (path, sha256, mtime),
            )
        self.conn.commit()

    def iter_files(self):
        for row in self.conn.execute("SELECT path, sha256, mtime FROM files"):
            yield dict(row)

    def remove_file(self, path: str):
        c = self.conn
        with self._lock:
            c.execute(
                "DELETE FROM bm25_terms WHERE chunk_id IN "
                "(SELECT id FROM chunks WHERE path = ?)",
                (path,),
            )
            for t in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vec_%'"
            ):
                c.execute(
                    f"DELETE FROM {t['name']} WHERE rowid IN "
                    "(SELECT id FROM chunks WHERE path = ?)",
                    (path,),
                )
            for t in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'tok_%'"
            ):
                c.execute(
                    f"DELETE FROM {t['name']} WHERE chunk_id IN "
                    "(SELECT id FROM chunks WHERE path = ?)",
                    (path,),
                )
            c.execute("DELETE FROM chunks WHERE path = ?", (path,))
            c.execute("DELETE FROM files WHERE path = ?", (path,))
            c.commit()

    # ------------------------------------------------------------------ models
    def ensure_model_tables(self, model_name: str, dim: int):
        slug = slugify(model_name)
        c = self.conn
        if self.vec_backend == "sqlite-vec":
            c.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_{slug} "
                f"USING vec0(rowid INTEGER PRIMARY KEY, embedding FLOAT[{dim}])"
            )
        else:
            c.execute(
                f"CREATE TABLE IF NOT EXISTS vec_{slug} "
                f"(rowid INTEGER PRIMARY KEY, dim INTEGER, blob BLOB)"
            )
        c.execute(
            f"CREATE TABLE IF NOT EXISTS tok_{slug} "
            f"(chunk_id INTEGER PRIMARY KEY, blob BLOB)"
        )
        c.commit()

    # ------------------------------------------------------------------ chunks
    def insert_chunk(
        self,
        path: str,
        start_line: int,
        start_char: int,
        text: str,
        token_count: int,
        tf: dict[str, int],
    ) -> int:
        c = self.conn
        cur = c.execute(
            "INSERT INTO chunks (path, start_line, start_char, text, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (path, start_line, start_char, text, token_count),
        )
        chunk_id = cur.lastrowid
        for term, t in tf.items():
            c.execute(
                "INSERT OR REPLACE INTO bm25_terms (term, chunk_id, tf) VALUES (?, ?, ?)",
                (term, chunk_id, t),
            )
        c.commit()
        return chunk_id

    def chunks_for_path(self, path: str) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM chunks WHERE path = ?", (path,)))

    def get_chunk(self, chunk_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()

    def remove_chunk(self, chunk_id: int):
        c = self.conn
        c.execute("DELETE FROM bm25_terms WHERE chunk_id = ?", (chunk_id,))
        c.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
        for t in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'vec_%'"
        ):
            c.execute(f"DELETE FROM {t['name']} WHERE rowid = ?", (chunk_id,))
        for t in c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'tok_%'"
        ):
            c.execute(f"DELETE FROM {t['name']} WHERE chunk_id = ?", (chunk_id,))
        c.commit()

    def chunk_count(self, path: Optional[str] = None) -> int:
        if path is None:
            return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return self.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE path = ?", (path,)
        ).fetchone()[0]

    # ------------------------------------------------------------------ vectors
    def store_vectors(self, model_name: str, chunk_id: int, pooled: np.ndarray,
                      tokens: Optional[np.ndarray] = None):
        slug = slugify(model_name)
        pooled = _l2_normalize(pooled.astype(np.float32))
        c = self.conn
        if self.vec_backend == "sqlite-vec":
            # sqlite-vec stores raw float32 bytes
            c.execute(
                f"INSERT OR REPLACE INTO vec_{slug} (rowid, embedding) VALUES (?, ?)",
                (chunk_id, pooled.tobytes()),
            )
        else:
            c.execute(
                f"INSERT OR REPLACE INTO vec_{slug} (rowid, dim, blob) VALUES (?, ?, ?)",
                (chunk_id, int(pooled.shape[-1]), pooled.tobytes()),
            )
        if tokens is not None:
            tokens = _l2_normalize(tokens.astype(np.float32))
            c.execute(
                f"INSERT OR REPLACE INTO tok_{slug} (chunk_id, blob) VALUES (?, ?)",
                (chunk_id, np.ascontiguousarray(tokens).tobytes()),
            )
        c.commit()

    def _table_exists(self, c: sqlite3.Connection, name: str) -> bool:
        row = c.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
            (name,),
        ).fetchone()
        return row is not None

    def search_dense(self, model_name: str, query_vec: np.ndarray, k: int):
        """Return list of (chunk_id, score) — higher is better."""
        query_vec = _l2_normalize(np.asarray(query_vec, dtype=np.float32))
        slug = slugify(model_name)
        c = self.conn
        if not self._table_exists(c, f"vec_{slug}"):
            return []
        if self.vec_backend == "sqlite-vec":
            # sqlite-vec: MATCH against a raw float32 blob, ordered by L2
            # distance. Stored vectors and the query are L2-normalized, so
            # cosine = 1 - d^2 / 2.
            rows = c.execute(
                f"SELECT rowid AS id, distance(vec_{slug}, embedding) AS d "
                f"FROM vec_{slug} WHERE embedding MATCH ? "
                f"ORDER BY distance(vec_{slug}, embedding) LIMIT ?",
                (query_vec.tobytes(), k),
            ).fetchall()
            return [(r["id"], 1.0 - r["d"] * r["d"] / 2.0) for r in rows]
        rows = c.execute(
            f"SELECT rowid AS id, blob FROM vec_{slug}"
        ).fetchall()
        if not rows:
            return []
        mat = np.stack(
            [np.frombuffer(r["blob"], dtype=np.float32).reshape(-1) for r in rows]
        )
        scores = mat @ query_vec
        idx = np.argsort(-scores)[:k]
        return [(rows[i]["id"], float(scores[i])) for i in idx]

    def load_token_vectors(self, model_name: str, chunk_ids: list[int],
                           dim: Optional[int] = None) -> dict[int, np.ndarray]:
        """Return {chunk_id: (seq, dim) array} of token-level vectors."""
        if not chunk_ids:
            return {}
        slug = slugify(model_name)
        c = self.conn
        out: dict[int, np.ndarray] = {}
        for row in c.execute(
            f"SELECT chunk_id, blob FROM tok_{slug} WHERE chunk_id IN ({','.join('?' * len(chunk_ids))})",
            tuple(chunk_ids),
        ):
            arr = np.frombuffer(row["blob"], dtype=np.float32)
            if dim is not None:
                arr = arr.reshape(-1, dim)
            out[row["chunk_id"]] = arr
        return out

    # ------------------------------------------------------------------ bm25
    def bm25_search(self, terms: list[str], k: int):
        """BM25 over the precomputed postings list. Returns (chunk_id, score)."""
        terms = [t.lower() for t in terms if t]
        if not terms:
            return []
        c = self.conn
        n_docs = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if n_docs == 0:
            return []
        avgdl = c.execute(
            "SELECT COALESCE(AVG(token_count), 1) FROM chunks"
        ).fetchone()[0]
        placeholders = ",".join("?" * len(terms))
        df = dict(
            c.execute(
                f"SELECT term, COUNT(*) AS df FROM bm25_terms WHERE term IN ({placeholders}) GROUP BY term",
                tuple(terms),
            )
        )
        postings: dict[int, dict[str, int]] = {}
        for term in terms:
            for row in c.execute(
                f"SELECT chunk_id, tf FROM bm25_terms WHERE term = ?", (term,)
            ):
                postings.setdefault(row["chunk_id"], {})[term] = row["tf"]
        if not postings:
            return []
        doclens = {
            r["id"]: r["token_count"]
            for r in c.execute(
                f"SELECT id, token_count FROM chunks WHERE id IN ({','.join('?' * len(postings))})",
                tuple(postings.keys()),
            )
        }
        k1, b = 1.5, 0.75
        scores: dict[int, float] = {}
        for cid, tfs in postings.items():
            dl = doclens.get(cid, 1)
            s = 0.0
            for term, tf in tfs.items():
                d = df.get(term, 0)
                if d == 0:
                    continue
                idf = np.log((n_docs - d + 0.5) / (d + 0.5) + 1.0)
                s += idf * tf * (k1 + 1.0) / (
                    tf + k1 * (1.0 - b + b * dl / max(avgdl, 1.0))
                )
            scores[cid] = s
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return ranked[:k]

    # ------------------------------------------------------------------ stats
    def index_stats(self) -> dict:
        stats: dict = {
            "files": self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0],
            "chunks": self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "bm25_terms": self.conn.execute(
                "SELECT COUNT(DISTINCT term) FROM bm25_terms"
            ).fetchone()[0],
            "vec_backend": self.vec_backend,
            "models": {},
        }
        for row in self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name LIKE 'vec_%' ORDER BY name"
        ):
            tname = row["name"]
            try:
                cnt = self.conn.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
            except sqlite3.OperationalError:
                continue
            stats["models"][tname] = cnt
        return stats
