"""Two-layer configuration model.

Install-time config and runtime config (ingestion dirs) both live in a single
YAML file written by `rag install` (default ~/.rag/config.yaml, overridable
via the RAG_CONFIG environment variable).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

#: Overridable config location (env var wins).
CONFIG_PATH_ENV = "RAG_CONFIG"

#: Embedding model categories -> default free-form model names.
DEFAULT_EMBEDDING_MODELS: dict[str, str] = {
    "late-interaction": "lightonai/GTE-ModernColBERT-v1",
    "llm-retrieval": "jina-ai/jina-embeddings-v3",
    "standard": "nomic-embed-text-v2",
}

DEFAULT_LLM_MODEL = "Qwen3-27B"
DEFAULT_OCR_MODEL = "maternion/LightOnOCR-2:latest"
DEFAULT_OCR_URL = "http://localhost:11434"
DEFAULT_MCP_PORT = 8931


def default_config_path() -> Path:
    env = os.environ.get(CONFIG_PATH_ENV)
    if env:
        return Path(env)
    return Path.home() / ".rag" / "config.yaml"


@dataclass
class DirEntry:
    path: str
    watch: bool = False


@dataclass
class Config:
    # LLM (chat) endpoint — used by external consumers of the MCP server.
    llm_url: str = "http://localhost:8000"
    llm_model: str = DEFAULT_LLM_MODEL

    # Embedding models, one per category.
    embedding_models: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_EMBEDDING_MODELS))

    # OCR (Ollama).
    ocr_model: str = DEFAULT_OCR_MODEL
    ocr_url: str = DEFAULT_OCR_URL

    # Storage locations.
    data_dir: str = str(Path.home() / ".rag" / "data")
    db_path: str = str(Path.home() / ".rag" / "data" / "index.db")

    # MCP server.
    mcp_host: str = "0.0.0.0"
    mcp_port: int = DEFAULT_MCP_PORT
    # External IdP: issuer URL + JWKS endpoint used to verify JWTs.
    auth_issuer: Optional[str] = None
    auth_jwks_uri: Optional[str] = None
    auth_audience: Optional[str] = None

    # Runtime layer: ingestion directories.
    dirs: list[DirEntry] = field(default_factory=list)

    # ------------------------------------------------------------------ paths
    @property
    def data_dir_path(self) -> Path:
        return Path(self.data_dir).expanduser()

    @property
    def db_path_obj(self) -> Path:
        return Path(self.db_path).expanduser()

    # ------------------------------------------------------------------ model helpers
    def model_names(self) -> list[str]:
        return list(self.embedding_models.values())

    def model_for(self, selector: str) -> Optional[str]:
        """Resolve a user selector (category key or model name) to a model name."""
        selector = selector.strip()
        if selector in self.embedding_models:
            return self.embedding_models[selector]
        if selector in self.embedding_models.values():
            return selector
        return None

    def category_of(self, model: str) -> Optional[str]:
        for cat, name in self.embedding_models.items():
            if name == model:
                return cat
        return None

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "llm": {"url": self.llm_url, "model": self.llm_model},
            "embedding_models": dict(self.embedding_models),
            "ocr": {"model": self.ocr_model, "url": self.ocr_url},
            "paths": {"data": self.data_dir, "db": self.db_path},
            "mcp": {
                "host": self.mcp_host,
                "port": self.mcp_port,
                "auth": {
                    "issuer": self.auth_issuer,
                    "jwks_uri": self.auth_jwks_uri,
                    "audience": self.auth_audience,
                },
            },
            "dirs": [{"path": d.path, "watch": d.watch} for d in self.dirs],
        }

    def save(self, path: Optional[Path] = None) -> None:
        path = path or default_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "Config":
        path = path or default_config_path()
        if not path.exists():
            raise FileNotFoundError(
                f"No config at {path}. Run `rag install` first."
            )
        data = yaml.safe_load(path.read_text()) or {}
        cfg = cls()
        cfg.llm_url = data.get("llm", {}).get("url", cfg.llm_url)
        cfg.llm_model = data.get("llm", {}).get("model", cfg.llm_model)
        em = data.get("embedding_models", {})
        for cat in DEFAULT_EMBEDDING_MODELS:
            if cat in em:
                cfg.embedding_models[cat] = em[cat]
        cfg.ocr_model = data.get("ocr", {}).get("model", cfg.ocr_model)
        cfg.ocr_url = data.get("ocr", {}).get("url", cfg.ocr_url)
        cfg.data_dir = data.get("paths", {}).get("data", cfg.data_dir)
        cfg.db_path = data.get("paths", {}).get("db", cfg.db_path)
        mcp = data.get("mcp", {})
        cfg.mcp_host = mcp.get("host", cfg.mcp_host)
        cfg.mcp_port = int(mcp.get("port", cfg.mcp_port))
        auth = mcp.get("auth", {})
        cfg.auth_issuer = auth.get("issuer")
        cfg.auth_jwks_uri = auth.get("jwks_uri")
        cfg.auth_audience = auth.get("audience")
        cfg.dirs = [DirEntry(d["path"], bool(d.get("watch", False))) for d in data.get("dirs", [])]
        return cfg

    def add_dir(self, path: str, watch: bool = False) -> None:
        path = str(Path(path).expanduser().resolve())
        for d in self.dirs:
            if d.path == path:
                d.watch = d.watch or watch
                return
        self.dirs.append(DirEntry(path, watch))

    def remove_dir(self, path: str) -> bool:
        path = str(Path(path).expanduser().resolve())
        before = len(self.dirs)
        self.dirs = [d for d in self.dirs if d.path != path]
        return len(self.dirs) != before
