"""Config tests: YAML round-trip and reload semantics (SIGHUP path)."""

import pytest

from rag.config import Config, DirEntry


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "config.yaml"
    cfg = Config()
    cfg.auth_issuer = "https://idp.example.com"
    cfg.auth_jwks_uri = "https://idp.example.com/.well-known/jwks.json"
    cfg.auth_audience = "rag-mcp"
    cfg.add_dir("/tmp/corpus1", watch=True)
    cfg.add_dir("/tmp/corpus2")
    cfg.save(path)

    loaded = Config.load(path)
    assert loaded.auth_issuer == "https://idp.example.com"
    assert loaded.auth_jwks_uri == "https://idp.example.com/.well-known/jwks.json"
    assert loaded.auth_audience == "rag-mcp"
    assert loaded.dirs == [DirEntry("/tmp/corpus1", True), DirEntry("/tmp/corpus2", False)]
    assert loaded.embedding_models == cfg.embedding_models


def test_env_var_overrides_path(tmp_path, monkeypatch):
    path = tmp_path / "cfg.yaml"
    cfg = Config()
    cfg.save(path)
    monkeypatch.setenv("RAG_CONFIG", str(path))
    loaded = Config.load()  # no explicit path -> env var used
    assert loaded is not None
    assert loaded.data_dir == cfg.data_dir


def test_reload_picks_up_dir_changes(tmp_path):
    path = tmp_path / "config.yaml"
    cfg = Config()
    cfg.add_dir("/tmp/a")
    cfg.save(path)

    # "edit" the YAML the way an operator would after SIGHUP
    import yaml

    data = yaml.safe_load(path.read_text())
    data["dirs"].append({"path": "/tmp/b", "watch": True})
    path.write_text(yaml.safe_dump(data))

    reloaded = Config.load(path)
    assert [d.path for d in reloaded.dirs] == ["/tmp/a", "/tmp/b"]


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.load(tmp_path / "nope.yaml")
