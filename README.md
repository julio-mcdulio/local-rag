# rag-cli

Local, offline RAG CLI. Ingests local text files (markdown, PDF, JSON, plain text), precomputes per-model search indices, and serves semantic + BM25 + grep search — from the terminal or as an MCP server for agentic clients.

Everything runs locally except two optional HTTP endpoints you configure: an LLM (chat) and an OCR endpoint (Ollama, PDF fallback only).

## Features

- **Ingestion**
  - Text extraction for markdown, PDF (pypdf + pypdfium2), JSON, plain text.
  - OCR fallback for PDF **only**: pages with <20 embedded chars are rasterized and sent to a configurable Ollama OCR model.
  - Structure-aware chunking: markdown by headings, JSON by top-level records, text by paragraphs; ~512-token windows (50 overlap), merge small units, cap 1024 tokens.
  - Every chunk carries precise source coordinates (`path:start_line:start_char`) for citations.
  - Per-file hash skip: unchanged files are not re-ingested.
- **Precomputed search indices** (one SQLite DB, sqlite-vec)
  - Dense vectors per configured embedding model (`vec_<slug>` tables).
  - Late-interaction / ColBERT token vectors (`tok_<slug>` tables) for maxSim re-ranking.
  - BM25 inverted-index postings (`bm25_terms`).
- **Search**
  - `search_embedding` — dense search, model selectable per query.
  - `search_late_interaction` — ColBERT maxSim (coarse top-200 → token-level re-rank).
  - `search_bm25` — keyword search over the precomputed postings.
  - `grep` — regex over the raw original files, in place.
  - Results include chunk text, path, line, char offset, and score — enough to build an exact citation.
- **Watch mode** — polls configured directories every 5 s (mtime + sha256): new files ingested, modified files re-ingested, deleted files removed from all indices.
- **MCP server** — streamable-HTTP (stateless) on `mcp_host:mcp_port` (default `8931`). Tools: `search_embedding`, `search_embedding_model`, `search_late_interaction`, `search_bm25`, `grep`, `status`, plus a `/health` endpoint. Optional JWT auth validated against an external IdP (issuer + JWKS); unauthenticated when not configured.

## Installation

Requires Python ≥ 3.10.

```bash
uv venv .venv
uv pip install .            # or: pip install -e .
rag --help
```

## Quickstart

```bash
# 1. Interactive setup — writes ~/.rag/config.yaml
rag install

# 2. Add an ingestion directory (initial sync runs automatically)
rag dir add ~/docs --watch

# 3. Search from the terminal
rag search "deployment strategy" --model standard -k 5
rag search "deploy" --bm25
rag search "def [a-z_]+" --grep --json

# 4. Inspect state
rag status

# 5. Serve MCP (or run everything in one process)
rag mcp
rag daemon            # file watcher + MCP server in one process
```

### CLI reference

| Command | Description |
| --- | --- |
| `rag install` | Interactive setup; writes the config file |
| `rag dir add <path> [--watch]` | Add an ingestion directory (persisted, initial sync) |
| `rag dir remove <path>` | Remove an ingestion directory |
| `rag search <query> [--model NAME] [-k K] [--bm25] [--grep] [--json]` | Human-facing search with citations |
| `rag status` | Config summary + index statistics |
| `rag remove <file>` | Remove one file from all indexes |
| `rag mcp` | Start the MCP server only |
| `rag daemon` | Watcher + MCP server in one process |

## Configuration

Single YAML file at `~/.rag/config.yaml` (path overridable via `RAG_CONFIG`), created by `rag install`.

- **Install layer** (set once): LLM endpoint URL/model, three embedding models (`standard`, `llm-retrieval`, `late-interaction`), OCR model + URL, data dir / DB path, MCP host/port, optional JWT auth (`auth_issuer`, `auth_jwks_uri`, `auth_audience`).
- **Runtime layer** (via CLI): ingestion directories (`rag dir add/remove`), persisted in the same file.

Defaults: `standard` = `nomic-embed-text-v2`, `llm-retrieval` = `jina-ai/jina-embeddings-v3`, `late-interaction` = `lightonai/GTE-ModernColBERT-v1`, OCR = `maternion/LightOnOCR-2:latest`.

## Docker deployment

The primary deployment is a single container (GPU-enabled; CUDA torch wheels with transparent CPU fallback):

```bash
# One-time host setup: internal network connecting to your Ollama / LLM
docker network create --internal services
docker network connect services ollama
docker network connect services llamacpp

# Config (~/.rag/config.yaml) should point at the services, e.g.:
#   llm.url:  http://llamacpp:8080
#   ocr.url:  http://ollama:11434

docker compose up -d    # exposes 8931, mounts ~/.rag and rag-data volume
```

- Entrypoint is `rag daemon` (watcher + MCP in one process).
- Ollama and the LLM are **not** bundled — run them separately; the daemon talks to them over HTTP.
- Runtime config changes (`rag dir add/remove`) don't require a restart:
  `docker exec -it rag kill -HUP 1` reloads the config and picks up new directories.

## Project layout

```
rag/
  cli.py          # `rag` entrypoint and subcommands
  config.py       # Config dataclass, YAML load/save (~/.rag/config.yaml)
  ingest.py       # extract → chunk → embed/index pipeline
  text_extract.py # markdown/PDF/JSON/text extraction + OCR fallback
  chunker.py      # structure-aware chunking
  embeddings.py   # sentence-transformers provider registry
  store.py        # SQLite + sqlite-vec storage, dense/ColBERT/BM25
  search.py       # dense, late-interaction, BM25, grep
  watcher.py      # polling file watcher
  mcp_server.py   # streamable-HTTP MCP server + JWT auth
  daemon.py       # watcher + MCP in one process (SIGHUP config reload)
tests/            # pytest suite
docs/             # requirements document
Dockerfile        # GPU-capable container image
docker-compose.yml
```

## Development

```bash
uv pip install -e .[dev]   # includes pytest
pytest
```

See `TODO.md` for implementation status per requirement and `docs/rag-cli-requirements.md` for the full spec.
