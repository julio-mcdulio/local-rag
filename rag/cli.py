"""CLI entry point: `rag` command.

Subcommands:
  rag install                 interactive setup, writes the config file
  rag dir add <path> [--watch]    add an ingestion directory (persisted)
  rag dir remove <path>         remove an ingestion directory
  rag search <query> [--model NAME] [-k K] [--bm25] [--grep] [--json]
  rag status                    config summary + index statistics
  rag remove <file>           remove one file from all indexes
  rag mcp                     start the MCP server only
  rag daemon                  watcher + MCP server in one process
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config, default_config_path
from .embeddings import slugify
from .store import Store


def _load_config() -> Config:
    try:
        return Config.load()
    except FileNotFoundError:
        print("No config found. Run `rag install` first.", file=sys.stderr)
        sys.exit(1)


def _providers(cfg: Config):
    from .embeddings import ProviderRegistry

    return ProviderRegistry(cfg.embedding_models)


def _ocr(cfg: Config):
    from .text_extract import OcrClient

    return OcrClient(cfg.ocr_url, cfg.ocr_model)


# ------------------------------------------------------------------ install
def cmd_install(args: argparse.Namespace) -> None:
    cfg = Config()

    def ask(prompt: str, default: str) -> str:
        if default:
            prompt = f"{prompt} [{default}]"
        try:
            val = input(prompt).strip()
        except EOFError:
            return default
        return val or default

    print("Local RAG CLI — install (all local/offline)\n")
    cfg.llm_url = ask("LLM endpoint URL (chat endpoint base, e.g. http://localhost:8000):", cfg.llm_url)
    cfg.llm_model = ask("LLM model name:", cfg.llm_model)
    cfg.embedding_models["standard"] = ask(
        "Standard embedding model (Hugging Face name):", cfg.embedding_models["standard"]
    )
    cfg.embedding_models["llm-retrieval"] = ask(
        "LLM-retrieval embedding model:", cfg.embedding_models["llm-retrieval"]
    )
    cfg.embedding_models["late-interaction"] = ask(
        "Late-interaction (ColBERT) embedding model:", cfg.embedding_models["late-interaction"]
    )
    cfg.ocr_model = ask("OCR model (Ollama tag):", cfg.ocr_model)
    cfg.ocr_url = ask("OCR endpoint URL (Ollama):", cfg.ocr_url)
    cfg.data_dir = ask("Data directory:", cfg.data_dir)
    cfg.db_path = ask("Index database path:", cfg.db_path)
    host = ask("MCP server host:", cfg.mcp_host)
    port = ask("MCP server port:", str(cfg.mcp_port))
    cfg.mcp_host = host
    try:
        cfg.mcp_port = int(port)
    except ValueError:
        cfg.mcp_port = 8931
    cfg.auth_issuer = ask("IdP issuer (blank for no JWT auth):", cfg.auth_issuer)
    cfg.auth_jwks_uri = ask("IdP JWKS URI:", cfg.auth_jwks_uri)
    cfg.auth_audience = ask("Expected JWT audience (blank for any):", cfg.auth_audience)

    cfg.save()
    print(f"\nConfig written to {default_config_path()}")
    print("Add ingestion directories with `rag dir add <path> [--watch]`.")


# ------------------------------------------------------------------ dir
def cmd_dir_add(args: argparse.Namespace) -> None:
    cfg = _load_config()
    p = Path(args.path).expanduser()
    if not p.is_dir():
        print(f"Not a directory: {p}", file=sys.stderr)
        sys.exit(1)
    resolved = str(p.resolve())
    if any(d.path == resolved for d in cfg.dirs):
        print(f"Already configured: {resolved}")
        return
    cfg.add_dir(resolved, watch=args.watch)
    cfg.save()
    print(f"Added ingestion dir {resolved} (watch={'yes' if args.watch else 'no'})")

    if cfg.dirs:
        store = Store(cfg.db_path_obj)
        providers = _providers(cfg)
        ocr = _ocr(cfg)
        from .ingest import ingest_dir

        print("Initial sync...")
        for d in cfg.dirs:
            results = ingest_dir(d.path, store, cfg, providers, ocr)
            for r in results:
                if r.error:
                    print(f"  {r.path}: error {r.error}", file=sys.stderr)
                elif not r.skipped:
                    print(f"  {r.path}: {r.chunks} chunks")
        store.close()
        print("Done. Run `rag daemon` to keep the index up to date.")


def cmd_dir_remove(args: argparse.Namespace) -> None:
    cfg = _load_config()
    p = str(Path(args.path).expanduser().resolve())
    if not cfg.remove_dir(p):
        print(f"Not configured: {p}", file=sys.stderr)
        sys.exit(1)
    cfg.save()
    print(f"Removed ingestion dir {p}")
    if cfg.dirs:
        store = Store(cfg.db_path_obj)
        for f in list(store.iter_files()):
            if not any(
                f["path"] == d.path or f["path"].startswith(d.path.rstrip("/") + "/")
                for d in cfg.dirs
            ):
                store.remove_file(f["path"])
        store.close()
        print("Removed files from the index that no longer belong to a configured dir.")


# ------------------------------------------------------------------ search
def _print_results(results) -> None:
    if not results:
        print("No results.")
        return
    for i, r in enumerate(results, 1):
        score = f"  score={r.score:.4f}" if r.score is not None else ""
        print(f"[{i}] {r.path}:{r.start_line}:{r.start_char}{score}")
        print("    " + r.text.replace("\n", "\n    "))


def cmd_search(args: argparse.Namespace) -> None:
    cfg = _load_config()
    store = Store(cfg.db_path_obj)
    providers = _providers(cfg)
    from .search import grep_files, search_bm25, search_embedding, search_late_interaction

    k = max(1, args.k)
    if args.grep:
        results = grep_files(store, args.query, k, dirs=[d.path for d in cfg.dirs])
    elif args.bm25:
        results = search_bm25(store, args.query, k)
    else:
        model = cfg.model_for(args.model)
        if model is None:
            print(f"Unknown model {args.model!r}. Available: {cfg.model_names()}", file=sys.stderr)
            sys.exit(1)
        prov = providers.get(model)
        if prov.late_interaction:
            results = search_late_interaction(store, providers, model, args.query, k)
        else:
            results = search_embedding(store, providers, model, args.query, k)
    if args.json:
        print(json.dumps([r.to_dict() for r in results]))
    else:
        _print_results(results)
    store.close()


# ------------------------------------------------------------------ status
def cmd_status(args: argparse.Namespace) -> None:
    cfg = _load_config()
    store = Store(cfg.db_path_obj)
    stats = store.index_stats()
    store.close()
    print(f"config: {default_config_path()}")
    print(f"llm:    {cfg.llm_url} (model {cfg.llm_model})")
    print(f"ocr:    {cfg.ocr_url} (model {cfg.ocr_model})")
    print(f"db:     {cfg.db_path_obj}")
    print(f"mcp:    http://{cfg.mcp_host}:{cfg.mcp_port}/mcp"
          + ("  [JWT auth]" if cfg.auth_jwks_uri else "  [no auth]"))
    print("dirs:")
    for d in cfg.dirs:
        flag = " (watch)" if d.watch else ""
        print(f"  - {d.path}{flag}")
    if not cfg.dirs:
        print("  (none)")
    print(f"index:  {stats['files']} files, {stats['chunks']} chunks, "
          f"{stats['bm25_terms']} bm25 terms, backend={stats['vec_backend']}")
    for model in cfg.embedding_models.values():
        slug = slugify(model)
        print(f"  model {model}: {stats['models'].get('vec_' + slug, 0)} vectors")


# ------------------------------------------------------------------ remove
def cmd_remove(args: argparse.Namespace) -> None:
    cfg = _load_config()
    store = Store(cfg.db_path_obj)
    target = str(Path(args.file).expanduser().resolve())
    key = None
    for f in store.iter_files():
        if f["path"] == target or str(Path(f["path"]).resolve()) == target:
            key = f["path"]
            break
    if key is None:
        print(f"Not in the index: {target}", file=sys.stderr)
        sys.exit(1)
    store.remove_file(key)
    print(f"Removed {key} from all indexes.")
    store.close()


# ------------------------------------------------------------------ mcp / daemon
def cmd_mcp(args: argparse.Namespace) -> None:
    cfg = _load_config()
    cfg.data_dir_path.mkdir(parents=True, exist_ok=True)
    store = Store(cfg.db_path_obj)
    providers = _providers(cfg)
    from .mcp_server import run_mcp_server

    run_mcp_server(cfg, store, providers)


def cmd_daemon(args: argparse.Namespace) -> None:
    cfg = _load_config()
    cfg.data_dir_path.mkdir(parents=True, exist_ok=True)
    store = Store(cfg.db_path_obj)
    providers = _providers(cfg)
    from .daemon import run_daemon

    run_daemon(cfg, store, providers)


# ------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag", description="Local RAG CLI (offline ingestion, search, MCP)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("install", help="interactive setup; writes the config file")
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("dir", help="manage ingestion directories")
    dsub = p.add_subparsers(dest="dir_command", required=True)
    d = dsub.add_parser("add", help="add an ingestion directory")
    d.add_argument("path")
    d.add_argument("--watch", action="store_true", help="watch for changes")
    d.set_defaults(func=cmd_dir_add)
    d = dsub.add_parser("remove", help="remove an ingestion directory")
    d.add_argument("path")
    d.set_defaults(func=cmd_dir_remove)

    p = sub.add_parser("search", help="search the index (human-facing, with citations)")
    p.add_argument("query")
    p.add_argument("--model", default="standard",
                   help="category key or model name (default: standard)")
    p.add_argument("-k", type=int, default=5, help="max results")
    p.add_argument("--bm25", action="store_true", help="use BM25 keyword search")
    p.add_argument("--grep", action="store_true", help="regex grep over raw files")
    p.add_argument("--json", action="store_true", help="JSON output")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("status", help="show config and index statistics")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("remove", help="remove one file from all indexes")
    p.add_argument("file")
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser("mcp", help="start the MCP server only")
    p.set_defaults(func=cmd_mcp)

    p = sub.add_parser("daemon", help="watcher + MCP server in one process")
    p.set_defaults(func=cmd_daemon)
    return parser


def main() -> None:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
