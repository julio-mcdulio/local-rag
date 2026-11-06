"""Single daemon process: file watcher + MCP server.

The watcher runs on a background thread (initial sync, then polling);
the MCP server runs in the main thread. This is the container entrypoint
(`rag daemon`).

Runtime config changes (e.g. `rag dir add/remove` while the daemon runs) are
applied on SIGHUP: the config file is reloaded and the watcher rescans its
dirs without a restart. Send with `kill -HUP <pid>`.
"""

from __future__ import annotations

import logging
import signal
import threading

from .config import Config
from .embeddings import ProviderRegistry
from .mcp_server import run_mcp_server
from .store import Store
from .text_extract import OcrClient
from .watcher import Watcher

log = logging.getLogger("rag.daemon")


def run_daemon(config: Config, store: Store, providers: ProviderRegistry) -> None:
    ocr = OcrClient(config.ocr_url, config.ocr_model)
    watcher = Watcher(store, config, providers, ocr)

    def _on_sighup(signum, frame):
        try:
            new_cfg = Config.load()
        except Exception as e:
            log.warning("SIGHUP: config reload failed: %s", e)
            return
        config.dirs = new_cfg.dirs
        watcher.on_config_reload()
        log.info("SIGHUP: reloaded config (%d dirs); watcher will rescan", len(config.dirs))

    try:
        signal.signal(signal.SIGHUP, _on_sighup)
    except (ValueError, OSError) as e:
        log.debug("SIGHUP handler not registered: %s", e)

    t = threading.Thread(target=watcher.run_forever, name="rag-watcher", daemon=True)
    t.start()
    log.info("Watcher thread started; MCP server serving on %s:%d",
             config.mcp_host, config.mcp_port)
    run_mcp_server(config, store, providers)
