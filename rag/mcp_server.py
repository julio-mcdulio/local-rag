"""MCP server (streamable-HTTP transport) exposing the search tools.

External chat agents (e.g. pi) authenticate with a JWT issued by the
configured external IdP; the token is verified against the IdP JWKS
endpoints. When no IdP is configured, the server runs unauthenticated.

Tools:
- search_embedding(query, k)
- search_embedding_model(query, model, k)
- search_late_interaction(query, k)
- search_bm25(query, k)
- grep(pattern, k)
- status()
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from mcp.server import MCPServer
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from .config import Config
from .embeddings import ProviderRegistry
from .search import (
    grep_files,
    search_bm25,
    search_embedding,
    search_late_interaction,
)
from .store import Store

log = logging.getLogger("rag.mcp")

DEFAULT_SEARCH_K = 5
MAX_K = 50


class JwtAuth:
    """Verifies Bearer JWTs against the configured IdP JWKS endpoint."""

    def __init__(
        self,
        jwks_uri: str,
        audience: Optional[str] = None,
        issuer: Optional[str] = None,
    ):
        from jwt import PyJWKClient

        self._jwks_uri = jwks_uri
        self._audience = audience
        self._issuer = issuer
        self._client = PyJWKClient(jwks_uri)
        self._algorithms = ["RS256", "ES256", "PS256"]

    def verify(self, token: str) -> bool:
        import jwt as _jwt

        try:
            key = self._client.get_signing_key_from_jwt(token)
            kwargs: dict = {"algorithms": self._algorithms}
            if self._audience:
                kwargs["audience"] = self._audience
            if self._issuer:
                kwargs["issuer"] = self._issuer
            _jwt.decode(token, key.key, **kwargs)
            return True
        except Exception as e:
            log.debug("JWT verification failed: %s", e)
            return False


def _results_payload(results) -> dict:
    return {
        "count": len(results),
        "results": [
            {
                "path": r.path,
                "start_line": r.start_line,
                "start_char": r.start_char,
                "score": r.score,
                "text": r.text,
                "citation": r.citation(),
            }
            for r in results
        ],
    }


def build_app(config: Config, store: Store, providers: ProviderRegistry) -> Starlette:
    server = MCPServer("rag")

    # ------------------------------------------------------------------ tools
    def _k(k: int) -> int:
        return max(1, min(int(k), MAX_K))

    @server.tool("search_embedding")
    def tool_search_embedding(query: str, k: int = DEFAULT_SEARCH_K) -> dict:
        """Semantic (dense) search using the standard embedding model."""
        model = config.embedding_models["standard"]
        results = search_embedding(store, providers, model, query, _k(k))
        return _results_payload(results)

    @server.tool("search_embedding_model")
    def tool_search_embedding_model(
        query: str, model: str, k: int = DEFAULT_SEARCH_K
    ) -> dict:
        """Search with an explicit embedding model (category key or model name)."""
        m = config.model_for(model)
        if m is None:
            return {"error": f"unknown model {model!r}", "models": config.model_names()}
        if providers.get(m).late_interaction:
            results = search_late_interaction(store, providers, m, query, _k(k))
        else:
            results = search_embedding(store, providers, m, query, _k(k))
        return _results_payload(results)

    @server.tool("search_late_interaction")
    def tool_search_late_interaction(query: str, k: int = DEFAULT_SEARCH_K) -> dict:
        """Late-interaction (ColBERT MaxSim) search."""
        model = config.embedding_models["late-interaction"]
        results = search_late_interaction(store, providers, model, query, _k(k))
        return _results_payload(results)

    @server.tool("search_bm25")
    def tool_search_bm25(query: str, k: int = DEFAULT_SEARCH_K) -> dict:
        """BM25 keyword search over the precomputed postings index."""
        results = search_bm25(store, providers, query, _k(k))
        return _results_payload(results)

    @server.tool("grep")
    def tool_grep(pattern: str, k: int = DEFAULT_SEARCH_K) -> dict:
        """Regex grep over the raw original files (per-file, full-text)."""
        results = grep_files(store, pattern, _k(k), dirs=[d.path for d in config.dirs])
        return _results_payload(results)

    @server.tool("status")
    def tool_status() -> dict:
        return {
            "config": {
                "llm": {"url": config.llm_url, "model": config.llm_model},
                "embedding_models": dict(config.embedding_models),
                "ocr": {"model": config.ocr_model, "url": config.ocr_url},
                "dirs": [d.path for d in config.dirs],
            },
            "index": store.index_stats(),
        }

    # ------------------------------------------------------------------ app wiring
    # mcp 2.x: MCPServer.streamable_http_app() returns a Starlette app that
    # owns the StreamableHTTP session manager (stateless, no event store).
    mcp_app = server.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=False,
    )

    @asynccontextmanager
    async def lifespan(app: Starlette):
        # Mount does not run sub-app lifespans; run the MCP app's lifespan
        # (session manager) from the outer app.
        async with mcp_app.router.lifespan_context(mcp_app):
            yield

    # Simple health check (not behind auth)
    async def health(request) -> JSONResponse:
        return JSONResponse({"ok": True, "index": store.index_stats()})

    middleware: list[Middleware] = []
    auth = (
        JwtAuth(config.auth_jwks_uri, config.auth_audience, config.auth_issuer)
        if config.auth_jwks_uri
        else None
    )
    if auth is not None:
        class AuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                if request.url.path.startswith("/mcp"):
                    header = request.headers.get("authorization", "")
                    if not header.lower().startswith("bearer "):
                        return JSONResponse({"error": "unauthorized"}, status_code=401)
                    if not auth.verify(header[7:]):
                        return JSONResponse({"error": "invalid token"}, status_code=401)
                return await call_next(request)

        middleware.append(Middleware(AuthMiddleware))

    app = Starlette(
        routes=[Route("/health", health, methods=["GET"]), Mount("/", mcp_app)],
        middleware=middleware,
        lifespan=lifespan,
    )
    return app


def run_mcp_server(config: Config, store: Store, providers: ProviderRegistry) -> None:
    import uvicorn

    app = build_app(config, store, providers)
    log.info("Starting MCP server on %s:%d", config.mcp_host, config.mcp_port)
    uvicorn.run(app, host=config.mcp_host, port=config.mcp_port, log_level="info")
