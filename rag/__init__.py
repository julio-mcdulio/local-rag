"""rag: local RAG CLI tool.

Ingests local text files (markdown, PDF, JSON, plain text), precomputes
dense / late-interaction / BM25 indices per configured embedding model,
and serves semantic search through a CLI and an authenticated MCP server.
"""

__version__ = "0.1.0"
