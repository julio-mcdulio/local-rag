# RAG CLI — container image
#
# Entrypoint: `rag daemon` (file watcher + MCP server in one process).
# GPU: torch is installed from the CUDA 12.1 wheel index so the same image
# runs on GPU hosts (NVIDIA driver >= 525) and falls back to CPU otherwise.
# Ollama is deliberately NOT included: OCR/LLM endpoints are configured in
# ~/.rag/config.yaml and point at any local Ollama instance.

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# pypdfium2 (PDF rasterization for OCR fallback) needs these system libs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# CUDA wheels for torch (GPU). If no GPU is present the CUDA build
# transparently falls back to CPU at runtime.
RUN uv pip install --system --no-cache torch --default-index https://download.pytorch.org/whl/cu121

# The rest of the dependencies from PyPI, then the package itself.
# (torch is already satisfied, so `rag-cli` install does not reinstall it.)
COPY pyproject.toml ./
COPY rag ./rag
RUN uv pip install --system --no-cache .

RUN useradd --create-home --shell /bin/bash rag \
    && mkdir -p /home/rag/.rag /home/rag/data
USER rag
WORKDIR /home/rag

EXPOSE 8931

ENTRYPOINT ["rag", "daemon"]
