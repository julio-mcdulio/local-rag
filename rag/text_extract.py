"""Text extraction per file type, with an OCR fallback for PDFs only.

Supported inputs: markdown, plain text / other text, JSON, PDF.
OCR (via a configured Ollama endpoint) is used ONLY for PDF pages whose
embedded text is missing or negligible. Other file types are never OCR'd.
"""

from __future__ import annotations

import base64
import binascii
import json
from pathlib import Path

import requests

#: PDF pages with less embedded text than this many characters are considered
#: "negligible" and are sent through the OCR pipeline.
OCR_MIN_CHARS_PER_PAGE = 20

#: Max page dimension / zoom used when rendering PDF pages to PNG for OCR.
OCR_MAX_DIM = 1600
OCR_ZOOM = 2.0

OCR_PROMPT = "Extract all visible text from this document image. Return only the text, formatted as markdown."


def is_pdf(path: Path) -> bool:
    return path.suffix.lower() in {".pdf"}


def is_json(path: Path) -> bool:
    return path.suffix.lower() in {".json"}


def is_markdown(path: Path) -> bool:
    return path.suffix.lower() in {".md", ".markdown"}


def file_kind(path: Path) -> str:
    if is_pdf(path):
        return "pdf"
    if is_json(path):
        return "json"
    if is_markdown(path):
        return "markdown"
    return "text"


def extract_text(path: Path, ocr) -> str:
    """Extract text from a file. `ocr` is an OcrClient (used only for PDFs)."""
    kind = file_kind(path)
    if kind == "pdf":
        return extract_pdf(path, ocr)
    return path.read_text(encoding="utf-8", errors="replace")


def extract_pdf(path: Path, ocr) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages_text: list[str] = []
    pages_blank: list[int] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages_text.append(text)
        if len(text.strip()) < OCR_MIN_CHARS_PER_PAGE:
            pages_blank.append(i)

    if not pages_blank:
        return "\n\n".join(pages_text)

    # OCR fallback: render the blank/negligible pages to images and call Ollama.
    from pypdfium2 import PdfDocument

    pdf = PdfDocument(str(path))
    ocr_parts: dict[int, str] = {}
    for i in pages_blank:
        img = pdf[i].render(scale=OCR_ZOOM).to_pil()
        if max(img.size) > OCR_MAX_DIM:
            img = img.resize(max(1, round(OCR_MAX_DIM * img.size[0] / img.size[1])),
                            max(1, round(OCR_MAX_DIM * img.size[1] / img.size[0])))
        buf = _pil_to_png_base64(img)
        ocr_parts[i] = ocr.run(buf)
    pdf.close()

    out: list[str] = []
    for i, text in enumerate(pages_text):
        if i in ocr_parts:
            out.append(f"[page {i + 1} OCR]\n{ocr_parts[i]}")
        else:
            out.append(text)
    return "\n\n".join(out)


def _pil_to_png_base64(img) -> str:
    import io

    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class OcrClient:
    """Client for an Ollama OCR model (e.g. maternion/LightOnOCR-2).

    Sends a base64 PNG to the Ollama chat endpoint and returns the extracted
    markdown text.
    """

    def __init__(self, url: str, model: str, timeout: float = 300.0):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def run(self, image_b64: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": OCR_PROMPT,
                    "images": [image_b64],
                }
            ],
        }
        resp = requests.post(
            f"{self.url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["message"]["content"]
        except (KeyError, TypeError):
            # Fallback: concatenate any content blocks.
            parts = []
            for msg in data.get("messages", []):
                content = msg.get("content")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    parts.extend(
                        b.get("text", "") for b in content if isinstance(b, dict)
                    )
            return "\n".join(p for p in parts if p)
