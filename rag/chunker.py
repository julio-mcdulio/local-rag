"""Structure-aware chunking with source coordinates.

- markdown  -> split by headings
- JSON      -> split by top-level records
- plain txt -> split by paragraphs

Units larger than 1024 tokens are split into ~512-token windows with a
50-token overlap. Units smaller than 128 tokens are merged with a neighbor
(capped at 1024 tokens). Every chunk records source coordinates: file path,
start line, start character offset.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

MAX_TOKENS = 1024
MERGE_MIN_TOKENS = 128
WINDOW_TOKENS = 512
WINDOW_OVERLAP = 50

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class Chunk:
    text: str
    path: str
    start_line: int   # 1-based
    start_char: int   # 0-based char offset within the file
    token_count: int


def tokenize(text: str) -> list[tuple[int, int]]:
    """Return (start, end) char offsets of word-ish tokens."""
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def count_tokens(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def word_tokens(text: str) -> list[str]:
    """Word-ish tokens as strings (for BM25 postings and queries)."""
    return _TOKEN_RE.findall(text)


# ---------------------------------------------------------------------- units
def _split_markdown(text: str) -> list[tuple[int, str]]:
    """Return list of (start_char, unit_text) sections split by headings."""
    units: list[tuple[int, str]] = []
    lines = text.split("\n")
    pos = 0
    cur_start: int | None = None
    cur: list[str] = []

    def flush():
        nonlocal cur, cur_start
        while cur and not cur[-1].strip():
            cur.pop()
        if cur:
            units.append((cur_start, "\n".join(cur)))
            cur = []
            cur_start = None

    for line in lines:
        if re.match(r"^#{1,6}\s", line):
            flush()
            cur_start = pos
            cur = [line]
        else:
            if cur_start is None:
                cur_start = pos
            cur.append(line)
        pos += len(line) + 1  # +1 for newline
    flush()
    return units


def _skip_ws(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i].isspace():
        i += 1
    return i


def _split_json_top_level(text: str) -> list[tuple[int, str]]:
    """Split raw JSON text into top-level records with char offsets.

    Top-level array  -> each element is a record.
    Top-level object -> each "key": value pair is a record (scalar values
    included).
    Falls back to the whole text if the file is not valid JSON or the
    top level is a scalar. Offsets are produced by walking
    ``json.JSONDecoder.raw_decode`` so they match the source text exactly.
    """
    try:
        data = json.loads(text)
    except ValueError:
        return [(0, text)]

    dec = json.JSONDecoder()
    units: list[tuple[int, str]] = []
    n = len(text)

    def _try(i: int, want: str):
        """Return True if text[i] == want after skipping whitespace."""
        i = _skip_ws(text, i)
        return i < n and text[i] == want

    if isinstance(data, list):
        i = 0
        if not _try(i, "["):
            return [(0, text)]
        i += 1
        while True:
            if not _try(i, "]"):
                start = _skip_ws(text, i)
                _, i = dec.raw_decode(text, start)
                units.append((start, text[start:i]))
            if not _try(i, ","):
                break  # closing ']' or malformed input
            i += 1
    elif isinstance(data, dict):
        i = 0
        if not _try(i, "{"):
            return [(0, text)]
        i += 1
        while True:
            i = _skip_ws(text, i)
            if i < n and text[i] == '"':
                key_start = i
                i = dec.raw_decode(text, i)[1]  # key
                if not _try(i, ":"):
                    break
                i += 1
                start = _skip_ws(text, i)
                _, i = dec.raw_decode(text, start)  # value
                units.append((key_start, text[key_start:i]))
            if not _try(i, ","):
                break  # closing '}' or malformed input
            i += 1

    if not units:
        return [(0, text)]
    return units


def _split_paragraphs(text: str) -> list[tuple[int, str]]:
    units: list[tuple[int, str]] = []
    pos = 0
    para: list[str] = []
    para_start = 0

    def flush():
        nonlocal para, para_start
        if para:
            joined = "\n".join(para)
            units.append((para_start, joined))
            para = []
            para_start = 0

    for line in text.split("\n"):
        if line.strip():
            if not para:
                para_start = pos
            para.append(line)
        else:
            flush()
        pos += len(line) + 1
    flush()
    return units


def split_units(text: str, kind: str) -> list[tuple[int, str]]:
    if kind == "markdown":
        return _split_markdown(text)
    if kind == "json":
        return _split_json_top_level(text)
    return _split_paragraphs(text)


# ---------------------------------------------------------------------- assembly
def _split_large(unit: tuple[int, str], base_line: int) -> list[Chunk]:
    off, text = unit
    toks = tokenize(text)
    step = WINDOW_TOKENS - WINDOW_OVERLAP
    out: list[Chunk] = []
    i = 0
    while i < len(toks):
        # First window keeps the unit's leading chars (e.g. the "#" of a
        # heading); later windows start at their first token.
        seg_start = 0 if i == 0 else toks[i][0]
        end = min(i + WINDOW_TOKENS, len(toks))
        seg = text[seg_start : toks[end - 1][1]]
        line_delta = text[:seg_start].count("\n")
        out.append(
            Chunk(
                text=seg,
                path="",
                start_line=base_line + line_delta,
                start_char=off + seg_start,
                token_count=end - i,
            )
        )
        if end == len(toks):
            break
        i += step
    return out


def _merge_small(units: list[tuple[int, str]]) -> list[tuple[int, str]]:
    merged: list[tuple[int, str]] = []
    for off, text in units:
        if merged and count_tokens(merged[-1][1]) < MERGE_MIN_TOKENS:
            off0, prev = merged[-1]
            cand = prev + "\n\n" + text
            if count_tokens(cand) <= MAX_TOKENS:
                merged[-1] = (off0, cand)
                continue
        merged.append((off, text))
    return merged


def chunk_file(path: str, text: str, kind: str) -> list[Chunk]:
    if not text.strip():
        return []
    units = split_units(text, kind)
    units = _merge_small(units)
    chunks: list[Chunk] = []
    for off, unit in units:
        base_line = text[:off].count("\n") + 1
        chunks.extend(_split_large((off, unit), base_line))
    for c in chunks:
        c.path = path
    return chunks
