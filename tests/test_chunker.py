"""Chunker tests: structure-aware splitting, coordinates, windowing, merging."""

import json

from rag.chunker import (
    MAX_TOKENS,
    MERGE_MIN_TOKENS,
    WINDOW_OVERLAP,
    WINDOW_TOKENS,
    chunk_file,
    count_tokens,
    split_units,
)

# ------------------------------------------------------------------ markdown
def test_markdown_splits_by_headings():
    text = "# A\nalpha text\n## B\nbeta text\ngamma tail\n"
    units = split_units(text, "markdown")
    assert units[0][1].startswith("# A\nalpha text")
    assert units[1][1].startswith("## B\nbeta text\ngamma tail")
    # offsets point at the heading lines
    assert text[units[0][0]] == "#"
    assert text[units[1][0]] == "#"
    assert units[1][0] == text.index("## B")


def test_markdown_no_headings_single_unit():
    text = "hello\nworld\n"
    units = split_units(text, "markdown")
    assert len(units) == 1
    assert units[0][1] == text.rstrip("\n") or units[0][1] == "hello\nworld"

# ------------------------------------------------------------------ JSON
def test_json_top_level_array_one_unit_per_element():
    records = [
        {"id": 1, "name": "alpha"},
        {"id": 2, "name": "beta", "tags": ["x", "y"]},
        [1, 2, 3],
    ]
    text = json.dumps(records, indent=2)
    units = split_units(text, "json")
    # Regression: previously the whole array was one giant unit.
    assert len(units) == len(records)
    for (off, unit), rec in zip(units, records):
        assert json.loads(unit) == rec
        # char offsets match the source text exactly
        assert text[off : off + len(unit)] == unit


def test_json_top_level_object_includes_scalar_values():
    text = json.dumps(
        {"a": 1, "b": "hello world", "c": [1, 2], "d": {"nested": True}},
        indent=2,
    )
    units = split_units(text, "json")
    # Regression: scalar key/value pairs were dropped before the rewrite.
    assert len(units) == 4
    values = [json.loads(u.split(":", 1)[1]) for _, u in units]
    assert values == [1, "hello world", [1, 2], {"nested": True}]
    for off, unit in units:
        assert text[off : off + len(unit)] == unit


def test_json_invalid_falls_back_to_whole_text():
    text = "{not valid json"
    units = split_units(text, "json")
    assert units == [(0, text)]


def test_json_scalar_top_level_falls_back():
    units = split_units("42", "json")
    assert units == [(0, "42")]

# ------------------------------------------------------------------ paragraphs
def test_paragraph_split():
    text = "one\n\nthree\nfour\n\n"
    units = split_units(text, "plain")
    assert [u[1] for u in units] == ["one", "three\nfour"]
    assert units[1][0] == text.index("three")

# ------------------------------------------------------------------ windowing
def test_large_unit_windowed_with_overlap():
    # 2000-token unit -> windows of ~512 with 50 overlap.
    text = " ".join(f"w{i}" for i in range(2000))
    chunks = chunk_file("f.txt", text, "plain")
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= WINDOW_TOKENS
    assert chunks[0].start_char == 0
    # overlap: consecutive windows must start strictly after the previous one
    for a, b in zip(chunks, chunks[1:]):
        assert b.start_char > a.start_char
    # window size check on a clean case
    toks = count_tokens(text)
    expected = -(-(toks - WINDOW_OVERLAP) // (WINDOW_TOKENS - WINDOW_OVERLAP))
    assert len(chunks) == expected


def test_small_units_merged_and_capped():
    # 3 tiny units should merge pairwise within the token cap.
    text = "a\n\nb\n\nc\n"
    chunks = chunk_file("f.txt", text, "plain")
    assert len(chunks) == 1
    assert count_tokens(chunks[0].text) == 3


def test_merge_respects_cap():
    # small unit followed by a large one: candidate exceeds the 1024 cap,
    # so they stay separate.
    small = "tiny\n"
    big = " ".join(f"t{i}" for i in range(1024))
    text = small + "\n" + big + "\n"
    units = split_units(text, "plain")
    assert len(units) == 2
    chunks = chunk_file("f.txt", text, "plain")
    # first unit stays small; second is windowed into 512-token chunks
    assert count_tokens(chunks[0].text) == 1
    assert all(c.token_count <= WINDOW_TOKENS for c in chunks[1:])
    assert sum(count_tokens(c.text) for c in chunks[1:]) >= 1000

# ------------------------------------------------------------------ coordinates
def test_chunk_coordinates_match_source():
    text = "# H\n" + "x" * 10 + "\n" + "second line\n"
    chunks = chunk_file("doc.md", text, "markdown")
    assert len(chunks) == 1
    c = chunks[0]
    assert c.path == "doc.md"
    assert c.start_line == 1
    assert c.start_char == 0
    assert text[c.start_char : c.start_char + len(c.text)] == c.text


def test_empty_text_no_chunks():
    assert chunk_file("f.txt", "   \n  ", "plain") == []


def test_token_counts_consistent():
    text = "foo bar\nbaz\n"
    chunks = chunk_file("f.txt", text, "plain")
    assert chunks[0].token_count == count_tokens(chunks[0].text) == 3
