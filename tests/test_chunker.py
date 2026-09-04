import json

import pytest

from rag_chunker.chunker import chunk_markdown, chunks_to_jsonl


def test_empty_input():
    assert chunk_markdown("") == []


def test_heading_with_no_body_produces_no_chunk():
    assert chunk_markdown("# Just a heading\n") == []


def test_single_small_section_with_heading_prefix():
    text = "# Title\n\nHello world.\n"
    chunks = chunk_markdown(text)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.index == 0
    assert chunk.heading_path == ["Title"]
    assert chunk.body == "Hello world."
    assert chunk.text == "Title\n\nHello world."
    assert chunk.start_line == 3
    assert chunk.end_line == 3
    assert chunk.oversized is False
    assert chunk.token_estimate == 6


def test_heading_prefix_disabled():
    text = "# Title\n\nHello world.\n"
    chunks = chunk_markdown(text, heading_prefix=False)
    assert len(chunks) == 1
    assert chunks[0].text == chunks[0].body == "Hello world."


def test_heading_stack_tracks_nesting_and_sibling_replacement():
    text = (
        "# A\n\nfoo\n\n"
        "## B\n\nbar\n\n"
        "### C\n\nbaz\n\n"
        "## D\n\nqux\n"
    )
    chunks = chunk_markdown(text)
    assert [c.heading_path for c in chunks] == [
        ["A"],
        ["A", "B"],
        ["A", "B", "C"],
        ["A", "D"],
    ]
    assert [c.body for c in chunks] == ["foo", "bar", "baz", "qux"]
    assert [c.index for c in chunks] == [0, 1, 2, 3]


def test_list_block_is_never_split_and_can_be_oversized():
    text = "- item one\n- item two\n- item three\n"
    chunks = chunk_markdown(text, max_tokens=1, overlap=0)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.body == "- item one\n- item two\n- item three"
    assert chunk.oversized is True


def test_table_block_is_never_split():
    text = "| A | B |\n| - | - |\n| 1 | 2 |\n"
    chunks = chunk_markdown(text, max_tokens=1, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].body == "| A | B |\n| - | - |\n| 1 | 2 |"
    assert chunks[0].oversized is True


def test_code_block_is_never_split():
    text = "```\naaaa bbbb cccc\n```\n"
    chunks = chunk_markdown(text, max_tokens=2, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].body == "```\naaaa bbbb cccc\n```"
    assert chunks[0].token_estimate == 8
    assert chunks[0].oversized is True


def test_paragraph_splits_into_sentences_and_packs_with_overlap():
    text = "One two three. Four five six. Seven eight nine.\n"
    chunks = chunk_markdown(text, max_tokens=9, overlap=4)
    assert len(chunks) == 2
    assert chunks[0].body == "One two three.\n\nFour five six."
    assert chunks[0].token_estimate == 9
    assert chunks[0].oversized is False
    # the last sentence of chunk 0 reappears at the start of chunk 1
    assert chunks[1].body == "Four five six.\n\nSeven eight nine."
    assert chunks[1].token_estimate == 9
    assert [c.index for c in chunks] == [0, 1]


def test_overlap_dropped_when_it_would_bust_the_budget():
    text = "One two three. Four five six. Seven eight nine.\n"
    chunks = chunk_markdown(text, max_tokens=6, overlap=3)
    # each sentence lands in its own chunk, no room for carried-over prose
    assert [c.body for c in chunks] == [
        "One two three.",
        "Four five six.",
        "Seven eight nine.",
    ]


def test_max_tokens_must_be_positive():
    with pytest.raises(ValueError):
        chunk_markdown("text", max_tokens=0)


def test_overlap_must_not_be_negative():
    with pytest.raises(ValueError):
        chunk_markdown("text", overlap=-1)


def test_overlap_must_be_smaller_than_max_tokens():
    with pytest.raises(ValueError):
        chunk_markdown("text", max_tokens=10, overlap=10)


def test_chunks_to_jsonl_round_trips_through_to_dict():
    text = "# Title\n\nHello world.\n"
    chunks = chunk_markdown(text)
    lines = chunks_to_jsonl(chunks).split("\n")
    assert len(lines) == len(chunks)
    for line, chunk in zip(lines, chunks):
        assert json.loads(line) == chunk.to_dict()


def test_chunks_to_jsonl_empty_list():
    assert chunks_to_jsonl([]) == ""
