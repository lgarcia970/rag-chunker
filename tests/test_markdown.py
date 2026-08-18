from rag_chunker.markdown import parse_blocks


def kinds(blocks):
    return [b.kind for b in blocks]


def test_empty_input():
    assert parse_blocks("") == []
    assert parse_blocks("   \n\n\t\n") == []


def test_heading_levels_and_trailing_hashes():
    blocks = parse_blocks("# One\n## Two ##\n####### Seven\n")
    assert kinds(blocks) == ["heading", "heading", "paragraph"]
    assert blocks[0].level == 1
    assert blocks[0].text == "One"
    assert blocks[1].level == 2
    assert blocks[1].text == "Two"
    # more than six leading '#' is not a heading at all
    assert blocks[2].text == "####### Seven"


def test_heading_requires_space_after_hashes():
    blocks = parse_blocks("#no-space\n")
    assert kinds(blocks) == ["paragraph"]
    assert blocks[0].text == "#no-space"


def test_empty_heading():
    blocks = parse_blocks("###\n")
    assert kinds(blocks) == ["heading"]
    assert blocks[0].level == 3
    assert blocks[0].text == ""


def test_paragraph_line_numbers():
    blocks = parse_blocks("# Title\n\nFirst line.\nSecond line.\n")
    assert kinds(blocks) == ["heading", "paragraph"]
    para = blocks[1]
    assert para.text == "First line.\nSecond line."
    assert para.start_line == 3
    assert para.end_line == 4


def test_fenced_code_block_is_atomic():
    text = "```python\nprint('hi')\n```\n"
    blocks = parse_blocks(text)
    assert kinds(blocks) == ["code"]
    assert blocks[0].text == "```python\nprint('hi')\n```"
    assert blocks[0].start_line == 1
    assert blocks[0].end_line == 3


def test_unterminated_fence_runs_to_end_of_document():
    text = "```\nno closing fence here\nmore text\n"
    blocks = parse_blocks(text)
    assert kinds(blocks) == ["code"]
    assert blocks[0].end_line == 3


def test_fence_can_contain_a_shorter_fence():
    text = "````\n```\nstill inside\n````\n"
    blocks = parse_blocks(text)
    assert kinds(blocks) == ["code"]
    assert blocks[0].start_line == 1
    assert blocks[0].end_line == 4


def test_tilde_fence():
    text = "~~~\ncode\n~~~\n"
    blocks = parse_blocks(text)
    assert kinds(blocks) == ["code"]
    assert blocks[0].text == "~~~\ncode\n~~~"


def test_pipe_table_is_atomic():
    text = "| A | B |\n| - | - |\n| 1 | 2 |\n| 3 | 4 |\n"
    blocks = parse_blocks(text)
    assert kinds(blocks) == ["table"]
    assert blocks[0].start_line == 1
    assert blocks[0].end_line == 4


def test_table_ends_at_blank_line():
    text = "| A | B |\n| - | - |\n| 1 | 2 |\n\nAfter.\n"
    blocks = parse_blocks(text)
    assert kinds(blocks) == ["table", "paragraph"]
    assert blocks[0].end_line == 3
    assert blocks[1].text == "After."


def test_bullet_list_with_continuation():
    text = "- one\n- two\n  wrapped\n- three\n"
    blocks = parse_blocks(text)
    assert kinds(blocks) == ["list"]
    assert blocks[0].text == text.strip()
    assert blocks[0].end_line == 4


def test_ordered_list():
    text = "1. first\n2. second\n3. third\n"
    blocks = parse_blocks(text)
    assert kinds(blocks) == ["list"]
    assert blocks[0].start_line == 1
    assert blocks[0].end_line == 3


def test_list_followed_by_paragraph():
    text = "- one\n- two\n\nNot part of the list.\n"
    blocks = parse_blocks(text)
    assert kinds(blocks) == ["list", "paragraph"]
    assert blocks[0].end_line == 2
    assert blocks[1].text == "Not part of the list."


def test_setext_underline_is_read_as_paragraph_text():
    text = "Title\n=====\n\nBody\n"
    blocks = parse_blocks(text)
    assert kinds(blocks) == ["paragraph", "paragraph"]
    assert blocks[0].text == "Title\n====="
    assert blocks[1].text == "Body"


def test_paragraph_stops_at_next_block_without_blank_line():
    text = "Some intro text\n# Heading right after\n"
    blocks = parse_blocks(text)
    assert kinds(blocks) == ["paragraph", "heading"]
    assert blocks[0].text == "Some intro text"
    assert blocks[1].text == "Heading right after"


def test_mixed_document():
    text = (
        "# Runbook\n"
        "\n"
        "Intro paragraph.\n"
        "\n"
        "## Checks\n"
        "\n"
        "| Check | Status |\n"
        "| --- | --- |\n"
        "| disk | ok |\n"
        "\n"
        "- step one\n"
        "- step two\n"
        "\n"
        "```bash\n"
        "echo done\n"
        "```\n"
    )
    blocks = parse_blocks(text)
    assert kinds(blocks) == [
        "heading",
        "paragraph",
        "heading",
        "table",
        "list",
        "code",
    ]
    assert blocks[0].level == 1
    assert blocks[2].level == 2
    assert blocks[-1].text == "```bash\necho done\n```"
