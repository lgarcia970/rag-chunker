"""Markdown block parsing: headings, code, tables, lists, paragraphs.

A chunker needs to know where the safe cut points are -- block boundaries --
without understanding markdown as a whole document tree. This module walks
the source once and yields a flat, ordered list of blocks. Inline syntax
(links, emphasis, inline code) is left untouched, because a chunker never
needs to interpret it.
"""

import re

__all__ = ["Block", "parse_blocks"]

_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*))?$")
_HEADING_TRAILING_HASHES_RE = re.compile(r"(?:^|[ \t])#+[ \t]*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*(.*)$")
_TABLE_SEPARATOR_RE = re.compile(
    r"^ {0,3}\|?[ \t]*:?-+:?[ \t]*(\|[ \t]*:?-+:?[ \t]*)*\|?[ \t]*$"
)
_LIST_ITEM_RE = re.compile(r"^ {0,3}(?:[-*+]|\d{1,9}[.)])[ \t]+\S")
_LIST_CONTINUATION_RE = re.compile(r"^[ \t]+\S")


class Block:
    """One structural unit of a document.

    ``kind`` is one of ``"heading"``, ``"paragraph"``, ``"list"``, ``"code"``
    or ``"table"``. ``level`` is the heading depth (1-6) for headings and
    ``None`` otherwise. ``text`` is the block's source text, verbatim for
    code and tables, heading-marker-stripped for headings. Line numbers are
    1-based and inclusive.
    """

    __slots__ = ("kind", "text", "level", "start_line", "end_line")

    def __init__(self, kind, text, start_line, end_line, level=None):
        self.kind = kind
        self.text = text
        self.level = level
        self.start_line = start_line
        self.end_line = end_line

    def __repr__(self):
        return (
            f"Block(kind={self.kind!r}, level={self.level!r}, "
            f"start_line={self.start_line}, end_line={self.end_line})"
        )

    def __eq__(self, other):
        if not isinstance(other, Block):
            return NotImplemented
        return (
            self.kind == other.kind
            and self.text == other.text
            and self.level == other.level
            and self.start_line == other.start_line
            and self.end_line == other.end_line
        )


def _is_blank(line):
    return line.strip() == ""


def _heading_match(line):
    """Return ``(level, text)`` if ``line`` is an ATX heading, else ``None``."""
    match = _HEADING_RE.match(line)
    if not match:
        return None
    level = len(match.group(1))
    content = match.group(2) or ""
    content = _HEADING_TRAILING_HASHES_RE.sub("", content).strip()
    return level, content


def _fence_start(line):
    """Return ``(fence_char, fence_len)`` if ``line`` opens a code fence."""
    match = _FENCE_RE.match(line)
    if not match:
        return None
    marker = match.group(1)
    return marker[0], len(marker)


def _fence_end(line, fence_char, fence_len):
    match = _FENCE_RE.match(line)
    if not match:
        return False
    marker = match.group(1)
    if marker[0] != fence_char or len(marker) < fence_len:
        return False
    return match.group(2).strip() == ""


def _is_table_start(lines, i):
    if i + 1 >= len(lines):
        return False
    header, separator = lines[i], lines[i + 1]
    if _is_blank(header) or "|" not in header:
        return False
    return bool(_TABLE_SEPARATOR_RE.match(separator))


def _is_list_start(line):
    return bool(_LIST_ITEM_RE.match(line))


def _starts_new_block(lines, i):
    line = lines[i]
    return (
        _heading_match(line) is not None
        or _fence_start(line) is not None
        or _is_table_start(lines, i)
        or _is_list_start(line)
    )


def parse_blocks(text):
    """Split ``text`` into a flat, ordered list of :class:`Block` objects.

    Recognises ATX headings (``#`` through ``######``), fenced code
    (backtick and tilde, including a fence left unterminated to end of
    document), pipe tables with a separator row, bullet and ordered list
    runs, and paragraphs. Setext headings are not special-cased and read as
    ordinary paragraph text, since a document that relies on them is rare
    enough in technical docs that adding a rule for it is not worth the
    false positives it would create against ``---`` used as a divider.
    """
    if not text:
        return []
    lines = text.splitlines()
    n = len(lines)
    blocks = []
    i = 0
    while i < n:
        line = lines[i]
        if _is_blank(line):
            i += 1
            continue

        heading = _heading_match(line)
        if heading is not None:
            level, content = heading
            blocks.append(Block("heading", content, i + 1, i + 1, level=level))
            i += 1
            continue

        fence = _fence_start(line)
        if fence is not None:
            fence_char, fence_len = fence
            start = i
            j = i + 1
            while j < n and not _fence_end(lines[j], fence_char, fence_len):
                j += 1
            end = j if j < n else n - 1
            blocks.append(
                Block("code", "\n".join(lines[start : end + 1]), start + 1, end + 1)
            )
            i = end + 1
            continue

        if _is_table_start(lines, i):
            start = i
            j = i + 2
            while j < n and not _is_blank(lines[j]) and "|" in lines[j]:
                j += 1
            end = j - 1
            blocks.append(
                Block("table", "\n".join(lines[start : end + 1]), start + 1, end + 1)
            )
            i = end + 1
            continue

        if _is_list_start(line):
            start = i
            j = i + 1
            while j < n:
                if _is_blank(lines[j]):
                    k = j + 1
                    while k < n and _is_blank(lines[k]):
                        k += 1
                    if k < n and (
                        _is_list_start(lines[k]) or _LIST_CONTINUATION_RE.match(lines[k])
                    ):
                        j = k
                        continue
                    break
                if _is_list_start(lines[j]) or _LIST_CONTINUATION_RE.match(lines[j]):
                    j += 1
                    continue
                break
            end = j - 1
            while end > start and _is_blank(lines[end]):
                end -= 1
            blocks.append(
                Block("list", "\n".join(lines[start : end + 1]), start + 1, end + 1)
            )
            i = end + 1
            continue

        # paragraph: everything up to a blank line or the start of another block
        start = i
        j = i + 1
        while j < n and not _is_blank(lines[j]) and not _starts_new_block(lines, j):
            j += 1
        end = j - 1
        blocks.append(
            Block("paragraph", "\n".join(lines[start : end + 1]), start + 1, end + 1)
        )
        i = end + 1

    return blocks
