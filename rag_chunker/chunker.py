"""Assemble parsed markdown blocks into token-budgeted chunks for embedding.

A block-level split alone is not a chunker: a one-line paragraph and a
forty-line code block are both single blocks, so blocks still need packing
into chunks that respect a token budget, carry a heading-path prefix for
context, and overlap at their prose boundaries so a chunk boundary never
strands a sentence outside the section it belongs to.
"""

import json

from .markdown import parse_blocks
from .sentences import split_sentences
from .tokens import estimate_tokens

__all__ = ["Chunk", "DEFAULT_MAX_TOKENS", "DEFAULT_OVERLAP", "chunk_markdown", "chunks_to_jsonl"]

DEFAULT_MAX_TOKENS = 512
DEFAULT_OVERLAP = 64

# Code, tables and lists are emitted whole or not at all -- splitting a table
# off its header row or a fence in the middle produces a fragment nobody can
# use. A paragraph is the only block kind allowed to break apart, and only
# along sentence boundaries.
_ATOMIC_KINDS = frozenset({"code", "table", "list"})


class Chunk:
    """A budget-sized piece of a document, ready to embed.

    ``text`` is the heading-path prefix (unless disabled) followed by a
    blank line and ``body`` -- ``body`` alone otherwise. ``oversized`` is
    true when ``token_estimate`` exceeds the budget anyway, which only
    happens when a single atomic block or an unsplittable sentence is bigger
    than the budget on its own.
    """

    __slots__ = (
        "index",
        "text",
        "body",
        "heading_path",
        "start_line",
        "end_line",
        "token_estimate",
        "oversized",
    )

    def __init__(
        self, index, text, body, heading_path, start_line, end_line, token_estimate, oversized
    ):
        self.index = index
        self.text = text
        self.body = body
        self.heading_path = heading_path
        self.start_line = start_line
        self.end_line = end_line
        self.token_estimate = token_estimate
        self.oversized = oversized

    def __repr__(self):
        return (
            f"Chunk(index={self.index!r}, heading_path={self.heading_path!r}, "
            f"start_line={self.start_line}, end_line={self.end_line}, "
            f"token_estimate={self.token_estimate}, oversized={self.oversized})"
        )

    def to_dict(self):
        return {
            "index": self.index,
            "text": self.text,
            "heading_path": self.heading_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "token_estimate": self.token_estimate,
        }


def chunk_markdown(text, max_tokens=DEFAULT_MAX_TOKENS, overlap=DEFAULT_OVERLAP, heading_prefix=True):
    """Split ``text`` into a flat, ordered list of :class:`Chunk` objects.

    Blocks are grouped by the heading path in effect when they appear, so a
    chunk never spans a heading. Within a section, blocks are packed
    greedily up to ``max_tokens`` (prefix included); a paragraph too large
    to fit on its own is broken into sentences first. ``overlap`` repeats up
    to that many trailing tokens of prose at the start of the next chunk of
    the same section, and never carries across a heading.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")

    chunks = []
    heading_stack = []  # list of (level, text), one entry per open heading
    section_blocks = []

    def flush_section():
        if section_blocks:
            path = [heading_text for _, heading_text in heading_stack]
            chunks.extend(_chunk_section(path, section_blocks, max_tokens, overlap, heading_prefix))
            section_blocks.clear()

    for block in parse_blocks(text):
        if block.kind == "heading":
            flush_section()
            while heading_stack and heading_stack[-1][0] >= block.level:
                heading_stack.pop()
            heading_stack.append((block.level, block.text))
            continue
        section_blocks.append(block)
    flush_section()

    for i, chunk in enumerate(chunks):
        chunk.index = i
    return chunks


def chunks_to_jsonl(chunks):
    """Serialise ``chunks`` as newline-delimited JSON, one object per line."""
    return "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in chunks)


def _make_pieces(block, prefix, max_tokens):
    """Split one non-heading ``block`` into packing units.

    Code, tables and lists always come back as a single atomic piece. A
    paragraph is one piece too, unless it alone -- with the heading prefix
    it will carry -- would already bust the budget, in which case it is
    broken into sentences so packing can still make progress.
    """
    if block.kind in _ATOMIC_KINDS:
        return [(block.text, "atomic", block.start_line, block.end_line)]

    solo = f"{prefix}\n\n{block.text}" if prefix else block.text
    if estimate_tokens(solo) <= max_tokens:
        return [(block.text, "prose", block.start_line, block.end_line)]

    sentences = split_sentences(block.text)
    if not sentences:
        return [(block.text, "prose", block.start_line, block.end_line)]
    return [(sentence, "prose", block.start_line, block.end_line) for sentence in sentences]


def _render(parts):
    return "\n\n".join(piece_text for piece_text, _ in parts)


def _trailing_overlap(parts, overlap):
    """Up to ``overlap`` tokens of whole sentences from the prose in ``parts``."""
    prose = [piece_text for piece_text, kind in parts if kind == "prose"]
    if not prose:
        return None
    sentences = split_sentences("\n\n".join(prose))
    if not sentences:
        return None
    tail = []
    tokens = 0
    for sentence in reversed(sentences):
        sentence_tokens = estimate_tokens(sentence)
        if tail and tokens + sentence_tokens > overlap:
            break
        tail.insert(0, sentence)
        tokens += sentence_tokens
    return " ".join(tail) if tail else None


def _chunk_section(heading_path, blocks, max_tokens, overlap, heading_prefix):
    prefix = " > ".join(heading_path) if heading_prefix and heading_path else ""

    pieces = []
    for block in blocks:
        pieces.extend(_make_pieces(block, prefix, max_tokens))

    chunks = []
    parts = []
    start_line = None
    end_line = None

    def emit():
        body = _render(parts)
        text = f"{prefix}\n\n{body}" if prefix else body
        token_estimate = estimate_tokens(text)
        chunks.append(
            Chunk(
                index=0,
                text=text,
                body=body,
                heading_path=list(heading_path),
                start_line=start_line,
                end_line=end_line,
                token_estimate=token_estimate,
                oversized=token_estimate > max_tokens,
            )
        )

    for piece_text, piece_kind, piece_start, piece_end in pieces:
        candidate = parts + [(piece_text, piece_kind)]
        candidate_text = f"{prefix}\n\n{_render(candidate)}" if prefix else _render(candidate)
        if not parts or estimate_tokens(candidate_text) <= max_tokens:
            parts = candidate
            end_line = piece_end
            if start_line is None:
                start_line = piece_start
            continue

        emit()
        overlap_text = _trailing_overlap(parts, overlap) if overlap > 0 else None
        parts = []
        if overlap_text is not None:
            trial = [(overlap_text, "prose"), (piece_text, piece_kind)]
            trial_text = f"{prefix}\n\n{_render(trial)}" if prefix else _render(trial)
            if estimate_tokens(trial_text) <= max_tokens:
                parts = trial
        if not parts:
            parts = [(piece_text, piece_kind)]
        start_line = piece_start
        end_line = piece_end

    if parts:
        emit()

    return chunks
