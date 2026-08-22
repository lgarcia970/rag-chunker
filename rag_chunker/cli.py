"""Command-line entry point: read a markdown file, print chunks as JSON.

Kept thin on purpose -- argument parsing and output formatting only. All the
chunking logic lives in :mod:`rag_chunker.chunker`, so ``chunk_markdown`` and
the CLI can never disagree about what a chunk looks like.
"""

import argparse
import json
import sys

from .chunker import DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP, chunk_markdown, chunks_to_jsonl

__all__ = ["main"]


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="rag-chunker",
        description="Split a markdown document into token-budgeted chunks for retrieval.",
    )
    parser.add_argument("path", help="markdown file to chunk, or - to read stdin")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"chunk size ceiling, heading prefix included (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"trailing tokens repeated in the next chunk of a section (default: {DEFAULT_OVERLAP})",
    )
    parser.add_argument(
        "--no-heading-prefix",
        action="store_true",
        help="do not prepend the heading path to the chunk text",
    )
    parser.add_argument(
        "--array",
        action="store_true",
        help="emit one indented JSON array instead of JSON lines",
    )
    parser.add_argument("--stats", action="store_true", help="print a size summary to stderr")
    parser.add_argument("-o", "--output", metavar="PATH", help="write the result to a file (default: stdout)")
    return parser


def _read_source(path):
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _format_stats(chunks):
    if not chunks:
        return "0 chunks | tokens min 0 avg 0 max 0 | 0 oversized"
    tokens = [chunk.token_estimate for chunk in chunks]
    oversized = sum(1 for chunk in chunks if chunk.oversized)
    avg = round(sum(tokens) / len(tokens))
    return (
        f"{len(chunks)} chunks | tokens min {min(tokens)} avg {avg} max {max(tokens)} "
        f"| {oversized} oversized"
    )


def main(argv=None):
    args = _build_parser().parse_args(argv)

    try:
        text = _read_source(args.path)
    except OSError as exc:
        print(f"rag-chunker: {exc}", file=sys.stderr)
        return 1

    try:
        chunks = chunk_markdown(
            text,
            max_tokens=args.max_tokens,
            overlap=args.overlap,
            heading_prefix=not args.no_heading_prefix,
        )
    except ValueError as exc:
        print(f"rag-chunker: {exc}", file=sys.stderr)
        return 1

    if args.array:
        output = json.dumps([chunk.to_dict() for chunk in chunks], indent=2, ensure_ascii=False)
    else:
        output = chunks_to_jsonl(chunks)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
            handle.write("\n")
    else:
        print(output)

    if args.stats:
        print(_format_stats(chunks), file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
