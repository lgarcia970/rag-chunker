import io
import json

import pytest

from rag_chunker.cli import main


def _write(tmp_path, text):
    path = tmp_path / "doc.md"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_reads_file_and_prints_jsonl(tmp_path, capsys):
    path = _write(tmp_path, "# Title\n\nHello world.\n")
    rc = main([path])
    assert rc == 0
    out = capsys.readouterr().out
    lines = out.rstrip("\n").split("\n")
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["heading_path"] == ["Title"]


def test_reads_stdin_when_path_is_dash(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("# T\n\nBody text.\n"))
    rc = main(["-"])
    assert rc == 0
    out = capsys.readouterr().out
    record = json.loads(out.strip())
    assert record["heading_path"] == ["T"]


def test_missing_file_reports_error_and_returns_1(tmp_path, capsys):
    missing = str(tmp_path / "does-not-exist.md")
    rc = main([missing])
    assert rc == 1
    err = capsys.readouterr().err
    assert "rag-chunker:" in err


def test_array_output_is_indented_json_list(tmp_path, capsys):
    path = _write(tmp_path, "# Title\n\nHello world.\n")
    rc = main([path, "--array"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    assert "\n" in out  # indented, not a single line


def test_no_heading_prefix_flag_is_forwarded(tmp_path, capsys):
    path = _write(tmp_path, "# Title\n\nHello world.\n")
    main([path])
    with_prefix = json.loads(capsys.readouterr().out.strip())

    main([path, "--no-heading-prefix"])
    without_prefix = json.loads(capsys.readouterr().out.strip())

    # heading_path is reported either way, but token_estimate is computed
    # over chunk.text, which drops the prefix when it is disabled
    assert with_prefix["token_estimate"] > without_prefix["token_estimate"]


def test_max_tokens_and_overlap_are_forwarded(tmp_path, capsys):
    text = "One two three. Four five six. Seven eight nine.\n"
    path = _write(tmp_path, text)
    rc = main([path, "--max-tokens", "9", "--overlap", "4"])
    assert rc == 0
    out = capsys.readouterr().out
    records = [json.loads(line) for line in out.rstrip("\n").split("\n")]
    assert len(records) == 2


def test_invalid_max_tokens_reports_error_and_returns_1(tmp_path, capsys):
    path = _write(tmp_path, "text\n")
    rc = main([path, "--max-tokens", "0"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "rag-chunker:" in err


def test_stats_flag_prints_summary_to_stderr(tmp_path, capsys):
    path = _write(tmp_path, "# Title\n\nHello world.\n")
    rc = main([path, "--stats"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "1 chunks" in err
    assert "oversized" in err


def test_stats_on_empty_document(tmp_path, capsys):
    path = _write(tmp_path, "")
    rc = main([path, "--stats"])
    assert rc == 0
    out, err = capsys.readouterr()
    assert out == "\n"
    assert "0 chunks | tokens min 0 avg 0 max 0 | 0 oversized" in err


def test_output_flag_writes_to_file_instead_of_stdout(tmp_path, capsys):
    path = _write(tmp_path, "# Title\n\nHello world.\n")
    out_path = tmp_path / "out.jsonl"
    rc = main([path, "-o", str(out_path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    content = out_path.read_text(encoding="utf-8")
    record = json.loads(content.strip())
    assert record["heading_path"] == ["Title"]


def test_no_path_argument_exits_nonzero():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code != 0
