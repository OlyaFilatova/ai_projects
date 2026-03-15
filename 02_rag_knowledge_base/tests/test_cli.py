"""CLI smoke tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from rag_knb.cli import run_cli
from rag_knb.service import KnowledgeBaseService


def test_module_help_returns_success() -> None:
    """The CLI should print help successfully."""
    result = subprocess.run(
        [sys.executable, "-m", "rag_knb", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Local retrieval-augmented knowledge base." in result.stdout


def test_status_command_returns_bootstrap_message() -> None:
    """The bootstrap CLI should expose a simple status command."""
    result = subprocess.run(
        [sys.executable, "-m", "rag_knb", "status"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "ready but empty" in result.stdout


def test_ingest_command_reports_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI should report ingest counts after loading documents."""
    source_path = tmp_path / "notes.txt"
    source_path.write_text("Cats chase strings.", encoding="utf-8")
    service = KnowledgeBaseService()

    exit_code = run_cli(["ingest", str(source_path)], service=service)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Ingested 1 document(s)" in captured.out
    assert service.documents


def test_ingest_command_persists_to_data_dir_when_requested(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should save a persisted knowledge base when --data-dir is provided."""
    source_path = tmp_path / "notes.txt"
    source_path.write_text("Cats nap in warm windows.", encoding="utf-8")
    data_dir = tmp_path / "kb-data"

    ingest_exit_code = run_cli(["ingest", "--data-dir", str(data_dir), str(source_path)])
    ingest_output = capsys.readouterr()
    ask_exit_code = run_cli(
        ["ask", "--data-dir", str(data_dir), "Where do cats nap?"],
        service=KnowledgeBaseService(),
    )
    ask_output = capsys.readouterr()

    assert ingest_exit_code == 0
    assert "Ingested 1 document(s)" in ingest_output.out
    assert ask_exit_code == 0
    assert "warm windows" in ask_output.out.lower()


def test_ask_command_returns_grounded_answer_after_ingest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should answer from the shared service layer after ingest."""
    source_path = tmp_path / "guide.txt"
    source_path.write_text("The museum opens at nine.", encoding="utf-8")
    service = KnowledgeBaseService()
    run_cli(["ingest", str(source_path)], service=service)

    exit_code = run_cli(["ask", "When does the museum open?"], service=service)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Grounded answer:" in captured.out
    assert "museum opens at nine" in captured.out.lower()


def test_ask_command_reports_empty_knowledge_base(capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI should distinguish an empty knowledge base from a no-match query."""
    exit_code = run_cli(["ask", "What is indexed?"], service=KnowledgeBaseService())
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "knowledge base is empty" in captured.out.lower()


def test_ask_command_reports_no_match(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI should report no-match cases clearly."""
    source_path = tmp_path / "history.txt"
    source_path.write_text("Ancient Rome built roads.", encoding="utf-8")
    service = KnowledgeBaseService()
    run_cli(["ingest", str(source_path)], service=service)

    exit_code = run_cli(["ask", "How do volcanoes erupt?"], service=service)
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "No grounded answer found" in captured.out


def test_ask_command_can_ingest_markdown_inline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The ask command should support Markdown documents without prior ingest state."""
    source_path = tmp_path / "guide.md"
    source_path.write_text("# Birds\n\nRobins build nests.", encoding="utf-8")

    exit_code = run_cli(
        ["ask", "--document", str(source_path), "Which birds build nests?"],
        service=KnowledgeBaseService(),
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "robins build nests" in captured.out.lower()


def test_ask_command_answers_docs_style_cat_question(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The README-style cat question should produce a grounded answer by default."""
    source_path = tmp_path / "cats.txt"
    source_path.write_text(
        "Cats are independent but affectionate on their terms.\nCats are energetic during the day.",
        encoding="utf-8",
    )

    exit_code = run_cli(
        ["ask", "--document", str(source_path), "What does the document say about cats?"],
        service=KnowledgeBaseService(),
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Grounded answer:" in captured.out
    assert "cats are independent" in captured.out.lower()


def test_ask_command_returns_focused_supporting_sentences(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI answers should prefer concise supporting sentences over whole chunk dumps."""
    source_path = tmp_path / "notes.txt"
    source_path.write_text(
        (
            "Cats nap in warm sunlight. "
            "Dogs patrol the yard. "
            "Cats also scratch old sofas when bored."
        ),
        encoding="utf-8",
    )

    exit_code = run_cli(
        ["ask", "--document", str(source_path), "Where do cats nap?"],
        service=KnowledgeBaseService(),
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Cats nap in warm sunlight. [notes:0]" in captured.out
    assert "Dogs patrol the yard." not in captured.out


def test_ask_command_supports_verbose_answer_verbosity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI callers should be able to request a richer grounded answer."""
    source_path = tmp_path / "notes.txt"
    source_path.write_text(
        "Cats nap in warm sunlight. Cats scratch old sofas. Cats watch birds from the window.",
        encoding="utf-8",
    )

    exit_code = run_cli(
        [
            "ask",
            "--answer-verbosity",
            "verbose",
            "--document",
            str(source_path),
            "What do cats do?",
        ],
        service=KnowledgeBaseService(),
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Cats nap in warm sunlight. [notes:0]" in captured.out
    assert "Cats scratch old sofas. [notes:0]" in captured.out


def test_ask_command_rejects_blank_question(capsys: pytest.CaptureFixture[str]) -> None:
    """Blank questions should fail with a clear validation message."""
    exit_code = run_cli(["ask", "   "], service=KnowledgeBaseService())
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Question cannot be empty." in captured.out


def test_ask_command_can_load_persisted_data_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI should answer from a persisted data directory without source files."""
    source_path = tmp_path / "travel.txt"
    source_path.write_text("Trains leave from platform seven.", encoding="utf-8")
    data_dir = tmp_path / "kb-data"
    service = KnowledgeBaseService()
    service.ingest_paths([source_path])
    service.save(data_dir)

    exit_code = run_cli(
        ["ask", "--data-dir", str(data_dir), "Which platform do trains leave from?"],
        service=KnowledgeBaseService(),
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "platform seven" in captured.out.lower()


def test_ask_command_can_show_diagnostics(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI diagnostics should be available on demand without cluttering the default path."""
    source_path = tmp_path / "diagnostics.txt"
    source_path.write_text("Saturn has rings.", encoding="utf-8")

    exit_code = run_cli(
        ["ask", "--document", str(source_path), "--show-diagnostics", "What has rings?"],
        service=KnowledgeBaseService(),
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    normalized_output = captured.out.replace(
        f'"retrieval_duration_ms": {service_time(captured.out)}',
        '"retrieval_duration_ms": "<timing>"',
    )
    answer_text, diagnostics_text = normalized_output.split("Diagnostics:\n", maxsplit=1)
    diagnostics = json.loads(diagnostics_text)

    assert answer_text == "Grounded answer:\n- Saturn has rings. [diagnostics:0]\nMatches: 1\n"
    assert diagnostics["match_count"] == 1
    assert diagnostics["matched_document_ids"] == ["diagnostics"]
    assert diagnostics["matches"][0]["chunk_id"] == "diagnostics:0"
    assert diagnostics["matches"][0]["snippet"] == "Saturn has rings."
    assert diagnostics["original_question"] == "What has rings?"
    assert diagnostics["rewritten_question"] == "What has rings"
    assert diagnostics["retrieval_queries"] == ["What has rings"]
    assert diagnostics["retrieval_duration_ms"] == "<timing>"


def service_time(output: str) -> str:
    """Extract the timing suffix from CLI diagnostics output for normalization."""
    prefix = '"retrieval_duration_ms": '
    start_index = output.index(prefix) + len(prefix)
    end_index = output.index("\n", start_index)
    return output[start_index:end_index]


def test_ask_command_can_filter_by_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI metadata filters should narrow the retrieval set."""
    cats_path = tmp_path / "cats.txt"
    cats_path.write_text("Cats nap on blankets.", encoding="utf-8")
    dogs_path = tmp_path / "dogs.txt"
    dogs_path.write_text("Dogs nap by doors.", encoding="utf-8")

    exit_code = run_cli(
        [
            "ask",
            "--document",
            str(cats_path),
            "--document",
            str(dogs_path),
            "--filter",
            "file_name=cats.txt",
            "Cats nap",
        ],
        service=KnowledgeBaseService(),
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Cats nap on blankets." in captured.out


def test_ask_command_can_report_clarification_needed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI should surface clarification-needed responses for ambiguous questions."""
    cats_path = tmp_path / "cats.txt"
    cats_path.write_text("Cats are energetic and playful.", encoding="utf-8")
    dogs_path = tmp_path / "dogs.txt"
    dogs_path.write_text("Dogs are energetic and playful.", encoding="utf-8")

    exit_code = run_cli(
        [
            "ask",
            "--document",
            str(cats_path),
            "--document",
            str(dogs_path),
            "Which pet is playful?",
        ],
        service=KnowledgeBaseService(),
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Do you mean document 'cats.txt' or document 'dogs.txt'?" in captured.out


def test_list_and_remove_document_commands_use_persisted_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI document management commands should operate on persisted knowledge-base data."""
    source_path = tmp_path / "notes.txt"
    source_path.write_text("Notes about Saturn.", encoding="utf-8")
    data_dir = tmp_path / "kb-data"
    service = KnowledgeBaseService()
    service.ingest_paths([source_path])
    service.save(data_dir)

    list_exit_code = run_cli(["list-documents", "--data-dir", str(data_dir)])
    list_output = capsys.readouterr()
    remove_exit_code = run_cli(
        ["remove-document", "--data-dir", str(data_dir), "notes"],
    )
    remove_output = capsys.readouterr()

    assert list_exit_code == 0
    assert "notes" in list_output.out
    assert remove_exit_code == 0
    assert "Remaining documents: 0" in remove_output.out


def test_ingest_command_accepts_runtime_config_overrides(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI config flags should feed the runtime config layer."""
    source_path = tmp_path / "configurable.txt"
    source_path.write_text("Alpha beta gamma delta epsilon zeta.", encoding="utf-8")

    exit_code = run_cli(
        [
            "ingest",
            "--data-dir",
            str(tmp_path / "kb-data"),
            "--chunk-size",
            "8",
            "--chunk-overlap",
            "2",
            str(source_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "chunk(s)." in captured.out


def test_ingest_command_rejects_invalid_runtime_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI should surface config validation errors clearly."""
    source_path = tmp_path / "invalid.txt"
    source_path.write_text("Alpha beta gamma.", encoding="utf-8")

    exit_code = run_cli(
        [
            "ingest",
            "--chunk-size",
            "10",
            "--chunk-overlap",
            "10",
            str(source_path),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Chunk overlap must be smaller than chunk size." in captured.out


def test_ask_command_rejects_question_longer_than_configured_limit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI should surface question-length limit failures clearly."""
    source_path = tmp_path / "limits.txt"
    source_path.write_text("Cats nap in sunny spots.", encoding="utf-8")

    exit_code = run_cli(
        [
            "ask",
            "--document",
            str(source_path),
            "--max-question-length",
            "10",
            "This question is definitely too long.",
        ],
        service=KnowledgeBaseService(),
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "configured maximum of 10 characters" in captured.out


def test_ask_command_rejects_documents_outside_allowed_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI should surface allowed-root violations clearly."""
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    source_path = tmp_path / "outside.txt"
    source_path.write_text("Cats nap in sunny spots.", encoding="utf-8")

    exit_code = run_cli(
        [
            "ask",
            "--document",
            str(source_path),
            "--allowed-root",
            str(allowed_root),
            "Where do cats nap?",
        ],
        service=KnowledgeBaseService(),
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "outside the configured allowed root" in captured.out
