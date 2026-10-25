from pathlib import Path

import pytest

from src.data.librispeech_adapter import (
    LibriSpeechDataError,
    build_rows,
    build_summary,
)


def make_source(tmp_path: Path) -> Path:
    root = tmp_path / "LibriSpeech"
    (root / "SPEAKERS.TXT").parent.mkdir(parents=True)
    (root / "SPEAKERS.TXT").write_text(
        "; comment\n"
        ";ID |SEX| SUBSET |MINUTES| NAME\n"
        "19 | F | train-clean-100 | 1.0 | Reader One\n"
        "20 | M | dev-clean | 1.0 | Reader Two\n"
        "21 | F | test-clean | 1.0 | Reader Three\n",
        encoding="utf-8",
    )
    fixtures = {
        "train-clean-100": ("19", "198"),
        "dev-clean": ("20", "199"),
        "test-clean": ("21", "200"),
    }
    for subset, (speaker, chapter) in fixtures.items():
        chapter_dir = root / subset / speaker / chapter
        chapter_dir.mkdir(parents=True)
        utterance_id = f"{speaker}-{chapter}-0000"
        (chapter_dir / f"{speaker}-{chapter}.trans.txt").write_text(
            f"{utterance_id}  HELLO   WORLD  \n", encoding="utf-8"
        )
        (chapter_dir / f"{utterance_id}.flac").touch()
    return root


def inventory_row(subset: str, speaker: str, chapter: str, duration: float) -> dict:
    utterance_id = f"{speaker}-{chapter}-0000"
    return {
        "source": subset,
        "relative_path": f"{subset}/{speaker}/{chapter}/{utterance_id}.flac",
        "status": "ok",
        "duration_sec": duration,
        "sample_rate": 16000,
        "channels": 1,
        "checksum_xxh3_128": f"checksum-{utterance_id}",
    }


def test_build_rows_joins_metadata_and_preserves_official_splits(
    tmp_path: Path,
) -> None:
    root = make_source(tmp_path)
    inventory = [
        inventory_row("train-clean-100", "19", "198", 2.5),
        inventory_row("dev-clean", "20", "199", 0.8),
        inventory_row("test-clean", "21", "200", 31.0),
    ]

    rows = build_rows(root, inventory, min_duration_sec=1.0, max_duration_sec=30.0)

    assert len(rows) == 3
    by_split = {row["split"]: row for row in rows}
    train = by_split["train"]
    assert train["id"] == "librispeech_19-198-0000"
    assert train["speaker_id"] == "librispeech_19"
    assert train["speaker_gender"] == "f"
    assert train["language"] == "en"
    assert train["text"] == "HELLO   WORLD  "
    assert train["text_normalized"] == "HELLO WORLD"
    assert train["qc_status"] == "accepted"
    assert by_split["dev"]["qc_reasons"] == ["duration_too_short"]
    assert by_split["test"]["qc_reasons"] == ["duration_too_long"]

    summary = build_summary(rows, 1.0, 30.0)
    assert summary["accepted_utterances"] == 1
    assert summary["quarantined_utterances"] == 2
    assert summary["speaker_overlap"] == {
        "train_dev": 0,
        "train_test": 0,
        "dev_test": 0,
    }


def test_audio_without_transcript_fails_closed(tmp_path: Path) -> None:
    root = make_source(tmp_path)
    extra = root / "train-clean-100/19/198/19-198-9999.flac"
    extra.touch()
    inventory = [
        inventory_row("train-clean-100", "19", "198", 2.5),
        inventory_row("dev-clean", "20", "199", 2.5),
        inventory_row("test-clean", "21", "200", 2.5),
        {
            **inventory_row("train-clean-100", "19", "198", 2.5),
            "relative_path": "train-clean-100/19/198/19-198-9999.flac",
        },
    ]

    with pytest.raises(LibriSpeechDataError, match="Audio/transcript mismatch"):
        build_rows(root, inventory, min_duration_sec=1.0, max_duration_sec=30.0)
