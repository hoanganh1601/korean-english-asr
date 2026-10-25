import json
from pathlib import Path

import pytest

from src.data.zeroth_adapter import ZerothDataError, build_rows, build_summary


def make_source(tmp_path: Path) -> Path:
    root = tmp_path / "zeroth"
    speaker_dir = root / "train_data_01" / "003" / "106"
    speaker_dir.mkdir(parents=True)
    (root / "AUDIO_INFO").write_text(
        "SPEAKERID|NAME|SEX|SCRIPTID|DATASET\n"
        "106|KimJiWon|f|003|train_data_01\n",
        encoding="utf-8",
    )
    (speaker_dir / "106_003.trans.txt").write_text(
        "106_003_0001  안녕   하세요  \n"
        "106_003_0002 두 번째 문장\n",
        encoding="utf-8",
    )
    (speaker_dir / "106_003_0001.flac").touch()
    (speaker_dir / "106_003_0002.flac").touch()
    return root


def inventory_row(utterance_id: str, duration: float) -> dict:
    return {
        "source": "zeroth",
        "relative_path": (
            f"zeroth/train_data_01/003/106/{utterance_id}.flac"
        ),
        "status": "ok",
        "duration_sec": duration,
        "sample_rate": 16000,
        "channels": 1,
        "checksum_xxh3_128": f"checksum-{utterance_id}",
    }


def test_build_rows_joins_metadata_and_quarantines_duration(tmp_path: Path) -> None:
    root = make_source(tmp_path)
    inventory = [
        inventory_row("106_003_0001", 2.5),
        inventory_row("106_003_0002", 31.0),
    ]

    rows = build_rows(root, inventory, min_duration_sec=1.0, max_duration_sec=30.0)

    assert len(rows) == 2
    first = rows[0]
    assert first["id"] == "zeroth_106_003_0001"
    assert first["speaker_id"] == "zeroth_106"
    assert first["speaker_gender"] == "f"
    assert first["text"] == "안녕   하세요  "
    assert first["text_normalized"] == "안녕 하세요"
    assert first["qc_status"] == "accepted"
    assert rows[1]["qc_reasons"] == ["duration_too_long"]

    summary = build_summary(rows, 1.0, 30.0)
    assert summary["accepted_utterances"] == 1
    assert summary["quarantined_utterances"] == 1
    assert summary["quarantine_reasons"] == {"duration_too_long": 1}
    assert summary["speakers_by_split"] == {"train": 1}


def test_audio_without_transcript_fails_closed(tmp_path: Path) -> None:
    root = make_source(tmp_path)
    extra_audio = root / "train_data_01/003/106/106_003_9999.flac"
    extra_audio.touch()
    inventory = [
        inventory_row("106_003_0001", 2.5),
        inventory_row("106_003_0002", 3.5),
        inventory_row("106_003_9999", 4.5),
    ]

    with pytest.raises(ZerothDataError, match="Audio/transcript mismatch"):
        build_rows(root, inventory, min_duration_sec=1.0, max_duration_sec=30.0)
