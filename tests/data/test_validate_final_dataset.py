from pathlib import Path

import src.data.validate_final_dataset as validator


def make_row(utterance_id: str, speaker: str, split: str, text: str) -> dict:
    return {
        "id": utterance_id,
        "audio_path": "audio.flac",
        "text": text,
        "text_normalized": text,
        "language": "ko",
        "duration_sec": 2.0,
        "sample_rate": 16000,
        "channels": 1,
        "speaker_id": speaker,
        "speaker_gender": "f",
        "source": "zeroth",
        "split": split,
        "license": "CC-BY-4.0",
        "checksum_xxh3_128": utterance_id,
    }


def test_valid_dataset_has_no_split_leakage(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "audio.flac").touch()
    monkeypatch.setattr(validator, "PROJECT_ROOT", tmp_path)
    train = [make_row("train_1", "speaker_train", "train", "훈련 문장")]
    dev = [make_row("dev_1", "speaker_dev", "dev", "검증 문장")]
    test = [make_row("test_1", "speaker_test", "test", "평가 문장")]

    summary = validator.validate_dataset(train, dev, test, test, [])

    assert summary["valid"] is True
    assert summary["error_count"] == 0
    assert not any(summary["speaker_overlap"].values())
