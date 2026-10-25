import pytest

from src.data.prepare_librispeech_transcript_qc import (
    ReviewQueueError,
    merge_repaired_qc,
    select_review_rows,
)


def row(identifier: str, split: str, reasons: list[str], warnings: list[str]) -> dict:
    return {
        "id": identifier,
        "split": split,
        "audio_qc": {
            "qc_status": "review" if warnings else "pass",
            "qc_reasons": reasons,
            "qc_warnings": warnings,
        },
    }


def test_merge_replaces_dc_row_and_selects_only_train_low_snr() -> None:
    original = [
        row("a", "train", ["excessive_dc_offset"], ["low_estimated_snr"]),
        row("b", "train", [], ["low_estimated_snr"]),
        row("c", "test", [], ["low_estimated_snr"]),
    ]
    repaired = [row("a", "train", [], [])]

    merged = merge_repaired_qc(original, repaired)
    selected = select_review_rows(merged, "low_estimated_snr", "train")

    assert [item["id"] for item in selected] == ["b"]
    assert merged[0]["audio_qc"]["qc_reasons"] == []


def test_merge_refuses_unflagged_replacement() -> None:
    original = [row("a", "train", [], [])]
    repaired = [row("a", "train", [], [])]

    with pytest.raises(ReviewQueueError, match="not originally flagged"):
        merge_repaired_qc(original, repaired)
