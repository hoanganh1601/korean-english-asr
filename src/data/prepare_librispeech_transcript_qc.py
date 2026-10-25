"""Merge post-repair acoustic QC and prepare an English teacher-review queue."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORIGINAL_QC = PROJECT_ROOT / "data/interim/librispeech/audio_qc.jsonl"
DEFAULT_REPAIRED_QC = (
    PROJECT_ROOT / "data/interim/librispeech/audio_qc.dc_repaired.jsonl"
)
DEFAULT_MERGED_OUTPUT = (
    PROJECT_ROOT / "data/interim/librispeech/audio_qc.final.jsonl"
)
DEFAULT_REVIEW_OUTPUT = (
    PROJECT_ROOT / "data/interim/librispeech/transcript_qc_candidates.jsonl"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "reports/data_qc/librispeech_transcript_qc_candidates.summary.json"
)


class ReviewQueueError(ValueError):
    """Raised when pre- and post-repair QC manifests cannot be merged safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge repaired QC and select low-SNR LibriSpeech train rows."
    )
    parser.add_argument("--original-qc", type=Path, default=DEFAULT_ORIGINAL_QC)
    parser.add_argument("--repaired-qc", type=Path, default=DEFAULT_REPAIRED_QC)
    parser.add_argument("--merged-output", type=Path, default=DEFAULT_MERGED_OUTPUT)
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--warning", default="low_estimated_snr")
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReviewQueueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ReviewQueueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def index_unique(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str):
            raise ReviewQueueError(f"{label} row is missing a string ID")
        if identifier in indexed:
            raise ReviewQueueError(f"Duplicate {label} ID: {identifier}")
        indexed[identifier] = row
    return indexed


def merge_repaired_qc(
    original_rows: list[dict[str, Any]], repaired_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    original = index_unique(original_rows, "original")
    repaired = index_unique(repaired_rows, "repaired")
    unexpected = sorted(set(repaired) - set(original))
    if unexpected:
        raise ReviewQueueError(f"Repaired IDs absent from original QC: {unexpected[:5]}")
    for identifier in repaired:
        reasons = original[identifier].get("audio_qc", {}).get("qc_reasons", [])
        if "excessive_dc_offset" not in reasons:
            raise ReviewQueueError(
                f"Repaired row {identifier} was not originally flagged for DC offset"
            )
        if repaired[identifier].get("split") != "train":
            raise ReviewQueueError(f"Repaired row {identifier} is not in train")
    return [repaired.get(row["id"], row) for row in original_rows]


def select_review_rows(
    rows: Iterable[dict[str, Any]], warning: str, split: str
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("split") == split
        and warning in row.get("audio_qc", {}).get("qc_warnings", [])
    ]
    return sorted(selected, key=lambda row: row["id"])


def build_summary(
    merged_rows: list[dict[str, Any]],
    repaired_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    warning: str,
    split: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": "librispeech",
        "merged_utterances": len(merged_rows),
        "repaired_qc_replacements": len(repaired_rows),
        "final_status_counts": dict(
            sorted(Counter(row["audio_qc"]["qc_status"] for row in merged_rows).items())
        ),
        "final_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in merged_rows
                    for reason in row["audio_qc"]["qc_reasons"]
                ).items()
            )
        ),
        "final_warning_counts": dict(
            sorted(
                Counter(
                    item
                    for row in merged_rows
                    for item in row["audio_qc"]["qc_warnings"]
                ).items()
            )
        ),
        "review_selection": {
            "warning": warning,
            "split": split,
            "utterances": len(review_rows),
            "duration_hours": round(
                sum(float(row["duration_sec"]) for row in review_rows) / 3600.0, 6
            ),
            "speakers": len({row["speaker_id"] for row in review_rows}),
        },
    }


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    original_rows = read_jsonl(args.original_qc.resolve())
    repaired_rows = read_jsonl(args.repaired_qc.resolve())
    merged_rows = merge_repaired_qc(original_rows, repaired_rows)
    review_rows = select_review_rows(merged_rows, args.warning, args.split)
    summary = build_summary(
        merged_rows, repaired_rows, review_rows, args.warning, args.split
    )
    write_jsonl_atomic(args.merged_output.resolve(), merged_rows)
    write_jsonl_atomic(args.review_output.resolve(), review_rows)
    write_json_atomic(args.summary.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Merged acoustic QC: {args.merged_output.resolve()}")
    print(f"Teacher candidates: {args.review_output.resolve()}")
    print(f"Selection summary:  {args.summary.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
