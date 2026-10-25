"""Join Zeroth Korean audio, transcripts, and speaker metadata.

The adapter is intentionally read-only with respect to the source corpus. It
uses the audio inventory as the source of technical metadata, validates every
cross-file relationship, and writes deterministic JSONL manifests atomically.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data/raw/zeroth"
DEFAULT_INVENTORY = PROJECT_ROOT / "data/manifests/raw_audio_inventory.jsonl"
DEFAULT_RAW_OUTPUT = PROJECT_ROOT / "data/manifests/zeroth.raw.jsonl"
DEFAULT_ACCEPTED_OUTPUT = PROJECT_ROOT / "data/manifests/zeroth.accepted.jsonl"
DEFAULT_QUARANTINE_OUTPUT = PROJECT_ROOT / "data/manifests/zeroth.quarantine.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports/data_qc/zeroth.summary.json"

REQUIRED_SPEAKER_COLUMNS = {"SPEAKERID", "NAME", "SEX", "SCRIPTID", "DATASET"}
VALID_DATASETS = {"train_data_01": "train", "test_data_01": "test"}
LICENSE = "CC-BY-4.0"


class ZerothDataError(ValueError):
    """Raised when the source corpus has an inconsistent relationship."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and duration-QC a Zeroth Korean JSONL manifest."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--accepted-output", type=Path, default=DEFAULT_ACCEPTED_OUTPUT)
    parser.add_argument(
        "--quarantine-output", type=Path, default=DEFAULT_QUARANTINE_OUTPUT
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--min-duration-sec", type=float, default=1.0)
    parser.add_argument("--max-duration-sec", type=float, default=30.0)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Apply the lossless v1 normalization used at ingestion time."""

    return " ".join(unicodedata.normalize("NFC", text).split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ZerothDataError(f"Invalid JSON at {path}:{line_number}: {exc}")
            if not isinstance(value, dict):
                raise ZerothDataError(
                    f"Expected a JSON object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def read_speakers(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_SPEAKER_COLUMNS - columns
        if missing:
            raise ZerothDataError(
                f"Missing AUDIO_INFO columns: {', '.join(sorted(missing))}"
            )

        speakers: dict[tuple[str, str, str], dict[str, str]] = {}
        for line_number, raw in enumerate(reader, start=2):
            row = {key: (value or "").strip() for key, value in raw.items()}
            key = (row["DATASET"], row["SCRIPTID"], row["SPEAKERID"])
            if row["DATASET"] not in VALID_DATASETS:
                raise ZerothDataError(
                    f"Unknown DATASET {row['DATASET']!r} at {path}:{line_number}"
                )
            if row["SEX"].lower() not in {"m", "f"}:
                raise ZerothDataError(
                    f"Invalid SEX {row['SEX']!r} at {path}:{line_number}"
                )
            if key in speakers:
                raise ZerothDataError(f"Duplicate speaker metadata key {key!r}")
            speakers[key] = row
    return speakers


def inventory_key(relative_path: str) -> str:
    path = Path(relative_path)
    parts = path.parts
    if parts and parts[0] == "zeroth":
        path = Path(*parts[1:])
    return path.as_posix()


def index_inventory(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("source") != "zeroth":
            continue
        relative_path = row.get("relative_path")
        if not isinstance(relative_path, str):
            raise ZerothDataError("Inventory row is missing relative_path")
        key = inventory_key(relative_path)
        if key in indexed:
            raise ZerothDataError(f"Duplicate inventory path: {key}")
        indexed[key] = row
    return indexed


def parse_transcript(path: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\r\n").split(maxsplit=1)
            if len(parts) != 2 or not parts[1].strip():
                raise ZerothDataError(
                    f"Missing utterance text at {path}:{line_number}"
                )
            utterance_id, text = parts
            if utterance_id in transcripts:
                raise ZerothDataError(
                    f"Duplicate utterance ID {utterance_id!r} in {path}"
                )
            transcripts[utterance_id] = text
    return transcripts


def qc_reasons(
    inventory_row: dict[str, Any], min_duration_sec: float, max_duration_sec: float
) -> list[str]:
    reasons: list[str] = []
    if inventory_row.get("status") != "ok":
        reasons.append("inventory_error")

    duration = inventory_row.get("duration_sec")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration):
        reasons.append("invalid_duration")
    elif duration < min_duration_sec:
        reasons.append("duration_too_short")
    elif duration > max_duration_sec:
        reasons.append("duration_too_long")
    return reasons


def build_rows(
    source_root: Path,
    inventory_rows: Iterable[dict[str, Any]],
    min_duration_sec: float,
    max_duration_sec: float,
) -> list[dict[str, Any]]:
    if min_duration_sec < 0 or max_duration_sec <= min_duration_sec:
        raise ZerothDataError("Duration bounds must satisfy 0 <= min < max")

    speakers = read_speakers(source_root / "AUDIO_INFO")
    inventory = index_inventory(inventory_rows)
    rows: list[dict[str, Any]] = []
    seen_utterances: set[str] = set()
    expected_inventory_paths: set[str] = set()

    for key, speaker in sorted(speakers.items()):
        dataset, script_id, speaker_id = key
        speaker_dir = source_root / dataset / script_id / speaker_id
        transcript_path = speaker_dir / f"{speaker_id}_{script_id}.trans.txt"
        if not speaker_dir.is_dir():
            raise ZerothDataError(f"Missing speaker directory: {speaker_dir}")
        if not transcript_path.is_file():
            raise ZerothDataError(f"Missing transcript file: {transcript_path}")

        transcripts = parse_transcript(transcript_path)
        audio_ids = {path.stem for path in speaker_dir.glob("*.flac")}
        transcript_ids = set(transcripts)
        if audio_ids != transcript_ids:
            missing_audio = sorted(transcript_ids - audio_ids)[:5]
            missing_text = sorted(audio_ids - transcript_ids)[:5]
            raise ZerothDataError(
                f"Audio/transcript mismatch in {speaker_dir}: "
                f"missing_audio={missing_audio}, missing_text={missing_text}"
            )

        for utterance_id, text in sorted(transcripts.items()):
            prefix = f"{speaker_id}_{script_id}_"
            if not utterance_id.startswith(prefix):
                raise ZerothDataError(
                    f"Utterance {utterance_id!r} does not match speaker/script {prefix!r}"
                )
            if utterance_id in seen_utterances:
                raise ZerothDataError(f"Duplicate corpus utterance ID: {utterance_id}")
            seen_utterances.add(utterance_id)

            relative_source_path = Path(dataset) / script_id / speaker_id / f"{utterance_id}.flac"
            inventory_path = relative_source_path.as_posix()
            expected_inventory_paths.add(inventory_path)
            if inventory_path not in inventory:
                raise ZerothDataError(
                    f"Audio is absent from inventory: {inventory_path}"
                )

            technical = inventory[inventory_path]
            reasons = qc_reasons(technical, min_duration_sec, max_duration_sec)
            split = VALID_DATASETS[dataset]
            rows.append(
                {
                    "schema_version": 1,
                    "id": f"zeroth_{utterance_id}",
                    "utterance_id": utterance_id,
                    "audio_path": (
                        Path("data/raw/zeroth") / relative_source_path
                    ).as_posix(),
                    "text": text,
                    "text_normalized": normalize_text(text),
                    "text_normalization": "ingest_nfc_whitespace_v1",
                    "language": "ko",
                    "duration_sec": technical.get("duration_sec"),
                    "sample_rate": technical.get("sample_rate"),
                    "channels": technical.get("channels"),
                    "speaker_id": f"zeroth_{speaker_id}",
                    "speaker_name": speaker["NAME"],
                    "speaker_gender": speaker["SEX"].lower(),
                    "source": "zeroth",
                    "source_dataset": dataset,
                    "source_script_id": script_id,
                    "split": split,
                    "license": LICENSE,
                    "is_clean": True,
                    "checksum_xxh3_128": technical.get("checksum_xxh3_128"),
                    "qc_status": "accepted" if not reasons else "quarantine",
                    "qc_reasons": reasons,
                }
            )

    unexpected_inventory = sorted(set(inventory) - expected_inventory_paths)
    if unexpected_inventory:
        raise ZerothDataError(
            "Inventory contains Zeroth audio without metadata/transcript: "
            f"{unexpected_inventory[:5]}"
        )
    return sorted(rows, key=lambda row: row["id"])


def percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def build_summary(
    rows: list[dict[str, Any]], min_duration_sec: float, max_duration_sec: float
) -> dict[str, Any]:
    accepted = [row for row in rows if row["qc_status"] == "accepted"]
    quarantined = [row for row in rows if row["qc_status"] == "quarantine"]
    durations = sorted(
        float(row["duration_sec"])
        for row in rows
        if isinstance(row.get("duration_sec"), (int, float))
        and math.isfinite(row["duration_sec"])
    )
    reason_counts = Counter(
        reason for row in quarantined for reason in row["qc_reasons"]
    )
    split_texts = {
        split: {row["text_normalized"] for row in rows if row["split"] == split}
        for split in {row["split"] for row in rows}
    }
    train_texts = split_texts.get("train", set())
    test_rows = [row for row in rows if row["split"] == "test"]
    test_texts = split_texts.get("test", set())
    seen_test_rows = sum(row["text_normalized"] in train_texts for row in test_rows)
    return {
        "schema_version": 1,
        "source": "zeroth",
        "duration_policy_sec": {
            "minimum_inclusive": min_duration_sec,
            "maximum_inclusive": max_duration_sec,
        },
        "total_utterances": len(rows),
        "accepted_utterances": len(accepted),
        "quarantined_utterances": len(quarantined),
        "accepted_duration_hours": round(
            sum(float(row["duration_sec"]) for row in accepted) / 3600.0, 6
        ),
        "quarantine_reasons": dict(sorted(reason_counts.items())),
        "source_splits": dict(sorted(Counter(row["split"] for row in rows).items())),
        "speakers_by_split": {
            split: len(
                {row["speaker_id"] for row in rows if row["split"] == split}
            )
            for split in sorted({row["split"] for row in rows})
        },
        "speakers": len({row["speaker_id"] for row in rows}),
        "speaker_gender": dict(
            sorted(
                Counter(
                    {
                        row["speaker_id"]: row["speaker_gender"] for row in rows
                    }.values()
                ).items()
            )
        ),
        "duration_sec": {
            "minimum": round(durations[0], 6) if durations else None,
            "p01": round(percentile(durations, 0.01), 6) if durations else None,
            "median": round(median(durations), 6) if durations else None,
            "mean": round(mean(durations), 6) if durations else None,
            "p99": round(percentile(durations, 0.99), 6) if durations else None,
            "maximum": round(durations[-1], 6) if durations else None,
        },
        "cross_split_text_overlap": {
            "unique_train_texts": len(train_texts),
            "unique_test_texts": len(test_texts),
            "unique_texts_in_both": len(train_texts & test_texts),
            "test_rows_seen_in_train": seen_test_rows,
            "test_rows_seen_in_train_pct": (
                round(100.0 * seen_test_rows / len(test_rows), 6)
                if test_rows
                else None
            ),
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
    source_root = args.source_root.resolve()
    rows = build_rows(
        source_root=source_root,
        inventory_rows=read_jsonl(args.inventory.resolve()),
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
    )
    accepted = [row for row in rows if row["qc_status"] == "accepted"]
    quarantined = [row for row in rows if row["qc_status"] == "quarantine"]
    summary = build_summary(rows, args.min_duration_sec, args.max_duration_sec)

    write_jsonl_atomic(args.raw_output.resolve(), rows)
    write_jsonl_atomic(args.accepted_output.resolve(), accepted)
    write_jsonl_atomic(args.quarantine_output.resolve(), quarantined)
    write_json_atomic(args.summary.resolve(), summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Raw manifest:        {args.raw_output.resolve()}")
    print(f"Accepted manifest:   {args.accepted_output.resolve()}")
    print(f"Quarantine manifest: {args.quarantine_output.resolve()}")
    print(f"QC summary:          {args.summary.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
