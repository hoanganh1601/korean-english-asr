"""Join LibriSpeech audio, transcripts, and speaker metadata.

The adapter never modifies source files. It validates the complete relationship
between the selected official subsets, the technical audio inventory, chapter
transcripts, and SPEAKERS.TXT before writing deterministic JSONL manifests.
"""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data/raw/librispeech"
DEFAULT_INVENTORY = (
    PROJECT_ROOT / "data/manifests/librispeech_audio_inventory.jsonl"
)
DEFAULT_RAW_OUTPUT = PROJECT_ROOT / "data/manifests/librispeech.raw.jsonl"
DEFAULT_ACCEPTED_OUTPUT = PROJECT_ROOT / "data/manifests/librispeech.accepted.jsonl"
DEFAULT_QUARANTINE_OUTPUT = (
    PROJECT_ROOT / "data/manifests/librispeech.quarantine.jsonl"
)
DEFAULT_TRAIN_OUTPUT = PROJECT_ROOT / "data/manifests/librispeech.train.jsonl"
DEFAULT_DEV_OUTPUT = PROJECT_ROOT / "data/manifests/librispeech.dev.jsonl"
DEFAULT_TEST_OUTPUT = PROJECT_ROOT / "data/manifests/librispeech.test.jsonl"
DEFAULT_OFFICIAL_DEV_OUTPUT = (
    PROJECT_ROOT / "data/manifests/librispeech.official_dev.jsonl"
)
DEFAULT_OFFICIAL_TEST_OUTPUT = (
    PROJECT_ROOT / "data/manifests/librispeech.official_test.jsonl"
)
DEFAULT_SUMMARY = PROJECT_ROOT / "reports/data_qc/librispeech.summary.json"

SUBSET_TO_SPLIT = {
    "train-clean-100": "train",
    "dev-clean": "dev",
    "test-clean": "test",
}
LICENSE = "CC-BY-4.0"


class LibriSpeechDataError(ValueError):
    """Raised when the source corpus has an inconsistent relationship."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and duration-QC LibriSpeech JSONL manifests."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--accepted-output", type=Path, default=DEFAULT_ACCEPTED_OUTPUT)
    parser.add_argument(
        "--quarantine-output", type=Path, default=DEFAULT_QUARANTINE_OUTPUT
    )
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--dev-output", type=Path, default=DEFAULT_DEV_OUTPUT)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST_OUTPUT)
    parser.add_argument(
        "--official-dev-output", type=Path, default=DEFAULT_OFFICIAL_DEV_OUTPUT
    )
    parser.add_argument(
        "--official-test-output", type=Path, default=DEFAULT_OFFICIAL_TEST_OUTPUT
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--min-duration-sec", type=float, default=1.0)
    parser.add_argument("--max-duration-sec", type=float, default=30.0)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """Apply the project's lossless ingestion normalization."""

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
                raise LibriSpeechDataError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise LibriSpeechDataError(
                    f"Expected a JSON object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def read_speakers(path: Path) -> dict[str, dict[str, str]]:
    speakers: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith(";"):
                continue
            parts = [part.strip() for part in line.rstrip("\r\n").split("|", 4)]
            if len(parts) != 5:
                raise LibriSpeechDataError(
                    f"Malformed speaker metadata at {path}:{line_number}"
                )
            speaker_id, gender, subset, minutes, name = parts
            if not speaker_id.isdigit():
                raise LibriSpeechDataError(
                    f"Invalid speaker ID {speaker_id!r} at {path}:{line_number}"
                )
            if gender.lower() not in {"m", "f"}:
                raise LibriSpeechDataError(
                    f"Invalid speaker gender {gender!r} at {path}:{line_number}"
                )
            try:
                float(minutes)
            except ValueError as exc:
                raise LibriSpeechDataError(
                    f"Invalid speaker duration {minutes!r} at {path}:{line_number}"
                ) from exc
            if speaker_id in speakers:
                raise LibriSpeechDataError(f"Duplicate speaker ID: {speaker_id}")
            speakers[speaker_id] = {
                "speaker_id": speaker_id,
                "gender": gender.lower(),
                "subset": subset,
                "minutes": minutes,
                "name": name,
            }
    return speakers


def parse_transcript(path: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\r\n").split(maxsplit=1)
            if len(parts) != 2 or not parts[1].strip():
                raise LibriSpeechDataError(
                    f"Missing utterance text at {path}:{line_number}"
                )
            utterance_id, text = parts
            if utterance_id in transcripts:
                raise LibriSpeechDataError(
                    f"Duplicate utterance ID {utterance_id!r} in {path}"
                )
            transcripts[utterance_id] = text
    return transcripts


def inventory_key(relative_path: str) -> str:
    path = Path(relative_path)
    if path.parts and path.parts[0] == "librispeech":
        path = Path(*path.parts[1:])
    return path.as_posix()


def index_inventory(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        relative_path = row.get("relative_path")
        if not isinstance(relative_path, str):
            raise LibriSpeechDataError("Inventory row is missing relative_path")
        key = inventory_key(relative_path)
        if not key.startswith(tuple(f"{subset}/" for subset in SUBSET_TO_SPLIT)):
            raise LibriSpeechDataError(f"Unexpected inventory subset: {key}")
        if key in indexed:
            raise LibriSpeechDataError(f"Duplicate inventory path: {key}")
        indexed[key] = row
    return indexed


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
        raise LibriSpeechDataError("Duration bounds must satisfy 0 <= min < max")

    speakers = read_speakers(source_root / "SPEAKERS.TXT")
    inventory = index_inventory(inventory_rows)
    rows: list[dict[str, Any]] = []
    seen_utterances: set[str] = set()
    expected_inventory_paths: set[str] = set()
    speaker_splits: dict[str, str] = {}

    for subset, split in SUBSET_TO_SPLIT.items():
        subset_root = source_root / subset
        if not subset_root.is_dir():
            raise LibriSpeechDataError(f"Missing LibriSpeech subset: {subset_root}")

        transcript_paths = sorted(subset_root.glob("*/*/*.trans.txt"))
        if not transcript_paths:
            raise LibriSpeechDataError(f"No transcripts found in {subset_root}")

        for transcript_path in transcript_paths:
            chapter_dir = transcript_path.parent
            speaker_id = chapter_dir.parent.name
            chapter_id = chapter_dir.name
            expected_name = f"{speaker_id}-{chapter_id}.trans.txt"
            if transcript_path.name != expected_name:
                raise LibriSpeechDataError(
                    f"Unexpected transcript name {transcript_path}; expected {expected_name}"
                )
            if speaker_id not in speakers:
                raise LibriSpeechDataError(
                    f"Speaker {speaker_id} is absent from SPEAKERS.TXT"
                )
            speaker = speakers[speaker_id]
            if speaker["subset"] != subset:
                raise LibriSpeechDataError(
                    f"Speaker {speaker_id} metadata subset {speaker['subset']!r} "
                    f"does not match {subset!r}"
                )
            previous_split = speaker_splits.setdefault(speaker_id, split)
            if previous_split != split:
                raise LibriSpeechDataError(
                    f"Speaker {speaker_id} occurs in both {previous_split} and {split}"
                )

            transcripts = parse_transcript(transcript_path)
            audio_ids = {path.stem for path in chapter_dir.glob("*.flac")}
            transcript_ids = set(transcripts)
            if audio_ids != transcript_ids:
                missing_audio = sorted(transcript_ids - audio_ids)[:5]
                missing_text = sorted(audio_ids - transcript_ids)[:5]
                raise LibriSpeechDataError(
                    f"Audio/transcript mismatch in {chapter_dir}: "
                    f"missing_audio={missing_audio}, missing_text={missing_text}"
                )

            for utterance_id, text in sorted(transcripts.items()):
                prefix = f"{speaker_id}-{chapter_id}-"
                if not utterance_id.startswith(prefix):
                    raise LibriSpeechDataError(
                        f"Utterance {utterance_id!r} does not match {prefix!r}"
                    )
                if utterance_id in seen_utterances:
                    raise LibriSpeechDataError(
                        f"Duplicate corpus utterance ID: {utterance_id}"
                    )
                seen_utterances.add(utterance_id)

                relative_path = (
                    Path(subset)
                    / speaker_id
                    / chapter_id
                    / f"{utterance_id}.flac"
                )
                inventory_path = relative_path.as_posix()
                expected_inventory_paths.add(inventory_path)
                if inventory_path not in inventory:
                    raise LibriSpeechDataError(
                        f"Audio is absent from inventory: {inventory_path}"
                    )
                technical = inventory[inventory_path]
                reasons = qc_reasons(
                    technical, min_duration_sec, max_duration_sec
                )
                rows.append(
                    {
                        "schema_version": 1,
                        "id": f"librispeech_{utterance_id}",
                        "utterance_id": utterance_id,
                        "audio_path": (
                            Path("data/raw/librispeech") / relative_path
                        ).as_posix(),
                        "text": text,
                        "text_normalized": normalize_text(text),
                        "text_normalization": "ingest_nfc_whitespace_v1",
                        "language": "en",
                        "duration_sec": technical.get("duration_sec"),
                        "sample_rate": technical.get("sample_rate"),
                        "channels": technical.get("channels"),
                        "speaker_id": f"librispeech_{speaker_id}",
                        "speaker_name": speaker["name"],
                        "speaker_gender": speaker["gender"],
                        "source": "librispeech",
                        "source_dataset": subset,
                        "source_chapter_id": chapter_id,
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
        raise LibriSpeechDataError(
            "Inventory contains audio without transcript/metadata: "
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
    speakers_by_split = {
        split: {row["speaker_id"] for row in rows if row["split"] == split}
        for split in ("train", "dev", "test")
    }
    texts_by_split = {
        split: {row["text_normalized"] for row in rows if row["split"] == split}
        for split in ("train", "dev", "test")
    }
    return {
        "schema_version": 1,
        "source": "librispeech",
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
        "source_subsets": dict(
            sorted(Counter(row["source_dataset"] for row in rows).items())
        ),
        "accepted_by_split": {
            split: sum(
                row["split"] == split and row["qc_status"] == "accepted"
                for row in rows
            )
            for split in ("train", "dev", "test")
        },
        "speakers_by_split": {
            split: len(speakers) for split, speakers in speakers_by_split.items()
        },
        "speakers": len({row["speaker_id"] for row in rows}),
        "speaker_gender": dict(
            sorted(
                Counter(
                    {
                        row["speaker_id"]: row["speaker_gender"]
                        for row in rows
                    }.values()
                ).items()
            )
        ),
        "speaker_overlap": {
            "train_dev": len(speakers_by_split["train"] & speakers_by_split["dev"]),
            "train_test": len(speakers_by_split["train"] & speakers_by_split["test"]),
            "dev_test": len(speakers_by_split["dev"] & speakers_by_split["test"]),
        },
        "text_overlap": {
            "train_dev": len(texts_by_split["train"] & texts_by_split["dev"]),
            "train_test": len(texts_by_split["train"] & texts_by_split["test"]),
            "dev_test": len(texts_by_split["dev"] & texts_by_split["test"]),
        },
        "duration_sec": {
            "minimum": round(durations[0], 6) if durations else None,
            "p01": round(percentile(durations, 0.01), 6) if durations else None,
            "median": round(median(durations), 6) if durations else None,
            "mean": round(mean(durations), 6) if durations else None,
            "p99": round(percentile(durations, 0.99), 6) if durations else None,
            "maximum": round(durations[-1], 6) if durations else None,
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
    rows = build_rows(
        source_root=args.source_root.resolve(),
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
    split_outputs = {
        "train": args.train_output,
        "dev": args.dev_output,
        "test": args.test_output,
    }
    for split, output in split_outputs.items():
        write_jsonl_atomic(
            output.resolve(), [row for row in accepted if row["split"] == split]
        )
    write_jsonl_atomic(
        args.official_dev_output.resolve(),
        [row for row in rows if row["split"] == "dev"],
    )
    write_jsonl_atomic(
        args.official_test_output.resolve(),
        [row for row in rows if row["split"] == "test"],
    )
    write_json_atomic(args.summary.resolve(), summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Raw manifest:        {args.raw_output.resolve()}")
    print(f"Accepted manifest:   {args.accepted_output.resolve()}")
    print(f"Quarantine manifest: {args.quarantine_output.resolve()}")
    print(f"Train manifest:      {args.train_output.resolve()}")
    print(f"Dev manifest:        {args.dev_output.resolve()}")
    print(f"Test manifest:       {args.test_output.resolve()}")
    print(f"Official dev:        {args.official_dev_output.resolve()}")
    print(f"Official test:       {args.official_test_output.resolve()}")
    print(f"QC summary:          {args.summary.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
