"""Validate and fingerprint the final Zeroth data handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = PROJECT_ROOT / "data/manifests/zeroth.train.jsonl"
DEFAULT_DEV = PROJECT_ROOT / "data/manifests/zeroth.dev.jsonl"
DEFAULT_OFFICIAL_TEST = PROJECT_ROOT / "data/manifests/zeroth.official_test.jsonl"
DEFAULT_CURATED_TEST = PROJECT_ROOT / "data/manifests/zeroth.curated_clean_test.jsonl"
DEFAULT_HOLDOUT = PROJECT_ROOT / "data/manifests/zeroth.transcript_review_holdout.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports/data_qc/zeroth_final_validation.summary.json"
DEFAULT_CHECKSUMS = PROJECT_ROOT / "data/manifests/zeroth_v1.sha256"
REQUIRED_FIELDS = {
    "id",
    "audio_path",
    "text",
    "text_normalized",
    "language",
    "duration_sec",
    "sample_rate",
    "channels",
    "speaker_id",
    "source",
    "split",
    "license",
    "checksum_xxh3_128",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate final Zeroth manifests.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--official-test", type=Path, default=DEFAULT_OFFICIAL_TEST)
    parser.add_argument("--curated-test", type=Path, default=DEFAULT_CURATED_TEST)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--checksums", type=Path, default=DEFAULT_CHECKSUMS)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def validate_rows(
    name: str, rows: list[dict[str, Any]], expected_split: str | None
) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            errors.append(f"{name}:{index}: missing fields {sorted(missing)}")
            continue
        if row["id"] in seen_ids:
            errors.append(f"{name}:{index}: duplicate ID {row['id']}")
        seen_ids.add(row["id"])
        if expected_split is not None and row["split"] != expected_split:
            errors.append(
                f"{name}:{index}: expected split {expected_split}, got {row['split']}"
            )
        audio_path = Path(row["audio_path"])
        resolved = audio_path if audio_path.is_absolute() else PROJECT_ROOT / audio_path
        if not resolved.is_file():
            errors.append(f"{name}:{index}: missing audio {resolved}")
        duration = row["duration_sec"]
        if not isinstance(duration, (int, float)) or not 1.0 <= duration <= 30.0:
            errors.append(f"{name}:{index}: invalid duration {duration!r}")
        if row["sample_rate"] != 16000:
            errors.append(f"{name}:{index}: sample rate is not 16000")
        if row["channels"] != 1:
            errors.append(f"{name}:{index}: audio is not mono")
        if row["language"] != "ko":
            errors.append(f"{name}:{index}: language is not ko")
        if row["source"] != "zeroth":
            errors.append(f"{name}:{index}: source is not zeroth")
        if not str(row["text"]).strip() or not str(row["text_normalized"]).strip():
            errors.append(f"{name}:{index}: empty transcript")
        expected_text = " ".join(unicodedata.normalize("NFC", row["text"]).split())
        if row["text_normalized"] != expected_text:
            errors.append(f"{name}:{index}: text normalization mismatch")
    return errors


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    speakers = {row["speaker_id"] for row in rows}
    return {
        "utterances": len(rows),
        "duration_hours": round(
            sum(float(row["duration_sec"]) for row in rows) / 3600.0, 6
        ),
        "speakers": len(speakers),
        "speaker_gender": dict(
            sorted(
                Counter(
                    {row["speaker_id"]: row["speaker_gender"] for row in rows}.values()
                ).items()
            )
        ),
    }


def normalized_texts(rows: list[dict[str, Any]]) -> set[str]:
    return {
        re.sub(r"\s+", " ", row["text_normalized"].strip())
        for row in rows
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_checksums_atomic(path: Path, targets: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    lines = [
        f"{sha256_file(target)}  {target.relative_to(PROJECT_ROOT).as_posix()}"
        for target in targets
    ]
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_dataset(
    train: list[dict[str, Any]],
    dev: list[dict[str, Any]],
    official_test: list[dict[str, Any]],
    curated_test: list[dict[str, Any]],
    holdout: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    errors.extend(validate_rows("train", train, "train"))
    errors.extend(validate_rows("dev", dev, "dev"))
    errors.extend(validate_rows("official_test", official_test, "test"))
    errors.extend(validate_rows("curated_test", curated_test, "test"))
    errors.extend(validate_rows("holdout", holdout, None))

    primary = {"train": train, "dev": dev, "official_test": official_test}
    ids = {name: {row["id"] for row in rows} for name, rows in primary.items()}
    speakers = {
        name: {row["speaker_id"] for row in rows} for name, rows in primary.items()
    }
    id_overlap: dict[str, list[str]] = {}
    speaker_overlap: dict[str, list[str]] = {}
    names = list(primary)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            key = f"{first}__{second}"
            id_overlap[key] = sorted(ids[first] & ids[second])
            speaker_overlap[key] = sorted(speakers[first] & speakers[second])
    if any(id_overlap.values()):
        errors.append("Primary splits contain duplicate utterance IDs")
    if any(speaker_overlap.values()):
        errors.append("Primary splits contain speaker leakage")

    official_ids = ids["official_test"]
    curated_ids = {row["id"] for row in curated_test}
    if not curated_ids <= official_ids:
        errors.append("Curated test is not a subset of official test")
    clean_ids = ids["train"] | ids["dev"] | curated_ids
    train_holdout_ids = {row["id"] for row in holdout if row["split"] == "train"}
    if clean_ids & train_holdout_ids:
        errors.append("Train review holdout leaked into clean splits")

    train_texts = normalized_texts(train)
    official_texts = normalized_texts(official_test)
    official_seen_rows = sum(
        row["text_normalized"] in train_texts for row in official_test
    )
    return {
        "schema_version": 1,
        "source": "zeroth",
        "valid": not errors,
        "errors": errors[:100],
        "error_count": len(errors),
        "splits": {
            "train": stats(train),
            "dev": stats(dev),
            "official_test": stats(official_test),
            "curated_test": stats(curated_test),
            "review_holdout": stats(holdout),
        },
        "id_overlap": id_overlap,
        "speaker_overlap": speaker_overlap,
        "curated_test_is_official_subset": curated_ids <= official_ids,
        "official_test_text_overlap": {
            "unique_train_texts": len(train_texts),
            "unique_official_test_texts": len(official_texts),
            "unique_texts_in_both": len(train_texts & official_texts),
            "official_test_rows_seen_in_train": official_seen_rows,
            "official_test_rows_seen_in_train_pct": round(
                100.0 * official_seen_rows / len(official_test), 6
            ),
        },
    }


def main() -> int:
    args = parse_args()
    paths = {
        "train": args.train.resolve(),
        "dev": args.dev.resolve(),
        "official_test": args.official_test.resolve(),
        "curated_test": args.curated_test.resolve(),
        "holdout": args.holdout.resolve(),
    }
    rows = {name: read_jsonl(path) for name, path in paths.items()}
    summary = validate_dataset(
        rows["train"],
        rows["dev"],
        rows["official_test"],
        rows["curated_test"],
        rows["holdout"],
    )
    write_json_atomic(args.summary.resolve(), summary)
    checksum_targets = list(paths.values()) + [
        PROJECT_ROOT / "configs/text_normalization.yaml",
        PROJECT_ROOT / "configs/clean_manifest_policy.yaml",
        PROJECT_ROOT / "configs/data_decisions/zeroth_audio_qc.yaml",
        PROJECT_ROOT / "configs/data_decisions/zeroth_speaker_split.yaml",
    ]
    write_checksums_atomic(args.checksums.resolve(), checksum_targets)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Checksums: {args.checksums.resolve()}")
    return 0 if summary["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
