"""Create a deterministic, gender-stratified, speaker-disjoint train/dev split."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/zeroth_split.yaml"
DEFAULT_INPUT = PROJECT_ROOT / "data/manifests/zeroth.train_clean_pool.jsonl"
DEFAULT_TRAIN = PROJECT_ROOT / "data/manifests/zeroth.train.jsonl"
DEFAULT_DEV = PROJECT_ROOT / "data/manifests/zeroth.dev.jsonl"
DEFAULT_ASSIGNMENTS = PROJECT_ROOT / "configs/data_decisions/zeroth_speaker_split.yaml"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports/data_qc/zeroth_split.summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a clean manifest into speaker-disjoint train/dev sets."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dev-output", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--assignments-output", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}")
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def gender_quotas(
    speaker_genders: dict[str, str], target_count: int
) -> dict[str, int]:
    counts = Counter(speaker_genders.values())
    total = sum(counts.values())
    exact = {gender: target_count * count / total for gender, count in counts.items()}
    quotas = {gender: math.floor(value) for gender, value in exact.items()}
    remaining = target_count - sum(quotas.values())
    order = sorted(counts, key=lambda gender: (-(exact[gender] - quotas[gender]), gender))
    for gender in order[:remaining]:
        quotas[gender] += 1
    return quotas


def choose_dev_speakers(
    rows: list[dict[str, Any]], fraction: float, seed: int, trials: int
) -> set[str]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("dev_speaker_fraction must be between zero and one")
    if trials <= 0:
        raise ValueError("optimization_trials must be greater than zero")

    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker_id"]].append(row)
    if len(by_speaker) < 2:
        raise ValueError("At least two speakers are required")

    speaker_genders: dict[str, str] = {}
    for speaker, speaker_rows in by_speaker.items():
        genders = {row["speaker_gender"] for row in speaker_rows}
        if len(genders) != 1:
            raise ValueError(f"Inconsistent gender metadata for {speaker}")
        speaker_genders[speaker] = next(iter(genders))

    target_count = max(1, min(len(by_speaker) - 1, round(len(by_speaker) * fraction)))
    quotas = gender_quotas(speaker_genders, target_count)
    by_gender: dict[str, list[str]] = defaultdict(list)
    for speaker, gender in speaker_genders.items():
        by_gender[gender].append(speaker)
    for speakers in by_gender.values():
        speakers.sort()

    speaker_duration = {
        speaker: sum(float(row["duration_sec"]) for row in speaker_rows)
        for speaker, speaker_rows in by_speaker.items()
    }
    speaker_rows_count = {speaker: len(speaker_rows) for speaker, speaker_rows in by_speaker.items()}
    total_duration = sum(speaker_duration.values())
    total_rows = len(rows)
    rng = random.Random(seed)
    best: tuple[tuple[float, float, tuple[str, ...]], set[str]] | None = None
    for _ in range(trials):
        selected: set[str] = set()
        for gender in sorted(quotas):
            selected.update(rng.sample(by_gender[gender], quotas[gender]))
        duration_fraction = sum(speaker_duration[s] for s in selected) / total_duration
        row_fraction = sum(speaker_rows_count[s] for s in selected) / total_rows
        score = (
            abs(duration_fraction - fraction) + abs(row_fraction - fraction),
            max(abs(duration_fraction - fraction), abs(row_fraction - fraction)),
            tuple(sorted(selected)),
        )
        if best is None or score < best[0]:
            best = (score, selected)
    assert best is not None
    return best[1]


def apply_split(
    rows: list[dict[str, Any]], dev_speakers: set[str], policy_version: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        row["source_split"] = row.get("source_split", row.get("split"))
        row["split"] = "dev" if row["speaker_id"] in dev_speakers else "train"
        row["split_policy_version"] = policy_version
        (dev if row["split"] == "dev" else train).append(row)
    return train, dev


def hours(rows: Iterable[dict[str, Any]]) -> float:
    return round(sum(float(row["duration_sec"]) for row in rows) / 3600.0, 6)


def split_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    speakers = {row["speaker_id"] for row in rows}
    gender_by_speaker = {row["speaker_id"]: row["speaker_gender"] for row in rows}
    return {
        "utterances": len(rows),
        "duration_hours": hours(rows),
        "speakers": len(speakers),
        "speaker_gender": dict(sorted(Counter(gender_by_speaker.values()).items())),
    }


def build_summary(
    train: list[dict[str, Any]], dev: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    train_speakers = {row["speaker_id"] for row in train}
    dev_speakers = {row["speaker_id"] for row in dev}
    overlap = sorted(train_speakers & dev_speakers)
    if overlap:
        raise ValueError(f"Speaker leakage between train/dev: {overlap[:5]}")
    return {
        "schema_version": 1,
        "source": "zeroth",
        "policy_version": config["policy_version"],
        "random_seed": config["random_seed"],
        "speaker_overlap": overlap,
        "train": split_stats(train),
        "dev": split_stats(dev),
    }


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_yaml_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, allow_unicode=True, sort_keys=False)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    config = read_yaml(args.config.resolve())
    rows = read_jsonl(args.input.resolve())
    dev_speakers = choose_dev_speakers(
        rows,
        float(config["dev_speaker_fraction"]),
        int(config["random_seed"]),
        int(config["optimization_trials"]),
    )
    train, dev = apply_split(rows, dev_speakers, str(config["policy_version"]))
    summary = build_summary(train, dev, config)
    assignments = {
        "schema_version": 1,
        "policy_version": config["policy_version"],
        "random_seed": config["random_seed"],
        "train_speakers": sorted({row["speaker_id"] for row in train}),
        "dev_speakers": sorted(dev_speakers),
    }
    write_jsonl_atomic(args.train_output.resolve(), train)
    write_jsonl_atomic(args.dev_output.resolve(), dev)
    write_yaml_atomic(args.assignments_output.resolve(), assignments)
    write_json_atomic(args.summary.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Train manifest:      {args.train_output.resolve()}")
    print(f"Dev manifest:        {args.dev_output.resolve()}")
    print(f"Speaker assignments: {args.assignments_output.resolve()}")
    print(f"Summary:             {args.summary.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
