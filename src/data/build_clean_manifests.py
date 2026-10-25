"""Build conservative clean and review manifests after transcript QC."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = PROJECT_ROOT / "configs/clean_manifest_policy.yaml"
DEFAULT_FULL = PROJECT_ROOT / "data/interim/zeroth/transcript_qc.jsonl"
DEFAULT_RECONCILED = PROJECT_ROOT / "data/interim/zeroth/transcript_qc_reconciled.jsonl"
DEFAULT_CLEAN = PROJECT_ROOT / "data/manifests/zeroth.clean_high_precision.jsonl"
DEFAULT_TRAIN_POOL = PROJECT_ROOT / "data/manifests/zeroth.train_clean_pool.jsonl"
DEFAULT_HOLDOUT = PROJECT_ROOT / "data/manifests/zeroth.transcript_review_holdout.jsonl"
DEFAULT_OFFICIAL_TEST = PROJECT_ROOT / "data/manifests/zeroth.official_test.jsonl"
DEFAULT_CURATED_TEST = PROJECT_ROOT / "data/manifests/zeroth.curated_clean_test.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports/data_qc/zeroth_clean_manifests.summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build high-precision Zeroth manifests without deleting data."
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--full-input", type=Path, default=DEFAULT_FULL)
    parser.add_argument("--reconciled-input", type=Path, default=DEFAULT_RECONCILED)
    parser.add_argument("--clean-output", type=Path, default=DEFAULT_CLEAN)
    parser.add_argument("--train-pool-output", type=Path, default=DEFAULT_TRAIN_POOL)
    parser.add_argument("--holdout-output", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--official-test-output", type=Path, default=DEFAULT_OFFICIAL_TEST)
    parser.add_argument("--curated-test-output", type=Path, default=DEFAULT_CURATED_TEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
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
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}")
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def read_policy(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    if value.get("transcript_review_action") != "holdout":
        raise ValueError("Only the conservative holdout policy is supported")
    return value


def index_unique(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        utterance_id = row.get("id")
        if not isinstance(utterance_id, str) or not utterance_id:
            raise ValueError(f"{label} row is missing id")
        if utterance_id in result:
            raise ValueError(f"Duplicate {label} ID: {utterance_id}")
        result[utterance_id] = row
    return result


def add_selection(
    row: dict[str, Any], status: str, reason: str, policy_version: str
) -> dict[str, Any]:
    value = dict(row)
    value["data_selection"] = {
        "policy_version": policy_version,
        "status": status,
        "reason": reason,
        "source_text_changed": False,
    }
    return value


def build_manifests(
    full_rows: list[dict[str, Any]],
    reconciled_rows: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    full = index_unique(full_rows, "full")
    reconciled = index_unique(reconciled_rows, "reconciled")
    unknown = sorted(set(reconciled) - set(full))
    if unknown:
        raise ValueError(f"Reconciled rows are absent from full input: {unknown[:5]}")

    policy_version = str(policy["policy_version"])
    clean: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for utterance_id, base in full.items():
        if utterance_id in reconciled:
            source = reconciled[utterance_id]
            reconciliation = source["transcript_reconciliation"]
            if reconciliation["provisional_action"] == "review":
                holdout.append(
                    add_selection(
                        source,
                        "transcript_review_holdout",
                        reconciliation["category"],
                        policy_version,
                    )
                )
                continue
            clean.append(
                add_selection(
                    source,
                    "clean_high_precision",
                    reconciliation["category"],
                    policy_version,
                )
            )
        else:
            clean.append(
                add_selection(
                    base,
                    "clean_high_precision",
                    "teacher1_pass_or_number_group_consensus",
                    policy_version,
                )
            )

    clean.sort(key=lambda row: row["id"])
    holdout.sort(key=lambda row: row["id"])
    train_pool = [row for row in clean if row["split"] == "train"]
    official_test = [full[key] for key in sorted(full) if full[key]["split"] == "test"]
    curated_test = [row for row in clean if row["split"] == "test"]
    return {
        "clean": clean,
        "train_pool": train_pool,
        "holdout": holdout,
        "official_test": official_test,
        "curated_test": curated_test,
    }


def hours(rows: Iterable[dict[str, Any]]) -> float:
    return round(sum(float(row["duration_sec"]) for row in rows) / 3600.0, 6)


def build_summary(
    manifests: dict[str, list[dict[str, Any]]], policy: dict[str, Any]
) -> dict[str, Any]:
    full_count = len(manifests["clean"]) + len(manifests["holdout"])
    clean_ids = {row["id"] for row in manifests["clean"]}
    holdout_ids = {row["id"] for row in manifests["holdout"]}
    if clean_ids & holdout_ids:
        raise ValueError("Clean and holdout manifests overlap")
    if len(clean_ids | holdout_ids) != full_count:
        raise ValueError("Clean and holdout manifests do not partition the input")

    return {
        "schema_version": 1,
        "source": "zeroth",
        "policy_version": policy["policy_version"],
        "source_texts_modified": 0,
        "input_utterances_after_audio_qc": full_count,
        "manifests": {
            name: {
                "utterances": len(rows),
                "duration_hours": hours(rows),
                "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
                "speakers": len({row["speaker_id"] for row in rows}),
            }
            for name, rows in manifests.items()
        },
        "holdout_reasons": dict(
            sorted(Counter(row["data_selection"]["reason"] for row in manifests["holdout"]).items())
        ),
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


def main() -> int:
    args = parse_args()
    policy = read_policy(args.policy.resolve())
    manifests = build_manifests(
        read_jsonl(args.full_input.resolve()),
        read_jsonl(args.reconciled_input.resolve()),
        policy,
    )
    outputs = {
        "clean": args.clean_output.resolve(),
        "train_pool": args.train_pool_output.resolve(),
        "holdout": args.holdout_output.resolve(),
        "official_test": args.official_test_output.resolve(),
        "curated_test": args.curated_test_output.resolve(),
    }
    for name, path in outputs.items():
        write_jsonl_atomic(path, manifests[name])
    summary = build_summary(manifests, policy)
    write_json_atomic(args.summary.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    for name, path in outputs.items():
        print(f"{name:14s}: {path}")
    print(f"summary       : {args.summary.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
