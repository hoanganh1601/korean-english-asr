"""Triage Whisper disagreements before any transcript is changed or removed."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data/interim/zeroth/transcript_qc.jsonl"
DEFAULT_CANDIDATES = (
    PROJECT_ROOT / "data/interim/zeroth/transcript_qc_second_teacher_candidates.jsonl"
)
DEFAULT_INCONSISTENT = (
    PROJECT_ROOT / "reports/data_qc/zeroth_transcript_inconsistent_scripts.json"
)
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "reports/data_qc/zeroth_transcript_qc.triage.summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a second-teacher queue from transcript QC results."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--candidates-output", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--inconsistent-output", type=Path, default=DEFAULT_INCONSISTENT)
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


def script_key(row: dict[str, Any]) -> str:
    parts = str(row["utterance_id"]).split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected utterance ID: {row['utterance_id']!r}")
    return "_".join(parts[1:])


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def triage_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[script_key(row)].append(row)

    inconsistent_groups = {
        key: group
        for key, group in groups.items()
        if len({row["text_normalized"] for row in group}) > 1
    }
    inconsistent_report: list[dict[str, Any]] = []
    for key, group in sorted(inconsistent_groups.items()):
        variants = Counter(row["text_normalized"] for row in group)
        inconsistent_report.append(
            {
                "script_key": key,
                "utterances": len(group),
                "variants": [
                    {"text": text, "count": count}
                    for text, count in variants.most_common()
                ],
                "ids": sorted(row["id"] for row in group),
            }
        )

    candidates: list[dict[str, Any]] = []
    number_consensus_rows = 0
    reason_counts: Counter[str] = Counter()
    for original in rows:
        row = dict(original)
        key = script_key(row)
        flags = set(row["transcript_qc"].get("flags") or [])
        prediction = compact(row["transcript_qc"]["prediction_normalized"])
        prediction_consensus = sum(
            compact(peer["transcript_qc"]["prediction_normalized"]) == prediction
            for peer in groups[key]
        )
        reasons: list[str] = []

        if flags & {"teacher_disagreement", "high_teacher_disagreement"}:
            reasons.append("non_number_teacher_disagreement")
        if any("number_format" in flag for flag in flags):
            if prediction_consensus >= 2:
                number_consensus_rows += 1
            else:
                reasons.append("number_format_without_group_consensus")
        if key in inconsistent_groups:
            reasons.append("source_transcript_inconsistent_within_script")

        if reasons:
            row["transcript_triage"] = {
                "script_key": key,
                "teacher_prediction_group_consensus": prediction_consensus,
                "reference_variants_in_group": len(
                    {peer["text_normalized"] for peer in groups[key]}
                ),
                "second_teacher_reasons": reasons,
            }
            candidates.append(row)
            reason_counts.update(reasons)

    summary = {
        "schema_version": 1,
        "source": "zeroth",
        "input_utterances": len(rows),
        "script_groups": len(groups),
        "inconsistent_script_groups": len(inconsistent_groups),
        "rows_in_inconsistent_script_groups": sum(
            len(group) for group in inconsistent_groups.values()
        ),
        "number_format_rows_with_group_consensus": number_consensus_rows,
        "second_teacher_candidates": len(candidates),
        "candidate_splits": dict(
            sorted(Counter(row["split"] for row in candidates).items())
        ),
        "candidate_reasons": dict(sorted(reason_counts.items())),
    }
    return candidates, inconsistent_report, summary


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
    candidates, inconsistent, summary = triage_rows(read_jsonl(args.input.resolve()))
    write_jsonl_atomic(args.candidates_output.resolve(), candidates)
    write_json_atomic(args.inconsistent_output.resolve(), inconsistent)
    write_json_atomic(args.summary.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Second-teacher candidates: {args.candidates_output.resolve()}")
    print(f"Inconsistent scripts:      {args.inconsistent_output.resolve()}")
    print(f"Triage summary:            {args.summary.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
