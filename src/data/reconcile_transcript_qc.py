"""Reconcile two local Whisper teachers into an auditable review queue."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from jiwer import cer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data/interim/zeroth/transcript_qc_teacher2.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/interim/zeroth/transcript_qc_reconciled.jsonl"
DEFAULT_REVIEW = PROJECT_ROOT / "data/interim/zeroth/transcript_qc_priority_review.jsonl"
DEFAULT_SUMMARY = (
    PROJECT_ROOT / "reports/data_qc/zeroth_transcript_qc.reconciled.summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconcile two Whisper teachers without changing source labels."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--teacher-agreement-cer", type=float, default=0.10)
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


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def teacher_agreement_cer(first: str, second: str) -> float | None:
    first_compact = compact(first)
    second_compact = compact(second)
    if not first_compact:
        return None
    return float(cer(first_compact, second_compact))


def classify_row(
    row: dict[str, Any], agreement_threshold: float
) -> dict[str, Any]:
    first = row["transcript_qc"]
    second = row["transcript_qc_teacher2"]
    first_prediction = first["prediction_normalized"]
    second_prediction = second["prediction_normalized"]
    agreement = teacher_agreement_cer(first_prediction, second_prediction)
    has_digits = any(
        character.isdigit() for character in first_prediction + second_prediction
    )
    triage_reasons = set(
        (row.get("transcript_triage") or {}).get("second_teacher_reasons") or []
    )
    source_inconsistent = "source_transcript_inconsistent_within_script" in triage_reasons
    teachers_agree = agreement is not None and agreement <= agreement_threshold
    both_high = first["cer"] > 0.30 and second["cer"] > 0.30

    if source_inconsistent:
        category = "source_transcript_inconsistent"
        action = "review"
        priority = "critical"
    elif second["cer"] <= 0.10:
        category = "teacher2_matches_reference"
        action = "provisional_accept"
        priority = "none"
    elif has_digits and teachers_agree:
        category = "number_format_teacher_consensus"
        action = "provisional_accept"
        priority = "none"
    elif both_high and teachers_agree:
        category = "both_teachers_high_disagreement_consensus"
        action = "review"
        priority = "critical"
    elif both_high:
        category = "both_teachers_high_disagreement_no_consensus"
        action = "review"
        priority = "high"
    elif teachers_agree:
        category = "both_teachers_moderate_disagreement_consensus"
        action = "review"
        priority = "medium"
    else:
        category = "teacher_predictions_ambiguous"
        action = "review"
        priority = "high"

    reference_chars = len(compact(second["reference_normalized"]))
    first_chars = len(compact(first_prediction))
    second_chars = len(compact(second_prediction))
    return {
        "category": category,
        "provisional_action": action,
        "priority": priority,
        "teacher_agreement_cer": (
            round(agreement, 8)
            if agreement is not None and math.isfinite(agreement)
            else None
        ),
        "has_digit_formatting": has_digits,
        "teacher1_reference_coverage": (
            round(first_chars / reference_chars, 8) if reference_chars else None
        ),
        "teacher2_reference_coverage": (
            round(second_chars / reference_chars, 8) if reference_chars else None
        ),
        "source_text_changed": False,
    }


def reconcile_rows(
    rows: list[dict[str, Any]], agreement_threshold: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    reconciled: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        result = classify_row(row, agreement_threshold)
        row["transcript_reconciliation"] = result
        reconciled.append(row)
        if result["provisional_action"] == "review":
            review.append(row)

    category_counts = Counter(
        row["transcript_reconciliation"]["category"] for row in reconciled
    )
    action_counts = Counter(
        row["transcript_reconciliation"]["provisional_action"] for row in reconciled
    )
    priority_counts = Counter(
        row["transcript_reconciliation"]["priority"] for row in review
    )
    summary = {
        "schema_version": 1,
        "source": "zeroth",
        "input_candidates": len(rows),
        "teacher_agreement_cer_threshold": agreement_threshold,
        "category_counts": dict(sorted(category_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "review_priority_counts": dict(sorted(priority_counts.items())),
        "review_splits": dict(sorted(Counter(row["split"] for row in review).items())),
        "source_texts_modified": 0,
    }
    return reconciled, review, summary


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
    if not 0.0 <= args.teacher_agreement_cer <= 1.0:
        raise ValueError("--teacher-agreement-cer must be between zero and one")
    reconciled, review, summary = reconcile_rows(
        read_jsonl(args.input.resolve()), args.teacher_agreement_cer
    )
    write_jsonl_atomic(args.output.resolve(), reconciled)
    write_jsonl_atomic(args.review_output.resolve(), review)
    write_json_atomic(args.summary.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Reconciled candidates: {args.output.resolve()}")
    print(f"Priority review queue: {args.review_output.resolve()}")
    print(f"Summary:               {args.summary.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
