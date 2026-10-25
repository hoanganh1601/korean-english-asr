"""Apply auditable manual decisions to an acoustic-QC manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data/interim/zeroth/audio_qc.jsonl"
DEFAULT_DECISIONS = PROJECT_ROOT / "configs/data_decisions/zeroth_audio_qc.yaml"
DEFAULT_ACCEPTED = PROJECT_ROOT / "data/manifests/zeroth.audio_accepted.jsonl"
DEFAULT_QUARANTINE = PROJECT_ROOT / "data/manifests/zeroth.audio_quarantine.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports/data_qc/zeroth_audio_final.summary.json"
VALID_DECISIONS = {"accept", "quarantine"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize acoustic QC using explicit manual review decisions."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--accepted-output", type=Path, default=DEFAULT_ACCEPTED)
    parser.add_argument("--quarantine-output", type=Path, default=DEFAULT_QUARANTINE)
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
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def index_decisions(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    decisions: dict[str, dict[str, Any]] = {}
    for row in rows:
        utterance_id = row.get("id")
        decision = row.get("decision")
        reason = row.get("reason")
        if not isinstance(utterance_id, str) or not utterance_id:
            raise ValueError("Every manual decision requires a non-empty id")
        if decision not in VALID_DECISIONS:
            raise ValueError(
                f"Invalid decision for {utterance_id}: {decision!r}; "
                f"expected one of {sorted(VALID_DECISIONS)}"
            )
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"Manual decision {utterance_id} requires a reason")
        if utterance_id in decisions:
            raise ValueError(f"Duplicate manual decision: {utterance_id}")
        decisions[utterance_id] = row
    return decisions


def read_decisions(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() not in {".yaml", ".yml"}:
        return read_jsonl(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("decisions"), list):
        raise ValueError(f"Expected a YAML mapping with a decisions list: {path}")
    if not all(isinstance(row, dict) for row in payload["decisions"]):
        raise ValueError(f"Every decision must be a mapping: {path}")
    return payload["decisions"]


def finalize_rows(
    rows: list[dict[str, Any]], decisions: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_ids = {row.get("id") for row in rows}
    unknown_decisions = sorted(set(decisions) - input_ids)
    if unknown_decisions:
        raise ValueError(f"Manual decisions reference unknown IDs: {unknown_decisions[:5]}")

    accepted: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        audio_qc = dict(row.get("audio_qc") or {})
        decision = decisions.get(row["id"])
        automatic_status = audio_qc.get("qc_status")

        if decision is not None:
            final_status = decision["decision"]
            final_reason = decision["reason"]
            row["manual_review"] = decision
        elif automatic_status == "quarantine_candidate":
            final_status = "quarantine"
            final_reason = "automatic_hard_qc_failure"
        else:
            final_status = "accept"
            final_reason = (
                "accepted_with_warning" if automatic_status == "review" else "passed_audio_qc"
            )

        audio_qc["final_status"] = final_status
        audio_qc["final_reason"] = final_reason
        row["audio_qc"] = audio_qc
        (accepted if final_status == "accept" else quarantined).append(row)

    return accepted, quarantined


def build_summary(
    accepted: list[dict[str, Any]], quarantined: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": "zeroth",
        "total_utterances": len(accepted) + len(quarantined),
        "accepted_utterances": len(accepted),
        "quarantined_utterances": len(quarantined),
        "accepted_with_warning": sum(
            row["audio_qc"]["final_reason"] == "accepted_with_warning"
            for row in accepted
        ),
        "quarantine_reasons": dict(
            sorted(
                Counter(
                    row["audio_qc"]["final_reason"] for row in quarantined
                ).items()
            )
        ),
        "splits": {
            "accepted": dict(sorted(Counter(row["split"] for row in accepted).items())),
            "quarantined": dict(
                sorted(Counter(row["split"] for row in quarantined).items())
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
    rows = read_jsonl(args.input.resolve())
    decisions = index_decisions(read_decisions(args.decisions.resolve()))
    accepted, quarantined = finalize_rows(rows, decisions)
    summary = build_summary(accepted, quarantined)
    write_jsonl_atomic(args.accepted_output.resolve(), accepted)
    write_jsonl_atomic(args.quarantine_output.resolve(), quarantined)
    write_json_atomic(args.summary.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Accepted manifest:   {args.accepted_output.resolve()}")
    print(f"Quarantine manifest: {args.quarantine_output.resolve()}")
    print(f"Summary:             {args.summary.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
