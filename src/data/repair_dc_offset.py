"""Create auditable, DC-corrected copies of flagged training audio.

Raw corpus files are never modified. Only rows explicitly flagged with
``excessive_dc_offset`` are selected, and by default the tool refuses to touch
anything outside the training split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import xxhash
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data/interim/librispeech/audio_qc.jsonl"
DEFAULT_BASE_MANIFEST = PROJECT_ROOT / "data/manifests/librispeech.raw.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/librispeech/dc_corrected"
DEFAULT_REPAIRED_OUTPUT = (
    PROJECT_ROOT / "data/interim/librispeech/dc_repaired.jsonl"
)
DEFAULT_COMBINED_OUTPUT = (
    PROJECT_ROOT / "data/interim/librispeech/audio_repaired_manifest.jsonl"
)
DEFAULT_SUMMARY = PROJECT_ROOT / "reports/data_qc/librispeech_dc_repair.summary.json"


class DcRepairError(ValueError):
    """Raised when repair inputs or outputs violate the safety policy."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair DC offset in flagged training audio without changing raw files."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repaired-output", type=Path, default=DEFAULT_REPAIRED_OUTPUT)
    parser.add_argument("--combined-output", type=Path, default=DEFAULT_COMBINED_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--split", default="train")
    parser.add_argument("--reason", default="excessive_dc_offset")
    parser.add_argument("--peak-target", type=float, default=0.99)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
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
                raise DcRepairError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise DcRepairError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def select_candidates(
    rows: Iterable[dict[str, Any]], reason: str, allowed_split: str
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        audio_qc = row.get("audio_qc")
        if not isinstance(audio_qc, dict):
            raise DcRepairError(f"Row {row.get('id')!r} has no audio_qc object")
        reasons = audio_qc.get("qc_reasons", [])
        if reason not in reasons:
            continue
        if row.get("split") != allowed_split:
            raise DcRepairError(
                f"Refusing to repair {row.get('id')!r}: split is "
                f"{row.get('split')!r}, expected {allowed_split!r}"
            )
        candidates.append(row)
    ids = [str(row.get("id")) for row in candidates]
    if len(ids) != len(set(ids)):
        raise DcRepairError("Candidate manifest contains duplicate IDs")
    return sorted(candidates, key=lambda row: str(row["id"]))


def resolve_audio_path(audio_path: str) -> Path:
    path = Path(audio_path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def output_relative_path(row: dict[str, Any]) -> Path:
    required = ("source_dataset", "source_chapter_id", "utterance_id", "speaker_id")
    missing = [key for key in required if not isinstance(row.get(key), str)]
    if missing:
        raise DcRepairError(
            f"Row {row.get('id')!r} is missing fields: {', '.join(missing)}"
        )
    speaker_id = row["speaker_id"]
    prefix = "librispeech_"
    if not speaker_id.startswith(prefix):
        raise DcRepairError(f"Unexpected LibriSpeech speaker ID: {speaker_id!r}")
    raw_speaker_id = speaker_id[len(prefix) :]
    return (
        Path(row["source_dataset"])
        / raw_speaker_id
        / row["source_chapter_id"]
        / f"{row['utterance_id']}.flac"
    )


def checksum_xxh3_128(path: Path) -> str:
    digest = xxhash.xxh3_128()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repair_audio_file(
    input_path: Path,
    output_path: Path,
    peak_target: float,
    overwrite: bool,
) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    if not 0.0 < peak_target < 1.0:
        raise DcRepairError("peak_target must satisfy 0 < value < 1")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Generated output already exists: {output_path}; use --overwrite to rebuild"
        )

    audio, sample_rate = sf.read(input_path, dtype="float64", always_2d=True)
    if audio.size == 0:
        raise DcRepairError(f"Cannot repair empty audio: {input_path}")
    if not np.isfinite(audio).all():
        raise DcRepairError(f"Cannot repair non-finite audio: {input_path}")

    channel_means = np.mean(audio, axis=0, dtype=np.float64)
    corrected = audio - channel_means
    peak_before_gain = float(np.max(np.abs(corrected)))
    gain = min(1.0, peak_target / peak_before_gain) if peak_before_gain > 0 else 1.0
    corrected *= gain

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    sf.write(
        temporary,
        corrected,
        sample_rate,
        format="FLAC",
        subtype="PCM_16",
    )
    temporary.replace(output_path)

    decoded, decoded_rate = sf.read(output_path, dtype="float64", always_2d=True)
    if decoded_rate != sample_rate or decoded.shape != audio.shape:
        raise DcRepairError(f"Repaired audio shape/rate mismatch: {output_path}")
    decoded_means = np.mean(decoded, axis=0, dtype=np.float64)
    return {
        "method": "subtract_per_channel_mean_v1",
        "output_format": "flac_pcm16",
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "num_frames": int(audio.shape[0]),
        "mean_before": [round(float(value), 10) for value in channel_means],
        "abs_mean_before_max": round(float(np.max(np.abs(channel_means))), 10),
        "mean_after": [round(float(value), 10) for value in decoded_means],
        "abs_mean_after_max": round(float(np.max(np.abs(decoded_means))), 10),
        "peak_after_centering_before_gain": round(peak_before_gain, 10),
        "applied_gain": round(float(gain), 10),
        "peak_after_write": round(float(np.max(np.abs(decoded))), 10),
        "checksum_xxh3_128": checksum_xxh3_128(output_path),
    }


def _repair_task(
    task: tuple[str, str, float, bool]
) -> dict[str, Any]:
    input_text, output_text, peak_target, overwrite = task
    return repair_audio_file(
        Path(input_text), Path(output_text), peak_target, overwrite
    )


def manifest_audio_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


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


def summarize(
    scored_rows: list[dict[str, Any]],
    repaired_rows: list[dict[str, Any]],
    peak_target: float,
) -> dict[str, Any]:
    repair_values = [row["audio_repair"] for row in repaired_rows]
    before = sorted(float(value["abs_mean_before_max"]) for value in repair_values)
    after = sorted(float(value["abs_mean_after_max"]) for value in repair_values)
    return {
        "schema_version": 1,
        "source": "librispeech",
        "method": "subtract_per_channel_mean_v1",
        "raw_audio_modified": False,
        "input_utterances": len(scored_rows),
        "repaired_utterances": len(repaired_rows),
        "repaired_duration_hours": round(
            sum(float(row["duration_sec"]) for row in repaired_rows) / 3600.0, 6
        ),
        "splits": dict(sorted(Counter(row["split"] for row in repaired_rows).items())),
        "peak_target": peak_target,
        "gain_reduced_utterances": sum(
            float(value["applied_gain"]) < 1.0 for value in repair_values
        ),
        "minimum_applied_gain": min(
            (float(value["applied_gain"]) for value in repair_values), default=None
        ),
        "abs_dc_offset_before": {
            "minimum": min(before, default=None),
            "median": median(before) if before else None,
            "mean": mean(before) if before else None,
            "maximum": max(before, default=None),
        },
        "abs_dc_offset_after": {
            "minimum": min(after, default=None),
            "median": median(after) if after else None,
            "mean": mean(after) if after else None,
            "maximum": max(after, default=None),
        },
    }


def main() -> int:
    args = parse_args()
    if not 0.0 < args.peak_target < 1.0:
        raise DcRepairError("--peak-target must satisfy 0 < value < 1")
    workers = args.workers or min(8, max(1, (os.cpu_count() or 2) - 1))
    if workers <= 0:
        raise DcRepairError("--workers must be greater than zero")

    scored_rows = read_jsonl(args.input.resolve())
    base_rows = read_jsonl(args.base_manifest.resolve())
    candidates = select_candidates(scored_rows, args.reason, args.split)
    if not candidates:
        raise DcRepairError(f"No rows are flagged with {args.reason!r}")

    base_by_id = {str(row.get("id")): row for row in base_rows}
    if len(base_by_id) != len(base_rows):
        raise DcRepairError("Base manifest contains duplicate IDs")
    missing = [row["id"] for row in candidates if row["id"] not in base_by_id]
    if missing:
        raise DcRepairError(f"Repair candidates absent from base manifest: {missing[:5]}")

    output_dir = args.output_dir.resolve()
    output_paths = [output_dir / output_relative_path(row) for row in candidates]
    tasks = [
        (
            str(resolve_audio_path(row["audio_path"])),
            str(output_path),
            args.peak_target,
            args.overwrite,
        )
        for row, output_path in zip(candidates, output_paths, strict=True)
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        repair_metrics = list(
            tqdm(
                executor.map(_repair_task, tasks, chunksize=4),
                total=len(tasks),
                desc="DC repair",
                unit="file",
            )
        )

    repaired_rows: list[dict[str, Any]] = []
    for candidate, output_path, metrics in zip(
        candidates, output_paths, repair_metrics, strict=True
    ):
        base = dict(base_by_id[candidate["id"]])
        original_path = base["audio_path"]
        original_checksum = base.get("checksum_xxh3_128")
        base["audio_path"] = manifest_audio_path(output_path)
        base["checksum_xxh3_128"] = metrics["checksum_xxh3_128"]
        base["audio_repair"] = {
            **metrics,
            "reason": args.reason,
            "original_audio_path": original_path,
            "original_checksum_xxh3_128": original_checksum,
        }
        repaired_rows.append(base)

    repaired_by_id = {row["id"]: row for row in repaired_rows}
    combined_rows = [repaired_by_id.get(row["id"], row) for row in base_rows]
    summary = summarize(scored_rows, repaired_rows, args.peak_target)

    write_jsonl_atomic(args.repaired_output.resolve(), repaired_rows)
    write_jsonl_atomic(args.combined_output.resolve(), combined_rows)
    write_json_atomic(args.summary.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Repaired-only manifest: {args.repaired_output.resolve()}")
    print(f"Combined manifest:      {args.combined_output.resolve()}")
    print(f"Repair summary:         {args.summary.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
