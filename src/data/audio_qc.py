"""Score audio quality and speech activity without modifying source audio.

This stage deliberately produces scores and review candidates instead of
deleting data. Thresholds come from ``configs/data_pipeline.yaml`` and every
decision is recorded in the output JSONL for later audit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/data_pipeline.yaml"
DEFAULT_INPUT = PROJECT_ROOT / "data/manifests/zeroth.accepted.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/interim/zeroth/audio_qc.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports/data_qc/zeroth_audio_qc.summary.json"

_VAD_MODEL: Any = None
_WORKER_SETTINGS: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score signal quality and Silero VAD metrics for a manifest."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes; default is min(8, CPU count - 1).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N rows (for smoke testing).",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
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
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def resolve_audio_path(audio_path: str) -> Path:
    path = Path(audio_path)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def dbfs(amplitude: float) -> float | None:
    if not math.isfinite(amplitude) or amplitude <= 0.0:
        return None
    return 20.0 * math.log10(amplitude)


def calculate_signal_metrics(samples: Any) -> dict[str, float | bool | None]:
    """Calculate full-scale metrics for a one-dimensional NumPy signal."""

    import numpy as np

    if samples.size == 0:
        return {
            "empty_audio": True,
            "finite_samples": True,
            "rms_dbfs": None,
            "peak_dbfs": None,
            "clipping_fraction": None,
            "abs_dc_offset": None,
        }

    finite = bool(np.isfinite(samples).all())
    if not finite:
        return {
            "empty_audio": False,
            "finite_samples": False,
            "rms_dbfs": None,
            "peak_dbfs": None,
            "clipping_fraction": None,
            "abs_dc_offset": None,
        }

    samples64 = samples.astype(np.float64, copy=False)
    rms = float(np.sqrt(np.mean(np.square(samples64))))
    peak = float(np.max(np.abs(samples64)))
    return {
        "empty_audio": False,
        "finite_samples": True,
        "rms_dbfs": dbfs(rms),
        "peak_dbfs": dbfs(peak),
        "clipping_fraction": float(np.mean(np.abs(samples64) >= 0.999)),
        "abs_dc_offset": abs(float(np.mean(samples64))),
    }


def estimate_snr_db(samples: Any, timestamps: list[dict[str, int]]) -> float | None:
    """Estimate speech/noise power ratio using Silero speech timestamps."""

    import numpy as np

    if samples.size == 0 or not timestamps:
        return None
    speech_mask = np.zeros(samples.shape[0], dtype=bool)
    for timestamp in timestamps:
        start = max(0, min(samples.shape[0], int(timestamp["start"])))
        end = max(start, min(samples.shape[0], int(timestamp["end"])))
        speech_mask[start:end] = True
    if not speech_mask.any() or speech_mask.all():
        return None

    speech_power = float(np.mean(np.square(samples[speech_mask], dtype=np.float64)))
    noise_power = float(np.mean(np.square(samples[~speech_mask], dtype=np.float64)))
    if speech_power <= 0.0 or noise_power <= 0.0:
        return None
    return 10.0 * math.log10(speech_power / noise_power)


def classify_metrics(
    metrics: dict[str, Any], quality: dict[str, Any]
) -> tuple[list[str], list[str], str]:
    reject = quality["reject"]
    warn = quality["warn"]
    reasons: list[str] = []
    warnings: list[str] = []

    if metrics.get("decode_error"):
        reasons.append("decode_error")
    if metrics.get("empty_audio"):
        reasons.append("empty_audio")
    if metrics.get("finite_samples") is False:
        reasons.append("non_finite_samples")
    if metrics.get("sample_rate_matches") is False:
        reasons.append("unexpected_sample_rate")
    if metrics.get("channels_match") is False:
        reasons.append("unexpected_channels")

    rms = metrics.get("rms_dbfs")
    if rms is not None and rms < float(reject["min_rms_dbfs"]):
        reasons.append("rms_below_minimum")
    clipping = metrics.get("clipping_fraction")
    if clipping is not None and clipping > float(reject["max_clipping_fraction"]):
        reasons.append("excessive_clipping")
    dc_offset = metrics.get("abs_dc_offset")
    if dc_offset is not None and dc_offset > float(reject["max_abs_dc_offset"]):
        reasons.append("excessive_dc_offset")

    speech_ratio = metrics.get("speech_ratio")
    if speech_ratio is not None:
        if speech_ratio < float(warn["speech_ratio_below"]):
            warnings.append("low_speech_ratio")
        if speech_ratio > float(warn["speech_ratio_above"]):
            warnings.append("high_speech_ratio")
    if metrics.get("speech_segments") == 0:
        warnings.append("no_speech_detected")
    snr = metrics.get("estimated_snr_db")
    if snr is not None and snr < float(warn["estimated_snr_below_db"]):
        warnings.append("low_estimated_snr")
    if metrics.get("duration_matches_manifest") is False:
        warnings.append("duration_mismatch")

    status = "quarantine_candidate" if reasons else "review" if warnings else "pass"
    return reasons, warnings, status


def _init_worker(settings: dict[str, Any]) -> None:
    global _VAD_MODEL, _WORKER_SETTINGS
    import torch
    from silero_vad import load_silero_vad

    torch.set_num_threads(1)
    _WORKER_SETTINGS = settings
    _VAD_MODEL = load_silero_vad()


def inspect_one(task: tuple[int, str, float | None]) -> tuple[int, dict[str, Any]]:
    index, audio_path_text, manifest_duration = task
    import numpy as np
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps

    settings = _WORKER_SETTINGS
    audio_path = Path(audio_path_text)
    metrics: dict[str, Any] = {
        "decode_error": None,
        "sample_rate": None,
        "channels": None,
        "num_frames": None,
        "decoded_duration_sec": None,
        "duration_matches_manifest": None,
        "sample_rate_matches": None,
        "channels_match": None,
        "empty_audio": None,
        "finite_samples": None,
        "rms_dbfs": None,
        "peak_dbfs": None,
        "clipping_fraction": None,
        "abs_dc_offset": None,
        "speech_segments": None,
        "speech_duration_sec": None,
        "speech_ratio": None,
        "leading_silence_sec": None,
        "trailing_silence_sec": None,
        "estimated_snr_db": None,
    }

    try:
        audio, sample_rate = sf.read(
            audio_path, dtype="float32", always_2d=True
        )
        num_frames, channels = audio.shape
        duration = num_frames / sample_rate if sample_rate else None
        metrics.update(
            {
                "sample_rate": int(sample_rate),
                "channels": int(channels),
                "num_frames": int(num_frames),
                "decoded_duration_sec": duration,
                "sample_rate_matches": sample_rate == settings["sample_rate"],
                "channels_match": channels == settings["channels"],
                "duration_matches_manifest": (
                    manifest_duration is None
                    or duration is not None
                    and abs(duration - manifest_duration) <= 0.02
                ),
            }
        )
        mono = np.mean(audio, axis=1, dtype=np.float32)
        metrics.update(calculate_signal_metrics(mono))

        if metrics["empty_audio"] or not metrics["finite_samples"]:
            reasons, warnings, status = classify_metrics(metrics, settings["quality"])
            metrics.update(
                {"qc_reasons": reasons, "qc_warnings": warnings, "qc_status": status}
            )
            return index, metrics

        if sample_rate != settings["sample_rate"]:
            reasons, warnings, status = classify_metrics(metrics, settings["quality"])
            metrics.update(
                {"qc_reasons": reasons, "qc_warnings": warnings, "qc_status": status}
            )
            return index, metrics

        waveform = torch.from_numpy(np.ascontiguousarray(mono))
        timestamps = get_speech_timestamps(
            waveform,
            _VAD_MODEL,
            threshold=settings["vad"]["threshold"],
            sampling_rate=sample_rate,
            min_speech_duration_ms=settings["vad"]["min_speech_duration_ms"],
            min_silence_duration_ms=settings["vad"]["min_silence_duration_ms"],
            speech_pad_ms=settings["vad"]["speech_pad_ms"],
            return_seconds=False,
        )
        speech_samples = sum(item["end"] - item["start"] for item in timestamps)
        speech_ratio = speech_samples / num_frames if num_frames else 0.0
        leading_silence = timestamps[0]["start"] / sample_rate if timestamps else duration
        trailing_silence = (
            (num_frames - timestamps[-1]["end"]) / sample_rate
            if timestamps
            else duration
        )
        metrics.update(
            {
                "speech_segments": len(timestamps),
                "speech_duration_sec": speech_samples / sample_rate,
                "speech_ratio": speech_ratio,
                "leading_silence_sec": leading_silence,
                "trailing_silence_sec": trailing_silence,
                "estimated_snr_db": estimate_snr_db(mono, timestamps),
            }
        )
    except Exception as exc:
        metrics["decode_error"] = f"{type(exc).__name__}: {exc}"

    reasons, warnings, status = classify_metrics(metrics, settings["quality"])
    metrics.update(
        {"qc_reasons": reasons, "qc_warnings": warnings, "qc_status": status}
    )
    for key, value in list(metrics.items()):
        if isinstance(value, float) and math.isfinite(value):
            metrics[key] = round(value, 8)
    return index, metrics


def percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    values = sorted(
        float(row["audio_qc"][key])
        for row in rows
        if isinstance(row["audio_qc"].get(key), (int, float))
        and math.isfinite(row["audio_qc"][key])
    )
    if not values:
        return None
    return {
        "minimum": round(values[0], 6),
        "p01": round(percentile(values, 0.01), 6),
        "median": round(median(values), 6),
        "mean": round(mean(values), 6),
        "p99": round(percentile(values, 0.99), 6),
        "maximum": round(values[-1], 6),
    }


def build_summary(
    rows: list[dict[str, Any]], settings: dict[str, Any], workers: int
) -> dict[str, Any]:
    source_counts = Counter(
        str(row.get("source", "unknown")) for row in rows
    )
    source = next(iter(source_counts)) if len(source_counts) == 1 else "mixed"
    status_counts = Counter(row["audio_qc"]["qc_status"] for row in rows)
    reason_counts = Counter(
        reason for row in rows for reason in row["audio_qc"]["qc_reasons"]
    )
    warning_counts = Counter(
        warning for row in rows for warning in row["audio_qc"]["qc_warnings"]
    )
    return {
        "schema_version": 1,
        "pipeline_version": settings["pipeline_version"],
        "source": source,
        "source_counts": dict(sorted(source_counts.items())),
        "total_utterances": len(rows),
        "workers": workers,
        "vad": {
            "engine": "silero-vad",
            "version": version("silero-vad"),
            **settings["vad"],
        },
        "thresholds": settings["quality"],
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "metrics": {
            key: metric_summary(rows, key)
            for key in (
                "rms_dbfs",
                "peak_dbfs",
                "clipping_fraction",
                "abs_dc_offset",
                "speech_ratio",
                "leading_silence_sec",
                "trailing_silence_sec",
                "estimated_snr_db",
            )
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
    config = load_yaml(args.config.resolve())
    rows = read_jsonl(args.input.resolve())
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero")
        rows = rows[: args.limit]

    settings = {
        "pipeline_version": config["pipeline_version"],
        "sample_rate": int(config["audio"]["sample_rate"]),
        "channels": int(config["audio"]["channels"]),
        "vad": {
            key: config["vad"][key]
            for key in (
                "threshold",
                "min_speech_duration_ms",
                "min_silence_duration_ms",
                "speech_pad_ms",
            )
        },
        "quality": config["quality"],
    }
    automatic_workers = min(8, max(1, (os.cpu_count() or 2) - 1))
    workers = args.workers if args.workers is not None else automatic_workers
    if workers <= 0:
        raise ValueError("--workers must be greater than zero")

    tasks = [
        (
            index,
            str(resolve_audio_path(row["audio_path"])),
            row.get("duration_sec"),
        )
        for index, row in enumerate(rows)
    ]
    scored_rows: list[dict[str, Any]] = [dict(row) for row in rows]
    if tasks:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(settings,),
        ) as executor:
            results = executor.map(inspect_one, tasks, chunksize=8)
            for index, metrics in tqdm(
                results, total=len(tasks), unit="file", desc="Audio QC"
            ):
                scored_rows[index]["audio_qc"] = metrics

    summary = build_summary(scored_rows, settings, workers)
    write_jsonl_atomic(args.output.resolve(), scored_rows)
    write_json_atomic(args.summary.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Scored manifest: {args.output.resolve()}")
    print(f"QC summary:      {args.summary.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
