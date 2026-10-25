"""Cross-check labeled transcripts with a local Whisper teacher model.

Teacher disagreement is evidence for review, never permission to overwrite the
human/source transcript. The script preserves raw predictions and the exact
normalization policy used for CER/WER.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import torch
import yaml
from jiwer import cer, wer
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/transcript_qc.yaml"
DEFAULT_INPUT = PROJECT_ROOT / "data/manifests/zeroth.audio_accepted.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/interim/zeroth/transcript_qc.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "reports/data_qc/zeroth_transcript_qc.summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-check a labeled audio manifest with local Whisper."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Run a deterministic random sample instead of the full manifest.",
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


def select_rows(
    rows: list[dict[str, Any]], sample_size: int | None, seed: int
) -> list[dict[str, Any]]:
    if sample_size is None:
        return rows
    if sample_size <= 0:
        raise ValueError("--sample-size must be greater than zero")
    if sample_size > len(rows):
        raise ValueError("--sample-size cannot exceed the input row count")
    indices = sorted(random.Random(seed).sample(range(len(rows)), sample_size))
    return [rows[index] for index in indices]


def normalize_for_evaluation(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).lower()
    characters = [
        " " if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in normalized
    ]
    return " ".join("".join(characters).split())


def score_pair(reference: str, prediction: str) -> dict[str, float | str | None]:
    reference_normalized = normalize_for_evaluation(reference)
    prediction_normalized = normalize_for_evaluation(prediction)
    reference_characters = re.sub(r"\s+", "", reference_normalized)
    prediction_characters = re.sub(r"\s+", "", prediction_normalized)
    return {
        "reference_normalized": reference_normalized,
        "prediction_normalized": prediction_normalized,
        "cer": (
            float(cer(reference_characters, prediction_characters))
            if reference_characters
            else None
        ),
        "wer": (
            float(wer(reference_normalized, prediction_normalized))
            if reference_normalized
            else None
        ),
    }


def classify_score(
    score: dict[str, Any], thresholds: dict[str, Any]
) -> tuple[str, list[str]]:
    flags: list[str] = []
    prediction = score.get("prediction_normalized") or ""
    primary_metric = str(thresholds.get("primary_metric", "cer"))
    if primary_metric not in {"cer", "wer"}:
        raise ValueError("primary_metric must be 'cer' or 'wer'")
    metric_value = score.get(primary_metric)
    review_above = float(
        thresholds.get("review_above", thresholds.get("cer_review_above", 0.10))
    )
    high_above = float(
        thresholds.get(
            "high_disagreement_above",
            thresholds.get("cer_high_disagreement_above", 0.30),
        )
    )
    if high_above <= review_above:
        raise ValueError("high disagreement threshold must exceed review threshold")
    if not prediction:
        flags.append("empty_teacher_prediction")
    if metric_value is not None:
        if metric_value > high_above:
            flags.append(
                "high_teacher_disagreement_number_format"
                if any(character.isdigit() for character in prediction)
                else "high_teacher_disagreement"
            )
        elif metric_value > review_above:
            flags.append(
                "teacher_disagreement_number_format"
                if any(character.isdigit() for character in prediction)
                else "teacher_disagreement"
            )
    return ("review" if flags else "pass"), flags


def resolve_audio_path(audio_path: str) -> str:
    path = Path(audio_path)
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {resolved}")
    return str(resolved)


def batches(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


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


def metric_summary(
    rows: list[dict[str, Any]], section_key: str, metric_key: str
) -> dict[str, float] | None:
    values = sorted(
        float(row[section_key][metric_key])
        for row in rows
        if isinstance(row[section_key].get(metric_key), (int, float))
        and math.isfinite(row[section_key][metric_key])
    )
    if not values:
        return None
    return {
        "minimum": round(values[0], 6),
        "p01": round(percentile(values, 0.01), 6),
        "median": round(median(values), 6),
        "mean": round(mean(values), 6),
        "p90": round(percentile(values, 0.90), 6),
        "p99": round(percentile(values, 0.99), 6),
        "maximum": round(values[-1], 6),
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
    all_rows = read_jsonl(args.input.resolve())
    rows = select_rows(all_rows, args.sample_size, int(config["random_seed"]))
    batch_size = args.batch_size or int(config["inference"]["batch_size"])
    if batch_size <= 0:
        raise ValueError("Batch size must be greater than zero")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Whisper transcript QC")

    model_config = config["model"]
    output_key = str(config.get("output_key", "transcript_qc"))
    if not output_key:
        raise ValueError("output_key must be a non-empty string")
    model_id = model_config["id"]
    revision = model_config["revision"]
    dtype = torch.float16
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    processor.tokenizer.clean_up_tokenization_spaces = False
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to("cuda:0")
    resolved_revision = getattr(model.config, "_commit_hash", None) or revision
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=dtype,
        device="cuda:0",
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    scored_rows: list[dict[str, Any]] = []
    progress = tqdm(total=len(rows), unit="file", desc="Transcript QC")
    for row_batch in batches(rows, batch_size):
        paths = [resolve_audio_path(row["audio_path"]) for row in row_batch]
        outputs = transcriber(
            paths,
            batch_size=batch_size,
            generate_kwargs={
                "language": model_config["language"],
                "task": model_config["task"],
            },
        )
        if isinstance(outputs, dict):
            outputs = [outputs]
        for original, output in zip(row_batch, outputs, strict=True):
            row = dict(original)
            prediction = str(output.get("text") or "").strip()
            score = score_pair(row["text_normalized"], prediction)
            status, flags = classify_score(score, config["thresholds"])
            row[output_key] = {
                "teacher_prediction": prediction,
                **score,
                "status": status,
                "flags": flags,
            }
            scored_rows.append(row)
        progress.update(len(row_batch))
    progress.close()

    elapsed = time.perf_counter() - started
    total_audio_sec = sum(float(row["duration_sec"]) for row in rows)
    status_counts = Counter(row[output_key]["status"] for row in scored_rows)
    flag_counts = Counter(
        flag for row in scored_rows for flag in row[output_key]["flags"]
    )
    summary = {
        "schema_version": 1,
        "pipeline_version": config["pipeline_version"],
        "output_key": output_key,
        "selection": {
            "input_utterances": len(all_rows),
            "processed_utterances": len(rows),
            "sample_size": args.sample_size,
            "random_seed": config["random_seed"],
        },
        "model": {
            "id": model_id,
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "dtype": model_config["dtype"],
            "language": model_config["language"],
            "task": model_config["task"],
        },
        "normalization": config["normalization"],
        "thresholds": config["thresholds"],
        "batch_size": batch_size,
        "status_counts": dict(sorted(status_counts.items())),
        "flag_counts": dict(sorted(flag_counts.items())),
        "metrics": {
            "cer": metric_summary(scored_rows, output_key, "cer"),
            "wer": metric_summary(scored_rows, output_key, "wer"),
        },
        "performance": {
            "audio_hours": round(total_audio_sec / 3600.0, 6),
            "elapsed_sec": round(elapsed, 3),
            "real_time_factor": round(elapsed / total_audio_sec, 6),
            "peak_vram_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        },
    }
    write_jsonl_atomic(args.output.resolve(), scored_rows)
    write_json_atomic(args.summary.resolve(), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Scored manifest: {args.output.resolve()}")
    print(f"QC summary:      {args.summary.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
