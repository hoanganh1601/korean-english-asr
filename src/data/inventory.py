"""Build a read-only inventory of raw audio files.

The scanner never modifies files below ``data/raw``. It probes the first audio
stream with ffprobe and calculates an XXH3-128 checksum for duplicate detection.
Results are written atomically as JSON Lines plus a compact JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import xxhash
import yaml
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/data_pipeline.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/manifests/raw_audio_inventory.jsonl"
DEFAULT_SUMMARY = PROJECT_ROOT / "data/manifests/raw_audio_inventory.summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inventory raw audio without modifying source files."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes; 0 or omitted uses an automatic value.",
    )
    parser.add_argument("--ffprobe-timeout-sec", type=float, default=60.0)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return config


def resolve_project_path(path: Path | str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def discover_files(raw_dir: Path, allowed_extensions: Iterable[str]) -> list[Path]:
    allowed = {f".{ext.lower().lstrip('.')}" for ext in allowed_extensions}
    discovered: list[Path] = []
    visited_directories: set[tuple[int, int]] = set()

    # Raw corpora are commonly mounted outside the repository and exposed below
    # data/raw with symlinks. os.walk can follow those links; inode tracking keeps
    # an accidental cyclic link from making the scan recurse forever.
    for directory, child_directories, filenames in os.walk(raw_dir, followlinks=True):
        directory_path = Path(directory)
        try:
            stat = directory_path.stat()
        except OSError:
            child_directories.clear()
            continue

        directory_key = (stat.st_dev, stat.st_ino)
        if directory_key in visited_directories:
            child_directories.clear()
            continue
        visited_directories.add(directory_key)

        for filename in filenames:
            path = directory_path / filename
            if path.suffix.lower() in allowed and path.is_file():
                discovered.append(path)

    return sorted(discovered)


def xxh3_128_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = xxhash.xxh3_128()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def inspect_one(task: tuple[str, str, float, str]) -> dict[str, Any]:
    path_text, raw_dir_text, timeout_sec, ffprobe_text = task
    path = Path(path_text)
    raw_dir = Path(raw_dir_text)
    relative_path = path.relative_to(raw_dir)
    source = relative_path.parts[0] if len(relative_path.parts) > 1 else "_root"

    result: dict[str, Any] = {
        "source": source,
        "relative_path": relative_path.as_posix(),
        "extension": path.suffix.lower(),
        "status": "error",
        "error": None,
    }

    try:
        stat = path.stat()
        result.update(
            {
                "file_size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                "checksum_xxh3_128": xxh3_128_file(path),
            }
        )

        command = [
            ffprobe_text,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            (
                "format=duration,size,bit_rate:"
                "stream=codec_name,sample_rate,channels,channel_layout,"
                "sample_fmt,bits_per_sample"
            ),
            "-of",
            "json",
            str(path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "ffprobe returned a non-zero code"
            raise RuntimeError(detail)

        payload = json.loads(completed.stdout)
        streams = payload.get("streams") or []
        if not streams:
            raise RuntimeError("No audio stream found")

        stream = streams[0]
        format_info = payload.get("format") or {}
        result.update(
            {
                "duration_sec": optional_float(format_info.get("duration")),
                "codec_name": stream.get("codec_name"),
                "sample_rate": optional_int(stream.get("sample_rate")),
                "channels": optional_int(stream.get("channels")),
                "channel_layout": stream.get("channel_layout"),
                "sample_format": stream.get("sample_fmt"),
                "bits_per_sample": optional_int(stream.get("bits_per_sample")),
                "bit_rate": optional_int(format_info.get("bit_rate")),
                "status": "ok",
            }
        )
    except Exception as exc:  # Keep the batch alive and report the bad file.
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def build_summary(
    rows: list[dict[str, Any]], raw_dir: Path, workers: int
) -> dict[str, Any]:
    valid_durations = [
        row["duration_sec"]
        for row in rows
        if row.get("status") == "ok" and row.get("duration_sec") is not None
    ]
    duplicate_groups = Counter(
        row["checksum_xxh3_128"]
        for row in rows
        if row.get("checksum_xxh3_128") is not None
    )
    duplicate_files = sum(count - 1 for count in duplicate_groups.values() if count > 1)

    return {
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "workers": workers,
        "total_files": len(rows),
        "ok_files": sum(row.get("status") == "ok" for row in rows),
        "error_files": sum(row.get("status") != "ok" for row in rows),
        "total_duration_hours": round(sum(valid_durations) / 3600.0, 6),
        "duplicate_files_by_checksum": duplicate_files,
        "sources": dict(sorted(Counter(row["source"] for row in rows).items())),
        "extensions": dict(
            sorted(Counter(row["extension"] for row in rows).items())
        ),
        "codecs": dict(
            sorted(Counter(row.get("codec_name") or "unknown" for row in rows).items())
        ),
        "sample_rates": dict(
            sorted(
                Counter(str(row.get("sample_rate") or "unknown") for row in rows).items()
            )
        ),
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config.resolve())
    raw_dir = resolve_project_path(args.raw_dir or config["paths"]["raw_dir"])
    output = resolve_project_path(args.output)
    summary_path = resolve_project_path(args.summary)
    allowed_extensions = config["audio"]["allowed_extensions"]

    environment_ffprobe = Path(sys.executable).with_name("ffprobe")
    ffprobe = shutil.which("ffprobe")
    if environment_ffprobe.is_file() and os.access(environment_ffprobe, os.X_OK):
        ffprobe = str(environment_ffprobe)
    if ffprobe is None:
        raise FileNotFoundError(
            "ffprobe was not found in PATH or beside the active Python interpreter"
        )

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    configured_workers = config["processing"].get("workers", 0)
    requested_workers = args.workers if args.workers is not None else configured_workers
    auto_workers = min(32, max(1, (os.cpu_count() or 2) - 1))
    workers = requested_workers if requested_workers and requested_workers > 0 else auto_workers

    files = discover_files(raw_dir, allowed_extensions)
    tasks = [
        (str(path), str(raw_dir), args.ffprobe_timeout_sec, ffprobe)
        for path in files
    ]

    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(
                tqdm(
                    executor.map(inspect_one, tasks, chunksize=16),
                    total=len(tasks),
                    unit="file",
                    desc="Inventory",
                )
            )
    else:
        rows = []

    rows.sort(key=lambda row: row["relative_path"])
    summary = build_summary(rows, raw_dir, workers)
    write_jsonl_atomic(output, rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Inventory: {output}")
    print(f"Summary:   {summary_path}")
    return 0 if summary["error_files"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
