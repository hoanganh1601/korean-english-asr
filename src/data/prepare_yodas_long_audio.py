"""Prepare Long-Form Conversation Audio & Context Windows from yodas-ko-200h-v1.

Builds:
  1. Reconstructed full-video FLAC files + manifests for dev (119 videos) and
     test (123 videos) at data/processed/yodas_long_v1/{dev,test}/*.flac and
     artifacts/manifests/yodas_long_v1.{dev,test}.jsonl.
  2. Zero-copy indexed Long-Form 30s training windows manifest
     (artifacts/manifests/yodas_long_windows_v1.train.jsonl) pointing directly
     to byte offsets inside incoming-shards/shards/*.tar, packing consecutive
     utterances of the same video_id up to 29.0s with exact 0.02s timestamps
     and preceding conversation context (`prev_text`) for <|startofprev|>.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import os
import re
import sys
import tarfile
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
YODAS_ROOT = PROJECT_ROOT / "data" / "longaudio" / "yodas-ko-200h-v1"
SHARDS_DIR = YODAS_ROOT / "incoming-shards" / "shards"
RELEASE_DIR = YODAS_ROOT / "release" / "yodas-ko-200h-v1"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "yodas_long_v1"
MANIFESTS_DIR = PROJECT_ROOT / "artifacts" / "manifests"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "data_qc" / "yodas_long_v1.summary.json"

SAMPLE_RATE = 16000
MAX_WINDOW_SEC = 29.0
MAX_GAP_MERGE_SEC = 0.8
QC_SKIP_GAP_SEC = 0.55
FADE_SAMPLES = int(0.008 * SAMPLE_RATE)  # 8 ms gentle edge fade to avoid clicks
MAX_PREV_CHARS = 500


_NON_SPEECH_PAREN_RE = re.compile(
    r"\((?:웃음|박수|음악|효과음|배경음|자막|노래|함성|탄성|한숨|기침|침묵|BGM|bgm)[^\)]*\)|\[[^\]]{1,40}\]"
)


def normalize_ko_text(text: str, clean_captions: bool = False) -> str:
    text = unicodedata.normalize("NFC", text or "")
    if clean_captions:
        text = html.unescape(text)
        if any(marker in text for marker in ("♪", "♬", "저작권자")):
            return ""
        text = _NON_SPEECH_PAREN_RE.sub(" ", text)
        text = re.sub(r"</?[^>]+>", " ", text)
        # Fix missing space after sentence-ending punctuation between Hangul/digits
        text = re.sub(r"([가-힣\d])([\.!\?])([가-힣\d])", r"\1\2 \3", text)
        # Drop pure Hangul laugh/reaction markers like ㅎㅎㅎ / ㅋㅋㅋ
        text = re.sub(r"[ㅎㅋㅠㅜ]{2,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_utt_ordering(utt_id: str) -> tuple[int, float, float]:
    """Extract (utterance_index, orig_start_sec, orig_end_sec) from YODAS utt_id."""
    # Format: {video_id}-{idx:05d}-{start_cs:08d}-{end_cs:08d}
    parts = utt_id.rsplit("-", 3)
    if len(parts) == 4:
        try:
            idx = int(parts[1])
            start_s = int(parts[2]) / 100.0
            end_s = int(parts[3]) / 100.0
            return idx, start_s, end_s
        except ValueError:
            pass
    return 0, 0.0, 0.0


def index_tar_shards(shards_dir: Path) -> dict[tuple[str, str], tuple[int, int]]:
    """Scan all TAR shards once and map (archive_name, member_name) -> (offset_data, size)."""
    t0 = time.time()
    tar_files = sorted(shards_dir.glob("*.tar"))
    print(f"Indexing {len(tar_files)} TAR shards in {shards_dir}...")
    index: dict[tuple[str, str], tuple[int, int]] = {}
    for i, tar_path in enumerate(tar_files, 1):
        archive_name = tar_path.name
        with tarfile.open(tar_path, "r") as tf:
            for member in tf:
                if member.isfile() and member.name.endswith(".flac"):
                    index[(archive_name, member.name)] = (member.offset_data, member.size)
        if i % 6 == 0 or i == len(tar_files):
            print(f"  Indexed {i}/{len(tar_files)} shards ({len(index):,} FLAC members)")
    print(f"Finished shard indexing in {time.time() - t0:.1f}s.")
    return index


def apply_edge_fade(audio: np.ndarray, fade_samples: int = FADE_SAMPLES) -> np.ndarray:
    if len(audio) <= 2 * fade_samples or fade_samples <= 0:
        return audio
    out = audio.copy()
    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    out[:fade_samples] *= ramp
    out[-fade_samples:] *= ramp[::-1]
    return out


def compute_inserted_gap(prev_orig_end: float, cur_orig_start: float) -> float:
    raw_gap = cur_orig_start - prev_orig_end
    if raw_gap <= 0.0:
        return 0.0
    if raw_gap <= MAX_GAP_MERGE_SEC:
        return round(raw_gap, 2)
    return QC_SKIP_GAP_SEC


def load_grouped_videos(jsonl_path: Path) -> dict[str, list[dict[str, Any]]]:
    vids: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx, orig_s, orig_e = parse_utt_ordering(row["utt_id"])
            row["_utt_idx"] = idx
            row["_orig_start"] = orig_s
            row["_orig_end"] = orig_e
            vids[row["video_id"]].append(row)
    for vid in vids:
        vids[vid].sort(key=lambda r: (r["_utt_idx"], r["_orig_start"], r["utt_id"]))
    return vids


def read_flac_from_tar(
    tar_handles: dict[str, Any],
    shards_dir: Path,
    archive_name: str,
    offset_data: int,
    size: int,
) -> np.ndarray:
    fh = tar_handles.get(archive_name)
    if fh is None:
        fh = open(shards_dir / archive_name, "rb")
        tar_handles[archive_name] = fh
    fh.seek(offset_data)
    raw = fh.read(size)
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr} Hz in {archive_name}")
    return audio


def build_eval_long_videos(
    split: str,
    jsonl_path: Path,
    shard_index: dict[tuple[str, str], tuple[int, int]],
    tar_handles: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    print(f"\nBuilding reconstructed long-form videos for split='{split}'...")
    t0 = time.time()
    vids = load_grouped_videos(jsonl_path)
    out_audio_dir = PROCESSED_DIR / split
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    out_manifest = MANIFESTS_DIR / f"yodas_long_v1.{split}.jsonl"

    durations: list[float] = []
    utt_counts: list[int] = []
    records_written = 0

    with open(out_manifest, "w", encoding="utf-8") as mf:
        for i, (video_id, rows) in enumerate(sorted(vids.items()), 1):
            chunks: list[np.ndarray] = []
            segments_meta: list[dict[str, Any]] = []
            texts: list[str] = []
            cursor_samples = 0

            for k, r in enumerate(rows):
                key = (r["archive"], r["audio_member"])
                offset_data, size = shard_index[key]
                audio = read_flac_from_tar(tar_handles, SHARDS_DIR, r["archive"], offset_data, size)
                audio = apply_edge_fade(audio)

                if k > 0:
                    gap_sec = compute_inserted_gap(rows[k - 1]["_orig_end"], r["_orig_start"])
                    if gap_sec > 0:
                        gap_samples = int(round(gap_sec * SAMPLE_RATE))
                        chunks.append(np.zeros(gap_samples, dtype=np.float32))
                        cursor_samples += gap_samples

                seg_start_sec = round(cursor_samples / SAMPLE_RATE, 3)
                chunks.append(audio)
                cursor_samples += len(audio)
                seg_end_sec = round(cursor_samples / SAMPLE_RATE, 3)

                clean_txt = normalize_ko_text(r["text"])
                if clean_txt:
                    texts.append(clean_txt)
                    segments_meta.append({
                        "utt_id": r["utt_id"],
                        "start_sec": seg_start_sec,
                        "end_sec": seg_end_sec,
                        "text": clean_txt,
                    })

            if not chunks or not texts:
                continue

            full_audio = np.concatenate(chunks)
            total_dur = len(full_audio) / SAMPLE_RATE
            out_flac = out_audio_dir / f"{video_id}.flac"
            sf.write(str(out_flac), full_audio, SAMPLE_RATE, subtype="PCM_16")

            full_text = " ".join(texts)
            record = {
                "id": f"yodas_long_{split}_{video_id}",
                "video_id": video_id,
                "audio_path": str(out_flac.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "text": full_text,
                "text_normalized": full_text,
                "language": "ko",
                "duration_sec": round(total_dur, 3),
                "sample_rate": SAMPLE_RATE,
                "num_utterances": len(segments_meta),
                "segments": segments_meta,
                "source": "yodas-ko-200h-v1",
                "split": split,
            }
            mf.write(json.dumps(record, ensure_ascii=False) + "\n")
            durations.append(total_dur)
            utt_counts.append(len(segments_meta))
            records_written += 1

            if i % 30 == 0 or i == len(vids):
                print(f"  [{split}] {i}/{len(vids)} videos built...")

    durations_sorted = sorted(durations)
    stats = {
        "split": split,
        "num_videos": records_written,
        "total_utterances": sum(utt_counts),
        "total_hours": round(sum(durations) / 3600.0, 4),
        "min_duration_sec": round(durations_sorted[0], 2) if durations_sorted else 0.0,
        "median_duration_sec": round(durations_sorted[len(durations_sorted) // 2], 2) if durations_sorted else 0.0,
        "max_duration_sec": round(durations_sorted[-1], 2) if durations_sorted else 0.0,
        "videos_over_30s": sum(1 for d in durations if d > 30.0),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    print(f"  -> {out_manifest}: {stats}")
    return out_manifest, stats


def build_train_long_windows(
    jsonl_path: Path,
    shard_index: dict[tuple[str, str], tuple[int, int]],
    max_window_sec: float = MAX_WINDOW_SEC,
    *,
    out_manifest: Path = MANIFESTS_DIR / "yodas_long_windows_v1.train.jsonl",
    clean_captions: bool = False,
    min_window_sec: float = 0.0,
    min_segments: int = 1,
    min_video_sec: float = 0.0,
    min_chars_per_sec: float = 0.0,
    max_chars_per_sec: float = 999.0,
    quarantine_video_ids: set[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    print(f"\nPacking training long-form 30s windows with conversation context -> {out_manifest.name}...")
    t0 = time.time()
    vids = load_grouped_videos(jsonl_path)
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

    window_durations: list[float] = []
    segs_per_window: list[int] = []
    windows_with_prev = 0
    total_windows = 0
    total_utts = 0
    kept_videos = 0
    skipped_videos = 0
    skipped_short_windows = 0
    skipped_bad_utts = 0

    def quantize_02(val: float) -> float:
        return round(round(val / 0.02) * 0.02, 2)

    with open(out_manifest, "w", encoding="utf-8") as mf:
        for video_id, rows in sorted(vids.items()):
            if quarantine_video_ids and video_id in quarantine_video_ids:
                skipped_videos += 1
                continue
            vid_dur = sum(float(r["duration"]) for r in rows)
            if vid_dur < min_video_sec:
                skipped_videos += 1
                continue

            # Filter valid rows present in shard_index
            valid_rows = []
            for r in rows:
                txt = normalize_ko_text(r["text"], clean_captions=clean_captions)
                dur_r = float(r["duration"])
                cps = len(re.sub(r"\s+", "", txt)) / max(0.1, dur_r)
                if clean_captions and (cps < min_chars_per_sec or cps > max_chars_per_sec):
                    skipped_bad_utts += 1
                    continue
                key = (r["archive"], r["audio_member"])
                if txt and key in shard_index and dur_r <= max_window_sec:
                    offset_data, size = shard_index[key]
                    r_copy = dict(r)
                    r_copy["_clean_text"] = txt
                    r_copy["_offset_data"] = offset_data
                    r_copy["_size"] = size
                    valid_rows.append(r_copy)
                elif not txt:
                    skipped_bad_utts += 1

            if not valid_rows:
                skipped_videos += 1
                continue

            # Greedily pack consecutive utterances into <= max_window_sec windows
            windows: list[list[dict[str, Any]]] = []
            cur_win: list[dict[str, Any]] = []
            cur_dur = 0.0

            for r in valid_rows:
                dur = float(r["duration"])
                gap = compute_inserted_gap(cur_win[-1]["_orig_end"], r["_orig_start"]) if cur_win else 0.0
                projected = cur_dur + gap + dur
                if cur_win and projected > max_window_sec:
                    windows.append(cur_win)
                    cur_win = [r]
                    cur_dur = dur
                else:
                    cur_win.append(r)
                    cur_dur = projected

            if cur_win:
                windows.append(cur_win)

            history_texts: list[str] = []
            vid_wrote_any = False
            for win_idx, win_rows in enumerate(windows):
                # Build preceding context string from recent previous utterances in same video
                prev_text = " ".join(history_texts).strip()
                if len(prev_text) > MAX_PREV_CHARS:
                    prev_text = prev_text[-MAX_PREV_CHARS:]
                    # Trim to word boundary to avoid broken syllable tokens
                    first_space = prev_text.find(" ")
                    if first_space != -1 and first_space < len(prev_text) - 10:
                        prev_text = prev_text[first_space + 1 :]

                cursor = 0.0
                win_segments: list[dict[str, Any]] = []
                win_texts: list[str] = []
                archive_name = win_rows[0]["archive"]

                for k, r in enumerate(win_rows):
                    dur = float(r["duration"])
                    gap = compute_inserted_gap(win_rows[k - 1]["_orig_end"], r["_orig_start"]) if k > 0 else 0.0
                    cursor += gap
                    s_ts = quantize_02(cursor)
                    e_ts = quantize_02(min(30.0, cursor + dur))
                    if e_ts <= s_ts:
                        e_ts = quantize_02(min(30.0, s_ts + 0.02))
                    cursor += dur

                    win_segments.append({
                        "utt_id": r["utt_id"],
                        "offset_data": r["_offset_data"],
                        "size": r["_size"],
                        "gap_before_sec": round(gap, 3),
                        "start_sec": s_ts,
                        "end_sec": e_ts,
                        "text": r["_clean_text"],
                    })
                    win_texts.append(r["_clean_text"])
                    history_texts.append(r["_clean_text"])

                win_dur = round(min(30.0, cursor), 3)
                if win_dur < min_window_sec or len(win_segments) < min_segments:
                    skipped_short_windows += 1
                    continue

                full_win_text = " ".join(win_texts)
                rec = {
                    "id": f"yodas_win_{video_id}_{win_idx:04d}",
                    "video_id": video_id,
                    "window_index": win_idx,
                    "archive": archive_name,
                    "duration_sec": win_dur,
                    "prev_text": prev_text,
                    "text_normalized": full_win_text,
                    "language": "ko",
                    "segments": win_segments,
                }
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                window_durations.append(rec["duration_sec"])
                segs_per_window.append(len(win_segments))
                if prev_text:
                    windows_with_prev += 1
                total_windows += 1
                total_utts += len(win_segments)
                vid_wrote_any = True

            if vid_wrote_any:
                kept_videos += 1

    stats = {
        "split": "train_windows",
        "num_videos": kept_videos,
        "skipped_videos": skipped_videos,
        "skipped_short_windows": skipped_short_windows,
        "skipped_bad_utts": skipped_bad_utts,
        "num_windows": total_windows,
        "total_utterances_packed": total_utts,
        "total_hours": round(sum(window_durations) / 3600.0, 4),
        "avg_window_duration_sec": round(sum(window_durations) / max(1, total_windows), 2),
        "avg_utterances_per_window": round(sum(segs_per_window) / max(1, total_windows), 2),
        "windows_with_prev_context": windows_with_prev,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    print(f"  -> {out_manifest}: {stats}")
    return out_manifest, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare YODAS-ko-200h-v1 Long-Form audio & manifests.")
    p.add_argument("--skip-dev", action="store_true", help="Skip building reconstructed dev FLAC videos.")
    p.add_argument("--skip-test", action="store_true", help="Skip building reconstructed test FLAC videos.")
    p.add_argument("--skip-train", action="store_true", help="Skip packing train long-form windows manifest.")
    p.add_argument(
        "--clean-long-v2",
        action="store_true",
        help="Build cleaned long-audio-only train windows (>=15s, >=2 segs, video>=45s, cleaned captions).",
    )
    p.add_argument("--train-manifest", type=Path, default=None, help="Override output train windows JSONL path.")
    p.add_argument("--quarantine-videos", type=Path, default=None, help="JSON/TXT list of quarantined video_ids.")
    p.add_argument("--min-window-sec", type=float, default=0.0)
    p.add_argument("--min-segments", type=int, default=1)
    p.add_argument("--min-video-sec", type=float, default=0.0)
    return p.parse_args()


def _load_quarantine_ids(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return set(data.get("quarantine_video_ids", []))
        return set(data)
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> int:
    args = parse_args()
    shard_index = index_tar_shards(SHARDS_DIR)
    tar_handles: dict[str, Any] = {}
    summary: dict[str, Any] = {}

    if args.clean_long_v2:
        out_train = args.train_manifest or (MANIFESTS_DIR / "yodas_long_windows_clean_v2.train.jsonl")
        clean_caps = True
        min_win = args.min_window_sec if args.min_window_sec > 0 else 15.0
        min_segs = args.min_segments if args.min_segments > 1 else 2
        min_vid = args.min_video_sec if args.min_video_sec > 0 else 45.0
        min_cps, max_cps = 1.2, 16.0
    else:
        out_train = args.train_manifest or (MANIFESTS_DIR / "yodas_long_windows_v1.train.jsonl")
        clean_caps = False
        min_win = args.min_window_sec
        min_segs = args.min_segments
        min_vid = args.min_video_sec
        min_cps, max_cps = 0.0, 999.0

    quarantine_ids = _load_quarantine_ids(args.quarantine_videos)

    try:
        if not args.skip_test:
            _, summary["test"] = build_eval_long_videos(
                "test", RELEASE_DIR / "test.jsonl", shard_index, tar_handles
            )
        if not args.skip_dev:
            _, summary["dev"] = build_eval_long_videos(
                "dev", RELEASE_DIR / "dev.jsonl", shard_index, tar_handles
            )
        if not args.skip_train:
            _, summary["train_windows"] = build_train_long_windows(
                RELEASE_DIR / "train.jsonl",
                shard_index,
                out_manifest=out_train,
                clean_captions=clean_caps,
                min_window_sec=min_win,
                min_segments=min_segs,
                min_video_sec=min_vid,
                min_chars_per_sec=min_cps,
                max_chars_per_sec=max_cps,
                quarantine_video_ids=quarantine_ids,
            )
    finally:
        for fh in tar_handles.values():
            fh.close()

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nSummary saved -> {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
