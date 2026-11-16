"""Long-Form Video/Conversation ASR Benchmark Runner.

Evaluates WhisperASRRunner (with Sequential Timestamp + Context Carry-Over)
on reconstructed full-video audio files (>30s, typically 2-10+ minutes) in
artifacts/manifests/yodas_long_v1.test.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.evaluation.buzzasr_runner import (
    WhisperASRRunner,
    compute_cer,
    compute_edit_distance,
    compute_wer,
)
from src.evaluation.text_normalizer import normalize_pair_for_evaluation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark ASR on Long-Form reconstructed videos.")
    p.add_argument(
        "--adapter",
        type=str,
        default="artifacts/checkpoints/buzzasr_lora_v5_full/final",
        help="Path to LoRA adapter directory (or 'base' for unadapted BuzzASR/korean).",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "manifests" / "yodas_long_v1.test.jsonl",
    )
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--limit", type=int, default=0, help="Max videos to evaluate (<=0 for all).")
    p.add_argument("--min-dur", type=float, default=30.1, help="Min video duration in seconds (default: >30s).")
    p.add_argument("--max-dur", type=float, default=99999.0, help="Max video duration in seconds.")
    p.add_argument("--language", type=str, default=None, help="Optional language hint ('ko', 'en', or None).")
    p.add_argument(
        "--longform-mode",
        choices=["vad_split", "native"],
        default="vad_split",
        help="Long-form strategy: 'vad_split' (silence-aware chunking + merge) or 'native' (HF 30s seek).",
    )
    p.add_argument("--no-context-carryover", action="store_true", help="Disable condition_on_prev_tokens.")
    p.add_argument(
        "--exclude-caption-artifacts",
        action="store_true",
        help="Skip YouTube caption-artifact videos containing music/copyright markers.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    adapter = None if args.adapter.lower() == "base" else args.adapter

    rows: list[dict[str, Any]] = []
    with open(args.manifest, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            dur = float(r["duration_sec"])
            txt = r.get("text_normalized") or r.get("text", "")
            if args.exclude_caption_artifacts and any(
                marker in txt for marker in ("♪", "♬", "&lt;", "&gt;", "저작권자")
            ):
                continue
            if args.min_dur <= dur <= args.max_dur:
                rows.append(r)

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    print(f"Loaded {len(rows)} long-form videos (dur in [{args.min_dur}s, {args.max_dur}s]).", flush=True)
    runner = WhisperASRRunner(adapter_path=adapter)
    runner.warmup()

    details: list[dict[str, Any]] = []
    total_audio = 0.0
    total_infer = 0.0
    peak_vram = 0.0
    total_word_edits = 0
    total_ref_words = 0
    total_char_edits = 0
    total_ref_chars = 0

    for idx, row in enumerate(rows, 1):
        audio_abs = PROJECT_ROOT / row["audio_path"]
        out = runner.transcribe_file(
            str(audio_abs),
            language=args.language,
            condition_on_prev_tokens=not args.no_context_carryover,
            force_long_form=True,
            longform_mode=args.longform_mode,
        )
        ref_raw = row.get("text_normalized") or row.get("text", "")
        pred_raw = out["transcription"]
        ref_norm, pred_norm = normalize_pair_for_evaluation(ref_raw, pred_raw)

        ref_words = ref_norm.split()
        pred_words = pred_norm.split()
        w_edits = compute_edit_distance(ref_words, pred_words)
        wer_val = w_edits / len(ref_words) if ref_words else (0.0 if not pred_words else 1.0)

        ref_chars = list(re.sub(r"\s+", "", ref_norm))
        pred_chars = list(re.sub(r"\s+", "", pred_norm))
        c_edits = compute_edit_distance(ref_chars, pred_chars)
        cer_val = c_edits / len(ref_chars) if ref_chars else (0.0 if not pred_chars else 1.0)

        total_word_edits += w_edits
        total_ref_words += len(ref_words)
        total_char_edits += c_edits
        total_ref_chars += len(ref_chars)
        total_audio += out["audio_duration_sec"]
        total_infer += out["inference_time_sec"]
        peak_vram = max(peak_vram, out["peak_vram_mb"])

        print(
            f"[{idx}/{len(rows)}] {row['video_id']} | dur={out['audio_duration_sec']:.1f}s "
            f"| utts={row.get('num_utterances', '?')} | pred_segs={len(out.get('segments', []))} "
            f"| infer={out['inference_time_sec']:.1f}s (RTF {out['real_time_factor']:.3f}) "
            f"| WER={wer_val*100:.2f}% | CER={cer_val*100:.2f}%",
            flush=True,
        )
        details.append({
            "video_id": row["video_id"],
            "duration_sec": out["audio_duration_sec"],
            "num_ref_utterances": row.get("num_utterances", 0),
            "num_pred_segments": len(out.get("segments", [])),
            "infer_sec": round(out["inference_time_sec"], 3),
            "rtf": round(out["real_time_factor"], 4),
            "wer_pct": round(wer_val * 100, 2),
            "cer_pct": round(cer_val * 100, 2),
            "ref_words": len(ref_words),
            "word_edits": w_edits,
            "ref_chars": len(ref_chars),
            "char_edits": c_edits,
            "ref_raw": ref_raw,
            "pred_raw": pred_raw,
        })

        if args.output_json and (idx % 5 == 0 or idx == len(rows)):
            _save_results(
                args.output_json, args, details, total_audio, total_infer,
                peak_vram, total_word_edits, total_ref_words, total_char_edits, total_ref_chars,
            )

    if args.output_json:
        _save_results(
            args.output_json, args, details, total_audio, total_infer,
            peak_vram, total_word_edits, total_ref_words, total_char_edits, total_ref_chars,
        )

    corpus_wer = (total_word_edits / max(1, total_ref_words)) * 100
    corpus_cer = (total_char_edits / max(1, total_ref_chars)) * 100
    macro_wer = sum(d["wer_pct"] for d in details) / max(1, len(details))
    macro_cer = sum(d["cer_pct"] for d in details) / max(1, len(details))
    overall_rtf = total_infer / total_audio if total_audio > 0 else 0.0

    print("\n================ LONG-FORM BENCHMARK SUMMARY ================", flush=True)
    print(f"Adapter:             {args.adapter}", flush=True)
    print(f"Long-Form Mode:      {args.longform_mode}", flush=True)
    print(f"Context Carry-Over:  {not args.no_context_carryover}", flush=True)
    print(f"Videos Evaluated:    {len(details)}", flush=True)
    print(f"Total Audio:         {total_audio/60:.2f} min ({total_audio:.1f}s)", flush=True)
    print(f"Total Infer Time:    {total_infer:.1f}s (RTF: {overall_rtf:.3f})", flush=True)
    print(f"Corpus WER (Micro):  {corpus_wer:.2f}%", flush=True)
    print(f"Corpus CER (Micro):  {corpus_cer:.2f}%", flush=True)
    print(f"Avg Video WER (Mac): {macro_wer:.2f}%", flush=True)
    print(f"Avg Video CER (Mac): {macro_cer:.2f}%", flush=True)
    print(f"Peak VRAM:           {peak_vram:.1f} MB ({peak_vram/1024:.2f} GB)", flush=True)
    print("=============================================================\n", flush=True)
    return 0


def _save_results(
    out_path: Path,
    args: argparse.Namespace,
    details: list[dict[str, Any]],
    total_audio: float,
    total_infer: float,
    peak_vram: float,
    total_word_edits: int,
    total_ref_words: int,
    total_char_edits: int,
    total_ref_chars: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "adapter": args.adapter,
        "manifest": str(args.manifest),
        "longform_mode": args.longform_mode,
        "condition_on_prev_tokens": not args.no_context_carryover,
        "language_hint": args.language,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "videos_evaluated": len(details),
        "total_audio_sec": round(total_audio, 2),
        "total_infer_sec": round(total_infer, 2),
        "overall_rtf": round(total_infer / total_audio, 4) if total_audio > 0 else 0.0,
        "corpus_wer_pct": round((total_word_edits / max(1, total_ref_words)) * 100, 2),
        "corpus_cer_pct": round((total_char_edits / max(1, total_ref_chars)) * 100, 2),
        "macro_wer_pct": round(sum(d["wer_pct"] for d in details) / max(1, len(details)), 2),
        "macro_cer_pct": round(sum(d["cer_pct"] for d in details) / max(1, len(details)), 2),
        "peak_vram_mb": round(peak_vram, 1),
        "details": details,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
