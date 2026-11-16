"""Batch-GPU audio-text consistency QC filter for yodas-ko-200h-v1 videos.

Samples up to K representative utterances per video_id across train/dev/test,
transcribes them in batches with Whisper (unconstrained language), and flags
videos whose audio clearly mismatches the YouTube caption (e.g. English speech
with Korean-translated captions, or music/BGM slideshows with overlay text).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "hub")

from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.data.prepare_yodas_long_audio import (
    RELEASE_DIR,
    SAMPLE_RATE,
    SHARDS_DIR,
    index_tar_shards,
    load_grouped_videos,
    normalize_ko_text,
    read_flac_from_tar,
)
from src.evaluation.buzzasr_runner import compute_edit_distance
from src.evaluation.text_normalizer import normalize_pair_for_evaluation


def _pick_representative_utts(rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    candidates = [r for r in rows if float(r["duration"]) >= 2.0 and normalize_ko_text(r["text"], clean_captions=True)]
    if not candidates:
        candidates = [r for r in rows if normalize_ko_text(r["text"], clean_captions=True)]
    if len(candidates) <= k:
        return candidates
    idxs = sorted({int(round(i * (len(candidates) - 1) / max(1, k - 1))) for i in range(k)})
    return [candidates[i] for i in idxs]


def _latin_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-z가-힣]", text)
    if not letters:
        return 0.0
    latin = sum(1 for ch in letters if ("A" <= ch <= "Z" or "a" <= ch <= "z"))
    return latin / len(letters)


def _compute_cer(ref_raw: str, pred_raw: str) -> float:
    ref_n, pred_n = normalize_pair_for_evaluation(ref_raw, pred_raw)
    rc = list(re.sub(r"\s+", "", ref_n))
    pc = list(re.sub(r"\s+", "", pred_n))
    if not rc:
        return 0.0 if not pc else 1.0
    return compute_edit_distance(rc, pc) / len(rc)


def main() -> int:
    p = argparse.ArgumentParser(description="Fast batched GPU audio-text QC for YODAS long videos.")
    p.add_argument(
        "--adapter",
        type=str,
        default="artifacts/checkpoints/buzzasr_lora_v5_full/final",
        help="LoRA adapter path used for QC transcription.",
    )
    p.add_argument("--splits", nargs="+", default=["test", "dev", "train"])
    p.add_argument("--samples-per-video", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--cer-threshold", type=float, default=0.70, help="Quarantine if avg CER >= threshold.")
    p.add_argument("--min-cer-floor", type=float, default=0.50, help="Require all sampled utts >= this CER.")
    p.add_argument(
        "--output-qc",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "manifests" / "yodas_v2_video_qc.json",
    )
    args = p.parse_args()

    t0 = time.time()
    shard_index = index_tar_shards(SHARDS_DIR)
    tar_handles: dict[str, Any] = {}

    print(f"Loading BuzzASR/korean + {args.adapter} for batched QC...")
    processor = WhisperProcessor.from_pretrained("BuzzASR/korean")
    model = WhisperForConditionalGeneration.from_pretrained(
        "BuzzASR/korean", torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    adapter_abs = (PROJECT_ROOT / args.adapter).resolve()
    if adapter_abs.exists():
        model = PeftModel.from_pretrained(model, str(adapter_abs))
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []
    model.generation_config.language = None
    model = model.to("cuda").eval()

    probes: list[dict[str, Any]] = []
    for split in args.splits:
        vids = load_grouped_videos(RELEASE_DIR / f"{split}.jsonl")
        for vid, rows in sorted(vids.items()):
            for r in _pick_representative_utts(rows, args.samples_per_video):
                key = (r["archive"], r["audio_member"])
                if key in shard_index:
                    probes.append({
                        "split": split,
                        "video_id": vid,
                        "archive": r["archive"],
                        "offset_data": shard_index[key][0],
                        "size": shard_index[key][1],
                        "ref": normalize_ko_text(r["text"], clean_captions=True),
                    })

    probes.sort(key=lambda x: len(x["ref"]))
    print(f"Probing {len(probes):,} utterances across splits={args.splits} (batch_size={args.batch_size})...")
    per_video: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    try:
        for i in range(0, len(probes), args.batch_size):
            batch_items = probes[i : i + args.batch_size]
            audios = [
                read_flac_from_tar(tar_handles, SHARDS_DIR, item["archive"], item["offset_data"], item["size"])
                for item in batch_items
            ]
            feats = processor.feature_extractor(
                audios, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            ).input_features.to("cuda", dtype=torch.float16)

            max_ref_len = max(len(x["ref"]) for x in batch_items)
            batch_max_tok = min(96, max(24, int(max_ref_len * 1.4)))
            with torch.inference_mode():
                gen_ids = model.generate(
                    feats,
                    num_beams=1,
                    use_cache=True,
                    max_new_tokens=batch_max_tok,
                    no_repeat_ngram_size=6,
                    repetition_penalty=1.05,
                )
            preds = processor.batch_decode(gen_ids, skip_special_tokens=True)

            for item, pred in zip(batch_items, preds):
                cer = _compute_cer(item["ref"], pred)
                per_video[(item["split"], item["video_id"])].append({
                    "cer": round(cer, 4),
                    "pred_latin": round(_latin_ratio(pred), 3),
                    "ref_latin": round(_latin_ratio(item["ref"]), 3),
                    "ref": item["ref"][:80],
                    "pred": pred.strip()[:80],
                })

            done = min(i + args.batch_size, len(probes))
            if done % (args.batch_size * 10) == 0 or done == len(probes):
                print(f"  QC progress: {done:,}/{len(probes):,} ({time.time() - t0:.1f}s)", flush=True)
    finally:
        for fh in tar_handles.values():
            fh.close()

    quarantine_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    quarantine_ids: list[str] = []
    for (split, vid), results in sorted(per_video.items()):
        cers = [r["cer"] for r in results]
        avg_cer = sum(cers) / len(cers)
        min_cer = min(cers)
        avg_pred_lat = sum(r["pred_latin"] for r in results) / len(results)
        avg_ref_lat = sum(r["ref_latin"] for r in results) / len(results)

        reason = None
        if avg_pred_lat >= 0.45 and avg_ref_lat <= 0.15:
            reason = "non_korean_speech_translated_caption"
        elif avg_cer >= args.cer_threshold and min_cer >= args.min_cer_floor:
            reason = "high_audio_text_mismatch"

        if reason:
            quarantine_ids.append(vid)
            quarantine_by_split[split].append({
                "video_id": vid,
                "reason": reason,
                "avg_cer": round(avg_cer, 3),
                "min_cer": round(min_cer, 3),
                "avg_pred_latin": round(avg_pred_lat, 3),
                "samples": results,
            })

    summary = {
        "total_videos_probed": len(per_video),
        "total_quarantined": len(quarantine_ids),
        "quarantined_by_split": {k: len(v) for k, v in quarantine_by_split.items()},
        "quarantine_video_ids": sorted(quarantine_ids),
        "details_by_split": dict(quarantine_by_split),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    args.output_qc.parent.mkdir(parents=True, exist_ok=True)
    args.output_qc.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nQC complete in {summary['elapsed_sec']}s -> quarantined {len(quarantine_ids)}/{len(per_video)} videos:")
    print(f"  Breakdown: {summary['quarantined_by_split']}")
    print(f"  Saved -> {args.output_qc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
