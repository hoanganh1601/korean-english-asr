"""Comprehensive Full Benchmark Runner for BuzzASR + LoRA v4
Evaluates 1,267 test samples across 3 domains:
1. HiKE Code-Switching Eval (321 samples)
2. Zeroth Korean Clean Test (446 samples)
3. LibriSpeech English Clean Test (500 samples)
"""

import json
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.buzzasr_runner import (
    WhisperASRRunner,
    compute_cer,
    compute_wer,
    normalize_for_evaluation,
)

BENCHMARKS = [
    {
        "name": "HiKE Conversational Code-Switching",
        "domain": "Korean-English Code-Switching",
        "manifest": PROJECT_ROOT / "artifacts" / "manifests" / "hike_v1.eval.jsonl",
        "limit": 321,
        "language": None,
    },
    {
        "name": "Zeroth Korean Clean Test",
        "domain": "Pure Korean Monolingual",
        "manifest": PROJECT_ROOT / "artifacts" / "manifests" / "zeroth_v1.clean_test.jsonl",
        "limit": 446,
        "language": "ko",
    },
    {
        "name": "LibriSpeech English Clean Test",
        "domain": "Pure English Monolingual",
        "manifest": PROJECT_ROOT / "artifacts" / "manifests" / "librispeech_v1.clean_test.jsonl",
        "limit": 500,
        "language": "en",
    },
]


def evaluate_manifest(runner: WhisperASRRunner, config: dict) -> dict:
    name = config["name"]
    manifest_path = config["manifest"]
    limit = config["limit"]
    lang = config["language"]

    print(f"\n{'='*75}")
    print(f"  RUNNING BENCHMARK: {name}")
    print(f"  Manifest: {manifest_path.name} | Limit: {limit} | Language Hint: {lang}")
    print(f"{'='*75}\n")

    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit > 0 and idx >= limit:
                break
            if line.strip():
                items.append(json.loads(line))

    results = []
    total_audio_sec = 0.0
    total_infer_sec = 0.0
    max_peak_vram = 0.0

    t_start = time.time()
    for idx, item in enumerate(items, 1):
        rel_path = item["audio_path"]
        audio_abs = PROJECT_ROOT / rel_path
        ref_text = item.get("text", "") or item.get("text_normalized", "")

        out = runner.transcribe_file(str(audio_abs), language=lang)
        pred_text = out["transcription"]

        ref_norm = normalize_for_evaluation(ref_text)
        pred_norm = normalize_for_evaluation(pred_text)

        wer_val = compute_wer(ref_norm, pred_norm)
        cer_val = compute_cer(ref_norm, pred_norm)

        total_audio_sec += out["audio_duration_sec"]
        total_infer_sec += out["inference_time_sec"]
        if out["peak_vram_mb"] > max_peak_vram:
            max_peak_vram = out["peak_vram_mb"]

        results.append({
            "idx": idx,
            "id": item.get("id", f"sample_{idx}"),
            "audio_path": rel_path,
            "ref_raw": ref_text,
            "pred_raw": pred_text,
            "ref_norm": ref_norm,
            "pred_norm": pred_norm,
            "wer": wer_val,
            "cer": cer_val,
            "duration_sec": out["audio_duration_sec"],
            "infer_sec": out["inference_time_sec"],
            "rtf": out["real_time_factor"],
        })

        if idx % 50 == 0 or idx == len(items) or idx <= 5:
            curr_avg_wer = sum(r["wer"] for r in results) / len(results) * 100
            curr_avg_cer = sum(r["cer"] for r in results) / len(results) * 100
            print(f"[{idx:3d}/{len(items)}] Current Avg WER: {curr_avg_wer:5.2f}% | CER: {curr_avg_cer:5.2f}% | RTF: {out['real_time_factor']:.3f} | Audio: {out['audio_duration_sec']:.1f}s")
            if idx <= 3:
                print(f"   REF:  {ref_text[:75]}")
                print(f"   PRED: {pred_text[:75]}")

    elapsed = time.time() - t_start
    avg_wer = sum(r["wer"] for r in results) / len(results) * 100 if results else 0.0
    avg_cer = sum(r["cer"] for r in results) / len(results) * 100 if results else 0.0
    overall_rtf = total_infer_sec / total_audio_sec if total_audio_sec > 0 else 0.0

    print(f"\n--- {name} Results ---")
    print(f"  Samples:          {len(results)}")
    print(f"  Average WER:      {avg_wer:.2f}%")
    print(f"  Average CER:      {avg_cer:.2f}%")
    print(f"  Total Audio:      {total_audio_sec:.1f}s ({total_audio_sec/60:.1f} min)")
    print(f"  Total Latency:    {total_infer_sec:.1f}s ({total_infer_sec/60:.1f} min)")
    print(f"  Overall RTF:      {overall_rtf:.3f}")
    print(f"  Wall-clock Time:  {elapsed:.1f}s")
    print(f"-----------------------------------------\n")

    return {
        "name": name,
        "domain": config["domain"],
        "manifest": str(manifest_path.name),
        "samples_evaluated": len(results),
        "avg_wer_pct": round(avg_wer, 2),
        "avg_cer_pct": round(avg_cer, 2),
        "accuracy_pct": round(max(0.0, 100.0 - avg_wer), 2),
        "total_audio_sec": round(total_audio_sec, 2),
        "total_infer_sec": round(total_infer_sec, 2),
        "overall_rtf": round(overall_rtf, 3),
        "peak_vram_mb": round(max_peak_vram, 1),
        "sample_details": results[:10],  # first 10 for inspectability
    }


def main():
    adapter_path = PROJECT_ROOT / "artifacts" / "checkpoints" / "buzzasr_lora_v4" / "final"
    print(f"\nInitializing BuzzASR + LoRA v4 from {adapter_path}...")
    runner = WhisperASRRunner(
        model_id="BuzzASR/korean",
        adapter_path=str(adapter_path),
        device="cuda"
    )
    runner.warmup()
    print("Warmup complete. Starting full evaluation on 1,267 samples...\n")

    all_summaries = []
    t_global_start = time.time()

    for config in BENCHMARKS:
        summary = evaluate_manifest(runner, config)
        all_summaries.append(summary)

    total_time = time.time() - t_global_start
    total_samples = sum(s["samples_evaluated"] for s in all_summaries)
    total_audio = sum(s["total_audio_sec"] for s in all_summaries)
    total_infer = sum(s["total_infer_sec"] for s in all_summaries)
    weighted_wer = sum(s["avg_wer_pct"] * s["samples_evaluated"] for s in all_summaries) / total_samples
    weighted_cer = sum(s["avg_cer_pct"] * s["samples_evaluated"] for s in all_summaries) / total_samples
    overall_rtf = total_infer / total_audio if total_audio > 0 else 0.0

    final_report = {
        "model": "BuzzASR/korean + LoRA v4 (Synthetic CS + Loanwords 592)",
        "adapter_path": str(adapter_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": total_samples,
        "total_audio_hours": round(total_audio / 3600, 3),
        "total_wall_clock_sec": round(total_time, 2),
        "weighted_avg_wer_pct": round(weighted_wer, 2),
        "weighted_avg_cer_pct": round(weighted_cer, 2),
        "weighted_accuracy_pct": round(100.0 - weighted_wer, 2),
        "overall_rtf": round(overall_rtf, 3),
        "benchmarks": all_summaries,
    }

    out_file = PROJECT_ROOT / "artifacts" / "benchmark_lora_v4_full_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("  🏆 FINAL FULL BENCHMARK RESULTS (BuzzASR + LoRA v4)")
    print("=" * 80)
    print(f"Total Test Samples:      {total_samples:,}")
    print(f"Total Audio Evaluated:   {total_audio:.1f}s ({total_audio/3600:.2f} hours)")
    print(f"Total Evaluation Time:   {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Overall RTF:             {overall_rtf:.3f}")
    print(f"--------------------------------------------------------------------------------")
    print(f"{'Domain / Test Set':<36} | {'Samples':<8} | {'WER (%)':<10} | {'CER (%)':<10} | {'Acc (%)':<8}")
    print(f"{'-'*36} | {'-'*8} | {'-'*10} | {'-'*10} | {'-'*8}")
    for s in all_summaries:
        print(f"{s['name']:<36} | {s['samples_evaluated']:<8} | {s['avg_wer_pct']:>8.2f}% | {s['avg_cer_pct']:>8.2f}% | {s['accuracy_pct']:>6.2f}%")
    print(f"{'-'*36} | {'-'*8} | {'-'*10} | {'-'*10} | {'-'*8}")
    print(f"{'WEIGHTED AVERAGE OVERALL':<36} | {total_samples:<8} | {weighted_wer:>8.2f}% | {weighted_cer:>8.2f}% | {100.0-weighted_wer:>6.2f}%")
    print("=" * 80)
    print(f"Results saved to: {out_file}\n")


if __name__ == "__main__":
    main()

