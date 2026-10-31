"""Test all downloaded audio samples with BuzzASR/korean model."""

import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.evaluation.buzzasr_runner import WhisperASRRunner, normalize_for_evaluation, compute_wer, compute_cer

def main():
    runner = WhisperASRRunner(model_id="BuzzASR/korean", device="cuda")

    manifest_file = Path("data/sample_audios/sample_manifest.json")
    with open(manifest_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print("\n" + "="*80)
    print(f"{'FILE':<20} | {'DUR':<6} | {'LATENCY':<8} | {'WER':<7} | {'CER':<7} | {'SOURCE'}")
    print("="*80)

    for item in samples:
        fpath = item["file"]
        ref = item["reference"]
        source = item["source"]

        out = runner.transcribe_file(fpath)
        pred = out["transcription"]

        ref_n = normalize_for_evaluation(ref)
        pred_n = normalize_for_evaluation(pred)

        wer_val = compute_wer(ref_n, pred_n) * 100
        cer_val = compute_cer(ref_n, pred_n) * 100

        fname = Path(fpath).name
        print(f"{fname:<20} | {out['audio_duration_sec']:<5.1f}s | {out['inference_time_sec']*1000:<6.0f}ms | {wer_val:<6.1f}% | {cer_val:<6.1f}% | {source}")
        print(f"   [REF]  {ref}")
        print(f"   [PRED] {pred}\n")

    print("="*80)
    print("Benchmark complete!")

if __name__ == "__main__":
    main()
