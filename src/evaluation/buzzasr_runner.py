"""Runner and Benchmark for BuzzASR/korean (and other Whisper-based candidates).

Designed for Korean and Korean-English Code-Switching ASR evaluation.
Measures: WER, CER, Peak VRAM, Processing Latency & Real-Time Factor (RTF).
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Keep all Hugging Face cache strictly inside this project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "hub")

import torch


# English contractions, abbreviations, and compound normalizations
EN_NORM_MAP = {
    "mr": "mister",
    "mrs": "missus",
    "ms": "miss",
    "dr": "doctor",
    "st": "saint",
    "youre": "you are",
    "you're": "you are",
    "im": "i am",
    "i'm": "i am",
    "ive": "i have",
    "i've": "i have",
    "id": "i would",
    "i'd": "i would",
    "itd": "it would",
    "it'd": "it would",
    "its": "it is",
    "it's": "it is",
    "were": "we are",
    "we're": "we are",
    "theyre": "they are",
    "they're": "they are",
    "didnt": "did not",
    "didn't": "did not",
    "dont": "do not",
    "don't": "do not",
    "doesnt": "does not",
    "doesn't": "does not",
    "cant": "cannot",
    "can't": "cannot",
    "wont": "will not",
    "won't": "will not",
    "isnt": "is not",
    "isn't": "is not",
    "arent": "are not",
    "aren't": "are not",
    "wasnt": "was not",
    "wasn't": "was not",
    "werent": "were not",
    "weren't": "were not",
    "hasnt": "has not",
    "hasn't": "has not",
    "havent": "have not",
    "haven't": "have not",
    "hadnt": "had not",
    "hadn't": "had not",
    "wouldnt": "would not",
    "wouldn't": "would not",
    "shouldnt": "should not",
    "shouldn't": "should not",
    "couldnt": "could not",
    "couldn't": "could not",
    "thats": "that is",
    "that's": "that is",
    "theres": "there is",
    "there's": "there is",
    "whats": "what is",
    "what's": "what is",
    "whos": "who is",
    "who's": "who is",
    "lets": "let us",
    "let's": "let us",
}

# Compound English & Korean spacing patterns normalized to canonical form
COMPOUND_NORM_PATTERNS = [
    (re.compile(r"\bthird\s+party\b"), "thirdparty"),
    (re.compile(r"\blarge\s+scale\b"), "largescale"),
    (re.compile(r"\bmicro\s+services\b"), "microservices"),
    (re.compile(r"\breal\s+time\b"), "realtime"),
    (re.compile(r"\bopen\s+source\b"), "opensource"),
    (re.compile(r"\bdeep\s+learning\b"), "deeplearning"),
    (re.compile(r"\bmachine\s+learning\b"), "machinelearning"),
    (re.compile(r"\bdata\s+base\b"), "database"),
    (re.compile(r"\bdata\s+set\b"), "dataset"),
    (re.compile(r"\bweb\s+toon\b"), "webtoon"),
    (re.compile(r"\bpower\s+point\b"), "powerpoint"),
    (re.compile(r"\bteam\s+work\b"), "teamwork"),
    (re.compile(r"\bback\s+end\b"), "backend"),
    (re.compile(r"\bfront\s+end\b"), "frontend"),
    (re.compile(r"\bcode\s+base\b"), "codebase"),
    (re.compile(r"\bwork\s+shop\b"), "workshop"),
    (re.compile(r"\bguide\s+line\b"), "guideline"),
    (re.compile(r"\btime\s+line\b"), "timeline"),
    (re.compile(r"\bmile\s+stone\b"), "milestone"),
    (re.compile(r"\bbase\s+line\b"), "baseline"),
    (re.compile(r"\bpipe\s+line\b"), "pipeline"),
    (re.compile(r"\bdead\s+line\b"), "deadline"),
    (re.compile(r"\bon\s+line\b"), "online"),
    (re.compile(r"\boff\s+line\b"), "offline"),
    # Korean colloquial & auxiliary verb spacing normalization
    (re.compile(r"([가-힣a-z0-9]+)한거\b"), r"\1한 것"),
    (re.compile(r"([가-힣a-z0-9]+)할거\b"), r"\1할 것"),
    (re.compile(r"([가-힣a-z0-9]+)인거\b"), r"\1인 것"),
    (re.compile(r"해\s+봐야"), "해봐야"),
    (re.compile(r"해\s+보았"), "해보았"),
    (re.compile(r"해\s+봤"), "해봤"),
    (re.compile(r"해\s+주세"), "해주세"),
    (re.compile(r"해\s+달라"), "해달라"),
]


def normalize_for_evaluation(text: str) -> str:
    """Normalize text for WER/CER evaluation:
    - Unicode NFC
    - Lowercase Latin characters
    - Collapse hyphenated compounds (e.g. third-party -> thirdparty)
    - Expand English contractions & titles (mr -> mister, youre -> you are)
    - Normalize Korean auxiliary verb & colloquial spacing
    - Remove punctuation and symbols
    - Collapse extra whitespace
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", text).lower()
    # Join hyphenated English words (e.g. third-party -> thirdparty, large-scale -> largescale)
    normalized = re.sub(r"([a-z0-9])[-–—]([a-z0-9])", r"\1\2", normalized)
    # Handle apostrophes inside contractions before stripping punctuation
    words_pre = normalized.split()
    expanded_words = []
    for w in words_pre:
        w_clean = w.strip(".,!?;:\"'()[]{}")
        if w_clean in EN_NORM_MAP:
            expanded_words.append(EN_NORM_MAP[w_clean])
        else:
            expanded_words.append(w)
    normalized = " ".join(expanded_words)

    characters = [
        " " if unicodedata.category(c)[0] in {"P", "S"} else c
        for c in normalized
    ]
    cleaned = " ".join("".join(characters).split())

    # Second pass token map after punctuation removal
    tokens = [EN_NORM_MAP.get(tok, tok) for tok in cleaned.split()]
    cleaned = " ".join(tokens)

    # Apply compound & Korean auxiliary spacing rules
    for pattern, repl in COMPOUND_NORM_PATTERNS:
        cleaned = pattern.sub(repl, cleaned)

    return " ".join(cleaned.split())


def compute_edit_distance(ref_tokens: List[str], pred_tokens: List[str]) -> int:
    """Standard Levenshtein distance."""
    d = [[0] * (len(pred_tokens) + 1) for _ in range(len(ref_tokens) + 1)]
    for i in range(len(ref_tokens) + 1):
        d[i][0] = i
    for j in range(len(pred_tokens) + 1):
        d[0][j] = j

    for i in range(1, len(ref_tokens) + 1):
        for j in range(1, len(pred_tokens) + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[len(ref_tokens)][len(pred_tokens)]


def compute_wer(ref: str, pred: str) -> float:
    ref_words = ref.split()
    pred_words = pred.split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    return compute_edit_distance(ref_words, pred_words) / len(ref_words)


def compute_cer(ref: str, pred: str) -> float:
    ref_chars = list(re.sub(r"\s+", "", ref))
    pred_chars = list(re.sub(r"\s+", "", pred))
    if not ref_chars:
        return 0.0 if not pred_chars else 1.0
    return compute_edit_distance(ref_chars, pred_chars) / len(ref_chars)


class WhisperASRRunner:
    def __init__(self, model_id: str = "BuzzASR/korean", adapter_path: Optional[str] = None, device: str = "cuda"):
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        print(f"\n=======================================================")
        print(f"Loading Model: {self.model_id}")
        if self.adapter_path:
            print(f"LoRA Adapter:  {self.adapter_path}")
        print(f"Device: {self.device}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            torch.cuda.reset_peak_memory_stats()
            vram_start = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"Initial VRAM Allocated: {vram_start:.2f} MB")
        print(f"=======================================================")

        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        load_t0 = time.time()
        self.processor = WhisperProcessor.from_pretrained(self.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )

        if self.adapter_path:
            from peft import PeftModel
            print(f"Loading LoRA adapter from {self.adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        # Clear forced Korean decoder bias so the model can freely emit Latin & Hangul
        # and avoid translating pure English speech into Korean.
        self.model.config.forced_decoder_ids = None
        self.model.generation_config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        self.model.generation_config.suppress_tokens = []
        self.model.generation_config.language = None

        self.model = self.model.to(self.device).eval()
        load_time = time.time() - load_t0

        if self.device == "cuda":
            vram_model = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"Model loaded in {load_time:.2f}s. VRAM allocated: {vram_model:.2f} MB")
        else:
            print(f"Model loaded in {load_time:.2f}s.")

    def warmup(self):
        """Warm up CUDA kernels with a short dummy audio."""
        import numpy as np
        dummy = np.zeros(16000, dtype=np.float32)
        try:
            self.transcribe_audio_array(dummy)
        except Exception:
            pass

    def transcribe_audio_array(
        self,
        audio_array,
        sampling_rate: int = 16000,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe a 1D numpy array of audio at 16kHz with accelerated inference."""
        t0 = time.time()
        audio_duration = len(audio_array) / sampling_rate if sampling_rate > 0 else 0.0

        inputs = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)
        if self.device == "cuda":
            input_features = input_features.half()

        # Dynamic max_new_tokens based on audio length to prevent runaway generation
        max_tokens = min(196, max(48, int(audio_duration * 18)))

        gen_kwargs: Dict[str, Any] = {
            "num_beams": 1,
            "use_cache": True,
            "max_new_tokens": max_tokens,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.2,
        }
        if language in {"en", "english"}:
            gen_kwargs["language"] = "en"
            gen_kwargs["task"] = "transcribe"
        elif language in {"ko", "korean"}:
            gen_kwargs["language"] = "ko"
            gen_kwargs["task"] = "transcribe"

        with torch.inference_mode():
            gen_tokens = self.model.generate(input_features, **gen_kwargs)

        transcription = self.processor.batch_decode(gen_tokens, skip_special_tokens=True)[0]
        inference_time = time.time() - t0
        rtf = inference_time / audio_duration if audio_duration > 0 else 0.0

        peak_vram_mb = 0.0
        if self.device == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        return {
            "transcription": transcription.strip(),
            "inference_time_sec": inference_time,
            "audio_duration_sec": audio_duration,
            "real_time_factor": rtf,
            "peak_vram_mb": peak_vram_mb
        }

    def transcribe_file(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        import soundfile as sf
        audio_data, sr = sf.read(audio_path)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)  # to mono
        if sr != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            sr = 16000
        return self.transcribe_audio_array(audio_data, sampling_rate=sr, language=language)


def run_benchmark_on_dataset(runner: WhisperASRRunner, dataset_name: str, split: str = "test", limit: int = 10):
    """Benchmark model on a Hugging Face dataset (e.g. thetaone-ai/HiKE or Bingsu/zeroth-korean)."""
    from datasets import load_dataset

    print(f"\n--- Running benchmark on HF dataset: {dataset_name} ({split} split, limit={limit}) ---")
    ds = load_dataset(dataset_name, split=split, streaming=True)

    results = []
    total_audio_sec = 0.0
    total_time_sec = 0.0
    max_peak_vram_mb = 0.0

    count = 0
    for idx, item in enumerate(ds):
        if count >= limit:
            break

        # Extract audio and reference
        audio_info = item.get("audio")
        if not audio_info:
            continue

        audio_array = audio_info["array"]
        sr = audio_info["sampling_rate"]

        # Resample to 16kHz if needed
        if sr != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            sr = 16000

        ref_text = item.get("text", "") or item.get("transcription", "") or item.get("sentence", "")

        out = runner.transcribe_audio_array(audio_array, sampling_rate=sr)
        pred_text = out["transcription"]

        ref_norm = normalize_for_evaluation(ref_text)
        pred_norm = normalize_for_evaluation(pred_text)

        wer_val = compute_wer(ref_norm, pred_norm)
        cer_val = compute_cer(ref_norm, pred_norm)

        total_audio_sec += out["audio_duration_sec"]
        total_time_sec += out["inference_time_sec"]
        if out["peak_vram_mb"] > max_peak_vram_mb:
            max_peak_vram_mb = out["peak_vram_mb"]

        print(f"\n[Sample {count + 1}/{limit}] (Duration: {out['audio_duration_sec']:.2f}s | Latency: {out['inference_time_sec']*1000:.1f}ms | RTF: {out['real_time_factor']:.3f})")
        print(f"  REF:  {ref_text}")
        print(f"  PRED: {pred_text}")
        print(f"  WER: {wer_val * 100:.2f}% | CER: {cer_val * 100:.2f}%")

        results.append({
            "index": count,
            "ref": ref_text,
            "pred": pred_text,
            "wer": wer_val,
            "cer": cer_val,
            "audio_duration_sec": out["audio_duration_sec"],
            "inference_time_sec": out["inference_time_sec"]
        })
        count += 1

    if results:
        avg_wer = sum(r["wer"] for r in results) / len(results) * 100
        avg_cer = sum(r["cer"] for r in results) / len(results) * 100
        overall_rtf = total_time_sec / total_audio_sec if total_audio_sec > 0 else 0.0

        print(f"\n==================== BENCHMARK SUMMARY ====================")
        print(f"Dataset:            {dataset_name}")
        print(f"Samples evaluated:  {len(results)}")
        print(f"Average WER:        {avg_wer:.2f}%")
        print(f"Average CER:        {avg_cer:.2f}%")
        print(f"Overall RTF:        {overall_rtf:.3f} (Audio: {total_audio_sec:.1f}s, Time: {total_time_sec:.1f}s)")
        print(f"Peak VRAM:          {max_peak_vram_mb:.1f} MB ({max_peak_vram_mb / 1024:.2f} GB)")
        print(f"VRAM Target (<10GB): {'PASSED ✅' if max_peak_vram_mb < 10240 else 'FAILED ❌'}")
        print(f"===========================================================\n")


def run_benchmark_on_manifest(
    runner: WhisperASRRunner,
    manifest_path: str,
    limit: int = 200,
    language: Optional[str] = None,
):
    """Benchmark model on a local JSONL manifest file."""
    limit_str = "ALL" if limit <= 0 else str(limit)
    print(f"\n--- Running benchmark on local manifest: {manifest_path} (limit={limit_str}) ---")
    results = []
    total_audio_sec = 0.0
    total_time_sec = 0.0
    max_peak_vram_mb = 0.0

    count = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if limit > 0 and count >= limit:
                break
            item = json.loads(line)
            audio_rel = item.get("audio_path")
            ref_text = item.get("text_normalized") or item.get("text", "")
            if not audio_rel:
                continue

            audio_abs = PROJECT_ROOT / audio_rel
            if not audio_abs.exists():
                print(f"Skipping missing audio: {audio_abs}")
                continue

            item_lang = language or item.get("language")
            # Only force explicit language token for pure monolingual en/ko; keep None for code-switching (ko-en)
            req_lang = item_lang if item_lang in {"en", "ko"} else None

            out = runner.transcribe_file(str(audio_abs), language=req_lang)
            pred_text = out["transcription"]

            ref_norm = normalize_for_evaluation(ref_text)
            pred_norm = normalize_for_evaluation(pred_text)

            wer_val = compute_wer(ref_norm, pred_norm)
            cer_val = compute_cer(ref_norm, pred_norm)

            total_audio_sec += out["audio_duration_sec"]
            total_time_sec += out["inference_time_sec"]
            if out["peak_vram_mb"] > max_peak_vram_mb:
                max_peak_vram_mb = out["peak_vram_mb"]

            print(f"\n[Sample {count + 1}/{limit_str}] (Duration: {out['audio_duration_sec']:.2f}s | Latency: {out['inference_time_sec']*1000:.1f}ms | RTF: {out['real_time_factor']:.3f})")
            print(f"  REF:  {ref_text}")
            print(f"  PRED: {pred_text}")
            print(f"  WER:  {wer_val * 100:.2f}% | CER: {cer_val * 100:.2f}%")

            results.append({
                "index": count,
                "ref": ref_text,
                "pred": pred_text,
                "wer": wer_val,
                "cer": cer_val,
                "audio_duration_sec": out["audio_duration_sec"],
                "inference_time_sec": out["inference_time_sec"]
            })
            count += 1

    if results:
        avg_wer = sum(r["wer"] for r in results) / len(results) * 100
        avg_cer = sum(r["cer"] for r in results) / len(results) * 100
        overall_rtf = total_time_sec / total_audio_sec if total_audio_sec > 0 else 0.0

        print(f"\n==================== BENCHMARK SUMMARY ====================")
        print(f"Manifest:           {manifest_path}")
        print(f"Samples evaluated:  {len(results)}")
        print(f"Average WER:        {avg_wer:.2f}%")
        print(f"Average CER:        {avg_cer:.2f}%")
        print(f"Overall RTF:        {overall_rtf:.3f} (Audio: {total_audio_sec:.1f}s, Time: {total_time_sec:.1f}s)")
        print(f"Peak VRAM:          {max_peak_vram_mb:.1f} MB ({max_peak_vram_mb / 1024:.2f} GB)")
        print(f"===========================================================\n")


def main():
    parser = argparse.ArgumentParser(description="BuzzASR/korean Evaluation & Inference Runner")
    parser.add_argument("--model", type=str, default="BuzzASR/korean", help="Hugging Face model ID")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--audio", type=str, default=None, help="Path to single audio file (wav/flac/mp3)")
    parser.add_argument("--manifest", type=str, default=None, help="Path to local JSONL manifest")
    parser.add_argument("--hf-dataset", type=str, default=None, help="Hugging Face dataset name")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--limit", type=int, default=200, help="Number of samples to evaluate (<=0 for full dataset)")
    parser.add_argument("--language", type=str, default=None, help="Optional language hint: en, ko, or None (auto/bilingual)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    runner = WhisperASRRunner(model_id=args.model, adapter_path=args.adapter, device=args.device)

    if args.audio:
        print(f"\nTranscribing single file: {args.audio}")
        out = runner.transcribe_file(args.audio, language=args.language)
        print(f"\nResult:")
        print(f"  Transcription:      {out['transcription']}")
        print(f"  Audio Duration:     {out['audio_duration_sec']:.2f}s")
        print(f"  Latency:            {out['inference_time_sec']*1000:.1f}ms")
        print(f"  Real-Time Factor:   {out['real_time_factor']:.3f}")
        print(f"  Peak VRAM:          {out['peak_vram_mb']:.1f} MB")
    elif args.manifest:
        run_benchmark_on_manifest(runner, manifest_path=args.manifest, limit=args.limit, language=args.language)
    elif args.hf_dataset:
        run_benchmark_on_dataset(runner, dataset_name=args.hf_dataset, split=args.split, limit=args.limit)
    else:
        print("Please provide --audio <path>, --manifest <path>, or --hf-dataset <name>.")


if __name__ == "__main__":
    main()
