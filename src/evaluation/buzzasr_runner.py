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


from src.evaluation.text_normalizer import (
    COMPOUND_NORM_PATTERNS,
    EN_NORM_MAP,
    UK_TO_US_MAP,
    normalize_for_evaluation,
    normalize_pair_for_evaluation,
)


def segment_audio_by_silence(
    audio_array,
    sampling_rate: int = 16000,
    target_max_sec: float = 24.0,
    hard_max_sec: float = 28.0,
    min_silence_sec: float = 0.35,
    min_speech_sec: float = 0.15,
    max_internal_gap_sec: float = 1.5,
    pad_sec: float = 0.15,
) -> List[tuple]:
    """Split long audio on natural silence/low-energy pauses into clean <=hard_max_sec speech chunks."""
    import numpy as np

    audio = np.asarray(audio_array, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    total_dur = len(audio) / sampling_rate if sampling_rate > 0 else 0.0
    if total_dur <= 0.0:
        return []

    frame_sec = 0.02  # 20ms frames
    frame_len = max(1, int(round(sampling_rate * frame_sec)))
    n_frames = (len(audio) + frame_len - 1) // frame_len
    padded = np.zeros(n_frames * frame_len, dtype=np.float32)
    padded[: len(audio)] = audio
    rms = np.sqrt(np.mean(padded.reshape(n_frames, frame_len) ** 2, axis=1))

    if float(rms.max()) < 1e-4:
        return []

    p10 = float(np.percentile(rms, 10))
    p95 = float(np.percentile(rms, 95))
    thresh = min(p95 * 0.30, max(p10 * 2.0, p95 * 0.07, 1e-4))
    active = rms >= thresh

    # Extract raw active spans and bridge micro-pauses < min_silence_sec
    raw_spans: List[tuple] = []
    in_seg = False
    seg_start_idx = 0
    for idx, flag in enumerate(active):
        if flag and not in_seg:
            in_seg = True
            seg_start_idx = idx
        elif not flag and in_seg:
            in_seg = False
            raw_spans.append((seg_start_idx * frame_sec, min(total_dur, idx * frame_sec)))
    if in_seg:
        raw_spans.append((seg_start_idx * frame_sec, total_dur))

    merged_spans: List[tuple] = []
    for s_sec, e_sec in raw_spans:
        if merged_spans and (s_sec - merged_spans[-1][1]) < min_silence_sec:
            merged_spans[-1] = (merged_spans[-1][0], e_sec)
        else:
            merged_spans.append((s_sec, e_sec))

    speech_spans = [(s, e) for s, e in merged_spans if (e - s) >= min_speech_sec]
    if not speech_spans:
        speech_spans = [(0.0, total_dur)]

    # Smoothed RMS (100ms window) for finding clean micro-pause valleys inside long monologues
    smooth_win = 5
    if len(rms) >= smooth_win:
        kernel = np.ones(smooth_win, dtype=np.float32) / smooth_win
        smooth_rms = np.convolve(rms, kernel, mode="same")
    else:
        smooth_rms = rms

    # Split any continuous speech region exceeding hard_max_sec at lowest-energy valley
    bounded_spans: List[tuple] = []
    for s_sec, e_sec in speech_spans:
        cur_s = s_sec
        while (e_sec - cur_s) > hard_max_sec:
            search_lo = cur_s + max(10.0, target_max_sec * 0.5)
            search_hi = min(e_sec - 1.5, cur_s + target_max_sec)
            if search_hi <= search_lo:
                split_sec = min(e_sec, cur_s + target_max_sec)
            else:
                lo_f = max(0, int(search_lo / frame_sec))
                hi_f = min(len(smooth_rms) - 1, int(search_hi / frame_sec))
                valley_f = lo_f + int(np.argmin(smooth_rms[lo_f : hi_f + 1]))
                split_sec = min(e_sec, max(cur_s + 5.0, valley_f * frame_sec))
            bounded_spans.append((cur_s, split_sec))
            cur_s = split_sec
        if (e_sec - cur_s) >= 0.1:
            bounded_spans.append((cur_s, e_sec))

    # Greedily pack adjacent speech spans into <= target_max_sec chunks with non-overlapping padding
    packed: List[tuple] = []
    for s_sec, e_sec in bounded_spans:
        if (
            packed
            and (e_sec - packed[-1][0]) <= target_max_sec
            and (s_sec - packed[-1][1]) <= max_internal_gap_sec
        ):
            packed[-1] = (packed[-1][0], e_sec)
        else:
            packed.append((s_sec, e_sec))

    chunks: List[tuple] = []
    for i, (s_sec, e_sec) in enumerate(packed):
        prev_end = packed[i - 1][1] if i > 0 else 0.0
        next_start = packed[i + 1][0] if i + 1 < len(packed) else total_dur
        c_start = max(0.0, s_sec - min(pad_sec, max(0.0, (s_sec - prev_end) * 0.5))) if i > 0 else max(0.0, s_sec - pad_sec)
        c_end = min(total_dur, e_sec + min(pad_sec, max(0.0, (next_start - e_sec) * 0.5))) if i + 1 < len(packed) else min(total_dur, e_sec + pad_sec)
        if c_end - c_start >= 0.1:
            chunks.append((round(c_start, 3), round(c_end, 3)))
    return chunks


def _sanitize_chunk_repetition(text: str) -> str:
    """Collapse consecutive loop hallucinations and ensure spacing after sentence punctuation."""
    cleaned = re.sub(r"(.{4,}?)(?:\s*\1){2,}", r"\1", text.strip())
    cleaned = re.sub(r"([.!?])([가-힣])", r"\1 \2", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _is_safe_context_prompt(text: str) -> bool:
    """Reject repetitive decoded text from poisoning subsequent chunks via prompt_ids."""
    words = text.strip().split()
    if len(words) < 6:
        return bool(words)
    return (len(set(words)) / len(words)) >= 0.45


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
    ref_aligned, pred_aligned = normalize_pair_for_evaluation(ref, pred)
    ref_words = ref_aligned.split()
    pred_words = pred_aligned.split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    return compute_edit_distance(ref_words, pred_words) / len(ref_words)


def compute_cer(ref: str, pred: str) -> float:
    ref_aligned, pred_aligned = normalize_pair_for_evaluation(ref, pred)
    ref_chars = list(re.sub(r"\s+", "", ref_aligned))
    pred_chars = list(re.sub(r"\s+", "", pred_aligned))
    if not ref_chars:
        return 0.0 if not pred_chars else 1.0
    return compute_edit_distance(ref_chars, pred_chars) / len(ref_chars)


class WhisperASRRunner:
    def __init__(self, model_id: str = "BuzzASR/korean", adapter_path: Optional[str] = None, device: str = "cuda"):
        self.model_id = model_id
        if adapter_path:
            p = Path(adapter_path)
            if not p.is_absolute():
                p = (PROJECT_ROOT / p).resolve()
            self.adapter_path = str(p) if p.exists() else adapter_path
        else:
            self.adapter_path = None
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
        self.model.generation_config.prev_sot_token_id = 50362  # <|startofprev|>

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

    def _transcribe_single_window(
        self,
        audio_array,
        sampling_rate: int = 16000,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        max_tokens_cap: int = 196,
        no_repeat_ngram_size: int = 3,
        repetition_penalty: float = 1.2,
    ) -> str:
        """Transcribe a single <=30s window cleanly without HF 30s seek."""
        audio_duration = len(audio_array) / sampling_rate if sampling_rate > 0 else 0.0
        inputs = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device)
        if self.device == "cuda":
            input_features = input_features.half()

        max_tokens = min(max_tokens_cap, max(48, int(audio_duration * 18)))
        gen_kwargs: Dict[str, Any] = {
            "num_beams": 1,
            "use_cache": True,
            "max_new_tokens": max_tokens,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "repetition_penalty": repetition_penalty,
        }
        if language in {"en", "english"}:
            gen_kwargs["language"] = "en"
            gen_kwargs["task"] = "transcribe"
        elif language in {"ko", "korean"}:
            gen_kwargs["language"] = "ko"
            gen_kwargs["task"] = "transcribe"

        if prompt and prompt.strip():
            try:
                prompt_ids = self.processor.get_prompt_ids(prompt.strip(), return_tensors="pt")
                gen_kwargs["prompt_ids"] = prompt_ids.to(self.device)
            except Exception:
                pass

        with torch.inference_mode():
            gen_tokens = self.model.generate(input_features, **gen_kwargs)
        return self.processor.batch_decode(gen_tokens, skip_special_tokens=True)[0].strip()

    def _transcribe_vad_split(
        self,
        audio_array,
        sampling_rate: int = 16000,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        condition_on_prev_tokens: bool = False,
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Split long audio by silence/energy pauses, transcribe each clean chunk, and merge."""
        chunks = segment_audio_by_silence(
            audio_array,
            sampling_rate=sampling_rate,
            target_max_sec=18.0,
            hard_max_sec=26.0,
        )
        segments_out: List[Dict[str, Any]] = []
        rolling_tail = (prompt or "").strip()

        for c_start, c_end in chunks:
            s_idx = int(round(c_start * sampling_rate))
            e_idx = min(len(audio_array), int(round(c_end * sampling_rate)))
            if e_idx <= s_idx:
                continue
            chunk_audio = audio_array[s_idx:e_idx]
            chunk_prompt = rolling_tail if (condition_on_prev_tokens and rolling_tail) else (prompt or None)

            raw_text = self._transcribe_single_window(
                chunk_audio,
                sampling_rate=sampling_rate,
                language=language,
                prompt=chunk_prompt,
                max_tokens_cap=280,
                no_repeat_ngram_size=6,
                repetition_penalty=1.05,
            )
            seg_text = _sanitize_chunk_repetition(raw_text)
            if seg_text:
                segments_out.append({
                    "start_sec": round(float(c_start), 2),
                    "end_sec": round(float(c_end), 2),
                    "text": seg_text,
                })
                if condition_on_prev_tokens and _is_safe_context_prompt(seg_text):
                    tail_words = seg_text.split()[-35:]
                    rolling_tail = " ".join(tail_words)

        transcription = " ".join(s["text"] for s in segments_out).strip()
        return transcription, segments_out

    def transcribe_audio_array(
        self,
        audio_array,
        sampling_rate: int = 16000,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        condition_on_prev_tokens: bool = False,
        force_long_form: bool = False,
        longform_mode: str = "vad_split",
    ) -> Dict[str, Any]:
        """Transcribe a 1D numpy array of audio at 16kHz (short-form or long-form)."""
        t0 = time.time()
        audio_duration = len(audio_array) / sampling_rate if sampling_rate > 0 else 0.0
        is_long_form = force_long_form or (audio_duration > 30.0)

        segments_out: List[Dict[str, Any]] = []
        if is_long_form and longform_mode == "vad_split":
            transcription, segments_out = self._transcribe_vad_split(
                audio_array,
                sampling_rate=sampling_rate,
                language=language,
                prompt=prompt,
                condition_on_prev_tokens=condition_on_prev_tokens,
            )
        elif is_long_form:
            inputs = self.processor(
                audio_array,
                sampling_rate=sampling_rate,
                truncation=False,
                padding="longest",
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(self.device)
            if self.device == "cuda":
                input_features = input_features.half()

            attention_mask = getattr(inputs, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            gen_kwargs: Dict[str, Any] = {
                "num_beams": 1,
                "use_cache": True,
                "return_timestamps": True,
                "condition_on_prev_tokens": condition_on_prev_tokens,
                "return_segments": True,
                "max_length": 448,
                "temperature": (0.0, 0.2),
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
            }
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask
            if language in {"en", "english"}:
                gen_kwargs["language"] = "en"
                gen_kwargs["task"] = "transcribe"
            elif language in {"ko", "korean"}:
                gen_kwargs["language"] = "ko"
                gen_kwargs["task"] = "transcribe"
            else:
                gen_kwargs["task"] = "transcribe"

            if prompt and prompt.strip():
                try:
                    prompt_ids = self.processor.get_prompt_ids(prompt.strip(), return_tensors="pt")
                    gen_kwargs["prompt_ids"] = prompt_ids.to(self.device)
                    gen_kwargs["prompt_condition_type"] = "first-segment"
                except Exception:
                    pass

            with torch.inference_mode():
                outputs = self.model.generate(input_features, **gen_kwargs)

            if isinstance(outputs, dict):
                gen_tokens = outputs["sequences"]
                raw_segments = outputs.get("segments", [[]])[0]
                for seg in raw_segments:
                    seg_text = self.processor.decode(seg["tokens"], skip_special_tokens=True).strip()
                    if seg_text:
                        segments_out.append({
                            "start_sec": round(float(seg["start"]), 2),
                            "end_sec": round(float(seg["end"]), 2),
                            "text": seg_text,
                        })
                if segments_out:
                    transcription = " ".join(s["text"] for s in segments_out)
                else:
                    transcription = self.processor.batch_decode(gen_tokens, skip_special_tokens=True)[0]
            else:
                transcription = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            transcription = self._transcribe_single_window(
                audio_array,
                sampling_rate=sampling_rate,
                language=language,
                prompt=prompt,
                max_tokens_cap=196,
            )

        inference_time = time.time() - t0
        rtf = inference_time / audio_duration if audio_duration > 0 else 0.0

        peak_vram_mb = 0.0
        if self.device == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        return {
            "transcription": transcription.strip(),
            "segments": segments_out,
            "is_long_form": is_long_form,
            "longform_mode": longform_mode if is_long_form else "short",
            "inference_time_sec": inference_time,
            "audio_duration_sec": audio_duration,
            "real_time_factor": rtf,
            "peak_vram_mb": peak_vram_mb
        }

    def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        condition_on_prev_tokens: bool = False,
        force_long_form: bool = False,
        longform_mode: str = "vad_split",
    ) -> Dict[str, Any]:
        import soundfile as sf
        audio_data, sr = sf.read(audio_path)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)  # to mono
        if sr != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            sr = 16000
        return self.transcribe_audio_array(
            audio_data,
            sampling_rate=sr,
            language=language,
            prompt=prompt,
            condition_on_prev_tokens=condition_on_prev_tokens,
            force_long_form=force_long_form,
            longform_mode=longform_mode,
        )


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
    parser.add_argument("--prompt", type=str, default=None, help="Optional context / hotwords prompt")
    parser.add_argument("--no-context-carryover", action="store_true", help="Disable conditioning on previous tokens in long-form")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    runner = WhisperASRRunner(model_id=args.model, adapter_path=args.adapter, device=args.device)

    if args.audio:
        print(f"\nTranscribing single file: {args.audio}")
        out = runner.transcribe_file(
            args.audio,
            language=args.language,
            prompt=args.prompt,
            condition_on_prev_tokens=not args.no_context_carryover,
        )
        print(f"\nResult:")
        print(f"  Transcription:      {out['transcription']}")
        print(f"  Audio Duration:     {out['audio_duration_sec']:.2f}s")
        print(f"  Long-Form Mode:     {out.get('is_long_form', False)}")
        print(f"  Latency:            {out['inference_time_sec']*1000:.1f}ms")
        print(f"  Real-Time Factor:   {out['real_time_factor']:.3f}")
        print(f"  Peak VRAM:          {out['peak_vram_mb']:.1f} MB")
        if out.get("segments"):
            print(f"\nTimestamped Segments ({len(out['segments'])}):")
            for seg in out["segments"]:
                print(f"  [{seg['start_sec']:6.2f}s -> {seg['end_sec']:6.2f}s] {seg['text']}")
    elif args.manifest:
        run_benchmark_on_manifest(runner, manifest_path=args.manifest, limit=args.limit, language=args.language)
    elif args.hf_dataset:
        run_benchmark_on_dataset(runner, dataset_name=args.hf_dataset, split=args.split, limit=args.limit)
    else:
        print("Please provide --audio <path>, --manifest <path>, or --hf-dataset <name>.")


if __name__ == "__main__":
    main()
