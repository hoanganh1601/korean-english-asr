"""PyTorch Dataset backed by local slim_v1 JSONL manifests.

Reads audio from disk using soundfile.  Each row must have:
    audio_path      : str  – relative to PROJECT_ROOT
    text_normalized : str  – normalised transcript used for training
    duration_sec    : float
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import WhisperProcessor

from src.data.prepare_yodas_long_audio import (
    FADE_SAMPLES,
    SAMPLE_RATE,
    SHARDS_DIR,
    apply_edge_fade,
    read_flac_from_tar,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EOS_TOKEN = 50257
SOT_TOKEN = 50258
EN_TOKEN = 50259
KO_TOKEN = 50264
TRANSCRIBE_TOKEN = 50360
STARTOFPREV_TOKEN = 50362
NOTIMESTAMPS_TOKEN = 50364
TIMESTAMP_BASE = 50365  # <|0.00|> .. 51865 (<|30.00|>), step 0.02s
MAX_TARGET_POSITIONS = 448


def sec_to_timestamp_token(sec: float) -> int:
    """Quantize a second offset into Whisper's 0.02s timestamp token ID range."""
    step = max(0, min(1500, int(round(sec / 0.02))))
    return TIMESTAMP_BASE + step


def build_decoder_and_labels_with_prev(
    target_seq: list[int],
    prev_tokens: list[int],
    *,
    max_target_positions: int = MAX_TARGET_POSITIONS,
    max_prev_tokens: int = 180,
) -> tuple[list[int], list[int]]:
    """Build aligned `(decoder_input_ids, labels)` masking any `<|startofprev|>` prefix."""
    if len(target_seq) > max_target_positions:
        # Preserve EOS at tail while keeping total target sequence inside decoder limit
        target_seq = target_seq[: max_target_positions - 1] + [target_seq[-1]]

    avail_for_prev = max_target_positions - len(target_seq) - 1
    if prev_tokens and avail_for_prev > 0 and max_prev_tokens > 0:
        kept_prev = prev_tokens[-min(max_prev_tokens, avail_for_prev):]
        prompt_ids = [STARTOFPREV_TOKEN] + kept_prev
    else:
        prompt_ids = []

    full_seq = prompt_ids + target_seq
    decoder_input_ids = full_seq[:-1]
    labels = ([-100] * len(prompt_ids)) + target_seq[1:]
    return decoder_input_ids, labels


class ManifestDataset(Dataset):
    """Dataset that reads from a slim_v1 JSONL manifest file."""

    def __init__(
        self,
        manifest_path: str | Path,
        processor: WhisperProcessor,
        *,
        max_duration_sec: float = 30.0,
        min_duration_sec: float = 1.0,
        language: str | None = None,
        shuffle: bool = False,
        seed: int = 42,
        loanwords_dict: dict[str, str] | None = None,
        loanwords_prob: float = 0.0,
        timestamp_prob: float = 0.0,
        default_ts_lang_token: int = KO_TOKEN,
    ) -> None:
        self.processor = processor
        self.project_root = PROJECT_ROOT
        self.language = language
        self.loanwords_dict = loanwords_dict
        self.loanwords_prob = loanwords_prob
        self.timestamp_prob = timestamp_prob
        self.default_ts_lang_token = EN_TOKEN if language == "en" else default_ts_lang_token
        self.rng = random.Random(seed)
        if self.loanwords_dict:
            self.pattern = re.compile(
                "|".join(re.escape(k) for k in sorted(self.loanwords_dict.keys(), key=len, reverse=True))
            )
        else:
            self.pattern = None

        rows: list[dict[str, Any]] = []
        skipped = 0
        with open(manifest_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                dur = float(row["duration_sec"])
                if not (min_duration_sec <= dur <= max_duration_sec):
                    skipped += 1
                    continue
                if language is not None and row.get("language") != language:
                    skipped += 1
                    continue
                rows.append(row)

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(rows)

        self.rows = rows
        print(
            f"  Loaded {len(rows):,} rows from {Path(manifest_path).name}"
            + (f" (skipped {skipped})" if skipped else "")
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        audio_path = self.project_root / row["audio_path"]

        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed (should always be 16 kHz per manifest contract)
        if sr != SAMPLE_RATE:
            import librosa  # lazy import
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        input_features = self.processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE
        ).input_features[0]

        text = row["text_normalized"]
        if self.pattern and self.loanwords_prob > 0 and self.rng.random() < self.loanwords_prob:
            text = self.pattern.sub(lambda m: self.loanwords_dict[m.group(0)].lower(), text)

        dur_sec = len(audio) / float(SAMPLE_RATE)
        if self.timestamp_prob > 0 and self.rng.random() < self.timestamp_prob:
            text_toks = self.processor.tokenizer(text, add_special_tokens=False).input_ids
            # Cap text tokens so prefix + <|0.00|> + text + <|dur|> + <|eos|> <= 448
            max_txt = MAX_TARGET_POSITIONS - 6
            text_toks = text_toks[:max_txt]
            labels = [
                SOT_TOKEN,
                self.default_ts_lang_token,
                TRANSCRIBE_TOKEN,
                TIMESTAMP_BASE,
                *text_toks,
                sec_to_timestamp_token(dur_sec),
                EOS_TOKEN,
            ]
        else:
            labels = self.processor.tokenizer(text).input_ids
            # Optional explicit language conditioning for pure monolingual utterances (e.g. 'en' -> <|en|>=50259)
            if self.language == "en" and len(labels) >= 2 and labels[0] == SOT_TOKEN:
                # Prepend <|en|> (50259) and <|transcribe|> (50360) right after <|startoftranscript|> (50258)
                labels = [SOT_TOKEN, EN_TOKEN, TRANSCRIBE_TOKEN] + [
                    t for t in labels[1:] if t not in (EN_TOKEN, KO_TOKEN, TRANSCRIBE_TOKEN)
                ]

        return {
            "input_features": input_features,
            "labels": labels,
            # Expose length for optional group-by-length sorting
            "length": len(audio),
        }


class YodasLongWindowDataset(Dataset):
    """Dataset of <=30s conversation windows packed with 0.02s timestamps and `<|startofprev|>` context."""

    def __init__(
        self,
        manifest_path: str | Path,
        processor: WhisperProcessor,
        *,
        shards_dir: str | Path = SHARDS_DIR,
        shuffle: bool = True,
        seed: int = 42,
        max_prev_tokens: int = 180,
        prev_drop_prob: float = 0.10,
        lang_token: int = KO_TOKEN,
        loanwords_dict: dict[str, str] | None = None,
        loanwords_prob: float = 0.0,
        min_window_sec: float = 0.0,
        min_segments: int = 1,
    ) -> None:
        self.processor = processor
        self.shards_dir = Path(shards_dir)
        self.max_prev_tokens = max_prev_tokens
        self.prev_drop_prob = prev_drop_prob
        self.lang_token = lang_token
        self.loanwords_dict = loanwords_dict
        self.loanwords_prob = loanwords_prob
        self.rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._tar_handles: dict[str, Any] = {}
        if self.loanwords_dict:
            self.pattern = re.compile(
                "|".join(re.escape(k) for k in sorted(self.loanwords_dict.keys(), key=len, reverse=True))
            )
        else:
            self.pattern = None

        rows: list[dict[str, Any]] = []
        skipped_short = 0
        with open(manifest_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if float(row.get("duration_sec", 0.0)) < min_window_sec or len(row.get("segments", [])) < min_segments:
                    skipped_short += 1
                    continue
                rows.append(row)

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(rows)

        self.rows = rows
        print(
            f"  Loaded {len(rows):,} long-form windows from {Path(manifest_path).name}"
            + (f" (filtered {skipped_short} short/single-seg)" if skipped_short else "")
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        archive_name = row["archive"]
        segments = row["segments"]

        pieces: list[np.ndarray] = []
        target_body: list[int] = []
        for seg in segments:
            gap_sec = float(seg.get("gap_before_sec", 0.0))
            if gap_sec > 0.0:
                gap_samples = int(round(gap_sec * SAMPLE_RATE))
                if gap_samples > 0:
                    # Subtle ambient room noise (~ -58 dBFS) instead of dead digital zero
                    # so decoder learns realistic pause robustness for noisy real-world audio.
                    noise_std = self.rng.uniform(0.0004, 0.0018)
                    pieces.append(
                        self._np_rng.normal(0.0, noise_std, size=gap_samples).astype(np.float32)
                    )

            utt_audio = read_flac_from_tar(
                self._tar_handles,
                self.shards_dir,
                archive_name,
                int(seg["offset_data"]),
                int(seg["size"]),
            )
            pieces.append(apply_edge_fade(utt_audio, FADE_SAMPLES))

            seg_text = seg.get("text", "").strip()
            if seg_text:
                if self.pattern and self.loanwords_prob > 0 and self.rng.random() < self.loanwords_prob:
                    seg_text = self.pattern.sub(lambda m: self.loanwords_dict[m.group(0)].lower(), seg_text)
                seg_toks = self.processor.tokenizer(seg_text, add_special_tokens=False).input_ids
                target_body.append(sec_to_timestamp_token(float(seg["start_sec"])))
                target_body.extend(seg_toks)
                target_body.append(sec_to_timestamp_token(float(seg["end_sec"])))

        if not pieces:
            audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        else:
            audio = np.concatenate(pieces)
            max_samples = 30 * SAMPLE_RATE
            if len(audio) > max_samples:
                audio = audio[:max_samples]

        input_features = self.processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE
        ).input_features[0]

        prefix = [SOT_TOKEN, self.lang_token, TRANSCRIBE_TOKEN]
        max_body = MAX_TARGET_POSITIONS - len(prefix) - 1
        if len(target_body) > max_body:
            target_body = target_body[:max_body]
        target_seq = prefix + target_body + [EOS_TOKEN]

        prev_toks: list[int] = []
        prev_text = (row.get("prev_text") or "").strip()
        if prev_text and (self.prev_drop_prob <= 0.0 or self.rng.random() >= self.prev_drop_prob):
            prev_toks = self.processor.tokenizer(prev_text, add_special_tokens=False).input_ids

        decoder_input_ids, labels = build_decoder_and_labels_with_prev(
            target_seq,
            prev_toks,
            max_target_positions=MAX_TARGET_POSITIONS,
            max_prev_tokens=self.max_prev_tokens,
        )

        return {
            "input_features": input_features,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "length": len(audio),
        }
