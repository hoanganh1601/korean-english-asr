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


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    ) -> None:
        self.processor = processor
        self.project_root = PROJECT_ROOT
        self.language = language
        self.loanwords_dict = loanwords_dict
        self.loanwords_prob = loanwords_prob
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
        if sr != 16000:
            import librosa  # lazy import
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        input_features = self.processor.feature_extractor(
            audio, sampling_rate=16000
        ).input_features[0]

        text = row["text_normalized"]
        if self.pattern and self.loanwords_prob > 0 and self.rng.random() < self.loanwords_prob:
            text = self.pattern.sub(lambda m: self.loanwords_dict[m.group(0)].lower(), text)

        labels = self.processor.tokenizer(text).input_ids
        # Optional explicit language conditioning for pure monolingual utterances (e.g. 'en' -> <|en|>=50259)
        if self.language == "en" and len(labels) >= 2 and labels[0] == 50258:
            # Prepend <|en|> (50259) and <|transcribe|> (50360) right after <|startoftranscript|> (50258)
            labels = [50258, 50259, 50360] + [t for t in labels[1:] if t not in (50259, 50264, 50360)]

        return {
            "input_features": input_features,
            "labels": labels,
            # Expose length for optional group-by-length sorting
            "length": len(audio),
        }
