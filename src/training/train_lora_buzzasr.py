"""Fine-tune BuzzASR/korean with LoRA for Korean-English Code-Switching ASR.

Reads directly from local slim_v1 JSONL manifests in artifacts/manifests/.
Mixes Korean (60%) + English (40%) training data as specified in
configs/training_data_mix.yaml (ko:0.6 / en:0.4, choose_language_then_uniform_utterance).

Objectives
----------
- Korean clean-test WER < 3% after fine-tuning
- Preserve proper Latin transcription for code-switched English terms
- Peak VRAM <= 10 GB on RTX A4000 (16 GB total)

Usage
-----
    python -m src.training.train_lora_buzzasr

Optional flags
--------------
    --pilot          Run a short 500-step sanity check (~30 min)
    --output-dir     Override checkpoint output directory
    --max-steps      Override max training steps
    --seed           Random seed (default: 42)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Project root and cache setup (must happen before any HF import)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR / "hub")

import torch
import numpy as np
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from src.evaluation.buzzasr_runner import compute_cer, compute_wer, normalize_for_evaluation
from src.training.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from src.training.manifest_dataset import (
    STARTOFPREV_TOKEN,
    ManifestDataset,
    YodasLongWindowDataset,
)


# ---------------------------------------------------------------------------
# Manifest paths
# ---------------------------------------------------------------------------
MANIFESTS = PROJECT_ROOT / "artifacts" / "manifests"
KO_TRAIN  = MANIFESTS / "zeroth_v1.train.jsonl"
EN_TRAIN  = MANIFESTS / "librispeech_v1.train.jsonl"
CS_TRAIN  = MANIFESTS / "hike_v1.train.jsonl"
SYN_TRAIN   = MANIFESTS / "koen_synthetic_v1.train.jsonl"
YODAS_TRAIN = MANIFESTS / "yodas_long_windows_v1.train.jsonl"
KO_DEV      = MANIFESTS / "zeroth_v1.dev.jsonl"
EN_DEV    = MANIFESTS / "librispeech_v1.dev.jsonl"
CS_DEV    = MANIFESTS / "hike_v1.eval.jsonl"
LOANWORDS_PATH = PROJECT_ROOT / "artifacts" / "loanwords_dict.json"

# Checkpoint output
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints" / "buzzasr_lora_v4"

# ---------------------------------------------------------------------------
# Mixed dataset (40% KO / 25% EN / 15% CS Real / 20% CS Synthetic)
# ---------------------------------------------------------------------------

class MixedDataset(torch.utils.data.Dataset):
    """Interleaves Korean, English, Code-Switching, and optional YODAS Long-Form window streams."""

    def __init__(
        self,
        ko_dataset: ManifestDataset,
        en_dataset: ManifestDataset,
        cs_dataset: ManifestDataset,
        syn_dataset: ManifestDataset | None = None,
        yodas_dataset: torch.utils.data.Dataset | None = None,
        *,
        ko_prob: float = 0.40,
        en_prob: float = 0.25,
        cs_prob: float = 0.15,
        syn_prob: float = 0.20,
        yodas_prob: float = 0.0,
        total_virtual_samples: int,
        seed: int = 42,
        full_dataset: bool = False,
    ) -> None:
        self.full_dataset = full_dataset
        candidates: list[tuple[str, torch.utils.data.Dataset | None, float]] = [
            ("ko", ko_dataset, ko_prob),
            ("en", en_dataset, en_prob),
            ("cs", cs_dataset, cs_prob),
            ("syn", syn_dataset, syn_prob),
            ("yodas", yodas_dataset, yodas_prob),
        ]
        active = [(name, ds, prob) for name, ds, prob in candidates if ds is not None and prob > 0.0]
        if not active:
            raise ValueError("At least one active dataset stream with positive probability is required.")

        total_prob = sum(prob for _, _, prob in active)
        self.datasets: dict[str, torch.utils.data.Dataset] = {name: ds for name, ds, _ in active}
        self.stream_probs: list[tuple[str, float]] = [(name, prob / total_prob) for name, _, prob in active]
        self.total = total_virtual_samples
        self.rng = random.Random(seed)
        self._build_index()

    def _build_index(self) -> None:
        self._index: list[tuple[str, int]] = []
        if self.full_dataset:
            for name, ds in self.datasets.items():
                for i in range(len(ds)):
                    self._index.append((name, i))
            self.rng.shuffle(self._index)
            self.total = len(self._index)
            return

        # Shuffled permutation queues without replacement per stream
        queues: dict[str, list[int]] = {name: [] for name in self.datasets}

        def _next_from_stream(name: str) -> int:
            ds = self.datasets[name]
            if not queues[name]:
                idxs = list(range(len(ds)))
                self.rng.shuffle(idxs)
                queues[name] = idxs
            return queues[name].pop()

        cumulative: list[tuple[float, str]] = []
        acc = 0.0
        for name, prob in self.stream_probs:
            acc += prob
            cumulative.append((acc, name))
        cumulative[-1] = (1.0, cumulative[-1][1])

        for _ in range(self.total):
            r = self.rng.random()
            chosen = cumulative[-1][1]
            for threshold, name in cumulative:
                if r < threshold:
                    chosen = name
                    break
            self._index.append((chosen, _next_from_stream(chosen)))

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, i: int) -> dict[str, Any]:
        stream, idx = self._index[i]
        return self.datasets[stream][idx]


# ---------------------------------------------------------------------------
# Dev dataset: small fixed slice for in-training evaluation
# ---------------------------------------------------------------------------

class SubsetDataset(torch.utils.data.Dataset):
    """A fixed-size subset drawn from a ManifestDataset."""

    def __init__(self, base: ManifestDataset, n: int, seed: int = 42) -> None:
        rng = random.Random(seed)
        indices = rng.sample(range(len(base)), min(n, len(base)))
        self._items = [base[i] for i in indices]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self._items[i]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_SPECIAL_TAG_RE = re.compile(r"<\|[^|]+\|>")


def compute_metrics_fn(pred: Any, processor: WhisperProcessor) -> dict[str, float]:
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # -100 padding → pad_token_id so tokenizer can decode
    label_ids = np.where(
        label_ids == -100,
        processor.tokenizer.pad_token_id,
        label_ids,
    )

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wers, cers = [], []
    for ref, hyp in zip(label_str, pred_str):
        ref_clean = _SPECIAL_TAG_RE.sub("", ref)
        hyp_clean = _SPECIAL_TAG_RE.sub("", hyp)
        ref_n = normalize_for_evaluation(ref_clean)
        hyp_n = normalize_for_evaluation(hyp_clean)
        wers.append(compute_wer(ref_n, hyp_n))
        cers.append(compute_cer(ref_n, hyp_n))

    avg_wer = (sum(wers) / len(wers)) * 100 if wers else 0.0
    avg_cer = (sum(cers) / len(cers)) * 100 if cers else 0.0
    return {"wer": avg_wer, "cer": avg_cer}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune BuzzASR/korean with LoRA.")
    p.add_argument(
        "--pilot",
        action="store_true",
        help="Short 500-step sanity check run (~30 min on RTX A4000).",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--init-adapter", type=str, default=None,
        help="Path to an existing LoRA checkpoint (e.g. artifacts/checkpoints/buzzasr_lora_v3/final) for continual fine-tuning.",
    )
    p.add_argument(
        "--learning-rate", type=float, default=None,
        help="Learning rate (default: 5e-5 for continual fine-tuning, 1e-4 for fresh training).",
    )
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override max training steps. Default: full 1-epoch equivalent.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--eval-samples", type=int, default=600,
        help="Number of dev utterances evaluated in-training (default: 600: 200 KO, 200 EN, 200 CS).",
    )
    p.add_argument(
        "--lora-r", type=int, default=64,
        help="LoRA rank (default: 64 for strong bilingual & code-switching capability).",
    )
    p.add_argument(
        "--loanwords-prob", type=float, default=0.75,
        help="Probability of replacing recognized Korean loanwords with English script in train (default: 0.75).",
    )
    p.add_argument(
        "--ko-prob", type=float, default=0.40,
        help="Probability of drawing a Korean utterance (default: 0.40).",
    )
    p.add_argument(
        "--en-prob", type=float, default=0.25,
        help="Probability of drawing an English utterance (default: 0.25).",
    )
    p.add_argument(
        "--cs-prob", type=float, default=0.15,
        help="Probability of drawing a HiKE real Code-Switching utterance (default: 0.15).",
    )
    p.add_argument(
        "--syn-prob", type=float, default=0.20,
        help="Probability of drawing a synthetic Code-Switching utterance (default: 0.20).",
    )
    p.add_argument(
        "--yodas-prob", type=float, default=0.0,
        help="Probability of drawing a YODAS Long-Form timestamped context window (default: 0.0).",
    )
    p.add_argument(
        "--yodas-manifest", type=Path, default=YODAS_TRAIN,
        help="Path to YODAS Long-Form training windows JSONL manifest.",
    )
    p.add_argument(
        "--timestamp-prob", type=float, default=0.0,
        help="Probability of training short-form KO/EN/CS utterances with 0.02s timestamp tokens (default: 0.0).",
    )
    p.add_argument(
        "--prev-drop-prob", type=float, default=0.10,
        help="Probability of dropping <|startofprev|> context during YODAS window training (default: 0.10).",
    )
    p.add_argument(
        "--yodas-max-prev-tokens", type=int, default=180,
        help="Max previous context tokens kept after <|startofprev|> (default: 180).",
    )
    p.add_argument(
        "--yodas-loanwords-prob", type=float, default=0.0,
        help="Probability of applying Hangul->English loanword augmentation inside YODAS windows (default: 0.0).",
    )
    p.add_argument(
        "--yodas-min-window-sec", type=float, default=0.0,
        help="Minimum window duration in seconds for YodasLongWindowDataset (default: 0.0).",
    )
    p.add_argument(
        "--yodas-min-segments", type=int, default=1,
        help="Minimum utterance count per window in YodasLongWindowDataset (default: 1).",
    )
    p.add_argument(
        "--num-workers", type=int, default=4,
        help="DataLoader worker processes (default: 4).",
    )
    p.add_argument(
        "--eval-steps", type=int, default=None,
        help="Override evaluation interval in steps.",
    )
    p.add_argument(
        "--save-steps", type=int, default=None,
        help="Override checkpoint save interval in steps.",
    )
    p.add_argument(
        "--batch-size", type=int, default=4,
        help="Per-device train batch size (default: 4). Reduce if OOM.",
    )
    p.add_argument(
        "--grad-accum", type=int, default=4,
        help="Gradient accumulation steps (default: 4). Effective batch = batch_size * grad_accum.",
    )
    p.add_argument(
        "--full-dataset", action="store_true",
        help="Train exhaustively over 100% of all 54,595 unique training utterances across all 4 manifests.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_id   = "BuzzASR/korean"
    output_dir = str(args.output_dir)

    effective_batch = args.batch_size * args.grad_accum

    # Determine max_steps
    # Default: 1 full epoch over the virtual mixed training set or exact 54,595 full dataset
    ko_count = sum(1 for _ in open(KO_TRAIN, encoding="utf-8") if _.strip())
    if args.full_dataset:
        en_count = sum(1 for _ in open(EN_TRAIN, encoding="utf-8") if _.strip())
        cs_count = sum(1 for _ in open(CS_TRAIN, encoding="utf-8") if _.strip())
        syn_count = sum(1 for _ in open(SYN_TRAIN, encoding="utf-8") if _.strip()) if SYN_TRAIN.exists() else 0
        yodas_count = (
            sum(1 for _ in open(args.yodas_manifest, encoding="utf-8") if _.strip())
            if (args.yodas_prob > 0 and args.yodas_manifest.exists())
            else 0
        )
        virtual_epoch = ko_count + en_count + cs_count + syn_count + yodas_count
    else:
        virtual_epoch = math.ceil(ko_count / 0.50)  # implied total at 50% KO ratio
    default_steps = math.ceil(virtual_epoch / effective_batch)

    if args.pilot:
        max_steps = 500
        eval_steps = 100
        save_steps = 500
        warmup_steps = 50
        print("\n[PILOT MODE] Running 500-step sanity check.")
    else:
        max_steps = args.max_steps if args.max_steps else default_steps
        eval_steps = args.eval_steps or (400 if max_steps > 1000 else 100)
        save_steps = args.save_steps or (400 if max_steps > 1000 else 100)
        warmup_steps = max(50, max_steps // 20)

    virtual_total = max_steps * effective_batch

    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(f"  Fine-Tuning: {model_id}")
    print(f"  Output Dir:  {output_dir}")
    print(f"  Device:      {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:         {props.name}  ({props.total_memory / 1024**3:.1f} GB VRAM)")
    print(f"  Effective Batch:  {effective_batch}  ({args.batch_size} × {args.grad_accum} accum)")
    print(f"  Max Steps:        {max_steps:,}  (~{max_steps*effective_batch/ko_count:.1f} epoch(s))")
    print(f"  Eval every:       {eval_steps} steps")
    print(f"  Save every:       {save_steps} steps")
    print(f"  Warmup steps:     {warmup_steps}")
    print(f"  LoRA rank (r):    {args.lora_r}")
    print(f"  Loanwords prob:   {args.loanwords_prob}")
    print(f"  Seed:             {args.seed}")
    print(f"{'='*72}\n")

    # ------------------------------------------------------------------
    # 1. Processor
    # ------------------------------------------------------------------
    print("Loading processor...")
    processor = WhisperProcessor.from_pretrained(model_id, cache_dir=str(CACHE_DIR))

    # ------------------------------------------------------------------
    # 2. Base model in FP16
    # ------------------------------------------------------------------
    print("Loading base model (FP16)...")
    t0 = time.time()
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        cache_dir=str(CACHE_DIR),
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Disable forced language tokens — the model must learn to emit both
    # Korean Hangul and Latin script freely based on the audio.
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.suppress_tokens = []
    model.generation_config.prev_sot_token_id = STARTOFPREV_TOKEN
    # Prevent unconstrained greedy loops on short-form validation sentences during trainer.evaluate()
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.repetition_penalty = 1.2

    # Gradient checkpointing saves ~30% VRAM at the cost of ~15% speed
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # ------------------------------------------------------------------
    # 3. LoRA adapter (Full linear target coverage or Warm-start fine-tuning)
    # ------------------------------------------------------------------
    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    if args.init_adapter and Path(args.init_adapter).exists():
        from peft import PeftModel
        print(f"\n[Continual Fine-Tuning] Loading existing LoRA adapter from {args.init_adapter}...")
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    print("\n--- Trainable Parameters ---")
    model.print_trainable_parameters()
    print("----------------------------\n")

    # ------------------------------------------------------------------
    # 4. Datasets
    # ------------------------------------------------------------------
    print("Loading datasets from local manifests...")

    loanwords_dict = None
    if LOANWORDS_PATH.exists() and args.loanwords_prob > 0:
        with open(LOANWORDS_PATH, "r", encoding="utf-8") as f:
            raw_loanwords = json.load(f)
        # Exclude hyper-localized everyday Korean words from forced Latin replacement
        # so the model preserves natural Hangul in daily contexts (e.g. '카페에서', '커피를')
        localized_everyday_ko = {
            "카페", "커피", "버스", "택시", "호텔", "아파트",
            "드라마", "치킨", "샐러드", "피자", "캠핑",
        }
        loanwords_dict = {
            k: v for k, v in raw_loanwords.items() if k not in localized_everyday_ko
        }
        print(f"  Loaded {len(loanwords_dict)} technical/business loanword pairs for synthetic CS augmentation.")

    ko_train_ds = ManifestDataset(
        KO_TRAIN, processor, shuffle=True, seed=args.seed,
        loanwords_dict=loanwords_dict, loanwords_prob=args.loanwords_prob,
        timestamp_prob=args.timestamp_prob,
    )
    en_train_ds = ManifestDataset(
        EN_TRAIN, processor, language="en", shuffle=True, seed=args.seed,
        timestamp_prob=args.timestamp_prob,
    )
    cs_train_ds = ManifestDataset(
        CS_TRAIN, processor, shuffle=True, seed=args.seed,
        timestamp_prob=args.timestamp_prob,
    )

    syn_train_ds = None
    if SYN_TRAIN.exists() and args.syn_prob > 0:
        syn_train_ds = ManifestDataset(
            SYN_TRAIN, processor, shuffle=True, seed=args.seed,
            timestamp_prob=args.timestamp_prob,
        )
        print(f"  Loaded {len(syn_train_ds)} synthetic CS utterances from {SYN_TRAIN.name}.")

    yodas_train_ds = None
    if args.yodas_prob > 0 and args.yodas_manifest.exists():
        yodas_train_ds = YodasLongWindowDataset(
            args.yodas_manifest,
            processor,
            shuffle=True,
            seed=args.seed,
            max_prev_tokens=args.yodas_max_prev_tokens,
            prev_drop_prob=args.prev_drop_prob,
            loanwords_dict=loanwords_dict,
            loanwords_prob=args.yodas_loanwords_prob,
            min_window_sec=args.yodas_min_window_sec,
            min_segments=args.yodas_min_segments,
        )

    train_ds = MixedDataset(
        ko_train_ds,
        en_train_ds,
        cs_train_ds,
        syn_dataset=syn_train_ds,
        yodas_dataset=yodas_train_ds,
        ko_prob=args.ko_prob,
        en_prob=args.en_prob,
        cs_prob=args.cs_prob,
        syn_prob=args.syn_prob,
        yodas_prob=args.yodas_prob,
        total_virtual_samples=virtual_total,
        seed=args.seed,
        full_dataset=args.full_dataset,
    )

    # Dev: 100 KO + 100 EN + 100 CS fixed sample for in-training eval
    n_each = max(1, args.eval_samples // 3)
    ko_dev_base = ManifestDataset(KO_DEV, processor, seed=args.seed)
    en_dev_base = ManifestDataset(EN_DEV, processor, seed=args.seed)
    cs_dev_base = ManifestDataset(CS_DEV, processor, seed=args.seed)
    ko_dev = SubsetDataset(ko_dev_base, n_each, seed=args.seed)
    en_dev = SubsetDataset(en_dev_base, n_each, seed=args.seed)
    cs_dev = SubsetDataset(cs_dev_base, n_each, seed=args.seed)
    eval_ds = torch.utils.data.ConcatDataset([ko_dev, en_dev, cs_dev])

    print(f"\n  Train virtual size : {len(train_ds):,} samples")
    print(f"  Eval size          : {len(eval_ds):,} samples (KO={len(ko_dev)}, EN={len(en_dev)}, CS={len(cs_dev)})\n")

    # ------------------------------------------------------------------
    # 5. Data Collator
    # ------------------------------------------------------------------
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # ------------------------------------------------------------------
    # 6. Training arguments
    # ------------------------------------------------------------------
    lr = args.learning_rate
    if lr is None:
        lr = 5e-5 if args.init_adapter else 1e-4
    print(f"  Configured Learning Rate: {lr}")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        # Batch
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        # Schedule
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        # Mixed precision
        fp16=True,
        gradient_checkpointing=True,
        # Data loading — overlap disk/TAR I/O with GPU compute
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        # Evaluation
        eval_strategy="steps",
        eval_steps=eval_steps,
        predict_with_generate=True,
        generation_max_length=320,
        # Saving
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        # Logging
        logging_steps=20,
        logging_first_step=True,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # 7. Trainer
    # ------------------------------------------------------------------
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics_fn(pred, processor),
        processing_class=processor.feature_extractor,
    )

    # ------------------------------------------------------------------
    # 8. Train (Auto-resume if checkpoint exists)
    # ------------------------------------------------------------------
    resume_checkpoint = None
    if Path(args.output_dir).exists():
        checkpoints = [
            p for p in Path(args.output_dir).glob("checkpoint-*")
            if p.is_dir() and p.name.split("-")[-1].isdigit()
        ]
        if checkpoints:
            checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]))
            resume_checkpoint = str(checkpoints[-1])
            print(f"Resuming training from checkpoint: {resume_checkpoint}\n")

    print("Starting training...\n")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # ------------------------------------------------------------------
    # 9. Save final adapter
    # ------------------------------------------------------------------
    final_dir = args.output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving final LoRA adapter → {final_dir}")
    model.save_pretrained(str(final_dir))
    processor.save_pretrained(str(final_dir))

    # Write a small run summary
    summary = {
        "model_id": model_id,
        "init_adapter": str(args.init_adapter) if args.init_adapter else None,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_r * 2,
        "learning_rate": lr,
        "target_modules": target_modules,
        "max_steps": max_steps,
        "effective_batch": effective_batch,
        "ko_train_rows": len(ko_train_ds),
        "en_train_rows": len(en_train_ds),
        "cs_train_rows": len(cs_train_ds),
        "syn_train_rows": len(syn_train_ds) if syn_train_ds else 0,
        "yodas_train_rows": len(yodas_train_ds) if yodas_train_ds else 0,
        "mix_ratios": {
            "ko": args.ko_prob,
            "en": args.en_prob,
            "cs": args.cs_prob,
            "syn": args.syn_prob,
            "yodas": args.yodas_prob,
        },
        "timestamp_prob": args.timestamp_prob,
        "prev_drop_prob": args.prev_drop_prob,
        "loanwords_prob": args.loanwords_prob,
        "seed": args.seed,
        "pilot": args.pilot,
    }
    summary_path = args.output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Run summary saved → {summary_path}")
    print("\nTraining complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
