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
from src.training.manifest_dataset import ManifestDataset


# ---------------------------------------------------------------------------
# Manifest paths
# ---------------------------------------------------------------------------
MANIFESTS = PROJECT_ROOT / "artifacts" / "manifests"
KO_TRAIN  = MANIFESTS / "zeroth_v1.train.jsonl"
EN_TRAIN  = MANIFESTS / "librispeech_v1.train.jsonl"
CS_TRAIN  = MANIFESTS / "hike_v1.train.jsonl"
SYN_TRAIN = MANIFESTS / "koen_synthetic_v1.train.jsonl"
KO_DEV    = MANIFESTS / "zeroth_v1.dev.jsonl"
EN_DEV    = MANIFESTS / "librispeech_v1.dev.jsonl"
CS_DEV    = MANIFESTS / "hike_v1.eval.jsonl"
LOANWORDS_PATH = PROJECT_ROOT / "artifacts" / "loanwords_dict.json"

# Checkpoint output
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "checkpoints" / "buzzasr_lora_v4"

# ---------------------------------------------------------------------------
# Mixed dataset (40% KO / 25% EN / 15% CS Real / 20% CS Synthetic)
# ---------------------------------------------------------------------------

class MixedDataset(torch.utils.data.Dataset):
    """Interleaves Korean, English, Real Code-Switching, and Synthetic Code-Switching utterances.

    Strategy: choose_stream_then_uniform_utterance
      - Draw stream with P(ko)=0.40, P(en)=0.25, P(cs)=0.15, P(syn)=0.20
      - Then draw a uniformly random utterance from that pool
    """

    def __init__(
        self,
        ko_dataset: ManifestDataset,
        en_dataset: ManifestDataset,
        cs_dataset: ManifestDataset,
        syn_dataset: ManifestDataset | None = None,
        *,
        ko_prob: float = 0.40,
        en_prob: float = 0.25,
        cs_prob: float = 0.15,
        syn_prob: float = 0.20,
        total_virtual_samples: int,
        seed: int = 42,
    ) -> None:
        self.ko = ko_dataset
        self.en = en_dataset
        self.cs = cs_dataset
        self.syn = syn_dataset

        if self.syn is None:
            # Rebalance without synthetic
            total_prob = ko_prob + en_prob + cs_prob
            self.ko_prob = ko_prob / total_prob
            self.en_prob = en_prob / total_prob
            self.cs_prob = cs_prob / total_prob
            self.syn_prob = 0.0
        else:
            total_prob = ko_prob + en_prob + cs_prob + syn_prob
            self.ko_prob = ko_prob / total_prob
            self.en_prob = en_prob / total_prob
            self.cs_prob = cs_prob / total_prob
            self.syn_prob = syn_prob / total_prob

        self.total = total_virtual_samples
        self.rng = random.Random(seed)
        self._build_index()

    def _build_index(self) -> None:
        self._index: list[tuple[str, int]] = []
        p_ko = self.ko_prob
        p_en = p_ko + self.en_prob
        p_cs = p_en + self.cs_prob

        for _ in range(self.total):
            r = self.rng.random()
            if r < p_ko:
                idx = self.rng.randrange(len(self.ko))
                self._index.append(("ko", idx))
            elif r < p_en:
                idx = self.rng.randrange(len(self.en))
                self._index.append(("en", idx))
            elif r < p_cs or self.syn is None:
                idx = self.rng.randrange(len(self.cs))
                self._index.append(("cs", idx))
            else:
                idx = self.rng.randrange(len(self.syn))
                self._index.append(("syn", idx))

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, i: int) -> dict[str, Any]:
        stream, idx = self._index[i]
        if stream == "ko":
            return self.ko[idx]
        elif stream == "en":
            return self.en[idx]
        elif stream == "cs":
            return self.cs[idx]
        else:
            return self.syn[idx]


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

def compute_metrics_fn(pred: Any, processor: WhisperProcessor) -> dict[str, float]:
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # -100 padding → pad_token_id so tokenizer can decode
    label_ids = np.where(
        label_ids == -100,
        processor.tokenizer.pad_token_id,
        label_ids,
    )

    pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wers, cers = [], []
    for ref, hyp in zip(label_str, pred_str):
        ref_n = normalize_for_evaluation(ref)
        hyp_n = normalize_for_evaluation(hyp)
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
        "--batch-size", type=int, default=4,
        help="Per-device train batch size (default: 4). Reduce if OOM.",
    )
    p.add_argument(
        "--grad-accum", type=int, default=4,
        help="Gradient accumulation steps (default: 4). Effective batch = batch_size * grad_accum.",
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
    # Default: 1 full epoch over the virtual mixed training set
    ko_count = sum(1 for _ in open(KO_TRAIN, encoding="utf-8") if _.strip())
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
        eval_steps = 100
        save_steps = 100
        warmup_steps = max(100, max_steps // 30)

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
            loanwords_dict = json.load(f)
        print(f"  Loaded {len(loanwords_dict)} loanword pairs for synthetic CS augmentation.")

    ko_train_ds = ManifestDataset(
        KO_TRAIN, processor, shuffle=True, seed=args.seed,
        loanwords_dict=loanwords_dict, loanwords_prob=args.loanwords_prob
    )
    en_train_ds = ManifestDataset(EN_TRAIN, processor, language="en", shuffle=True, seed=args.seed)
    cs_train_ds = ManifestDataset(CS_TRAIN, processor, shuffle=True, seed=args.seed)

    syn_train_ds = None
    if SYN_TRAIN.exists() and args.syn_prob > 0:
        syn_train_ds = ManifestDataset(SYN_TRAIN, processor, shuffle=True, seed=args.seed)
        print(f"  Loaded {len(syn_train_ds)} synthetic CS utterances from {SYN_TRAIN.name}.")

    train_ds = MixedDataset(
        ko_train_ds,
        en_train_ds,
        cs_train_ds,
        syn_dataset=syn_train_ds,
        ko_prob=args.ko_prob,
        en_prob=args.en_prob,
        cs_prob=args.cs_prob,
        syn_prob=args.syn_prob,
        total_virtual_samples=virtual_total,
        seed=args.seed,
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
        # Data loading — 4 workers overlap disk I/O with GPU compute
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        # Evaluation
        eval_strategy="steps",
        eval_steps=eval_steps,
        predict_with_generate=True,
        generation_max_length=225,
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
        "mix_ratios": {"ko": args.ko_prob, "en": args.en_prob, "cs": args.cs_prob, "syn": args.syn_prob},
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
