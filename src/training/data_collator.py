"""Data collator for Whisper speech-to-text fine-tuning (short-form & long-form context)."""

from dataclasses import dataclass
from typing import Any, Dict, List
import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        batch["input_features"] = batch["input_features"].to(torch.float16)

        pad_id = self.processor.tokenizer.pad_token_id
        # Whisper decoder_start_token_id is <|startoftranscript|> (50258), NOT eos/bos (50257).
        decoder_start_id = 50258

        dec_seqs: List[List[int]] = []
        lbl_seqs: List[List[int]] = []

        for f in features:
            if "decoder_input_ids" in f:
                dec = list(f["decoder_input_ids"])
                lbl = list(f["labels"])
            else:
                full = list(f["labels"])
                if full and full[0] == decoder_start_id:
                    # Standard tokenizer output already begins with <|startoftranscript|> (50258);
                    # shift right once instead of prepending a duplicate 50258 token.
                    dec = full[:-1]
                    lbl = full[1:]
                else:
                    dec = [decoder_start_id] + full[:-1]
                    lbl = full
            dec_seqs.append(dec)
            lbl_seqs.append(lbl)

        max_len = max(len(x) for x in lbl_seqs)
        padded_dec = [x + [pad_id] * (max_len - len(x)) for x in dec_seqs]
        padded_lbl = [x + [-100] * (max_len - len(x)) for x in lbl_seqs]

        batch["decoder_input_ids"] = torch.tensor(padded_dec, dtype=torch.long)
        batch["labels"] = torch.tensor(padded_lbl, dtype=torch.long)
        return batch
