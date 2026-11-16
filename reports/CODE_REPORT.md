# BÁO CÁO MÃ NGUỒN (CODE REPORT) — KOREAN-ENGLISH CODE-SWITCHING & LONG-FORM ASR

**Repository:** `hoanganh1601/korean-english-asr` (Nhánh `code`)  
**Mô hình nền:** `BuzzASR/korean` (Whisper Large 1.55B) + LoRA PEFT (`r=64, lora_alpha=128`)  
**Bản mặc định chính thức:** [`artifacts/checkpoints/buzzasr_lora_v6_1_long/final`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/artifacts/checkpoints/buzzasr_lora_v6_1_long/final) (`443.90 MB`, `adapter_model.safetensors` `440.14 MB` qua Git LFS)  
**Bản dự phòng chuyên chép nguyên văn (Verbatim) & Short-Form:** [`artifacts/checkpoints/buzzasr_lora_v5_full/final`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/artifacts/checkpoints/buzzasr_lora_v5_full/final) (`444.15 MB`)  
**Báo cáo thực nghiệm & số liệu chi tiết:** [`reports/korean_english_asr_report.md`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/reports/korean_english_asr_report.md)

---

## 1. Cấu trúc Thư mục Mã nguồn sau Dọn dẹp

```text
korean-english-asr/
├── app.py                                # Web Studio Server (aiohttp, 567 dòng)
├── run_ui.bat                            # Script khởi chạy Web UI trên Windows
├── .gitignore                            # Bỏ qua 15.5 GB raw/longaudio shards, uploads, cache
├── .gitattributes                        # Cấu hình Git LFS cho *.safetensors
├── configs/                              # Cấu hình môi trường & training
├── data/sample_audios/                   # Mẫu audio demo ngắn cho Web UI
├── reports/
│   ├── korean_english_asr_report.md      # Báo cáo kỹ thuật & nghiệm thu thực nghiệm (Section 1–15)
│   ├── CODE_REPORT.md                    # Báo cáo kiến trúc mã nguồn (tài liệu này)
│   └── data_qc/yodas_long_v1.summary.json
├── scripts/
│   ├── filter_yodas_long_videos.py       # Tier-2 GPU audio-text QC cho YODAS (219 dòng)
│   ├── run_full_benchmark_v4.py          # Benchmark Short-Form 1,267 mẫu HiKE/Zeroth/Libri (269 dòng)
│   ├── run_longform_benchmark.py         # Benchmark Long-Form vad_split / native (231 dòng)
│   └── setup_data_links_windows.py       # Liên kết dữ liệu Windows (48 dòng)
├── src/
│   ├── data/
│   │   ├── prepare_yodas_long_audio.py   # Giải nén TAR YODAS, lọc sạch & đóng gói cửa sổ dài (524 dòng)
│   │   └── ...                           # Các module QC/adapter Zeroth, HiKE, LibriSpeech, audio_qc
│   ├── evaluation/
│   │   ├── buzzasr_runner.py             # WhisperASRRunner + vad_split Silence Pipeline (714 dòng)
│   │   └── text_normalizer.py            # Chuẩn hóa song ngữ Hàn-Anh & loanword alignment (646 dòng)
│   ├── training/
│   │   ├── data_collator.py              # Padding batch & fix double <|startoftranscript|> (47 dòng)
│   │   ├── manifest_dataset.py           # ManifestDataset + YodasLongWindowDataset (326 dòng)
│   │   └── train_lora_buzzasr.py         # Huấn luyện 5-Way Mixture LoRA FP16 (673 dòng)
│   └── ui/
│       ├── index.html                    # Giao diện Web Studio + chọn cặp LoRA v6.1 / v5 Full (353 dòng)
│       └── static/{script.js, style.css} # Timeline segments bấm tua audio & hiển thị WER/CER
└── tests/
    ├── test_text_normalizer.py           # 7 unit tests chuẩn hóa văn bản song ngữ (73 dòng)
    └── test_vad_longform.py              # 4 unit tests tách khoảng lặng VAD & chống lặp (59 dòng)
```

> Đã dọn dẹp toàn bộ thư mục tạm `__pycache__`, `.pytest_cache`, file upload thử nghiệm `data/sample_audios/uploads/`, và các file JSON pilot cũ không nằm trong báo cáo nghiệm thu.

---

## 2. Chi tiết Các Module Cốt lõi & Nâng cấp Kỹ thuật

### 2.1 Suy luận & Tách đoạn theo Khoảng lặng (`vad_split`)
* **File:** [`src/evaluation/buzzasr_runner.py`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/src/evaluation/buzzasr_runner.py) (`714` dòng)
* **Các thành phần chính:**
  * `segment_audio_by_silence(...)`: Tính RMS năng lượng theo khung `20ms` với ngưỡng thích nghi giữa noise floor (`p10`) và speech peak (`p95`), gộp khoảng nghỉ `< 0.35s`, bỏ qua đoạn im lặng kéo dài, tìm đáy năng lượng thấp nhất nếu đoạn nói liền vượt ngưỡng và đóng gói thành các chunk sạch **`target_max_sec=18.0s` (`hard_max_sec=26.0s`)**.
  * `_sanitize_chunk_repetition(...)` & `_is_safe_context_prompt(...)`: Gỡ vòng lặp 3+ lần liên tiếp, chuẩn hóa dấu cách sau `.!?` trước ký tự Hangul, và kiểm tra tỷ lệ từ duy nhất (`>= 0.45`) trước khi đưa context sang chunk kế tiếp.
  * `WhisperASRRunner`:
    * Hỗ trợ linh hoạt `longform_mode="vad_split"` (mặc định cho audio `> 30s`) và `longform_mode="native"` (HF 30s seek).
    * `_transcribe_vad_split(...)` gọi `_transcribe_single_window(...)` với **`max_tokens_cap=280`, `no_repeat_ngram_size=6`, `repetition_penalty=1.05`** (tránh hiện tượng `no_repeat_ngram_size=3, repetition_penalty=1.2` làm biến âm danh từ chuyên ngành lặp lại).
    * Tự động vô hiệu `forced_decoder_ids`, mở rộng `suppress_tokens=[]`, đặt `prev_sot_token_id=50362` (`<|startofprev|>`) để giải mã tự do chữ Hangul + Latin.

### 2.2 Chuẩn bị Dữ liệu Long-Audio & Lọc QC Tier-2
* [`src/data/prepare_yodas_long_audio.py`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/src/data/prepare_yodas_long_audio.py) (`524` dòng): `index_tar_shards`, `read_flac_from_tar`, `normalize_ko_text` (hỗ trợ `clean_captions=True` loại ký tự `♪`/`♬`/`저작권자` và chú thích ngoặc), `build_train_long_windows`, `build_eval_long_videos`, `_load_quarantine_ids`. Đóng gói tập sạch `yodas_long_windows_clean_v2.train.jsonl` (**21,171 cửa sổ**, **154.27 giờ**, **1,310 video >=45s**).
* [`scripts/filter_yodas_long_videos.py`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/scripts/filter_yodas_long_videos.py) (`219` dòng): Batch GPU probe (`batch_size=32`, sort độ dài batch, dynamic `max_new_tokens`), phát hiện và cách ly **994/2,520 video** lỗi audio–text (BGM/slideshow hoặc nói tiếng Anh gắn sub Hàn) vào [`artifacts/manifests/yodas_v2_video_qc_strict.json`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/artifacts/manifests/yodas_v2_video_qc_strict.json).

### 2.3 Pipeline Huấn luyện & Sửa lỗi Collator
* [`src/training/data_collator.py`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/src/training/data_collator.py) (`47` dòng): Lớp `DataCollatorSpeechSeq2SeqWithPadding` chuẩn hóa dịch `decoder_input_ids = full[:-1]` và `labels = full[1:]`, khắc phục lỗi chèn trùng hai lần token `<|startoftranscript|>` (`50258, 50258`) trên các mẫu short-form.
* [`src/training/manifest_dataset.py`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/src/training/manifest_dataset.py) (`326` dòng): `ManifestDataset` (short-form + loanwords augmentation) và `YodasLongWindowDataset` (thay khoảng lặng `np.zeros` bằng nhiễu nền ~`-58 dBFS`, hỗ trợ `--yodas-loanwords-prob`, `--yodas-min-window-sec`, `--yodas-min-segments`, `<|startofprev|>`).
* [`src/training/train_lora_buzzasr.py`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/src/training/train_lora_buzzasr.py) (`673` dòng): `MixedDataset` phối trộn 5 nguồn (`YODAS clean + KO + EN + HiKE CS + Syn CS`), `SubsetDataset`, hỗ trợ warm-start qua `--init-adapter`, `gradient_checkpointing=True`, `fp16=True`.

### 2.4 Web Studio & Benchmark Scripts
* [`app.py`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/app.py) (`567` dòng) & [`src/ui/index.html`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/src/ui/index.html) (`353` dòng):
  * Mặc định dùng **`lora_v6_1`** (`artifacts/checkpoints/buzzasr_lora_v6_1_long/final`), cho phép chuyển đổi nhanh sang **`lora_v5`** (`artifacts/checkpoints/buzzasr_lora_v5_full/final`), tự động chạy `vad_split` với audio `>30s` và trả mảng `segments` bấm tua theo timestamp.
* [`scripts/run_longform_benchmark.py`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/scripts/run_longform_benchmark.py) (`231` dòng) & [`scripts/run_full_benchmark_v4.py`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/scripts/run_full_benchmark_v4.py) (`269` dòng).

---

## 3. Tóm tắt Kết quả Kiểm thử Code & Benchmark Nghiệm thu

* **Unit Tests & Compilation:**
  * [`tests/test_text_normalizer.py`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/tests/test_text_normalizer.py): **7/7 pass** (`All 7 text normalizer tests passed successfully!`).
  * [`tests/test_vad_longform.py`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/tests/test_vad_longform.py): **4/4 pass** (`All 4 VAD Long-Form unit tests passed successfully!`).
  * `py_compile`: Pass trên `app.py`, `src/evaluation/buzzasr_runner.py`, `src/training/*.py`, `src/data/prepare_yodas_long_audio.py`, `scripts/*.py`.
  * Web HTTP Smoke Check: `GET /` HTTP `200` (xác thực có đủ `lora_v6_1` & `lora_v5`), `GET /api/samples` HTTP `200`, và inference smoke test `app.get_asr_runner("lora_v6_1")` trên video 5 phút `2jwkCBzthu8` (`18` segments `vad_split`, `RTF=0.435`).

* **Kết quả Mô hình tốt nhất (`LoRA v6.1` + `vad_split`):**
  * **Long-Form `Gold-16` ([`yodas_long_gold_v1.test.jsonl`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/artifacts/manifests/yodas_long_gold_v1.test.jsonl), 16 video / 79.79 phút):** Corpus WER **14.33%** | Corpus CER **7.32%** *(trên 13/16 video khớp lời thoại chuẩn: **9.43% WER / 3.64% CER**, RTF `0.471`, Peak VRAM `3.58 GB`)*.
  * **Short-Form (`1,267` mẫu):** Weighted WER **3.91%** | CER **1.79%** *(HiKE CS `5.11%`, Zeroth KO `4.20%`, LibriSpeech EN `2.89%`)*.
  * **Adapter đối chứng chuyên nguyên văn (`LoRA v5 Full`):** Short-Form **3.86% WER / 1.63% CER**; tốt nhất trên khẩu ngữ tự do lặp từ (`9ZlBLH7Q19u` 7.7m: **13.10% WER / 7.88% CER**).

---

## 4. Hướng dẫn Chạy Nhanh

```powershell
# 1. Chạy Unit Tests (CPU)
.\.venv\Scripts\python.exe tests/test_text_normalizer.py
.\.venv\Scripts\python.exe tests/test_vad_longform.py

# 2. Khởi chạy Web Studio (mặc định LoRA v6.1 + vad_split)
.\.venv\Scripts\python.exe app.py

# 3. Benchmark Long-Form trên bộ Gold-16
.\.venv\Scripts\python.exe -u scripts/run_longform_benchmark.py `
  --adapter artifacts/checkpoints/buzzasr_lora_v6_1_long/final `
  --manifest artifacts/manifests/yodas_long_gold_v1.test.jsonl `
  --longform-mode vad_split --no-context-carryover

# 4. Benchmark Short-Form 1,267 mẫu
.\.venv\Scripts\python.exe -u scripts/run_full_benchmark_v4.py `
  --adapter artifacts/checkpoints/buzzasr_lora_v6_1_long/final
```
