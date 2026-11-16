# BÁO CÁO TỔNG QUAN HỆ THỐNG (MODEL, PHƯƠNG PHÁP, CHECKPOINT & DATA)

**Nhánh Git:** `code` (`hoanganh1601/korean-english-asr`)  
**Báo cáo thực nghiệm chi tiết:** [`reports/korean_english_asr_report.md`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/reports/korean_english_asr_report.md)

---

## 1. Model sử dụng

* **Mô hình nền (Base Model):** `BuzzASR/korean` (kiến trúc Whisper Large, ~1.55 tỷ tham số, 32 lớp Encoder / 32 lớp Decoder).
* **Phương pháp tinh chỉnh:** LoRA PEFT (`r=64`, `lora_alpha=128`, `lora_dropout=0.05`) gắn trên 6 ma trận (`q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`), tổng **115,343,360** tham số huấn luyện (**6.95%** toàn bộ mô hình).
* **Giải mã song ngữ tự do:** Tắt ép ngôn ngữ (`forced_decoder_ids=None`, `suppress_tokens=[]`) để mô hình xuất đồng thời chữ Hangul và chữ Latin trong câu Code-Switching, đồng thời cấu hình `prev_sot_token_id=50362` (`<|startofprev|>`).
* **Adapter chính thức duy nhất được giữ lại & push lên GitHub (Git LFS):**
  * **`LoRA v6.1 Long-Form` (`final/`):** [`artifacts/checkpoints/buzzasr_lora_v6_1_long/final`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/artifacts/checkpoints/buzzasr_lora_v6_1_long/final) (`adapter_model.safetensors` `440.14 MB` / `462 MB` trên Git LFS). *(Các phiên bản cũ `v1`–`v6` đã được dọn bỏ khỏi nhánh `code` và ổ đĩa; base model `BuzzASR/korean` tự động tải từ HuggingFace Hub khi chạy).*

---

## 2. Cách đang làm (Huấn luyện & Suy luận Long-Form `vad_split`)

### 2.1 Quy trình Huấn luyện (`LoRA v6.1`)
* **Khởi tạo (Warm-Start):** Nối tiếp từ `LoRA v6/final`, huấn luyện **1,200 steps** trên 01 GPU RTX A4000 16 GB (`3h 58m`, `lr=3e-5`, cosine scheduler, `effective_batch=16`, FP16 + gradient checkpointing).
* **Phối trộn 5 nguồn dữ liệu (5-Way Mixture):** **45%** YODAS Long Clean (>=15s, >=2 câu/cửa sổ) + **14%** Zeroth Korean + **12%** LibriSpeech English + **14%** HiKE CS + **15%** Synthetic CS Loanwords.
* **Các lỗi đã khắc phục trước khi train `v6.1`:**
  * Sửa lỗi chèn trùng 2 lần token `<|startoftranscript|>` (`50258, 50258`) ở các mẫu Short-Form trong `DataCollatorSpeechSeq2SeqWithPadding`.
  * Thay khoảng lặng nhân tạo `np.zeros` giữa các câu YODAS bằng nhiễu nền nhẹ (~`-58 dBFS`), giảm ảo giác khi gặp khoảng nghỉ thật.

### 2.2 Quy trình Suy luận cho Audio/Video dài 5–10+ phút (`vad_split`)
Thay vì cắt cứng cửa sổ trượt `30.0s` của Hugging Face (dễ cắt ngang chữ và lọt khoảng lặng gây lặp câu), mọi file audio `> 30s` được xử lý tự động theo pipeline **`vad_split`**:
1. **Tách theo khoảng lặng & đáy năng lượng (`segment_audio_by_silence`):** Tính RMS theo khung `20ms` với ngưỡng thích nghi theo noise floor (`p10`) & speech peak (`p95`), gộp khoảng nghỉ `< 0.35s`, bỏ qua đoạn im lặng kéo dài, và đóng gói thành các chunk sạch **`target_max_sec=18.0s` (`hard_max_sec=26.0s`)**.
2. **Nhận dạng từng chunk & chống méo từ:** Đưa từng chunk `<= 26s` qua nhánh giải mã cửa sổ đơn với `no_repeat_ngram_size=6`, `repetition_penalty=1.05`, `max_tokens_cap=280` (giữ nguyên danh từ lặp tự nhiên thay vì `no_repeat_ngram_size=3` làm biến âm từ), giữ `condition_on_prev_tokens=False` để tránh lan truyền vòng lặp.
3. **Hậu kiểm & ghép kết quả:** Lọc bỏ vòng lặp 3+ lần liên tiếp, chuẩn hóa dấu cách sau dấu câu, và ghép lại thành 1 văn bản hoàn chỉnh kèm mảng `segments` (`start_sec`, `end_sec`, `text`).

---

## 3. Checkpoint của Phiên bản Mới nhất (`LoRA v6.1`)

### 3.1 Danh sách Checkpoint trên đĩa (`artifacts/checkpoints/buzzasr_lora_v6_1_long/`)

| Checkpoint / Thư mục | Step | Train Loss | Eval Loss | Eval WER | Eval CER | Dung lượng | Ghi chú |
|---|---:|---:|---:|---:|---:|---:|---|
| 🏆 **`final/`** *(Git LFS)* | 400 *(best)* | 1.521 | 0.1369 | 20.84% | 18.32% | **443.90 MB** | **Bản triển khai chính thức** (`adapter_model.safetensors`: `440.14 MB`) |
| `checkpoint-400/` | 400 | 1.521 | 0.1369 | 20.84% | 18.32% | 1,321.00 MB | Best metric theo `load_best_model_at_end` |
| `checkpoint-800/` | 800 | 1.232 | 0.1349 | 23.36% | 20.74% | 1,321.00 MB | Checkpoint trung gian (`eval_loss` thấp nhất) |
| `checkpoint-1200/` | 1,200 | 1.465 | 0.1379 | 23.93% | 21.30% | 1,321.01 MB | Hoàn thành 100% lịch trình (đủ state để train tiếp `v6.2`) |

### 3.2 Kết quả Nghiệm thu Chính thức của `LoRA v6.1/final`

| Bộ kiểm thử | Quy mô mẫu | WER (%) | CER (%) | Tốc độ (RTF) / VRAM |
|---|---|---:|---:|---|
| **Long-Form `Gold-16`** *(toàn bộ)* | **16 video** (2.5–9.5m, **79.79 phút**) | **14.33%** | **7.32%** | `RTF 0.471` \| `3,663.9 MB` |
| **Long-Form `Gold-16`** *(13/16 video khớp lời thoại chuẩn)* | **13 video** (**67.1 phút**) | **9.43%** | **3.64%** | Tiêu biểu: `5.27%–9.64%` WER |
| **Short-Form Weighted Average** | **1,267 mẫu** (**2.84 giờ**) | **3.91%** | **1.79%** | `RTF 0.474` \| `3,665.0 MB` |
| ↳ *HiKE Conversational Code-Switching* | 321 mẫu | 5.11% | 2.89% | `RTF 0.349` |
| ↳ *Zeroth Korean Clean Test* | 446 mẫu | 4.20% | 1.86% | `RTF 0.444` |
| ↳ *LibriSpeech English Clean Test* | 500 mẫu | 2.89% | 1.01% | `RTF 0.584` |

*(So với bản đối chứng `LoRA v5 Full`: Short-Form `3.86% WER / 1.63% CER`; Long-Form 6 video 5–10m `27.69% WER / 15.91% CER` vs `v6.1` `22.34% WER / 12.76% CER`, nhưng `v5 Full` thắng riêng video khẩu ngữ lặp tự do `9ZlBLH7Q19u`: `13.10%` vs `26.06%`.)*

---

## 4. Dữ liệu (Data)

### 4.1 Dữ liệu Short-Form & Code-Switching
* **Zeroth Korean:** `19,651` mẫu train / `2,183` dev / `446` test sạch (tiếng Hàn thuần).
* **LibriSpeech English Clean:** `28,539` mẫu train / `2,694` dev / `500` test sạch (tiếng Anh thuần).
* **HiKE Conversational Code-Switching:** `800` mẫu train / `321` eval (hội thoại trộn Hàn – Anh thực tế).
* **Synthetic CS Loanwords:** `5,605` mẫu tăng cường + bộ từ điển cặp từ mượn Anh – Hàn thay thế động trong quá trình train.

### 4.2 Dữ liệu Long-Audio (`yodas-ko-200h-v1`) & Bộ lọc Tier-2 QC
* **Nguồn gốc:** `yodas-ko-200h-v1` (`15.51 GB`, `2,521` video, `199,334` utterances sau dedupe, tổng `228.07` giờ).
* **Kiểm định chất lượng chéo Audio–Text (Tier-2 GPU QC):**
  * Quét `2,520` video trên toàn bộ 3 tập `train/dev/test`, phát hiện và cách ly **994/2,520 video (~39.4%)** bị lệch nhãn nặng (`min_cer >= 0.40` — video nói tiếng Anh gắn sub Hàn, nhạc nền BGM/slideshow hoặc chữ phủ màn hình không đọc).
* **Tập huấn luyện Long-Audio sạch (`yodas_long_windows_clean_v2.train.jsonl`):**
  * Đóng gói **21,171 cửa sổ đa câu** (**154.27 giờ**, từ **1,310 video >=45s**), trung bình **`26.23s`** & **`6.57` câu/cửa sổ**, trong đó `19,871` cửa sổ có ngữ cảnh `<|startofprev|>`, đã lọc ký tự rác (`♪`/`♬`/`저작권자`/chú thích ngoặc) và tách câu sạch.
* **Bộ Test Long-Audio Nhãn Chuẩn (`yodas_long_gold_v1.test.jsonl`):**
  * **16 video dài `2.5–9.5 phút` (tổng `79.79` phút)** lọc theo tiêu chí khách quan: loại 100% video lỗi Tier-2 QC, `0` chú thích cảnh `(...)`/`[...]`, `0` ký tự nhạc/jamo, mật độ lời thoại tự nhiên `3.8 <= chars/sec <= 6.2`.
