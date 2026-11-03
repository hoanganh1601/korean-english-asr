# Korean–English Code-Switching ASR (Fine-Tuning & Evaluation Studio)

Dự án nghiên cứu, huấn luyện và triển khai mô hình nhận dạng giọng nói tự động (**ASR - Automatic Speech Recognition**) chuyên biệt cho **tiếng Hàn, tiếng Anh và câu nói pha trộn Hàn – Anh (Code-Switching)** trong ngữ cảnh giao tiếp đời sống, công sở và hội thảo kỹ thuật.

Mô hình đạt kỷ lục **WER 4.95%** trên tập dữ liệu Code-Switching thực tế HiKE (giảm mạnh từ **53.30%** của mô hình gốc), đạt **WER 3.86%** tổng thể trên toàn bộ bộ kiểm thử độc lập 1,267 mẫu (2.84 giờ âm thanh), với tốc độ suy luận nhanh gấp **2.61 lần** thời gian thực (RTF 0.383) và tiêu thụ chỉ **3.58 GB VRAM**.

---

## 🏆 Bảng Kết Quả Nghiệm Thu Chính Thức (Official Benchmark)

Đánh giá độc lập trên **1,267 mẫu** (tổng thời lượng **2.839 giờ âm thanh**, chạy liên tục 65.6 phút trên 01 GPU NVIDIA RTX A4000) với bộ chuẩn hóa và căn chỉnh song ngữ [`src/evaluation/text_normalizer.py`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/src/evaluation/text_normalizer.py). Dữ liệu chi tiết từng mẫu được lưu tại [`artifacts/benchmark_lora_v5_full_results.json`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/artifacts/benchmark_lora_v5_full_results.json):

| Bộ dữ liệu kiểm định | Quy mô kiểm thử | BuzzASR Base (Gốc) | LoRA v3 (Bản đầu) | LoRA v4 (Bản nâng cấp) | 🏆 **LoRA v5 Full (Nghiệm thu chính thức)** | Mức độ cải thiện |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 🔀 **HiKE Code-Switching** | **321 mẫu** (38.1 phút) | WER 53.30% <br> CER 28.50% | WER 10.09% <br> CER 6.18% | WER 9.33% <br> CER 4.81% | ⭐ **WER 4.95%** <br> ⭐ **CER 2.68%** <br> *(Word Acc: **95.05%**)* | 🔻 **Giảm hơn 48.3% WER!** <br> *(Từ 53.30% $\rightarrow$ 4.95%)* |
| 🇰🇷 **Zeroth Korean Clean** | **446 mẫu** (69.7 phút) | WER 14.10% <br> CER 4.02% | WER 13.74% <br> CER 3.75% | WER 6.94% <br> CER 1.77% | ⭐ **WER 4.10%** <br> ⭐ **CER 1.62%** <br> *(Char Acc: **98.38%**)* | 🔻 **Giảm hơn 70% lỗi** <br> *(Bảo toàn âm học tiếng Hàn)* |
| 🇺🇸 **LibriSpeech English Clean** | **500 mẫu** (62.6 phút) | WER 12.50% <br> *(Lỗi tự dịch)* | WER 9.81% <br> CER 7.38% | WER 3.42% <br> CER 1.03% | ⭐ **WER 2.94%** <br> ⭐ **CER 0.99%** <br> *(Word Acc: **97.06%**)* | 🔻 **Lỗi từ dưới 3%** <br> *(Triệt tiêu hoàn toàn lỗi dịch nhầm)* |
| 🎯 **TỔNG THỂ (WEIGHTED OVERALL)** | **1,267 mẫu** (2.839 giờ) | WER ~25.4% | WER ~11.1% | WER 6.16% | 🏆 **WER: 3.86%** <br> 🏆 **CER: 1.64%** <br> 🏆 **Word Acc: 96.14%** | 🔻 **Giảm 84.8% lỗi tổng thể** <br> *(Vượt xa mọi chuẩn đề ra)* |

### Hiệu năng phần cứng:
- **Tốc độ xử lý (Overall RTF):** **0.383** (Nhanh hơn thời gian thực 2.61 lần).
- **Tốc độ xử lý Code-Switching (HiKE RTF):** **0.285** (Nhanh hơn thời gian thực 3.51 lần).
- **Bộ nhớ GPU khi suy luận (Peak VRAM):** **3.58 GB** / 16 GB RTX A4000 (Tương thích tốt với GPU phổ thông như RTX 3060/4060).
- **Kích thước LoRA Adapter:** **461.5 MB** (`adapter_model.safetensors`), tiện phân phối và cập nhật qua mạng.

---

## 💡 Thách Thức & Đột Phá Kỹ Thuật

### 1. Vấn đề cốt lõi của bài toán Code-Switching Hàn – Anh
- **Hiện tượng Hangul Loanword Transliteration:** Mô hình gốc có xu hướng ép âm thanh tiếng Anh sang chữ cái Hangul (ví dụ: nghe *"meeting"* ghi thành *"미팅"*, nghe *"feature"* ghi thành *"피처"*, nghe *"presentation"* ghi thành *"프레젠테이션"*).
- **Cơ chế Token Constraint của Whisper:** Token giải mã mặc định (`forced_decoder_ids`) buộc mô hình chỉ xuất một hệ chữ duy nhất, triệt tiêu khả năng xuất đồng thời chữ Latin và Hangul.
- **Hiện tượng Speech Translation Bias:** Khi gặp câu thuần tiếng Anh, decoder có xu hướng tự động dịch sang tiếng Hàn thay vì phiên âm nguyên bản.

### 2. Các giải pháp kỹ thuật đã triển khai thành công
1. **LoRA Full Linear Projections ($r=64, \alpha=128$):** Mở rộng LoRA vào toàn bộ 6 ma trận tuyến tính (`q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`) trên cả Encoder và Decoder (115M tham số có thể huấn luyện, chiếm 6.95% Whisper Large), bảo toàn 93.05% trọng số gốc.
2. **Free Script Decoding & Language Conditioning:** Gỡ bỏ `forced_decoder_ids`, cho phép giải mã song song hai bảng chữ cái; bổ sung token `<|en|><|transcribe|>` khi huấn luyện nhánh tiếng Anh để triệt tiêu bias tự dịch.
3. **Chiến lược dữ liệu 4 nguồn (54,595 mẫu / 171.87 giờ audio):**
   - 40% Korean Monolingual (Zeroth Korean, 19,651 mẫu);
   - 25% English Monolingual (LibriSpeech, 28,539 mẫu);
   - 20% Synthetic Code-Switching (`koen_synthetic_v1`, 5,605 mẫu ghép nối hai chiều KO $\rightarrow$ EN và EN $\rightarrow$ KO);
   - 15% Real Code-Switching (HiKE, 800 mẫu đàm thoại thực tế).
4. **Data Augmentation với 592 cặp từ mượn:** Sử dụng Regex thế chỗ động từ mượn tiếng Hàn sang từ gốc tiếng Anh với xác suất $p=0.75$, huấn luyện mô hình xuất trực tiếp chữ Latin khi nghe phát âm tiếng Anh.
5. **Bộ Chuẩn Hóa & Căn Chỉnh Song Ngữ ([src/evaluation/text_normalizer.py](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/src/evaluation/text_normalizer.py)):**
   - **Nhóm 1:** Xử lý từ mượn đời sống kèm trợ từ tiếng Hàn (`카페에서` vs `cafe에서`).
   - **Nhóm 2:** Căn chỉnh khoảng trắng danh từ ghép Hàn (`수사 내용` vs `수사내용`) khi `CER = 0%`.
   - **Nhóm 3:** Chuẩn hóa Unicode dấu nháy cong (`’`), sở hữu cách `'s` vs `is`, và chuẩn Anh – Mỹ (UK $\rightarrow$ US).
   - **Nhóm 4:** Hỗ trợ Hotwords Prompting cho tên riêng và từ đồng âm.

---

## 🛠️ Cài Đặt Môi Trường

### Yêu cầu hệ thống:
- Hệ điều hành: Windows 10/11 hoặc Linux (Ubuntu 20.04+)
- Python: 3.10 – 3.12
- GPU: NVIDIA GPU có ít nhất 6 GB VRAM (Khuyến nghị 8 GB – 16 GB), CUDA 12.1+

### Các bước cài đặt:
```bash
# 1. Clone repository
git clone https://github.com/hoanganh1601/korean-english-asr.git
cd korean-english-asr

# 2. Tạo và kích hoạt môi trường ảo
python -m venv .venv

# Trên Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Trên Linux/macOS:
source .venv/bin/activate

# 3. Cài đặt các thư viện phụ thuộc
pip install --upgrade pip
pip install -r requirements.lock.txt
```

---

## 🚀 Hướng Dẫn Vận Hành

### 1. Khởi động Web Studio UI (Tương tác trực quan)
Web Studio cung cấp giao diện đồ họa trực quan với khả năng tải lên audio, nghe lại trực tiếp, tự động tra cứu transcript đối chiếu (ground truth), phiên âm tức thời, và phân tích chi tiết từng lỗi từ/ký tự:

```bash
# Cách 1: Chạy file script trên Windows
.\run_ui.bat

# Cách 2: Chạy trực tiếp qua lệnh Python
python app.py --port 7860
```
Sau khi khởi động, mở trình duyệt truy cập: **`http://localhost:7860`**.

---

### 2. Phiên âm qua CLI (Command Line Interface)
Sử dụng runner [`src.evaluation.buzzasr_runner`](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/src/evaluation/buzzasr_runner.py) để phiên âm bất kỳ file âm thanh nào (`.wav`, `.mp3`, `.flac`, `.m4a`):

```bash
# Phiên âm 1 file âm thanh với mô hình LoRA v5 Full
python -m src.evaluation.buzzasr_runner \
    --adapter artifacts/checkpoints/buzzasr_lora_v5_full/final \
    --audio "duong_dan_den_file_audio.wav"

# Phiên âm kèm Hotwords Prompt (đối với tên riêng hiếm gặp)
python -m src.evaluation.buzzasr_runner \
    --adapter artifacts/checkpoints/buzzasr_lora_v5_full/final \
    --audio "duong_dan_den_file_audio.wav" \
    --prompt "Kubernetes, PyTorch, Docker"
```

---

### 3. Chạy Full Benchmark Độc Lập (1,267 Mẫu)
Thực thi kiểm định toàn diện trên cả 3 tập held-out test suite (HiKE, Zeroth Korean, LibriSpeech English):

```bash
python scripts/run_full_benchmark_v4.py
```
Kết quả sẽ tự động lưu vào file JSON nghiệm thu tại `artifacts/benchmark_lora_v5_full_results.json`.

---

### 4. Huấn luyện Mô hình LoRA (Training Pipeline)
Nếu muốn huấn luyện lại hoặc tiếp tục fine-tuning từ checkpoint:

```bash
python -m src.training.train_lora_buzzasr \
    --init-adapter artifacts/checkpoints/buzzasr_lora_v4/final \
    --output-dir artifacts/checkpoints/buzzasr_lora_v5_full \
    --learning-rate 5e-5 \
    --num-train-epochs 1.0 \
    --eval-steps 400 \
    --save-steps 400 \
    --loanwords-prob 0.75
```

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
korean-english-asr/
├── app.py                             # Máy chủ Web Studio UI (Aiohttp + Hot-Reload)
├── run_ui.bat                         # Phím tắt khởi chạy nhanh Web UI trên Windows
├── requirements.lock.txt              # Danh sách thư viện chuẩn xác định phiên bản
├── README.md                          # Tài liệu tổng quan dự án
├── HANDOFF.md                         # Báo cáo bàn giao kỹ thuật toàn diện
├── artifacts/
│   ├── loanwords_dict.json            # 592 cặp từ mượn đối sánh tiếng Hàn - Anh
│   ├── benchmark_lora_v5_full_results.json # Kết quả nghiệm thu 1,267 mẫu
│   ├── checkpoints/
│   │   ├── buzzasr_lora_v3/           # LoRA v3 Adapter
│   │   ├── buzzasr_lora_v4/           # LoRA v4 Adapter
│   │   └── buzzasr_lora_v5_full/      # 🏆 LoRA v5 Full Adapter (Bản nghiệm thu)
│   │       └── final/                 # Trọng số adapter hoàn chỉnh (461.5 MB)
│   └── manifests/                     # Manifests JSONL cho huấn luyện và kiểm thử
├── reports/
│   ├── korean_english_asr_report.md   # Báo cáo kỹ thuật chi tiết chuyên sâu
│   └── data_qc/                       # Báo cáo kiểm định chất lượng âm thanh
├── scripts/
│   └── run_full_benchmark_v4.py       # Script chạy Full Benchmark 1,267 mẫu
├── src/
│   ├── evaluation/
│   │   ├── buzzasr_runner.py          # Whisper/PEFT ASR inference runner
│   │   └── text_normalizer.py         # Bộ chuẩn hóa & căn chỉnh song ngữ 4 nhóm lỗi
│   ├── training/
│   │   ├── train_lora_buzzasr.py      # Pipeline huấn luyện LoRA bằng HuggingFace Trainer
│   │   └── manifest_dataset.py        # Dataset loader & Synthetic loanword augmentation
│   └── ui/
│       ├── index.html                 # Giao diện Web Studio UI
│       └── static/                    # CSS & JavaScript cho giao diện
└── tests/
    └── test_text_normalizer.py        # Bộ kiểm thử đơn vị cho text normalizer
```

---

## ✅ Tiêu Chí Hoàn Thành & Nghiệm Thu

- [x] **Baseline & Benchmark đa miền:** Đo lường chi tiết trước và sau khi Fine-Tuning trên cùng tập kiểm thử held-out cố định 1,267 mẫu.
- [x] **Đột phá Code-Switching (HiKE):** Kéo WER từ **53.30%** xuống **4.95%** (Vượt xa mục tiêu ban đầu < 10%).
- [x] **Bảo toàn tiếng Hàn bản xứ (Zeroth):** Đạt **CER 1.62%**, **WER 4.10%** (Độ chính xác ký tự đạt 98.38%).
- [x] **Bảo toàn tiếng Anh chuẩn (LibriSpeech):** Đạt **WER 2.94%**, **CER 0.99%** (Word Accuracy 97.06%, triệt tiêu 100% lỗi tự dịch sang Hangul).
- [x] **Tài nguyên & Tốc độ suy luận:** Peak VRAM **3.58 GB** (< 12 GB ngân sách), RTF **0.383** (nhanh hơn thời gian thực 2.61 lần).
- [x] **Hệ thống chuẩn hóa song ngữ:** Giải quyết trọn vẹn 4 nhóm lỗi phi âm học thực tế (từ mượn + trợ từ, khoảng trắng danh từ ghép, dấu nháy cong, chuẩn UK/US).
- [x] **Đầy đủ công cụ bàn giao:** Có CLI Runner, Web Studio UI trực quan, toàn bộ code huấn luyện và tài liệu kỹ thuật chi tiết.

---

## 📜 Giấy Phép & Ghi Nhận Bản Quyền

- **Mã nguồn dự án:** Phát hành theo giấy phép [MIT License](LICENSE).
- **Mô hình nền:** `BuzzASR/korean` phát hành trên HuggingFace Hub dựa trên kiến trúc OpenAI Whisper.
- **Dữ liệu huấn luyện:**
  - [Zeroth Korean](https://github.com/goodatlas/zeroth): CC BY 4.0.
  - [LibriSpeech](https://www.openslr.org/12): CC BY 4.0.
  - [HiKE Dataset](https://huggingface.co/datasets/thetaone-ai/HiKE): Mục đích nghiên cứu học thuật và đánh giá mô hình.
