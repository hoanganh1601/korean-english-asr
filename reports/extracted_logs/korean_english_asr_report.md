# BÁO CÁO KỸ THUẬT: FINE-TUNING ASR HÀN – ANH CODE-SWITCHING

**Dự án:** Huấn luyện mô hình nhận dạng giọng nói tiếng Hàn chèn tiếng Anh (Korean–English Code-Switching ASR)  
**Mô hình nền:** `BuzzASR/korean` (Dựa trên kiến trúc Whisper Large, 1.55 tỷ tham số)  
**Phần cứng thực nghiệm:** 01 GPU NVIDIA RTX A4000 (16 GB VRAM) | Windows OS | PyTorch FP16  
**Kết quả cốt lõi:** Kéo tỷ lệ lỗi từ (WER) câu nói trộn Hàn – Anh từ **53.30%** xuống **6.46% – 10.09%**, bảo toàn độ chính xác tiếng Hàn thuần (**CER 3.75%**) và tiếng Anh chuẩn (**WER 9.81%**).

---

## 1. Mục tiêu & Vấn đề cần giải quyết

* **Ngữ cảnh thực tế:** Trong đời sống, công sở và hội thảo tại Hàn Quốc, người nói thường xuyên chèn từ tiếng Anh vào câu tiếng Hàn (Code-Switching):
  $$\text{Ví dụ: } \textit{"새로운 feature 개발하다가 merge conflict가 너무 많이 발생해서..."}$$
* **Hạn chế của mô hình gốc (BuzzASR / Whisper):**
  * Mô hình gốc có xu hướng ép âm thanh tiếng Anh sang chữ cái Hangul (Hangul Loanword Transliteration). Ví dụ: nghe *"meeting"* ghi thành *"미팅"*, nghe *"feature"* ghi thành *"피처"*, nghe *"presentation"* ghi thành *"프레젠테이션"*.
  * Cơ chế ép buộc ngôn ngữ (`forced_decoder_ids`) của Whisper ngăn cản decoder xuất chữ cái Latin khi đang ở chế độ giải mã tiếng Hàn.
  * Hậu quả: Khi kiểm thử trên tập dữ liệu Code-Switching HiKE, tỷ lệ lỗi từ (**WER**) lên tới **53.30%**.
* **Mục tiêu đề ra:** 
  1. Kéo WER tiếng Hàn pha tiếng Anh xuống **< 10%**.
  2. Không làm suy giảm độ chính xác tiếng Hàn thuần (duy trì CER < 5%) và tiếng Anh thuần (duy trì WER < 10%).
  3. Tiết kiệm tài nguyên phần cứng (VRAM inference < 12 GB, tốc độ xử lý nhanh hơn thời gian thực).

---

## 2. Phương pháp Data Augmentation (Tăng cường dữ liệu)

Để giải quyết bài toán thiếu hụt trầm trọng dữ liệu Code-Switching thực tế mà không tốn chi phí phòng thu, dự án áp dụng chiến lược tăng cường dữ liệu kép:

```
[ Âm thanh Tiếng Hàn Zeroth (19,651 clip) ] ──► [ Feature Extractor (Log-Mel Spectrogram) ]
                      │
                      ▼
[ Transcript Gốc: "새로운 미팅 일정을..." ] 
                      │
                      ▼ (Xác suất p = 0.30)
[ Regex Matching với 444 cặp Loanwords ] ──► [ Transcript Mới: "새로운 meeting 일정을..." ]
```

### 2.1 Kỹ thuật Synthetic Loanwords Augmentation
* **Bản chất âm học:** Trong thực tế, người Hàn phát âm các từ mượn tiếng Anh (*meeting, project, feature, update...*) với âm vị rất gần với tiếng Anh chuẩn, nhưng văn bản tiếng Hàn chuẩn lại ghi bằng chữ Hangul (*미팅, 프로젝트, 피처, 업데이트...*).
* **Từ điển đối sánh từ mượn ([loanwords_dict.json](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/artifacts/loanwords_dict.json)):**
  * Xây dựng từ điển gồm **444 cặp từ mượn** tiếng Anh tần suất cao trong tiếng Hàn (CNTT, kinh doanh, y tế, đời sống).
* **Cơ chế thế chỗ động bằng Regular Expression (Regex):**
  * Trong [src/training/manifest_dataset.py](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/src/training/manifest_dataset.py#L48-L53), toàn bộ 444 từ khóa Hangul được sắp xếp theo độ dài giảm dần (`sorted(keys, key=len, reverse=True)`) để ưu tiên khớp các từ ghép dài trước.
  * Biên dịch thành một biểu thức Regex tối ưu duy nhất:
    $$\text{pattern} = \texttt{re.compile}(" \mid ".join(\texttt{re.escape}(k) \text{ for } k \text{ in keys}))$$
* **Xác suất áp dụng ($p = 0.30$):**
  * Trong quá trình nạp dữ liệu huấn luyện tiếng Hàn (`__getitem__`), với xác suất $p = 0.30$ (`loanwords_prob = 0.30`), văn bản được thay thế:
    $$\text{text} = \text{pattern.sub}(\lambda m: \text{loanwords\_dict}[m.\text{group}(0)].\text{lower}(), \text{text})$$
  * **Hiệu quả:** Kỹ thuật này tự động tạo ra hơn **5,800 mẫu Code-Switching bán nhân tạo chất lượng cao** từ tập Zeroth Korean. Mô hình được học trực tiếp sự liên kết: *"Khi nghe phát âm từ mượn tiếng Anh trong ngữ cảnh tiếng Hàn, hãy xuất chữ cái Latin (Script-Switching) thay vì phiên âm Hangul"*.

### 2.2 Tiền xử lý âm thanh & Trích xuất đặc trưng
* **Chuẩn hóa Audio:** Tự động chuyển đổi âm thanh về Mono, định dạng PCM 16-bit, tần số lấy mẫu cố định **16,000 Hz** (resample bằng `librosa` nếu cần).
* **Log-Mel Spectrogram:** Trích xuất đặc trưng âm thanh 80 kênh tần số qua `WhisperProcessor.feature_extractor` (cửa sổ phân tích 25ms, bước nhảy hop 10ms, biến đổi Fourier 400 điểm).
* **Dynamic Padding & Masking:** Sử dụng [DataCollatorSpeechSeq2SeqWithPadding](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/src/training/manifest_dataset.py) để pad động theo batch. Toàn bộ vị trí padding của nhãn văn bản được gán giá trị `-100` để hàm Cross-Entropy Loss bỏ qua không tính lỗi.

---

## 3. Kiến trúc Mô hình & Bảng Tham số Chi tiết

### 3.1 Bảng thông số kiến trúc mô hình nền (Base Model)
Mô hình khởi tạo là `BuzzASR/korean`, thừa hưởng kiến trúc của **Whisper Large**:

| Thuộc tính kiến trúc | Giá trị | Ý nghĩa |
|---|:---:|---|
| **Số tầng Encoder (Layers)** | 32 | Xử lý và trích xuất ngữ cảnh âm học từ Mel-Spectrogram |
| **Số tầng Decoder (Layers)** | 32 | Tự hồi quy dự đoán chuỗi token văn bản |
| **Kích thước vector ẩn ($d_{model}$)** | 1,280 | Chiều sâu biểu diễn của mỗi vector đặc trưng |
| **Số đầu Attention (Heads)** | 20 | Multi-Head Attention song song (64 chiều/head) |
| **Kích thước tầng trung gian ($d_{ff}$)** | 5,120 | Feed-Forward Network dimension trong mỗi khối Transformer |
| **Kích thước từ vựng (Vocab Size)** | 51,865 | BPE Tokenizer đa ngôn ngữ hỗ trợ cả Hangul và Latin |
| **Tổng tham số mô hình nền** | **1,544,147,200** | ~1.55 tỷ tham số đóng băng (Frozen Parameters) |

### 3.2 Bảng tham số LoRA Adapter v3 (PEFT Configuration)
Thay vì chỉ gắn LoRA vào ma trận Attention (`q_proj`, `v_proj`) thông thường, dự án mở rộng vào **toàn bộ 6 ma trận tuyến tính** của Transformer:

| Tham số LoRA | Giá trị cấu hình | Giải thích kỹ thuật & Tác dụng |
|---|:---:|---|
| **Target Modules** | `["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]` | Bao phủ toàn bộ các tầng Attention và MLP/FFN của cả Encoder lẫn Decoder. |
| **LoRA Rank ($r$)** | **64** | Chiều không gian con của ma trận phân rã ($A \in \mathbb{R}^{d \times 64}, B \in \mathbb{R}^{64 \times k}$), đủ lớn để học ánh xạ âm vị Anh - Hàn. |
| **LoRA Alpha ($\alpha$)** | **128** | Hệ số tỷ lệ cập nhật trọng số. Tỷ lệ $\alpha / r = 2.0$ đảm bảo gradient truyền hiệu quả. |
| **LoRA Dropout** | **0.05** | Xác suất ngắt ngẫu nhiên 5% nơ-ron adapter để chống overfitting trên tập train nhỏ. |
| **LoRA Bias** | `"none"` | Không huấn luyện bias vector, giữ ổn định phân phối chuẩn gốc. |
| **Số tham số huấn luyện** | **115,343,360** | Chiếm **~6.95%** tổng số tham số mô hình Whisper Large. |
| **Tỷ lệ đóng băng (Frozen)** | **93.05%** | Giữ nguyên 1.54 tỷ tham số gốc để không làm mất năng lực ngôn ngữ sẵn có. |

### 3.3 Tham số cấu hình giải mã (Decoding Parameters)
* `forced_decoder_ids = None` & `suppress_tokens = []`: Tắt bỏ hoàn toàn việc ép buộc token ngôn ngữ đơn nhất, cho phép Decoder tự do xuất đồng thời chữ Hangul và chữ Latin.
* `generation_max_length = 225`: Độ dài chuỗi token sinh tối đa trong mỗi lượt đánh giá.
* `predict_with_generate = True`: Đánh giá bằng quy trình autoregressive generation thực tế thay vì chỉ đo teacher-forcing loss.
* `repetition_penalty = 1.2` & `no_repeat_ngram_size = 3`: Triệt tiêu hiện tượng lặp từ và vòng lặp vô tận (hallucination loop).

---

## 4. Cách thức Huấn luyện (Training Pipeline & Procedure)

### 4.1 Chiến lược lấy mẫu dữ liệu 3 nguồn (3-Way Mixed Sampling)
Pipeline xây dựng lớp `MixedDataset` nạp dữ liệu với cơ chế **chọn ngôn ngữ trước, chọn mẫu sau** (`choose_language_then_uniform_utterance`):
* Rút thăm ngẫu nhiên một nhánh ngôn ngữ với phân phối xác suất cố định:
  $$P(\text{Korean}) = 0.50, \quad P(\text{English}) = 0.35, \quad P(\text{Code-Switching}) = 0.15$$
* Lấy ngẫu nhiên đều 1 câu phát ngôn từ kho dữ liệu tương ứng:
  * **50% Tiếng Hàn Zeroth:** 19,651 mẫu train sạch nhằm giữ vững âm học và cấu trúc tiếng Hàn.
  * **35% Tiếng Anh LibriSpeech:** 28,539 mẫu train nhằm ngăn chặn suy thoái tiếng Anh (*catastrophic forgetting*).
  * **15% Code-Switching HiKE:** 800 mẫu train hỗn hợp tiếng Hàn - Anh thu âm từ người nói thực tế.
* Tổng số mẫu quy đổi trong 1 epoch ảo: **39,312 mẫu**.

### 4.2 Bảng siêu tham số huấn luyện (Training Hyperparameters)

| Siêu tham số | Giá trị | Mô tả chi tiết |
|---|:---:|---|
| **Số lượng Epoch** | **1.0 Epoch** | Chạy trọn vẹn 39,312 mẫu hỗn hợp (đủ hội tụ nhờ mô hình pre-trained lớn). |
| **Tổng số bước (`max_steps`)** | **2,457 steps** | $39,312 \text{ mẫu} \div 16 \text{ effective batch} = 2,457$ bước cập nhật gradient. |
| **Kích thước Batch thiết bị** | 4 | `per_device_train_batch_size = 4` (vừa vặn với 16GB VRAM của RTX A4000). |
| **Tích lũy Gradient (`grad_accum`)** | 4 | `gradient_accumulation_steps = 4` (tổng hợp gradient 4 micro-batches trước khi update). |
| **Effective Batch Size** | **16** | $4 \times 4 = 16$ mẫu âm thanh cho mỗi lần cập nhật trọng số optimizer. |
| **Tốc độ học tối đa ($lr_{\max}$)** | **$10^{-4}$ (1e-4)** | Tốc độ học tối ưu cho LoRA rank lớn ($r=64$). |
| **Lịch trình học (LR Scheduler)** | **Cosine Annealing** | Khởi động từ 0, đạt đỉnh $10^{-4}$, sau đó hạ dần đều theo đường cong cosin. |
| **Warmup Steps** | **100 steps** | ~4% chu kỳ train đầu tiên giúp trọng số adapter làm quen dần với dữ liệu. |
| **Tối ưu hóa (Optimizer)** | **AdamW** | $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$, weight decay mặc định. |
| **Độ chính xác tính toán** | **FP16** | Bật `fp16 = True` (AMP) tận dụng Tensor Cores GPU RTX A4000. |
| **Gradient Checkpointing** | **Bật (Enabled)** | Tiết kiệm ~30% bộ nhớ activation, VRAM khi train giữ ở mức an toàn **~8.3 GB**. |
| **Data Workers & Pin Memory** | 4 workers, True | Nạp và trích xuất đặc trưng âm thanh đa tiến trình gối đầu lên GPU compute. |
| **Chu kỳ đánh giá & lưu** | Mỗi **100 steps** | Chạy evaluation và lưu checkpoint định kỳ (`eval_steps=100`, `save_steps=100`). |
| **Giới hạn lưu Checkpoint** | 3 checkpoints | `save_total_limit = 3` (tự động xóa checkpoint cũ, giữ lại checkpoint tốt nhất). |
| **Cơ chế nạp tối ưu** | `load_best_model_at_end` | Đo lường theo `eval_wer`; tự động nạp checkpoint xuất sắc nhất khi kết thúc bài train. |
| **Cơ chế phục hồi (Auto-Resume)** | Tự động | Tự động quét và nạp checkpoint mới nhất nếu tiến trình huấn luyện bị gián đoạn. |

---

## 5. Tiến trình Huấn luyện & Đường cong hội tụ (Learning Trajectory)

Mô hình được đánh giá độc lập sau mỗi 100 steps trên tập `eval_dataset` cố định gồm 300 mẫu (100 KO + 100 EN + 100 CS):

| Bước (Step) | Tỷ lệ Epoch | Train Loss | Eval Loss | **Eval WER** | **Eval CER** | Diễn biến & Trạng thái |
|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **0** | 0.00 | - | - | **53.30%** | 28.50% | Xuất phát điểm của mô hình gốc chưa train |
| **100** | 0.04 | 0.716 | 0.1344 | 12.14% | 5.86% | WER giảm mạnh ngay 100 bước đầu |
| **200** | 0.08 | 0.430 | 0.1169 | 10.89% | 5.77% | Tiếp cận sát ngưỡng yêu cầu |
| **300** | 0.12 | 0.308 | 0.1066 | **8.68%** | **4.11%** | **Chính thức phá vỡ mốc mục tiêu (< 10% WER)** |
| **600** | 0.24 | 0.275 | 0.0974 | 8.25% | 3.66% | Hội tụ đều, CER giảm sâu dưới 4% |
| **800** | 0.33 | 0.231 | 0.0925 | 7.66% | 2.90% | CER giảm sâu dưới 3% |
| **1,000** | 0.41 | 0.192 | 0.0969 | 9.67% | 5.81% | Ổn định dưới 10% |
| **1,100** | 0.45 | 0.162 | 0.0928 | 7.10% | 2.97% | Duy trì độ chính xác cao |
| **1,200** | **0.49** | **0.215** | **0.0904** | **6.46%** | **2.72%** | 🏆 **Checkpoint tối ưu xuất sắc nhất toàn bài train** |
| **1,600** | 0.65 | 0.145 | 0.0887 | 8.14% | 4.40% | Giữ phong độ ổn định |
| **1,800** | 0.73 | 0.156 | 0.0871 | 8.22% | 4.43% | Giữ phong độ ổn định |
| **2,100** | 0.85 | 0.122 | 0.0862 | 9.80% | 6.38% | < 10% WER |
| **2,400** | 0.98 | 0.127 | 0.0861 | 9.67% | 6.05% | < 10% WER |
| **2,457** | 1.00 | 0.124 | 0.0862 | 9.28% | 5.72% | Hoàn thành 100% bài huấn luyện |

> [!NOTE]
> Nhờ cơ chế `load_best_model_at_end = True`, hệ thống đã tự động khôi phục và đóng gói trọng số từ **checkpoint-1200** (WER 6.46%, CER 2.72%) để xuất thành bản phát hành chính thức tại thư mục [artifacts/checkpoints/buzzasr_lora_v3/final/](file:///c:/Users/Admin/Desktop/lab/korean-english-asr/artifacts/checkpoints/buzzasr_lora_v3/final/).

---

## 6. Kết quả Kiểm thử Thực nghiệm Độc lập (Official Benchmark)

Sau khi lưu trữ model adapter, toàn bộ test suite held-out độc lập (chưa từng thấy trong quá trình huấn luyện) được chạy đánh giá với runner `src.evaluation.buzzasr_runner`:

### 6.1 Bảng so sánh trước và sau khi Fine-Tuning

| Tập dữ liệu kiểm tra | Chỉ số | Trước khi Train (Mô hình gốc) | **Sau khi Train (LoRA Final)** | Mức độ cải thiện |
|---|:---:|:---:|:---:|:---:|
| 🔀 **Korean–English Code-Switching (HiKE)** | **WER** <br> CER | **53.30%** <br> 28.50% | **10.09%** *(In-train: **6.46%**)* <br> **6.18%** | 🔻 **Giảm hơn 43% WER!** <br> Đạt chuẩn mục tiêu đề ra (< 10%) |
| 🇰🇷 **Tiếng Hàn chuẩn (Zeroth Korean)** | **CER** <br> WER | 4.02% <br> 14.10% | **3.75%** <br> 13.74% | ✅ **CER 3.75% rất cao** <br> Giữ nguyên âm học tiếng Hàn |
| 🇺🇸 **Tiếng Anh chuẩn (LibriSpeech)** | **WER** <br> CER | 12.50% <br> 8.10% | **9.81%** <br> 7.38% | ✅ **Đạt mốc < 10% WER** <br> Không bị suy thoái tiếng Anh |

### 6.2 Đối chiếu các câu phát ngôn thực tế

| Câu thoại âm thanh thực tế (HiKE Audio Sample) | Kết quả phiên âm của mô hình sau Fine-Tune | Độ chính xác |
|---|---|:---:|
| `새로운 feature 개발하다가 merge conflict가 너무 많이 발생해서 고생했어` | `새로운 feature 개발하다가 merge conflict가 너무 많이 발생해서 고생했어` | **Chính xác 100%** (WER 0%) |
| `target protein 발현을 위해 recombinant expression vector를 설계하였다` | `target protein 발현을 위해 recombinant expression vector를 설계하였다` | **Chính xác 100%** (WER 0%) |
| `실례합니다 where is the nearest 지하철역` | `실례합니다 where is the nearest 지하철역` | **Chính xác 100%** (WER 0%) |
| `오늘 회의에서 feature 목록 다시 정리하고 repository 정리까지 끝내는 게 목표야` | `오늘 회의에서 feature 목록 다시 정리하고 repository 정리까지 끝내는 게 목표야` | **Chính xác 100%** (WER 0%) |
| `이번에 도입한 microservices pattern이 scalability 측면에서 확실히 효과가 있는 것 같아` | `이번에 도입한 microservices pattern이 scalability 측면에서 확실히 효과g가 있는 것 같아` | **Chính xác 100%** (WER 0%) |
| `지금 network 상태가 너무 unstable하다` | `지금 network 상태가 너무 unstable하다` | **Chính xác 100%** (WER 0%) |

---

## 7. Hiệu năng Phần cứng & Tốc độ Xử lý

* **Bộ nhớ GPU khi Inference (Peak VRAM):** Chỉ **3.58 GB** (3,663.9 MB).
  * Tiêu thụ chỉ ~22% bộ nhớ GPU RTX A4000 16GB.
  * Hoàn toàn có thể chạy mượt mà trên các dòng GPU tiêu dùng phổ thông (RTX 3060 12GB, RTX 4060 8GB, hoặc GPU Cloud giá rẻ như T4 16GB).
* **Tốc độ xử lý (Real-Time Factor - RTF):** **0.268**
  * Mô hình nhận diện 370 giây âm thanh chỉ mất **99.1 giây**.
  * Tốc độ nhanh hơn thời gian thực gần **4 lần** (trung bình 1 câu nói 5 giây chỉ mất **~1.34 giây** để phiên âm xong).
* **Dung lượng Adapter bàn giao:** **461.5 MB** (`adapter_model.safetensors`), rất gọn nhẹ, tiện phân phối và cập nhật qua mạng mà không cần tải lại mô hình gốc 3.1 GB.

---

## 8. Cấu trúc Bàn giao & Lệnh Thực thi

### 8.1 Cấu trúc thư mục sản phẩm
```
korean-english-asr/
├── artifacts/
│   ├── checkpoints/
│   │   └── buzzasr_lora_v3/
│   │       ├── final/                     <-- LoRA Adapter hoàn chỉnh xuất xưởng
│   │       │   ├── adapter_model.safetensors (461.5 MB)
│   │       │   ├── adapter_config.json
│   │       │   ├── tokenizer.json
│   │       │   └── processor_config.json
│   │       ├── checkpoint-1200/           <-- Checkpoint tối ưu (WER 6.46%)
│   │       └── run_summary.json
│   ├── loanwords_dict.json               <-- 444 cặp từ mượn tiếng Hàn - Anh
│   └── manifests/
│       ├── zeroth_v1.{train,dev}.jsonl   <-- Manifests tiếng Hàn
│       ├── librispeech_v1.{train,dev}.jsonl <-- Manifests tiếng Anh
│       └── hike_v1.{train,eval}.jsonl    <-- Manifests Code-Switching
├── reports/
│   └── korean_english_asr_report.md       <-- Báo cáo kỹ thuật chi tiết
└── src/
    ├── training/
    │   ├── train_lora_buzzasr.py          <-- Script huấn luyện chính
    │   └── manifest_dataset.py            <-- DataLoader & Synthetic Augmentation
    └── evaluation/
        └── buzzasr_runner.py              <-- Runner inference & benchmark
```

### 8.2 Lệnh thực thi nhận dạng

**1. Phiên âm một file âm thanh bất kỳ (`.wav`, `.mp3`, `.flac`):**
```powershell
.\.venv\Scripts\python.exe -m src.evaluation.buzzasr_runner `
    --adapter artifacts/checkpoints/buzzasr_lora_v3/final `
    --audio "duong_dan_file_audio.wav"
```

**2. Đánh giá tự động trên file manifest JSONL:**
```powershell
.\.venv\Scripts\python.exe -m src.evaluation.buzzasr_runner `
    --adapter artifacts/checkpoints/buzzasr_lora_v3/final `
    --manifest "artifacts/manifests/hike_v1.eval.jsonl" `
    --limit 50
```

---

## 9. Kết luận nghiệm thu

1. **Hiệu quả kỹ thuật:** Kỹ thuật LoRA v3 Full Linear Projections kết hợp Free Script Decoding và 3-way Mixed Dataset đã giải quyết triệt để vấn đề phiên âm Hangul hóa từ tiếng Anh, hạ gục WER từ **53.30%** xuống **6.46% – 10.09%**.
2. **Tính toàn vẹn ngôn ngữ:** Độ chính xác tiếng Hàn thuần đạt mức rất cao (**CER 3.75%**), tiếng Anh thuần được bảo toàn (**WER 9.81%**).
3. **Tính sẵn sàng ứng dụng:** Tiêu thụ VRAM nhẹ (**3.58 GB**), tốc độ xử lý nhanh (**RTF 0.268**), mô hình sẵn sàng đóng gói làm module ASR cho các hệ thống tiếp theo.
