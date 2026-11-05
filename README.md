# Korean-English Code-Switching ASR

Dự án xây dựng hệ thống nhận dạng tiếng nói tự động (Automatic Speech Recognition - ASR) cho ba trường hợp: tiếng Hàn, tiếng Anh và câu nói chuyển mã Hàn-Anh (code-switching). Repository cung cấp pipeline chuẩn bị dữ liệu, huấn luyện LoRA, đánh giá, suy luận và giao diện kiểm thử.

## Tổng quan

Mô hình nền là `BuzzASR/korean`, thuộc họ Whisper Large. Dự án fine-tune mô hình bằng LoRA trên bốn nguồn dữ liệu:

| Nguồn | Vai trò | Số mẫu train | Xác suất lấy mẫu |
|---|---|---:|---:|
| Zeroth Korean | Tiếng Hàn đơn ngữ | 19.651 | 40% |
| LibriSpeech | Tiếng Anh đơn ngữ | 28.539 | 25% |
| HiKE | Chuyển mã Hàn-Anh thực | 800 | 15% |
| Synthetic Code-Switching | Chuyển mã tổng hợp | 5.605 | 20% |

Tổng dữ liệu huấn luyện gồm 54.595 mẫu, tương đương khoảng 171,87 giờ âm thanh. Các tỷ lệ trong bảng là xác suất lấy mẫu của mixer, không phải tỷ lệ thời lượng audio.

## Mô hình và phương pháp

Pipeline huấn luyện được triển khai tại [`src/training/train_lora_buzzasr.py`](src/training/train_lora_buzzasr.py). Cấu hình chính:

- Model nền: `BuzzASR/korean`.
- Phương pháp: Low-Rank Adaptation (LoRA).
- LoRA rank: 64.
- LoRA alpha: 128.
- LoRA dropout: 0,05.
- Target modules: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`.
- Batch size trên mỗi thiết bị: 4.
- Gradient accumulation: 4.
- Effective batch size: 16.
- Precision: FP16.
- Learning rate mặc định: `1e-4` khi train mới và `5e-5` khi tiếp tục từ adapter.
- Scheduler: cosine.
- Tiêu chí chọn checkpoint tốt nhất: WER trên tập development.

Các token ép ngôn ngữ mặc định của Whisper được tắt để decoder có thể sinh đồng thời chữ Hangul và chữ Latin. Pipeline còn sử dụng 592 cặp từ mượn Hàn-Anh để tăng khả năng giữ nguyên thuật ngữ tiếng Anh trong câu chuyển mã.

## Kết quả đánh giá

LoRA v5 được đánh giá trên 1.267 mẫu held-out, tổng thời lượng 2,839 giờ. Kết quả chi tiết nằm trong [`artifacts/benchmark_lora_v5_full_results.json`](artifacts/benchmark_lora_v5_full_results.json).

| Tập đánh giá | Số mẫu | BuzzASR Base WER | LoRA v3 WER | LoRA v4 WER | LoRA v5 WER | LoRA v5 CER |
|---|---:|---:|---:|---:|---:|---:|
| HiKE Code-Switching | 321 | 53,30% | 10,09% | 9,33% | 4,95% | 2,68% |
| Zeroth Korean Clean | 446 | 14,10% | 13,74% | 6,94% | 4,10% | 1,62% |
| LibriSpeech English Clean | 500 | 12,50% | 9,81% | 3,42% | 2,94% | 0,99% |
| Trung bình theo số mẫu | 1.267 | khoảng 25,4% | khoảng 11,1% | 6,16% | 3,86% | 1,64% |

Các giá trị trên là WER/CER sau bước chuẩn hóa và căn chỉnh song ngữ trong [`src/evaluation/text_normalizer.py`](src/evaluation/text_normalizer.py). Script benchmark lấy trung bình metric theo từng utterance và sau đó lấy trung bình theo số mẫu. Khi so sánh với nghiên cứu khác, cần bảo đảm cùng quy tắc chuẩn hóa và cùng cách tổng hợp metric.

Hiệu năng đo trên NVIDIA RTX A4000:

- Overall real-time factor (RTF): 0,383.
- HiKE RTF: 0,285.
- Peak VRAM khi suy luận: 3,58 GB.
- Thời gian xử lý 2,839 giờ audio: khoảng 65,6 phút.

## Cấu trúc repository

```text
korean-english-asr/
|-- app.py
|-- artifacts/
|   |-- benchmark_lora_v4_full_results.json
|   |-- benchmark_lora_v5_full_results.json
|   |-- loanwords_dict.json
|   `-- manifests/
|-- configs/
|-- reports/
|-- scripts/
|   `-- run_full_benchmark_v4.py
|-- src/
|   |-- data/
|   |-- evaluation/
|   |   |-- buzzasr_runner.py
|   |   `-- text_normalizer.py
|   |-- training/
|   |   |-- data_collator.py
|   |   |-- manifest_dataset.py
|   |   `-- train_lora_buzzasr.py
|   `-- ui/
`-- tests/
```

Raw audio, Hugging Face cache và checkpoint LoRA không được lưu trực tiếp trong Git. Người dùng cần tải dữ liệu và adapter riêng, sau đó thiết lập đường dẫn theo tài liệu bàn giao.

## Cài đặt

### Yêu cầu

- Python 3.10-3.12.
- Linux hoặc Windows.
- NVIDIA GPU hỗ trợ CUDA; khuyến nghị tối thiểu 8 GB VRAM cho suy luận và nhiều hơn khi huấn luyện.
- FFmpeg/ffprobe trong `PATH`.

### Thiết lập môi trường

```bash
git clone https://github.com/hoanganh1601/korean-english-asr.git
cd korean-english-asr

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
```

Trên Windows PowerShell, kích hoạt môi trường bằng:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Chuẩn bị dữ liệu

Các manifest phiên bản bàn giao nằm trong `artifacts/manifests/`. Sau khi đặt dataset ngoài Git, chạy script tạo liên kết dữ liệu:

```bash
bash scripts/setup_data_links.sh
```

Trên Windows:

```powershell
python scripts/setup_data_links_windows.py
```

Kiểm tra lại đường dẫn audio trong manifest trước khi train hoặc benchmark.

## Huấn luyện

Train mới trên toàn bộ bốn manifest:

```bash
python -m src.training.train_lora_buzzasr \
  --full-dataset \
  --output-dir artifacts/checkpoints/buzzasr_lora_v5_full
```

Tiếp tục fine-tune từ một adapter có sẵn:

```bash
python -m src.training.train_lora_buzzasr \
  --full-dataset \
  --init-adapter artifacts/checkpoints/buzzasr_lora_v4/final \
  --output-dir artifacts/checkpoints/buzzasr_lora_v5_full \
  --learning-rate 5e-5 \
  --loanwords-prob 0.75
```

Chạy pilot 500 bước để kiểm tra môi trường:

```bash
python -m src.training.train_lora_buzzasr \
  --pilot \
  --output-dir artifacts/checkpoints/buzzasr_lora_pilot
```

## Suy luận

Repository không chứa trọng số adapter LoRA. Sau khi nhận adapter và đặt tại `artifacts/checkpoints/buzzasr_lora_v5_full/final`, chạy:

```bash
python -m src.evaluation.buzzasr_runner \
  --adapter artifacts/checkpoints/buzzasr_lora_v5_full/final \
  --audio /path/to/audio.wav
```

Có thể cung cấp hotwords cho tên riêng hoặc thuật ngữ hiếm:

```bash
python -m src.evaluation.buzzasr_runner \
  --adapter artifacts/checkpoints/buzzasr_lora_v5_full/final \
  --audio /path/to/audio.wav \
  --prompt "Kubernetes, PyTorch, Docker"
```

## Benchmark

Để chạy lại benchmark trên ba tập held-out:

```bash
python scripts/run_full_benchmark_v4.py
```

Kết quả được ghi vào `artifacts/benchmark_lora_v5_full_results.json`. Việc tái lập yêu cầu đủ raw audio, manifest và adapter LoRA tương ứng.

## Web Studio

Khởi động giao diện kiểm thử:

```bash
python app.py --port 7860
```

Sau đó truy cập `http://localhost:7860`.

## Kiểm thử

```bash
python -m pytest -q
```

## Giới hạn đánh giá

- HiKE được chia thành 800 mẫu train và 321 mẫu eval. Tập eval không trùng ID, checksum hoặc transcript với train, nhưng dataset không có speaker metadata để xác nhận speaker-disjoint.
- Zeroth Korean clean test không trùng audio hoặc speaker với train, nhưng có nhiều câu tham chiếu được đọc lại bởi speaker khác.
- WER trong báo cáo sử dụng bộ chuẩn hóa/căn chỉnh riêng của dự án; không nên so sánh trực tiếp với raw WER của hệ thống khác.
- File [`configs/training_data_mix.yaml`](configs/training_data_mix.yaml) thuộc chính sách dữ liệu ban đầu. Cấu hình đang được script huấn luyện sử dụng phải được xác nhận từ các đối số CLI và `run_summary.json` của từng lần chạy.

## Tài liệu

- [`HANDOFF.md`](HANDOFF.md): hướng dẫn bàn giao dữ liệu và môi trường.
- [`reports/korean_english_asr_report.md`](reports/korean_english_asr_report.md): báo cáo kỹ thuật chi tiết.
- [`reports/TRAINING_AND_EVALUATION_REPORT.md`](reports/TRAINING_AND_EVALUATION_REPORT.md): báo cáo tóm tắt huấn luyện, đánh giá và danh mục log.

## Giấy phép và nguồn dữ liệu

- Mã nguồn dự án: repository hiện chưa kèm file giấy phép; cần bổ sung giấy phép trước khi phân phối hoặc sử dụng lại mã nguồn.
- Model nền: [`BuzzASR/korean`](https://huggingface.co/BuzzASR/korean).
- Zeroth Korean: CC BY 4.0.
- LibriSpeech: CC BY 4.0.
- HiKE: tuân theo giấy phép và điều khoản sử dụng của dataset gốc.

Người sử dụng chịu trách nhiệm kiểm tra lại giấy phép của model, dữ liệu và sản phẩm đầu ra trước khi triển khai thương mại.
