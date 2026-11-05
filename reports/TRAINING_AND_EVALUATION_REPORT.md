# Báo cáo huấn luyện và đánh giá Korean-English ASR

## 1. Phạm vi báo cáo

Báo cáo này tổng hợp cấu hình mô hình, dữ liệu huấn luyện, kết quả benchmark và các log hiện có trong repository. Mục tiêu là tạo một gói bàn giao có thể kiểm tra, không suy diễn các log hoặc checkpoint không còn lưu trên máy.

## 2. Mô hình

- Model nền: `BuzzASR/korean`, thuộc họ Whisper Large.
- Phương pháp fine-tune: LoRA.
- LoRA rank: 64.
- LoRA alpha: 128.
- LoRA dropout: 0,05.
- Target modules: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`.
- Số tham số trainable được báo cáo: 115.343.360, tương đương khoảng 6,95% mô hình.
- Precision: FP16.
- Effective batch size: 16, gồm batch size 4 và gradient accumulation 4.
- Scheduler: cosine.
- Tiêu chí chọn checkpoint: WER thấp nhất trên development set.

## 3. Dữ liệu huấn luyện

Phiên bản V5 sử dụng bốn nguồn dữ liệu:

| Nguồn | Số mẫu | Xác suất lấy mẫu |
|---|---:|---:|
| Zeroth Korean | 19.651 | 40% |
| LibriSpeech English | 28.539 | 25% |
| HiKE real code-switching | 800 | 15% |
| Synthetic code-switching | 5.605 | 20% |
| Tổng | 54.595 | 100% |

Tổng thời lượng được báo cáo là khoảng 171,87 giờ. Dữ liệu synthetic kết hợp các đoạn Hàn-Anh theo hai hướng. Ngoài ra pipeline sử dụng 592 cặp từ mượn với xác suất thay thế mặc định 0,75.

## 4. Lịch sử huấn luyện có log

Các log console thô hiện lưu trong archive cũ mô tả lần chạy LoRA V3:

- `training_initial_step0_to_100.log`: giai đoạn khởi động từ step 0 đến 100.
- `training_main_step100_to_2457.log`: giai đoạn chính từ step 100 đến 2.457.
- `training_metrics_history.json`: 145 bản ghi metric từ Hugging Face Trainer.
- `run_summary.json`: cấu hình tóm tắt của lần chạy.

Cấu hình V3 trong `run_summary.json`:

| Thông số | Giá trị |
|---|---:|
| Max steps | 2.457 |
| Effective batch | 16 |
| Korean sampling | 50% |
| English sampling | 35% |
| HiKE sampling | 15% |
| Loanword probability | 0,30 |
| Random seed | 42 |

Log console thô riêng cho V4 và V5 không còn trên máy tại thời điểm đóng gói. Thông tin V5 còn lại gồm bảng lịch sử metric trong báo cáo kỹ thuật và file benchmark chi tiết từng mẫu. Gói bàn giao không tạo log giả để thay thế dữ liệu bị thiếu.

Theo báo cáo V5, quá trình full-dataset chạy 3.413 bước. Checkpoint tốt nhất được chọn tại step 400 với eval WER 3,88% và eval CER 2,32%. Giá trị tại step 3.413 là eval WER 4,98% và eval CER 3,65%. Cơ chế `load_best_model_at_end=True` chọn checkpoint có WER thấp nhất thay vì mặc định sử dụng trạng thái ở bước cuối.

## 5. Kết quả benchmark V5

Benchmark gồm 1.267 mẫu, tổng thời lượng 2,839 giờ:

| Tập đánh giá | Số mẫu | WER | CER | RTF |
|---|---:|---:|---:|---:|
| HiKE Code-Switching | 321 | 4,95% | 2,68% | 0,285 |
| Zeroth Korean Clean | 446 | 4,10% | 1,62% | [không tách riêng trong summary] |
| LibriSpeech English Clean | 500 | 2,94% | 0,99% | [không tách riêng trong summary] |
| Trung bình theo số mẫu | 1.267 | 3,86% | 1,64% | 0,383 |

Peak VRAM khi benchmark là khoảng 3,58 GB trên NVIDIA RTX A4000. Tổng thời gian xử lý là khoảng 65,6 phút.

## 6. Cách hiểu metric

Các con số chính thức được tính sau khi chuẩn hóa và căn chỉnh cặp reference-prediction bằng `src/evaluation/text_normalizer.py`. Benchmark lấy trung bình WER/CER của từng utterance, sau đó tổng hợp theo số mẫu. Vì vậy đây là normalized/aligned utterance-average WER, không phải raw corpus WER.

Tính lại trực tiếp từ prediction đã lưu cho kết quả tham khảo:

| Cách tính | WER tổng thể |
|---|---:|
| Chỉ số được repository công bố | 3,86% |
| Corpus WER trên text đã căn chỉnh | 3,30% |
| Corpus WER với chuẩn hóa cơ bản, không căn chỉnh chuyên biệt | 5,57% |

Các cách tính đều có giá trị khi được mô tả rõ. Không nên so sánh trực tiếp chúng với nhau hoặc với benchmark bên ngoài nếu preprocessing khác nhau.

## 7. Đánh giá tính toàn vẹn

- HiKE train và eval không trùng ID, checksum hoặc transcript.
- HiKE không có speaker metadata, do đó chưa xác nhận được speaker-disjoint.
- Zeroth và LibriSpeech không trùng speaker giữa train và clean test.
- Zeroth clean test có 412/446 transcript xuất hiện trong train dưới audio/speaker khác. Đây không phải trùng audio, nhưng làm giảm độ mới về nội dung của tập test.
- Mục tiêu ban đầu trong `configs/training_data_mix.yaml` là Korean clean WER dưới 3%; kết quả V5 được báo cáo là 4,10%, nên mục tiêu này chưa đạt.
- Cấu hình YAML ban đầu ghi tỷ lệ 60/40 và cấm dùng HiKE để train, trong khi pipeline V5 dùng 40/25/15/20. Cần đồng bộ cấu hình quản trị dữ liệu trước lần chạy tiếp theo.

## 8. Kết luận

LoRA V5 cải thiện rõ rệt so với model nền trên cả ba miền, đặc biệt với câu chuyển mã Hàn-Anh. Kết quả phù hợp cho nghiệm thu nội bộ và tiếp tục nghiên cứu. Để công bố hoặc so sánh bên ngoài, nên bổ sung raw corpus WER, một tập code-switching hoàn toàn độc lập và log console đầy đủ của chính lần chạy V5.

## 9. Thành phần gói log

Gói ZIP bàn giao gồm:

- Báo cáo này.
- Báo cáo kỹ thuật chi tiết của dự án.
- Hai log console huấn luyện V3 hiện có.
- Ba log benchmark hiện có.
- `run_summary.json`.
- `training_metrics_history.json`.
- Kết quả benchmark V4 và V5 ở dạng JSON.
- File manifest liệt kê tên file, kích thước và SHA-256 của từng thành phần.

Checkpoint LoRA, raw audio, cache mô hình và dataset không nằm trong gói log.
