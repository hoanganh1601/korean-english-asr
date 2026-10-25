# Korean–English ASR Fine-tuning

Dự án xây dựng mô hình nhận dạng giọng nói (ASR) cho **tiếng Hàn, tiếng Anh và câu nói trộn Hàn–Anh**, với trọng tâm là chuẩn hóa dữ liệu, fine-tune mô hình và đánh giá bằng WER trên tập test cố định.

## Mục tiêu

| Hạng mục | Yêu cầu |
|---|---|
| Đầu vào ASR | Audio tiếng Hàn, tiếng Anh hoặc mixed Hàn–Anh |
| Đầu ra ASR | Transcript giữ nguyên ngôn ngữ được nói |
| Chất lượng đích | Korean clean test: **WER < 3%** |
| Phần cứng inference | **< 12 GB VRAM** với batch size 1 |
| Dữ liệu | Mọi nguồn được chuyển về một schema thống nhất và được kiểm tra chất lượng |
| Đánh giá | Có baseline trước fine-tune; báo cáo WER theo từng ngôn ngữ và điều kiện âm thanh |
| Thời gian dự kiến | 10 ngày, từ 20/10/2026 đến 29/10/2026 |

> WER < 3% là tiêu chí nghiệm thu trên tập **Korean clean test** đã khóa trước khi huấn luyện. Kết quả trên dữ liệu nhiễu, hội thoại tự nhiên và code-switch được báo cáo riêng; không được dùng tập test để chọn checkpoint hoặc chỉnh siêu tham số.

## Phạm vi Audio → Text → Audio

```text
Audio đầu vào
    │
    ▼
VAD / chuẩn hóa audio
    │
    ▼
ASR Hàn–Anh ───────────────► Text + timestamp + confidence
    │
    ▼
TTS Hàn–Anh (module downstream)
    │
    ▼
Audio đầu ra
```

Repository này tập trung vào bước **Audio → Text**. Bước **Text → Audio** là một module TTS độc lập, dùng transcript từ ASR và được tích hợp sau khi ASR đạt tiêu chí. Tách hai module giúp WER phản ánh đúng chất lượng nhận dạng và tránh nhầm lỗi ASR với lỗi tổng hợp tiếng nói.

## Mô hình đề xuất

Baseline và checkpoint khởi tạo: [`openai/whisper-large-v3-turbo`](https://huggingface.co/openai/whisper-large-v3-turbo), mô hình đa ngôn ngữ 809M tham số. Đây là điểm khởi đầu phù hợp cho tiếng Hàn–Anh và nhẹ hơn `whisper-large-v3`.

- Đo zero-shot baseline trước khi fine-tune.
- Fine-tune bằng LoRA/QLoRA; ưu tiên adapter riêng thay vì cập nhật toàn bộ trọng số trong vòng 10 ngày.
- Inference FP16 trước, sau đó benchmark INT8 nếu VRAM vượt ngân sách hoặc cần giảm độ trễ.
- Không ép một language token duy nhất cho các đoạn code-switch; giữ Unicode tiếng Hàn và chữ Latin trong cùng transcript.
- Chỉ chấp nhận checkpoint khi vừa đạt chất lượng, vừa có peak VRAM dưới 12 GB ở cấu hình benchmark đã công bố.

## Nguồn dữ liệu

| Nguồn | Nội dung dùng trong dự án | Lưu ý |
|---|---|---|
| [OLKAVS](https://github.com/IIP-Sogang/olkavs-avspeech) | 1.150 giờ audio tiếng Hàn, 1.107 người nói, transcript và metadata môi trường | Dataset có điều khoản riêng trên AIHub; mã nguồn MIT. Chỉ dùng audio cho ASR, video là tùy chọn. |
| [KMSAV](https://github.com/etri/kmsav) | 83 giờ hội thoại tiếng Hàn tự nhiên từ YouTube, nhiều người nói và nhiều mức nhiễu | Dataset CC BY-NC-SA 4.0, chỉ phù hợp mục đích nghiên cứu/phi thương mại; mã nguồn MIT. URL nguồn có thể mất hiệu lực nên cần ghi log các mẫu tải lỗi. |
| [Zeroth Korean](https://github.com/goodatlas/zeroth) | Dữ liệu đọc tiếng Hàn để bổ sung miền clean và làm test tham chiếu | Bản công khai được mô tả là 51,6 giờ trên OpenSLR; audio CC BY 4.0. Repo đã archive, dùng như nguồn dữ liệu/recipe tham khảo. |
| English corpus | Bổ sung sau khi chốt miền sử dụng và giấy phép | Cần có cả English-only và mixed Hàn–Anh; không đánh giá code-switch bằng cách ghép transcript giả đơn thuần. |

Không commit audio, video, checkpoint hoặc dữ liệu có hạn chế giấy phép vào Git. Mỗi nguồn phải có manifest riêng, phiên bản/ngày tải và file ghi nhận giấy phép.

## Schema dữ liệu thống nhất

Mỗi dòng trong `manifest.jsonl` biểu diễn một utterance:

```json
{
  "id": "kmsav_train_000001",
  "audio_path": "data/processed/kmsav/train/kmsav_train_000001.wav",
  "text": "오늘 meeting은 세 시에 시작합니다",
  "text_normalized": "오늘 meeting은 세 시에 시작합니다",
  "language": "ko-en",
  "duration_sec": 4.27,
  "sample_rate": 16000,
  "speaker_id": "kmsav_spk_0042",
  "source": "kmsav",
  "split": "train",
  "license": "CC-BY-NC-SA-4.0",
  "is_clean": false
}
```

Quy ước bắt buộc:

- Audio: WAV mono, PCM 16-bit, 16 kHz; loudness/peak được kiểm tra nhưng không ghi đè bản raw.
- `text` giữ transcript gốc; `text_normalized` dùng cho huấn luyện và tính metric.
- `language` chỉ nhận `ko`, `en` hoặc `ko-en`.
- ID có namespace theo nguồn để không trùng sau khi merge.
- Chia train/dev/test theo `speaker_id`; cùng một người nói không xuất hiện ở nhiều split.
- Loại mẫu rỗng, hỏng, trùng audio, transcript lệch nghiêm trọng hoặc clip quá ngắn/dài so với ngưỡng cấu hình.
- Lưu checksum và báo cáo số mẫu bị loại để pipeline có thể tái lập.

## Chuẩn hóa transcript

Pipeline normalization phải có phiên bản và chạy giống nhau cho reference lẫn prediction trước khi tính WER:

1. Chuẩn hóa Unicode NFC và khoảng trắng.
2. Giữ nguyên Hangul và từ tiếng Anh; không dịch transcript.
3. Chuẩn hóa dấu câu, ký tự đặc biệt, số và đơn vị theo một policy cố định.
4. Chuẩn hóa cách viết tiếng Hàn ở mức có thể tái lập; không sửa thủ công tập test sau khi xem prediction.
5. Lưu cả văn bản gốc và văn bản đã chuẩn hóa để audit.

Với tiếng Hàn, báo cáo cả **WER** và **CER** vì ranh giới từ phụ thuộc quy tắc spacing. Với mixed Hàn–Anh, báo cáo thêm WER theo từng vùng ngôn ngữ nếu alignment cho phép.

## Tạo manifest Zeroth và QC độ dài

Adapter ghép từng FLAC với dòng transcript và `AUDIO_INFO`, đồng thời đối chiếu
inventory kỹ thuật đã tạo trước đó. Dữ liệu nguồn không bị sửa. Chạy từ thư mục
gốc repository:

```bash
python -m src.data.zeroth_adapter
```

Lệnh tạo bốn artifact có tính quyết định (deterministic):

- `data/manifests/zeroth.raw.jsonl`: toàn bộ utterance sau khi ghép metadata.
- `data/manifests/zeroth.accepted.jsonl`: chỉ clip dài từ 1 đến 30 giây, tính cả hai biên.
- `data/manifests/zeroth.quarantine.jsonl`: clip ngoài ngưỡng hoặc có lỗi inventory.
- `reports/data_qc/zeroth.summary.json`: thống kê QC để audit.

Adapter dừng ngay nếu thiếu/thừa audio, transcript, speaker metadata, ID bị trùng
hoặc cấu trúc thư mục không khớp. Thay ngưỡng bằng
`--min-duration-sec`/`--max-duration-sec`; không xóa mẫu quarantine.

Zeroth dùng chung nhiều câu đọc giữa các speaker. Báo cáo QC vì vậy ghi rõ tỷ lệ
text overlap giữa train/test. Split gốc vẫn hữu ích để đo tổng quát hóa sang giọng
người mới, nhưng không được diễn giải là đánh giá hoàn toàn trên câu chưa từng thấy.

## Tạo manifest LibriSpeech và QC độ dài

Ba subset English clean được giữ nguyên split chính thức và ghép với
`SPEAKERS.TXT`, transcript theo chapter và inventory kỹ thuật bằng:

```bash
python -m src.data.librispeech_adapter
```

Adapter không sửa dữ liệu nguồn. Nó dừng ngay nếu audio, transcript, speaker
metadata hoặc inventory không khớp; đồng thời kiểm tra speaker không xuất hiện ở
nhiều split. Các artifact chính:

- `data/manifests/librispeech.raw.jsonl`: toàn bộ utterance đã ghép.
- `data/manifests/librispeech.accepted.jsonl`: clip đạt ngưỡng 1–30 giây.
- `data/manifests/librispeech.quarantine.jsonl`: clip cần cách ly.
- `data/manifests/librispeech.{train,dev,test}.jsonl`: các split sau QC 1–30 giây.
- `data/manifests/librispeech.official_{dev,test}.jsonl`: benchmark chính thức đầy
  đủ, kể cả clip dài hơn 30 giây; cần chunking khi đánh giá bằng Whisper.
- `reports/data_qc/librispeech.summary.json`: thống kê và kiểm tra overlap.

Sau acoustic QC, sửa các clip train bị DC offset bằng bản sao có provenance:

```bash
python -m src.data.repair_dc_offset
```

Công cụ chỉ chọn hàng có cờ `excessive_dc_offset` và mặc định từ chối mọi split
khác `train`. Raw FLAC được giữ nguyên; bản sửa nằm trong
`data/processed/librispeech/dc_corrected/`. Manifest repaired-only dùng để chạy
lại QC, còn combined manifest thay đường dẫn đúng các mẫu đã sửa.

Sau khi chạy lại acoustic QC cho repaired-only manifest, hợp nhất kết quả và tạo
hàng đợi teacher chỉ từ các mẫu train còn SNR thấp:

```bash
python -m src.data.prepare_librispeech_transcript_qc
```

English teacher QC dùng WER làm metric chính và chỉ chạy trên queue SNR thấp:

```bash
python -m src.data.transcript_qc \
  --config configs/transcript_qc_english.yaml \
  --input data/interim/librispeech/transcript_qc_candidates.jsonl \
  --output data/interim/librispeech/transcript_qc.smoke.jsonl \
  --summary reports/data_qc/librispeech_transcript_qc.smoke.summary.json \
  --sample-size 16
```

## Acoustic QC và Silero VAD

Chạy smoke test trên 10 file trước:

```bash
python -m src.data.audio_qc \
  --limit 10 \
  --output data/interim/zeroth/audio_qc.smoke.jsonl \
  --summary reports/data_qc/zeroth_audio_qc.smoke.summary.json
```

Nếu không có `decode_error`, chạy toàn bộ corpus:

```bash
python -m src.data.audio_qc
```

Stage này chỉ ghi signal metrics, Silero VAD metrics và các cờ
`pass`/`review`/`quarantine_candidate`; không sửa hay xóa audio. Cần xem phân phối
và nghe các mẫu biên trước khi tạo manifest lọc cuối cùng.

Quyết định nghe thủ công được ghi trong
`configs/data_decisions/zeroth_audio_qc.yaml`. Áp dụng quyết định và tạo manifest sau
acoustic QC bằng:

```bash
python -m src.data.finalize_audio_qc
```

## Transcript QC bằng Whisper teacher

Chạy smoke test có lấy mẫu cố định trước khi xử lý toàn corpus:

```bash
python -m src.data.transcript_qc \
  --sample-size 16 \
  --output data/interim/zeroth/transcript_qc.smoke.jsonl \
  --summary reports/data_qc/zeroth_transcript_qc.smoke.summary.json
```

Prediction chỉ dùng để tính CER/WER và gắn cờ review; không được tự động ghi đè
transcript nguồn. Khi smoke test đạt về VRAM và kết quả, bỏ `--sample-size` để chạy
toàn bộ manifest sau acoustic QC.

Triage kết quả teacher đầu tiên trước khi loại dữ liệu:

```bash
python -m src.data.triage_transcript_qc
```

Các khác biệt cách viết số có cùng prediction ở ít nhất hai speaker được giữ lại
ở vòng đầu. Các bất đồng không liên quan số, số không có đồng thuận giữa speaker,
và script có transcript nguồn không nhất quán được đưa sang hàng đợi teacher thứ
hai; không mẫu nào bị xóa tự động.

Teacher thứ hai (`whisper-large-v3`) chỉ chạy trên hàng đợi triage và ghi kết quả
vào `transcript_qc_teacher2`, không ghi đè kết quả teacher đầu tiên.

Hợp nhất hai teacher thành quyết định tạm và hàng đợi review:

```bash
python -m src.data.reconcile_transcript_qc
```

Reconciliation không sửa transcript. Hai teacher cùng khớp reference hoặc cùng
thể hiện một cách viết số được `provisional_accept`; các trường hợp còn lại được
xếp ưu tiên review dựa trên mức đồng thuận giữa hai prediction.

Tạo clean pool bảo thủ và giữ riêng toàn bộ mẫu nghi vấn:

```bash
python -m src.data.build_clean_manifests
```

Policy nằm trong `configs/clean_manifest_policy.yaml`. Official test được giữ đủ
để so sánh benchmark gốc, đồng thời có một curated clean test riêng.

Khóa normalization policy và tạo train/dev speaker-disjoint:

```bash
python -m src.data.split_manifest
```

Danh sách speaker được lưu tại `configs/data_decisions/zeroth_speaker_split.yaml`;
mọi mẫu được cứu lại từ review holdout sau này phải theo đúng assignment này.

Chạy cổng validation và đóng băng checksum:

```bash
python -m src.data.validate_final_dataset
sha256sum -c data/manifests/zeroth_v1.sha256
```

## Thiết kế tập đánh giá

| Tập | Mục đích | Metric chính |
|---|---|---|
| `ko_clean_test` | Tiêu chí WER < 3% | WER, CER |
| `ko_noisy_test` | Độ bền với nhiễu và hội thoại tự nhiên | WER, CER |
| `en_test` | Ngăn suy giảm tiếng Anh sau fine-tune | WER |
| `ko_en_mixed_test` | Khả năng code-switch | WER tổng, WER theo ngôn ngữ |

Mỗi báo cáo phải ghi: commit, checkpoint, phiên bản manifest, normalization policy, decoding config, GPU, precision, peak VRAM, real-time factor và WER/CER. Công thức WER:

```text
WER = (substitutions + deletions + insertions) / số từ trong reference × 100%
```

## Kế hoạch 10 ngày

| Ngày | Công việc | Đầu ra |
|---|---|---|
| 1 | Chốt tiêu chí, kiểm tra giấy phép và quyền truy cập dữ liệu | Data inventory + acceptance criteria |
| 2 | Viết adapter cho OLKAVS, KMSAV và Zeroth | Manifest thô theo từng nguồn |
| 3 | Chuyển audio và transcript về schema chung | Dataset chuẩn hóa v1 |
| 4 | QC tự động, deduplicate, chia speaker-disjoint | Train/dev/test manifests đã khóa |
| 5 | Chạy zero-shot baseline trên toàn bộ test suite | Báo cáo baseline WER/CER |
| 6 | Fine-tune vòng 1 với dữ liệu clean cân bằng Hàn–Anh | Checkpoint và learning curves |
| 7 | Fine-tune vòng 2 với noisy/mixed data và hard samples | Candidate checkpoints |
| 8 | Error analysis; chỉnh data/decoding bằng dev set | Model candidate cuối |
| 9 | Benchmark chất lượng, tốc độ và VRAM | Báo cáo test chính thức |
| 10 | Đóng gói inference, tài liệu hóa và tích hợp stub TTS | Release candidate |

## Tiêu chí hoàn thành

- [ ] Có baseline zero-shot và kết quả sau fine-tune trên cùng tập test.
- [ ] Korean clean WER < 3%; có CER đi kèm.
- [ ] Không làm giảm English WER quá ngưỡng đã chốt từ baseline.
- [ ] Có kết quả riêng cho Korean noisy và Hàn–Anh mixed.
- [ ] Peak VRAM inference < 12 GB, batch size 1, trên audio benchmark cố định.
- [ ] Manifest, split, normalization và checkpoint có version để tái lập.
- [ ] Các nguồn dữ liệu được dùng đúng giấy phép và có attribution.
- [ ] Có API/CLI ASR nhận audio và trả text; TTS được tích hợp qua interface riêng.

## Cấu trúc repository dự kiến

```text
.
├── README.md
├── configs/             # Cấu hình data, training, decoding
├── data/
│   ├── manifests/       # JSONL được version hóa
│   └── README.md        # Hướng dẫn tải dữ liệu, không chứa audio
├── src/
│   ├── data/            # Adapter và normalization
│   ├── training/        # Fine-tune pipeline
│   ├── inference/       # ASR API/CLI
│   └── evaluation/      # WER/CER và benchmark VRAM
├── tests/
└── reports/             # Baseline, ablation và kết quả cuối
```

## Rủi ro cần theo dõi

- Mục tiêu WER < 3% có thể chỉ khả thi trên miền clean, hẹp và có transcript nhất quán; cần giữ riêng kết quả noisy/mixed để tránh báo cáo sai mức tổng quát hóa.
- KMSAV mang giấy phép phi thương mại; nếu sản phẩm có mục đích thương mại, phải loại nguồn này hoặc xin quyền bổ sung.
- OLKAVS yêu cầu tuân thủ điều khoản AIHub, không mặc định theo giấy phép MIT của phần mã nguồn.
- Dữ liệu code-switch thực thường ít; dữ liệu ghép/tổng hợp chỉ nên dùng để tăng cường, không thay thế test set do người nói thật tạo ra.
- Fine-tune chủ yếu tiếng Hàn có thể làm giảm chất lượng tiếng Anh; cần sampling cân bằng và regression test tiếng Anh sau mỗi checkpoint.

## Tài liệu tham khảo

- [OLKAVS: Open Large-Scale Korean Audio-Visual Speech Dataset](https://github.com/IIP-Sogang/olkavs-avspeech)
- [KMSAV: Korean Multi-speaker Spontaneous Audio-Visual Speech Recognition Dataset](https://github.com/etri/kmsav)
- [Zeroth: Kaldi-based Korean ASR](https://github.com/goodatlas/zeroth)
- [Whisper large-v3-turbo model card](https://huggingface.co/openai/whisper-large-v3-turbo)
