# Data Handoff — Korean–English ASR

## Phạm vi bàn giao

Pipeline chuẩn bị và QC dữ liệu đã hoàn tất.

- Korean train: 19,651 mẫu, 45.695823 giờ.
- English train: 28,539 mẫu, 100.590880 giờ.
- Tổng train: 48,190 mẫu, 146.286702 giờ.
- HiKE code-switch test: 1,121 mẫu, evaluation-only.
- Raw audio, cache, checkpoint và intermediate data không nằm trong Git.

Repository bàn giao dữ liệu đã chuẩn hóa. Phần trainer/evaluator thuộc giai đoạn tiếp theo.

## Thiết lập môi trường

```bash
git clone <REPOSITORY_URL>
cd korean-english-asr

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
```