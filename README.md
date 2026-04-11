# Student Feedback

Huấn luyện mô hình đa nhiệm cho 2 bài toán:
- Sentiment classification: negative, positive
- Topic classification: lecturer, training_program, facility

Dự án hỗ trợ 3 pipeline:
- LSTM
- RNN
- Transformer

## 1. Chạy nhanh theo luồng mặc định (luôn dùng wandb)

Luồng chuẩn:
1. Cài môi trường
2. Đăng nhập wandb
3. Tải dữ liệu
4. Vẽ biểu đồ dữ liệu (tùy chọn)
5. Train pipeline theo model
6. Kiểm tra kết quả ở logs, images, outputs và dashboard wandb

Lệnh chạy nhanh:

  conda activate student-feedback
  wandb login
  python run.py --task download
  python run.py --task data-vis
  python run.py --task pipeline --model LSTM

Bạn có thể thay LSTM bằng RNN hoặc Transformer.

## 2. Yêu cầu hệ thống

- Python 3.11
- Conda hoặc Miniconda
- Git

## 3. Cài đặt môi trường

### Cách A (khuyến nghị): dùng environment.yml

Tạo môi trường:

  conda env create -f environment.yml

Kích hoạt:

  conda activate student-feedback

### Cách B: dùng requirements.txt

Tạo môi trường rỗng:

  conda create -n student-feedback python=3.11 -y

Kích hoạt:

  conda activate student-feedback

Cài thư viện:

  pip install -r requirements.txt

## 4. Cấu hình wandb (mặc định bật)

Trong config.yml, cả 3 model đều đang để:
- pipelines.LSTM.wandb.enabled: true
- pipelines.RNN.wandb.enabled: true
- pipelines.TRANSFORMER.wandb.enabled: true

Bạn nên làm 3 việc trước khi train:

1. Đăng nhập:

  wandb login

2. Mở config.yml để đặt project cho dễ quản lý (nếu cần):
- project: student-feedback
- tags: nhãn cho run

3. Chạy pipeline bình thường, hệ thống tự log metric theo epoch và tự finish run cuối quá trình.

Tên run W&B được tự sinh theo format:
- LSTM-YYYYMMDD-HHMMSS
- RNN-YYYYMMDD-HHMMSS
- TRANSFORMER-YYYYMMDD-HHMMSS

## 5. Cấu trúc dữ liệu bắt buộc

Thư mục data cần có đúng 9 file sau:
- train_sents.txt, train_sens.txt, train_tops.txt
- validation_sents.txt, validation_sens.txt, validation_tops.txt
- test_sents.txt, test_sens.txt, test_tops.txt

Nếu chưa có dữ liệu, chạy:

  python run.py --task download

Nếu muốn tải đè dữ liệu cũ:

  python run.py --task download --overwrite-data

## 6. Cách chạy từng task

Run script chính:

  python run.py --task <task_name> [--model <model_name>] [--overwrite-data]

Task hợp lệ:
- download
- data-vis
- pipeline

Model hợp lệ cho pipeline:
- LSTM
- RNN
- Transformer

Lưu ý logic tham số từ run.py:
- Nếu task là pipeline thì bắt buộc có --model
- Nếu task không phải pipeline thì không được truyền --model

Ví dụ chạy:

Tải dữ liệu:

  python run.py --task download

Vẽ biểu đồ phân bố nhãn:

  python run.py --task data-vis

Train LSTM:

  python run.py --task pipeline --model LSTM

Train RNN:

  python run.py --task pipeline --model RNN

Train Transformer:

  python run.py --task pipeline --model Transformer

## 7. Kết quả sau khi chạy

Mỗi lần chạy pipeline sẽ tạo/ghi đè các artifact theo config:

- Best checkpoint: thư mục outputs
- Classification report sentiment/topic: thư mục logs
- Confusion matrix: thư mục images theo từng model
- Runtime log tổng: logs/execution.log
- Dashboard thực nghiệm: wandb project tương ứng

Đường dẫn chi tiết nằm trong block pipelines.<MODEL>.output và pipelines.<MODEL>.early_stopping.checkpoint_path của config.yml.

## 8. Tùy chỉnh hyperparameter

Tất cả tham số nằm trong config.yml theo từng model:
- model: kiến trúc mạng
- training: epochs, batch_size, learning_rate, scheduler, grad clip
- loss: label smoothing và loss weight
- early_stopping: patience, delta, checkpoint_path
- output: report path và confusion matrix dir
- wandb: project, tags, enabled
