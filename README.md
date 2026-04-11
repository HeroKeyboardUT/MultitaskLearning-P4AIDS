# Student Feedback - Hướng dẫn cài đặt và sử dụng

## 1. Tổng quan
Dự án này huấn luyện và đánh giá mô hình đa nhiệm cho 2 bài toán:
- Phân loại cảm xúc: negative, positive
- Phân loại chủ đề: lecturer, training_program, facility

Hệ thống hỗ trợ 3 pipeline:
- LSTM
- RNN
- Transformer

Các thành phần chính:
- run.py: Điểm vào chính để chạy task
- config.yml: Cấu hình dự án
- dataloader/: Xử lý vocab và DataLoader
- model/: Định nghĩa kiến trúc mô hình
- pipeline/: Logic train, eval, save cho từng mô hình
- utils/: Hàm tiện ích (xử lý dữ liệu, early stopping, vẽ biểu đồ)

## 2. Yêu cầu hệ thống
- Python 3.11
- Conda hoặc Miniconda
- Git

## 3. Cài đặt môi trường
Bạn có 2 cách:

### Cách A - Dùng environment.yml (khuyến nghị)
Bước 1: Tạo môi trường
  conda env create -f environment.yml

Bước 2: Kích hoạt môi trường
  conda activate student-feedback

### Cách B - Dùng requirements.txt
Bước 1: Tạo môi trường rỗng
  conda create -n student-feedback python=3.11 -y

Bước 2: Kích hoạt
  conda activate student-feedback

Bước 3: Cài thư viện
  pip install -r requirements.txt

## 4. Lưu ý về CUDA trong environment.yml
File environment.yml hiện có cả:
- pytorch-cuda=12.1
- cpuonly

Hai tùy chọn này không nên dùng đồng thời. Nếu gặp lỗi giải dependency:
- Nếu dùng GPU NVIDIA: giữ pytorch-cuda, bỏ cpuonly
- Nếu dùng CPU: giữ cpuonly, bỏ pytorch-cuda

Sau đó tạo lại môi trường.

## 5. Cấu trúc thư mục dữ liệu
Sau khi tải dữ liệu, thư mục data cần có:
- train_sents.txt, train_sens.txt, train_tops.txt
- validation_sents.txt, validation_sens.txt, validation_tops.txt
- test_sents.txt, test_sens.txt, test_tops.txt

## 6. Các task hỗ trợ
run.py hỗ trợ 3 task:
- download: Tải dữ liệu từ Google Drive theo file_id trong config
- data-vis: Vẽ và lưu biểu đồ phân bố nhãn
- pipeline: Train, eval, lưu kết quả theo model được chọn

## 7. Cách chạy nhanh
### 7.1 Tải dữ liệu
  python run.py --task download

Tải lại dữ liệu (ghi đè file cũ)
  python run.py --task download --overwrite-data

### 7.2 Vẽ biểu đồ phân bố
  python run.py --task data-vis

### 7.3 Chạy pipeline theo model
LSTM:
  python run.py --task pipeline --model LSTM

RNN:
  python run.py --task pipeline --model RNN

Transformer:
  python run.py --task pipeline --model Transformer

## 8. Kết quả đầu ra
Mỗi model sẽ lưu:
- Checkpoint tốt nhất trong outputs/
- Báo cáo classification trong logs/
- Confusion matrix trong images/

Đường dẫn cụ thể được cấu hình trong block pipelines của config.yml.

## 9. Bật W&B để theo dõi thực nghiệm
Dự án đã tích hợp W&B cho cả 3 pipeline.

Bước 1: Đăng nhập W&B
  wandb login

Bước 2: Mở config.yml, tìm block pipelines.<MODEL>.wandb và bật:
- enabled: true
- project: tên project W&B
- run_name: có thể để null hoặc đặt tên run
- tags: danh sách nhãn

Bước 3: Chạy pipeline bình thường.

Mỗi epoch sẽ được log các chỉ số train/val. Kết thúc run sẽ được finish tự động.

## 10. Tùy chỉnh thông số
Tất cả hyperparameter nằm trong config.yml theo từng model:
- model: Kiến trúc
- training: epoch, batch_size, learning rate, scheduler
- loss: label smoothing, loss weight
- early_stopping: patience, delta, đường dẫn checkpoint
- output: đường dẫn report và ảnh
- wandb: tracking thực nghiệm

## 11. Khắc phục lỗi thường gặp
1. Lỗi import torch, sklearn, datasets, wandb trong IDE
- Kiểm tra đã chọn đúng Python interpreter của môi trường conda.
- Chạy lại lệnh cài đặt thư viện.

2. Lỗi tạo env do xung đột CUDA và CPU
- Chọn một trong hai hướng GPU hoặc CPU như mục 4.

3. Lỗi không tìm thấy dữ liệu
- Chạy lại task download.
- Kiểm tra tên file trong data/ có đúng pattern trong config.yml.

4. Không thấy log W&B
- Kiểm tra wandb.enabled đã là true.
- Kiểm tra đã đăng nhập wandb login.

## 12. Luồng chạy đề xuất
1. Tạo môi trường và cài đặt thư viện
2. Tải dữ liệu
3. Vẽ biểu đồ phân bố nhãn
4. Chạy pipeline model cần thử
5. Xem report và confusion matrix
6. Điều chỉnh config và lặp lại
