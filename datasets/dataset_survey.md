# Dataset Survey — Drowning & Swimming Behaviors

Tài liệu này tổng hợp các bộ dữ liệu công khai (Public Datasets) **đã được kiểm tra đường dẫn thực tế (verified & active)** và chiến lược xây dựng tập dữ liệu cho dự án.

---

## 1. Danh sách Datasets Thực tế & Đang Hoạt động

### 🌐 Nhóm 1: Roboflow Universe (Khuyên dùng nhất — Có sẵn nhãn YOLO)
Roboflow Universe cho phép xem trực tiếp ảnh, bounding box trên web và xuất dữ liệu định dạng **YOLOv8 / YOLO11** chỉ với 1 click:

1. **[DrowningDetectionTracking](https://universe.roboflow.com/drowningdetectiontracking/drowningdetectiontracking)**
   - **Quy mô:** ~9,530 ảnh.
   - **Các lớp (Classes):** `drowning`, `swimming`, `out of water`.
   - **Tình trạng:** Link hoạt động tốt, có phân chia Train/Val/Test sẵn.
   - **Ưu điểm:** Có nhãn bounding box chuẩn cho cả bơi và đuối nước.

2. **[Swimming and Drowning Detection](https://universe.roboflow.com/university-g3h71/swimming-and-drowning-detection)**
   - **Quy mô:** ~7,300 ảnh.
   - **Các lớp:** `drowning`, `swimming`.
   - **Ưu điểm:** Đa dạng góc nhìn, rất thích hợp huấn luyện mô hình phân biệt 2 trạng thái cơ bản.

3. **[Drowning Detection Project (wqiom)](https://universe.roboflow.com/object-detection-model/drowning-detection-wqiom)**
   - **Quy mô:** Gần 10,000 ảnh.
   - **Các lớp:** `drowning`, `person out of water`, `swimming`.

4. **[Drowning Datection (UAEU)](https://universe.roboflow.com/uaeu/drowning-datection)**
   - **Quy mô:** ~6,130 ảnh.
   - **Các lớp:** `drowning`, `swimming`.

---

### 📊 Nhóm 2: Kaggle Datasets (Cần tài khoản Kaggle để tải)
1. **[Swimming and Drowning DataSet](https://www.kaggle.com/datasets/alanoud/swimming-and-drowning)**
   - Tập dữ liệu phân loại hình ảnh người bơi và người có dấu hiệu đuối nước.
2. **[Kaggle Search: Drowning Detection](https://www.kaggle.com/search?q=drowning+detection)**
   - Trang tìm kiếm trực tiếp các bộ dữ liệu mới nhất được cập nhật trên Kaggle.

---

### 💻 Nhóm 3: Mã nguồn & Dữ liệu Video Thực tế trên GitHub
Các repository này cung cấp mã nguồn huấn luyện YOLO cùng video test thực tế:

1. **[Hasibwajid / Automated-Drowning-Detection-YOLOV8](https://github.com/Hasibwajid/Automated-Drowning-Detection-YOLOV8)**
   - Phát hiện đuối nước bằng YOLOv8 trong môi trường nước thực tế.
2. **[randhana / Drowning-Detection-](https://github.com/randhana/Drowning-Detection-)**
   - Phân tích chuyển động và phát hiện người trong nước bằng YOLO + tracking.
3. **[zseng0912 / Drowning-Detection-System](https://github.com/zseng0912/Drowning-Detection-System)**
   - Hệ thống cảnh báo thời gian thực kết hợp xử lý ảnh dưới nước và YOLO.
4. **[FranklineMisango / Drowning_Detection](https://github.com/FranklineMisango/Drowning_Detection)**
   - Sử dụng YOLOv7 & YOLOv8 để phân loại trạng thái đuối nước qua video.
5. **[Koushik0901 / AI-Lifeguard-for-Active-Drowning-Detection](https://github.com/Koushik0901/AI-Lifeguard-for-Active-Drowning-Detection)**
   - Phát hiện đuối nước chủ động (Active Drowning) thời gian thực sử dụng YOLOv11.
6. **[Reema1234ag / Drowning-Risk-Analysis](https://github.com/Reema1234ag/Drowning-Risk-Analysis)**
   - Phân tích chuyển động tâm bounding box theo thời gian để đánh giá nguy cơ đuối nước.

---

## 2. Hướng dẫn Tải Dataset Nhanh từ Roboflow về Máy

Để tải tập dữ liệu về thư mục `datasets/raw/` của dự án:
1. Truy cập vào link Roboflow: [DrowningDetectionTracking](https://universe.roboflow.com/drowningdetectiontracking/drowningdetectiontracking)
2. Bấm nút **"Download Dataset"** ở góc phải trên.
3. Chọn định dạng: **YOLOv8** (tương thích hoàn toàn với YOLO11).
4. Chọn **"Show download code"** -> Sao chép đoạn mã `curl` hoặc `python` để tải tự động về thư mục dự án.

---

## 3. Chiến lược Xây dựng Dataset của Đề tài

Theo quy định trong **Mục 10 & 12 của plan.md**:
1. **Tuyệt đối không gộp chung ảnh rồi chia train/test ngẫu nhiên (No Random Frame Split)** nhằm tránh data leakage.
2. **Cấu trúc dữ liệu hỗn hợp (Hybrid Data):**
   - **Tập Train/Val:** Sử dụng kết hợp Public Datasets (Roboflow) + Video thu thập tự nhiên.
   - **Tập Hard Negative Test Set (Độc lập):** Bộ video riêng biệt chứa các hành vi gây nhiễu:
     - FP-01: Bơi sải, bơi bướm (chuyển động tay mạnh).
     - FP-02: Lặn nín thở (người chìm dưới nước nhưng có chủ đích).
     - FP-03: Thả nổi ngửa (bất động trên mặt nước).
     - FP-04: Té nước, quẫy bọt (splash, strong water disturbance).
     - FP-05: Ánh sáng mặt trời phản chiếu lóa mặt hồ (water reflection/glare).
     - FP-06: Nhóm nhiều người bơi tụ tập (multiple people, occlusion).

---

## 4. Quy chuẩn Metadata cho Video (`metadata.csv`)
Tất cả video/tập dữ liệu đưa vào dự án phải được lập chỉ mục trong file [metadata.csv](file:///Users/hohoangson/Documents/SienceResearch/datasets/metadata.csv).
