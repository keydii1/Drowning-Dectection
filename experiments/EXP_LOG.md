# Experiment Tracking Log (Nhật ký Thực nghiệm)

Tài liệu này lưu lại toàn bộ các thử nghiệm theo quy chuẩn khoa học (Reproducibility) đã nêu trong `plan.md`.

---

## 1. Danh sách Thử nghiệm Dự kiến

| Mã EXP | Tên thử nghiệm | Mục tiêu | Model / Pipeline | Dữ liệu Test | False Alarm (%) | Recall (%) | F1 | FPS | Trạng thái |
|:---:|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **EXP-001** | Baseline Detection | Đánh giá frame-level detection thuần túy | YOLO11n / YOLOv8n | Test Set chuẩn | - | - | - | - | ⏳ Chưa chạy |
| **EXP-002** | Baseline + Hard Negative | Đánh giá tỉ lệ báo động giả của baseline | YOLO11n / YOLOv8n | Hard Negative Set | - | - | - | - | ⏳ Chưa chạy |
| **EXP-003** | YOLO + Tracking | Giữ ID liên tục & lọc đối tượng nhiễu | YOLO11 + ByteTrack | Test Set + Hard Neg | - | - | - | - | ⏳ Chưa chạy |
| **EXP-004** | YOLO + Tracking + Persistence Rule | Lọc FP bằng ngưỡng thời gian N frames ($N \in \{5, 10, 15, 30\}$) | YOLO + ByteTrack + Rule | Hard Negative Set | - | - | - | - | ⏳ Chưa chạy |
| **EXP-005** | Proposed Spatio-Temporal Method | Mô hình đề xuất (Confidence + BBox kinematics + Temporal window) | Proposed Architecture | Toàn bộ Test Sets | - | - | - | - | ⏳ Chưa chạy |
| **EXP-006** | Cross-Scene Generalization | Kiểm tra mô hình trên hồ bơi mới chưa xuất hiện ở Train | Proposed Model | Unseen Pool Videos | - | - | - | - | ⏳ Chưa chạy |
| **EXP-007** | Real-Time Latency Benchmark | Đo lường FPS, CPU/GPU latency trên video 1080p | Final Framework | Benchmark Video | - | - | - | - | ⏳ Chưa chạy |

---

### EXP-000: Pipeline Sanity & Environment Verification
- **Ngày chạy:** 2026-09-26
- **Môi trường & Phần cứng:** Apple Silicon Mac (ARM64), Python 3.13.11 in `.venv`, PyTorch 2.14, Ultralytics 8.4 (YOLO11n), Supervision 0.30 (ByteTrack)
- **Tập dữ liệu:** `datasets/raw/demo_pool_sample.mp4` (150 frames, 640x480, 30 fps)
- **Kết quả:** Xử lý thành công 150/150 frames, xuất video có chú thích tại `results/videos/output_annotated.mp4`.
- **Trạng thái:** ✅ PASSED. Toàn bộ luồng kết nối YOLO -> ByteTrack -> Temporal Filter -> Video Visualizer hoạt động trơn tru.

### EXP-001: Baseline Detection
- **Ngày chạy:** Chưa
- **Môi trường & Phần cứng:** 
- **Hyperparameters:** `imgsz=640`, `batch=16`, `epochs=100`, `optimizer=auto`, `seed=42`
- **Kết quả:**
- **Nhận xét & Phân tích lỗi (Failure Modes):**
