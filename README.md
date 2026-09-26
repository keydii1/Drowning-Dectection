# False Alarm Reduction in Real-Time Drowning Detection Using Computer Vision

> **Vietnamese Title:** Phát hiện đuối nước thời gian thực và giảm báo động giả bằng mô hình thị giác máy tính kết hợp phân tích thông tin không gian–thời gian  
> **English Title:** A Spatio-Temporal Computer Vision Framework for False Alarm Reduction in Real-Time Drowning Detection

---

## 🎯 Giới thiệu Dự án (Project Overview)
Dự án nghiên cứu khoa học và khóa luận tốt nghiệp tập trung giải quyết bài toán:
$$\text{Drowning Detection} + \text{False Alarm Reduction}$$

Hệ thống phát hiện đuối nước bằng thị giác máy tính thường dễ gặp hiện tượng **báo động giả (false alarm / false positive)** trong các tình huống bình thường như: bơi lội (swimming), lặn (diving), nổi ngửa (floating), té nước (splashing), ánh sáng phản chiếu mặt nước (water reflection), bọt nước, và che khuất (occlusion). 

Mục tiêu cốt lõi của nghiên cứu là xây dựng pipeline kết hợp **Object Detection + Tracking + Spatio-Temporal Analysis** để giảm thiểu tối đa báo động giả trong khi vẫn duy trì độ nhạy phát hiện (recall) cao trong điều kiện thời gian thực (real-time).

---

## 📁 Cấu trúc Thư mục (Project Structure)

```text
.
├── plan.md                     # Bản kế hoạch nghiên cứu khoa học chi tiết
├── README.md                   # Tổng quan dự án và hướng dẫn sử dụng
├── requirements.txt            # Thư viện và môi trường phụ thuộc
├── configs/                    # File cấu hình (YAML)
│   └── default.yaml
├── datasets/                   # Dữ liệu nghiên cứu (tuân thủ bản quyền)
│   ├── raw/                    # Video / ảnh gốc chưa xử lý
│   ├── processed/              # Dữ liệu sau chuẩn hóa (frame rate, size)
│   ├── annotations/            # Nhãn YOLO / COCO định dạng chuẩn
│   ├── metadata.csv            # Metadata quản lý video theo kịch bản & split
│   └── dataset_survey.md       # Khảo sát các nguồn public dataset hiện có
├── notebooks/                  # Notebooks phục vụ EDA và phân tích định tính
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_baseline_evaluation.ipynb
│   └── 03_false_positive_analysis.ipynb
├── src/                        # Mã nguồn triển khai framework
│   ├── detection/              # Module phát hiện đối tượng (YOLOv8/11)
│   ├── tracking/               # Module theo dõi đa đối tượng (ByteTrack/BoT-SORT)
│   ├── temporal/               # Module phân tích chuỗi thời gian & bộ lọc FP
│   ├── evaluation/             # Metrics đánh giá (Frame-level & Event-level)
│   └── visualization/          # Render video overlay cảnh báo
├── experiments/                # Quản lý cấu hình & kết quả các lần thử nghiệm
│   └── EXP_LOG.md              # Bảng ghi nhật ký các thực nghiệm (EXP-001, ...)
├── results/                    # Kết quả xuất ra (metrics, hình ảnh, video demo)
│   ├── metrics/
│   ├── figures/
│   ├── confusion_matrix/
│   └── videos/
├── paper/                      # Tài liệu phục vụ viết bài báo khoa học
│   ├── literature_matrix.md    # Ma trận tổng quan các bài báo (20–30 papers)
│   ├── research_gap.md         # Phân tích khoảng trống nghiên cứu & giả thuyết
│   ├── definitions.md          # Định nghĩa khoa học về Drowning, False Alarm, Latency
│   └── paper_draft.md          # Bản thảo bài báo
└── thesis/                     # Đề cương & nội dung khóa luận tốt nghiệp
    ├── chapter_01_intro.md
    ├── chapter_02_related_work.md
    ├── chapter_03_methodology.md
    ├── chapter_04_experiments.md
    └── chapter_05_conclusion.md
```

---

## 🚦 Tiến độ Hiện tại (Current Status)

- [x] **Xác định đề tài & mục tiêu nghiên cứu**
- [x] **Xây dựng khung dự án chuẩn nghiên cứu**
- [ ] **Phase 1: Literature Review & Literature Matrix (Đang thực hiện - 20–30 papers)**
- [ ] **Phase 2: Dataset Discovery & Hard-Negative Collection**
- [ ] **Phase 3: Data Annotation & Split theo Video/Scene**
- [ ] **Phase 4: Baseline Implementation (YOLO)**
- [ ] **Phase 5: Phân tích & Phân loại False Positives (FP-01 -> FP-10)**
- [ ] **Phase 6: Object Tracking (ByteTrack / BoT-SORT)**
- [ ] **Phase 7 & 8: Spatio-Temporal Decision Modeling**
- [ ] **Phase 9: Proposed Framework & Ablation Study**
- [ ] **Phase 10: Prototype Video / Real-time Alert**
- [ ] **Phase 11 & 12: Hoàn thiện Paper Draft & Khóa luận**
