# Literature Matrix — Drowning Detection & False Alarm Reduction

Bảng tổng hợp các bài báo khoa học liên quan trực tiếp đến đề tài.  
**Mục tiêu:** 20–30 bài báo (Tập trung giai đoạn 2023–2026 và các bài báo nền tảng).

---

## 1. Ma trận Phân tích Tổng hợp (Literature Matrix)

| # | Bài báo & Tác giả | Năm | Nguồn / Venue | Tập dữ liệu (Dataset) | Kiến trúc Mô hình | Có Tracking? | Có Temporal? | Phân tích False Alarm? | Hard Negative? | Metrics chính | Hạn chế chính (Limitations) |
|---|---|:---:|---|---|---|:---:|:---:|:---:|:---:|---|---|
| **01** | **YOLO11-LiB: Lightweight Model for Drowning Detection in Pools** | 2025 | MDPI Sensors | Swimming Pool Dataset (Private/Public) | YOLO11 + LGCBlock + BiFF-Net | ❌ Không | ❌ Không (Frame-level) | ⚠️ Sơ lược | ❌ Chưa có tập riêng | mAP50 (94.1%), Size (4.25MB), FPS (88) | Chưa xét chuỗi thời gian; dễ báo động giả khi có bọt nước mạnh hoặc người lặn ngụp nhanh. |
| **02** | **SS-YOLOv8: Real-time Drowning Detection with Spatial-Channel Attention** | 2025 | Eng. Letters | Water Safety & Drowning Custom Dataset | YOLOv8 + SPPF + ECA Attention | ❌ Không | ❌ Không | ⚠️ Đề cập FPS & FP | ⚠️ Có xét splash | Precision, Recall, mAP50, FPS | Vẫn thuần 2D object detection; không theo dõi được cùng một nạn nhân qua các frame. |
| **03** | **Spatio-Temporal Video Analysis for Drowning Victim Behavior Detection** | 2024 | PLOS ONE / IEEE | Custom Surveillance Video Dataset | YOLO + 3D CNN / Temporal Conv | ⚠️ SORT | ✅ Có (3D-CNN) | ✅ Có phân tích | ⚠️ Bơi & lặn | Accuracy, Sensitivity, Specificity, Latency | Chi phí tính toán cao; khó đạt tốc độ real-time cao (>30 FPS) trên phần cứng nhúng/edge. |
| **04** | **Decoupled Detection and Temporal Intent Inference for Water Safety** | 2024 | NIH / Pattern Recogn. | Multi-scene pool videos | Cascaded 2-stage: YOLOv8 + LSTM | ✅ ByteTrack | ✅ Có (LSTM trên BBox seq) | ✅ Giảm FP do bơi lội | ✅ Đầy đủ | F1-Score, Detection Latency, False Alarm Rate | Dễ mất dấu khi nạn nhân chìm sâu hoặc bị người khác che khuất (Occlusion); chưa tối ưu latency. |
| **05** | **MS-YOLO: Lightweight High-Precision Drowning Detection** | 2024 | MDPI Appl. Sci. | Roboflow & Kaggle Drowning Set | Multi-scale YOLOv8 variant | ❌ Không | ❌ Không | ❌ Chỉ tính Precision | ❌ Không | mAP50-95, GFLOPs, Inference time | Đánh giá ngẫu nhiên theo frame (Random Frame Split), có nguy cơ data leakage giữa tập train/test. |
| **06** | **AI-Assisted UAV Multimodal Vision for Open-Water Rescue** | 2026 | MDPI Drones | Thermal + RGB Beach Video Dataset | Multi-modal YOLO11 + Kalman Tracking | ✅ Kalman / DeepSORT | ⚠️ Rule-based Duration | ✅ Giảm lóa sáng & sóng biển | ✅ Glare, Waves | Precision, Recall, Miss Rate, Rescue Latency | Cần cảm biến nhiệt đắt tiền; tập trung môi trường biển mở thay vì hồ bơi gia đình/công cộng. |
| **07** | **Standard Specification for CV Drowning Detection (ASTM F3698-24)** | 2024 | ASTM International | Standard Compliance Benchmarks | Quy chuẩn công nghiệp | N/A | ✅ Bắt buộc cửa sổ 30s | ✅ Yêu cầu chống Alert Fatigue | ✅ Nước đục, bơi lội, lóa sáng | Alert Latency (<30s), Visibility Warnings | Không phải thuật toán cụ thể, là bộ tiêu chuẩn bắt buộc hệ thống thương mại/nghiên cứu phải đối chiếu. |
| **08** | **Pose Estimation Assisted Aquatic Distress Recognition** | 2023 | IEEE Access | Underwater & Overhead pool dataset | YOLOv7-Pose + Rule Classifier | ⚠️ DeepSORT | ⚠️ Động học khớp (Velocity) | ✅ Phân biệt bơi vs quẫy đạp | ✅ Swimming, Treading water | Keypoint OKS, Precision, Recall | Keypoint dễ bị gãy/mất khi nước sủi bọt mạnh hoặc cơ thể bị chìm dưới mặt nước (submerged). |

---

## 2. Các Bài báo Cần Bổ sung Tiếp theo (Checklist)

- [ ] Tìm thêm 12–15 bài báo (năm 2024–2026) trên IEEE Xplore, ScienceDirect, ACM Digital Library.
- [ ] Tập trung các bài có từ khóa:
  - `ByteTrack drowning detection`
  - `False alarm reduction computer vision surveillance`
  - `Temporal action recognition swimming pool`
  - `Event-based evaluation video surveillance`
- [ ] Điền vào ma trận và đối chiếu với các mục trong `paper/research_gap.md`.
