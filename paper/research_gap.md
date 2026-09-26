# Research Gap & Research Questions Analysis

Tài liệu này dùng để theo dõi và đối chiếu các giả thuyết nghiên cứu với các bằng chứng thực tế thu thập từ Literature Review.

---

## 1. Research Questions (RQ)

- **RQ1:** Các mô hình thị giác máy tính hiện tại thường tạo ra báo động giả (False Alarm) trong những kịch bản cụ thể nào (Hard-negative scenarios)?
- **RQ2:** Việc bổ sung thông tin chuỗi thời gian (Temporal Information) có giảm thiểu đáng kể báo động giả so với việc chỉ phân loại từng khung hình đơn lẻ (Frame-level) hay không?
- **RQ3:** Việc kết hợp theo dõi đa đối tượng (Multi-Object Tracking - MOT) với phát hiện (Detection) đóng vai trò thế nào trong việc duy trì định danh và phân tích hành vi đuối nước liên tục?
- **RQ4:** Có thể thiết kế một cơ chế lọc/phân loại thời gian giúp giảm tỉ lệ báo động giả (False Alarm Rate) mà không làm suy giảm độ nhạy phát hiện đuối nước (Drowning Recall) hay không?
- **RQ5:** Hệ thống đề xuất có duy trì được tính ổn định và khả năng chạy thời gian thực (Real-time FPS) khi thay đổi góc quay camera, ánh sáng, số lượng người và chuyển động mặt nước hay không?

---

## 2. Research Gaps & Bằng chứng Kiểm chứng (Literature Verification)

| Mã Gap | Tên khoảng trống nghiên cứu | Mô tả vấn đề hiện tại | Bằng chứng từ Literature Review | Hướng giải quyết của đề tài |
| :--- | :--- | :--- | :--- | :--- |
| **Gap 1** | Thiếu phân tích hệ thống về False Alarm | Đa số nghiên cứu chỉ báo cáo mAP, Precision, Recall chung chung; ít công trình phân loại chi tiết các nguồn gốc gây ra báo động giả. | Các bài báo 2023–2025 chỉ ra rằng bọt nước (splashing) và bóng phản chiếu (reflection) là nguyên nhân gây nhiễu hàng đầu. | Xây dựng bộ phân loại lỗi FP (FP-01 đến FP-10) và đo lường trực tiếp tỉ lệ báo động giả. |
| **Gap 2** | Đánh giá thiếu Hard Negative Scenarios | Các tập test thường chỉ gồm người bơi êm ả vs. người đuối nước, thiếu các hành vi gây nhiễu mạnh: lặn sâu, té nước, thả nổi ngửa, che khuất một phần. | Nhiều mô hình đạt >95% mAP trong điều kiện phòng thí nghiệm nhưng thất bại ở hồ bơi thực tế đông người. | Xây dựng tập thử nghiệm riêng biệt: **Hard Negative Evaluation Set**. |
| **Gap 3** | Phụ thuộc vào Frame-level Detection | Phân loại frame đơn lẻ dễ nhầm lẫn giữa *hành vi bất thường thoáng qua* (transient anomaly) và *đuối nước kéo dài* (sustained drowning). | Các nghiên cứu temporal (LSTM, 3D-CNN) đòi hỏi tính toán nặng, khó chạy real-time trên thiết bị biên. | Đề xuất cơ chế Temporal Filtering kết hợp Sliding Window / Feature History gọn nhẹ, đáp ứng real-time. |
| **Gap 4** | Đánh giá Cross-Scene & Data Leakage | Phổ biến lỗi chia tập dữ liệu ngẫu nhiên theo frame (Random Frame Split), dẫn đến việc các frame liền kề cùng nằm ở cả Train và Test. | Gây ra hiện tượng overfitting trầm trọng, kết quả mAP ảo cao nhưng sang hồ bơi khác thì giảm sút. | Bắt buộc chia Train / Val / Test theo từng **Video / Scene** độc lập. |
| **Gap 5** | Thiếu Event-level False Alarm Metrics | Chỉ đo Frame-level FPR mà không tính đến Event-level False Alarm và Alert Latency. | Tiêu chuẩn ASTM F3698-24 yêu cầu phát cảnh báo trong 30s và tránh Alert Fatigue. | Bổ sung metric: Số sự kiện báo động giả trên giờ quan sát (False Alarm Events / Hour) và Detection Latency ($\Delta t$). |
