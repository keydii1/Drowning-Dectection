# Core Definitions & Standards in Drowning Detection

Tài liệu này chuẩn hóa các khái niệm khoa học nền tảng nhằm phục vụ phương pháp luận (Methodology) và đánh giá định lượng (Evaluation) của bài báo và khóa luận.

---

## 1. Định nghĩa Tình trạng Đuối nước (Drowning Event)

Theo định nghĩa y khoa của **Tổ chức Y tế Thế giới (WHO)** và y văn cấp cứu:
> *"Drowning is the process of experiencing respiratory impairment from submersion/immersion in liquid."*

Trong thị giác máy tính và giám sát hồ bơi:
- **Active Drowning (Đuối nước chủ động / vùng vẫy):** Nạn nhân có hành vi phản xạ không chủ ý (Instinctive Drowning Response):
  - Đầu ngửa ra sau, miệng ở sát hoặc nhấp nhô ngang mặt nước.
  - Hai tay dang ngang đập nước liên tục theo phương thẳng đứng (không thể vẫy tay cầu cứu có chủ đích).
  - Thân người ở tư thế thẳng đứng trong nước, hầu như không có lực đẩy về phía trước.
  - Thời gian vùng vẫy thường ngắn (20–60 giây) trước khi chìm.
- **Passive Drowning (Đuối nước thụ động / bất động):** Nạn nhân chìm xuống đáy hoặc nổi bất động dưới mặt nước do mất ý thức, chấn thương hoặc kiệt sức đột ngột.

---

## 2. Định nghĩa Báo động giả (False Alarm)

Nghiên cứu phân biệt rõ **2 cấp độ báo động giả**:

### 2.1. Frame-Level False Positive (FP cấp khung hình)
- Là trường hợp mô hình dự đoán nhãn `drowning` trên một khung hình riêng lẻ (hoặc bounding box của một người tại một thời điểm $t$) trong khi nhãn thực tế (Ground Truth) là `normal` (đang bơi, lặn, nổi, hoặc nước bắn).
- **Hạn chế của Frame-level metric:** Chuyển động nước dữ dội hoặc một pha ngụp đầu ngẫu nhiên có thể tạo ra 1–2 frame FP chớp nhoáng, nhưng hệ thống giám sát thực tế chưa phát còi báo động. Đánh giá thuần túy theo frame sẽ không phản ánh đúng trải nghiệm thực tế.

### 2.2. Event-Level False Alarm (FA cấp sự kiện - Quan trọng nhất)
- Một **Sự kiện Báo động giả** xảy ra khi hệ thống đưa ra cảnh báo khẩn cấp (Trigger Alert) gửi đến nhân viên cứu hộ/người dùng cho một người **không hề bị đuối nước**.
- Sự kiện này được kích hoạt khi trạng thái `drowning` duy trì hoặc tích lũy vượt qua ngưỡng thời gian $\tau_{alert}$ (ví dụ: $\tau_{alert} \ge 3.0\text{s}$ hoặc $N$ frame liên tục/chiếm tỉ lệ lớn trong sliding window).
- **Tác hại:** Gây ra hiện tượng **Alert Fatigue (bão hòa báo động)**, khiến người cứu hộ xem nhẹ cảnh báo hoặc tắt hệ thống.

---

## 3. Độ trễ phát hiện (Detection Latency)

$$\Delta t_{detect} = t_{alert} - t_{onset}$$

Trong đó:
- $t_{onset}$: Thời điểm bắt đầu xuất hiện hành vi đuối nước thực tế (dựa trên ground truth video).
- $t_{alert}$: Thời điểm hệ thống kích hoạt cảnh báo `🚨 ALERT`.
- Yêu cầu thực tế: $\Delta t_{detect}$ cần đủ nhỏ (thường $< 5\text{s}$ đến $< 10\text{s}$) để cứu hộ kịp thời, nhưng không được quá nhỏ đến mức kích hoạt sai do các biến động thoáng qua (transient noise).

---

## 4. Tiêu chuẩn Quốc tế Tham chiếu: ASTM F3698-24

Tiêu chuẩn quốc tế ban hành tháng 5/2024:
> **ASTM F3698-24:** *Standard Specification for Computer Vision Drowning Detection Systems in Residential Pools.*

Các yêu cầu kỹ thuật then chốt:
1. **Thời gian phát cảnh báo (Alert Window):** Phải phát tín hiệu cảnh báo trong vòng tối đa **30 giây** kể từ khi xảy ra đuối nước.
2. **Cảnh báo suy giảm tầm nhìn (Visibility Warning):** Hệ thống phải cảnh báo người dùng khi điều kiện môi trường (ánh sáng yếu, nước đục, lóa sáng) làm giảm độ tin cậy của thuật toán.
3. **Giám sát hoạt động (Active Protection):** Hệ thống phải duy trì khả năng phát hiện ngay cả khi hồ bơi đang có nhiều người hoạt động sôi nổi (swimming, splashing) mà không bị tê liệt bởi báo động giả.
