# Research Project Plan

# False Alarm Reduction in Real-Time Drowning Detection Using Computer Vision

> **Tên tiếng Việt:** Phát hiện đuối nước thời gian thực và giảm báo động giả bằng mô hình thị giác máy tính kết hợp phân tích thông tin không gian–thời gian
>
> **Tên tiếng Anh:** A Spatio-Temporal Computer Vision Framework for False Alarm Reduction in Real-Time Drowning Detection
>
> **Mục tiêu:** Thực hiện bài báo khoa học và sử dụng kết quả nghiên cứu làm nền tảng cho khóa luận tốt nghiệp.

---

# 1. Project Overview

## 1.1. Problem

Drowning detection bằng Computer Vision đã được nghiên cứu khá nhiều. Tuy nhiên, hệ thống phát hiện đuối nước có thể tạo ra **false alarm** trong những tình huống không phải đuối nước nhưng có đặc điểm hình ảnh hoặc chuyển động tương tự.

Các tình huống có thể gây báo động giả:

* Swimming
* Diving
* Floating
* Playing in water
* Splashing
* Water reflection
* Strong water movement
* Person partially submerged
* Occlusion
* Multiple people
* Low-light conditions
* Camera angle khó quan sát

Do đó, bài toán của project không chỉ là:

```text
Drowning Detection
```

mà là:

```text
Drowning Detection
        +
False Alarm Reduction
```

---

# 2. Research Objective

## 2.1. Main Objective

Xây dựng một hệ thống Computer Vision có khả năng:

1. Phát hiện người trong môi trường nước.
2. Xác định khả năng người đó đang gặp tình trạng đuối nước.
3. Theo dõi người theo thời gian.
4. Phân tích hành vi trong một khoảng thời gian thay vì chỉ dựa trên một frame.
5. Giảm false alarm trong các tình huống dễ gây nhầm lẫn.
6. Duy trì khả năng phát hiện drowning ở mức phù hợp.
7. Có khả năng chạy gần real-time.

---

# 3. Research Questions

## RQ1

Các mô hình Computer Vision hiện tại có tạo ra false alarm trong những tình huống nào?

## RQ2

Việc sử dụng thông tin temporal có giúp giảm false alarm so với frame-level detection hay không?

## RQ3

Việc kết hợp object detection và tracking có cải thiện khả năng xác định trạng thái drowning hay không?

## RQ4

Có thể giảm false alarm mà không làm giảm đáng kể drowning recall hay không?

## RQ5

Phương pháp đề xuất có hoạt động ổn định khi thay đổi:

* camera angle
* lighting
* number of people
* swimming environment
* water condition
* scene

hay không?

---

# 4. Research Gap

Research gap dự kiến cần kiểm chứng bằng Literature Review.

## Gap 1 — False Alarm

Nhiều nghiên cứu tập trung vào detection performance nhưng chưa phân tích đầy đủ các loại false alarm.

## Gap 2 — Hard Negative

Các tình huống như:

* swimming
* diving
* floating
* splash
* reflection
* occlusion

cần được đánh giá có hệ thống hơn.

## Gap 3 — Temporal Information

Frame-level classification có thể không đủ để phân biệt:

```text
Transient abnormal behavior
```

với:

```text
Sustained drowning behavior
```

## Gap 4 — Generalization

Cần đánh giá model trên các video/scene khác với dữ liệu training.

## Gap 5 — Event-level False Alarm

Không chỉ đánh giá false positive theo từng frame mà cần xem xét:

```text
False Alarm Event
```

và:

```text
Detection Latency
```

> **Lưu ý:** Các research gap trên chỉ là giả thuyết ban đầu. Phải xác nhận bằng Literature Review trước khi đưa vào paper chính thức.

---

# 5. Expected Contribution

Dự kiến project có thể đóng góp:

## Contribution 1

Xây dựng baseline drowning detection system.

## Contribution 2

Xây dựng tập dữ liệu/validation set tập trung vào các **hard-negative scenarios**.

## Contribution 3

Đề xuất cơ chế giảm false alarm dựa trên thông tin temporal.

## Contribution 4

Đánh giá false alarm một cách có hệ thống.

## Contribution 5

So sánh:

```text
Baseline
        vs
Baseline + Tracking
        vs
Baseline + Temporal Filtering
        vs
Proposed Method
```

> Contribution cuối cùng sẽ được xác định lại sau Literature Review và thực nghiệm.

---

# 6. Overall System Architecture

```text
                    VIDEO
                      |
                      v
              +---------------+
              | Object        |
              | Detection     |
              | YOLO          |
              +-------+-------+
                      |
                      v
              +---------------+
              | Object        |
              | Tracking      |
              | ByteTrack /   |
              | BoT-SORT      |
              +-------+-------+
                      |
                      v
              Person ID + BBox
                      |
          +-----------+-----------+
          |                       |
          v                       v
+-------------------+   +-------------------+
| Spatial Features  |   | Temporal Features |
|                   |   |                   |
| Position          |   | Movement          |
| Bounding Box      |   | Duration          |
| Posture           |   | Confidence        |
| Head Position     |   | State changes     |
+---------+---------+   +---------+---------+
          |                       |
          +-----------+-----------+
                      |
                      v
              Drowning Score
                      |
                      v
              Temporal Decision
                      |
             +--------+--------+
             |                 |
             v                 v
          NORMAL           DROWNING
                               |
                               v
                       False Alarm Filter
                               |
                               v
                           🚨 ALERT
```

---

# 7. Phase 1 — Literature Review

## Objective

Tìm hiểu:

* Drowning detection
* Computer Vision
* YOLO
* Object detection
* Object tracking
* Pose estimation
* Temporal modeling
* False alarm reduction
* Real-time detection
* Drowning datasets

## Target

Tối thiểu:

```text
20–30 papers
```

Ưu tiên:

```text
2024
2025
2026
```

Có thể sử dụng paper cũ hơn nếu có tính nền tảng.

## Search Keywords

```text
drowning detection computer vision

real-time drowning detection

drowning detection YOLO

drowning detection deep learning

false alarm drowning detection

false positive drowning detection

drowning detection temporal

drowning detection tracking

drowning detection pose estimation

swimming drowning classification

drowning detection dataset

real-world drowning detection
```

## Literature Matrix

Tạo bảng:

| Paper | Year | Dataset | Model | Detection | Tracking | Temporal | False Alarm | Hard Negative | Metrics | Limitation |
| ----- | ---: | ------- | ----- | --------- | -------- | -------- | ----------- | ------------- | ------- | ---------- |

## Deliverables

* [ ] 20–30 papers
* [ ] Literature Matrix
* [ ] Related Work
* [ ] Research Gap
* [ ] Research Questions
* [ ] Preliminary methodology

---

# 8. Phase 2 — Dataset Research

## 8.1. Public Dataset

Tìm các dataset có:

### Positive

```text
Drowning
```

### Normal

```text
Swimming
Floating
Diving
Playing
Standing
Walking in water
```

### Hard Negative

```text
Splash
Reflection
Occlusion
Low light
Multiple people
Strong water movement
Partially submerged person
```

---

# 9. Dataset Sources

Ưu tiên tìm từ:

* Kaggle
* Roboflow Universe
* GitHub
* Papers With Code
* Hugging Face
* Public research datasets
* Dataset được tác giả paper công khai

Không sử dụng dataset nếu license không cho phép mục đích nghiên cứu/phân phối phù hợp.

---

# 10. Dataset Strategy

Không phụ thuộc hoàn toàn vào một dataset.

Dự kiến:

```text
Public Drowning Dataset
          +
Public Normal/Swimming Dataset
          +
Hard Negative Dataset
          +
Self-collected videos
```

---

# 11. Self-Collected Data

Có thể tự thu thập:

```text
Swimming
Diving
Floating
Splash
Playing
Multiple people
Occlusion
Different camera angles
Different lighting
```

Không cần tạo tình huống nguy hiểm thật.

Drowning có thể sử dụng dữ liệu mô phỏng có kiểm soát hoặc public datasets phù hợp.

---

# 12. Dataset Split

Không split random theo frame.

Phải ưu tiên split theo video/scene:

```text
TRAIN
├── Video 01
├── Video 02
├── Video 03
└── ...

VALIDATION
├── Video 10
└── ...

TEST
├── Video 11
├── Video 12
└── ...
```

Mục tiêu:

> Tránh data leakage giữa train và test.

---

# 13. Dataset Metadata

Tạo:

```text
metadata.csv
```

Các field dự kiến:

```text
video_id
source
scene
camera_angle
lighting
number_of_people
water_condition
scenario
label
difficulty
split
```

Ví dụ:

```csv
video_id,source,scene,camera_angle,lighting,number_of_people,scenario,label,difficulty,split
V001,public,pool,front,day,1,swimming,normal,easy,train
V002,self,pool,side,day,2,splash,normal,hard,test
V003,public,pool,front,day,1,drowning,drowning,hard,test
```

---

# 14. Phase 3 — Data Annotation

## Annotation Type

Nếu sử dụng object detection:

```text
Bounding Box
```

Class dự kiến có thể bắt đầu đơn giản:

```text
person
```

Sau đó classification/drowning decision có thể xử lý ở stage tiếp theo.

Hoặc thử:

```text
drowning
normal
```

depending on methodology.

## Annotation Tools

Có thể sử dụng:

* CVAT
* Label Studio
* Roboflow
* LabelImg

## Annotation Rules

Phải có annotation guideline thống nhất:

```text
Person partially visible
Person underwater
Multiple people
Occluded person
Difficult visibility
```

---

# 15. Phase 4 — Baseline

## Goal

Xây dựng hệ thống đơn giản trước khi đề xuất phương pháp mới.

Baseline:

```text
Video
  ↓
YOLO
  ↓
Drowning / Normal
```

## Candidate Model

Có thể bắt đầu với:

```text
YOLO11n
YOLO11s
```

Model cụ thể sẽ được quyết định sau Literature Review.

Không cần thay đổi architecture ngay từ đầu.

---

# 16. Baseline Metrics

Đánh giá:

```text
Precision
Recall
F1-score
mAP
FPS
Inference latency
False Positive
False Negative
```

Đặc biệt:

```text
False Alarm Rate
```

---

# 17. Phase 5 — False Positive Analysis

Sau khi train baseline:

```text
Prediction
    ↓
Compare Ground Truth
    ↓
Collect False Positives
```

Phân loại:

```text
FP-01: Swimming
FP-02: Diving
FP-03: Floating
FP-04: Splash
FP-05: Reflection
FP-06: Occlusion
FP-07: Low light
FP-08: Multiple people
FP-09: Partially submerged
FP-10: Other
```

Tạo bảng:

| Scenario        | False Alarm Count | Percentage |
| --------------- | ----------------: | ---------: |
| Swimming        |                   |            |
| Diving          |                   |            |
| Floating        |                   |            |
| Splash          |                   |            |
| Reflection      |                   |            |
| Occlusion       |                   |            |
| Low Light       |                   |            |
| Multiple People |                   |            |

## Objective

Xác định:

> Model đang sai ở đâu và tại sao?

---

# 18. Phase 6 — Tracking

Thử thêm:

```text
YOLO
+
ByteTrack / BoT-SORT
```

Mục tiêu:

```text
Person #1
Person #2
Person #3
```

và duy trì identity qua nhiều frame.

Ví dụ:

```text
Frame 01 → Person #5
Frame 02 → Person #5
Frame 03 → Person #5
Frame 04 → Person #5
```

---

# 19. Phase 7 — Temporal Analysis

## Approach 1 — Temporal Persistence

Thay vì:

```text
1 frame drowning
→ ALERT
```

dùng:

```text
Drowning detected
        ↓
Persist for N frames
        ↓
ALERT
```

Test nhiều giá trị:

```text
N = 3
N = 5
N = 10
N = 15
N = 30
```

Không chọn giá trị tùy ý.

Giá trị cuối cùng phải được xác định bằng validation/experiment.

---

# 20. Approach 2 — Temporal Features

Có thể sử dụng:

```text
Confidence history
Position history
Bounding box history
Movement
Velocity
Duration
Posture
Head position
```

Ví dụ:

```text
Person #5

t1 → score = 0.81
t2 → score = 0.84
t3 → score = 0.88
t4 → score = 0.91
t5 → score = 0.94
```

so với:

```text
t1 → 0.82
t2 → 0.31
t3 → 0.78
t4 → 0.25
t5 → 0.40
```

Mục tiêu:

> Phân biệt persistent abnormal behavior với transient false detection.

---

# 21. Phase 8 — Advanced Temporal Model

Chỉ thực hiện nếu kết quả baseline + tracking + temporal filtering cho thấy cần thiết.

Các candidate:

```text
LSTM
GRU
Temporal CNN
Transformer
Temporal Transformer
```

Pipeline:

```text
YOLO
 ↓
Tracking
 ↓
Feature Sequence
 ↓
Temporal Model
 ↓
Drowning / Normal
```

Không sử dụng tất cả.

Chọn 1 phương pháp dựa trên:

* Literature Review
* Dataset
* Computational cost
* Experimental results

---

# 22. Optional — Pose Estimation

Nếu false alarm vẫn xảy ra do posture:

```text
YOLO
+
Pose Estimation
```

Có thể lấy:

```text
Head
Shoulder
Elbow
Wrist
Hip
Knee
Ankle
```

Sau đó nghiên cứu:

> Posture có giúp phân biệt drowning với swimming/diving/floating không?

Đây là extension, không bắt buộc trong phiên bản đầu.

---

# 23. Phase 9 — Proposed Method

Sau khi có baseline:

```text
Baseline
   ↓
False Positive Analysis
   ↓
Identify Main Failure Modes
   ↓
Choose Method
   ↓
Proposed Framework
```

Không quyết định methodology trước khi có evidence.

---

# 24. Ablation Study

Đây là phần rất quan trọng đối với paper.

So sánh:

```text
Experiment A
YOLO

Experiment B
YOLO + Tracking

Experiment C
YOLO + Tracking + Temporal Rule

Experiment D
YOLO + Tracking + Proposed Temporal Method
```

Bảng:

| Method          | Precision | Recall | F1 | False Alarm | FPS | Latency |
| --------------- | --------: | -----: | -: | ----------: | --: | ------: |
| YOLO            |           |        |    |             |     |         |
| YOLO + Tracking |           |        |    |             |     |         |
| + Temporal Rule |           |        |    |             |     |         |
| Proposed        |           |        |    |             |     |         |

---

# 25. Hard Negative Evaluation

Tạo một test set riêng:

```text
Hard Negative Test Set
```

Bao gồm:

```text
Swimming
Diving
Floating
Splash
Reflection
Occlusion
Low-light
Multiple people
Partially submerged
```

Không dùng test set này để train.

Mục tiêu:

> Đo khả năng chống false alarm.

---

# 26. Cross-Scene Evaluation

Train:

```text
Pool A
```

Test:

```text
Pool B
```

hoặc:

```text
Camera A → TRAIN

Camera B → TEST
```

Mục tiêu:

> Kiểm tra generalization.

---

# 27. Real-Time Evaluation

Đánh giá:

```text
FPS
Latency
GPU/CPU usage
Memory usage
Resolution
Model size
```

Mục tiêu:

```text
Real-time inference
```

Không chỉ đạt accuracy cao.

---

# 28. False Alarm Metrics

Không chỉ sử dụng mAP.

Các metric quan trọng:

## Precision

```text
TP / (TP + FP)
```

## Recall

```text
TP / (TP + FN)
```

## F1

```text
2 × Precision × Recall
-----------------------
Precision + Recall
```

## False Positive Rate

```text
FP / (FP + TN)
```

## False Alarm Rate

Cần định nghĩa rõ trong methodology.

Ví dụ:

```text
False Alarm Events / Observation Time
```

hoặc một định nghĩa event-level phù hợp với dataset.

## Detection Latency

```text
Drowning onset
      ↓
System Alert
```

Đo khoảng thời gian giữa hai thời điểm này.

---

# 29. Important Trade-off

Không tối ưu:

```text
False Alarm = 0
```

một cách độc lập.

Vì:

```text
Always NORMAL
```

cũng có:

```text
False Alarm = 0
```

nhưng:

```text
Recall = 0
```

Mục tiêu:

```text
↓ False Alarm
+
Maintain high Recall
```

---

# 30. Event-Level Evaluation

Phải quyết định rõ:

### Frame-level

```text
Frame 01 → FP
Frame 02 → FP
Frame 03 → FP
```

hay:

### Event-level

```text
01–03 seconds
     ↓
1 false alarm event
```

Khuyến nghị nghiên cứu cả hai nếu dataset cho phép.

---

# 31. Phase 10 — Prototype

## Input

```text
MP4
```

Sau đó:

```text
Video
 ↓
OpenCV
 ↓
YOLO
 ↓
Tracking
 ↓
Temporal Analysis
 ↓
Decision
```

Output:

```text
Processed Video
```

Overlay:

```text
Person ID: 12
Status: NORMAL
Drowning Score: 0.23
```

hoặc:

```text
Person ID: 12
Status: DROWNING
Drowning Score: 0.94
Duration: 3.4s

🚨 ALERT
```

---

# 32. Optional Real-Time System

Sau khi model ổn định:

```text
Camera
 ↓
RTSP
 ↓
Inference Server
 ↓
Detection
 ↓
Tracking
 ↓
Temporal Decision
 ↓
Alert
```

Alert có thể:

```text
Telegram
Web Dashboard
Email
Local Alarm
```

Đây là phần demo, không phải contribution chính.

---

# 33. Suggested Repository Structure

```text
drowning-false-alarm/
│
├── README.md
├── plan.md
├── requirements.txt
├── configs/
│
├── datasets/
│   ├── raw/
│   ├── processed/
│   ├── annotations/
│   └── metadata.csv
│
├── notebooks/
│   ├── dataset_analysis.ipynb
│   ├── baseline_analysis.ipynb
│   └── false_positive_analysis.ipynb
│
├── src/
│   ├── detection/
│   ├── tracking/
│   ├── temporal/
│   ├── evaluation/
│   └── visualization/
│
├── experiments/
│   ├── baseline/
│   ├── tracking/
│   ├── temporal/
│   └── proposed/
│
├── results/
│   ├── metrics/
│   ├── figures/
│   ├── confusion_matrix/
│   └── videos/
│
├── paper/
│   ├── literature_review.md
│   ├── methodology.md
│   ├── experiments.md
│   └── paper_draft.md
│
└── thesis/
    ├── chapter_01.md
    ├── chapter_02.md
    ├── chapter_03.md
    ├── chapter_04.md
    └── chapter_05.md
```

---

# 34. Experiment Naming

Sử dụng tên rõ ràng:

```text
EXP-001-baseline
EXP-002-tracking
EXP-003-temporal-threshold
EXP-004-temporal-model
EXP-005-proposed
EXP-006-hard-negative
EXP-007-cross-scene
EXP-008-real-time
```

Mỗi experiment lưu:

```text
Model
Dataset
Hyperparameters
Training configuration
Metrics
Checkpoint
Results
Notes
```

---

# 35. Reproducibility

Mỗi experiment phải lưu:

```text
Random seed
Dataset version
Model version
Image size
Batch size
Learning rate
Epoch
Optimizer
Hardware
Python version
Library versions
```

Mục tiêu:

> Có thể chạy lại experiment và thu được kết quả tương đương.

---

# 36. Phase 11 — Statistical Analysis

Nếu có đủ dữ liệu:

So sánh:

```text
Baseline
vs
Proposed
```

trên cùng test set.

Có thể báo cáo:

```text
Mean
Standard deviation
Confidence interval
```

nếu thực nghiệm nhiều lần.

Không chỉ báo:

```text
95%
```

mà không biết kết quả có ổn định hay không.

---

# 37. Phase 12 — Paper Structure

## 1. Abstract

```text
Problem
Method
Dataset
Results
Contribution
```

## 2. Introduction

```text
Drowning problem
↓
Need automatic detection
↓
Existing Computer Vision methods
↓
False alarm problem
↓
Research gap
↓
Research questions
↓
Contribution
```

## 3. Related Work

```text
Drowning Detection
Object Detection
Tracking
Temporal Analysis
False Alarm Reduction
```

## 4. Dataset

```text
Data Sources
Annotation
Dataset Statistics
Train/Val/Test
Hard Negative
```

## 5. Methodology

```text
Baseline
Tracking
Spatial Features
Temporal Features
Proposed Method
Decision Mechanism
```

## 6. Experiments

```text
Baseline
Ablation
Hard Negative
Cross-scene
Real-time
```

## 7. Results

```text
Metrics
Tables
Figures
Qualitative Results
Failure Cases
```

## 8. Discussion

```text
Why method works
Why method fails
Trade-offs
Limitations
Generalization
```

## 9. Conclusion

```text
What was achieved
Research contribution
Future work
```

---

# 38. Thesis Structure

## Chapter 1 — Introduction

* Motivation
* Problem statement
* Objectives
* Research questions
* Scope
* Contribution

## Chapter 2 — Background and Related Work

* Computer Vision
* Object Detection
* YOLO
* Object Tracking
* Temporal Modeling
* Drowning Detection
* False Alarm

## Chapter 3 — Methodology

* Dataset
* Preprocessing
* Baseline
* Proposed Method
* Temporal analysis
* False alarm reduction

## Chapter 4 — Experiments and Results

* Experimental setup
* Baseline results
* Ablation study
* Hard-negative evaluation
* Cross-scene evaluation
* Real-time evaluation

## Chapter 5 — Conclusion

* Results
* Contributions
* Limitations
* Future work

---

# 39. Project Milestones

## Milestone 1 — Literature Review

* [ ] 20–30 papers
* [ ] Research Matrix
* [ ] Identify existing methods
* [ ] Identify limitations
* [ ] Confirm research gap

---

## Milestone 2 — Dataset

* [ ] Find public datasets
* [ ] Check licenses
* [ ] Collect normal videos
* [ ] Collect hard-negative videos
* [ ] Define annotation guideline
* [ ] Annotate
* [ ] Create train/validation/test
* [ ] Create metadata

---

## Milestone 3 — Baseline

* [ ] Train baseline
* [ ] Evaluate
* [ ] Generate confusion matrix
* [ ] Analyze false positives
* [ ] Categorize false alarms

---

## Milestone 4 — Proposed Method

* [ ] Implement tracking
* [ ] Implement temporal persistence
* [ ] Test different thresholds
* [ ] Implement temporal features
* [ ] Evaluate candidate temporal model
* [ ] Select final proposed method

---

## Milestone 5 — Experiments

* [ ] Baseline experiment
* [ ] Tracking experiment
* [ ] Temporal experiment
* [ ] Proposed method
* [ ] Ablation study
* [ ] Hard-negative evaluation
* [ ] Cross-scene evaluation
* [ ] Real-time evaluation

---

## Milestone 6 — Paper

* [ ] Abstract
* [ ] Introduction
* [ ] Related Work
* [ ] Dataset
* [ ] Methodology
* [ ] Experiments
* [ ] Results
* [ ] Discussion
* [ ] Conclusion
* [ ] References

---

# 40. Definition of Done

Project được xem là hoàn thành khi:

* [ ] Research gap được chứng minh bằng literature.
* [ ] Dataset có nguồn gốc rõ ràng.
* [ ] Dataset có train/validation/test hợp lý.
* [ ] Có baseline reproducible.
* [ ] False alarm được phân tích theo scenario.
* [ ] Có proposed method.
* [ ] Có ablation study.
* [ ] Có hard-negative evaluation.
* [ ] Có comparison với baseline.
* [ ] Có precision/recall/F1.
* [ ] Có false alarm metric.
* [ ] Có detection latency.
* [ ] Có FPS/inference performance.
* [ ] Có failure case analysis.
* [ ] Có demo real-time hoặc near-real-time.
* [ ] Có paper draft.
* [ ] Có thesis draft.

---

# 41. Immediate Next Steps

## Không train model ngay.

### Step 1

Tìm:

```text
20–30 papers
```

về:

```text
Drowning Detection
False Alarm
Temporal Analysis
Computer Vision
YOLO
Tracking
```

### Step 2

Tạo:

```text
literature_matrix.xlsx
```

### Step 3

Xác định:

```text
Existing Methods
        ↓
Limitations
        ↓
Research Gap
```

### Step 4

Xác định chính xác:

```text
What is a drowning event?
What is a false alarm?
What is a false alarm event?
```

### Step 5

Tìm dataset.

### Step 6

Xây dựng baseline.

### Step 7

Chạy baseline và phân tích false alarm.

### Step 8

Chỉ sau đó mới quyết định:

```text
Tracking?
Temporal rule?
Pose?
LSTM?
GRU?
Transformer?
```

---

# 42. Research Philosophy

Project không hướng tới:

> "Model có accuracy cao nhất."

Mà hướng tới:

> **"Có thể giảm false alarm trong drowning detection mà vẫn duy trì khả năng phát hiện drowning ở mức phù hợp hay không?"**

Do đó mọi quyết định về model phải phục vụ research question.

```text
Research Question
        ↓
Dataset
        ↓
Baseline
        ↓
Failure Analysis
        ↓
Hypothesis
        ↓
Proposed Method
        ↓
Experiment
        ↓
Evidence
        ↓
Conclusion
```

Đây là workflow chính của toàn bộ project.

---

# 43. Current Project Status

```text
[✓] Topic selected
[✓] Main objective identified
[✓] Initial research direction identified

[ ] Literature Review
[ ] Research Matrix
[ ] Confirm Research Gap
[ ] Dataset Selection
[ ] Dataset Collection
[ ] Annotation
[ ] Baseline
[ ] False Positive Analysis
[ ] Tracking
[ ] Temporal Method
[ ] Proposed Method
[ ] Experiments
[ ] Ablation Study
[ ] Hard Negative Evaluation
[ ] Cross-Scene Evaluation
[ ] Real-Time Evaluation
[ ] Paper
[ ] Thesis
```

---

# 44. Current Priority

> **PRIORITY #1: Literature Review + Research Gap**

Không bắt đầu bằng:

```text
YOLO11
```

Không bắt đầu bằng:

```text
LSTM
```

Không bắt đầu bằng:

```text
Coding
```

Mà bắt đầu bằng:

```text
Paper
 ↓
Paper
 ↓
Paper
 ↓
Research Matrix
 ↓
Research Gap
 ↓
Methodology
 ↓
Dataset
 ↓
Baseline
 ↓
Experiments
```

**Mục tiêu của giai đoạn đầu là chứng minh rằng bài toán mà nhóm chọn thực sự còn vấn đề cần giải quyết.**

---
