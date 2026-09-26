# Jupyter Notebooks for Research & Analysis

Thư mục chứa các Jupyter Notebook phục vụ quá trình khám phá dữ liệu, phân tích định tính và trực quan hóa kết quả:

- `01_dataset_eda.ipynb`: Thống kê số lượng frame, phân bố lớp, độ dài video và kiểm tra chất lượng nhãn.
- `02_baseline_evaluation.ipynb`: Chạy thử nghiệm baseline YOLO, xuất ma trận nhầm lẫn (confusion matrix).
- `03_false_positive_analysis.ipynb`: Trích xuất các khung hình bị báo động giả, phân loại theo kịch bản (FP-01 đến FP-10) và trực quan hóa lỗi.
- `04_temporal_filtering_study.ipynb`: Khảo sát tác động của kích thước cửa sổ thời gian (window size) và ngưỡng duy trì (persistence threshold) tới tỉ lệ FP.
