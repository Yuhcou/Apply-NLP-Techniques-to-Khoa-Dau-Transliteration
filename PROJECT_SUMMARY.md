# Tổng quan Dự án: Applying NLP Techniques to Khoa Dau Transliteration

Dự án tập trung nghiên cứu và thực hiện chuyển tự (transliteration) giữa chữ Khoa Đẩu và chữ Quốc ngữ sử dụng các kỹ thuật Xử lý ngôn ngữ tự nhiên (NLP).

## 1. Module: Quốc ngữ -> Khoa Đẩu (Hoàn thành)
Đây là bài toán chuyển tự ánh xạ 1-1 có tính cục bộ cao. Hệ thống đã triển khai và thực nghiệm so sánh các phương pháp trên tập dữ liệu kiểm thử ~200,000 âm tiết:

| Phương pháp | Accuracy | Latency (ms) | Parameters | Dung lượng | Ghi chú |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Rule-based** | **100.00%** | ~1140ms | N/A | 5 KB | Phương pháp dựa trên luật, ổn định trên CPU. |
| **Dictionary (GPU)** | **100.00%** | **~330ms** | N/A | 315 KB | Tối ưu hóa song song hóa trên GPU, độ trễ thấp nhất. |
| **CNN-Shallow-Big** | **100.00%** | **~820ms** | **3,326** | **16 KB** | Cấu hình đề xuất (Độ chính xác tuyệt đối / Tài nguyên tối thiểu). |
| **CNN-Shallow-Small**| 75.27% | ~830ms | 1,202 | 8 KB | Mô hình thực nghiệm (Dưới ngưỡng hội tụ). |
| **CNN-Big** | 99.93% | ~830ms | 3,342 | 17 KB | Kiến trúc CNN 3 lớp tiêu chuẩn. |
| **CNN-Small** | 66.73% | ~850ms | 1,210 | 8 KB | Mô hình thực nghiệm (Dưới ngưỡng hội tụ). |

**Phân tích kỹ thuật:**
- **Hiệu năng:** Thuật toán Dictionary kết hợp song song hóa GPU (Vectorized Lookup) cho thấy hiệu quả vượt trội về tốc độ xử lý hàng loạt.
- **Tối ưu hóa kiến trúc:** Mô hình `CNN-Shallow-Big` (2 Layers, Kernel 5) đạt độ chính xác tuyệt đối, khẳng định tầm quan trọng của vùng nhận cảm (Receptive Field) so với độ sâu của mạng trong bài toán này.
- **Tính thực tiễn:** Hệ thống đã tích hợp bộ lọc từ điển (Dictionary-based Filtering) để xử lý các thực thể không phải tiếng Việt (ngoại ngữ, số, ký hiệu), đảm bảo tính nguyên bản của văn bản đầu vào.

## 2. Module: Khoa Đẩu -> Quốc ngữ (Đang triển khai)
- **Đặc điểm:** Bài toán ánh xạ 1-nhiều (Đồng âm khác hình), yêu cầu xử lý ngữ cảnh để xác định từ vựng chính xác.
- **Giải pháp:** Kiến trúc mạng nơ-ron Sequence-to-Sequence dựa trên Transformer (Character-level).
- **Trạng thái:** Hoàn thành giai đoạn Prototype với 100,000 mẫu (CER ~ 0.21. Đang chuẩn bị huấn luyện quy mô lớn trên 4 triệu mẫu.

## 3. Nhật ký cập nhật kỹ thuật
- **02/04/2026:**
    - Thực thi thuật toán Dictionary-only tối ưu hóa GPU, đạt hiệu suất xử lý ~600,000 từ/giây.
    - Chuẩn hóa hệ thống phân cấp mô hình (Model Hierarchy): **Shallow_Big, Big, Shallow_Small, Small**.
    - Tích hợp cơ chế Dừng sớm (Early Stopping) và đánh giá tham số (Parameter Count) vào quy trình huấn luyện và kiểm định.
