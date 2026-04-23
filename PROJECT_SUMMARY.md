# Tổng quan Dự án: Applying NLP Techniques to Khoa Dau Transliteration

Dự án tập trung nghiên cứu và thực hiện chuyển tự (transliteration) giữa chữ Khoa Đẩu và chữ Quốc ngữ sử dụng các kỹ thuật Xử lý ngôn ngữ tự nhiên (NLP).

## 1. Module: Quốc ngữ -> Khoa Đẩu (Hoàn thành)
Đây là bài toán chuyển tự ánh xạ 1-1 có tính cục bộ cao. Hệ thống đã triển khai và thực nghiệm so sánh các phương pháp trên tập dữ liệu benchmark **213,118 từ** (`source.txt`):

| Phương pháp | Accuracy | Latency (ms) | Parameters | Dung lượng | Ghi chú |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dict (GPU-Vectorized)** | **100.00%** | **~399ms** | N/A | 315 KB | **Nhanh nhất** nhờ tối ưu hóa Tensor Mapping trên GPU. |
| **Dict (CPU Serial)** | 100.00% | ~413ms | N/A | 315 KB | Tra cứu bảng băm Python, cực nhanh cho CPU đơn nhân. |
| **CNN-Shallow-Big** | **100.00%** | **~885ms** | **3,326** | **16 KB** | **Champion:** AI Model đạt độ chính xác tuyệt đối. |
| **CNN-Big** | 99.93% | ~1253ms | 3,342 | 17 KB | Kiến trúc CNN 3 lớp (13 lỗi trên 18k âm tiết). |
| **Rule-based (Serial)** | 100.00% | ~1295ms | N/A | 5 KB | Phương pháp dựa trên luật, ổn định nhất về logic. |
| **CNN-Shallow-Small** | 75.27% | ~877ms | 1,202 | 8 KB | Mô hình thực nghiệm (Dưới ngưỡng hội tụ). |
| **CNN-Small** | 66.73% | ~902ms | 1,210 | 8 KB | Mô hình thực nghiệm (Dưới ngưỡng hội tụ). |

**Phân tích kỹ thuật & Hiệu chỉnh (04/04/2026):**
- **Sức mạnh AI:** Thực nghiệm cho thấy AI Model (`Shallow-Big`) nhanh gấp 2 lần so với các ước tính cũ (~415ms/100k từ).
- **GPU Optimization:** Khẳng định `GPU Vectorized Lookup` đạt hiệu suất cao nhất (~530,000 từ/giây), vượt qua CPU Serial trong các tác vụ xử lý hàng loạt lớn.
- **Vấn đề Parallel (Windows):** Trên môi trường Windows, việc song song hóa CPU (ProcessPool) chậm hơn Serial do chi phí khởi tạo process lớn. Thông tin "Speedup 2.7x" chỉ áp dụng cho môi trường Linux/Unix.
- **Dictionary Filtering:** Các mô hình AI đã được tích hợp bộ lọc từ điển ở tầng Inference để giữ nguyên 100% từ ngoại ngữ, số và ký hiệu, thay vì bắt model tự học copy.

## 2. Module: Khoa Đẩu -> Quốc ngữ (Đang triển khai)
- **Đặc điểm:** Bài toán ánh xạ 1-nhiều (Đồng âm khác hình), yêu cầu xử lý ngữ cảnh.
- **Giải pháp:** Kiến trúc **Seq2Seq Transformer** (Character-level) kết hợp **Beam Search (width=3)**.
- **Trạng thái:** Hoàn thành giai đoạn Prototype (100,000 mẫu). CER đạt **0.21**.
- **Cập nhật Kỹ thuật V2 (Inference):**
    - **Punctuation Padding:** Tự động tách dấu câu trước khi đưa vào AI để model nhận diện ranh giới từ tốt hơn.
    - **UI Alignment (SequenceMatcher):** Tự động so khớp kết quả AI với bản gốc để highlight lỗi (màu đỏ) và chỉ ra từ bị thiếu (marker xanh `[ ]`).
    - **Beam Search:** Cải thiện đáng kể độ mượt của câu so với Greedy Decoding.
- **Hạn chế hiện tại:** Mô hình vẫn gặp lỗi "Identity Mapping" (khó giữ nguyên các cụm từ ngoại ngữ lạ). Đang chuẩn bị huấn luyện lại trên tập 4 triệu mẫu (Phase 3) để khắc phục.

## 3. Nhật ký cập nhật kỹ thuật
- **04/04/2026:**
    - **Module 1:** Đính chính toàn bộ số liệu Latency dựa trên benchmark thực tế 213k từ. Xác nhận `Shallow-Big` là kiến trúc AI tối ưu nhất.
    - **Module 2:** Hoàn thiện Giao diện V2 với cơ chế highlight lỗi word-level. Ghi nhận vai trò của Punctuation Padding trong việc giảm CER.
    - **System:** Đồng bộ hóa logic "Dictionary Filtering" cho cả Rule-based và AI Inference.
- **23/04/2026 (Hoàn thành Phase 3 trên Cloud A100):**
    - **Kiến trúc & Kích thước:** Nâng cấp Transformer `D_MODEL=512`, `N_HEAD=16`.
    - **Tối ưu Tốc độ GPU (A100):** Áp dụng *Automatic Mixed Precision (AMP)* và cơ chế *Dynamic Padding* trong quá trình DataLoader (tránh thắt cổ chai CPU bằng Pre-tokenization). Giảm VRAM và tăng tốc độ xử lý từ 1.13 it/s lên ~7 it/s.
    - **Kết quả Đánh giá (500 mẫu ngẫu nhiên):** Cải thiện CER giảm đột phá xuống còn **0.0065** (0.65%), WER đạt **0.0218** (2.18%).
    - **Lỗi độ dài (OOM & Crash):** Đã phân tích giới hạn của `MAX_LEN` đối với Positional Encoding. Loại bỏ các hack không an toàn với CUDA SDPA (Flash Attention) và quay về phương pháp lọc dữ liệu chuẩn (loại bỏ câu dài hơn 192 ký tự).
    - **Giao diện (Inference V2):** Hỗ trợ nạp đa mô hình bằng Dropdown (Auto-detect cấu hình cũ 256 và cấu hình mới 512). Hỗ trợ tự động cắt cụt input dài để chống Crash GPU.
