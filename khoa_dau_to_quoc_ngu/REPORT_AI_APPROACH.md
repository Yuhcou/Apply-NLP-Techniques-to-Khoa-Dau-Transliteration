# Báo cáo Phương pháp Tiếp cận AI: Chuyển tự Khoa Đẩu sang Quốc ngữ

## 1. Thách thức cốt lõi (The Core Challenge)
Bài toán chuyển ngược từ **Khoa Đẩu -> Quốc ngữ** là bài toán **1-nhiều (One-to-Many)**. 
*   **Sự nhập nhằng (Ambiguity):** Một mã ký tự Khoa Đẩu (ví dụ: `e012`) có thể đại diện cho nhiều chữ Quốc ngữ khác nhau (`c`, `k`, `q`).
*   **Mất mát thông tin:** Hệ thống chữ Khoa Đẩu lịch sử thường đơn giản hóa các biến thể chính tả, dẫn đến việc thiếu thông tin để xác định từ đúng nếu không dựa vào ngữ cảnh (Context).

## 2. Các phương án đề xuất (Proposed Strategies)

### Phương án 1: Hệ thống lai (Rule-based + Spell Checker)
*   **Cơ chế:** Dùng bộ quy tắc cố định chuyển Khoa Đẩu sang Quốc ngữ "thô" kết hợp với bộ sửa lỗi chính tả.
*   **Ưu điểm:** Tận dụng được các mô hình mạnh có sẵn về ngôn ngữ tiếng Việt.

### Phương án 2: Phân loại dựa trên ngữ cảnh (Contextual Classification)
*   **Cơ chế:** Sử dụng mô hình BERT (PhoBERT) để dự đoán ký tự đúng tại các vị trí nhập nhằng.

### Phương án 3: Dịch máy đầu-cuối (End-to-End Seq2Seq Transformer) - **ĐÃ TRIỂN KHAI**
*   **Kiến trúc:** Seq2Seq Transformer (Encoder-Decoder) với 3 lớp Encoder, 3 lớp Decoder, 8 Head Attention và kích thước Embedding 256.
*   **Tokenizer:** Character-level Tokenizer giúp mô hình xử lý 100% từ vựng mà không gặp lỗi OOV.
*   **Chiến lược Giải mã (Decoding):** Áp dụng **Beam Search** với `beam_width=3` để chọn ra chuỗi tối ưu nhất theo ngữ cảnh.
*   **Ưu điểm:** 
    *   Xử lý hoàn hảo các trường hợp nhập nhằng dựa trên ngữ cảnh chuỗi.
    *   Khả năng "Copy" thông minh các từ ngoại ngữ và ký số.
    *   Mô hình nhẹ (~10MB), tốc độ cực nhanh, chạy được offline.

### Phương án 4: Sử dụng Mô hình ngôn ngữ lớn (LLM Prompting)
*   **Cơ chế:** Sử dụng GPT-4, Gemini hoặc Llama-3 để "dịch" và sửa lỗi.

## 3. Bảng so sánh các phương án

| Tiêu chí | Phương án 1 (Hybrid) | Phương án 2 (BERT) | Phương án 3 (Transformer) | Phương án 4 (LLM) |
| :--- | :--- | :--- | :--- | :--- |
| **Độ chính xác** | Khá | Cao | **Rất cao** | Xuất sắc |
| **Tốc độ** | Nhanh | Trung bình | **Rất nhanh** | Chậm |
| **Khả năng Offline**| Có | Khó | **Có** | Không |
| **Tài nguyên** | Thấp | Cao | **Thấp (sau training)** | Rất cao |

## 4. Lộ trình thực hiện (Roadmap) - Trạng thái: **Giai đoạn 3 & 4**
1.  **Giai đoạn 1 (Hoàn thành):** Xây dựng Dataset khổng lồ (~4 triệu câu) từ Wikipedia và báo chí.
2.  **Giai đoạn 2 (Hoàn thành):** Thiết kế kiến trúc **Seq2Seq Transformer** tối ưu.
3.  **Giai đoạn 3 (Đang thực hiện):** Huấn luyện Prototype trên **100,000 mẫu**.
4.  **Giai đoạn 4 (Tiếp theo):** Mở rộng huấn luyện lên toàn bộ tập dữ liệu và đóng gói hệ thống Inference.

---
*Cập nhật ngày: 20/03/2026*
