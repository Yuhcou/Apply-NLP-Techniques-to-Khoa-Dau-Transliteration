# Báo cáo Phương pháp Tiếp cận AI: Chuyển tự Khoa Đẩu sang Quốc ngữ

## 1. Thách thức cốt lõi (The Core Challenge)
Bài toán chuyển ngược từ **Khoa Đẩu -> Quốc ngữ** là bài toán **1-nhiều (One-to-Many)**. 
*   **Sự nhập nhằng (Ambiguity):** Một mã ký tự Khoa Đẩu (ví dụ: `e012`) có thể đại diện cho nhiều chữ Quốc ngữ khác nhau (`c`, `k`, `q`).
*   **Mất mát thông tin:** Hệ thống chữ Khoa Đẩu lịch sử thường đơn giản hóa các biến thể chính tả, dẫn đến việc thiếu thông tin để xác định từ đúng nếu không dựa vào ngữ cảnh (Context).

## 2. Các phương án đề xuất (Proposed Strategies)

### Phương án 1: Hệ thống lai (Rule-based + Spell Checker)
*   **Cơ chế:**
    1.  Dùng bộ quy tắc cố định chuyển Khoa Đẩu sang Quốc ngữ "thô" (Naive Quoc Ngu). Ví dụ: luôn chọn `c` cho mã `e012`.
    2.  Sử dụng một mô hình sửa lỗi chính tả (Spell Checker) tiền huấn luyện cho tiếng Việt (như PhoBERT hoặc mô hình N-gram) để sửa lại các từ sai chính tả dựa trên ngữ cảnh câu.
*   **Ưu điểm:** Tận dụng được các mô hình mạnh có sẵn về ngôn ngữ tiếng Việt.
*   **Nhược điểm:** Phụ thuộc vào chất lượng của bộ sửa lỗi bên thứ ba.

### Phương án 2: Phân loại dựa trên ngữ cảnh (Contextual Classification)
*   **Cơ chế:** Sử dụng mô hình BERT (PhoBERT) để dự đoán ký tự đúng tại các vị trí nhập nhằng.
*   **Cách làm:** Với mỗi từ Khoa Đẩu có nhiều lựa chọn, mô hình sẽ tính toán xác suất của từng lựa chọn dựa trên các từ đứng trước và đứng sau nó.
*   **Ưu điểm:** Độ chính xác về ngữ pháp cực cao.
*   **Nhược điểm:** Tốc độ xử lý chậm, khó chạy thời gian thực trên thiết bị yếu.

### Phương án 3: Dịch máy đầu-cuối (End-to-End Seq2Seq Transformer) - **KHUYÊN DÙNG**
*   **Cơ chế:** Huấn luyện một mô hình Transformer nhỏ (Small Transformer) từ đầu trên tập dữ liệu song ngữ tổng hợp (Synthetic Data).
*   **Cách làm:** Đầu vào là chuỗi ký tự Khoa Đẩu, đầu ra là chuỗi ký tự Quốc ngữ hoàn chỉnh.
*   **Ưu điểm:** 
    *   Tối ưu hóa hoàn toàn cho các mã Unicode đặc biệt của Khoa Đẩu.
    *   Mô hình nhẹ (vài MB), tốc độ cực nhanh, chạy được offline.
    *   Có khả năng học được cả các quy luật ngữ pháp và từ vựng cổ.
*   **Nhược điểm:** Cần lượng dữ liệu lớn (ít nhất 50,000 - 100,000 câu).

### Phương án 4: Sử dụng Mô hình ngôn ngữ lớn (LLM Prompting)
*   **Cơ chế:** Sử dụng GPT-4, Gemini hoặc Llama-3 để "dịch" và sửa lỗi.
*   **Cách làm:** Đưa câu thô vào và yêu cầu AI chỉnh sửa lại cho đúng chính tả tiếng Việt.
*   **Ưu điểm:** Không cần huấn luyện, kết quả rất tự nhiên.
*   **Nhược điểm:** Chi phí cao (API), yêu cầu kết nối mạng, tốc độ chậm nhất.

## 3. Bảng so sánh các phương án

| Tiêu chí | Phương án 1 (Hybrid) | Phương án 2 (BERT) | Phương án 3 (Transformer) | Phương án 4 (LLM) |
| :--- | :--- | :--- | :--- | :--- |
| **Độ chính xác** | Khá | Cao | **Rất cao** | Xuất sắc |
| **Tốc độ** | Nhanh | Trung bình | **Rất nhanh** | Chậm |
| **Khả năng Offline**| Có | Khó | **Có** | Không |
| **Độ phức tạp** | Thấp | Trung bình | Cao | Rất thấp |
| **Tài nguyên** | Thấp | Cao | **Thấp (sau training)** | Rất cao |

## 4. Lộ trình thực hiện (Roadmap)
1.  **Giai đoạn 1:** Xây dựng Dataset tổng hợp từ 100,000 câu tiếng Việt đa dạng nguồn (Báo chí, Văn học, Hội thoại).
2.  **Giai đoạn 2:** Thiết kế kiến trúc **Small Transformer** (độ sâu 4-6 lớp, số đầu chú ý 4-8).
3.  **Giai đoạn 3:** Huấn luyện mô hình và tối ưu hóa hàm mất mát (Loss function) cho bài toán Character-level.
4.  **Giai đoạn 4:** Đánh giá độ chính xác (Accuracy) và tốc độ (Inference Time).

---
*Cập nhật ngày: 18/03/2026*
