# Báo cáo Chiến lược AI: Chuyển tự Khoa Đẩu sang Quốc ngữ

Báo cáo này trình bày chi tiết về kiến trúc, quy trình thực thi và kết quả thực nghiệm của hệ thống AI chuyển tự từ **chữ Khoa Đẩu** sang **chữ Quốc ngữ**. Đây là bài toán **Dịch máy cấp ký tự (Character-level Machine Translation)** với thách thức lớn nhất là xử lý sự nhập nhằng (ambiguity) của các từ đồng âm khác hình.

---

## 1. Kiến trúc Hệ thống (Model Architecture)

Chúng tôi lựa chọn kiến trúc **Seq2Seq Transformer (Encoder-Decoder)** vì khả năng học ngữ cảnh toàn cục cực tốt, vượt trội so với các mô hình RNN truyền thống.

*   **Tokenizer (Character-level):** Sử dụng bộ mã hóa cấp ký tự thay vì cấp từ/subword. 
    *   *Lý do:* Số lượng ký tự Khoa Đẩu và Latinh hữu hạn (~200 ký tự), giúp loại bỏ hoàn toàn lỗi OOV (Out-of-Vocabulary) và hỗ trợ cực tốt cho các từ ngoại ngữ/số (cơ chế Copy).
*   **Transformer Hyperparameters:**
    *   **Layers:** 3-6 lớp Encoder/Decoder (tùy phase).
    *   **Attention Heads:** 8-16 heads.
    *   **Embedding Dim (d_model):** 256-512.
    *   **Activation:** GeLU (mượt mà hơn ReLU trong Transformer).
*   **Decoding Strategy:** **Beam Search (Width=3)**.
    *   *Lý do:* Thay vì chọn ký tự có xác suất cao nhất tại mỗi bước (Greedy), Beam Search duy trì các chuỗi ứng viên tiềm năng nhất để chọn ra kết quả có ý nghĩa ngữ pháp và ngữ cảnh tốt nhất.

---

## 2. Quy trình Thực thi 4 Giai đoạn (Phased Roadmap)

Dự án tuân thủ nghiêm ngặt mô hình **Gate-driven Development** (Chỉ chuyển giai đoạn khi đạt chỉ số mục tiêu).

| Giai đoạn | Quy mô (Mẫu) | Mục tiêu chính | Trạng thái |
| :--- | :--- | :--- | :--- |
| **Phase 1: Sanity Check** | 1,000 | Chứng minh pipeline logic (Overfitting test) | **PASS** |
| **Phase 2: Prototype** | 100,000 | Kiểm tra khả năng tổng quát hóa (Generalization) | **PASS** |
| **Phase 3: Production** | 4,000,000 | Huấn luyện quy mô lớn đạt độ chính xác thương mại | **WAITING** |
| **Phase 4: Optimization** | Toàn bộ | Tối ưu tốc độ Inference (Quantization/JIT) | **TODO** |

---

## 3. Chi tiết Triển khai và Kết quả (Phase Details)

### Phase 1: Sanity Check (Surgical Validation)
*   **Mục tiêu:** Ép mô hình "học thuộc lòng" 1,000 câu để xác nhận Tokenizer, Masks và Loss function hoạt động đúng.
*   **Cấu hình:** `D_MODEL=128`, `DROPOUT=0.0`, `LR=1e-3`, `Epochs=200`.
*   **Kết quả:** 
    *   **Loss:** 0.15 (Sát ngưỡng 0.1).
    *   **CER (Character Error Rate):** **0.04** (Mục tiêu < 0.05).
*   **Giải thích:** Mô hình đã học được cách ánh xạ 1-1 và bắt đầu nhận diện được các từ ngoại ngữ đơn giản.

### Phase 2: Prototype (Scale-up & Generalization)
*   **Mục tiêu:** Huấn luyện trên tập dữ liệu đủ lớn để mô hình bắt đầu học ngữ cảnh tiếng Việt và xử lý nhập nhằng.
*   **Cấu hình (Tối ưu cho GPU 4GB VRAM):** 
    *   `D_MODEL=256`, `N_HEAD=8`, `BATCH_SIZE=128`, `MAX_LEN=128`.
    *   Kỹ thuật: **Gradient Clipping (1.0)** và **Early Stopping (Patience=10)**.
*   **Kết quả:**
    *   **Validation Loss:** **0.1439** (Mục tiêu < 0.8).
    *   **CER:** **0.2126** (Mục tiêu < 0.3).
*   **Giải thích:** Với CER ~21%, mô hình đã dịch đúng khung xương chính của câu. Các lỗi còn lại chủ yếu nằm ở các từ vựng cổ hoặc cấu trúc câu cực kỳ phức tạp. **Pipeline chính thức mở khóa cho Phase 3.**

### Phase 3: Production (Cloud-based Training)
*   **Mục tiêu:** Đạt độ chính xác CER < 0.05 (Tương đương 95% ký tự chính xác).
*   **Cấu hình Đề xuất:**
    *   **Dữ liệu:** 4 Triệu mẫu (Đã chuẩn hóa dấu mới).
    *   **Kiến trúc:** Tăng lên 6 lớp Encoder/Decoder, `D_MODEL=512`, `N_HEAD=16`.
    *   **Môi trường:** Google Colab/Kaggle (P100/T4/A100 GPU).
*   **Tiêu chí Chấp nhận (Gate):** 
    *   **Val Loss < 0.1**.
    *   **CER < 0.05**.
    *   Thử nghiệm thực tế với văn bản hỗn hợp (Tiếng Việt + Anh + Số) phải đạt độ copy chính xác 100% cho phần ngoại ngữ.

---

## 4. Kết luận Kỹ thuật (Technical Inferences)

1.  **Character-level vs Word-level:** Cấp ký tự là chìa khóa để xử lý chữ Khoa Đẩu vì cấu trúc âm tiết Khoa Đẩu vốn đã mang tính chất ký tự lẻ (glyph-based).
2.  **Beam Search:** Là bắt buộc. Trong tiếng Việt, sự khác biệt giữa "c/k/q" (đều là mẫ e012) chỉ có thể giải quyết bằng cách nhìn cả "vần" đi sau, Beam Search giúp model duy trì các lựa chọn này.
3.  **Dữ liệu Synthetic:** Việc trộn 20% dữ liệu từ điển đơn lẻ với 80% dữ liệu ngữ cảnh báo chí giúp mô hình vừa giỏi ngữ pháp vừa không bị quên các từ hiếm.

---
*Cập nhật cuối: 21/03/2026 - Sau khi hoàn thành Phase 2 thành công.*
