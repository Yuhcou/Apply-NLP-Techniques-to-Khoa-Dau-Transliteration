# Tổng quan Dự án: Applying NLP Techniques to Khoa Dau Transliteration

Dự án tập trung nghiên cứu và ứng dụng các kỹ thuật Xử lý Ngôn ngữ Tự nhiên (NLP) tiên tiến nhằm xây dựng hệ thống chuyển tự (transliteration) hai chiều giữa chữ Quốc Ngữ và chữ Khoa Đẩu. 

## 1. Module: Quốc ngữ -> Khoa Đẩu (Hoàn thành)
Đây là bài toán chuyển tự có tính cục bộ cao, đặc trưng bởi cơ chế ánh xạ 1-1 hoặc 1-nhiều nhưng quy luật rõ ràng. Hệ thống đã triển khai và thực nghiệm so sánh các phương pháp trên tập dữ liệu benchmark **213,118 từ** (`source.txt`):

| Phương pháp | Độ chính xác | Thời gian trễ (ms) | Số tham số | Dung lượng | Ghi chú |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dict (GPU-Vectorized)** | **100.00%** | **~399ms** | N/A | 315 KB | **Tối ưu nhất** nhờ tối ưu hóa Tensor Mapping trên GPU. |
| **Dict (CPU Serial)** | 100.00% | ~413ms | N/A | 315 KB | Tốc độ rất cao cho môi trường CPU đơn nhân. |
| **CNN-Shallow-Big** | **100.00%** | **~885ms** | **3,326** | **16 KB** | **Champion AI:** Đạt độ chính xác tuyệt đối với dung lượng cực thấp. |
| **CNN-Big** | 99.93% | ~1253ms | 3,342 | 17 KB | Kiến trúc CNN 3 lớp. |
| **Rule-based (Serial)** | 100.00% | ~1295ms | N/A | 5 KB | Phương pháp dựa trên luật, ổn định nhất về logic cốt lõi. |

**Phân tích kỹ thuật & Đóng góp:**
- **Giải pháp Lọc Từ Điển (Dictionary Filtering):** Tích hợp từ điển 18,000 âm tiết để bảo vệ tính nguyên bản của từ ngoại ngữ, số và ký tự đặc biệt, khắc phục nhược điểm nhận diện sai của mô hình học máy.
- **Tối ưu Tốc độ:** Kỹ thuật GPU Vectorized Lookup đạt thông lượng lên tới ~530,000 từ/giây. Việc song song hóa trên Windows gặp rào cản chi phí khởi tạo (overhead), do đó giải pháp Vectorized trên GPU/CPU đơn nhân được ưu tiên cho môi trường Windows.

## 2. Module: Khoa Đẩu -> Quốc ngữ (Hoàn thành Phase 3)
Đặc thù bài toán ánh xạ 1-nhiều (đồng âm khác hình), đòi hỏi hệ thống phải hiểu ngữ cảnh (context-aware) để khử nhập nhằng (Ví dụ: `c / k / q` dùng chung một ký tự Khoa Đẩu).

### 2.1. Quá trình Chuẩn bị Dữ liệu (Data Preparation)
- **Khai thác dữ liệu:** Tập ngữ liệu khổng lồ tiếng Việt ban đầu có quy mô lên tới **~4 triệu câu** (ngữ liệu thô từ HuggingFace).
- **Chuẩn hóa & Làm sạch (Cleaning):** Trải qua quá trình làm sạch nghiêm ngặt: loại bỏ câu chứa ký tự không hợp lệ, đồng nhất bộ chuẩn dấu tiếng Việt mới, tách dấu câu (Punctuation Padding), và đặc biệt là **loại bỏ các câu trùng lặp (drop_duplicates) và rỗng (dropna)**.
- **Dữ liệu thực tế đưa vào Huấn luyện:** Quá trình tinh tinh lọc đã giữ lại những mẫu dữ liệu cốt lõi và đa dạng nhất, cho ra tập dữ liệu chất lượng cao quy mô **~2.27 triệu câu duy nhất** (2.04M Train, 113K Val, 113K Test). Việc giảm số lượng từ 4 triệu thô xuống 2.27 triệu tinh lọc là tiêu chuẩn bắt buộc để mô hình không bị thiên lệch (overfit) vào các câu lặp lại.

### 2.2. Cấu trúc Mô hình & Kỹ thuật Tối ưu (State-of-the-Art)
Sử dụng kiến trúc **Seq2Seq Transformer (Character-level)**, kết hợp cùng các kỹ thuật tinh chỉnh tiên tiến nhất để tối đa hóa sức mạnh của phần cứng (NVIDIA A100 GPU):

- **Kiến trúc mạng (Architecture):** 
  - `D_MODEL = 512`, `N_HEAD = 16`, `NUM_LAYERS = 3`, `MAX_LEN = 192`. Không gian biểu diễn được mở rộng gấp đôi (so với Prototype Phase 2) giúp mô hình lưu trữ được nhiều quy tắc ngữ âm phức tạp.
- **Tối ưu Thông lượng DataLoader (I/O Optimization):**
  - Kỹ thuật **Pre-tokenization:** Số hóa toàn bộ 2 triệu câu thành mảng số nguyên lưu trên RAM trước khi huấn luyện, triệt tiêu hoàn toàn nút thắt cổ chai ở CPU.
  - Kỹ thuật **Dynamic Padding:** Trong hàm `collate_fn`, tensor chỉ được đệm (pad) bằng với độ dài lớn nhất của câu trong *batch hiện tại*, giảm tải hàng chục gigabyte VRAM so với việc đệm cứng 192 ký tự cho toàn bộ tập dữ liệu.
- **Tối ưu Hỗ trợ Tính toán GPU (Compute Optimization):**
  - Tích hợp **Automatic Mixed Precision (AMP):** Ép kiểu `torch.float16` tự động giúp đẩy tốc độ huấn luyện (throughput) tăng vọt từ 1.13 it/s lên ~7.1 it/s.
  - Phòng vệ **CUDA SDPA Crash:** Xử lý triệt để lỗi "Illegal Memory Access" của nhân Flash Attention bằng cách vô hiệu hóa mask boolean lỗi thời, sử dụng cơ chế sinh mask mặc định của PyTorch và lọc nghiêm ngặt các câu vượt quá giới hạn `MAX_LEN`.
- **Chiến lược Giải mã (Decoding Strategy):** Triển khai thuật toán **Beam Search (K=3)** được vector hóa toàn diện trên GPU. Thực nghiệm chứng minh `K=3` là điểm cân bằng hoàn hảo giữa độ chính xác và hiệu năng giải mã thời gian thực.

### 2.3. Kết quả Đánh giá (Evaluation Metrics)
Đánh giá khách quan trên tập kiểm thử (Test Set) chưa từng xuất hiện trong quá trình huấn luyện:

| Mô hình | Số lượng Mẫu Huấn luyện | D_MODEL | CER (Char Error Rate) | WER (Word Error Rate) |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 2 (Prototype)** | 100,000 | 256 | 0.2126 | 0.3980 |
| **Phase 3 (Production)** | **~2.27 Triệu (Cleaned)** | **512** | **0.0065** | **0.0218** |

**Thành tựu đột phá:** Mô hình **Phase 3 đạt mức CER ấn tượng là 0.65%**, dịch chính xác đến 99.35% ký tự. Độ lỗi từ (WER) giảm xuống còn 2.18%, vượt tiêu chuẩn thương mại. Mô hình xử lý mượt mà hầu hết các trường hợp đồng âm và cấu trúc câu phức tạp của tiếng Việt.

## 3. Hệ sinh thái Ứng dụng & Giao diện (Inference V2)
- Cung cấp giao diện trực quan hỗ trợ dịch hai chiều (Double Translation).
- Tích hợp **SequenceMatcher Alignment** giúp phát hiện và highlight chính xác từ bị dịch sai (màu đỏ) hoặc bị thiếu (marker xanh), hỗ trợ đánh giá minh bạch.
- Cơ chế **Auto-detect Model Architecture:** Hệ thống UI có khả năng nạp và tự động nhận diện cấu trúc của nhiều file weights khác nhau (tự động phát hiện D_MODEL 256 hay 512) từ định dạng state_dict, cho phép người dùng chuyển đổi linh hoạt các phiên bản mô hình mà không cần tái cấu trúc mã nguồn.
