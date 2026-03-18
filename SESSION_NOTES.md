# Nhật ký Kỹ thuật & Bài học Kinh nghiệm

## 1. Các lỗi đã gặp và Cách xử lý
- **Dữ liệu:** File CSV chứa giá trị `NaN` ở cuối file gây lỗi `TypeError` khi build vocab. -> **Xử lý:** Luôn dùng `.dropna()` hoặc `str(x) if pd.notnull(x) else ""`.
- **Mô hình:** Lỗi `size mismatch` khi load `state_dict` do định nghĩa class trong file `evaluate` lệch so với file `train`. -> **Xử lý:** Đồng bộ hóa kiến trúc (số lớp, hidden_dim) giữa hai file.
- **NLP:** Regex cơ bản `[a-zA-Z]` bỏ sót các ký tự tiếng Việt có dấu, khiến model trả về text gốc. -> **Xử lý:** Sử dụng dải Regex đầy đủ hoặc kiểm tra `c.isalpha()`.
- **Ngữ cảnh:** Dấu câu (., /) dính liền vào từ làm hỏng quy tắc "kết thúc từ" (final locking). -> **Xử lý:** Tách tuyệt đối dấu câu ra trước khi đưa vào mô hình hoặc Rule-based.

## 2. Bí kíp tối ưu hiệu năng
- **Batch Inference:** Đưa cả danh sách từ vào mô hình trong 1 lần gọi (Batch) thay vì dùng vòng lặp `for`. Tốc độ tăng ~30 lần.
- **Python Overhead:** Với các tác vụ siêu nhanh (như bản Nano), thời gian thực thi của AI bị chiếm 90% bởi khâu chuẩn hóa chuỗi và tạo Tensor. Nếu muốn nhanh hơn nữa, khâu này phải được viết bằng NumPy hoặc C++.
- **CPU vs GPU:** Với mô hình tí hon (Nano), chạy trên CPU đôi khi nhanh hơn GPU vì loại bỏ được chi phí vận chuyển dữ liệu qua PCIe bus.
- **PyTorch JIT:** Sử dụng `torch.jit.script` và `optimize_for_inference` giúp mô hình chạy trên CPU mượt mà hơn.

## 3. Cách tiếp cận cho Session sau (Khoa Đẩu -> Quốc ngữ)
- **Data:** Tái sử dụng tập 18k âm tiết, nhưng cần tạo thêm dữ liệu cấp **câu** để mô hình học ngữ cảnh (VD: "con cá" vs "kêu ca" để phân biệt c/k).
- **Model:** Chuyển từ CNN đơn giản sang kiến trúc có khả năng ghi nhớ trật tự tốt hơn như Transformer hoặc GRU với Attention.
