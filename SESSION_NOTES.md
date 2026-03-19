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

## 4. Ý tưởng cải tiến Bộ quy tắc (Rule-based) & Xử lý từ nước ngoài
- **Vấn đề:** Các từ nước ngoài (Python, Facebook...) khi đi qua bộ `rule_based.py` hiện tại sẽ bị biến đổi sai quy tắc, gây lỗi logic và làm nhiễu dữ liệu huấn luyện.
- **Giải pháp - Dictionary-based Filtering:**
    - Sử dụng file `all-vietnamese-syllables.txt` làm bộ lọc (từ điển âm tiết tiếng Việt).
    - Chỉ áp dụng chuyển tự Khoa Dấu cho những từ xuất hiện trong từ điển.
    - Giữ nguyên 100% các từ nước ngoài, tên riêng, thuật ngữ kỹ thuật, số và ký hiệu.
- **Lợi ích cho Model:**
    - Giúp model học được cơ chế "Copy" (giữ nguyên các ký tự Latinh lạ như f, j, w, z).
    - Tăng tính thực tế khi xử lý văn bản hỗn hợp (mixed-language).
    - Tạo ra bộ dữ liệu huấn luyện (từ dữ liệu crawl) nhất quán và chất lượng cao hơn.
- **Chiến lược Data:** Khi làm chiều `khoa_dau_to_quoc_ngu`, cần trộn dữ liệu Crawl (ngữ cảnh thực tế) với dữ liệu Synthetic (âm tiết đơn lẻ) để model vừa giỏi tiếng Việt vừa thông minh với từ ngoại ngữ.

## 3. Chuẩn hóa tiếng Việt và Xử lý Dữ liệu lớn (Session 19/03/2026)
- **Vấn đề Dấu tiếng Việt:** Sự khác biệt giữa Chuẩn cũ (`hoà, tuỳ`) và Chuẩn mới (`hòa, tùy`) gây nhiễu dữ liệu nghiêm trọng.
- **Giải pháp:** 
    - Chuyển đổi vĩnh viễn file `all-vietnamese-syllables.txt` sang Chuẩn mới (Dấu ở nguyên âm chính).
    - Mọi dữ liệu đầu vào (Input) và nhãn (Target) đều được chuẩn hóa về Chuẩn mới trước khi lưu.
- **Xử lý Parquet quy mô lớn:** 
    - Với dữ liệu hàng chục GB, không thể đọc toàn bộ vào RAM. 
    - Chiến lược: Đọc từng tệp Shard -> Tách câu -> Xử lý từng câu -> Lưu CSV trung gian -> Gộp file cuối cùng.
- **Kết quả Data:** Thu được ~3.8 triệu cặp câu (Khoa Đẩu - Quốc ngữ) chất lượng cao, sạch dấu câu và nhất quán về quy tắc đặt dấu.

## 4. Kế hoạch Session tới: Huấn luyện Mô hình Seq2Seq
- **Model:** Ưu tiên kiến trúc Transformer (Encoder-Decoder) vì khả năng học ngữ cảnh câu cực tốt.
- **Tokenizer:** Cần xây dựng Tokenizer cấp ký tự (Character-level) hoặc Subword cho cả Khoa Đẩu và Quốc ngữ.
- **Training:** Cần script huấn luyện hỗ trợ GPU (nếu có) và cơ chế lưu checkpoint.
