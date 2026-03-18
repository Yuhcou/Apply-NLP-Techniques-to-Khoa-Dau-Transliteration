# Tổng quan Dự án: Applying NLP Techniques to Khoa Dau Transliteration

Dự án này tập trung vào việc áp dụng các kỹ thuật Xử lý ngôn ngữ tự nhiên (NLP) và Trí tuệ Nhân tạo (AI) để thực hiện chuyển tự (transliteration) giữa **chữ Khoa Đẩu** và **chữ Quốc ngữ**.

> **Ghi chú:** File này được cập nhật sau mỗi session.

## 1. Module: Quốc ngữ -> Khoa Đẩu (Hoàn thành)
Đây là bài toán chuyển tự **1-1** (tính cục bộ cao). Chúng ta đã xây dựng và so sánh các phương pháp sau trên tập 141,000 từ:

| Phương pháp | Accuracy | Thời gian (ms) | Dung lượng | Ghi chú |
| :--- | :--- | :--- | :--- | :--- |
| **Rule-based** | **100%** | **~700ms** | **5 KB** | Tối ưu nhất cho CPU và ứng dụng thông thường. |
| **CNN-Small** | 99.91% | ~4000ms | 1.8 MB | Điểm cân bằng tốt nhất nếu cần dùng AI. |
| **CNN-Large** | 99.97% | ~15000ms | 7.0 MB | Quá nặng, bị nghẽn cổ chai bởi overhead của Python. |
| **CNN-Nano** | 74.64% | **~600ms** | **53 KB** | Nhanh hơn Rule-based nhờ Batching nhưng Accuracy thấp. |

**Kết luận:**
- Với bài toán 1-1, Rule-based vẫn là giải pháp tối ưu về mọi mặt trên CPU.
- AI (CNN) chỉ thực sự phát huy sức mạnh tốc độ khi xử lý Batch cực lớn và khâu tiền xử lý được tối ưu hóa bằng ngôn ngữ bậc thấp (C++/Rust).
- Điểm yếu của AI trong bài toán này là chi phí đóng gói Tensor (Overhead) lớn hơn chi phí tính toán logic.

## 2. Module: Khoa Đẩu -> Quốc ngữ (Kế hoạch)
- **Thách thức:** Bài toán **1-nhiều** (Đồng âm khác hình: c/k/q -> e012). Rule-based không thể xử lý.
- **Giải pháp:** Sử dụng AI (Language Model / Seq2Seq) để dựa vào ngữ cảnh chọn từ đúng.

## 3. Nhật ký cập nhật
- **17/03/2026:** 
    - Dọn dẹp project, xây dựng thành công 3 phiên bản mô hình 1D-CNN.
    - Triển khai Batch Inference tối ưu hóa tốc độ AI gấp 30 lần so với tuần tự.
    - Tìm ra giới hạn vật lý của AI trên CPU so với Rule-based.
    - Hoàn thiện bộ công cụ đánh giá (`evaluate_model.py`) công bằng và chính xác.
