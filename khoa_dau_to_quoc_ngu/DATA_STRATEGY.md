# Chiến lược Xử lý Dữ liệu: Khoa Đẩu - Quốc Ngữ

Tài liệu này ghi lại các quyết định kỹ thuật và logic xử lý dữ liệu để xây dựng bộ dataset huấn luyện cho mô hình chuyển tự từ Khoa Đẩu sang Quốc Ngữ.

## 1. Mục tiêu (Objectives)
- Xây dựng dataset từ văn bản thô (raw text) có chứa nhiều tạp chất (tên riêng nước ngoài, ký tự lạ, số, dấu câu).
- Đảm bảo dữ liệu "sạch" để mô hình học đúng quy tắc Khoa Đẩu nhưng vẫn đủ "linh hoạt" để xử lý văn bản thực tế.

## 2. Quy tắc Xử lý Ký tự Ngoại lai

### Ký tự Phi Latin (Trung, Nhật, Hy Lạp,...)
- **Hướng xử lý:** Loại bỏ hoàn toàn (Stripping).
- **Lý do:** Tránh bùng nổ từ điển (Vocabulary Explosion) và nhiễu dữ liệu. Các ký tự này không có quy luật âm tiết tương đồng với tiếng Việt hay Khoa Đẩu.
- **Thực hiện:** Sử dụng Regex whitelist chỉ giữ lại bảng chữ cái Latin, chữ tiếng Việt có dấu, số và dấu câu.

### Ký tự Hệ Latin (Tên riêng, từ mượn: Fukui, Hepburn, Apple,...)
- **Hướng xử lý:** Giữ nguyên (Identity Mapping).
- **Lý do:** Giúp mô hình học cơ chế "Copy". Khi gặp một từ không thuộc hệ thống Khoa Đẩu, mô hình sẽ học cách giữ nguyên từ đó ở đầu ra thay vì cố gắng dịch sai.
- **Thực hiện:** Kiểm tra từ bằng từ điển `all-vietnamese-syllables.txt`. Nếu không có trong từ điển nhưng là chữ Latin -> Không áp dụng `encode_custom`.

## 3. Xử lý Dấu câu và Số (Punctuation & Numbers)

### Bảo toàn Dấu câu
- Giữ lại các dấu câu cơ bản: `. , ! ? ( ) : ;` và các chữ số `0-9`.
- **Punctuation Padding:** Thêm khoảng trắng xung quanh dấu câu (ví dụ: `đi học.` -> `đi học .`).
- **Lý do:** Ngăn chặn việc dấu câu "dính" liền vào từ tạo thành các token sai lệch (ví dụ: `học.` và `học` sẽ bị coi là hai từ khác nhau nếu không tách dấu).

## 4. Cấu trúc Dataset (Input/Output)

- **Cột `quoc_ngu` (Target):** Giữ nguyên dấu thanh và chữ gốc để mô hình học được ngữ cảnh (Ví dụ: "Nhật", "Phúc").
- **Cột `khoa_dau` (Input):** 
    - Từ tiếng Việt: Chuyển sang Khoa Đẩu không dấu theo bộ `rule_based.py`.
    - Từ nước ngoài/Số/Dấu câu: Giữ nguyên dạng gốc.

## 5. Luồng xử lý trong `data_maker.py`
1. `clean_text`: Lọc bỏ ký tự phi Latin, đưa về lowercase.
2. `Punctuation Padding`: Tách dấu câu bằng khoảng trắng.
3. `Tokenization`: Tách câu thành danh sách các từ (tokens).
4. `Syllable Check`: 
    - Nếu nằm trong `all-vietnamese-syllables.txt` -> `encode_custom`.
    - Nếu không nằm trong danh sách -> Giữ nguyên (Identity).
5. `Reconstruct`: Ghép lại thành chuỗi và lưu vào CSV.

---
*Cập nhật lần cuối: 19/03/2026*
