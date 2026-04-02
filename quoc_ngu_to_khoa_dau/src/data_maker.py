from .rule_based import encode_custom
from .config import SYLLABLES_PLAIN_PATH, CSV_PATH
import os
import csv

def make_data():
    print(f"Đang tạo dữ liệu từ {SYLLABLES_PLAIN_PATH}...")
    if not os.path.exists(SYLLABLES_PLAIN_PATH):
        print("Lỗi: Không tìm thấy file âm tiết gốc.")
        return

    with open(SYLLABLES_PLAIN_PATH, "r", encoding="utf-8") as fin, open(
        CSV_PATH, "w", encoding="utf-8", newline=""
    ) as fout:
        writer = csv.writer(fout)
        writer.writerow(["original", "encoded"]) # Thêm header
        for raw in fin:
            word = raw.strip()
            if not word:
                continue
            writer.writerow([word, encode_custom(word)])
    print(f"Đã tạo xong dữ liệu tại {CSV_PATH}")

if __name__ == "__main__":
    make_data()
