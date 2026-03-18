# Đoạn code này xuất ra ảnh tất cả ký tự trong bảng chữ.

import os
from PIL import Image, ImageDraw, ImageFont

def export_unicode_range(start_hex, end_hex, exclude_hex, font_path, output_dir="khoa_dau_script"):
    # Tạo thư mục đầu ra nếu chưa có
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        font_size = 150
        image_size = (200, 200)
        font = ImageFont.truetype(font_path, font_size)

        # Chuyển đổi dải hex sang số nguyên
        start = int(start_hex, 16)
        end = int(end_hex, 16)
        exclude = [int(h, 16) for h in exclude_hex]

        for code_point in range(start, end + 1):
            if code_point in exclude:
                continue

            char = chr(code_point)
            char_hex = f"{code_point:04x}" # Định dạng tên file (vd: e000.png)

            # Tạo ảnh
            image = Image.new('RGBA', image_size, color=(255, 255, 255, 0)) # Nền trong suốt
            draw = ImageDraw.Draw(image)

            # Căn giữa
            left, top, right, bottom = draw.textbbox((0, 0), char, font=font)
            text_width = right - left
            text_height = bottom - top
            position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2 - top)

            # Vẽ và lưu
            draw.text(position, char, fill='black', font=font)
            image.save(os.path.join(output_dir, f"char_{char_hex}.png"))
            print(f"Đã xuất: {char_hex}")

    except Exception as e:
        print(f"Lỗi: {e}")

# --- Cấu hình ---
FONT_FILE = "misc/font/KhoaDau-Regular_v3.otf"  # THAY ĐỔI ĐƯỜNG DẪN FONT CỦA BẠN
EXCLUDE = ["e00d", "e010", "e023"]

export_unicode_range("e000", "e026", EXCLUDE, FONT_FILE)