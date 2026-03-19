import os
import sys
import pandas as pd
import re
from tqdm import tqdm
import glob

# Thêm đường dẫn để import rule_based từ module quoc_ngu_to_khoa_dau
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "quoc_ngu_to_khoa_dau"))
from src.rule_based import encode_custom

def normalize_tone_to_modern(word):
    """Chuẩn hóa vị trí đặt dấu tiếng Việt sang CHUẨN MỚI (Dấu ở nguyên âm 2).
    Ví dụ: hòa (dấu ở a), hòe (dấu ở e), thúy (dấu ở y), toán, hoét..."""
    # Bản đồ đưa dấu từ nguyên âm đệm sang nguyên âm chính (Chuẩn mới)
    to_modern = {
        "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
        "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
        "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
    }
    for old, new in to_modern.items():
        word = word.replace(old, new)
    return word

def load_vietnamese_syllables(syllable_file):
    """Tải danh sách âm tiết đã được chuẩn hóa (Chuẩn mới)."""
    if not os.path.exists(syllable_file):
        print(f"Cảnh báo: Không tìm thấy file âm tiết tại {syllable_file}")
        return set()
    
    with open(syllable_file, 'r', encoding='utf-8') as f:
        # File đã được chuẩn hóa sang Chuẩn mới ở Bước 1
        syllables = {line.strip().lower() for line in f if line.strip()}
    return syllables

# Cấu hình đường dẫn
RAW_DIR = "khoa_dau_to_quoc_ngu/data/raw_corpus"
OUTPUT_DIR = "khoa_dau_to_quoc_ngu/data/processed"
SYLLABLE_DICT_PATH = "quoc_ngu_to_khoa_dau/data/all-vietnamese-syllables.txt"

# Tải từ điển âm tiết
VIETNAMESE_SYLLABLES = load_vietnamese_syllables(SYLLABLE_DICT_PATH)

def clean_text(text):
    """Làm sạch văn bản: giữ lại chữ cái tiếng Việt, số, dấu câu."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-zàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ0-9\s.,!?():;]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_sentences(text):
    """Tách văn bản dài thành các câu."""
    sentences = re.split(r'(?<=[.?!])\s+|\n+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]

def process_sentence(sentence):
    """Chuyển đổi một câu sang cặp Khoa Đẩu - Quốc ngữ theo CHUẨN MỚI duy nhất."""
    clean_s = clean_text(sentence)
    if not clean_s: return None
    
    # Punctuation Padding
    clean_s = re.sub(r'([.,!?():;])', r' \1 ', clean_s)
    clean_s = re.sub(r'\s+', ' ', clean_s).strip()
    
    words = clean_s.split()
    kd_words, qn_words = [], []
    
    for w in words:
        if re.match(r'^[0-9.,!?():;]+$', w):
            kd_words.append(w)
            qn_words.append(w)
            continue
            
        # 1. Chuẩn hóa về Chuẩn mới (Dấu ở nguyên âm 2)
        w_modern = normalize_tone_to_modern(w)
        
        # 2. Kiểm tra trong từ điển
        if w_modern in VIETNAMESE_SYLLABLES:
            # 3. Mã hóa Khoa Đẩu dựa trên chuẩn mới
            kd = encode_custom(w_modern)
            if kd:
                kd_words.append(kd)
                # CHỈ LƯU CHUẨN MỚI vào Target để mô hình viết đúng quy tắc
                qn_words.append(w_modern)
        else:
            # Từ ngoại ngữ, số, dấu câu -> Giữ nguyên
            kd_words.append(w)
            qn_words.append(w)
            
    if not kd_words or len(kd_words) < 3:
        return None
        
    return " ".join(kd_words), " ".join(qn_words)

def process_parquet_files():
    """Lặp qua các file parquet và xử lý hàng loạt."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    parquet_files = glob.glob(os.path.join(RAW_DIR, "*.parquet"))
    
    if not parquet_files:
        print(f"Không tìm thấy file parquet nào tại {RAW_DIR}")
        return

    print(f"Tìm thấy {len(parquet_files)} file shards. Bắt đầu xử lý với CHUẨN MỚI (Dấu ở nguyên âm chính)...")
    
    for i, file_path in enumerate(parquet_files):
        print(f"[{i+1}/{len(parquet_files)}] Đang xử lý: {os.path.basename(file_path)}")
        df_raw = pd.read_parquet(file_path)
        
        processed_pairs = []
        # Chạy 10,000 dòng để kiểm tra chất lượng
        for text in tqdm(df_raw['text'].head(10000), desc="Đang tách và chuyển tự câu"):
            sentences = split_into_sentences(text)
            for s in sentences:
                res = process_sentence(s)
                if res:
                    processed_pairs.append({"khoa_dau": res[0], "quoc_ngu": res[1]})
        
        if processed_pairs:
            df_out = pd.DataFrame(processed_pairs).drop_duplicates()
            out_name = os.path.basename(file_path).replace(".parquet", "_processed.csv")
            df_out.to_csv(os.path.join(OUTPUT_DIR, out_name), index=False, encoding='utf-8')
            print(f"   -> Đã lưu {len(df_out)} câu CHUẨN MỚI vào {out_name}")

if __name__ == "__main__":
    process_parquet_files()
