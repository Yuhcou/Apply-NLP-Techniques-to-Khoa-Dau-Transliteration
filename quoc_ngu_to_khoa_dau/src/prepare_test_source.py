import pandas as pd
import os
import re

# Đường dẫn
RAW_CORPUS_DIR = "khoa_dau_to_quoc_ngu/data/raw_corpus"
OUTPUT_FILE = "quoc_ngu_to_khoa_dau/data/source.txt"
TARGET_SYLLABLES = 200000

def prepare_test_data():
    if not os.path.exists(RAW_CORPUS_DIR):
        print(f"Error: Directory {RAW_CORPUS_DIR} not found.")
        return

    # Lấy file shard đầu tiên
    shard_files = sorted([f for f in os.listdir(RAW_CORPUS_DIR) if f.endswith('.parquet')])
    if not shard_files:
        print("Error: No parquet files found.")
        return

    shard_path = os.path.join(RAW_CORPUS_DIR, shard_files[0])
    print(f"Reading from {shard_path}...")

    # Đọc theo từng chunk nếu có thể, hoặc đọc một phần nhỏ
    # Vì mỗi shard có 1 triệu dòng, ta chỉ cần vài nghìn dòng đầu
    df = pd.read_parquet(shard_path, columns=['text'])
    
    total_syllables = 0
    selected_texts = []
    
    for text in df['text']:
        if not text:
            continue
        
        # Làm sạch cơ bản: bỏ dấu xuống dòng thừa
        clean_text = text.replace('\n', ' ').strip()
        if not clean_text:
            continue
            
        # Đếm âm tiết
        syllables = clean_text.split()
        count = len(syllables)
        
        selected_texts.append(clean_text)
        total_syllables += count
        
        if total_syllables >= TARGET_SYLLABLES:
            break
            
    print(f"Collected {len(selected_texts)} sentences with total {total_syllables} syllables.")
    
    # Lưu ra file source.txt
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(selected_texts))
    
    print(f"Successfully saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_test_data()
