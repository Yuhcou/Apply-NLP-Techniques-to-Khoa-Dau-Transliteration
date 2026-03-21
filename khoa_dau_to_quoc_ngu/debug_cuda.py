import torch
import os
import pandas as pd
from src.config import *
from src.vocab_utils import CharTokenizer
from src.models import Seq2SeqTransformer

def debug():
    # 1. Kiểm tra Tokenizer
    tokenizer = CharTokenizer()
    if not tokenizer.load(VOCAB_PATH):
        print("Lỗi: Không tìm thấy file từ điển.")
        return
    
    print(f"--- THÔNG SỐ TOKENIZER ---")
    print(f"Vocab Size thực tế: {tokenizer.vocab_size}")
    max_id = max(tokenizer.id2char.keys())
    print(f"ID lớn nhất trong vocab: {max_id}")
    
    # 2. Kiểm tra Model
    model = Seq2SeqTransformer(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        emb_size=D_MODEL,
        nhead=N_HEAD,
        vocab_size=tokenizer.vocab_size,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    )
    
    # Kích thước embedding của model
    emb_size = model.src_tok_emb.embedding.num_embeddings
    print(f"\n--- THÔNG SỐ MÔ HÌNH ---")
    print(f"Embedding size của mô hình: {emb_size}")
    
    if max_id >= emb_size:
        print(f"!!! PHÁT HIỆN LỖI: ID lớn nhất ({max_id}) vượt quá kích thước Embedding ({emb_size})!")
    else:
        print("Kích thước Embedding và Vocab khớp nhau (OK).")

    # 3. Kiểm tra dữ liệu bị lỗi (Giả thuyết có ký tự NaN hoặc Inf)
    print(f"\n--- KIỂM TRA DỮ LIỆU ---")
    df = pd.read_csv(TRAIN_DATA_PATH).sample(100000, random_state=42)
    # Lấy 100k mẫu như khi train
    
    found_invalid = False
    for i, row in df.iterrows():
        src_text = str(row['khoa_dau'])
        tgt_text = str(row['quoc_ngu'])
        
        src_ids = tokenizer.encode(src_text)
        tgt_ids = tokenizer.encode(tgt_text)
        
        if any(idx >= emb_size for idx in src_ids) or any(idx >= emb_size for idx in tgt_ids):
            print(f"Phát hiện chuỗi chứa ID không hợp lệ tại dòng {i}!")
            found_invalid = True
            break
            
    if not found_invalid:
        print("Không tìm thấy ID không hợp lệ trong 100k mẫu test.")

if __name__ == "__main__":
    debug()
