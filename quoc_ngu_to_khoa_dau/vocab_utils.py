import torch
import os
import pandas as pd

VOCAB_PATH = "quoc_ngu_to_khoa_dau/vocab.pth"

def build_vocab(df):
    """Xây dựng từ điển từ dữ liệu thô."""
    qn_texts = df['original'].dropna().astype(str).tolist()
    kd_texts = df['encoded'].dropna().astype(str).tolist()
    
    qn_chars = sorted(list(set("".join(qn_texts))))
    kd_chars = sorted(list(set("".join(kd_texts))))
    
    qn_v = {char: i+2 for i, char in enumerate(qn_chars)}
    qn_v["<PAD>"], qn_v["<UNK>"] = 0, 1
    
    kd_v = {char: i+2 for i, char in enumerate(kd_chars)}
    kd_v["<PAD>"], kd_v["<UNK>"] = 0, 1
    
    return qn_v, kd_v

def get_or_build_vocab(df=None, force_rebuild=False):
    """
    Lấy vocab hiện có hoặc xây mới nếu chưa có.
    Đảm bảo tính nhất quán ID giữa các model.
    """
    if not force_rebuild and os.path.exists(VOCAB_PATH):
        print(f"Loading existing vocab from {VOCAB_PATH}")
        v = torch.load(VOCAB_PATH)
        return v['qn'], v['kd']
    
    if df is not None:
        print("Building new master vocab...")
        qn_v, kd_v = build_vocab(df)
        torch.save({'qn': qn_v, 'kd': kd_v}, VOCAB_PATH)
        return qn_v, kd_v
    
    raise ValueError("Vocab file not found and no DataFrame provided to build one.")
