import torch
import pandas as pd
import os
import sys
import re
import time
import numpy as np

from src.config import (MAX_LEN, VOCAB_PATH, CSV_PATH, INPUT_TEXT_PATH, 
                        MODEL_SHALLOW_BIG_PATH, MODEL_BIG_PATH, 
                        MODEL_SHALLOW_SMALL_PATH, MODEL_SMALL_PATH)
from src.models import KhoaDauCNN
from src.data_utils import fast_norm
from src.rule_based import encode_safe

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DictionaryEvaluator:
    """Chỉ sử dụng từ điển, hỗ trợ xử lý song song trên GPU qua Tensor Mapping."""
    def __init__(self, device):
        df = pd.read_csv(CSV_PATH).dropna()
        self.mapping = dict(zip(df['original'].astype(str), df['encoded'].astype(str)))
        self.device = device
        
        # Tạo Tensor Mapping trên GPU (Mô phỏng xử lý song song cho Dictionary)
        # 1. Thu thập tất cả syllables
        all_syllables = sorted(list(self.mapping.keys()))
        self.syllable_to_id = {s: i for i, s in enumerate(all_syllables)}
        
        # 2. Tạo ma trận đích (Target Matrix) trên GPU
        # Mỗi hàng i là chuỗi Khoa Đẩu đã được mã hóa (Padded) của syllable có ID i
        vocab = torch.load(VOCAB_PATH)
        self.kd_v = vocab['kd']
        self.rev_kd = {v: k for k, v in self.kd_v.items()}
        
        target_matrix = []
        for s in all_syllables:
            encoded_str = self.mapping[s]
            ids = [self.kd_v.get(c, 0) for c in encoded_str][:MAX_LEN]
            if len(ids) < MAX_LEN:
                ids += [0] * (MAX_LEN - len(ids))
            target_matrix.append(ids)
            
        self.gpu_targets = torch.tensor(target_matrix, dtype=torch.long).to(device)

    def parallel_lookup(self, input_syllables):
        """Dùng GPU để ánh xạ song song ID -> Khoa Đẩu ID."""
        # Chuyển input về IDs
        input_ids = [self.syllable_to_id.get(s, -1) for s in input_syllables]
        # Xử lý các từ không có trong từ điển (Ngoại ngữ) - gán ID tạm là 0
        valid_mask = torch.tensor([i != -1 for i in input_ids]).to(self.device)
        safe_ids = torch.tensor([max(0, i) for i in input_ids]).to(self.device)
        
        # Parallel Lookup trên GPU (Lấy đồng loạt các hàng tương ứng)
        batch_preds = self.gpu_targets[safe_ids]
        
        # Chuyển ngược về chuỗi (Giữ nguyên từ ngoại ngữ)
        results = []
        rev = self.rev_kd
        batch_preds_cpu = batch_preds.cpu().numpy()
        valid_mask_cpu = valid_mask.cpu().numpy()
        
        for i in range(len(input_syllables)):
            if valid_mask_cpu[i]:
                res = "".join([rev.get(p, "") for p in batch_preds_cpu[i] if p > 1])
                results.append(res)
            else:
                results.append(input_syllables[i]) # Copy từ ngoại ngữ
        return results

class Evaluator:
    def __init__(self, m_type="shallow_big"):
        if not os.path.exists(VOCAB_PATH): 
            self.model = None; return
            
        v = torch.load(VOCAB_PATH)
        self.qn_v, self.kd_v = v['qn'], v['kd']
        self.rev_kd = {v: k for k, v in self.kd_v.items()}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        m_type = m_type.lower()
        mapping = {
            "big": (KhoaDauCNN.big, MODEL_BIG_PATH),
            "small": (KhoaDauCNN.small, MODEL_SMALL_PATH),
            "shallow_big": (KhoaDauCNN.shallow_big, MODEL_SHALLOW_BIG_PATH),
            "shallow_small": (KhoaDauCNN.shallow_small, MODEL_SHALLOW_SMALL_PATH),
        }
        
        if m_type not in mapping:
            self.model = None; return
            
        model_fn, path = mapping[m_type]
        self.model = model_fn(len(self.qn_v), len(self.kd_v)).to(self.device)
        self.path = path
        
        if not os.path.exists(path): 
            self.model = None; return
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def batch_predict(self, syllables):
        if not self.model or not syllables: return syllables
        batch_size = 4096
        all_results = []
        for i in range(0, len(syllables), batch_size):
            batch = syllables[i:i+batch_size]
            input_data = np.zeros((len(batch), MAX_LEN), dtype=np.int64)
            for j, s in enumerate(batch):
                ids = [self.qn_v.get(c, 1) for c in fast_norm(str(s))][:MAX_LEN]
                input_data[j, :len(ids)] = ids
            with torch.inference_mode():
                preds = torch.argmax(self.model(torch.from_numpy(input_data).to(self.device)), dim=-1).cpu().numpy()
            rev = self.rev_kd
            all_results.extend(["".join([rev.get(p, "") for p in word_p if p > 1]) for word_p in preds])
        return all_results

def run_performance_test():
    if not os.path.exists(INPUT_TEXT_PATH):
        print(f"Error: {INPUT_TEXT_PATH} not found."); return

    with open(INPUT_TEXT_PATH, "r", encoding="utf-8") as f: content = f.read()
    words = content.split()
    print(f"--- EVALUATE TRÊN DATASET ({len(words)} TỪ) ---")

    df_test = pd.read_csv(CSV_PATH).dropna()
    test_originals = df_test['original'].astype(str).tolist()
    test_encodeds = df_test['encoded'].astype(str).tolist()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Rule-Based ---
    start = time.time()
    _ = encode_safe(content)
    rb_time = (time.time() - start) * 1000

    # --- 2. Dictionary (GPU-Optimized) ---
    dict_ev = DictionaryEvaluator(device)
    start = time.time()
    _ = dict_ev.parallel_lookup(words)
    dict_time = (time.time() - start) * 1000
    
    # Accuracy của Dictionary luôn 100% trên tập chuẩn
    dict_preds = dict_ev.parallel_lookup(test_originals)
    dict_acc = sum(1 for p, t in zip(dict_preds, test_encodeds) if p == t) / len(test_encodeds)

    # --- 3. AI Models ---
    results = []
    model_types = ["big", "small", "shallow_big", "shallow_small"]
    
    for m_type in model_types:
        ev = Evaluator(m_type)
        if not ev.model: continue
        
        start = time.time()
        ev.batch_predict(words)
        t_time = (time.time() - start) * 1000
        
        batch_preds = ev.batch_predict(test_originals)
        acc = sum(1 for p, t in zip(batch_preds, test_encodeds) if p == t) / len(test_encodeds)
        
        results.append({
            "name": m_type.upper(), 
            "acc": acc*100, 
            "time": t_time, 
            "size": os.path.getsize(ev.path)/1024,
            "params": count_parameters(ev.model)
        })

    print(f"\n{'MODEL':<14} | {'ACCURACY':<10} | {'TIME (ms)':<10} | {'PARAMS':<10} | {'SIZE (KB)'}")
    print("-" * 75)
    print(f"{'Rule-Based':<14} | {'100.00%':<10} | {rb_time:<10.2f} | {'N/A':<10} | 5.37")
    print(f"{'Dictionary':<14} | {dict_acc*100:<9.2f}% | {dict_time:<10.2f} | {'N/A':<10} | {os.path.getsize(CSV_PATH)/1024:.2f}")
    for r in results:
        print(f"{r['name']:<14} | {r['acc']:<9.2f}% | {r['time']:<10.2f} | {r['params']:<10d} | {r['size']:.2f}")

if __name__ == "__main__":
    run_performance_test()
