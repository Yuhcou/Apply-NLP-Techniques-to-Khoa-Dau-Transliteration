import torch
import pandas as pd
import os
import sys
import re
import time
import numpy as np

from src.config import MAX_LEN, VOCAB_PATH, CSV_PATH, INPUT_TEXT_PATH, MODEL_LARGE_PATH, MODEL_SMALL_PATH, MODEL_NANO_PATH
from src.models import KhoaDauCNNLarge, KhoaDauCNNSmall, KhoaDauCNNNano
from src.data_utils import fast_norm
from src.rule_based import encode_custom

class Evaluator:
    def __init__(self, m_type="large"):
        if not os.path.exists(VOCAB_PATH): 
            self.model = None; return
            
        v = torch.load(VOCAB_PATH)
        self.qn_v, self.kd_v = v['qn'], v['kd']
        self.rev_kd = {v: k for k, v in self.kd_v.items()}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if m_type == "large": 
            self.model = KhoaDauCNNLarge(len(self.qn_v), len(self.kd_v)).to(self.device)
            path = MODEL_LARGE_PATH
        elif m_type == "small": 
            self.model = KhoaDauCNNSmall(len(self.qn_v), len(self.kd_v)).to(self.device)
            path = MODEL_SMALL_PATH
        else: 
            self.model = KhoaDauCNNNano(len(self.qn_v), len(self.kd_v)).to(self.device)
            path = MODEL_NANO_PATH
        
        if not os.path.exists(path): 
            self.model = None; return
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def batch_predict(self, syllables):
        if not self.model or not syllables: return syllables
        input_data = np.zeros((len(syllables), MAX_LEN), dtype=np.int64)
        for i, s in enumerate(syllables):
            ids = [self.qn_v.get(c, 1) for c in fast_norm(s)][:MAX_LEN]
            input_data[i, :len(ids)] = ids
        
        with torch.inference_mode():
            preds = torch.argmax(self.model(torch.from_numpy(input_data).to(self.device)), dim=-1).cpu().numpy()
            
        rev = self.rev_kd
        return ["".join([rev.get(p, "") for p in word_p if p > 1]) for word_p in preds]

def run_performance_test():
    if not os.path.exists(INPUT_TEXT_PATH):
        print(f"Error: {INPUT_TEXT_PATH} not found.")
        return

    with open(INPUT_TEXT_PATH, "r", encoding="utf-8") as f: content = f.read()
    content = content * 1000 
    words = content.split()
    print(f"--- ĐẠI CHIẾN TỐC ĐỘ ({len(words)} TỪ) ---")

    start = time.time()
    for w in words: _ = encode_custom(w)
    rb_time = (time.time() - start) * 1000

    results = []
    pattern = re.compile(r'([a-zA-ZàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ]+)')
    
    for m_type in ["large", "small", "nano"]:
        ev = Evaluator(m_type)
        if not ev.model: continue
        
        start = time.time()
        syllables, structs = [], []
        for w in words:
            parts = pattern.split(w)
            s = []
            for p in parts:
                if p and p[0].isalpha(): s.append(len(syllables)); syllables.append(p)
                elif p: s.append(p)
            structs.append(s)
        
        translated = ev.batch_predict(syllables)
        _ = ["".join([translated[item] if isinstance(item, int) else item for item in s]) for s in structs]
        t_time = (time.time() - start) * 1000
        
        df = pd.read_csv(CSV_PATH).dropna()
        batch_preds = ev.batch_predict(df['original'].astype(str).tolist())
        acc = sum(1 for p, t in zip(batch_preds, df['encoded'].astype(str).tolist()) if p == t) / len(df)
        
        path = MODEL_LARGE_PATH if m_type == "large" else (MODEL_SMALL_PATH if m_type == "small" else MODEL_NANO_PATH)
        results.append({"name": m_type.upper(), "acc": acc*100, "time": t_time, "size": os.path.getsize(path)/1024})

    print(f"\n{'MODEL':<12} | {'ACCURACY':<10} | {'TIME (ms)':<12} | {'SIZE (KB)'}")
    print("-" * 55)
    print(f"{'Rule-Based':<12} | {'100.00%':<10} | {rb_time:<12.2f} | 5.37")
    for r in results:
        print(f"{r['name']:<12} | {r['acc']:<9.2f}% | {r['time']:<12.2f} | {r['size']:.2f}")

if __name__ == "__main__":
    run_performance_test()
