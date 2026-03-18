import torch
import torch.nn as nn
import pandas as pd
import unicodedata
import os
import sys
import re
import time
import numpy as np

# Thêm thư mục hiện tại vào path để import rule_based
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rule_based import encode_custom

MAX_LEN = 7
VOCAB_PATH = "quoc_ngu_to_khoa_dau/vocab.pth"
CSV_PATH = "quoc_ngu_to_khoa_dau/data/all-vietnamese-syllables-encoded.csv"
INPUT_TEXT_PATH = "quoc_ngu_to_khoa_dau/test_input.txt"

# Định nghĩa các kiến trúc
class KhoaDauCNNLarge(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_dim=128, hidden_dim=512):
        super(KhoaDauCNNLarge, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(); self.fc = nn.Linear(hidden_dim, output_vocab_size)
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.relu(self.conv1(x)); x = self.relu(self.conv2(x)); x = self.relu(self.conv3(x))
        return self.fc(x.transpose(1, 2))

class KhoaDauCNNSmall(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_dim=64, hidden_dim=256):
        super(KhoaDauCNNSmall, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(); self.fc = nn.Linear(hidden_dim, output_vocab_size)
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.relu(self.conv1(x)); x = self.relu(self.conv2(x)); x = self.relu(self.conv3(x))
        return self.fc(x.transpose(1, 2))

class KhoaDauCNNNano(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_dim=16, hidden_dim=48):
        super(KhoaDauCNNNano, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(); self.fc = nn.Linear(hidden_dim, output_vocab_size)
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.relu(self.conv1(x)); x = self.relu(self.conv2(x))
        return self.fc(x.transpose(1, 2))

TONE_MARKS_MAP = {ord(c): None for c in "\u0300\u0301\u0303\u0309\u0323"}
def fast_norm(text):
    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", text.lower()).translate(TONE_MARKS_MAP))

class Evaluator:
    def __init__(self, m_type="large"):
        v = torch.load(VOCAB_PATH)
        self.qn_v, self.kd_v = v['qn'], v['kd']
        self.rev_kd = {v: k for k, v in self.kd_v.items()}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if m_type == "large": self.model = KhoaDauCNNLarge(len(self.qn_v), len(self.kd_v)).to(self.device)
        elif m_type == "small": self.model = KhoaDauCNNSmall(len(self.qn_v), len(self.kd_v)).to(self.device)
        else: self.model = KhoaDauCNNNano(len(self.qn_v), len(self.kd_v)).to(self.device)
        
        path = f"quoc_ngu_to_khoa_dau/khoa_dau_cnn_{m_type}.pth"
        if not os.path.exists(path): self.model = None; return
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
    with open(INPUT_TEXT_PATH, "r", encoding="utf-8") as f: content = f.read()
    content = content * 1000 # NHÂN BẢN 1000 LẦN (~140,000 từ)
    words = content.split()
    print(f"--- ĐẠI CHIẾN TỐC ĐỘ ({len(words)} TỪ) ---")
    print(f"Thiết bị AI: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    # Rule-based
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
        
        results.append({"name": m_type.upper(), "acc": acc*100, "time": t_time, "size": os.path.getsize(f"quoc_ngu_to_khoa_dau/khoa_dau_cnn_{m_type}.pth")/1024})

    print(f"\n{'MODEL':<12} | {'ACCURACY':<10} | {'TIME (ms)':<12} | {'SIZE (KB)'}")
    print("-" * 55)
    print(f"{'Rule-Based':<12} | {'100.00%':<10} | {rb_time:<12.2f} | 5.37")
    for r in results:
        print(f"{r['name']:<12} | {r['acc']:<9.2f}% | {r['time']:<12.2f} | {r['size']:.2f}")

if __name__ == "__main__":
    run_performance_test()
