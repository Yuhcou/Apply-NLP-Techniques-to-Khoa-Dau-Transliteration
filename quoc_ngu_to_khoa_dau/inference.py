import torch
import tkinter as tk
from tkinter import ttk, messagebox
import os
import re
import random
import unicodedata
import ctypes
import pandas as pd
from src.config import (MAX_LEN, VOCAB_PATH, MODEL_SHALLOW_BIG_PATH, 
                        MODEL_BIG_PATH, MODEL_SHALLOW_SMALL_PATH, MODEL_SMALL_PATH,
                        INPUT_TEXT_PATH, CSV_PATH)
from src.models import KhoaDauCNN
from src.data_utils import fast_norm
from src.rule_based import encode_custom

# --- LOAD FONT KHOA DAU TẠM THỜI (Windows) ---
FONT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "misc", "font", "KhoaDau-Regular_v3.otf"))

def load_custom_font(font_path):
    if os.path.exists(font_path):
        FR_PRIVATE = 0x10
        ctypes.windll.gdi32.AddFontResourceExW(font_path, FR_PRIVATE, 0)
    else:
        print(f"Cảnh báo: Không tìm thấy font tại {font_path}")

class KhoaDauInferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Khoa Đẩu Transliteration Studio v1.0")
        self.root.geometry("900x700")
        self.root.attributes("-topmost", True)

        load_custom_font(FONT_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load vocab
        v = torch.load(VOCAB_PATH)
        self.qn_v, self.kd_v = v['qn'], v['kd']
        self.rev_kd = {v: k for k, v in self.kd_v.items()}
        
        # Load dictionary mapping
        df = pd.read_csv(CSV_PATH).dropna()
        self.dict_mapping = dict(zip(df['original'].astype(str), df['encoded'].astype(str)))
        
        # Load real sentences from source.txt
        self.sentences = self._load_corpus()

        # Cache cho các mô hình AI
        self.models = {}
        
        self.setup_ui()
        self.change_model()

    def _load_corpus(self):
        """Đọc và làm sạch các câu từ file source.txt."""
        sentences = []
        if os.path.exists(INPUT_TEXT_PATH):
            try:
                with open(INPUT_TEXT_PATH, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Tách thành các đoạn văn/câu dựa trên dấu chấm hoặc dòng mới
                    raw_lines = re.split(r'[\.\n]+', content)
                    for line in raw_lines:
                        line = line.strip()
                        # Chỉ lấy những câu có độ dài hợp lý (từ 5 đến 30 từ) và là tiếng Việt
                        if 5 <= len(line.split()) <= 30:
                            sentences.append(line)
            except Exception as e:
                print(f"Lỗi đọc file corpus: {e}")
        
        if not sentences:
            sentences = ["Chào mừng bạn đến với công cụ chuyển tự chữ Khoa Đẩu.", "Hệ thống sử dụng trí tuệ nhân tạo để nhận diện âm tiết."]
        return sentences

    def setup_ui(self):
        # ... (Phần UI giữ nguyên như bản trước nhưng mở rộng kích thước)
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill="x")
        
        ttk.Label(top_frame, text="Phương pháp:").pack(side="left")
        self.model_var = tk.StringVar(value="Shallow_Big")
        self.model_combo = ttk.Combobox(top_frame, textvariable=self.model_var, state="readonly", width=20)
        self.model_combo['values'] = ("Rule-based", "Dictionary", "Shallow_Big", "Big", "Shallow_Small", "Small")
        self.model_combo.pack(side="left", padx=5)
        self.model_combo.bind("<<ComboboxSelected>>", lambda e: self.change_model())

        ttk.Button(top_frame, text="🎲 Câu ngẫu nhiên", command=self.random_sample).pack(side="left", padx=10)
        
        input_frame = ttk.Frame(self.root, padding=10)
        input_frame.pack(fill="both", expand=True)
        ttk.Label(input_frame, text="Quốc ngữ:").pack(anchor="nw")
        self.input_text = tk.Text(input_frame, height=8, font=("Segoe UI", 13), undo=True)
        self.input_text.pack(fill="both", expand=True, pady=5)
        self.input_text.bind("<KeyRelease>", lambda e: self.on_input_change())

        output_frame = ttk.Frame(self.root, padding=10)
        output_frame.pack(fill="both", expand=True)
        ttk.Label(output_frame, text="Khoa Đẩu (Màu đỏ = Sai lệch so với Rule-based):").pack(anchor="nw")
        self.output_text = tk.Text(output_frame, height=12, font=("Khoa Dau Regular v3", 26), wrap="word")
        self.output_text.pack(fill="both", expand=True, pady=5)
        self.output_text.tag_config("mismatch", foreground="red")
        
        self.status_var = tk.StringVar(value="Sẵn sàng")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")

    def change_model(self):
        m_name = self.model_var.get()
        self.status_var.set(f"Đang kích hoạt {m_name}...")
        self.root.update_idletasks()
        
        if m_name not in ["Rule-based", "Dictionary"] and m_name not in self.models:
            mapping = {
                "Shallow_Big": (KhoaDauCNN.shallow_big, MODEL_SHALLOW_BIG_PATH),
                "Big": (KhoaDauCNN.big, MODEL_BIG_PATH),
                "Shallow_Small": (KhoaDauCNN.shallow_small, MODEL_SHALLOW_SMALL_PATH),
                "Small": (KhoaDauCNN.small, MODEL_SMALL_PATH)
            }
            model_fn, path = mapping[m_name]
            model = model_fn(len(self.qn_v), len(self.kd_v)).to(self.device)
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            self.models[m_name] = model
            
        self.status_var.set(f"Đã kích hoạt {m_name}")
        self.on_input_change()

    def transliterate_syllable_ai(self, model, syllable):
        clean_s = fast_norm(syllable)
        encoded = [self.qn_v.get(c, 1) for c in clean_s][:MAX_LEN]
        if len(encoded) < MAX_LEN:
            encoded += [self.qn_v["<PAD>"]] * (MAX_LEN - len(encoded))
        
        input_tensor = torch.tensor([encoded]).to(self.device)
        with torch.no_grad():
            preds = torch.argmax(model(input_tensor), dim=-1)[0]
        return "".join([self.rev_kd.get(p.item(), "") for p in preds if p > 1])

    def on_input_change(self):
        content = self.input_text.get("1.0", tk.END).strip()
        if not content:
            self.output_text.delete("1.0", tk.END)
            return

        m_name = self.model_var.get()
        tokens = re.findall(r'\w+|[^\w\s]|\s+', content)
        self.output_text.delete("1.0", tk.END)
        
        for token in tokens:
            if re.match(r'\w+', token):
                clean_token = unicodedata.normalize("NFC", token.lower())
                is_vn = clean_token in self.dict_mapping
                truth = encode_custom(token) if is_vn else token
                
                if not is_vn: pred = token
                elif m_name == "Rule-based": pred = truth
                elif m_name == "Dictionary": pred = self.dict_mapping.get(clean_token, token)
                else: pred = self.transliterate_syllable_ai(self.models[m_name], token)
                
                tag = "mismatch" if pred != truth else ""
                self.output_text.insert(tk.END, pred, tag)
            else:
                self.output_text.insert(tk.END, token)

    def random_sample(self):
        sample = random.choice(self.sentences)
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert(tk.END, sample)
        self.on_input_change()

if __name__ == "__main__":
    root = tk.Tk()
    app = KhoaDauInferenceApp(root)
    root.mainloop()
