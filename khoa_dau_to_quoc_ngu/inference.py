import os
import sys
import torch
import tkinter as tk
from tkinter import ttk, messagebox
import re
import random
import unicodedata
import ctypes
from difflib import SequenceMatcher

# Thêm đường dẫn để import từ các module khác
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *
from src.vocab_utils import CharTokenizer, SOS_IDX, EOS_IDX
from src.models import Seq2SeqTransformer
from evaluate import beam_search_decode

# Import Rule-based từ Module 1
from quoc_ngu_to_khoa_dau.src.rule_based import encode_safe

# --- LOAD FONT KHOA DAU (Windows) ---
FONT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "misc", "font", "KhoaDau-Regular_v3.otf"))

def load_custom_font(font_path):
    if os.path.exists(font_path):
        FR_PRIVATE = 0x10
        ctypes.windll.gdi32.AddFontResourceExW(font_path, FR_PRIVATE, 0)
    else:
        print(f"Cảnh báo: Không tìm thấy font tại {font_path}")

class KhoaDauToQuocNguApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Khoa Đẩu to Quốc Ngữ - AI Transliteration v2.0")
        self.root.geometry("1000x800")
        
        load_custom_font(FONT_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Tokenizer & Model
        self.tokenizer = CharTokenizer()
        if not self.tokenizer.load(VOCAB_PATH):
            messagebox.showerror("Lỗi", f"Không tìm thấy từ điển tại {VOCAB_PATH}")
            sys.exit(1)
            
        self.model = Seq2SeqTransformer(
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            emb_size=D_MODEL,
            nhead=N_HEAD,
            vocab_size=self.tokenizer.vocab_size,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=0.0
        ).to(self.device)
        
        model_path = os.path.join(WEIGHTS_DIR, "best_model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        else:
            messagebox.showwarning("Cảnh báo", "Không tìm thấy file trọng số best_model.pth")

        # 2. Load Corpus cho câu ngẫu nhiên
        self.corpus = self._load_corpus()
        
        self.setup_ui()

    def _load_corpus(self):
        source_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "quoc_ngu_to_khoa_dau", "data", "source.txt"))
        if os.path.exists(source_path):
            with open(source_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Tách câu đơn giản
                sentences = re.split(r'[.\n]+', content)
                return [s.strip() for s in sentences if 5 <= len(s.split()) <= 25]
        return ["Chào mừng bạn đến với hệ thống chuyển tự chữ Khoa Đẩu ."]

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=15)
        main_frame.pack(fill="both", expand=True)

        # --- Phía trên: Điều khiển ---
        ctrl_frame = ttk.Frame(main_frame)
        ctrl_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(ctrl_frame, text="Nhập Quốc ngữ để tự động chuyển sang Khoa Đẩu:", font=("Segoe UI", 10, "italic")).pack(side="left")
        ttk.Button(ctrl_frame, text="🎲 Câu ngẫu nhiên", command=self.random_sample).pack(side="right")

        # --- 1. Input Box (Quốc ngữ) ---
        ttk.Label(main_frame, text="1. Văn bản Gốc (Quốc ngữ):").pack(anchor="w")
        self.input_text = tk.Text(main_frame, height=5, font=("Segoe UI", 13))
        self.input_text.pack(fill="x", pady=5)
        self.input_text.bind("<KeyRelease>", lambda e: self.on_content_change())

        # --- 2. Middle Box (Khoa Đẩu - Hiển thị trung gian) ---
        ttk.Label(main_frame, text="2. Chữ Khoa Đẩu (Trung gian - Rule-based):").pack(anchor="w")
        self.mid_text = tk.Text(main_frame, height=5, font=("Khoa Dau Regular v3", 24), wrap="word", bg="#f9f9f9")
        self.mid_text.pack(fill="x", pady=5)
        
        # --- 3. Output Box (AI Quốc ngữ) ---
        ttk.Label(main_frame, text="3. Kết quả AI (Quốc ngữ - Highlight lỗi so với gốc):").pack(anchor="w")
        self.output_text = tk.Text(main_frame, height=8, font=("Segoe UI", 14), wrap="word")
        self.output_text.pack(fill="both", expand=True, pady=5)
        self.output_text.tag_config("error", foreground="red", underline=True)
        self.output_text.tag_config("missing", foreground="blue", font=("Segoe UI", 12, "bold"))

        # Thanh trạng thái
        self.status_var = tk.StringVar(value="Sẵn sàng")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w").pack(side="bottom", fill="x")

    def preprocess_for_ai(self, text):
        """Tách dấu câu để AI xử lý đúng theo logic data_maker."""
        text = text.lower()
        # Punctuation Padding
        text = re.sub(r'([.,!?():;])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def postprocess_from_ai(self, text):
        """Gộp lại dấu câu cho tự nhiên."""
        text = re.sub(r'\s+([.,!?():;])', r'\1', text)
        return text

    def postprocess_from_ai(self, text):
        """Gộp lại dấu câu cho tự nhiên theo quy tắc tiếng Việt."""
        # 1. Xóa khoảng trắng TRƯỚC các dấu ngắt: . , ! ? : ; )
        text = re.sub(r'\s+([.,!?\):;])', r'\1', text)
        # 2. Xóa khoảng trắng SAU dấu mở ngoặc: (
        text = re.sub(r'\(\s+', r'(', text)
        # 3. Đảm bảo có khoảng trắng TRƯỚC dấu mở ngoặc (ví dụ: word( -> word ()
        text = re.sub(r'(\w)\(', r'\1 (', text)
        return text

    def on_content_change(self):
        qn_input = self.input_text.get("1.0", tk.END).strip()
        if not qn_input:
            self.mid_text.delete("1.0", tk.END)
            self.output_text.delete("1.0", tk.END)
            return

        # Bước 1: Quốc ngữ -> Khoa Đẩu (Rule-based)
        kd_mid = encode_safe(qn_input)
        self.mid_text.delete("1.0", tk.END)
        self.mid_text.insert(tk.END, kd_mid)

        # Bước 2: Tách đoạn văn thành các câu để dịch riêng biệt
        sentences = re.split(r'(?<=[.?!])\s+|\n+', qn_input)
        
        full_ai_output = []
        self.status_var.set(f"AI đang dịch {len(sentences)} câu...")
        self.root.update_idletasks()
        
        try:
            for s_qn in sentences:
                if not s_qn.strip(): continue
                
                # Chuyển từng câu sang Khoa Đẩu trung gian
                s_kd = encode_safe(s_qn)
                
                # Tiền xử lý (Padding dấu câu cho đúng chuẩn data huấn luyện)
                ai_input = self.preprocess_for_ai(s_kd)
                
                # Thêm dấu chấm giả nếu thiếu
                has_end_punc = re.search(r'[.?!]$', ai_input)
                if not has_end_punc:
                    ai_input += " ."
                
                src_ids = self.tokenizer.encode(ai_input)
                src_tensor = torch.tensor(src_ids).unsqueeze(1).to(self.device)
                num_tokens = src_tensor.shape[0]
                src_mask = torch.zeros((num_tokens, num_tokens), device=self.device).type(torch.bool)
                
                with torch.no_grad():
                    out_ids = beam_search_decode(
                        self.model, src_tensor, src_mask, 
                        max_len=MAX_LEN, 
                        start_symbol=SOS_IDX, 
                        end_symbol=EOS_IDX, 
                        device=self.device, 
                        beam_width=3
                    )
                
                s_output_raw = self.tokenizer.decode(out_ids)
                s_output = self.postprocess_from_ai(s_output_raw)
                
                # Xóa dấu chấm giả
                if not has_end_punc and s_output.endswith("."):
                    s_output = s_output[:-1].strip()
                
                full_ai_output.append(s_output)
            
            # Ghép các câu lại
            final_output = " ".join(full_ai_output)
            
            # Bước 4: Hiển thị so sánh (Highlight lỗi thực tế của AI)
            self.display_comparison(qn_input, final_output)
            self.status_var.set(f"Hoàn tất ({len(sentences)} câu)")
            
        except Exception as e:
            self.status_var.set(f"Lỗi: {str(e)}")
            import traceback
            traceback.print_exc()

    def display_comparison(self, original, predicted):
        self.output_text.delete("1.0", tk.END)
        
        orig_tokens = re.findall(r'\w+|[^\w\s]|\s+', original.lower())
        pred_tokens = re.findall(r'\w+|[^\w\s]|\s+', predicted.lower())
        
        orig_words = [t for t in orig_tokens if re.match(r'\w+', t)]
        pred_words = [t for t in pred_tokens if re.match(r'\w+', t)]
        
        sm = SequenceMatcher(None, orig_words, pred_words)
        
        # word_status: index trong pred_words -> tag ("" hoặc "error")
        word_status = {}
        # missing_markers: index trong pred_words -> danh sách các cụm từ bị thiếu TRƯỚC từ đó
        missing_markers = {i: [] for i in range(len(pred_words) + 1)}
        
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                for j in range(j1, j2):
                    word_status[j] = ""
            elif tag == 'replace':
                for j in range(j1, j2):
                    word_status[j] = "error"
                missing_markers[j1].append(f"[ {' '.join(orig_words[i1:i2])} ]")
            elif tag == 'insert':
                for j in range(j1, j2):
                    word_status[j] = "error"
            elif tag == 'delete':
                missing_markers[j1].append(f"[ {' '.join(orig_words[i1:i2])} ]")

        # Duyệt qua pred_tokens để hiển thị lên UI
        curr_word_idx = 0
        for token in pred_tokens:
            if re.match(r'\w+', token):
                # Chèn marker thiếu TRƯỚC từ hiện tại nếu có
                for marker in missing_markers[curr_word_idx]:
                    self.output_text.insert(tk.END, marker + " ", "missing")
                
                status = word_status.get(curr_word_idx, "error")
                self.output_text.insert(tk.END, token, status)
                curr_word_idx += 1
            else:
                self.output_text.insert(tk.END, token)
        
        # Chèn marker thiếu ở cuối văn bản nếu AI dịch thiếu phần cuối
        for marker in missing_markers[curr_word_idx]:
            self.output_text.insert(tk.END, " " + marker, "missing")


    def random_sample(self):
        if self.corpus:
            sample = random.choice(self.corpus)
            self.input_text.delete("1.0", tk.END)
            self.input_text.insert(tk.END, sample)
            self.on_content_change()

if __name__ == "__main__":
    root = tk.Tk()
    # Giảm kích thước cửa sổ một chút để vừa màn hình
    root.geometry("1000x700")
    app = KhoaDauToQuocNguApp(root)
    root.mainloop()
