import os
import torch
import torch.nn as nn
from src.config import *
from src.vocab_utils import CharTokenizer, PAD_IDX, SOS_IDX, EOS_IDX
from src.models import Seq2SeqTransformer, generate_square_subsequent_mask
from evaluate import beam_search_decode

class KhoaDauTranslator:
    def __init__(self, model_path=None, device=None):
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.tokenizer = CharTokenizer()
        if not self.tokenizer.load(VOCAB_PATH):
            raise FileNotFoundError(f"Không tìm thấy file từ điển tại {VOCAB_PATH}")
            
        self.model = Seq2SeqTransformer(
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            emb_size=D_MODEL,
            nhead=N_HEAD,
            vocab_size=self.tokenizer.vocab_size,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=0.0 # Tắt dropout khi inference
        ).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"-> Đã nạp mô hình từ {model_path}")
        else:
            best_model = os.path.join(WEIGHTS_DIR, "best_model.pth")
            if os.path.exists(best_model):
                self.model.load_state_dict(torch.load(best_model, map_location=self.device))
                print(f"-> Đã nạp mô hình tốt nhất từ {best_model}")
            else:
                print("! Cảnh báo: Chưa có file trọng số. Vui lòng huấn luyện mô hình trước.")
        
        self.model.eval()

    def translate(self, text, beam_width=3, max_len=MAX_LEN):
        if not text.strip():
            return ""
            
        src_ids = self.tokenizer.encode(text)
        src_tensor = torch.tensor(src_ids).unsqueeze(1).to(self.device)
        num_tokens = src_tensor.shape[0]
        src_mask = torch.zeros((num_tokens, num_tokens), device=self.device).type(torch.bool)
        
        with torch.no_grad():
            out_ids = beam_search_decode(
                self.model, src_tensor, src_mask, 
                max_len=max_len, 
                start_symbol=SOS_IDX, 
                end_symbol=EOS_IDX, 
                device=self.device, 
                beam_width=beam_width
            )
            
        return self.tokenizer.decode(out_ids)

def main():
    translator = KhoaDauTranslator()
    print("\n--- Hệ thống Chuyển tự Khoa Đẩu -> Quốc ngữ (AI) ---")
    print("Nhập 'exit' để thoát.\n")
    
    while True:
        text = input("Khoa Đẩu: ")
        if text.lower() == 'exit':
            break
            
        result = translator.translate(text)
        print(f"Quốc ngữ: {result}\n")

if __name__ == "__main__":
    main()
