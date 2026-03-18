import torch
import pandas as pd
import os
from src.config import MAX_LEN, VOCAB_PATH, MODEL_LARGE_PATH
from src.models import KhoaDauCNNLarge
from src.data_utils import fast_norm

class TransliterativeInference:
    def __init__(self, model_path=MODEL_LARGE_PATH):
        if not os.path.exists(VOCAB_PATH):
            raise FileNotFoundError(f"Vocab file not found at {VOCAB_PATH}")
            
        # Load vocab
        vocab_data = torch.load(VOCAB_PATH)
        self.qn_vocab = vocab_data['qn']
        self.kd_vocab = vocab_data['kd']
        self.rev_kd_vocab = {v: k for k, v in self.kd_vocab.items()}
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = KhoaDauCNNLarge(len(self.qn_vocab), len(self.kd_vocab)).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict_syllable(self, syllable):
        clean_s = fast_norm(syllable)
        encoded = [self.qn_vocab.get(c, self.qn_vocab["<UNK>"]) for c in clean_s]
        if len(encoded) < MAX_LEN:
            encoded += [self.qn_vocab["<PAD>"]] * (MAX_LEN - len(encoded))
        else:
            encoded = encoded[:MAX_LEN]
        
        input_tensor = torch.tensor([encoded]).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor)
            predictions = torch.argmax(outputs, dim=-1)[0]
            
        result = ""
        for p in predictions.tolist():
            char = self.rev_kd_vocab.get(p, "")
            if char not in ["<PAD>", "<UNK>"]:
                result += char
        return result

    def transliterate(self, text):
        words = text.split()
        results = [self.predict_syllable(w) for w in words]
        return " ".join(results)

if __name__ == "__main__":
    translator = TransliterativeInference()
    print("Mô hình đã sẵn sàng. Nhập 'exit' để thoát.")
    while True:
        text = input("Quốc ngữ: ")
        if text.lower() == 'exit':
            break
        result = translator.transliterate(text)
        print(f"Khoa Đẩu: {result}")
        hex_codes = " ".join(["-".join([f"{ord(c):04x}" for c in w]) for w in result.split()])
        print(f"Hex:      {hex_codes}")
