import torch
import torch.nn as nn
import pandas as pd
import unicodedata
import os

# --- SAO CHÉP LẠI CÁC THÔNG SỐ VÀ KIẾN TRÚC ---
MAX_LEN = 7
MODEL_PATH = "quoc_ngu_to_khoa_dau/khoa_dau_cnn_large.pth"
VOCAB_PATH = "quoc_ngu_to_khoa_dau/vocab.pth"

class KhoaDauCNNLarge(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_dim=128, hidden_dim=256):
        super(KhoaDauCNNLarge, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, output_vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.transpose(1, 2)
        logits = self.fc(x)
        return logits

# --- TIỀN XỬ LÝ ---
def remove_tone_marks(text):
    TONE_MARKS = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}
    normalized = unicodedata.normalize("NFD", text.lower())
    result = "".join([ch for ch in normalized if ch not in TONE_MARKS])
    return unicodedata.normalize("NFC", result)

class TransliterativeInference:
    def __init__(self):
        # Load vocab
        vocab_data = torch.load(VOCAB_PATH)
        self.qn_vocab = vocab_data['qn']
        self.kd_vocab = vocab_data['kd']
        self.rev_kd_vocab = {v: k for k, v in self.kd_vocab.items()}
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = KhoaDauCNNLarge(len(self.qn_vocab), len(self.kd_vocab)).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.eval()

    def predict_syllable(self, syllable):
        clean_s = remove_tone_marks(syllable)
        encoded = [self.qn_vocab.get(c, self.qn_vocab["<UNK>"]) for c in clean_s]
        if len(encoded) < MAX_LEN:
            encoded += [self.qn_vocab["<PAD>"]] * (MAX_LEN - len(encoded))
        else:
            encoded = encoded[:MAX_LEN]
        
        input_tensor = torch.tensor([encoded]).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_tensor) # [1, 7, vocab_size]
            predictions = torch.argmax(outputs, dim=-1)[0] # [7]
            
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

# --- CHƯƠNG TRÌNH CHÍNH ---
if __name__ == "__main__":
    translator = TransliterativeInference()
    print("Mô hình đã sẵn sàng. Nhập 'exit' để thoát.")
    while True:
        text = input("Quốc ngữ: ")
        if text.lower() == 'exit':
            break
        result = translator.transliterate(text)
        print(f"Khoa Đẩu: {result}")
        # In mã hex để kiểm tra nếu không có font
        hex_codes = " ".join(["-".join([f"{ord(c):04x}" for c in w]) for w in result.split()])
        print(f"Hex:      {hex_codes}")
