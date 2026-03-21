import json
import os
from tqdm import tqdm

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

class CharTokenizer:
    def __init__(self):
        self.char2id = {PAD_TOKEN: PAD_IDX, SOS_TOKEN: SOS_IDX, EOS_TOKEN: EOS_IDX, UNK_TOKEN: UNK_IDX}
        self.id2char = {PAD_IDX: PAD_TOKEN, SOS_IDX: SOS_TOKEN, EOS_IDX: EOS_TOKEN, UNK_IDX: UNK_TOKEN}
        self.vocab_size = len(self.char2id)

    def build_vocab(self, texts):
        """Xây dựng từ điển dựa trên một danh sách các chuỗi."""
        print("Đang xây dựng từ điển ký tự...")
        unique_chars = set()
        for text in tqdm(texts, desc="Trích xuất ký tự"):
            if isinstance(text, str):
                unique_chars.update(list(text))
        
        for char in sorted(list(unique_chars)):
            if char not in self.char2id:
                self.char2id[char] = self.vocab_size
                self.id2char[self.vocab_size] = char
                self.vocab_size += 1
                
        print(f"Hoàn tất! Tổng số ký tự trong từ điển: {self.vocab_size}")

    def encode(self, text, add_special_tokens=True):
        """Mã hóa một chuỗi thành list of IDs."""
        if not isinstance(text, str):
            text = ""
        ids = [self.char2id.get(c, UNK_IDX) for c in text]
        if add_special_tokens:
            ids = [SOS_IDX] + ids + [EOS_IDX]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        """Giải mã list of IDs thành chuỗi ký tự."""
        chars = []
        for i in ids:
            # Nếu truyền vào tensor thay vì số int
            if hasattr(i, "item"):
                i = i.item()
            
            if skip_special_tokens and i in [PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX]:
                continue
            chars.append(self.id2char.get(i, UNK_TOKEN))
        return "".join(chars)
        
    def save(self, filepath):
        """Lưu từ điển ra file JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "char2id": self.char2id, 
                "id2char": {str(k): v for k, v in self.id2char.items()}
            }, f, ensure_ascii=False, indent=2)
            
    def load(self, filepath):
        """Tải từ điển từ file JSON."""
        if not os.path.exists(filepath):
            return False
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.char2id = data["char2id"]
            self.id2char = {int(k): v for k, v in data["id2char"].items()}
            self.vocab_size = len(self.char2id)
        return True
