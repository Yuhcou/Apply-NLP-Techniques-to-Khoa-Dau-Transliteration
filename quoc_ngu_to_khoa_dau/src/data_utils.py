import torch
import unicodedata
import pandas as pd
from torch.utils.data import Dataset
from .config import MAX_LEN

def fast_norm(text):
    """Chuẩn hóa tiếng Việt và loại bỏ dấu."""
    TONE_MARKS_MAP = {ord(c): None for c in "\u0300\u0301\u0303\u0309\u0323"}
    normalized = unicodedata.normalize("NFD", str(text).lower())
    result = "".join([ch for ch in normalized if ch not in TONE_MARKS_MAP])
    return unicodedata.normalize("NFC", result)

class KhoaDauDataset(Dataset):
    def __init__(self, df, qn_vocab, kd_vocab):
        self.data = df.dropna()
        self.qn_v = qn_vocab
        self.kd_v = kd_vocab

    def encode(self, text, vocab):
        encoded = [vocab.get(c, vocab["<UNK>"]) for c in str(text)]
        if len(encoded) < MAX_LEN:
            encoded += [vocab["<PAD>"]] * (MAX_LEN - len(encoded))
        else:
            encoded = encoded[:MAX_LEN]
        return torch.tensor(encoded)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        qn = str(self.data.iloc[idx]['original'])
        kd = str(self.data.iloc[idx]['encoded'])
        return self.encode(qn, self.qn_v), self.encode(kd, self.kd_v)
