import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from .config import MAX_LEN
from .vocab_utils import PAD_IDX

class TranslationDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=MAX_LEN):
        # Đọc dữ liệu và bỏ các dòng NaN
        print(f"Đang tải dữ liệu từ {csv_file}...")
        # Sử dụng low_memory=True và chỉ nạp các cột cần thiết để tiết kiệm RAM
        self.df = pd.read_csv(csv_file, usecols=["khoa_dau", "quoc_ngu"]).dropna()
        # Chuyển sang danh sách python để truy cập index nhanh hơn so với iloc trên DataFrame lớn
        self.khoa_dau = self.df["khoa_dau"].astype(str).tolist()
        self.quoc_ngu = self.df["quoc_ngu"].astype(str).tolist()
        
        # Giải phóng DataFrame gốc để tiết kiệm RAM
        del self.df
        
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.khoa_dau)

    def __getitem__(self, idx):
        src_text = self.khoa_dau[idx]
        tgt_text = self.quoc_ngu[idx]

        src_ids = self.tokenizer.encode(src_text)
        tgt_ids = self.tokenizer.encode(tgt_text)

        # Cắt bớt nếu vượt quá max_len (giữ lại EOS token ở cuối)
        if len(src_ids) > self.max_len:
            src_ids = src_ids[:self.max_len-1] + [src_ids[-1]]
        if len(tgt_ids) > self.max_len:
            tgt_ids = tgt_ids[:self.max_len-1] + [tgt_ids[-1]]

        # Pad chuỗi cho đủ max_len
        src_padded = src_ids + [PAD_IDX] * (self.max_len - len(src_ids))
        tgt_padded = tgt_ids + [PAD_IDX] * (self.max_len - len(tgt_ids))

        return {
            # Transformer ở PyTorch yêu cầu input dạng [seq_len, batch_size] 
            # Dataloader sẽ batch lại thành [batch_size, seq_len], ta sẽ transpose sau.
            "src": torch.tensor(src_padded, dtype=torch.long),
            "tgt": torch.tensor(tgt_padded, dtype=torch.long)
        }

def get_dataloader(csv_file, tokenizer, batch_size, shuffle=True, num_workers=0, limit=None):
    dataset = TranslationDataset(csv_file, tokenizer)
    
    if limit and limit < len(dataset):
        # Lấy mẫu ngẫu nhiên để đảm bảo đa dạng
        indices = torch.randperm(len(dataset))[:limit]
        dataset = torch.utils.data.Subset(dataset, indices)
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)
