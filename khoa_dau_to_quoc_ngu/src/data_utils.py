import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from .config import MAX_LEN
from .vocab_utils import PAD_IDX

class TranslationDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=MAX_LEN):
        # Đọc dữ liệu
        print(f"Đang tải dữ liệu từ {csv_file}...")
        df = pd.read_csv(csv_file, usecols=["khoa_dau", "quoc_ngu"]).dropna()
        
        khoa_dau_texts = df["khoa_dau"].astype(str).tolist()
        quoc_ngu_texts = df["quoc_ngu"].astype(str).tolist()
        del df # Giải phóng RAM
        
        print(f"Đang Pre-tokenize (mã hóa) {len(khoa_dau_texts)} câu vào RAM...")
        self.src_data = []
        self.tgt_data = []
        
        for i in tqdm(range(len(khoa_dau_texts)), desc="Tokenizing"):
            src_ids = tokenizer.encode(khoa_dau_texts[i])
            tgt_ids = tokenizer.encode(quoc_ngu_texts[i])
            
            # Cắt bớt nếu vượt quá max_len (giữ lại EOS token ở cuối)
            if len(src_ids) > max_len:
                src_ids = src_ids[:max_len-1] + [src_ids[-1]]
            if len(tgt_ids) > max_len:
                tgt_ids = tgt_ids[:max_len-1] + [tgt_ids[-1]]
                
            # Lưu dưới dạng mảng số nguyên (List of int) thay vì Tensor để tránh lỗi tràn File Descriptor khi chạy đa luồng
            self.src_data.append(src_ids)
            self.tgt_data.append(tgt_ids)
            
    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        # Trả về chuỗi độ dài thực (Chưa đệm PAD)
        return {
            "src": self.src_data[idx],
            "tgt": self.tgt_data[idx]
        }

def collate_fn(batch):
    """
    Kỹ thuật Dynamic Padding (Đệm động): 
    Tìm câu dài nhất trong TỪNG BATCH và chỉ đệm PAD cho các câu khác bằng đúng độ dài đó.
    Tiết kiệm hàng chục lần khối lượng tính toán cho GPU.
    """
    # Chuyển Python List thành Tensor ngay tại đây (an toàn cho đa luồng)
    src_list = [torch.tensor(item['src'], dtype=torch.long) for item in batch]
    tgt_list = [torch.tensor(item['tgt'], dtype=torch.long) for item in batch]
    
    # pad_sequence tự động đệm tới chiều dài lớn nhất trong list
    # batch_first=True trả về [batch_size, max_seq_len]
    src_padded = pad_sequence(src_list, padding_value=PAD_IDX, batch_first=True)
    tgt_padded = pad_sequence(tgt_list, padding_value=PAD_IDX, batch_first=True)
    
    return {
        "src": src_padded,
        "tgt": tgt_padded
    }

def get_dataloader(csv_file, tokenizer, batch_size, shuffle=True, limit=None):
    dataset = TranslationDataset(csv_file, tokenizer)
    
    if limit and limit < len(dataset):
        # Lấy mẫu ngẫu nhiên để đảm bảo đa dạng
        indices = torch.randperm(len(dataset))[:limit]
        dataset = torch.utils.data.Subset(dataset, indices)
        
    # Linux (Cloud/Colab) hỗ trợ multiprocessing rất tốt, Windows có thể bị nghẽn
    num_workers = 4 if os.name != 'nt' else 0
        
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True,
        collate_fn=collate_fn  # Kích hoạt Dynamic Padding
    )
