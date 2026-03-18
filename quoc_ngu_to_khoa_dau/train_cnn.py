import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

# --- CẤU HÌNH ---
MAX_LEN = 7
CSV_PATH = "quoc_ngu_to_khoa_dau/data/all-vietnamese-syllables-encoded.csv"
MODEL_SAVE_PATH = "quoc_ngu_to_khoa_dau/khoa_dau_cnn.pth"
VOCAB_SAVE_PATH = "quoc_ngu_to_khoa_dau/vocab.pth"

# --- TẠO VOCAB ---
def build_vocab(df):
    # Loại bỏ NaN và chuyển về string
    qn_texts = df['original'].dropna().astype(str).tolist()
    kd_texts = df['encoded'].dropna().astype(str).tolist()
    
    qn_chars = sorted(list(set("".join(qn_texts))))
    kd_chars = sorted(list(set("".join(kd_texts))))
    
    # Thêm ký tự PADDING (<PAD>) và ký tự lạ (<UNK>)
    qn_vocab = {char: i+2 for i, char in enumerate(qn_chars)}
    qn_vocab["<PAD>"] = 0
    qn_vocab["<UNK>"] = 1
    
    kd_vocab = {char: i+2 for i, char in enumerate(kd_chars)}
    kd_vocab["<PAD>"] = 0
    kd_vocab["<UNK>"] = 1
    
    return qn_vocab, kd_vocab

# --- DATASET ---
class KhoaDauDataset(Dataset):
    def __init__(self, df, qn_vocab, kd_vocab, max_len=MAX_LEN):
        self.data = df
        self.qn_vocab = qn_vocab
        self.kd_vocab = kd_vocab
        self.max_len = max_len

    def encode(self, text, vocab):
        encoded = [vocab.get(c, vocab["<UNK>"]) for c in text]
        # Padding hoặc cắt ngắn về MAX_LEN
        if len(encoded) < self.max_len:
            encoded += [vocab["<PAD>"]] * (self.max_len - len(encoded))
        else:
            encoded = encoded[:self.max_len]
        return torch.tensor(encoded)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        qn = str(self.data.iloc[idx]['original']) if pd.notnull(self.data.iloc[idx]['original']) else ""
        kd = str(self.data.iloc[idx]['encoded']) if pd.notnull(self.data.iloc[idx]['encoded']) else ""
        
        return self.encode(qn, self.qn_vocab), self.encode(kd, self.kd_vocab)

# --- KIẾN TRÚC MÔ HÌNH 1D-CNN ---
class KhoaDauCNN(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_dim=128, hidden_dim=512):
        super(KhoaDauCNN, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim)
        
        # Lớp CNN rộng hơn
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1) # Giảm dropout để mô hình nhớ tốt hơn
        
        self.fc = nn.Linear(hidden_dim, output_vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.transpose(1, 2)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

# --- HUẤN LUYỆN ---
def train():
    df = pd.read_csv(CSV_PATH)
    
    qn_vocab, kd_vocab = build_vocab(df)
    dataset = KhoaDauDataset(df, qn_vocab, kd_vocab)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KhoaDauCNN(len(qn_vocab), len(kd_vocab)).to(device)
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    print(f"Bắt đầu huấn luyện trên {device}...")
    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            
            loss = criterion(outputs.view(-1, outputs.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.5f}, LR: {optimizer.param_groups[0]['lr']}")

    # Lưu Model và Vocab
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    torch.save({'qn': qn_vocab, 'kd': kd_vocab}, VOCAB_SAVE_PATH)
    print("Huấn luyện hoàn tất và đã lưu model.")

if __name__ == "__main__":
    train()
