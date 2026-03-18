import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from vocab_utils import get_or_build_vocab

# --- CẤU HÌNH ---
MAX_LEN = 7
CSV_PATH = "quoc_ngu_to_khoa_dau/data/all-vietnamese-syllables-encoded.csv"
MODEL_SAVE_PATH = "quoc_ngu_to_khoa_dau/khoa_dau_cnn_large.pth"

# --- DATASET ---
class KhoaDauDataset(Dataset):
    def __init__(self, df, qn_vocab, kd_vocab, max_len=MAX_LEN):
        self.data = df
        self.qn_vocab = qn_vocab
        self.kd_vocab = kd_vocab
        self.max_len = max_len

    def encode(self, text, vocab):
        encoded = [vocab.get(c, vocab["<UNK>"]) for c in str(text)]
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
class KhoaDauCNNLarge(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_dim=128, hidden_dim=512):
        super(KhoaDauCNNLarge, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_dim, output_vocab_size)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.transpose(1, 2)
        return self.fc(self.dropout(x))

# --- HUẤN LUYỆN ---
def train():
    df = pd.read_csv(CSV_PATH)
    qn_vocab, kd_vocab = get_or_build_vocab(df)
    
    dataset = KhoaDauDataset(df, qn_vocab, kd_vocab)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KhoaDauCNNLarge(len(qn_vocab), len(kd_vocab)).to(device)
    
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    print(f"Bắt đầu huấn luyện LARGE trên {device}...")
    for epoch in range(100):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).view(-1, len(kd_vocab)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {avg_loss:.5f}, LR: {optimizer.param_groups[0]['lr']}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Hoàn tất. Model lưu tại: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
