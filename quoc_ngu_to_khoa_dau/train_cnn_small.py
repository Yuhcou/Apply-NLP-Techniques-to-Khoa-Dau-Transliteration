import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from vocab_utils import get_or_build_vocab

# --- CẤU HÌNH ---
MAX_LEN = 7
CSV_PATH = "quoc_ngu_to_khoa_dau/data/all-vietnamese-syllables-encoded.csv"
MODEL_SAVE_PATH = "quoc_ngu_to_khoa_dau/khoa_dau_cnn_small.pth"

class KhoaDauDataset(Dataset):
    def __init__(self, df, qn_vocab, kd_vocab, max_len=MAX_LEN):
        self.data = df.dropna()
        self.qn_v, self.kd_v = qn_vocab, kd_vocab
        self.max_len = max_len

    def encode(self, text, vocab):
        encoded = [vocab.get(c, vocab["<UNK>"]) for c in str(text)]
        if len(encoded) < self.max_len:
            encoded += [vocab["<PAD>"]] * (self.max_len - len(encoded))
        else:
            encoded = encoded[:self.max_len]
        return torch.tensor(encoded)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return self.encode(self.data.iloc[idx]['original'], self.qn_v), self.encode(self.data.iloc[idx]['encoded'], self.kd_v)

class KhoaDauCNNSmall(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_dim=64, hidden_dim=256):
        super(KhoaDauCNNSmall, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_vocab_size)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return self.fc(x.transpose(1, 2))

def train():
    df = pd.read_csv(CSV_PATH)
    qn_v, kd_v = get_or_build_vocab(df)
    
    dataset = KhoaDauDataset(df, qn_v, kd_v)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = KhoaDauCNNSmall(len(qn_v), len(kd_v)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Train SMALL trên {device}...")
    for epoch in range(100):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).view(-1, len(kd_v)), y.view(-1))
            loss.backward(); optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {total_loss/len(dataloader):.5f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Done SMALL. Model tại: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
