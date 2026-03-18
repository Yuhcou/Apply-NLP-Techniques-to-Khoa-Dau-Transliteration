import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

MAX_LEN = 7
CSV_PATH = "quoc_ngu_to_khoa_dau/data/all-vietnamese-syllables-encoded.csv"
MODEL_SAVE_PATH = "quoc_ngu_to_khoa_dau/khoa_dau_cnn_nano.pth"
VOCAB_SAVE_PATH = "quoc_ngu_to_khoa_dau/vocab.pth"

class KhoaDauCNNNano(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_dim=16, hidden_dim=48):
        super(KhoaDauCNNNano, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_vocab_size)
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.fc(x.transpose(1, 2))

def build_vocab(df):
    qn_texts = df['original'].dropna().astype(str).tolist()
    kd_texts = df['encoded'].dropna().astype(str).tolist()
    qn_v = {c: i+2 for i, c in enumerate(sorted(list(set("".join(qn_texts)))))}
    qn_v["<PAD>"], qn_v["<UNK>"] = 0, 1
    kd_v = {c: i+2 for i, c in enumerate(sorted(list(set("".join(kd_texts)))))}
    kd_v["<PAD>"], kd_v["<UNK>"] = 0, 1
    return qn_v, kd_v

class KhoaDauDataset(Dataset):
    def __init__(self, df, qn_vocab, kd_vocab):
        self.data = df.dropna()
        self.qn_v, self.kd_v = qn_vocab, kd_vocab
    def encode(self, text, vocab):
        encoded = [vocab.get(c, 1) for c in str(text)][:MAX_LEN]
        return torch.tensor(encoded + [0] * (MAX_LEN - len(encoded)))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return self.encode(self.data.iloc[idx]['original'], self.qn_v), self.encode(self.data.iloc[idx]['encoded'], self.kd_v)

def train():
    df = pd.read_csv(CSV_PATH)
    qn_v, kd_v = build_vocab(df)
    dataset = KhoaDauDataset(df, qn_v, kd_v)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KhoaDauCNNNano(len(qn_v), len(kd_v)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    print(f"Train NANO-v4 (Hidden 48) on {device}...")
    for epoch in range(200): # Tăng epoch để ép Accuracy lên cao
        model.train()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).view(-1, len(kd_v)), y.view(-1))
            loss.backward(); optimizer.step()
        if (epoch+1) % 50 == 0: print(f"Epoch {epoch+1}/200")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    torch.save({'qn': qn_v, 'kd': kd_v}, VOCAB_SAVE_PATH)
    print("Done NANO-v4.")

if __name__ == "__main__":
    train()
