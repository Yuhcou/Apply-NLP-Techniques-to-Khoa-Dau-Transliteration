import torch
import torch.nn as nn

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
