import torch
import torch.nn as nn

class KhoaDauCNN(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, num_layers=3, embed_dim=128, hidden_dim=512, kernel_size=3, dropout=0.0):
        super(KhoaDauCNN, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_dim)
        
        layers = []
        in_channels = embed_dim
        padding = kernel_size // 2
        
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size=kernel_size, padding=padding))
            layers.append(nn.ReLU())
            in_channels = hidden_dim
            
        self.conv_blocks = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(hidden_dim, output_vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x).transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = self.conv_blocks(x)                # (batch, hidden_dim, seq_len)
        x = x.transpose(1, 2)                  # (batch, seq_len, hidden_dim)
        return self.fc(self.dropout(x))

    @staticmethod
    def big(input_vocab_size, output_vocab_size):
        """3 Layers, Kernel 3, Dims 8/16. (Formerly Micro/Ultra)"""
        return KhoaDauCNN(input_vocab_size, output_vocab_size, num_layers=3, embed_dim=8, hidden_dim=16, kernel_size=3)

    @staticmethod
    def small(input_vocab_size, output_vocab_size):
        """3 Layers, Kernel 3, Dims 4/8. (Formerly Femto/Atomic)"""
        return KhoaDauCNN(input_vocab_size, output_vocab_size, num_layers=3, embed_dim=4, hidden_dim=8, kernel_size=3)

    @staticmethod
    def shallow_big(input_vocab_size, output_vocab_size):
        """2 Layers, Kernel 5, Dims 8/16. (Formerly Nano/Shallow_Small - CHAMPION)"""
        return KhoaDauCNN(input_vocab_size, output_vocab_size, num_layers=2, embed_dim=8, hidden_dim=16, kernel_size=5)

    @staticmethod
    def shallow_small(input_vocab_size, output_vocab_size):
        """2 Layers, Kernel 5, Dims 4/8. (Formerly Pico/Shallow_Tiny)"""
        return KhoaDauCNN(input_vocab_size, output_vocab_size, num_layers=2, embed_dim=4, hidden_dim=8, kernel_size=5)
