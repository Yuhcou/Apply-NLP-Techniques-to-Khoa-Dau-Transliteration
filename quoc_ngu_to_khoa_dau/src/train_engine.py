import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import os
from .config import CSV_PATH
from .vocab_utils import get_or_build_vocab
from .data_utils import KhoaDauDataset

def run_training(model_instance, save_path, batch_size=64, epochs=100, lr=0.001, label="MODEL", patience=20):
    """Hàm huấn luyện mô hình AI chung với Early Stopping."""
    df = pd.read_csv(CSV_PATH)
    qn_v, kd_v = get_or_build_vocab(df)
    
    dataset = KhoaDauDataset(df, qn_v, kd_v)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_instance.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"--- BẮT ĐẦU HUẤN LUYỆN {label} TRÊN {device} ---")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs.view(-1, len(kd_v)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        # In log định kỳ
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs:3d} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early Stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            # Lưu model tốt nhất tạm thời
            torch.save(model.state_dict(), save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"--- DỪNG SỚM TẠI EPOCH {epoch+1} (Loss không giảm trong {patience} epochs) ---")
                break

    print(f"--- HOÀN TẤT {label}. Best Loss: {best_loss:.6f}. Model lưu tại: {save_path} ---")
