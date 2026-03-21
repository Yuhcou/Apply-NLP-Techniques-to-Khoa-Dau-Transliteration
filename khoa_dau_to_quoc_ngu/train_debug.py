import os
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import sys

from src.config import *
from src.vocab_utils import CharTokenizer, PAD_IDX
from src.data_utils import get_dataloader
from src.models import Seq2SeqTransformer, create_mask

# Ép CUDA báo lỗi đồng bộ
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def train_epoch(model, optimizer, criterion, dataloader, device, tokenizer):
    model.train()
    losses = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Huấn luyện Debug")):
        # 1. Kiểm tra Tensor đầu vào (Input Validation)
        src = batch['src'].transpose(0, 1).to(device)
        tgt = batch['tgt'].transpose(0, 1).to(device)
        
        # Kiểm tra Index out of bounds trước khi nạp vào GPU
        if (src >= tokenizer.vocab_size).any() or (tgt >= tokenizer.vocab_size).any():
            print(f"\n[LỖI NGHIÊM TRỌNG] Batch {batch_idx}: Phát hiện Token ID vượt ngưỡng!")
            print(f"Max src ID: {src.max().item()}, Max tgt ID: {tgt.max().item()}")
            sys.exit(1)
            
        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:, :]
        
        # 2. Tạo Mask
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, PAD_IDX, device)
        
        # 3. Forward Pass với Anomaly Detection (Nếu cần)
        try:
            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            
            optimizer.zero_grad()
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            
            if torch.isnan(loss):
                print(f"\n[LỖI] Batch {batch_idx}: Loss là NaN!")
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses += loss.item()
            
        except Exception as e:
            print(f"\n[CRASH] Lỗi xảy ra tại Batch {batch_idx}: {str(e)}")
            # In ra thông số tại thời điểm crash
            print(f"src shape: {src.shape}")
            print(f"tgt_input shape: {tgt_input.shape}")
            print(f"src_mask shape: {src_mask.shape}")
            raise e
    
    return losses / len(dataloader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Debug Mode: Đang sử dụng thiết bị {device}")
    
    tokenizer = CharTokenizer()
    tokenizer.load(VOCAB_PATH)
    
    # Giảm Batch size để dễ debug và tránh OOM bất ngờ
    DEBUG_BATCH_SIZE = 32 
    train_dataloader = get_dataloader(TRAIN_DATA_PATH, tokenizer, DEBUG_BATCH_SIZE, shuffle=True, limit=5000)
    
    model = Seq2SeqTransformer(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        emb_size=D_MODEL,
        nhead=N_HEAD,
        vocab_size=tokenizer.vocab_size,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)
    
    # Load model vừa crash (để tiếp tục từ trạng thái đó)
    model_path = os.path.join(WEIGHTS_DIR, "best_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Đã nạp trọng số Epoch 15 để bắt đầu debug từ Epoch 16.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    print("Bắt đầu chạy Debug 1 Epoch...")
    train_epoch(model, optimizer, criterion, train_dataloader, device, tokenizer)
    print("Hoàn thành Debug. Không phát hiện lỗi nghiêm trọng.")

if __name__ == "__main__":
    main()
