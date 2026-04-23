import os
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd

from src.config import *
from src.vocab_utils import CharTokenizer, PAD_IDX
from src.data_utils import get_dataloader
from src.models import Seq2SeqTransformer, create_mask

def train_epoch(model, optimizer, criterion, dataloader, device, scaler=None):
    model.train()
    losses = 0

    for batch in tqdm(dataloader, desc="Huấn luyện"):
        # Chuyển đổi tensor sang [seq_len, batch_size] theo yêu cầu của nn.Transformer
        src = batch['src'].transpose(0, 1).to(device)
        tgt = batch['tgt'].transpose(0, 1).to(device)

        # tgt_input bỏ token cuối cùng (EOS)
        tgt_input = tgt[:-1, :]
        # tgt_out bỏ token đầu tiên (SOS) làm target
        tgt_out = tgt[1:, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, PAD_IDX, device)

        optimizer.zero_grad()

        # Automatic Mixed Precision (AMP)
        if scaler is not None:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                
            # Ép kiểu logits về float32 trước khi tính Loss để tránh tràn số (Overflow gây ra NaN/Inf)
            loss = criterion(logits.float().reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if torch.isnan(loss) or torch.isinf(loss):
            print("CẢNH BÁO: Phát hiện Loss là NaN hoặc Inf. Dừng huấn luyện.")
            return None

        losses += loss.item()

    if len(dataloader) == 0:
        print("CẢNH BÁO: Dataloader rỗng (có thể do limit quá nhỏ so với batch_size).")
        return 0.0
    return losses / len(dataloader)

def evaluate(model, criterion, dataloader, device):
    model.eval()
    losses = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Đánh giá"):
            src = batch['src'].transpose(0, 1).to(device)
            tgt = batch['tgt'].transpose(0, 1).to(device)
            
            tgt_input = tgt[:-1, :]
            tgt_out = tgt[1:, :]
            
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, PAD_IDX, device)
            
            # AMP cho Evaluation
            if device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                    loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            else:
                logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            losses += loss.item()
            
    if len(dataloader) == 0:
        return 0.0
    return losses / len(dataloader)

def main(data_limit=None):
    # Cấu hình thiết bị
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    print(f"Đang sử dụng thiết bị: {device}")
    
    # 1. Chuẩn bị Tokenizer
    tokenizer = CharTokenizer()
    if not tokenizer.load(VOCAB_PATH):
        print("Không tìm thấy từ điển hoặc cần xây dựng mới. Đang nạp toàn bộ dữ liệu để đảm bảo bao phủ 100% ký tự...")
        all_texts = []
        for path in [TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH]:
            if os.path.exists(path):
                df_temp = pd.read_csv(path).dropna(subset=['khoa_dau', 'quoc_ngu'])
                all_texts.extend(df_temp['khoa_dau'].tolist())
                all_texts.extend(df_temp['quoc_ngu'].tolist())
        
        tokenizer.build_vocab(all_texts)
        tokenizer.save(VOCAB_PATH)
    else:
        print(f"Đã tải từ điển. Kích thước: {tokenizer.vocab_size} ký tự")

    # 2. Tải DataLoader (với giới hạn nếu có)
    print("Đang chuẩn bị DataLoader...")
    train_dataloader = get_dataloader(TRAIN_DATA_PATH, tokenizer, BATCH_SIZE, shuffle=True, limit=data_limit)
    # Val set cũng giới hạn tương ứng để nhanh
    val_limit = int(data_limit * 0.1) if data_limit else None
    val_dataloader = get_dataloader(VAL_DATA_PATH, tokenizer, BATCH_SIZE, shuffle=False, limit=val_limit)
    
    # 3. Khởi tạo mô hình
    model = Seq2SeqTransformer(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        emb_size=D_MODEL,
        nhead=N_HEAD,
        vocab_size=tokenizer.vocab_size,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # Khởi tạo Scaler cho Automatic Mixed Precision (AMP) để tăng tốc trên GPU
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # 4. Nạp Checkpoint nếu có (Hỗ trợ Resume)
    start_epoch = 1
    best_val_loss = float('inf')
    model_path = os.path.join(WEIGHTS_DIR, "best_model.pth")
    checkpoint_path = os.path.join(WEIGHTS_DIR, "checkpoint.pth")
    
    patience_counter = 0
    if os.path.exists(checkpoint_path):
        print(f"-> Phát hiện checkpoint tại {checkpoint_path}. Đang nạp để tiếp tục...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # Không tự động nạp lại optimizer_state_dict nếu muốn đổi Learning Rate
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"-> Tiếp tục từ Epoch {start_epoch} (Best Val Loss trước đó: {best_val_loss:.4f}, Patience: {patience_counter})")
        print("-> Đã áp dụng Learning Rate mới từ config.py (Bỏ qua Optimizer State cũ).")
    elif os.path.exists(model_path):
        print(f"-> Không thấy checkpoint đầy đủ, nhưng thấy best_model.pth. Đang nạp trọng số...")
        ckpt = torch.load(model_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
    
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{NUM_EPOCHS}]")
        train_loss = train_epoch(model, optimizer, criterion, train_dataloader, device, scaler=scaler)
        val_loss = evaluate(model, criterion, val_dataloader, device)
        
        if train_loss is None: break # Lỗi NaN/Inf

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Dọn dẹp VRAM sau mỗi epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Lưu model tốt nhất (để Inference)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'num_encoder_layers': NUM_ENCODER_LAYERS,
                    'num_decoder_layers': NUM_DECODER_LAYERS,
                    'emb_size': D_MODEL,
                    'nhead': N_HEAD,
                    'dim_feedforward': DIM_FEEDFORWARD
                }
            }, model_path)
            print(f"-> Đã lưu mô hình tốt nhất (Loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # Lưu checkpoint hàng epoch (để Resume)
        ckpt_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
        }
        if scaler is not None:
            ckpt_dict['scaler_state_dict'] = scaler.state_dict()
            
        torch.save(ckpt_dict, checkpoint_path)
            
        if patience_counter >= PATIENCE:
            print("Early Stopping!")
            break
    
    # Tự động đánh giá sau khi train xong
    print("\n" + "="*50)
    print("BẮT ĐẦU QUÁ TRÌNH ĐÁNH GIÁ CUỐI CÙNG...")
    from evaluate import run_evaluation
    run_evaluation(os.path.join(WEIGHTS_DIR, "best_model.pth"), TEST_DATA_PATH, num_samples=100)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số câu để train thử")
    args = parser.parse_args()
    
    # Nếu đang ở trên máy cục bộ, hãy để limit khoảng 50000 - 100000 để thấy kết quả nhanh
    main(data_limit=args.limit)
