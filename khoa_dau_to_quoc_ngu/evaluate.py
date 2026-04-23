import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import numpy as np

from src.config import *
from src.vocab_utils import CharTokenizer, PAD_IDX, SOS_IDX, EOS_IDX
from src.models import Seq2SeqTransformer, create_mask, generate_square_subsequent_mask

def levenshtein_distance(s1, s2):
    """Tính khoảng cách Levenshtein để dùng cho CER/WER."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def calculate_metrics(target, predicted):
    """Tính toán CER và WER cho một cặp câu."""
    # CER
    dist_char = levenshtein_distance(target, predicted)
    cer = dist_char / max(len(target), 1)
    
    # WER
    t_words = target.split()
    p_words = predicted.split()
    dist_word = levenshtein_distance(t_words, p_words)
    wer = dist_word / max(len(t_words), 1)
    
    return cer, wer

def beam_search_decode(model, src, src_mask, max_len, start_symbol, end_symbol, device, beam_width=3):
    """Giải mã bằng thuật toán Beam Search (Đã tối ưu hóa Vectorized để tránh thắt cổ chai GPU)."""
    src = src.to(device)
    if src_mask is not None:
        src_mask = src_mask.to(device)
    
    # memory: [src_len, 1, d_model]
    memory = model.encode(src, src_mask)
    
    # Khởi tạo beam: (chuỗi_ids, log_score)
    beams = [([start_symbol], 0.0)]
    
    for i in range(max_len - 1):
        active_beams = []
        finished_beams = []
        
        for seq, score in beams:
            if seq[-1] == end_symbol:
                finished_beams.append((seq, score))
            else:
                active_beams.append((seq, score))
                
        if not active_beams:
            break
            
        num_active = len(active_beams)
        curr_len = len(active_beams[0][0])
        
        # Gom tất cả các chuỗi đang active thành 1 batch tensor: [curr_len, num_active]
        seq_tensors = [torch.tensor(b[0], dtype=torch.long) for b in active_beams]
        tgt_tensor = torch.stack(seq_tensors, dim=1).to(device)
        tgt_mask = generate_square_subsequent_mask(curr_len, device).type(torch.bool)
        
        # Mở rộng memory cho khớp với num_active beams hiện tại
        # [src_len, 1, d_model] -> [src_len, num_active, d_model]
        batched_memory = memory.expand(-1, num_active, -1).contiguous()
        
        # 1 lần suy luận duy nhất trên GPU cho toàn bộ k beams
        out = model.decode(tgt_tensor, batched_memory, tgt_mask)
        
        # Chỉ lấy hidden states của token cuối cùng: [num_active, d_model]
        prob = model.generator(out[-1, :, :])
        
        # Nếu model sinh ra NaN do khác biệt phân phối độ dài (Train MAX_LEN=128 nhưng Eval > 128),
        # dừng giải mã nhánh này để tránh lỗi CUDA Illegal Memory Access
        if torch.isnan(prob).any() or torch.isinf(prob).any():
            break
            
        log_probs = torch.log_softmax(prob, dim=1) # [num_active, vocab_size]
        top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=1)
        
        new_beams = finished_beams.copy()
        for b_idx in range(num_active):
            seq, current_score = active_beams[b_idx]
            for j in range(beam_width):
                next_word = top_indices[b_idx][j].item()
                next_score = top_log_probs[b_idx][j].item()
                new_beams.append((seq + [next_word], current_score + next_score))
        
        # Sắp xếp và giữ lại top beam_width (Áp dụng Length Penalty)
        beams = sorted(new_beams, key=lambda x: x[1] / (len(x[0])**0.7), reverse=True)[:beam_width]
            
    return beams[0][0]

def run_evaluation(model_path, data_path, num_samples=500, beam_width=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = CharTokenizer()
    tokenizer.load(VOCAB_PATH)
    
    # Load checkpoint to check for config
    ckpt = None
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        
    if isinstance(ckpt, dict) and 'config' in ckpt:
        config = ckpt['config']
        enc_layers = config.get('num_encoder_layers', NUM_ENCODER_LAYERS)
        dec_layers = config.get('num_decoder_layers', NUM_DECODER_LAYERS)
        emb_size = config.get('emb_size', D_MODEL)
        nhead = config.get('nhead', N_HEAD)
        dim_ff = config.get('dim_feedforward', DIM_FEEDFORWARD)
    else:
        state_dict = ckpt['model_state_dict'] if (isinstance(ckpt, dict) and 'model_state_dict' in ckpt) else ckpt
        
        enc_layers = NUM_ENCODER_LAYERS
        dec_layers = NUM_DECODER_LAYERS
        emb_size = D_MODEL
        nhead = N_HEAD
        dim_ff = DIM_FEEDFORWARD
        
        if state_dict is not None:
            try:
                if 'src_tok_emb.embedding.weight' in state_dict:
                    emb_size = state_dict['src_tok_emb.embedding.weight'].shape[1]
                enc_keys = [k for k in state_dict.keys() if k.startswith('transformer.encoder.layers.')]
                if enc_keys:
                    enc_layers = max([int(k.split('.')[3]) for k in enc_keys]) + 1
                dec_keys = [k for k in state_dict.keys() if k.startswith('transformer.decoder.layers.')]
                if dec_keys:
                    dec_layers = max([int(k.split('.')[3]) for k in dec_keys]) + 1
                if 'transformer.encoder.layers.0.linear1.weight' in state_dict:
                    dim_ff = state_dict['transformer.encoder.layers.0.linear1.weight'].shape[0]
                if emb_size == 256:
                    nhead = 8
                elif emb_size == 512:
                    nhead = 16
            except Exception as e:
                print(f"Lỗi Auto-detect: {e}")
    
    model = Seq2SeqTransformer(
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        emb_size=emb_size,
        nhead=nhead,
        vocab_size=tokenizer.vocab_size,
        dim_feedforward=dim_ff,
        dropout=0.0
    ).to(device)
    
    if ckpt is not None:
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        print(f"Đã tải trọng số từ {model_path} (D_MODEL: {emb_size}, Lớp: {enc_layers})")
    model.eval()
    
    # Đọc và lọc bỏ các câu dài hơn MAX_LEN để tránh lỗi CUDA và đánh giá chính xác
    df_full = pd.read_csv(data_path).dropna()
    valid_rows = []
    for _, row in df_full.iterrows():
        if len(tokenizer.encode(str(row['khoa_dau']))) <= MAX_LEN and len(tokenizer.encode(str(row['quoc_ngu']))) <= MAX_LEN:
            valid_rows.append(row)
            
    df = pd.DataFrame(valid_rows)
    if len(df) == 0:
        print(f"CẢNH BÁO: Không có câu nào trong tập test thỏa mãn điều kiện <= {MAX_LEN} ký tự.")
        return 0.0, 0.0
        
    df = df.sample(min(num_samples, len(df)), random_state=42)
    
    total_cer, total_wer = [], []
    print(f"Đang đánh giá {len(df)} câu (Beam Width: {beam_width}, MAX_LEN: {MAX_LEN})...")
    
    results = []
    
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Giải mã"):
            src_text = str(row['khoa_dau'])
            tgt_text = str(row['quoc_ngu'])
            
            src_ids = tokenizer.encode(src_text)
            src_tensor = torch.tensor(src_ids).unsqueeze(1).to(device)
            
            # Đối với nn.Transformer, nếu src_mask không che gì cả (all False), tốt nhất truyền None 
            src_mask = None
            
            # Beam Search Decode tuân thủ tuyệt đối giới hạn MAX_LEN của mô hình
            out_ids = beam_search_decode(model, src_tensor, src_mask, MAX_LEN, SOS_IDX, EOS_IDX, device, beam_width)
            predicted_text = tokenizer.decode(out_ids)
            
            cer, wer = calculate_metrics(tgt_text, predicted_text)
            total_cer.append(cer)
            total_wer.append(wer)
            
            if len(results) < 5:
                results.append({
                    "src": src_text,
                    "target": tgt_text,
                    "pred": predicted_text,
                    "cer": cer
                })

    avg_cer = np.mean(total_cer)
    avg_wer = np.mean(total_wer)
    
    print("\n" + "="*30)
    print(f"KẾT QUẢ ĐÁNH GIÁ (Metrics):")
    print(f" - Character Error Rate (CER): {avg_cer:.4f} (Càng thấp càng tốt)")
    print(f" - Word Error Rate (WER):      {avg_wer:.4f}")
    print("="*30)
    
    print("\nVí dụ thực tế:")
    for res in results:
        print(f"  Input:  {res['src'][:50]}...")
        print(f"  Target: {res['target']}")
        print(f"  Pred:   {res['pred']}")
        print(f"  CER:    {res['cer']:.4f}")
        print("-" * 20)

    return avg_cer, avg_wer

if __name__ == "__main__":
    import sys
    beam_width = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    # Mặc định dùng tham số thứ 2 làm tên file, nếu không có thì mặc định lấy best_model.pth
    model_name = sys.argv[2] if len(sys.argv) > 2 else "best_model.pth"
    model_file = os.path.join(WEIGHTS_DIR, model_name)
    if os.path.exists(model_file):
        run_evaluation(model_file, TEST_DATA_PATH, beam_width=beam_width)
    else:
        print(f"Lỗi: Không tìm thấy file trọng số tại {model_file}")
