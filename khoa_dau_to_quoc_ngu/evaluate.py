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
    """Giải mã bằng thuật toán Beam Search."""
    src = src.to(device)
    src_mask = src_mask.to(device)
    
    memory = model.encode(src, src_mask)
    
    # Khởi tạo beam: (chuỗi_ids, log_score)
    # log_score khởi tạo là 0 (xác suất = 1)
    beams = [([start_symbol], 0.0)]
    
    for i in range(max_len - 1):
        new_beams = []
        for seq, score in beams:
            if seq[-1] == end_symbol:
                new_beams.append((seq, score))
                continue
                
            # Tạo target tensor cho chuỗi hiện tại
            tgt_tensor = torch.tensor(seq).unsqueeze(1).to(device)
            tgt_mask = generate_square_subsequent_mask(tgt_tensor.size(0), device).type(torch.bool)
            
            # Dự đoán
            out = model.decode(tgt_tensor, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            
            # Lấy log_softmax để tính log_score (cộng thay vì nhân)
            log_probs = torch.log_softmax(prob, dim=1)
            
            # Lấy top k ứng viên tiếp theo
            top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=1)
            
            for j in range(beam_width):
                next_word = top_indices[0][j].item()
                next_score = top_log_probs[0][j].item()
                new_beams.append((seq + [next_word], score + next_score))
        
        # Sắp xếp và giữ lại top beam_width
        # Áp dụng Length Penalty đơn giản: score / len(seq)^0.7
        beams = sorted(new_beams, key=lambda x: x[1] / (len(x[0])**0.7), reverse=True)[:beam_width]
        
        # Nếu tất cả các beam đều đã kết thúc bằng EOS
        if all(seq[-1] == end_symbol for seq, _ in beams):
            break
            
    return beams[0][0]

def run_evaluation(model_path, data_path, num_samples=500, beam_width=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = CharTokenizer()
    tokenizer.load(VOCAB_PATH)
    
    model = Seq2SeqTransformer(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        emb_size=D_MODEL,
        nhead=N_HEAD,
        vocab_size=tokenizer.vocab_size,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=0.0
    ).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Đã tải trọng số từ {model_path}")
    model.eval()
    
    df = pd.read_csv(data_path).dropna().sample(min(num_samples, 2000))
    
    total_cer, total_wer = [], []
    print(f"Đang đánh giá {len(df)} câu (Beam Width: {beam_width})...")
    
    results = []
    
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Giải mã"):
            src_text = str(row['khoa_dau'])
            tgt_text = str(row['quoc_ngu'])
            
            src_ids = tokenizer.encode(src_text)
            src_tensor = torch.tensor(src_ids).unsqueeze(1).to(device)
            num_tokens = src_tensor.shape[0]
            src_mask = torch.zeros((num_tokens, num_tokens), device=device).type(torch.bool)
            
            # Beam Search Decode
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
    model_file = os.path.join(WEIGHTS_DIR, "best_model.pth")
    if os.path.exists(model_file):
        run_evaluation(model_file, TEST_DATA_PATH)
    else:
        print(f"Lỗi: Không tìm thấy file trọng số tại {model_file}")
