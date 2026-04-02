import torch
import pandas as pd
import os
import time
from src.config import (MAX_LEN, VOCAB_PATH, MODEL_SHALLOW_BIG_PATH, MODEL_BIG_PATH, 
                        MODEL_SHALLOW_SMALL_PATH, MODEL_SMALL_PATH, SYLLABLES_PLAIN_PATH)
from src.models import KhoaDauCNN
from src.data_utils import fast_norm
from src.rule_based import encode_custom

class ModelMismatchEvaluator:
    def __init__(self):
        if not os.path.exists(VOCAB_PATH):
            raise FileNotFoundError(f"Vocab file not found at {VOCAB_PATH}")
            
        # Load vocab
        vocab_data = torch.load(VOCAB_PATH)
        self.qn_vocab = vocab_data['qn']
        self.kd_vocab = vocab_data['kd']
        self.rev_kd_vocab = {v: k for k, v in self.kd_vocab.items()}
        self.device = torch.device("cuda" if torch.available() else "cpu")
        
        # Load all syllables
        with open(SYLLABLES_PLAIN_PATH, 'r', encoding='utf-8') as f:
            self.syllables = [line.strip() for line in f if line.strip()]
            
    def load_model(self, model_type, path):
        input_dim = len(self.qn_vocab)
        output_dim = len(self.kd_vocab)
        
        mapping = {
            "shallow_big": KhoaDauCNN.shallow_big,
            "big": KhoaDauCNN.big,
            "shallow_small": KhoaDauCNN.shallow_small,
            "small": KhoaDauCNN.small,
        }
        
        if model_type not in mapping:
            raise ValueError(f"Invalid model type: {model_type}")
            
        model = mapping[model_type](input_dim, output_dim)
            
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            print(f"Warning: Model weight not found at {path}")
            return None
            
        model.to(self.device)
        model.eval()
        return model

    def predict_batch(self, model, syllables):
        all_encoded = []
        for s in syllables:
            clean_s = fast_norm(str(s))
            encoded = [self.qn_vocab.get(c, self.qn_vocab["<UNK>"]) for c in clean_s]
            if len(encoded) < MAX_LEN:
                encoded += [self.qn_vocab["<PAD>"]] * (MAX_LEN - len(encoded))
            else:
                encoded = encoded[:MAX_LEN]
            all_encoded.append(encoded)
            
        input_tensor = torch.tensor(all_encoded).to(self.device)
        with torch.no_grad():
            outputs = model(input_tensor)
            predictions = torch.argmax(outputs, dim=-1)
            
        results = []
        for i in range(len(predictions)):
            res = ""
            for p in predictions[i].tolist():
                char = self.rev_kd_vocab.get(p, "")
                if char not in ["<PAD>", "<UNK>"]:
                    res += char
            results.append(res)
        return results

    def evaluate_model(self, name, model):
        if model is None: return
        
        print(f"\n--- ĐANG ĐÁNH GIÁ SAI LỆCH (MISMATCH): {name.upper()} ---")
        
        # Ground truth from rule-based
        batch_size = 1024
        all_preds = []
        for i in range(0, len(self.syllables), batch_size):
            batch = self.syllables[i:i+batch_size]
            all_preds.extend(self.predict_batch(model, batch))
            
        mismatches = []
        for i, (syllable, pred) in enumerate(zip(self.syllables, all_preds)):
            truth = encode_custom(syllable)
            if pred != truth:
                mismatches.append({
                    "syllable": syllable,
                    "truth": truth,
                    "pred": pred,
                    "truth_hex": "-".join([f"{ord(c):04x}" for c in truth]),
                    "pred_hex": "-".join([f"{ord(c):04x}" for c in pred])
                })
        
        accuracy = (len(self.syllables) - len(mismatches)) / len(self.syllables) * 100
        print(f"Accuracy: {accuracy:.4f}% | Số lỗi: {len(mismatches)}")
        
        if mismatches:
            print(f"{'Âm tiết':<12} | {'Thực tế (Hex)':<20} | {'Dự đoán (Hex)':<20}")
            print("-" * 60)
            for m in mismatches[:10]: # In 10 lỗi đầu
                print(f"{m['syllable']:<12} | {m['truth_hex']:<20} | {m['pred_hex']:<20}")
            if len(mismatches) > 10:
                print(f"... và {len(mismatches)-10} lỗi khác.")

if __name__ == "__main__":
    evaluator = ModelMismatchEvaluator()
    
    models_to_test = [
        ("shallow_big", MODEL_SHALLOW_BIG_PATH),
        ("big", MODEL_BIG_PATH),
        ("shallow_small", MODEL_SHALLOW_SMALL_PATH),
        ("small", MODEL_SMALL_PATH)
    ]
    
    for name, path in models_to_test:
        model = evaluator.load_model(name, path)
        evaluator.evaluate_model(name, model)
