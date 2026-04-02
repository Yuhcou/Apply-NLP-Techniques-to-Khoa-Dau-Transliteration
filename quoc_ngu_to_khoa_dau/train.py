import argparse
import pandas as pd
from src.models import KhoaDauCNN
from src.vocab_utils import get_or_build_vocab
from src.train_engine import run_training
from src.config import (CSV_PATH, MODEL_SHALLOW_BIG_PATH, MODEL_BIG_PATH, 
                        MODEL_SHALLOW_SMALL_PATH, MODEL_SMALL_PATH)

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện các mô hình Khoa Dau CNN.")
    parser.add_argument("--model", type=str, default="shallow_big", 
                        choices=["shallow_big", "big", "shallow_small", "small"],
                        help="Kích thước mô hình (shallow_big, big, shallow_small, small).")
    parser.add_argument("--epochs", type=int, default=None, help="Số lượng epoch.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size.")
    
    args = parser.parse_args()

    # Load data
    print(f"Đang tải dữ liệu từ {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    qn_v, kd_v = get_or_build_vocab(df)
    
    input_dim = len(qn_v)
    output_dim = len(kd_v)

    # Cấu hình mapping mới
    configs = {
        "shallow_big": {
            "model_fn": KhoaDauCNN.shallow_big,
            "path": MODEL_SHALLOW_BIG_PATH,
            "epochs": 500,
            "label": "SHALLOW_BIG (Champion - 100% Acc)"
        },
        "big": {
            "model_fn": KhoaDauCNN.big,
            "path": MODEL_BIG_PATH,
            "epochs": 500,
            "label": "BIG (3 Layers, Kernel 3)"
        },
        "shallow_small": {
            "model_fn": KhoaDauCNN.shallow_small,
            "path": MODEL_SHALLOW_SMALL_PATH,
            "epochs": 500,
            "label": "SHALLOW_SMALL (2 Layers, Kernel 5, Dims 8)"
        },
        "small": {
            "model_fn": KhoaDauCNN.small,
            "path": MODEL_SMALL_PATH,
            "epochs": 500,
            "label": "SMALL (3 Layers, Kernel 3, Dims 8)"
        }
    }

    cfg = configs[args.model]
    model = cfg["model_fn"](input_dim, output_dim)
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    
    print(f"Bắt đầu huấn luyện mô hình {cfg['label']}...")
    run_training(
        model, 
        cfg["path"], 
        batch_size=args.batch_size, 
        epochs=epochs, 
        lr=args.lr, 
        label=cfg["label"],
        patience=30
    )

if __name__ == "__main__":
    main()
