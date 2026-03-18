from src.models import KhoaDauCNNNano
from src.vocab_utils import get_or_build_vocab
from src.train_engine import run_training
from src.config import MODEL_NANO_PATH, CSV_PATH
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    qn_v, kd_v = get_or_build_vocab(df)
    model = KhoaDauCNNNano(len(qn_v), len(kd_v))
    run_training(model, MODEL_NANO_PATH, batch_size=512, epochs=200, lr=0.005, label="NANO")
