import os

# Đường dẫn gốc (Từ src lên 1 bậc)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

# Tham số mô hình
MAX_LEN = 7

# Đường dẫn file dữ liệu
CSV_PATH = os.path.join(DATA_DIR, "all-vietnamese-syllables-encoded.csv")
SYLLABLES_PLAIN_PATH = os.path.join(DATA_DIR, "all-vietnamese-syllables.txt")
VOCAB_PATH = os.path.join(WEIGHTS_DIR, "vocab.pth")
INPUT_TEXT_PATH = os.path.join(DATA_DIR, "source.txt")

# New Hierarchy: Shallow_Big (100% Acc) > Big (99.9% Acc) > Shallow_Small (75% Acc) > Small (66% Acc)
MODEL_SHALLOW_BIG_PATH = os.path.join(WEIGHTS_DIR, "khoa_dau_cnn_shallow_big.pth")
MODEL_BIG_PATH = os.path.join(WEIGHTS_DIR, "khoa_dau_cnn_big.pth")
MODEL_SHALLOW_SMALL_PATH = os.path.join(WEIGHTS_DIR, "khoa_dau_cnn_shallow_small.pth")
MODEL_SMALL_PATH = os.path.join(WEIGHTS_DIR, "khoa_dau_cnn_small.pth")

# Mặc định sử dụng Shallow_Big (Champion: 100% Acc, 16KB)
MODEL_BEST_PATH = MODEL_SHALLOW_BIG_PATH
