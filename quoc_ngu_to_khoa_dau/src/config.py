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
VOCAB_PATH = os.path.join(WEIGHTS_DIR, "vocab.pth")
INPUT_TEXT_PATH = os.path.join(BASE_DIR, "test_input.txt")

# Đường dẫn lưu model
MODEL_LARGE_PATH = os.path.join(WEIGHTS_DIR, "khoa_dau_cnn_large.pth")
MODEL_SMALL_PATH = os.path.join(WEIGHTS_DIR, "khoa_dau_cnn_small.pth")
MODEL_NANO_PATH = os.path.join(WEIGHTS_DIR, "khoa_dau_cnn_nano.pth")
